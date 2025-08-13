import streamlit as st
import pandas as pd
import requests
from azure.identity import InteractiveBrowserCredential, DefaultAzureCredential
from openai import OpenAI
import os
from dotenv import load_dotenv
import traceback
import re
import io
import matplotlib.pyplot as plt
import datetime
import csv
import pyodbc
import sqlalchemy as sa
import struct
from itertools import chain, repeat
import urllib
from azure import identity
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss

# --- Example Queries (customize as needed) ---
QUERIES = {
    "EPC Non-Domestic Scotland": '''
        query($first: Int!) {
            epcNonDomesticScotlands(first: $first) {
                items {
                    CURRENT_ENERGY_PERFORMANCE_BAND
                    CURRENT_ENERGY_PERFORMANCE_RATING
                    LODGEMENT_DATE
                    PRIMARY_ENERGY_VALUE
                    BUILDING_EMISSIONS
                    FLOOR_AREA
                    PROPERTY_TYPE
                }
            }
        }
    ''',
    # Add more queries as needed
}

# --- Table schemas for column types ---
TABLE_SCHEMAS = {
    "EPC Non-Domestic Scotland": {
        "numeric": [
            "CURRENT_ENERGY_PERFORMANCE_RATING",
            "PRIMARY_ENERGY_VALUE",
            "BUILDING_EMISSIONS",
            "FLOOR_AREA"
        ],
        "categorical": [
            "CURRENT_ENERGY_PERFORMANCE_BAND",
            "PROPERTY_TYPE"
        ],
        "datetime": [
            "LODGEMENT_DATE"
        ]
    },
    # Add more tables here as needed
}

def process_table_types(df, table_name):
    schema = TABLE_SCHEMAS.get(table_name, {})
    # Convert numeric columns
    for col in schema.get("numeric", []):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    # Convert datetime columns
    for col in schema.get("datetime", []):
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    # Convert categorical columns to category dtype (optional)
    for col in schema.get("categorical", []):
        if col in df.columns:
            df[col] = df[col].astype("category")
    return df

@st.cache_data
def get_cached_embeddings(_engine, date_range=None):
    return fetch_embeddings(_engine, date_range)

@st.cache_resource
def get_faiss_index(emb_df):
    embeddings_matrix = np.vstack(emb_df['embedding_np'].values).astype('float32')
    dim = embeddings_matrix.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings_matrix)
    return index 

# --- Hybrid RAG Embedding and Semantic Search Functions ---
def parse_embedding(embedding_str):
    if isinstance(embedding_str, str):
        return np.array([float(x) for x in embedding_str.strip('[]').split(',')])
    elif isinstance(embedding_str, list):
        return np.array([float(x) for x in embedding_str])
    else:
        raise ValueError("Unknown embedding format")

def embed_question(question, model):
    return np.array(model.encode([question])[0])

def vector_search(question_embedding, embeddings_matrix, top_n=100):
    norms = np.linalg.norm(embeddings_matrix, axis=1) * np.linalg.norm(question_embedding)
    sims = np.dot(embeddings_matrix, question_embedding) / (norms + 1e-8)
    top_indices = np.argsort(sims)[-top_n:][::-1]
    return top_indices, sims[top_indices]

def fetch_embeddings(engine, date_range=None):
    sql = """
    SELECT POSTCODE, PROPERTY_TYPE, MAIN_HEATING_FUEL, embedding, LODGEMENT_DATE
    FROM LH_external_datasets.embedding.epcNonDomesticScotlandVE
    """
    if date_range:
        start, end = date_range
        sql += f" WHERE LODGEMENT_DATE >= '{start}' AND LODGEMENT_DATE <= '{end}'"
    df = pd.read_sql(sql, engine)
    df['embedding_np'] = df['embedding'].apply(parse_embedding)
    return df

def fetch_raw_data(engine, keys, date_range=None):
    if not keys:
        return pd.DataFrame()
    conditions = [
        f"(POSTCODE = '{postcode}' AND PROPERTY_TYPE = '{ptype}' AND MAIN_HEATING_FUEL = '{fuel}')"
        for postcode, ptype, fuel in keys
    ]
    where_clause = ' OR '.join(conditions)
    sql = f'''
    SELECT POSTCODE, MAIN_HEATING_FUEL, CURRENT_ENERGY_PERFORMANCE_BAND, CURRENT_ENERGY_PERFORMANCE_RATING,
           LODGEMENT_DATE, PRIMARY_ENERGY_VALUE, BUILDING_EMISSIONS, FLOOR_AREA, PROPERTY_TYPE
    FROM LH_external_datasets.epc.epcNonDomesticScotland
    WHERE {where_clause}
    '''
    if date_range:
        start, end = date_range
        sql += f" AND LODGEMENT_DATE >= '{start}' AND LODGEMENT_DATE <= '{end}'"
    return pd.read_sql(sql, engine)

def hybrid_rag(question, model, engine, top_n=100, date_range=None):
    emb_df = fetch_embeddings(engine, date_range)
    if emb_df.empty:
        return pd.DataFrame()
    embeddings_matrix = np.vstack(emb_df['embedding_np'].values)
    q_emb = embed_question(question, model)
    top_indices, _ = vector_search(q_emb, embeddings_matrix, top_n=top_n)
    top_keys = emb_df.iloc[top_indices][['POSTCODE', 'PROPERTY_TYPE', 'MAIN_HEATING_FUEL']].values.tolist()
    df_subset = fetch_raw_data(engine, top_keys, date_range)
    return df_subset

def log_audit_entry(question, reasoning, code, result, is_retry=False):
    log_file = "rag_audit_log.csv"
    now = datetime.datetime.now().isoformat()
    # Handle different result types
    if result is None:
        row_count = col_count = columns = None
    elif isinstance(result, pd.DataFrame):
        row_count = len(result)
        col_count = len(result.columns)
        columns = ','.join(result.columns)
    elif isinstance(result, pd.Series):
        row_count = len(result)
        col_count = 1
        columns = result.name if result.name else ''
    else:  # scalar (number, string, etc.)
        row_count = 1
        col_count = 1
        columns = ''
    with open(log_file, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        if f.tell() == 0:
            writer.writerow(["timestamp", "is_retry", "question", "reasoning", "code", "row_count", "col_count", "columns"])
        writer.writerow([now, is_retry, question, reasoning, code, row_count, col_count, columns])

@st.cache_resource
def get_credential():
    app = InteractiveBrowserCredential()
    scp = 'https://analysis.windows.net/powerbi/api/user_impersonation'
    return app, scp

# --- Model Selection UI ---
st.sidebar.header("Embedding Model")
model_options = {
    "Default (all-MiniLM-L6-v2)": "all-MiniLM-L6-v2",
    "Fine-tuned: epc-ndscotland-finetuned": "models/epc-ndscotland-finetuned"
}
selected_model_name = st.sidebar.selectbox("Select embedding model", list(model_options.keys()), index=0)
selected_model_path = model_options[selected_model_name]

# --- Main Navigation as Tabs ---
tabs = st.tabs(["RAG QA", "SQL Editor"])

def get_embedding_model(model_path=None):
    if model_path is None:
        model_path = "all-MiniLM-L6-v2"
    return SentenceTransformer(model_path)

with tabs[0]:
    st.header("RAG QA")
    # Model selection in main window
    model_options = {
        "Default (all-MiniLM-L6-v2)": "all-MiniLM-L6-v2",
        "Fine-tuned: epc-ndscotland-finetuned": "models/epc-ndscotland-finetuned"
    }
    selected_model_name = st.selectbox("Select embedding model", list(model_options.keys()), index=0)
    selected_model_path = model_options[selected_model_name]
    model = get_embedding_model(selected_model_path)

    # Table selection in main window
    st.subheader("Data Source")
    selected_tables = st.multiselect("Select Table(s)/Query(ies)", list(QUERIES.keys()), default=list(QUERIES.keys())[:1])
    batch_size = st.number_input("Records to fetch per table", min_value=100, max_value=5000, value=1000, step=100)

    # Date filter in main window
    min_date = datetime.date(2000, 1, 1)
    max_date = datetime.date.today()
    date_range = st.date_input("Select date range for LODGEMENT_DATE", [min_date, max_date])
    if len(date_range) == 2:
        start_date, end_date = date_range
    else:
        start_date, end_date = min_date, max_date

    # --- Place the rest of the RAG QA logic here, using model, selected_tables, batch_size, start_date, end_date ---
    # ...

with tabs[1]:
    st.header("SQL Editor (Microsoft Fabric SQL Endpoint)")
    # Place all SQL Editor logic here
    # ...

# Update get_embedding_model to accept a model_path argument
def get_embedding_model(model_path=None):
    if model_path is None:
        model_path = "all-MiniLM-L6-v2"
    return SentenceTransformer(model_path)

# In RAG QA tab, use:
# model = get_embedding_model(selected_model_path)

@st.cache_resource
def get_fabric_engine():
    import urllib
    from azure.identity import DefaultAzureCredential
    import struct
    from itertools import chain, repeat
    import sqlalchemy as sa

    sql_endpoint = "twprzboxbsruppjvrglogyufxu-uywupunqqmwepeeo6rptcas3jq.datawarehouse.fabric.microsoft.com"
    database = "default"  # Change if needed
    driver = "ODBC Driver 18 for SQL Server"  # Or allow user selection

    # Use DefaultAzureCredential for consistency with SQL Editor
    credential = DefaultAzureCredential()
    resource_url = "https://database.windows.net/.default"
    token_object = credential.get_token(resource_url)

    # Prepare connection string
    connection_string = (
        f"Driver={{{driver}}};"
        f"Server={sql_endpoint},1433;"
        f"Database={database};"
        "Encrypt=Yes;"
        "TrustServerCertificate=No"
    )
    params = urllib.parse.quote(connection_string)

    # Encode the access token for ODBC
    token_as_bytes = bytes(token_object.token, "UTF-8")
    encoded_bytes = bytes(chain.from_iterable(zip(token_as_bytes, repeat(0))))
    token_bytes = struct.pack("<i", len(encoded_bytes)) + encoded_bytes
    attrs_before = {1256: token_bytes}

    engine = sa.create_engine(
        f"mssql+pyodbc:///?odbc_connect={params}",
        connect_args={'attrs_before': attrs_before}
    )
    return engine

# --- Main Navigation ---
page = st.sidebar.radio("Navigation", ["RAG QA", "SQL Editor"])

def sql_editor():
    st.title("SQL Editor (Microsoft Fabric SQL Endpoint)")
    st.info("You are connected to the Fabric SQL endpoint using your Azure credentials. Enter a SQL query below and click 'Run Query'.")
    sql_endpoint = "twprzboxbsruppjvrglogyufxu-uywupunqqmwepeeo6rptcas3jq.datawarehouse.fabric.microsoft.com"
    database = "default"  # Change if needed

    # Driver selection
    driver = st.selectbox(
        "Select ODBC Driver",
        ["ODBC Driver 18 for SQL Server", "ODBC Driver 17 for SQL Server"],
        index=0
    )
    st.write(f"Using driver: {driver}")

    # Authentication
    resource_url = "https://database.windows.net/.default"
    azure_credentials = identity.DefaultAzureCredential()
    token_object = azure_credentials.get_token(resource_url)

    # Prepare connection string
    connection_string = (
        f"Driver={{{driver}}};"
        f"Server={sql_endpoint},1433;"
        f"Database={database};"
        "Encrypt=Yes;"
        "TrustServerCertificate=No"
    )
    params = urllib.parse.quote(connection_string)

    # Encode the access token for ODBC
    token_as_bytes = bytes(token_object.token, "UTF-8")
    encoded_bytes = bytes(chain.from_iterable(zip(token_as_bytes, repeat(0))))
    token_bytes = struct.pack("<i", len(encoded_bytes)) + encoded_bytes
    attrs_before = {1256: token_bytes}

    query = st.text_area("Enter SQL query", "SELECT TOP 10 * FROM INFORMATION_SCHEMA.TABLES;")
    if st.button("Run Query"):
        try:
            engine = sa.create_engine(
                f"mssql+pyodbc:///?odbc_connect={params}",
                connect_args={'attrs_before': attrs_before}
            )
            df = pd.read_sql(query, engine)
            MAX_DISPLAY_ROWS = 1000
            if len(df) > MAX_DISPLAY_ROWS:
                st.warning(f"Result has {len(df)} rows. Displaying only the first {MAX_DISPLAY_ROWS} rows.")
                st.dataframe(df.head(MAX_DISPLAY_ROWS))
            else:
                st.dataframe(df)
            # Add download button for full results
            if not df.empty:
                csv = df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="Download results as CSV",
                    data=csv,
                    file_name="sql_query_results.csv",
                    mime="text/csv"
                )
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button("Download full result as CSV", data=csv, file_name="result.csv", mime="text/csv")
        except Exception as e:
            st.error(f"SQL query failed: {e}")

if page == "SQL Editor":
    sql_editor()
    st.stop()

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
DEFAULT_MODEL = os.getenv('DEFAULT_MODEL', 'gpt-3.5-turbo')
MAX_TOKENS = int(os.getenv('MAX_TOKENS', '600'))
TEMPERATURE = float(os.getenv('TEMPERATURE', '0.2'))

# Initialize OpenAI client
client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

st.set_page_config(page_title="Fabric RAG QA", layout="wide")
st.title("Fabric RAG (Retrieval-Augmented Generation) QA App")

# --- Safe built-ins for code execution ---
SAFE_BUILTINS = {
    "zip": zip,
    "len": len,
    "range": range,
    "min": min,
    "max": max,
    "sum": sum,
    "abs": abs,
    "sorted": sorted,
    "list": list,
    "dict": dict,
    "set": set,
    "float": float,
    "int": int,
    "str": str,
    "enumerate": enumerate,
    "any": any,
    "all": all,
    "round": round,
}

# --- Data summarization for LLM context ---
def summarize_dataframe(df):
    summary = []
    summary.append(f"Shape: {df.shape[0]} rows x {df.shape[1]} columns\n")
    summary.append("Column types:")
    summary.append(str(df.dtypes))
    summary.append("\nMissing values per column:")
    summary.append(str(df.isnull().sum()))
    # Numeric summary
    num_cols = df.select_dtypes(include='number').columns
    if len(num_cols) > 0:
        summary.append("\nNumeric columns summary:")
        summary.append(str(df[num_cols].describe().T))
    # Categorical summary
    cat_cols = df.select_dtypes(include='object').columns
    for col in cat_cols:
        vc = df[col].value_counts(dropna=False)
        summary.append(f"\nTop values for {col}:\n{vc.head(5)}")
    return "\n".join(summary)

# --- Helper to clean LLM code output ---
def clean_code(code):
    # Remove triple backticks and optional 'python' from start/end
    code = re.sub(r"^```(?:python)?\s*", "", code.strip(), flags=re.IGNORECASE)
    code = re.sub(r"```$", "", code.strip())
    return code.strip()

# --- Helper to check for dangerous code ---
FORBIDDEN_KEYWORDS = [
    '__import__', 'open(', 'exec(', 'eval(', 'os.', 'sys.', 'subprocess', 'import os', 'import sys', 'import subprocess',
    'shutil', 'socket', 'input(', 'globals(', 'locals(', 'compile(', 'del ', 'setattr(', 'getattr(', 'exit(', 'quit('
]
def is_code_safe(code):
    code_lower = code.lower()
    for keyword in FORBIDDEN_KEYWORDS:
        if keyword in code_lower:
            return False, keyword
    return True, None

# --- Helper to validate code uses actual DataFrame and column names ---
def validate_code_uses_actual_names(code, available_dfs, available_columns, user_question):
    """Check if generated code uses only actual DataFrame and column names, allowing dynamically created columns and 'result' as a variable."""
    import re
    
    # Extract DataFrame names from code (e.g., df1, df2, etc.); allow 'result' as a variable
    df_pattern = r'\b(df\d+|result)\b'
    found_dfs = re.findall(df_pattern, code)
    
    # Track columns: start with available columns, then add dynamically created ones as we parse
    valid_columns = set(available_columns)
    created_columns = set()
    
    # Split code into lines for sequential parsing
    code_lines = code.split('\n')
    
    # Pattern to match column assignment: df['COL'] = ...
    assign_pattern = re.compile(r"(df\d+|result)\[['\"]([^'\"]+)['\"]\]\s*=")
    # Pattern to match column references: df['COL']
    col_pattern = re.compile(r"(df\d+|result)\[['\"]([^'\"]+)['\"]\]")
    
    for line in code_lines:
        # First, check for assignments and add to created_columns
        for m in assign_pattern.finditer(line):
            df_name, col_name = m.groups()
            created_columns.add(col_name)
            valid_columns.add(col_name)
        # Then, check for column references
        for m in col_pattern.finditer(line):
            df_name, col_name = m.groups()
            # Only flag as invalid if not in valid_columns (original or created)
            if col_name not in valid_columns:
                # Allow some flexibility for common exploration patterns
                if any(pattern in col_name.lower() for pattern in ['temp', 'result', 'filtered', 'search']):
                    continue  # Might be a dynamic variable
                else:
                    return False, f"Code references non-existent columns: [{col_name}]. Available: {sorted(list(valid_columns))}"
    # Check for invalid DataFrame names, but ignore 'result' as it is allowed as a variable
    invalid_dfs = [df for df in found_dfs if df != 'result' and df not in available_dfs]
    if invalid_dfs:
        return False, f"Code references non-existent DataFrames: {invalid_dfs}. Available: {available_dfs}"
    return True, "Code uses valid DataFrame and column names (including dynamically created columns and 'result' as a variable)"

# --- Visualization keyword detection ---
VISUALIZATION_KEYWORDS = [
    'plot', 'chart', 'visualize', 'visualisation', 'visualization', 'graph', 'bar', 'line', 'scatter', 'histogram', 'pie', 'distribution', 'trend', 'heatmap'
]
def is_visualization_request(question):
    q = question.lower()
    return any(word in q for word in VISUALIZATION_KEYWORDS)

# --- Hybrid RAG: Keyword-based filtering for large datasets ---
LARGE_DATA_THRESHOLD = 50000
FILTERED_SAMPLE_SIZE = 500

def extract_keywords(question):
    # Simple keyword extraction: split on spaces, remove stopwords, short words, and punctuation
    stopwords = set(['the','is','a','an','of','and','to','in','on','for','by','with','as','at','from','that','this','it','be','or','are','was','were','has','have','had','but','not','which','can','will','would','should','could','if','then','so','do','does','did','about','into','over','after','before','between','under','above','more','less','most','least','all','any','each','other','such','no','yes','than','when','where','who','what','how'])
    words = re.findall(r'\b\w+\b', question.lower())
    keywords = [w for w in words if w not in stopwords and len(w) > 2]
    return keywords

def filter_df_by_keywords(df, keywords):
    if not keywords:
        return df
    str_cols = df.select_dtypes(include='object').columns
    if len(str_cols) == 0:
        return df
    mask = pd.Series([False] * len(df))
    for col in str_cols:
        for kw in keywords:
            mask = mask | df[col].str.contains(kw, case=False, na=False)
    filtered = df[mask]
    return filtered

# --- Dynamic dataset searchability for RAG ---
def prepare_comprehensive_context(dfs, df1_context, user_question):
    """Prepare comprehensive context that makes entire dataset searchable"""
    # Create detailed context for each DataFrame
    context_parts = []
    for name, df in dfs.items():
        # Get column info
        columns_info = f"DataFrame '{name}' columns: {', '.join(df.columns)}"
        # Get sample data with unique values for categorical columns
        sample_data = df.head(20).to_csv(index=False)
        # Get unique values for categorical columns to help LLM understand data
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        unique_values_info = []
        for col in categorical_cols[:5]:  # Limit to first 5 categorical columns
            unique_vals = df[col].value_counts().head(10).index.tolist()
            unique_values_info.append(f"'{col}' unique values: {unique_vals}")
        # Add guidance for substring/case-insensitive matching
        if len(categorical_cols) > 0:
            unique_values_info.append("Note: When filtering for these columns, use substring and case-insensitive matching (e.g., str.contains('value', case=False, na=False)). Do not assume exact matches.")
        # Get numeric column statistics
        numeric_cols = df.select_dtypes(include=['number']).columns
        numeric_info = []
        for col in numeric_cols[:5]:  # Limit to first 5 numeric columns
            stats = df[col].describe()
            numeric_info.append(f"'{col}' stats: min={stats['min']:.2f}, max={stats['max']:.2f}, mean={stats['mean']:.2f}")
        # Combine all info for this DataFrame
        df_context = f"""
{columns_info}
Sample data:
{sample_data}
{chr(10).join(unique_values_info)}
{chr(10).join(numeric_info)}
"""
        context_parts.append(df_context)
    # Add df1_context (semantic search results) if available
    if df1_context is not None and len(df1_context) > 0:
        df1_info = f"""
Semantic search results (df1) - most relevant to your question:
Columns: {', '.join(df1_context.columns)}
Sample data:
{df1_context.head(50).to_csv(index=False)}
"""
        context_parts.insert(0, df1_info)  # Put semantic results first
    return "\n\n".join(context_parts)

def create_intelligent_prompt(user_question, comprehensive_context):
    """Create an intelligent prompt that encourages exploration of all data"""
    
    prompt = f"""
You are a data analyst with access to comprehensive dataset information. Your task is to answer the user's question by exploring ALL available data.

CRITICAL: You MUST follow this exact step-by-step approach with robust data type handling:

STEP 1: ALWAYS check data types first
```python
# Check data types before any operations
print("Data types:")
print(df1.dtypes)
```

STEP 2: Identify and convert data types safely
```python
# Get numeric columns (int64, float64) - NOT category or object
numeric_cols = df1.select_dtypes(include=['int64', 'float64']).columns
print("Numeric columns (safe for mean/sum):", list(numeric_cols))

# Get category/object columns (NOT safe for mean/sum)
category_cols = df1.select_dtypes(include=['category', 'object']).columns
print("Category/Object columns:", list(category_cols))

# Convert string columns to numeric when needed (with error handling)
for column_name in category_cols:
    if column_name in ['FLOOR_AREA', 'PRIMARY_ENERGY_VALUE', 'BUILDING_EMISSIONS', 'CURRENT_ENERGY_PERFORMANCE_RATING']:
        try:
            df1[column_name] = pd.to_numeric(df1[column_name], errors='coerce')
            print(f"Converted {{column_name}} to numeric")
        except Exception as e:
            print(f"Could not convert {{column_name}}: {{e}}")
```

STEP 3: Safe aggregation approach with error handling
```python
# For numeric comparisons, ONLY use numeric columns
numeric_cols = df1.select_dtypes(include=['int64', 'float64']).columns
if len(numeric_cols) > 0:
    # Safe: only aggregate numeric columns
    result = df1.groupby('PROPERTY_TYPE')[numeric_cols].agg(['mean', 'count'])
else:
    # If no numeric columns, use categorical methods
    result = df1.groupby('PROPERTY_TYPE').size()
```

STEP 4: For visualization, handle data type conversion
```python
# For plotting, ensure numeric data
if 'FLOOR_AREA' in df1.columns:
    # Convert to numeric if needed
    df1['FLOOR_AREA'] = pd.to_numeric(df1['FLOOR_AREA'], errors='coerce')
    # Remove NaN values for plotting
    df1_clean = df1.dropna(subset=['FLOOR_AREA'])
    if len(df1_clean) > 0:
        result = df1_clean.groupby('PROPERTY_TYPE')['FLOOR_AREA'].mean()
        fig = result.plot(kind='bar', figsize=(12, 6))
    else:
        print("No valid numeric data for visualization")
```

IMPORTANT RULES:
- ALWAYS convert string columns to numeric using pd.to_numeric(column, errors='coerce')
- Handle conversion errors gracefully with try/except
- Remove NaN values before plotting with .dropna()
- Use .describe() only on numeric columns
- For visualization, ensure data is numeric and clean

ROBUST DATA TYPE HANDLING:
- Convert string columns: pd.to_numeric(column, errors='coerce')
- Handle NaN values: .dropna(subset=[column])
- Check for valid data: len(df_clean) > 0
- Use error handling: try/except blocks

AVAILABLE DATA:
{comprehensive_context}

USER QUESTION: {user_question}

Write Python code that:
1. ALWAYS starts with checking .dtypes
2. Converts string columns to numeric when needed
3. Handles conversion errors gracefully
4. Removes NaN values before operations
5. Uses appropriate operations for each data type
6. Returns the answer to the user's question

IMPORTANT: Include robust data type conversion and error handling!
"""
    return prompt

# --- Microsoft Fabric Authentication and Data Fetching ---
def fetch_fabric_data(query, variables=None):
    app, scp = get_credential()
    result = app.get_token(scp)
    if not result.token:
        st.error("Could not get access token")
        return None
    headers = {
        'Authorization': f'Bearer {result.token}',
        'Content-Type': 'application/json'
    }
    endpoint = 'https://d1472da683b0472c908ef45f31025b4c.zd1.graphql.fabric.microsoft.com/v1/workspaces/d1472da6-83b0-472c-908e-f45f31025b4c/graphqlapis/8d0c3265-e4de-49ec-bd12-615f3551ef6f/graphql'
    payload = {'query': query}
    if variables:
        payload['variables'] = variables
    response = requests.post(endpoint, json=payload, headers=headers)
    response.raise_for_status()
    data = response.json()
    if 'errors' in data:
        st.error(f"GraphQL errors: {data['errors']}")
        return None
    return data

# --- Intelligent table selection for questions ---
def determine_relevant_tables_for_question(user_question, available_tables):
    """Determine which tables are most relevant for a given question"""
    import re
    
    # Keywords that might indicate which tables are relevant
    table_keywords = {
        "epcNonDomesticScotlands": ["non-domestic", "scotland", "commercial", "office", "restaurant", "shop", "building"],
        "epcDomesticEngWales": ["domestic", "england", "wales", "home", "house", "residential", "property"],
        "epcDomesticScotlands": ["domestic", "scotland", "home", "house", "residential", "property"],
        "epcNonDomesticEngWales": ["non-domestic", "england", "wales", "commercial", "office", "restaurant", "shop", "building"],
        "scotDomChangesOverTimes": ["scotland", "domestic", "changes", "trends", "over time", "historical"]
    }
    
    question_lower = user_question.lower()
    relevant_tables = []
    
    # Check which tables match the question keywords
    for table_name, keywords in table_keywords.items():
        if table_name in available_tables:
            for keyword in keywords:
                if keyword in question_lower:
                    relevant_tables.append(table_name)
                    break
    
    # If no specific matches, return all available tables
    if not relevant_tables:
        relevant_tables = list(available_tables)
    
    return relevant_tables

# --- Data Selection UI ---
st.sidebar.header("Data Source")
selected_tables = st.sidebar.multiselect("Select Table(s)/Query(ies)", list(QUERIES.keys()), default=list(QUERIES.keys())[:1])
batch_size = st.sidebar.number_input("Records to fetch per table", min_value=100, max_value=5000, value=1000, step=100)

# Add date range filter to sidebar before Fetch Data button
min_date = datetime.date(2000, 1, 1)
max_date = datetime.date.today()
date_range = st.sidebar.date_input("Select date range for LODGEMENT_DATE", [min_date, max_date])
if len(date_range) == 2:
    start_date, end_date = date_range
else:
    start_date, end_date = min_date, max_date

# --- Preview Table Button (On-Demand EDA) ---
if st.sidebar.button("Preview Table"):
    dfs = {}
    summaries = {}
    preview_progress = st.progress(0, text="Fetching preview sample...")
    with st.spinner("Fetching preview sample from Fabric for selected tables..."):
        for i, table in enumerate(selected_tables):
            query = QUERIES[table]
            variables = {"first": batch_size}
            variables["lodgementDateStart"] = str(start_date)
            variables["lodgementDateEnd"] = str(end_date)
            data = fetch_fabric_data(query, variables=variables)
            if data and 'data' in data and list(data['data'].keys()):
                key = list(data['data'].keys())[0]
                items = data['data'][key]['items']
                df = pd.DataFrame(items)
                if 'LODGEMENT_DATE' in df.columns:
                    df['LODGEMENT_DATE'] = pd.to_datetime(df['LODGEMENT_DATE'], errors='coerce')
                    df = df[(df['LODGEMENT_DATE'] >= pd.to_datetime(start_date)) & (df['LODGEMENT_DATE'] <= pd.to_datetime(end_date))]
                elif 'lodgement_date' in df.columns:
                    df['lodgement_date'] = pd.to_datetime(df['lodgement_date'], errors='coerce')
                    df = df[(df['lodgement_date'] >= pd.to_datetime(start_date)) & (df['lodgement_date'] <= pd.to_datetime(end_date))]
                df = process_table_types(df, table)
                # --- Sampling for large datasets ---
                if len(df) > FILTERED_SAMPLE_SIZE:
                    st.warning(f"Sampling {FILTERED_SAMPLE_SIZE} rows from {len(df)} for table {table} to improve performance.")
                    df = df.sample(n=FILTERED_SAMPLE_SIZE, random_state=42)
                dfs[f'df{i+1}'] = df
                summaries[f'df{i+1}'] = summarize_dataframe(df)
            preview_progress.progress((i + 1) / len(selected_tables), text=f"Fetched {i+1}/{len(selected_tables)} tables")
    preview_progress.empty()
    st.session_state['fabric_dfs'] = dfs
    st.session_state['fabric_summaries'] = summaries
    st.success(f"Fetched {len(dfs)} table(s) for preview.")

# --- Data Preview and Summaries (On-Demand) ---
dfs = st.session_state.get('fabric_dfs', {})
summaries = st.session_state.get('fabric_summaries', {})
if dfs:
    st.subheader("Data Preview (All Loaded Tables)")
    for name, df in dfs.items():
        st.markdown(f"**{name}**")
        # Pagination for DataFrame display
        total_rows = len(df)
        rows_per_page = 20
        total_pages = (total_rows - 1) // rows_per_page + 1
        page = st.number_input(f"Page for {name}", min_value=1, max_value=total_pages, value=1, key=f"page_{name}")
        start_idx = (page - 1) * rows_per_page
        end_idx = start_idx + rows_per_page
        st.dataframe(df.iloc[start_idx:end_idx], use_container_width=True)
        with st.expander(f"Show Data Summary for {name} (used for LLM context)", expanded=False):
            st.text(summaries[name])

# --- Fetch tables with progress bar ---
def fetch_tables_for_question(user_question, selected_tables, batch_size, start_date, end_date):
    all_available_tables = list(QUERIES.keys())
    relevant_tables = determine_relevant_tables_for_question(user_question, all_available_tables)
    tables_to_fetch = list(set(selected_tables + relevant_tables))
    if not selected_tables:
        tables_to_fetch = relevant_tables[:2]
    st.info(f"Fetching data for question from tables: {', '.join(tables_to_fetch)}")
    dfs = {}
    summaries = {}
    progress = st.progress(0, text="Fetching tables...")
    with st.spinner(f"Fetching data from {len(tables_to_fetch)} table(s) for your question..."):
        for i, table in enumerate(tables_to_fetch):
            query = QUERIES[table]
            variables = {"first": batch_size}
            variables["lodgementDateStart"] = str(start_date)
            variables["lodgementDateEnd"] = str(end_date)
            data = fetch_fabric_data(query, variables=variables)
            if data and 'data' in data and list(data['data'].keys()):
                key = list(data['data'].keys())[0]
                items = data['data'][key]['items']
                df = pd.DataFrame(items)
                if 'LODGEMENT_DATE' in df.columns:
                    df['LODGEMENT_DATE'] = pd.to_datetime(df['LODGEMENT_DATE'], errors='coerce')
                    df = df[(df['LODGEMENT_DATE'] >= pd.to_datetime(start_date)) & (df['LODGEMENT_DATE'] <= pd.to_datetime(end_date))]
                elif 'lodgement_date' in df.columns:
                    df['lodgement_date'] = pd.to_datetime(df['lodgement_date'], errors='coerce')
                    df = df[(df['lodgement_date'] >= pd.to_datetime(start_date)) & (df['lodgement_date'] <= pd.to_datetime(end_date))]
                df = process_table_types(df, table)
                # --- Sampling for large datasets ---
                if len(df) > FILTERED_SAMPLE_SIZE:
                    st.warning(f"Sampling {FILTERED_SAMPLE_SIZE} rows from {len(df)} for table {table} to improve performance.")
                    df = df.sample(n=FILTERED_SAMPLE_SIZE, random_state=42)
                dfs[f'df{len(dfs)+1}'] = df
                summaries[f'df{len(dfs)}'] = summarize_dataframe(df)
            progress.progress((i + 1) / len(tables_to_fetch), text=f"Fetched {i+1}/{len(tables_to_fetch)} tables")
    progress.empty()
    return dfs, summaries

# --- RAG Q&A Section (On-Demand Hybrid RAG with FAISS) ---
st.subheader("Ask a Question about the Data (powered by OpenAI, multi-table)")
st.info("Note: All queries, code, and AI reasoning are logged for audit and improvement purposes.")

with st.form(key="rag_qa_form"):
    user_question = st.text_input("Enter your question about the data (you can reference df1, df2, etc.):")
    show_code = st.checkbox("Show generated code", value=True)
    auto_fetch = st.checkbox("Auto-fetch relevant tables for question", value=True, 
                            help="Automatically fetch tables that are relevant to your question, even if not previewed")
    submit_rag = st.form_submit_button("Submit Question")

download_placeholder = st.empty()

# --- Step 9: Input Validation ---
MIN_QUESTION_LENGTH = 8
MAX_DISPLAY_ROWS = 1000
MAX_DOWNLOAD_ROWS = 10000
ambiguous = False
if submit_rag:
    if user_question:
        if len(user_question.strip()) < MIN_QUESTION_LENGTH:
            st.warning("Your question is very short. Please provide more detail for better results.")
            ambiguous = True
        # Optionally, check for table/column references
        if not any([col in user_question for df in dfs.values() for col in df.columns]):
            st.info("Tip: Reference specific columns or tables (e.g., df1, df2, column names) for more accurate answers.")
    if not user_question or ambiguous:
        st.stop()

    if user_question and client:
        status_placeholder = st.empty()
        status_placeholder.info("Step 1: Embedding user question...")
        is_viz = is_visualization_request(user_question)
        try:
            model = get_embedding_model(selected_model_path)
            status_placeholder.info("Step 2: Connecting to Fabric SQL endpoint...")
            engine = get_fabric_engine()
            status_placeholder.info("Step 3: Fetching and caching embeddings...")
            emb_df = get_cached_embeddings(engine, date_range=(start_date, end_date))
            status_placeholder.info("Step 4: Building/using FAISS index for semantic search...")
            index = get_faiss_index(emb_df)
            q_emb = embed_question(user_question, model)
            top_n = 5000
            D, I = index.search(q_emb.reshape(1, -1).astype('float32'), top_n)
            top_indices = I[0]
            top_keys = emb_df.iloc[top_indices][['POSTCODE', 'PROPERTY_TYPE', 'MAIN_HEATING_FUEL']].values.tolist()
            status_placeholder.info("Step 5: Fetching raw data for top semantic matches...")
            df1_context = fetch_raw_data(engine, top_keys, date_range=(start_date, end_date))
            context_info = f"Using top {min(len(df1_context), top_n)} semantically relevant rows for context in df1 (via FAISS)."
            st.info(context_info)
        except Exception as e:
            st.error(f"Error in hybrid RAG semantic search: {e}")
            df1_context = dfs.get('df1')
        
        # Auto-fetch relevant tables if enabled
        if auto_fetch and (not dfs or len(dfs) == 0):
            status_placeholder.info("Step 5.5: Auto-fetching relevant tables for your question...")
            dfs, summaries = fetch_tables_for_question(user_question, selected_tables, batch_size, start_date, end_date)
            st.session_state['fabric_dfs'] = dfs
            st.session_state['fabric_summaries'] = summaries
            st.success(f"Auto-fetched {len(dfs)} relevant table(s) for your question.")
        
        status_placeholder.info("Step 6: Preparing LLM context and prompt...")
        if df1_context is not None:
            df1_columns = list(df1_context.columns)
            n_context_rows = min(len(df1_context), top_n)
            df1_sample_csv = df1_context.head(n_context_rows).to_csv(index=False)
            
            # Create comprehensive context that makes entire dataset searchable
            comprehensive_context = prepare_comprehensive_context(dfs, df1_context, user_question)
            
            # Create execution environment with actual DataFrames
            exec_env = {
                'pd': pd, 
                'plt': plt, 
                '__builtins__': SAFE_BUILTINS
            }
            # Add actual DataFrames to execution environment
            for name, df in dfs.items():
                exec_env[name] = df
            # Add df1_context for semantic search results
            exec_env['df1'] = df1_context

            # --- Pre-process: Convert relevant columns in df1 to numeric ---
            if 'df1' in exec_env and exec_env['df1'] is not None:
                for col in ['FLOOR_AREA', 'PRIMARY_ENERGY_VALUE', 'BUILDING_EMISSIONS', 'CURRENT_ENERGY_PERFORMANCE_RATING']:
                    if col in exec_env['df1'].columns:
                        exec_env['df1'][col] = pd.to_numeric(exec_env['df1'][col], errors='coerce')

            status_placeholder.info("Step 7: Generating reasoning with LLM...")
            
            # Create intelligent prompt that encourages data exploration
            reasoning_prompt = create_intelligent_prompt(user_question, comprehensive_context)
            with st.spinner("Generating reasoning with OpenAI..."):
                try:
                    reasoning_response = client.chat.completions.create(
                        model=DEFAULT_MODEL,
                        messages=[
                            {"role": "system", "content": "You are a senior data analyst. Explain your reasoning step by step for the user's question, referencing the DataFrames and columns you would use. Do not write code yet."},
                            {"role": "user", "content": reasoning_prompt}
                        ],
                        temperature=0.2,
                        max_tokens=400
                    )
                    reasoning = reasoning_response.choices[0].message.content.strip()
                    st.markdown("**AI Reasoning (Chain-of-Thought):**\n" + reasoning)
                except Exception as e:
                    st.error(f"OpenAI API error during reasoning: {str(e)}")
                    reasoning = ""

            # --- Now generate code, using the reasoning as context ---
            if is_viz:
                prompt = f"""
You are a Python data analyst. The following pandas DataFrames are loaded: {', '.join(dfs.keys())}.

IMPORTANT: Use ONLY the actual column names provided below. Do NOT invent or assume column names.

{comprehensive_context}

Here is your reasoning for how to answer the question:\n{reasoning}\n
Write Python code to answer the following question. Use ONLY the actual column names from the DataFrames above. If a plot is required, use matplotlib and assign the figure to a variable called fig. If a tabular result is also needed, assign it to a variable called result. Only output the code (no explanations, no print statements, no comments). Question: {user_question}
"""
            else:
                prompt = f"""
You are a Python data analyst. The following pandas DataFrames are loaded: {', '.join(dfs.keys())}.

IMPORTANT: Use ONLY the actual column names provided below. Do NOT invent or assume column names.

{comprehensive_context}

Here is your reasoning for how to answer the question:\n{reasoning}\n
Write Python pandas code to answer the following question. Use ONLY the actual column names from the DataFrames above. Assign the final answer to a variable called result. Only output the code (no explanations, no print statements, no comments). Question: {user_question}
"""
            with st.spinner("Generating code with OpenAI..."):
                try:
                    code_response = client.chat.completions.create(
                        model=DEFAULT_MODEL,
                        messages=[
                            {"role": "system", "content": "You are a helpful Python data analyst. Only output code that produces the answer as a DataFrame or Series, and assign it to a variable called result. If a plot is required, assign the matplotlib figure to a variable called fig. You can use any of the loaded DataFrames (df1, df2, etc.)."},
                            {"role": "user", "content": prompt}
                        ],
                        temperature=0.0,
                        max_tokens=700
                    )
                    pandas_code = code_response.choices[0].message.content.strip()
                    pandas_code = clean_code(pandas_code)
                    if show_code:
                        st.code(pandas_code, language="python")
                    
                    # --- Validate code uses actual names before execution ---
                    all_available_columns = set()
                    for name, df in dfs.items():
                        all_available_columns.update(df.columns)
                    if 'df1' in exec_env:
                        all_available_columns.update(exec_env['df1'].columns)
                    
                    is_valid, validation_msg = validate_code_uses_actual_names(
                        pandas_code, 
                        list(exec_env.keys()), 
                        list(all_available_columns),
                        user_question
                    )
                    
                    if not is_valid:
                        st.error(f"Generated code uses invalid names: {validation_msg}")
                        st.error("Please rephrase your question to use only the actual DataFrame and column names.")
                        st.stop()
                    
                    # --- Safety check before execution ---
                    is_safe, keyword = is_code_safe(pandas_code)
                    if not is_safe:
                        st.error(f"Blocked potentially unsafe code: found forbidden keyword '{keyword}'. Please rephrase your question.")
                    else:
                        # 2. Try to execute the code safely using exec
                        try:
                            local_vars = exec_env.copy()
                            exec(pandas_code, {"pd": pd, "plt": plt, "__builtins__": SAFE_BUILTINS}, local_vars)
                            result = local_vars.get("result")
                            fig = local_vars.get("fig")
                            # --- Audit log for initial attempt ---
                            log_audit_entry(user_question, reasoning, pandas_code, result, is_retry=False)
                            if fig is not None:
                                st.write("**Visualization:**")
                                st.pyplot(fig)
                            if result is not None:
                                if isinstance(result, pd.DataFrame):
                                    nrows = len(result)
                                    if nrows > MAX_DISPLAY_ROWS:
                                        st.warning(f"Result has {nrows} rows. Displaying only the first {MAX_DISPLAY_ROWS} rows.")
                                        st.dataframe(result.head(MAX_DISPLAY_ROWS))
                                    else:
                                        st.dataframe(result)
                                    # Download button for DataFrame
                                    if nrows > MAX_DOWNLOAD_ROWS:
                                        st.warning(f"Result has {nrows} rows. Download will be limited to the first {MAX_DOWNLOAD_ROWS} rows.")
                                        csv = result.head(MAX_DOWNLOAD_ROWS).to_csv(index=False).encode('utf-8')
                                    else:
                                        csv = result.to_csv(index=False).encode('utf-8')
                                    download_placeholder.download_button(
                                        label="Download result as CSV",
                                        data=csv,
                                        file_name="rag_result.csv",
                                        mime="text/csv"
                                    )
                                    # 3. Optionally, summarize the result with LLM
                                    summary_prompt = f"Summarize the following table in plain English for the question: '{user_question}'.\n\n{result.head(10).to_csv(index=False)}"
                                    summary_response = client.chat.completions.create(
                                        model=DEFAULT_MODEL,
                                        messages=[
                                            {"role": "system", "content": "You are a data analyst. Summarize the table for the user question."},
                                            {"role": "user", "content": summary_prompt}
                                        ],
                                        temperature=TEMPERATURE,
                                        max_tokens=MAX_TOKENS
                                    )
                                    st.markdown("**Summary:**\n" + summary_response.choices[0].message.content)
                                elif isinstance(result, pd.Series):
                                    nrows = len(result)
                                    st.dataframe(result)
                                    # Download button for Series
                                    csv = result.to_frame().to_csv(index=False).encode('utf-8')
                                    download_placeholder.download_button(
                                        label="Download result as CSV",
                                        data=csv,
                                        file_name="rag_result.csv",
                                        mime="text/csv"
                                    )
                                    summary_prompt = f"Summarize the following table in plain English for the question: '{user_question}'.\n\n{result.head(10).to_csv(index=False)}"
                                    summary_response = client.chat.completions.create(
                                        model=DEFAULT_MODEL,
                                        messages=[
                                            {"role": "system", "content": "You are a data analyst. Summarize the table for the user question."},
                                            {"role": "user", "content": summary_prompt}
                                        ],
                                        temperature=TEMPERATURE,
                                        max_tokens=MAX_TOKENS
                                    )
                                    st.markdown("**Summary:**\n" + summary_response.choices[0].message.content)
                                else:  # scalar (number, string, etc.)
                                    st.write("**Result:**", result)
                            if fig is None and result is None:
                                st.warning("No result or visualization was produced. Please try rephrasing your question.")
                        except Exception as e:
                            st.error(f"Error executing generated code: {e}\n{traceback.format_exc()}")
                            
                            # Check if it's a data type error and provide helpful guidance
                            error_str = str(e).lower()
                            if any(keyword in error_str for keyword in ['unsupported operand', 'type', 'arithmetic', 'string', 'agg function failed', 'could not convert string']):
                                st.warning("""
**Data Type Error Detected:**
The code tried to perform numeric operations on non-numeric data.

**The Problem:**
String columns need to be converted to numeric before operations. Even if .dtypes shows numeric, the actual data might be strings.

**Robust Solution Template:**
```python
# Step 1: Check data types
print("Data types:")
print(df1.dtypes)

# Step 2: Convert string columns to numeric with error handling
for col in ['FLOOR_AREA', 'PRIMARY_ENERGY_VALUE', 'BUILDING_EMISSIONS', 'CURRENT_ENERGY_PERFORMANCE_RATING']:
    if col in df1.columns:
        try:
            df1[col] = pd.to_numeric(df1[col], errors='coerce')
            print(f"Converted {col} to numeric")
        except Exception as e:
            print(f"Could not convert {col}: {e}")

# Step 3: Remove NaN values for clean data
df1_clean = df1.dropna(subset=['FLOOR_AREA'])  # or relevant column
print(f"Clean data rows: {len(df1_clean)}")

# Step 4: Safe aggregation on clean numeric data
if len(df1_clean) > 0:
    result = df1_clean.groupby('PROPERTY_TYPE')['FLOOR_AREA'].mean()
    fig = result.plot(kind='bar', figsize=(12, 6))
else:
    print("No valid numeric data for visualization")
```

**Key Rules:**
- ALWAYS convert string columns: `pd.to_numeric(column, errors='coerce')`
- Handle NaN values: `.dropna(subset=[column])`
- Check for valid data: `len(df_clean) > 0`
- Use error handling: `try/except` blocks
- For visualization, ensure data is numeric and clean
""")
                            
                            # --- Error Feedback Loop: Ask LLM to fix the code ---
                            with st.spinner("Code execution failed. Asking OpenAI to fix the code and retrying..."):
                                fix_prompt = f"""
The following code was generated to answer the question: '{user_question}'. However, it failed with the error: {e}.

IMPORTANT: This appears to be a data type error. Please fix the code by:
1. Checking data types with .dtypes before operations
2. Using appropriate methods for each data type (strings vs numbers)
3. Using .groupby() and aggregation for categorical comparisons
4. Avoiding arithmetic operations on string columns

Here is the reasoning for how to answer the question:\n{reasoning}\n
Please correct the code and assign the final answer to a variable called result and/or fig. Only output the corrected code.

Original code:
{pandas_code}
"""
                                try:
                                    fix_response = client.chat.completions.create(
                                        model=DEFAULT_MODEL,
                                        messages=[
                                            {"role": "system", "content": "You are a helpful Python data analyst. Only output corrected code that produces the answer as a DataFrame or Series, and assign it to a variable called result. If a plot is required, assign the matplotlib figure to a variable called fig. You can use any of the loaded DataFrames (df1, df2, etc.)."},
                                            {"role": "user", "content": fix_prompt}
                                        ],
                                        temperature=0.0,
                                        max_tokens=900
                                    )
                                    fixed_code = fix_response.choices[0].message.content.strip()
                                    fixed_code = clean_code(fixed_code)
                                    st.info("Retrying with corrected code:")
                                    if show_code:
                                        st.code(fixed_code, language="python")
                                    is_safe2, keyword2 = is_code_safe(fixed_code)
                                    if not is_safe2:
                                        st.error(f"Blocked potentially unsafe code in retry: found forbidden keyword '{keyword2}'. Please rephrase your question.")
                                    else:
                                        try:
                                            local_vars2 = exec_env.copy()
                                            exec(fixed_code, {"pd": pd, "plt": plt, "__builtins__": SAFE_BUILTINS}, local_vars2)
                                            result2 = local_vars2.get("result")
                                            fig2 = local_vars2.get("fig")
                                            # --- Audit log for retry ---
                                            log_audit_entry(user_question, reasoning, fixed_code, result2, is_retry=True)
                                            if fig2 is not None:
                                                st.write("**Visualization (after retry):**")
                                                st.pyplot(fig2)
                                            if result2 is not None:
                                                if isinstance(result2, pd.DataFrame):
                                                    nrows2 = len(result2)
                                                    if nrows2 > MAX_DISPLAY_ROWS:
                                                        st.warning(f"Result has {nrows2} rows. Displaying only the first {MAX_DISPLAY_ROWS} rows.")
                                                        st.dataframe(result2.head(MAX_DISPLAY_ROWS))
                                                    else:
                                                        st.dataframe(result2)
                                                    # Download button for retry DataFrame
                                                    if nrows2 > MAX_DOWNLOAD_ROWS:
                                                        st.warning(f"Result has {nrows2} rows. Download will be limited to the first {MAX_DOWNLOAD_ROWS} rows.")
                                                        csv2 = result2.head(MAX_DOWNLOAD_ROWS).to_csv(index=False).encode('utf-8')
                                                    else:
                                                        csv2 = result2.to_csv(index=False).encode('utf-8')
                                                    download_placeholder.download_button(
                                                        label="Download result as CSV (retry)",
                                                        data=csv2,
                                                        file_name="rag_result_retry.csv",
                                                        mime="text/csv"
                                                    )
                                                    summary_prompt2 = f"Summarize the following table in plain English for the question: '{user_question}'.\n\n{result2.head(10).to_csv(index=False)}"
                                                    summary_response2 = client.chat.completions.create(
                                                        model=DEFAULT_MODEL,
                                                        messages=[
                                                            {"role": "system", "content": "You are a data analyst. Summarize the table for the user question."},
                                                            {"role": "user", "content": summary_prompt2}
                                                        ],
                                                        temperature=TEMPERATURE,
                                                        max_tokens=MAX_TOKENS
                                                    )
                                                    st.markdown("**Summary:**\n" + summary_response2.choices[0].message.content)
                                                elif isinstance(result2, pd.Series):
                                                    nrows2 = len(result2)
                                                    st.dataframe(result2)
                                                    # Download button for retry Series
                                                    csv2 = result2.to_frame().to_csv(index=False).encode('utf-8')
                                                    download_placeholder.download_button(
                                                        label="Download result as CSV (retry)",
                                                        data=csv2,
                                                        file_name="rag_result_retry.csv",
                                                        mime="text/csv"
                                                    )
                                                    summary_prompt2 = f"Summarize the following table in plain English for the question: '{user_question}'.\n\n{result2.head(10).to_csv(index=False)}"
                                                    summary_response2 = client.chat.completions.create(
                                                        model=DEFAULT_MODEL,
                                                        messages=[
                                                            {"role": "system", "content": "You are a data analyst. Summarize the table for the user question."},
                                                            {"role": "user", "content": summary_prompt2}
                                                        ],
                                                        temperature=TEMPERATURE,
                                                        max_tokens=MAX_TOKENS
                                                    )
                                                    st.markdown("**Summary:**\n" + summary_response2.choices[0].message.content)
                                                else:  # scalar (number, string, etc.)
                                                    st.write("**Result (retry):**", result2)
                                            if fig2 is None and result2 is None:
                                                st.warning("No result or visualization was produced after retry. Please try rephrasing your question.")
                                        except Exception as e2:
                                            st.error(f"Retry also failed: {e2}\n{traceback.format_exc()}")
                                except Exception as e_fix:
                                    st.error(f"OpenAI API error during code fix: {str(e_fix)}")
                except Exception as e:
                    st.error(f"OpenAI API error: {str(e)}")
elif user_question:
    st.warning("OpenAI API key not configured. Please set it in your .env file.")
else:
    st.info("Fetch data from Fabric to begin.") 