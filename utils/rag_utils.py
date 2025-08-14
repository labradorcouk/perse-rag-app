import pandas as pd
import numpy as np
import requests
import json
import re
import struct
from itertools import chain, repeat
import urllib.parse
import streamlit as st

# Conditional imports for ODBC support
try:
    import sqlalchemy as sa
    ODBC_AVAILABLE = True
except ImportError:
    ODBC_AVAILABLE = False
    sa = None

class RAGUtils:
    @staticmethod
    def get_fabric_engine():
        if not ODBC_AVAILABLE:
            raise ImportError("ODBC drivers not available. SQL functionality is limited.")
        
        from azure.identity import DefaultAzureCredential
        sql_endpoint = "twprzboxbsruppjvrglogyufxu-uywupunqqmwepeeo6rptcas3jq.datawarehouse.fabric.microsoft.com"
        database = "default"
        driver = "ODBC Driver 18 for SQL Server"
        credential = DefaultAzureCredential()
        resource_url = "https://database.windows.net/.default"
        token_object = credential.get_token(resource_url)
        connection_string = (
            f"Driver={{{driver}}};"
            f"Server={sql_endpoint},1433;"
            f"Database={database};"
            "Encrypt=Yes;"
            "TrustServerCertificate=No"
        )
        params = urllib.parse.quote(connection_string)
        token_as_bytes = bytes(token_object.token, "UTF-8")
        encoded_bytes = bytes(chain.from_iterable(zip(token_as_bytes, repeat(0))))
        token_bytes = struct.pack("<i", len(encoded_bytes)) + encoded_bytes
        attrs_before = {1256: token_bytes}
        engine = sa.create_engine(
            f"mssql+pyodbc:///?odbc_connect={params}",
            connect_args={'attrs_before': attrs_before}
        )
        return engine

    @staticmethod
    def process_table_types(df, table_name, table_schemas):
        schema = table_schemas.get(table_name, {})
        for col in schema.get("numeric", []):
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        for col in schema.get("datetime", []):
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors="coerce")
        for col in schema.get("categorical", []):
            if col in df.columns:
                df[col] = df[col].astype("category")
        return df

    @staticmethod
    def parse_embedding(embedding_str):
        if isinstance(embedding_str, str):
            return np.array([float(x) for x in embedding_str.strip('[]').split(',')])
        elif isinstance(embedding_str, list):
            return np.array([float(x) for x in embedding_str])
        else:
            raise ValueError("Unknown embedding format")

    @staticmethod
    def embed_question(question, model):
        return np.array(model.encode([question])[0])

    @staticmethod
    def vector_search(question_embedding, embeddings_matrix, top_n=100):
        norms = np.linalg.norm(embeddings_matrix, axis=1) * np.linalg.norm(question_embedding)
        sims = np.dot(embeddings_matrix, question_embedding) / (norms + 1e-8)
        top_indices = np.argsort(sims)[-top_n:][::-1]
        return top_indices, sims[top_indices]

    @staticmethod
    def fetch_embeddings(engine, date_range=None):
        sql = """
        SELECT POSTCODE, PROPERTY_TYPE, MAIN_HEATING_FUEL, embedding, LODGEMENT_DATE
        FROM LH_external_datasets.embedding.epcNonDomesticScotlandVE
        """
        if date_range:
            start, end = date_range
            sql += f" WHERE LODGEMENT_DATE >= '{start}' AND LODGEMENT_DATE <= '{end}'"
        df = pd.read_sql(sql, engine)
        # Do NOT add embedding_np here!
        return df

    @staticmethod
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

    @staticmethod
    def hybrid_rag(question, model, engine, top_n=100, date_range=None):
        # --- Debug: Step 1: Fetch embeddings ---
        try:
            emb_df = RAGUtils.fetch_embeddings(engine, date_range)
            # Add embedding_np column here, after fetching (and after any caching)
            emb_df['embedding_np'] = emb_df['embedding'].apply(RAGUtils.parse_embedding)
            print("[hybrid_rag] Step 1: Fetched embeddings shape:", emb_df.shape)
            if emb_df.empty:
                print("[hybrid_rag] No embeddings found. Returning empty DataFrame.")
                return pd.DataFrame()
            print("[hybrid_rag] Sample embedding (first row):", type(emb_df['embedding_np'].iloc[0]), emb_df['embedding_np'].iloc[0].shape if hasattr(emb_df['embedding_np'].iloc[0], 'shape') else emb_df['embedding_np'].iloc[0])
        except Exception as e:
            print(f"[hybrid_rag] Error fetching embeddings: {e}")
            return pd.DataFrame()

        # --- Debug: Step 2: Stack embeddings matrix ---
        try:
            embeddings_matrix = np.vstack(emb_df['embedding_np'].values)
            print("[hybrid_rag] Step 2: Embeddings matrix shape:", embeddings_matrix.shape)
        except Exception as e:
            print(f"[hybrid_rag] Error stacking embeddings: {e}")
            return pd.DataFrame()

        # --- Debug: Step 3: Embed question ---
        try:
            q_emb = RAGUtils.embed_question(question, model)
            print("[hybrid_rag] Step 3: Question embedding shape:", q_emb.shape)
        except Exception as e:
            print(f"[hybrid_rag] Error embedding question: {e}")
            return pd.DataFrame()

        # --- Debug: Step 4: Vector search ---
        try:
            top_indices, sims = RAGUtils.vector_search(q_emb, embeddings_matrix, top_n=top_n)
            print("[hybrid_rag] Step 4: Top indices:", top_indices)
            print("[hybrid_rag] Step 4: Top similarities:", sims)
        except Exception as e:
            print(f"[hybrid_rag] Error in vector search: {e}")
            return pd.DataFrame()

        # --- Debug: Step 5: Extract top keys ---
        try:
            top_keys = emb_df.iloc[top_indices][['POSTCODE', 'PROPERTY_TYPE', 'MAIN_HEATING_FUEL']].values.tolist()
            print("[hybrid_rag] Step 5: Top keys (first 5):", top_keys[:5])
        except Exception as e:
            print(f"[hybrid_rag] Error extracting top keys: {e}")
            return pd.DataFrame()

        # --- Debug: Step 6: Fetch raw data ---
        try:
            df_subset = RAGUtils.fetch_raw_data(engine, top_keys, date_range)
            print("[hybrid_rag] Step 6: Fetched raw data shape:", df_subset.shape)
        except Exception as e:
            print(f"[hybrid_rag] Error fetching raw data: {e}")
            return pd.DataFrame()
        return df_subset

    @staticmethod
    def summarize_dataframe(df):
        summary = []
        summary.append(f"Shape: {df.shape[0]} rows x {df.shape[1]} columns\n")
        summary.append("Column types:")
        summary.append(str(df.dtypes))
        summary.append("\nMissing values per column:")
        summary.append(str(df.isnull().sum()))
        num_cols = df.select_dtypes(include='number').columns
        if len(num_cols) > 0:
            summary.append("\nNumeric columns summary:")
            summary.append(str(df[num_cols].describe().T))
        cat_cols = df.select_dtypes(include='object').columns
        for col in cat_cols:
            vc = df[col].value_counts(dropna=False)
            summary.append(f"\nTop values for {col}:\n{vc.head(5)}")
        return "\n".join(summary)

    @staticmethod
    def extract_keywords(question):
        stopwords = set(['the','is','a','an','of','and','to','in','on','for','by','with','as','at','from','that','this','it','be','or','are','was','were','has','have','had','but','not','which','can','will','would','should','could','if','then','so','do','does','did','about','into','over','after','before','between','under','above','more','less','most','least','all','any','each','other','such','no','yes','than','when','where','who','what','how'])
        words = re.findall(r'\b\w+\b', question.lower())
        keywords = [w for w in words if w not in stopwords and len(w) > 2]
        return keywords

    @staticmethod
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

    @staticmethod
    def prepare_comprehensive_context(dfs, df1_context, user_question, context_sample_size=20):
        context_parts = []
        for name, df in dfs.items():
            columns_info = f"DataFrame '{name}' columns: {', '.join(df.columns)}"
            sample_data = df.head(20).to_csv(index=False)
            categorical_cols = df.select_dtypes(include=['object', 'category']).columns
            unique_values_info = []
            for col in categorical_cols[:5]:
                unique_vals = df[col].value_counts().head(10).index.tolist()
                unique_values_info.append(f"'{col}' unique values: {unique_vals}")
            if len(categorical_cols) > 0:
                unique_values_info.append("Note: When filtering for these columns, use substring and case-insensitive matching (e.g., str.contains('value', case=False, na=False)). Do not assume exact matches.")
            numeric_cols = df.select_dtypes(include=['number']).columns
            numeric_info = []
            for col in numeric_cols[:5]:
                stats = df[col].describe()
                numeric_info.append(f"'{col}' stats: min={stats['min']:.2f}, max={stats['max']:.2f}, mean={stats['mean']:.2f}")
            df_context = f"""
{columns_info}
Sample data:
{sample_data}
{chr(10).join(unique_values_info)}
{chr(10).join(numeric_info)}
"""
            context_parts.append(df_context)
        if df1_context is not None and len(df1_context) > 0:
            df1_info = f"""
Semantic search results (df1) - most relevant to your question:
Columns: {', '.join(df1_context.columns)}
Sample data:
{df1_context.head(context_sample_size).to_csv(index=False)}
"""
            context_parts.insert(0, df1_info)
        return "\n\n".join(context_parts)

    @staticmethod
    def create_intelligent_prompt(user_question, comprehensive_context):
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

# ALWAYS check and convert date columns (such as 'LODGEMENT_DATE') to datetime before using .dt accessors or time-based grouping
if 'LODGEMENT_DATE' in df1.columns:
    df1['LODGEMENT_DATE'] = pd.to_datetime(df1['LODGEMENT_DATE'], errors='coerce')

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
- ALWAYS check and convert date columns (such as 'LODGEMENT_DATE') to datetime using pd.to_datetime before using .dt accessors or time-based grouping
- Handle conversion errors gracefully with try/except
- Remove NaN values before plotting with .dropna()
- Use .describe() only on numeric columns
- For visualization, ensure data is numeric and clean

ROBUST DATA TYPE HANDLING:
- Convert string columns: pd.to_numeric(column, errors='coerce')
- Convert date columns: pd.to_datetime(column, errors='coerce') before using .dt
- Handle NaN values: .dropna(subset=[column])
- Check for valid data: len(df_clean) > 0
- Use error handling: try/except blocks

AVAILABLE DATA:
{comprehensive_context}

USER QUESTION: {user_question}

Write Python code that:
1. ALWAYS starts with checking .dtypes
2. Converts string columns to numeric when needed
3. Converts date columns (such as 'LODGEMENT_DATE') to datetime before using .dt or time-based grouping
4. Handles conversion errors gracefully
5. Removes NaN values before operations
6. Uses appropriate operations for each data type
7. Returns the answer to the user's question

IMPORTANT: Include robust data type conversion and error handling, especially for date columns!
"""
        return prompt

    @staticmethod
    def fetch_fabric_data(query, variables=None, get_credential=None):
        app, scp = get_credential()
        result = app.get_token(scp)
        if not result.token:
            raise RuntimeError("Could not get access token")
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
            raise RuntimeError(f"GraphQL errors: {data['errors']}")
        return data

    @staticmethod
    def is_visualization_request(question):
        VISUALIZATION_KEYWORDS = [
            'plot', 'chart', 'visualize', 'visualisation', 'visualization', 'graph', 'bar', 'line', 'scatter', 'histogram', 'pie', 'distribution', 'trend', 'heatmap'
        ]
        q = question.lower()
        return any(word in q for word in VISUALIZATION_KEYWORDS)

    @staticmethod
    def get_embedding_model(model_path=None):
        from sentence_transformers import SentenceTransformer
        if model_path is None:
            model_path = "all-MiniLM-L6-v2"
        return SentenceTransformer(model_path)

    @staticmethod
    def fetch_tables_for_question(user_question, selected_tables, batch_size, start_date, end_date, QUERIES, get_credential):
        # This is a refactored version of the original fetch_tables_for_question
        def determine_relevant_tables_for_question(user_question, available_tables):
            table_keywords = {
                "epcNonDomesticScotlands": ["non-domestic", "scotland", "commercial", "office", "restaurant", "shop", "building"],
                "epcDomesticEngWales": ["domestic", "england", "wales", "home", "house", "residential", "property"],
                "epcDomesticScotlands": ["domestic", "scotland", "home", "house", "residential", "property"],
                "epcNonDomesticEngWales": ["non-domestic", "england", "wales", "commercial", "office", "restaurant", "shop", "building"],
                "scotDomChangesOverTimes": ["scotland", "domestic", "changes", "trends", "over time", "historical"]
            }
            question_lower = user_question.lower()
            relevant_tables = []
            for table_name, keywords in table_keywords.items():
                if table_name in available_tables:
                    for keyword in keywords:
                        if keyword in question_lower:
                            relevant_tables.append(table_name)
                            break
            if not relevant_tables:
                relevant_tables = list(available_tables)
            return relevant_tables

        import streamlit as st
        import pandas as pd
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
                data = RAGUtils.fetch_fabric_data(query, variables=variables, get_credential=get_credential)
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
                    # You may want to pass table_schemas as an argument for process_table_types
                    # df = RAGUtils.process_table_types(df, table, table_schemas)
                    # --- Sampling for large datasets ---
                    FILTERED_SAMPLE_SIZE = 500
                    if len(df) > FILTERED_SAMPLE_SIZE:
                        st.warning(f"Sampling {FILTERED_SAMPLE_SIZE} rows from {len(df)} for table {table} to improve performance.")
                        df = df.sample(n=FILTERED_SAMPLE_SIZE, random_state=42)
                    dfs[f'df{len(dfs)+1}'] = df
                    # summaries[f'df{len(dfs)}'] = RAGUtils.summarize_dataframe(df)
                progress.progress((i + 1) / len(tables_to_fetch), text=f"Fetched {i+1}/{len(tables_to_fetch)} tables")
        progress.empty()
        return dfs, summaries

    @staticmethod
    def clean_code(code):
        # Remove triple backticks and optional 'python' from start/end
        import re
        code = re.sub(r"^```(?:python)?\s*", "", code.strip(), flags=re.IGNORECASE)
        code = re.sub(r"```$", "", code.strip())
        return code.strip()

    @staticmethod
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

    @staticmethod
    def is_code_safe(code):
        """
        Smart pattern-based code safety validation.
        Instead of blocking individual keywords, recognizes safe code patterns.
        """
        import re
        
        # Remove comments and strings to avoid false positives
        code_no_comments = re.sub(r'#.*$', '', code, flags=re.MULTILINE)
        code_no_comments = re.sub(r'""".*?"""', '', code_no_comments, flags=re.DOTALL)
        code_no_comments = re.sub(r"'''.*?'''", '', code_no_comments, flags=re.DOTALL)
        code_no_comments = re.sub(r'"[^"]*"', '', code_no_comments)
        code_no_comments = re.sub(r"'[^']*'", '', code_no_comments)
        
        code_lower = code_no_comments.lower()
        
        # 🚀 SMART PATTERN-BASED VALIDATION SYSTEM
        
        # 1. SAFE MONGODB ARRAY SEARCH PATTERNS (Allow these completely)
        safe_mongo_patterns = [
            # Basic MongoDB array search
            r"df\d+\[df\d+\['[^']+'\].*?\.apply\(lambda\s+\w+:\s*'[^']+'\s+in\s+\w+.*?\)\]",
            # MongoDB array search with isinstance check
            r"df\d+\[df\d+\['[^']+'\].*?\.apply\(lambda\s+\w+:\s*'[^']+'\s+in\s+\w+\s+if\s+isinstance\(\w+,\s*list\)\s+else\s+false\)\]",
            # MongoDB array search with any conditional logic
            r"df\d+\[df\d+\['[^']+'\].*?\.apply\(lambda\s+\w+:\s*.*?\)\]",
            # DataFrame filtering with apply
            r"df\d+\[df\d+\['[^']+'\].*?\.apply\(.*?\)\]",
            # Column selection from filtered DataFrame
            r"df\d+\[.*?\]\['[^']+'\]",
        ]
        
        # Check if code matches any safe MongoDB pattern
        for pattern in safe_mongo_patterns:
            if re.search(pattern, code_lower):
                return True, None  # Safe MongoDB pattern - allow completely
        
        # 2. SAFE DATA ANALYSIS PATTERNS (Allow these completely)
        safe_analysis_patterns = [
            # Basic DataFrame operations
            r"df\d+\[.*?\]",
            r"df\d+\.groupby\(.*?\)",
            r"df\d+\.sort_values\(.*?\)",
            r"df\d+\.head\(.*?\)",
            r"df\d+\.tail\(.*?\)",
            r"df\d+\.describe\(\)",
            r"df\d+\.info\(\)",
            r"df\d+\.columns",
            r"df\d+\.shape",
            r"df\d+\.dtypes",
            
            # Data type conversions
            r"pd\.to_numeric\(.*?\)",
            r"pd\.to_datetime\(.*?\)",
            r"pd\.to_timedelta\(.*?\)",
            r"\.astype\(.*?\)",
            
            # String operations
            r"\.str\.",
            r"\.str\[.*?\]",
            r"\.str\.contains\(.*?\)",
            r"\.str\.replace\(.*?\)",
            r"\.str\.split\(.*?\)",
            
            # Mathematical operations
            r"\.mean\(\)",
            r"\.median\(\)",
            r"\.sum\(\)",
            r"\.count\(\)",
            r"\.min\(\)",
            r"\.max\(\)",
            r"\.std\(\)",
            r"\.var\(\)",
            
            # Aggregation
            r"\.agg\(.*?\)",
            r"\.aggregate\(.*?\)",
            
            # Plotting
            r"\.plot\(.*?\)",
            r"plt\.",
            r"\.figure\(.*?\)",
            r"\.subplot\(.*?\)",
            
            # Result assignment
            r"result\s*=",
            r"df_result\s*=",
            r"filtered_df\s*=",
        ]
        
        # Check if code matches any safe analysis pattern
        for pattern in safe_analysis_patterns:
            if re.search(pattern, code_lower):
                return True, None  # Safe analysis pattern - allow completely
        
        # 3. SAFE CONDITIONAL PATTERNS (Allow these completely)
        safe_conditional_patterns = [
            # Safe if-else patterns in data analysis context
            r"if\s+\w+\s+in\s+\w+",
            r"if\s+isinstance\(\w+,\s*\w+\)",
            r"if\s+\w+\s*==\s*'[^']*'",
            r"if\s+\w+\s*!=\s*'[^']*'",
            r"if\s+\w+\s*>\s*\d+",
            r"if\s+\w+\s*<\s*\d+",
            r"if\s+\w+\s*>=\s*\d+",
            r"if\s+\w+\s*<=\s*\d+",
            
            # Safe lambda patterns
            r"lambda\s+\w+:\s*\w+\s+in\s+\w+",
            r"lambda\s+\w+:\s*isinstance\(\w+,\s*\w+\)",
            r"lambda\s+\w+:\s*\w+\s*==\s*'[^']*'",
            r"lambda\s+\w+:\s*\w+\s*!=\s*'[^']*'",
            r"lambda\s+\w+:\s*\w+\s*>\s*\d+",
            r"lambda\s+\w+:\s*\w+\s*<\s*\d+",
        ]
        
        # Check if code matches any safe conditional pattern
        for pattern in safe_conditional_patterns:
            if re.search(pattern, code_lower):
                return True, None  # Safe conditional pattern - allow completely
        
        # 4. DANGEROUS PATTERNS (Block these completely)
        dangerous_patterns = [
            # Code execution
            r"\bexec\s*\(",
            r"\beval\s*\(",
            r"\bcompile\s*\(",
            r"\b__import__\s*\(",
            
            # File operations
            r"\bopen\s*\(",
            r"\bfile\s*\(",
            r"\binput\s*\(",
            r"\braw_input\s*\(",
            
            # System operations
            r"\bsystem\s*\(",
            r"\bsubprocess\.",
            r"\bos\.system\s*\(",
            
            # Network operations
            r"\burllib\.",
            r"\bhttplib\.",
            r"\bsocket\.",
            r"\brequests\.",
            
            # Serialization
            r"\bpickle\.",
            r"\bmarshal\.",
            
            # Reflection/metaprogramming
            r"\bgetattr\s*\(",
            r"\bsetattr\s*\(",
            r"\bdelattr\s*\(",
            r"\bhasattr\s*\(",
            r"\bglobals\s*\(",
            r"\blocals\s*\(",
            r"\bvars\s*\(",
            r"\bdir\s*\(",
            r"\bhelp\s*\(",
            r"\btype\s*\(",
            
            # Class definition (block to prevent code injection)
            r"\bclass\s+\w+",
            r"\bdef\s+\w+",
            
            # Control flow (block to prevent infinite loops)
            r"\bwhile\s+True:",
            r"\bfor\s+\w+\s+in\s+range\(\d+,\s*\d+\):",
            
            # Import statements (block to prevent module injection)
            r"\bimport\s+\w+",
            r"\bfrom\s+\w+\s+import",
        ]
        
        # Check if code contains any dangerous patterns
        for pattern in dangerous_patterns:
            if re.search(pattern, code_lower):
                # Extract the dangerous keyword for error message
                match = re.search(pattern, code_lower)
                if match:
                    dangerous_part = match.group(0)
                    # Extract the main keyword
                    for keyword in ['exec', 'eval', 'compile', 'open', 'file', 'input', 'system', 'subprocess', 'urllib', 'pickle', 'marshal', 'getattr', 'setattr', 'delattr', 'hasattr', 'globals', 'locals', 'vars', 'dir', 'help', 'type', 'class', 'def', 'while', 'for', 'import', 'from']:
                        if keyword in dangerous_part:
                            return False, keyword
        
        # 5. IF NO PATTERN MATCHES, CHECK FOR MIXED SAFE/DANGEROUS CODE
        # This catches edge cases where safe and dangerous code are mixed
        
        # Count safe vs dangerous elements
        safe_count = 0
        dangerous_count = 0
        
        # Count safe elements
        for pattern in safe_mongo_patterns + safe_analysis_patterns + safe_conditional_patterns:
            if re.search(pattern, code_lower):
                safe_count += 1
        
        # Count dangerous elements
        for pattern in dangerous_patterns:
            if re.search(pattern, code_lower):
                dangerous_count += 1
        
        # If mostly safe, allow it
        if safe_count > dangerous_count:
            return True, None
        
        # If mostly dangerous, block it
        if dangerous_count > safe_count:
            # Find the most dangerous pattern
            for pattern in dangerous_patterns:
                if re.search(pattern, code_lower):
                    match = re.search(pattern, code_lower)
                    if match:
                        dangerous_part = match.group(0)
                        for keyword in ['exec', 'eval', 'compile', 'open', 'file', 'input', 'system', 'subprocess', 'urllib', 'pickle', 'marshal', 'getattr', 'setattr', 'delattr', 'hasattr', 'globals', 'locals', 'vars', 'dir', 'help', 'type', 'class', 'def', 'while', 'for', 'import', 'from']:
                            if keyword in dangerous_part:
                                return False, keyword
        
        # Default: if we can't determine, allow it (safer to allow than block)
        return True, None 

    @staticmethod
    def run_deepseek_llm(prompt, max_new_tokens=512, temperature=0.2):
        """
        Run the locally finetuned Deepseek model for text generation.
        Loads the model and tokenizer from models/finetuned-deepseek-coder.
        Returns the generated text (first sequence).
        """
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
        model_dir = "models/finetuned-deepseek-coder"
        device = 0 if torch.cuda.is_available() else -1
        tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(model_dir, trust_remote_code=True)
        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            device=device,
        )
        outputs = pipe(
            prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
        return outputs[0]["generated_text"] if "generated_text" in outputs[0] else outputs[0]["text"] 