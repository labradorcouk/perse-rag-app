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
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import jwt
import time
from auth import init_auth_session_state
from config.queries import QUERIES
from config.table_schemas import TABLE_SCHEMAS
from config.safe_builtins import SAFE_BUILTINS
from utils.rag_utils import RAGUtils
from utils.qdrant_utils import QdrantIndex
from qdrant_client import models
import yaml

load_dotenv()
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
DEFAULT_MODEL = os.getenv('DEFAULT_MODEL', 'gpt-3.5-turbo')
MAX_TOKENS = int(os.getenv('MAX_TOKENS', '600'))
TEMPERATURE = float(os.getenv('TEMPERATURE', '0.2'))
client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

# --- Initialize authentication session state at the top ---
init_auth_session_state()

# --- Load RAG Tables Config ---
with open('config/rag_tables_config.yaml', 'r') as f:
    RAG_TABLES_CONFIG = yaml.safe_load(f)

TABLES_META = {t['name']: t for t in RAG_TABLES_CONFIG['tables']}
TABLE_DISPLAY_NAMES = {t['name']: t['display_name'] for t in RAG_TABLES_CONFIG['tables']}

def get_date_filter_column(table_name):
    """
    Safely get the date filter column for a table.
    
    Args:
        table_name: Name of the table
        
    Returns:
        date_filter_column if exists and not empty, None otherwise
    """
    if table_name not in TABLES_META:
        return None
    
    date_column = TABLES_META[table_name].get('date_filter_column')
    if date_column and date_column.strip():
        return date_column.strip()
    return None

def has_date_filtering(table_name):
    """
    Check if a table has date filtering configured.
    
    Args:
        table_name: Name of the table
        
    Returns:
        True if table has date filtering, False otherwise
    """
    return get_date_filter_column(table_name) is not None

# --- Example Queries (customize as needed) ---
# QUERIES = {
#     "EPC Non-Domestic Scotland": '''
#         query($first: Int!) {
#             epcNonDomesticScotlands(first: $first) {
#                 items {
#                     CURRENT_ENERGY_PERFORMANCE_BAND
#                     CURRENT_ENERGY_PERFORMANCE_RATING
#                     LODGEMENT_DATE
#                     PRIMARY_ENERGY_VALUE
#                     BUILDING_EMISSIONS
#                     FLOOR_AREA
#                     PROPERTY_TYPE
#                 }
#             }
#         }
#     ''',
#     # Add more queries as needed
# }

# --- Table schemas for column types ---
# TABLE_SCHEMAS = {
#     "EPC Non-Domestic Scotland": {
#         "numeric": [
#             "CURRENT_ENERGY_PERFORMANCE_RATING",
#             "PRIMARY_ENERGY_VALUE",
#             "BUILDING_EMISSIONS",
#             "FLOOR_AREA"
#         ],
#         "categorical": [
#             "CURRENT_ENERGY_PERFORMANCE_BAND",
#             "PROPERTY_TYPE"
#         ],
#         "datetime": [
#             "LODGEMENT_DATE"
#         ]
#     },
#     # Add more tables here as needed
# }

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
    df = fetch_embeddings(_engine, date_range)
    # Remove 'embedding_np' if present (should not be, but for safety)
    if 'embedding_np' in df.columns:
        df = df.drop(columns=['embedding_np'])
    return df

def get_embeddings_with_np(emb_df):
    if 'embedding_np' not in emb_df.columns:
        emb_df = emb_df.copy()
        emb_df['embedding_np'] = emb_df['embedding'].apply(RAGUtils.parse_embedding)
    return emb_df

@st.cache_resource
def get_faiss_index(emb_df):
    if 'embedding_np' not in emb_df.columns:
        raise ValueError("DataFrame must have 'embedding_np' column.")
    embeddings_matrix = np.vstack(emb_df['embedding_np'].values).astype('float32')
    index = faiss.IndexFlatL2(embeddings_matrix.shape[1])
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

# Patch fetch_embeddings to NOT add 'embedding_np'
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
    top_indices, _ = RAGUtils.vector_search(q_emb, embeddings_matrix, top_n=top_n)
    top_keys = emb_df.iloc[top_indices][['POSTCODE', 'PROPERTY_TYPE', 'MAIN_HEATING_FUEL']].values.tolist()
    df_subset = RAGUtils.fetch_raw_data(engine, top_keys, date_range=(start_date, end_date))
    return df_subset

def detect_column_type_and_bounds(df, column_name):
    """Dynamically detect column type and appropriate bounds based on data analysis"""
    if column_name not in df.columns:
        return None, None, None
    
    # Get sample of non-null values
    sample_data = df[column_name].dropna().astype(str).head(100)
    
    # Try to convert to numeric
    try:
        numeric_data = pd.to_numeric(sample_data, errors='coerce')
        numeric_data = numeric_data.dropna()
        
        if len(numeric_data) == 0:
            return 'string', None, None
        
        # Analyze the data to determine type and bounds
        min_val = numeric_data.min()
        max_val = numeric_data.max()
        mean_val = numeric_data.mean()
        std_val = numeric_data.std()
        
        # Detect column type based on patterns
        column_lower = column_name.lower()
        
        # Energy ratings (typically 0-100)
        if any(keyword in column_lower for keyword in ['rating', 'efficiency', 'performance']) and 'energy' in column_lower:
            return 'energy_rating', 0, 100
        
        # Percentages (0-100)
        elif any(keyword in column_lower for keyword in ['percentage', 'percent', 'ratio']) or max_val <= 100:
            return 'percentage', 0, 100
        
        # Floor areas (typically 1-100,000 m²)
        elif any(keyword in column_lower for keyword in ['floor', 'area', 'size']) and mean_val > 50:
            return 'floor_area', 1, 100000
        
        # Energy values (typically 0-10,000)
        elif any(keyword in column_lower for keyword in ['energy', 'primary', 'consumption']) and mean_val > 0:
            return 'energy_value', 0, 10000
        
        # Costs (typically 0-10,000)
        elif any(keyword in column_lower for keyword in ['cost', 'price', 'expense']) and mean_val > 0:
            return 'cost', 0, 10000
        
        # Emissions (typically 0-10,000)
        elif any(keyword in column_lower for keyword in ['emission', 'co2', 'carbon']) and mean_val > 0:
            return 'emission', 0, 10000
        
        # Counts (typically 0-1000)
        elif any(keyword in column_lower for keyword in ['count', 'number', 'quantity']) and mean_val < 1000:
            return 'count', 0, 1000
        
        # Generic numeric with statistical bounds
        else:
            # Use statistical bounds (mean ± 3*std)
            lower_bound = max(0, mean_val - 3 * std_val)
            upper_bound = mean_val + 3 * std_val
            return 'generic_numeric', lower_bound, upper_bound
            
    except Exception:
        return 'string', None, None

def clean_numeric_column_dynamic(df, column_name):
    """Dynamically clean any numeric column with automatic type detection"""
    if column_name not in df.columns:
        return df
    
    # Create a copy to avoid modifying original
    df_clean = df.copy()
    
    # Detect column type and bounds
    col_type, min_val, max_val = detect_column_type_and_bounds(df, column_name)
    
    if col_type == 'string':
        return df_clean  # Skip string columns
    
    # Convert to numeric with coerce
    df_clean[column_name] = pd.to_numeric(df_clean[column_name], errors='coerce')
    
    # Remove NaN values
    df_clean = df_clean[df_clean[column_name].notna()]
    
    # Apply bounds if detected
    if min_val is not None:
        df_clean = df_clean[df_clean[column_name] >= min_val]
    if max_val is not None:
        df_clean = df_clean[df_clean[column_name] <= max_val]
    
    return df_clean

def detect_and_clean_dataframe_dynamic(df):
    """Dynamically detect and clean all columns in a DataFrame using pattern recognition"""
    if df.empty:
        return df
    
    df_clean = df.copy()
    
    # Analyze each column
    for col in df_clean.columns:
        # Skip non-numeric columns (dates, strings, etc.)
        sample_values = df_clean[col].dropna().astype(str).head(10)
        
        # Check if column contains numeric data
        has_numeric_data = any(
            any(char.isdigit() for char in str(val)) 
            for val in sample_values if pd.notna(val)
        )
        
        if has_numeric_data:
            # Apply dynamic cleaning
            df_clean = clean_numeric_column_dynamic(df_clean, col)
    
    return df_clean

def robust_groupby_analysis_dynamic(df, group_column, value_column, agg_funcs=['mean', 'count', 'std']):
    """Robust groupby analysis with dynamic data cleaning"""
    if df.empty:
        return pd.DataFrame()
    
    # Clean the DataFrame dynamically
    df_clean = detect_and_clean_dataframe_dynamic(df)
    
    if df_clean.empty:
        return pd.DataFrame()
    
    # Ensure required columns exist
    if group_column not in df_clean.columns or value_column not in df_clean.columns:
        return pd.DataFrame()
    
    # Perform groupby analysis
    try:
        result = df_clean.groupby(group_column)[value_column].agg(agg_funcs).reset_index()
        return result
    except Exception as e:
        print(f"Error in groupby analysis: {e}")
        return pd.DataFrame()

def compare_datasets_analysis_dynamic(df1, dataset_identifiers=None):
    """Dynamic comparison of multiple datasets with automatic column detection and cleaning"""
    if df1.empty:
        return pd.DataFrame(), None
    
    # Clean the main DataFrame dynamically
    df_clean = detect_and_clean_dataframe_dynamic(df1)
    
    if df_clean.empty:
        return pd.DataFrame(), None
    
    # Auto-detect dataset types if not provided
    if dataset_identifiers is None:
        # Use pattern matching to identify dataset types
        domestic_patterns = ['total_floor_area', 'current_energy_rating', 'main_fuel', 'domestic']
        non_domestic_patterns = ['FLOOR_AREA', 'CURRENT_ENERGY_PERFORMANCE_BAND', 'MAIN_HEATING_FUEL', 'non_domestic']
        
        dataset_identifiers = {
            'domestic': domestic_patterns,
            'non_domestic': non_domestic_patterns
        }
    
    results = []
    
    # Analyze each dataset type
    for dataset_type, identifier_patterns in dataset_identifiers.items():
        # Find columns that match the patterns
        matching_cols = []
        for pattern in identifier_patterns:
            matching_cols.extend([col for col in df_clean.columns if pattern.lower() in col.lower()])
        
        if matching_cols:
            # Filter data for this dataset type
            mask = df_clean[matching_cols].notna().any(axis=1)
            dataset_data = df_clean[mask]
            
            if len(dataset_data) > 0:
                # Find analysis columns dynamically
                floor_area_cols = [col for col in dataset_data.columns 
                                 if any(keyword in col.lower() for keyword in ['floor', 'area', 'size'])]
                energy_rating_cols = [col for col in dataset_data.columns 
                                    if any(keyword in col.lower() for keyword in ['energy', 'rating', 'efficiency'])]
                
                if floor_area_cols and energy_rating_cols:
                    floor_area_col = floor_area_cols[0]
                    energy_rating_col = energy_rating_cols[0]
                    
                    # Perform analysis with dynamic cleaning
                    analysis = robust_groupby_analysis_dynamic(dataset_data, energy_rating_col, floor_area_col)
                    if not analysis.empty:
                        analysis['dataset_type'] = dataset_type
                        results.append(analysis)
    
    if results:
        combined_results = pd.concat(results, ignore_index=True)
        
        # Create visualization
        fig = create_comparison_visualization(combined_results)
        
        return combined_results, fig
    else:
        return pd.DataFrame(), None

def analyze_dataframe_structure(df):
    """Analyze DataFrame structure and provide insights about data types and cleaning needs"""
    analysis = {
        'total_columns': len(df.columns),
        'total_rows': len(df),
        'column_types': {},
        'cleaning_recommendations': []
    }
    
    for col in df.columns:
        col_analysis = {
            'dtype': str(df[col].dtype),
            'null_count': df[col].isnull().sum(),
            'unique_count': df[col].nunique(),
            'sample_values': df[col].dropna().head(3).tolist()
        }
        
        # Detect if column is numeric
        sample_values = df[col].dropna().astype(str).head(10)
        has_numeric_data = any(
            any(char.isdigit() for char in str(val)) 
            for val in sample_values if pd.notna(val)
        )
        
        if has_numeric_data:
            col_type, min_val, max_val = detect_column_type_and_bounds(df, col)
            col_analysis['detected_type'] = col_type
            col_analysis['suggested_bounds'] = (min_val, max_val)
            
            if col_type != 'string':
                analysis['cleaning_recommendations'].append({
                    'column': col,
                    'type': col_type,
                    'bounds': (min_val, max_val),
                    'action': f'Clean as {col_type} with bounds {min_val} to {max_val}'
                })
        
        analysis['column_types'][col] = col_analysis
    
    return analysis

def create_comparison_visualization(results_df):
    """Create comprehensive visualization for dataset comparison"""
    if results_df.empty:
        return None
    
    # Create subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Plot 1: Average values by category
    if 'mean' in results_df.columns:
        pivot_mean = results_df.pivot(index=results_df.columns[0], columns='dataset_type', values='mean')
        pivot_mean.plot(kind='bar', ax=axes[0,0], title='Average Values by Category')
        axes[0,0].set_xlabel('Category')
        axes[0,0].set_ylabel('Average Value')
        axes[0,0].legend()
        axes[0,0].tick_params(axis='x', rotation=45)
    
    # Plot 2: Count of records by category
    if 'count' in results_df.columns:
        pivot_count = results_df.pivot(index=results_df.columns[0], columns='dataset_type', values='count')
        pivot_count.plot(kind='bar', ax=axes[0,1], title='Number of Records by Category')
        axes[0,1].set_xlabel('Category')
        axes[0,1].set_ylabel('Count')
        axes[0,1].legend()
        axes[0,1].tick_params(axis='x', rotation=45)
    
    # Plot 3: Standard deviation by category
    if 'std' in results_df.columns:
        pivot_std = results_df.pivot(index=results_df.columns[0], columns='dataset_type', values='std')
        pivot_std.plot(kind='bar', ax=axes[1,0], title='Standard Deviation by Category')
        axes[1,0].set_xlabel('Category')
        axes[1,0].set_ylabel('Standard Deviation')
        axes[1,0].legend()
        axes[1,0].tick_params(axis='x', rotation=45)
    
    # Plot 4: Summary statistics
    axes[1,1].axis('off')
    summary_text = f"Dataset Summary:\n"
    summary_text += f"Total records: {len(results_df)}\n"
    summary_text += f"Categories: {results_df.iloc[:, 0].nunique()}\n"
    summary_text += f"Dataset types: {results_df['dataset_type'].nunique()}"
    axes[1,1].text(0.1, 0.5, summary_text, transform=axes[1,1].transAxes, fontsize=12, verticalalignment='center')
    
    plt.tight_layout()
    return fig

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
    
    try:
        with open(log_file, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            if f.tell() == 0:
                writer.writerow(["timestamp", "is_retry", "question", "reasoning", "code", "row_count", "col_count", "columns"])
            writer.writerow([now, is_retry, question, reasoning, code, row_count, col_count, columns])
    except Exception as e:
        # Silently fail logging to avoid disrupting the main flow
        print(f"Warning: Could not write to audit log: {e}")
        pass

@st.cache_resource
def get_credential():
    app = InteractiveBrowserCredential()
    scp = 'https://analysis.windows.net/powerbi/api/user_impersonation'
    return app, scp

# --- Main Authentication Flow ---
from auth import main_auth
main_auth(get_credential)

# --- Only after authentication, set up the UI ---
st.set_page_config(page_title="Fabric RAG QA", layout="wide")
st.title("Fabric RAG (Retrieval-Augmented Generation) QA App")

# --- Only show sidebar and navigation after authentication ---
if st.session_state.authenticated:
    # --- Sidebar Configuration (only after authentication) ---
    with st.sidebar:
        st.header("Embedding Model")
        model_options = {
            "all-MiniLM-L6-v2": "all-MiniLM-L6-v2",
            "all-mpnet-base-v2": "all-mpnet-base-v2",
            "multi-qa-MiniLM-L6-cos-v1": "multi-qa-MiniLM-L6-cos-v1"
        }
        
        llm_provider = st.sidebar.radio(
            "Select LLM Provider",
            ["OpenAI GPT-3.5/4", "Local Deepseek LLM"],
            index=0,
            key='sidebar_llm_select'
        )
        
        selected_model_name = st.sidebar.selectbox(
            "Select embedding model",
            list(model_options.keys()),
            key='sidebar_model_select'
        )
        st.session_state['selected_model_name'] = selected_model_name
        selected_model_path = model_options[selected_model_name]

        # Table selection
        selected_tables = st.sidebar.multiselect(
            "Select Table(s)",
            list(TABLES_META.keys()),
            default=list(TABLES_META.keys())[:1],
            key='sidebar_table_select'
        )
        st.session_state['selected_tables'] = selected_tables

# --- Main Navigation as Tabs ---
tabs = st.tabs(["RAG QA", "SQL Editor"])

with tabs[0]:
    st.header("RAG QA")
    # Always define dfs before using it
    dfs = st.session_state.get('fabric_dfs', {})
    # Embedding model selection in main tab (synchronized with sidebar)
    selected_model_name_tab = st.selectbox(
        "Select embedding model",
        list(model_options.keys()),
        index=list(model_options.keys()).index(st.session_state['selected_model_name']) if st.session_state['selected_model_name'] in model_options else 0,
        key='main_tab_model_select'
    )
    st.session_state['selected_model_name'] = selected_model_name_tab
    selected_model_path_tab = model_options[selected_model_name_tab]
    # Ensure selected_tables is defined
    # selected_tables = st.multiselect("Select Table(s)/Query(ies)", list(QUERIES.keys()), default=list(QUERIES.keys())[:1])
    # Ensure date range and batch_size are defined
    min_date = datetime.date(2000, 1, 1)
    max_date = datetime.date.today()
        
    # Check if any selected tables have date filtering
    selected_tables_with_date_filter = [table for table in selected_tables if has_date_filtering(table)]
        
    if selected_tables_with_date_filter:
        # Show date range input for tables that support it
        date_range = st.date_input("Select date range for date-filtered tables", [min_date, max_date])
        if len(date_range) == 2:
            start_date, end_date = date_range
        else:
            start_date, end_date = min_date, max_date
            
        # Show info about which tables support date filtering
        st.info(f"Date filtering available for: {', '.join([TABLE_DISPLAY_NAMES.get(t, t) for t in selected_tables_with_date_filter])}")
    else:
        # No date filtering available
        start_date, end_date = min_date, max_date
        st.info("No date filtering configured for selected tables - all available data will be fetched")
        
    batch_size = st.number_input("Records to fetch per table", min_value=100, max_value=5000, value=1000, step=100)
        
    vector_search_engine = st.radio(
        "Select Vector Search Engine",
        ("FAISS", "Qdrant", "MongoDB"),
        index=0,
        key="vector_search_engine_select",
        help="Choose the backend for semantic search."
    )

    # Place all RAG QA logic here (do not include SQL Editor logic)
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
            is_viz = RAGUtils.is_visualization_request(user_question)
            df1_context = None # Initialize df1_context
            
            # Debug: Check which vector search engine is selected
            st.info(f"Selected vector search engine: {vector_search_engine}")
            
            try:
                model = RAGUtils.get_embedding_model(selected_model_path_tab)
                
                # Adjust top_n based on vector search engine
                if vector_search_engine == "MongoDB":
                    top_n = 100  # MongoDB Atlas has stricter limits
                else:
                    top_n = 5000  # FAISS and Qdrant can handle more

                # Helper: join DataFrames on config relationships
                def join_dataframes(dfs, relationships, backend):
                    if len(dfs) == 1:
                        return list(dfs.values())[0]
                    
                    # If no relationships defined, just combine all tables
                    if not relationships:
                        st.info(f"No relationships defined in config. Combining {len(dfs)} tables without joins.")
                        # Add a source column to identify which table each row came from
                        combined_dfs = []
                        for name, df in dfs.items():
                            # Create a copy to avoid modifying the original
                            df_copy = df.copy()
                            df_copy['_source_table'] = name
                            # Reset index to avoid conflicts
                            df_copy = df_copy.reset_index(drop=True)
                            combined_dfs.append(df_copy)
                        
                        # Concatenate all DataFrames with proper index handling
                        try:
                            # Debug: Show DataFrame info before concatenation
                            for i, df in enumerate(combined_dfs):
                                st.info(f"DataFrame {i+1}: {df.shape[0]} rows, {df.shape[1]} columns, index: {type(df.index)}")
                            
                            # Check for duplicate column names and add suffixes if needed
                            all_columns = []
                            for df in combined_dfs:
                                all_columns.extend(df.columns.tolist())
                            
                            # Find duplicate columns (excluding _source_table)
                            from collections import Counter
                            column_counts = Counter(all_columns)
                            duplicate_columns = [col for col, count in column_counts.items() if count > 1 and col != '_source_table']
                            
                            if duplicate_columns:
                                st.warning(f"Found duplicate columns: {duplicate_columns}. Adding table suffixes to distinguish them.")
                                # Add suffixes to duplicate columns in each DataFrame
                                for i, df in enumerate(combined_dfs):
                                    df_copy = df.copy()
                                    # Add suffix to duplicate columns (except _source_table)
                                    for col in df_copy.columns:
                                        if col in duplicate_columns and col != '_source_table':
                                            new_col_name = f"{col}_{df_copy['_source_table'].iloc[0] if len(df_copy) > 0 else f'table_{i}'}"
                                            df_copy = df_copy.rename(columns={col: new_col_name})
                                    combined_dfs[i] = df_copy
                            
                            combined_df = pd.concat(combined_dfs, ignore_index=True, sort=False)
                            st.success(f"Combined {len(combined_df)} rows from {len(dfs)} tables")
                            return combined_df
                        except Exception as e:
                            st.error(f"Error combining DataFrames: {e}")
                            # Fallback: return the first DataFrame if concatenation fails
                            st.warning("Falling back to first DataFrame due to concatenation error")
                            return list(dfs.values())[0]
                    
                    # Process relationships to join tables
                    joined_dfs = {}
                    processed_tables = set()
                    
                    for relationship in relationships:
                        left = relationship['left_table']
                        right = relationship['right_table']
                        left_keys = relationship['left_keys']
                        right_keys = relationship['right_keys']
                        
                        # Check if both tables are available
                        if left not in dfs or right not in dfs:
                            st.warning(f"Cannot join {left} and {right}: one or both tables not available")
                            continue
                        
                        df_left = dfs[left]
                        df_right = dfs[right]
                        
                        # Check if join keys exist in both tables
                        missing_left = [key for key in left_keys if key not in df_left.columns]
                        missing_right = [key for key in right_keys if key not in df_right.columns]
                        
                        if missing_left or missing_right:
                            st.warning(f"Cannot join {left} and {right}: missing columns. Left missing: {missing_left}, Right missing: {missing_right}")
                            st.warning(f"Available columns in {left}: {list(df_left.columns)}")
                            st.warning(f"Available columns in {right}: {list(df_right.columns)}")
                            # Keep tables separate if join fails
                            if left not in processed_tables:
                                joined_dfs[left] = df_left
                                processed_tables.add(left)
                            if right not in processed_tables:
                                joined_dfs[right] = df_right
                                processed_tables.add(right)
                            continue
                        
                        try:
                            joined = pd.merge(df_left, df_right, left_on=left_keys, right_on=right_keys, suffixes=(f'_{left}', f'_{right}'))
                            st.success(f"Successfully joined {left} and {right} on keys: {left_keys}")
                            # Store the joined result with a combined name
                            joined_name = f"{left}_{right}_joined"
                            joined_dfs[joined_name] = joined
                            processed_tables.add(left)
                            processed_tables.add(right)
                        except Exception as e:
                            st.error(f"Error joining {left} and {right}: {e}")
                            # Keep tables separate if join fails
                            if left not in processed_tables:
                                joined_dfs[left] = df_left
                                processed_tables.add(left)
                            if right not in processed_tables:
                                joined_dfs[right] = df_right
                                processed_tables.add(right)
                    
                    # Add any remaining tables that weren't processed
                    for table_name, df in dfs.items():
                        if table_name not in processed_tables:
                            joined_dfs[table_name] = df
                    
                    # Ensure we have at least one DataFrame
                    if not joined_dfs:
                        st.error("No DataFrames available for joining")
                        return pd.DataFrame()
                    
                    # If we have multiple DataFrames, concatenate them vertically
                    if len(joined_dfs) > 1:
                        st.info(f"Combining {len(joined_dfs)} tables without joins: {list(joined_dfs.keys())}")
                        # Add a source column to identify which table each row came from
                        combined_dfs = []
                        for name, df in joined_dfs.items():
                            # Create a copy to avoid modifying the original
                            df_copy = df.copy()
                            df_copy['_source_table'] = name
                            # Reset index to avoid conflicts
                            df_copy = df_copy.reset_index(drop=True)
                            combined_dfs.append(df_copy)
                        
                        # Concatenate all DataFrames with proper index handling
                        try:
                            # Debug: Show DataFrame info before concatenation
                            for i, df in enumerate(combined_dfs):
                                st.info(f"DataFrame {i+1}: {df.shape[0]} rows, {df.shape[1]} columns, index: {type(df.index)}")
                            
                            # Check for duplicate column names and add suffixes if needed
                            all_columns = []
                            for df in combined_dfs:
                                all_columns.extend(df.columns.tolist())
                            
                            # Find duplicate columns (excluding _source_table)
                            from collections import Counter
                            column_counts = Counter(all_columns)
                            duplicate_columns = [col for col, count in column_counts.items() if count > 1 and col != '_source_table']
                            
                            if duplicate_columns:
                                st.warning(f"Found duplicate columns: {duplicate_columns}. Adding table suffixes to distinguish them.")
                                # Add suffixes to duplicate columns in each DataFrame
                                for i, df in enumerate(combined_dfs):
                                    df_copy = df.copy()
                                    # Add suffix to duplicate columns (except _source_table)
                                    for col in df_copy.columns:
                                        if col in duplicate_columns and col != '_source_table':
                                            new_col_name = f"{col}_{df_copy['_source_table'].iloc[0] if len(df_copy) > 0 else f'table_{i}'}"
                                            df_copy = df_copy.rename(columns={col: new_col_name})
                                    combined_dfs[i] = df_copy
                            
                            combined_df = pd.concat(combined_dfs, ignore_index=True, sort=False)
                            st.success(f"Combined {len(combined_df)} rows from {len(joined_dfs)} tables")
                            return combined_df
                        except Exception as e:
                            st.error(f"Error combining DataFrames: {e}")
                            # Fallback: return the first DataFrame if concatenation fails
                            st.warning("Falling back to first DataFrame due to concatenation error")
                            return list(joined_dfs.values())[0]
                    else:
                        # Return the single DataFrame
                        return list(joined_dfs.values())[0]

                if vector_search_engine == "FAISS":
                    st.info("Executing FAISS path...")
                    status_placeholder.info("Step 2: Connecting to Fabric SQL endpoint (FAISS path)...")
                    engine = RAGUtils.get_fabric_engine()
                    status_placeholder.info("Step 3: Fetching and caching embeddings (FAISS path)...")
                    dfs = {}
                    for table_name in selected_tables:
                        table_meta = TABLES_META[table_name]
                        emb_df = get_cached_embeddings(engine, date_range=(start_date, end_date))
                        emb_df = get_embeddings_with_np(emb_df)
                        status_placeholder.info(f"Building FAISS index for {table_meta['display_name']}...")
                        index = get_faiss_index(emb_df)
                        q_emb = RAGUtils.embed_question(user_question, model)
                        _, I = index.search(q_emb.reshape(1, -1).astype('float32'), top_n)
                        top_indices = I[0]
                        top_keys = emb_df.iloc[top_indices][table_meta['faiss_columns']].values.tolist()
                        status_placeholder.info(f"Fetching raw data for top semantic matches ({table_meta['display_name']})...")
                        df_context = RAGUtils.fetch_raw_data(engine, top_keys, date_range=(start_date, end_date))
                        dfs[table_name] = df_context
                    df1_context = join_dataframes(dfs, RAG_TABLES_CONFIG.get('relationships', []), backend='faiss')
                    context_info = f"Using top {min(len(df1_context), top_n)} semantically relevant rows for context in df1 (via FAISS, {', '.join(selected_tables)})."
                    st.info(context_info)

                elif vector_search_engine == "Qdrant":
                    st.info("Executing Qdrant path...")
                    status_placeholder.info("Step 2: Initializing Qdrant and performing search...")
                    dfs = {}
                    for table_name in selected_tables:
                        table_meta = TABLES_META[table_name]
                        qdrant_index = QdrantIndex(
                            collection_name=table_meta['collection'],
                            embedding_model=model
                        )
                        search_results = qdrant_index.search(
                            user_question,
                            limit=top_n,
                            with_payload=True
                        )
                        if not search_results:
                            st.warning(f"No relevant documents found in Qdrant for {table_meta['display_name']}.")
                            continue
                        status_placeholder.info(f"Building context from Qdrant search results for {table_meta['display_name']}...")
                        payloads = [result.payload for result in search_results]
                        df_context = pd.DataFrame(payloads)
                        
                        # Debug: Show available columns
                        st.info(f"Available columns in {table_meta['display_name']}: {list(df_context.columns)}")
                        
                        # Only keep columns defined for Qdrant for this table
                        available_qdrant_columns = [col for col in table_meta['qdrant_columns'] if col in df_context.columns]
                        if available_qdrant_columns:
                            df_context = df_context[available_qdrant_columns]
                        else:
                            st.warning(f"No matching columns found for {table_meta['display_name']}. Using all available columns.")
                        
                        dfs[table_name] = df_context
                    
                    # Debug: Show what we're trying to join
                    st.info(f"DataFrames to join: {list(dfs.keys())}")
                    for name, df in dfs.items():
                        st.info(f"{name} columns: {list(df.columns)}")
                    
                    df1_context = join_dataframes(dfs, RAG_TABLES_CONFIG.get('relationships', []), backend='qdrant')
                    if df1_context is not None and not df1_context.empty:
                        context_info = f"Using top {len(df1_context)} semantically relevant rows for context in df1 (via Qdrant, {', '.join(selected_tables)})."
                        st.info(context_info)
                    else:
                        st.warning("No context data available from Qdrant search. Will proceed with auto-fetch.")
                        df1_context = None

                elif vector_search_engine == "MongoDB":
                    st.info("Executing MongoDB path...")
                    status_placeholder.info("Step 2: Initializing MongoDB Atlas and performing search...")
                    dfs = {}
                    
                    # Debug: Check environment variables
                    st.info(f"🔍 Debug: MONGODB_URI = {os.getenv('MONGODB_URI', 'Not set')}")
                    st.info(f"🔍 Debug: MONGODB_DB_NAME = {os.getenv('MONGODB_DB_NAME', 'perse-data-network')}")
                    
                    try:
                        # Import MongoDBIndex only when needed
                        st.info("🔍 Debug: Importing MongoDBIndex...")
                        from utils.mongodb_utils import MongoDBIndex
                        from utils.mongodb_schema_manager import MongoDBSchemaManager
                        st.success("✅ MongoDBIndex import successful")
                    except Exception as e:
                        st.error(f"❌ MongoDBIndex import failed: {str(e)}")
                        st.error(f"Error type: {type(e).__name__}")
                        st.stop()
                    
                    # Initialize MongoDB Schema Manager for enhanced contextual awareness
                    schema_manager = MongoDBSchemaManager()
                    st.info("✅ MongoDB Schema Manager initialized with contextual awareness")
                    
                    # Process each selected table
                    for table_name in selected_tables:
                        table_meta = TABLES_META[table_name]
                        st.info(f"🔍 Debug: Processing table {table_name}")
                        st.info(f"🔍 Debug: Table meta = {table_meta}")
                        
                        try:
                            # Use table name as collection name for MongoDB (as per config)
                            collection_name = table_meta.get('collection', table_name)
                            st.info(f"Connecting to MongoDB collection: {collection_name}")
                            
                            # Test MongoDB connection
                            st.info("🔍 Debug: Testing MongoDB connection...")
                            mongodb_index = MongoDBIndex(
                                collection_name=collection_name,
                                embedding_model=model
                            )
                            
                            # Test connection
                            if mongodb_index.test_connection():
                                st.success(f"✅ MongoDB connection successful for collection: {collection_name}")
                            else:
                                st.error(f"❌ MongoDB connection failed for collection: {collection_name}")
                                st.error("Please check your MongoDB URI and network connectivity.")
                                st.error("You can use FAISS or Qdrant instead by changing the vector search engine.")
                                continue
                            
                            # Get collection info
                            try:
                                collection_info = mongodb_index.get_collection_info()
                                st.info(f"Collection info: {collection_info}")
                            except Exception as e:
                                st.error(f"❌ Failed to get collection info: {str(e)}")
                                continue
                            
                            # 🚀 ENHANCED: Query Enhancement and Contextual Awareness
                            st.info("🔍 Debug: Enhancing user query with business context...")
                            enhanced_query_info = schema_manager.enhance_user_query(collection_name, user_question)
                            
                            # Display query enhancement details
                            if enhanced_query_info['original_query'] != enhanced_query_info['enhanced_query']:
                                st.success(f"✅ Query enhanced for better search results!")
                                st.info(f"Original: '{enhanced_query_info['original_query']}'")
                                st.info(f"Enhanced: '{enhanced_query_info['enhanced_query']}'")
                                st.info(f"Business domain: {enhanced_query_info['business_domain']}")
                                st.info(f"Purpose: {enhanced_query_info['purpose']}")
                            
                            if enhanced_query_info['detected_intent']:
                                st.info(f"🎯 Detected intent: {', '.join(enhanced_query_info['detected_intent'])}")
                            
                            if enhanced_query_info['semantic_expansions']:
                                st.info(f"🔍 Semantic expansions: {', '.join(enhanced_query_info['semantic_expansions'])}")
                            
                            # Use enhanced query for search
                            search_query = enhanced_query_info['enhanced_query']
                            st.info(f"🔍 Debug: Performing enhanced search with: '{search_query}'")
                            
                            try:
                                search_results = mongodb_index.search(
                                    search_query,
                                    limit=top_n,
                                    score_threshold=0.01
                                )
                                st.success(f"✅ Search successful: {len(search_results)} results")
                            except Exception as e:
                                st.error(f"❌ Search failed: {str(e)}")
                                continue
                            
                            if not search_results:
                                st.warning(f"No relevant documents found in MongoDB for {table_meta['display_name']}.")
                                continue
                            
                            status_placeholder.info(f"Building context from MongoDB search results for {table_meta['display_name']}...")
                            
                            # Get raw data from MongoDB for context building
                            st.info("🔍 Debug: Fetching raw data from MongoDB...")
                            try:
                                # Get raw data from MongoDB collection
                                raw_data = mongodb_index.get_raw_data(limit=5000)
                                st.success(f"✅ Raw data fetched: {len(raw_data)} documents")
                                
                                if not raw_data.empty:
                                    # Debug: Show available columns
                                    st.info(f"Available columns in {table_meta['display_name']}: {list(raw_data.columns)}")
                                    
                                    # 🚀 ENHANCED: Use MongoDB Schema Manager to optimize data for context
                                    st.info("🔍 Debug: Optimizing data using enhanced schema configuration...")
                                    try:
                                        # Optimize DataFrame using enhanced schema configuration
                                        raw_data_optimized = schema_manager.optimize_dataframe_for_context(
                                            raw_data, 
                                            collection_name
                                        )
                                        
                                        # Get optimization details from enhanced schema
                                        essential_cols = schema_manager.get_essential_columns(collection_name)
                                        exclude_cols = schema_manager.get_exclude_columns(collection_name)
                                        max_rows = schema_manager.get_max_context_rows(collection_name)
                                        
                                        # Get business context information
                                        business_context = schema_manager.get_business_context(collection_name)
                                        business_keywords = schema_manager.get_business_keywords(collection_name)
                                        semantic_boost_fields = schema_manager.get_semantic_boost_fields(collection_name)
                                        
                                        st.success(f"✅ Data optimized using enhanced schema: {len(raw_data_optimized)} rows x {len(raw_data_optimized.columns)} columns")
                                        st.info(f"Schema settings: Essential columns: {essential_cols}, Exclude: {exclude_cols}, Max rows: {max_rows}")
                                        st.info(f"Business context: Domain: {business_context.get('domain', 'Unknown')}")
                                        st.info(f"Business keywords: {business_keywords}")
                                        st.info(f"Semantic boost fields: {semantic_boost_fields}")
                                        
                                        # Store the optimized data DataFrame
                                        dfs[table_name] = raw_data_optimized
                                        st.success(f"✅ Successfully processed {table_meta['display_name']}")
                                        
                                    except Exception as schema_error:
                                        st.warning(f"Enhanced schema optimization failed, using fallback optimization: {str(schema_error)}")
                                        # Fallback to basic optimization if schema fails
                                        if len(raw_data) > 10:
                                            raw_data_fallback = raw_data.sample(n=10, random_state=42)
                                        else:
                                            raw_data_fallback = raw_data.copy()
                                        
                                        # Keep only first 2 columns as fallback
                                        if len(raw_data_fallback.columns) > 2:
                                            raw_data_fallback = raw_data_fallback.iloc[:, :2]
                                        
                                        dfs[table_name] = raw_data_fallback
                                        st.success(f"✅ Successfully processed {table_meta['display_name']} (fallback mode)")
                                    
                                else:
                                    st.warning(f"No raw data found in MongoDB collection: {collection_name}")
                                    continue
                                    
                            except Exception as e:
                                st.error(f"❌ Failed to fetch raw data from MongoDB: {str(e)}")
                                diagnostics_logger.log_error(f"MongoDB raw data fetch failed for {table_name}: {str(e)}")
                                continue
                            
                            # Build context from search results for additional context
                            documents = []
                            for result in search_results:
                                # Handle MongoDB document format directly instead of trying to parse as JSON
                                try:
                                    # The payload is already a string representation of the MongoDB document
                                    # We don't need to parse it as JSON since it's not valid JSON format
                                    doc_data = {
                                        'payload': result.get('payload', ''),
                                        'score': result.get('score', 0),
                                        'metadata': result.get('metadata', {})
                                    }
                                    documents.append(doc_data)
                                except Exception as e:
                                    st.warning(f"Could not process search result document: {e}")
                                    continue
                            
                            if documents:
                                search_df = pd.DataFrame(documents)
                                st.info(f"Search results DataFrame shape: {search_df.shape}")
                                
                                # Combine raw data with search results for better context
                                if not raw_data.empty:
                                    # Use raw data as primary source, search results as supplementary
                                    st.info(f"Using raw data from MongoDB collection: {collection_name}")
                                else:
                                    # Fallback to search results if no raw data
                                    dfs[table_name] = search_df
                                    st.info(f"Using search results as fallback for {table_name}")
                            else:
                                st.warning(f"No search results could be parsed for {table_meta['display_name']}")
                                continue
                            
                        except Exception as e:
                            st.error(f"❌ Error processing MongoDB for {table_meta['display_name']}: {str(e)}")
                            diagnostics_logger.log_error(f"MongoDB processing failed for {table_name}: {str(e)}")
                            continue
                        
                        # Debug: Show what we're trying to join
                        st.info(f"DataFrames to join: {list(dfs.keys())}")
                        for name, df in dfs.items():
                            st.info(f"{name} columns: {list(df.columns)}")
                        
                        # Check if we have any DataFrames before trying to join
                        if not dfs:
                            st.error("❌ No DataFrames available from MongoDB. Please check your connection and try again.")
                            st.error("You can use FAISS or Qdrant instead by changing the vector search engine.")
                            st.stop()
                        
                        df1_context = join_dataframes(dfs, RAG_TABLES_CONFIG.get('relationships', []), backend='mongodb')
                        if df1_context is not None and not df1_context.empty:
                            context_info = f"Using top {len(df1_context)} semantically relevant rows for context in df1 (via MongoDB, {', '.join(selected_tables)})."
                            st.info(context_info)
                        else:
                            st.warning("No context data available from MongoDB search. Will proceed with auto-fetch.")
                            df1_context = None

            except Exception as e:
                st.error(f"Error in hybrid RAG semantic search: {e}")
                df1_context = dfs.get('df1')
            
                # Auto-fetch relevant tables if enabled and no context was found
                if auto_fetch and (df1_context is None or len(df1_context) == 0):
                    status_placeholder.info("Step 5.5: Auto-fetching relevant tables for your question...")
                    
                    # Use config-based SQL queries instead of GraphQL
                    engine = RAGUtils.get_fabric_engine()
                    dfs = {}
                    
                    for table_name in selected_tables:
                        table_meta = TABLES_META[table_name]
                        try:
                            # Use the utility function to safely get date filter column
                            date_column = get_date_filter_column(table_name)
                            
                            if date_column:  # Check if date_column exists and is not empty
                                # Fetch sample data from raw table for context with date filtering
                                sample_query = f"""
                                SELECT TOP 1000 *
                                FROM {table_meta['raw_table']}
                                WHERE {date_column} >= '{start_date}' AND {date_column} <= '{end_date}'
                                """
                            else:
                                # No date filtering - fetch all data (with limit for performance)
                                sample_query = f"""
                                SELECT TOP 1000 *
                                FROM {table_meta['raw_table']}
                                """
                                st.info(f"No date filtering configured for {table_meta['display_name']} - fetching all available data")
                            
                            df_sample = pd.read_sql(sample_query, engine)
                            dfs[f'df_{table_name}'] = df_sample
                            st.success(f"Auto-fetched {len(df_sample)} rows from {table_meta['display_name']}")
                        except Exception as e:
                            st.warning(f"Could not fetch data for {table_meta['display_name']}: {e}")
                    
                st.session_state['fabric_dfs'] = dfs
                st.success(f"Auto-fetched {len(dfs)} relevant table(s) for your question.")
            
            status_placeholder.info("Step 6: Preparing LLM context and prompt...")
            if df1_context is not None:
                df1_columns = list(df1_context.columns)
                n_context_rows = min(len(df1_context), top_n)
                df1_sample_csv = df1_context.head(n_context_rows).to_csv(index=False)
                
                # Create comprehensive context that makes entire dataset searchable
                comprehensive_context = RAGUtils.prepare_comprehensive_context(
                    dfs, 
                    df1_context, 
                    user_question,
                    context_sample_size=20 # Limit context sample to 20 rows
                )
                
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
                reasoning_prompt = RAGUtils.create_intelligent_prompt(user_question, comprehensive_context)
                with st.spinner(f"Generating reasoning with {llm_provider}..."):
                    try:
                        if llm_provider == "Local Deepseek LLM":
                            import os
                            if not os.path.exists("models/finetuned-deepseek-coder"):
                                st.warning("Local Deepseek model not found in models/finetuned-deepseek-coder. Please train or place the model there.")
                                reasoning = "[ERROR: Deepseek model not found]"
                            else:
                                # Build short context: column names and sample data
                                colnames = ', '.join(df1_context.columns)
                                sample_rows = df1_context.head(5).to_csv(index=False)
                                
                                # 🚀 ENHANCED: Add business context for Deepseek LLM reasoning
                                if vector_search_engine == "MongoDB" and 'schema_manager' in locals():
                                    try:
                                        first_table = list(dfs.keys())[0] if dfs else None
                                        if first_table:
                                            business_context = schema_manager.get_business_context(first_table)
                                            business_purpose = schema_manager.get_business_purpose(first_table)
                                            essential_cols = schema_manager.get_essential_columns(first_table)
                                            
                                            enhanced_prompt = f"<|user|>\nBusiness Context: {business_context.get('domain', 'Unknown')} - {business_purpose}\nEssential Columns: {', '.join(essential_cols)}\nContext: {colnames}\nSample data:\n{sample_rows}\nQuestion: {user_question}\n<|assistant|>\n"
                                        else:
                                            enhanced_prompt = f"<|user|>\nContext: {colnames}\nSample data:\n{sample_rows}\nQuestion: {user_question}\n<|assistant|>\n"
                                    except Exception as e:
                                        enhanced_prompt = f"<|user|>\nContext: {colnames}\nSample data:\n{sample_rows}\nQuestion: {user_question}\n<|assistant|>\n"
                                else:
                                    enhanced_prompt = f"<|user|>\nContext: {colnames}\nSample data:\n{sample_rows}\nQuestion: {user_question}\n<|assistant|>\n"
                                
                                reasoning = RAGUtils.run_deepseek_llm(
                                    enhanced_prompt,
                                    max_new_tokens=400,
                                    temperature=0.2
                                ).strip()
                        else:
                            # 🚀 ENHANCED: Use enhanced system message with business context for reasoning
                            system_message = "You are a senior data analyst. Explain your reasoning step by step for the user's question, referencing the DataFrames and columns you would use. Do not write code yet."
                            
                            if vector_search_engine == "MongoDB" and 'schema_manager' in locals():
                                try:
                                    first_table = list(dfs.keys())[0] if dfs else None
                                    if first_table:
                                        business_context = schema_manager.get_business_context(first_table)
                                        business_purpose = schema_manager.get_business_purpose(first_table)
                                        system_message = f"You are a senior data analyst with expertise in {business_context.get('domain', 'data analysis')}. Explain your reasoning step by step for the user's question, considering the business context and key entities. Reference the DataFrames and columns you would use. Do not write code yet."
                                except Exception as e:
                                    st.warning(f"Enhanced reasoning system message generation failed: {str(e)}")
                            
                            reasoning_response = client.chat.completions.create(
                                model=DEFAULT_MODEL,
                                messages=[
                                    {"role": "system", "content": system_message},
                                    {"role": "user", "content": reasoning_prompt}
                                ],
                                temperature=0.2,
                                max_tokens=400
                            )
                            reasoning = reasoning_response.choices[0].message.content.strip()
                        st.markdown("**AI Reasoning (Chain-of-Thought):**\n" + reasoning)
                    except Exception as e:
                        st.error(f"LLM error during reasoning: {str(e)}")
                        reasoning = ""

                # --- Now generate code, using the reasoning as context ---
                # 🚀 ENHANCED: Use MongoDB Schema Manager for context optimization if available
                if vector_search_engine == "MongoDB" and 'schema_manager' in locals() and df1_context is not None and not df1_context.empty:
                    try:
                        # Get the collection name from the first table
                        first_table = list(dfs.keys())[0] if dfs else None
                        if first_table:
                            # Use enhanced schema manager to optimize context
                            st.info("🔍 Debug: Using enhanced schema manager for context optimization...")
                            
                            # Get business context information for better reasoning
                            business_context = schema_manager.get_business_context(first_table)
                            business_purpose = schema_manager.get_business_purpose(first_table)
                            common_queries = schema_manager.get_common_queries(first_table)
                            
                            st.info(f"🎯 Business context: {business_context.get('domain', 'Unknown')} - {business_purpose}")
                            if common_queries:
                                st.info(f"💡 Common query patterns: {len(common_queries)} examples available")
                            
                            # Use enhanced schema manager to optimize context
                            df1_context_sample = schema_manager.optimize_dataframe_for_context(
                                df1_context, 
                                first_table
                            )
                            
                            # Get enhanced optimization details
                            essential_cols = schema_manager.get_essential_columns(first_table)
                            exclude_cols = schema_manager.get_exclude_columns(first_table)
                            max_rows = schema_manager.get_max_context_rows(first_table)
                            
                            # Get search optimization settings
                            search_settings = schema_manager.get_search_optimization_settings(first_table)
                            business_keywords = schema_manager.get_business_keywords(first_table)
                            semantic_boost_fields = schema_manager.get_semantic_boost_fields(first_table)
                            
                            st.success(f"✅ Context optimized using enhanced schema: {len(df1_context_sample)} rows x {len(df1_context_sample.columns)} columns")
                            st.info(f"Enhanced schema settings: Essential columns: {essential_cols}, Exclude: {exclude_cols}, Max rows: {max_rows}")
                            st.info(f"Search optimization: Business keywords: {business_keywords}, Semantic boost fields: {semantic_boost_fields}")
                            
                        else:
                            # Fallback to basic optimization
                            st.warning("No table name available for enhanced schema optimization, using fallback")
                            # Limit to 10 rows for context (absolute minimum)
                            max_context_rows = 10
                            if len(df1_context) > max_context_rows:
                                st.info(f"⚠️ Large dataset detected ({len(df1_context)} rows). Sampling {max_context_rows} rows for context to avoid token limits.")
                                df1_context_sample = df1_context.sample(n=max_context_rows, random_state=42)
                            else:
                                df1_context_sample = df1_context
                            
                            # Further reduce context by limiting column content for very long fields
                            if not df1_context_sample.empty:
                                # Create a simplified context DataFrame with limited content
                                # Only use columns that actually exist in the DataFrame
                                available_columns = list(df1_context_sample.columns)
                                context_columns = []
                                
                                # Add essential columns if they exist
                                if '_id_oid' in available_columns:
                                    context_columns.append('_id_oid')
                                if 'type' in available_columns:
                                    context_columns.append('type')
                                if 'value' in available_columns:
                                    context_columns.append('value')
                                
                                # If no essential columns found, use the first few available columns
                                if not context_columns:
                                    context_columns = available_columns[:1]  # Use only 1 column maximum
                                
                                # Limit Results field to first 50 characters if it exists (reduced from 100)
                                if 'Results' in df1_context_sample.columns:
                                    df1_context_sample['Results'] = df1_context_sample['Results'].astype(str).str[:50] + '...'
                                
                                # Only keep selected context columns
                                df1_context_sample = df1_context_sample[context_columns]
                                
                                st.info(f"✅ Context optimized: {len(df1_context_sample)} rows with {len(df1_context_sample.columns)} essential columns: {context_columns}")
                            
                    except Exception as schema_error:
                        st.warning(f"Enhanced schema optimization failed, using fallback: {str(schema_error)}")
                        # Fallback to basic optimization
                        max_context_rows = 10  # Limit to 10 rows for context (absolute minimum)
                        if df1_context is not None and len(df1_context) > max_context_rows:
                            st.info(f"⚠️ Large dataset detected ({len(df1_context)} rows). Sampling {max_context_rows} rows for context to avoid token limits.")
                            df1_context_sample = df1_context.sample(n=max_context_rows, random_state=42)
                        else:
                            df1_context_sample = df1_context if df1_context is not None else pd.DataFrame()
                        
                        # Further reduce context by limiting column content for very long fields
                        if not df1_context_sample.empty:
                            # Create a simplified context DataFrame with limited content
                            # Only use columns that actually exist in the DataFrame
                            available_columns = list(df1_context_sample.columns)
                            context_columns = []
                            
                            # Add essential columns if they exist
                            if '_id_oid' in available_columns:
                                context_columns.append('_id_oid')
                            if 'type' in available_columns:
                                context_columns.append('type')
                            if 'value' in available_columns:
                                context_columns.append('value')
                            
                            # If no essential columns found, use the first few available columns
                            if not context_columns:
                                context_columns = available_columns[:1]  # Use only 1 column maximum
                            
                            # Limit Results field to first 50 characters if it exists (reduced from 100)
                            if 'Results' in df1_context_sample.columns:
                                df1_context_sample['Results'] = df1_context_sample['Results'].astype(str).str[:50] + '...'
                            
                            # Only keep selected context columns
                            df1_context_sample = df1_context_sample[context_columns]
                            
                            st.info(f"✅ Context optimized: {len(df1_context_sample)} rows with {len(df1_context_sample.columns)} essential columns: {context_columns}")
                        
                else:
                    # Standard context optimization for non-MongoDB engines
                    # Limit the context size to prevent token limit exceeded errors
                    max_context_rows = 10  # Limit to 10 rows for context (absolute minimum)
                    if df1_context is not None and len(df1_context) > max_context_rows:
                        st.info(f"⚠️ Large dataset detected ({len(df1_context)} rows). Sampling {max_context_rows} rows for context to avoid token limits.")
                        df1_context_sample = df1_context.sample(n=max_context_rows, random_state=42)
                    else:
                        df1_context_sample = df1_context if df1_context is not None else pd.DataFrame()
                    
                    # Further reduce context by limiting column content for very long fields
                    if not df1_context_sample.empty:
                        # Create a simplified context DataFrame with limited content
                        # Only use columns that actually exist in the DataFrame
                        available_columns = list(df1_context_sample.columns)
                        context_columns = []
                        
                        # Add essential columns if they exist
                        if '_id_oid' in available_columns:
                            context_columns.append('_id_oid')
                        if 'type' in available_columns:
                            context_columns.append('type')
                        if 'value' in available_columns:
                            context_columns.append('value')
                        
                        # If no essential columns found, use the first few available columns
                        if not context_columns:
                            context_columns = available_columns[:1]  # Use only 1 column maximum
                        
                        # Limit Results field to first 50 characters if it exists (reduced from 100)
                        if 'Results' in df1_context_sample.columns:
                            df1_context_sample['Results'] = df1_context_sample['Results'].astype(str).str[:50] + '...'
                        
                        # Only keep selected context columns
                        df1_context_sample = df1_context_sample[context_columns]
                        
                        st.info(f"✅ Context optimized: {len(df1_context_sample)} rows with {len(df1_context_sample.columns)} essential columns: {context_columns}")

                if is_viz:
                    # 🚀 ENHANCED: Add business context for visualization prompts
                    if vector_search_engine == "MongoDB" and 'schema_manager' in locals():
                        try:
                            first_table = list(dfs.keys())[0] if dfs else None
                            if first_table:
                                business_context = schema_manager.get_business_context(first_table)
                                business_purpose = schema_manager.get_business_purpose(first_table)
                                essential_cols = schema_manager.get_essential_columns(first_table)
                                
                                enhanced_prompt = f"""
You are a Python data analyst with expertise in {business_context.get('domain', 'data analysis')}. The following pandas DataFrames are loaded: {', '.join(dfs.keys())}.

BUSINESS CONTEXT: {business_context.get('domain', 'Unknown')} - {business_purpose}
ESSENTIAL BUSINESS COLUMNS: {', '.join(essential_cols)}

IMPORTANT: Use ONLY the actual DataFrame names listed above and the actual column names provided below. Do NOT invent or assume column names or DataFrame names.

Available DataFrames: {list(dfs.keys())}
Available columns: {list(set([col for df in dfs.values() for col in df.columns]))}

Sample data from df1 (showing first 5 rows):
{df1_context_sample.head(5).to_string() if not df1_context_sample.empty else "No data available"}

Here is your reasoning for how to answer the question:\n{reasoning}\n
Write Python code to answer the following question. Use ONLY the actual DataFrame names from the list above and the actual column names from the DataFrames. Focus on business-relevant columns when available. If a plot is required, use matplotlib and assign the figure to a variable called fig. If a tabular result is also needed, assign it to a variable called result. Only output the code (no explanations, no print statements, no comments). Question: {user_question}
"""
                            else:
                                enhanced_prompt = prompt
                        except Exception as e:
                            st.warning(f"Enhanced prompt generation failed: {str(e)}")
                            enhanced_prompt = prompt
                    else:
                        enhanced_prompt = prompt
                    
                    prompt = enhanced_prompt
                else:
                    # 🚀 ENHANCED: Add business context for standard code prompts
                    if vector_search_engine == "MongoDB" and 'schema_manager' in locals():
                        try:
                            first_table = list(dfs.keys())[0] if dfs else None
                            if first_table:
                                business_context = schema_manager.get_business_context(first_table)
                                business_purpose = schema_manager.get_business_purpose(first_table)
                                essential_cols = schema_manager.get_essential_columns(first_table)
                                
                                enhanced_prompt = f"""
You are a Python data analyst with expertise in {business_context.get('domain', 'data analysis')}. The following pandas DataFrames are loaded: {', '.join(dfs.keys())}.

BUSINESS CONTEXT: {business_context.get('domain', 'Unknown')} - {business_purpose}
ESSENTIAL BUSINESS COLUMNS: {', '.join(essential_cols)}

IMPORTANT: Use ONLY the actual DataFrame names listed above and the actual column names provided below. Do NOT invent or assume column names or DataFrame names.

Available DataFrames: {list(dfs.keys())}
Available columns: {list(set([col for df in dfs.values() for col in df.columns]))}

Sample data from df1 (showing first 5 rows):
{df1_context_sample.head(5).to_string() if not df1_context_sample.empty else "No data available"}

Here is your reasoning for how to answer the question:\n{reasoning}\n
Write Python pandas code to answer the following question. Use ONLY the actual DataFrame names from the list above and the actual column names from the DataFrames. Focus on business-relevant columns when available. Assign the final answer to a variable called result. Only output the code (no explanations, no print statements, no comments). Question: {user_question}
"""
                            else:
                                enhanced_prompt = prompt
                        except Exception as e:
                            st.warning(f"Enhanced prompt generation failed: {str(e)}")
                            enhanced_prompt = prompt
                    else:
                        enhanced_prompt = prompt
                    
                    prompt = enhanced_prompt

                with st.spinner(f"Generating code with {llm_provider}..."):
                    try:
                        if llm_provider == "Local Deepseek LLM":
                            import os
                            if not os.path.exists("models/finetuned-deepseek-coder"):
                                st.warning("Local Deepseek model not found in models/finetuned-deepseek-coder. Please train or place the model there.")
                                pandas_code = "[ERROR: Deepseek model not found]"
                            else:
                                # Build short context: column names and sample data
                                colnames = ', '.join(df1_context.columns)
                                sample_rows = df1_context.head(5).to_csv(index=False)
                                
                                # 🚀 ENHANCED: Add business context for Deepseek LLM
                                if vector_search_engine == "MongoDB" and 'schema_manager' in locals():
                                    try:
                                        first_table = list(dfs.keys())[0] if dfs else None
                                        if first_table:
                                            business_context = schema_manager.get_business_context(first_table)
                                            business_purpose = schema_manager.get_business_purpose(first_table)
                                            essential_cols = schema_manager.get_essential_columns(first_table)
                                            
                                            enhanced_prompt = f"<|user|>\nBusiness Context: {business_context.get('domain', 'Unknown')} - {business_purpose}\nEssential Columns: {', '.join(essential_cols)}\nContext: {colnames}\nSample data:\n{sample_rows}\nQuestion: {user_question}\n<|assistant|>\n"
                                        else:
                                            enhanced_prompt = f"<|user|>\nContext: {colnames}\nSample data:\n{sample_rows}\nQuestion: {user_question}\n<|assistant|>\n"
                                    except Exception as e:
                                        enhanced_prompt = f"<|user|>\nContext: {colnames}\nSample data:\n{sample_rows}\nQuestion: {user_question}\n<|assistant|>\n"
                                else:
                                    enhanced_prompt = f"<|user|>\nContext: {colnames}\nSample data:\n{sample_rows}\nQuestion: {user_question}\n<|assistant|>\n"
                                
                                pandas_code = RAGUtils.run_deepseek_llm(
                                    enhanced_prompt,
                                    max_new_tokens=700,
                                    temperature=0.0
                                )
                                pandas_code = RAGUtils.clean_code(pandas_code)
                        else:
                            # 🚀 ENHANCED: Use enhanced system message with business context
                            system_message = "You are a helpful Python data analyst. Only output code that produces the answer as a DataFrame or Series, and assign it to a variable called result. If a plot is required, assign the matplotlib figure to a variable called fig. You can use any of the loaded DataFrames (df1, df2, etc.)."
                            
                            if vector_search_engine == "MongoDB" and 'schema_manager' in locals():
                                try:
                                    first_table = list(dfs.keys())[0] if dfs else None
                                    if first_table:
                                        business_context = schema_manager.get_business_context(first_table)
                                        business_purpose = schema_manager.get_business_purpose(first_table)
                                        system_message = f"You are a helpful Python data analyst with expertise in {business_context.get('domain', 'data analysis')}. Only output code that produces the answer as a DataFrame or Series, and assign it to a variable called result. If a plot is required, assign the matplotlib figure to a variable called fig. You can use any of the loaded DataFrames (df1, df2, etc.). Focus on business-relevant columns when available."
                                except Exception as e:
                                    st.warning(f"Enhanced system message generation failed: {str(e)}")
                            
                            code_response = client.chat.completions.create(
                                model=DEFAULT_MODEL,
                                messages=[
                                    {"role": "system", "content": system_message},
                                    {"role": "user", "content": prompt}
                                ],
                                temperature=0.0,
                                max_tokens=700
                            )
                            pandas_code = code_response.choices[0].message.content.strip()
                            pandas_code = RAGUtils.clean_code(pandas_code)
                        if show_code:
                            st.code(pandas_code, language="python")
                        
                        # --- Validate code uses actual names before execution ---
                        all_available_columns = set()
                        for name, df in dfs.items():
                            all_available_columns.update(df.columns)
                        if 'df1' in exec_env:
                            all_available_columns.update(exec_env['df1'].columns)
                        
                            # Check for undefined DataFrame references
                            import re
                            df_pattern = r'\b(df_\w+)\b'
                            found_dfs = re.findall(df_pattern, pandas_code)
                            undefined_dfs = [df_name for df_name in found_dfs if df_name not in exec_env]
                            
                            if undefined_dfs:
                                st.error(f"Generated code references undefined DataFrames: {undefined_dfs}")
                                st.error(f"Available DataFrames: {list(exec_env.keys())}")
                                st.error("Please rephrase your question to use only the available DataFrame names.")
                                st.stop()
                        
                        is_valid, validation_msg = RAGUtils.validate_code_uses_actual_names(
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
                        is_safe, keyword = RAGUtils.is_code_safe(pandas_code)
                        if not is_safe:
                            st.error(f"Blocked potentially unsafe code: found forbidden keyword '{keyword}'. Please rephrase your question.")
                        else:
                                # Debug: Show available DataFrames
                                st.info(f"Available DataFrames for execution: {list(exec_env.keys())}")
                                
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
                                        elif isinstance(result, tuple):
                                            st.write("**Result:**")
                                            for item in result:
                                                if isinstance(item, (pd.DataFrame, pd.Series)):
                                                    st.dataframe(item)
                                                else:
                                                    st.write(item)
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
                                    error_message = str(e) if 'e' in locals() else "Unknown error"
                                    fix_prompt = f"""
The following code was generated to answer the question: '{user_question}'. However, it failed with the error: {error_message}.

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
                                        fixed_code = RAGUtils.clean_code(fixed_code)
                                        st.info("Retrying with corrected code:")
                                        if show_code:
                                            st.code(fixed_code, language="python")
                                        is_safe2, keyword2 = RAGUtils.is_code_safe(fixed_code)
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
                                                    elif isinstance(result2, tuple):
                                                        st.write("**Result (retry):**")
                                                        for item in result2:
                                                            if isinstance(item, (pd.DataFrame, pd.Series)):
                                                                st.dataframe(item)
                                                            else:
                                                                st.write(item)
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

with tabs[1]:
    st.header("SQL Editor (Microsoft Fabric SQL Endpoint)")
    # Place all SQL Editor logic here (do not include RAG QA logic)
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

    engine = sa.create_engine(
        f"mssql+pyodbc:///?odbc_connect={params}",
        connect_args={'attrs_before': attrs_before}
    )
    st.info(f"Connected to SQL endpoint: {sql_endpoint}")

    # SQL Query Input
    sql_query = st.text_area("Enter your SQL query:", height=200)
    if st.button("Run Query"):
        if sql_query:
            try:
                df = pd.read_sql(sql_query, engine)
                st.write("**Query Result:**")
                st.dataframe(df)
                # Download button for DataFrame
                csv_data = df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="Download result as CSV",
                    data=csv_data,
                    file_name="sql_result.csv",
                    mime="text/csv"
                )
            except Exception as e:
                st.error(f"Error executing SQL query: {e}\n{traceback.format_exc()}")
        else:
            st.warning("Please enter a SQL query.") 

def _basic_context_optimization(df):
    """Basic context optimization when schema manager is not available."""
    if df is None or df.empty:
        return df
    
    # Sample to 10 rows maximum
    if len(df) > 10:
        df_optimized = df.sample(n=10, random_state=42)
    else:
        df_optimized = df.copy()
    
    # Keep only first 2 columns to minimize tokens
    if len(df_optimized.columns) > 2:
        df_optimized = df_optimized.iloc[:, :2]
    
    return df_optimized