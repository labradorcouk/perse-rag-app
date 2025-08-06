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
from auth import azure_auth
from utils.sql_connection import sql_manager
from config.queries import QUERIES
from config.table_schemas import TABLE_SCHEMAS
from config.safe_builtins import SAFE_BUILTINS
from utils.rag_utils import RAGUtils
# QdrantIndex will be imported conditionally when needed
from qdrant_client import models
import yaml
import time
from utils.diagnostics_logger import diagnostics_logger, EventType, LogLevel
from utils.diagnostics_dashboard import diagnostics_dashboard
from utils.startup_monitor import startup_monitor
import threading

# Performance optimization imports
from utils.performance_optimizer import performance_optimizer

# Start monitoring
startup_monitor.mark_milestone("Application imports")

# Conditional imports for ODBC support
try:
    import pyodbc
    import sqlalchemy as sa
    ODBC_AVAILABLE = True
except ImportError as e:
    st.warning(f"ODBC drivers not available: {e}")
    ODBC_AVAILABLE = False
    pyodbc = None
    sa = None

load_dotenv()
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
DEFAULT_MODEL = os.getenv('DEFAULT_MODEL', 'gpt-3.5-turbo')
MAX_TOKENS = int(os.getenv('MAX_TOKENS', '600'))
TEMPERATURE = float(os.getenv('TEMPERATURE', '0.2'))
client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

startup_monitor.mark_milestone("Environment setup")

# --- Main Azure Authentication Flow ---
# azure_auth is now imported from the auth package

# Initialize authentication session state first
azure_auth.init_session_state()

startup_monitor.mark_milestone("Authentication initialization")

# Background initialization for heavy components
def initialize_background():
    """Initialize heavy components in background."""
    def background_init():
        try:
            startup_monitor.mark_milestone("Background init start")
            
            # Use performance optimizer for model loading
            model = performance_optimizer.get_cached_embedding_model('all-MiniLM-L6-v2')
            st.session_state['embedding_model'] = model
            
            startup_monitor.mark_milestone("Embedding model loaded")
            
            # Use performance optimizer for Qdrant client
            qdrant_url = os.getenv('QDRANT_URL', 'http://localhost:6333')
            qdrant_client = performance_optimizer.get_cached_qdrant_client(qdrant_url)
            st.session_state['qdrant_client'] = qdrant_client
            
            startup_monitor.mark_milestone("Qdrant client loaded")
            
            # Optimize memory usage
            performance_optimizer.optimize_memory_usage()
            
            print("✅ Background initialization completed with performance optimizations")
            
        except Exception as e:
            print(f"❌ Background initialization failed: {e}")
    
    thread = threading.Thread(target=background_init, daemon=True)
    thread.start()

# Start background initialization
initialize_background()

# Performance monitoring: Periodic memory optimization
def periodic_optimization():
    """Run periodic performance optimizations."""
    import threading
    import time
    
    def optimization_loop():
        while True:
            try:
                # Optimize memory usage every 5 minutes
                performance_optimizer.optimize_memory_usage()
                performance_optimizer.cleanup_expired_cache()
                
                # Log performance metrics
                metrics = performance_optimizer.monitor_performance()
                print(f"Performance metrics: {metrics}")
                
                time.sleep(300)  # 5 minutes
            except Exception as e:
                print(f"Error in periodic optimization: {e}")
                time.sleep(60)
    
    thread = threading.Thread(target=optimization_loop, daemon=True)
    thread.start()

# Start periodic optimization
periodic_optimization()

# Check if we're in a callback scenario
query_params = st.query_params
if 'code' in query_params and 'state' in query_params:
    # We're in an OAuth callback - handle it
    if azure_auth.handle_auth_callback():
        st.success("✅ Authentication successful! Redirecting to main application...")
        st.rerun()
    else:
        st.error("❌ Authentication failed. Please try again.")
        st.rerun()

# Check if user is authenticated
if not azure_auth.login_button():
    st.stop()

# Get access token for SQL connections
access_token = azure_auth.get_access_token()
user_display_name = azure_auth.get_user_display_name()

# Display user info
if user_display_name:
    st.success(f"Welcome, {user_display_name}!")

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

# --- Copy all the functions from rag_fabric_app.py ---
# (I'll include the key functions here, but you can copy the rest from the original file)

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

# --- Only after authentication, set up the UI ---
st.set_page_config(page_title="Fabric RAG QA (Docker)", layout="wide")
st.title("Fabric RAG (Retrieval-Augmented Generation) QA App - Docker Version")

# --- Only show sidebar and navigation after authentication ---
# Check if user is authenticated using Azure auth state
is_authenticated = st.session_state.get('azure_auth_state', {}).get('is_authenticated', False)

if is_authenticated:
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
    tabs = st.tabs(["RAG QA", "SQL Editor", "🔍 Diagnostics", "📊 Performance"])

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
                # Log RAG query start
                start_time = time.time()
                diagnostics_logger.log_rag_query(
                    question=user_question,
                    selected_tables=selected_tables,
                    vector_search_engine=vector_search_engine,
                    llm_provider=llm_provider
                )
                
                # Performance optimization: Clean up expired cache
                performance_optimizer.cleanup_expired_cache()
                
                status_placeholder = st.empty()
                status_placeholder.info("Step 1: Embedding user question...")
                is_viz = RAGUtils.is_visualization_request(user_question)
                df1_context = None # Initialize df1_context
                
                # Debug: Check which vector search engine is selected
                st.info(f"Selected vector search engine: {vector_search_engine}")
                
                try:
                    model = RAGUtils.get_embedding_model(selected_model_path_tab)
                    top_n = 5000

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
                        
                        # Debug: Check environment variables
                        st.info(f"🔍 Debug: QDRANT_URL = {os.getenv('QDRANT_URL')}")
                        st.info(f"🔍 Debug: QDRANT_HOST = {os.getenv('QDRANT_HOST')}")
                        st.info(f"🔍 Debug: QDRANT_PORT = {os.getenv('QDRANT_PORT')}")
                        
                        try:
                            # Import QdrantIndex only when needed
                            st.info("🔍 Debug: Importing QdrantIndex...")
                            from utils.qdrant_utils import QdrantIndex
                            st.success("✅ QdrantIndex import successful")
                        except Exception as e:
                            st.error(f"❌ QdrantIndex import failed: {str(e)}")
                            st.error(f"Error type: {type(e).__name__}")
                            st.stop()
                        
                        # Let QdrantIndex handle the connection directly
                        for table_name in selected_tables:
                            table_meta = TABLES_META[table_name]
                            st.info(f"🔍 Debug: Processing table {table_name}")
                            st.info(f"🔍 Debug: Table meta = {table_meta}")
                            
                            try:
                                st.info(f"Connecting to Qdrant collection: {table_meta['collection']}")
                                
                                # Debug: Test direct QdrantClient connection first
                                st.info("🔍 Debug: Testing direct QdrantClient connection...")
                                from qdrant_client import QdrantClient
                                qdrant_url = os.getenv('QDRANT_URL')
                                if qdrant_url:
                                    try:
                                        test_client = QdrantClient(url=qdrant_url, prefer_grpc=True)
                                        collections = test_client.get_collections()
                                        st.success(f"✅ Direct QdrantClient connection successful: {len(collections.collections)} collections")
                                    except Exception as e:
                                        st.error(f"❌ Direct QdrantClient connection failed: {str(e)}")
                                        st.error(f"Error type: {type(e).__name__}")
                                        st.error(f"Full error: {e}")
                                        st.stop()
                                else:
                                    st.error("❌ QDRANT_URL not set")
                                    st.stop()
                                
                                # Now try QdrantIndex
                                st.info("🔍 Debug: Creating QdrantIndex...")
                                qdrant_index = QdrantIndex(
                                    collection_name=table_meta['collection'],
                                    embedding_model=model
                                )
                                st.success(f"✅ Connected to Qdrant collection: {table_meta['collection']}")
                                
                                st.info("🔍 Debug: Performing search...")
                                search_results = qdrant_index.search(
                                    user_question,
                                    limit=top_n,
                                    with_payload=True
                                )
                                st.success(f"✅ Search successful: {len(search_results)} results")
                                
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
                                
                                # Performance optimization: Cache DataFrame
                                cache_key = f"qdrant_{table_name}_{user_question[:50]}"
                                performance_optimizer.cache_dataframe(cache_key, df_context, ttl=1800)  # 30 minutes
                                
                                dfs[table_name] = df_context
                                st.success(f"✅ Successfully processed {table_meta['display_name']}")
                                
                            except Exception as e:
                                # Log the error
                                diagnostics_logger.log_error(
                                    component="Qdrant_Search",
                                    error=e,
                                    context={
                                        'table_name': table_name,
                                        'vector_search_engine': vector_search_engine,
                                        'user_question': user_question
                                    }
                                )
                                
                                # Show the actual error instead of generic message
                                st.error(f"❌ Error searching Qdrant for {table_meta['display_name']}: {str(e)}")
                                st.error(f"❌ Error type: {type(e).__name__}")
                                st.error(f"❌ Full error details: {e}")
                                st.error(f"❌ Error location: {traceback.format_exc()}")
                                
                                # Only show connection-related suggestions if it's actually a connection error
                                if "Connection refused" in str(e) or "Connection" in str(e):
                                    st.warning("Qdrant connection failed. Possible solutions:")
                                    st.info("1. Check if Qdrant is running: docker ps | grep qdrant")
                                    st.info("2. Use FAISS instead: Select 'FAISS' as vector search engine")
                                    st.info("3. Use auto-fetch: Enable 'Auto-fetch relevant tables'")
                                    st.info("4. Restart containers: docker-compose restart")
                                else:
                                    st.warning("Qdrant search failed. Possible solutions:")
                                    st.info("1. Use FAISS instead: Select 'FAISS' as vector search engine")
                                    st.info("2. Use auto-fetch: Enable 'Auto-fetch relevant tables'")
                                    st.info("3. Check the collection configuration")
                                continue
                        
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
                        st.info(f"🔍 Debug: MONGODB_COLLECTION_NAME = {os.getenv('MONGODB_COLLECTION_NAME', 'addressMatches')}")
                        
                        try:
                            # Import MongoDBIndex only when needed
                            st.info("🔍 Debug: Importing MongoDBIndex...")
                            from utils.mongodb_utils import MongoDBIndex
                            st.success("✅ MongoDBIndex import successful")
                        except Exception as e:
                            st.error(f"❌ MongoDBIndex import failed: {str(e)}")
                            st.error(f"Error type: {type(e).__name__}")
                            st.stop()
                        
                        # Process each selected table
                        for table_name in selected_tables:
                            table_meta = TABLES_META[table_name]
                            st.info(f"🔍 Debug: Processing table {table_name}")
                            st.info(f"🔍 Debug: Table meta = {table_meta}")
                            
                            try:
                                # Use table name as collection name for MongoDB
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
                                    continue
                                
                                # Get collection info
                                collection_info = mongodb_index.get_collection_info()
                                st.info(f"Collection info: {collection_info}")
                                
                                st.info("🔍 Debug: Performing search...")
                                search_results = mongodb_index.search(
                                    user_question,
                                    limit=top_n,
                                    score_threshold=0.01
                                )
                                st.success(f"✅ Search successful: {len(search_results)} results")
                                
                                if not search_results:
                                    st.warning(f"No relevant documents found in MongoDB for {table_meta['display_name']}.")
                                    continue
                                
                                status_placeholder.info(f"Building context from MongoDB search results for {table_meta['display_name']}...")
                                
                                # Convert search results to DataFrame
                                # MongoDB results contain 'payload' which is the document content
                                documents = []
                                for result in search_results:
                                    # Parse the payload (assuming it's JSON or structured data)
                                    try:
                                        import json
                                        if isinstance(result['payload'], str):
                                            doc_data = json.loads(result['payload'])
                                        else:
                                            doc_data = result['payload']
                                        
                                        # Add metadata
                                        doc_data['_score'] = result.get('score', 0)
                                        doc_data['_metadata'] = result.get('metadata', {})
                                        documents.append(doc_data)
                                    except Exception as e:
                                        st.warning(f"Could not parse document: {e}")
                                        continue
                                
                                if documents:
                                    df_context = pd.DataFrame(documents)
                                    
                                    # Debug: Show available columns
                                    st.info(f"Available columns in {table_meta['display_name']}: {list(df_context.columns)}")
                                    
                                    # Only keep columns defined for MongoDB for this table (if specified)
                                    mongodb_columns = table_meta.get('mongodb_columns', table_meta.get('qdrant_columns', []))
                                    if mongodb_columns:
                                        available_mongodb_columns = [col for col in mongodb_columns if col in df_context.columns]
                                        if available_mongodb_columns:
                                            df_context = df_context[available_mongodb_columns]
                                        else:
                                            st.warning(f"No matching columns found for {table_meta['display_name']}. Using all available columns.")
                                    else:
                                        st.info(f"No specific MongoDB columns defined for {table_meta['display_name']}. Using all available columns.")
                                    
                                    # Performance optimization: Cache DataFrame
                                    cache_key = f"mongodb_{table_name}_{user_question[:50]}"
                                    performance_optimizer.cache_dataframe(cache_key, df_context, ttl=1800)  # 30 minutes
                                    
                                    dfs[table_name] = df_context
                                    st.success(f"✅ Successfully processed {table_meta['display_name']}")
                                else:
                                    st.warning(f"No valid documents found in MongoDB for {table_meta['display_name']}.")
                                    continue
                                
                            except Exception as e:
                                # Log the error
                                diagnostics_logger.log_error(
                                    component="MongoDB_Search",
                                    error=e,
                                    context={
                                        'table_name': table_name,
                                        'vector_search_engine': vector_search_engine,
                                        'user_question': user_question
                                    }
                                )
                                
                                # Show the actual error instead of generic message
                                st.error(f"❌ Error searching MongoDB for {table_meta['display_name']}: {str(e)}")
                                st.error(f"❌ Error type: {type(e).__name__}")
                                st.error(f"❌ Full error details: {e}")
                                
                                # Show connection-related suggestions if it's actually a connection error
                                if "Connection" in str(e) or "Authentication" in str(e):
                                    st.warning("MongoDB connection failed. Possible solutions:")
                                    st.info("1. Check MONGODB_URI environment variable")
                                    st.info("2. Verify MongoDB Atlas credentials")
                                    st.info("3. Use FAISS instead: Select 'FAISS' as vector search engine")
                                    st.info("4. Use Qdrant instead: Select 'Qdrant' as vector search engine")
                                else:
                                    st.warning("MongoDB search failed. Possible solutions:")
                                    st.info("1. Use FAISS instead: Select 'FAISS' as vector search engine")
                                    st.info("2. Use Qdrant instead: Select 'Qdrant' as vector search engine")
                                    st.info("3. Check the collection configuration")
                                continue
                        
                        # Debug: Show what we're trying to join
                        st.info(f"DataFrames to join: {list(dfs.keys())}")
                        for name, df in dfs.items():
                            st.info(f"{name} columns: {list(df.columns)}")
                        
                        df1_context = join_dataframes(dfs, RAG_TABLES_CONFIG.get('relationships', []), backend='mongodb')
                        if df1_context is not None and not df1_context.empty:
                            context_info = f"Using top {len(df1_context)} semantically relevant rows for context in df1 (via MongoDB, {', '.join(selected_tables)})."
                            st.info(context_info)
                        else:
                            st.warning("No context data available from MongoDB search. Will proceed with auto-fetch.")
                            df1_context = None

                except Exception as e:
                    # Log the error
                    diagnostics_logger.log_error(
                        component="RAG_Search",
                        error=e,
                        context={
                            'vector_search_engine': vector_search_engine,
                            'selected_tables': selected_tables,
                            'user_question': user_question
                        }
                    )
                    
                    st.error(f"❌ Error in hybrid RAG semantic search: {str(e)}")
                    st.error(f"❌ Error type: {type(e).__name__}")
                    st.error(f"❌ Full error details: {e}")
                    st.error(f"❌ Vector search engine: {vector_search_engine}")
                    st.error(f"❌ Selected tables: {selected_tables}")
                    df1_context = dfs.get('df1')
                
                # Auto-fetch relevant tables if enabled and no context was found
                if auto_fetch and (df1_context is None or len(df1_context) == 0):
                    status_placeholder.info("Step 5.5: Auto-fetching relevant tables for your question...")
                    
                    # Check if ODBC is available for SQL queries
                    if not ODBC_AVAILABLE:
                        st.warning("ODBC not available - cannot fetch data from Fabric SQL endpoint.")
                        st.info("To enable data fetching, rebuild with ODBC support:")
                        st.code("docker build -f Dockerfile.docker -t rag-fabric-app .")
                        st.info("For now, you can use the RAG QA with sample data or Qdrant.")
                        st.stop()
                    
                    # Use config-based SQL queries instead of GraphQL
                    try:
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
                    except ImportError as e:
                        st.error(f"ODBC not available: {e}")
                        st.info("Please rebuild with ODBC support or use Qdrant/FAISS for RAG functionality.")
                        st.stop()
                    except Exception as e:
                        st.error(f"Error connecting to Fabric: {e}")
                        st.info("Please check your Azure credentials and network connectivity.")
                        st.stop()
                
                # Continue with the rest of the RAG logic...
                # (Copy the rest of the logic from the original rag_fabric_app.py)
                
                # Step 3: Generate reasoning and code
                if df1_context is not None and not df1_context.empty:
                    status_placeholder.info("Step 3: Generating reasoning and code...")
                    
                    # Prepare the context for the LLM
                    comprehensive_context = f"""
Available DataFrames:
- df1: {len(df1_context)} rows with columns: {list(df1_context.columns)}

Sample data from df1:
{df1_context.head(3).to_string()}

IMPORTANT: The main DataFrame is called 'df1'. Use 'df1' for all operations.
"""
                    
                    # Generate reasoning
                    reasoning_prompt = f"""
You are a data analyst. Given the following data and question, provide step-by-step reasoning for how to answer it.

Data Context:
{comprehensive_context}

Question: {user_question}

Provide clear, step-by-step reasoning for how to answer this question using the available data. Focus on:
1. What specific analysis is needed
2. Which columns to use
3. What operations to perform
4. How to interpret the results

Reasoning:
"""
                    
                    try:
                        if llm_provider == "Local Deepseek LLM":
                            # Use local Deepseek LLM for reasoning
                            reasoning = RAGUtils.run_deepseek_llm(
                                reasoning_prompt,
                                max_new_tokens=300,
                                temperature=0.1
                            )
                        else:
                            # Use OpenAI for reasoning
                            reasoning_response = client.chat.completions.create(
                                model=DEFAULT_MODEL,
                                messages=[
                                    {"role": "system", "content": "You are a helpful data analyst. Provide clear, step-by-step reasoning for how to answer data analysis questions."},
                                    {"role": "user", "content": reasoning_prompt}
                                ],
                                temperature=0.1,
                                max_tokens=300
                            )
                            reasoning = reasoning_response.choices[0].message.content.strip()
                        
                        if show_code:
                            st.write("**Reasoning:**")
                            st.write(reasoning)
                        
                        # Generate code
                        code_prompt = f"""
CRITICAL: Use ONLY the exact DataFrame names and column names provided below. Do NOT invent or assume any names.

AVAILABLE DATAFRAMES:
{chr(10).join([f"- {name}: {len(df)} rows with columns: {list(df.columns)}" for name, df in dfs.items()])}

MAIN DATAFRAME: df1 ({len(df1_context)} rows)
Sample data from df1:
{df1_context.head(3).to_string()}

AVAILABLE COLUMNS: {list(set([col for df in dfs.values() for col in df.columns]))}

Your reasoning: {reasoning}

Write Python pandas code to answer this question. Use ONLY the DataFrame names listed above (especially 'df1') and the exact column names provided. Assign the final answer to a variable called 'result'.

Question: {user_question}

Code (only code, no comments or explanations):
"""
                        
                        # Generate code with better DataFrame information
                        from utils.dataframe_corrector import DataFrameCorrector
                        
                        dataframe_corrector = DataFrameCorrector()
                        dataframe_info = dataframe_corrector.get_dataframe_info(dfs)
                        
                        code_prompt = f"""
CRITICAL: Use ONLY the exact DataFrame names and column names provided below. Do NOT invent or assume any names.

AVAILABLE DATAFRAMES:
{dataframe_info}

MAIN DATAFRAME: df1 ({len(df1_context)} rows)
Sample data from df1:
{df1_context.head(3).to_string()}

AVAILABLE COLUMNS: {list(set([col for df in dfs.values() for col in df.columns]))}

Your reasoning: {reasoning}

Write Python pandas code to answer this question. Use ONLY the DataFrame names listed above (especially 'df1') and the exact column names provided. Assign the final answer to a variable called 'result'.

Question: {user_question}

Code (only code, no comments or explanations):
"""
                        
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
                                        prompt = f"<|user|>\nContext: {colnames}\nSample data:\n{sample_rows}\nQuestion: {user_question}\n<|assistant|>\n"
                                        pandas_code = RAGUtils.run_deepseek_llm(
                                            prompt,
                                            max_new_tokens=700,
                                            temperature=0.0
                                        )
                                        pandas_code = RAGUtils.clean_code(pandas_code)
                                else:
                                    code_response = client.chat.completions.create(
                                        model=DEFAULT_MODEL,
                                        messages=[
                                            {"role": "system", "content": "You are a helpful Python data analyst. Only output code that produces the answer as a DataFrame or Series, and assign it to a variable called result. If a plot is required, assign the matplotlib figure to a variable called fig. You can use any of the loaded DataFrames (df1, df2, etc.)."},
                                            {"role": "user", "content": code_prompt}
                                        ],
                                        temperature=0.0,
                                        max_tokens=700
                                    )
                                    pandas_code = code_response.choices[0].message.content.strip()
                                    pandas_code = RAGUtils.clean_code(pandas_code)
                                
                                if show_code:
                                    st.code(pandas_code, language="python")
                                
                                # Prepare execution environment
                                exec_env = {
                                    'df1': df1_context,
                                    'pd': pd,
                                    'plt': plt,
                                    'np': np
                                }
                                
                                # Add other DataFrames if available
                                for name, df in dfs.items():
                                    if name != 'bev':  # df1 is already the main context
                                        exec_env[f'df_{name}'] = df
                                
                                # --- Validate code uses actual names before execution ---
                                all_available_columns = set()
                                for name, df in dfs.items():
                                    all_available_columns.update(df.columns)
                                if 'df1' in exec_env:
                                    all_available_columns.update(exec_env['df1'].columns)
                                
                                # Use the intelligent DataFrameCorrector utility
                                from utils.dataframe_corrector import DataFrameCorrector
                                
                                # Create an instance of the corrector
                                dataframe_corrector = DataFrameCorrector()
                                
                                # Correct DataFrame names using intelligent fixing
                                pandas_code, corrections_made = dataframe_corrector.correct_dataframe_names(
                                    pandas_code, 
                                    exec_env
                                )
                                
                                if corrections_made:
                                    st.warning(f"Fixed DataFrame name references: {', '.join(corrections_made)}")
                                    st.info(f"Available DataFrames: {list(exec_env.keys())}")
                                
                                # Validate remaining DataFrame references
                                undefined_refs = dataframe_corrector.validate_dataframe_references(
                                    pandas_code, 
                                    exec_env
                                )
                                
                                if undefined_refs:
                                    st.error(f"Could not resolve DataFrame references: {undefined_refs}")
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
                                    
                                    # Execute the code safely
                                    try:
                                        local_vars = exec_env.copy()
                                        exec(pandas_code, {"pd": pd, "plt": plt, "__builtins__": SAFE_BUILTINS}, local_vars)
                                        result = local_vars.get("result")
                                        fig = local_vars.get("fig")
                                        
                                        if fig is not None:
                                            st.write("**Visualization:**")
                                            st.pyplot(fig)
                                        
                                        if result is not None:
                                            if isinstance(result, pd.DataFrame):
                                                nrows = len(result)
                                                if nrows > 1000:  # MAX_DISPLAY_ROWS
                                                    st.warning(f"Result has {nrows} rows. Displaying only the first 1000 rows.")
                                                    st.dataframe(result.head(1000))
                                                else:
                                                    st.dataframe(result)
                                                
                                                # Download button
                                                csv_data = result.to_csv(index=False).encode('utf-8')
                                                st.download_button(
                                                    label="Download result as CSV",
                                                    data=csv_data,
                                                    file_name="rag_result.csv",
                                                    mime="text/csv"
                                                )
                                            elif isinstance(result, pd.Series):
                                                st.write("**Result:**")
                                                st.write(result)
                                            else:
                                                st.write("**Result:**")
                                                st.write(result)
                                        else:
                                            st.warning("No result variable found in the generated code.")
                                            
                                        # Log successful code execution
                                        execution_time = time.time() - start_time
                                        diagnostics_logger.log_code_generation(
                                            question=user_question,
                                            generated_code=pandas_code,
                                            llm_provider=llm_provider,
                                            execution_success=True
                                        )
                                        diagnostics_logger.log_performance(
                                            component="Code_Execution",
                                            operation="RAG_Query_Complete",
                                            duration_seconds=execution_time
                                        )
                                            
                                    except NameError as e:
                                        # Handle DataFrame name errors
                                        error_msg = str(e)
                                        if "name" in error_msg and "is not defined" in error_msg:
                                            st.error(f"DataFrame name error: {error_msg}")
                                            st.error("The generated code is trying to use a DataFrame name that doesn't exist.")
                                            st.info("Available DataFrames: df1")
                                            st.info("Please try rephrasing your question or contact support if this persists.")
                                            
                                            # Log the error
                                            diagnostics_logger.log_code_generation(
                                                question=user_question,
                                                generated_code=pandas_code,
                                                llm_provider=llm_provider,
                                                execution_success=False,
                                                execution_error=f"DataFrame name error: {error_msg}"
                                            )
                                        else:
                                            raise e
                                            
                                    except Exception as e:
                                        # Log code execution error
                                        diagnostics_logger.log_code_generation(
                                            question=user_question,
                                            generated_code=pandas_code,
                                            llm_provider=llm_provider,
                                            execution_success=False,
                                            execution_error=str(e)
                                        )
                                        
                                        st.error(f"Error executing generated code: {str(e)}")
                                        st.error(f"Error type: {type(e).__name__}")
                                        st.error(f"Full error: {e}")
                                        
                            except Exception as e:
                                # Log code generation error
                                diagnostics_logger.log_error(
                                    component="Code_Generation",
                                    error=e,
                                    context={
                                        'user_question': user_question,
                                        'llm_provider': llm_provider
                                    }
                                )
                                
                                st.error(f"Error generating code: {str(e)}")
                                st.error(f"Error type: {type(e).__name__}")
                                st.error(f"Full error: {e}")
                                
                    except Exception as e:
                        # Log reasoning error
                        diagnostics_logger.log_error(
                            component="Reasoning_Generation",
                            error=e,
                            context={
                                'user_question': user_question,
                                'llm_provider': llm_provider
                            }
                        )
                        
                        st.error(f"Error in reasoning/code generation: {str(e)}")
                        st.error(f"Error type: {type(e).__name__}")
                        st.error(f"Full error: {e}")
                else:
                    st.warning("No context data available for code generation.")
                
            elif user_question:
                st.warning("OpenAI API key not configured. Please set it in your .env file.")
            else:
                st.info("Fetch data from Fabric to begin.")

    with tabs[1]:
        st.header("SQL Editor")
        
        if not ODBC_AVAILABLE:
            st.warning("SQL Editor functionality is limited because ODBC drivers are not installed.")
            st.info("To enable full SQL functionality, rebuild the Docker image with ODBC support:")
            st.code("docker build -f Dockerfile.docker -t rag-fabric-app .")
            st.info("Or use the alternative build method:")
            st.code("docker build -f Dockerfile.docker.alternative -t rag-fabric-app .")
            st.stop()
        
        # Check if we have a valid access token
        if not access_token:
            st.error("No valid access token available. Please login again.")
            st.stop()
        
        st.info("You are connected to the Fabric SQL endpoint using your Azure credentials. Enter a SQL query below and click 'Run Query'.")
        
        # Test connection
        if st.button("Test Connection"):
            if sql_manager.test_connection(access_token):
                st.success("✅ Database connection successful!")
            else:
                st.error("❌ Database connection failed!")
        
        # SQL Query Input
        sql_query = st.text_area("Enter your SQL query:", height=200)
        if st.button("Run Query"):
            if sql_query:
                try:
                    start_time = time.time()
                    
                    # Performance optimization: Check cache first
                    cache_key = f"sql_{hash(sql_query)}"
                    cached_result = performance_optimizer.get_cached_dataframe(cache_key)
                    
                    if cached_result is not None:
                        df = cached_result
                        st.info("📋 Using cached result")
                    else:
                        df = sql_manager.execute_query(access_token, sql_query)
                        # Cache the result for 1 hour
                        performance_optimizer.cache_dataframe(cache_key, df, ttl=3600)
                    
                    execution_time = time.time() - start_time
                    
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
                    
                    # Log successful SQL query
                    diagnostics_logger.log_sql_query(
                        sql_query=sql_query,
                        execution_time=execution_time,
                        row_count=len(df)
                    )
                    
                except Exception as e:
                    # Log SQL query error
                    diagnostics_logger.log_error(
                        component="SQL_Editor",
                        error=e,
                        context={'sql_query': sql_query},
                        user_message="Error executing SQL query"
                    )
                    
                    st.error(f"Error executing SQL query: {e}")
                    st.error(f"Full error details: {traceback.format_exc()}")
            else:
                st.warning("Please enter a SQL query.")

    with tabs[2]:
        # Diagnostics Dashboard
        diagnostics_dashboard.render_dashboard()
        
    with tabs[3]:
        # Performance Monitoring Dashboard
        st.header("📊 Performance Monitoring")
        
        # Get current performance metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("CPU Usage", f"{performance_optimizer.monitor_performance()['cpu_percent']:.1f}%")
        
        with col2:
            st.metric("Memory Usage", f"{performance_optimizer.monitor_performance()['memory_percent']:.1f}%")
        
        with col3:
            st.metric("Disk Usage", f"{performance_optimizer.monitor_performance()['disk_percent']:.1f}%")
        
        # Cache statistics
        st.subheader("Cache Statistics")
        cache_info = {
            "Total cached items": len(performance_optimizer.cache),
            "Cache TTL": f"{performance_optimizer.cache_ttl} seconds",
            "Last cleanup": f"{performance_optimizer.last_cleanup:.0f} seconds ago"
        }
        
        for key, value in cache_info.items():
            st.text(f"{key}: {value}")
        
        # Performance actions
        st.subheader("Performance Actions")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("🧹 Clear Cache"):
                performance_optimizer.cache.clear()
                st.success("Cache cleared!")
        
        with col2:
            if st.button("💾 Optimize Memory"):
                performance_optimizer.optimize_memory_usage()
                st.success("Memory optimized!")
        
        with col3:
            if st.button("📊 Refresh Metrics"):
                st.rerun()
        
        # Performance logs
        st.subheader("Recent Performance Logs")
        try:
            with open('/opt/rag-app/logs/performance.log', 'r') as f:
                logs = f.readlines()[-10:]  # Last 10 lines
                for log in logs:
                    st.text(log.strip())
        except FileNotFoundError:
            st.info("Performance logs not available yet.") 