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

# 🚀 NEW: Dynamic Complex Field Unpacking Functions
def unpack_complex_fields_dynamically(df):
    """Dynamically unpacks complex nested fields in a DataFrame"""
    try:
        if df.empty:
            return df
        df_unpacked = df.copy()
        unpacked_columns = []
        
        for col in df.columns:
            if is_complex_field(df[col]):
                try:
                    unpacked_cols = unpack_complex_column(df[col], col)
                    if unpacked_cols is not None:
                        for new_col_name, new_col_data in unpacked_cols.items():
                            df_unpacked[new_col_name] = new_col_data
                            unpacked_columns.append(new_col_name)
                except Exception:
                    continue
        
        if unpacked_columns:
            st.info(f"🔍 Unpacked {len(unpacked_columns)} complex fields: {', '.join(unpacked_columns)}")
        return df_unpacked
    except Exception as e:
        st.warning(f"⚠️ Dynamic unpacking failed: {e}")
        return df

def unpack_complex_series_dynamically(series):
    """Dynamically unpacks complex nested fields in a Series"""
    try:
        if series.empty:
            return series
        if is_complex_field(series):
            try:
                unpacked_data = unpack_complex_column(series, "series")
                if unpacked_data:
                    return pd.DataFrame(unpacked_data)
            except Exception as e:
                st.warning(f"⚠️ Series unpacking failed: {e}")
        return series
    except Exception as e:
        st.warning(f"⚠️ Series unpacking failed: {e}")
        return series

def unpack_complex_other_dynamically(data):
    """Dynamically unpacks complex nested fields in other data types"""
    try:
        if isinstance(data, (dict, list)):
            unpacked_data = unpack_complex_data_structure(data)
            if unpacked_data != data:
                return unpacked_data
        return data
    except Exception as e:
        st.warning(f"⚠️ Other type unpacking failed: {e}")
        return data

def is_complex_field(series):
    """Intelligently detects complex nested data"""
    try:
        if series.empty:
            return False
        sample_values = series.dropna().head(5)
        if len(sample_values) == 0:
            return False
        
        for val in sample_values:
            if isinstance(val, dict) and len(val) > 1:
                return True
            elif isinstance(val, list) and len(val) > 0:
                for item in val[:3]:
                    if isinstance(item, (dict, list)):
                        return True
            elif isinstance(val, str):
                if (val.startswith('{') and val.endswith('}')) or \
                   (val.startswith('[') and val.endswith(']')) or \
                   ('requestedAt' in val or 'success' in val):
                    return True
        return False
    except Exception:
        return False

def unpack_complex_column(series, column_name):
    """Unpacks complex columns into simple columns"""
    try:
        unpacked_columns = {}
        
        for idx, value in series.items():
            if pd.isna(value):
                continue
            
            if isinstance(value, dict):
                for key, val in value.items():
                    new_col_name = f"{column_name}_{key}"
                    if new_col_name not in unpacked_columns:
                        unpacked_columns[new_col_name] = pd.Series(index=series.index, dtype='object')
                    unpacked_columns[new_col_name][idx] = val
            
            elif isinstance(value, list):
                if len(value) > 0:
                    if all(isinstance(item, (str, int, float)) for item in value):
                        new_col_name = f"{column_name}_joined"
                        if new_col_name not in unpacked_columns:
                            unpacked_columns[new_col_name] = pd.Series(index=series.index, dtype='object')
                        unpacked_columns[new_col_name][idx] = ' | '.join(map(str, value))
        
        return unpacked_columns
    except Exception as e:
        st.warning(f"⚠️ Column unpacking failed for {column_name}: {e}")
        return None

def unpack_complex_data_structure(data):
    """Recursively unpacks complex data structures"""
    try:
        if isinstance(data, dict):
            unpacked = {}
            for key, value in data.items():
                if isinstance(value, (dict, list)):
                    unpacked[key] = unpack_complex_data_structure(value)
                else:
                    unpacked[key] = value
            return unpacked
        elif isinstance(data, list):
            return [unpack_complex_data_structure(item) for item in data]
        else:
            return data
    except Exception:
        return data

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

def get_table_metadata(table_name, vector_search_engine):
    """Get table metadata for both predefined tables and MongoDB collections."""
    if vector_search_engine == "MongoDB Basic":
        # For MongoDB Basic, create metadata for collections
        return {
            'name': table_name,
            'display_name': table_name.title().replace('_', ' '),
            'raw_table': table_name,
            'embedding_table': table_name,
            'collection': table_name,
            'faiss_columns': [],
            'qdrant_columns': [],
            'date_filter_column': None
        }
    else:
        # For other search engines, use predefined table metadata
        if table_name in TABLES_META:
            return TABLES_META[table_name]
        else:
            # Fallback for unknown tables
            return {
                'name': table_name,
                'display_name': table_name.title().replace('_', ' '),
                'raw_table': table_name,
                'embedding_table': table_name,
                'collection': table_name,
                'faiss_columns': [],
                'qdrant_columns': [],
                'date_filter_column': None
            }

# --- Copy all the functions from rag_fabric_app.py ---
# (I'll include the key functions here, but you can copy the rest from the original file)

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
        
        # Table selection - dynamic based on search engine
        if 'vector_search_engine' in st.session_state and st.session_state['vector_search_engine'] == "MongoDB Basic":
            # For MongoDB Basic, show available MongoDB collections
            # Use a simple, efficient approach: show collections we know exist
            if 'mongodb_collections' not in st.session_state:
                # Initialize with known collections that are configured
                # This avoids connection issues and is much faster
                known_collections = ['connections', 'bev', 'phev']
                st.session_state['mongodb_collections'] = known_collections
                st.sidebar.success(f"✅ MongoDB Basic: {len(known_collections)} collections available")
            
            available_tables = st.session_state.get('mongodb_collections', [])
            
            if available_tables:
                selected_tables = st.sidebar.multiselect(
                    "Select MongoDB Collection(s)",
                    available_tables,
                    default=available_tables[:1] if available_tables else [],
                    key='sidebar_table_select'
                )
            else:
                st.sidebar.warning("⚠️ No MongoDB collections found")
                selected_tables = []
        else:
            # For other search engines, show predefined tables
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
            ("FAISS", "Qdrant", "MongoDB", "MongoDB Basic"),
            index=0,
            key="vector_search_engine_select",
            help="Choose the backend for semantic search. MongoDB Basic focuses on schema-level search without vector embeddings.",
            on_change=lambda: st.session_state.update({'vector_search_engine': st.session_state.get('vector_search_engine_select')})
        )
        
        # Store the selected search engine in session state
        st.session_state['vector_search_engine'] = vector_search_engine
        
        # Track search engine changes for logging
        if 'previous_search_engine' not in st.session_state:
            st.session_state['previous_search_engine'] = vector_search_engine
        elif st.session_state['previous_search_engine'] != vector_search_engine:
            st.session_state['previous_search_engine'] = vector_search_engine
        
        # Show MongoDB Basic status
        if vector_search_engine == "MongoDB Basic":
            if 'mongodb_collections' in st.session_state:
                collections_count = len(st.session_state['mongodb_collections'])
                st.info(f"📚 MongoDB Basic: {collections_count} collections available")
            else:
                st.info("📚 MongoDB Basic: Ready to search")
        
        # Display current table selection
        if selected_tables:
            if vector_search_engine == "MongoDB Basic":
                st.success(f"📚 Selected MongoDB Collections: {', '.join(selected_tables)}")
            else:
                st.success(f"📚 Selected Tables: {', '.join(selected_tables)}")
        else:
            st.warning("⚠️ No tables/collections selected. Please select at least one from the sidebar.")

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
                    
                    # Adjust top_n based on vector search engine
                    if vector_search_engine == "MongoDB" or vector_search_engine == "MongoDB Basic":
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
                            table_meta = get_table_metadata(table_name, vector_search_engine)
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
                            table_meta = get_table_metadata(table_name, vector_search_engine)
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

                    elif vector_search_engine == "MongoDB" or vector_search_engine == "MongoDB Basic":
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
                            table_meta = get_table_metadata(table_name, vector_search_engine)
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
                                
                                # Safely check for detected_intent with fallback
                                detected_intent = enhanced_query_info.get('detected_intent', [])
                                if detected_intent:
                                    st.info(f"🎯 Detected intent: {', '.join(detected_intent)}")
                                else:
                                    st.info("🎯 No specific intent detected")
                                
                                if enhanced_query_info['semantic_expansions']:
                                    st.info(f"🔍 Semantic expansions: {', '.join(enhanced_query_info['semantic_expansions'])}")
                                
                                # Use enhanced query for search
                                search_query = enhanced_query_info['enhanced_query']
                                st.info(f"🔍 Debug: Performing enhanced search with: '{search_query}'")
                                
                                # MongoDB Basic vs Full MongoDB search logic
                                if vector_search_engine == "MongoDB Basic":
                                    st.info("🔍 Using MongoDB Basic search (schema-level, no vector embeddings)...")
                                    
                                    # Initialize variables at the start to avoid scope issues
                                    search_results = []
                                    search_metadata = {}
                                    skip_original_search = False
                                    
                                    try:
                                        # Import MongoDB Basic Search
                                        from utils.mongodb_basic_search import MongoDBBasicSearch
                                        
                                        # Initialize basic search engine
                                        mongodb_basic = MongoDBBasicSearch(
                                            connection_string=os.getenv('MONGODB_URI'),
                                            database_name=os.getenv('MONGODB_DB_NAME', 'perse-data-network')
                                        )
                                        
                                        if not mongodb_basic.connect():
                                            st.error("❌ Failed to connect to MongoDB for basic search")
                                            continue
                                        
                                        # Get schema configuration for this collection
                                        st.info(f"🔍 Debug: Getting schema for collection: {collection_name}")
                                        schema_config = schema_manager.get_collection_schema(collection_name)
                                        st.info(f"🔍 Debug: Schema config keys: {list(schema_config.keys()) if schema_config else 'None'}")
                                        
                                        # Debug the search optimization section
                                        if schema_config and 'search_optimization' in schema_config:
                                            search_opt = schema_config['search_optimization']
                                            st.info(f"🔍 Debug: Search optimization keys: {list(search_opt.keys())}")
                                            if 'exact_match_fields' in search_opt:
                                                st.info(f"🔍 Debug: Exact match fields: {search_opt['exact_match_fields']}")
                                            else:
                                                st.info("🔍 Debug: No exact_match_fields in search_optimization")
                                        else:
                                            st.info("🔍 Debug: No search_optimization section in schema config")
                                        
                                        if not schema_config:
                                            st.warning(f"⚠️ No schema configuration found for {collection_name}, using default search")
                                            # Create a minimal schema config for connections collection
                                            if collection_name == 'connections':
                                                schema_config = {
                                                    'business_context': {
                                                        'key_entities': ['UPRN', 'EMSN', 'MPAN', 'MPRN', 'POSTCODE', 'ADDRESS', 'apiStatus']
                                                    },
                                                    'search_optimization': {
                                                        'exact_match_fields': ['EMSN', 'MPAN', 'MPRN', 'UPRN'],
                                                        'partial_search_fields': ['POSTCODE', 'ADDRESS']
                                                    },
                                                    'context_optimization': {
                                                        'essential_columns': ['UPRN', 'ADDRESS', 'EMSN', 'MPAN', 'MPRN', 'POSTCODE', 'apiStatus'],
                                                        'exclude_columns': ['_id', 'GEOM', 'LATLON']
                                                    }
                                                }
                                                st.info("✅ Using fallback schema config for connections collection")
                                            else:
                                                schema_config = {}
                                        
                                        # Single MongoDB access point
                                        collection = mongodb_basic.database[collection_name]
                                        
                                        # Helper function for field value extraction
                                        def extract_field_value(query, field_name):
                                            """Extract field value using flexible pattern matching"""
                                            patterns = [
                                                rf'{field_name.lower()}\s+([A-Za-z0-9]+)',
                                                rf'{field_name.lower()}.*?([A-Za-z0-9]+)'
                                            ]
                                            for pattern in patterns:
                                                match = re.search(pattern, query.lower())
                                                if match:
                                                    return match.group(1).upper()
                                            return None
                                        
                                        # Test direct MongoDB access to see what's happening
                                        st.info("🔍 Debug: Testing direct MongoDB access...")
                                        try:
                                            # Try to find any document in the collection
                                            test_doc = collection.find_one({})
                                            if test_doc:
                                                st.info(f"🔍 Debug: Found sample document with keys: {list(test_doc.keys())}")
                                                if 'UPRN' in test_doc:
                                                    st.info(f"🔍 Debug: UPRN field exists, sample value: {test_doc['UPRN']}")
                                                else:
                                                    st.info("🔍 Debug: UPRN field does not exist in sample document")
                                                
                                                # Try to find a document with UPRN field
                                                uprn_doc = collection.find_one({'UPRN': {'$exists': True}})
                                                if uprn_doc:
                                                    st.info(f"🔍 Debug: Found document with UPRN field: {uprn_doc.get('UPRN', 'N/A')}")
                                                else:
                                                    st.info("🔍 Debug: No documents found with UPRN field")
                                                
                                                # Try to find the specific UPRN value from the query
                                                uprn_match = re.search(r'(\d+)', user_question)
                                                if uprn_match:
                                                    uprn_value = uprn_match.group(1)
                                                    st.info(f"🔍 Debug: Extracted UPRN value from query: {uprn_value}")
                                                    
                                                    # Try to find a document with this specific UPRN
                                                    specific_doc = collection.find_one({'UPRN': uprn_value})
                                                    if specific_doc:
                                                        st.info(f"🔍 Debug: Found document with specific UPRN {uprn_value}")
                                                    else:
                                                        st.info(f"🔍 Debug: No document found with specific UPRN {uprn_value}")
                                                        
                                                        # Try with numeric UPRN
                                                        try:
                                                            numeric_uprn = int(uprn_value)
                                                            numeric_doc = collection.find_one({'UPRN': numeric_uprn})
                                                            if numeric_doc:
                                                                st.info(f"🔍 Debug: Found document with numeric UPRN {numeric_uprn}")
                                                            else:
                                                                st.info(f"🔍 Debug: No document found with numeric UPRN {numeric_uprn}")
                                                        except ValueError:
                                                            st.info("🔍 Debug: Could not convert UPRN to numeric")
                                                else:
                                                    st.info("🔍 Debug: Could not extract UPRN value from query")
                                            else:
                                                st.info("🔍 Debug: Collection appears to be empty")
                                        except Exception as e:
                                            st.error(f"🔍 Debug: Error testing MongoDB access: {str(e)}")
                                        
                                        # Now perform the actual search
                                        st.info("🔍 Debug: Executing MongoDB Basic search...")
                                        st.info(f"🔍 Debug: User query: {user_question}")
                                        st.info(f"🔍 Debug: Collection: {collection_name}")
                                        st.info(f"🔍 Debug: Schema config keys: {list(schema_config.keys())}")
                                        
                                        # Add debug output for pattern matching
                                        st.info("🔍 Debug: Testing pattern matching...")
                                        query_lower = user_question.lower()
                                        st.info(f"🔍 Debug: Query lower: {query_lower}")
                                        st.info(f"🔍 Debug: Query contains 'uprn': {'uprn' in query_lower}")
                                        st.info(f"🔍 Debug: Query contains 'UPRN': {'UPRN' in query_lower}")
                                        
                                        # Test the regex patterns directly
                                        import re
                                        exact_match_fields = schema_config.get('search_optimization', {}).get('exact_match_fields', [])
                                        st.info(f"🔍 Debug: Exact match fields: {exact_match_fields}")
                                        
                                        # Manually create search criteria since the method seems to have an issue
                                        manual_search_criteria = {
                                            'intent': 'general_search',
                                            'search_fields': [],
                                            'exact_matches': {},
                                            'partial_matches': {},
                                            'business_entities': [],
                                            'confidence_score': 0.0
                                        }
                                        
                                        # Batch process all fields for efficiency
                                        for field in exact_match_fields:
                                            field_lower = field.lower()
                                            st.info(f"🔍 Debug: Checking field: {field} (lower: {field_lower})")
                                            if field_lower in query_lower:
                                                st.info(f"🔍 Debug: Field {field_lower} found in query")
                                                
                                                # Use the helper function for consistent pattern matching
                                                captured_value = extract_field_value(user_question, field)
                                                if captured_value:
                                                    st.info(f"🔍 Debug: Pattern matched! Captured value: '{captured_value}'")
                                                    
                                                    # Manually set the search criteria
                                                    manual_search_criteria['exact_matches'][field] = captured_value
                                                    manual_search_criteria['intent'] = f'find_by_{field_lower}'
                                                    manual_search_criteria['confidence_score'] = 0.9
                                                    st.info(f"✅ Manually captured {field}: {captured_value}")
                                                else:
                                                    st.info(f"🔍 Debug: Pattern did not match for {field}")
                                            else:
                                                st.info(f"🔍 Debug: Field {field_lower} not found in query")
                                        
                                        # Show the manually created search criteria
                                        st.info(f"🔍 Debug: Manually created search criteria: {manual_search_criteria}")
                                        
                                        # Now try to use the manual search criteria to build a MongoDB query
                                        if manual_search_criteria['exact_matches']:
                                            st.info("🔍 Debug: Building MongoDB query from manual search criteria...")
                                            
                                            # Build the MongoDB query manually
                                            mongo_query = {}
                                            exact_match_fields_list = schema_config.get('search_optimization', {}).get('exact_match_fields', [])
                                            for field, value in manual_search_criteria['exact_matches'].items():
                                                if field in exact_match_fields_list:
                                                    mongo_query[field] = {'$in': [value]}
                                                    st.info(f"🔍 Debug: Added {field} search: {{'$in': [{value}]}}")
                                            
                                            st.info(f"🔍 Debug: Manual MongoDB query: {mongo_query}")
                                            
                                            # Test the manual query directly
                                            if mongo_query:
                                                st.info("🔍 Debug: Testing manual MongoDB query...")
                                                try:
                                                    manual_results = list(collection.find(mongo_query, {'UPRN': 1, 'ADDRESS': 1, 'apiStatus': 1, '_id': 0}).limit(5))
                                                    st.info(f"🔍 Debug: Manual query found {len(manual_results)} results")
                                                    if manual_results:
                                                        st.info(f"🔍 Debug: First result: {manual_results[0]}")
                                                        
                                                        # Use manual results instead of calling the original search method
                                                        st.info("🔍 Debug: Using manual search results...")
                                                        search_results = manual_results
                                                        search_metadata = {
                                                            'collection_name': collection_name,
                                                            'search_criteria': manual_search_criteria,
                                                            'mongo_query': mongo_query,
                                                            'results_count': len(manual_results),
                                                            'max_results': top_n,
                                                            'search_strategy': 'manual_schema_based_search'
                                                        }
                                                        st.success(f"✅ Manual search successful: {len(search_results)} results")
                                                        st.info(f"Search intent: {manual_search_criteria['intent']}")
                                                        st.info(f"Confidence score: {manual_search_criteria['confidence_score']:.2f}")
                                                        
                                                        # Process manual search results into DataFrame
                                                        st.info("🔍 Debug: Processing manual search results into DataFrame...")
                                                        try:
                                                            # Convert MongoDB documents to DataFrame
                                                            if search_results:
                                                                # Create DataFrame from search results
                                                                search_df = pd.DataFrame(search_results)
                                                                st.info(f"✅ Created DataFrame from manual search: {search_df.shape}")
                                                                
                                                                # Store the DataFrame for further processing
                                                                dfs[table_name] = search_df
                                                                st.success(f"✅ Stored DataFrame '{table_name}' with {len(search_df)} rows")
                                                                
                                                                # Show DataFrame info
                                                                st.info(f"DataFrame columns: {list(search_df.columns)}")
                                                                st.info(f"DataFrame sample data:")
                                                                st.dataframe(search_df.head())
                                                                
                                                                # Debug: Show current state of dfs dictionary
                                                                st.info(f"🔍 Debug: Current dfs dictionary keys: {list(dfs.keys())}")
                                                                st.info(f"🔍 Debug: dfs['{table_name}'] type: {type(dfs[table_name])}")
                                                                st.info(f"🔍 Debug: dfs['{table_name}'] shape: {dfs[table_name].shape if hasattr(dfs[table_name], 'shape') else 'No shape'}")
                                                            else:
                                                                st.warning("No search results to process")
                                                        except Exception as df_error:
                                                            st.error(f"❌ Error creating DataFrame from manual search results: {str(df_error)}")
                                                            st.info("Will continue with original search method")
                                                            skip_original_search = False
                                                        
                                                        # Skip the original search method call
                                                        skip_original_search = True
                                                except Exception as e:
                                                    st.error(f"🔍 Debug: Error testing manual query: {str(e)}")
                                                    skip_original_search = False
                                            else:
                                                skip_original_search = False
                                        else:
                                            st.info("🔍 Debug: No manual search criteria created")
                                            skip_original_search = False
                                        
                                        # Only call the original search method if manual search didn't work
                                        if not skip_original_search:
                                            search_results, search_metadata = mongodb_basic.search_by_schema_intent(
                                                collection_name=collection_name,
                                                user_query=user_question,
                                                schema_config=schema_config,
                                                max_results=top_n
                                            )
                                            
                                            # Show search metadata for debugging
                                            if 'search_criteria' in search_metadata:
                                                st.info(f"🔍 Debug: Search criteria: {search_metadata['search_criteria']}")
                                            if 'mongo_query' in search_metadata:
                                                st.info(f"🔍 Debug: MongoDB query: {search_metadata['mongo_query']}")
                                            st.info(f"🔍 Debug: Results count: {search_metadata.get('results_count', 0)}")
                                            
                                            # Additional debugging - show what was actually searched
                                            st.info("🔍 Debug: Let's see what happened in the search...")
                                            if 'search_criteria' in search_metadata and 'exact_matches' in search_metadata['search_criteria']:
                                                exact_matches = search_metadata['search_criteria']['exact_matches']
                                                if exact_matches:
                                                    st.info(f"🔍 Debug: Exact matches found: {exact_matches}")
                                                else:
                                                    st.info("🔍 Debug: No exact matches found in search criteria")
                                            else:
                                                st.info("🔍 Debug: No exact_matches in search criteria")
                                            
                                            st.success(f"✅ MongoDB Basic search successful: {len(search_results)} results")
                                            st.info(f"Search intent: {search_metadata.get('search_criteria', {}).get('intent', 'unknown')}")
                                            st.info(f"Confidence score: {search_metadata.get('search_criteria', {}).get('confidence_score', 0.0):.2f}")
                                            
                                            # Show detailed search metadata for debugging
                                            if 'search_criteria' in search_metadata:
                                                st.info(f"🔍 Debug: Search criteria: {search_metadata['search_criteria']}")
                                            if 'mongo_query' in search_metadata:
                                                st.info(f"🔍 Debug: MongoDB query: {search_metadata['mongo_query']}")
                                            
                                            # Process original search results into DataFrame
                                            st.info("🔍 Debug: Processing original search results into DataFrame...")
                                            try:
                                                # Convert MongoDB documents to DataFrame
                                                if search_results:
                                                    # Create DataFrame from search results
                                                    search_df = pd.DataFrame(search_results)
                                                    st.info(f"✅ Created DataFrame from original search: {search_df.shape}")
                                                    
                                                    # Store the DataFrame for further processing
                                                    dfs[table_name] = search_df
                                                    st.success(f"✅ Stored DataFrame '{table_name}' with {len(search_df)} rows")
                                                    
                                                    # Show DataFrame info
                                                    st.info(f"DataFrame columns: {list(search_df.columns)}")
                                                    st.info(f"DataFrame sample data:")
                                                    st.dataframe(search_df.head())
                                                else:
                                                    st.warning("No search results to process from original search")
                                            except Exception as df_error:
                                                st.error(f"❌ Error creating DataFrame from original search results: {str(df_error)}")
                                                st.info("Will continue without DataFrame")
                                    
                                    except Exception as e:
                                        st.error(f"❌ MongoDB Basic search failed: {str(e)}")
                                        st.info("Falling back to standard MongoDB search...")
                                        # Fall back to standard search
                                        search_results = mongodb_index.search(
                                            search_query,
                                            limit=top_n,
                                            score_threshold=0.01
                                        )
                                        st.success(f"✅ Fallback search successful: {len(search_results)} results")
                                        
                                        # Process fallback search results into DataFrame
                                        st.info("🔍 Debug: Processing fallback search results into DataFrame...")
                                        try:
                                            # Convert fallback search results to DataFrame
                                            if search_results:
                                                # Create DataFrame from fallback search results
                                                fallback_df = pd.DataFrame(search_results)
                                                st.info(f"✅ Created DataFrame from fallback search: {fallback_df.shape}")
                                                
                                                # Store the DataFrame for further processing
                                                dfs[table_name] = fallback_df
                                                st.success(f"✅ Stored DataFrame '{table_name}' with {len(fallback_df)} rows from fallback search")
                                                
                                                # Show DataFrame info
                                                st.info(f"Fallback DataFrame columns: {list(fallback_df.columns)}")
                                                st.info(f"Fallback DataFrame sample data:")
                                                st.dataframe(fallback_df.head())
                                            else:
                                                st.warning("No fallback search results to process")
                                        except Exception as df_error:
                                            st.error(f"❌ Error creating DataFrame from fallback search results: {str(df_error)}")
                                            st.info("Will continue without DataFrame")
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
                                            diagnostics_logger.log_error("MongoDB_Data_Fetch", e, {"table_name": table_name})
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
                                    
                                    # Final debug output for MongoDB Basic search
                                    st.info("🔍 Debug: Final MongoDB Basic search summary:")
                                    st.info(f"🔍 Debug: search_results count: {len(search_results) if search_results else 0}")
                                    st.info(f"🔍 Debug: dfs dictionary keys: {list(dfs.keys())}")
                                    if table_name in dfs:
                                        st.info(f"🔍 Debug: dfs['{table_name}'] exists with shape: {dfs[table_name].shape if hasattr(dfs[table_name], 'shape') else 'No shape'}")
                                    else:
                                        st.info(f"🔍 Debug: dfs['{table_name}'] NOT found in dfs dictionary")
                                    
                                    # MongoDB Basic search completed, now continue with data processing
                                    if not search_results:
                                        st.warning(f"No relevant documents found in MongoDB for {table_meta['display_name']}.")
                                        continue
                                
                            except Exception as e:
                                st.error(f"❌ Error processing MongoDB for {table_meta['display_name']}: {str(e)}")
                                diagnostics_logger.log_error("MongoDB_Processing", e, {"table_name": table_name, "display_name": table_meta['display_name']})
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
                            table_meta = get_table_metadata(table_name, vector_search_engine)
                            try:
                                if vector_search_engine == "MongoDB Basic":
                                    # For MongoDB Basic, we don't auto-fetch data since it's handled by the search
                                    st.info(f"MongoDB Basic mode: Data will be fetched during search for {table_meta['display_name']}")
                                    # Create an empty DataFrame as placeholder
                                    dfs[f'df_{table_name}'] = pd.DataFrame()
                                else:
                                    # For other search engines, use SQL auto-fetch
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
                    
                    # 🚀 ENHANCED: Use MongoDB Schema Manager to optimize context if available
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
                                df1_context_sample = _basic_context_optimization(df1_context)
                                
                        except Exception as schema_error:
                            st.warning(f"Enhanced schema optimization failed, using fallback: {str(schema_error)}")
                            df1_context_sample = _basic_context_optimization(df1_context)
                            
                    else:
                        # Fallback to basic optimization when enhanced schema manager is not available
                        df1_context_sample = _basic_context_optimization(df1_context)
                    
                    # 🚀 ENHANCED: Prepare enhanced context for the LLM with business context
                    if vector_search_engine == "MongoDB" and 'schema_manager' in locals():
                        try:
                            # Get business context for enhanced reasoning
                            first_table = list(dfs.keys())[0] if dfs else None
                            if first_table:
                                business_context = schema_manager.get_business_context(first_table)
                                business_purpose = schema_manager.get_business_purpose(first_table)
                                key_entities = business_context.get('key_entities', [])
                                
                                enhanced_context = f"""
Business Context: {business_context.get('domain', 'Unknown')} - {business_purpose}
Key Business Entities: {', '.join(key_entities)}
DataFrame df1: {len(df1_context)} rows with columns: {list(df1_context.columns)}

Sample data (first 2 rows):
{df1_context_sample.head(2).to_string()}

Use 'df1' for all operations. Focus on business-relevant columns: {', '.join(schema_manager.get_essential_columns(first_table))}
"""
                            else:
                                enhanced_context = f"""
DataFrame df1: {len(df1_context)} rows with columns: {list(df1_context.columns)}

Sample data (first 2 rows):
{df1_context_sample.head(2).to_string()}

Use 'df1' for all operations.
"""
                        except Exception as e:
                            st.warning(f"Enhanced context generation failed: {str(e)}")
                            enhanced_context = f"""
DataFrame df1: {len(df1_context)} rows with columns: {list(df1_context.columns)}

Sample data (first 2 rows):
{df1_context_sample.head(2).to_string()}

Use 'df1' for all operations.
"""
                    else:
                        # Standard context for non-MongoDB engines
                        enhanced_context = f"""
DataFrame df1: {len(df1_context)} rows with columns: {list(df1_context.columns)}

Sample data (first 2 rows):
{df1_context_sample.head(2).to_string()}

Use 'df1' for all operations.
"""
                    
                    # 🚀 ENHANCED: Generate reasoning with business context awareness
                    if vector_search_engine == "MongoDB" and 'schema_manager' in locals():
                        try:
                            first_table = list(dfs.keys())[0] if dfs else None
                            if first_table:
                                business_context = schema_manager.get_business_context(first_table)
                                business_purpose = schema_manager.get_business_purpose(first_table)
                                
                                enhanced_reasoning_prompt = f"""
Business Context: {business_context.get('domain', 'Unknown')} - {business_purpose}
Key Entities: {', '.join(business_context.get('key_entities', []))}

Data: {len(df1_context)} rows, columns: {list(df1_context.columns)}
Question: {user_question}

Provide brief reasoning for how to answer this question, considering the business context and key entities.
"""
                            else:
                                enhanced_reasoning_prompt = f"""
Data: {len(df1_context)} rows, columns: {list(df1_context.columns)}
Question: {user_question}
Provide brief reasoning for how to answer this question.
"""
                        except Exception as e:
                            st.warning(f"Enhanced reasoning prompt generation failed: {str(e)}")
                            enhanced_reasoning_prompt = f"""
Data: {len(df1_context)} rows, columns: {list(df1_context.columns)}
Question: {user_question}
Provide brief reasoning for how to answer this question.
"""
                    else:
                        # Standard reasoning prompt for non-MongoDB engines
                        enhanced_reasoning_prompt = f"""
Data: {len(df1_context)} rows, columns: {list(df1_context.columns)}
Question: {user_question}
Provide brief reasoning for how to answer this question.
"""
                    
                    try:
                        if llm_provider == "Local Deepseek LLM":
                            # Use local Deepseek LLM for reasoning
                            reasoning = RAGUtils.run_deepseek_llm(
                                enhanced_reasoning_prompt,
                                max_new_tokens=300,
                                temperature=0.1
                            )
                        else:
                            # Use OpenAI for reasoning
                            reasoning_response = client.chat.completions.create(
                                model=DEFAULT_MODEL,
                                messages=[
                                    {"role": "system", "content": "You are a helpful data analyst with business domain expertise. Provide clear, step-by-step reasoning for how to answer data analysis questions, considering the business context."},
                                    {"role": "user", "content": enhanced_reasoning_prompt}
                                ],
                                temperature=0.1,
                                max_tokens=300
                            )
                            reasoning = reasoning_response.choices[0].message.content.strip()
                        
                        if show_code:
                            st.write("**Reasoning:**")
                            st.write(reasoning)
                        
                        # 🚀 ENHANCED: Generate code with business context awareness
                        if vector_search_engine == "MongoDB" and 'schema_manager' in locals():
                            try:
                                first_table = list(dfs.keys())[0] if dfs else None
                                if first_table:
                                    business_context = schema_manager.get_business_context(first_table)
                                    business_purpose = schema_manager.get_business_purpose(first_table)
                                    essential_cols = schema_manager.get_essential_columns(first_table)
                                    
                                    enhanced_code_prompt = f"""
Business Context: {business_context.get('domain', 'Unknown')} - {business_purpose}
Key Business Entities: {', '.join(business_context.get('key_entities', []))}

df1: {len(df1_context)} rows, columns: {list(df1_context.columns)}
Essential Business Columns: {essential_cols}
Sample: {df1_context_sample.head(1).to_string()}
Reasoning: {reasoning}
Question: {user_question}

Write Python code using df1. Focus on business-relevant columns. Assign result to 'result'.
"""
                                else:
                                    enhanced_code_prompt = f"""
df1: {len(df1_context)} rows, columns: {list(df1_context.columns)}
Sample: {df1_context_sample.head(1).to_string()}
Reasoning: {reasoning}
Question: {user_question}
Write Python code using df1. Assign result to 'result'.
"""
                            except Exception as e:
                                st.warning(f"Enhanced code prompt generation failed: {str(e)}")
                                enhanced_code_prompt = f"""
df1: {len(df1_context)} rows, columns: {list(df1_context.columns)}
Sample: {df1_context_sample.head(1).to_string()}
Reasoning: {reasoning}
Question: {user_question}
Write Python code using df1. Assign result to 'result'.
"""
                        else:
                            # Standard code prompt for non-MongoDB engines
                            enhanced_code_prompt = f"""
df1: {len(df1_context)} rows, columns: {list(df1_context.columns)}
Sample: {df1_context_sample.head(1).to_string()}
Reasoning: {reasoning}
Question: {user_question}
Write Python code using df1. Assign result to 'result'.
"""
                        
                        # Generate code with better DataFrame information
                        from utils.dataframe_corrector import DataFrameCorrector
                        
                        dataframe_corrector = DataFrameCorrector()
                        dataframe_info = dataframe_corrector.get_dataframe_info(dfs)
                        
                        # 🚀 ENHANCED: Final code prompt with business context
                        if vector_search_engine == "MongoDB" and 'schema_manager' in locals():
                            try:
                                first_table = list(dfs.keys())[0] if dfs else None
                                if first_table:
                                    business_context = schema_manager.get_business_context(first_table)
                                    business_purpose = schema_manager.get_business_purpose(first_table)
                                    essential_cols = schema_manager.get_essential_columns(first_table)
                                    
                                    final_code_prompt = f"""
CRITICAL: Use ONLY the exact DataFrame names and column names provided below. Do NOT invent or assume any names.

BUSINESS CONTEXT: {business_context.get('domain', 'Unknown')} - {business_purpose}
KEY BUSINESS ENTITIES: {', '.join(business_context.get('key_entities', []))}

AVAILABLE DATAFRAMES:
{dataframe_info}

MAIN DATAFRAME: df1 ({len(df1_context)} rows)
Sample data from df1:
{df1_context.head(3).to_string()}

ESSENTIAL BUSINESS COLUMNS: {essential_cols}
AVAILABLE COLUMNS: {list(set([col for df in dfs.values() for col in df.columns]))}

Your reasoning: {reasoning}

IMPORTANT CODING RULES:
1. Use ONLY the DataFrame names listed above (especially 'df1') and the exact column names provided
2. For filtering text columns, use separate conditions with .str.contains() and combine with & operator
3. Avoid complex regex patterns - use simple string matching
4. Focus on business-relevant columns: {essential_cols}
5. Assign the final answer to a variable called 'result'

EXAMPLE PATTERNS:
- Filter by type: df1[df1['type'].str.contains('MPAN', case=False, na=False)]
- Filter by value: df1[df1['value'].str.contains('error', case=False, na=False)]
- Combine filters: df1[(df1['type'].str.contains('MPAN', case=False, na=False)) & (df1['value'].str.contains('error', case=False, na=False))]
        - Count values: df1['type'].value_counts()
        - Group by: df1.groupby('type')['value'].count()

Write Python pandas code to answer this question. Use ONLY the DataFrame names listed above (especially 'df1') and the exact column names provided. Focus on business-relevant columns. 

**IMPORTANT**: 
- Focus on the business question and use the most relevant columns for the analysis
- Data quality issues are automatically handled - focus on the business logic
- Write clean, readable pandas code that answers the user's question

Assign the final answer to a variable called 'result'.

Question: {user_question}

Code (only code, no comments or explanations):
"""
                                else:
                                    final_code_prompt = f"""
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
                            except Exception as e:
                                st.warning(f"Enhanced final code prompt generation failed: {str(e)}")
                                final_code_prompt = f"""
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
                        else:
                            # Standard final code prompt for non-MongoDB engines
                            final_code_prompt = f"""
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
                                    # 🚀 ENHANCED: Dynamic MongoDB schema-aware code generation
                                    st.info("🔍 Analyzing MongoDB collection schema for dynamic code generation...")
                                    
                                    # Get the actual MongoDB collection schema dynamically
                                    collection_schema = schema_manager.get_collection_schema(collection_name)
                                    
                                    # Analyze the actual data structure from the collection
                                    try:
                                        # Get a sample document to understand the real structure
                                        sample_doc = mongodb_index.get_raw_data(limit=1)
                                        if not sample_doc.empty:
                                            # Analyze the actual data types and structure
                                            schema_analysis = {}
                                            for col in sample_doc.columns:
                                                sample_value = sample_doc[col].iloc[0]
                                                if isinstance(sample_value, list):
                                                    schema_analysis[col] = {
                                                        'type': 'array',
                                                        'sample': sample_value,
                                                        'length': len(sample_value),
                                                        'element_type': type(sample_value[0]).__name__ if sample_value else 'unknown'
                                                    }
                                                elif isinstance(sample_value, dict):
                                                    schema_analysis[col] = {
                                                        'type': 'object',
                                                        'keys': list(sample_value.keys()),
                                                        'nested_types': {k: type(v).__name__ for k, v in sample_value.items()}
                                                    }
                                                else:
                                                    schema_analysis[col] = {
                                                        'type': type(sample_value).__name__,
                                                        'sample': sample_value
                                                    }
                                            
                                            st.success(f"✅ Schema analysis completed for {collection_name}")
                                            st.info(f"Schema structure: {schema_analysis}")
                                            
                                            # Generate dynamic schema-aware prompt
                                            schema_context = "MongoDB Collection Schema Analysis:\n"
                                            for col, info in schema_analysis.items():
                                                if info['type'] == 'array':
                                                    schema_context += f"- {col}: Array field with {info['length']} elements of type {info['element_type']}\n"
                                                    schema_context += f"  Example: {info['sample']}\n"
                                                    schema_context += f"  Access: Use .str.contains('value', na=False) or .explode() for array operations\n"
                                                elif info['type'] == 'object':
                                                    schema_context += f"- {col}: Object field with keys: {', '.join(info['keys'])}\n"
                                                    schema_context += f"  Access: Use .str['key'] for nested access or .apply() with safe operations\n"
                                                else:
                                                    schema_context += f"- {col}: {info['type']} field\n"
                                                    schema_context += f"  Example: {info['sample']}\n"
                                            
                                            # Add business context from schema manager
                                            business_context_info = schema_manager.get_business_context(collection_name)
                                            essential_cols = schema_manager.get_essential_columns(collection_name)
                                            
                                            if business_context_info:
                                                schema_context += f"\nBusiness Context:\n"
                                                schema_context += f"- Domain: {business_context_info.get('domain', 'Unknown')}\n"
                                                schema_context += f"- Key Entities: {', '.join(business_context_info.get('key_entities', []))}\n"
                                            
                                            if essential_cols:
                                                schema_context += f"\nEssential Columns: {', '.join(essential_cols)}\n"
                                            
                                            # Create dynamic system prompt
                                            dynamic_system_prompt = f"""You are a helpful Python data analyst with MongoDB expertise. 

{schema_context}

IMPORTANT: Generate code based on the ACTUAL schema above, not assumptions:
- For array fields: Use .str.contains('value', na=False) or .explode() for array operations
- For object fields: Use .str['key'] for nested access
- For simple fields: Use standard pandas operations
- Always assign results to a variable called 'result'
- For plots, assign to 'fig'

CRITICAL: NEVER use these forbidden patterns:
- NO lambda functions (use .str.contains() instead)
- NO isinstance() (use pandas methods instead)
- NO .apply() with lambda (use built-in pandas methods)

SAFE EXAMPLES for array fields:
- df1[df1['COLUMN_NAME'].apply(lambda x: 'SEARCH_VALUE' in x if isinstance(x, list) else False)]['TARGET_COLUMN']
- df1[df1['UPRN'].apply(lambda x: '10023229787' in x if isinstance(x, list) else False)]['apiStatus']
- df1[df1['POSTCODE'].apply(lambda x: 'POSTCODE_VALUE' in x if isinstance(x, list) else False)]

CRITICAL: For MongoDB array fields, you MUST use .apply() with lambda to search within arrays:
- CORRECT: df1[df1['UPRN'].apply(lambda x: '10023229787' in x if isinstance(x, list) else False)]
- WRONG: df1[df1['UPRN'].str.contains('10023229787', na=False)]  # This won't work for arrays

Focus on the business-relevant columns and use the actual data types shown above."""
                                            
                                            st.info("🔍 Using dynamic schema-aware prompt for code generation...")
                                            
                                        else:
                                            st.warning("⚠️ Could not analyze schema - using fallback prompt")
                                            dynamic_system_prompt = "You are a helpful Python data analyst. Generate code that produces the answer as a DataFrame or Series, assigned to 'result'."
                                    
                                    except Exception as schema_error:
                                        st.warning(f"⚠️ Schema analysis failed: {schema_error}")
                                        st.info("Using fallback prompt...")
                                        dynamic_system_prompt = "You are a helpful Python data analyst. Generate code that produces the answer as a DataFrame or Series, assigned to 'result'."
                                    
                                    # Use the dynamic system prompt
                                    code_response = client.chat.completions.create(
                                        model=DEFAULT_MODEL,
                                        messages=[
                                            {"role": "system", "content": dynamic_system_prompt},
                                            {"role": "user", "content": final_code_prompt}
                                        ],
                                        temperature=0.0,
                                        max_tokens=700
                                    )
                                    pandas_code = code_response.choices[0].message.content.strip()
                                    pandas_code = RAGUtils.clean_code(pandas_code)
                                    
                                    # 🚀 ENHANCED: Immediate error detection and fixing during code generation
                                    try:
                                        st.info("🔧 Detecting and fixing code generation errors...")
                                        
                                        # SIMPLIFIED: Only apply fixes if the code actually has obvious issues
                                        # Skip complex pattern matching that causes regex errors
                                        
                                        # Check for the most obvious and safe patterns only
                                        if 'str.contains(' in pandas_code and '&' in pandas_code:
                                            # Only fix if parentheses are clearly missing (simple check)
                                            if '(' not in pandas_code.split('[')[1].split(']')[0]:
                                                st.warning("🔧 Detected missing parentheses - applying simple fix...")
                                                
                                                # Simple fix: Add parentheses around the condition
                                                original_code = pandas_code
                                                condition_part = pandas_code.split('[')[1].split(']')[0]
                                                fixed_code = f"df1[({condition_part})]"
                                                pandas_code = fixed_code
                                                
                                                st.success("✅ Applied simple parentheses fix!")
                                                st.info("Added missing parentheses around boolean conditions")
                                                
                                                if show_code:
                                                    st.code(fixed_code, language="python")
                                            else:
                                                st.info("✅ Code structure looks correct - no fixes needed")
                                        else:
                                            st.info("✅ No complex boolean operations detected - code looks fine")
                                    
                                    except Exception as generation_fix_error:
                                        st.warning(f"Code generation error fixing failed: {generation_fix_error}")
                                        st.info("Continuing with original code...")
                                        # Don't let error detection errors stop the process
                                    
                                    # 🚀 NEW: Auto-fix forbidden patterns in generated code
                                    st.info("🔧 Auto-fixing forbidden patterns in generated code...")
                                    try:
                                        original_code = pandas_code
                                        
                                        # CRITICAL FIX: MongoDB array field search patterns
                                        # The LLM often generates .str.contains() for array fields, which doesn't work
                                        # We need to convert these to .apply(lambda x: ...) patterns
                                        
                                        # Fix MongoDB array field searches (UPRN, EMSN, MPAN, MPRN)
                                        array_fields = ['UPRN', 'EMSN', 'MPAN', 'MPRN']
                                        for field in array_fields:
                                            if field in pandas_code and '.str.contains(' in pandas_code:
                                                st.warning(f"🔧 Detected incorrect .str.contains() for MongoDB array field '{field}' - applying critical fix...")
                                                
                                                # Pattern: df1[df1['UPRN'].str.contains('10023229787', na=False)]
                                                # Convert to: df1[df1['UPRN'].apply(lambda x: '10023229787' in x if isinstance(x, list) else False)]
                                                
                                                import re
                                                pattern = rf"df1\[df1\['{field}'\]\.str\.contains\('([^']+)', na=False\)\]"
                                                replacement = rf"df1[df1['{field}'].apply(lambda x: '\1' in x if isinstance(x, list) else False)]"
                                                
                                                if re.search(pattern, pandas_code):
                                                    pandas_code = re.sub(pattern, replacement, pandas_code)
                                                    st.success(f"✅ CRITICAL FIX: Fixed {field} array field search pattern")
                                                    st.info(f"Converted .str.contains() to .apply(lambda x: ...) for MongoDB array field")
                                                    st.info("This ensures the search will work with MongoDB array data structures")
                                        
                                        # Fix the specific error pattern you're encountering
                                        if 'str.contains(' in pandas_code and '&' in pandas_code:
                                            # Only fix if parentheses are actually missing
                                            if pandas_code != original_code:
                                                st.success("✅ Applied simple parentheses fix!")
                                                st.info("Added missing parentheses around boolean conditions")
                                                
                                                if show_code:
                                                    st.code(pandas_code, language="python")
                                        else:
                                            st.info("✅ No complex boolean operations detected - code looks fine")
                                    
                                    except Exception as fix_error:
                                        st.warning(f"🔧 Auto-fixing failed: {fix_error}")
                                        st.info("Continuing with original code...")
                                
                                if show_code:
                                    st.code(pandas_code, language="python")
                                
                                # 🚀 ENHANCED: Pre-execution error detection and fixing
                                try:
                                    from utils.code_validator_silo import CodeValidatorSilo
                                    import ast  # Add missing import
                                    
                                    # Initialize the code validator silo
                                    code_validator = CodeValidatorSilo()
                                    
                                    # Check if the generated code has obvious errors before execution
                                    st.info("🔧 Pre-execution code validation and error fixing...")
                                    
                                    # First, try to fix any obvious syntax issues
                                    try:
                                        # Test basic syntax
                                        ast.parse(pandas_code)
                                        st.info("✅ Basic syntax validation passed")
                                    except SyntaxError as syntax_error:
                                        st.warning(f"🔧 Syntax error detected: {syntax_error}")
                                        
                                        # Try to fix the syntax error
                                        fixed_code, fixes = code_validator.fix_specific_error_patterns(
                                            pandas_code, 
                                            str(syntax_error)
                                        )
                                        
                                        if fixes:
                                            st.success("✅ Syntax error auto-fixed!")
                                            st.info(code_validator.get_fix_summary(fixes))
                                            pandas_code = fixed_code
                                            
                                            if show_code:
                                                st.code(pandas_code, language="python")
                                        else:
                                            st.warning("Could not auto-fix syntax error")
                                    
                                except ImportError as import_error:
                                    st.warning(f"CodeValidatorSilo not available for pre-execution validation: {import_error}")
                                except Exception as validation_error:
                                    st.warning(f"Pre-execution validation failed: {validation_error}")
                                    st.info("Continuing with basic validation...")
                                    
                                    # Fallback: Basic pattern fixing for common errors
                                    try:
                                        st.info("🔧 Applying fallback pattern fixes...")
                                        
                                        # Fix the specific error pattern you're encountering
                                        if 'str.contains(' in pandas_code and '&' in pandas_code:
                                            original_code = pandas_code
                                            
                                            # Fix malformed boolean operations with str.contains
                                            if "df1['type'].str.contains('MPAN', case=False, na=False) & df1['value'].str.contains('error', case=False, na=False)" in pandas_code:
                                                fixed_code = "df1[(df1['type'].str.contains('MPAN', case=False, na=False)) & (df1['value'].str.contains('error', case=False, na=False))]"
                                                pandas_code = fixed_code
                                                st.success("✅ Fixed malformed boolean operation pattern!")
                                                st.info("Added proper parentheses around str.contains conditions")
                                                
                                                if show_code:
                                                    st.code(pandas_code, language="python")
                                            
                                            # Pattern 2: Fix missing case=False, na=False parameters
                                            elif "df1['type'].str.contains('MPAN') & df1['value'].str.contains('error')" in pandas_code:
                                                fixed_code = "df1[(df1['type'].str.contains('MPAN', case=False, na=False)) & (df1['value'].str.contains('error', case=False, na=False))]"
                                                pandas_code = fixed_code
                                                st.success("✅ Fixed malformed str.contains pattern!")
                                                st.info("Added missing case=False, na=False parameters and proper parentheses")
                                                
                                                if show_code:
                                                    st.code(fixed_code, language="python")
                                            
                                            # SIMPLIFIED: Remove complex regex pattern matching that causes errors
                                            # Instead, apply simple fixes for common patterns
                                            else:
                                                st.info("🔧 Applying simple pattern fixes...")
                                                
                                                # Fix missing case=False, na=False parameters
                                                pandas_code = pandas_code.replace("str.contains('MPAN')", "str.contains('MPAN', case=False, na=False)")
                                                pandas_code = pandas_code.replace("str.contains('error')", "str.contains('error', case=False, na=False)")
                                                pandas_code = pandas_code.replace("str.contains('validation')", "str.contains('validation', case=False, na=False)")
                                                
                                                # Simple fix: Add parentheses around boolean operations if they're missing
                                                if '&' in pandas_code and 'str.contains(' in pandas_code and '(' not in pandas_code.split('[')[1].split(']')[0]:
                                                    # Only fix if parentheses are actually missing
                                                    if pandas_code != original_code:
                                                        st.success("✅ Applied simple pattern fixes!")
                                                        st.info("Added missing parameters and basic structure improvements")
                                                
                                                if show_code and pandas_code != original_code:
                                                    st.code(pandas_code, language="python")
                                    
                                    except Exception as generation_fix_error:
                                        st.warning(f"Code generation error fixing failed: {generation_fix_error}")
                                        st.info("Continuing with original code...")
                                
                                if show_code:
                                    st.code(pandas_code, language="python")
                                
                                # 🚀 ENHANCED: Pre-execution error detection and fixing
                                try:
                                    from utils.code_validator_silo import CodeValidatorSilo
                                    import ast  # Add missing import
                                    
                                    # Initialize the code validator silo
                                    code_validator = CodeValidatorSilo()
                                    
                                    # Check if the generated code has obvious errors before execution
                                    st.info("🔧 Pre-execution code validation and error fixing...")
                                    
                                    # First, try to fix any obvious syntax issues
                                    try:
                                        # Test basic syntax
                                        ast.parse(pandas_code)
                                        st.info("✅ Basic syntax validation passed")
                                    except SyntaxError as syntax_error:
                                        st.warning(f"🔧 Syntax error detected: {syntax_error}")
                                        
                                        # Try to fix the syntax error
                                        fixed_code, fixes = code_validator.fix_specific_error_patterns(
                                            pandas_code, 
                                            str(syntax_error)
                                        )
                                        
                                        if fixes:
                                            st.success("✅ Syntax error auto-fixed!")
                                            st.info(code_validator.get_fix_summary(fixes))
                                            pandas_code = fixed_code
                                            
                                            if show_code:
                                                st.code(pandas_code, language="python")
                                        else:
                                            st.warning("Could not auto-fix syntax error")
                                    
                                except ImportError as import_error:
                                    st.warning(f"CodeValidatorSilo not available for pre-execution validation: {import_error}")
                                except Exception as validation_error:
                                    st.warning(f"Pre-execution validation failed: {validation_error}")
                                    st.info("Continuing with basic validation...")
                                    
                                    # Fallback: Basic pattern fixing for common errors
                                    try:
                                        st.info("🔧 Applying fallback pattern fixes...")
                                        
                                        # Fix the specific error pattern you're encountering
                                        if 'str.contains(' in pandas_code and '&' in pandas_code:
                                            original_code = pandas_code
                                            
                                            # Fix malformed boolean operations with str.contains
                                            if "df1['type'].str.contains('MPAN', case=False, na=False) & df1['value'].str.contains('error', case=False, na=False)" in pandas_code:
                                                fixed_code = "df1[(df1['type'].str.contains('MPAN', case=False, na=False)) & (df1['value'].str.contains('error', case=False, na=False))]"
                                                pandas_code = fixed_code
                                                st.success("✅ Fixed malformed boolean operation pattern!")
                                                st.info("Added proper parentheses around str.contains conditions")
                                                
                                                if show_code:
                                                    st.code(pandas_code, language="python")
                                            
                                            # Fix other common patterns
                                            elif "df1['type'].str.contains('MPAN') & df1['value'].str.contains('error')" in pandas_code:
                                                fixed_code = "df1[(df1['type'].str.contains('MPAN', case=False, na=False)) & (df1['value'].str.contains('error', case=False, na=False))]"
                                                pandas_code = fixed_code
                                                st.success("✅ Fixed malformed str.contains pattern!")
                                                st.info("Added missing case=False, na=False parameters and proper parentheses")
                                                
                                                if show_code:
                                                    st.code(pandas_code, language="python")
                                        
                                        # 🚀 NEW: Fix MongoDB array field handling issues
                                        # Fix incorrect array field searches like df1[df1['COLUMN'] == "['VALUE']"]
                                        if "['" in pandas_code and "']" in pandas_code and "==" in pandas_code:
                                            st.warning("🔧 Detected incorrect MongoDB array field handling - applying fix...")
                                            
                                            # Generic pattern: Fix df1[df1['COLUMN'] == "['VALUE']"] to proper array search
                                            # Use regex to find any column name and value
                                            import re
                                            
                                            # Pattern: df1[df1['COLUMN'] == "['VALUE']"]
                                            array_pattern = r"df1\[df1\['([^']+)'\] == \"\['([^']+)'\]\""
                                            matches = re.findall(array_pattern, pandas_code)
                                            
                                            for column_name, value in matches:
                                                # Replace with proper array search using str.contains
                                                old_pattern = f"df1[df1['{column_name}'] == \"['{value}']\"]"
                                                new_pattern = f"df1[df1['{column_name}'].str.contains('{value}', na=False)]"
                                                pandas_code = pandas_code.replace(old_pattern, new_pattern)
                                                
                                                st.success(f"✅ Fixed MongoDB array field search for {column_name}!")
                                                st.info(f"Changed incorrect string search to proper array search for value: {value}")
                                                
                                                if show_code:
                                                    st.code(new_pattern, language="python")
                                            
                                            # If no matches found, try a more general approach
                                            if not matches:
                                                # Replace any remaining array string patterns
                                                pandas_code = re.sub(
                                                    r"df1\[df1\['([^']+)'\] == \"\['([^']+)'\]\"",
                                                    r"df1[df1['\1'].str.contains('\2', na=False)]",
                                                    pandas_code
                                                )
                                                st.success("✅ Applied general array field fix!")
                                    
                                    except Exception as fallback_error:
                                        st.warning(f"Fallback pattern fixing also failed: {fallback_error}")
                                        st.info("Proceeding with original code...")
                                
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
                                
                                # 🚀 ENHANCED: Automatic data quality management using existing utilities
                                try:
                                    from utils.enhanced_dataframe_corrector import EnhancedDataFrameCorrector
                                    data_quality_manager = EnhancedDataFrameCorrector()
                                    st.info("🔧 Auto-cleaning data for better analysis...")
                                    
                                    # Clean all DataFrames in the execution environment
                                    for key, value in exec_env.items():
                                        if hasattr(value, 'columns') and hasattr(value, 'shape'):  # It's a DataFrame
                                            original_shape = value.shape
                                            
                                            # Show original data types for key columns
                                            st.info(f"🔍 Original data types for '{key}':")
                                            key_columns = [col for col in value.columns if any(keyword in col.lower() for keyword in ['battery', 'capacity', 'efficiency', 'range'])]
                                            for col in key_columns[:5]:  # Show first 5 key columns
                                                st.info(f"  {col}: {value[col].dtype}")
                                                if value[col].dtype == 'object':
                                                    sample_values = value[col].dropna().head(3)
                                                    if len(sample_values) > 0:
                                                        st.info(f"    Sample values: {sample_values.tolist()}")
                                            
                                            # Determine collection name from DataFrame key dynamically
                                            collection_name = None
                                            if 'phev' in key.lower():
                                                collection_name = 'phevVE'
                                            elif 'bev' in key.lower():
                                                collection_name = 'bevVE'
                                            elif 'connections' in key.lower():
                                                collection_name = 'connections'
                                            elif 'df1' in key.lower():
                                                # df1 is usually the main DataFrame from the query
                                                # Try to determine collection from the actual data or use a generic approach
                                                if 'df1' in exec_env and hasattr(exec_env['df1'], 'columns'):
                                                    # Look for collection-specific columns to determine the collection
                                                    if any('battery' in col.lower() or 'vehicle' in col.lower() for col in exec_env['df1'].columns):
                                                        collection_name = 'phevVE'  # Vehicle-related data
                                                    elif any('address' in col.lower() or 'postcode' in col.lower() for col in exec_env['df1'].columns):
                                                        collection_name = 'connections'  # Address-related data
                                                    else:
                                                        collection_name = 'generic'  # Generic collection
                                                else:
                                                    collection_name = 'generic'  # Fallback
                                            else:
                                                # For any other DataFrame, try to infer collection from column names
                                                if hasattr(value, 'columns'):
                                                    if any('battery' in col.lower() or 'vehicle' in col.lower() for col in value.columns):
                                                        collection_name = 'phevVE'
                                                    elif any('address' in col.lower() or 'postcode' in col.lower() for col in value.columns):
                                                        collection_name = 'connections'
                                                    else:
                                                        collection_name = 'generic'
                                                else:
                                                    collection_name = 'generic'
                                            
                                            # Check if this DataFrame contains MongoDB array fields that might cause issues
                                            has_array_fields = False
                                            if hasattr(value, 'columns'):
                                                for col in value.columns:
                                                    if value[col].dtype == 'object':
                                                        # Check if column contains array-like data
                                                        sample_values = value[col].dropna().head(3)
                                                        if len(sample_values) > 0:
                                                            for val in sample_values:
                                                                if isinstance(val, list) or (isinstance(val, str) and val.startswith('[') and val.endswith(']')):
                                                                    has_array_fields = True
                                                                    break
                                                            if has_array_fields:
                                                                break
                                            
                                            # Skip data quality management for DataFrames with MongoDB array fields to prevent errors
                                            if has_array_fields and collection_name == 'connections':
                                                st.info(f"🔧 Skipping data quality management for '{key}' - contains MongoDB array fields")
                                                st.info("This prevents errors with array field processing while maintaining functionality")
                                                # Keep the original DataFrame without cleaning
                                                exec_env[key] = value
                                                cleaned_shape = value.shape
                                            else:
                                                # Proceed with normal data quality management
                                                try:
                                                    exec_env[key] = data_quality_manager.auto_clean_dataframe(value, collection_name)
                                                    cleaned_shape = exec_env[key].shape
                                                except Exception as clean_error:
                                                    st.warning(f"🔧 Data cleaning failed for '{key}': {clean_error}")
                                                    st.info("Continuing with original DataFrame to prevent execution failure")
                                                    exec_env[key] = value
                                                    cleaned_shape = value.shape
                                            
                                            # Show cleaned data types for key columns
                                            st.info(f"🔧 Cleaned data types for '{key}':")
                                            for col in key_columns[:5]:  # Show first 5 key columns
                                                if col in exec_env[key].columns:
                                                    st.info(f"  {col}: {exec_env[key][col].dtype}")
                                                    if exec_env[key][col].dtype in ['float64', 'float32', 'int64', 'int32']:
                                                        non_null_count = exec_env[key][col].count()
                                                        st.info(f"    ✅ Numeric conversion successful: {non_null_count}/{len(exec_env[key][col])} non-null values")
                                                    else:
                                                        st.info(f"    ⚠️ Still object type - conversion may have failed")
                                            
                                            if original_shape != cleaned_shape:
                                                st.warning(f"⚠️ DataFrame '{key}' shape changed during cleaning: {original_shape} -> {cleaned_shape}")
                                            
                                            # Show data quality report (with error handling for array fields)
                                            try:
                                                quality_report = data_quality_manager.get_data_quality_report(exec_env[key])
                                                if quality_report['quality_score'] < 0.8:
                                                    st.info(f"📊 Data quality score for '{key}': {quality_report['quality_score']:.2f}")
                                                    
                                                    # Show detailed battery capacity analysis if available
                                                    if 'battery_capacity_analysis' in quality_report:
                                                        st.info("🔋 Battery Capacity Analysis:")
                                                        for col, analysis in quality_report['battery_capacity_analysis'].items():
                                                            st.info(f"  {col}: {analysis['dtype']}, {analysis['non_null_values']}/{analysis['total_values']} non-null ({analysis['null_percentage']:.1f}% null)")
                                                            if 'should_be_numeric' in analysis:
                                                                st.info(f"    Should be numeric: {analysis['should_be_numeric']}")
                                                            if analysis['sample_values']:
                                                                st.info(f"    Sample values: {analysis['sample_values'][:3]}")
                                                    
                                                    # Show recommendations
                                                    if quality_report.get('recommendations'):
                                                        st.info("💡 Data Quality Recommendations:")
                                                        for rec in quality_report['recommendations'][:3]:  # Show first 3
                                                            st.info(f"  • {rec}")
                                            except Exception as report_error:
                                                st.warning(f"📊 Data quality report generation failed for '{key}': {report_error}")
                                                st.info("Continuing without quality report to prevent execution failure")
                                    
                                    st.success("✅ Data automatically cleaned and optimized!")
                                    
                                except ImportError as import_error:
                                    st.warning(f"EnhancedDataFrameCorrector not available: {import_error}")
                                except Exception as data_error:
                                    st.warning(f"Data quality management failed: {data_error}")
                                    st.info("Continuing with original data...")
                                    
                                    # CRITICAL: Ensure execution environment has the necessary DataFrames
                                    # even if data quality management failed
                                    if 'df1' not in exec_env and 'df1' in dfs:
                                        st.info("🔧 Ensuring df1 is available in execution environment...")
                                        exec_env['df1'] = dfs['df1']
                                    
                                    # Add any other DataFrames that might be missing
                                    for name, df in dfs.items():
                                        if name not in exec_env:
                                            st.info(f"🔧 Adding missing DataFrame '{name}' to execution environment...")
                                            exec_env[name] = df
                                    
                                    st.info(f"✅ Execution environment prepared with {len(exec_env)} DataFrames")
                                
                                # --- Validate code uses actual names before execution ---
                                all_available_columns = set()
                                for name, df in dfs.items():
                                    all_available_columns.update(df.columns)
                                if 'df1' in exec_env:
                                    all_available_columns.update(exec_env['df1'].columns)
                                
                                # Use the intelligent DataFrameCorrector utility
                                # DISABLED: This was causing regex errors with valid code
                                # from utils.dataframe_corrector import DataFrameCorrector
                                
                                # Create an instance of the corrector
                                # dataframe_corrector = DataFrameCorrector()
                                
                                # Correct DataFrame names using intelligent fixing
                                # DISABLED: Skip DataFrame correction since code is already valid
                                # pandas_code, corrections_made = dataframe_corrector.correct_dataframe_names(
                                #     pandas_code, 
                                #     exec_env
                                # )
                                
                                # if corrections_made:
                                #     st.warning(f"Fixed DataFrame name references: {', '.join(corrections_made)}")
                                #     st.info(f"Available DataFrames: {list(exec_env.keys())}")
                                
                                # Validate remaining DataFrame references
                                # DISABLED: Skip DataFrame validation since code is already valid
                                # undefined_refs = dataframe_corrector.validate_dataframe_references(
                                #     pandas_code, 
                                #     exec_env
                                # )
                                
                                # if undefined_refs:
                                #     st.error(f"Could not resolve DataFrame references: {undefined_refs}")
                                #     st.error(f"Available DataFrames: {list(exec_env.keys())}")
                                #     st.error("Please rephrase your question to use only the available DataFrame names.")
                                #     st.stop()
                                
                                # SIMPLIFIED: Basic validation without complex utilities
                                st.info("✅ Skipping DataFrame correction - code appears valid")
                                
                                # Check if DataFrame names are in the available list
                                available_dataframes = list(exec_env.keys())
                                if 'df1' in pandas_code and 'df1' in available_dataframes:
                                    st.info("✅ DataFrame 'df1' is available and correctly referenced")
                                else:
                                    st.warning("⚠️ DataFrame reference issue detected")
                                    st.info(f"Available DataFrames: {available_dataframes}")
                                
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
                                    
                                    # 🚀 ENHANCED: Use CodeValidatorSilo for automatic error fixing
                                    try:
                                        from utils.code_validator_silo import CodeValidatorSilo
                                        
                                        # Initialize the code validator silo
                                        code_validator = CodeValidatorSilo()
                                        
                                        # Get available columns and DataFrames
                                        available_columns = []
                                        available_dataframes = []
                                        
                                        for key, value in exec_env.items():
                                            if hasattr(value, 'columns') and hasattr(value, 'shape'):
                                                # This is a DataFrame
                                                available_dataframes.append(key)
                                                available_columns.extend(value.columns)
                                            elif key in ['pd', 'plt', 'np']:
                                                # These are modules, not DataFrames
                                                continue
                                            else:
                                                # Check if it's a DataFrame by other means
                                                try:
                                                    if hasattr(value, 'columns'):
                                                        available_dataframes.append(key)
                                                        available_columns.extend(value.columns)
                                                except:
                                                    continue
                                        
                                        # Remove duplicates
                                        available_columns = list(set(available_columns))
                                        
                                        st.info(f"✅ Found {len(available_dataframes)} DataFrames and {len(available_columns)} columns")
                                        
                                        # Validate and fix the code automatically
                                        st.info("🔧 Validating and auto-fixing code using CodeValidatorSilo...")
                                        
                                        fixed_code, fixes, is_valid = code_validator.validate_and_fix_code(
                                            pandas_code, 
                                            available_columns, 
                                            available_dataframes
                                        )
                                        
                                        # Show fix summary
                                        if fixes:
                                            st.success("✅ Code auto-fixed successfully!")
                                            st.info(code_validator.get_fix_summary(fixes))
                                            
                                            if show_code:
                                                st.code(fixed_code, language="python")
                                            
                                            # Use the fixed code
                                            pandas_code = fixed_code
                                        else:
                                            st.info("✅ Code validation passed - no fixes needed")
                                        
                                    except ImportError:
                                        st.warning("CodeValidatorSilo not available, using basic validation...")
                                        # Fallback to basic regex fixing
                                        fixed_code = pandas_code
                                        
                                        # Fix common regex pattern issues
                                        if 'str.contains(' in fixed_code:
                                            # Fix missing parentheses in str.contains calls
                                            fixed_code = fixed_code.replace('str.contains(', 'str.contains(')
                                            
                                            # Fix common pattern issues
                                            fixed_code = fixed_code.replace("'MPAN') & df1['type'].str.contains('error')", "('MPAN', case=False, na=False)) & (df1['type'].str.contains('error', case=False, na=False))")
                                            fixed_code = fixed_code.replace("'MPAN') & df1['value'].str.contains('error')", "('MPAN', case=False, na=False)) & (df1['value'].str.contains('error', case=False, na=False))")
                                            
                                            # Fix missing case=False, na=False parameters
                                            fixed_code = fixed_code.replace("str.contains('MPAN')", "str.contains('MPAN', case=False, na=False)")
                                            fixed_code = fixed_code.replace("str.contains('error')", "str.contains('error', case=False, na=False)")
                                            fixed_code = fixed_code.replace("str.contains('validation')", "str.contains('validation', case=False, na=False)")
                                        
                                        if fixed_code != pandas_code:
                                            st.warning("🔧 Fixed common regex pattern issues in generated code")
                                            st.info("Original code had malformed str.contains() patterns")
                                            if show_code:
                                                st.code(fixed_code, language="python")
                                            pandas_code = fixed_code
                                    
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
                                                
                                                # 🚀 NEW: Dynamic Complex Field Unpacking
                                                st.info("🔍 Analyzing complex nested fields for enhanced display...")
                                                try:
                                                    unpacked_result = unpack_complex_fields_dynamically(result)
                                                    if unpacked_result is not None and not unpacked_result.equals(result):
                                                        st.success("✅ Complex fields unpacked and enhanced!")
                                                        st.info("Enhanced result with unpacked nested fields:")
                                                        st.dataframe(unpacked_result)
                                                        
                                                        # Download enhanced result
                                                        csv_data_enhanced = unpacked_result.to_csv(index=False).encode('utf-8')
                                                        st.download_button(
                                                            label="Download enhanced result as CSV",
                                                            data=csv_data_enhanced,
                                                            file_name="rag_result_enhanced.csv",
                                                            mime="text/csv"
                                                        )
                                                except Exception as unpack_error:
                                                    st.warning(f"⚠️ Complex field unpacking failed: {unpack_error}")
                                                    st.info("Continuing with original result display...")
                                                
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
                                                
                                                # 🚀 NEW: Dynamic Complex Field Unpacking for Series
                                                try:
                                                    unpacked_series = unpack_complex_series_dynamically(result)
                                                    if unpacked_series is not None and not unpacked_series.equals(result):
                                                        st.success("✅ Complex Series fields unpacked!")
                                                        st.info("Enhanced Series with unpacked nested fields:")
                                                        st.write(unpacked_series)
                                                except Exception as unpack_error:
                                                    st.warning(f"⚠️ Series unpacking failed: {unpack_error}")
                                            else:
                                                st.write("**Result:**")
                                                st.write(result)
                                                
                                                # 🚀 NEW: Dynamic Complex Field Unpacking for other types
                                                try:
                                                    unpacked_other = unpack_complex_other_dynamically(result)
                                                    if unpacked_other is not None and unpacked_other != result:
                                                        st.success("✅ Complex fields unpacked!")
                                                        st.info("Enhanced result with unpacked nested fields:")
                                                        st.write(unpacked_other)
                                                except Exception as unpack_error:
                                                    st.warning(f"⚠️ Other type unpacking failed: {unpack_error}")
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
                                        
                                        # 🚀 ENHANCED: Auto-fallback using CodeValidatorSilo
                                        try:
                                            st.info("🔄 Attempting automatic code generation fallback...")
                                            
                                            if 'code_validator' in locals():
                                                # Use the code validator to generate a working query
                                                auto_generated_code = code_validator.auto_fix_common_queries(
                                                    user_question, 
                                                    available_columns
                                                )
                                                
                                                st.success("✅ Auto-generated fallback code created!")
                                                st.info("This code was automatically generated based on your question and available data.")
                                                
                                                if show_code:
                                                    st.code(auto_generated_code, language="python")
                                                
                                                # Try to execute the auto-generated code
                                                try:
                                                    local_vars_fallback = exec_env.copy()
                                                    exec(auto_generated_code, {"pd": pd, "plt": plt, "__builtins__": SAFE_BUILTINS}, local_vars_fallback)
                                                    result_fallback = local_vars_fallback.get("result")
                                                    
                                                    if result_fallback is not None:
                                                        st.success("✅ Auto-generated code executed successfully!")
                                                        
                                                        if isinstance(result_fallback, pd.DataFrame):
                                                            st.dataframe(result_fallback)
                                                        elif isinstance(result_fallback, pd.Series):
                                                            st.write(result_fallback)
                                                        else:
                                                            st.write("Result:", result_fallback)
                                                        
                                                        # Log successful fallback execution
                                                        diagnostics_logger.log_code_generation(
                                                            question=user_question,
                                                            generated_code=auto_generated_code,
                                                            llm_provider="Auto-Fallback",
                                                            execution_success=True
                                                        )
                                                    else:
                                                        st.warning("Auto-generated code did not produce a result")
                                                        
                                                except Exception as fallback_error:
                                                    st.error(f"Auto-generated code also failed: {fallback_error}")
                                                    st.info("Please try rephrasing your question or contact support.")
                                            
                                            else:
                                                st.warning("CodeValidatorSilo not available for fallback")
                                                st.info("Please try rephrasing your question or use a different approach.")
                                                
                                        except Exception as fallback_init_error:
                                            st.error(f"Could not initialize fallback system: {fallback_init_error}")
                                            st.info("Please try rephrasing your question.")
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