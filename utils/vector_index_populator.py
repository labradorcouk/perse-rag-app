#!/usr/bin/env python3
"""
Vector Index Populator Script
Populates Qdrant collections from SQL endpoint with pre-computed embeddings
"""

import os
import sys
import yaml
import traceback
import time
from pathlib import Path
from tqdm import tqdm
import pandas as pd
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance
from utils.rag_utils import RAGUtils

# Configuration
CONFIG_PATH = "config/vector_index_population.yaml"
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "")

def load_config(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def get_sql_engine():
    return RAGUtils.get_fabric_engine()

def resume_from_offset(collection_name, offset_file="resume_offsets.json"):
    """Load the last processed offset for a collection"""
    import json
    try:
        with open(offset_file, 'r') as f:
            offsets = json.load(f)
            return offsets.get(collection_name, 0)
    except FileNotFoundError:
        return 0

def save_offset(collection_name, offset, offset_file="resume_offsets.json"):
    """Save the current offset for a collection"""
    import json
    try:
        with open(offset_file, 'r') as f:
            offsets = json.load(f)
    except FileNotFoundError:
        offsets = {}
    
    offsets[collection_name] = offset
    
    with open(offset_file, 'w') as f:
        json.dump(offsets, f, indent=2)

def mark_collection_complete(collection_name, offset_file="resume_offsets.json"):
    """Mark a collection as complete by setting offset to -1"""
    import json
    try:
        with open(offset_file, 'r') as f:
            offsets = json.load(f)
    except FileNotFoundError:
        offsets = {}
    
    offsets[collection_name] = -1  # -1 indicates completion
    
    with open(offset_file, 'w') as f:
        json.dump(offsets, f, indent=2)

def is_collection_complete(collection_name, offset_file="resume_offsets.json"):
    """Check if a collection is marked as complete"""
    import json
    try:
        with open(offset_file, 'r') as f:
            offsets = json.load(f)
            return offsets.get(collection_name, 0) == -1
    except FileNotFoundError:
        return False

def main():
    print("--- Starting Vector Index Population Script ---")
    
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description='Vector Index Population Script')
    parser.add_argument('--table', type=str, help='Process only specific table (by collection name)')
    parser.add_argument('--force', action='store_true', help='Force reprocess even if collection exists')
    parser.add_argument('--skip-completed', action='store_true', help='Skip tables marked as complete')
    args = parser.parse_args()
    
    try:
        print(f"Loading configuration from: {CONFIG_PATH}")
        config = load_config(CONFIG_PATH)
    except FileNotFoundError:
        print(f"ERROR: Configuration file not found at {CONFIG_PATH}. Exiting.")
        return
    except Exception as e:
        print(f"ERROR: Could not load or parse YAML configuration: {e}. Exiting.")
        return

    print(f"Initializing Qdrant client for Qdrant at: {QDRANT_URL}")
    client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
    
    print("Connecting to SQL endpoint...")
    try:
        engine = get_sql_engine()
        print("Successfully connected to SQL endpoint.")
    except Exception as e:
        print(f"ERROR: Could not connect to SQL endpoint: {e}. Exiting.")
        return

    # Filter tables if specific table is requested
    tables_to_process = config['tables']
    if args.table:
        tables_to_process = [t for t in config['tables'] if t['collection'] == args.table]
        if not tables_to_process:
            print(f"ERROR: Table with collection name '{args.table}' not found in config.")
            return
        print(f"Processing only table: {args.table}")

    for table_cfg in tables_to_process:
        collection = table_cfg['collection']
        force = table_cfg.get('force', False) or args.force
        query = table_cfg['query']
        id_column = table_cfg['id_column']
        batch_size = table_cfg.get('batch_size', 1000)

        print(f"\n--- Processing table: {table_cfg['name']} ---")
        print(f"Target Qdrant collection: {collection}")

        # Check if collection is marked as complete
        if args.skip_completed and is_collection_complete(collection):
            print(f"Collection '{collection}' is marked as complete. Skipping.")
            continue

        # Check if collection exists and has points
        collection_exists = False
        collection_size = 0
        try:
            info = client.get_collection(collection)
            collection_exists = True
            collection_size = info.points_count
        except Exception:
            collection_exists = False

        if collection_exists and collection_size > 0 and not force:
            print(f"Collection '{collection}' already exists with {collection_size} points. Skipping (use --force to overwrite).")
            continue

        # Get resume offset
        resume_offset = resume_from_offset(collection)
        if resume_offset > 0:
            print(f"Resuming from offset: {resume_offset}")
        elif resume_offset == -1:
            print(f"Collection '{collection}' is marked as complete. Skipping.")
            continue
        
        first_batch = True
        offset = resume_offset if resume_offset >= 0 else 0
        total_rows = 0
        consecutive_errors = 0
        max_consecutive_errors = 5
        
        with tqdm(desc=f"Processing batches for {collection}", initial=offset//batch_size) as pbar:
            while True:
                try:
                    print(f"\nFetching batch starting at offset {offset}...")
                    
                    # Add retry logic for connection issues
                    max_retries = 3
                    retry_delay = 5
                    
                    for retry in range(max_retries):
                        try:
                            # ORDER BY is required for OFFSET FETCH to work reliably
                            batch_query = f"""
                            WITH OrderedResults AS ({query})
                            SELECT * FROM OrderedResults
                            ORDER BY {id_column}
                            OFFSET {offset} ROWS FETCH NEXT {batch_size} ROWS ONLY
                            """
                            df = pd.read_sql(batch_query, engine)
                            break  # Success, exit retry loop
                        except Exception as e:
                            if "connection" in str(e).lower() or "timeout" in str(e).lower():
                                if retry < max_retries - 1:
                                    print(f"Connection error, retrying in {retry_delay} seconds... (attempt {retry + 1}/{max_retries})")
                                    time.sleep(retry_delay)
                                    retry_delay *= 2  # Exponential backoff
                                    continue
                                else:
                                    print(f"Max retries reached. Saving offset and exiting.")
                                    save_offset(collection, offset)
                                    raise e
                            else:
                                raise e

                    if df.empty:
                        print("No more rows to fetch. Marking collection as complete.")
                        mark_collection_complete(collection)
                        break
                    
                    print(f"Fetched {len(df)} rows. DataFrame shape: {df.shape}")

                    if 'embedding' not in df.columns:
                        print("ERROR: The SQL query must return an 'embedding' column.")
                        break
                    
                    if id_column not in df.columns:
                        print(f"ERROR: The specified id_column '{id_column}' was not found.")
                        break

                    if first_batch:
                        print("First batch: creating or recreating Qdrant collection...")
                        sample_emb = RAGUtils.parse_embedding(df['embedding'].iloc[0])
                        print(f"Inferred vector size: {len(sample_emb)}")
                        client.recreate_collection(
                            collection_name=collection,
                            vectors_config=VectorParams(size=len(sample_emb), distance=Distance.COSINE)
                        )
                        print(f"Collection '{collection}' is ready.")
                        first_batch = False

                    print("Parsing embeddings from the 'embedding' column...")
                    embeddings = df['embedding'].apply(RAGUtils.parse_embedding).tolist()
                    
                    print("Preparing points for upsert...")
                    points = [
                        {
                            'id': offset + i,
                            'vector': embeddings[i],
                            'payload': df.drop(columns=['embedding']).iloc[i].to_dict()
                        }
                        for i in range(len(df))
                    ]
                    
                    if points:
                        print(f"Sample point payload (first point): {points[0]['payload']}")

                    print(f"Upserting {len(points)} points to Qdrant...")
                    client.upsert(collection_name=collection, points=points, wait=True)
                    print(f"Upsert successful for batch.")

                    offset += len(df)
                    total_rows += len(df)
                    consecutive_errors = 0  # Reset error counter on success
                    
                    # Save progress after each successful batch
                    save_offset(collection, offset)
                    
                    pbar.update(1)

                except Exception as e:
                    consecutive_errors += 1
                    print(f"\nERROR processing batch for collection '{collection}' at offset {offset}:")
                    print(traceback.format_exc())
                    
                    if consecutive_errors >= max_consecutive_errors:
                        print(f"Too many consecutive errors ({consecutive_errors}). Saving offset and stopping.")
                        save_offset(collection, offset)
                        break
                    
                    print(f"Consecutive errors: {consecutive_errors}/{max_consecutive_errors}")
                    print("Waiting 30 seconds before retrying...")
                    time.sleep(30)
                    continue
        
        print(f"--- Finished processing for table: {table_cfg['name']}. Total rows indexed: {total_rows} ---")

    print("\n--- Vector Index Population Script Finished ---")

if __name__ == "__main__":
    main() 