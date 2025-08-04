import os
import pandas as pd
from sentence_transformers import SentenceTransformer
import torch
from utils.rag_utils import RAGUtils

# --- CONFIG ---
BATCH_SIZE = 640  # Optimized for 8GB VRAM; reduce to 32 if OOM, try 128 if possible
EXCLUDED_COLUMNS = {"lmk_key", "building_reference_number", "uprn", "uprn_source", "report_type"}
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # Correct HF model name
OUTPUT_DIR = "vector_embeddings"

os.makedirs(OUTPUT_DIR, exist_ok=True)

def get_sql_engine():
    # Use the same authentication as the SQL Editor
    return RAGUtils.get_fabric_engine()

def is_integer_pk(pk_val):
    try:
        int(pk_val)
        return True
    except Exception:
        return False

def process_data_in_batches(table_name, selected_columns=None, pk_col="id", source_type="sql"):
    print(f"Processing table: {table_name} (source_type={source_type})")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"[DEBUG] Using device: {device}")
    # --- Optimization: Use float16 for even more speed/memory efficiency ---
    model = SentenceTransformer(EMBEDDING_MODEL, device=device)
    if device == 'cuda':
        from transformers import AutoModel
        model._first_module().auto_model = AutoModel.from_pretrained(
            EMBEDDING_MODEL, torch_dtype=torch.float16
        ).to('cuda')
    print(f"[DEBUG] Model loaded on device: {model.device}")
    batch_count = 0
    if source_type == "sql":
        engine = get_sql_engine()
        # Get min and max PK and total rows
        minmax = pd.read_sql(f"SELECT MIN({pk_col}) as min_id, MAX({pk_col}) as max_id, COUNT(*) as cnt FROM {table_name}", engine)
        min_id, max_id, total_rows = minmax.iloc[0]["min_id"], minmax.iloc[0]["max_id"], minmax.iloc[0]["cnt"]
        print(f"[DEBUG] PK range: {min_id} to {max_id}, total rows: {total_rows}")
        # Decide batching method
        if is_integer_pk(min_id) and is_integer_pk(max_id):
            print("[DEBUG] Using PK range batching (integer PK detected)")
            for batch_start in range(int(min_id), int(max_id)+1, BATCH_SIZE):
                batch_end = min(batch_start + BATCH_SIZE - 1, int(max_id))
                print(f"[DEBUG] Processing PK batch: {batch_start} - {batch_end}")
                batch_df = pd.read_sql(
                    f"SELECT * FROM {table_name} WHERE {pk_col} >= {batch_start} AND {pk_col} <= {batch_end}",
                    engine
                )
                print(f"[DEBUG] Data fetched: {len(batch_df)} rows")
                if batch_df.empty:
                    continue
                columns_to_concat = [col for col in batch_df.columns if col not in EXCLUDED_COLUMNS]
                batch_df["combined_text"] = batch_df[columns_to_concat].astype(str).agg(" | ".join, axis=1)
                # --- Optimization: Use batch_size=64, convert_to_numpy=True ---
                batch_df["embedding"] = model.encode(
                    batch_df["combined_text"].tolist(),
                    show_progress_bar=True,
                    batch_size=BATCH_SIZE,
                    convert_to_numpy=True
                ).tolist()
                print(f"[DEBUG] Embeddings generated for batch {batch_count}")
                if selected_columns is None:
                    selected_columns = [col for col in batch_df.columns if col in EXCLUDED_COLUMNS] + ["embedding"]
                final_batch = batch_df[selected_columns]
                print(f"[DEBUG] Columns selected: {selected_columns}")
                out_path = os.path.join(OUTPUT_DIR, f"{table_name.replace('.', '_')}.csv")
                header = not os.path.exists(out_path) and batch_count == 0
                final_batch.to_csv(out_path, mode='a', header=header, index=False)
                print(f"[DEBUG] Saved batch {batch_count} to {out_path}")
                batch_count += 1
        else:
            print("[WARNING] String PK detected. Using OFFSET/FETCH batching. This may be slow for large tables!")
            num_batches = (total_rows + BATCH_SIZE - 1) // BATCH_SIZE
            for batch_idx in range(num_batches):
                offset = batch_idx * BATCH_SIZE
                print(f"[DEBUG] Processing OFFSET batch: {offset} - {offset+BATCH_SIZE}")
                batch_df = pd.read_sql(
                    f"SELECT * FROM {table_name} ORDER BY {pk_col} OFFSET {offset} ROWS FETCH NEXT {BATCH_SIZE} ROWS ONLY",
                    engine
                )
                print(f"[DEBUG] Data fetched: {len(batch_df)} rows")
                if batch_df.empty:
                    continue
                columns_to_concat = [col for col in batch_df.columns if col not in EXCLUDED_COLUMNS]
                batch_df["combined_text"] = batch_df[columns_to_concat].astype(str).agg(" | ".join, axis=1)
                # --- Optimization: Use batch_size=64, convert_to_numpy=True ---
                batch_df["embedding"] = model.encode(
                    batch_df["combined_text"].tolist(),
                    show_progress_bar=True,
                    batch_size=BATCH_SIZE,
                    convert_to_numpy=True
                ).tolist()
                print(f"[DEBUG] Embeddings generated for batch {batch_count}")
                if selected_columns is None:
                    selected_columns = [col for col in batch_df.columns if col in EXCLUDED_COLUMNS] + ["embedding"]
                final_batch = batch_df[selected_columns]
                print(f"[DEBUG] Columns selected: {selected_columns}")
                out_path = os.path.join(OUTPUT_DIR, f"{table_name.replace('.', '_')}.csv")
                header = not os.path.exists(out_path) and batch_count == 0
                final_batch.to_csv(out_path, mode='a', header=header, index=False)
                print(f"[DEBUG] Saved batch {batch_count} to {out_path}")
                batch_count += 1
    elif source_type == "parquet":
        # Read and process each parquet file one at a time to avoid OOM
        parquet_dir = os.path.join("downloads", table_name)
        print(f"[DEBUG] Reading parquet files from {parquet_dir}")
        all_files = [os.path.join(parquet_dir, f) for f in os.listdir(parquet_dir) if f.endswith(".parquet")]
        if not all_files:
            print(f"[ERROR] No parquet files found in {parquet_dir}")
            return
        for file_idx, f in enumerate(all_files):
            print(f"[DEBUG] Processing parquet file {file_idx+1}/{len(all_files)}: {f}")
            df = pd.read_parquet(f)
            print(f"[DEBUG] Loaded {len(df)} rows from {f}")
            for batch_start in range(0, len(df), BATCH_SIZE):
                batch_end = min(batch_start + BATCH_SIZE, len(df))
                print(f"[DEBUG] Processing parquet batch: {batch_start} - {batch_end}")
                batch_df = df.iloc[batch_start:batch_end].copy()
                columns_to_concat = [col for col in batch_df.columns if col not in EXCLUDED_COLUMNS]
                # --- Optimization: Use batch_size=64, convert_to_numpy=True ---
                batch_df["combined_text"] = batch_df[columns_to_concat].astype(str).agg(" | ".join, axis=1)
                batch_df["embedding"] = model.encode(
                    batch_df["combined_text"].tolist(),
                    show_progress_bar=True,
                    batch_size=BATCH_SIZE,
                    convert_to_numpy=True
                ).tolist()
                print(f"[DEBUG] Embeddings generated for batch {batch_count}")
                if selected_columns is None:
                    selected_columns = [col for col in batch_df.columns if col in EXCLUDED_COLUMNS] + ["embedding"]
                final_batch = batch_df[selected_columns]
                print(f"[DEBUG] Columns selected: {selected_columns}")
                out_path = os.path.join(OUTPUT_DIR, f"{table_name.replace('.', '_')}.csv")
                header = not os.path.exists(out_path) and batch_count == 0
                final_batch.to_csv(out_path, mode='a', header=header, index=False)
                print(f"[DEBUG] Saved batch {batch_count} to {out_path}")
                batch_count += 1
    else:
        print(f"[ERROR] Unknown source_type: {source_type}")
        return
    print(f"Completed processing for {table_name}")

if __name__ == "__main__":
    # Example usage: process a table from parquet files
    process_data_in_batches(
        table_name="epcDomesticEngWalesRec",
        selected_columns=["lmk_key", "embedding"],
        source_type="parquet"
    )
    process_data_in_batches(
        table_name="LH_external_datasets.epc.epcDomesticEngWales",
        selected_columns=["lmk_key", "uprn", "building_reference_number", "postcode_trim", "embedding"],
        pk_col="lmk_key",  # Change to your actual PK column
        source_type="sql"
    )
    # To process more tables, call process_data_in_batches() with different args

# ---
# For scalable, production-grade vector search, consider using FAISS (in-memory, fast, open source),
# Qdrant, or Milvus (both open source, scalable, and support disk persistence and REST APIs).
# This script saves to CSV for simplicity and ECS compatibility. 