import os
import yaml
import json
import pandas as pd
import traceback
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader
from utils.rag_utils import RAGUtils

# Utility to load YAML config
def load_config(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

# Utility to load or initialize used data tracker
def load_used_tracker(path):
    if os.path.exists(path):
        with open(path, 'r') as f:
            return json.load(f)
    return {}

def save_used_tracker(path, tracker):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        json.dump(tracker, f, indent=2)

# Get unique IDs for a DataFrame (for deduplication)
def get_unique_ids(df, id_cols):
    if not id_cols or not all(col in df.columns for col in id_cols):
        # Fallback: hash the row as a string
        return df.astype(str).apply(lambda row: hash(tuple(row)), axis=1).tolist()
    return df[id_cols].astype(str).agg('-'.join, axis=1).tolist()

# Save DataFrame to CSV (for QA pairs)
def save_qa_pairs(df, table_name, debug=False):
    folder = os.path.join('data', 'qa_pairs')
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, f'{table_name}.csv')
    df.to_csv(path, index=False)
    if debug:
        print(f"Saved QA pairs for {table_name} to {path}")

# Save DataFrame to Parquet (for embeddings)
def save_embeddings(df, table_name, debug=False):
    folder = os.path.join('data', 'embeddings')
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, f'{table_name}.parquet')
    df.to_parquet(path, index=False)
    if debug:
        print(f"Saved embeddings for {table_name} to {path}")

# Main incremental fine-tuning utility
def main(config_path, debug=False):
    def debug_print(*args, **kwargs):
        if debug:
            print(*args, **kwargs)

    try:
        config = load_config(config_path)
        debug_print(f"Loaded config from {config_path}")
    except Exception as e:
        print(f"[ERROR] Failed to load config: {e}")
        if debug:
            traceback.print_exc()
        return

    try:
        tracker = load_used_tracker(config['used_data_tracker'])
        debug_print(f"Loaded used data tracker from {config['used_data_tracker']}")
    except Exception as e:
        print(f"[ERROR] Failed to load used data tracker: {e}")
        if debug:
            traceback.print_exc()
        tracker = {}

    try:
        engine = RAGUtils.get_fabric_engine()
        debug_print("SQL engine initialized.")
    except Exception as e:
        print(f"[ERROR] Failed to initialize SQL engine: {e}")
        if debug:
            traceback.print_exc()
        return

    train_examples = []
    table_id_cols = {}  # Optionally specify unique ID columns per table

    for table in config['tables']:
        name = table['name']
        debug_print(f"\nProcessing table: {name}")
        tracker.setdefault(name, {})
        # --- QA pairs ---
        try:
            qa_df = pd.read_sql(table['qa_query'], engine)
            debug_print(f"Loaded {len(qa_df)} QA pairs from {name}")
            if len(qa_df) > 0:
                save_qa_pairs(qa_df, name, debug=debug)
        except Exception as e:
            print(f"[ERROR] Failed to load QA pairs for {name}: {e}")
            if debug:
                traceback.print_exc()
            qa_df = pd.DataFrame()
        qa_id_col = 'question_id' if 'question_id' in qa_df.columns else (qa_df.columns[0] if len(qa_df.columns) > 0 else None)
        qa_ids = get_unique_ids(qa_df, [qa_id_col]) if qa_id_col else []
        used_qa_ids = set(tracker[name].get('qa_ids', []))
        new_qa_mask = [i for i, id_ in enumerate(qa_ids) if id_ not in used_qa_ids]
        new_qa = qa_df.iloc[new_qa_mask] if len(new_qa_mask) > 0 else pd.DataFrame()
        debug_print(f"QA: {len(new_qa)} new, {len(qa_df)-len(new_qa)} skipped (already used)")
        for _, row in new_qa.iterrows():
            if 'question' in row and 'answer' in row:
                train_examples.append(InputExample(texts=[str(row['question']), str(row['answer'])], label=1.0))
        tracker[name]['qa_ids'] = list(used_qa_ids.union([qa_ids[i] for i in new_qa_mask]))
        if debug:
            debug_print(f"Updated tracker QA IDs for {name}: {tracker[name]['qa_ids'][-5:]}")

        # --- Raw data (optional: self-supervised pairs) ---
        try:
            raw_df = pd.read_sql(table['raw_query'], engine)
            debug_print(f"Loaded {len(raw_df)} raw rows from {name}")
        except Exception as e:
            print(f"[ERROR] Failed to load raw data for {name}: {e}")
            if debug:
                traceback.print_exc()
            raw_df = pd.DataFrame()
        raw_id_col = 'row_id' if 'row_id' in raw_df.columns else (raw_df.columns[0] if len(raw_df.columns) > 0 else None)
        raw_ids = get_unique_ids(raw_df, [raw_id_col]) if raw_id_col else []
        used_raw_ids = set(tracker[name].get('raw_ids', []))
        new_raw_mask = [i for i, id_ in enumerate(raw_ids) if id_ not in used_raw_ids]
        new_raw = raw_df.iloc[new_raw_mask] if len(new_raw_mask) > 0 else pd.DataFrame()
        debug_print(f"Raw: {len(new_raw)} new, {len(raw_df)-len(new_raw)} skipped (already used)")
        for _, row in new_raw.iterrows():
            cols = list(raw_df.columns)
            if len(cols) >= 2:
                train_examples.append(InputExample(texts=[str(row[cols[0]]), str(row[cols[1]])], label=1.0))
        tracker[name]['raw_ids'] = list(used_raw_ids.union([raw_ids[i] for i in new_raw_mask]))
        if debug:
            debug_print(f"Updated tracker raw IDs for {name}: {tracker[name]['raw_ids'][-5:]}")

        # --- Vector embeddings (optional: use as additional positives/negatives) ---
        try:
            emb_df = pd.read_sql(table['embedding_query'], engine)
            debug_print(f"Loaded {len(emb_df)} embeddings from {name}")
            if len(emb_df) > 0:
                save_embeddings(emb_df, name, debug=debug)
        except Exception as e:
            print(f"[ERROR] Failed to load embeddings for {name}: {e}")
            if debug:
                traceback.print_exc()
            emb_df = pd.DataFrame()
        emb_id_col = 'embedding_id' if 'embedding_id' in emb_df.columns else (emb_df.columns[0] if len(emb_df.columns) > 0 else None)
        emb_ids = get_unique_ids(emb_df, [emb_id_col]) if emb_id_col else []
        used_emb_ids = set(tracker[name].get('embedding_ids', []))
        new_emb_mask = [i for i, id_ in enumerate(emb_ids) if id_ not in used_emb_ids]
        new_emb = emb_df.iloc[new_emb_mask] if len(new_emb_mask) > 0 else pd.DataFrame()
        debug_print(f"Embeddings: {len(new_emb)} new, {len(emb_df)-len(new_emb)} skipped (already used)")
        for _, row in new_emb.iterrows():
            cols = list(emb_df.columns)
            if len(cols) >= 2:
                train_examples.append(InputExample(texts=[str(row[cols[0]]), str(row[cols[1]])], label=1.0))
        tracker[name]['embedding_ids'] = list(used_emb_ids.union([emb_ids[i] for i in new_emb_mask]))
        if debug:
            debug_print(f"Updated tracker embedding IDs for {name}: {tracker[name]['embedding_ids'][-5:]}")

    if not train_examples:
        print("No new data to train on. Exiting.")
        return

    print(f"Training on {len(train_examples)} new examples...")
    try:
        model = SentenceTransformer(config['base_model'])
        train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=config['batch_size'])
        train_loss = losses.MultipleNegativesRankingLoss(model)
        model.fit(train_objectives=[(train_dataloader, train_loss)], epochs=config['epochs'], show_progress_bar=True)
        os.makedirs(config['output_dir'], exist_ok=True)
        model.save(config['output_dir'])
        print(f"Model saved to {config['output_dir']}")
    except Exception as e:
        print(f"[ERROR] Training or saving model failed: {e}")
        if debug:
            traceback.print_exc()
        return

    try:
        save_used_tracker(config['used_data_tracker'], tracker)
        print("Used data tracker updated.")
    except Exception as e:
        print(f"[ERROR] Failed to update used data tracker: {e}")
        if debug:
            traceback.print_exc()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Grounded incremental fine-tuning across multiple tables.")
    parser.add_argument('--config', type=str, required=True, help='Path to YAML config file')
    parser.add_argument('--debug', action='store_true', help='Enable debug output')
    args = parser.parse_args()
    main(args.config, debug=args.debug) 