#!/usr/bin/env python3
import os
import sys
import traceback
import pandas as pd

def test_error_capture():
    print("🔍 Capturing Exact Error from Main App\n" + "=" * 50)
    
    # Simulate the exact same conditions as the main app
    try:
        # Step 1: Import and setup (like main app)
        print("1. Setting up imports...")
        from utils.qdrant_utils import QdrantIndex
        from utils.rag_utils import RAGUtils
        
        # Step 2: Get model (like main app)
        print("2. Getting embedding model...")
        model = RAGUtils.get_embedding_model("all-MiniLM-L6-v2")
        
        # Step 3: Simulate the exact same loop as main app
        print("3. Simulating main app loop...")
        
        # Simulate the selected_tables loop from main app
        selected_tables = ["bev"]  # Use existing collection
        
        for table_name in selected_tables:
            print(f"Processing table: {table_name}")
            
            # Simulate table_meta (this might be the issue!)
            table_meta = {
                'collection': 'bev',  # This should exist
                'display_name': 'Battery Electric Vehicles',
                'qdrant_columns': ['make', 'model', 'year']  # These might not exist
            }
            
            try:
                print(f"Connecting to Qdrant collection: {table_meta['collection']}")
                qdrant_index = QdrantIndex(
                    collection_name=table_meta['collection'],
                    embedding_model=model
                )
                print(f"✅ Connected to Qdrant collection: {table_meta['collection']}")
                
                # This is where the error might be happening
                search_results = qdrant_index.search(
                    "test question",
                    limit=5,
                    with_payload=True
                )
                print(f"✅ Search successful: {len(search_results)} results")
                
                if not search_results:
                    print(f"No relevant documents found in Qdrant for {table_meta['display_name']}.")
                    continue
                
                print(f"Building context from Qdrant search results for {table_meta['display_name']}...")
                payloads = [result.payload for result in search_results]
                df_context = pd.DataFrame(payloads)
                
                print(f"Available columns: {list(df_context.columns)}")
                
                # This might be causing the error - checking columns that don't exist
                available_qdrant_columns = [col for col in table_meta['qdrant_columns'] if col in df_context.columns]
                if available_qdrant_columns:
                    df_context = df_context[available_qdrant_columns]
                else:
                    print(f"No matching columns found for {table_meta['display_name']}. Using all available columns.")
                
                print("✅ Processing successful")
                
            except Exception as e:
                print(f"❌ Error in table processing: {e}")
                print("Full traceback:")
                traceback.print_exc()
                print("\nThis is the exact error that's happening in the main app!")
                return False
        
        print("\n✅ Error capture test completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Error capture test failed: {e}")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_error_capture()
    if success:
        print("\n✅ Error capture test successful!")
        sys.exit(0)
    else:
        print("\n❌ Error capture test failed!")
        sys.exit(1) 