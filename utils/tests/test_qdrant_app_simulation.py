#!/usr/bin/env python3
import os
import sys

def test_app_simulation():
    print("🔍 Testing App Simulation\n" + "=" * 40)
    
    # Simulate the exact same code path as the main app
    try:
        # Step 1: Import QdrantIndex (like the app does)
        print("1. Importing QdrantIndex...")
        from utils.qdrant_utils import QdrantIndex
        print("✅ QdrantIndex import successful")
        
        # Step 2: Get model (like the app does)
        print("2. Getting embedding model...")
        from utils.rag_utils import RAGUtils
        model = RAGUtils.get_embedding_model("all-MiniLM-L6-v2")
        print("✅ Model creation successful")
        
        # Step 3: Create QdrantIndex (like the app does)
        print("3. Creating QdrantIndex...")
        qdrant_index = QdrantIndex(
            collection_name="bev",  # Use existing collection
            embedding_model=model
        )
        print("✅ QdrantIndex creation successful")
        
        # Step 4: Perform search (like the app does)
        print("4. Performing search...")
        search_results = qdrant_index.search(
            "test question",
            limit=5,
            with_payload=True
        )
        print(f"✅ Search successful: {len(search_results)} results")
        
        # Step 5: Process results (like the app does)
        print("5. Processing results...")
        if search_results:
            payloads = [result.payload for result in search_results]
            print(f"✅ Results processing successful: {len(payloads)} payloads")
        else:
            print("⚠️ No search results found")
        
        print("\n✅ App simulation successful!")
        return True
        
    except Exception as e:
        print(f"❌ App simulation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_app_simulation()
    if success:
        print("\n✅ App simulation test successful!")
        sys.exit(0)
    else:
        print("\n❌ App simulation test failed!")
        sys.exit(1) 