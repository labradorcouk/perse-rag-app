#!/usr/bin/env python3
import os
import sys
from qdrant_client import QdrantClient

def test_qdrant_debug():
    print("🔍 Comprehensive Qdrant Debug Test\n" + "=" * 50)
    
    # Test 1: Environment variables
    print("\n1. Testing Environment Variables...")
    qdrant_url = os.getenv("QDRANT_URL")
    print(f"QDRANT_URL: {qdrant_url}")
    
    if not qdrant_url:
        print("❌ QDRANT_URL not set")
        return False
    
    # Test 2: Direct QdrantClient connection
    print("\n2. Testing Direct QdrantClient Connection...")
    try:
        client = QdrantClient(url=qdrant_url, prefer_grpc=True)
        collections = client.get_collections()
        print(f"✅ Direct connection successful: {len(collections.collections)} collections")
        print(f"Collections: {[c.name for c in collections.collections]}")
    except Exception as e:
        print(f"❌ Direct connection failed: {e}")
        return False
    
    # Test 3: QdrantIndex import
    print("\n3. Testing QdrantIndex Import...")
    try:
        from utils.qdrant_utils import QdrantIndex
        print("✅ QdrantIndex import successful")
    except Exception as e:
        print(f"❌ QdrantIndex import failed: {e}")
        return False
    
    # Test 4: QdrantIndex instantiation
    print("\n4. Testing QdrantIndex Instantiation...")
    try:
        qdrant_index = QdrantIndex(
            collection_name="bev",  # Use an existing collection
            embedding_model=None
        )
        print("✅ QdrantIndex instantiation successful")
    except Exception as e:
        print(f"❌ QdrantIndex instantiation failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test 5: Client property access
    print("\n5. Testing Client Property Access...")
    try:
        client = qdrant_index.client
        print("✅ Client property access successful")
    except Exception as e:
        print(f"❌ Client property access failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test 6: Collections access through QdrantIndex
    print("\n6. Testing Collections Access through QdrantIndex...")
    try:
        collections = qdrant_index.client.get_collections()
        print(f"✅ Collections access successful: {len(collections.collections)} collections")
    except Exception as e:
        print(f"❌ Collections access failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test 7: Search method (without embedding model)
    print("\n7. Testing Search Method (without embedding model)...")
    try:
        # This should fail because embedding_model is None
        search_results = qdrant_index.search("test query", limit=5)
        print("✅ Search method successful")
    except Exception as e:
        print(f"❌ Search method failed (expected): {e}")
        print("This is expected because embedding_model is None")
    
    # Test 8: RAGUtils import and model creation
    print("\n8. Testing RAGUtils and Model Creation...")
    try:
        from utils.rag_utils import RAGUtils
        print("✅ RAGUtils import successful")
        
        # Test model creation
        model = RAGUtils.get_embedding_model("all-MiniLM-L6-v2")
        print("✅ Model creation successful")
        
        # Test QdrantIndex with model
        qdrant_index_with_model = QdrantIndex(
            collection_name="bev",
            embedding_model=model
        )
        print("✅ QdrantIndex with model successful")
        
        # Test search with model
        search_results = qdrant_index_with_model.search("test query", limit=5)
        print(f"✅ Search with model successful: {len(search_results)} results")
        
    except Exception as e:
        print(f"❌ RAGUtils/model test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n✅ All tests passed!")
    return True

if __name__ == "__main__":
    success = test_qdrant_debug()
    if success:
        print("\n✅ Qdrant debug test successful!")
        sys.exit(0)
    else:
        print("\n❌ Qdrant debug test failed!")
        sys.exit(1) 