#!/usr/bin/env python3
import os
import sys
from qdrant_client import QdrantClient

def test_qdrant_connection():
    print("🔍 Testing Qdrant Connection (App Style)\n" + "=" * 40)
    
    # Test 1: Direct connection (like our working test)
    try:
        print("Test 1: Direct connection...")
        qdrant_url = os.getenv("QDRANT_URL")
        print(f"QDRANT_URL: {qdrant_url}")
        
        if qdrant_url:
            client = QdrantClient(url=qdrant_url, prefer_grpc=True)
            collections = client.get_collections()
            print(f"✅ Direct connection successful: {len(collections.collections)} collections")
        else:
            print("❌ QDRANT_URL not set")
            return False
    except Exception as e:
        print(f"❌ Direct connection failed: {e}")
        return False
    
    # Test 2: QdrantIndex import and instantiation (like the app)
    try:
        print("\nTest 2: QdrantIndex import and instantiation...")
        from utils.qdrant_utils import QdrantIndex
        
        print("✅ QdrantIndex import successful")
        
        # Try to create a QdrantIndex instance
        qdrant_index = QdrantIndex(
            collection_name="test_collection",
            embedding_model=None
        )
        print("✅ QdrantIndex instantiation successful")
        
        # Try to access the client property
        client = qdrant_index.client
        print("✅ QdrantIndex client access successful")
        
        # Try to get collections
        collections = client.get_collections()
        print(f"✅ QdrantIndex collections access successful: {len(collections.collections)} collections")
        
    except Exception as e:
        print(f"❌ QdrantIndex test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n✅ All tests passed!")
    return True

if __name__ == "__main__":
    success = test_qdrant_connection()
    if success:
        print("\n✅ Qdrant connection test successful!")
        sys.exit(0)
    else:
        print("\n❌ Qdrant connection test failed!")
        sys.exit(1) 