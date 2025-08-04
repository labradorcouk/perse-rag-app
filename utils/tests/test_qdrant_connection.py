#!/usr/bin/env python3
"""
Test Qdrant Connection Script
This script tests the connection to Qdrant from within the Docker container.
"""

import os
import sys
from qdrant_client import QdrantClient

def test_qdrant_connection():
    """Test connection to Qdrant using different methods"""
    
    print("🔍 Testing Qdrant Connection")
    print("=" * 40)
    
    # Try different connection methods
    connection_methods = [
        ("Environment Variables", lambda: QdrantClient(url=os.getenv("QDRANT_URL"), prefer_grpc=True) if os.getenv("QDRANT_URL") else None),
        ("Docker Service Name", lambda: QdrantClient("qdrant", port=6333, prefer_grpc=True)),
        ("localhost", lambda: QdrantClient("localhost", port=6333, prefer_grpc=True)),
        ("host.docker.internal", lambda: QdrantClient("host.docker.internal", port=6333, prefer_grpc=True)),
    ]
    
    for method_name, method in connection_methods:
        try:
            print(f"Trying {method_name}...")
            client = method()
            if client:
                # Test the connection
                collections = client.get_collections()
                print(f"✅ Successfully connected using {method_name}")
                print(f"   Available collections: {len(collections.collections)}")
                return client
        except Exception as e:
            print(f"❌ Failed with {method_name}: {e}")
            continue
    
    print("❌ Could not connect to Qdrant using any method")
    return None

if __name__ == "__main__":
    client = test_qdrant_connection()
    if client:
        print("\n✅ Qdrant connection test successful!")
        sys.exit(0)
    else:
        print("\n❌ Qdrant connection test failed!")
        sys.exit(1) 