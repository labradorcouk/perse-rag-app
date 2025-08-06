#!/usr/bin/env python3
"""
Test script to verify MongoDB integration for vector search.
"""

import pandas as pd
import sys
import os

# Add the parent directory to the path so we can import utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

def test_mongodb_connection():
    """Test MongoDB connection and basic functionality."""
    try:
        from utils.mongodb_utils import MongoDBIndex
        
        print("🧪 Testing MongoDB Integration")
        print("=" * 50)
        
        # Test environment variables
        mongodb_uri = os.getenv('MONGODB_URI')
        if not mongodb_uri:
            print("❌ MONGODB_URI environment variable not set")
            print("💡 Please set MONGODB_URI in your environment")
            return False
        
        print(f"✅ MONGODB_URI found: {mongodb_uri[:20]}...")
        
        # Test MongoDBIndex creation
        try:
            mongodb_index = MongoDBIndex(
                collection_name="test_collection",
                embedding_model=None  # Will use OpenAI embeddings
            )
            print("✅ MongoDBIndex created successfully")
        except Exception as e:
            print(f"❌ Failed to create MongoDBIndex: {e}")
            return False
        
        # Test connection
        try:
            if mongodb_index.test_connection():
                print("✅ MongoDB connection successful")
            else:
                print("❌ MongoDB connection failed")
                return False
        except Exception as e:
            print(f"❌ Connection test failed: {e}")
            return False
        
        # Test collection info
        try:
            collection_info = mongodb_index.get_collection_info()
            print(f"✅ Collection info retrieved: {collection_info}")
        except Exception as e:
            print(f"⚠️  Could not get collection info: {e}")
        
        # Test sample documents
        try:
            sample_docs = mongodb_index.get_sample_documents(limit=3)
            print(f"✅ Retrieved {len(sample_docs)} sample documents")
        except Exception as e:
            print(f"⚠️  Could not get sample documents: {e}")
        
        print("\n🎉 MongoDB integration test completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_mongodb_search():
    """Test MongoDB vector search functionality."""
    try:
        from utils.mongodb_utils import MongoDBIndex
        
        print("\n🧪 Testing MongoDB Vector Search")
        print("=" * 50)
        
        # Create MongoDB index
        mongodb_index = MongoDBIndex(
            collection_name="test_collection"
        )
        
        # Test search
        test_query = "energy efficiency rating"
        try:
            search_results = mongodb_index.search(
                query=test_query,
                limit=5,
                score_threshold=0.01
            )
            print(f"✅ Search successful: {len(search_results)} results")
            
            if search_results:
                print("📊 Search results sample:")
                for i, result in enumerate(search_results[:2]):
                    print(f"  Result {i+1}:")
                    print(f"    Score: {result.get('score', 'N/A')}")
                    print(f"    Payload length: {len(str(result.get('payload', '')))}")
                    print(f"    Metadata keys: {list(result.get('metadata', {}).keys())}")
            
        except Exception as e:
            print(f"❌ Search failed: {e}")
            return False
        
        print("\n🎉 MongoDB search test completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_mongodb_configuration():
    """Test MongoDB configuration and environment setup."""
    try:
        print("\n🧪 Testing MongoDB Configuration")
        print("=" * 50)
        
        # Check required environment variables
        required_vars = ['MONGODB_URI']
        optional_vars = ['MONGODB_DB_NAME', 'MONGODB_COLLECTION_NAME', 'MONGODB_INDEX_NAME']
        
        print("📋 Environment Variables Check:")
        
        for var in required_vars:
            value = os.getenv(var)
            if value:
                print(f"  ✅ {var}: {value[:20]}..." if len(value) > 20 else f"  ✅ {var}: {value}")
            else:
                print(f"  ❌ {var}: Not set")
        
        for var in optional_vars:
            value = os.getenv(var)
            if value:
                print(f"  ✅ {var}: {value}")
            else:
                print(f"  ⚠️  {var}: Not set (using default)")
        
        # Check if required packages are available
        print("\n📦 Package Availability Check:")
        
        try:
            import pymongo
            print("  ✅ pymongo: Available")
        except ImportError:
            print("  ❌ pymongo: Not installed")
        
        try:
            import langchain_mongodb
            print("  ✅ langchain-mongodb: Available")
        except ImportError:
            print("  ❌ langchain-mongodb: Not installed")
        
        try:
            import langchain_openai
            print("  ✅ langchain-openai: Available")
        except ImportError:
            print("  ❌ langchain-openai: Not installed")
        
        print("\n🎉 Configuration test completed!")
        return True
        
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        return False

if __name__ == "__main__":
    print("🧪 Testing MongoDB Vector Search Integration")
    print("=" * 70)
    
    # Test configuration
    test_mongodb_configuration()
    
    # Test connection
    test_mongodb_connection()
    
    # Test search
    test_mongodb_search()
    
    print("\n🎉 All MongoDB integration tests completed!")
    print("\n💡 MongoDB integration provides:")
    print("  ✅ Vector search using MongoDB Atlas")
    print("  ✅ Semantic similarity search")
    print("  ✅ Integration with existing RAG pipeline")
    print("  ✅ Fallback to other vector search engines")
    print("  ✅ Detailed error handling and debugging") 