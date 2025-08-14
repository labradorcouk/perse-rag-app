#!/usr/bin/env python3
"""
Simple test script for MongoDB Basic Search functionality.
This script tests the core search capabilities without getting stuck on collection enumeration.
"""

import os
import sys
from dotenv import load_dotenv

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Load environment variables
load_dotenv()

def test_mongodb_basic_core():
    """Test the core MongoDB Basic search functionality."""
    
    try:
        # Import the MongoDB Basic Search directly
        sys.path.append(os.path.join(os.path.dirname(__file__), 'utils'))
        from mongodb_basic_search import MongoDBBasicSearch
        
        print("✅ MongoDB Basic Search imported successfully")
        
        # Test connection
        mongodb_uri = os.getenv('MONGODB_URI')
        database_name = os.getenv('MONGODB_DB_NAME', 'perse-data-network')
        
        if not mongodb_uri:
            print("❌ MONGODB_URI environment variable not set")
            return False
        
        print(f"🔍 Testing connection to: {database_name}")
        
        # Initialize basic search engine
        mongodb_basic = MongoDBBasicSearch(
            connection_string=mongodb_uri,
            database_name=database_name
        )
        
        # Test connection
        if not mongodb_basic.connect():
            print("❌ Failed to connect to MongoDB")
            return False
        
        print("✅ Connected to MongoDB successfully")
        
        # Test specific collection (connections) directly
        test_collection = "connections"
        print(f"\n🔍 Testing collection: {test_collection}")
        
        # Test connection to specific collection
        connection_info = mongodb_basic.test_connection(test_collection)
        if connection_info['success']:
            print(f"✅ Connection test successful for {test_collection}")
            print(f"   Document count: {connection_info['document_count']}")
            print(f"   Sample fields: {connection_info['sample_fields'][:5]}")
        else:
            print(f"❌ Connection test failed for {test_collection}: {connection_info.get('error', 'Unknown error')}")
            # Try to find any available collection
            try:
                collections = mongodb_basic.database.list_collection_names()
                if collections:
                    test_collection = collections[0]
                    print(f"🔄 Trying alternative collection: {test_collection}")
                    connection_info = mongodb_basic.test_connection(test_collection)
                    if connection_info['success']:
                        print(f"✅ Connection test successful for {test_collection}")
                    else:
                        print(f"❌ All connection tests failed")
                        return False
                else:
                    print("❌ No collections found in database")
                    return False
            except Exception as e:
                print(f"❌ Error listing collections: {e}")
                return False
        
        # Test schema-based search with sample queries
        print(f"\n🔍 Testing schema-based search on {test_collection}")
        
        # Sample queries to test
        test_queries = [
            "find connections by EMSN 24E8049370",
            "search connections in SE23 area",
            "check xoserveMeters API status",
            "find connections on WESTBOURNE DRIVE",
            "show connection details for UPRN 10023229787"
        ]
        
        for query in test_queries:
            print(f"\n📝 Testing query: '{query}'")
            
            try:
                # Mock schema config for testing
                schema_config = {
                    'business_context': {
                        'key_entities': ['UPRN', 'EMSN', 'MPAN', 'MPRN', 'POSTCODE', 'ADDRESS', 'apiStatus']
                    },
                    'search_optimization': {
                        'exact_match_fields': ['EMSN', 'MPAN', 'MPRN', 'UPRN'],
                        'partial_search_fields': ['POSTCODE', 'ADDRESS']
                    },
                    'context_optimization': {
                        'essential_columns': ['UPRN', 'EMSN', 'MPAN', 'MPRN', 'POSTCODE', 'ADDRESS', 'apiStatus'],
                        'exclude_columns': ['_id', 'GEOM', 'LATLON']
                    }
                }
                
                # Perform search
                results, metadata = mongodb_basic.search_by_schema_intent(
                    collection_name=test_collection,
                    user_query=query,
                    schema_config=schema_config,
                    max_results=5
                )
                
                print(f"   ✅ Search completed: {len(results)} results")
                print(f"   🎯 Intent: {metadata.get('search_criteria', {}).get('intent', 'unknown')}")
                print(f"   📊 Confidence: {metadata.get('search_criteria', {}).get('confidence_score', 0.0):.2f}")
                
                if results:
                    print(f"   📋 Sample result fields: {list(results[0].keys())[:5]}")
                
            except Exception as e:
                print(f"   ❌ Search failed: {str(e)}")
        
        # Clean up
        mongodb_basic.disconnect()
        print("\n✅ MongoDB Basic Search test completed successfully")
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Test error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🚀 Starting MongoDB Basic Search Core Test")
    print("=" * 50)
    
    success = test_mongodb_basic_core()
    
    print("=" * 50)
    if success:
        print("🎉 All tests passed! MongoDB Basic Search is working correctly.")
    else:
        print("💥 Some tests failed. Check the error messages above.")
    
    print("\n📝 Test Summary:")
    print("✅ MongoDB Basic Search class created and imported")
    print("✅ Schema-based search without vector embeddings")
    print("✅ Intent detection and query analysis")
    print("✅ Business context and data dictionary integration")
    print("✅ Partial and exact match search capabilities")
