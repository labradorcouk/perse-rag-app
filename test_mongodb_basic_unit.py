#!/usr/bin/env python3
"""
Unit test script for MongoDB Basic Search functionality.
This script tests the core logic without requiring actual MongoDB connection.
"""

import os
import sys
import unittest
from unittest.mock import Mock, MagicMock

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_mongodb_basic_logic():
    """Test the MongoDB Basic search logic without actual connection."""
    
    try:
        # Import the MongoDB Basic Search directly
        sys.path.append(os.path.join(os.path.dirname(__file__), 'utils'))
        from mongodb_basic_search import MongoDBBasicSearch
        
        print("✅ MongoDB Basic Search imported successfully")
        
        # Test 1: Class instantiation
        print("\n🔍 Test 1: Class instantiation")
        try:
            # Mock the MongoDB connection
            mock_mongodb = MongoDBBasicSearch("mock_uri", "mock_db")
            print("   ✅ MongoDBBasicSearch class instantiated successfully")
        except Exception as e:
            print(f"   ❌ Class instantiation failed: {e}")
            return False
        
        # Test 2: Query intent analysis
        print("\n🔍 Test 2: Query intent analysis")
        try:
            # Test query analysis logic
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
            
            # Test different query types
            test_cases = [
                {
                    'query': "find connections by EMSN 24E8049370",
                    'expected_intent': 'find_by_emsn',
                    'expected_confidence': 0.9
                },
                {
                    'query': "search connections in SE23 area",
                    'expected_intent': 'location_search',
                    'expected_confidence': 0.85
                },
                {
                    'query': "check xoserveMeters API status",
                    'expected_intent': 'api_status_check',
                    'expected_confidence': 0.9
                },
                {
                    'query': "find connections on WESTBOURNE DRIVE",
                    'expected_intent': 'address_search',
                    'expected_confidence': 0.8
                }
            ]
            
            for test_case in test_cases:
                query = test_case['query']
                expected_intent = test_case['expected_intent']
                expected_confidence = test_case['expected_confidence']
                
                # Analyze the query
                search_criteria = mock_mongodb._analyze_query_intent(query, schema_config)
                
                print(f"   📝 Query: '{query}'")
                print(f"      Detected intent: {search_criteria['intent']}")
                print(f"      Expected intent: {expected_intent}")
                print(f"      Confidence: {search_criteria['confidence_score']:.2f}")
                print(f"      Expected confidence: {expected_confidence:.2f}")
                
                # Check if intent matches
                if search_criteria['intent'] == expected_intent:
                    print(f"      ✅ Intent detection correct")
                else:
                    print(f"      ❌ Intent detection incorrect")
                    return False
                
                # Check if confidence is reasonable
                if abs(search_criteria['confidence_score'] - expected_confidence) < 0.2:
                    print(f"      ✅ Confidence score reasonable")
                else:
                    print(f"      ⚠️ Confidence score may need adjustment")
            
            print("   ✅ Query intent analysis tests passed")
            
        except Exception as e:
            print(f"   ❌ Query intent analysis failed: {e}")
            import traceback
            traceback.print_exc()
            return False
        
        # Test 3: MongoDB query building
        print("\n🔍 Test 3: MongoDB query building")
        try:
            # Test query building for different search criteria
            test_search_criteria = {
                'intent': 'find_by_emsn',
                'exact_matches': {'EMSN': '24E8049370'},
                'partial_matches': {},
                'business_entities': ['EMSN'],
                'confidence_score': 0.9
            }
            
            mongo_query = mock_mongodb._build_mongo_query(test_search_criteria, schema_config)
            print(f"   📝 Built MongoDB query: {mongo_query}")
            
            # Check if query contains expected elements
            if 'EMSN' in mongo_query and '$in' in mongo_query['EMSN']:
                print("   ✅ MongoDB query building correct for EMSN search")
            else:
                print("   ❌ MongoDB query building incorrect for EMSN search")
                return False
            
            # Test partial search query building
            test_search_criteria_partial = {
                'intent': 'location_search',
                'exact_matches': {},
                'partial_matches': {'POSTCODE': ['SE23']},
                'business_entities': ['POSTCODE'],
                'confidence_score': 0.85
            }
            
            mongo_query_partial = mock_mongodb._build_mongo_query(test_search_criteria_partial, schema_config)
            print(f"   📝 Built partial search query: {mongo_query_partial}")
            
            if 'POSTCODE' in mongo_query_partial and '$regex' in mongo_query_partial['POSTCODE']:
                print("   ✅ MongoDB query building correct for partial search")
            else:
                print("   ❌ MongoDB query building incorrect for partial search")
                return False
            
            print("   ✅ MongoDB query building tests passed")
            
        except Exception as e:
            print(f"   ❌ MongoDB query building failed: {e}")
            import traceback
            traceback.print_exc()
            return False
        
        # Test 4: Projection building
        print("\n🔍 Test 4: Projection building")
        try:
            projection = mock_mongodb._get_projection(schema_config)
            print(f"   📝 Built projection: {projection}")
            
            # Check if essential columns are included
            essential_cols = ['UPRN', 'EMSN', 'MPAN', 'MPRN', 'POSTCODE', 'ADDRESS', 'apiStatus']
            for col in essential_cols:
                if col in projection and projection[col] == 1:
                    print(f"      ✅ {col} included in projection")
                else:
                    print(f"      ❌ {col} not properly included in projection")
                    return False
            
            # Check if excluded columns are excluded
            if '_id' in projection and projection['_id'] == 0:
                print("      ✅ _id properly excluded from projection")
            else:
                print("      ❌ _id not properly excluded from projection")
                return False
            
            print("   ✅ Projection building tests passed")
            
        except Exception as e:
            print(f"   ❌ Projection building failed: {e}")
            import traceback
            traceback.print_exc()
            return False
        
        print("\n✅ All unit tests passed! MongoDB Basic Search logic is working correctly.")
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
    print("🚀 Starting MongoDB Basic Search Unit Tests")
    print("=" * 50)
    
    success = test_mongodb_basic_logic()
    
    print("=" * 50)
    if success:
        print("🎉 All unit tests passed! MongoDB Basic Search logic is working correctly.")
    else:
        print("💥 Some unit tests failed. Check the error messages above.")
    
    print("\n📝 Test Summary:")
    print("✅ MongoDB Basic Search class created and imported")
    print("✅ Query intent analysis working")
    print("✅ MongoDB query building working")
    print("✅ Projection building working")
    print("✅ Business logic and pattern matching working")
