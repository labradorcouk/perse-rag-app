import re
import json
from typing import Dict, List, Any, Optional, Tuple
from pymongo import MongoClient
from pymongo.collection import Collection
from pymongo.database import Database
import logging

class MongoDBBasicSearch:
    """
    MongoDB Basic Search Engine that focuses on schema-level search and data dictionary
    instead of row-level vector embeddings. This allows querying collections without
    vector embedding columns.
    """
    
    def __init__(self, connection_string: str, database_name: str):
        self.connection_string = connection_string
        self.database_name = database_name
        self.client: Optional[MongoClient] = None
        self.database: Optional[Database] = None
        self.logger = logging.getLogger(__name__)
        
    def connect(self) -> bool:
        """Establish connection to MongoDB with timeout."""
        try:
            # Set connection timeout to 5 seconds
            self.client = MongoClient(
                self.connection_string,
                serverSelectionTimeoutMS=5000,  # 5 second timeout
                connectTimeoutMS=5000,
                socketTimeoutMS=5000
            )
            self.database = self.client[self.database_name]
            # Test connection
            self.client.admin.command('ping')
            self.logger.info(f"✅ Connected to MongoDB database: {self.database_name}")
            return True
        except Exception as e:
            self.logger.error(f"❌ Failed to connect to MongoDB: {e}")
            return False
    
    def disconnect(self):
        """Close MongoDB connection."""
        if self.client:
            self.client.close()
            self.logger.info("🔌 MongoDB connection closed")
    
    def get_collection_info(self, collection_name: str) -> Dict[str, Any]:
        """Get basic information about a collection."""
        try:
            collection = self.database[collection_name]
            count = collection.count_documents({})
            
            # Get sample document to understand structure
            sample_doc = collection.find_one({})
            
            info = {
                'collection_name': collection_name,
                'database_name': self.database_name,
                'document_count': count,
                'has_sample_document': sample_doc is not None,
                'sample_fields': list(sample_doc.keys()) if sample_doc else []
            }
            
            if sample_doc:
                # Calculate approximate size - handle ObjectId serialization
                try:
                    # Convert ObjectId to string for JSON serialization
                    sample_doc_serializable = {}
                    for key, value in sample_doc.items():
                        if hasattr(value, '__class__') and value.__class__.__name__ == 'ObjectId':
                            sample_doc_serializable[key] = str(value)
                        else:
                            sample_doc_serializable[key] = value
                    
                    sample_size = len(json.dumps(sample_doc_serializable))
                    info['avg_document_size'] = sample_size
                    info['size_bytes'] = count * sample_size
                except Exception as json_error:
                    # Fallback: estimate size without JSON serialization
                    info['avg_document_size'] = 1000  # Default estimate
                    info['size_bytes'] = count * 1000
            
            return info
            
        except Exception as e:
            self.logger.error(f"❌ Error getting collection info for {collection_name}: {e}")
            return {}
    
    def search_by_schema_intent(self, collection_name: str, user_query: str, 
                               schema_config: Dict[str, Any], max_results: int = 20) -> Tuple[List[Dict], Dict[str, Any]]:
        """
        Search collection based on schema intent and data dictionary understanding.
        This method uses business logic and field patterns instead of vector embeddings.
        """
        try:
            collection = self.database[collection_name]
            
            # Analyze user query for intent and extract search criteria
            search_criteria = self._analyze_query_intent(user_query, schema_config)
            self.logger.info(f"🔍 Search criteria: {search_criteria}")
            
            # Build MongoDB query based on extracted criteria
            mongo_query = self._build_mongo_query(search_criteria, schema_config)
            self.logger.info(f"🔍 MongoDB query: {mongo_query}")
            
            # Execute search with appropriate projection
            projection = self._get_projection(schema_config)
            self.logger.info(f"🔍 Projection: {projection}")
            
            # Execute the search
            self.logger.info(f"🔍 Executing MongoDB query: {mongo_query}")
            self.logger.info(f"🔍 Using projection: {projection}")
            self.logger.info(f"🔍 Collection: {collection_name}")
            self.logger.info(f"🔍 Max results: {max_results}")
            
            cursor = collection.find(mongo_query, projection).limit(max_results)
            results = list(cursor)
            self.logger.info(f"🔍 Found {len(results)} results")
            
            # If no results, try a simple test to see if the collection has any data
            if not results:
                self.logger.info("🔍 No results found, testing collection access...")
                # Try to find any document without any query
                test_doc = collection.find_one({})
                if test_doc:
                    self.logger.info(f"🔍 Collection has data, sample document keys: {list(test_doc.keys())}")
                    # Check if UPRN field exists
                    if 'UPRN' in test_doc:
                        self.logger.info(f"🔍 UPRN field exists, sample value: {test_doc['UPRN']}")
                    else:
                        self.logger.info("🔍 UPRN field does not exist in sample document")
                else:
                    self.logger.info("🔍 Collection appears to be empty")
            
            # If no results, try a simple test query to see what's in the collection
            if not results and mongo_query:
                self.logger.info("🔍 No results found, testing with simple query...")
                # Try to find any document to see the data structure
                test_doc = collection.find_one({}, projection)
                if test_doc:
                    self.logger.info(f"🔍 Sample document fields: {list(test_doc.keys())}")
                    if 'UPRN' in test_doc:
                        self.logger.info(f"🔍 Sample UPRN value: {test_doc['UPRN']} (type: {type(test_doc['UPRN'])})")
                    if 'apiStatus' in test_doc:
                        self.logger.info(f"🔍 Sample apiStatus value: {test_doc['apiStatus']} (type: {type(test_doc['apiStatus'])})")
                
                # Also try a simple UPRN search to see if the field exists
                self.logger.info("🔍 Testing simple UPRN search...")
                
                simple_test = collection.find_one({'UPRN': {'$exists': True}}, projection)
                if simple_test:
                    self.logger.info(f"🔍 Found document with UPRN field: {simple_test.get('UPRN', 'N/A')}")
                else:
                    self.logger.info("🔍 No documents found with UPRN field")
                
                # Try to find the specific UPRN value mentioned in the query
                if 'UPRN' in search_criteria.get('exact_matches', {}):
                    uprn_value = search_criteria['exact_matches']['UPRN']
                    self.logger.info(f"🔍 Testing specific UPRN search for: {uprn_value}")
                    specific_test = collection.find_one({'UPRN': uprn_value}, projection)
                    if specific_test:
                        self.logger.info(f"🔍 Found document with specific UPRN {uprn_value}")
                    else:
                        self.logger.info(f"🔍 No document found with specific UPRN {uprn_value}")
            
            # Enhance results with business context
            enhanced_results = self._enhance_results_with_context(results, search_criteria, schema_config)
            
            search_metadata = {
                'collection_name': collection_name,
                'search_criteria': search_criteria,
                'mongo_query': mongo_query,
                'results_count': len(results),
                'max_results': max_results,
                'search_strategy': 'schema_based_intent_search'
            }
            
            return enhanced_results, search_metadata
            
        except Exception as e:
            self.logger.error(f"❌ Error in schema-based search: {e}")
            return [], {'error': str(e)}
    
    def _analyze_query_intent(self, user_query: str, schema_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze user query to extract search intent and criteria based on schema configuration.
        """
        query_lower = user_query.lower()
        search_criteria = {
            'intent': 'general_search',
            'search_fields': [],
            'exact_matches': {},
            'partial_matches': {},
            'business_entities': [],
            'confidence_score': 0.0
        }
        
        # Extract business context from schema
        business_context = schema_config.get('business_context', {})
        business_keywords = business_context.get('key_entities', [])
        
        # Check for exact identifier searches
        exact_match_fields = schema_config.get('search_optimization', {}).get('exact_match_fields', [])
        self.logger.info(f"🔍 Debug: Exact match fields from schema: {exact_match_fields}")
        self.logger.info(f"🔍 Debug: Query lower: {query_lower}")
        self.logger.info(f"🔍 Debug: Query contains 'uprn': {'uprn' in query_lower}")
        self.logger.info(f"🔍 Debug: Query contains 'UPRN': {'UPRN' in query_lower}")
        
        for field in exact_match_fields:
            field_lower = field.lower()
            self.logger.info(f"🔍 Debug: Checking field: {field} (lower: {field_lower})")
            if field_lower in query_lower:
                self.logger.info(f"🔍 Debug: Field {field_lower} found in query")
                # Look for patterns like "EMSN 24E8049370" or "UPRN 10023229787"
                # Use a more flexible pattern that captures the full identifier
                pattern = rf'{field_lower}\s+([A-Za-z0-9]+)'
                self.logger.info(f"🔍 Debug: Using pattern: {pattern}")
                self.logger.info(f"🔍 Debug: Testing pattern '{pattern}' against query: '{query_lower}'")
                match = re.search(pattern, query_lower)
                self.logger.info(f"🔍 Debug: Pattern match result: {match}")
                if match:
                    self.logger.info(f"🔍 Debug: Pattern matched! Group 1: '{match.group(1)}'")
                if match:
                    captured_value = match.group(1).upper()
                    search_criteria['exact_matches'][field] = captured_value
                    search_criteria['intent'] = f'find_by_{field_lower}'
                    search_criteria['confidence_score'] = 0.9
                    self.logger.info(f"✅ Captured {field}: {captured_value}")
                else:
                    self.logger.info(f"🔍 Debug: Pattern did not match for {field}")
                    # Try alternative pattern for UPRN (might be longer numbers)
                    if field_lower == 'uprn':
                        self.logger.info("🔍 Debug: Trying alternative UPRN pattern...")
                        # Look for UPRN followed by numbers (more flexible)
                        uprn_pattern = r'uprn\s+(\d+)'
                        uprn_match = re.search(uprn_pattern, query_lower)
                        self.logger.info(f"🔍 Debug: Alternative UPRN pattern match: {uprn_match}")
                        if uprn_match:
                            captured_value = uprn_match.group(1)
                            search_criteria['exact_matches']['UPRN'] = captured_value
                            search_criteria['intent'] = 'find_by_uprn'
                            search_criteria['confidence_score'] = 0.9
                            self.logger.info(f"✅ Captured UPRN with alternative pattern: {captured_value}")
                        
                        # Also try a more flexible pattern that might catch variations
                        self.logger.info("🔍 Debug: Trying flexible UPRN pattern...")
                        flexible_pattern = r'uprn.*?(\d+)'
                        flexible_match = re.search(flexible_pattern, query_lower)
                        self.logger.info(f"🔍 Debug: Flexible UPRN pattern match: {flexible_match}")
                        if flexible_match:
                            captured_value = flexible_match.group(1)
                            search_criteria['exact_matches']['UPRN'] = captured_value
                            search_criteria['intent'] = 'find_by_uprn'
                            search_criteria['confidence_score'] = 0.9
                            self.logger.info(f"✅ Captured UPRN with flexible pattern: {captured_value}")
            else:
                self.logger.info(f"🔍 Debug: Field {field_lower} not found in query")
        
        # Check for partial searches (postcode, address)
        # First, look for postcode patterns regardless of whether "postcode" is mentioned
        # Use a simpler pattern that matches UK postcode formats like SE23, SE23 2UN, etc.
        postcode_pattern = r'\b([A-Z]{1,2}\d{1,2}(?:\s*\d[A-Z]{2})?)\b'
        postcode_matches = re.findall(postcode_pattern, query_lower.upper())
        if postcode_matches:
            search_criteria['partial_matches']['POSTCODE'] = postcode_matches
            search_criteria['intent'] = 'location_search'
            search_criteria['confidence_score'] = 0.85
        
        # Then check for address patterns
        address_keywords = ['drive', 'road', 'street', 'avenue', 'lane', 'close']
        for keyword in address_keywords:
            if keyword in query_lower:
                # Extract the street name
                pattern = rf'(\w+)\s+{keyword}'
                match = re.search(pattern, query_lower)
                if match:
                    search_criteria['partial_matches']['ADDRESS'] = [match.group(1).upper()]
                    search_criteria['intent'] = 'address_search'
                    search_criteria['confidence_score'] = 0.8
                    break
        
        # Also check for location-related keywords that might indicate postcode search
        location_keywords = ['area', 'region', 'location', 'postcode', 'postal']
        if any(keyword in query_lower for keyword in location_keywords) and postcode_matches:
            # If location keywords are present and we found postcodes, boost confidence
            search_criteria['confidence_score'] = 0.9
        
        # Check for API status queries
        if 'api' in query_lower and 'status' in query_lower:
            api_services = ['xoserve', 'building', 'elink', 'gbg', 'ecoes']
            for service in api_services:
                if service in query_lower:
                    search_criteria['intent'] = 'api_status_check'
                    search_criteria['business_entities'].append(f'{service}_api')
                    search_criteria['confidence_score'] = 0.9
                    break
        
        # Extract business entities mentioned in query
        for keyword in business_keywords:
            if keyword.lower() in query_lower:
                search_criteria['business_entities'].append(keyword)
        
        # Determine search fields based on intent
        search_criteria['search_fields'] = self._get_search_fields_for_intent(
            search_criteria['intent'], schema_config
        )
        
        return search_criteria
    
    def _get_search_fields_for_intent(self, intent: str, schema_config: Dict[str, Any]) -> List[str]:
        """Get relevant search fields based on detected intent."""
        intent_mapping = {
            'find_by_emsn': ['EMSN', 'ADDRESS', 'POSTCODE', 'UPRN', 'apiStatus'],
            'find_by_mpan': ['MPAN', 'ADDRESS', 'POSTCODE', 'UPRN', 'apiStatus'],
            'find_by_mprn': ['MPRN', 'ADDRESS', 'POSTCODE', 'UPRN', 'apiStatus'],
            'find_by_uprn': ['UPRN', 'ADDRESS', 'EMSN', 'MPAN', 'MPRN', 'apiStatus'],
            'location_search': ['POSTCODE', 'ADDRESS', 'UPRN', 'EMSN', 'MPAN'],
            'address_search': ['ADDRESS', 'POSTCODE', 'UPRN', 'EMSN', 'MPAN'],
            'api_status_check': ['UPRN', 'apiStatus', 'ADDRESS'],
            'general_search': schema_config.get('context_optimization', {}).get('essential_columns', [])
        }
        
        return intent_mapping.get(intent, [])
    
    def _build_mongo_query(self, search_criteria: Dict[str, Any], schema_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Build MongoDB query based on extracted search criteria.
        """
        mongo_query = {}
        
        # Add exact matches
        for field, value in search_criteria['exact_matches'].items():
            if field in ['EMSN', 'MPAN', 'MPRN', 'UPRN']:
                # These fields are arrays, so we search within them
                mongo_query[field] = {'$in': [value]}
                self.logger.info(f"🔍 {field} search: {value} (searching within array)")
            else:
                mongo_query[field] = value
        
        # Add partial matches for postcode
        if 'POSTCODE' in search_criteria['partial_matches']:
            postcodes = search_criteria['partial_matches']['POSTCODE']
            # Use regex for partial postcode matching
            postcode_regex = '|'.join([f'^{postcode}' for postcode in postcodes])
            mongo_query['POSTCODE'] = {'$regex': postcode_regex, '$options': 'i'}
        
        # Add partial matches for address
        if 'ADDRESS' in search_criteria['partial_matches']:
            addresses = search_criteria['partial_matches']['ADDRESS']
            # Use regex for partial address matching
            address_regex = '|'.join([f'.*{address}.*' for address in addresses])
            mongo_query['ADDRESS'] = {'$regex': address_regex, '$options': 'i'}
        
        # If no specific criteria, try a fallback search
        if not mongo_query:
            # Try to find any document with the UPRN mentioned in the query
            uprn_match = re.search(r'(\d+)', user_query)
            if uprn_match:
                uprn_value = uprn_match.group(1)
                mongo_query = {'UPRN': {'$in': [uprn_value]}}
                self.logger.info(f"🔍 Fallback UPRN search: {uprn_value} (searching within array)")
        
        return mongo_query
    
    def _get_projection(self, schema_config: Dict[str, Any]) -> Dict[str, Any]:
        """Get MongoDB projection based on schema configuration."""
        essential_columns = schema_config.get('context_optimization', {}).get('essential_columns', [])
        exclude_columns = schema_config.get('context_optimization', {}).get('exclude_columns', [])
        
        projection = {}
        
        # Include essential columns
        for col in essential_columns:
            projection[col] = 1
        
        # Exclude specified columns
        for col in exclude_columns:
            projection[col] = 0
        
        # Always exclude _id unless specifically requested
        if '_id' not in essential_columns:
            projection['_id'] = 0
        
        # If no essential columns specified, include common fields for debugging
        if not essential_columns:
            projection = {
                'UPRN': 1,
                'ADDRESS': 1,
                'POSTCODE': 1,
                'EMSN': 1,
                'MPAN': 1,
                'MPRN': 1,
                'apiStatus': 1,
                '_id': 0
            }
            self.logger.info("🔍 Using default projection for debugging")
        
        return projection
    
    def _enhance_results_with_context(self, results: List[Dict], search_criteria: Dict[str, Any], 
                                    schema_config: Dict[str, Any]) -> List[Dict]:
        """
        Enhance search results with business context and data dictionary information.
        """
        enhanced_results = []
        
        for result in results:
            enhanced_result = result.copy()
            
            # Add business context
            enhanced_result['_business_context'] = {
                'search_intent': search_criteria['intent'],
                'confidence_score': search_criteria['confidence_score'],
                'business_entities': search_criteria['business_entities']
            }
            
            # Add field descriptions from data dictionary
            data_dictionary = schema_config.get('data_dictionary', {})
            enhanced_result['_field_descriptions'] = {}
            
            for field, value in result.items():
                if field in data_dictionary:
                    field_info = data_dictionary[field]
                    enhanced_result['_field_descriptions'][field] = {
                        'description': field_info.get('description', ''),
                        'data_type': field_info.get('data_type', ''),
                        'business_importance': field_info.get('business_importance', '')
                    }
            
            # Add search relevance score
            enhanced_result['_search_relevance'] = self._calculate_relevance_score(
                result, search_criteria, schema_config
            )
            
            enhanced_results.append(enhanced_result)
        
        # Sort by relevance score
        enhanced_results.sort(key=lambda x: x.get('_search_relevance', 0), reverse=True)
        
        return enhanced_results
    
    def _calculate_relevance_score(self, result: Dict[str, Any], search_criteria: Dict[str, Any], 
                                 schema_config: Dict[str, Any]) -> float:
        """
        Calculate relevance score based on how well the result matches search criteria.
        """
        score = 0.0
        
        # Base score from intent confidence
        score += search_criteria['confidence_score'] * 0.4
        
        # Score for exact matches
        for field, expected_value in search_criteria['exact_matches'].items():
            if field in result:
                if isinstance(result[field], list):
                    if expected_value in result[field]:
                        score += 0.3
                elif str(result[field]) == expected_value:
                    score += 0.3
        
        # Score for partial matches
        for field, expected_values in search_criteria['partial_matches'].items():
            if field in result:
                if isinstance(result[field], list):
                    for expected_value in expected_values:
                        for actual_value in result[field]:
                            if expected_value.lower() in str(actual_value).lower():
                                score += 0.2
                                break
                else:
                    for expected_value in expected_values:
                        if expected_value.lower() in str(result[field]).lower():
                            score += 0.2
                            break
        
        # Score for business entity presence
        business_entities = search_criteria['business_entities']
        for entity in business_entities:
            if any(entity.lower() in str(value).lower() for value in result.values()):
                score += 0.1
        
        return min(score, 1.0)  # Cap at 1.0
    
    def get_collections_list(self) -> List[Dict[str, str]]:
        """Get list of available collections."""
        try:
            collections = []
            for collection_name in self.database.list_collection_names():
                # Get basic info for each collection
                info = self.get_collection_info(collection_name)
                collections.append({
                    'name': collection_name,
                    'display_name': collection_name.title().replace('_', ' '),
                    'document_count': info.get('document_count', 0),
                    'has_embeddings': False  # Basic search doesn't use embeddings
                })
            return collections
        except Exception as e:
            self.logger.error(f"❌ Error getting collections list: {e}")
            return []
    
    def test_connection(self, collection_name: str) -> Dict[str, Any]:
        """Test connection to a specific collection."""
        try:
            collection = self.database[collection_name]
            count = collection.count_documents({})
            
            # Try to get a sample document
            sample_doc = collection.find_one({})
            
            return {
                'success': True,
                'collection_name': collection_name,
                'database_name': self.database_name,
                'document_count': count,
                'has_sample_document': sample_doc is not None,
                'sample_fields': list(sample_doc.keys()) if sample_doc else []
            }
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'collection_name': collection_name
            }
