import yaml
import os
from typing import Dict, List, Any, Optional
import pandas as pd
import re
from datetime import datetime
from pymongo import MongoClient

class MongoDBSchemaManager:
    """
    Manages MongoDB collection schemas and optimizes context generation
    to prevent parsing issues and optimize token usage in RAG applications.
    Now includes enhanced contextual awareness, semantic search capabilities,
    Q&A pattern matching for better intent understanding, and dynamic learning
    capabilities for continuous improvement.
    """
    
    def __init__(self, config_path: str = "config/mongodb_schema_config.yaml", qa_config_path: str = "config/mongodb_qa_collections.yaml"):
        """
        Initialize the MongoDB schema manager.
        
        Args:
            config_path: Path to the MongoDB schema configuration file
            qa_config_path: Path to the MongoDB Q&A collections configuration file
        """
        self.config_path = config_path
        self.qa_config_path = qa_config_path
        self.schema_config = self._load_schema_config()
        self.qa_config = self._load_qa_config()
        
        # Initialize MongoDB connection for Q&A collections
        self.mongo_client = None
        self.qa_db = None
        self._init_mongodb_connection()
    
    def _init_mongodb_connection(self):
        """Initialize MongoDB connection for Q&A collections."""
        try:
            # Get MongoDB connection from environment or config
            mongo_uri = os.getenv('MONGODB_URI', 'mongodb://localhost:27017/')
            self.mongo_client = MongoClient(mongo_uri)
            self.qa_db = self.mongo_client['perse-data-network']
            
            # Test connection
            self.mongo_client.admin.command('ping')
            print("✅ Connected to MongoDB Q&A collections")
            
        except Exception as e:
            print(f"⚠️ MongoDB connection failed: {e}")
            print("⚠️ Dynamic learning features will be disabled")
            self.mongo_client = None
            self.qa_db = None
    
    def is_mongodb_available(self) -> bool:
        """Check if MongoDB connection is available."""
        return self.qa_db is not None and self.mongo_client is not None
    
    def _load_schema_config(self) -> Dict[str, Any]:
        """Load the MongoDB schema configuration file."""
        try:
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
            return config
        except FileNotFoundError:
            print(f"Warning: MongoDB schema config not found at {self.config_path}")
            return {}
        except Exception as e:
            print(f"Error loading MongoDB schema config: {e}")
            return {}
    
    def _load_qa_config(self) -> Dict[str, Any]:
        """Load the MongoDB Q&A collections configuration file."""
        try:
            with open(self.qa_config_path, 'r') as f:
                config = yaml.safe_load(f)
            return config
        except FileNotFoundError:
            print(f"Warning: MongoDB Q&A config not found at {self.qa_config_path}")
            return {}
        except Exception as e:
            print(f"Error loading MongoDB Q&A config: {e}")
            return {}
    
    def get_collection_schema(self, collection_name: str) -> Optional[Dict[str, Any]]:
        """
        Get the schema configuration for a specific collection.
        
        Args:
            collection_name: Name of the MongoDB collection
            
        Returns:
            Schema configuration dictionary or None if not found
        """
        return self.schema_config.get('mongodb_collections', {}).get(collection_name)
    
    def get_qa_patterns(self, collection_name: str) -> List[Dict[str, Any]]:
        """
        Get Q&A patterns for a specific collection.
        
        Args:
            collection_name: Name of the MongoDB collection
            
        Returns:
            List of Q&A patterns
        """
        schema = self.get_collection_schema(collection_name)
        if schema and 'question_answer_patterns' in schema:
            return schema.get('question_answer_patterns', {}).get('core_patterns', [])
        return []
    
    def get_hybrid_qa_patterns(self, collection_name: str) -> List[Dict[str, Any]]:
        """
        Get Q&A patterns from both YAML (core) and MongoDB (extended).
        This creates a hybrid system for optimal performance.
        
        Args:
            collection_name: Name of the MongoDB collection
            
        Returns:
            List of Q&A patterns from both sources
        """
        patterns = []
        
        # 1. Get core patterns from YAML (fast, reliable)
        core_patterns = self.get_qa_patterns(collection_name)
        patterns.extend(core_patterns)
        
        # 2. Get extended patterns from MongoDB (dynamic, learnable)
        if self.qa_db is not None:
            try:
                extended_patterns = list(
                    self.qa_db.extended_qa_patterns.find({
                        "collection_name": collection_name,
                        "is_active": True
                    }).sort("confidence_score", -1)
                )
                patterns.extend(extended_patterns)
                print(f"✅ Loaded {len(extended_patterns)} extended patterns from MongoDB")
            except Exception as e:
                print(f"⚠️ Failed to load extended patterns: {e}")
        
        return patterns
    
    def get_intent_categories(self, collection_name: str) -> Dict[str, Any]:
        """
        Get intent categories for a specific collection.
        
        Args:
            collection_name: Name of the MongoDB collection
            
        Returns:
            Dictionary of intent categories
        """
        schema = self.get_collection_schema(collection_name)
        if schema and 'question_answer_patterns' in schema:
            return schema.get('question_answer_patterns', {}).get('intent_categories', {})
        return {}
    
    def match_qa_pattern(self, collection_name: str, user_query: str) -> Optional[Dict[str, Any]]:
        """
        Match a user query against Q&A patterns to find the best match.
        Now uses hybrid patterns from both YAML and MongoDB.
        
        Args:
            collection_name: Name of the MongoDB collection
            user_query: The user's query
            
        Returns:
            Best matching Q&A pattern or None if no match found
        """
        # Use hybrid patterns for better coverage
        patterns = self.get_hybrid_qa_patterns(collection_name)
        if not patterns:
            return None
        
        best_match = None
        best_score = 0.0
        
        for pattern in patterns:
            score = self._calculate_pattern_match_score(pattern, user_query, collection_name)
            if score > best_score and score >= 0.7:  # Minimum confidence threshold
                best_score = score
                best_match = pattern.copy()
                best_match['match_score'] = score
        
        return best_match
    
    def _calculate_pattern_match_score(self, pattern: Dict[str, Any], user_query: str, collection_name: str) -> float:
        """
        Calculate how well a pattern matches a user query.
        
        Args:
            pattern: The Q&A pattern to match against
            user_query: The user's query
            collection_name: Name of the MongoDB collection
            
        Returns:
            Match score between 0.0 and 1.0
        """
        query_lower = user_query.lower()
        pattern_lower = pattern.get('question_pattern', '').lower()
        
        # Check exact pattern match
        if pattern_lower in query_lower or query_lower in pattern_lower:
            return 0.95
        
        # Check sample queries
        sample_queries = pattern.get('sample_queries', [])
        for sample in sample_queries:
            sample_lower = sample.lower()
            if sample_lower in query_lower or query_lower in sample_lower:
                return 0.9
        
        # Check business entities
        business_entities = pattern.get('business_entities', [])
        entity_matches = 0
        for entity in business_entities:
            if entity.lower() in query_lower:
                entity_matches += 1
        
        if business_entities:
            entity_score = entity_matches / len(business_entities)
        else:
            entity_score = 0.0
        
        # Check intent keywords
        intent_categories = self.get_intent_categories(collection_name)
        intent_keywords = []
        for category in intent_categories.values():
            intent_keywords.extend(category.get('keywords', []))
        
        keyword_matches = 0
        for keyword in intent_keywords:
            if keyword.lower() in query_lower:
                keyword_matches += 1
        
        if intent_keywords:
            keyword_score = keyword_matches / len(intent_keywords)
        else:
            keyword_score = 0.0
        
        # Calculate final score
        final_score = (entity_score * 0.4) + (keyword_score * 0.6)
        
        return min(final_score, 0.85)  # Cap at 0.85 for non-exact matches
    
    def get_business_context(self, collection_name: str) -> Dict[str, Any]:
        """
        Get the business context for a collection.
        
        Args:
            collection_name: Name of the MongoDB collection
            
        Returns:
            Business context dictionary
        """
        schema = self.get_collection_schema(collection_name)
        if schema and 'business_context' in schema:
            return schema['business_context']
        return {}
    
    def get_data_dictionary(self, collection_name: str) -> Dict[str, Any]:
        """
        Get the data dictionary for a collection.
        
        Args:
            collection_name: Name of the MongoDB collection
            
        Returns:
            Data dictionary dictionary
        """
        schema = self.get_collection_schema(collection_name)
        if schema and 'data_dictionary' in schema:
            return schema['data_dictionary']
        return {}
    
    def get_essential_columns(self, collection_name: str) -> List[str]:
        """
        Get the essential columns for a collection that should be included in context.
        
        Args:
            collection_name: Name of the MongoDB collection
            
        Returns:
            List of essential column names
        """
        schema = self.get_collection_schema(collection_name)
        if schema and 'context_optimization' in schema:
            return schema['context_optimization'].get('essential_columns', [])
        return []
    
    def get_exclude_columns(self, collection_name: str) -> List[str]:
        """
        Get the columns that should be excluded from context.
        
        Args:
            collection_name: Name of the MongoDB collection
            
        Returns:
            List of excluded column names
        """
        schema = self.get_collection_schema(collection_name)
        if schema and 'context_optimization' in schema:
            return schema['context_optimization'].get('exclude_columns', [])
        return []
    
    def get_max_context_rows(self, collection_name: str) -> int:
        """
        Get the maximum number of rows to include in context for a collection.
        
        Args:
            collection_name: Name of the MongoDB collection
            
        Returns:
            Maximum number of context rows
        """
        schema = self.get_collection_schema(collection_name)
        if schema and 'context_optimization' in schema:
            return schema['context_optimization'].get('max_rows', 10)
        return self.schema_config.get('global_settings', {}).get('default_context_rows', 10)
    
    def get_max_field_length(self, collection_name: str) -> int:
        """
        Get the maximum field length for a collection.
        
        Args:
            collection_name: Name of the MongoDB collection
            
        Returns:
            Maximum field length
        """
        schema = self.get_collection_schema(collection_name)
        if schema and 'data_processing' in schema:
            return schema['data_processing'].get('max_field_length', 50)
        return self.schema_config.get('global_settings', {}).get('default_max_field_length', 50)
    
    def should_include_column_in_context(self, collection_name: str, column_name: str) -> bool:
        """
        Check if a column should be included in context for a collection.
        
        Args:
            collection_name: Name of the MongoDB collection
            column_name: Name of the column
            
        Returns:
            True if column should be included, False otherwise
        """
        data_dict = self.get_data_dictionary(collection_name)
        if data_dict and column_name in data_dict:
            return data_dict[column_name].get('include_in_context', True)
        return True
    
    def get_column_max_length(self, collection_name: str, column_name: str) -> int:
        """
        Get the maximum length for a specific column in a collection.
        
        Args:
            collection_name: Name of the MongoDB collection
            column_name: Name of the column
            
        Returns:
            Maximum length for the column
        """
        data_dict = self.get_data_dictionary(collection_name)
        if data_dict and column_name in data_dict:
            return data_dict[column_name].get('max_length', 50)
        return self.get_max_field_length(collection_name)
    
    def get_search_relevance(self, collection_name: str, column_name: str) -> str:
        """
        Get the search relevance for a specific column.
        
        Args:
            collection_name: Name of the MongoDB collection
            column_name: Name of the column
            
        Returns:
            Search relevance level: 'high', 'medium', 'low', or 'none'
        """
        data_dict = self.get_data_dictionary(collection_name)
        if data_dict and column_name in data_dict:
            return data_dict[column_name].get('search_relevance', 'medium')
        return 'medium'
    
    def get_semantic_keywords(self, collection_name: str, column_name: str) -> List[str]:
        """
        Get semantic keywords for a specific column.
        
        Args:
            collection_name: Name of the MongoDB collection
            column_name: Name of the column
            
        Returns:
            List of semantic keywords
        """
        data_dict = self.get_data_dictionary(collection_name)
        if data_dict and column_name in data_dict:
            return data_dict[column_name].get('semantic_keywords', [])
        return []
    
    def enhance_user_query(self, collection_name: str, user_query: str) -> Dict[str, Any]:
        """
        Enhance user query using business context, semantic understanding, and Q&A patterns.
        
        Args:
            collection_name: Name of the MongoDB collection
            user_query: Original user query
            
        Returns:
            Enhanced query information
        """
        # Initialize default enhanced info to ensure all keys are always present
        default_enhanced_info = {
            'original_query': user_query,
            'enhanced_query': user_query,
            'business_domain': 'Unknown',
            'purpose': 'Data analysis',
            'detected_intent': [],
            'semantic_expansions': [],
            'relevant_columns': [],
            'search_strategy': 'vector_search',
            'qa_pattern_match': None,
            'confidence_score': 0.0
        }
        
        try:
            schema = self.get_collection_schema(collection_name)
            if not schema:
                return default_enhanced_info
            
            business_context = self.get_business_context(collection_name)
            query_enhancement = schema.get('query_enhancement', {})
            
            enhanced_info = {
                'original_query': user_query,
                'enhanced_query': user_query,
                'business_domain': business_context.get('domain', 'Unknown'),
                'purpose': business_context.get('purpose', 'Data analysis'),
                'detected_intent': [],
                'semantic_expansions': [],
                'relevant_columns': [],
                'search_strategy': 'vector_search',
                'qa_pattern_match': None,
                'confidence_score': 0.0
            }
        
        # 🚀 ENHANCED: Q&A Pattern Matching
        if query_enhancement.get('enable_qa_pattern_matching', False):
            qa_pattern = self.match_qa_pattern(collection_name, user_query)
            if qa_pattern:
                enhanced_info['qa_pattern_match'] = qa_pattern
                enhanced_info['confidence_score'] = qa_pattern.get('match_score', 0.0)
                enhanced_info['detected_intent'].append(qa_pattern.get('answer_intent', ''))
                
                # Use expected columns from the pattern
                expected_columns = qa_pattern.get('expected_columns', [])
                if expected_columns:
                    enhanced_info['relevant_columns'] = [
                        {
                            'name': col,
                            'relevance': 'high',
                            'business_meaning': f'Expected column for {qa_pattern.get("answer_intent", "")} intent',
                            'keywords': []
                        }
                        for col in expected_columns
                    ]
                
                # Use search strategy from the pattern
                search_strategy = qa_pattern.get('search_strategy', '')
                if search_strategy:
                    enhanced_info['search_strategy'] = search_strategy
        
        # 🚀 ENHANCED: Intent Detection
        if query_enhancement.get('enable_intent_detection', False):
            intent_categories = self.get_intent_categories(collection_name)
            for intent_name, intent_info in intent_categories.items():
                keywords = intent_info.get('keywords', [])
                if any(keyword.lower() in user_query.lower() for keyword in keywords):
                    if intent_name not in enhanced_info['detected_intent']:
                        enhanced_info['detected_intent'].append(intent_name)
        
        # Semantic expansion (existing logic)
        if query_enhancement.get('enable_semantic_expansion', False):
            patterns = query_enhancement.get('common_question_patterns', {})
            aliases = query_enhancement.get('business_aliases', {})
            
            # Detect intent patterns
            for intent, keywords in patterns.items():
                if any(keyword.lower() in user_query.lower() for keyword in keywords):
                    if intent not in enhanced_info['detected_intent']:
                        enhanced_info['detected_intent'].append(intent)
            
            # Apply semantic expansions
            expanded_query = user_query
            for business_term, synonyms in aliases.items():
                if business_term.lower() in user_query.lower():
                    enhanced_info['semantic_expansions'].extend(synonyms)
                    # Add synonyms to query for better search
                    for synonym in synonyms:
                        if synonym.lower() not in expanded_query.lower():
                            expanded_query += f" {synonym}"
            
            enhanced_info['enhanced_query'] = expanded_query
        
        # Identify relevant columns based on query content
        data_dict = self.get_data_dictionary(collection_name)
        for column_name, column_info in data_dict.items():
            if column_info.get('include_in_context', False):
                # Check if column is relevant to the query
                relevance = column_info.get('search_relevance', 'medium')
                keywords = column_info.get('semantic_keywords', [])
                
                # Check if query contains relevant keywords
                query_lower = user_query.lower()
                if any(keyword.lower() in query_lower for keyword in keywords):
                    # Check if column is already in relevant_columns
                    if not any(col['name'] == column_name for col in enhanced_info['relevant_columns']):
                        enhanced_info['relevant_columns'].append({
                            'name': column_name,
                            'relevance': relevance,
                            'business_meaning': column_info.get('business_meaning', ''),
                            'keywords': keywords
                        })
        
            # Determine search strategy
            if enhanced_info['detected_intent']:
                enhanced_info['search_strategy'] = 'semantic_search'
            
            return enhanced_info
            
        except Exception as e:
            # Log the error and return default info
            print(f"Error in enhance_user_query for {collection_name}: {str(e)}")
            return default_enhanced_info
    
    def get_search_optimization_settings(self, collection_name: str) -> Dict[str, Any]:
        """
        Get search optimization settings for a collection.
        
        Args:
            collection_name: Name of the MongoDB collection
            
        Returns:
            Search optimization settings
        """
        schema = self.get_collection_schema(collection_name)
        if schema and 'search_optimization' in schema:
            return schema['search_optimization']
        return {}
    
    def get_business_keywords(self, collection_name: str) -> List[str]:
        """
        Get business keywords for a collection.
        
        Args:
            collection_name: Name of the MongoDB collection
            
        Returns:
            List of business keywords
        """
        search_opt = self.get_search_optimization_settings(collection_name)
        return search_opt.get('business_keywords', [])
    
    def get_semantic_boost_fields(self, collection_name: str) -> List[str]:
        """
        Get fields that should be boosted in semantic search.
        
        Args:
            collection_name: Name of the MongoDB collection
            
        Returns:
            List of fields to boost
        """
        search_opt = self.get_search_optimization_settings(collection_name)
        return search_opt.get('semantic_boost_fields', [])
    
    def optimize_dataframe_for_context(self, df: pd.DataFrame, collection_name: str) -> pd.DataFrame:
        """
        Optimize a DataFrame for context generation based on collection schema.
        
        Args:
            df: Input DataFrame
            collection_name: Name of the MongoDB collection
            
        Returns:
            Optimized DataFrame for context
        """
        if df.empty:
            return df
        
        schema = self.get_collection_schema(collection_name)
        if not schema:
            # Fallback to basic optimization
            return self._basic_optimization(df)
        
        # Get optimization settings
        max_rows = self.get_max_context_rows(collection_name)
        essential_columns = self.get_essential_columns(collection_name)
        exclude_columns = self.get_exclude_columns(collection_name)
        max_field_length = self.get_max_field_length(collection_name)
        
        # Sample rows if needed
        if len(df) > max_rows:
            df_optimized = df.sample(n=max_rows, random_state=42)
        else:
            df_optimized = df.copy()
        
        # Select columns based on schema
        if essential_columns:
            # Use essential columns if they exist
            available_essential = [col for col in essential_columns if col in df_optimized.columns]
            if available_essential:
                df_optimized = df_optimized[available_essential]
            else:
                # Fallback to first few columns
                df_optimized = df_optimized.iloc[:, :2]
        else:
            # Remove excluded columns
            available_columns = [col for col in df_optimized.columns if col not in exclude_columns]
            if available_columns:
                df_optimized = df_optimized[available_columns]
        
        # Truncate long fields
        if schema.get('data_processing', {}).get('truncate_long_fields', True):
            df_optimized = self._truncate_long_fields(df_optimized, collection_name, max_field_length)
        
        return df_optimized
    
    def _basic_optimization(self, df: pd.DataFrame) -> pd.DataFrame:
        """Basic DataFrame optimization when no schema is available."""
        if df.empty:
            return df
        
        # Sample to 10 rows maximum
        if len(df) > 10:
            df_optimized = df.sample(n=10, random_state=42)
        else:
            df_optimized = df.copy()
        
        # Keep only first 2 columns to minimize tokens
        if len(df_optimized.columns) > 2:
            df_optimized = df_optimized.iloc[:, :2]
        
        return df_optimized
    
    def _truncate_long_fields(self, df: pd.DataFrame, collection_name: str, max_length: int) -> pd.DataFrame:
        """Truncate long fields in the DataFrame based on schema configuration."""
        df_truncated = df.copy()
        
        for column in df_truncated.columns:
            if df_truncated[column].dtype == 'object':
                # Check if this column should be truncated
                column_max_length = self.get_column_max_length(collection_name, column)
                if column_max_length < max_length:
                    max_length = column_max_length
                
                # Truncate string fields
                df_truncated[column] = df_truncated[column].astype(str).str[:max_length] + '...'
        
        return df_truncated
    
    def get_collection_display_name(self, collection_name: str) -> str:
        """Get the display name for a collection."""
        schema = self.get_collection_schema(collection_name)
        if schema:
            return schema.get('display_name', collection_name)
        return collection_name
    
    def get_collection_description(self, collection_name: str) -> str:
        """Get the description for a collection."""
        schema = self.get_collection_schema(collection_name)
        if schema:
            return schema.get('description', 'No description available')
        return 'No description available'
    
    def get_business_purpose(self, collection_name: str) -> str:
        """Get the business purpose for a collection."""
        business_context = self.get_business_context(collection_name)
        return business_context.get('purpose', 'Data analysis and reporting')
    
    def get_common_queries(self, collection_name: str) -> List[str]:
        """Get common queries for a collection."""
        business_context = self.get_business_context(collection_name)
        return business_context.get('common_queries', [])
    
    def validate_collection_schema(self, collection_name: str) -> Dict[str, Any]:
        """
        Validate the schema configuration for a collection.
        
        Args:
            collection_name: Name of the MongoDB collection
            
        Returns:
            Validation results dictionary
        """
        schema = self.get_collection_schema(collection_name)
        if not schema:
            return {
                'valid': False,
                'error': f'No schema found for collection: {collection_name}'
            }
        
        validation_results = {
            'valid': True,
            'warnings': [],
            'collection_name': collection_name,
            'data_dictionary_fields': len(schema.get('data_dictionary', {})),
            'business_context': 'business_context' in schema,
            'query_enhancement': 'query_enhancement' in schema,
            'qa_patterns': 'question_answer_patterns' in schema,
            'essential_columns': len(schema.get('context_optimization', {}).get('essential_columns', [])),
            'exclude_columns': len(schema.get('context_optimization', {}).get('exclude_columns', []))
        }
        
        # Check for required sections
        required_sections = ['data_dictionary', 'context_optimization', 'search_optimization']
        for section in required_sections:
            if section not in schema:
                validation_results['warnings'].append(f'Missing section: {section}')
        
        # Check for business context
        if 'business_context' not in schema:
            validation_results['warnings'].append('Missing business context section')
        
        # Check for Q&A patterns
        if 'question_answer_patterns' not in schema:
            validation_results['warnings'].append('Missing question/answer patterns section')
        
        # Check for essential columns
        if not schema.get('context_optimization', {}).get('essential_columns'):
            validation_results['warnings'].append('No essential columns defined')
        
        return validation_results
    
    def learn_from_query(self, collection_name: str, user_query: str, 
                         detected_intent: str, user_satisfaction: int = None):
        """
        Learn from user query for pattern improvement.
        This is called automatically during query processing.
        
        Args:
            collection_name: Name of the MongoDB collection
            user_query: The user's original query
            detected_intent: The intent that was detected
            user_satisfaction: User satisfaction rating (1-5) if available
        """
        if not self.is_mongodb_available():
            return
        
        try:
            # Extract query features
            query_features = self._extract_query_features(user_query)
            
            # Record learning data
            learning_data = {
                "collection_name": collection_name,
                "user_query": user_query,
                "detected_intent": detected_intent,
                "query_features": query_features,
                "created_at": datetime.now(),
                "session_id": self._get_session_id()
            }
            
            if user_satisfaction is not None:
                learning_data["user_satisfaction"] = user_satisfaction
            
            # Insert into learning collection
            self.qa_db.intent_learning_data.insert_one(learning_data)
            
            # Update pattern usage statistics
            self._update_pattern_usage(collection_name, detected_intent)
            
            print(f"✅ Learning data recorded for {collection_name}")
            
        except Exception as e:
            print(f"⚠️ Failed to record learning data: {e}")
    
    def _extract_query_features(self, user_query: str) -> Dict[str, Any]:
        """
        Extract features from user query for learning.
        
        Args:
            user_query: The user's query
            
        Returns:
            Dictionary of extracted features
        """
        query_lower = user_query.lower()
        
        # Extract business keywords from the schema
        business_keywords = []
        technical_keywords = []
        
        # Check for common business terms
        business_terms = ['mpan', 'meter', 'supplier', 'error', 'validation', 'postcode', 'location']
        for term in business_terms:
            if term in query_lower:
                business_keywords.append(term)
        
        # Check for technical terms
        technical_terms = ['analyze', 'find', 'show', 'count', 'pattern', 'trend', 'frequency']
        for term in technical_terms:
            if term in query_lower:
                technical_keywords.append(term)
        
        return {
            "word_count": len(user_query.split()),
            "has_numbers": any(char.isdigit() for char in user_query),
            "has_dates": any(word in query_lower for word in ['today', 'yesterday', 'date', 'time']),
            "business_keywords": business_keywords,
            "technical_keywords": technical_keywords
        }
    
    def _get_session_id(self) -> str:
        """Generate a simple session ID for grouping related queries."""
        return f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    def _update_pattern_usage(self, collection_name: str, intent: str):
        """
        Update usage statistics for patterns.
        
        Args:
            collection_name: Name of the MongoDB collection
            intent: The detected intent
        """
        if self.qa_db is None:
            return
            
        try:
            self.qa_db.extended_qa_patterns.update_many(
                {
                    "collection_name": collection_name,
                    "answer_intent": intent
                },
                {
                    "$inc": {"usage_count": 1},
                    "$set": {"updated_at": datetime.now()}
                }
            )
        except Exception as e:
            print(f"⚠️ Failed to update pattern usage: {e}")
    
    def evolve_patterns(self, collection_name: str):
        """
        Evolve patterns based on learning data and feedback.
        This can be called periodically or after certain thresholds.
        
        Args:
            collection_name: Name of the MongoDB collection
        """
        if self.qa_db is None:
            return
        
        try:
            # Analyze learning data
            learning_data = list(
                self.qa_db.intent_learning_data.find({
                    "collection_name": collection_name
                })
            )
            
            if len(learning_data) < 10:  # Need minimum data
                print(f"⚠️ Insufficient learning data for {collection_name}: {len(learning_data)} records")
                return
            
            # Calculate pattern performance
            pattern_performance = self._calculate_pattern_performance(learning_data)
            
            # Update confidence scores
            updated_count = 0
            for intent, performance in pattern_performance.items():
                if performance['count'] >= 5:  # Minimum usage threshold
                    new_confidence = performance['avg_satisfaction'] / 5.0
                    
                    # Update pattern confidence
                    result = self.qa_db.extended_qa_patterns.update_many(
                        {
                            "collection_name": collection_name,
                            "answer_intent": intent
                        },
                        {
                            "$set": {
                                "confidence_score": new_confidence,
                                "success_rate": performance['success_rate'],
                                "updated_at": datetime.now()
                            }
                        }
                    )
                    
                    if result.modified_count > 0:
                        updated_count += result.modified_count
            
            print(f"✅ Patterns evolved for {collection_name}: {updated_count} patterns updated")
            
        except Exception as e:
            print(f"⚠️ Failed to evolve patterns: {e}")
    
    def _calculate_pattern_performance(self, learning_data: List[Dict]) -> Dict[str, Any]:
        """
        Calculate performance metrics for each intent.
        
        Args:
            learning_data: List of learning data records
            
        Returns:
            Dictionary of performance metrics by intent
        """
        performance = {}
        
        for data in learning_data:
            intent = data.get('detected_intent')
            if not intent:
                continue
                
            if intent not in performance:
                performance[intent] = {
                    'count': 0,
                    'satisfaction_scores': [],
                    'success_count': 0
                }
            
            performance[intent]['count'] += 1
            
            if 'user_satisfaction' in data:
                performance[intent]['satisfaction_scores'].append(data['user_satisfaction'])
                
                # Consider satisfaction >= 4 as successful
                if data['user_satisfaction'] >= 4:
                    performance[intent]['success_count'] += 1
        
        # Calculate averages
        for intent, metrics in performance.items():
            if metrics['satisfaction_scores']:
                metrics['avg_satisfaction'] = sum(metrics['satisfaction_scores']) / len(metrics['satisfaction_scores'])
                metrics['success_rate'] = metrics['success_count'] / metrics['count']
            else:
                metrics['avg_satisfaction'] = 0
                metrics['success_rate'] = 0
        
        return performance
    
    def add_user_feedback(self, pattern_id: str, feedback_data: Dict[str, Any]):
        """
        Add user feedback for pattern improvement.
        
        Args:
            pattern_id: ID of the pattern being rated
            feedback_data: Dictionary containing feedback information
        """
        if self.qa_db is None:
            print("⚠️ MongoDB not connected, cannot add feedback")
            return
        
        try:
            # Ensure required fields
            feedback_data['created_at'] = datetime.now()
            
            # Insert feedback
            result = self.qa_db.qa_pattern_feedback.insert_one(feedback_data)
            print(f"✅ Feedback added for pattern {pattern_id}")
            return result.inserted_id
            
        except Exception as e:
            print(f"⚠️ Error adding feedback: {e}")
            return None
    
    def get_pattern_analytics(self, collection_name: str) -> Dict[str, Any]:
        """
        Get analytics and performance metrics for patterns.
        
        Args:
            collection_name: Name of the MongoDB collection
            
        Returns:
            Dictionary containing analytics data
        """
        if self.qa_db is None:
            return {}
        
        try:
            analytics = {
                'total_patterns': 0,
                'active_patterns': 0,
                'total_usage': 0,
                'avg_confidence': 0,
                'avg_success_rate': 0,
                'intent_distribution': {},
                'recent_feedback': []
            }
            
            # Get pattern statistics
            patterns = list(self.qa_db.extended_qa_patterns.find({"collection_name": collection_name}))
            analytics['total_patterns'] = len(patterns)
            analytics['active_patterns'] = len([p for p in patterns if p.get('is_active', False)])
            
            if patterns:
                analytics['total_usage'] = sum(p.get('usage_count', 0) for p in patterns)
                analytics['avg_confidence'] = sum(p.get('confidence_score', 0) for p in patterns) / len(patterns)
                analytics['avg_success_rate'] = sum(p.get('success_rate', 0) for p in patterns) / len(patterns)
                
                # Intent distribution
                for pattern in patterns:
                    intent = pattern.get('answer_intent', 'unknown')
                    if intent not in analytics['intent_distribution']:
                        analytics['intent_distribution'][intent] = 0
                    analytics['intent_distribution'][intent] += 1
            
            # Get recent feedback
            recent_feedback = list(
                self.qa_db.qa_pattern_feedback.find().sort("created_at", -1).limit(5)
            )
            analytics['recent_feedback'] = recent_feedback
            
            return analytics
            
        except Exception as e:
            print(f"⚠️ Failed to get pattern analytics: {e}")
            return {} 