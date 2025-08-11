import yaml
import os
from typing import Dict, List, Any, Optional
import pandas as pd
import re

class MongoDBSchemaManager:
    """
    Manages MongoDB collection schemas and optimizes context generation
    to prevent parsing issues and optimize token usage.
    Now includes contextual awareness and semantic search capabilities.
    """
    
    def __init__(self, config_path: str = "config/mongodb_schema_config.yaml"):
        """
        Initialize the MongoDB schema manager.
        
        Args:
            config_path: Path to the MongoDB schema configuration file
        """
        self.config_path = config_path
        self.schema_config = self._load_schema_config()
    
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
    
    def get_collection_schema(self, collection_name: str) -> Optional[Dict[str, Any]]:
        """
        Get the schema configuration for a specific collection.
        
        Args:
            collection_name: Name of the MongoDB collection
            
        Returns:
            Schema configuration dictionary or None if not found
        """
        return self.schema_config.get('mongodb_collections', {}).get(collection_name)
    
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
        Enhance user query using business context and semantic understanding.
        
        Args:
            collection_name: Name of the MongoDB collection
            user_query: Original user query
            
        Returns:
            Enhanced query information
        """
        schema = self.get_collection_schema(collection_name)
        if not schema:
            return {'original_query': user_query, 'enhanced_query': user_query}
        
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
            'search_strategy': 'vector_search'
        }
        
        # Detect query intent based on patterns
        if query_enhancement.get('enable_semantic_expansion', False):
            patterns = query_enhancement.get('common_question_patterns', {})
            aliases = query_enhancement.get('business_aliases', {})
            
            # Detect intent patterns
            for intent, keywords in patterns.items():
                if any(keyword.lower() in user_query.lower() for keyword in keywords):
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
        
        # Check for essential columns
        if not schema.get('context_optimization', {}).get('essential_columns'):
            validation_results['warnings'].append('No essential columns defined')
        
        return validation_results 