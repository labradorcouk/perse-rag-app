#!/usr/bin/env python3
"""
Test script for Enhanced MongoDB Schema Manager
Demonstrates contextual awareness, semantic search, and query understanding capabilities.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mongodb_schema_manager import MongoDBSchemaManager
import pandas as pd

def test_enhanced_schema_manager():
    """Test the enhanced MongoDB Schema Manager functionality."""
    
    print("Testing Enhanced MongoDB Schema Manager")
    print("=" * 60)
    
    # Initialize schema manager
    schema_manager = MongoDBSchemaManager()
    
    # Test collection schema retrieval
    collection_name = "ecoesTechDetailsWithEmbedding"
    
    print(f"\nCollection: {collection_name}")
    print("-" * 40)
    
    # Get enhanced schema information
    schema = schema_manager.get_collection_schema(collection_name)
    if schema:
        print(f"Schema found with {len(schema.get('data_dictionary', {}))} fields")
        print(f"Display name: {schema_manager.get_collection_display_name(collection_name)}")
        print(f"Description: {schema_manager.get_collection_description(collection_name)}")
        print(f"Business purpose: {schema_manager.get_business_purpose(collection_name)}")
        
        # Get business context
        business_context = schema_manager.get_business_context(collection_name)
        if business_context:
            print(f"Domain: {business_context.get('domain', 'Unknown')}")
            print(f"Key entities: {business_context.get('key_entities', [])}")
            print(f"Common queries: {len(business_context.get('common_queries', []))} examples")
        
        # Get optimization settings
        essential_cols = schema_manager.get_essential_columns(collection_name)
        exclude_cols = schema_manager.get_exclude_columns(collection_name)
        max_rows = schema_manager.get_max_context_rows(collection_name)
        max_field_length = schema_manager.get_max_field_length(collection_name)
        
        print(f"\nOptimization Settings:")
        print(f"   Essential columns: {essential_cols}")
        print(f"   Exclude columns: {exclude_cols}")
        print(f"   Max context rows: {max_rows}")
        print(f"   Max field length: {max_field_length}")
        
        # Validate schema
        validation = schema_manager.validate_collection_schema(collection_name)
        print(f"\nSchema validation: {'Valid' if validation['valid'] else 'Invalid'}")
        if validation['warnings']:
            print(f"Warnings: {validation['warnings']}")
            
    else:
        print(f"No schema found for collection: {collection_name}")
        return
    
    # Test query enhancement capabilities
    print(f"\nTesting Query Enhancement and Contextual Awareness")
    print("-" * 50)
    
    test_queries = [
        "What are the top 5 most common errors in mpan type records?",
        "Find MPAN records with validation issues",
        "Show me meter problems in the system",
        "Analyze error patterns in the data",
        "Find records by postcode location",
        "What are the most frequent technical issues?"
    ]
    
    for query in test_queries:
        print(f"\nQuery: {query}")
        enhanced_info = schema_manager.enhance_user_query(collection_name, query)
        
        print(f"   Business domain: {enhanced_info['business_domain']}")
        print(f"   Purpose: {enhanced_info['purpose']}")
        print(f"   Detected intent: {enhanced_info['detected_intent']}")
        print(f"   Semantic expansions: {enhanced_info['semantic_expansions']}")
        print(f"   Search strategy: {enhanced_info['search_strategy']}")
        
        if enhanced_info['relevant_columns']:
            print(f"   Relevant columns:")
            for col in enhanced_info['relevant_columns']:
                print(f"     - {col['name']}: {col['relevance']} relevance")
                print(f"       Business meaning: {col['business_meaning']}")
                print(f"       Keywords: {col['keywords']}")
    
    # Test semantic search capabilities
    print(f"\nTesting Semantic Search Capabilities")
    print("-" * 40)
    
    # Get search optimization settings
    search_settings = schema_manager.get_search_optimization_settings(collection_name)
    business_keywords = schema_manager.get_business_keywords(collection_name)
    semantic_boost_fields = schema_manager.get_semantic_boost_fields(collection_name)
    
    print(f"Search optimization enabled: {search_settings.get('vector_search_enabled', False)}")
    print(f"Business keywords: {business_keywords}")
    print(f"Semantic boost fields: {semantic_boost_fields}")
    
    # Test column search relevance
    print(f"\nTesting Column Search Relevance")
    print("-" * 35)
    
    test_columns = ['_id_oid', 'type', 'value', 'Results', 'embedding', 'createdAt_date']
    for col in test_columns:
        relevance = schema_manager.get_search_relevance(collection_name, col)
        keywords = schema_manager.get_semantic_keywords(collection_name, col)
        print(f"   {col}: {relevance} relevance")
        if keywords:
            print(f"     Keywords: {keywords}")
    
    # Test DataFrame optimization
    print(f"\nTesting Enhanced DataFrame Optimization")
    print("-" * 40)
    
    # Create a sample DataFrame similar to MongoDB data
    sample_data = {
        '_id_oid': ['123', '456', '789'],
        'type': ['MPAN', 'MPAN', 'MPAN'],
        'value': ['1580001422451', '1580001422452', '1580001422453'],
        'Results': [
            'Very long complex nested structure with lots of data that would cause token issues...',
            'Another very long complex nested structure with lots of data that would cause token issues...',
            'Yet another very long complex nested structure with lots of data that would cause token issues...'
        ],
        'embedding': [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]],
        'createdAt_date': ['2025-01-01', '2025-01-02', '2025-01-03'],
        'salt': ['salt1', 'salt2', 'salt3']
    }
    
    df = pd.DataFrame(sample_data)
    print(f"Original DataFrame: {df.shape[0]} rows x {df.shape[1]} columns")
    print(f"   Columns: {list(df.columns)}")
    
    # Optimize DataFrame using enhanced schema
    df_optimized = schema_manager.optimize_dataframe_for_context(df, collection_name)
    print(f"\nOptimized DataFrame: {df_optimized.shape[0]} rows x {df_optimized.shape[1]} columns")
    print(f"   Columns: {list(df_optimized.columns)}")
    
    # Show sample of optimized data
    print(f"\nSample optimized data:")
    print(df_optimized.head(2).to_string())
    
    print(f"\nEnhanced Schema Manager Test Complete!")
    print(f"\nKey Benefits of Enhanced Schema:")
    print(f"   1. Contextual awareness - understands business domain and purpose")
    print(f"   2. Semantic search - automatically expands queries with business terms")
    print(f"   3. Intent detection - recognizes what users are trying to accomplish")
    print(f"   4. Column relevance - knows which fields are most important for queries")
    print(f"   5. Business aliases - understands synonyms and alternative terms")
    print(f"   6. Query enhancement - automatically improves vague or unclear questions")

if __name__ == "__main__":
    test_enhanced_schema_manager() 