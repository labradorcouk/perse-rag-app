#!/usr/bin/env python3
"""
Test script for MongoDB Schema Manager
Demonstrates how the schema configuration prevents parsing issues and optimizes context.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mongodb_schema_manager import MongoDBSchemaManager
import pandas as pd

def test_schema_manager():
    """Test the MongoDB Schema Manager functionality."""
    
    print("Testing MongoDB Schema Manager")
    print("=" * 50)
    
    # Initialize schema manager
    schema_manager = MongoDBSchemaManager()
    
    # Test collection schema retrieval
    collection_name = "ecoesTechDetailsWithEmbedding"
    
    print(f"\nCollection: {collection_name}")
    print("-" * 30)
    
    # Get schema information
    schema = schema_manager.get_collection_schema(collection_name)
    if schema:
        print(f"Schema found with {len(schema.get('schema', {}))} fields")
        print(f"Display name: {schema_manager.get_collection_display_name(collection_name)}")
        print(f"Description: {schema_manager.get_collection_description(collection_name)}")
        
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
    
    # Test DataFrame optimization
    print(f"\nTesting DataFrame Optimization")
    print("-" * 30)
    
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
    
    # Optimize DataFrame using schema
    df_optimized = schema_manager.optimize_dataframe_for_context(df, collection_name)
    print(f"\nOptimized DataFrame: {df_optimized.shape[0]} rows x {df_optimized.shape[1]} columns")
    print(f"   Columns: {list(df_optimized.columns)}")
    
    # Show sample of optimized data
    print(f"\nSample optimized data:")
    print(df_optimized.head(2).to_string())
    
    # Test column inclusion logic
    print(f"\nTesting Column Inclusion Logic")
    print("-" * 30)
    
    test_columns = ['_id_oid', 'type', 'value', 'Results', 'embedding', 'createdAt_date']
    for col in test_columns:
        should_include = schema_manager.should_include_column_in_context(collection_name, col)
        max_len = schema_manager.get_column_max_length(collection_name, col)
        print(f"   {col}: {'Include' if should_include else 'Exclude'} (max length: {max_len})")
    
    print(f"\nSchema Manager Test Complete!")
    print(f"\nBenefits of this approach:")
    print(f"   1. No more parsing issues - schema is predefined")
    print(f"   2. Optimized token usage - only essential data included")
    print(f"   3. Consistent data handling - standardized across collections")
    print(f"   4. Easy maintenance - add new collections via config")
    print(f"   5. Future-proof - schema evolves without code changes")

if __name__ == "__main__":
    test_schema_manager() 