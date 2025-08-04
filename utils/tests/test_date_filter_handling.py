#!/usr/bin/env python3
"""
Test script to verify date filter column handling.

This script tests the date filter column handling for tables with and without date filtering.
"""

import yaml

def test_date_filter_handling():
    """Test the date filter column handling logic."""
    
    # Load the config
    with open('config/rag_tables_config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    tables_meta = {t['name']: t for t in config['tables']}
    
    def get_date_filter_column(table_name):
        """Safely get the date filter column for a table."""
        if table_name not in tables_meta:
            return None
        
        date_column = tables_meta[table_name].get('date_filter_column')
        if date_column and date_column.strip():
            return date_column.strip()
        return None
    
    def has_date_filtering(table_name):
        """Check if a table has date filtering configured."""
        return get_date_filter_column(table_name) is not None
    
    # Test cases
    test_cases = [
        ('epc_non_domestic_scotland', True, 'LODGEMENT_DATE'),
        ('epc_domestic_scotland', True, 'lodgement_date'),
        ('bev', False, None),
        ('phev', False, None),
        ('nonexistent_table', False, None)
    ]
    
    print("🧪 Testing Date Filter Column Handling")
    print("=" * 50)
    
    all_passed = True
    
    for table_name, expected_has_filtering, expected_column in test_cases:
        print(f"\n📊 Testing table: {table_name}")
        
        # Test has_date_filtering
        actual_has_filtering = has_date_filtering(table_name)
        print(f"   Has date filtering: {actual_has_filtering} (expected: {expected_has_filtering})")
        
        # Test get_date_filter_column
        actual_column = get_date_filter_column(table_name)
        print(f"   Date filter column: '{actual_column}' (expected: '{expected_column}')")
        
        # Check results
        if actual_has_filtering == expected_has_filtering and actual_column == expected_column:
            print("   ✅ PASS")
        else:
            print("   ❌ FAIL")
            all_passed = False
    
    print(f"\n📊 Overall Result: {'✅ ALL TESTS PASSED' if all_passed else '❌ SOME TESTS FAILED'}")
    
    # Show summary
    print(f"\n📋 Summary:")
    tables_with_filtering = [name for name, has_filtering, _ in test_cases if has_filtering]
    tables_without_filtering = [name for name, has_filtering, _ in test_cases if not has_filtering]
    
    print(f"   Tables with date filtering: {tables_with_filtering}")
    print(f"   Tables without date filtering: {tables_without_filtering}")
    
    return all_passed

if __name__ == "__main__":
    test_date_filter_handling() 