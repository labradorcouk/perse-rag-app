#!/usr/bin/env python3
"""
Test script to demonstrate the intelligent DataFrame fixing system.
This shows how the system can dynamically handle any DataFrame reference issues
without hardcoding specific mappings.
"""

import pandas as pd
import numpy as np
from utils.enhanced_dataframe_corrector import EnhancedDataFrameCorrector

def create_sample_data():
    """Create sample DataFrames for testing."""
    # Sample PHEV data
    phev_data = {
        'Vehicle_ID': ['V001', 'V002', 'V003', 'V004', 'V005'],
        'Vehicle_Make': ['Tesla', 'BMW', 'Audi', 'Mercedes', 'Volkswagen'],
        'Vehicle_Model': ['Model S', 'i3', 'e-tron', 'EQC', 'ID.4'],
        'Battery_Capacity_Full': [100.0, 42.2, 95.0, 80.0, 77.0],
        'Battery_Capacity_Useable': [95.0, 37.9, 86.5, 75.0, 70.0],
        'Range_WLTP': [610, 285, 436, 471, 520],
        'Efficiency_WLTP': [16.4, 14.8, 21.8, 17.0, 14.9]
    }
    
    df1 = pd.DataFrame(phev_data)
    return {'df1': df1}

def test_intelligent_fixing():
    """Test the intelligent DataFrame fixing system."""
    print("🧪 Testing Intelligent DataFrame Fixing System")
    print("=" * 60)
    
    # Create sample data
    available_dataframes = create_sample_data()
    corrector = EnhancedDataFrameCorrector()
    
    # Test cases with different issues
    test_cases = [
        {
            'name': 'Undefined DataFrame Reference',
            'code': '''
# Find the car with highest battery capacity
max_battery_idx = phev['Battery_Capacity_Full'].idxmax()
result = phev.loc[max_battery_idx, ['Vehicle_ID', 'Vehicle_Make', 'Vehicle_Model', 'Battery_Capacity_Full']]
print(result)
''',
            'query': 'What is the car with highest battery capacity?'
        },
        {
            'name': 'Undefined Column Reference',
            'code': '''
# Find vehicle with highest battery charge time
df1['Battery_Charge_Time'] = df1['Battery_Capacity_Full'] / df1['Charging_Rate']
max_charge_time_idx = df1['Battery_Charge_Time'].idxmax()
result = df1.loc[max_charge_time_idx, ['Vehicle_ID', 'Vehicle_Make', 'Vehicle_Model']]
''',
            'query': 'What is the car with highest battery charge time?'
        },
        {
            'name': 'Incorrect Method Usage',
            'code': '''
# Get top 5 vehicles by battery capacity
top_5 = df1.sort_values('Battery_Capacity_Full', ascending=False).head(5)
print(top_5[['Vehicle_ID', 'Vehicle_Make', 'Vehicle_Model', 'Battery_Capacity_Full']])
''',
            'query': 'Show me the top 5 vehicles by battery capacity'
        },
        {
            'name': 'Mixed Issues',
            'code': '''
# Analyze battery data
phev_clean = phev.dropna(subset=['Battery_Capacity_Full'])
max_battery = phev_clean['Battery_Capacity_Full'].max()
max_vehicle = phev_clean[phev_clean['Battery_Capacity_Full'] == max_battery]
print(f"Vehicle with highest battery capacity: {max_vehicle['Vehicle_Make'].iloc[0]} {max_vehicle['Vehicle_Model'].iloc[0]}")
''',
            'query': 'Find the vehicle with the highest battery capacity'
        }
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n🔍 Test Case {i}: {test_case['name']}")
        print("-" * 40)
        
        print("❌ Original Code:")
        print(test_case['code'])
        
        # Apply intelligent fixing
        fixed_code, fixes, context = corrector.fix_code_with_context(
            test_case['code'], 
            test_case['query'], 
            available_dataframes
        )
        
        print("\n✅ Fixed Code:")
        print(fixed_code)
        
        print("\n🔧 Fixes Applied:")
        for fix in fixes:
            print(f"  • {fix}")
        
        print("\n📊 Context Analysis:")
        print(f"  • Intent: {context.get('intent', 'Unknown')}")
        print(f"  • Issues Found: {len(context.get('potential_issues', []))}")
        print(f"  • Available DataFrames: {context.get('available_dataframes', [])}")
        
        # Validate the fixed code
        unresolved = corrector.validate_dataframe_references(fixed_code, available_dataframes)
        if unresolved:
            print("\n⚠️  Unresolved Issues:")
            for issue in unresolved:
                print(f"  • {issue}")
        else:
            print("\n✅ All DataFrame references resolved!")
        
        print("\n💡 Smart Suggestions:")
        suggestions = corrector.generate_smart_suggestions(fixed_code, available_dataframes)
        for suggestion in suggestions[:3]:  # Show top 3 suggestions
            print(f"  • {suggestion}")

def test_dynamic_column_fixing():
    """Test dynamic column fixing capabilities."""
    print("\n\n🧪 Testing Dynamic Column Fixing")
    print("=" * 60)
    
    available_dataframes = create_sample_data()
    corrector = EnhancedDataFrameCorrector()
    
    # Test with various column name variations
    problematic_code = '''
# Try to access battery capacity with different column names
battery_capacity = df1['Battery_Capacity']  # Wrong column name
full_capacity = df1['Full_Capacity']        # Wrong column name
charge_time = df1['Charge_Time']            # Non-existent column
'''
    
    print("❌ Problematic Code:")
    print(problematic_code)
    
    # Apply intelligent fixing
    fixed_code, fixes = corrector.correct_dataframe_names(problematic_code, available_dataframes)
    
    print("\n✅ Fixed Code:")
    print(fixed_code)
    
    print("\n🔧 Fixes Applied:")
    for fix in fixes:
        print(f"  • {fix}")

def test_context_aware_fixing():
    """Test context-aware fixing based on user query."""
    print("\n\n🧪 Testing Context-Aware Fixing")
    print("=" * 60)
    
    available_dataframes = create_sample_data()
    corrector = EnhancedDataFrameCorrector()
    
    # Test with battery-related query
    battery_query = "What is the car with highest battery capacity?"
    battery_code = '''
# Find maximum battery capacity
max_capacity = df1['Battery_Capacity_Full'].max()
print(f"Maximum battery capacity: {max_capacity} kWh")
'''
    
    print(f"🔍 User Query: {battery_query}")
    print("\n❌ Original Code:")
    print(battery_code)
    
    # Apply context-aware fixing
    fixed_code, fixes, context = corrector.fix_code_with_context(
        battery_code, 
        battery_query, 
        available_dataframes
    )
    
    print("\n✅ Context-Aware Fixed Code:")
    print(fixed_code)
    
    print("\n🔧 Context-Aware Fixes:")
    for fix in fixes:
        print(f"  • {fix}")

if __name__ == "__main__":
    test_intelligent_fixing()
    test_dynamic_column_fixing()
    test_context_aware_fixing()
    
    print("\n\n🎉 All tests completed!")
    print("\n💡 Key Benefits of This System:")
    print("  • No hardcoded mappings - works with any DataFrame/column names")
    print("  • Intelligent pattern recognition")
    print("  • Context-aware suggestions")
    print("  • Dynamic issue detection and resolution")
    print("  • Scalable to any use case") 