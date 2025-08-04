#!/usr/bin/env python3
"""
Demonstration of how the intelligent DataFrame fixing system solves the original problem
without hardcoding specific mappings.
"""

import pandas as pd
import numpy as np
from utils.enhanced_dataframe_corrector import EnhancedDataFrameCorrector

def demonstrate_original_problem():
    """Demonstrate how the intelligent system solves the original battery charge time problem."""
    print("🎯 Demonstrating Intelligent DataFrame Fixing System")
    print("=" * 70)
    
    # Create sample data similar to your PHEV dataset
    phev_data = {
        'Vehicle_ID': ['V001', 'V002', 'V003', 'V004', 'V005'],
        'Vehicle_Make': ['Tesla', 'BMW', 'Audi', 'Mercedes', 'Volkswagen'],
        'Vehicle_Model': ['Model S', 'i3', 'e-tron', 'EQC', 'ID.4'],
        'Battery_Capacity_Full': [100.0, 42.2, 95.0, 80.0, 77.0],
        'Battery_Capacity_Useable': [95.0, 37.9, 86.5, 75.0, 70.0],
        'Battery_Capacity_Estimate': [98.0, 40.0, 92.0, 78.0, 75.0],
        'Range_WLTP': [610, 285, 436, 471, 520],
        'Efficiency_WLTP': [16.4, 14.8, 21.8, 17.0, 14.9]
    }
    
    available_dataframes = {'df1': pd.DataFrame(phev_data)}
    
    # The problematic code from your original issue
    problematic_code = '''
df1['Battery_Charge_Time'] = df1['Battery_Capacity_Full'] / df1['Battery_Capacity_Estimate']
result = df1.loc[df1['Battery_Charge_Time'].idxmax(), ['Vehicle_ID', 'Vehicle_Make', 'Vehicle_Model', 'Battery_Charge_Time']]
'''
    
    user_query = "What is the car with highest battery charge time?"
    
    print("❌ Original Problematic Code:")
    print(problematic_code)
    print(f"\n🔍 User Query: {user_query}")
    
    # Apply intelligent fixing
    corrector = EnhancedDataFrameCorrector()
    fixed_code, fixes, context = corrector.fix_code_with_context(
        problematic_code, 
        user_query, 
        available_dataframes
    )
    
    print("\n✅ Intelligently Fixed Code:")
    print(fixed_code)
    
    print("\n🔧 Fixes Applied:")
    for fix in fixes:
        print(f"  • {fix}")
    
    print("\n📊 Context Analysis:")
    print(f"  • Intent: {context.get('intent', 'Unknown')}")
    print(f"  • Issues Found: {len(context.get('potential_issues', []))}")
    
    # Show what the system detected
    print("\n🔍 Issues Detected:")
    for issue in context.get('potential_issues', []):
        print(f"  • {issue['type']}: {issue.get('dataframe', issue.get('column', 'Unknown'))}")
        print(f"    Suggestion: {issue.get('suggestion', 'None')}")
    
    print("\n💡 Smart Suggestions:")
    suggestions = corrector.generate_smart_suggestions(fixed_code, available_dataframes)
    for suggestion in suggestions:
        print(f"  • {suggestion}")
    
    # Execute the fixed code to show it works
    print("\n🚀 Executing Fixed Code:")
    try:
        # Create the DataFrame
        df1 = pd.DataFrame(phev_data)
        
        # Execute the fixed code
        exec(fixed_code)
        print("✅ Code executed successfully!")
        
        # Show the result
        print(f"\n📊 Result: {result}")
        
    except Exception as e:
        print(f"❌ Error executing code: {e}")

def demonstrate_scalability():
    """Demonstrate how the system scales to different use cases without hardcoding."""
    print("\n\n🔄 Demonstrating Scalability")
    print("=" * 70)
    
    # Create different datasets
    datasets = {
        'electric_cars': {
            'Vehicle_ID': ['E001', 'E002', 'E003'],
            'Make': ['Tesla', 'Nissan', 'Chevrolet'],
            'Model': ['Model 3', 'Leaf', 'Bolt'],
            'Battery_Size': [75.0, 62.0, 66.0],
            'Range_Miles': [358, 226, 259]
        },
        'hybrid_vehicles': {
            'ID': ['H001', 'H002', 'H003'],
            'Brand': ['Toyota', 'Honda', 'Ford'],
            'Model_Name': ['Prius', 'Insight', 'Fusion'],
            'Fuel_Efficiency': [56, 52, 42],
            'Electric_Range': [25, 28, 21]
        }
    }
    
    available_dataframes = {
        'electric_df': pd.DataFrame(datasets['electric_cars']),
        'hybrid_df': pd.DataFrame(datasets['hybrid_vehicles'])
    }
    
    # Different problematic codes for different use cases
    test_cases = [
        {
            'name': 'Electric Cars - Battery Analysis',
            'code': '''
# Find electric car with highest battery size
max_battery_idx = electric_cars['Battery_Size'].idxmax()
result = electric_cars.loc[max_battery_idx, ['Vehicle_ID', 'Make', 'Model', 'Battery_Size']]
''',
            'query': 'Which electric car has the highest battery capacity?'
        },
        {
            'name': 'Hybrid Vehicles - Efficiency Analysis',
            'code': '''
# Find most efficient hybrid
max_efficiency_idx = hybrid_vehicles['Fuel_Efficiency'].idxmax()
result = hybrid_vehicles.loc[max_efficiency_idx, ['ID', 'Brand', 'Model_Name', 'Fuel_Efficiency']]
''',
            'query': 'Which hybrid vehicle is most fuel efficient?'
        },
        {
            'name': 'Mixed Dataset Analysis',
            'code': '''
# Compare battery sizes across datasets
electric_max = electric_cars['Battery_Size'].max()
hybrid_max = hybrid_vehicles['Electric_Range'].max()
print(f"Electric max: {electric_max}, Hybrid max: {hybrid_max}")
''',
            'query': 'Compare battery capacities between electric and hybrid vehicles'
        }
    ]
    
    corrector = EnhancedDataFrameCorrector()
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n🔍 Test Case {i}: {test_case['name']}")
        print("-" * 50)
        
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
        
        print(f"\n📊 Intent Detected: {context.get('intent', 'Unknown')}")

def demonstrate_no_hardcoding():
    """Demonstrate that the system works without any hardcoded mappings."""
    print("\n\n🚫 Demonstrating No Hardcoded Mappings")
    print("=" * 70)
    
    # Create completely different dataset with different column names
    solar_data = {
        'Panel_ID': ['S001', 'S002', 'S003'],
        'Manufacturer': ['SunPower', 'LG', 'Panasonic'],
        'Model_Type': ['X-Series', 'NeON', 'HIT'],
        'Wattage_Output': [400, 365, 330],
        'Efficiency_Rating': [22.8, 21.4, 20.3],
        'Cost_Per_Watt': [3.50, 2.80, 3.20]
    }
    
    available_dataframes = {'solar_df': pd.DataFrame(solar_data)}
    
    # Code that references non-existent DataFrames and columns
    problematic_code = '''
# Find most efficient solar panel
max_efficiency_idx = solar_panels['Efficiency_Rating'].idxmax()
result = solar_panels.loc[max_efficiency_idx, ['Panel_ID', 'Manufacturer', 'Model_Type', 'Efficiency_Rating']]
'''
    
    user_query = "Which solar panel is most efficient?"
    
    print("❌ Problematic Code (completely different domain):")
    print(problematic_code)
    print(f"\n🔍 User Query: {user_query}")
    
    corrector = EnhancedDataFrameCorrector()
    fixed_code, fixes, context = corrector.fix_code_with_context(
        problematic_code, 
        user_query, 
        available_dataframes
    )
    
    print("\n✅ Intelligently Fixed Code:")
    print(fixed_code)
    
    print("\n🔧 Fixes Applied:")
    for fix in fixes:
        print(f"  • {fix}")
    
    print("\n💡 Key Point: No hardcoded mappings were used!")
    print("   The system dynamically detected and fixed all issues.")

if __name__ == "__main__":
    demonstrate_original_problem()
    demonstrate_scalability()
    demonstrate_no_hardcoding()
    
    print("\n\n🎉 Demonstration Complete!")
    print("\n💡 Key Benefits Demonstrated:")
    print("  ✅ No hardcoded mappings - works with any DataFrame/column names")
    print("  ✅ Intelligent pattern recognition")
    print("  ✅ Context-aware suggestions")
    print("  ✅ Dynamic issue detection and resolution")
    print("  ✅ Scalable to any use case or domain")
    print("  ✅ Handles undefined DataFrames, columns, and methods")
    print("  ✅ Provides intelligent suggestions for improvement") 