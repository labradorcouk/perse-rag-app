#!/usr/bin/env python3
"""
Test script for dynamic DataFrame name correction.
"""

import re

def test_dynamic_dataframe_correction():
    """Test the dynamic DataFrame name correction logic."""
    print("🧪 Testing Dynamic DataFrame Name Correction")
    print("=" * 50)
    
    # Test cases with different scenarios
    test_cases = [
        {
            'name': 'single table - phev',
            'code': 'result = phev[phev["Battery_Capacity_Useable"] > 50]',
            'available_dfs': {'df1', 'pd', 'plt', 'np'},
            'expected': 'result = df1[df1["Battery_Capacity_Useable"] > 50]'
        },
        {
            'name': 'multiple tables - bev',
            'code': 'result = bev.groupby("Vehicle_Make").mean()',
            'available_dfs': {'df1', 'df_bev', 'pd', 'plt', 'np'},
            'expected': 'result = df1.groupby("Vehicle_Make").mean()'
        },
        {
            'name': 'epc table',
            'code': 'result = epc_non_domestic_scotland["POSTCODE"].value_counts()',
            'available_dfs': {'df1', 'pd', 'plt', 'np'},
            'expected': 'result = df1["POSTCODE"].value_counts()'
        },
        {
            'name': 'mixed references',
            'code': 'phev_data = phev.copy(); bev_data = bev.copy()',
            'available_dfs': {'df1', 'df_bev', 'pd', 'plt', 'np'},
            'expected': 'phev_data = df1.copy(); bev_data = df1.copy()'
        },
        {
            'name': 'correct df1 reference',
            'code': 'result = df1["Vehicle_Make"].unique()',
            'available_dfs': {'df1', 'pd', 'plt', 'np'},
            'expected': 'result = df1["Vehicle_Make"].unique()'
        },
        {
            'name': 'new table - solar',
            'code': 'result = solar_panels["Efficiency"].mean()',
            'available_dfs': {'df1', 'df_solar', 'pd', 'plt', 'np'},
            'expected': 'result = df1["Efficiency"].mean()'
        },
        {
            'name': 'wind energy table',
            'code': 'result = wind_turbines.groupby("Location").sum()',
            'available_dfs': {'df1', 'df_wind', 'pd', 'plt', 'np'},
            'expected': 'result = df1.groupby("Location").sum()'
        }
    ]
    
    # Python keywords to filter out
    python_keywords = {
        'pd', 'np', 'plt', 'result', 'fig', 'df1', 'df2', 'df3', 'df4', 'df5',
        'import', 'from', 'as', 'in', 'if', 'else', 'for', 'while', 'def', 'class',
        'return', 'True', 'False', 'None', 'and', 'or', 'not', 'is', 'lambda',
        'try', 'except', 'finally', 'with', 'raise', 'assert', 'del', 'global',
        'nonlocal', 'pass', 'break', 'continue', 'yield', 'async', 'await'
    }
    
    passed = 0
    total = len(test_cases)
    
    for test_case in test_cases:
        print(f"\n📝 Testing: {test_case['name']}")
        print(f"Input:  {test_case['code']}")
        print(f"Available DFs: {test_case['available_dfs']}")
        
        # Apply dynamic correction
        pandas_code = test_case['code']
        available_dfs = test_case['available_dfs']
        
        # Find all potential DataFrame references
        df_pattern = r'\b([a-zA-Z_][a-zA-Z0-9_]*)\b'
        potential_dfs = re.findall(df_pattern, pandas_code)
        
        corrections_made = []
        
        for potential_df in potential_dfs:
            # Skip if it's a Python keyword, built-in, or already available
            if (potential_df in python_keywords or 
                potential_df in available_dfs or 
                potential_df.startswith('df_') or
                potential_df == 'df1'):
                continue
            
            # Check if this looks like a DataFrame reference
            df_usage_pattern = rf'\b{re.escape(potential_df)}\b'
            if re.search(df_usage_pattern, pandas_code):
                # Try to find the most appropriate replacement
                if 'df1' in available_dfs:
                    replacement = 'df1'
                else:
                    # Use the first available DataFrame
                    available_list = [df for df in available_dfs if df.startswith('df')]
                    replacement = available_list[0] if available_list else 'df1'
                
                # Make the replacement
                pandas_code = re.sub(rf'\b{re.escape(potential_df)}\b', replacement, pandas_code)
                corrections_made.append(f"'{potential_df}' → '{replacement}'")
        
        print(f"Output: {pandas_code}")
        print(f"Expected: {test_case['expected']}")
        
        if corrections_made:
            print(f"Corrections: {', '.join(corrections_made)}")
        
        if pandas_code == test_case['expected']:
            print("✅ PASS")
            passed += 1
        else:
            print("❌ FAIL")
    
    print(f"\n📊 Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All dynamic DataFrame corrections working correctly!")
    else:
        print("⚠️ Some corrections need attention")
    
    return passed == total

def test_edge_cases():
    """Test edge cases for the dynamic correction."""
    print("\n🔍 Testing Edge Cases")
    print("=" * 30)
    
    edge_cases = [
        {
            'name': 'no df1 available',
            'code': 'result = phev.head()',
            'available_dfs': {'df_bev', 'df_solar', 'pd', 'plt', 'np'},
            'expected': 'result = df_bev.head()'
        },
        {
            'name': 'column names not DataFrames',
            'code': 'result = df1["phev_column"].sum()',
            'available_dfs': {'df1', 'pd', 'plt', 'np'},
            'expected': 'result = df1["phev_column"].sum()'
        },
        {
            'name': 'function calls not DataFrames',
            'code': 'result = pd.DataFrame(phev_data)',
            'available_dfs': {'df1', 'pd', 'plt', 'np'},
            'expected': 'result = pd.DataFrame(phev_data)'
        }
    ]
    
    python_keywords = {
        'pd', 'np', 'plt', 'result', 'fig', 'df1', 'df2', 'df3', 'df4', 'df5',
        'import', 'from', 'as', 'in', 'if', 'else', 'for', 'while', 'def', 'class',
        'return', 'True', 'False', 'None', 'and', 'or', 'not', 'is', 'lambda',
        'try', 'except', 'finally', 'with', 'raise', 'assert', 'del', 'global',
        'nonlocal', 'pass', 'break', 'continue', 'yield', 'async', 'await'
    }
    
    for case in edge_cases:
        print(f"\n📝 Testing: {case['name']}")
        print(f"Input:  {case['code']}")
        
        pandas_code = case['code']
        available_dfs = case['available_dfs']
        
        # Apply the same logic as in the main function
        df_pattern = r'\b([a-zA-Z_][a-zA-Z0-9_]*)\b'
        potential_dfs = re.findall(df_pattern, pandas_code)
        
        for potential_df in potential_dfs:
            if (potential_df in python_keywords or 
                potential_df in available_dfs or 
                potential_df.startswith('df_') or
                potential_df == 'df1'):
                continue
            
            df_usage_pattern = rf'\b{re.escape(potential_df)}\b'
            if re.search(df_usage_pattern, pandas_code):
                if 'df1' in available_dfs:
                    replacement = 'df1'
                else:
                    available_list = [df for df in available_dfs if df.startswith('df')]
                    replacement = available_list[0] if available_list else 'df1'
                
                pandas_code = re.sub(rf'\b{re.escape(potential_df)}\b', replacement, pandas_code)
        
        print(f"Output: {pandas_code}")
        print(f"Expected: {case['expected']}")
        
        if pandas_code == case['expected']:
            print("✅ PASS")
        else:
            print("❌ FAIL")

if __name__ == "__main__":
    test_dynamic_dataframe_correction()
    test_edge_cases() 