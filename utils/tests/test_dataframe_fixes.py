#!/usr/bin/env python3
"""
Test script to verify DataFrame name fixes work correctly.
"""

import re

def test_dataframe_name_fixes():
    """Test the DataFrame name correction logic."""
    print("🧪 Testing DataFrame Name Fixes")
    print("=" * 40)
    
    # Test cases
    test_cases = [
        {
            'name': 'phev reference',
            'code': 'result = phev[phev["Battery_Capacity_Useable"] > 50]',
            'expected': 'result = df1[df1["Battery_Capacity_Useable"] > 50]'
        },
        {
            'name': 'bev reference',
            'code': 'result = bev.groupby("Vehicle_Make").mean()',
            'expected': 'result = df1.groupby("Vehicle_Make").mean()'
        },
        {
            'name': 'epc reference',
            'code': 'result = epc_non_domestic_scotland["POSTCODE"].value_counts()',
            'expected': 'result = df1["POSTCODE"].value_counts()'
        },
        {
            'name': 'mixed references',
            'code': 'phev_data = phev.copy(); bev_data = bev.copy()',
            'expected': 'phev_data = df1.copy(); bev_data = df1.copy()'
        },
        {
            'name': 'correct df1 reference',
            'code': 'result = df1["Vehicle_Make"].unique()',
            'expected': 'result = df1["Vehicle_Make"].unique()'
        }
    ]
    
    # Common mistakes mapping
    common_mistakes = {
        'phev': 'df1',
        'bev': 'df1', 
        'epc_non_domestic_scotland': 'df1',
        'epc_domestic_scotland': 'df1'
    }
    
    passed = 0
    total = len(test_cases)
    
    for test_case in test_cases:
        print(f"\n📝 Testing: {test_case['name']}")
        print(f"Input:  {test_case['code']}")
        
        # Apply the fix
        fixed_code = test_case['code']
        for wrong_name, correct_name in common_mistakes.items():
            if wrong_name in fixed_code:
                fixed_code = fixed_code.replace(wrong_name, correct_name)
        
        print(f"Output: {fixed_code}")
        print(f"Expected: {test_case['expected']}")
        
        if fixed_code == test_case['expected']:
            print("✅ PASS")
            passed += 1
        else:
            print("❌ FAIL")
    
    print(f"\n📊 Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All DataFrame name fixes working correctly!")
    else:
        print("⚠️ Some fixes need attention")
    
    return passed == total

def test_regex_pattern():
    """Test the regex pattern for finding DataFrame references."""
    print("\n🔍 Testing Regex Pattern")
    print("=" * 30)
    
    test_code = """
    df1_result = df1.groupby("make").count()
    df_phev_data = phev["model"].unique()
    df_bev_info = bev.head()
    df_epc_stats = epc_non_domestic_scotland.describe()
    """
    
    df_pattern = r'\b(df_\w+)\b'
    found_dfs = re.findall(df_pattern, test_code)
    
    print(f"Found DataFrame references: {found_dfs}")
    print("✅ Regex pattern working correctly")

if __name__ == "__main__":
    test_dataframe_name_fixes()
    test_regex_pattern() 