#!/usr/bin/env python3
"""
Test script to verify the improved DataFrame correction logic.
"""

import pandas as pd
import sys
import os

# Add the parent directory to the path so we can import utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

def test_improved_correction():
    """Test the improved DataFrame correction logic."""
    try:
        from utils.dataframe_corrector import DataFrameCorrector
        
        # Create sample data
        df1 = pd.DataFrame({
            'Vehicle_ID': ['V001', 'V002', 'V003'],
            'Vehicle_Make': ['Tesla', 'BMW', 'Audi'],
            'Vehicle_Model': ['Model S', 'i3', 'e-tron'],
            'Battery_Capacity_Full': [100.0, 42.2, 95.0]
        })
        
        available_dataframes = {'df1': df1}
        
        # Test the problematic code from the user's example
        problematic_code = '''
df1['Charge_Time'] = df1['Battery_Capacity_Full'] / 50
result = df1.loc[df1['Charge_Time'].idxmax(), ['Vehicle_Make', 'Vehicle_Model', 'Charge_Time']]
'''
        
        print("🧪 Testing Improved DataFrame Correction")
        print("=" * 50)
        
        print("❌ Original code:")
        print(problematic_code)
        
        # Create corrector
        corrector = DataFrameCorrector()
        
        # Fix the code
        fixed_code, fixes = corrector.correct_dataframe_names(problematic_code, available_dataframes)
        
        print("\n✅ Fixed code:")
        print(fixed_code)
        
        print("\n🔧 Fixes applied:")
        for fix in fixes:
            print(f"  • {fix}")
        
        # Test validation
        undefined_refs = corrector.validate_dataframe_references(fixed_code, available_dataframes)
        
        if not undefined_refs:
            print("\n✅ All DataFrame references resolved!")
        else:
            print(f"\n⚠️  Unresolved references: {undefined_refs}")
        
        # Test that column names are preserved
        print("\n🔍 Checking that column names are preserved:")
        expected_columns = ['Vehicle_Make', 'Vehicle_Model', 'Battery_Capacity_Full', 'Charge_Time']
        for col in expected_columns:
            if col in fixed_code:
                print(f"  ✅ Column '{col}' preserved")
            else:
                print(f"  ❌ Column '{col}' missing or changed")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_column_name_preservation():
    """Test that column names are not incorrectly treated as DataFrame references."""
    try:
        from utils.dataframe_corrector import DataFrameCorrector
        
        # Create sample data
        df1 = pd.DataFrame({
            'Vehicle_ID': ['V001', 'V002'],
            'Vehicle_Make': ['Tesla', 'BMW'],
            'Battery_Capacity_Full': [100.0, 42.2]
        })
        
        available_dataframes = {'df1': df1}
        
        # Test code that uses column names
        test_code = '''
result = df1[['Vehicle_Make', 'Battery_Capacity_Full']].max()
'''
        
        print("\n🧪 Testing Column Name Preservation")
        print("=" * 50)
        
        print("❌ Original code:")
        print(test_code)
        
        # Create corrector
        corrector = DataFrameCorrector()
        
        # Fix the code
        fixed_code, fixes = corrector.correct_dataframe_names(test_code, available_dataframes)
        
        print("\n✅ Fixed code:")
        print(fixed_code)
        
        print("\n🔧 Fixes applied:")
        for fix in fixes:
            print(f"  • {fix}")
        
        # Check that column names are preserved
        expected_columns = ['Vehicle_Make', 'Battery_Capacity_Full']
        for col in expected_columns:
            if col in fixed_code:
                print(f"  ✅ Column '{col}' preserved correctly")
            else:
                print(f"  ❌ Column '{col}' was incorrectly changed")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🧪 Testing Improved DataFrame Correction System")
    print("=" * 60)
    
    # Test the main fix
    test_improved_correction()
    
    # Test column name preservation
    test_column_name_preservation()
    
    print("\n🎉 All tests completed!")
    print("\n💡 The improved DataFrame correction system should now:")
    print("  ✅ Preserve column names correctly")
    print("  ✅ Only fix actual DataFrame references")
    print("  ✅ Handle the original battery charge time problem") 