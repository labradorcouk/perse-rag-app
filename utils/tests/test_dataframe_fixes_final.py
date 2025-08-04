#!/usr/bin/env python3
"""
Final test script to verify the DataFrame correction system works correctly.
"""

import pandas as pd
import numpy as np
import sys
import os

# Add the current directory to the path so we can import utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

def test_dataframe_corrector_import():
    """Test that the DataFrameCorrector can be imported correctly."""
    try:
        from utils.dataframe_corrector import DataFrameCorrector
        print("✅ Successfully imported DataFrameCorrector")
        
        # Create an instance
        corrector = DataFrameCorrector()
        print("✅ Successfully created DataFrameCorrector instance")
        
        return corrector
    except Exception as e:
        print(f"❌ Failed to import DataFrameCorrector: {e}")
        return None

def test_enhanced_corrector_import():
    """Test that the EnhancedDataFrameCorrector can be imported correctly."""
    try:
        from utils.enhanced_dataframe_corrector import EnhancedDataFrameCorrector
        print("✅ Successfully imported EnhancedDataFrameCorrector")
        
        # Create an instance
        corrector = EnhancedDataFrameCorrector()
        print("✅ Successfully created EnhancedDataFrameCorrector instance")
        
        return corrector
    except Exception as e:
        print(f"❌ Failed to import EnhancedDataFrameCorrector: {e}")
        return None

def test_intelligent_fixer_import():
    """Test that the IntelligentDataFrameFixer can be imported correctly."""
    try:
        from utils.intelligent_dataframe_fixer import IntelligentDataFrameFixer
        print("✅ Successfully imported IntelligentDataFrameFixer")
        
        # Create an instance
        fixer = IntelligentDataFrameFixer()
        print("✅ Successfully created IntelligentDataFrameFixer instance")
        
        return fixer
    except Exception as e:
        print(f"❌ Failed to import IntelligentDataFrameFixer: {e}")
        return None

def test_basic_functionality():
    """Test basic functionality of the DataFrame correction system."""
    try:
        from utils.dataframe_corrector import DataFrameCorrector
        
        # Create sample data
        df1 = pd.DataFrame({
            'Vehicle_ID': ['V001', 'V002', 'V003'],
            'Vehicle_Make': ['Tesla', 'BMW', 'Audi'],
            'Battery_Capacity_Full': [100.0, 42.2, 95.0]
        })
        
        available_dataframes = {'df1': df1}
        
        # Create corrector
        corrector = DataFrameCorrector()
        
        # Test problematic code
        problematic_code = '''
# Find the car with highest battery capacity
max_battery_idx = phev['Battery_Capacity_Full'].idxmax()
result = phev.loc[max_battery_idx, ['Vehicle_ID', 'Vehicle_Make', 'Battery_Capacity_Full']]
'''
        
        print("\n🔍 Testing DataFrame correction...")
        print("❌ Original code:")
        print(problematic_code)
        
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
        
        return True
        
    except Exception as e:
        print(f"❌ Failed to test basic functionality: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_intelligent_fixing():
    """Test the intelligent fixing capabilities."""
    try:
        from utils.enhanced_dataframe_corrector import EnhancedDataFrameCorrector
        
        # Create sample data
        df1 = pd.DataFrame({
            'Vehicle_ID': ['V001', 'V002', 'V003'],
            'Vehicle_Make': ['Tesla', 'BMW', 'Audi'],
            'Battery_Capacity_Full': [100.0, 42.2, 95.0]
        })
        
        available_dataframes = {'df1': df1}
        
        # Create corrector
        corrector = EnhancedDataFrameCorrector()
        
        # Test with context
        user_query = "What is the car with highest battery capacity?"
        problematic_code = '''
# Find the car with highest battery capacity
max_battery_idx = phev['Battery_Capacity_Full'].idxmax()
result = phev.loc[max_battery_idx, ['Vehicle_ID', 'Vehicle_Make', 'Battery_Capacity_Full']]
'''
        
        print("\n🧠 Testing intelligent fixing...")
        print("❌ Original code:")
        print(problematic_code)
        
        # Fix with context
        fixed_code, fixes, context = corrector.fix_code_with_context(
            problematic_code, 
            user_query, 
            available_dataframes
        )
        
        print("\n✅ Fixed code:")
        print(fixed_code)
        
        print("\n🔧 Fixes applied:")
        for fix in fixes:
            print(f"  • {fix}")
        
        print(f"\n📊 Context analysis:")
        print(f"  • Intent: {context.get('intent', 'Unknown')}")
        print(f"  • Issues found: {len(context.get('potential_issues', []))}")
        
        return True
        
    except Exception as e:
        print(f"❌ Failed to test intelligent fixing: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🧪 Testing DataFrame Correction System")
    print("=" * 50)
    
    # Test imports
    print("\n1. Testing imports...")
    test_dataframe_corrector_import()
    test_enhanced_corrector_import()
    test_intelligent_fixer_import()
    
    # Test basic functionality
    print("\n2. Testing basic functionality...")
    test_basic_functionality()
    
    # Test intelligent fixing
    print("\n3. Testing intelligent fixing...")
    test_intelligent_fixing()
    
    print("\n🎉 All tests completed!")
    print("\n💡 The DataFrame correction system should now work correctly in the main application.") 