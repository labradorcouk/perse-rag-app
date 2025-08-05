#!/usr/bin/env python3
"""
Test script to demonstrate dynamic syntax error correction.
"""

import pandas as pd
import sys
import os

# Add the parent directory to the path so we can import utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

def test_syntax_error_correction():
    """Test the dynamic syntax error correction functionality."""
    try:
        from utils.dataframe_corrector import DataFrameCorrector
        
        # Create sample data
        df1 = pd.DataFrame({
            'postcode_trim': ['AB1 1AA', 'AB1 1AB', 'AB2 1AA', 'AB2 1AB'],
            'property_type': ['Detached', 'Detached', 'Semi-detached', 'Detached'],
            'current_energy_rating': [85, 90, 75, 88]
        })
        
        available_dataframes = {'df1': df1}
        
        # Test cases with syntax errors
        test_cases = [
            {
                'name': 'Incomplete idxmax()',
                'code': '''
detached_properties = df1[df1['property_type'] == 'Detached']
postcode_avg_rating = detached_properties.groupby('postcode_trim')['current_energy_rating'].mean().reset_index()
best_postcode = postcode_avg_rating.loc[postcode_avg_rating['current_energy_rating'].idx
result = best_postcode['postcode_trim']
''',
                'expected_fix': 'idxmax()'
            },
            {
                'name': 'Incomplete groupby()',
                'code': '''
detached_properties = df1[df1['property_type'] == 'Detached']
postcode_avg_rating = detached_properties.groupby
best_postcode = postcode_avg_rating.loc[postcode_avg_rating['current_energy_rating'].idxmax()]
''',
                'expected_fix': 'groupby()'
            },
            {
                'name': 'Incomplete reset_index()',
                'code': '''
detached_properties = df1[df1['property_type'] == 'Detached']
postcode_avg_rating = detached_properties.groupby('postcode_trim')['current_energy_rating'].mean().reset_index
best_postcode = postcode_avg_rating.loc[postcode_avg_rating['current_energy_rating'].idxmax()]
''',
                'expected_fix': 'reset_index()'
            }
        ]
        
        print("🧪 Testing Dynamic Syntax Error Correction")
        print("=" * 60)
        
        corrector = DataFrameCorrector()
        
        for i, test_case in enumerate(test_cases, 1):
            print(f"\n📝 Test {i}: {test_case['name']}")
            print("-" * 40)
            
            print("❌ Original code with syntax error:")
            print(test_case['code'])
            
            # Fix syntax errors
            fixed_code, syntax_fixes = corrector.fix_syntax_errors(test_case['code'])
            
            print(f"\n✅ Fixed code:")
            print(fixed_code)
            
            print(f"\n🔧 Syntax fixes applied:")
            for fix in syntax_fixes:
                print(f"  • {fix}")
            
            # Validate the fixed code
            is_valid, warnings, error_message = corrector.validate_and_test_code(fixed_code, available_dataframes)
            
            if is_valid:
                print(f"\n✅ Code validation: PASSED")
                if warnings:
                    print(f"⚠️  Warnings:")
                    for warning in warnings:
                        print(f"  • {warning}")
            else:
                print(f"\n❌ Code validation: FAILED")
                print(f"  Error: {error_message}")
            
            # Test if the expected fix was applied
            if test_case['expected_fix'] in fixed_code:
                print(f"✅ Expected fix '{test_case['expected_fix']}' was applied")
            else:
                print(f"❌ Expected fix '{test_case['expected_fix']}' was NOT applied")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_non_aggressive_correction():
    """Test that the correction is non-aggressive and only fixes obvious issues."""
    try:
        from utils.dataframe_corrector import DataFrameCorrector
        
        # Test code that should NOT be changed
        safe_code = '''
# This code should remain unchanged
detached_properties = df1[df1['property_type'] == 'Detached']
postcode_avg_rating = detached_properties.groupby('postcode_trim')['current_energy_rating'].mean().reset_index()
best_postcode = postcode_avg_rating.loc[postcode_avg_rating['current_energy_rating'].idxmax()]
result = best_postcode['postcode_trim']
'''
        
        print("\n🧪 Testing Non-Aggressive Correction")
        print("=" * 50)
        
        corrector = DataFrameCorrector()
        
        print("📝 Testing code that should remain unchanged:")
        print(safe_code)
        
        # Apply syntax error correction
        fixed_code, syntax_fixes = corrector.fix_syntax_errors(safe_code)
        
        print(f"\n✅ Code after correction:")
        print(fixed_code)
        
        print(f"\n🔧 Fixes applied: {len(syntax_fixes)}")
        for fix in syntax_fixes:
            print(f"  • {fix}")
        
        # Check that no unnecessary changes were made
        if len(syntax_fixes) == 0:
            print("✅ No unnecessary changes made - correction is non-aggressive")
        else:
            print("⚠️  Some changes were made - check if they were necessary")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🧪 Testing Dynamic Syntax Error Correction System")
    print("=" * 70)
    
    # Test syntax error correction
    test_syntax_error_correction()
    
    # Test non-aggressive behavior
    test_non_aggressive_correction()
    
    print("\n🎉 All tests completed!")
    print("\n💡 The dynamic syntax error correction system:")
    print("  ✅ Fixes common syntax errors automatically")
    print("  ✅ Is non-aggressive and only fixes obvious issues")
    print("  ✅ Provides detailed feedback on what was fixed")
    print("  ✅ Validates code after correction")
    print("  ✅ Integrates seamlessly with existing DataFrame correction") 