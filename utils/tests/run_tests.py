#!/usr/bin/env python3
"""
Test runner for utils tests.
"""

import sys
import os
import importlib
import glob

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

def run_all_tests():
    """Run all test files in the tests directory."""
    
    print("🧪 Running all utils tests...")
    print("=" * 60)
    
    # Get all test files
    test_files = glob.glob(os.path.join(os.path.dirname(__file__), "test_*.py"))
    
    passed = 0
    failed = 0
    
    for test_file in test_files:
        test_name = os.path.basename(test_file)[:-3]  # Remove .py extension
        
        print(f"\n🔍 Running {test_name}...")
        
        try:
            # Import and run the test module
            module = importlib.import_module(f"utils.tests.{test_name}")
            
            # Look for test functions
            test_functions = [name for name in dir(module) if name.startswith('test_')]
            
            if test_functions:
                for func_name in test_functions:
                    test_func = getattr(module, func_name)
                    if callable(test_func):
                        try:
                            result = test_func()
                            if result:
                                print(f"  ✅ {func_name} passed")
                                passed += 1
                            else:
                                print(f"  ❌ {func_name} failed")
                                failed += 1
                        except Exception as e:
                            print(f"  ❌ {func_name} failed with error: {e}")
                            failed += 1
            else:
                # If no test functions found, try to run the module directly
                if hasattr(module, '__main__'):
                    print(f"  ⚠️  No test functions found in {test_name}")
                else:
                    print(f"  ✅ {test_name} completed")
                    passed += 1
                    
        except Exception as e:
            print(f"  ❌ Failed to run {test_name}: {e}")
            failed += 1
    
    print("\n" + "=" * 60)
    print(f"📊 Test Results:")
    print(f"  ✅ Passed: {passed}")
    print(f"  ❌ Failed: {failed}")
    print(f"  📈 Total: {passed + failed}")
    
    if failed == 0:
        print("🎉 All tests passed!")
    else:
        print("⚠️  Some tests failed. Check the output above for details.")
    
    return failed == 0

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1) 