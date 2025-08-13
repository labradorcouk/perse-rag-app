#!/usr/bin/env python3
"""
Test Execution Script for Dynamic Learning System
Runs all tests and provides a comprehensive summary.
"""

import sys
import os
import subprocess
import time
from datetime import datetime

def run_test(test_name, test_file, description):
    """Run a specific test and return results."""
    print(f"\n{'='*60}")
    print(f"🧪 Running: {test_name}")
    print(f"📝 Description: {description}")
    print(f"{'='*60}")
    
    start_time = time.time()
    
    try:
        # Run the test
        result = subprocess.run([
            sys.executable, test_file
        ], capture_output=True, text=True, timeout=300)  # 5 minute timeout
        
        end_time = time.time()
        duration = end_time - start_time
        
        if result.returncode == 0:
            print("✅ Test PASSED")
            print(f"⏱️  Duration: {duration:.2f}s")
            return True, duration, result.stdout
        else:
            print("❌ Test FAILED")
            print(f"⏱️  Duration: {duration:.2f}s")
            print(f"🔍 Error Output:")
            print(result.stderr)
            return False, duration, result.stderr
            
    except subprocess.TimeoutExpired:
        print("⏰ Test TIMEOUT (exceeded 5 minutes)")
        return False, 300, "Test timeout"
    except Exception as e:
        print(f"💥 Test ERROR: {e}")
        return False, 0, str(e)

def main():
    """Main function to run all tests."""
    print("🚀 Dynamic Learning System - Test Suite")
    print("=" * 60)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Define tests to run
    tests = [
        {
            'name': 'Enhanced Q&A Patterns',
            'file': 'utils/tests/test_enhanced_qa_patterns.py',
            'description': 'Tests basic Q&A pattern matching functionality'
        },
        {
            'name': 'Dynamic Learning',
            'file': 'utils/tests/test_dynamic_learning.py',
            'description': 'Tests dynamic learning and pattern evolution'
        },
        {
            'name': 'Comprehensive Dynamic Learning',
            'file': 'utils/tests/test_comprehensive_dynamic_learning.py',
            'description': 'Comprehensive test suite for all dynamic learning features'
        },
        {
            'name': 'Integration Tests',
            'file': 'utils/tests/test_integration.py',
            'description': 'End-to-end integration tests for the complete system'
        }
    ]
    
    # Results tracking
    results = []
    total_tests = len(tests)
    passed_tests = 0
    total_duration = 0
    
    # Run each test
    for test in tests:
        if os.path.exists(test['file']):
            success, duration, output = run_test(
                test['name'], 
                test['file'], 
                test['description']
            )
            
            results.append({
                'name': test['name'],
                'success': success,
                'duration': duration,
                'output': output
            })
            
            if success:
                passed_tests += 1
            total_duration += duration
            
            # Save individual test output
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"test_output_{test['name'].lower().replace(' ', '_')}_{timestamp}.txt"
            
            try:
                with open(output_file, 'w', encoding='utf-8') as f:
                    f.write(f"Test: {test['name']}\n")
                    f.write(f"Description: {test['description']}\n")
                    f.write(f"Success: {success}\n")
                    f.write(f"Duration: {duration:.2f}s\n")
                    f.write(f"Timestamp: {datetime.now().isoformat()}\n")
                    f.write("\n" + "="*50 + "\n")
                    f.write("OUTPUT:\n")
                    f.write("="*50 + "\n")
                    f.write(output)
                
                print(f"📄 Test output saved to: {output_file}")
            except Exception as e:
                print(f"⚠️ Failed to save test output: {e}")
        else:
            print(f"\n⚠️ Test file not found: {test['file']}")
            results.append({
                'name': test['name'],
                'success': False,
                'duration': 0,
                'output': f"Test file not found: {test['file']}"
            })
    
    # Generate summary report
    print(f"\n{'='*60}")
    print("📊 TEST SUMMARY REPORT")
    print(f"{'='*60}")
    
    print(f"Total Tests: {total_tests}")
    print(f"Passed: {passed_tests}")
    print(f"Failed: {total_tests - passed_tests}")
    print(f"Success Rate: {(passed_tests / total_tests * 100):.1f}%")
    print(f"Total Duration: {total_duration:.2f}s")
    print(f"Average Duration: {(total_duration / total_tests):.2f}s per test")
    
    # Detailed results
    print(f"\n📋 DETAILED RESULTS:")
    print("-" * 40)
    
    for result in results:
        status = "✅ PASS" if result['success'] else "❌ FAIL"
        duration_str = f"{result['duration']:.2f}s" if result['duration'] > 0 else "N/A"
        print(f"{status} {result['name']} ({duration_str})")
    
    # Save comprehensive report
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = f"comprehensive_test_report_{timestamp}.txt"
    
    try:
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("Dynamic Learning System - Comprehensive Test Report\n")
            f.write("=" * 60 + "\n")
            f.write(f"Generated: {datetime.now().isoformat()}\n")
            f.write(f"Total Tests: {total_tests}\n")
            f.write(f"Passed: {passed_tests}\n")
            f.write(f"Failed: {total_tests - passed_tests}\n")
            f.write(f"Success Rate: {(passed_tests / total_tests * 100):.1f}%\n")
            f.write(f"Total Duration: {total_duration:.2f}s\n")
            f.write(f"Average Duration: {(total_duration / total_tests):.2f}s per test\n\n")
            
            f.write("DETAILED RESULTS:\n")
            f.write("-" * 40 + "\n")
            
            for result in results:
                status = "PASS" if result['success'] else "FAIL"
                duration_str = f"{result['duration']:.2f}s" if result['duration'] > 0 else "N/A"
                f.write(f"{status} {result['name']} ({duration_str})\n")
                f.write(f"Output: {result['output'][:200]}...\n\n")
        
        print(f"\n📄 Comprehensive report saved to: {report_file}")
    except Exception as e:
        print(f"\n⚠️ Failed to save comprehensive report: {e}")
    
    # Final status
    if passed_tests == total_tests:
        print(f"\n🎉 All tests passed successfully!")
        return 0
    else:
        print(f"\n⚠️ {total_tests - passed_tests} test(s) failed. Check individual output files for details.")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code) 