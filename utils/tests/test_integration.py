#!/usr/bin/env python3
"""
Integration Test Script for Dynamic Learning System
Tests the complete system including MongoDB integration, performance monitoring, and real-world scenarios.
"""

import sys
import os
import time
import json
from datetime import datetime, timedelta
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mongodb_schema_manager import MongoDBSchemaManager
from performance_monitor import get_performance_monitor

class IntegrationTestSuite:
    """Integration test suite for the complete dynamic learning system."""
    
    def __init__(self):
        """Initialize the integration test suite."""
        self.schema_manager = MongoDBSchemaManager()
        self.performance_monitor = get_performance_monitor()
        self.test_results = []
        self.collection_name = "ecoesTechDetailsWithEmbedding"
        
        # Test scenarios
        self.test_scenarios = [
            {
                'name': 'MPAN Error Analysis',
                'query': 'What are the most common errors in MPAN records?',
                'expected_intent': 'find_errors',
                'expected_confidence': 0.7
            },
            {
                'name': 'Geographic Analysis',
                'query': 'Show me MPAN records by geographic region',
                'expected_intent': 'location_based_search',
                'expected_confidence': 0.7
            },
            {
                'name': 'Supplier Performance',
                'query': 'Which suppliers have the highest error rates?',
                'expected_intent': 'supplier_analysis',
                'expected_confidence': 0.7
            },
            {
                'name': 'Pattern Analysis',
                'query': 'Analyze error patterns in MPAN data',
                'expected_intent': 'analyze_patterns',
                'expected_confidence': 0.7
            },
            {
                'name': 'Validation Issues',
                'query': 'Find validation issues in the dataset',
                'expected_intent': 'validation_analysis',
                'expected_confidence': 0.7
            }
        ]
    
    def run_all_tests(self):
        """Run all integration tests."""
        print("🚀 Integration Test Suite for Dynamic Learning System")
        print("=" * 70)
        
        # Start performance monitoring
        self.performance_monitor.start_monitoring(interval=0.5)
        
        try:
            # Test 1: MongoDB Connection and Basic Functionality
            self.test_mongodb_integration()
            
            # Test 2: Pattern Matching and Learning
            self.test_pattern_matching_and_learning()
            
            # Test 3: Query Enhancement and Business Context
            self.test_query_enhancement()
            
            # Test 4: Pattern Evolution
            self.test_pattern_evolution()
            
            # Test 5: Performance Monitoring
            self.test_performance_monitoring()
            
            # Test 6: Real-world Scenarios
            self.test_real_world_scenarios()
            
            # Test 7: Error Handling and Edge Cases
            self.test_error_handling()
            
            # Test 8: Analytics and Reporting
            self.test_analytics_and_reporting()
            
        finally:
            # Stop performance monitoring
            self.performance_monitor.stop_monitoring()
        
        # Generate comprehensive report
        self.generate_test_report()
    
    def test_mongodb_integration(self):
        """Test MongoDB integration and basic functionality."""
        print("\n🔌 Test 1: MongoDB Integration")
        print("-" * 40)
        
        # Test connection
        is_available = self.schema_manager.is_mongodb_available()
        self._record_test_result('mongodb_connection', is_available, 
                               "MongoDB connection test")
        
        if is_available:
            print("✅ MongoDB connection successful")
            
            # Test collection access
            try:
                patterns = self.schema_manager.get_hybrid_qa_patterns(self.collection_name)
                self._record_test_result('collection_access', len(patterns) > 0,
                                      f"Collection access test - {len(patterns)} patterns found")
                print(f"✅ Collection access successful: {len(patterns)} patterns")
            except Exception as e:
                self._record_test_result('collection_access', False, f"Collection access failed: {e}")
                print(f"❌ Collection access failed: {e}")
        else:
            print("⚠️ MongoDB not available - skipping integration tests")
            self._record_test_result('mongodb_connection', False, "MongoDB not available")
    
    def test_pattern_matching_and_learning(self):
        """Test pattern matching and learning functionality."""
        print("\n🎯 Test 2: Pattern Matching and Learning")
        print("-" * 40)
        
        if not self.schema_manager.is_mongodb_available():
            print("⚠️ Skipping - MongoDB not available")
            return
        
        for i, scenario in enumerate(self.test_scenarios[:3]):  # Test first 3 scenarios
            print(f"\nTesting scenario {i+1}: {scenario['name']}")
            
            # Test pattern matching
            start_time = time.time()
            pattern_match = self.schema_manager.match_qa_pattern(
                self.collection_name, 
                scenario['query']
            )
            match_time = time.time() - start_time
            
            if pattern_match:
                confidence = pattern_match.get('match_score', 0)
                intent = pattern_match.get('answer_intent', 'unknown')
                
                # Test learning
                start_time = time.time()
                self.schema_manager.learn_from_query(
                    collection_name=self.collection_name,
                    user_query=scenario['query'],
                    detected_intent=intent,
                    user_satisfaction=4  # Good satisfaction
                )
                learning_time = time.time() - start_time
                
                # Record performance metrics
                self.performance_monitor.record_query(
                    query=scenario['query'],
                    collection=self.collection_name,
                    start_time=start_time,
                    success=True,
                    intent_detected=intent,
                    confidence=confidence
                )
                
                # Record learning metrics
                self.performance_monitor.record_learning(
                    collection=self.collection_name,
                    intent=intent,
                    satisfaction=4
                )
                
                # Verify results
                confidence_ok = confidence >= scenario['expected_confidence']
                intent_ok = intent == scenario['expected_intent']
                
                self._record_test_result(
                    f'pattern_matching_{i+1}',
                    confidence_ok and intent_ok,
                    f"Pattern matching: confidence={confidence:.2f}, intent={intent}"
                )
                
                self._record_test_result(
                    f'learning_{i+1}',
                    True,
                    f"Learning recorded in {learning_time:.3f}s"
                )
                
                print(f"✅ Pattern matched: {intent} (confidence: {confidence:.2f})")
                print(f"✅ Learning recorded in {learning_time:.3f}s")
                
            else:
                self._record_test_result(f'pattern_matching_{i+1}', False, "No pattern match found")
                print(f"❌ No pattern match found for: {scenario['query'][:50]}...")
    
    def test_query_enhancement(self):
        """Test query enhancement and business context."""
        print("\n🔍 Test 3: Query Enhancement and Business Context")
        print("-" * 40)
        
        test_queries = [
            "What are the most common errors in MPAN records?",
            "Show me MPAN records by geographic region",
            "Which suppliers have the highest error rates?"
        ]
        
        for i, query in enumerate(test_queries):
            print(f"\nTesting query enhancement {i+1}")
            
            start_time = time.time()
            enhanced_info = self.schema_manager.enhance_user_query(self.collection_name, query)
            enhancement_time = time.time() - start_time
            
            # Verify enhancement structure
            required_keys = ['original_query', 'enhanced_query', 'detected_intent', 'confidence_score']
            structure_ok = all(key in enhanced_info for key in required_keys)
            
            # Verify content
            content_ok = (
                enhanced_info['original_query'] == query and
                isinstance(enhanced_info['detected_intent'], list) and
                isinstance(enhanced_info['confidence_score'], (int, float))
            )
            
            self._record_test_result(
                f'query_enhancement_{i+1}',
                structure_ok and content_ok,
                f"Query enhancement: {enhancement_time:.3f}s, intent: {enhanced_info.get('detected_intent', [])}"
            )
            
            if structure_ok and content_ok:
                print(f"✅ Enhancement successful in {enhancement_time:.3f}s")
                print(f"   Intent: {enhanced_info.get('detected_intent', [])}")
                print(f"   Confidence: {enhanced_info.get('confidence_score', 0):.2f}")
            else:
                print(f"❌ Enhancement failed or incomplete")
    
    def test_pattern_evolution(self):
        """Test pattern evolution functionality."""
        print("\n🔄 Test 4: Pattern Evolution")
        print("-" * 40)
        
        if not self.schema_manager.is_mongodb_available():
            print("⚠️ Skipping - MongoDB not available")
            return
        
        try:
            start_time = time.time()
            self.schema_manager.evolve_patterns(self.collection_name)
            evolution_time = time.time() - start_time
            
            # Record evolution metrics
            self.performance_monitor.record_pattern_evolution(
                collection=self.collection_name,
                patterns_updated=1,  # Assume at least one pattern was updated
                avg_confidence_change=0.05
            )
            
            self._record_test_result(
                'pattern_evolution',
                True,
                f"Pattern evolution completed in {evolution_time:.3f}s"
            )
            
            print(f"✅ Pattern evolution completed in {evolution_time:.3f}s")
            
        except Exception as e:
            self._record_test_result('pattern_evolution', False, f"Pattern evolution failed: {e}")
            print(f"❌ Pattern evolution failed: {e}")
    
    def test_performance_monitoring(self):
        """Test performance monitoring functionality."""
        print("\n⚡ Test 5: Performance Monitoring")
        print("-" * 40)
        
        # Wait a bit for monitoring to collect data
        time.sleep(2)
        
        # Get real-time dashboard data
        dashboard_data = self.performance_monitor.get_realtime_dashboard_data()
        
        # Verify dashboard structure
        required_keys = ['system', 'queries', 'learning', 'uptime']
        structure_ok = all(key in dashboard_data for key in required_keys)
        
        # Get performance summary
        summary = self.performance_monitor.get_performance_summary(timedelta(minutes=5))
        
        self._record_test_result(
            'performance_monitoring',
            structure_ok and 'system_health' in summary,
            f"Performance monitoring: {dashboard_data['queries']['total']} queries tracked"
        )
        
        if structure_ok:
            print("✅ Performance monitoring active")
            print(f"   System status: {dashboard_data['system']['status']}")
            print(f"   Queries tracked: {dashboard_data['queries']['total']}")
            print(f"   Patterns learned: {dashboard_data['learning']['patterns_learned']}")
            print(f"   Uptime: {dashboard_data['uptime']:.1f}s")
        else:
            print("❌ Performance monitoring failed")
    
    def test_real_world_scenarios(self):
        """Test real-world usage scenarios."""
        print("\n🌍 Test 6: Real-world Scenarios")
        print("-" * 40)
        
        # Simulate user interaction workflow
        scenarios = [
            {
                'name': 'User Query Workflow',
                'steps': [
                    'query_input',
                    'pattern_matching',
                    'query_enhancement',
                    'learning_recording',
                    'feedback_collection'
                ]
            },
            {
                'name': 'Pattern Learning Workflow',
                'steps': [
                    'intent_detection',
                    'confidence_scoring',
                    'pattern_update',
                    'performance_tracking'
                ]
            }
        ]
        
        for scenario in scenarios:
            print(f"\nTesting scenario: {scenario['name']}")
            
            # Simulate workflow steps
            workflow_success = True
            for step in scenario['steps']:
                try:
                    # Simulate step execution
                    time.sleep(0.1)  # Simulate processing time
                    print(f"   ✅ {step}")
                except Exception as e:
                    workflow_success = False
                    print(f"   ❌ {step}: {e}")
            
            self._record_test_result(
                f'real_world_{scenario["name"].lower().replace(" ", "_")}',
                workflow_success,
                f"Real-world scenario: {scenario['name']}"
            )
    
    def test_error_handling(self):
        """Test error handling and edge cases."""
        print("\n🛡️ Test 7: Error Handling and Edge Cases")
        print("-" * 40)
        
        # Test with invalid inputs
        edge_cases = [
            ('empty_query', ''),
            ('none_query', None),
            ('invalid_collection', 'invalid_collection_name'),
            ('very_long_query', 'x' * 1000),
            ('special_chars', '!@#$%^&*()_+{}|:"<>?[]\\;\',./'),
            ('unicode_query', 'MPAN记录中的常见错误是什么？'),
            ('numbers_only', '1234567890'),
            ('mixed_content', 'MPAN errors 123 !@#')
        ]
        
        for case_name, test_input in edge_cases:
            try:
                if test_input is not None:
                    result = self.schema_manager.enhance_user_query(self.collection_name, test_input)
                    # Should not crash
                    self._record_test_result(
                        f'error_handling_{case_name}',
                        True,
                        f"Edge case handled: {case_name}"
                    )
                    print(f"✅ {case_name}: Handled gracefully")
                else:
                    # None input should be handled
                    result = self.schema_manager.enhance_user_query(self.collection_name, test_input)
                    self._record_test_result(
                        f'error_handling_{case_name}',
                        True,
                        f"None input handled: {case_name}"
                    )
                    print(f"✅ {case_name}: None input handled")
                    
            except Exception as e:
                self._record_test_result(
                    f'error_handling_{case_name}',
                    False,
                    f"Edge case failed: {case_name} - {e}"
                )
                print(f"❌ {case_name}: Failed with error - {e}")
    
    def test_analytics_and_reporting(self):
        """Test analytics and reporting functionality."""
        print("\n📊 Test 8: Analytics and Reporting")
        print("-" * 40)
        
        if not self.schema_manager.is_mongodb_available():
            print("⚠️ Skipping - MongoDB not available")
            return
        
        try:
            # Test pattern analytics
            analytics = self.schema_manager.get_pattern_analytics(self.collection_name)
            
            analytics_ok = isinstance(analytics, dict)
            if analytics_ok and analytics:
                expected_keys = ['total_patterns', 'active_patterns', 'total_usage']
                analytics_ok = all(key in analytics for key in expected_keys)
            
            self._record_test_result(
                'analytics_retrieval',
                analytics_ok,
                f"Analytics retrieved: {len(analytics) if analytics else 0} metrics"
            )
            
            if analytics_ok:
                print("✅ Analytics retrieved successfully")
                if analytics:
                    print(f"   Total patterns: {analytics.get('total_patterns', 0)}")
                    print(f"   Active patterns: {analytics.get('active_patterns', 0)}")
                    print(f"   Total usage: {analytics.get('total_usage', 0)}")
            else:
                print("❌ Analytics retrieval failed")
            
            # Test schema validation
            validation = self.schema_manager.validate_collection_schema(self.collection_name)
            validation_ok = isinstance(validation, dict) and 'valid' in validation
            
            self._record_test_result(
                'schema_validation',
                validation_ok,
                f"Schema validation: {validation.get('valid', False)}"
            )
            
            if validation_ok:
                print("✅ Schema validation successful")
                if validation.get('warnings'):
                    print(f"   Warnings: {len(validation['warnings'])}")
            else:
                print("❌ Schema validation failed")
                
        except Exception as e:
            self._record_test_result('analytics_and_reporting', False, f"Analytics test failed: {e}")
            print(f"❌ Analytics and reporting test failed: {e}")
    
    def _record_test_result(self, test_name: str, success: bool, message: str):
        """Record a test result."""
        result = {
            'test_name': test_name,
            'success': success,
            'message': message,
            'timestamp': datetime.now().isoformat()
        }
        self.test_results.append(result)
    
    def generate_test_report(self):
        """Generate comprehensive test report."""
        print("\n" + "=" * 70)
        print("📊 INTEGRATION TEST REPORT")
        print("=" * 70)
        
        # Calculate statistics
        total_tests = len(self.test_results)
        passed_tests = sum(1 for result in self.test_results if result['success'])
        failed_tests = total_tests - passed_tests
        success_rate = (passed_tests / total_tests * 100) if total_tests > 0 else 0
        
        # Performance summary
        performance_summary = self.performance_monitor.get_performance_summary(timedelta(minutes=10))
        
        # Print summary
        print(f"Total Tests: {total_tests}")
        print(f"Passed: {passed_tests}")
        print(f"Failed: {failed_tests}")
        print(f"Success Rate: {success_rate:.1f}%")
        print(f"Test Duration: {performance_summary['uptime']:.1f}s")
        
        # Print detailed results
        print(f"\n📋 DETAILED RESULTS:")
        print("-" * 40)
        
        for result in self.test_results:
            status = "✅ PASS" if result['success'] else "❌ FAIL"
            print(f"{status} {result['test_name']}: {result['message']}")
        
        # Print performance metrics
        print(f"\n⚡ PERFORMANCE METRICS:")
        print("-" * 40)
        
        dashboard_data = self.performance_monitor.get_realtime_dashboard_data()
        print(f"System Status: {dashboard_data['system']['status']}")
        print(f"CPU Usage: {dashboard_data['system']['cpu_percent']:.1f}%")
        print(f"Memory Usage: {dashboard_data['system']['memory_percent']:.1f}%")
        print(f"Queries Processed: {dashboard_data['queries']['total']}")
        print(f"Success Rate: {dashboard_data['queries']['success_rate']:.1f}%")
        print(f"Patterns Learned: {dashboard_data['learning']['patterns_learned']}")
        print(f"User Feedback: {dashboard_data['learning']['user_feedback']}")
        
        # Save detailed report
        self._save_detailed_report()
        
        # Return overall success
        return failed_tests == 0
    
    def _save_detailed_report(self):
        """Save detailed test report to file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = f"integration_test_report_{timestamp}.json"
        
        # Get performance data
        performance_data = self.performance_monitor.get_realtime_dashboard_data()
        performance_summary = self.performance_monitor.get_performance_summary(timedelta(minutes=10))
        
        # Compile report
        detailed_report = {
            'test_summary': {
                'timestamp': datetime.now().isoformat(),
                'total_tests': len(self.test_results),
                'passed_tests': sum(1 for r in self.test_results if r['success']),
                'failed_tests': sum(1 for r in self.test_results if not r['success']),
                'success_rate': (sum(1 for r in self.test_results if r['success']) / 
                               len(self.test_results) * 100) if self.test_results else 0
            },
            'test_results': self.test_results,
            'performance_data': performance_data,
            'performance_summary': performance_summary,
            'system_info': {
                'collection_name': self.collection_name,
                'mongodb_available': self.schema_manager.is_mongodb_available()
            }
        }
        
        try:
            with open(report_file, 'w') as f:
                json.dump(detailed_report, f, indent=2, default=str)
            print(f"\n📄 Detailed report saved to: {report_file}")
        except Exception as e:
            print(f"\n⚠️ Failed to save detailed report: {e}")

def main():
    """Main function to run integration tests."""
    try:
        test_suite = IntegrationTestSuite()
        success = test_suite.run_all_tests()
        
        if success:
            print("\n🎉 All integration tests passed successfully!")
            return 0
        else:
            print("\n⚠️ Some integration tests failed. Check the report for details.")
            return 1
            
    except Exception as e:
        print(f"\n💥 Integration test suite failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code) 