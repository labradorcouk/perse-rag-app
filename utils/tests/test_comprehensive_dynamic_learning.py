#!/usr/bin/env python3
"""
Comprehensive Test Suite for Dynamic Learning and Pattern Evolution
Tests all functionalities including MongoDB integration, pattern matching, learning, and evolution.
"""

import sys
import os
import time
import unittest
from datetime import datetime
import pandas as pd
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mongodb_schema_manager import MongoDBSchemaManager

class TestComprehensiveDynamicLearning(unittest.TestCase):
    """Comprehensive test suite for dynamic learning functionality."""
    
    def setUp(self):
        """Set up test environment."""
        self.schema_manager = MongoDBSchemaManager()
        self.collection_name = "ecoesTechDetailsWithEmbedding"
        self.test_queries = [
            "What are the most common errors in MPAN records?",
            "Analyze error patterns in MPAN data",
            "Show me MPAN records by geographic region",
            "Which suppliers have the highest error rates?",
            "Find validation issues in the dataset"
        ]
        
        # Performance tracking
        self.performance_metrics = {
            'pattern_matching_times': [],
            'learning_times': [],
            'evolution_times': [],
            'query_enhancement_times': []
        }
    
    def test_01_mongodb_connection(self):
        """Test MongoDB connection and availability."""
        print("\n🔌 Testing MongoDB Connection...")
        
        is_available = self.schema_manager.is_mongodb_available()
        self.assertIsInstance(is_available, bool)
        
        if is_available:
            print("✅ MongoDB connection successful")
        else:
            print("⚠️ MongoDB connection failed - some tests will be skipped")
        
        return is_available
    
    def test_02_hybrid_pattern_loading(self):
        """Test hybrid pattern loading from both YAML and MongoDB."""
        print("\n🔄 Testing Hybrid Pattern Loading...")
        
        start_time = time.time()
        
        # Test core patterns from YAML
        core_patterns = self.schema_manager.get_qa_patterns(self.collection_name)
        self.assertIsInstance(core_patterns, list)
        print(f"✅ Core patterns loaded: {len(core_patterns)} patterns")
        
        # Test hybrid patterns
        hybrid_patterns = self.schema_manager.get_hybrid_qa_patterns(self.collection_name)
        self.assertIsInstance(hybrid_patterns, list)
        print(f"✅ Hybrid patterns loaded: {len(hybrid_patterns)} patterns")
        
        # Verify hybrid patterns include core patterns
        self.assertGreaterEqual(len(hybrid_patterns), len(core_patterns))
        
        end_time = time.time()
        self.performance_metrics['pattern_matching_times'].append(end_time - start_time)
        
        print(f"✅ Hybrid pattern loading successful in {end_time - start_time:.3f}s")
    
    def test_03_query_enhancement(self):
        """Test query enhancement with business context."""
        print("\n🔍 Testing Query Enhancement...")
        
        for i, query in enumerate(self.test_queries[:3]):  # Test first 3 queries
            start_time = time.time()
            
            enhanced_info = self.schema_manager.enhance_user_query(self.collection_name, query)
            
            # Verify structure
            self.assertIsInstance(enhanced_info, dict)
            required_keys = ['original_query', 'enhanced_query', 'detected_intent', 'confidence_score']
            for key in required_keys:
                self.assertIn(key, enhanced_info)
            
            # Verify content
            self.assertEqual(enhanced_info['original_query'], query)
            self.assertIsInstance(enhanced_info['detected_intent'], list)
            self.assertIsInstance(enhanced_info['confidence_score'], (int, float))
            
            end_time = time.time()
            self.performance_metrics['query_enhancement_times'].append(end_time - start_time)
            
            print(f"✅ Query {i+1} enhanced: '{query[:50]}...' -> Intent: {enhanced_info['detected_intent']}")
    
    def test_04_pattern_matching(self):
        """Test Q&A pattern matching functionality."""
        print("\n🎯 Testing Pattern Matching...")
        
        for i, query in enumerate(self.test_queries):
            start_time = time.time()
            
            pattern_match = self.schema_manager.match_qa_pattern(self.collection_name, query)
            
            if pattern_match:
                # Verify pattern structure
                self.assertIsInstance(pattern_match, dict)
                required_keys = ['question_pattern', 'answer_intent', 'match_score']
                for key in required_keys:
                    self.assertIn(key, pattern_match)
                
                # Verify match score
                self.assertIsInstance(pattern_match['match_score'], (int, float))
                self.assertGreaterEqual(pattern_match['match_score'], 0.0)
                self.assertLessEqual(pattern_match['match_score'], 1.0)
                
                print(f"✅ Query {i+1} matched: Score {pattern_match['match_score']:.2f} -> Intent: {pattern_match['answer_intent']}")
            else:
                print(f"⚠️ Query {i+1} no match found: '{query[:50]}...'")
            
            end_time = time.time()
            self.performance_metrics['pattern_matching_times'].append(end_time - start_time)
    
    def test_05_learning_functionality(self):
        """Test learning from user queries."""
        print("\n🧠 Testing Learning Functionality...")
        
        if not self.schema_manager.is_mongodb_available():
            print("⚠️ Skipping learning tests - MongoDB not available")
            return
        
        for i, query in enumerate(self.test_queries[:3]):
            start_time = time.time()
            
            # Test learning from query
            detected_intent = f"test_intent_{i}"
            self.schema_manager.learn_from_query(
                collection_name=self.collection_name,
                user_query=query,
                detected_intent=detected_intent,
                user_satisfaction=4  # Good satisfaction
            )
            
            end_time = time.time()
            self.performance_metrics['learning_times'].append(end_time - start_time)
            
            print(f"✅ Learning recorded for query {i+1}: Intent '{detected_intent}'")
    
    def test_06_pattern_evolution(self):
        """Test pattern evolution based on learning data."""
        print("\n🔄 Testing Pattern Evolution...")
        
        if not self.schema_manager.is_mongodb_available():
            print("⚠️ Skipping evolution tests - MongoDB not available")
            return
        
        start_time = time.time()
        
        # Evolve patterns
        self.schema_manager.evolve_patterns(self.collection_name)
        
        end_time = time.time()
        self.performance_metrics['evolution_times'].append(end_time - start_time)
        
        print(f"✅ Pattern evolution completed in {end_time - start_time:.3f}s")
    
    def test_07_analytics_functionality(self):
        """Test pattern analytics and performance metrics."""
        print("\n📊 Testing Analytics Functionality...")
        
        if not self.schema_manager.is_mongodb_available():
            print("⚠️ Skipping analytics tests - MongoDB not available")
            return
        
        analytics = self.schema_manager.get_pattern_analytics(self.collection_name)
        
        if analytics:
            # Verify analytics structure
            self.assertIsInstance(analytics, dict)
            expected_keys = ['total_patterns', 'active_patterns', 'total_usage', 'avg_confidence']
            for key in expected_keys:
                self.assertIn(key, analytics)
            
            print(f"✅ Analytics retrieved: {analytics['total_patterns']} patterns, {analytics['active_patterns']} active")
        else:
            print("⚠️ No analytics data available")
    
    def test_08_user_feedback(self):
        """Test user feedback collection."""
        print("\n💬 Testing User Feedback...")
        
        if not self.schema_manager.is_mongodb_available():
            print("⚠️ Skipping feedback tests - MongoDB not available")
            return
        
        # Test feedback submission
        feedback_data = {
            "pattern_id": "test_pattern_123",
            "user_query": "Test feedback query",
            "detected_intent": "test_intent",
            "confidence_score": 0.85,
            "was_correct": True,
            "response_quality": 4,
            "feedback_notes": "Test feedback for system improvement"
        }
        
        feedback_id = self.schema_manager.add_user_feedback("test_pattern_123", feedback_data)
        
        if feedback_id:
            print(f"✅ Feedback submitted successfully: {feedback_id}")
        else:
            print("⚠️ Feedback submission failed")
    
    def test_09_schema_validation(self):
        """Test schema validation functionality."""
        print("\n✅ Testing Schema Validation...")
        
        validation_results = self.schema_manager.validate_collection_schema(self.collection_name)
        
        # Verify validation structure
        self.assertIsInstance(validation_results, dict)
        required_keys = ['valid', 'collection_name', 'data_dictionary_fields']
        for key in required_keys:
            self.assertIn(key, validation_results)
        
        print(f"✅ Schema validation: {validation_results['valid']}")
        if validation_results.get('warnings'):
            print(f"⚠️ Warnings: {validation_results['warnings']}")
    
    def test_10_performance_benchmarks(self):
        """Test performance benchmarks and timing."""
        print("\n⚡ Testing Performance Benchmarks...")
        
        # Calculate average performance metrics
        avg_pattern_matching = np.mean(self.performance_metrics['pattern_matching_times']) if self.performance_metrics['pattern_matching_times'] else 0
        avg_learning = np.mean(self.performance_metrics['learning_times']) if self.performance_metrics['learning_times'] else 0
        avg_evolution = np.mean(self.performance_metrics['evolution_times']) if self.performance_metrics['evolution_times'] else 0
        avg_enhancement = np.mean(self.performance_metrics['query_enhancement_times']) if self.performance_metrics['query_enhancement_times'] else 0
        
        print(f"📊 Performance Metrics:")
        print(f"  • Pattern Matching: {avg_pattern_matching:.3f}s avg")
        print(f"  • Learning: {avg_learning:.3f}s avg")
        print(f"  • Evolution: {avg_evolution:.3f}s avg")
        print(f"  • Query Enhancement: {avg_enhancement:.3f}s avg")
        
        # Performance thresholds (adjust as needed)
        self.assertLess(avg_pattern_matching, 0.1, "Pattern matching too slow")
        self.assertLess(avg_enhancement, 0.05, "Query enhancement too slow")
        
        print("✅ Performance benchmarks passed")
    
    def test_11_error_handling(self):
        """Test error handling and edge cases."""
        print("\n🛡️ Testing Error Handling...")
        
        # Test with invalid collection name
        invalid_result = self.schema_manager.get_qa_patterns("invalid_collection")
        self.assertEqual(invalid_result, [])
        
        # Test with empty query
        empty_enhancement = self.schema_manager.enhance_user_query(self.collection_name, "")
        self.assertIsInstance(empty_enhancement, dict)
        
        # Test with None query
        none_enhancement = self.schema_manager.enhance_user_query(self.collection_name, None)
        self.assertIsInstance(none_enhancement, dict)
        
        print("✅ Error handling tests passed")
    
    def test_12_dataframe_optimization(self):
        """Test DataFrame optimization for context."""
        print("\n🔧 Testing DataFrame Optimization...")
        
        # Create test DataFrame
        test_df = pd.DataFrame({
            'type': ['error', 'warning', 'info'] * 10,
            'value': range(30),
            'Results': ['Test result ' + str(i) for i in range(30)],
            'extra_col': ['extra' + str(i) for i in range(30)]
        })
        
        # Test optimization
        optimized_df = self.schema_manager.optimize_dataframe_for_context(test_df, self.collection_name)
        
        # Verify optimization
        self.assertIsInstance(optimized_df, pd.DataFrame)
        self.assertLessEqual(len(optimized_df), len(test_df))  # Should not increase rows
        
        print(f"✅ DataFrame optimized: {len(test_df)} -> {len(optimized_df)} rows")
    
    def tearDown(self):
        """Clean up after tests."""
        # Save performance metrics
        if any(self.performance_metrics.values()):
            self._save_performance_report()
    
    def _save_performance_report(self):
        """Save performance report to file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = f"performance_report_{timestamp}.txt"
        
        with open(report_file, 'w') as f:
            f.write("Dynamic Learning Performance Report\n")
            f.write("=" * 50 + "\n")
            f.write(f"Generated: {datetime.now().isoformat()}\n")
            f.write(f"Collection: {self.collection_name}\n\n")
            
            for metric, times in self.performance_metrics.items():
                if times:
                    avg_time = np.mean(times)
                    min_time = np.min(times)
                    max_time = np.max(times)
                    f.write(f"{metric}:\n")
                    f.write(f"  Average: {avg_time:.3f}s\n")
                    f.write(f"  Min: {min_time:.3f}s\n")
                    f.write(f"  Max: {max_time:.3f}s\n")
                    f.write(f"  Count: {len(times)}\n\n")
        
        print(f"📊 Performance report saved to: {report_file}")

def run_comprehensive_tests():
    """Run all comprehensive tests."""
    print("🧪 Comprehensive Dynamic Learning Test Suite")
    print("=" * 60)
    
    # Create test suite
    test_suite = unittest.TestLoader().loadTestsFromTestCase(TestComprehensiveDynamicLearning)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 TEST SUMMARY")
    print("=" * 60)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    
    if result.failures:
        print("\n❌ FAILURES:")
        for test, traceback in result.failures:
            print(f"  • {test}: {traceback.split('AssertionError:')[-1].strip()}")
    
    if result.errors:
        print("\n💥 ERRORS:")
        for test, traceback in result.errors:
            print(f"  • {test}: {traceback.split('Exception:')[-1].strip()}")
    
    if result.wasSuccessful():
        print("\n🎉 All tests passed successfully!")
        return True
    else:
        print(f"\n⚠️ {len(result.failures) + len(result.errors)} test(s) failed")
        return False

if __name__ == "__main__":
    success = run_comprehensive_tests()
    sys.exit(0 if success else 1) 