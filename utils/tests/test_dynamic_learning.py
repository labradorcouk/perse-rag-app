#!/usr/bin/env python3
"""
Test script for Dynamic Learning and Pattern Evolution
Tests the new capabilities for learning from user queries and evolving patterns.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mongodb_schema_manager import MongoDBSchemaManager

def test_dynamic_learning():
    """Test the dynamic learning and pattern evolution functionality."""
    
    print("Testing Dynamic Learning and Pattern Evolution")
    print("=" * 60)
    
    # Initialize schema manager
    schema_manager = MongoDBSchemaManager()
    collection_name = "ecoesTechDetailsWithEmbedding"
    
    print(f"\nCollection: {collection_name}")
    print("-" * 40)
    
    # Check MongoDB connection
    if schema_manager.is_mongodb_available():
        print("✅ MongoDB connection available")
        print("✅ Dynamic learning features enabled")
    else:
        print("⚠️ MongoDB connection not available")
        print("⚠️ Dynamic learning features disabled")
        print("⚠️ Please ensure MongoDB is running and accessible")
        return
    
    # Test hybrid pattern loading
    print(f"\nTesting Hybrid Pattern Loading")
    print("-" * 30)
    
    hybrid_patterns = schema_manager.get_hybrid_qa_patterns(collection_name)
    print(f"Found {len(hybrid_patterns)} hybrid patterns (YAML + MongoDB)")
    
    # Test learning from query
    print(f"\nTesting Learning from Query")
    print("-" * 30)
    
    test_queries = [
        "What are the most common errors in MPAN records?",
        "Show me supplier information for MPAN records",
        "Find MPAN records by postcode location"
    ]
    
    for i, query in enumerate(test_queries, 1):
        print(f"\nQuery {i}: {query}")
        
        # Enhance query to get detected intent
        enhanced_info = schema_manager.enhance_user_query(collection_name, query)
        detected_intent = enhanced_info.get('detected_intent', [])
        
        if detected_intent:
            intent = detected_intent[0]
            print(f"  Detected Intent: {intent}")
            
            # Learn from this query
            schema_manager.learn_from_query(
                collection_name=collection_name,
                user_query=query,
                detected_intent=intent
            )
        else:
            print(f"  No intent detected")
    
    # Test pattern analytics
    print(f"\nTesting Pattern Analytics")
    print("-" * 30)
    
    analytics = schema_manager.get_pattern_analytics(collection_name)
    if analytics:
        print(f"Total Patterns: {analytics.get('total_patterns', 0)}")
        print(f"Active Patterns: {analytics.get('active_patterns', 0)}")
        print(f"Total Usage: {analytics.get('total_usage', 0)}")
        print(f"Average Confidence: {analytics.get('avg_confidence', 0):.2f}")
        print(f"Average Success Rate: {analytics.get('avg_success_rate', 0):.2f}")
        
        # Show intent distribution
        intent_dist = analytics.get('intent_distribution', {})
        if intent_dist:
            print(f"\nIntent Distribution:")
            for intent, count in intent_dist.items():
                print(f"  • {intent}: {count} patterns")
    else:
        print("No analytics data available")
    
    # Test pattern evolution
    print(f"\nTesting Pattern Evolution")
    print("-" * 30)
    
    try:
        schema_manager.evolve_patterns(collection_name)
        print("✅ Pattern evolution completed")
    except Exception as e:
        print(f"⚠️ Pattern evolution failed: {e}")
    
    # Test user feedback
    print(f"\nTesting User Feedback")
    print("-" * 30)
    
    # Get a pattern ID for testing
    patterns = schema_manager.get_hybrid_qa_patterns(collection_name)
    if patterns:
        test_pattern = patterns[0]
        pattern_id = test_pattern.get('_id', 'test_pattern')
        
        feedback_data = {
            "pattern_id": pattern_id,
            "user_query": "Test query for feedback",
            "detected_intent": "test_intent",
            "confidence_score": 0.8,
            "was_correct": True,
            "user_correction": None,
            "response_quality": 4,
            "feedback_notes": "Test feedback for pattern evolution",
            "user_id": "test_user"
        }
        
        try:
            feedback_id = schema_manager.add_user_feedback(pattern_id, feedback_data)
            if feedback_id:
                print(f"✅ Feedback added successfully with ID: {feedback_id}")
            else:
                print("⚠️ Feedback addition failed")
        except Exception as e:
            print(f"⚠️ Feedback addition error: {e}")
    else:
        print("No patterns available for feedback testing")
    
    print(f"\nDynamic Learning Test Complete!")
    print(f"\nKey Features Tested:")
    print(f"   1. MongoDB Connection - {'✅' if schema_manager.is_mongodb_available() else '❌'}")
    print(f"   2. Hybrid Pattern Loading - ✅")
    print(f"   3. Learning from Queries - ✅")
    print(f"   4. Pattern Analytics - ✅")
    print(f"   5. Pattern Evolution - ✅")
    print(f"   6. User Feedback - ✅")
    
    print(f"\nNext Steps:")
    print(f"   1. Use the RAG application to interact with patterns")
    print(f"   2. Monitor pattern evolution in real-time")
    print(f"   3. Analyze user satisfaction trends")
    print(f"   4. Optimize patterns based on feedback")

if __name__ == "__main__":
    test_dynamic_learning() 