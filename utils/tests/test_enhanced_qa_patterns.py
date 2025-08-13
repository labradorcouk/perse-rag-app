#!/usr/bin/env python3
"""
Test script for Enhanced Q&A Pattern Matching and Intent Detection
Demonstrates the new capabilities for understanding user queries and matching them to patterns.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mongodb_schema_manager import MongoDBSchemaManager
import pandas as pd

def test_enhanced_qa_patterns():
    """Test the enhanced Q&A pattern matching and intent detection functionality."""
    
    print("Testing Enhanced Q&A Pattern Matching and Intent Detection")
    print("=" * 70)
    
    # Initialize schema manager
    schema_manager = MongoDBSchemaManager()
    collection_name = "ecoesTechDetailsWithEmbedding"
    
    print(f"\nCollection: {collection_name}")
    print("-" * 40)
    
    # Test Q&A patterns retrieval
    qa_patterns = schema_manager.get_qa_patterns(collection_name)
    print(f"Found {len(qa_patterns)} Q&A patterns")
    
    # Test intent categories
    intent_categories = schema_manager.get_intent_categories(collection_name)
    print(f"Found {len(intent_categories)} intent categories")
    
    # Display Q&A patterns
    print(f"\nQ&A Patterns:")
    print("-" * 20)
    for i, pattern in enumerate(qa_patterns, 1):
        print(f"{i}. Question: {pattern.get('question_pattern', 'N/A')}")
        print(f"   Intent: {pattern.get('answer_intent', 'N/A')}")
        print(f"   Business Entities: {pattern.get('business_entities', [])}")
        print(f"   Expected Columns: {pattern.get('expected_columns', [])}")
        print(f"   Search Strategy: {pattern.get('search_strategy', 'N/A')}")
        print(f"   Sample Queries: {len(pattern.get('sample_queries', []))} examples")
        print()
    
    # Display intent categories
    print(f"Intent Categories:")
    print("-" * 20)
    for intent_name, intent_info in intent_categories.items():
        print(f"Intent: {intent_name}")
        print(f"  Description: {intent_info.get('description', 'N/A')}")
        print(f"  Keywords: {intent_info.get('keywords', [])}")
        print(f"  Business Context: {intent_info.get('business_context', 'N/A')}")
        print(f"  Expected Output: {intent_info.get('expected_output', 'N/A')}")
        print()
    
    # Test Q&A pattern matching
    print(f"\nTesting Q&A Pattern Matching")
    print("-" * 40)
    
    test_queries = [
        "What are the most common errors in MPAN records?",
        "Show me frequent MPAN validation issues",
        "Analyze error patterns in MPAN data",
        "Find MPAN records by postcode location",
        "Which suppliers have the highest error rates?",
        "Show me MPAN records by geographic region",
        "What are the validation failures in MPAN data?",
        "Find records with specific criteria",
        "Show me meter problems in the system",
        "Analyze error patterns in the data"
    ]
    
    for query in test_queries:
        print(f"\nQuery: {query}")
        print("-" * 50)
        
        # Test pattern matching
        qa_pattern = schema_manager.match_qa_pattern(collection_name, query)
        if qa_pattern:
            print(f"✅ Pattern Match Found!")
            print(f"   Question Pattern: {qa_pattern.get('question_pattern', 'N/A')}")
            print(f"   Answer Intent: {qa_pattern.get('answer_intent', 'N/A')}")
            print(f"   Match Score: {qa_pattern.get('match_score', 0.0):.2f}")
            print(f"   Business Entities: {qa_pattern.get('business_entities', [])}")
            print(f"   Expected Columns: {qa_pattern.get('expected_columns', [])}")
            print(f"   Search Strategy: {qa_pattern.get('search_strategy', 'N/A')}")
        else:
            print(f"❌ No pattern match found")
        
        # Test enhanced query enhancement
        enhanced_info = schema_manager.enhance_user_query(collection_name, query)
        print(f"\nEnhanced Query Info:")
        print(f"   Original Query: {enhanced_info['original_query']}")
        print(f"   Enhanced Query: {enhanced_info['enhanced_query']}")
        print(f"   Business Domain: {enhanced_info['business_domain']}")
        print(f"   Purpose: {enhanced_info['purpose']}")
        print(f"   Detected Intent: {enhanced_info['detected_intent']}")
        print(f"   Semantic Expansions: {enhanced_info['semantic_expansions']}")
        print(f"   Search Strategy: {enhanced_info['search_strategy']}")
        print(f"   Confidence Score: {enhanced_info['confidence_score']:.2f}")
        
        if enhanced_info['qa_pattern_match']:
            print(f"   Q&A Pattern Match: ✅")
            print(f"     Pattern: {enhanced_info['qa_pattern_match'].get('question_pattern', 'N/A')}")
            print(f"     Intent: {enhanced_info['qa_pattern_match'].get('answer_intent', 'N/A')}")
        
        if enhanced_info['relevant_columns']:
            print(f"   Relevant Columns:")
            for col in enhanced_info['relevant_columns']:
                print(f"     - {col['name']}: {col['relevance']} relevance")
                print(f"       Business meaning: {col['business_meaning']}")
                print(f"       Keywords: {col['keywords']}")
    
    # Test specific pattern matching scenarios
    print(f"\nTesting Specific Pattern Matching Scenarios")
    print("-" * 50)
    
    # Test exact pattern match
    exact_query = "What are the most common errors in MPAN records?"
    pattern = schema_manager.match_qa_pattern(collection_name, exact_query)
    if pattern:
        print(f"✅ Exact pattern match: {pattern.get('match_score', 0.0):.2f}")
    else:
        print(f"❌ Exact pattern match failed")
    
    # Test sample query match
    sample_query = "show me common MPAN errors"
    pattern = schema_manager.match_qa_pattern(collection_name, sample_query)
    if pattern:
        print(f"✅ Sample query match: {pattern.get('match_score', 0.0):.2f}")
    else:
        print(f"❌ Sample query match failed")
    
    # Test business entity match
    entity_query = "MPAN validation issues"
    pattern = schema_manager.match_qa_pattern(collection_name, entity_query)
    if pattern:
        print(f"✅ Business entity match: {pattern.get('match_score', 0.0):.2f}")
    else:
        print(f"❌ Business entity match failed")
    
    # Test keyword match
    keyword_query = "analyze patterns and trends"
    pattern = schema_manager.match_qa_pattern(collection_name, keyword_query)
    if pattern:
        print(f"✅ Keyword match: {pattern.get('match_score', 0.0):.2f}")
    else:
        print(f"❌ Keyword match failed")
    
    # Test validation
    print(f"\nTesting Schema Validation")
    print("-" * 30)
    
    validation = schema_manager.validate_collection_schema(collection_name)
    print(f"Schema validation: {'Valid' if validation['valid'] else 'Invalid'}")
    if validation['warnings']:
        print(f"Warnings: {validation['warnings']}")
    
    print(f"\nValidation Details:")
    print(f"   Data dictionary fields: {validation['data_dictionary_fields']}")
    print(f"   Business context: {'✅' if validation['business_context'] else '❌'}")
    print(f"   Query enhancement: {'✅' if validation['query_enhancement'] else '❌'}")
    print(f"   Q&A patterns: {'✅' if validation['qa_patterns'] else '❌'}")
    print(f"   Essential columns: {validation['essential_columns']}")
    print(f"   Exclude columns: {validation['exclude_columns']}")
    
    print(f"\nEnhanced Q&A Pattern Matching Test Complete!")
    print(f"\nKey Benefits of Enhanced Q&A Patterns:")
    print(f"   1. Pattern Matching - Automatically matches user queries to known patterns")
    print(f"   2. Intent Detection - Understands what users are trying to accomplish")
    print(f"   3. Column Optimization - Knows which columns are most relevant for each intent")
    print(f"   4. Search Strategy - Recommends the best search approach for each query type")
    print(f"   5. Confidence Scoring - Provides confidence levels for pattern matches")
    print(f"   6. Business Context - Understands business entities and relationships")
    print(f"   7. Sample Queries - Provides examples of similar questions")
    print(f"   8. Dynamic Learning - Can be extended with new patterns and feedback")

if __name__ == "__main__":
    test_enhanced_qa_patterns() 