#!/usr/bin/env python3
"""
Test script for the diagnostics logging system.
"""

import os
import time
from dotenv import load_dotenv
from utils.diagnostics_logger import diagnostics_logger, EventType, LogLevel

def test_diagnostics_logging():
    """Test the diagnostics logging functionality."""
    print("🧪 Testing Diagnostics Logging System")
    print("=" * 50)
    
    # Load environment variables
    load_dotenv()
    
    # Test basic logging
    print("1. Testing basic event logging...")
    try:
        diagnostics_logger.log_event(
            event_type=EventType.SYSTEM,
            log_level=LogLevel.INFO,
            component="Test_Component",
            message="Test diagnostic event",
            details={"test": True, "timestamp": time.time()}
        )
        print("✅ Basic event logging successful")
    except Exception as e:
        print(f"❌ Basic event logging failed: {e}")
    
    # Test error logging
    print("\n2. Testing error logging...")
    try:
        test_error = Exception("Test error for diagnostics")
        diagnostics_logger.log_error(
            component="Test_Component",
            error=test_error,
            context={"test": True},
            user_message="Test error message"
        )
        print("✅ Error logging successful")
    except Exception as e:
        print(f"❌ Error logging failed: {e}")
    
    # Test RAG query logging
    print("\n3. Testing RAG query logging...")
    try:
        diagnostics_logger.log_rag_query(
            question="What is the average energy consumption?",
            selected_tables=["epc_non_domestic_scotland"],
            vector_search_engine="Qdrant",
            llm_provider="OpenAI GPT-3.5/4",
            performance_metrics={"duration_seconds": 2.5}
        )
        print("✅ RAG query logging successful")
    except Exception as e:
        print(f"❌ RAG query logging failed: {e}")
    
    # Test SQL query logging
    print("\n4. Testing SQL query logging...")
    try:
        diagnostics_logger.log_sql_query(
            sql_query="SELECT TOP 10 * FROM epc_non_domestic_scotland",
            execution_time=1.2,
            row_count=10
        )
        print("✅ SQL query logging successful")
    except Exception as e:
        print(f"❌ SQL query logging failed: {e}")
    
    # Test authentication logging
    print("\n5. Testing authentication logging...")
    try:
        diagnostics_logger.log_authentication(
            auth_method="OAuth",
            success=True,
            user_info={
                "userPrincipalName": "test@opendata.energy",
                "displayName": "Test User",
                "mail": "test@opendata.energy"
            }
        )
        print("✅ Authentication logging successful")
    except Exception as e:
        print(f"❌ Authentication logging failed: {e}")
    
    # Test performance logging
    print("\n6. Testing performance logging...")
    try:
        diagnostics_logger.log_performance(
            component="Test_Component",
            operation="test_operation",
            duration_seconds=1.5,
            additional_metrics={"memory_usage": "100MB"}
        )
        print("✅ Performance logging successful")
    except Exception as e:
        print(f"❌ Performance logging failed: {e}")
    
    # Test logs summary
    print("\n7. Testing logs summary...")
    try:
        summary = diagnostics_logger.get_logs_summary()
        print(f"✅ Logs summary retrieved: {summary.get('total_logs', 0)} total logs")
        print(f"   - Event types: {summary.get('event_types', {})}")
        print(f"   - Log levels: {summary.get('log_levels', {})}")
        print(f"   - Components: {summary.get('components', {})}")
    except Exception as e:
        print(f"❌ Logs summary failed: {e}")
    
    # Test log search
    print("\n8. Testing log search...")
    try:
        results = diagnostics_logger.search_logs("test", limit=10)
        print(f"✅ Log search successful: {len(results)} results found")
    except Exception as e:
        print(f"❌ Log search failed: {e}")
    
    print("\n" + "=" * 50)
    print("🎉 Diagnostics logging test completed!")
    print("\nNext steps:")
    print("1. Check the diagnostics dashboard in the Streamlit app")
    print("2. Verify logs are being stored in Qdrant")
    print("3. Test the search functionality in the dashboard")

if __name__ == "__main__":
    test_diagnostics_logging() 