#!/usr/bin/env python3
"""
Test script to verify performance optimization integration.
"""

def test_performance_optimizer_import():
    """Test that performance optimizer can be imported."""
    try:
        from utils.performance_optimizer import performance_optimizer
        print("✅ Performance optimizer import successful")
        return True
    except Exception as e:
        print(f"❌ Performance optimizer import failed: {e}")
        return False

def test_performance_optimizer_functionality():
    """Test performance optimizer functionality."""
    try:
        from utils.performance_optimizer import performance_optimizer
        
        # Test memory optimization
        performance_optimizer.optimize_memory_usage()
        print("✅ Memory optimization working")
        
        # Test cache functionality
        test_data = {"test": "data"}
        performance_optimizer.cache_dataframe("test_key", test_data, ttl=60)
        
        cached_data = performance_optimizer.get_cached_dataframe("test_key")
        if cached_data == test_data:
            print("✅ Cache functionality working")
        else:
            print("❌ Cache functionality failed")
            return False
        
        # Test cache cleanup
        performance_optimizer.cleanup_expired_cache()
        print("✅ Cache cleanup working")
        
        return True
    except Exception as e:
        print(f"❌ Performance optimizer functionality test failed: {e}")
        return False

def test_monitor_performance_import():
    """Test that monitor performance can be imported."""
    try:
        import monitor_performance
        print("✅ Monitor performance import successful")
        return True
    except Exception as e:
        print(f"❌ Monitor performance import failed: {e}")
        return False

def test_rag_app_integration():
    """Test that RAG app has performance optimizations."""
    try:
        with open('rag_fabric_app_docker.py', 'r') as f:
            content = f.read()
        
        # Check for performance optimizer import
        if 'from utils.performance_optimizer import performance_optimizer' in content:
            print("✅ Performance optimizer import found in RAG app")
        else:
            print("❌ Performance optimizer import not found in RAG app")
            return False
        
        # Check for performance optimizer usage
        if 'performance_optimizer.' in content:
            print("✅ Performance optimizer usage found in RAG app")
        else:
            print("❌ Performance optimizer usage not found in RAG app")
            return False
        
        # Check for performance monitoring tab
        if '📊 Performance' in content:
            print("✅ Performance monitoring tab found in RAG app")
        else:
            print("❌ Performance monitoring tab not found in RAG app")
            return False
        
        return True
    except Exception as e:
        print(f"❌ RAG app integration test failed: {e}")
        return False

def main():
    """Run all performance integration tests."""
    print("🧪 Testing Performance Optimization Integration")
    print("=" * 50)
    
    tests = [
        ("Performance Optimizer Import", test_performance_optimizer_import),
        ("Performance Optimizer Functionality", test_performance_optimizer_functionality),
        ("Monitor Performance Import", test_monitor_performance_import),
        ("RAG App Integration", test_rag_app_integration)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n📝 Testing: {test_name}")
        if test_func():
            print("✅ PASS")
            passed += 1
        else:
            print("❌ FAIL")
    
    print(f"\n📊 Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All performance optimization integration tests passed!")
    else:
        print("⚠️ Some integration tests failed")
    
    return passed == total

if __name__ == "__main__":
    main() 