#!/usr/bin/env python3
"""
Performance Optimization Utility

This module provides utilities to optimize I/O operations and reduce system load.
"""

import os
import gc
import time
import threading
from typing import Dict, Any, Optional
from functools import lru_cache
import streamlit as st

class PerformanceOptimizer:
    """Performance optimization utility for reducing I/O and improving caching."""
    
    def __init__(self):
        self.cache = {}
        self.cache_ttl = 3600  # 1 hour
        self.last_cleanup = time.time()
        self.cleanup_interval = 300  # 5 minutes
        
    @lru_cache(maxsize=100)
    def get_cached_embedding_model(self, model_name: str):
        """Get cached embedding model with LRU cache."""
        from sentence_transformers import SentenceTransformer
        return SentenceTransformer(model_name)
    
    @lru_cache(maxsize=10)
    def get_cached_qdrant_client(self, url: str):
        """Get cached Qdrant client with LRU cache."""
        from qdrant_client import QdrantClient
        return QdrantClient(url=url, prefer_grpc=True)
    
    def optimize_memory_usage(self):
        """Optimize memory usage by running garbage collection."""
        gc.collect()
        
    def cache_dataframe(self, key: str, df, ttl: int = 3600):
        """Cache DataFrame with TTL."""
        self.cache[key] = {
            'data': df,
            'timestamp': time.time(),
            'ttl': ttl
        }
        
    def get_cached_dataframe(self, key: str):
        """Get cached DataFrame if still valid."""
        if key in self.cache:
            cache_entry = self.cache[key]
            if time.time() - cache_entry['timestamp'] < cache_entry['ttl']:
                return cache_entry['data']
            else:
                del self.cache[key]
        return None
    
    def cleanup_expired_cache(self):
        """Clean up expired cache entries."""
        current_time = time.time()
        if current_time - self.last_cleanup > self.cleanup_interval:
            expired_keys = []
            for key, entry in self.cache.items():
                if current_time - entry['timestamp'] > entry['ttl']:
                    expired_keys.append(key)
            
            for key in expired_keys:
                del self.cache[key]
            
            self.last_cleanup = current_time
    
    def optimize_streamlit_cache(self):
        """Optimize Streamlit cache settings."""
        # Set cache TTL to 1 hour
        st.cache_data.clear()
        st.cache_resource.clear()
    
    def batch_operations(self, operations: list, batch_size: int = 100):
        """Execute operations in batches to reduce I/O."""
        results = []
        for i in range(0, len(operations), batch_size):
            batch = operations[i:i + batch_size]
            # Process batch
            batch_results = self._process_batch(batch)
            results.extend(batch_results)
        return results
    
    def _process_batch(self, batch):
        """Process a batch of operations."""
        # Implement batch processing logic here
        return batch
    
    def monitor_performance(self):
        """Monitor system performance and log metrics."""
        import psutil
        
        # Get system metrics
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        # Log performance metrics
        print(f"CPU Usage: {cpu_percent}%")
        print(f"Memory Usage: {memory.percent}%")
        print(f"Disk Usage: {disk.percent}%")
        
        return {
            'cpu_percent': cpu_percent,
            'memory_percent': memory.percent,
            'disk_percent': disk.percent
        }

# Global optimizer instance
performance_optimizer = PerformanceOptimizer() 