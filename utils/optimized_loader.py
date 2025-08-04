#!/usr/bin/env python3
"""
Optimized Loader for Fast Startup

This module implements lazy loading and caching strategies to improve
application startup time and reduce memory usage.
"""

import os
import time
import threading
from typing import Optional, Dict, Any
from functools import lru_cache
import streamlit as st

class OptimizedLoader:
    """Optimized loader with lazy loading and caching."""
    
    def __init__(self):
        self._embedding_model = None
        self._qdrant_client = None
        self._diagnostics_logger = None
        self._rag_utils = None
        self._initialized = False
        self._init_lock = threading.Lock()
    
    def initialize_async(self):
        """Initialize components asynchronously in background."""
        if self._initialized:
            return
        
        with self._init_lock:
            if self._initialized:
                return
            
            # Start background initialization
            thread = threading.Thread(target=self._background_init, daemon=True)
            thread.start()
    
    def _background_init(self):
        """Background initialization of heavy components."""
        try:
            # Initialize embedding model in background
            self._load_embedding_model()
            
            # Initialize Qdrant client in background
            self._load_qdrant_client()
            
            # Initialize diagnostics logger in background
            self._load_diagnostics_logger()
            
            self._initialized = True
            print("✅ Background initialization completed")
            
        except Exception as e:
            print(f"❌ Background initialization failed: {e}")
    
    def _load_embedding_model(self):
        """Load embedding model with caching."""
        if self._embedding_model is None:
            try:
                from sentence_transformers import SentenceTransformer
                print("🔄 Loading embedding model...")
                self._embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
                print("✅ Embedding model loaded")
            except Exception as e:
                print(f"❌ Failed to load embedding model: {e}")
    
    def _load_qdrant_client(self):
        """Load Qdrant client with connection pooling."""
        if self._qdrant_client is None:
            try:
                from qdrant_client import QdrantClient
                qdrant_url = os.getenv('QDRANT_URL', 'http://localhost:6333')
                print("🔄 Connecting to Qdrant...")
                self._qdrant_client = QdrantClient(url=qdrant_url, prefer_grpc=True)
                # Test connection
                self._qdrant_client.get_collections()
                print("✅ Qdrant client connected")
            except Exception as e:
                print(f"❌ Failed to connect to Qdrant: {e}")
    
    def _load_diagnostics_logger(self):
        """Load diagnostics logger with lazy initialization."""
        if self._diagnostics_logger is None:
            try:
                from utils.diagnostics_logger import diagnostics_logger
                self._diagnostics_logger = diagnostics_logger
                print("✅ Diagnostics logger loaded")
            except Exception as e:
                print(f"❌ Failed to load diagnostics logger: {e}")
    
    @property
    def embedding_model(self):
        """Get embedding model with lazy loading."""
        if self._embedding_model is None:
            self._load_embedding_model()
        return self._embedding_model
    
    @property
    def qdrant_client(self):
        """Get Qdrant client with lazy loading."""
        if self._qdrant_client is None:
            self._load_qdrant_client()
        return self._qdrant_client
    
    @property
    def diagnostics_logger(self):
        """Get diagnostics logger with lazy loading."""
        if self._diagnostics_logger is None:
            self._load_diagnostics_logger()
        return self._diagnostics_logger

# Global optimized loader instance
optimized_loader = OptimizedLoader()

@lru_cache(maxsize=1)
def get_cached_embedding_model():
    """Get cached embedding model."""
    return optimized_loader.embedding_model

@lru_cache(maxsize=1)
def get_cached_qdrant_client():
    """Get cached Qdrant client."""
    return optimized_loader.qdrant_client 