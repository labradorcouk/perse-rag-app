#!/usr/bin/env python3
"""
Utils package for the RAG application.
"""

import sys
import os

# Add the project root to the path so utils can import other modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import core utilities that are actually used by the main application
from .rag_utils import RAGUtils
from .qdrant_utils import QdrantIndex
from .mongodb_utils import MongoDBIndex
from .diagnostics_logger import diagnostics_logger, EventType, LogLevel
from .diagnostics_dashboard import diagnostics_dashboard
from .startup_monitor import startup_monitor
from .performance_optimizer import performance_optimizer
from .optimized_loader import optimized_loader
from .sql_connection import sql_manager

# Import DataFrame correction utilities
from .dataframe_corrector import DataFrameCorrector
from .enhanced_dataframe_corrector import EnhancedDataFrameCorrector
from .intelligent_dataframe_fixer import IntelligentDataFrameFixer

__all__ = [
    'RAGUtils',
    'QdrantIndex',
    'MongoDBIndex',
    'diagnostics_logger',
    'EventType',
    'LogLevel',
    'diagnostics_dashboard',
    'startup_monitor',
    'performance_optimizer',
    'optimized_loader',
    'sql_manager',
    'DataFrameCorrector',
    'EnhancedDataFrameCorrector',
    'IntelligentDataFrameFixer',
] 