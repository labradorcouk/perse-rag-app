#!/usr/bin/env python3
"""
Utils package for the RAG application.
"""

import sys
import os

# Add the project root to the path so utils can import other modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import core utilities that are actually used by the main application
from .mongodb_utils import MongoDBIndex
from .mongodb_schema_manager import MongoDBSchemaManager
from .qdrant_utils import QdrantIndex
from .dataframe_corrector import DataFrameCorrector
from .enhanced_dataframe_corrector import EnhancedDataFrameCorrector
from .rag_utils import RAGUtils
from .sql_connection import sql_manager
from .diagnostics_logger import diagnostics_logger, EventType, LogLevel
from .diagnostics_dashboard import diagnostics_dashboard
from .startup_monitor import startup_monitor
from .performance_optimizer import performance_optimizer

__all__ = [
    'MongoDBIndex',
    'MongoDBSchemaManager',
    'QdrantIndex',
    'DataFrameCorrector',
    'EnhancedDataFrameCorrector',
    'RAGUtils',
    'sql_manager',
    'diagnostics_logger',
    'EventType',
    'LogLevel',
    'diagnostics_dashboard',
    'startup_monitor',
    'performance_optimizer'
] 