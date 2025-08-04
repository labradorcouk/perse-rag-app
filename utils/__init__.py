#!/usr/bin/env python3
"""
Utils package for Fabric RAG application.
"""

# Import main utilities
from .dataframe_corrector import DataFrameCorrector
from .enhanced_dataframe_corrector import EnhancedDataFrameCorrector
from .intelligent_dataframe_fixer import IntelligentDataFrameFixer
from .performance_optimizer import PerformanceOptimizer
from .diagnostics_logger import DiagnosticsLogger, EventType, LogLevel
from .diagnostics_dashboard import DiagnosticsDashboard
from .startup_monitor import StartupMonitor
from .optimized_loader import OptimizedLoader
from .sql_connection import SQLConnectionManager
from .rag_utils import RAGUtils
from .qdrant_utils import QdrantIndex
from .verisk_qdrant_processor import VeriskQdrantProcessor

# Create instances for backward compatibility
performance_optimizer = PerformanceOptimizer()
diagnostics_logger = DiagnosticsLogger()
diagnostics_dashboard = DiagnosticsDashboard()
startup_monitor = StartupMonitor()
optimized_loader = OptimizedLoader()
sql_manager = SQLConnectionManager()

__all__ = [
    'DataFrameCorrector',
    'EnhancedDataFrameCorrector', 
    'IntelligentDataFrameFixer',
    'PerformanceOptimizer',
    'DiagnosticsLogger',
    'DiagnosticsDashboard',
    'StartupMonitor',
    'OptimizedLoader',
    'SQLConnectionManager',
    'RAGUtils',
    'QdrantIndex',
    'VeriskQdrantProcessor',
    'EventType',
    'LogLevel',
    # Backward compatibility instances
    'performance_optimizer',
    'diagnostics_logger',
    'diagnostics_dashboard',
    'startup_monitor',
    'optimized_loader',
    'sql_manager'
] 