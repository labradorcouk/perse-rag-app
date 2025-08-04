#!/usr/bin/env python3
"""
Diagnostics Logger for Qdrant Storage

This module provides comprehensive logging and diagnostics functionality
that stores all application logs, errors, and diagnostics into a Qdrant collection.
"""

import os
import json
import traceback
import datetime
import uuid
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict
from enum import Enum
import streamlit as st
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, Distance, VectorParams
import numpy as np
from sentence_transformers import SentenceTransformer

class LogLevel(Enum):
    """Log levels for diagnostics."""
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

class EventType(Enum):
    """Event types for diagnostics."""
    AUTHENTICATION = "authentication"
    RAG_QUERY = "rag_query"
    SQL_QUERY = "sql_query"
    VECTOR_SEARCH = "vector_search"
    CODE_GENERATION = "code_generation"
    CODE_EXECUTION = "code_execution"
    ERROR = "error"
    PERFORMANCE = "performance"
    USER_ACTION = "user_action"
    SYSTEM = "system"

@dataclass
class DiagnosticEvent:
    """Data class for diagnostic events."""
    event_id: str
    timestamp: str
    event_type: str
    log_level: str
    user_id: Optional[str]
    session_id: Optional[str]
    component: str
    message: str
    details: Dict[str, Any]
    error_info: Optional[Dict[str, Any]]
    performance_metrics: Optional[Dict[str, Any]]
    metadata: Dict[str, Any]

class DiagnosticsLogger:
    """Main diagnostics logger class for Qdrant storage."""
    
    def __init__(self, collection_name: str = "diagnostics_logs"):
        self.collection_name = collection_name
        self.qdrant_url = os.getenv('QDRANT_URL', 'http://localhost:6333')
        self.client = None
        self.embedding_model = None
        self._initialize_qdrant()
        self._initialize_embedding_model()
    
    def _initialize_qdrant(self):
        """Initialize Qdrant client and create collection if needed."""
        try:
            self.client = QdrantClient(url=self.qdrant_url, prefer_grpc=True)
            
            # Check if collection exists
            collections = self.client.get_collections()
            collection_names = [col.name for col in collections.collections]
            
            if self.collection_name not in collection_names:
                # Create collection with vector search capabilities
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=384,  # Standard embedding size
                        distance=Distance.COSINE
                    )
                )
                print(f"✅ Created diagnostics collection: {self.collection_name}")
            else:
                print(f"✅ Using existing diagnostics collection: {self.collection_name}")
                
        except Exception as e:
            print(f"❌ Failed to initialize Qdrant: {e}")
            self.client = None
    
    def _initialize_embedding_model(self):
        """Initialize the embedding model for text search with lazy loading."""
        # Don't initialize immediately - wait until first use
        self.embedding_model = None
    
    def _get_embedding_model(self):
        """Get embedding model with lazy loading."""
        if self.embedding_model is None:
            try:
                from sentence_transformers import SentenceTransformer
                self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
                print("✅ Embedding model initialized for diagnostics")
            except Exception as e:
                print(f"❌ Failed to initialize embedding model: {e}")
                self.embedding_model = None
        return self.embedding_model
    
    def _generate_embedding(self, text: str) -> Optional[List[float]]:
        """Generate embedding for text search with lazy loading."""
        model = self._get_embedding_model()
        if model:
            try:
                embedding = model.encode(text)
                return embedding.tolist()
            except Exception as e:
                print(f"❌ Failed to generate embedding: {e}")
        return None
    
    def _get_user_info(self) -> Dict[str, Any]:
        """Get current user information from Streamlit session."""
        try:
            auth_state = st.session_state.get('azure_auth_state', {})
            user_info = auth_state.get('user_info', {})
            return {
                'user_id': user_info.get('userPrincipalName', 'unknown'),
                'display_name': user_info.get('displayName', 'Unknown User'),
                'email': user_info.get('mail', 'unknown@domain.com')
            }
        except Exception:
            return {
                'user_id': 'unknown',
                'display_name': 'Unknown User',
                'email': 'unknown@domain.com'
            }
    
    def _get_session_info(self) -> Dict[str, Any]:
        """Get current session information."""
        try:
            return {
                'session_id': str(id(st.session_state)),
                'timestamp': datetime.datetime.now().isoformat(),
                'page_url': st.get_script_run_ctx().page_script_hash if st.get_script_run_ctx() else None
            }
        except Exception:
            return {
                'session_id': 'unknown',
                'timestamp': datetime.datetime.now().isoformat(),
                'page_url': None
            }
    
    def log_event(
        self,
        event_type: EventType,
        log_level: LogLevel,
        component: str,
        message: str,
        details: Optional[Dict[str, Any]] = None,
        error_info: Optional[Dict[str, Any]] = None,
        performance_metrics: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Log a diagnostic event to Qdrant."""
        if not self.client:
            print("❌ Qdrant client not available for logging")
            return
        
        try:
            # Get user and session info
            user_info = self._get_user_info()
            session_info = self._get_session_info()
            
            # Create diagnostic event
            event = DiagnosticEvent(
                event_id=str(uuid.uuid4()),
                timestamp=session_info['timestamp'],
                event_type=event_type.value,
                log_level=log_level.value,
                user_id=user_info['user_id'],
                session_id=session_info['session_id'],
                component=component,
                message=message,
                details=details or {},
                error_info=error_info,
                performance_metrics=performance_metrics,
                metadata=metadata or {}
            )
            
            # Generate embedding for search
            search_text = f"{event_type.value} {log_level.value} {component} {message}"
            embedding = self._generate_embedding(search_text)
            
            # Prepare payload for Qdrant
            payload = asdict(event)
            
            # Add searchable text fields
            payload['search_text'] = search_text
            payload['user_display_name'] = user_info['display_name']
            payload['user_email'] = user_info['email']
            
            # Create point for Qdrant
            point = PointStruct(
                id=event.event_id,
                vector=embedding if embedding else [0.0] * 384,  # Default vector if embedding fails
                payload=payload
            )
            
            # Upsert to Qdrant
            self.client.upsert(
                collection_name=self.collection_name,
                points=[point]
            )
            
            # Also print to console for immediate visibility
            print(f"[{log_level.value.upper()}] {event_type.value}: {message}")
            
        except Exception as e:
            print(f"❌ Failed to log diagnostic event: {e}")
    
    def log_error(
        self,
        component: str,
        error: Exception,
        context: Optional[Dict[str, Any]] = None,
        user_message: Optional[str] = None
    ):
        """Log an error with full details."""
        error_info = {
            'error_type': type(error).__name__,
            'error_message': str(error),
            'traceback': traceback.format_exc(),
            'user_message': user_message
        }
        
        self.log_event(
            event_type=EventType.ERROR,
            log_level=LogLevel.ERROR,
            component=component,
            message=user_message or str(error),
            details=context,
            error_info=error_info
        )
    
    def log_rag_query(
        self,
        question: str,
        selected_tables: List[str],
        vector_search_engine: str,
        llm_provider: str,
        performance_metrics: Optional[Dict[str, Any]] = None
    ):
        """Log a RAG query event."""
        details = {
            'question': question,
            'selected_tables': selected_tables,
            'vector_search_engine': vector_search_engine,
            'llm_provider': llm_provider,
            'question_length': len(question)
        }
        
        self.log_event(
            event_type=EventType.RAG_QUERY,
            log_level=LogLevel.INFO,
            component="RAG_QA",
            message=f"RAG query: {question[:100]}{'...' if len(question) > 100 else ''}",
            details=details,
            performance_metrics=performance_metrics
        )
    
    def log_sql_query(
        self,
        sql_query: str,
        execution_time: Optional[float] = None,
        row_count: Optional[int] = None
    ):
        """Log a SQL query event."""
        details = {
            'sql_query': sql_query,
            'query_length': len(sql_query)
        }
        
        performance_metrics = {}
        if execution_time:
            performance_metrics['execution_time_seconds'] = execution_time
        if row_count:
            performance_metrics['row_count'] = row_count
        
        self.log_event(
            event_type=EventType.SQL_QUERY,
            log_level=LogLevel.INFO,
            component="SQL_Editor",
            message=f"SQL query executed: {row_count} rows returned",
            details=details,
            performance_metrics=performance_metrics
        )
    
    def log_authentication(
        self,
        auth_method: str,
        success: bool,
        user_info: Optional[Dict[str, Any]] = None,
        error_message: Optional[str] = None
    ):
        """Log an authentication event."""
        details = {
            'auth_method': auth_method,
            'success': success
        }
        
        if user_info:
            details['user_info'] = user_info
        
        error_info = None
        if not success and error_message:
            error_info = {'error_message': error_message}
        
        self.log_event(
            event_type=EventType.AUTHENTICATION,
            log_level=LogLevel.INFO if success else LogLevel.ERROR,
            component="Authentication",
            message=f"Authentication {'successful' if success else 'failed'} via {auth_method}",
            details=details,
            error_info=error_info
        )
    
    def log_code_generation(
        self,
        question: str,
        generated_code: str,
        llm_provider: str,
        execution_success: bool,
        execution_error: Optional[str] = None
    ):
        """Log code generation and execution."""
        details = {
            'question': question,
            'generated_code': generated_code,
            'llm_provider': llm_provider,
            'code_length': len(generated_code),
            'execution_success': execution_success
        }
        
        error_info = None
        if not execution_success and execution_error:
            error_info = {'execution_error': execution_error}
        
        self.log_event(
            event_type=EventType.CODE_GENERATION,
            log_level=LogLevel.INFO if execution_success else LogLevel.ERROR,
            component="Code_Generation",
            message=f"Code generation {'successful' if execution_success else 'failed'} with {llm_provider}",
            details=details,
            error_info=error_info
        )
    
    def log_performance(
        self,
        component: str,
        operation: str,
        duration_seconds: float,
        additional_metrics: Optional[Dict[str, Any]] = None
    ):
        """Log performance metrics."""
        performance_metrics = {
            'duration_seconds': duration_seconds,
            'operation': operation
        }
        
        if additional_metrics:
            performance_metrics.update(additional_metrics)
        
        self.log_event(
            event_type=EventType.PERFORMANCE,
            log_level=LogLevel.INFO,
            component=component,
            message=f"Performance: {operation} took {duration_seconds:.2f}s",
            performance_metrics=performance_metrics
        )
    
    def search_logs(
        self,
        query: str,
        limit: int = 50,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Search logs using semantic search."""
        if not self.client or not self._get_embedding_model(): # Use _get_embedding_model here
            print("❌ Qdrant client or embedding model not available")
            return []
        
        try:
            # Generate query embedding
            query_embedding = self._generate_embedding(query)
            if not query_embedding:
                return []
            
            # Build filter conditions
            filter_conditions = []
            if filters:
                for key, value in filters.items():
                    if isinstance(value, list):
                        filter_conditions.append({
                            "key": key,
                            "match": {"any": value}
                        })
                    else:
                        filter_conditions.append({
                            "key": key,
                            "match": {"value": value}
                        })
            
            # Search in Qdrant
            search_result = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding,
                limit=limit,
                query_filter={"must": filter_conditions} if filter_conditions else None
            )
            
            # Convert to list of dictionaries
            results = []
            for point in search_result:
                results.append(point.payload)
            
            return results
            
        except Exception as e:
            print(f"❌ Failed to search logs: {e}")
            return []
    
    def get_logs_summary(self) -> Dict[str, Any]:
        """Get a summary of logs for analytics."""
        if not self.client:
            return {}
        
        try:
            # Get collection info
            collection_info = self.client.get_collection(self.collection_name)
            total_points = collection_info.points_count
            
            # Get recent logs (last 1000 points)
            recent_logs = self.client.scroll(
                collection_name=self.collection_name,
                limit=1000,
                with_payload=True,
                with_vectors=False
            )[0]
            
            # Analyze logs
            event_types = {}
            log_levels = {}
            components = {}
            users = {}
            
            for point in recent_logs:
                payload = point.payload
                
                # Count event types
                event_type = payload.get('event_type', 'unknown')
                event_types[event_type] = event_types.get(event_type, 0) + 1
                
                # Count log levels
                log_level = payload.get('log_level', 'unknown')
                log_levels[log_level] = log_levels.get(log_level, 0) + 1
                
                # Count components
                component = payload.get('component', 'unknown')
                components[component] = components.get(component, 0) + 1
                
                # Count users
                user_id = payload.get('user_id', 'unknown')
                users[user_id] = users.get(user_id, 0) + 1
            
            return {
                'total_logs': total_points,
                'recent_logs_analyzed': len(recent_logs),
                'event_types': event_types,
                'log_levels': log_levels,
                'components': components,
                'active_users': len(users),
                'top_users': dict(sorted(users.items(), key=lambda x: x[1], reverse=True)[:10])
            }
            
        except Exception as e:
            print(f"❌ Failed to get logs summary: {e}")
            return {}

# Global logger instance
diagnostics_logger = DiagnosticsLogger() 