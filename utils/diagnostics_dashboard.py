#!/usr/bin/env python3
"""
Diagnostics Dashboard Component

This module provides a Streamlit dashboard for viewing and analyzing
diagnostic logs stored in Qdrant.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from typing import Dict, Any, List
from utils.diagnostics_logger import diagnostics_logger, EventType, LogLevel

class DiagnosticsDashboard:
    """Dashboard for viewing and analyzing diagnostic logs."""
    
    def __init__(self):
        self.logger = diagnostics_logger
    
    def render_dashboard(self):
        """Render the main diagnostics dashboard."""
        st.title("🔍 Diagnostics Dashboard")
        st.markdown("Monitor application logs, errors, and performance metrics stored in Qdrant.")
        
        # Check if diagnostics logger is available
        if not self.logger.client:
            st.error("❌ Diagnostics logger not available. Qdrant connection failed.")
            return
        
        # Get logs summary
        summary = self.logger.get_logs_summary()
        if not summary:
            st.warning("⚠️ No diagnostic logs found in Qdrant.")
            return
        
        # Overview metrics
        self._render_overview_metrics(summary)
        
        # Tabs for different views
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📊 Analytics", "🔍 Search Logs", "❌ Errors", "⚡ Performance", "👥 Users"
        ])
        
        with tab1:
            self._render_analytics_tab(summary)
        
        with tab2:
            self._render_search_tab()
        
        with tab3:
            self._render_errors_tab()
        
        with tab4:
            self._render_performance_tab()
        
        with tab5:
            self._render_users_tab(summary)
    
    def _render_overview_metrics(self, summary: Dict[str, Any]):
        """Render overview metrics cards."""
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                label="Total Logs",
                value=summary.get('total_logs', 0),
                help="Total number of diagnostic events stored"
            )
        
        with col2:
            st.metric(
                label="Active Users",
                value=summary.get('active_users', 0),
                help="Number of unique users with activity"
            )
        
        with col3:
            error_count = summary.get('log_levels', {}).get('error', 0)
            st.metric(
                label="Errors",
                value=error_count,
                delta=f"{error_count} errors detected",
                delta_color="inverse"
            )
        
        with col4:
            recent_logs = summary.get('recent_logs_analyzed', 0)
            st.metric(
                label="Recent Activity",
                value=recent_logs,
                help="Logs analyzed in summary"
            )
    
    def _render_analytics_tab(self, summary: Dict[str, Any]):
        """Render analytics tab with charts."""
        st.header("📊 Analytics Dashboard")
        
        # Event types chart
        event_types = summary.get('event_types', {})
        if event_types:
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Event Types Distribution")
                fig = px.pie(
                    values=list(event_types.values()),
                    names=list(event_types.keys()),
                    title="Event Types"
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.subheader("Log Levels Distribution")
                log_levels = summary.get('log_levels', {})
                if log_levels:
                    fig = px.bar(
                        x=list(log_levels.keys()),
                        y=list(log_levels.values()),
                        title="Log Levels",
                        color=list(log_levels.values()),
                        color_continuous_scale='RdYlGn'
                    )
                    st.plotly_chart(fig, use_container_width=True)
        
        # Components activity
        components = summary.get('components', {})
        if components:
            st.subheader("Component Activity")
            fig = px.bar(
                x=list(components.keys()),
                y=list(components.values()),
                title="Activity by Component",
                color=list(components.values()),
                color_continuous_scale='Blues'
            )
            st.plotly_chart(fig, use_container_width=True)
    
    def _render_search_tab(self):
        """Render search tab for querying logs."""
        st.header("🔍 Search Logs")
        
        # Search form
        with st.form("log_search_form"):
            search_query = st.text_input(
                "Search Query",
                placeholder="Enter search terms (e.g., 'error', 'authentication', 'performance')",
                help="Search logs using semantic search"
            )
            
            col1, col2 = st.columns(2)
            with col1:
                event_type_filter = st.multiselect(
                    "Event Type",
                    options=[e.value for e in EventType],
                    help="Filter by event type"
                )
            
            with col2:
                log_level_filter = st.multiselect(
                    "Log Level",
                    options=[l.value for l in LogLevel],
                    help="Filter by log level"
                )
            
            limit = st.slider("Max Results", min_value=10, max_value=100, value=50)
            
            if st.form_submit_button("🔍 Search Logs"):
                if search_query:
                    # Build filters
                    filters = {}
                    if event_type_filter:
                        filters['event_type'] = event_type_filter
                    if log_level_filter:
                        filters['log_level'] = log_level_filter
                    
                    # Search logs
                    results = self.logger.search_logs(search_query, limit=limit, filters=filters)
                    
                    if results:
                        st.success(f"✅ Found {len(results)} matching logs")
                        
                        # Convert to DataFrame for display
                        df = pd.DataFrame(results)
                        
                        # Format timestamp
                        if 'timestamp' in df.columns:
                            df['timestamp'] = pd.to_datetime(df['timestamp'])
                            df['timestamp'] = df['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
                        
                        # Display results
                        st.dataframe(
                            df[['timestamp', 'event_type', 'log_level', 'component', 'message', 'user_display_name']],
                            use_container_width=True
                        )
                        
                        # Show detailed view
                        if st.checkbox("Show detailed view"):
                            for i, result in enumerate(results):
                                with st.expander(f"Log {i+1}: {result.get('message', 'No message')}"):
                                    st.json(result)
                    else:
                        st.warning("No logs found matching your search criteria.")
                else:
                    st.warning("Please enter a search query.")
    
    def _render_errors_tab(self):
        """Render errors tab for viewing error logs."""
        st.header("❌ Error Analysis")
        
        # Search for errors
        error_results = self.logger.search_logs(
            "error exception failed",
            limit=100,
            filters={'log_level': ['error', 'critical']}
        )
        
        if error_results:
            st.success(f"✅ Found {len(error_results)} error logs")
            
            # Convert to DataFrame
            df = pd.DataFrame(error_results)
            
            # Format timestamp
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df['timestamp'] = df['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
            
            # Display errors
            st.dataframe(
                df[['timestamp', 'event_type', 'component', 'message', 'user_display_name']],
                use_container_width=True
            )
            
            # Error analysis
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Errors by Component")
                component_errors = df['component'].value_counts()
                fig = px.pie(
                    values=component_errors.values,
                    names=component_errors.index,
                    title="Errors by Component"
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.subheader("Errors by Event Type")
                event_errors = df['event_type'].value_counts()
                fig = px.bar(
                    x=event_errors.index,
                    y=event_errors.values,
                    title="Errors by Event Type"
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Show detailed error information
            if st.checkbox("Show detailed error information"):
                for i, error in enumerate(error_results):
                    with st.expander(f"Error {i+1}: {error.get('message', 'No message')}"):
                        if 'error_info' in error and error['error_info']:
                            st.error(f"Error Type: {error['error_info'].get('error_type', 'Unknown')}")
                            st.error(f"Error Message: {error['error_info'].get('error_message', 'No message')}")
                            if 'traceback' in error['error_info']:
                                st.code(error['error_info']['traceback'], language='python')
                        else:
                            st.json(error)
        else:
            st.info("🎉 No error logs found! The application is running smoothly.")
    
    def _render_performance_tab(self):
        """Render performance tab for viewing performance metrics."""
        st.header("⚡ Performance Metrics")
        
        # Search for performance logs
        perf_results = self.logger.search_logs(
            "performance duration execution time",
            limit=100,
            filters={'event_type': ['performance']}
        )
        
        if perf_results:
            st.success(f"✅ Found {len(perf_results)} performance logs")
            
            # Convert to DataFrame
            df = pd.DataFrame(perf_results)
            
            # Extract performance metrics
            durations = []
            operations = []
            components = []
            
            for result in perf_results:
                if 'performance_metrics' in result and result['performance_metrics']:
                    metrics = result['performance_metrics']
                    durations.append(metrics.get('duration_seconds', 0))
                    operations.append(metrics.get('operation', 'Unknown'))
                    components.append(result.get('component', 'Unknown'))
            
            if durations:
                # Performance summary
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Average Duration", f"{sum(durations)/len(durations):.2f}s")
                
                with col2:
                    st.metric("Max Duration", f"{max(durations):.2f}s")
                
                with col3:
                    st.metric("Min Duration", f"{min(durations):.2f}s")
                
                # Performance chart
                perf_df = pd.DataFrame({
                    'duration': durations,
                    'operation': operations,
                    'component': components
                })
                
                fig = px.scatter(
                    perf_df,
                    x='operation',
                    y='duration',
                    color='component',
                    title="Performance by Operation",
                    labels={'duration': 'Duration (seconds)', 'operation': 'Operation'}
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Performance distribution
                fig = px.histogram(
                    perf_df,
                    x='duration',
                    color='component',
                    title="Performance Distribution",
                    labels={'duration': 'Duration (seconds)'}
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Show detailed performance logs
            if st.checkbox("Show detailed performance logs"):
                for i, perf in enumerate(perf_results):
                    with st.expander(f"Performance Log {i+1}: {perf.get('message', 'No message')}"):
                        st.json(perf)
        else:
            st.info("No performance logs found.")
    
    def _render_users_tab(self, summary: Dict[str, Any]):
        """Render users tab for viewing user activity."""
        st.header("👥 User Activity")
        
        top_users = summary.get('top_users', {})
        if top_users:
            st.subheader("Most Active Users")
            
            # Convert to DataFrame
            user_df = pd.DataFrame([
                {'user_id': user_id, 'activity_count': count}
                for user_id, count in top_users.items()
            ])
            
            # Display user activity
            fig = px.bar(
                user_df,
                x='user_id',
                y='activity_count',
                title="User Activity",
                labels={'activity_count': 'Number of Events', 'user_id': 'User ID'}
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # User activity table
            st.dataframe(user_df, use_container_width=True)
        else:
            st.info("No user activity data available.")
        
        # Search for specific user activity
        st.subheader("Search User Activity")
        user_search = st.text_input("Enter user ID or email to search:")
        
        if user_search:
            user_results = self.logger.search_logs(
                user_search,
                limit=50,
                filters={'user_id': [user_search]}
            )
            
            if user_results:
                st.success(f"✅ Found {len(user_results)} events for user: {user_search}")
                
                # Convert to DataFrame
                df = pd.DataFrame(user_results)
                
                # Format timestamp
                if 'timestamp' in df.columns:
                    df['timestamp'] = pd.to_datetime(df['timestamp'])
                    df['timestamp'] = df['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
                
                # Display user activity
                st.dataframe(
                    df[['timestamp', 'event_type', 'component', 'message']],
                    use_container_width=True
                )
            else:
                st.warning(f"No activity found for user: {user_search}")

# Global dashboard instance
diagnostics_dashboard = DiagnosticsDashboard() 