#!/usr/bin/env python3
"""
Performance Monitoring Framework for Dynamic Learning System
Tracks system performance, user interactions, and learning metrics in real-time.
"""

import time
import psutil
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import json
import os
from collections import defaultdict, deque
import logging

class PerformanceMonitor:
    """
    Real-time performance monitoring for the dynamic learning system.
    Tracks system resources, query performance, and learning metrics.
    """
    
    def __init__(self, log_file: str = "performance_monitor.log"):
        """
        Initialize the performance monitor.
        
        Args:
            log_file: Path to the log file
        """
        self.log_file = log_file
        self.start_time = time.time()
        self.is_monitoring = False
        self.monitor_thread = None
        
        # Performance metrics storage
        self.metrics = {
            'system': defaultdict(list),
            'queries': defaultdict(list),
            'learning': defaultdict(list),
            'patterns': defaultdict(list),
            'errors': defaultdict(list)
        }
        
        # Real-time counters
        self.counters = {
            'total_queries': 0,
            'successful_queries': 0,
            'failed_queries': 0,
            'patterns_learned': 0,
            'patterns_evolved': 0,
            'user_feedback_count': 0
        }
        
        # Performance thresholds
        self.thresholds = {
            'query_time_warning': 2.0,  # seconds
            'query_time_critical': 5.0,  # seconds
            'memory_warning': 80.0,      # percentage
            'memory_critical': 90.0,     # percentage
            'cpu_warning': 70.0,         # percentage
            'cpu_critical': 85.0         # percentage
        }
        
        # Setup logging
        self._setup_logging()
        
        # Alert handlers
        self.alert_handlers = []
    
    def _setup_logging(self):
        """Setup logging configuration."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def start_monitoring(self, interval: float = 1.0):
        """
        Start continuous performance monitoring.
        
        Args:
            interval: Monitoring interval in seconds
        """
        if self.is_monitoring:
            self.logger.warning("Monitoring already started")
            return
        
        self.is_monitoring = True
        self.monitor_thread = threading.Thread(
            target=self._monitor_loop,
            args=(interval,),
            daemon=True
        )
        self.monitor_thread.start()
        self.logger.info(f"Performance monitoring started (interval: {interval}s)")
    
    def stop_monitoring(self):
        """Stop performance monitoring."""
        self.is_monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5.0)
        self.logger.info("Performance monitoring stopped")
    
    def _monitor_loop(self, interval: float):
        """Main monitoring loop."""
        while self.is_monitoring:
            try:
                self._collect_system_metrics()
                self._check_thresholds()
                time.sleep(interval)
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
    
    def _collect_system_metrics(self):
        """Collect system performance metrics."""
        timestamp = datetime.now()
        
        # CPU usage
        cpu_percent = psutil.cpu_percent(interval=0.1)
        self.metrics['system']['cpu_percent'].append({
            'timestamp': timestamp,
            'value': cpu_percent
        })
        
        # Memory usage
        memory = psutil.virtual_memory()
        self.metrics['system']['memory_percent'].append({
            'timestamp': timestamp,
            'value': memory.percent
        })
        
        # Disk usage
        disk = psutil.disk_usage('/')
        self.metrics['system']['disk_percent'].append({
            'timestamp': timestamp,
            'value': (disk.used / disk.total) * 100
        })
        
        # Network I/O
        network = psutil.net_io_counters()
        self.metrics['system']['network_bytes_sent'].append({
            'timestamp': timestamp,
            'value': network.bytes_sent
        })
        self.metrics['system']['network_bytes_recv'].append({
            'timestamp': timestamp,
            'value': network.bytes_recv
        })
        
        # Clean old metrics (keep last 1000 entries)
        for category in self.metrics.values():
            for metric_list in category.values():
                if len(metric_list) > 1000:
                    metric_list.pop(0)
    
    def _check_thresholds(self):
        """Check performance thresholds and trigger alerts."""
        current_metrics = self._get_current_metrics()
        
        # Check CPU
        if current_metrics['cpu_percent'] > self.thresholds['cpu_critical']:
            self._trigger_alert('CRITICAL', 'CPU usage critical', 
                              f"CPU: {current_metrics['cpu_percent']:.1f}%")
        elif current_metrics['cpu_percent'] > self.thresholds['cpu_warning']:
            self._trigger_alert('WARNING', 'CPU usage high', 
                              f"CPU: {current_metrics['cpu_percent']:.1f}%")
        
        # Check memory
        if current_metrics['memory_percent'] > self.thresholds['memory_critical']:
            self._trigger_alert('CRITICAL', 'Memory usage critical', 
                              f"Memory: {current_metrics['memory_percent']:.1f}%")
        elif current_metrics['memory_percent'] > self.thresholds['memory_warning']:
            self._trigger_alert('WARNING', 'Memory usage high', 
                              f"Memory: {current_metrics['memory_percent']:.1f}%")
    
    def _get_current_metrics(self) -> Dict[str, float]:
        """Get current system metrics."""
        current = {}
        
        # Get latest CPU
        if self.metrics['system']['cpu_percent']:
            current['cpu_percent'] = self.metrics['system']['cpu_percent'][-1]['value']
        else:
            current['cpu_percent'] = 0.0
        
        # Get latest memory
        if self.metrics['system']['memory_percent']:
            current['memory_percent'] = self.metrics['system']['memory_percent'][-1]['value']
        else:
            current['memory_percent'] = 0.0
        
        return current
    
    def _trigger_alert(self, level: str, title: str, message: str):
        """Trigger a performance alert."""
        alert = {
            'level': level,
            'title': title,
            'message': message,
            'timestamp': datetime.now().isoformat()
        }
        
        self.logger.warning(f"ALERT [{level}]: {title} - {message}")
        
        # Store alert
        self.metrics['errors'][f'{level.lower()}_alerts'].append(alert)
        
        # Notify handlers
        for handler in self.alert_handlers:
            try:
                handler(alert)
            except Exception as e:
                self.logger.error(f"Error in alert handler: {e}")
    
    def record_query(self, query: str, collection: str, start_time: float, 
                    success: bool, intent_detected: str = None, 
                    confidence: float = None, error: str = None):
        """
        Record query performance metrics.
        
        Args:
            query: The user query
            collection: MongoDB collection name
            start_time: Query start time
            success: Whether query was successful
            intent_detected: Detected intent
            confidence: Confidence score
            error: Error message if failed
        """
        end_time = time.time()
        duration = end_time - start_time
        
        # Update counters
        self.counters['total_queries'] += 1
        if success:
            self.counters['successful_queries'] += 1
        else:
            self.counters['failed_queries'] += 1
        
        # Record metrics
        query_metric = {
            'timestamp': datetime.now(),
            'query': query[:100],  # Truncate long queries
            'collection': collection,
            'duration': duration,
            'success': success,
            'intent_detected': intent_detected,
            'confidence': confidence,
            'error': error
        }
        
        self.metrics['queries']['performance'].append(query_metric)
        
        # Check performance thresholds
        if duration > self.thresholds['query_time_critical']:
            self._trigger_alert('CRITICAL', 'Query performance critical', 
                              f"Query took {duration:.2f}s")
        elif duration > self.thresholds['query_time_warning']:
            self._trigger_alert('WARNING', 'Query performance slow', 
                              f"Query took {duration:.2f}s")
        
        # Log query
        if success:
            self.logger.info(f"Query completed in {duration:.3f}s: {query[:50]}...")
        else:
            self.logger.error(f"Query failed in {duration:.3f}s: {error}")
    
    def record_learning(self, collection: str, intent: str, 
                       satisfaction: int = None, pattern_updated: bool = False):
        """
        Record learning metrics.
        
        Args:
            collection: MongoDB collection name
            intent: The detected intent
            satisfaction: User satisfaction rating
            pattern_updated: Whether pattern was updated
        """
        # Update counters
        self.counters['patterns_learned'] += 1
        if pattern_updated:
            self.counters['patterns_evolved'] += 1
        
        # Record metrics
        learning_metric = {
            'timestamp': datetime.now(),
            'collection': collection,
            'intent': intent,
            'satisfaction': satisfaction,
            'pattern_updated': pattern_updated
        }
        
        self.metrics['learning']['events'].append(learning_metric)
        
        self.logger.info(f"Learning recorded: {intent} for {collection}")
    
    def record_pattern_evolution(self, collection: str, patterns_updated: int, 
                               avg_confidence_change: float):
        """
        Record pattern evolution metrics.
        
        Args:
            collection: MongoDB collection name
            patterns_updated: Number of patterns updated
            avg_confidence_change: Average change in confidence scores
        """
        evolution_metric = {
            'timestamp': datetime.now(),
            'collection': collection,
            'patterns_updated': patterns_updated,
            'avg_confidence_change': avg_confidence_change
        }
        
        self.metrics['patterns']['evolution'].append(evolution_metric)
        
        self.logger.info(f"Pattern evolution: {patterns_updated} patterns updated, "
                        f"confidence change: {avg_confidence_change:.3f}")
    
    def record_user_feedback(self, pattern_id: str, rating: int, 
                           feedback_notes: str = None):
        """
        Record user feedback metrics.
        
        Args:
            pattern_id: ID of the pattern
            rating: User rating (1-5)
            feedback_notes: Additional feedback
        """
        # Update counter
        self.counters['user_feedback_count'] += 1
        
        # Record metrics
        feedback_metric = {
            'timestamp': datetime.now(),
            'pattern_id': pattern_id,
            'rating': rating,
            'feedback_notes': feedback_notes
        }
        
        self.metrics['patterns']['feedback'].append(feedback_metric)
        
        self.logger.info(f"User feedback recorded: Pattern {pattern_id}, Rating: {rating}/5")
    
    def get_performance_summary(self, time_window: timedelta = timedelta(hours=1)) -> Dict[str, Any]:
        """
        Get performance summary for the specified time window.
        
        Args:
            time_window: Time window for summary
            
        Returns:
            Performance summary dictionary
        """
        cutoff_time = datetime.now() - time_window
        
        summary = {
            'time_window': str(time_window),
            'system_health': self._get_system_health(),
            'query_performance': self._get_query_performance(cutoff_time),
            'learning_metrics': self._get_learning_metrics(cutoff_time),
            'pattern_metrics': self._get_pattern_metrics(cutoff_time),
            'counters': self.counters.copy(),
            'uptime': time.time() - self.start_time
        }
        
        return summary
    
    def _get_system_health(self) -> Dict[str, Any]:
        """Get current system health metrics."""
        current = self._get_current_metrics()
        
        return {
            'cpu_percent': current['cpu_percent'],
            'memory_percent': current['memory_percent'],
            'status': self._get_system_status(current)
        }
    
    def _get_system_status(self, metrics: Dict[str, float]) -> str:
        """Get system status based on metrics."""
        if (metrics['cpu_percent'] > self.thresholds['cpu_critical'] or 
            metrics['memory_percent'] > self.thresholds['memory_critical']):
            return 'CRITICAL'
        elif (metrics['cpu_percent'] > self.thresholds['cpu_warning'] or 
              metrics['memory_percent'] > self.thresholds['memory_warning']):
            return 'WARNING'
        else:
            return 'HEALTHY'
    
    def _get_query_performance(self, cutoff_time: datetime) -> Dict[str, Any]:
        """Get query performance metrics for the time window."""
        recent_queries = [
            q for q in self.metrics['queries']['performance']
            if q['timestamp'] > cutoff_time
        ]
        
        if not recent_queries:
            return {'count': 0, 'avg_duration': 0, 'success_rate': 0}
        
        durations = [q['duration'] for q in recent_queries]
        successful = [q for q in recent_queries if q['success']]
        
        return {
            'count': len(recent_queries),
            'avg_duration': sum(durations) / len(durations),
            'min_duration': min(durations),
            'max_duration': max(durations),
            'success_rate': len(successful) / len(recent_queries) * 100
        }
    
    def _get_learning_metrics(self, cutoff_time: datetime) -> Dict[str, Any]:
        """Get learning metrics for the time window."""
        recent_learning = [
            l for l in self.metrics['learning']['events']
            if l['timestamp'] > cutoff_time
        ]
        
        if not recent_learning:
            return {'events': 0, 'patterns_updated': 0}
        
        patterns_updated = sum(1 for l in recent_learning if l['pattern_updated'])
        
        return {
            'events': len(recent_learning),
            'patterns_updated': patterns_updated,
            'intent_distribution': self._get_intent_distribution(recent_learning)
        }
    
    def _get_pattern_metrics(self, cutoff_time: datetime) -> Dict[str, Any]:
        """Get pattern metrics for the time window."""
        recent_evolution = [
            p for p in self.metrics['patterns']['evolution']
            if p['timestamp'] > cutoff_time
        ]
        
        recent_feedback = [
            f for f in self.metrics['patterns']['feedback']
            if f['timestamp'] > cutoff_time
        ]
        
        return {
            'evolution_events': len(recent_evolution),
            'feedback_count': len(recent_feedback),
            'avg_rating': self._get_average_rating(recent_feedback) if recent_feedback else 0
        }
    
    def _get_intent_distribution(self, learning_events: List[Dict]) -> Dict[str, int]:
        """Get distribution of intents from learning events."""
        distribution = defaultdict(int)
        for event in learning_events:
            distribution[event['intent']] += 1
        return dict(distribution)
    
    def _get_average_rating(self, feedback_events: List[Dict]) -> float:
        """Get average rating from feedback events."""
        ratings = [f['rating'] for f in feedback_events if f['rating'] is not None]
        return sum(ratings) / len(ratings) if ratings else 0
    
    def add_alert_handler(self, handler):
        """Add an alert handler function."""
        self.alert_handlers.append(handler)
    
    def export_metrics(self, filepath: str):
        """Export all metrics to a JSON file."""
        try:
            export_data = {
                'export_timestamp': datetime.now().isoformat(),
                'metrics': self.metrics,
                'counters': self.counters,
                'uptime': time.time() - self.start_time
            }
            
            with open(filepath, 'w') as f:
                json.dump(export_data, f, indent=2, default=str)
            
            self.logger.info(f"Metrics exported to {filepath}")
            
        except Exception as e:
            self.logger.error(f"Failed to export metrics: {e}")
    
    def clear_old_metrics(self, days_to_keep: int = 7):
        """Clear metrics older than specified days."""
        cutoff_time = datetime.now() - timedelta(days=days_to_keep)
        
        cleared_count = 0
        for category in self.metrics.values():
            for metric_list in category.values():
                original_length = len(metric_list)
                metric_list[:] = [
                    m for m in metric_list 
                    if m['timestamp'] > cutoff_time
                ]
                cleared_count += original_length - len(metric_list)
        
        self.logger.info(f"Cleared {cleared_count} old metrics (older than {days_to_keep} days)")
    
    def get_realtime_dashboard_data(self) -> Dict[str, Any]:
        """Get real-time data for dashboard display."""
        current_metrics = self._get_current_metrics()
        
        return {
            'system': {
                'cpu_percent': current_metrics['cpu_percent'],
                'memory_percent': current_metrics['memory_percent'],
                'status': self._get_system_status(current_metrics)
            },
            'queries': {
                'total': self.counters['total_queries'],
                'successful': self.counters['successful_queries'],
                'failed': self.counters['failed_queries'],
                'success_rate': (self.counters['successful_queries'] / 
                               max(self.counters['total_queries'], 1)) * 100
            },
            'learning': {
                'patterns_learned': self.counters['patterns_learned'],
                'patterns_evolved': self.counters['patterns_evolved'],
                'user_feedback': self.counters['user_feedback_count']
            },
            'uptime': time.time() - self.start_time
        }

# Global performance monitor instance
performance_monitor = PerformanceMonitor()

def get_performance_monitor() -> PerformanceMonitor:
    """Get the global performance monitor instance."""
    return performance_monitor 