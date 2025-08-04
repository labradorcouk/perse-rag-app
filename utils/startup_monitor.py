#!/usr/bin/env python3
"""
Startup Performance Monitor

This module tracks startup times and provides optimization recommendations.
"""

import time
import threading
from typing import Dict, Any, List
import streamlit as st

class StartupMonitor:
    """Monitor startup performance and provide optimization insights."""
    
    def __init__(self):
        self.start_time = time.time()
        self.milestones = {}
        self.optimization_suggestions = []
    
    def mark_milestone(self, name: str):
        """Mark a startup milestone."""
        self.milestones[name] = time.time() - self.start_time
        print(f"⏱️ {name}: {self.milestones[name]:.2f}s")
    
    def get_startup_time(self) -> float:
        """Get total startup time."""
        return time.time() - self.start_time
    
    def analyze_performance(self) -> Dict[str, Any]:
        """Analyze startup performance and provide suggestions."""
        total_time = self.get_startup_time()
        
        # Performance thresholds
        if total_time > 30:
            self.optimization_suggestions.append("Startup time > 30s - Consider pre-loading models")
        if total_time > 60:
            self.optimization_suggestions.append("Startup time > 60s - Critical optimization needed")
        
        # Component analysis
        if 'embedding_model' in self.milestones and self.milestones['embedding_model'] > 10:
            self.optimization_suggestions.append("Embedding model loading > 10s - Consider caching")
        
        if 'qdrant_connection' in self.milestones and self.milestones['qdrant_connection'] > 5:
            self.optimization_suggestions.append("Qdrant connection > 5s - Check network connectivity")
        
        return {
            'total_time': total_time,
            'milestones': self.milestones,
            'suggestions': self.optimization_suggestions
        }
    
    def log_performance(self):
        """Log performance metrics."""
        analysis = self.analyze_performance()
        
        print(f"\n🚀 Startup Performance Report")
        print(f"Total time: {analysis['total_time']:.2f}s")
        print(f"Milestones: {analysis['milestones']}")
        
        if analysis['suggestions']:
            print(f"Optimization suggestions:")
            for suggestion in analysis['suggestions']:
                print(f"  - {suggestion}")

# Global monitor instance
startup_monitor = StartupMonitor() 