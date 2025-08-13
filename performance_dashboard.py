#!/usr/bin/env python3
"""
Performance Dashboard for Dynamic Learning System
Displays real-time performance metrics, system health, and learning analytics.
"""

import time
import os
import sys
from datetime import datetime, timedelta
import json

# Add utils to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'utils'))

try:
    from performance_monitor import get_performance_monitor
    from mongodb_schema_manager import MongoDBSchemaManager
except ImportError as e:
    print(f"Error importing modules: {e}")
    print("Make sure you're running this from the project root directory")
    sys.exit(1)

class PerformanceDashboard:
    """Real-time performance dashboard for the dynamic learning system."""
    
    def __init__(self):
        """Initialize the performance dashboard."""
        self.performance_monitor = get_performance_monitor()
        self.schema_manager = MongoDBSchemaManager()
        self.collection_name = "ecoesTechDetailsWithEmbedding"
        self.refresh_interval = 2.0  # seconds
        self.is_running = False
        
        # Start performance monitoring if not already running
        if not self.performance_monitor.is_monitoring:
            self.performance_monitor.start_monitoring(interval=1.0)
    
    def start_dashboard(self):
        """Start the interactive dashboard."""
        self.is_running = True
        print("🚀 Dynamic Learning System - Performance Dashboard")
        print("=" * 60)
        print("Press Ctrl+C to stop the dashboard")
        print("=" * 60)
        
        try:
            while self.is_running:
                self.clear_screen()
                self.display_dashboard()
                time.sleep(self.refresh_interval)
                
        except KeyboardInterrupt:
            print("\n\n🛑 Dashboard stopped by user")
            self.stop_dashboard()
        except Exception as e:
            print(f"\n💥 Dashboard error: {e}")
            self.stop_dashboard()
    
    def stop_dashboard(self):
        """Stop the dashboard."""
        self.is_running = False
        if self.performance_monitor.is_monitoring:
            self.performance_monitor.stop_monitoring()
        print("✅ Performance monitoring stopped")
    
    def clear_screen(self):
        """Clear the terminal screen."""
        os.system('cls' if os.name == 'nt' else 'clear')
    
    def display_dashboard(self):
        """Display the main dashboard."""
        # Header
        print("🚀 Dynamic Learning System - Performance Dashboard")
        print(f"📅 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 60)
        
        # System Health
        self.display_system_health()
        
        # Performance Metrics
        self.display_performance_metrics()
        
        # Learning Analytics
        self.display_learning_analytics()
        
        # MongoDB Status
        self.display_mongodb_status()
        
        # Recent Activity
        self.display_recent_activity()
        
        # Footer
        print("=" * 60)
        print(f"🔄 Auto-refresh every {self.refresh_interval}s | Press Ctrl+C to stop")
    
    def display_system_health(self):
        """Display system health information."""
        print("\n🏥 SYSTEM HEALTH")
        print("-" * 30)
        
        try:
            dashboard_data = self.performance_monitor.get_realtime_dashboard_data()
            system = dashboard_data['system']
            
            # System status with color coding
            status = system['status']
            if status == 'HEALTHY':
                status_display = "✅ HEALTHY"
            elif status == 'WARNING':
                status_display = "⚠️ WARNING"
            else:
                status_display = "🔴 CRITICAL"
            
            print(f"Status:     {status_display}")
            print(f"CPU Usage:  {system['cpu_percent']:6.1f}%")
            print(f"Memory:     {system['memory_percent']:6.1f}%")
            print(f"Uptime:     {self.format_duration(dashboard_data['uptime'])}")
            
        except Exception as e:
            print(f"❌ Error getting system health: {e}")
    
    def display_performance_metrics(self):
        """Display performance metrics."""
        print("\n⚡ PERFORMANCE METRICS")
        print("-" * 30)
        
        try:
            dashboard_data = self.performance_monitor.get_realtime_dashboard_data()
            queries = dashboard_data['queries']
            
            print(f"Total Queries:      {queries['total']:6d}")
            print(f"Successful:         {queries['successful']:6d}")
            print(f"Failed:             {queries['failed']:6d}")
            print(f"Success Rate:       {queries['success_rate']:6.1f}%")
            
            # Get detailed performance summary
            summary = self.performance_monitor.get_performance_summary(timedelta(minutes=5))
            if 'query_performance' in summary and summary['query_performance']['count'] > 0:
                perf = summary['query_performance']
                print(f"Avg Response Time:  {perf['avg_duration']:6.3f}s")
                print(f"Min Response Time:  {perf['min_duration']:6.3f}s")
                print(f"Max Response Time:  {perf['max_duration']:6.3f}s")
            
        except Exception as e:
            print(f"❌ Error getting performance metrics: {e}")
    
    def display_learning_analytics(self):
        """Display learning analytics."""
        print("\n🧠 LEARNING ANALYTICS")
        print("-" * 30)
        
        try:
            dashboard_data = self.performance_monitor.get_realtime_dashboard_data()
            learning = dashboard_data['learning']
            
            print(f"Patterns Learned:   {learning['patterns_learned']:6d}")
            print(f"Patterns Evolved:   {learning['patterns_evolved']:6d}")
            print(f"User Feedback:      {learning['user_feedback']:6d}")
            
            # Get learning metrics for recent time window
            summary = self.performance_monitor.get_performance_summary(timedelta(minutes=10))
            if 'learning_metrics' in summary:
                learning_metrics = summary['learning_metrics']
                if learning_metrics['events'] > 0:
                    print(f"Recent Learning:    {learning_metrics['events']:6d} events")
                    print(f"Patterns Updated:   {learning_metrics['patterns_updated']:6d}")
                    
                    # Show intent distribution
                    intent_dist = learning_metrics.get('intent_distribution', {})
                    if intent_dist:
                        print(f"Intent Distribution:")
                        for intent, count in sorted(intent_dist.items(), key=lambda x: x[1], reverse=True)[:3]:
                            print(f"  • {intent}: {count}")
            
        except Exception as e:
            print(f"❌ Error getting learning analytics: {e}")
    
    def display_mongodb_status(self):
        """Display MongoDB connection status."""
        print("\n🔌 MONGODB STATUS")
        print("-" * 30)
        
        try:
            is_available = self.schema_manager.is_mongodb_available()
            
            if is_available:
                print("Status:     ✅ CONNECTED")
                
                # Test collection access
                try:
                    patterns = self.schema_manager.get_hybrid_qa_patterns(self.collection_name)
                    print(f"Collection: {self.collection_name}")
                    print(f"Patterns:   {len(patterns)} total")
                    
                    # Get analytics if available
                    try:
                        analytics = self.schema_manager.get_pattern_analytics(self.collection_name)
                        if analytics and analytics.get('total_patterns', 0) > 0:
                            print(f"Active:     {analytics.get('active_patterns', 0)}")
                            print(f"Usage:      {analytics.get('total_usage', 0)}")
                            print(f"Avg Confidence: {analytics.get('avg_confidence', 0):.2f}")
                    except Exception:
                        pass  # Analytics might not be available yet
                        
                except Exception as e:
                    print(f"Collection: ❌ Access failed: {e}")
            else:
                print("Status:     ❌ DISCONNECTED")
                print("Note:       Dynamic learning features disabled")
                
        except Exception as e:
            print(f"❌ Error checking MongoDB status: {e}")
    
    def display_recent_activity(self):
        """Display recent system activity."""
        print("\n📊 RECENT ACTIVITY")
        print("-" * 30)
        
        try:
            # Get recent queries
            summary = self.performance_monitor.get_performance_summary(timedelta(minutes=5))
            
            if 'query_performance' in summary and summary['query_performance']['count'] > 0:
                print(f"Last 5 minutes: {summary['query_performance']['count']} queries")
                
                # Get recent learning events
                if 'learning_metrics' in summary and summary['learning_metrics']['events'] > 0:
                    print(f"Learning events: {summary['learning_metrics']['events']}")
                
                # Get recent pattern evolution
                if 'pattern_metrics' in summary and summary['pattern_metrics']['evolution_events'] > 0:
                    print(f"Pattern evolution: {summary['pattern_metrics']['evolution_events']} events")
                
                # Get recent feedback
                if 'pattern_metrics' in summary and summary['pattern_metrics']['feedback_count'] > 0:
                    avg_rating = summary['pattern_metrics']['avg_rating']
                    print(f"User feedback: {summary['pattern_metrics']['feedback_count']} ratings (avg: {avg_rating:.1f}/5)")
            else:
                print("No recent activity")
                
        except Exception as e:
            print(f"❌ Error getting recent activity: {e}")
    
    def format_duration(self, seconds):
        """Format duration in human-readable format."""
        if seconds < 60:
            return f"{seconds:.0f}s"
        elif seconds < 3600:
            minutes = seconds / 60
            return f"{minutes:.0f}m {seconds % 60:.0f}s"
        else:
            hours = seconds / 3600
            minutes = (seconds % 3600) / 60
            return f"{hours:.0f}h {minutes:.0f}m"
    
    def export_metrics(self):
        """Export current metrics to file."""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            export_file = f"dashboard_export_{timestamp}.json"
            
            # Get all available data
            dashboard_data = self.performance_monitor.get_realtime_dashboard_data()
            performance_summary = self.performance_monitor.get_performance_summary(timedelta(hours=1))
            
            export_data = {
                'export_timestamp': datetime.now().isoformat(),
                'dashboard_data': dashboard_data,
                'performance_summary': performance_summary,
                'mongodb_status': {
                    'available': self.schema_manager.is_mongodb_available(),
                    'collection_name': self.collection_name
                }
            }
            
            with open(export_file, 'w') as f:
                json.dump(export_data, f, indent=2, default=str)
            
            print(f"📄 Metrics exported to: {export_file}")
            return export_file
            
        except Exception as e:
            print(f"❌ Failed to export metrics: {e}")
            return None

def main():
    """Main function to run the dashboard."""
    try:
        dashboard = PerformanceDashboard()
        dashboard.start_dashboard()
    except Exception as e:
        print(f"💥 Dashboard failed to start: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code) 