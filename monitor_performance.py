#!/usr/bin/env python3
"""
Performance Monitoring Dashboard for RAG Application
"""

import psutil
import docker
import time
import json
from datetime import datetime
import subprocess

class PerformanceMonitor:
    """Monitor system and Docker performance."""
    
    def __init__(self):
        self.docker_client = docker.from_env()
        
    def get_system_metrics(self):
        """Get system performance metrics."""
        return {
            'cpu_percent': psutil.cpu_percent(interval=1),
            'memory_percent': psutil.virtual_memory().percent,
            'disk_percent': psutil.disk_usage('/').percent,
            'load_average': psutil.getloadavg(),
            'timestamp': datetime.now().isoformat()
        }
    
    def get_docker_metrics(self):
        """Get Docker container metrics."""
        try:
            containers = self.docker_client.containers.list()
            metrics = {}
            
            for container in containers:
                stats = container.stats(stream=False)
                metrics[container.name] = {
                    'cpu_percent': self._calculate_cpu_percent(stats),
                    'memory_usage': stats['memory_stats']['usage'],
                    'memory_limit': stats['memory_stats']['limit'],
                    'network_rx': stats['networks']['eth0']['rx_bytes'],
                    'network_tx': stats['networks']['eth0']['tx_bytes']
                }
            
            return metrics
        except Exception as e:
            print(f"Error getting Docker metrics: {e}")
            return {}
    
    def _calculate_cpu_percent(self, stats):
        """Calculate CPU percentage from Docker stats."""
        try:
            cpu_delta = stats['cpu_stats']['cpu_usage']['total_usage'] - \
                       stats['precpu_stats']['cpu_usage']['total_usage']
            system_delta = stats['cpu_stats']['system_cpu_usage'] - \
                          stats['precpu_stats']['system_cpu_usage']
            
            if system_delta > 0:
                return (cpu_delta / system_delta) * 100
            return 0
        except:
            return 0
    
    def get_io_stats(self):
        """Get I/O statistics."""
        try:
            result = subprocess.run(['iostat', '-x', '1', '1'], 
                                  capture_output=True, text=True)
            return result.stdout
        except:
            return "iostat not available"
    
    def log_metrics(self, metrics):
        """Log metrics to file."""
        with open('/opt/rag-app/logs/performance.log', 'a') as f:
            f.write(json.dumps(metrics) + '\n')
    
    def print_dashboard(self):
        """Print performance dashboard."""
        print("=" * 80)
        print("🚀 RAG Application Performance Dashboard")
        print("=" * 80)
        
        # System metrics
        system_metrics = self.get_system_metrics()
        print(f"\n📊 System Metrics:")
        print(f"   CPU Usage: {system_metrics['cpu_percent']:.1f}%")
        print(f"   Memory Usage: {system_metrics['memory_percent']:.1f}%")
        print(f"   Disk Usage: {system_metrics['disk_percent']:.1f}%")
        print(f"   Load Average: {system_metrics['load_average']}")
        
        # Docker metrics
        docker_metrics = self.get_docker_metrics()
        print(f"\n🐳 Docker Container Metrics:")
        for container_name, metrics in docker_metrics.items():
            memory_mb = metrics['memory_usage'] / 1024 / 1024
            memory_limit_mb = metrics['memory_limit'] / 1024 / 1024
            print(f"   {container_name}:")
            print(f"     CPU: {metrics['cpu_percent']:.1f}%")
            print(f"     Memory: {memory_mb:.1f}MB / {memory_limit_mb:.1f}MB")
            print(f"     Network RX: {metrics['network_rx'] / 1024:.1f}KB")
            print(f"     Network TX: {metrics['network_tx'] / 1024:.1f}KB")
        
        # I/O stats
        print(f"\n💾 I/O Statistics:")
        io_stats = self.get_io_stats()
        print(io_stats)
        
        print(f"\n⏰ Last Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 80)

def main():
    """Main monitoring loop."""
    monitor = PerformanceMonitor()
    
    print("Starting performance monitoring...")
    print("Press Ctrl+C to stop")
    
    try:
        while True:
            monitor.print_dashboard()
            
            # Log metrics
            metrics = {
                'system': monitor.get_system_metrics(),
                'docker': monitor.get_docker_metrics(),
                'timestamp': datetime.now().isoformat()
            }
            monitor.log_metrics(metrics)
            
            time.sleep(30)  # Update every 30 seconds
            
    except KeyboardInterrupt:
        print("\nPerformance monitoring stopped.")

if __name__ == "__main__":
    main() 