#!/bin/bash

# System Optimization Script for CentOS
# This script optimizes the system to reduce I/O pressure and improve Docker performance

echo "🔧 Starting system optimization for RAG application..."

# 1. Update system packages
echo "📦 Updating system packages..."
sudo yum update -y

# 2. Install performance monitoring tools
echo "📊 Installing performance monitoring tools..."
sudo yum install -y htop iotop iostat sysstat

# 3. Optimize disk I/O
echo "💾 Optimizing disk I/O..."

# Create optimized mount points for Docker data
sudo mkdir -p /opt/qdrant/data /opt/qdrant/snapshots
sudo mkdir -p /opt/rag-app/data /opt/rag-app/cache /opt/rag-app/logs

# Set proper permissions
sudo chown -R 1000:1000 /opt/qdrant/data /opt/qdrant/snapshots
sudo chown -R 1000:1000 /opt/rag-app/data /opt/rag-app/cache /opt/rag-app/logs

# 4. Optimize kernel parameters for better I/O performance
echo "⚙️ Optimizing kernel parameters..."

# Add kernel optimizations to sysctl.conf
cat << EOF | sudo tee -a /etc/sysctl.conf

# I/O Optimization
vm.dirty_ratio = 15
vm.dirty_background_ratio = 5
vm.dirty_expire_centisecs = 3000
vm.dirty_writeback_centisecs = 500

# Memory optimization
vm.swappiness = 10
vm.vfs_cache_pressure = 50

# Network optimization
net.core.rmem_max = 16777216
net.core.wmem_max = 16777216
net.ipv4.tcp_rmem = 4096 87380 16777216
net.ipv4.tcp_wmem = 4096 65536 16777216

# File system optimization
fs.file-max = 1000000
fs.inotify.max_user_watches = 524288

# Docker optimization
net.bridge.bridge-nf-call-iptables = 1
net.bridge.bridge-nf-call-ip6tables = 1
EOF

# Apply kernel parameters
sudo sysctl -p

# 5. Optimize Docker daemon
echo "🐳 Optimizing Docker daemon..."

# Create optimized Docker daemon configuration
sudo mkdir -p /etc/docker
cat << EOF | sudo tee /etc/docker/daemon.json
{
  "storage-driver": "overlay2",
  "storage-opts": [
    "overlay2.override_kernel_check=true"
  ],
  "log-driver": "json-file",
  "log-opts": {
    "max-size": "10m",
    "max-file": "3"
  },
  "default-ulimits": {
    "nofile": {
      "Hard": 64000,
      "Name": "nofile",
      "Soft": 64000
    }
  },
  "max-concurrent-downloads": 10,
  "max-concurrent-uploads": 5,
  "experimental": false,
  "metrics-addr": "127.0.0.1:9323",
  "insecure-registries": [],
  "registry-mirrors": []
}
EOF

# 6. Create systemd service for performance monitoring
echo "📈 Creating performance monitoring service..."

cat << EOF | sudo tee /etc/systemd/system/performance-monitor.service
[Unit]
Description=Performance Monitoring Service
After=network.target

[Service]
Type=simple
User=root
ExecStart=/usr/bin/python3 /opt/rag-app/monitor_performance.py
Restart=always
RestartSec=60

[Install]
WantedBy=multi-user.target
EOF

# 7. Optimize file system
echo "📁 Optimizing file system..."

# Add noatime option to reduce I/O
sudo sed -i 's/defaults/defaults,noatime/' /etc/fstab

# 8. Create performance monitoring script
echo "📊 Creating performance monitoring script..."

cat << 'EOF' > /opt/rag-app/monitor_performance.py
#!/usr/bin/env python3
"""
Performance monitoring script for RAG application.
"""

import time
import psutil
import subprocess
import json
from datetime import datetime

def get_docker_stats():
    """Get Docker container statistics."""
    try:
        result = subprocess.run(['docker', 'stats', '--no-stream', '--format', 'json'], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            return result.stdout.strip().split('\n')
    except Exception as e:
        print(f"Error getting Docker stats: {e}")
    return []

def log_performance():
    """Log system performance metrics."""
    timestamp = datetime.now().isoformat()
    
    # System metrics
    cpu_percent = psutil.cpu_percent(interval=1)
    memory = psutil.virtual_memory()
    disk = psutil.disk_usage('/')
    
    # Docker stats
    docker_stats = get_docker_stats()
    
    # Log metrics
    metrics = {
        'timestamp': timestamp,
        'cpu_percent': cpu_percent,
        'memory_percent': memory.percent,
        'disk_percent': disk.percent,
        'docker_stats': docker_stats
    }
    
    with open('/opt/rag-app/logs/performance.log', 'a') as f:
        f.write(json.dumps(metrics) + '\n')
    
    print(f"Performance logged: CPU={cpu_percent}%, Memory={memory.percent}%, Disk={disk.percent}%")

def main():
    """Main monitoring loop."""
    print("Starting performance monitoring...")
    
    while True:
        try:
            log_performance()
            time.sleep(60)  # Log every minute
        except KeyboardInterrupt:
            print("Performance monitoring stopped.")
            break
        except Exception as e:
            print(f"Error in performance monitoring: {e}")
            time.sleep(60)

if __name__ == "__main__":
    main()
EOF

chmod +x /opt/rag-app/monitor_performance.py

# 9. Create log rotation configuration
echo "📝 Creating log rotation configuration..."

cat << EOF | sudo tee /etc/logrotate.d/rag-app
/opt/rag-app/logs/*.log {
    daily
    missingok
    rotate 7
    compress
    delaycompress
    notifempty
    create 644 root root
    postrotate
        systemctl reload docker
    endscript
}
EOF

# 10. Restart services
echo "🔄 Restarting services..."

sudo systemctl daemon-reload
sudo systemctl restart docker
sudo systemctl enable performance-monitor.service
sudo systemctl start performance-monitor.service

# 11. Set up monitoring
echo "📊 Setting up monitoring..."

# Create monitoring directory
sudo mkdir -p /opt/rag-app/logs
sudo chown -R 1000:1000 /opt/rag-app/logs

# 12. Final optimizations
echo "🎯 Final optimizations..."

# Increase file descriptor limits
echo "* soft nofile 64000" | sudo tee -a /etc/security/limits.conf
echo "* hard nofile 64000" | sudo tee -a /etc/security/limits.conf

# Optimize swap
sudo swapon --show
if [ $? -eq 0 ]; then
    echo "Swap is enabled. Consider disabling if you have sufficient RAM."
fi

echo "✅ System optimization completed!"
echo ""
echo "📋 Summary of optimizations:"
echo "- Updated system packages"
echo "- Optimized kernel parameters for I/O"
echo "- Configured Docker daemon for better performance"
echo "- Created persistent storage directories"
echo "- Set up performance monitoring"
echo "- Configured log rotation"
echo "- Optimized file system settings"
echo ""
echo "🚀 Next steps:"
echo "1. Rebuild and deploy your Docker containers"
echo "2. Monitor performance with: docker stats"
echo "3. Check logs: tail -f /opt/rag-app/logs/performance.log"
echo "4. Monitor system: htop, iotop" 