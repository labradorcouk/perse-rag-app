#!/bin/bash

# Optimized Deployment Script for RAG Application
# This script deploys the application with performance optimizations

echo "🚀 Starting optimized deployment..."

# 1. Stop existing containers
echo "🛑 Stopping existing containers..."
docker-compose down

# 2. Clean up old images and containers
echo "🧹 Cleaning up old images and containers..."
docker system prune -f
docker volume prune -f

# 3. Create optimized directories
echo "📁 Creating optimized directories..."
sudo mkdir -p /opt/qdrant/data /opt/qdrant/snapshots
sudo mkdir -p /opt/rag-app/data /opt/rag-app/cache /opt/rag-app/logs
sudo chown -R 1000:1000 /opt/qdrant/data /opt/qdrant/snapshots
sudo chown -R 1000:1000 /opt/rag-app/data /opt/rag-app/cache /opt/rag-app/logs

# 4. Build optimized image
echo "🔨 Building optimized Docker image..."
docker build -f Dockerfile.docker.optimized -t rag-fabric-app:optimized .

# 5. Deploy with optimized configuration
echo "🚀 Deploying with optimized configuration..."
docker-compose -f docker-compose.optimized.yml up -d

# 6. Wait for services to be healthy
echo "⏳ Waiting for services to be healthy..."
sleep 30

# 7. Check service health
echo "🏥 Checking service health..."
docker-compose -f docker-compose.optimized.yml ps

# 8. Monitor performance
echo "📊 Monitoring performance..."
echo "Docker stats:"
docker stats --no-stream

echo "System resources:"
free -h
df -h

echo "✅ Optimized deployment completed!"
echo ""
echo "📋 Deployment summary:"
echo "- Built optimized Docker image"
echo "- Deployed with resource limits"
echo "- Configured persistent storage"
echo "- Set up health monitoring"
echo ""
echo "🔍 Monitoring commands:"
echo "- Container status: docker-compose -f docker-compose.optimized.yml ps"
echo "- Container logs: docker-compose -f docker-compose.optimized.yml logs"
echo "- Performance stats: docker stats"
echo "- System monitoring: htop, iotop"
echo ""
echo "🌐 Application URL: http://localhost:8501" 