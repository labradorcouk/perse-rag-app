#!/bin/bash

echo "🚀 Fabric RAG Application - Quick Install"
echo "=========================================="

# Check if package exists
if [ ! -f "fabric-rag-deployment-v1.0.0.tar.gz" ]; then
    echo "❌ Package not found. Please download fabric-rag-deployment-v1.0.0.tar.gz first."
    exit 1
fi

# Extract package
echo "📦 Extracting package..."
tar -xzf fabric-rag-deployment-v1.0.0.tar.gz
cd fabric-rag-deployment

# Run deployment
echo "🔧 Running deployment..."
chmod +x deploy.sh
./deploy.sh

echo ""
echo "✅ Installation completed!"
echo "🌐 Access the application at: http://localhost:8501"
echo "📋 For help, see README.md" 