#!/bin/bash

# Test Docker Build Script
# This script tests the Docker build without ODBC to verify basic functionality

echo "🧪 Testing Docker Build (No ODBC)"
echo "=================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

# Test the no-ODBC build
print_status "Building Docker image without ODBC support..."
docker build -f Dockerfile.docker.no-odbc -t rag-fabric-app:test .

if [ $? -eq 0 ]; then
    print_status "✅ Docker build successful!"
    
    print_info "Testing container startup..."
    docker run --rm -d --name test-rag-app -p 8502:8501 rag-fabric-app:test
    
    if [ $? -eq 0 ]; then
        print_status "✅ Container started successfully!"
        
        # Wait a bit for the app to start
        print_info "Waiting for application to start..."
        sleep 10
        
        # Test if the app is responding
        if curl -f http://localhost:8502/_stcore/health &> /dev/null; then
            print_status "✅ Application is responding!"
            print_info "🌐 Test the application at: http://localhost:8502"
            print_info "Login with: mawaz@opendata.energy"
        else
            print_warning "Application might still be starting..."
            print_info "Check logs with: docker logs test-rag-app"
        fi
        
        # Stop the test container
        docker stop test-rag-app
        print_status "Test container stopped."
        
    else
        print_error "Failed to start container."
    fi
    
    print_info "Build test completed successfully!"
    print_info "To run the full application:"
    echo "  docker-compose up -d"
    echo "  Access at: http://localhost:8501"
    
else
    print_error "Docker build failed!"
    print_info "Try the fix script: ./fix_odbc_issue.sh"
fi 