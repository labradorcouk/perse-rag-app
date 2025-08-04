#!/bin/bash

# Fabric RAG Application Deployment Script
# Version: 1.0.0

set -e

echo "🚀 Fabric RAG Application Deployment"
echo "====================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if Docker is installed
check_docker() {
    print_status "Checking Docker installation..."
    if ! command -v docker &> /dev/null; then
        print_error "Docker is not installed. Please install Docker first."
        exit 1
    fi
    
    if ! command -v docker-compose &> /dev/null; then
        print_error "Docker Compose is not installed. Please install Docker Compose first."
        exit 1
    fi
    
    print_status "Docker and Docker Compose are installed."
}

# Check if the Docker image exists
check_image() {
    print_status "Checking Docker image..."
    if ! docker images | grep -q "fabric-rag-app"; then
        print_warning "Docker image not found. Loading from tar file..."
        if [ -f "fabric-rag-app-v1.0.0.tar" ]; then
            docker load -i fabric-rag-app-v1.0.0.tar
            print_status "Docker image loaded successfully."
        else
            print_error "Docker image tar file not found: fabric-rag-app-v1.0.0.tar"
            exit 1
        fi
    else
        print_status "Docker image found."
    fi
}

# Check environment file
check_env() {
    print_status "Checking environment configuration..."
    if [ ! -f ".env" ]; then
        if [ -f ".env.example" ]; then
            print_warning ".env file not found. Creating from .env.example..."
            cp .env.example .env
            print_warning "Please edit .env file with your configuration before continuing."
            echo ""
            echo "Required environment variables:"
            echo "- AZURE_CLIENT_ID"
            echo "- AZURE_CLIENT_SECRET" 
            echo "- AZURE_TENANT_ID"
            echo "- OPENAI_API_KEY"
            echo "- AUTHORIZED_USERS"
            echo ""
            read -p "Press Enter after editing .env file to continue..."
        else
            print_error ".env.example file not found."
            exit 1
        fi
    else
        print_status ".env file found."
    fi
}

# Check port availability
check_port() {
    print_status "Checking port 8501 availability..."
    if netstat -tulpn 2>/dev/null | grep -q ":8501 "; then
        print_warning "Port 8501 is already in use. Please stop the conflicting service or change the port in docker-compose.yml"
        exit 1
    fi
    print_status "Port 8501 is available."
}

# Deploy the application
deploy_app() {
    print_status "Deploying Fabric RAG application..."
    
    # Stop any existing containers
    docker-compose down 2>/dev/null || true
    
    # Start the services
    docker-compose up -d
    
    # Wait for services to start
    print_status "Waiting for services to start..."
    sleep 10
    
    # Check if services are running
    if docker-compose ps | grep -q "Up"; then
        print_status "Application deployed successfully!"
        echo ""
        echo "🌐 Access the application at: http://localhost:8501"
        echo "📊 Check service status: docker-compose ps"
        echo "📋 View logs: docker-compose logs rag-fabric-app"
        echo ""
    else
        print_error "Failed to deploy application. Check logs with: docker-compose logs"
        exit 1
    fi
}

# Main deployment function
main() {
    echo ""
    check_docker
    check_image
    check_env
    check_port
    deploy_app
    echo ""
    print_status "Deployment completed successfully!"
}

# Run main function
main "$@" 