#!/bin/bash

# RAG Fabric App Docker Deployment Script
# This script automates the Docker deployment process

set -e  # Exit on any error

echo "🚀 RAG Fabric App Docker Deployment"
echo "=================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
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

print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
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

# Check if .env file exists
check_env_file() {
    print_status "Checking environment configuration..."
    if [ ! -f ".env" ]; then
        print_warning ".env file not found. Creating from template..."
        if [ -f ".env.example.docker" ]; then
            cp .env.example.docker .env
            print_warning "Please update the .env file with your actual values:"
            print_warning "  - AZURE_TENANT_ID"
            print_warning "  - AZURE_CLIENT_ID" 
            print_warning "  - AZURE_CLIENT_SECRET"
            print_warning "  - OPENAI_API_KEY"
            echo ""
            read -p "Press Enter after updating the .env file..."
        else
            print_error ".env.example.docker not found. Please create a .env file manually."
            exit 1
        fi
    else
        print_status ".env file found."
    fi
}

# Build Docker image with ODBC support
build_image_with_odbc() {
    print_status "Building Docker image with ODBC support..."
    docker build -f Dockerfile.docker -t rag-fabric-app .
    if [ $? -eq 0 ]; then
        print_status "Docker image built successfully with ODBC support."
    else
        print_error "Failed to build Docker image with ODBC support."
        return 1
    fi
}

# Build Docker image without ODBC (for testing)
build_image_without_odbc() {
    print_status "Building Docker image without ODBC support (for testing)..."
    docker build -f Dockerfile.docker.no-odbc -t rag-fabric-app:no-odbc .
    if [ $? -eq 0 ]; then
        print_status "Docker image built successfully without ODBC support."
    else
        print_error "Failed to build Docker image without ODBC support."
        return 1
    fi
}

# Build Docker image with alternative ODBC method
build_image_alternative_odbc() {
    print_status "Building Docker image with alternative ODBC method..."
    docker build -f Dockerfile.docker.alternative -t rag-fabric-app:alt-odbc .
    if [ $? -eq 0 ]; then
        print_status "Docker image built successfully with alternative ODBC method."
    else
        print_error "Failed to build Docker image with alternative ODBC method."
        return 1
    fi
}

# Build Docker image with smart fallback
build_image() {
    print_status "Attempting to build Docker image..."
    
    # Try the main Dockerfile first
    if build_image_with_odbc; then
        print_status "✅ Successfully built with ODBC support!"
        return 0
    fi
    
    print_warning "ODBC installation failed. Trying alternative method..."
    
    # Try alternative ODBC method
    if build_image_alternative_odbc; then
        print_status "✅ Successfully built with alternative ODBC method!"
        # Update docker-compose to use the alternative image
        sed -i 's/rag-fabric-app:/rag-fabric-app:alt-odbc:/' docker-compose.yml
        return 0
    fi
    
    print_warning "Alternative ODBC method also failed. Building without ODBC for testing..."
    
    # Build without ODBC as last resort
    if build_image_without_odbc; then
        print_status "✅ Successfully built without ODBC support (SQL features will be limited)."
        # Update docker-compose to use the no-odbc image
        sed -i 's/rag-fabric-app:/rag-fabric-app:no-odbc:/' docker-compose.yml
        return 0
    fi
    
    print_error "All build methods failed. Please check your Docker installation and try again."
    return 1
}

# Start services
start_services() {
    print_status "Starting services..."
    
    # Check if user wants to run with local Qdrant
    read -p "Do you want to run with local Qdrant? (y/n): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        print_status "Starting with local Qdrant..."
        docker-compose --profile qdrant-local up -d
    else
        print_status "Starting without local Qdrant (using external)..."
        docker-compose up rag-fabric-app -d
    fi
}

# Check service health
check_health() {
    print_status "Checking service health..."
    
    # Wait a bit for services to start
    sleep 10
    
    # Check if the app is responding
    if curl -f http://localhost:8501/_stcore/health &> /dev/null; then
        print_status "✅ Application is healthy and running!"
        print_status "🌐 Access the application at: http://localhost:8501"
    else
        print_warning "Application might still be starting up..."
        print_status "Check logs with: docker-compose logs rag-fabric-app"
    fi
}

# Show logs
show_logs() {
    print_status "Recent application logs:"
    docker-compose logs --tail=20 rag-fabric-app
}

# Fix ODBC issues
fix_odbc_issues() {
    print_info "ODBC Installation Troubleshooting Guide:"
    echo ""
    echo "If you're experiencing ODBC installation issues, try these solutions:"
    echo ""
    echo "1. Use the alternative Dockerfile:"
    echo "   docker build -f Dockerfile.docker.alternative -t rag-fabric-app ."
    echo ""
    echo "2. Build without ODBC for testing:"
    echo "   docker build -f Dockerfile.docker.no-odbc -t rag-fabric-app ."
    echo ""
    echo "3. Manual ODBC installation in container:"
    echo "   docker run -it rag-fabric-app:no-odbc bash"
    echo "   # Then manually install ODBC drivers inside the container"
    echo ""
    echo "4. Use a different base image:"
    echo "   # Modify Dockerfile to use ubuntu:20.04 instead of python:3.11-slim"
    echo ""
    echo "5. Install ODBC drivers at runtime:"
    echo "   # Add a startup script that installs ODBC drivers when container starts"
    echo ""
    print_warning "Note: Without ODBC drivers, SQL Editor functionality will be limited."
}

# Main deployment function
deploy() {
    print_status "Starting deployment process..."
    
    check_docker
    check_env_file
    
    # Ask user about ODBC build preference
    echo ""
    print_info "ODBC Driver Installation Options:"
    echo "1. Try standard build (recommended)"
    echo "2. Use alternative ODBC method"
    echo "3. Build without ODBC (for testing)"
    echo "4. Show troubleshooting guide"
    echo ""
    read -p "Choose an option (1-4): " -n 1 -r
    echo
    
    case $REPLY in
        1)
            build_image
            ;;
        2)
            build_image_alternative_odbc
            sed -i 's/rag-fabric-app:/rag-fabric-app:alt-odbc:/' docker-compose.yml
            ;;
        3)
            build_image_without_odbc
            sed -i 's/rag-fabric-app:/rag-fabric-app:no-odbc:/' docker-compose.yml
            ;;
        4)
            fix_odbc_issues
            exit 0
            ;;
        *)
            print_error "Invalid option. Using standard build."
            build_image
            ;;
    esac
    
    start_services
    check_health
    
    echo ""
    print_status "Deployment completed!"
    print_status "Useful commands:"
    echo "  View logs: docker-compose logs -f rag-fabric-app"
    echo "  Stop services: docker-compose down"
    echo "  Restart: docker-compose restart rag-fabric-app"
    echo "  Shell access: docker-compose exec rag-fabric-app bash"
}

# Cleanup function
cleanup() {
    print_status "Cleaning up Docker resources..."
    docker-compose down
    docker system prune -f
    print_status "Cleanup completed."
}

# Show help
show_help() {
    echo "Usage: $0 [COMMAND]"
    echo ""
    echo "Commands:"
    echo "  deploy    - Deploy the application (default)"
    echo "  build     - Build Docker image only"
    echo "  start     - Start services only"
    echo "  stop      - Stop all services"
    echo "  logs      - Show application logs"
    echo "  cleanup   - Clean up Docker resources"
    echo "  health    - Check service health"
    echo "  fix-odbc  - Show ODBC troubleshooting guide"
    echo "  help      - Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0              # Deploy the application"
    echo "  $0 build        # Build image only"
    echo "  $0 logs         # Show logs"
    echo "  $0 fix-odbc     # Show ODBC troubleshooting"
}

# Main script logic
case "${1:-deploy}" in
    "deploy")
        deploy
        ;;
    "build")
        check_docker
        build_image
        ;;
    "start")
        check_docker
        check_env_file
        start_services
        ;;
    "stop")
        print_status "Stopping services..."
        docker-compose down
        ;;
    "logs")
        show_logs
        ;;
    "cleanup")
        cleanup
        ;;
    "health")
        check_health
        ;;
    "fix-odbc")
        fix_odbc_issues
        ;;
    "help"|"-h"|"--help")
        show_help
        ;;
    *)
        print_error "Unknown command: $1"
        show_help
        exit 1
        ;;
esac 