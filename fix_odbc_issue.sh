#!/bin/bash

# Quick ODBC Issue Fix Script
# This script provides alternative methods to build the Docker image when ODBC installation fails

echo "🔧 ODBC Installation Issue Fix"
echo "=============================="

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

echo ""
print_info "The ODBC installation is failing due to package conflicts. Here are your options:"
echo ""

echo "Option 1: Build without ODBC (Quick Test)"
echo "  This will build the app without SQL Server connectivity for testing:"
echo "  docker build -f Dockerfile.docker.no-odbc -t rag-fabric-app ."
echo ""

echo "Option 2: Use Alternative ODBC Method"
echo "  This uses a different approach to install ODBC drivers:"
echo "  docker build -f Dockerfile.docker.alternative -t rag-fabric-app ."
echo ""

echo "Option 3: Manual ODBC Installation"
echo "  Build without ODBC first, then install manually:"
echo "  docker build -f Dockerfile.docker.no-odbc -t rag-fabric-app ."
echo "  docker run -it rag-fabric-app bash"
echo "  # Then manually install ODBC drivers inside the container"
echo ""

echo "Option 4: Use Ubuntu Base Image"
echo "  Create a new Dockerfile using ubuntu:20.04 as base"
echo ""

read -p "Which option would you like to try? (1-4): " -n 1 -r
echo

case $REPLY in
    1)
        print_status "Building without ODBC support..."
        docker build -f Dockerfile.docker.no-odbc -t rag-fabric-app .
        if [ $? -eq 0 ]; then
            print_status "✅ Successfully built without ODBC!"
            print_warning "Note: SQL Editor functionality will be limited."
            echo ""
            print_status "To run the application:"
            echo "docker-compose up -d"
            echo "Access at: http://localhost:8501"
        else
            print_error "Build failed. Please check your Docker installation."
        fi
        ;;
    2)
        print_status "Building with alternative ODBC method..."
        docker build -f Dockerfile.docker.alternative -t rag-fabric-app .
        if [ $? -eq 0 ]; then
            print_status "✅ Successfully built with alternative ODBC method!"
            echo ""
            print_status "To run the application:"
            echo "docker-compose up -d"
            echo "Access at: http://localhost:8501"
        else
            print_error "Alternative build failed. Try option 1 (without ODBC)."
        fi
        ;;
    3)
        print_status "Building base image without ODBC..."
        docker build -f Dockerfile.docker.no-odbc -t rag-fabric-app .
        if [ $? -eq 0 ]; then
            print_status "✅ Base image built successfully!"
            echo ""
            print_info "Now you can manually install ODBC drivers:"
            echo "docker run -it rag-fabric-app bash"
            echo ""
            echo "Inside the container, run:"
            echo "curl https://packages.microsoft.com/keys/microsoft.asc | apt-key add -"
            echo "curl https://packages.microsoft.com/config/debian/11/prod.list > /etc/apt/sources.list.d/mssql-release.list"
            echo "apt-get update"
            echo "ACCEPT_EULA=Y apt-get install -y msodbcsql18"
            echo ""
            print_warning "After installing ODBC, commit the container as a new image:"
            echo "docker commit <container_id> rag-fabric-app:with-odbc"
        else
            print_error "Base build failed. Please check your Docker installation."
        fi
        ;;
    4)
        print_status "Creating Ubuntu-based Dockerfile..."
        cat > Dockerfile.docker.ubuntu << 'EOF'
# Use Ubuntu 20.04 as base
FROM ubuntu:20.04

# Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    gcc \
    g++ \
    curl \
    gnupg2 \
    wget \
    ca-certificates \
    unixodbc \
    unixodbc-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Microsoft ODBC Driver 18
RUN curl https://packages.microsoft.com/keys/microsoft.asc | apt-key add - \
    && curl https://packages.microsoft.com/config/ubuntu/20.04/prod.list > /etc/apt/sources.list.d/mssql-release.list \
    && apt-get update \
    && ACCEPT_EULA=Y apt-get install -y msodbcsql18 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements files
COPY requirements.txt .
COPY requirements_verisk.txt .

# Install Python dependencies
RUN pip3 install --no-cache-dir -r requirements.txt
RUN pip3 install --no-cache-dir -r requirements_verisk.txt

# Copy application files
COPY . .

# Create necessary directories
RUN mkdir -p logs downloads models utils config

# Set environment variables
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0
ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_SERVER_ENABLE_CORS=false
ENV STREAMLIT_SERVER_ENABLE_XSRF_PROTECTION=false

# Expose port
EXPOSE 8501

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# Run the application
CMD ["python3", "-m", "streamlit", "run", "rag_fabric_app_docker.py", "--server.port=8501", "--server.address=0.0.0.0"]
EOF
        print_status "✅ Ubuntu-based Dockerfile created: Dockerfile.docker.ubuntu"
        echo ""
        print_status "To build with this Dockerfile:"
        echo "docker build -f Dockerfile.docker.ubuntu -t rag-fabric-app ."
        ;;
    *)
        print_error "Invalid option. Please run the script again and choose 1-4."
        ;;
esac

echo ""
print_info "If you continue to have issues, you can:"
echo "1. Use the deployment script: ./deploy_docker.sh"
echo "2. Check Docker logs: docker system prune -a"
echo "3. Try building on a different machine"
echo "4. Use the browser-based version instead of Docker" 