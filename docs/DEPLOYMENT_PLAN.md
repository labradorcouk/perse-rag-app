# Fabric RAG Application - Deployment Plan

## 🚀 Version 2.0.0 Deployment Guide

### **Overview**
This document provides comprehensive deployment instructions for the Fabric RAG application with FAISS vector search, intelligent table selection, and robust data type handling.

## 📋 Prerequisites

### **System Requirements**
- **OS**: Linux (Ubuntu 20.04+) / Windows Server 2019+ / macOS 10.15+
- **CPU**: 4+ cores (8+ recommended for production)
- **RAM**: 8GB minimum (16GB+ recommended for FAISS operations)
- **Storage**: 50GB+ SSD storage
- **Network**: Stable internet connection for API access

### **Software Dependencies**
- **Python**: 3.8+ (3.9+ recommended)
- **Docker**: 20.10+ (for containerized deployment)
- **Git**: Latest version
- **ODBC Driver**: 17 or 18 for SQL Server

### **Cloud Platform Requirements**
- **Azure**: Active subscription with Fabric access
- **OpenAI**: API key with sufficient credits
- **Container Registry**: Azure Container Registry or Docker Hub

## 🏗️ Architecture Overview

### **Component Stack**
```
┌─────────────────────────────────────────────────────────────┐
│                    User Interface Layer                     │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────┐ │
│  │   Streamlit UI  │  │   SQL Editor    │  │ Data Preview│ │
│  └─────────────────┘  └─────────────────┘  └─────────────┘ │
└─────────────────────────────────────────────────────────────┘
┌─────────────────────────────────────────────────────────────┐
│                   Application Logic Layer                   │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────┐ │
│  │  RAG Pipeline   │  │ FAISS Vector    │  │ Data Type   │ │
│  │                 │  │ Search Engine   │  │ Handler     │ │
│  └─────────────────┘  └─────────────────┘  └─────────────┘ │
└─────────────────────────────────────────────────────────────┘
┌─────────────────────────────────────────────────────────────┐
│                    Data Access Layer                       │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────┐ │
│  │ Microsoft Fabric│  │   Azure AD      │  │ OpenAI API  │ │
│  │   GraphQL API   │  │ Authentication  │  │             │ │
│  └─────────────────┘  └─────────────────┘  └─────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

## 🚀 Deployment Options

### **Option 1: Local Development Deployment**

#### **Step 1: Environment Setup**
```bash
# Clone repository
git clone <repository-url>
cd fabric-rag-app

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt
```

#### **Step 2: Configuration**
```bash
# Copy environment template
cp .env.example .env

# Edit .env file
nano .env
```

**Required Environment Variables:**
```env
# OpenAI Configuration
OPENAI_API_KEY=your_openai_api_key_here
DEFAULT_MODEL=gpt-3.5-turbo
MAX_TOKENS=600
TEMPERATURE=0.2

# Azure Configuration
AZURE_TENANT_ID=your_tenant_id
AZURE_CLIENT_ID=your_client_id
AZURE_CLIENT_SECRET=your_client_secret

# Application Settings
LOG_LEVEL=INFO
CACHE_TTL=3600
MAX_BATCH_SIZE=5000
```

#### **Step 3: Run Application**
```bash
# Development mode
streamlit run rag_fabric_app.py

# Production mode with custom config
streamlit run rag_fabric_app.py --server.port 8501 --server.address 0.0.0.0
```

### **Option 2: Docker Container Deployment**

#### **Step 1: Create Dockerfile**
```dockerfile
FROM python:3.9-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    unixodbc \
    unixodbc-dev \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create non-root user
RUN useradd -m -u 1000 streamlit
RUN chown -R streamlit:streamlit /app
USER streamlit

# Expose port
EXPOSE 8501

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# Run application
CMD ["streamlit", "run", "rag_fabric_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

#### **Step 2: Build and Run**
```bash
# Build image
docker build -t fabric-rag-app .

# Run container
docker run -d \
    --name fabric-rag-app \
    -p 8501:8501 \
    --env-file .env \
    --restart unless-stopped \
    fabric-rag-app
```

### **Option 3: Azure Container Instances**

#### **Step 1: Push to Azure Container Registry**
```bash
# Login to Azure
az login

# Create container registry
az acr create --resource-group myResourceGroup --name myacr --sku Basic

# Build and push image
az acr build --registry myacr --image fabric-rag-app .
```

#### **Step 2: Deploy to Container Instances**
```bash
# Deploy container instance
az container create \
    --resource-group myResourceGroup \
    --name fabric-rag-app \
    --image myacr.azurecr.io/fabric-rag-app:latest \
    --dns-name-label fabric-rag-app \
    --ports 8501 \
    --environment-variables \
        OPENAI_API_KEY=$OPENAI_API_KEY \
        AZURE_TENANT_ID=$AZURE_TENANT_ID
```

### **Option 4: Azure App Service**

#### **Step 1: Create App Service**
```bash
# Create resource group
az group create --name myResourceGroup --location eastus

# Create app service plan
az appservice plan create \
    --name myAppServicePlan \
    --resource-group myResourceGroup \
    --sku B1 \
    --is-linux

# Create web app
az webapp create \
    --resource-group myResourceGroup \
    --plan myAppServicePlan \
    --name fabric-rag-app \
    --deployment-local-git
```

#### **Step 2: Configure Environment Variables**
```bash
# Set environment variables
az webapp config appsettings set \
    --resource-group myResourceGroup \
    --name fabric-rag-app \
    --settings \
        OPENAI_API_KEY=$OPENAI_API_KEY \
        AZURE_TENANT_ID=$AZURE_TENANT_ID \
        AZURE_CLIENT_ID=$AZURE_CLIENT_ID \
        AZURE_CLIENT_SECRET=$AZURE_CLIENT_SECRET
```

## 🔧 Configuration Management

### **Environment-Specific Configs**

#### **Development Environment**
```env
LOG_LEVEL=DEBUG
CACHE_TTL=300
MAX_BATCH_SIZE=1000
AUTO_FETCH_TABLES=true
```

#### **Staging Environment**
```env
LOG_LEVEL=INFO
CACHE_TTL=1800
MAX_BATCH_SIZE=3000
AUTO_FETCH_TABLES=true
```

#### **Production Environment**
```env
LOG_LEVEL=WARNING
CACHE_TTL=3600
MAX_BATCH_SIZE=5000
AUTO_FETCH_TABLES=true
```

### **Performance Tuning**

#### **Memory Optimization**
```python
# In rag_fabric_app.py
import gc
import psutil

# Monitor memory usage
def monitor_memory():
    process = psutil.Process()
    memory_info = process.memory_info()
    return memory_info.rss / 1024 / 1024  # MB

# Clean up after large operations
def cleanup_memory():
    gc.collect()
```

#### **FAISS Configuration**
```python
# Optimize FAISS for your use case
import faiss

# For CPU-only deployment
index = faiss.IndexFlatL2(dimension)

# For GPU deployment (if available)
# res = faiss.StandardGpuResources()
# index = faiss.index_cpu_to_gpu(res, 0, index)
```

## 🔒 Security Configuration

### **Authentication Setup**

#### **Azure AD Configuration**
```bash
# Register application in Azure AD
az ad app create --display-name "Fabric RAG App" --identifier-uris "http://localhost:8501"

# Get application ID
APP_ID=$(az ad app list --display-name "Fabric RAG App" --query "[].appId" -o tsv)

# Create service principal
az ad sp create --id $APP_ID

# Assign permissions
az role assignment create \
    --assignee $APP_ID \
    --role "Contributor" \
    --scope "/subscriptions/$SUBSCRIPTION_ID"
```

#### **OpenAI API Security**
```python
# Rate limiting configuration
import time
from functools import wraps

def rate_limit(max_calls=60, time_window=60):
    def decorator(func):
        calls = []
        
        @wraps(func)
        def wrapper(*args, **kwargs):
            now = time.time()
            calls[:] = [call for call in calls if now - call < time_window]
            
            if len(calls) >= max_calls:
                raise Exception("Rate limit exceeded")
            
            calls.append(now)
            return func(*args, **kwargs)
        return wrapper
    return decorator
```

### **Network Security**

#### **Firewall Configuration**
```bash
# Allow only necessary ports
sudo ufw allow 8501/tcp
sudo ufw allow 22/tcp
sudo ufw enable
```

#### **SSL/TLS Configuration**
```bash
# Generate SSL certificate
openssl req -x509 -newkey rsa:4096 -keyout key.pem -out cert.pem -days 365 -nodes

# Configure Streamlit with SSL
streamlit run rag_fabric_app.py \
    --server.sslCertFile=cert.pem \
    --server.sslKeyFile=key.pem
```

## 📊 Monitoring and Logging

### **Application Monitoring**

#### **Health Checks**
```python
# Add health check endpoint
import streamlit as st
import psutil
import time

def health_check():
    return {
        "status": "healthy",
        "timestamp": time.time(),
        "memory_usage": psutil.virtual_memory().percent,
        "cpu_usage": psutil.cpu_percent()
    }

# In main app
if st.sidebar.button("Health Check"):
    health = health_check()
    st.json(health)
```

#### **Performance Monitoring**
```python
# Monitor key metrics
import time
from functools import wraps

def monitor_performance(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        execution_time = time.time() - start_time
        
        # Log performance metrics
        st.session_state['performance_metrics'] = {
            'function': func.__name__,
            'execution_time': execution_time,
            'timestamp': time.time()
        }
        
        return result
    return wrapper
```

### **Logging Configuration**

#### **Structured Logging**
```python
import logging
import json
from datetime import datetime

# Configure structured logging
class StructuredFormatter(logging.Formatter):
    def format(self, record):
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName
        }
        return json.dumps(log_entry)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
```

## 🚀 Deployment Checklist

### **Pre-Deployment**
- [ ] **Environment Variables**: All required variables set
- [ ] **Dependencies**: All packages installed and compatible
- [ ] **Authentication**: Azure AD and OpenAI credentials configured
- [ ] **Network**: Firewall rules and SSL certificates configured
- [ ] **Monitoring**: Logging and health checks implemented
- [ ] **Backup**: Database and configuration backups scheduled

### **Deployment**
- [ ] **Build**: Application successfully builds
- [ ] **Test**: All functionality works in staging
- [ ] **Deploy**: Application deployed to production
- [ ] **Verify**: Health checks pass
- [ ] **Monitor**: Performance metrics within acceptable ranges

### **Post-Deployment**
- [ ] **Documentation**: Update deployment documentation
- [ ] **Training**: Team trained on new features
- [ ] **Monitoring**: Alerts configured for critical issues
- [ ] **Backup**: Verify backup procedures work
- [ ] **Security**: Security scan completed

## 🔄 Update Procedures

### **Application Updates**
```bash
# Pull latest changes
git pull origin main

# Update dependencies
pip install -r requirements.txt

# Restart application
sudo systemctl restart fabric-rag-app
```

### **Database Migrations**
```python
# Handle schema changes
def migrate_data():
    # Add migration logic here
    pass
```

### **Rollback Procedures**
```bash
# Quick rollback
git checkout HEAD~1
pip install -r requirements.txt
sudo systemctl restart fabric-rag-app
```

## 📞 Support and Maintenance

### **Troubleshooting Guide**
1. **Authentication Issues**: Check Azure AD configuration
2. **Memory Issues**: Monitor FAISS memory usage
3. **Performance Issues**: Review batch sizes and caching
4. **API Errors**: Check rate limits and quotas

### **Maintenance Schedule**
- **Daily**: Health check monitoring
- **Weekly**: Performance review and optimization
- **Monthly**: Security updates and dependency updates
- **Quarterly**: Full system audit and capacity planning

### **Fine-Tuning Embedding Model (Optional)**

#### **Step 1: Export Q&A Data as CSV**
- Use the SQL Editor in the app to export Q&A/code pairs as a CSV file.

#### **Step 2: Run Fine-Tuning Utility**
- Use the provided script to fine-tune the embedding model using the exported CSV:
```bash
python utils/fine_tune_embedding_model.py --csv_path <your_exported.csv> --base_model <model_name> --output_dir <output_dir>
```
- The utility now works exclusively with local CSV files, avoiding ODBC/pyodbc driver issues.
- Ensure `accelerate` and `datasets` are installed (included in requirements.txt).

### **Troubleshooting**
See TROUBLESHOOTING.md for a list of known issues, ODBC/pyodbc driver troubleshooting, and solutions for common errors.

---

**Version**: 2.0.0  
**Last Updated**: 8th July 2025  
**Status**: Production Ready 