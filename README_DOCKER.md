# RAG Fabric App - Docker Deployment Guide

This guide explains how to deploy the RAG Fabric App using Docker with Azure Service Principal authentication.

## Overview

The Docker version of the RAG Fabric App uses Azure Service Principal authentication instead of browser-based authentication, making it suitable for containerized environments where browser redirects are not possible.

## Prerequisites

1. **Docker and Docker Compose** installed on your system
2. **Azure CLI** installed and configured
3. **Azure subscription** with appropriate permissions
4. **OpenAI API key** for LLM functionality

## Quick Start

### 1. Set up Azure Service Principal

Run the setup script to create and configure the Azure service principal:

```bash
python setup_azure_service_principal.py
```

This script will:
- Check if Azure CLI is installed and you're logged in
- Create a service principal for the application
- Assign necessary roles (Reader, Contributor)
- Create a `.env` file with the credentials

### 2. Configure Environment Variables

Update the `.env` file with your specific values:

```bash
# Edit the .env file
nano .env
```

Make sure to update:
- `OPENAI_API_KEY`: Your OpenAI API key
- `EMAIL_USER`: Your email for notifications (optional)
- `EMAIL_PASSWORD`: Your email password (optional)
- `ADMIN_EMAIL`: Admin email for notifications (optional)

### 3. Build and Run

```bash
# Build the Docker image
docker build -f Dockerfile.docker -t rag-fabric-app .

# Run with docker-compose
docker-compose up -d

# Or run just the app (without Qdrant)
docker-compose up rag-fabric-app -d
```

### 4. Access the Application

Open your browser and navigate to: `http://localhost:8501`

## Manual Setup (Alternative)

If you prefer to set up the service principal manually:

### 1. Create Service Principal

```bash
# Login to Azure
az login

# Create service principal
az ad sp create-for-rbac --name "rag-fabric-app" --skip-assignment

# Note the output - you'll need these values:
# - appId (Client ID)
# - password (Client Secret)
# - tenant (Tenant ID)
```

### 2. Assign Roles

```bash
# Get your subscription ID
az account show --query id -o tsv

# Assign roles (replace with your actual values)
az role assignment create --assignee <appId> --role Reader --scope /subscriptions/<subscription-id>
az role assignment create --assignee <appId> --role Contributor --scope /subscriptions/<subscription-id>
```

### 3. Create .env File

Create a `.env` file with the following content:

```env
# Azure Service Principal Configuration
AZURE_TENANT_ID=your-tenant-id
AZURE_CLIENT_ID=your-client-id
AZURE_CLIENT_SECRET=your-client-secret

# OpenAI Configuration
OPENAI_API_KEY=your-openai-api-key
DEFAULT_MODEL=gpt-3.5-turbo
MAX_TOKENS=600
TEMPERATURE=0.2

# Authorized Users
AUTHORIZED_USERS=mawaz@opendata.energy,jaipal@opendata.energy,vikesh@opendata.energy,sudheer@opendata.energy,selva@opendata.energy,ranjana.c@opendata.energy

# Qdrant Configuration
QDRANT_HOST=localhost
QDRANT_PORT=6333

# Streamlit Configuration
STREAMLIT_SERVER_PORT=8501
STREAMLIT_SERVER_ADDRESS=0.0.0.0
STREAMLIT_SERVER_HEADLESS=true
STREAMLIT_SERVER_ENABLE_CORS=false
STREAMLIT_SERVER_ENABLE_XSRF_PROTECTION=false
```

## Authentication Flow

### Docker Version vs Browser Version

| Feature | Browser Version | Docker Version |
|---------|----------------|----------------|
| Authentication | Interactive browser redirect | Service principal + email input |
| User Input | Automatic from browser session | Manual email entry |
| Role Assignment | GraphQL API calls | Microsoft Graph API |
| Deployment | Local development | Containerized production |

### How Docker Authentication Works

1. **User enters email** in the login form
2. **Service principal** authenticates with Azure
3. **Microsoft Graph API** is used to:
   - Get user information
   - Check user roles
   - Verify authorization
4. **Session state** is maintained for the user

## Configuration Options

### Environment Variables

| Variable | Description | Required | Default |
|----------|-------------|----------|---------|
| `AZURE_TENANT_ID` | Azure tenant ID | Yes | - |
| `AZURE_CLIENT_ID` | Service principal client ID | Yes | - |
| `AZURE_CLIENT_SECRET` | Service principal client secret | Yes | - |
| `OPENAI_API_KEY` | OpenAI API key | Yes | - |
| `AUTHORIZED_USERS` | Comma-separated list of authorized emails | No | Default list |
| `QDRANT_HOST` | Qdrant host address | No | localhost |
| `QDRANT_PORT` | Qdrant port | No | 6333 |

### Docker Compose Options

```yaml
# Run with local Qdrant
docker-compose --profile qdrant-local up

# Run without Qdrant (use external)
docker-compose up rag-fabric-app

# Run in detached mode
docker-compose up -d
```

## Troubleshooting

### Common Issues

1. **Authentication Failed**
   - Check service principal credentials in `.env`
   - Verify roles are assigned correctly
   - Ensure user email is in `AUTHORIZED_USERS`

2. **Connection to Azure Services Failed**
   - Verify service principal has correct permissions
   - Check network connectivity
   - Ensure tenant ID is correct

3. **Docker Build Fails**
   - Check Docker installation
   - Ensure all files are in the correct directory
   - Verify requirements.txt files exist

4. **Application Won't Start**
   - Check logs: `docker-compose logs rag-fabric-app`
   - Verify port 8501 is available
   - Check environment variables

### Debug Commands

```bash
# Check container logs
docker-compose logs rag-fabric-app

# Access container shell
docker-compose exec rag-fabric-app bash

# Check environment variables
docker-compose exec rag-fabric-app env

# Test Azure connection
docker-compose exec rag-fabric-app python -c "from auth_docker import get_service_principal_credential; print(get_service_principal_credential())"
```

## Security Considerations

1. **Service Principal Credentials**
   - Store securely (use Azure Key Vault in production)
   - Rotate regularly
   - Use minimal required permissions

2. **Environment Variables**
   - Never commit `.env` files to version control
   - Use secrets management in production
   - Validate all inputs

3. **Network Security**
   - Use HTTPS in production
   - Configure firewall rules appropriately
   - Consider using Azure Container Instances or AKS

## Production Deployment

For production deployment, consider:

1. **Azure Container Instances (ACI)**
   ```bash
   az container create \
     --resource-group myResourceGroup \
     --name rag-fabric-app \
     --image rag-fabric-app:latest \
     --dns-name-label rag-fabric-app \
     --ports 8501
   ```

2. **Azure Kubernetes Service (AKS)**
   - Create Kubernetes manifests
   - Use Azure Key Vault for secrets
   - Configure ingress controllers

3. **Azure App Service**
   - Use Docker container deployment
   - Configure environment variables in Azure portal
   - Use managed identity for Azure services

## Support

For issues or questions:
1. Check the troubleshooting section above
2. Review Docker and Azure logs
3. Verify service principal permissions
4. Test with a simple Azure connection first

## Files Overview

| File | Purpose |
|------|---------|
| `auth_docker.py` | Service principal authentication module |
| `rag_fabric_app_docker.py` | Main application for Docker deployment |
| `Dockerfile.docker` | Docker image definition |
| `docker-compose.yml` | Multi-service deployment configuration |
| `setup_azure_service_principal.py` | Automated setup script |
| `.env.example.docker` | Environment variables template |
| `README_DOCKER.md` | This documentation | 