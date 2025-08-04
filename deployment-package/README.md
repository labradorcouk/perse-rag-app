# Fabric RAG Application - Deployment Package

## 📋 Overview
This package contains the Docker image and configuration files for the Fabric RAG (Retrieval-Augmented Generation) application.

## 📦 Contents
- `fabric-rag-app-v1.0.0.tar` - Docker image
- `docker-compose.yml` - Docker Compose configuration
- `.env.example` - Environment variables template
- `deploy.sh` - Deployment script
- `README.md` - This file

## 🚀 Quick Start

### Prerequisites
- Docker and Docker Compose installed
- At least 4GB RAM available
- Port 8501 available

### Installation Steps

1. **Extract the package:**
   ```bash
   tar -xzf fabric-rag-deployment-v1.0.0.tar.gz
   cd fabric-rag-deployment
   ```

2. **Load the Docker image:**
   ```bash
   docker load -i fabric-rag-app-v1.0.0.tar
   ```

3. **Configure environment variables:**
   ```bash
   cp .env.example .env
   # Edit .env with your Azure AD and other configuration
   ```

4. **Deploy the application:**
   ```bash
   chmod +x deploy.sh
   ./deploy.sh
   ```

5. **Access the application:**
   - Open http://localhost:8501 in your browser
   - Login with your Azure AD credentials

## ⚙️ Configuration

### Required Environment Variables
- `AZURE_CLIENT_ID` - Azure AD application client ID
- `AZURE_CLIENT_SECRET` - Azure AD application client secret
- `AZURE_TENANT_ID` - Azure AD tenant ID
- `AZURE_REDIRECT_URI` - OAuth redirect URI (http://localhost:8501/callback)
- `AZURE_SCOPE` - Azure AD scope (https://analysis.windows.net/powerbi/api/.default)
- `OPENAI_API_KEY` - OpenAI API key for code generation
- `AUTHORIZED_USERS` - Comma-separated list of authorized user emails

### Optional Environment Variables
- `QDRANT_URL` - Qdrant vector database URL (default: http://localhost:6333)
- `EMAIL_USER` - Email username for notifications
- `EMAIL_PASSWORD` - Email password for notifications
- `ADMIN_EMAIL` - Admin email for notifications

## 🔧 Troubleshooting

### Common Issues

1. **Port 8501 already in use:**
   ```bash
   # Check what's using the port
   netstat -tulpn | grep 8501
   # Stop the conflicting service or change the port in docker-compose.yml
   ```

2. **Docker image not found:**
   ```bash
   # Verify the image was loaded correctly
   docker images | grep fabric-rag-app
   # If not found, reload the image
   docker load -i fabric-rag-app-v1.0.0.tar
   ```

3. **Azure AD authentication issues:**
   - Verify all Azure AD environment variables are set correctly
   - Check that the redirect URI is registered in Azure AD
   - Ensure the application has the correct permissions

4. **Qdrant connection issues:**
   ```bash
   # Check if Qdrant is running
   docker ps | grep qdrant
   # Restart the services
   docker-compose restart
   ```

## 📊 Features

- **RAG QA**: Natural language queries with AI-powered code generation
- **SQL Editor**: Direct SQL queries to Microsoft Fabric
- **Diagnostics Dashboard**: Monitor logs, errors, and performance
- **Azure AD Authentication**: Secure OAuth and direct authentication
- **Vector Search**: FAISS and Qdrant integration
- **Performance Monitoring**: Real-time metrics and analytics

## 🆘 Support

For issues or questions:
1. Check the diagnostics dashboard in the application
2. Review Docker logs: `docker-compose logs rag-fabric-app`
3. Contact the development team

## 📝 Version History

- **v1.0.0**: Initial release with RAG QA, SQL Editor, and Diagnostics Dashboard 