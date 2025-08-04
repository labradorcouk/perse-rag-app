# Azure Deployment Guide for RAG Fabric Application

This guide will help you deploy your RAG application to Azure using Docker containers.

## Prerequisites

1. **Azure CLI** installed and configured
2. **Docker** installed and running
3. **Azure subscription** with appropriate permissions
4. **Environment variables** configured (see below)

## Environment Variables Setup

Create a `.env` file with your configuration:

```bash
# OpenAI Configuration
OPENAI_API_KEY=your_openai_api_key_here

# Qdrant Configuration
QDRANT_URL=your_qdrant_url_here
QDRANT_API_KEY=your_qdrant_api_key_here

# Application Configuration
DEFAULT_MODEL=gpt-3.5-turbo
MAX_TOKENS=600
TEMPERATURE=0.2
```

## Deployment Options

### Option 1: Azure Container Instances (Recommended for testing)

1. **Make the deployment script executable:**
   ```bash
   chmod +x deploy_to_azure.sh
   ```

2. **Set your environment variables:**
   ```bash
   export OPENAI_API_KEY="your_openai_api_key"
   export QDRANT_URL="your_qdrant_url"
   export QDRANT_API_KEY="your_qdrant_api_key"
   ```

3. **Run the deployment script:**
   ```bash
   ./deploy_to_azure.sh
   ```

### Option 2: Azure App Service (Recommended for production)

1. **Create App Service Plan:**
   ```bash
   az appservice plan create --name "rag-fabric-plan" --resource-group "rag-fabric-rg" --sku B1 --is-linux
   ```

2. **Create Web App:**
   ```bash
   az webapp create --resource-group "rag-fabric-rg" --plan "rag-fabric-plan" --name "rag-fabric-webapp" --deployment-container-image-name your-acr.azurecr.io/rag-fabric-app:latest
   ```

3. **Configure environment variables:**
   ```bash
   az webapp config appsettings set --resource-group "rag-fabric-rg" --name "rag-fabric-webapp" --settings \
     OPENAI_API_KEY="your_openai_api_key" \
     QDRANT_URL="your_qdrant_url" \
     QDRANT_API_KEY="your_qdrant_api_key" \
     DEFAULT_MODEL="gpt-3.5-turbo" \
     MAX_TOKENS="600" \
     TEMPERATURE="0.2"
   ```

### Option 3: Local Docker Testing

1. **Build and run with docker-compose:**
   ```bash
   docker-compose up --build
   ```

2. **Access the application at:** `http://localhost:8501`

## Manual Docker Build and Push

If you prefer to build and push manually:

1. **Build the Docker image:**
   ```bash
   docker build -t rag-fabric-app .
   ```

2. **Tag for Azure Container Registry:**
   ```bash
   docker tag rag-fabric-app your-acr.azurecr.io/rag-fabric-app:latest
   ```

3. **Login to ACR:**
   ```bash
   az acr login --name your-acr
   ```

4. **Push to ACR:**
   ```bash
   docker push your-acr.azurecr.io/rag-fabric-app:latest
   ```

## Azure Resources Created

The deployment script will create:

- **Resource Group:** `rag-fabric-rg`
- **Container Registry:** `ragfabricacr`
- **Container Instance:** `rag-fabric-instance` (or App Service)
- **Network Security Groups** (if needed)

## Monitoring and Logs

### View Container Logs:
```bash
az container logs --resource-group rag-fabric-rg --name rag-fabric-instance
```

### View App Service Logs:
```bash
az webapp log tail --resource-group rag-fabric-rg --name rag-fabric-webapp
```

## Scaling and Performance

### Container Instances:
- **CPU:** 1-4 cores
- **Memory:** 1-16 GB
- **Cost:** Pay per second

### App Service:
- **B1 Plan:** 1 CPU, 1.75 GB RAM
- **S1 Plan:** 1 CPU, 1.75 GB RAM (with auto-scaling)
- **P1V2 Plan:** 1 CPU, 3.5 GB RAM (Premium)

## Troubleshooting

### Common Issues:

1. **Container fails to start:**
   - Check environment variables
   - Verify Qdrant connection
   - Check logs: `az container logs`

2. **Application not accessible:**
   - Verify port 8501 is exposed
   - Check network security groups
   - Verify DNS name resolution

3. **Authentication issues:**
   - Verify Azure credentials
   - Check ACR permissions
   - Ensure service principal has proper roles

### Debug Commands:

```bash
# Check container status
az container show --resource-group rag-fabric-rg --name rag-fabric-instance

# Get public IP
az container show --resource-group rag-fabric-rg --name rag-fabric-instance --query "ipAddress.ip" --output tsv

# Restart container
az container restart --resource-group rag-fabric-rg --name rag-fabric-instance
```

## Cost Optimization

1. **Use Container Instances** for development/testing
2. **Use App Service** for production (better scaling)
3. **Stop unused resources** to save costs
4. **Monitor usage** with Azure Cost Management

## Security Best Practices

1. **Use Azure Key Vault** for secrets
2. **Enable HTTPS** for production
3. **Configure network security groups**
4. **Use managed identities** when possible
5. **Regular security updates**

## Next Steps

1. **Set up monitoring** with Azure Monitor
2. **Configure alerts** for application health
3. **Set up CI/CD** with GitHub Actions
4. **Implement backup** strategies
5. **Plan for scaling** based on usage patterns 