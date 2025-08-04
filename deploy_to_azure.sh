#!/bin/bash

# Azure Deployment Script for RAG Application
# This script deploys the application to Azure Container Instances

set -e

# Configuration
RESOURCE_GROUP="rag-fabric-rg"
LOCATION="eastus"
CONTAINER_REGISTRY="ragfabricacr"
CONTAINER_IMAGE="rag-fabric-app"
CONTAINER_INSTANCE="rag-fabric-instance"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}Starting Azure deployment...${NC}"

# Check if Azure CLI is installed
if ! command -v az &> /dev/null; then
    echo -e "${RED}Azure CLI is not installed. Please install it first.${NC}"
    exit 1
fi

# Login to Azure
echo -e "${YELLOW}Logging into Azure...${NC}"
az login

# Create resource group
echo -e "${YELLOW}Creating resource group...${NC}"
az group create --name $RESOURCE_GROUP --location $LOCATION

# Create Azure Container Registry
echo -e "${YELLOW}Creating Azure Container Registry...${NC}"
az acr create --resource-group $RESOURCE_GROUP --name $CONTAINER_REGISTRY --sku Basic

# Enable admin user for ACR
echo -e "${YELLOW}Enabling admin user for ACR...${NC}"
az acr update -n $CONTAINER_REGISTRY --admin-enabled true

# Get ACR login server
ACR_LOGIN_SERVER=$(az acr show --name $CONTAINER_REGISTRY --resource-group $RESOURCE_GROUP --query "loginServer" --output tsv)
echo -e "${GREEN}ACR Login Server: $ACR_LOGIN_SERVER${NC}"

# Get ACR credentials
echo -e "${YELLOW}Getting ACR credentials...${NC}"
ACR_USERNAME=$(az acr credential show --name $CONTAINER_REGISTRY --query "username" --output tsv)
ACR_PASSWORD=$(az acr credential show --name $CONTAINER_REGISTRY --query "passwords[0].value" --output tsv)

# Build and push Docker image
echo -e "${YELLOW}Building Docker image...${NC}"
docker build -t $CONTAINER_IMAGE .

echo -e "${YELLOW}Tagging image for ACR...${NC}"
docker tag $CONTAINER_IMAGE $ACR_LOGIN_SERVER/$CONTAINER_IMAGE:latest

echo -e "${YELLOW}Logging into ACR...${NC}"
docker login $ACR_LOGIN_SERVER -u $ACR_USERNAME -p $ACR_PASSWORD

echo -e "${YELLOW}Pushing image to ACR...${NC}"
docker push $ACR_LOGIN_SERVER/$CONTAINER_IMAGE:latest

# Create Container Instance
echo -e "${YELLOW}Creating Container Instance...${NC}"
az container create \
    --resource-group $RESOURCE_GROUP \
    --name $CONTAINER_INSTANCE \
    --image $ACR_LOGIN_SERVER/$CONTAINER_IMAGE:latest \
    --dns-name-label rag-fabric-app \
    --ports 8501 \
    --environment-variables \
        OPENAI_API_KEY="$OPENAI_API_KEY" \
        DEFAULT_MODEL="gpt-3.5-turbo" \
        MAX_TOKENS="600" \
        TEMPERATURE="0.2" \
        QDRANT_URL="$QDRANT_URL" \
        QDRANT_API_KEY="$QDRANT_API_KEY"

# Get the public IP
echo -e "${YELLOW}Getting public IP...${NC}"
PUBLIC_IP=$(az container show --resource-group $RESOURCE_GROUP --name $CONTAINER_INSTANCE --query "ipAddress.ip" --output tsv)

echo -e "${GREEN}Deployment completed successfully!${NC}"
echo -e "${GREEN}Your application is available at: http://$PUBLIC_IP:8501${NC}"
echo -e "${GREEN}Container Instance Name: $CONTAINER_INSTANCE${NC}"
echo -e "${GREEN}Resource Group: $RESOURCE_GROUP${NC}"

# Optional: Create App Service deployment
echo -e "${YELLOW}Would you like to deploy to Azure App Service instead? (y/n)${NC}"
read -r response
if [[ "$response" =~ ^([yY][eE][sS]|[yY])$ ]]; then
    echo -e "${YELLOW}Creating App Service Plan...${NC}"
    az appservice plan create --name "rag-fabric-plan" --resource-group $RESOURCE_GROUP --sku B1 --is-linux

    echo -e "${YELLOW}Creating Web App...${NC}"
    az webapp create --resource-group $RESOURCE_GROUP --plan "rag-fabric-plan" --name "rag-fabric-webapp" --deployment-container-image-name $ACR_LOGIN_SERVER/$CONTAINER_IMAGE:latest

    echo -e "${YELLOW}Configuring environment variables...${NC}"
    az webapp config appsettings set --resource-group $RESOURCE_GROUP --name "rag-fabric-webapp" --settings \
        OPENAI_API_KEY="$OPENAI_API_KEY" \
        DEFAULT_MODEL="gpt-3.5-turbo" \
        MAX_TOKENS="600" \
        TEMPERATURE="0.2" \
        QDRANT_URL="$QDRANT_URL" \
        QDRANT_API_KEY="$QDRANT_API_KEY"

    echo -e "${GREEN}App Service deployment completed!${NC}"
    echo -e "${GREEN}Your application is available at: https://rag-fabric-webapp.azurewebsites.net${NC}"
fi 