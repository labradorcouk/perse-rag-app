# Azure Deployment Script for RAG Application (PowerShell)
# This script deploys the application to Azure Container Instances

param(
    [string]$ResourceGroup = "rag-fabric-rg",
    [string]$Location = "eastus",
    [string]$ContainerRegistry = "ragfabricacr",
    [string]$ContainerImage = "rag-fabric-app",
    [string]$ContainerInstance = "rag-fabric-instance"
)

# Check if Azure CLI is installed
if (-not (Get-Command az -ErrorAction SilentlyContinue)) {
    Write-Host "Azure CLI is not installed. Please install it first." -ForegroundColor Red
    exit 1
}

Write-Host "Starting Azure deployment..." -ForegroundColor Green

# Login to Azure
Write-Host "Logging into Azure..." -ForegroundColor Yellow
az login

# Create resource group
Write-Host "Creating resource group..." -ForegroundColor Yellow
az group create --name $ResourceGroup --location $Location

# Create Azure Container Registry
Write-Host "Creating Azure Container Registry..." -ForegroundColor Yellow
az acr create --resource-group $ResourceGroup --name $ContainerRegistry --sku Basic

# Enable admin user for ACR
Write-Host "Enabling admin user for ACR..." -ForegroundColor Yellow
az acr update -n $ContainerRegistry --admin-enabled true

# Get ACR login server
$ACRLoginServer = az acr show --name $ContainerRegistry --resource-group $ResourceGroup --query "loginServer" --output tsv
Write-Host "ACR Login Server: $ACRLoginServer" -ForegroundColor Green

# Get ACR credentials
Write-Host "Getting ACR credentials..." -ForegroundColor Yellow
$ACRUsername = az acr credential show --name $ContainerRegistry --query "username" --output tsv
$ACRPassword = az acr credential show --name $ContainerRegistry --query "passwords[0].value" --output tsv

# Build and push Docker image
Write-Host "Building Docker image..." -ForegroundColor Yellow
docker build -t $ContainerImage .

Write-Host "Tagging image for ACR..." -ForegroundColor Yellow
docker tag $ContainerImage $ACRLoginServer/$ContainerImage`:latest

Write-Host "Logging into ACR..." -ForegroundColor Yellow
docker login $ACRLoginServer -u $ACRUsername -p $ACRPassword

Write-Host "Pushing image to ACR..." -ForegroundColor Yellow
docker push $ACRLoginServer/$ContainerImage`:latest

# Create Container Instance
Write-Host "Creating Container Instance..." -ForegroundColor Yellow
az container create `
    --resource-group $ResourceGroup `
    --name $ContainerInstance `
    --image $ACRLoginServer/$ContainerImage`:latest `
    --dns-name-label rag-fabric-app `
    --ports 8501 `
    --environment-variables `
        OPENAI_API_KEY="$env:OPENAI_API_KEY" `
        DEFAULT_MODEL="gpt-3.5-turbo" `
        MAX_TOKENS="600" `
        TEMPERATURE="0.2" `
        QDRANT_URL="$env:QDRANT_URL" `
        QDRANT_API_KEY="$env:QDRANT_API_KEY"

# Get the public IP
Write-Host "Getting public IP..." -ForegroundColor Yellow
$PublicIP = az container show --resource-group $ResourceGroup --name $ContainerInstance --query "ipAddress.ip" --output tsv

Write-Host "Deployment completed successfully!" -ForegroundColor Green
Write-Host "Your application is available at: http://$PublicIP:8501" -ForegroundColor Green
Write-Host "Container Instance Name: $ContainerInstance" -ForegroundColor Green
Write-Host "Resource Group: $ResourceGroup" -ForegroundColor Green

# Optional: Create App Service deployment
$response = Read-Host "Would you like to deploy to Azure App Service instead? (y/n)"
if ($response -eq "y" -or $response -eq "Y") {
    Write-Host "Creating App Service Plan..." -ForegroundColor Yellow
    az appservice plan create --name "rag-fabric-plan" --resource-group $ResourceGroup --sku B1 --is-linux

    Write-Host "Creating Web App..." -ForegroundColor Yellow
    az webapp create --resource-group $ResourceGroup --plan "rag-fabric-plan" --name "rag-fabric-webapp" --deployment-container-image-name $ACRLoginServer/$ContainerImage`:latest

    Write-Host "Configuring environment variables..." -ForegroundColor Yellow
    az webapp config appsettings set --resource-group $ResourceGroup --name "rag-fabric-webapp" --settings `
        OPENAI_API_KEY="$env:OPENAI_API_KEY" `
        DEFAULT_MODEL="gpt-3.5-turbo" `
        MAX_TOKENS="600" `
        TEMPERATURE="0.2" `
        QDRANT_URL="$env:QDRANT_URL" `
        QDRANT_API_KEY="$env:QDRANT_API_KEY"

    Write-Host "App Service deployment completed!" -ForegroundColor Green
    Write-Host "Your application is available at: https://rag-fabric-webapp.azurewebsites.net" -ForegroundColor Green
} 