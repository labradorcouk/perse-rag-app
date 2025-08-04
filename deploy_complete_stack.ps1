# Complete Stack Deployment Script for RAG Application
# This script deploys both Qdrant and the RAG application to Azure

param(
    [string]$ResourceGroup = "rag-fabric-rg",
    [string]$Location = "eastus",
    [string]$ContainerRegistry = "ragfabricacr",
    [string]$ContainerImage = "rag-fabric-app",
    [string]$ContainerInstance = "rag-fabric-instance",
    [string]$QdrantInstance = "qdrant-instance",
    [string]$QdrantStorageAccount = "qdrantstorage",
    [string]$QdrantFileShare = "qdrant-data"
)

# Check if Azure CLI is installed
if (-not (Get-Command az -ErrorAction SilentlyContinue)) {
    Write-Host "Azure CLI is not installed. Please install it first." -ForegroundColor Red
    exit 1
}

Write-Host "Starting complete stack deployment..." -ForegroundColor Green

# Login to Azure
Write-Host "Logging into Azure..." -ForegroundColor Yellow
az login

# Create resource group
Write-Host "Creating resource group..." -ForegroundColor Yellow
az group create --name $ResourceGroup --location $Location

# Step 1: Deploy Qdrant
Write-Host "Step 1: Deploying Qdrant..." -ForegroundColor Yellow

# Create Storage Account for persistent Qdrant data
Write-Host "Creating Storage Account for Qdrant data..." -ForegroundColor Yellow
az storage account create `
    --resource-group $ResourceGroup `
    --name $QdrantStorageAccount `
    --location $Location `
    --sku Standard_LRS `
    --kind StorageV2

# Create File Share for Qdrant data
Write-Host "Creating File Share..." -ForegroundColor Yellow
az storage share create `
    --account-name $QdrantStorageAccount `
    --name $QdrantFileShare

# Get Storage Account Key
Write-Host "Getting Storage Account Key..." -ForegroundColor Yellow
$StorageKey = az storage account keys list --resource-group $ResourceGroup --account-name $QdrantStorageAccount --query "[0].value" --output tsv

# Create Qdrant Container Instance
Write-Host "Creating Qdrant Container Instance..." -ForegroundColor Yellow
az container create `
    --resource-group $ResourceGroup `
    --name $QdrantInstance `
    --image qdrant/qdrant:latest `
    --dns-name-label qdrant-rag-app `
    --ports 6333 6334 `
    --environment-variables `
        QDRANT__SERVICE__HTTP_PORT=6333 `
        QDRANT__SERVICE__GRPC_PORT=6334 `
    --azure-file-volume-account-name $QdrantStorageAccount `
    --azure-file-volume-account-key $StorageKey `
    --azure-file-volume-share-name $QdrantFileShare `
    --azure-file-volume-mount-path /qdrant/storage

# Get Qdrant public IP
Write-Host "Getting Qdrant public IP..." -ForegroundColor Yellow
$QdrantIP = az container show --resource-group $ResourceGroup --name $QdrantInstance --query "ipAddress.ip" --output tsv
$QdrantURL = "http://$QdrantIP:6333"

Write-Host "Qdrant deployed successfully!" -ForegroundColor Green
Write-Host "Qdrant URL: $QdrantURL" -ForegroundColor Green

# Step 2: Deploy RAG Application
Write-Host "Step 2: Deploying RAG Application..." -ForegroundColor Yellow

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

# Create RAG Application Container Instance
Write-Host "Creating RAG Application Container Instance..." -ForegroundColor Yellow
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
        QDRANT_URL="$QdrantURL" `
        QDRANT_API_KEY="$env:QDRANT_API_KEY"

# Get the RAG application public IP
Write-Host "Getting RAG application public IP..." -ForegroundColor Yellow
$RAGIP = az container show --resource-group $ResourceGroup --name $ContainerInstance --query "ipAddress.ip" --output tsv

Write-Host "Complete stack deployment finished!" -ForegroundColor Green
Write-Host "=" * 50 -ForegroundColor Green
Write-Host "Deployment Summary:" -ForegroundColor Green
Write-Host "  Qdrant URL: $QdrantURL" -ForegroundColor Green
Write-Host "  RAG Application: http://$RAGIP:8501" -ForegroundColor Green
Write-Host "  Resource Group: $ResourceGroup" -ForegroundColor Green
Write-Host "  Qdrant Container: $QdrantInstance" -ForegroundColor Green
Write-Host "  RAG Container: $ContainerInstance" -ForegroundColor Green
Write-Host "=" * 50 -ForegroundColor Green

# Optional: Create data migration script
Write-Host "Would you like to create a data migration script for your local Qdrant data? (y/n)" -ForegroundColor Yellow
$response = Read-Host
if ($response -eq "y" -or $response -eq "Y") {
    Write-Host "Creating migration script..." -ForegroundColor Yellow
    
    $MigrationScript = @"
# Set the Azure Qdrant URL for migration
`$env:AZURE_QDRANT_URL = "$QdrantURL"

# Run the migration script
python migrate_qdrant_data.py
"@
    
    $MigrationScript | Out-File -FilePath "run_migration.ps1" -Encoding UTF8
    Write-Host "Migration script created: run_migration.ps1" -ForegroundColor Green
    Write-Host "Run this script to migrate your local Qdrant data to Azure" -ForegroundColor Green
}

# Optional: Create App Service deployment
Write-Host "Would you like to deploy to Azure App Service instead? (y/n)" -ForegroundColor Yellow
$response = Read-Host
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
        QDRANT_URL="$QdrantURL" `
        QDRANT_API_KEY="$env:QDRANT_API_KEY"

    Write-Host "App Service deployment completed!" -ForegroundColor Green
    Write-Host "Your application is available at: https://rag-fabric-webapp.azurewebsites.net" -ForegroundColor Green
} 