# Qdrant Deployment Script for Azure Container Instances
# This script deploys Qdrant to Azure and configures it for the RAG application

param(
    [string]$ResourceGroup = "rag-fabric-rg",
    [string]$Location = "eastus",
    [string]$QdrantInstance = "qdrant-instance",
    [string]$QdrantStorageAccount = "qdrantstorage",
    [string]$QdrantFileShare = "qdrant-data"
)

# Check if Azure CLI is installed
if (-not (Get-Command az -ErrorAction SilentlyContinue)) {
    Write-Host "Azure CLI is not installed. Please install it first." -ForegroundColor Red
    exit 1
}

Write-Host "Starting Qdrant deployment to Azure..." -ForegroundColor Green

# Login to Azure
Write-Host "Logging into Azure..." -ForegroundColor Yellow
az login

# Create resource group if it doesn't exist
Write-Host "Creating resource group..." -ForegroundColor Yellow
az group create --name $ResourceGroup --location $Location

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

# Get the public IP
Write-Host "Getting Qdrant public IP..." -ForegroundColor Yellow
$QdrantIP = az container show --resource-group $ResourceGroup --name $QdrantInstance --query "ipAddress.ip" --output tsv

Write-Host "Qdrant deployment completed successfully!" -ForegroundColor Green
Write-Host "Qdrant is available at: http://$QdrantIP:6333" -ForegroundColor Green
Write-Host "Qdrant gRPC is available at: $QdrantIP:6334" -ForegroundColor Green
Write-Host "Container Instance Name: $QdrantInstance" -ForegroundColor Green
Write-Host "Resource Group: $ResourceGroup" -ForegroundColor Green

# Export the Qdrant URL for use in the main application
$QdrantURL = "http://$QdrantIP:6333"
Write-Host "Set this as your QDRANT_URL: $QdrantURL" -ForegroundColor Yellow

# Optional: Create a script to migrate your local Qdrant data
Write-Host "Would you like to create a data migration script? (y/n)" -ForegroundColor Yellow
$response = Read-Host
if ($response -eq "y" -or $response -eq "Y") {
    Write-Host "Creating data migration script..." -ForegroundColor Yellow
    
    # Create migration script
    $MigrationScript = @"
# Qdrant Data Migration Script
# Run this to migrate your local Qdrant data to Azure

# Export collections from local Qdrant
Write-Host "Exporting collections from local Qdrant..." -ForegroundColor Yellow

# You can use qdrant-client to export/import collections
# Example commands:
# python -c "from qdrant_client import QdrantClient; client = QdrantClient('localhost', port=6333); client.create_snapshot('collection_name', 'snapshot_name')"

# Or use the Qdrant REST API to export collections
# curl -X POST "http://localhost:6333/collections/{collection_name}/snapshots"

Write-Host "Migration script created. Please implement the specific export/import logic for your collections." -ForegroundColor Green
"@
    
    $MigrationScript | Out-File -FilePath "migrate_qdrant_data.ps1" -Encoding UTF8
    Write-Host "Migration script created: migrate_qdrant_data.ps1" -ForegroundColor Green
} 