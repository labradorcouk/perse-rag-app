#!/usr/bin/env python3
"""
Azure Service Principal Setup Script for Docker Deployment

This script helps you create and configure an Azure service principal
for Docker deployment of the RAG Fabric App.

Prerequisites:
1. Azure CLI installed and logged in
2. Appropriate permissions to create service principals
3. Python with azure-identity package installed

Usage:
    python setup_azure_service_principal.py
"""

import subprocess
import json
import os
import sys
from pathlib import Path

def run_command(command, capture_output=True):
    """Run a command and return the result"""
    try:
        result = subprocess.run(command, shell=True, capture_output=capture_output, text=True)
        return result
    except Exception as e:
        print(f"Error running command '{command}': {e}")
        return None

def check_azure_cli():
    """Check if Azure CLI is installed and user is logged in"""
    print("Checking Azure CLI installation...")
    
    # Check if az CLI is installed
    result = run_command("az --version")
    if result is None or result.returncode != 0:
        print("❌ Azure CLI is not installed. Please install it from: https://docs.microsoft.com/en-us/cli/azure/install-azure-cli")
        return False
    
    # Check if user is logged in
    result = run_command("az account show")
    if result is None or result.returncode != 0:
        print("❌ You are not logged in to Azure. Please run 'az login' first.")
        return False
    
    account_info = json.loads(result.stdout)
    print(f"✅ Logged in as: {account_info.get('user', {}).get('name', 'Unknown')}")
    print(f"   Tenant ID: {account_info.get('tenantId', 'Unknown')}")
    return True

def create_service_principal(app_name="rag-fabric-app"):
    """Create a service principal for the application"""
    print(f"\nCreating service principal '{app_name}'...")
    
    # Create the service principal
    result = run_command(f"az ad sp create-for-rbac --name {app_name} --skip-assignment")
    if result is None or result.returncode != 0:
        print("❌ Failed to create service principal")
        return None
    
    sp_info = json.loads(result.stdout)
    print("✅ Service principal created successfully!")
    return sp_info

def assign_roles(service_principal_id, roles=None):
    """Assign roles to the service principal"""
    if roles is None:
        roles = [
            "Reader",  # Basic read access
            "Contributor"  # For Fabric data access
        ]
    
    print(f"\nAssigning roles to service principal...")
    
    for role in roles:
        print(f"Assigning {role} role...")
        result = run_command(f"az role assignment create --assignee {service_principal_id} --role {role}")
        if result is None or result.returncode != 0:
            print(f"⚠️  Warning: Failed to assign {role} role. You may need to do this manually.")
        else:
            print(f"✅ {role} role assigned successfully")

def create_env_file(sp_info):
    """Create .env file with service principal credentials"""
    print("\nCreating .env file...")
    
    env_content = f"""# Azure Service Principal Configuration
AZURE_TENANT_ID={sp_info['tenant']}
AZURE_CLIENT_ID={sp_info['appId']}
AZURE_CLIENT_SECRET={sp_info['password']}

# OpenAI Configuration
OPENAI_API_KEY=your-openai-api-key-here
DEFAULT_MODEL=gpt-3.5-turbo
MAX_TOKENS=600
TEMPERATURE=0.2

# Authorized Users (comma-separated list)
AUTHORIZED_USERS=mawaz@opendata.energy,jaipal@opendata.energy,vikesh@opendata.energy,sudheer@opendata.energy,selva@opendata.energy,ranjana.c@opendata.energy

# Qdrant Configuration
QDRANT_HOST=localhost
QDRANT_PORT=6333

# Email Configuration (Optional)
EMAIL_USER=your-email@opendata.energy
EMAIL_PASSWORD=your-email-password
ADMIN_EMAIL=admin@opendata.energy

# Streamlit Configuration
STREAMLIT_SERVER_PORT=8501
STREAMLIT_SERVER_ADDRESS=0.0.0.0
STREAMLIT_SERVER_HEADLESS=true
STREAMLIT_SERVER_ENABLE_CORS=false
STREAMLIT_SERVER_ENABLE_XSRF_PROTECTION=false
"""
    
    with open('.env', 'w') as f:
        f.write(env_content)
    
    print("✅ .env file created successfully!")
    print("⚠️  IMPORTANT: Please update the following values in .env:")
    print("   - OPENAI_API_KEY: Your OpenAI API key")
    print("   - EMAIL_USER: Your email for notifications")
    print("   - EMAIL_PASSWORD: Your email password")
    print("   - ADMIN_EMAIL: Admin email for notifications")

def main():
    """Main setup function"""
    print("🚀 Azure Service Principal Setup for RAG Fabric App")
    print("=" * 50)
    
    # Check prerequisites
    if not check_azure_cli():
        sys.exit(1)
    
    # Get app name
    app_name = input("\nEnter service principal name (default: rag-fabric-app): ").strip()
    if not app_name:
        app_name = "rag-fabric-app"
    
    # Create service principal
    sp_info = create_service_principal(app_name)
    if not sp_info:
        sys.exit(1)
    
    # Assign roles
    assign_roles(sp_info['appId'])
    
    # Create .env file
    create_env_file(sp_info)
    
    print("\n" + "=" * 50)
    print("✅ Setup completed successfully!")
    print("\nNext steps:")
    print("1. Update the .env file with your OpenAI API key and email settings")
    print("2. Build the Docker image: docker build -f Dockerfile.docker -t rag-fabric-app .")
    print("3. Run with docker-compose: docker-compose up")
    print("4. Access the application at: http://localhost:8501")
    
    print(f"\nService Principal Details:")
    print(f"  App ID: {sp_info['appId']}")
    print(f"  Tenant ID: {sp_info['tenant']}")
    print(f"  Password: {sp_info['password']}")
    print("\n⚠️  IMPORTANT: Save these credentials securely!")

if __name__ == "__main__":
    main() 