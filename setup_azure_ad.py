#!/usr/bin/env python3
"""
Azure AD Setup Script for Docker Authentication

This script helps you set up Azure AD application registration for Docker authentication.
"""

import os
import requests
import json
from urllib.parse import urlencode

def print_setup_instructions():
    """Print step-by-step instructions for Azure AD setup."""
    
    print("Azure AD Setup for Docker Authentication")
    print("=" * 50)
    print()
    print("Follow these steps to set up Azure AD authentication:")
    print()
    
    print("1. Create Azure AD App Registration:")
    print("   - Go to Azure Portal: https://portal.azure.com")
    print("   - Navigate to 'Azure Active Directory' > 'App registrations'")
    print("   - Click 'New registration'")
    print("   - Name: 'Fabric RAG Docker App'")
    print("   - Supported account types: 'Accounts in this organizational directory only'")
    print("   - Redirect URI: Web > http://localhost:8501")
    print("   - Click 'Register'")
    print()
    
    print("2. Get Application (Client) ID:")
    print("   - Copy the 'Application (client) ID' from the app registration")
    print("   - This will be your AZURE_CLIENT_ID")
    print()
    
    print("3. Get Directory (Tenant) ID:")
    print("   - Copy the 'Directory (tenant) ID' from the app registration")
    print("   - This will be your AZURE_TENANT_ID")
    print()
    
    print("4. Create Client Secret:")
    print("   - In the app registration, go to 'Certificates & secrets'")
    print("   - Click 'New client secret'")
    print("   - Description: 'Docker App Secret'")
    print("   - Expiration: Choose appropriate duration")
    print("   - Copy the secret value (you won't see it again)")
    print("   - This will be your AZURE_CLIENT_SECRET")
    print()
    
    print("5. Configure API Permissions (CRITICAL):")
    print("   - Go to 'API permissions' in the app registration")
    print("   - Click 'Add a permission'")
    print("   - Select 'Microsoft Graph'")
    print("   - Choose 'Delegated permissions'")
    print("   - Add these permissions:")
    print("     - User.Read (to get user info)")
    print("   - Click 'Add permissions'")
    print()
    
    print("6. Configure Power BI Service Permissions (IMPORTANT):")
    print("   - Go to 'API permissions' again")
    print("   - Click 'Add a permission'")
    print("   - Select 'Power BI Service' (search for 'Power BI')")
    print("   - Choose 'Delegated permissions'")
    print("   - Add 'Dataset.Read.All' permission")
    print("   - Click 'Add permissions'")
    print()
    
    print("7. Grant Admin Consent:")
    print("   - Click 'Grant admin consent' for all permissions")
    print("   - This is required for the permissions to take effect")
    print()
    
    print("8. Update Environment Variables:")
    print("   - Copy .env.example.azure to .env")
    print("   - Update the following variables:")
    print("     AZURE_CLIENT_ID=your_client_id")
    print("     AZURE_CLIENT_SECRET=your_client_secret")
    print("     AZURE_TENANT_ID=your_tenant_id")
    print("     AZURE_REDIRECT_URI=http://localhost:8501")
    print("     AZURE_SCOPE=https://analysis.windows.net/powerbi/api/.default")
    print()
    
    print("9. Test the Setup:")
    print("   - Run: docker-compose up -d")
    print("   - Access: http://localhost:8501")
    print("   - Click 'Login with Azure AD'")
    print("   - Complete the authentication flow")
    print()

def test_azure_config():
    """Test if Azure configuration is properly set up."""
    
    print("Testing Azure Configuration")
    print("=" * 30)
    
    required_vars = [
        'AZURE_CLIENT_ID',
        'AZURE_CLIENT_SECRET', 
        'AZURE_TENANT_ID'
    ]
    
    missing_vars = []
    for var in required_vars:
        if not os.getenv(var):
            missing_vars.append(var)
    
    if missing_vars:
        print(f"ERROR: Missing environment variables: {', '.join(missing_vars)}")
        print("Please set these variables in your .env file")
        return False
    
    print("SUCCESS: All required environment variables are set")
    
    # Test Azure AD endpoints
    tenant_id = os.getenv('AZURE_TENANT_ID')
    client_id = os.getenv('AZURE_CLIENT_ID')
    
    print(f"Testing Azure AD endpoints for tenant: {tenant_id}")
    
    # Test authorization endpoint
    auth_url = f"https://login.microsoftonline.com/{tenant_id}/oauth2/v2.0/authorize"
    try:
        response = requests.get(auth_url, timeout=10)
        if response.status_code == 200:
            print("SUCCESS: Authorization endpoint is accessible")
        else:
            print(f"WARNING: Authorization endpoint returned status: {response.status_code}")
    except Exception as e:
        print(f"ERROR: Cannot access authorization endpoint: {e}")
        return False
    
    # Test token endpoint
    token_url = f"https://login.microsoftonline.com/{tenant_id}/oauth2/v2.0/token"
    try:
        response = requests.get(token_url, timeout=10)
        if response.status_code == 400:  # Expected for GET without parameters
            print("SUCCESS: Token endpoint is accessible")
        else:
            print(f"WARNING: Token endpoint returned status: {response.status_code}")
    except Exception as e:
        print(f"ERROR: Cannot access token endpoint: {e}")
        return False
    
    print("SUCCESS: Azure AD configuration appears to be correct")
    return True

def main():
    """Main function."""
    
    print("Azure AD Setup for Fabric RAG Docker App")
    print("=" * 50)
    print()
    
    # Check if .env file exists
    if not os.path.exists('.env'):
        print("ERROR: .env file not found!")
        print("Please copy .env.example.azure to .env and configure it")
        print()
        print_setup_instructions()
        return
    
    # Load environment variables
    from dotenv import load_dotenv
    load_dotenv()
    
    # Test configuration
    if test_azure_config():
        print()
        print("SUCCESS: Azure AD setup appears to be complete!")
        print("You can now run: docker-compose up -d")
    else:
        print()
        print("ERROR: Azure AD setup needs attention")
        print_setup_instructions()

if __name__ == "__main__":
    main() 