#!/usr/bin/env python3
"""
Test script to verify OAuth flow configuration.
"""

import os
from dotenv import load_dotenv

def test_oauth_config():
    """Test OAuth configuration."""
    print("Testing OAuth Flow Configuration")
    print("=" * 50)
    
    # Load environment variables
    load_dotenv()
    
    # Get Azure AD configuration
    client_id = os.getenv('AZURE_CLIENT_ID')
    client_secret = os.getenv('AZURE_CLIENT_SECRET')
    tenant_id = os.getenv('AZURE_TENANT_ID')
    redirect_uri = os.getenv('AZURE_REDIRECT_URI', 'http://localhost:8501/callback')
    scope = os.getenv('AZURE_SCOPE', 'https://analysis.windows.net/powerbi/api/.default')
    
    print(f"✅ Client ID: {'*' * len(client_id) if client_id else 'NOT SET'}")
    print(f"✅ Client Secret: {'*' * len(client_secret) if client_secret else 'NOT SET'}")
    print(f"✅ Tenant ID: {tenant_id if tenant_id else 'NOT SET'}")
    print(f"✅ Redirect URI: {redirect_uri}")
    print(f"✅ Scope: {scope}")
    
    if all([client_id, client_secret, tenant_id]):
        print("\n✅ All Azure AD configuration variables are set!")
        print("\nNext steps:")
        print("1. Update your Azure AD app registration:")
        print(f"   - Redirect URI: {redirect_uri}")
        print("   - Permissions: Microsoft Graph > User.Read, Power BI Service > Dataset.Read.All")
        print("2. Grant admin consent for all permissions")
        print("3. Test the OAuth flow in the application")
        return True
    else:
        print("\n❌ Missing Azure AD configuration variables")
        print("Please set AZURE_CLIENT_ID, AZURE_CLIENT_SECRET, and AZURE_TENANT_ID")
        return False

def main():
    """Run the test."""
    success = test_oauth_config()
    
    print("\n" + "=" * 50)
    if success:
        print("✅ OAuth configuration test completed!")
    else:
        print("❌ OAuth configuration test failed")

if __name__ == "__main__":
    main() 