#!/usr/bin/env python3
"""
Test script to verify Azure AD authentication configuration.
"""

import os
from dotenv import load_dotenv

def test_azure_config():
    """Test Azure AD configuration."""
    print("Testing Azure AD Configuration")
    print("=" * 40)
    
    # Load environment variables
    load_dotenv()
    
    # Check required variables
    required_vars = [
        'AZURE_CLIENT_ID',
        'AZURE_CLIENT_SECRET', 
        'AZURE_TENANT_ID'
    ]
    
    missing_vars = []
    for var in required_vars:
        value = os.getenv(var)
        if not value:
            missing_vars.append(var)
        else:
            print(f"✅ {var}: {'*' * len(value)} (length: {len(value)})")
    
    if missing_vars:
        print(f"❌ Missing environment variables: {', '.join(missing_vars)}")
        return False
    
    # Check optional variables
    redirect_uri = os.getenv('AZURE_REDIRECT_URI', 'http://localhost:8501')
    scope = os.getenv('AZURE_SCOPE', 'https://analysis.windows.net/powerbi/api/.default')
    
    print(f"✅ AZURE_REDIRECT_URI: {redirect_uri}")
    print(f"✅ AZURE_SCOPE: {scope}")
    
    print("\n✅ All Azure AD configuration variables are set!")
    print("\nNext steps:")
    print("1. Make sure your Azure AD app registration has the correct permissions:")
    print("   - Microsoft Graph > User.Read")
    print("   - Power BI Service > Dataset.Read.All")
    print("2. Grant admin consent for all permissions")
    print("3. Access the application at: http://localhost:8501")
    print("4. Click 'Login with Azure AD' to test authentication")
    
    return True

if __name__ == "__main__":
    test_azure_config() 