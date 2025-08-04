#!/usr/bin/env python3
"""
Test script to verify OAuth auth URL generation.
"""

import os
import secrets
import urllib.parse
from dotenv import load_dotenv

def test_auth_url():
    """Test OAuth auth URL generation."""
    print("Testing OAuth Auth URL Generation")
    print("=" * 50)
    
    # Load environment variables
    load_dotenv()
    
    # Get Azure AD configuration
    client_id = os.getenv('AZURE_CLIENT_ID')
    client_secret = os.getenv('AZURE_CLIENT_SECRET')
    tenant_id = os.getenv('AZURE_TENANT_ID')
    redirect_uri = os.getenv('AZURE_REDIRECT_URI', 'http://localhost:8501/callback')
    scope = os.getenv('AZURE_SCOPE', 'https://analysis.windows.net/powerbi/api/.default')
    
    if not all([client_id, client_secret, tenant_id]):
        print("❌ Missing Azure AD configuration variables")
        return False
    
    # Generate auth URL (same logic as in auth_azure_docker.py)
    state = secrets.token_urlsafe(32)
    
    params = {
        'client_id': client_id,
        'response_type': 'code',
        'redirect_uri': redirect_uri,
        'scope': scope,
        'state': state,
        'response_mode': 'query'
    }
    
    auth_url = f"https://login.microsoftonline.com/{tenant_id}/oauth2/v2.0/authorize"
    full_auth_url = f"{auth_url}?{urllib.parse.urlencode(params)}"
    
    print(f"✅ Client ID: {client_id}")
    print(f"✅ Tenant ID: {tenant_id}")
    print(f"✅ Redirect URI: {redirect_uri}")
    print(f"✅ Scope: {scope}")
    print(f"✅ State: {state}")
    print(f"\n🔗 Generated Auth URL:")
    print(f"{full_auth_url}")
    
    print(f"\n📋 URL Components:")
    print(f"   - Base URL: https://login.microsoftonline.com/{tenant_id}/oauth2/v2.0/authorize")
    print(f"   - Client ID: {client_id}")
    print(f"   - Redirect URI: {redirect_uri}")
    print(f"   - Scope: {scope}")
    print(f"   - State: {state}")
    
    return True

def main():
    """Run the test."""
    success = test_auth_url()
    
    print("\n" + "=" * 50)
    if success:
        print("✅ Auth URL generation test completed!")
        print("\nNext steps:")
        print("1. Copy the generated auth URL")
        print("2. Open it in a browser to test the OAuth flow")
        print("3. Verify it redirects to Microsoft login")
    else:
        print("❌ Auth URL generation test failed")

if __name__ == "__main__":
    main() 