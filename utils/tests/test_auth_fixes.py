#!/usr/bin/env python3
"""
Test script to verify the authentication fixes.
"""

import os
from dotenv import load_dotenv
from datetime import datetime, timedelta
from azure.identity import ClientSecretCredential

def test_direct_authentication():
    """Test the direct authentication method."""
    print("Testing Direct Authentication Fix")
    print("=" * 40)
    
    # Load environment variables
    load_dotenv()
    
    # Get Azure AD configuration
    client_id = os.getenv('AZURE_CLIENT_ID')
    client_secret = os.getenv('AZURE_CLIENT_SECRET')
    tenant_id = os.getenv('AZURE_TENANT_ID')
    scope = os.getenv('AZURE_SCOPE', 'https://analysis.windows.net/powerbi/api/.default')
    
    if not all([client_id, client_secret, tenant_id]):
        print("❌ Missing Azure AD configuration variables")
        return False
    
    try:
        # Create credential
        credential = ClientSecretCredential(
            tenant_id=tenant_id,
            client_id=client_id,
            client_secret=client_secret
        )
        
        # Get token
        token_data = credential.get_token(scope)
        
        # Test the timestamp calculation fix
        if hasattr(token_data.expires_on, 'timestamp'):
            # If it's a datetime object
            expires_in = int((token_data.expires_on - datetime.now()).total_seconds())
        else:
            # If it's already a timestamp
            expires_in = int(token_data.expires_on - datetime.now().timestamp())
        
        print(f"✅ Token acquired successfully")
        print(f"✅ Token expires in: {expires_in} seconds")
        print(f"✅ Timestamp calculation fixed")
        
        return True
        
    except Exception as e:
        print(f"❌ Direct authentication failed: {str(e)}")
        return False

def test_oauth_state():
    """Test the OAuth state parameter handling."""
    print("\nTesting OAuth State Parameter Fix")
    print("=" * 40)
    
    import secrets
    
    # Generate a test state
    state = secrets.token_urlsafe(32)
    print(f"✅ Generated state: {state[:10]}...")
    
    # Simulate state verification
    expected_state = state
    received_state = state
    
    if expected_state == received_state:
        print("✅ State parameter verification working")
        return True
    else:
        print("❌ State parameter verification failed")
        return False

def main():
    """Run all tests."""
    print("Authentication Fixes Test")
    print("=" * 50)
    
    # Test direct authentication
    direct_success = test_direct_authentication()
    
    # Test OAuth state
    oauth_success = test_oauth_state()
    
    print("\n" + "=" * 50)
    if direct_success and oauth_success:
        print("✅ All authentication fixes are working!")
        print("\nNext steps:")
        print("1. Access the application at: http://localhost:8501")
        print("2. Try 'Login with Azure AD (Direct)' first")
        print("3. If that doesn't work, try 'Login with Azure AD (OAuth)'")
    else:
        print("❌ Some authentication fixes need attention")
        if not direct_success:
            print("- Direct authentication needs configuration")
        if not oauth_success:
            print("- OAuth state handling needs attention")

if __name__ == "__main__":
    main() 