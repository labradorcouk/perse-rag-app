#!/usr/bin/env python3
"""
Test script to verify Microsoft Graph API permissions and token acquisition.
"""

import os
from dotenv import load_dotenv
from azure.identity import ClientSecretCredential
import requests

def test_graph_permissions():
    """Test Microsoft Graph API permissions."""
    print("Testing Microsoft Graph API Permissions")
    print("=" * 50)
    
    # Load environment variables
    load_dotenv()
    
    # Get Azure AD configuration
    client_id = os.getenv('AZURE_CLIENT_ID')
    client_secret = os.getenv('AZURE_CLIENT_SECRET')
    tenant_id = os.getenv('AZURE_TENANT_ID')
    
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
        
        # Test different scopes
        scopes = [
            'https://graph.microsoft.com/.default',
            'https://analysis.windows.net/powerbi/api/.default',
            'https://database.windows.net/.default'
        ]
        
        for scope in scopes:
            print(f"\n🔍 Testing scope: {scope}")
            try:
                token_data = credential.get_token(scope)
                print(f"✅ Token acquired successfully")
                print(f"   Token expires in: {token_data.expires_on}")
                
                # Test Microsoft Graph API call if it's a Graph scope
                if 'graph.microsoft.com' in scope:
                    headers = {
                        'Authorization': f'Bearer {token_data.token}',
                        'Content-Type': 'application/json'
                    }
                    
                    # Test getting users
                    response = requests.get(
                        'https://graph.microsoft.com/v1.0/users?$top=1',
                        headers=headers
                    )
                    
                    if response.status_code == 200:
                        print("✅ Microsoft Graph API call successful")
                        data = response.json()
                        if 'value' in data and len(data['value']) > 0:
                            user = data['value'][0]
                            print(f"   Sample user: {user.get('displayName', 'N/A')} ({user.get('userPrincipalName', 'N/A')})")
                    else:
                        print(f"❌ Microsoft Graph API call failed: {response.status_code}")
                        print(f"   Error: {response.text}")
                
            except Exception as e:
                print(f"❌ Failed to get token for scope {scope}: {str(e)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {str(e)}")
        return False

def main():
    """Run the test."""
    print("Microsoft Graph Permissions Test")
    print("=" * 50)
    
    success = test_graph_permissions()
    
    print("\n" + "=" * 50)
    if success:
        print("✅ Graph permissions test completed!")
        print("\nNext steps:")
        print("1. Check if your Azure AD app has Microsoft Graph permissions")
        print("2. Make sure 'User.Read.All' permission is granted")
        print("3. Grant admin consent for all permissions")
    else:
        print("❌ Graph permissions test failed")
        print("\nTroubleshooting:")
        print("1. Check Azure AD app registration permissions")
        print("2. Verify service principal has correct roles")
        print("3. Ensure admin consent is granted")

if __name__ == "__main__":
    main() 