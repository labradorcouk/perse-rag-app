#!/usr/bin/env python3
"""
Test script to verify auth imports work correctly after moving files.
"""

import sys
import os

# Add the current directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_auth_imports():
    """Test that auth imports work correctly."""
    try:
        print("🧪 Testing Auth Imports")
        print("=" * 40)
        
        # Test importing from auth package
        from auth import azure_auth
        print("✅ Successfully imported azure_auth from auth package")
        
        from auth import init_auth_session_state
        print("✅ Successfully imported init_auth_session_state from auth package")
        
        from auth import main_auth
        print("✅ Successfully imported main_auth from auth package")
        
        # Test importing specific modules
        from auth.auth_azure_docker import AzureAuthDocker
        print("✅ Successfully imported AzureAuthDocker from auth.auth_azure_docker")
        
        from auth.auth_docker import get_service_principal_credential
        print("✅ Successfully imported get_service_principal_credential from auth.auth_docker")
        
        from auth.auth import get_authorized_users
        print("✅ Successfully imported get_authorized_users from auth.auth")
        
        print("\n🎉 All auth imports working correctly!")
        return True
        
    except Exception as e:
        print(f"❌ Import test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_auth_imports() 