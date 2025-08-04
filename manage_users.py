#!/usr/bin/env python3
"""
Authorized Users Management Utility
"""

import os
from dotenv import load_dotenv

load_dotenv()

def get_authorized_users():
    """Get list of authorized users from environment variable or use default list"""
    env_users = os.getenv('AUTHORIZED_USERS')
    if env_users:
        return [user.strip() for user in env_users.split(',')]
    
    return [
        "mawaz@opendata.energy",
        "jaipal@opendata.energy",
        "vikesh@opendata.energy",
        "sudheer@opendata.energy",
        "selva@opendata.energy",
        "ranjana.c@opendata.energy",
    ]

def list_authorized_users():
    """List all currently authorized users"""
    users = get_authorized_users()
    print("Currently authorized users:")
    for i, user in enumerate(users, 1):
        print(f"{i}. {user}")
    return users

def check_user_authorized(email):
    """Check if a specific user is authorized"""
    users = get_authorized_users()
    is_authorized = email in users
    print(f"User {email}: {'AUTHORIZED' if is_authorized else 'NOT AUTHORIZED'}")
    return is_authorized

def show_environment_setup():
    """Show how to set up the environment variable"""
    print("\n" + "="*50)
    print("ENVIRONMENT VARIABLE SETUP")
    print("="*50)
    print("To manage authorized users via environment variable:")
    print("1. Add to your .env file:")
    print("   AUTHORIZED_USERS=user1@domain.com,user2@domain.com,user3@domain.com")
    print("2. Or set as environment variable:")
    print("   export AUTHORIZED_USERS='user1@domain.com,user2@domain.com'")
    print("3. Or set in your deployment environment")
    print("\nCurrent authorized users from environment:")
    users = get_authorized_users()
    if users:
        for user in users:
            print(f"   - {user}")
    else:
        print("   No users found in environment variable")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        command = sys.argv[1]
        
        if command == "list":
            list_authorized_users()
        
        elif command == "check":
            if len(sys.argv) > 2:
                email = sys.argv[2]
                check_user_authorized(email)
            else:
                print("Usage: python manage_users.py check <email>")
        
        elif command == "setup":
            show_environment_setup()
        
        else:
            print("Unknown command. Available commands:")
            print("  list   - List all authorized users")
            print("  check <email> - Check if user is authorized")
            print("  setup  - Show environment setup instructions")
    else:
        print("Authorized Users Management Utility")
        print("="*40)
        list_authorized_users()
        print("\nUse 'python manage_users.py setup' for environment setup instructions") 