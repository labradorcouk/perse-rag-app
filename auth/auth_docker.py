import streamlit as st
import os
import requests
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import jwt
import time
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from azure.identity import ClientSecretCredential, DefaultAzureCredential
from azure.core.exceptions import ClientAuthenticationError

def get_authorized_users():
    """Get list of authorized users from environment variable or use default list"""
    # Try to get from environment variable first
    env_users = os.getenv('AUTHORIZED_USERS')
    if env_users:
        # Split by comma and strip whitespace
        return [user.strip() for user in env_users.split(',')]
    
    # Default authorized users list
    return [
        "mawaz@opendata.energy",
        "jaipal@opendata.energy",
        "vikesh@opendata.energy",
        "sudheer@opendata.energy",
        "selva@opendata.energy",
        "ranjana.c@opendata.energy",
        # Add more authorized usernames here
    ]

def is_user_authorized(user_principal_name):
    """Check if a user is in the authorized list"""
    authorized_users = get_authorized_users()
    return user_principal_name in authorized_users

def get_service_principal_credential():
    """Get service principal credentials from environment variables"""
    tenant_id = os.getenv('AZURE_TENANT_ID')
    client_id = os.getenv('AZURE_CLIENT_ID')
    client_secret = os.getenv('AZURE_CLIENT_SECRET')
    
    if not all([tenant_id, client_id, client_secret]):
        st.error("Service principal credentials not configured. Please set AZURE_TENANT_ID, AZURE_CLIENT_ID, and AZURE_CLIENT_SECRET environment variables.")
        return None
    
    try:
        credential = ClientSecretCredential(
            tenant_id=tenant_id,
            client_id=client_id,
            client_secret=client_secret
        )
        return credential
    except Exception as e:
        st.error(f"Failed to create service principal credential: {str(e)}")
        return None

def get_user_info_from_service_principal(user_email, credential):
    """Get user information using service principal authentication"""
    try:
        # Get token for Microsoft Graph API
        graph_token = credential.get_token("https://graph.microsoft.com/.default")
        
        # Query Microsoft Graph API to get user information
        headers = {
            'Authorization': f'Bearer {graph_token.token}',
            'Content-Type': 'application/json'
        }
        
        # Get user details from Microsoft Graph
        graph_url = f"https://graph.microsoft.com/v1.0/users/{user_email}"
        response = requests.get(graph_url, headers=headers)
        
        if response.status_code == 200:
            user_data = response.json()
            return {
                'userPrincipalName': user_data.get('userPrincipalName'),
                'displayName': user_data.get('displayName'),
                'mail': user_data.get('mail'),
                'jobTitle': user_data.get('jobTitle'),
                'id': user_data.get('id')
            }
        else:
            st.error(f"Failed to get user info: {response.status_code} - {response.text}")
            return None
            
    except Exception as e:
        st.error(f"Error getting user info from service principal: {str(e)}")
        return None

def check_user_role_from_service_principal(user_email, credential):
    """Check user role using service principal authentication"""
    try:
        # Get token for Microsoft Graph API
        graph_token = credential.get_token("https://graph.microsoft.com/.default")
        
        headers = {
            'Authorization': f'Bearer {graph_token.token}',
            'Content-Type': 'application/json'
        }
        
        # Get user's app role assignments (this requires admin consent)
        graph_url = f"https://graph.microsoft.com/v1.0/users/{user_email}/appRoleAssignments"
        response = requests.get(graph_url, headers=headers)
        
        if response.status_code == 200:
            role_data = response.json()
            app_roles = role_data.get('value', [])
            
            # Check for admin role
            for role in app_roles:
                if role.get('appRole', {}).get('displayName') == 'Admin':
                    return 'Admin'
            
            # Check for other roles
            for role in app_roles:
                role_name = role.get('appRole', {}).get('displayName')
                if role_name in ['DNOAnalyst', 'EnergySystemAcademic', 'EnergyAppDev', 'Landlord']:
                    return role_name
            
            return 'User'  # Default role
        else:
            st.warning(f"Could not fetch user roles: {response.status_code}")
            return None
            
    except Exception as e:
        st.error(f"Error checking user role from service principal: {str(e)}")
        return None

def authenticate_with_service_principal(user_email):
    """Authenticate user using service principal"""
    try:
        # Get service principal credential
        credential = get_service_principal_credential()
        if not credential:
            return False
        
        # Get user information
        user_info = get_user_info_from_service_principal(user_email, credential)
        if not user_info:
            st.error(f"Could not get user information for {user_email}")
            return False
        
        user_principal_name = user_info.get('userPrincipalName')
        
        # Check if user is authorized
        if not is_user_authorized(user_principal_name):
            st.error(f"Access denied for {user_principal_name}. You don't have access to this dashboard.")
            return False
        
        # Get user role
        user_role = check_user_role_from_service_principal(user_email, credential)
        if not user_role:
            user_role = 'User'  # Default role
        
        # Set session state
        st.session_state.authenticated = True
        st.session_state.user_role = user_role
        st.session_state.user_info = user_info
        
        st.success(f"Welcome, {user_info.get('displayName', user_principal_name)}! Role: {user_role}")
        return True
        
    except Exception as e:
        st.error(f"Authentication failed: {str(e)}")
        return False

def init_auth_session_state():
    """Initialize authentication session state"""
    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = False
    if 'user_role' not in st.session_state:
        st.session_state.user_role = None
    if 'user_info' not in st.session_state:
        st.session_state.user_info = None

def clear_auth_session_state():
    """Clear authentication session state"""
    st.session_state.authenticated = False
    st.session_state.user_role = None
    st.session_state.user_info = None

def show_docker_login():
    """Show login form for Docker deployment"""
    st.subheader("Docker Authentication")
    st.info("This application is running in a Docker container. Please enter your email to authenticate.")
    
    with st.form("docker_login_form"):
        user_email = st.text_input("Email Address", placeholder="mawaz@opendata.energy")
        login_button = st.form_submit_button("Login")
        
        if login_button:
            if not user_email:
                st.error("Please enter your email address")
                return
            
            if not user_email.endswith('@opendata.energy'):
                st.error("Please use your @opendata.energy email address")
                return
            
            with st.spinner("Authenticating..."):
                if authenticate_with_service_principal(user_email):
                    st.rerun()
                else:
                    st.error("Authentication failed. Please check your email address and try again.")

def main_docker_auth():
    """Main authentication function for Docker deployment"""
    init_auth_session_state()
    
    if not st.session_state.authenticated:
        show_docker_login()
        st.stop()
    
    # Show user info in sidebar
    with st.sidebar:
        if st.session_state.user_info:
            st.write(f"Welcome, {st.session_state.user_info.get('displayName', 'User')}")
            if st.session_state.user_role == "Admin":
                st.write("Role: Admin (Full Access)")
            else:
                st.write(f"Role: {st.session_state.user_role}")
        
        if st.button("Logout"):
            clear_auth_session_state()
            st.rerun()

def get_docker_credential():
    """Get credential for Docker deployment using service principal"""
    try:
        # Try service principal first
        credential = get_service_principal_credential()
        if credential:
            return credential
        
        # Fallback to default credential
        return DefaultAzureCredential()
        
    except Exception as e:
        st.error(f"Failed to get credential: {str(e)}")
        return None 