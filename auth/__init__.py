"""
Authentication package for the RAG application.
"""

# Import the main authentication modules
from .auth_azure_docker import AzureAuthDocker
from .auth_docker import *
from .auth import *

# Create instances for easy access
azure_auth = AzureAuthDocker()

__all__ = [
    'AzureAuthDocker',
    'azure_auth',
    'get_authorized_users',
    'is_user_authorized',
    'add_authorized_user',
    'remove_authorized_user',
    'init_auth_session_state',
    'clear_credential_cache',
    'get_user_info_from_token',
    'make_graphql_request',
    'fetch_user_role',
    'authenticate_with_azure',
    'show_registration_form',
    'send_registration_email',
    'show_login',
    'main_auth',
    'get_service_principal_credential',
    'get_user_info_from_service_principal',
    'check_user_role_from_service_principal',
    'authenticate_with_service_principal',
    'clear_auth_session_state',
    'show_docker_login',
    'main_docker_auth',
    'get_docker_credential',
] 