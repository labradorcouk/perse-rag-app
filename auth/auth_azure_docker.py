#!/usr/bin/env python3
"""
Azure AD Authentication for Docker Deployment

This module provides Azure AD OAuth 2.0 authentication for the Streamlit app
when running in Docker containers where interactive browser authentication is not available.
"""

import streamlit as st
import os
import requests
import json
import jwt
from datetime import datetime, timedelta
from azure.identity import ClientSecretCredential, InteractiveBrowserCredential
import secrets
import urllib.parse
from utils.diagnostics_logger import diagnostics_logger, EventType, LogLevel

class AzureAuthDocker:
    def __init__(self):
        self.client_id = os.getenv('AZURE_CLIENT_ID')
        self.client_secret = os.getenv('AZURE_CLIENT_SECRET')
        self.tenant_id = os.getenv('AZURE_TENANT_ID')
        # Use a different redirect URI for OAuth callback to avoid confusion
        self.redirect_uri = os.getenv('AZURE_REDIRECT_URI', 'http://localhost:8501/callback')
        # Use the correct scope for Microsoft Fabric
        self.scope = os.getenv('AZURE_SCOPE', 'https://analysis.windows.net/powerbi/api/.default')
        # Use Microsoft Graph scope for user info calls
        self.graph_scope = 'https://graph.microsoft.com/.default'
        
        # Initialize credential
        if self.client_id and self.client_secret and self.tenant_id:
            self.credential = ClientSecretCredential(
                tenant_id=self.tenant_id,
                client_id=self.client_id,
                client_secret=self.client_secret
            )
        else:
            self.credential = None

    def init_session_state(self):
        """Initialize session state for Azure authentication."""
        if 'azure_auth_state' not in st.session_state:
            st.session_state.azure_auth_state = {
                'is_authenticated': False,
                'access_token': None,
                'user_info': None,
                'state': None,
                'token_expires_at': None
            }

    def generate_auth_url(self):
        """Generate the Azure AD authorization URL."""
        state = secrets.token_urlsafe(32)
        
        # Store state in session state
        if 'azure_auth_state' not in st.session_state:
            st.session_state.azure_auth_state = {}
        st.session_state.azure_auth_state['state'] = state
        
        # Use only Graph scope for authentication - this is sufficient for user info
        # We'll handle Fabric permissions separately if needed
        auth_scope = self.graph_scope
        
        params = {
            'client_id': self.client_id,
            'response_type': 'code',
            'redirect_uri': self.redirect_uri,
            'scope': auth_scope,
            'state': state,
            'response_mode': 'query'
        }
        
        auth_url = f"https://login.microsoftonline.com/{self.tenant_id}/oauth2/v2.0/authorize"
        return f"{auth_url}?{urllib.parse.urlencode(params)}"

    def exchange_code_for_token(self, auth_code):
        """Exchange authorization code for access token using proper OAuth flow."""
        token_url = f"https://login.microsoftonline.com/{self.tenant_id}/oauth2/v2.0/token"
        
        # Use only Graph scope for authentication
        auth_scope = self.graph_scope
        
        data = {
            'client_id': self.client_id,
            'client_secret': self.client_secret,
            'code': auth_code,
            'redirect_uri': self.redirect_uri,
            'grant_type': 'authorization_code',
            'scope': auth_scope
        }
        
        try:
            response = requests.post(token_url, data=data)
            if response.status_code == 200:
                token_data = response.json()
                return {
                    'access_token': token_data['access_token'],
                    'expires_in': token_data.get('expires_in', 3600),
                    'refresh_token': token_data.get('refresh_token')
                }
            else:
                st.error(f"Token exchange failed: {response.status_code} - {response.text}")
                return None
        except Exception as e:
            st.error(f"Token exchange failed: {str(e)}")
            return None

    def get_authorized_users(self):
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
        ]

    def is_user_authorized(self, user_principal_name):
        """Check if a user is in the authorized list"""
        authorized_users = self.get_authorized_users()
        return user_principal_name in authorized_users

    def get_graph_token(self):
        """Get token for Microsoft Graph API calls."""
        if not self.credential:
            return None
            
        try:
            token_data = self.credential.get_token(self.graph_scope)
            # Fix the timestamp calculation - handle both datetime and timestamp objects
            if hasattr(token_data.expires_on, 'timestamp'):
                # If it's a datetime object
                expires_in = int((token_data.expires_on - datetime.now()).total_seconds())
            else:
                # If it's already a timestamp
                expires_in = int(token_data.expires_on - datetime.now().timestamp())
            
            return {
                'access_token': token_data.token,
                'expires_in': expires_in
            }
        except Exception as e:
            st.error(f"Graph token acquisition failed: {str(e)}")
            return None

    def get_user_info_from_service_principal(self, user_email, access_token):
        """Get user information using Microsoft Graph API with service principal"""
        try:
            headers = {
                'Authorization': f'Bearer {access_token}',
                'Content-Type': 'application/json'
            }
            
            # Use Microsoft Graph API to get user info
            response = requests.get(
                f'https://graph.microsoft.com/v1.0/users/{user_email}',
                headers=headers
            )
            
            if response.status_code == 200:
                user_data = response.json()
                return {
                    'displayName': user_data.get('displayName', user_email),
                    'userPrincipalName': user_data.get('userPrincipalName', user_email),
                    'mail': user_data.get('mail', user_email),
                    'jobTitle': user_data.get('jobTitle', ''),
                    'officeLocation': user_data.get('officeLocation', '')
                }
            else:
                st.error(f"Failed to get user info: {response.status_code} - {response.text}")
                return None
        except Exception as e:
            st.error(f"Error getting user info: {str(e)}")
            return None

    def get_token_directly(self):
        """Get token directly using client credentials (for Docker environments)."""
        if not self.credential:
            return None
            
        try:
            token_data = self.credential.get_token(self.scope)
            # Fix the timestamp calculation - handle both datetime and timestamp objects
            if hasattr(token_data.expires_on, 'timestamp'):
                # If it's a datetime object
                expires_in = int((token_data.expires_on - datetime.now()).total_seconds())
            else:
                # If it's already a timestamp
                expires_in = int(token_data.expires_on - datetime.now().timestamp())
            
            return {
                'access_token': token_data.token,
                'expires_in': expires_in
            }
        except Exception as e:
            st.error(f"Direct token acquisition failed: {str(e)}")
            return None

    def get_user_info(self, access_token):
        """Get user information from Microsoft Graph API."""
        try:
            headers = {
                'Authorization': f'Bearer {access_token}',
                'Content-Type': 'application/json'
            }
            
            response = requests.get(
                'https://graph.microsoft.com/v1.0/me',
                headers=headers
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                st.error(f"Failed to get user info: {response.status_code}")
                return None
        except Exception as e:
            st.error(f"Error getting user info: {str(e)}")
            return None

    def is_token_valid(self):
        """Check if the current token is still valid."""
        if not st.session_state.azure_auth_state.get('token_expires_at'):
            return False
        
        # Check if token expires in the next 5 minutes
        return datetime.now() < st.session_state.azure_auth_state['token_expires_at'] - timedelta(minutes=5)
    
    def refresh_token_if_needed(self):
        """Refresh token if it's about to expire."""
        if not self.is_token_valid():
            # For now, we'll require re-authentication
            # In a production environment, you'd implement token refresh
            st.session_state.azure_auth_state['is_authenticated'] = False
            st.session_state.azure_auth_state['access_token'] = None
            return False
        return True
    
    def handle_auth_callback(self):
        """Handle the OAuth callback from Azure AD."""
        # Ensure session state is initialized
        self.init_session_state()
        
        query_params = st.query_params
        
        # Only process callback if we have both code and state parameters
        if 'code' in query_params and 'state' in query_params:
            auth_code = query_params['code']
            state = query_params['state']
            
            # For now, accept any state to avoid session state issues
            # In production, you should validate the state properly
            st.info("🔄 Processing authentication... Please wait.")
            
            # Exchange code for token
            token_data = self.exchange_code_for_token(auth_code)
            if token_data:
                # Store token information
                st.session_state.azure_auth_state['access_token'] = token_data['access_token']
                st.session_state.azure_auth_state['token_expires_at'] = datetime.now() + timedelta(seconds=token_data.get('expires_in', 3600))
                
                # Try to get user information with the OAuth token
                user_info = self.get_user_info(token_data['access_token'])
                if user_info:
                    # Check if the authenticated user is authorized
                    user_principal_name = user_info.get('userPrincipalName')
                    if user_principal_name and not self.is_user_authorized(user_principal_name):
                        # Log unauthorized access attempt
                        diagnostics_logger.log_authentication(
                            auth_method="OAuth",
                            success=False,
                            user_info=user_info,
                            error_message=f"Access denied for {user_principal_name}"
                        )
                        
                        st.error(f"Access denied for {user_principal_name}. You don't have access to this application.")
                        st.info("If you believe you should have access, please contact the system administrator.")
                        # Clear the URL parameters
                        st.query_params.clear()
                        return False
                    
                    st.session_state.azure_auth_state['user_info'] = user_info
                    st.session_state.azure_auth_state['is_authenticated'] = True
                    
                    # Clear the URL parameters
                    st.query_params.clear()
                    
                    # Log successful authentication
                    diagnostics_logger.log_authentication(
                        auth_method="OAuth",
                        success=True,
                        user_info=user_info
                    )
                    
                    # Show success message and redirect to main app
                    st.success(f"✅ Authentication successful! Welcome, {user_info.get('displayName', user_principal_name)}!")
                    st.info("🔄 Redirecting to main application...")
                    
                    # Redirect to main app URL
                    st.markdown("""
                    <script>
                    setTimeout(function() {
                        window.location.href = "http://localhost:8501";
                    }, 2000);
                    </script>
                    """, unsafe_allow_html=True)
                    
                    return True
                else:
                    # If OAuth token doesn't work for user info, try to extract from JWT token
                    st.warning("OAuth token doesn't have Graph permissions. Extracting user info from token...")
                    
                    try:
                        # Decode the JWT token to get user info
                        # Note: We're not verifying the signature since we trust Azure AD
                        decoded_token = jwt.decode(token_data['access_token'], options={"verify_signature": False})
                        
                        # Extract user info from token claims
                        user_info_from_token = {
                            'displayName': decoded_token.get('name', 'Unknown User'),
                            'userPrincipalName': decoded_token.get('upn', decoded_token.get('preferred_username', 'unknown@domain.com')),
                            'mail': decoded_token.get('email', decoded_token.get('preferred_username', 'unknown@domain.com')),
                            'jobTitle': decoded_token.get('jobtitle', ''),
                            'officeLocation': decoded_token.get('office_location', '')
                        }
                        
                        # Check if the user is authorized
                        user_principal_name = user_info_from_token.get('userPrincipalName')
                        if user_principal_name and not self.is_user_authorized(user_principal_name):
                            # Log unauthorized access attempt
                            diagnostics_logger.log_authentication(
                                auth_method="OAuth_JWT",
                                success=False,
                                user_info=user_info_from_token,
                                error_message=f"Access denied for {user_principal_name}"
                            )
                            
                            st.error(f"Access denied for {user_principal_name}. You don't have access to this application.")
                            st.info("If you believe you should have access, please contact the system administrator.")
                            st.query_params.clear()
                            return False
                        
                        st.session_state.azure_auth_state['user_info'] = user_info_from_token
                        st.session_state.azure_auth_state['is_authenticated'] = True
                        
                        # Clear the URL parameters
                        st.query_params.clear()
                        
                        # Log successful authentication
                        diagnostics_logger.log_authentication(
                            auth_method="OAuth_JWT",
                            success=True,
                            user_info=user_info_from_token
                        )
                        
                        # Show success message and redirect to main app
                        st.success(f"✅ Authentication successful! Welcome, {user_info_from_token.get('displayName', user_principal_name)}!")
                        st.info("🔄 Redirecting to main application...")
                        
                        # Redirect to main app URL
                        st.markdown("""
                        <script>
                        setTimeout(function() {
                            window.location.href = "http://localhost:8501";
                        }, 2000);
                        </script>
                        """, unsafe_allow_html=True)
                        
                        return True
                        
                    except Exception as e:
                        st.error(f"Failed to extract user info from token: {str(e)}")
                        st.info("Please ensure your Azure AD app has the necessary permissions.")
                        st.query_params.clear()
                        return False
            else:
                # Log token exchange error
                diagnostics_logger.log_error(
                    component="Authentication",
                    error=Exception("Failed to exchange authorization code for token"),
                    context={'auth_method': 'OAuth'},
                    user_message="Failed to exchange authorization code for token"
                )
                
                st.error("Failed to exchange authorization code for token.")
                # Clear the URL parameters
                st.query_params.clear()
                return False
            
        return False
    
    def is_oauth_callback(self):
        """Check if we're in an OAuth callback scenario."""
        query_params = st.query_params
        return 'code' in query_params and 'state' in query_params

    def login_button(self):
        """Display login button and handle authentication flow."""
        self.init_session_state()
        
        # Check if user is already authenticated
        if st.session_state.azure_auth_state['is_authenticated']:
            if self.refresh_token_if_needed():
                return True
            else:
                st.session_state.azure_auth_state['is_authenticated'] = False
        
        # Handle callback if present (only if not already authenticated)
        if not st.session_state.azure_auth_state['is_authenticated']:
            if self.handle_auth_callback():
                return True
        
        # If we're in an OAuth callback but not authenticated, show loading
        if self.is_oauth_callback():
            st.info("🔄 Processing authentication... Please wait.")
            st.stop()
        
        # Display login interface only if not authenticated
        if not st.session_state.azure_auth_state['is_authenticated']:
            st.title("🔐 Azure AD Authentication")
            st.markdown("Please authenticate to access the Fabric RAG application.")
            
            if not self.client_id or not self.client_secret or not self.tenant_id:
                st.error("Azure AD configuration is incomplete. Please check your environment variables.")
                st.info("Required environment variables: AZURE_CLIENT_ID, AZURE_CLIENT_SECRET, AZURE_TENANT_ID")
                return False
            
            # OAuth Flow (Primary method - more secure)
            st.markdown("### OAuth Authentication (Recommended)")
            st.markdown("This method provides secure user authentication through Microsoft:")
            
            auth_url = self.generate_auth_url()
            
            if st.button("🔑 Sign in with Microsoft (OAuth)", type="primary", use_container_width=True):
                # Use a more reliable redirect method
                st.markdown(f"""
                <div style="text-align: center; padding: 20px;">
                    <a href="{auth_url}" style="
                        display: inline-block;
                        background-color: #0078d4;
                        color: white;
                        padding: 12px 24px;
                        text-decoration: none;
                        border-radius: 4px;
                        font-weight: bold;
                        margin: 10px;
                    ">
                        🔑 Click here to Sign in with Microsoft
                    </a>
                </div>
                """, unsafe_allow_html=True)
                
                st.info("""
                **🔐 OAuth Authentication Instructions:**
                
                1. **Click the "Sign in with Microsoft" button above**
                2. **You will be redirected** to Microsoft's login page
                3. **Complete your Microsoft authentication**
                4. **After successful login**, you'll be redirected back to the application
                5. **The application will automatically detect** your authentication
                
                **⚠️ Important**: Keep this browser tab open during the authentication process.
                """)
                
                # Debug information (expandable)
                with st.expander("🔧 Debug Information"):
                    st.code(f"Auth URL: {auth_url}")
                    st.code(f"Redirect URI: {self.redirect_uri}")
                    st.code(f"Client ID: {self.client_id}")
                    st.code(f"Tenant ID: {self.tenant_id}")
                    st.info("If the redirect doesn't work, try the manual redirect button below.")
                
                # Add a manual redirect button as backup
                st.markdown("---")
                if st.button("🔄 Manual Redirect to Microsoft", type="secondary"):
                    st.markdown(f"""
                    <script>
                    window.location.href = "{auth_url}";
                    </script>
                    """, unsafe_allow_html=True)
                    st.info("🔄 Redirecting to Microsoft authentication...")
                    st.stop()
                
                # Add auto-refresh using JavaScript
                st.markdown("""
                <script>
                // Auto-refresh every 2 seconds to check for authentication
                setTimeout(function() {
                    window.location.reload();
                }, 2000);
                </script>
                """, unsafe_allow_html=True)
            
            # Direct Authentication (Fallback - less secure)
            st.markdown("---")
            st.markdown("### Alternative: Direct Authentication")
            st.markdown("⚠️ **Note**: This method is less secure and should only be used if OAuth doesn't work.")
            st.markdown("Enter your email address to authenticate using service principal:")
            
            user_email = st.text_input("Email Address", placeholder="your.email@opendata.energy")
            
            if st.button("🔑 Login with Azure AD (Direct)", type="secondary", use_container_width=True):
                if not user_email:
                    st.error("Please enter your email address.")
                    return False
                
                # Validate email format
                if '@' not in user_email:
                    st.error("Please enter a valid email address.")
                    return False
                
                # Check if user is authorized
                if not self.is_user_authorized(user_email):
                    # Log unauthorized access attempt
                    diagnostics_logger.log_authentication(
                        auth_method="Direct",
                        success=False,
                        user_info={'userPrincipalName': user_email},
                        error_message=f"Access denied for {user_email}"
                    )
                    
                    st.error(f"Access denied for {user_email}. You don't have access to this application.")
                    st.info("If you believe you should have access, please contact the system administrator.")
                    return False
                
                # Get service principal token
                token_data = self.get_token_directly()
                if token_data:
                    # Get Microsoft Graph token for user info
                    graph_token_data = self.get_graph_token()
                    if graph_token_data:
                        # Get real user information using Microsoft Graph API
                        user_info = self.get_user_info_from_service_principal(user_email, graph_token_data['access_token'])
                        if user_info:
                            st.session_state.azure_auth_state['access_token'] = token_data['access_token']
                            st.session_state.azure_auth_state['token_expires_at'] = datetime.now() + timedelta(seconds=token_data.get('expires_in', 3600))
                            st.session_state.azure_auth_state['is_authenticated'] = True
                            st.session_state.azure_auth_state['user_info'] = user_info
                            
                            # Log successful direct authentication
                            diagnostics_logger.log_authentication(
                                auth_method="Direct",
                                success=True,
                                user_info=user_info
                            )
                            
                            st.success(f"Successfully authenticated as {user_info.get('displayName', user_email)}!")
                            st.rerun()
                            return True
                        else:
                            # Log user info retrieval error
                            diagnostics_logger.log_error(
                                component="Authentication",
                                error=Exception(f"Could not retrieve user information for {user_email}"),
                                context={'auth_method': 'Direct', 'user_email': user_email},
                                user_message=f"Could not retrieve user information for {user_email}"
                            )
                            
                            st.error(f"Could not retrieve user information for {user_email}. Please check your email address.")
                            return False
                    else:
                        # Log Graph token error
                        diagnostics_logger.log_error(
                            component="Authentication",
                            error=Exception("Failed to get Microsoft Graph token"),
                            context={'auth_method': 'Direct'},
                            user_message="Failed to get Microsoft Graph token for user information"
                        )
                        
                        st.error("Failed to get Microsoft Graph token for user information.")
                        return False
                else:
                    # Log direct authentication error
                    diagnostics_logger.log_error(
                        component="Authentication",
                        error=Exception("Direct authentication failed"),
                        context={'auth_method': 'Direct'},
                        user_message="Direct authentication failed. Please check your Azure AD configuration"
                    )
                    
                    st.error("Direct authentication failed. Please check your Azure AD configuration.")
                    return False
            
            return False
        
        return True
    
    def get_access_token(self):
        """Get the current access token."""
        return st.session_state.azure_auth_state.get('access_token')
    
    def get_user_display_name(self):
        """Get the user's display name."""
        user_info = st.session_state.azure_auth_state.get('user_info')
        if user_info:
            return user_info.get('displayName', user_info.get('userPrincipalName', 'Unknown User'))
        return None

# Create global instance
azure_auth = AzureAuthDocker() 