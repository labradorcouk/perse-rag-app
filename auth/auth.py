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

def add_authorized_user(email):
    """Add a user to the authorized list (for admin use)"""
    # This could be extended to save to a database or config file
    st.info(f"To add {email} to authorized users, set the AUTHORIZED_USERS environment variable or update the code.")

def remove_authorized_user(email):
    """Remove a user from the authorized list (for admin use)"""
    # This could be extended to save to a database or config file
    st.info(f"To remove {email} from authorized users, update the AUTHORIZED_USERS environment variable or code.")

def init_auth_session_state():
    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = False
    if 'show_register' not in st.session_state:
        st.session_state.show_register = False
    if 'user_role' not in st.session_state:
        st.session_state.user_role = None
    if 'user_info' not in st.session_state:
        st.session_state.user_info = None

def clear_credential_cache(get_credential):
    get_credential.clear()
    st.session_state.authenticated = False
    st.session_state.user_role = None
    st.session_state.user_info = None
    st.session_state.show_register = False

def get_user_info_from_token(token):
    try:
        decoded = jwt.decode(token, options={"verify_signature": False})
        return decoded.get('upn')
    except Exception as e:
        st.error(f"Error decoding token: {str(e)}")
        return None

retry_strategy = Retry(total=3, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504])
adapter = HTTPAdapter(max_retries=retry_strategy)
session = requests.Session()
session.mount("https://", adapter)
session.mount("http://", adapter)

def make_graphql_request(endpoint, query, variables=None, headers=None):
    try:
        time.sleep(0.5)
        response = session.post(
            endpoint,
            json={'query': query, 'variables': variables} if variables else {'query': query},
            headers=headers
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        if e.response is not None and e.response.status_code == 429:
            st.warning("Rate limit reached. Please wait a moment and try again.")
        else:
            st.error(f"Error making GraphQL request: {str(e)}")
        return None

def fetch_user_role(user_principal_name, get_credential):
    try:
        app, scp = get_credential()
        result = app.get_token(scp)
        if not result.token:
            st.error("Could not get access token")
            return None
        endpoint = 'https://3b1565aabd8147d394c135b1d3761d87.z3b.graphql.fabric.microsoft.com/v1/workspaces/3b1565aa-bd81-47d3-94c1-35b1d3761d87/graphqlapis/2eae604b-a405-49f7-83ab-12dc279de279/graphql'
        query = """
        query($loggedInUser: String!){
            users (first: 300, filter:  {
               userPrincipalName:  {
                  eq: $loggedInUser
               }
            }){
                items {
                    businessPhones
                    displayName
                    givenName
                    jobTitle
                    mail
                    mobilePhone
                    officeLocation
                    preferredLanguage
                    surname
                    userPrincipalName
                    id
                    roles {
                        items {
                            description
                            roleID
                            roleName
                            userID
                            username
                        }
                    }
                }
            }
        }
        """
        headers = {
            'Authorization': f'Bearer {result.token}',
            'Content-Type': 'application/json'
        }
        variables = {"loggedInUser": user_principal_name}
        data = make_graphql_request(endpoint, query, variables, headers)
        if data is None:
            return None
        if 'errors' in data:
            st.error(f"GraphQL errors: {data['errors']}")
            return None
        users = data.get('data', {}).get('users', {}).get('items', [])
        if not users:
            return None
        user = users[0]
        roles = user.get('roles', {}).get('items', [])
        if not roles:
            return None
        return {
            'role': roles[0].get('roleName'),
            'user_info': user
        }
    except Exception as e:
        st.error(f"Error fetching user role: {str(e)}")
        return None

def authenticate_with_azure(get_credential, fetch_user_role_func):
    try:
        app, scp = get_credential()
        result = app.get_token(scp)
        if result.token:
            user_principal_name = get_user_info_from_token(result.token)
            if user_principal_name:
                # Check if the logged-in user is in the authorized list
                if is_user_authorized(user_principal_name):
                    st.session_state.authenticated = True
                    st.session_state.user_role = "Admin"
                    st.session_state.user_info = {
                        'userPrincipalName': user_principal_name,
                        'displayName': "Authorized User"
                    }
                    st.success(f"Welcome, {user_principal_name}! You are authorized to access this application.")
                    return True
                
                # If not in the authorized list, try to fetch role from GraphQL
                role_info = fetch_user_role_func(user_principal_name)
                if role_info:
                    st.session_state.authenticated = True
                    st.session_state.user_role = role_info['role']
                    st.session_state.user_info = role_info['user_info']
                    st.success(f"Welcome, {user_principal_name}! Role: {role_info['role']}")
                    return True
                else:
                    st.error(f"Access denied for {user_principal_name}. You don't have access to this dashboard. Please contact your administrator.")
                    st.info("If you believe you should have access, please contact the system administrator to be added to the authorized users list.")
                    return False
        return False
    except Exception as e:
        st.error(f"Authentication failed: {str(e)}")
        return False

def show_registration_form(get_credential, send_registration_email_func):
    st.subheader("Register")
    with st.form("registration_form"):
        email = st.text_input("Email")
        first_name = st.text_input("First Name")
        last_name = st.text_input("Last Name")
        job_title = st.text_input("Job Title")
        phone = st.text_input("Phone Number")
        role = st.selectbox("Select Role", ["DNOAnalyst", "EnergySystemAcademic", "EnergyAppDev", "Landlord"])
        submit_button = st.form_submit_button("Submit Registration")
        if submit_button:
            if not all([email, first_name, last_name, job_title, phone, role]):
                st.error("Please fill in all fields")
                return
            try:
                app, scp = get_credential()
                result = app.get_token(scp)
                if not result.token:
                    st.error("Could not get access token")
                    return
                endpoint = 'https://3b1565aabd8147d394c135b1d3761d87.z3b.graphql.fabric.microsoft.com/v1/workspaces/3b1565aa-bd81-47d3-94c1-35b1d3761d87/graphqlapis/2eae604b-a405-49f7-83ab-12dc279de279/graphql'
                mutation = """
                mutation($input: CreateUserInput!) {
                    createUser(input: $input) {
                        id
                        displayName
                        mail
                        jobTitle
                        businessPhones
                        roles {
                            items {
                                roleName
                            }
                        }
                    }
                }
                """
                variables = {
                    "input": {
                        "mail": email,
                        "givenName": first_name,
                        "surname": last_name,
                        "jobTitle": job_title,
                        "businessPhones": [phone],
                        "roleName": role
                    }
                }
                headers = {
                    'Authorization': f'Bearer {result.token}',
                    'Content-Type': 'application/json'
                }
                with st.spinner('Submitting registration...'):
                    response = session.post(
                        endpoint,
                        json={'query': mutation, 'variables': variables},
                        headers=headers
                    )
                    if response.status_code == 200:
                        data = response.json()
                        if 'errors' in data:
                            st.error(f"Registration failed: {data['errors'][0]['message']}")
                        else:
                            st.success("Registration submitted successfully! Please wait for admin approval.")
                            send_registration_email_func(email, first_name, last_name, role)
                            st.session_state.show_register = False
                            st.rerun()
                    else:
                        st.error(f"Registration failed with status code: {response.status_code}")
            except Exception as e:
                st.error(f"Error during registration: {str(e)}")
    if st.button("Back to Login"):
        st.session_state.show_register = False
        st.rerun()

def send_registration_email(email, first_name, last_name, role):
    try:
        sender_email = os.getenv("EMAIL_USER")
        sender_password = os.getenv("EMAIL_PASSWORD")
        admin_email = os.getenv("ADMIN_EMAIL")
        if not all([sender_email, sender_password, admin_email]):
            st.warning("Email configuration not complete. Skipping email notification.")
            return
        msg = MIMEMultipart()
        msg['From'] = sender_email
        msg['To'] = admin_email
        msg['Subject'] = "New User Registration - Fabric RAG App"
        body = f"""
        A new user has registered for the Fabric RAG App:\n\nName: {first_name} {last_name}\nEmail: {email}\nRole: {role}\n\nPlease review and approve this registration.
        """
        msg.attach(MIMEText(body, 'plain'))
        with smtplib.SMTP('smtp.office365.com', 587) as server:
            server.starttls()
            server.login(sender_email, sender_password)
            server.send_message(msg)
    except Exception as e:
        st.warning(f"Could not send registration email: {str(e)}")

def show_login(authenticate_with_azure_func):
    st.subheader("Login")
    with st.form("login_form"):
        col1, col2 = st.columns(2)
        with col1:
            login_button = st.form_submit_button("Login with Microsoft")
        with col2:
            register_button = st.form_submit_button("Register")
        if login_button:
            if authenticate_with_azure_func():
                st.rerun()
        if register_button:
            st.session_state.show_register = True
            st.rerun()

def main_auth(get_credential):
    # Removed relative import to avoid ImportError in Streamlit
    init_auth_session_state()
    if not st.session_state.authenticated:
        if st.session_state.show_register:
            show_registration_form(get_credential, send_registration_email)
        else:
            show_login(lambda: authenticate_with_azure(get_credential, lambda upn: fetch_user_role(upn, get_credential)))
        st.stop()
    with st.sidebar:
        if st.session_state.user_info:
            st.write(f"Welcome, {st.session_state.user_info.get('displayName', 'User')}")
            if st.session_state.user_role == "Admin":
                st.write("Role: Admin (Full Access)")
            else:
                st.write(f"Role: {st.session_state.user_role}")
        if st.button("Logout"):
            clear_credential_cache(get_credential)
            st.rerun() 