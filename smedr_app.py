import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import requests
from azure.identity import InteractiveBrowserCredential
import json
from datetime import datetime
import numpy as np
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import os
from dotenv import load_dotenv
import jwt
import time
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import plotly.graph_objects as go

# Load environment variables
load_dotenv()

# Configure retry strategy
retry_strategy = Retry(
    total=3,  # number of retries
    backoff_factor=1,  # wait 1, 2, 4 seconds between retries
    status_forcelist=[429, 500, 502, 503, 504]  # HTTP status codes to retry on
)
adapter = HTTPAdapter(max_retries=retry_strategy)
session = requests.Session()
session.mount("https://", adapter)
session.mount("http://", adapter)

# Set page config
st.set_page_config(
    page_title="SMEDR Demo",
    page_icon="📊",
    layout="wide"
)

# Title
st.title("Smart Meter Energy Data Repository (SMEDR) Demo")

# Initialize session state for authentication
if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False
if 'show_register' not in st.session_state:
    st.session_state.show_register = False
if 'user_role' not in st.session_state:
    st.session_state.user_role = None
if 'user_info' not in st.session_state:
    st.session_state.user_info = None

# Initialize the credential once
@st.cache_resource
def get_credential():
    app = InteractiveBrowserCredential()
    scp = 'https://analysis.windows.net/powerbi/api/user_impersonation'
    return app, scp

# Function to clear credential cache
def clear_credential_cache():
    # Clear the cached credential
    get_credential.clear()
    # Clear session state
    st.session_state.authenticated = False
    st.session_state.user_role = None
    st.session_state.user_info = None
    st.session_state.show_register = False

# Function to get user info from token
def get_user_info_from_token(token):
    try:
        # The token is already a JWT token, we can decode it directly
        decoded = jwt.decode(token, options={"verify_signature": False})
        return decoded.get('upn')  # userPrincipalName
    except Exception as e:
        st.error(f"Error decoding token: {str(e)}")
        return None

# Function to make GraphQL request with retry logic
def make_graphql_request(endpoint, query, variables=None, headers=None):
    try:
        # Add delay between requests to avoid rate limiting
        time.sleep(0.5)  # Reduced from 1 second to 0.5 seconds
        
        # Make the request
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

# Function to fetch user role from GraphQL
def fetch_user_role(user_principal_name):
    try:
        app, scp = get_credential()
        result = app.get_token(scp)

        if not result.token:
            st.error("Could not get access token")
            return None

        # GraphQL endpoint
        endpoint = 'https://3b1565aabd8147d394c135b1d3761d87.z3b.graphql.fabric.microsoft.com/v1/workspaces/3b1565aa-bd81-47d3-94c1-35b1d3761d87/graphqlapis/2eae604b-a405-49f7-83ab-12dc279de279/graphql'
        
        # GraphQL query
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
        
        # Prepare headers
        headers = {
            'Authorization': f'Bearer {result.token}',
            'Content-Type': 'application/json'
        }
        
        # Prepare variables
        variables = {
            "loggedInUser": user_principal_name
        }
        
        # Make the request with retry logic
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

# Cache the usage by DNO data
@st.cache_data(ttl=300)  # Cache for 5 minutes
def fetch_usage_by_dno():
    try:
        app, scp = get_credential()
        result = app.get_token(scp)
        
        if not result.token:
            st.error("Could not get access token")
            return None

        # GraphQL endpoint
        endpoint = 'https://3b1565aabd8147d394c135b1d3761d87.z3b.graphql.fabric.microsoft.com/v1/workspaces/3b1565aa-bd81-47d3-94c1-35b1d3761d87/graphqlapis/2eae604b-a405-49f7-83ab-12dc279de279/graphql'
        
        # GraphQL query to fetch all records
        query = """
        query {
            usageByDNOs(first: 10000) {
                items {
                    dno
                    createdDate
                    total_usage
                }
            }
        }
        """
        
        # Prepare headers
        headers = {
            'Authorization': f'Bearer {result.token}',
            'Content-Type': 'application/json'
        }
        
        # Make the request with retry logic
        data = make_graphql_request(endpoint, query, headers=headers)
        if data is None:
            return None
        
        if 'errors' in data:
            st.error(f"GraphQL errors: {data['errors']}")
            return None
        
        # Extract items
        items = data.get('data', {}).get('usageByDNOs', {}).get('items', [])
        
        if not items:
            st.warning("No data returned from the API")
            return None
        
        # Convert to DataFrame
        df = pd.DataFrame(items)
        
        # Convert createdDate to datetime
        df['createdDate'] = pd.to_datetime(df['createdDate'])
        
        return df
        
    except Exception as e:
        st.error(f"Error fetching usage by DNO data: {str(e)}")
        return None

# Cache the premise type data
@st.cache_data(ttl=300)  # Cache for 5 minutes
def fetch_premise_type_data():
    try:
        app, scp = get_credential()
        result = app.get_token(scp)
        
        if not result.token:
            st.error("Could not get access token")
            return None
        
        # GraphQL endpoint
        endpoint = 'https://3b1565aabd8147d394c135b1d3761d87.z3b.graphql.fabric.microsoft.com/v1/workspaces/3b1565aa-bd81-47d3-94c1-35b1d3761d87/graphqlapis/2eae604b-a405-49f7-83ab-12dc279de279/graphql'
        
        # GraphQL query
        query = """
        query {
            premiseTypeDNOs(first: 10000) {
                items {
                    premise_type
                    dno
                    energy_efficiency_rating
                    floor_type
                    glazing_type
                    main_heating_fuel
                    map_floors
                    sum_habitable_rooms
                    sum_open_fireplaces
                    sum_bathroom_count
                    sum_bedroom_count
                    sum_building_area
                    sum_height
                }
            }
        }
        """
        
        # Prepare headers
        headers = {
            'Authorization': f'Bearer {result.token}',
            'Content-Type': 'application/json'
        }
        
        # Make the request with retry logic
        data = make_graphql_request(endpoint, query, headers=headers)
        if data is None:
            return None
        
        if 'errors' in data:
            st.error(f"GraphQL errors: {data['errors']}")
            return None
        
        # Extract items
        items = data.get('data', {}).get('premiseTypeDNOs', {}).get('items', [])
        
        if not items:
            st.warning("No data returned from the API")
            return None
        
        # Convert to DataFrame
        df = pd.DataFrame(items)
        return df
        
    except Exception as e:
        st.error(f"Error fetching premise type data: {str(e)}")
        return None

# Cache the service request data
@st.cache_data(ttl=300)  # Cache for 5 minutes
def fetch_service_request_data():
    try:
        app, scp = get_credential()
        result = app.get_token(scp)
        
        if not result.token:
            st.error("Could not get access token")
            return None

        # GraphQL endpoint
        endpoint = 'https://3b1565aabd8147d394c135b1d3761d87.z3b.graphql.fabric.microsoft.com/v1/workspaces/3b1565aa-bd81-47d3-94c1-35b1d3761d87/graphqlapis/2eae604b-a405-49f7-83ab-12dc279de279/graphql'
        
        # GraphQL query
        query = """
        query {
            serviceRequestDNOs(first: 10000) {
                items {
                    dno
                    distinct_code_count
                }
            }
        }
        """
        
        # Prepare headers
        headers = {
            'Authorization': f'Bearer {result.token}',
            'Content-Type': 'application/json'
        }
        
        # Make the request with retry logic
        data = make_graphql_request(endpoint, query, headers=headers)
        if data is None:
            return None
        
        if 'errors' in data:
            st.error(f"GraphQL errors: {data['errors']}")
            return None
        
        # Extract items
        items = data.get('data', {}).get('serviceRequestDNOs', {}).get('items', [])
        
        if not items:
            st.warning("No data returned from the API")
            return None
        
        # Convert to DataFrame
        df = pd.DataFrame(items)
        return df
        
    except Exception as e:
        st.error(f"Error fetching service request data: {str(e)}")
        return None

# Function to show usage by DNO visualization
def show_usage_by_dno():
    st.subheader("Usage by DNO")
    
    # Fetch data with loading spinner
    with st.spinner('Loading data...'):
        df = fetch_usage_by_dno()
    
    if df is None:
        st.error("Could not fetch usage by DNO data")
        return
    
    # Create two columns for the layout
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Date range selector in a more compact format
        date_col1, date_col2 = st.columns(2)
        with date_col1:
            start_date = st.date_input("Start Date", df['createdDate'].min(), key="start_date_dno")
        with date_col2:
            end_date = st.date_input("End Date", df['createdDate'].max(), key="end_date_dno")
    
    # Filter data based on date range
    mask = (df['createdDate'].dt.date >= start_date) & (df['createdDate'].dt.date <= end_date)
    filtered_df = df[mask]
    
    # Group by DNO and sum total_usage
    dno_usage = filtered_df.groupby('dno')['total_usage'].sum().reset_index()
    dno_usage = dno_usage.sort_values('total_usage', ascending=False)
    
    # Create visualization with smaller size and font
    plt.figure(figsize=(8, 4))
    plt.rcParams.update({'font.size': 8})  # Reduce font size
    
    bars = plt.bar(dno_usage['dno'], dno_usage['total_usage'])
    
    # Add value labels on top of bars with smaller font
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:,.0f}',
            ha='center', va='bottom', fontsize=7)
    
    plt.title('Total Usage by DNO', fontsize=10)
    plt.xlabel('DNO', fontsize=8)
    plt.ylabel('Total Usage (kWh)', fontsize=8)
    plt.xticks(rotation=45, ha='right', fontsize=7)
    plt.tight_layout()
    
    # Display the plot in the first column
    with col1:
        st.pyplot(plt)
        
    # Display the raw data in the second column with compact styling
    with col2:
        st.subheader("Data Summary", divider='gray')
        # Format the numbers in the dataframe
        display_df = dno_usage.copy()
        display_df['total_usage'] = display_df['total_usage'].apply(lambda x: f'{x:,.0f}')
        display_df.columns = ['DNO', 'Total Usage']
        
        # Display with compact styling
        st.dataframe(
            display_df,
            height=300,
            use_container_width=True,
            hide_index=True
        )
        
        # Add some summary statistics
        st.markdown("---")
        st.markdown("**Summary Statistics**")
        st.markdown(f"Total DNOs: {len(dno_usage)}")
        st.markdown(f"Highest Usage: {dno_usage['total_usage'].max():,.0f}")
        st.markdown(f"Lowest Usage: {dno_usage['total_usage'].min():,.0f}")

# Function to show premise type visualization
def show_premise_type():
    st.subheader("Premise Type Analysis")
    
    # Fetch data with loading spinner
    with st.spinner('Loading premise type data...'):
        df = fetch_premise_type_data()
    
    if df is None:
        st.error("Could not fetch premise type data")
        return
    
    # Create two columns for the layout
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # DNO selector
        selected_dno = st.selectbox(
            "Select DNO",
            options=sorted(df['dno'].unique()),
            key="dno_selector_premise"
        )
    
    # Filter data for selected DNO
    filtered_df = df[df['dno'] == selected_dno]
    
    # Create visualization with smaller size and font
    plt.figure(figsize=(8, 4))
    plt.rcParams.update({'font.size': 8})  # Reduce font size
    
    # Group by premise_type and count
    premise_counts = filtered_df['premise_type'].value_counts()
    
    # Create bar chart
    bars = plt.bar(premise_counts.index, premise_counts.values)
    
    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
            f'{int(height):,}',
            ha='center', va='bottom', fontsize=7)
    
    plt.title(f'Premise Types for {selected_dno}', fontsize=10)
    plt.xlabel('Premise Type', fontsize=8)
    plt.ylabel('Count', fontsize=8)
    plt.xticks(rotation=45, ha='right', fontsize=7)
    plt.tight_layout()
    
    # Display the plot in the first column
    with col1:
        st.pyplot(plt)
        
    # Display the summary statistics in the second column
    with col2:
        st.subheader("Summary Statistics", divider='gray')
        
        # Calculate and display key metrics
        st.markdown(f"**Total Premises:** {len(filtered_df):,}")
        
        # EPC Rating Distribution
        epc_counts = filtered_df['energy_efficiency_rating'].value_counts()
        st.markdown("**EPC Rating Distribution:**")
        for rating, count in epc_counts.items():
            st.markdown(f"- {rating}: {count:,}")
        
        # Convert numeric columns to float, handling any non-numeric values
        numeric_columns = ['sum_habitable_rooms', 'sum_bedroom_count', 'sum_building_area']
        for col in numeric_columns:
            filtered_df[col] = pd.to_numeric(filtered_df[col], errors='coerce')
        
        # Display numeric summaries
        st.markdown("---")
        st.markdown("**Property Statistics:**")
        st.markdown(f"**Total Habitable Rooms:** {filtered_df['sum_habitable_rooms'].sum():,.0f}")
        st.markdown(f"**Total Bedrooms:** {filtered_df['sum_bedroom_count'].sum():,.0f}")
        st.markdown(f"**Total Building Area:** {filtered_df['sum_building_area'].sum():,.0f} m²")
        
        # Display top 5 premise types
        st.markdown("---")
        st.markdown("**Top 5 Premise Types**")
        top_premises = premise_counts.head()
        for premise, count in top_premises.items():
            st.markdown(f"- {premise}: {count:,}")
        
        # Additional property characteristics
        st.markdown("---")
        st.markdown("**Property Characteristics:**")
        
        # Floor type distribution
        floor_counts = filtered_df['floor_type'].value_counts().head(3)
        st.markdown("**Top Floor Types:**")
        for floor, count in floor_counts.items():
            st.markdown(f"- {floor}: {count:,}")
        
        # Glazing type distribution
        glazing_counts = filtered_df['glazing_type'].value_counts().head(3)
        st.markdown("**Top Glazing Types:**")
        for glazing, count in glazing_counts.items():
            st.markdown(f"- {glazing}: {count:,}")
        
        # Main heating fuel distribution
        heating_counts = filtered_df['main_heating_fuel'].value_counts().head(3)
        st.markdown("**Top Heating Fuels:**")
        for fuel, count in heating_counts.items():
            st.markdown(f"- {fuel}: {count:,}")
    
    # Add tabular view section
    st.markdown("---")
    st.subheader("Detailed Data View")
    
    # Create a pivot table for categorical columns
    categorical_cols = ['premise_type', 'energy_efficiency_rating', 'floor_type', 'glazing_type', 'main_heating_fuel']
    numeric_cols = ['sum_habitable_rooms', 'sum_bedroom_count', 'sum_building_area', 'sum_height']
    
    # Create tabs for different views
    tab1, tab2 = st.tabs(["Matrix View", "Raw Data"])
    
    with tab1:
        # Create a matrix view using pivot tables
        st.markdown("**Property Characteristics Matrix**")
        
        # Create a pivot table for premise type vs energy rating
        pivot1 = pd.pivot_table(
            filtered_df,
            values='sum_building_area',
            index='premise_type',
            columns='energy_efficiency_rating',
            aggfunc='count',
            fill_value=0
        )
        
        # Format the pivot table
        st.dataframe(
            pivot1.style.format("{:,.0f}"),
            use_container_width=True
        )
        
        st.markdown("**Property Types by Floor Type**")
        pivot2 = pd.pivot_table(
            filtered_df,
            values='sum_building_area',
            index='premise_type',
            columns='floor_type',
            aggfunc='count',
            fill_value=0
        )
        
        st.dataframe(
            pivot2.style.format("{:,.0f}"),
            use_container_width=True
        )
        
        st.markdown("**Property Types by Heating Fuel**")
        pivot3 = pd.pivot_table(
            filtered_df,
            values='sum_building_area',
            index='premise_type',
            columns='main_heating_fuel',
            aggfunc='count',
            fill_value=0
        )
        
        st.dataframe(
            pivot3.style.format("{:,.0f}"),
            use_container_width=True
        )
    
    with tab2:
        # Show raw data with selected columns
        display_cols = categorical_cols + numeric_cols
        st.dataframe(
            filtered_df[display_cols].sort_values('premise_type'),
            use_container_width=True,
            height=400
        )

# Function to show service request codes visualization
def show_service_request_codes():
    st.subheader("Codes by DNO")
    
    # Fetch data with loading spinner
    with st.spinner('Loading service request data...'):
        df = fetch_service_request_data()
    
    if df is None:
        st.error("Could not fetch service request data")
        return
    
    # Create two columns for the layout
    col1, col2 = st.columns([2, 1])
    
    # Sort data by distinct_code_count
    df = df.sort_values('distinct_code_count', ascending=False)
    
    # Create visualization with smaller size and font
    plt.figure(figsize=(8, 4))
    plt.rcParams.update({'font.size': 8})  # Reduce font size
    
    # Create bar chart
    bars = plt.bar(df['dno'], df['distinct_code_count'])
    
    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
            f'{int(height):,}',
            ha='center', va='bottom', fontsize=7)
    
    plt.title('Number of Distinct Service Request Codes by DNO', fontsize=10)
    plt.xlabel('DNO', fontsize=8)
    plt.ylabel('Number of Distinct Codes', fontsize=8)
    plt.xticks(rotation=45, ha='right', fontsize=7)
    plt.tight_layout()
    
    # Display the plot in the first column
    with col1:
        st.pyplot(plt)
        
    # Display the summary statistics in the second column
    with col2:
        st.subheader("Summary Statistics", divider='gray')
        
        # Calculate and display key metrics
        st.markdown(f"**Total DNOs:** {len(df):,}")
        st.markdown(f"**Total Distinct Codes:** {df['distinct_code_count'].sum():,}")
        st.markdown(f"**Average Codes per DNO:** {df['distinct_code_count'].mean():.1f}")
        
        # Display top 5 DNOs by code count
        st.markdown("---")
        st.markdown("**Top 5 DNOs by Code Count**")
        top_dnos = df.head()
        for _, row in top_dnos.iterrows():
            st.markdown(f"- {row['dno']}: {row['distinct_code_count']:,}")
        
        # Display bottom 5 DNOs by code count
        st.markdown("---")
        st.markdown("**Bottom 5 DNOs by Code Count**")
        bottom_dnos = df.tail()
        for _, row in bottom_dnos.iterrows():
            st.markdown(f"- {row['dno']}: {row['distinct_code_count']:,}")
    
    # Add tabular view section
    st.markdown("---")
    st.subheader("Detailed Data View")
    
    # Create tabs for different views
    tab1, tab2 = st.tabs(["Summary Table", "Raw Data"])
    
    with tab1:
        # Create a summary table with statistics
        summary_df = df.describe()
        summary_df.index = ['Count', 'Mean', 'Std Dev', 'Min', '25%', 'Median', '75%', 'Max']
        st.dataframe(
            summary_df.style.format("{:,.1f}"),
            use_container_width=True
        )
    
    with tab2:
        # Show raw data
        st.dataframe(
            df.sort_values('distinct_code_count', ascending=False),
            use_container_width=True,
            height=400
        )

# Function to show API testing interface
def show_api_testing_interface():
    st.subheader("GraphQL API Testing Interface")
    
    # API Endpoint (read-only)
    st.markdown("**API Endpoint**")
    endpoint = 'https://3b1565aabd8147d394c135b1d3761d87.z3b.graphql.fabric.microsoft.com/v1/workspaces/3b1565aa-bd81-47d3-94c1-35b1d3761d87/graphqlapis/2eae604b-a405-49f7-83ab-12dc279de279/graphql'
    st.code(endpoint, language="text")
    
    # Create tabs for different sections
    tab1, tab2 = st.tabs(["Query Builder", "Response Viewer"])
    
    with tab1:
        st.markdown("### Query Builder")
        
        # Query input
        query = st.text_area(
            "GraphQL Query",
            height=200,
            help="Enter your GraphQL query here. Example: query { usageByDNOs { items { dno total_usage } } }"
        )
        
        # Variables input (optional)
        variables = st.text_area(
            "Variables (Optional)",
            height=100,
            help="Enter your variables in JSON format. Example: { 'first': 10 }"
        )
        
        # Headers (read-only, showing the required headers)
        st.markdown("### Headers")
        st.markdown("The following headers are automatically included:")
        st.code("""
{
    'Authorization': 'Bearer <token>',
    'Content-Type': 'application/json'
}
        """, language="json")
        
        # Execute button
        if st.button("Execute Query", type="primary"):
            if not query:
                st.error("Please enter a GraphQL query")
                return
            
            try:
                # Get authentication token
                app, scp = get_credential()
                result = app.get_token(scp)
                
                if not result.token:
                    st.error("Could not get access token")
                    return
                
                # Prepare headers
                headers = {
                    'Authorization': f'Bearer {result.token}',
                    'Content-Type': 'application/json'
                }
                
                # Prepare request body
                request_body = {'query': query}
                if variables:
                    try:
                        request_body['variables'] = json.loads(variables)
                    except json.JSONDecodeError:
                        st.error("Invalid JSON in variables")
                        return
                
                # Make the request
                with st.spinner('Executing query...'):
                    response = session.post(
                        endpoint,
                        json=request_body,
                        headers=headers
                    )
                    
                    # Store response in session state
                    st.session_state.last_response = {
                        'status_code': response.status_code,
                        'headers': dict(response.headers),
                        'body': response.json() if response.headers.get('content-type', '').startswith('application/json') else response.text,
                        'request': {
                            'query': query,
                            'variables': variables if variables else None,
                            'headers': headers
                        }
                    }
                    
                    # Trigger rerun to show response in the Response Viewer tab
                    st.experimental_rerun()
            
            except Exception as e:
                st.error(f"Error executing query: {str(e)}")
    
    with tab2:
        st.markdown("### Response Viewer")
        
        if 'last_response' not in st.session_state:
            st.info("Execute a query to see the response here")
            return
        
        response = st.session_state.last_response
        
        # Show response status
        status_color = "green" if response['status_code'] == 200 else "red"
        st.markdown(f"**Status Code:** :{status_color}[{response['status_code']}]")
        
        # Show response headers
        with st.expander("Response Headers"):
            st.json(response['headers'])
        
        # Show request details
        with st.expander("Request Details"):
            st.markdown("**Query:**")
            st.code(response['request']['query'], language="graphql")
            if response['request']['variables']:
                st.markdown("**Variables:**")
                st.code(response['request']['variables'], language="json")
        
        # Show response body
        st.markdown("**Response Body:**")
        if isinstance(response['body'], dict):
            st.json(response['body'])
            
            # Add download options for JSON response
            st.markdown("### Download Options")
            col1, col2 = st.columns(2)
            
            with col1:
                # Download as JSON
                json_str = json.dumps(response['body'], indent=2)
                st.download_button(
                    "Download as JSON",
                    json_str,
                    file_name="graphql_response.json",
                    mime="application/json"
                )
            
            with col2:
                # Try to convert to DataFrame and download as CSV if possible
                try:
                    # Check if the response contains items that can be converted to a DataFrame
                    if 'data' in response['body']:
                        for key, value in response['body']['data'].items():
                            if isinstance(value, dict) and 'items' in value:
                                df = pd.DataFrame(value['items'])
                                csv = df.to_csv(index=False)
                                st.download_button(
                                    "Download as CSV",
                                    csv,
                                    file_name="graphql_response.csv",
                                    mime="text/csv"
                                )
                                break
                except Exception as e:
                    st.warning("CSV download not available for this response format")
        else:
            st.code(response['body'], language="text")
            # Download as text for non-JSON responses
            st.download_button(
                "Download as Text",
                response['body'],
                file_name="graphql_response.txt",
                mime="text/plain"
            )

# Function to authenticate user with Azure AD
def authenticate_with_azure():
    try:
        app, scp = get_credential()
        result = app.get_token(scp)
        if result.token:
            # Get user info from the token
            user_principal_name = get_user_info_from_token(result.token)
            if user_principal_name:
                # Check if user is admin
                if user_principal_name == "mawaz@opendata.energy":
                    st.session_state.authenticated = True
                    st.session_state.user_role = "Admin"
                    st.session_state.user_info = {
                        'userPrincipalName': user_principal_name,
                        'displayName': "Admin User"
                    }
                    return True
                
                # Fetch user role for non-admin users
                role_info = fetch_user_role(user_principal_name)
                if role_info:
                    st.session_state.authenticated = True
                    st.session_state.user_role = role_info['role']
                    st.session_state.user_info = role_info['user_info']
                    return True
                else:
                    st.error("You don't have access to any dashboard. Please contact your administrator.")
                    return False
        return False
    except Exception as e:
        st.error(f"Authentication failed: {str(e)}")
        return False

# Cache the meter building data
@st.cache_data(ttl=300)  # Cache for 5 minutes
def fetch_meter_building_data():
    try:
        app, scp = get_credential()
        result = app.get_token(scp)
        
        if not result.token:
            st.error("Could not get access token")
            return None

        # GraphQL endpoint
        endpoint = 'https://3b1565aabd8147d394c135b1d3761d87.z3b.graphql.fabric.microsoft.com/v1/workspaces/3b1565aa-bd81-47d3-94c1-35b1d3761d87/graphqlapis/2eae604b-a405-49f7-83ab-12dc279de279/graphql'
        
        # GraphQL query
        query = """
        query {
            anonMeterBuildingDatas(first: 10000) {
                items {
                    date
                    readType
                    fuelType
                    meterType
                    profileClass
                    supplierId
                    supplierName
                    isRenewable
                    granularity
                    usage
                    usageRI
                    supplyRE
                    supplyAE
                    lastMeterReadDate
                    actualCO2FootPrintSavedPerMeter
                    marketcarbonFootprint
                    locationcarbonFootprint
                    measurementQuantity
                    createdDate
                    updatedOn
                    readingtypeAE
                    readingtypeRI
                    readingtypeRE
                    eac
                    source_load_time
                    start_date
                    end_date
                    record_status
                    upn_link
                    premise_age
                    premise_year
                    premise_age_confidence
                    premise_use
                    premise_type
                    premise_type_confidence
                    premise_floor_count
                    bungalow
                    flat_conversion
                    height
                    premise_area
                    building_area
                    address_area
                    gross_area
                    basement
                    listed_grade
                    element_count
                    bathroom_count
                    bedroom_count
                    reception_room_count
                    roof_type
                    wall_type
                    wall_type_confidence
                    substructure_type
                    energy_efficiency_rating
                    glazing_type
                    wall_construction_type
                    extension_count
                    habitable_rooms
                    open_fireplaces
                    floor_type
                    main_heating_fuel
                    distance_building
                    site_area
                    site_non_built_area
                    site_building_count
                    distance_water
                    uprn_count
                    uprn_distance
                    map_age
                    map_floors
                    map_use
                    map_simple_use
                }
            }
        }
        """
        
        # Prepare headers
        headers = {
            'Authorization': f'Bearer {result.token}',
            'Content-Type': 'application/json'
        }
        
        # Make the request with retry logic
        data = make_graphql_request(endpoint, query, headers=headers)
        if data is None:
            return None
        
        if 'errors' in data:
            st.error(f"GraphQL errors: {data['errors']}")
            return None
        
        # Extract items
        items = data.get('data', {}).get('anonMeterBuildingDatas', {}).get('items', [])
        
        if not items:
            st.warning("No data returned from the API")
            return None
        
        # Convert to DataFrame
        df = pd.DataFrame(items)
        
        # Convert date columns to datetime
        date_columns = ['date', 'createdDate', 'updatedOn', 'lastMeterReadDate', 'start_date', 'end_date']
        for col in date_columns:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col])
        
        # Convert numeric columns
        numeric_columns = [
            'usage', 'usageRI', 'supplyRE', 'supplyAE', 'actualCO2FootPrintSavedPerMeter',
            'marketcarbonFootprint', 'locationcarbonFootprint', 'premise_age', 'premise_floor_count',
            'height', 'premise_area', 'building_area', 'address_area', 'gross_area',
            'bathroom_count', 'bedroom_count', 'reception_room_count', 'habitable_rooms',
            'open_fireplaces', 'distance_building', 'site_area', 'site_non_built_area',
            'site_building_count', 'distance_water', 'uprn_count', 'uprn_distance'
        ]
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        return df
        
    except Exception as e:
        st.error(f"Error fetching meter building data: {str(e)}")
        return None

# Function to show energy system academic dashboard
def show_energy_system_academic_dashboard():
    st.header("Energy System Academic Dashboard")
    
    # Fetch data with loading spinner
    with st.spinner('Loading data...'):
        df = fetch_meter_building_data()
    
    if df is None:
        st.error("Could not fetch meter building data")
        return
    
    # Create tabs for different views
    tab1, tab2, tab3 = st.tabs(["Raw Data", "Matrix Views", "Time Series"])
    
    with tab1:
        st.subheader("Raw Data View")
        
        # Add filters for the raw data
        col1, col2 = st.columns(2)
        
        with col1:
            # Date range filter
            date_cols = [col for col in df.columns if 'date' in col.lower()]
            if date_cols:
                selected_date_col = st.selectbox(
                    "Select date column for filtering",
                    options=date_cols,
                    key="date_filter"
                )
                
                if selected_date_col in df.columns:
                    min_date = df[selected_date_col].min()
                    max_date = df[selected_date_col].max()
                    date_range = st.date_input(
                        "Select date range",
                        value=(min_date, max_date),
                        key="date_range"
                    )
                    
                    if len(date_range) == 2:
                        mask = (df[selected_date_col].dt.date >= date_range[0]) & (df[selected_date_col].dt.date <= date_range[1])
                        df = df[mask]
        
        with col2:
            # Numeric column filter
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if numeric_cols:
                selected_numeric_col = st.selectbox(
                    "Select numeric column for filtering",
                    options=numeric_cols,
                    key="numeric_filter"
                )
                
                if selected_numeric_col in df.columns:
                    min_val = df[selected_numeric_col].min()
                    max_val = df[selected_numeric_col].max()
                    value_range = st.slider(
                        f"Select range for {selected_numeric_col}",
                        min_value=float(min_val),
                        max_value=float(max_val),
                        value=(float(min_val), float(max_val)),
                        key="numeric_range"
                    )
                    
                    mask = (df[selected_numeric_col] >= value_range[0]) & (df[selected_numeric_col] <= value_range[1])
                    df = df[mask]
        
        # Display the raw data
        st.dataframe(
            df,
            use_container_width=True,
            height=400
        )
        
        # Add download button for the filtered data
        csv = df.to_csv(index=False)
        st.download_button(
            "Download Filtered Data as CSV",
            csv,
            "filtered_data.csv",
            "text/csv",
            key='download-csv'
        )
    
    with tab2:
        st.subheader("Matrix Views")
        
        # Create matrix view options
        col1, col2 = st.columns(2)
        
        with col1:
            # Select columns for matrix view
            available_cols = df.columns.tolist()
            selected_cols = st.multiselect(
                "Select columns for matrix view",
                options=available_cols,
                default=available_cols[:2] if len(available_cols) >= 2 else available_cols,
                key="matrix_cols"
            )
        
        if len(selected_cols) >= 2:
            # Create matrix view
            matrix_tab1, matrix_tab2 = st.tabs(["Pivot Table", "Cross Tabulation"])
            
            with matrix_tab1:
                st.markdown("### Pivot Table View")
                
                # Pivot table configuration
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    index_col = st.selectbox(
                        "Select index column",
                        options=selected_cols,
                        key="pivot_index"
                    )
                
                with col2:
                    columns_col = st.selectbox(
                        "Select columns",
                        options=[col for col in selected_cols if col != index_col],
                        key="pivot_columns"
                    )
                
                with col3:
                    values_col = st.selectbox(
                        "Select values",
                        options=[col for col in selected_cols if col not in [index_col, columns_col]],
                        key="pivot_values"
                    )
                
                if values_col:
                    # Create pivot table
                    pivot_df = pd.pivot_table(
                        df,
                        values=values_col,
                        index=index_col,
                        columns=columns_col,
                        aggfunc='sum',
                        fill_value=0
                    )
                    
                    # Check if the values column is numeric
                    if pd.api.types.is_numeric_dtype(df[values_col]):
                        st.dataframe(
                            pivot_df.style.format("{:,.0f}"),
                            use_container_width=True
                        )
                    else:
                        st.dataframe(
                            pivot_df,
                            use_container_width=True
                        )
            
            with matrix_tab2:
                st.markdown("### Cross Tabulation View")
                
                # Cross tab configuration
                col1, col2 = st.columns(2)
                
                with col1:
                    row_col = st.selectbox(
                        "Select row variable",
                        options=selected_cols,
                        key="crosstab_row"
                    )
                
                with col2:
                    col_col = st.selectbox(
                        "Select column variable",
                        options=[col for col in selected_cols if col != row_col],
                        key="crosstab_col"
                    )
                
                if row_col and col_col:
                    # Create cross tabulation
                    crosstab_df = pd.crosstab(
                        df[row_col],
                        df[col_col],
                        margins=True
                    )
                    
                    # Check if the values are numeric
                    if pd.api.types.is_numeric_dtype(crosstab_df):
                        st.dataframe(
                            crosstab_df.style.format("{:,.0f}"),
                            use_container_width=True
                        )
                    else:
                        st.dataframe(
                            crosstab_df,
                            use_container_width=True
                        )
        else:
            st.warning("Please select at least two columns for matrix view")
    
    with tab3:
        st.subheader("Time Series Analysis")
        
        # Create filters
        col1, col2 = st.columns(2)
        
        with col1:
            # Date column selection
            date_cols = [col for col in df.columns if 'date' in col.lower()]
            selected_date_col = st.selectbox(
                "Select date column",
                options=date_cols,
                key="timeseries_date"
            )
            
            # Usage column selection
            usage_cols = ['usage', 'usageRI', 'supplyRE', 'supplyAE']
            available_usage_cols = [col for col in usage_cols if col in df.columns]
            selected_usage_col = st.selectbox(
                "Select usage metric",
                options=available_usage_cols,
                key="timeseries_usage"
            )
        
        with col2:
            # Categorical filters
            categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
            selected_cat_cols = st.multiselect(
                "Select filters",
                options=categorical_cols,
                key="timeseries_filters"
            )
        
        # Apply filters
        filtered_df = df.copy()
        
        # Apply categorical filters
        for col in selected_cat_cols:
            unique_values = sorted(filtered_df[col].unique())
            selected_values = st.multiselect(
                f"Select {col}",
                options=unique_values,
                default=unique_values[:5] if len(unique_values) > 5 else unique_values,
                key=f"filter_{col}"
            )
            if selected_values:
                filtered_df = filtered_df[filtered_df[col].isin(selected_values)]
        
        if not filtered_df.empty:
            # Group by date and calculate statistics
            time_series = filtered_df.groupby(selected_date_col)[selected_usage_col].agg(['mean', 'sum', 'count']).reset_index()
            time_series.columns = [selected_date_col, 'Average', 'Total', 'Count']
            
            # Create time series plot using Plotly
            from plotly.subplots import make_subplots
            
            # Create figure with secondary y-axis
            fig = make_subplots(specs=[[{"secondary_y": True}]])
            
            # Add average usage line
            fig.add_trace(
                go.Scatter(
                    x=time_series[selected_date_col],
                    y=time_series['Average'],
                    name="Average Usage",
                    line=dict(color='#1f77b4', width=2)
                ),
                secondary_y=False
            )
            
            # Add total usage bars
            fig.add_trace(
                go.Bar(
                    x=time_series[selected_date_col],
                    y=time_series['Total'],
                    name="Total Usage",
                    marker_color='rgba(46, 137, 205, 0.3)'
                ),
                secondary_y=True
            )
            
            # Update layout
            fig.update_layout(
                title=dict(
                    text=f'Time Series Analysis of {selected_usage_col}',
                    x=0.5,
                    y=0.95,
                    xanchor='center',
                    yanchor='top'
                ),
                xaxis_title="Date",
                hovermode='x unified',
                showlegend=True,
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                ),
                height=600,
                template="plotly_white"
            )
            
            # Update y-axes labels
            fig.update_yaxes(title_text="Average Usage (kWh)", secondary_y=False)
            fig.update_yaxes(title_text="Total Usage (kWh)", secondary_y=True)
            
            # Display the plot
            st.plotly_chart(fig, use_container_width=True)
            
            # Display summary statistics
            st.subheader("Summary Statistics")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Total Records", f"{len(filtered_df):,}")
            with col2:
                st.metric("Average Usage", f"{filtered_df[selected_usage_col].mean():,.2f}")
            with col3:
                st.metric("Total Usage", f"{filtered_df[selected_usage_col].sum():,.2f}")
            
            # Display the time series data
            st.subheader("Time Series Data")
            st.dataframe(
                time_series.style.format({
                    'Average': '{:,.2f}',
                    'Total': '{:,.2f}',
                    'Count': '{:,.0f}'
                }),
                use_container_width=True
            )
        else:
            st.warning("No data available with the selected filters")

# Cache the valid consents data
@st.cache_data(ttl=300)  # Cache for 5 minutes
def fetch_valid_consents():
    try:
        app, scp = get_credential()
        result = app.get_token(scp)
        
        if not result.token:
            st.error("Could not get access token")
            return None

        # GraphQL endpoint
        endpoint = 'https://3b1565aabd8147d394c135b1d3761d87.z3b.graphql.fabric.microsoft.com/v1/workspaces/3b1565aa-bd81-47d3-94c1-35b1d3761d87/graphqlapis/2eae604b-a405-49f7-83ab-12dc279de279/graphql'
        
        # GraphQL query
        query = """
        query {
            valid_consents(first: 10000) {
                items {
                    userPrincipalName
                    roleName
                    meterNo
                }
            }
        }
        """
        
        # Prepare headers
        headers = {
            'Authorization': f'Bearer {result.token}',
            'Content-Type': 'application/json'
        }
        
        # Make the request with retry logic
        data = make_graphql_request(endpoint, query, headers=headers)
        if data is None:
            return None
        
        if 'errors' in data:
            st.error(f"GraphQL errors: {data['errors']}")
            return None
        
        # Extract items
        items = data.get('data', {}).get('valid_consents', {}).get('items', [])
        
        if not items:
            st.warning("No data returned from the API")
            return None
        
        # Convert to DataFrame
        df = pd.DataFrame(items)
        return df
        
    except Exception as e:
        st.error(f"Error fetching valid consents data: {str(e)}")
        return None

# Cache the meter building data for landlord
@st.cache_data(ttl=300)  # Cache for 5 minutes
def fetch_landlord_meter_data(meter_numbers):
    try:
        app, scp = get_credential()
        result = app.get_token(scp)
        
        if not result.token:
            st.error("Could not get access token")
            return None

        # GraphQL endpoint
        endpoint = 'https://3b1565aabd8147d394c135b1d3761d87.z3b.graphql.fabric.microsoft.com/v1/workspaces/3b1565aa-bd81-47d3-94c1-35b1d3761d87/graphqlapis/2eae604b-a405-49f7-83ab-12dc279de279/graphql'
        
        # GraphQL query
        query = """
        query {
            meterBuildingDatas(first: 10000) {
                items {
                    date
                    meterNo
                    readType
                    fuelType
                    meterType
                    profileClass
                    supplierId
                    supplierName
                    isRenewable
                    granularity
                    usage
                    usageRI
                    supplyRE
                    supplyAE
                    lastMeterReadDate
                    actualCO2FootPrintSavedPerMeter
                    marketcarbonFootprint
                    locationcarbonFootprint
                    measurementQuantity
                    createdDate
                    updatedOn
                    readingtypeAE
                    readingtypeRI
                    readingtypeRE
                    eac
                    source_load_time
                    start_date
                    end_date
                    record_status
                    upn_link
                    premise_age
                    premise_year
                    premise_age_confidence
                    premise_use
                    premise_type
                    premise_type_confidence
                    premise_floor_count
                    bungalow
                    flat_conversion
                    height
                    premise_area
                    building_area
                    address_area
                    gross_area
                    basement
                    listed_grade
                    element_count
                    bathroom_count
                    bedroom_count
                    reception_room_count
                    roof_type
                    wall_type
                    wall_type_confidence
                    substructure_type
                    energy_efficiency_rating
                    glazing_type
                    wall_construction_type
                    extension_count
                    habitable_rooms
                    open_fireplaces
                    floor_type
                    main_heating_fuel
                    distance_building
                    site_area
                    site_non_built_area
                    site_building_count
                    distance_water
                    uprn_count
                    uprn_distance
                    map_age
                    map_floors
                    map_use
                    map_simple_use
                }
            }
        }
        """
        
        # Prepare headers
        headers = {
            'Authorization': f'Bearer {result.token}',
            'Content-Type': 'application/json'
        }
        
        # Make the request with retry logic
        data = make_graphql_request(endpoint, query, headers=headers)
        if data is None:
            return None
        
        if 'errors' in data:
            st.error(f"GraphQL errors: {data['errors']}")
            return None
        
        # Extract items
        items = data.get('data', {}).get('meterBuildingDatas', {}).get('items', [])
        
        if not items:
            st.warning("No data returned from the API")
            return None
        
        # Convert to DataFrame
        df = pd.DataFrame(items)
        
        # Filter for the landlord's meters
        df = df[df['meterNo'].isin(meter_numbers)]
        
        # Convert date columns to datetime
        date_columns = ['date', 'createdDate', 'updatedOn', 'lastMeterReadDate', 'start_date', 'end_date']
        for col in date_columns:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col])
        
        # Convert numeric columns
        numeric_columns = [
            'usage', 'usageRI', 'supplyRE', 'supplyAE', 'actualCO2FootPrintSavedPerMeter',
            'marketcarbonFootprint', 'locationcarbonFootprint', 'premise_age', 'premise_floor_count',
            'height', 'premise_area', 'building_area', 'address_area', 'gross_area',
            'bathroom_count', 'bedroom_count', 'reception_room_count', 'habitable_rooms',
            'open_fireplaces', 'distance_building', 'site_area', 'site_non_built_area',
            'site_building_count', 'distance_water', 'uprn_count', 'uprn_distance'
        ]
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        return df
        
    except Exception as e:
        st.error(f"Error fetching meter building data: {str(e)}")
        return None

# Function to anonymize meter numbers
def anonymize_meter_numbers(meter_numbers):
    """Create a consistent mapping of real meter numbers to anonymized IDs."""
    # Sort meter numbers to ensure consistent mapping
    sorted_meters = sorted(meter_numbers)
    # Create mapping with format "METER_XXX" where XXX is a zero-padded number
    return {meter: f"METER_{i+1:03d}" for i, meter in enumerate(sorted_meters)}

# Function to show landlord dashboard
def show_landlord_dashboard():
    st.header("Landlord Dashboard")
    
    # Get the logged-in user's UPN
    user_principal_name = st.session_state.user_info.get('userPrincipalName')
    if not user_principal_name:
        st.error("Could not get user information")
        return
    
    # Fetch valid consents with loading spinner
    with st.spinner('Loading consent data...'):
        consents_df = fetch_valid_consents()
    
    if consents_df is None:
        st.error("Could not fetch consent data")
        return
    
    # Filter consents for the logged-in user
    user_consents = consents_df[consents_df['userPrincipalName'] == user_principal_name]
    
    if user_consents.empty:
        st.warning("No meters found for your account")
        return
    
    # Get list of meter numbers and create anonymization mapping
    meter_numbers = user_consents['meterNo'].tolist()
    meter_mapping = anonymize_meter_numbers(meter_numbers)
    
    # Fetch meter building data with loading spinner
    with st.spinner('Loading meter data...'):
        df = fetch_landlord_meter_data(meter_numbers)
    
    if df is None:
        st.error("Could not fetch meter building data")
        return
    
    # Create anonymized meter number column
    df['anonymized_meter'] = df['meterNo'].map(meter_mapping)
    
    # Calculate carbon footprint (CO2E in metric tonnes)
    df['carbon_footprint'] = df['usage'] * 0.000207
    
    # Create tabs for different views
    tab1, tab2, tab3 = st.tabs(["Portfolio", "Matrix Views", "Time Series"])
    
    with tab1:
        st.subheader("Portfolio Overview")
        
        # Calculate usage and carbon footprint aggregation by meter
        usage_by_meter = df.groupby('anonymized_meter').agg({
            'usage': ['sum', 'mean', 'count'],
            'carbon_footprint': ['sum', 'mean']
        }).reset_index()
        
        # Flatten column names
        usage_by_meter.columns = ['Meter ID', 'Total Usage', 'Average Usage', 'Reading Count', 
                                'Total CO2E', 'Average CO2E']
        
        # Create two columns for layout
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Create bar chart for total usage and carbon footprint by meter
            fig = go.Figure()
            
            # Add usage bars
            fig.add_trace(
                go.Bar(
                    x=usage_by_meter['Meter ID'],
                    y=usage_by_meter['Total Usage'],
                    name='Total Usage',
                    marker_color='rgba(46, 137, 205, 0.7)',
                    yaxis='y1'
                )
            )
            
            # Add carbon footprint line
            fig.add_trace(
                go.Scatter(
                    x=usage_by_meter['Meter ID'],
                    y=usage_by_meter['Total CO2E'],
                    name='Total CO2E',
                    line=dict(color='#FF5733', width=2),
                    yaxis='y2'
                )
            )
            
            # Update layout with dual y-axes
            fig.update_layout(
                title=dict(
                    text='Total Usage and Carbon Footprint by Meter',
                    x=0.5,
                    y=0.95,
                    xanchor='center',
                    yanchor='top'
                ),
                xaxis_title="Meter ID",
                yaxis=dict(
                    title="Total Usage (kWh)",
                    side="left"
                ),
                yaxis2=dict(
                    title="Total CO2E (metric tonnes)",
                    side="right",
                    overlaying="y"
                ),
                showlegend=True,
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                ),
                height=400,
                template="plotly_white"
            )
            
            # Display the plot
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Display summary metrics
            st.metric("Total Meters", len(usage_by_meter))
            st.metric("Total Portfolio Usage", f"{usage_by_meter['Total Usage'].sum():,.2f}")
            st.metric("Total Carbon Footprint", f"{usage_by_meter['Total CO2E'].sum():,.2f} tonnes CO2E")
            st.metric("Average Usage per Meter", f"{usage_by_meter['Average Usage'].mean():,.2f}")
            st.metric("Average CO2E per Meter", f"{usage_by_meter['Average CO2E'].mean():,.2f} tonnes CO2E")
            
            # Display top 5 meters by usage
            st.subheader("Top 5 Meters by Usage")
            top_meters = usage_by_meter.nlargest(5, 'Total Usage')
            for _, row in top_meters.iterrows():
                st.markdown(f"**{row['Meter ID']}:**")
                st.markdown(f"- Usage: {row['Total Usage']:,.2f}")
                st.markdown(f"- CO2E: {row['Total CO2E']:,.2f} tonnes")
        
        # Add filters for the detailed data
        st.subheader("Detailed Data")
        col1, col2 = st.columns(2)
        
        with col1:
            # Date range filter
            date_cols = [col for col in df.columns if 'date' in col.lower()]
            if date_cols:
                selected_date_col = st.selectbox(
                    "Select date column for filtering",
                    options=date_cols,
                    key="landlord_date_filter"
                )
                
                if selected_date_col in df.columns:
                    min_date = df[selected_date_col].min()
                    max_date = df[selected_date_col].max()
                    date_range = st.date_input(
                        "Select date range",
                        value=(min_date, max_date),
                        key="landlord_date_range"
                    )
                    
                    if len(date_range) == 2:
                        mask = (df[selected_date_col].dt.date >= date_range[0]) & (df[selected_date_col].dt.date <= date_range[1])
                        df = df[mask]
        
        with col2:
            # Numeric column filter
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if numeric_cols:
                selected_numeric_col = st.selectbox(
                    "Select numeric column for filtering",
                    options=numeric_cols,
                    key="landlord_numeric_filter"
                )
                
                if selected_numeric_col in df.columns:
                    min_val = df[selected_numeric_col].min()
                    max_val = df[selected_numeric_col].max()
                    value_range = st.slider(
                        f"Select range for {selected_numeric_col}",
                        min_value=float(min_val),
                        max_value=float(max_val),
                        value=(float(min_val), float(max_val)),
                        key="landlord_numeric_range"
                    )
                    
                    mask = (df[selected_numeric_col] >= value_range[0]) & (df[selected_numeric_col] <= value_range[1])
                    df = df[mask]
        
        # Display the raw data with anonymized meter numbers
        display_df = df.copy()
        display_df['meterNo'] = display_df['anonymized_meter']
        display_df = display_df.drop('anonymized_meter', axis=1)
        
        st.dataframe(
            display_df,
            use_container_width=True,
            height=400
        )
        
        # Add download button for the filtered data
        csv = display_df.to_csv(index=False)
        st.download_button(
            "Download Filtered Data as CSV",
            csv,
            "landlord_filtered_data.csv",
            "text/csv",
            key='landlord-download-csv'
        )
    
    with tab2:
        st.subheader("Matrix Views")
        
        # Create matrix view options
        col1, col2 = st.columns(2)
        
        with col1:
            # Select columns for matrix view
            available_cols = df.columns.tolist()
            # Replace meterNo with anonymized_meter in available columns
            available_cols = ['anonymized_meter' if col == 'meterNo' else col for col in available_cols]
            selected_cols = st.multiselect(
                "Select columns for matrix view",
                options=available_cols,
                default=['anonymized_meter', 'usage', 'carbon_footprint'],
                key="landlord_matrix_cols"
            )
        
        if len(selected_cols) >= 2:
            # Create matrix view
            matrix_tab1, matrix_tab2 = st.tabs(["Pivot Table", "Cross Tabulation"])
            
            with matrix_tab1:
                st.markdown("### Pivot Table View")
                
                # Pivot table configuration
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    index_col = st.selectbox(
                        "Select index column",
                        options=selected_cols,
                        key="landlord_pivot_index"
                    )
                
                with col2:
                    columns_col = st.selectbox(
                        "Select columns",
                        options=[col for col in selected_cols if col != index_col],
                        key="landlord_pivot_columns"
                    )
                
                with col3:
                    values_col = st.selectbox(
                        "Select values",
                        options=[col for col in selected_cols if col not in [index_col, columns_col]],
                        key="landlord_pivot_values"
                    )
                
                if values_col:
                    # Create pivot table
                    pivot_df = pd.pivot_table(
                        df,
                        values=values_col,
                        index=index_col,
                        columns=columns_col,
                        aggfunc='sum',
                        fill_value=0
                    )
                    
                    # Check if the values column is numeric
                    if pd.api.types.is_numeric_dtype(df[values_col]):
                        st.dataframe(
                            pivot_df.style.format("{:,.2f}"),
                            use_container_width=True
                        )
                    else:
                        st.dataframe(
                            pivot_df,
                            use_container_width=True
                        )
            
            with matrix_tab2:
                st.markdown("### Cross Tabulation View")
                
                # Cross tab configuration
                col1, col2 = st.columns(2)
                
                with col1:
                    row_col = st.selectbox(
                        "Select row variable",
                        options=selected_cols,
                        key="landlord_crosstab_row"
                    )
                
                with col2:
                    col_col = st.selectbox(
                        "Select column variable",
                        options=[col for col in selected_cols if col != row_col],
                        key="landlord_crosstab_col"
                    )
                
                if row_col and col_col:
                    # Create cross tabulation
                    crosstab_df = pd.crosstab(
                        df[row_col],
                        df[col_col],
                        margins=True
                    )
                    
                    # Check if the values are numeric
                    if pd.api.types.is_numeric_dtype(crosstab_df):
                        st.dataframe(
                            crosstab_df.style.format("{:,.2f}"),
                            use_container_width=True
                        )
                    else:
                        st.dataframe(
                            crosstab_df,
                            use_container_width=True
                        )
        else:
            st.warning("Please select at least two columns for matrix view")
    
    with tab3:
        st.subheader("Time Series Analysis")
        
        # Create filters
        col1, col2 = st.columns(2)
        
        with col1:
            # Date column selection
            date_cols = [col for col in df.columns if 'date' in col.lower()]
            selected_date_col = st.selectbox(
                "Select date column",
                options=date_cols,
                key="landlord_timeseries_date"
            )
            
            # Usage and carbon footprint selection
            metric_cols = ['usage', 'carbon_footprint']
            selected_metric = st.selectbox(
                "Select metric",
                options=metric_cols,
                key="landlord_timeseries_metric"
            )
        
        with col2:
            # Categorical filters
            categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
            # Replace meterNo with anonymized_meter in categorical columns
            categorical_cols = ['anonymized_meter' if col == 'meterNo' else col for col in categorical_cols]
            selected_cat_cols = st.multiselect(
                "Select filters",
                options=categorical_cols,
                key="landlord_timeseries_filters"
            )
        
        # Apply filters
        filtered_df = df.copy()
        
        # Apply categorical filters
        for col in selected_cat_cols:
            unique_values = sorted(filtered_df[col].unique())
            selected_values = st.multiselect(
                f"Select {col}",
                options=unique_values,
                default=unique_values[:5] if len(unique_values) > 5 else unique_values,
                key=f"landlord_filter_{col}"
            )
            if selected_values:
                filtered_df = filtered_df[filtered_df[col].isin(selected_values)]
        
        if not filtered_df.empty:
            # Group by date and calculate statistics
            time_series = filtered_df.groupby(selected_date_col)[selected_metric].agg(['mean', 'sum', 'count']).reset_index()
            time_series.columns = [selected_date_col, 'Average', 'Total', 'Count']
            
            # Create time series plot using Plotly
            from plotly.subplots import make_subplots
            
            # Create figure with secondary y-axis
            fig = make_subplots(specs=[[{"secondary_y": True}]])
            
            # Add average line
            fig.add_trace(
                go.Scatter(
                    x=time_series[selected_date_col],
                    y=time_series['Average'],
                    name=f"Average {selected_metric}",
                    line=dict(color='#1f77b4', width=2)
                ),
                secondary_y=False
            )
            
            # Add total bars
            fig.add_trace(
                go.Bar(
                    x=time_series[selected_date_col],
                    y=time_series['Total'],
                    name=f"Total {selected_metric}",
                    marker_color='rgba(46, 137, 205, 0.3)'
                ),
                secondary_y=True
            )
            
            # Update layout
            metric_title = "Usage (kWh)" if selected_metric == "usage" else "CO2E (metric tonnes)"
            fig.update_layout(
                title=dict(
                    text=f'Time Series Analysis of {metric_title}',
                    x=0.5,
                    y=0.95,
                    xanchor='center',
                    yanchor='top'
                ),
                xaxis_title="Date",
                hovermode='x unified',
                showlegend=True,
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                ),
                height=600,
                template="plotly_white"
            )
            
            # Update y-axes labels
            fig.update_yaxes(title_text=f"Average {metric_title}", secondary_y=False)
            fig.update_yaxes(title_text=f"Total {metric_title}", secondary_y=True)
            
            # Display the plot
            st.plotly_chart(fig, use_container_width=True)
            
            # Display summary statistics
            st.subheader("Summary Statistics")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Total Records", f"{len(filtered_df):,}")
            with col2:
                st.metric(f"Average {metric_title}", f"{filtered_df[selected_metric].mean():,.2f}")
            with col3:
                st.metric(f"Total {metric_title}", f"{filtered_df[selected_metric].sum():,.2f}")
            
            # Display the time series data
            st.subheader("Time Series Data")
            st.dataframe(
                time_series.style.format({
                    'Average': '{:,.2f}',
                    'Total': '{:,.2f}',
                    'Count': '{:,.0f}'
                }),
                use_container_width=True
            )
        else:
            st.warning("No data available with the selected filters")

# Function to show role-specific dashboard
def show_role_dashboard():
    role = st.session_state.user_role
    user_principal_name = st.session_state.user_info.get('userPrincipalName')
    
    # Check if user is admin
    is_admin = user_principal_name == "mawaz@opendata.energy"
    
    if is_admin:
        st.header("Admin Dashboard")
        
        # Add dashboard selector in sidebar
        with st.sidebar:
            st.subheader("Dashboard Selection")
            selected_dashboard = st.selectbox(
                "Select Dashboard to View",
                ["DNO Analyst", "Energy System Academic", "Energy App Developer", "Landlord"],
                key="admin_dashboard_selector"
            )
            
            # Map selection to role
            role_mapping = {
                "DNO Analyst": "DNOAnalyst",
                "Energy System Academic": "EnergySystemAcademic",
                "Energy App Developer": "EnergyAppDev",
                "Landlord": "Landlord"
            }
            role = role_mapping[selected_dashboard]
    
    if role == "DNOAnalyst":
        st.header("DNO Analyst Dashboard")
        
        # Create tabs for different sections
        tab1, tab2, tab3 = st.tabs(["Usage", "Property Type", "Service Request Type"])
        
        with tab1:
            show_usage_by_dno()
            
        with tab2:
            show_premise_type()
            
        with tab3:
            show_service_request_codes()
        
    elif role == "EnergySystemAcademic":
        st.header("Energy System Academic Dashboard")
        show_energy_system_academic_dashboard()
        
    elif role == "EnergyAppDev":
        st.header("Energy App Developer Dashboard")
        show_api_testing_interface()
        
    elif role == "Landlord":
        show_landlord_dashboard()
        
    else:
        st.error("Unknown role. Please contact your administrator.")

# Function to show registration form
def show_registration_form():
    st.subheader("Register")
    
    with st.form("registration_form"):
        # Form fields
        email = st.text_input("Email")
        first_name = st.text_input("First Name")
        last_name = st.text_input("Last Name")
        job_title = st.text_input("Job Title")
        phone = st.text_input("Phone Number")
        
        # Role selection
        role = st.selectbox(
            "Select Role",
            ["DNOAnalyst", "EnergySystemAcademic", "EnergyAppDev", "Landlord"]
        )
        
        # Submit button
        submit_button = st.form_submit_button("Submit Registration")
        
        if submit_button:
            if not all([email, first_name, last_name, job_title, phone, role]):
                st.error("Please fill in all fields")
                return
            
            try:
                # Get authentication token
                app, scp = get_credential()
                result = app.get_token(scp)
                
                if not result.token:
                    st.error("Could not get access token")
                    return
                
                # GraphQL endpoint
                endpoint = 'https://3b1565aabd8147d394c135b1d3761d87.z3b.graphql.fabric.microsoft.com/v1/workspaces/3b1565aa-bd81-47d3-94c1-35b1d3761d87/graphqlapis/2eae604b-a405-49f7-83ab-12dc279de279/graphql'
                
                # GraphQL mutation for user registration
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
                
                # Prepare variables
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
                
                # Prepare headers
                headers = {
                    'Authorization': f'Bearer {result.token}',
                    'Content-Type': 'application/json'
                }
                
                # Make the request
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
                            # Send email notification
                            send_registration_email(email, first_name, last_name, role)
                            # Reset form state
                            st.session_state.show_register = False
                            st.experimental_rerun()
                    else:
                        st.error(f"Registration failed with status code: {response.status_code}")
            
            except Exception as e:
                st.error(f"Error during registration: {str(e)}")
    
    # Back to login button
    if st.button("Back to Login"):
        st.session_state.show_register = False
        st.experimental_rerun()

# Function to send registration email
def send_registration_email(email, first_name, last_name, role):
    try:
        # Email configuration
        sender_email = os.getenv("EMAIL_USER")
        sender_password = os.getenv("EMAIL_PASSWORD")
        admin_email = os.getenv("ADMIN_EMAIL")
        
        if not all([sender_email, sender_password, admin_email]):
            st.warning("Email configuration not complete. Skipping email notification.")
            return
        
        # Create message
        msg = MIMEMultipart()
        msg['From'] = sender_email
        msg['To'] = admin_email
        msg['Subject'] = "New User Registration - SMEDR Demo"
        
        # Email body
        body = f"""
        A new user has registered for the SMEDR Demo:
        
        Name: {first_name} {last_name}
        Email: {email}
        Role: {role}
        
        Please review and approve this registration.
        """
        
        msg.attach(MIMEText(body, 'plain'))
        
        # Send email
        with smtplib.SMTP('smtp.office365.com', 587) as server:
            server.starttls()
            server.login(sender_email, sender_password)
            server.send_message(msg)
            
    except Exception as e:
        st.warning(f"Could not send registration email: {str(e)}")

# Function to show login
def show_login():
    st.subheader("Login")
    
    with st.form("login_form"):
        col1, col2 = st.columns(2)
        with col1:
            login_button = st.form_submit_button("Login with Microsoft")
        with col2:
            register_button = st.form_submit_button("Register")
        
        if login_button:
            if authenticate_with_azure():
                st.experimental_rerun()
        
        if register_button:
            st.session_state.show_register = True
            st.experimental_rerun()

# Main content
def main():
    # Check authentication
    if not st.session_state.authenticated:
        if st.session_state.show_register:
            show_registration_form()
        else:
            show_login()
        return
    
    # Add logout button in the sidebar
    with st.sidebar:
        if st.session_state.user_info:
            st.write(f"Welcome, {st.session_state.user_info.get('displayName', 'User')}")
            if st.session_state.user_role == "Admin":
                st.write("Role: Admin (Full Access)")
            else:
                st.write(f"Role: {st.session_state.user_role}")
        if st.button("Logout"):
            clear_credential_cache()
            st.experimental_rerun()
    
    # Show role-specific dashboard
    show_role_dashboard()

if __name__ == "__main__":
    main() 