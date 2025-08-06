import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import requests
from azure.identity import InteractiveBrowserCredential
import json
from datetime import datetime
import numpy as np

# Set page config
st.set_page_config(
    page_title="EPC Statistics",
    page_icon="📊",
    layout="wide"
)

# Title
st.title("EPC Statistics Dashboard")

# Initialize the credential once
@st.cache_resource
def get_credential():
    app = InteractiveBrowserCredential()
    scp = 'https://analysis.windows.net/powerbi/api/user_impersonation'
    return app, scp

# Function to fetch data from Microsoft GraphQL API
def fetch_data(table_name="epcNonDomesticScotlands"):
    try:
        # Get the cached credential
        app, scp = get_credential()
        
        # Get token using the cached credential
        result = app.get_token(scp)

        if not result.token:
            st.error("Could not get access token")
            return None

        # Prepare headers
        headers = {
            'Authorization': f'Bearer {result.token}',
            'Content-Type': 'application/json'
        }

        # GraphQL endpoint
        endpoint = 'https://d1472da683b0472c908ef45f31025b4c.zd1.graphql.fabric.microsoft.com/v1/workspaces/d1472da6-83b0-472c-908e-f45f31025b4c/graphqlapis/8d0c3265-e4de-49ec-bd12-615f3551ef6f/graphql'
        
        # Define queries for different tables with pagination
        queries = {
            "epcNonDomesticScotlands": """
                query($first: Int!) {
                    epcNonDomesticScotlands(first: $first) {
                        items {
                            CURRENT_ENERGY_PERFORMANCE_BAND
                            CURRENT_ENERGY_PERFORMANCE_RATING
                            CREATED_AT
                            LODGEMENT_DATE
                            PRIMARY_ENERGY_VALUE
                            BUILDING_EMISSIONS
                            FLOOR_AREA
                            PROPERTY_TYPE
                            POST_TOWN
                            NEW_BUILD_ENERGY_PERFORMANCE_BAND
                            NEW_BUILD_ENERGY_PERFORMANCE_RATING
                        }
                    }
                }
            """,
            "epcDomesticEngWales": """
                query($first: Int!) {
                    epcDomesticEngWales(first: $first) {
                        items {
                            property_type
                            transaction_type
                            local_authority
                            constituency
                            county
                            energy_tariff
                            main_fuel
                            floor_level
                            tenure
                            report_type
                            mains_gas_flag
                            lodgement_date
                            energy_consumption_current
                            co2_emissions_current
                        }
                    }
                }
            """,
            "epcDomesticScotlands": """
                query($first: Int!) {
                    epcDomesticScotlands(first: $first) {
                        items {
                            lodgement_date
                            energy_consumption_current
                            co2_emissions_current
                            heating_cost_current
                            energy_consumption_potential
                            co2_emissions_potential
                            heating_cost_potential
                            property_type
                            main_heating_category
                            transaction_type
                            local_authority_label
                            low_energy_lighting
                            construction_age_band
                            current_energy_rating
                            potential_energy_rating
                            total_floor_area
                        }
                    }
                }
            """,
            "epcNonDomesticEngWales": """
                query($first: Int!) {
                    epcNonDomesticEngWales(first: $first) {
                        items {
                            lodgement_date
                            primary_energy_value
                            building_emissions
                            floor_area
                        }
                    }
                }
            """,
            "scotDomChangesOverTimes": """
                query {
                    scotDomChangesOverTimes {
                        items {
                            lodgement_year
                            median_co2_emissions_current
                            median_heating_cost_current
                            median_energy_consumption_potential
                            median_co2_emissions_potential
                            median_heating_cost_potential
                            lodgement_month
                            total_records
                            avg_energy_consumption_current
                            avg_co2_emissions_current
                            avg_heating_cost_current
                            avg_energy_consumption_potential
                            avg_co2_emissions_potential
                            avg_heating_cost_potential
                            median_energy_consumption_current
                        }
                    }
                }
            """
        }
        
        # Get the appropriate query
        query = queries.get(table_name)
        if query is None:
            st.error(f"Unknown table name: {table_name}")
            return None

        # For the aggregated table, we don't need pagination
        if table_name == "scotDomChangesOverTimes":
            response = requests.post(endpoint, json={'query': query}, headers=headers)
            response.raise_for_status()
            data = response.json()
            
            if 'errors' in data:
                error_messages = [error.get('message', 'Unknown error') for error in data['errors']]
                st.error(f"GraphQL errors: {', '.join(error_messages)}")
                return None
            
            if 'data' not in data or table_name not in data['data']:
                st.error(f"Unexpected response structure: {data}")
                return None
            
            items = data['data'][table_name]['items']
            df = pd.DataFrame(items)
            
            # Convert numeric columns
            numeric_columns = [col for col in df.columns if col not in ['lodgement_year', 'lodgement_month']]
            for col in numeric_columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            return df

        # For other tables, use pagination
        all_items = []
        batch_size = 1000  # Number of records to fetch per request
        has_more = True
        
        # Progress bar for data fetching
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Fetch all pages
        while has_more:
            try:
                # Prepare variables for the query
                variables = {
                    "first": batch_size
                }
                
                # Issue GraphQL request
                response = requests.post(endpoint, json={'query': query, 'variables': variables}, headers=headers)
                response.raise_for_status()
                data = response.json()
                
                # Check for GraphQL errors
                if 'errors' in data:
                    error_messages = [error.get('message', 'Unknown error') for error in data['errors']]
                    st.error(f"GraphQL errors: {', '.join(error_messages)}")
                    return None
                
                # Verify the response structure
                if 'data' not in data or table_name not in data['data']:
                    st.error(f"Unexpected response structure: {data}")
                    return None
                
                # Extract items
                table_data = data['data'][table_name]
                if 'items' not in table_data:
                    st.error(f"Missing required fields in response: {table_data}")
                    return None
                
                items = table_data['items']
                
                # Check if we got any items
                if not items:
                    has_more = False
                else:
                    # Add items to our collection
                    all_items.extend(items)
                    
                    # Update progress
                    status_text.text(f"Fetched {len(all_items)} records...")
                    progress_bar.progress(min(1.0, len(all_items) / 100000))  # Assuming max 100k records for progress bar
                
            except requests.exceptions.RequestException as e:
                st.error(f"Network error during data fetch: {str(e)}")
                return None
            except Exception as e:
                st.error(f"Error processing data: {str(e)}")
                return None
        
        # Clear progress indicators
        progress_bar.empty()
        status_text.empty()
        
        if not all_items:
            st.warning(f"No data found for table: {table_name}")
            return None
        
        # Convert the response to a DataFrame
        df = pd.DataFrame(all_items)
        
        # Convert date and numeric columns if they exist
        if 'LODGEMENT_DATE' in df.columns:
            df['LODGEMENT_DATE'] = pd.to_datetime(df['LODGEMENT_DATE'])
            df['lodgement_year'] = df['LODGEMENT_DATE'].dt.year
            df['lodgement_month'] = df['LODGEMENT_DATE'].dt.month
        elif 'lodgement_date' in df.columns:
            df['lodgement_date'] = pd.to_datetime(df['lodgement_date'])
            df['lodgement_year'] = df['lodgement_date'].dt.year
            df['lodgement_month'] = df['lodgement_date'].dt.month
        
        # Convert numeric columns if they exist
        numeric_columns = ['PRIMARY_ENERGY_VALUE', 'BUILDING_EMISSIONS', 'FLOOR_AREA', 
                         'CURRENT_ENERGY_PERFORMANCE_RATING', 'NEW_BUILD_ENERGY_PERFORMANCE_RATING',
                         'energy_consumption_current', 'co2_emissions_current',
                         'heating_cost_current', 'energy_consumption_potential',
                         'co2_emissions_potential', 'heating_cost_potential',
                         'total_floor_area', 'primary_energy_value', 'building_emissions',
                         'floor_area']
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Display total records fetched
        st.write(f"Total records fetched: {len(df)}")
        
        return df
        
    except Exception as e:
        st.error(f"Error fetching data: {str(e)}")
        return None

# Function to create categorical analysis visualization
def create_categorical_analysis_visualization(df):
    if df is not None:
        categorical_columns = ['property_type', 'transaction_type', 'local_authority', 
                             'constituency', 'county', 'energy_tariff', 'main_fuel', 
                             'floor_level', 'tenure', 'report_type', 'mains_gas_flag']
        
        for col_name in categorical_columns:
            if col_name in df.columns:  # Only process columns that exist in the DataFrame
                st.subheader(f"Analysis for {col_name.replace('_', ' ').title()}")
                
                # Calculate distinct count
                distinct_count = df[col_name].nunique()
                st.write(f"Distinct count: {distinct_count}")
                
                # Group and count
                grouped_df = df.groupby(col_name).size().reset_index(name='count')
                grouped_df = grouped_df.sort_values('count', ascending=False)
                
                # Create visualization
                plt.figure(figsize=(12, 6))
                bars = plt.bar(grouped_df[col_name], grouped_df['count'])
                
                # Add value labels on top of bars
                for bar in bars:
                    height = bar.get_height()
                    plt.text(bar.get_x() + bar.get_width()/2., height,
                            f'{int(height)}',
                            ha='center', va='bottom')
                
                plt.title(f'Distribution of {col_name.replace("_", " ").title()}')
                plt.xlabel(col_name.replace('_', ' ').title())
                plt.ylabel('Count')
                plt.xticks(rotation=45, ha='right')
                plt.tight_layout()
                
                # Display the plot
                st.pyplot(plt)
                
                # Display the raw data
                st.dataframe(grouped_df)
                
                # Add a separator between different categories
                st.markdown("---")

# Function to create energy performance bands visualization
def create_energy_bands_visualization(df):
    if df is not None:
        # Group and count the data
        band_counts = df.groupby("CURRENT_ENERGY_PERFORMANCE_BAND").size().reset_index(name="count")
        band_counts = band_counts.sort_values("count", ascending=False)
        
        # Create the plot
        plt.figure(figsize=(12, 6))
        sns.barplot(data=band_counts, x="CURRENT_ENERGY_PERFORMANCE_BAND", y="count")
        plt.title("Distribution of Energy Performance Bands")
        plt.xlabel("Energy Performance Band")
        plt.ylabel("Count")
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        # Display the plot in Streamlit
        st.pyplot(plt)
        
        # Display the raw data
        st.subheader("Raw Data")
        st.dataframe(band_counts)

# Function to create changes over time visualization
def create_changes_over_time_visualization(df):
    if df is not None:
        # Calculate monthly statistics
        changes_over_time = df.groupby(['lodgement_year', 'lodgement_month']).agg({
            'CURRENT_ENERGY_PERFORMANCE_BAND': 'count',
            'PRIMARY_ENERGY_VALUE': ['mean', 'median'],
            'BUILDING_EMISSIONS': ['mean', 'median'],
            'FLOOR_AREA': ['mean', 'median']
        }).reset_index()
        
        # Flatten column names
        changes_over_time.columns = ['lodgement_year', 'lodgement_month', 
                                   'total_certificates', 
                                   'avg_primary_energy_value', 'median_primary_energy_value',
                                   'avg_building_emissions', 'median_building_emissions',
                                   'avg_floor_area', 'median_floor_area']
        
        # Create date column for x-axis
        changes_over_time['date'] = pd.to_datetime({
            'year': changes_over_time['lodgement_year'],
            'month': changes_over_time['lodgement_month'],
            'day': 1
        })
        
        # Plot total certificates over time
        plt.figure(figsize=(15, 10))
        
        # Plot 1: Total Certificates
        plt.subplot(2, 2, 1)
        plt.plot(changes_over_time['date'], changes_over_time['total_certificates'])
        plt.title('Total Certificates Over Time')
        plt.xlabel('Date')
        plt.ylabel('Number of Certificates')
        plt.xticks(rotation=45)
        
        # Plot 2: Primary Energy Value
        plt.subplot(2, 2, 2)
        plt.plot(changes_over_time['date'], changes_over_time['avg_primary_energy_value'], label='Mean')
        plt.plot(changes_over_time['date'], changes_over_time['median_primary_energy_value'], label='Median')
        plt.title('Primary Energy Value Over Time')
        plt.xlabel('Date')
        plt.ylabel('Primary Energy Value')
        plt.legend()
        plt.xticks(rotation=45)
        
        # Plot 3: Building Emissions
        plt.subplot(2, 2, 3)
        plt.plot(changes_over_time['date'], changes_over_time['avg_building_emissions'], label='Mean')
        plt.plot(changes_over_time['date'], changes_over_time['median_building_emissions'], label='Median')
        plt.title('Building Emissions Over Time')
        plt.xlabel('Date')
        plt.ylabel('Building Emissions')
        plt.legend()
        plt.xticks(rotation=45)
        
        # Plot 4: Floor Area
        plt.subplot(2, 2, 4)
        plt.plot(changes_over_time['date'], changes_over_time['avg_floor_area'], label='Mean')
        plt.plot(changes_over_time['date'], changes_over_time['median_floor_area'], label='Median')
        plt.title('Floor Area Over Time')
        plt.xlabel('Date')
        plt.ylabel('Floor Area')
        plt.legend()
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        st.pyplot(plt)
        
        # Display the raw data
        st.subheader("Raw Data")
        st.dataframe(changes_over_time)

# Function to create average emissions visualization
def create_average_emissions_visualization(df):
    if df is not None:
        # Calculate averages
        avg_emissions = pd.DataFrame({
            'Metric': ['Building Emissions', 'Primary Energy Value'],
            'Value': [
                df['BUILDING_EMISSIONS'].mean(),
                df['PRIMARY_ENERGY_VALUE'].mean()
            ]
        })
        
        # Create visualization
        plt.figure(figsize=(10, 6))
        bars = plt.bar(avg_emissions['Metric'], avg_emissions['Value'])
        
        # Add value labels on top of bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2f}',
                    ha='center', va='bottom')
        
        plt.title('Average Building Emissions and Primary Energy Value')
        plt.ylabel('Value')
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        # Display the plot
        st.pyplot(plt)
        
        # Display the raw data
        st.subheader("Raw Data")
        st.dataframe(avg_emissions)

# Function to create yearly trends visualization
def create_yearly_trends_visualization(df):
    if df is not None:
        # Calculate yearly averages
        yearly_trends = df.groupby('lodgement_year')['CURRENT_ENERGY_PERFORMANCE_RATING'].mean().reset_index()
        
        # Create visualization
        plt.figure(figsize=(12, 6))
        plt.plot(yearly_trends['lodgement_year'], yearly_trends['CURRENT_ENERGY_PERFORMANCE_RATING'], 
                marker='o', linestyle='-', linewidth=2)
        
        # Add value labels
        for x, y in zip(yearly_trends['lodgement_year'], yearly_trends['CURRENT_ENERGY_PERFORMANCE_RATING']):
            plt.text(x, y, f'{y:.2f}', ha='center', va='bottom')
        
        plt.title('Yearly Energy Performance Trends')
        plt.xlabel('Year')
        plt.ylabel('Average Energy Performance Rating')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        
        # Display the plot
        st.pyplot(plt)
        
        # Display the raw data
        st.subheader("Raw Data")
        st.dataframe(yearly_trends)

# Function to create average floor area by property type visualization
def create_floor_area_visualization(df):
    if df is not None:
        # Calculate average floor area by property type
        avg_floor_area = df.groupby('PROPERTY_TYPE')['FLOOR_AREA'].mean().reset_index()
        avg_floor_area = avg_floor_area.sort_values('FLOOR_AREA', ascending=False)
        
        # Create visualization
        plt.figure(figsize=(12, 6))
        bars = plt.bar(avg_floor_area['PROPERTY_TYPE'], avg_floor_area['FLOOR_AREA'])
        
        # Add value labels on top of bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2f}',
                    ha='center', va='bottom')
        
        plt.title('Average Floor Area by Property Type')
        plt.xlabel('Property Type')
        plt.ylabel('Average Floor Area')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        # Display the plot
        st.pyplot(plt)
        
        # Display the raw data
        st.subheader("Raw Data")
        st.dataframe(avg_floor_area)

# Function to create top post towns visualization
def create_top_post_towns_visualization(df):
    if df is not None:
        # Calculate top 10 post towns
        top_post_towns = df.groupby('POST_TOWN').size().reset_index(name='count')
        top_post_towns = top_post_towns.sort_values('count', ascending=False).head(10)
        
        # Create visualization
        plt.figure(figsize=(12, 6))
        bars = plt.bar(top_post_towns['POST_TOWN'], top_post_towns['count'])
        
        # Add value labels on top of bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}',
                    ha='center', va='bottom')
        
        plt.title('Top 10 Post Towns by Record Count')
        plt.xlabel('Post Town')
        plt.ylabel('Number of Records')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        # Display the plot
        st.pyplot(plt)
        
        # Display the raw data
        st.subheader("Raw Data")
        st.dataframe(top_post_towns)

# Function to create new build energy performance visualization
def create_new_build_visualization(df):
    if df is not None:
        # Calculate average new build energy performance rating by band
        new_build_ratings = df.groupby('NEW_BUILD_ENERGY_PERFORMANCE_BAND')['NEW_BUILD_ENERGY_PERFORMANCE_RATING'].mean().reset_index()
        new_build_ratings = new_build_ratings.sort_values('NEW_BUILD_ENERGY_PERFORMANCE_RATING', ascending=False)
        
        # Create visualization
        plt.figure(figsize=(12, 6))
        bars = plt.bar(new_build_ratings['NEW_BUILD_ENERGY_PERFORMANCE_BAND'], 
                      new_build_ratings['NEW_BUILD_ENERGY_PERFORMANCE_RATING'])
        
        # Add value labels on top of bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2f}',
                    ha='center', va='bottom')
        
        plt.title('Average New Build Energy Performance Rating by Band')
        plt.xlabel('Energy Performance Band')
        plt.ylabel('Average Rating')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        # Display the plot
        st.pyplot(plt)
        
        # Display the raw data
        st.subheader("Raw Data")
        st.dataframe(new_build_ratings)

# Function to create time-based statistics visualization
def create_time_based_stats_visualization(df):
    if df is not None:
        # Calculate monthly statistics
        changes_over_time = df.groupby(['lodgement_year', 'lodgement_month']).agg({
            'property_type': 'count',  # Using any column for count
            'energy_consumption_current': ['mean', 'median'],
            'co2_emissions_current': ['mean', 'median']
        }).reset_index()
        
        # Flatten column names
        changes_over_time.columns = ['lodgement_year', 'lodgement_month', 
                                   'total_certificates',
                                   'avg_energy_consumption', 'median_energy_consumption',
                                   'avg_co2_emissions', 'median_co2_emissions']
        
        # Create date column for x-axis
        changes_over_time['date'] = pd.to_datetime({
            'year': changes_over_time['lodgement_year'],
            'month': changes_over_time['lodgement_month'],
            'day': 1
        })
        
        # Create visualization
        plt.figure(figsize=(15, 10))
        
        # Plot 1: Total Certificates
        plt.subplot(2, 2, 1)
        plt.plot(changes_over_time['date'], changes_over_time['total_certificates'])
        plt.title('Total Certificates Over Time')
        plt.xlabel('Date')
        plt.ylabel('Number of Certificates')
        plt.xticks(rotation=45)
        
        # Plot 2: Energy Consumption
        plt.subplot(2, 2, 2)
        plt.plot(changes_over_time['date'], changes_over_time['avg_energy_consumption'], label='Mean')
        plt.plot(changes_over_time['date'], changes_over_time['median_energy_consumption'], label='Median')
        plt.title('Energy Consumption Over Time')
        plt.xlabel('Date')
        plt.ylabel('Energy Consumption')
        plt.legend()
        plt.xticks(rotation=45)
        
        # Plot 3: CO2 Emissions
        plt.subplot(2, 2, 3)
        plt.plot(changes_over_time['date'], changes_over_time['avg_co2_emissions'], label='Mean')
        plt.plot(changes_over_time['date'], changes_over_time['median_co2_emissions'], label='Median')
        plt.title('CO2 Emissions Over Time')
        plt.xlabel('Date')
        plt.ylabel('CO2 Emissions')
        plt.legend()
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        st.pyplot(plt)
        
        # Display the raw data
        st.subheader("Raw Data")
        st.dataframe(changes_over_time)

# Function to create combined changes over time visualization
def create_combined_changes_visualization(df):
    if df is not None:
        # Calculate monthly statistics
        changes_over_time = df.groupby(['lodgement_year', 'lodgement_month']).agg({
            'property_type': 'count',  # Using any column for count
            'energy_consumption_current': ['mean', 'median'],
            'co2_emissions_current': ['mean', 'median']
        }).reset_index()
        
        # Flatten column names
        changes_over_time.columns = ['lodgement_year', 'lodgement_month', 
                                   'total_certificates',
                                   'avg_energy_consumption', 'median_energy_consumption',
                                   'avg_co2_emissions', 'median_co2_emissions']
        
        # Create year_month column for x-axis
        changes_over_time['year_month'] = pd.to_datetime(
            changes_over_time['lodgement_year'].astype(str) + '-' + 
            changes_over_time['lodgement_month'].astype(str).str.zfill(2)
        )
        
        # Create visualization
        plt.figure(figsize=(14, 7))
        
        # Plot all metrics
        plt.plot(changes_over_time['year_month'], changes_over_time['avg_energy_consumption'], 
                label="Avg Energy Consumption", marker='o')
        plt.plot(changes_over_time['year_month'], changes_over_time['avg_co2_emissions'], 
                label="Avg CO2 Emissions", marker='x')
        plt.plot(changes_over_time['year_month'], changes_over_time['median_energy_consumption'], 
                label="Median Energy Consumption", linestyle='--')
        plt.plot(changes_over_time['year_month'], changes_over_time['median_co2_emissions'], 
                label="Median CO2 Emissions", linestyle='--')
        
        plt.title("Changes Over Time: Energy Consumption and CO2 Emissions")
        plt.xlabel("Year-Month")
        plt.ylabel("Value")
        plt.legend()
        plt.grid(True)
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        # Display the plot
        st.pyplot(plt)
        
        # Display the raw data
        st.subheader("Raw Data")
        st.dataframe(changes_over_time)

# Function to create Scotland Domestic changes over time visualization
def create_scotland_domestic_changes_visualization(df):
    if df is not None:
        # Create year_month column for x-axis
        df['year_month'] = pd.to_datetime(
            df['lodgement_year'].astype(str) + '-' + 
            df['lodgement_month'].astype(str).str.zfill(2)
        )
        
        # Create visualizations
        plt.figure(figsize=(15, 15))
        
        # Plot 1: Energy Consumption
        plt.subplot(3, 1, 1)
        plt.plot(df['year_month'], df['avg_energy_consumption_current'], 
                label="Current Avg", marker='o')
        plt.plot(df['year_month'], df['median_energy_consumption_current'], 
                label="Current Median", linestyle='--')
        plt.plot(df['year_month'], df['avg_energy_consumption_potential'], 
                label="Potential Avg", marker='x')
        plt.plot(df['year_month'], df['median_energy_consumption_potential'], 
                label="Potential Median", linestyle=':')
        plt.title("Energy Consumption Over Time")
        plt.xlabel("Year-Month")
        plt.ylabel("Energy Consumption")
        plt.legend()
        plt.grid(True)
        plt.xticks(rotation=45)
        
        # Plot 2: CO2 Emissions
        plt.subplot(3, 1, 2)
        plt.plot(df['year_month'], df['avg_co2_emissions_current'], 
                label="Current Avg", marker='o')
        plt.plot(df['year_month'], df['median_co2_emissions_current'], 
                label="Current Median", linestyle='--')
        plt.plot(df['year_month'], df['avg_co2_emissions_potential'], 
                label="Potential Avg", marker='x')
        plt.plot(df['year_month'], df['median_co2_emissions_potential'], 
                label="Potential Median", linestyle=':')
        plt.title("CO2 Emissions Over Time")
        plt.xlabel("Year-Month")
        plt.ylabel("CO2 Emissions")
        plt.legend()
        plt.grid(True)
        plt.xticks(rotation=45)
        
        # Plot 3: Heating Cost
        plt.subplot(3, 1, 3)
        plt.plot(df['year_month'], df['avg_heating_cost_current'], 
                label="Current Avg", marker='o')
        plt.plot(df['year_month'], df['median_heating_cost_current'], 
                label="Current Median", linestyle='--')
        plt.plot(df['year_month'], df['avg_heating_cost_potential'], 
                label="Potential Avg", marker='x')
        plt.plot(df['year_month'], df['median_heating_cost_potential'], 
                label="Potential Median", linestyle=':')
        plt.title("Heating Cost Over Time")
        plt.xlabel("Year-Month")
        plt.ylabel("Heating Cost")
        plt.legend()
        plt.grid(True)
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        st.pyplot(plt)
        
        # Display the raw data
        st.subheader("Raw Data")
        st.dataframe(df)

# Function to create property types visualization
def create_property_types_visualization(df):
    if df is not None:
        # Calculate property type counts
        property_counts = df.groupby('property_type').size().reset_index(name='count')
        property_counts = property_counts.sort_values('count', ascending=False)
        
        # Create visualization
        plt.figure(figsize=(12, 6))
        bars = plt.bar(property_counts['property_type'], property_counts['count'])
        
        # Add value labels on top of bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}',
                    ha='center', va='bottom')
        
        plt.title('Most Common Property Types')
        plt.xlabel('Property Type')
        plt.ylabel('Count')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        # Display the plot
        st.pyplot(plt)
        
        # Display the raw data
        st.subheader("Raw Data")
        st.dataframe(property_counts)

# Function to create heating category visualization
def create_heating_category_visualization(df):
    if df is not None:
        # Calculate heating category counts
        heating_counts = df.groupby('main_heating_category').size().reset_index(name='count')
        heating_counts = heating_counts.sort_values('count', ascending=False)
        
        # Create visualization
        plt.figure(figsize=(12, 6))
        bars = plt.bar(heating_counts['main_heating_category'], heating_counts['count'])
        
        # Add value labels on top of bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}',
                    ha='center', va='bottom')
        
        plt.title('Main Heating Category Distribution')
        plt.xlabel('Heating Category')
        plt.ylabel('Count')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        # Display the plot
        st.pyplot(plt)
        
        # Display the raw data
        st.subheader("Raw Data")
        st.dataframe(heating_counts)

# Function to create transaction types visualization
def create_transaction_types_visualization(df):
    if df is not None:
        # Calculate transaction type counts
        tx_counts = df.groupby('transaction_type').size().reset_index(name='count')
        tx_counts = tx_counts.sort_values('count', ascending=False)
        
        # Create visualization
        plt.figure(figsize=(12, 6))
        bars = plt.bar(tx_counts['transaction_type'], tx_counts['count'])
        
        # Add value labels on top of bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}',
                    ha='center', va='bottom')
        
        plt.title('Transaction Types Distribution')
        plt.xlabel('Transaction Type')
        plt.ylabel('Count')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        # Display the plot
        st.pyplot(plt)
        
        # Display the raw data
        st.subheader("Raw Data")
        st.dataframe(tx_counts)

# Function to create local authority visualization
def create_local_authority_visualization(df):
    if df is not None:
        # Calculate local authority counts
        local_auth_counts = df.groupby('local_authority_label').size().reset_index(name='count')
        local_auth_counts = local_auth_counts.sort_values('count', ascending=False)
        
        # Create visualization
        plt.figure(figsize=(12, 6))
        bars = plt.bar(local_auth_counts['local_authority_label'], local_auth_counts['count'])
        
        # Add value labels on top of bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}',
                    ha='center', va='bottom')
        
        plt.title('Local Authority Breakdown')
        plt.xlabel('Local Authority')
        plt.ylabel('Count')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        # Display the plot
        st.pyplot(plt)
        
        # Display the raw data
        st.subheader("Raw Data")
        st.dataframe(local_auth_counts)

# Function to create low energy lighting visualization
def create_low_energy_lighting_visualization(df):
    if df is not None:
        # Calculate low energy lighting counts
        lighting_counts = df['low_energy_lighting'].value_counts().reset_index()
        lighting_counts.columns = ['status', 'count']
        
        # Create visualization
        plt.figure(figsize=(8, 8))
        plt.pie(lighting_counts['count'], labels=lighting_counts['status'], autopct='%1.1f%%')
        plt.title('Low Energy Lighting Distribution')
        plt.axis('equal')
        
        # Display the plot
        st.pyplot(plt)
        
        # Display the raw data
        st.subheader("Raw Data")
        st.dataframe(lighting_counts)

# Function to create construction age band visualization
def create_construction_age_band_visualization(df):
    if df is not None:
        # Calculate age band counts
        age_band_counts = df.groupby('construction_age_band').size().reset_index(name='count')
        age_band_counts = age_band_counts.sort_values('count', ascending=False)
        
        # Create visualization
        plt.figure(figsize=(12, 6))
        bars = plt.bar(age_band_counts['construction_age_band'], age_band_counts['count'])
        
        # Add value labels on top of bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}',
                    ha='center', va='bottom')
        
        plt.title('Construction Age Band Distribution')
        plt.xlabel('Age Band')
        plt.ylabel('Count')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        # Display the plot
        st.pyplot(plt)
        
        # Display the raw data
        st.subheader("Raw Data")
        st.dataframe(age_band_counts)

# Function to create EPC rating distribution visualization
def create_epc_rating_distribution_visualization(df):
    if df is not None:
        # Calculate current and potential rating counts
        current_rating_counts = df.groupby('current_energy_rating').size().reset_index(name='count')
        current_rating_counts = current_rating_counts.sort_values('current_energy_rating')
        
        potential_rating_counts = df.groupby('potential_energy_rating').size().reset_index(name='count')
        potential_rating_counts = potential_rating_counts.sort_values('potential_energy_rating')
        
        # Create visualization
        plt.figure(figsize=(15, 6))
        
        # Plot current ratings
        plt.subplot(1, 2, 1)
        bars1 = plt.bar(current_rating_counts['current_energy_rating'], current_rating_counts['count'])
        for bar in bars1:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}',
                    ha='center', va='bottom')
        plt.title('Current Energy Rating Distribution')
        plt.xlabel('Energy Rating')
        plt.ylabel('Count')
        plt.xticks(rotation=45)
        
        # Plot potential ratings
        plt.subplot(1, 2, 2)
        bars2 = plt.bar(potential_rating_counts['potential_energy_rating'], potential_rating_counts['count'])
        for bar in bars2:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}',
                    ha='center', va='bottom')
        plt.title('Potential Energy Rating Distribution')
        plt.xlabel('Energy Rating')
        plt.ylabel('Count')
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        st.pyplot(plt)
        
        # Display the raw data
        st.subheader("Raw Data")
        col1, col2 = st.columns(2)
        with col1:
            st.write("Current Energy Ratings")
            st.dataframe(current_rating_counts)
        with col2:
            st.write("Potential Energy Ratings")
            st.dataframe(potential_rating_counts)

# Function to create average CO2 and energy visualization
def create_average_co2_energy_visualization(df):
    if df is not None:
        # Calculate averages
        avg_stats = pd.DataFrame({
            'Metric': ['Average CO₂ Emissions', 'Average Energy Consumption'],
            'Value': [
                df['co2_emissions_current'].mean(),
                df['energy_consumption_current'].mean()
            ]
        })
        
        # Create visualization
        plt.figure(figsize=(10, 6))
        bars = plt.bar(avg_stats['Metric'], avg_stats['Value'])
        
        # Add value labels on top of bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2f}',
                    ha='center', va='bottom')
        
        plt.title('Average CO₂ Emissions and Energy Consumption')
        plt.ylabel('Value')
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        # Display the plot
        st.pyplot(plt)
        
        # Display the raw data
        st.subheader("Raw Data")
        st.dataframe(avg_stats)

# Function to create average floor area visualization
def create_average_floor_area_visualization(df):
    if df is not None:
        # Calculate average floor area by property type
        avg_floor_areas = df.groupby('property_type')['total_floor_area'].mean().reset_index()
        avg_floor_areas = avg_floor_areas.sort_values('total_floor_area', ascending=False)
        
        # Create visualization
        plt.figure(figsize=(12, 6))
        bars = plt.bar(avg_floor_areas['property_type'], avg_floor_areas['total_floor_area'])
        
        # Add value labels on top of bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2f}',
                    ha='center', va='bottom')
        
        plt.title('Average Floor Area by Property Type')
        plt.xlabel('Property Type')
        plt.ylabel('Average Floor Area')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        # Display the plot
        st.pyplot(plt)
        
        # Display the raw data
        st.subheader("Raw Data")
        st.dataframe(avg_floor_areas)

# Function to create EPC ratings by year visualization
def create_epc_ratings_by_year_visualization(df):
    if df is not None:
        # Calculate yearly rating counts
        yearly_ratings = df.groupby(['lodgement_year', 'current_energy_rating']).size().reset_index(name='count')
        yearly_ratings = yearly_ratings.sort_values(['lodgement_year', 'current_energy_rating'])
        
        # Create visualization
        plt.figure(figsize=(15, 8))
        
        # Get unique years and ratings
        years = yearly_ratings['lodgement_year'].unique()
        ratings = yearly_ratings['current_energy_rating'].unique()
        
        # Create grouped bar chart
        x = np.arange(len(years))
        width = 0.8 / len(ratings)
        
        for i, rating in enumerate(ratings):
            rating_data = yearly_ratings[yearly_ratings['current_energy_rating'] == rating]
            plt.bar(x + i * width, rating_data['count'], width, label=f'Rating {rating}')
        
        plt.title('EPC Ratings by Year')
        plt.xlabel('Year')
        plt.ylabel('Count')
        plt.xticks(x + width * (len(ratings) - 1) / 2, years)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Display the plot
        st.pyplot(plt)
        
        # Display the raw data
        st.subheader("Raw Data")
        st.dataframe(yearly_ratings)

# Function to create England & Wales Non-domestic changes over time visualization
def create_eng_wales_non_domestic_changes_visualization(df):
    if df is not None:
        # Calculate monthly statistics
        changes_over_time = df.groupby(['lodgement_year', 'lodgement_month']).agg({
            'primary_energy_value': ['mean', 'median'],
            'building_emissions': ['mean', 'median'],
            'floor_area': ['mean', 'median']
        }).reset_index()
        
        # Flatten column names
        changes_over_time.columns = ['lodgement_year', 'lodgement_month',
                                   'avg_primary_energy_value', 'median_primary_energy_value',
                                   'avg_building_emissions', 'median_building_emissions',
                                   'avg_floor_area', 'median_floor_area']
        
        # Create year_month column for x-axis
        changes_over_time['year_month'] = pd.to_datetime(
            changes_over_time['lodgement_year'].astype(str) + '-' + 
            changes_over_time['lodgement_month'].astype(str).str.zfill(2)
        )
        
        # Create visualization
        plt.figure(figsize=(15, 10))
        
        # Plot 1: Primary Energy Value
        plt.subplot(2, 2, 1)
        plt.plot(changes_over_time['year_month'], changes_over_time['avg_primary_energy_value'], 
                label='Mean', marker='o')
        plt.plot(changes_over_time['year_month'], changes_over_time['median_primary_energy_value'], 
                label='Median', linestyle='--')
        plt.title('Primary Energy Value Over Time')
        plt.xlabel('Year-Month')
        plt.ylabel('Primary Energy Value')
        plt.legend()
        plt.grid(True)
        plt.xticks(rotation=45)
        
        # Plot 2: Building Emissions
        plt.subplot(2, 2, 2)
        plt.plot(changes_over_time['year_month'], changes_over_time['avg_building_emissions'], 
                label='Mean', marker='o')
        plt.plot(changes_over_time['year_month'], changes_over_time['median_building_emissions'], 
                label='Median', linestyle='--')
        plt.title('Building Emissions Over Time')
        plt.xlabel('Year-Month')
        plt.ylabel('Building Emissions')
        plt.legend()
        plt.grid(True)
        plt.xticks(rotation=45)
        
        # Plot 3: Floor Area
        plt.subplot(2, 2, 3)
        plt.plot(changes_over_time['year_month'], changes_over_time['avg_floor_area'], 
                label='Mean', marker='o')
        plt.plot(changes_over_time['year_month'], changes_over_time['median_floor_area'], 
                label='Median', linestyle='--')
        plt.title('Floor Area Over Time')
        plt.xlabel('Year-Month')
        plt.ylabel('Floor Area')
        plt.legend()
        plt.grid(True)
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        st.pyplot(plt)
        
        # Display the raw data
        st.subheader("Raw Data")
        st.dataframe(changes_over_time)

# Main content
def main():
    # Create collapsible sections
    with st.expander("England & Wales", expanded=False):
        st.subheader("EPC Domestic")
        
        show_categorical_analysis = st.checkbox("Show Categorical Distinct Count Column Analysis", value=False, key="eng_wales_categorical")
        if show_categorical_analysis:
            df_eng_wales = fetch_data("epcDomesticEngWales")
            if df_eng_wales is not None:
                create_categorical_analysis_visualization(df_eng_wales)
            
        show_time_stats = st.checkbox("Show Time Based Statistics", value=False, key="eng_wales_time_stats")
        if show_time_stats:
            df_eng_wales = fetch_data("epcDomesticEngWales")
            if df_eng_wales is not None:
                create_time_based_stats_visualization(df_eng_wales)
            
        show_combined_changes = st.checkbox("Show Changes over Time: Energy Consumption and CO2 Emissions", value=False, key="eng_wales_combined_changes")
        if show_combined_changes:
            df_eng_wales = fetch_data("epcDomesticEngWales")
            if df_eng_wales is not None:
                create_combined_changes_visualization(df_eng_wales)
        
        st.subheader("EPC Non-domestic")
        
        show_eng_wales_changes = st.checkbox("Show Changes Over Time", value=False, key="eng_wales_non_domestic_changes")
        if show_eng_wales_changes:
            df_eng_wales_non_domestic = fetch_data("epcNonDomesticEngWales")
            if df_eng_wales_non_domestic is not None:
                create_eng_wales_non_domestic_changes_visualization(df_eng_wales_non_domestic)
    
    with st.expander("Scotland", expanded=False):
        st.subheader("EPC Domestic")
        
        show_scotland_changes = st.checkbox("Show Changes Over Time", value=False, key="scotland_domestic_changes")
        if show_scotland_changes:
            df_scotland_domestic_changes = fetch_data("scotDomChangesOverTimes")
            if df_scotland_domestic_changes is not None:
                create_scotland_domestic_changes_visualization(df_scotland_domestic_changes)
        
        show_property_types = st.checkbox("Show Most Common Property Types", value=False, key="scotland_domestic_property_types")
        if show_property_types:
            df_scotland_domestic = fetch_data("epcDomesticScotlands")
            if df_scotland_domestic is not None:
                create_property_types_visualization(df_scotland_domestic)
                
        show_heating_category = st.checkbox("Show Main Heating Category Distribution", value=False, key="scotland_domestic_heating")
        if show_heating_category:
            df_scotland_domestic = fetch_data("epcDomesticScotlands")
            if df_scotland_domestic is not None:
                create_heating_category_visualization(df_scotland_domestic)
                
        show_transaction_types = st.checkbox("Show Transaction Types", value=False, key="scotland_domestic_transaction")
        if show_transaction_types:
            df_scotland_domestic = fetch_data("epcDomesticScotlands")
            if df_scotland_domestic is not None:
                create_transaction_types_visualization(df_scotland_domestic)
                
        show_local_authority = st.checkbox("Show Local Authority Breakdown", value=False, key="scotland_domestic_local_auth")
        if show_local_authority:
            df_scotland_domestic = fetch_data("epcDomesticScotlands")
            if df_scotland_domestic is not None:
                create_local_authority_visualization(df_scotland_domestic)
                
        show_low_energy_lighting = st.checkbox("Show Low Energy Lighting", value=False, key="scotland_domestic_lighting")
        if show_low_energy_lighting:
            df_scotland_domestic = fetch_data("epcDomesticScotlands")
            if df_scotland_domestic is not None:
                create_low_energy_lighting_visualization(df_scotland_domestic)
                
        show_construction_age = st.checkbox("Show Construction Age Band Distribution", value=False, key="scotland_domestic_age")
        if show_construction_age:
            df_scotland_domestic = fetch_data("epcDomesticScotlands")
            if df_scotland_domestic is not None:
                create_construction_age_band_visualization(df_scotland_domestic)
                
        show_epc_ratings = st.checkbox("Show EPC Rating Distribution (Current & Potential)", value=False, key="scotland_domestic_epc")
        if show_epc_ratings:
            df_scotland_domestic = fetch_data("epcDomesticScotlands")
            if df_scotland_domestic is not None:
                create_epc_rating_distribution_visualization(df_scotland_domestic)
                
        show_avg_co2_energy = st.checkbox("Show Average CO2 Emissions & Energy Use", value=False, key="scotland_domestic_co2")
        if show_avg_co2_energy:
            df_scotland_domestic = fetch_data("epcDomesticScotlands")
            if df_scotland_domestic is not None:
                create_average_co2_energy_visualization(df_scotland_domestic)
                
        show_avg_floor_area = st.checkbox("Show Average Floor Area by Property Type", value=False, key="scotland_domestic_floor")
        if show_avg_floor_area:
            df_scotland_domestic = fetch_data("epcDomesticScotlands")
            if df_scotland_domestic is not None:
                create_average_floor_area_visualization(df_scotland_domestic)
                
        show_epc_by_year = st.checkbox("Show EPC Ratings by Year", value=False, key="scotland_domestic_epc_year")
        if show_epc_by_year:
            df_scotland_domestic = fetch_data("epcDomesticScotlands")
            if df_scotland_domestic is not None:
                create_epc_ratings_by_year_visualization(df_scotland_domestic)
        
        st.subheader("EPC Non-domestic")
        
        show_energy_bands = st.checkbox("Show Energy Performance Bands", value=False, key="scotland_non_domestic_bands")
        if show_energy_bands:
            df_scotland = fetch_data("epcNonDomesticScotlands")
            if df_scotland is not None:
                create_energy_bands_visualization(df_scotland)
                
        show_changes_over_time = st.checkbox("Show Changes Over Time", value=False, key="scotland_non_domestic_changes")
        if show_changes_over_time:
            df_scotland = fetch_data("epcNonDomesticScotlands")
            if df_scotland is not None:
                create_changes_over_time_visualization(df_scotland)
                
        show_avg_emissions = st.checkbox("Show Average Building Emissions and Primary Energy Value", value=False, key="scotland_non_domestic_emissions")
        if show_avg_emissions:
            df_scotland = fetch_data("epcNonDomesticScotlands")
            if df_scotland is not None:
                create_average_emissions_visualization(df_scotland)
                
        show_yearly_trends = st.checkbox("Show Yearly Energy Performance Trends", value=False, key="scotland_non_domestic_trends")
        if show_yearly_trends:
            df_scotland = fetch_data("epcNonDomesticScotlands")
            if df_scotland is not None:
                create_yearly_trends_visualization(df_scotland)
                
        show_floor_area = st.checkbox("Show Average Floor Area by Property Type", value=False, key="scotland_non_domestic_floor")
        if show_floor_area:
            df_scotland = fetch_data("epcNonDomesticScotlands")
            if df_scotland is not None:
                create_floor_area_visualization(df_scotland)
                
        show_post_towns = st.checkbox("Show Top 10 Post Towns by Record Count", value=False, key="scotland_non_domestic_towns")
        if show_post_towns:
            df_scotland = fetch_data("epcNonDomesticScotlands")
            if df_scotland is not None:
                create_top_post_towns_visualization(df_scotland)
                
        show_new_build = st.checkbox("Show Average New Build Energy Performance Rating by Band", value=False, key="scotland_non_domestic_new_build")
        if show_new_build:
            df_scotland = fetch_data("epcNonDomesticScotlands")
            if df_scotland is not None:
                create_new_build_visualization(df_scotland)

if __name__ == "__main__":
    main() 