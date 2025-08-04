import streamlit as st
import struct
from itertools import chain, repeat
import urllib.parse

# Conditional imports for ODBC support
try:
    import pyodbc
    import sqlalchemy as sa
    ODBC_AVAILABLE = True
except ImportError:
    ODBC_AVAILABLE = False
    pyodbc = None
    sa = None

class SQLConnectionManager:
    def __init__(self):
        self.sql_endpoint = "twprzboxbsruppjvrglogyufxu-uywupunqqmwepeeo6rptcas3jq.datawarehouse.fabric.microsoft.com"
        self.database = "default"
        self.driver = "ODBC Driver 18 for SQL Server"
        self.resource_url = "https://database.windows.net/.default"
    
    def create_engine_with_token(self, access_token):
        """
        Create SQLAlchemy engine using the provided access token.
        
        Args:
            access_token: Azure AD access token for database authentication
            
        Returns:
            SQLAlchemy engine object
        """
        if not ODBC_AVAILABLE:
            raise ImportError("ODBC drivers not available. SQL functionality is limited.")
        
        if not access_token:
            raise ValueError("Access token is required for database connection.")
        
        # Create connection string
        connection_string = (
            f"Driver={{{self.driver}}};"
            f"Server={self.sql_endpoint},1433;"
            f"Database={self.database};"
            "Encrypt=Yes;"
            "TrustServerCertificate=No"
        )
        
        # Encode connection string
        params = urllib.parse.quote(connection_string)
        
        # Prepare token for ODBC
        token_as_bytes = bytes(access_token, "UTF-8")
        encoded_bytes = bytes(chain.from_iterable(zip(token_as_bytes, repeat(0))))
        token_bytes = struct.pack("<i", len(encoded_bytes)) + encoded_bytes
        attrs_before = {1256: token_bytes}
        
        # Create engine
        engine = sa.create_engine(
            f"mssql+pyodbc:///?odbc_connect={params}",
            connect_args={'attrs_before': attrs_before}
        )
        
        return engine
    
    def test_connection(self, access_token):
        """
        Test the database connection with the provided token.
        
        Args:
            access_token: Azure AD access token
            
        Returns:
            bool: True if connection successful, False otherwise
        """
        try:
            engine = self.create_engine_with_token(access_token)
            
            # Test connection with a simple query
            with engine.connect() as conn:
                result = conn.execute(sa.text("SELECT 1 as test"))
                result.fetchone()
            
            return True
        except Exception as e:
            st.error(f"Database connection test failed: {str(e)}")
            return False
    
    def execute_query(self, access_token, query, params=None):
        """
        Execute a SQL query using the provided access token.
        
        Args:
            access_token: Azure AD access token
            query: SQL query to execute
            params: Optional parameters for the query
            
        Returns:
            pandas.DataFrame: Query results
        """
        if not ODBC_AVAILABLE:
            raise ImportError("ODBC drivers not available. SQL functionality is limited.")
        
        import pandas as pd
        
        engine = self.create_engine_with_token(access_token)
        
        try:
            if params:
                df = pd.read_sql(query, engine, params=params)
            else:
                df = pd.read_sql(query, engine)
            return df
        except Exception as e:
            st.error(f"Query execution failed: {str(e)}")
            raise
    
    def get_connection_info(self):
        """
        Get connection information for display purposes.
        
        Returns:
            dict: Connection information
        """
        return {
            'endpoint': self.sql_endpoint,
            'database': self.database,
            'driver': self.driver,
            'odbc_available': ODBC_AVAILABLE
        }

# Global instance
sql_manager = SQLConnectionManager() 