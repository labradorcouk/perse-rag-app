import pyodbc
from azure.identity import InteractiveBrowserCredential
import os

server = os.getenv("SQL_SERVER")
database = os.getenv("SQL_DATABASE", "default")
driver = os.getenv("SQL_DRIVER", "ODBC Driver 17 for SQL Server")

credential = InteractiveBrowserCredential()
resource_url = "https://database.windows.net/.default"
token_object = credential.get_token(resource_url)
access_token = token_object.token

conn_str = (
    f"Driver={{{driver}}};"
    f"Server={server},1433;"
    f"Database={database};"
    "Encrypt=Yes;"
    "TrustServerCertificate=No;"
    "Authentication=ActiveDirectoryAccessToken;"
)

conn = pyodbc.connect(conn_str, attrs_before={1256: access_token})
print("Connected!")