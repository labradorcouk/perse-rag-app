# Azure Authentication for Fabric RAG Docker App

This document explains how to set up Azure AD authentication for the Fabric RAG application running in Docker containers.

## 🎯 Overview

The application now supports Azure AD authentication that works in Docker containers and preserves access tokens for SQL endpoint connections. This replaces the previous service principal approach with a more user-friendly OAuth flow.

## 🔧 Key Features

- ✅ **Docker-Compatible**: Works in containerized environments
- ✅ **Token Preservation**: Access tokens are preserved throughout the session
- ✅ **SQL Integration**: Preserved tokens are used for Fabric SQL endpoint connections
- ✅ **User-Friendly**: Simple login button with Azure AD OAuth flow
- ✅ **Secure**: Proper token validation and expiration handling

## 📋 Prerequisites

1. **Azure AD Tenant**: You need access to an Azure AD tenant
2. **Azure Portal Access**: Ability to create app registrations
3. **Docker & Docker Compose**: For running the application
4. **Fabric SQL Endpoint**: Access to Microsoft Fabric SQL endpoint

## 🚀 Quick Setup

### 1. Run the Setup Script

```bash
python setup_azure_ad.py
```

This script will:
- Check your current configuration
- Provide step-by-step instructions
- Test Azure AD endpoints

### 2. Create Azure AD App Registration

Follow the instructions from the setup script, or manually:

1. Go to [Azure Portal](https://portal.azure.com)
2. Navigate to **Azure Active Directory** > **App registrations**
3. Click **New registration**
4. Configure:
   - **Name**: `Fabric RAG Docker App`
   - **Supported account types**: `Accounts in this organizational directory only`
   - **Redirect URI**: Web > `http://localhost:8501`
5. Click **Register**

### 3. Configure App Registration

#### Get Required Values

From your app registration, note:
- **Application (client) ID** → `AZURE_CLIENT_ID`
- **Directory (tenant) ID** → `AZURE_TENANT_ID`

#### Create Client Secret

1. Go to **Certificates & secrets**
2. Click **New client secret**
3. Description: `Docker App Secret`
4. Copy the secret value → `AZURE_CLIENT_SECRET`

#### Configure API Permissions

1. Go to **API permissions**
2. Add **Microsoft Graph** > **Delegated permissions**:
   - `User.Read` (for user info)
3. Add **Azure SQL Database** > **Delegated permissions**:
   - `user_impersonation` (for database access)
4. Click **Grant admin consent**

### 4. Configure Environment Variables

Copy the example environment file:

```bash
cp .env.example.azure .env
```

Edit `.env` with your values:

```env
# Azure AD Configuration
AZURE_CLIENT_ID=your_client_id_here
AZURE_CLIENT_SECRET=your_client_secret_here
AZURE_TENANT_ID=your_tenant_id_here
AZURE_REDIRECT_URI=http://localhost:8501
AZURE_SCOPE=https://database.windows.net/.default

# Other configurations...
OPENAI_API_KEY=your_openai_key
QDRANT_URL=http://qdrant:6333
```

### 5. Deploy the Application

```bash
# Build and start the containers
docker-compose up -d

# Check status
docker-compose ps

# View logs
docker-compose logs -f rag-fabric-app
```

### 6. Access the Application

1. Open your browser to `http://localhost:8501`
2. Click **Login with Azure AD**
3. Complete the authentication flow
4. You'll be redirected back to the application

## 🔐 Authentication Flow

1. **Login Button**: User clicks "Login with Azure AD"
2. **OAuth Redirect**: User is redirected to Azure AD login
3. **Authentication**: User authenticates with their Azure credentials
4. **Token Exchange**: App exchanges authorization code for access token
5. **User Info**: App fetches user information from Microsoft Graph
6. **Session Storage**: Token and user info are stored in session state
7. **SQL Connections**: Preserved token is used for database connections

## 🗄️ SQL Endpoint Integration

The preserved access token is automatically used for:

- **Connection Testing**: Verify database connectivity
- **Query Execution**: Execute SQL queries against Fabric SQL endpoint
- **Token Validation**: Check token expiration before operations

## 🔧 Configuration Options

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `AZURE_CLIENT_ID` | Azure AD app registration client ID | Required |
| `AZURE_CLIENT_SECRET` | Azure AD app registration client secret | Required |
| `AZURE_TENANT_ID` | Azure AD tenant ID | Required |
| `AZURE_REDIRECT_URI` | OAuth redirect URI | `http://localhost:8501` |
| `AZURE_SCOPE` | OAuth scope for database access | `https://database.windows.net/.default` |

### Docker Configuration

The application supports different Docker builds:

```bash
# Without ODBC (faster, limited SQL functionality)
docker-compose up -d

# With ODBC (full SQL functionality)
docker build -f Dockerfile.docker -t rag-fabric-app .
docker-compose up -d
```

## 🛠️ Troubleshooting

### Common Issues

1. **"Invalid redirect URI"**
   - Ensure `AZURE_REDIRECT_URI` matches exactly in Azure AD app registration
   - Check that the URI is configured as a Web redirect URI

2. **"Token exchange failed"**
   - Verify `AZURE_CLIENT_SECRET` is correct
   - Check that the client secret hasn't expired
   - Ensure API permissions are granted

3. **"Database connection failed"**
   - Verify the user has access to the Fabric SQL endpoint
   - Check that `user_impersonation` permission is granted
   - Ensure ODBC drivers are installed (if using SQL features)

4. **"Qdrant connection refused"**
   - Check that Qdrant container is running: `docker-compose ps`
   - View Qdrant logs: `docker-compose logs qdrant`
   - Restart containers: `docker-compose restart`

### Debug Commands

```bash
# Check container status
docker-compose ps

# View application logs
docker-compose logs rag-fabric-app

# View Qdrant logs
docker-compose logs qdrant

# Test Azure configuration
python setup_azure_ad.py

# Rebuild containers
docker-compose down
docker-compose build --no-cache
docker-compose up -d
```

## 🔒 Security Considerations

1. **Client Secrets**: Store securely, rotate regularly
2. **Redirect URIs**: Only use trusted redirect URIs
3. **Token Storage**: Tokens are stored in session state (in-memory)
4. **Network Security**: Use HTTPS in production
5. **User Permissions**: Grant minimal required permissions

## 📚 Additional Resources

- [Azure AD App Registration Documentation](https://docs.microsoft.com/en-us/azure/active-directory/develop/quickstart-register-app)
- [Microsoft Graph API Documentation](https://docs.microsoft.com/en-us/graph/)
- [Azure SQL Database Authentication](https://docs.microsoft.com/en-us/azure/azure-sql/database/authentication-aad-configure)

## 🆘 Support

If you encounter issues:

1. Check the troubleshooting section above
2. Review application logs: `docker-compose logs rag-fabric-app`
3. Test Azure configuration: `python setup_azure_ad.py`
4. Verify environment variables are set correctly
5. Ensure all API permissions are granted in Azure AD 