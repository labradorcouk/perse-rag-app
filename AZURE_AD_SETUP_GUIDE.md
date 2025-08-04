# Azure AD Setup Guide for Docker Authentication

This guide will help you set up Azure AD authentication for the Fabric RAG application running in Docker.

## 🔧 **Step 1: Azure AD App Registration Setup**

### 1.1 Create App Registration
1. Go to [Azure Portal](https://portal.azure.com)
2. Navigate to **Azure Active Directory** > **App registrations**
3. Click **New registration**
4. Fill in the details:
   - **Name**: `Fabric RAG Docker App`
   - **Supported account types**: `Accounts in this organizational directory only`
   - **Redirect URI**: `Web` > `http://localhost:8501/` (note the trailing slash)
5. Click **Register**

### 1.2 Get Application Details
- Copy the **Application (client) ID** - this is your `AZURE_CLIENT_ID`
- Copy the **Directory (tenant) ID** - this is your `AZURE_TENANT_ID`

### 1.3 Create Client Secret
1. In the app registration, go to **Certificates & secrets**
2. Click **New client secret**
3. Description: `Docker App Secret`
4. Expiration: Choose appropriate duration
5. Copy the **secret value** - this is your `AZURE_CLIENT_SECRET`

## 🔐 **Step 2: Configure API Permissions**

### 2.1 Microsoft Graph Permissions
1. Go to **API permissions** in the app registration
2. Click **Add a permission**
3. Select **Microsoft Graph**
4. Choose **Delegated permissions**
5. Add these permissions:
   - `User.Read` (to get user info)
6. Click **Add permissions**

### 2.2 Power BI Service Permissions
1. Go to **API permissions** again
2. Click **Add a permission**
3. Select **Power BI Service** (search for "Power BI")
4. Choose **Delegated permissions**
5. Add `Dataset.Read.All` permission
6. Click **Add permissions**

### 2.3 Grant Admin Consent
1. Click **Grant admin consent** for all permissions
2. This is **required** for the permissions to take effect

## ⚙️ **Step 3: Environment Configuration**

### 3.1 Update .env File
Copy `.env.example.azure` to `.env` and update with your values:

```bash
# Azure AD Configuration
AZURE_CLIENT_ID=your_client_id_here
AZURE_CLIENT_SECRET=your_client_secret_here
AZURE_TENANT_ID=your_tenant_id_here
AZURE_REDIRECT_URI=http://localhost:8501/
AZURE_SCOPE=https://analysis.windows.net/powerbi/api/.default
```

### 3.2 Test Configuration
Run the test script to verify your configuration:

```bash
python test_azure_auth.py
```

## 🚀 **Step 4: Deploy and Test**

### 4.1 Start the Application
```bash
docker-compose up -d
```

### 4.2 Access the Application
1. Open your browser to: `http://localhost:8501`
2. You should see the Azure AD authentication page
3. Click **"Login with Azure AD (Direct)"** first
4. If that doesn't work, try **"Login with Azure AD (OAuth)"**

## 🔍 **Troubleshooting**

### Issue: "No reply address is registered"
**Solution**: Make sure the redirect URI in Azure AD exactly matches your environment variable:
- Azure AD: `http://localhost:8501/` (with trailing slash)
- Environment: `AZURE_REDIRECT_URI=http://localhost:8501/`

### Issue: "Invalid resource" error
**Solution**: Make sure you've added the correct permissions:
- Microsoft Graph > User.Read
- Power BI Service > Dataset.Read.All

### Issue: "AADSTS650057: Invalid resource"
**Solution**: The scope should be `https://analysis.windows.net/powerbi/api/.default`, not `https://database.windows.net/.default`

### Issue: Perpetual redirects
**Solution**: 
1. Try the **Direct** authentication method first
2. If that doesn't work, use the **OAuth** method
3. Make sure admin consent is granted for all permissions

## 📋 **Authentication Methods**

### Method 1: Direct Authentication (Recommended for Docker)
- Uses service principal credentials
- No browser redirect required
- Works well in Docker containers
- Click **"Login with Azure AD (Direct)"**

### Method 2: OAuth Flow
- Uses browser-based authentication
- Requires proper redirect URI configuration
- May have issues in Docker environments
- Click **"Login with Azure AD (OAuth)"**

## ✅ **Verification Checklist**

- [ ] Azure AD app registration created
- [ ] Client ID, Client Secret, and Tenant ID copied
- [ ] Redirect URI set to `http://localhost:8501/`
- [ ] Microsoft Graph > User.Read permission added
- [ ] Power BI Service > Dataset.Read.All permission added
- [ ] Admin consent granted for all permissions
- [ ] Environment variables configured in `.env`
- [ ] Docker containers running (`docker-compose ps`)
- [ ] Application accessible at `http://localhost:8501`
- [ ] Authentication working (try both methods)

## 🆘 **Need Help?**

If you're still having issues:

1. **Check the logs**: `docker-compose logs rag-fabric-app`
2. **Test configuration**: `python test_azure_auth.py`
3. **Verify permissions**: Make sure admin consent is granted
4. **Check redirect URI**: Must match exactly in Azure AD and environment

## 📞 **Support**

For additional help, check:
- Azure AD documentation
- Docker networking issues
- Streamlit authentication documentation 