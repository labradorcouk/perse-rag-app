# Fixing Azure AD Permissions for OAuth Authentication

## 🔍 **Issue Analysis**

The error you're seeing:
```
🔄 Processing authentication... Please wait.
Failed to get user info: 401
OAuth token doesn't have Graph permissions. Trying service principal for user validation...
```

This occurs because your Azure AD app registration doesn't have the necessary Microsoft Graph API permissions to fetch user information.

## 🛠️ **Solution Steps**

### Step 1: Update Azure AD App Registration Permissions

1. **Go to Azure Portal**
   - Navigate to [Azure Portal](https://portal.azure.com)
   - Go to **Azure Active Directory** → **App registrations**
   - Find your app registration

2. **Add Microsoft Graph Permissions**
   - Click on your app registration
   - Go to **API permissions**
   - Click **Add a permission**
   - Select **Microsoft Graph**
   - Choose **Delegated permissions**
   - Add these permissions:
     - `User.Read` (to read user profile)
     - `User.ReadBasic.All` (to read basic user info)
     - `email` (to read email)
     - `profile` (to read profile info)
     - `openid` (for OpenID Connect)

3. **Grant Admin Consent**
   - Click **Grant admin consent for [Your Organization]**
   - This will apply the permissions to all users in your organization

### Step 2: Update Environment Variables

Make sure your `.env` file has the correct scopes:

```env
# Azure AD Configuration
AZURE_CLIENT_ID=your_client_id
AZURE_CLIENT_SECRET=your_client_secret
AZURE_TENANT_ID=your_tenant_id
AZURE_REDIRECT_URI=http://localhost:8501/callback
AZURE_SCOPE=https://analysis.windows.net/powerbi/api/.default https://graph.microsoft.com/.default
```

### Step 3: Alternative Solution - JWT Token Extraction

If you can't modify the Azure AD permissions, the updated code will now:

1. **Request both scopes** during OAuth flow
2. **Extract user info from JWT token** if Graph API fails
3. **Fall back gracefully** with proper error messages

## 🔧 **Code Changes Made**

### 1. Updated OAuth Scope Request
```python
# Request both Fabric and Graph scopes
combined_scope = f"{self.scope} {self.graph_scope}"
```

### 2. Improved User Info Extraction
```python
# Extract user info from JWT token claims
decoded_token = jwt.decode(token_data['access_token'], options={"verify_signature": False})
user_info_from_token = {
    'displayName': decoded_token.get('name', 'Unknown User'),
    'userPrincipalName': decoded_token.get('upn', decoded_token.get('preferred_username')),
    'mail': decoded_token.get('email', decoded_token.get('preferred_username')),
    'jobTitle': decoded_token.get('jobtitle', ''),
    'officeLocation': decoded_token.get('office_location', '')
}
```

## 🧪 **Testing the Fix**

### Option 1: With Graph Permissions (Recommended)
1. Update Azure AD app permissions as described above
2. Restart your Docker containers
3. Try logging in again

### Option 2: Without Graph Permissions
1. The updated code will automatically extract user info from the JWT token
2. No additional Azure AD changes needed
3. Restart your Docker containers and try again

## 📋 **Verification Steps**

1. **Check Azure AD App Permissions**
   ```bash
   # In Azure Portal, verify these permissions are added:
   - Microsoft Graph > Delegated > User.Read
   - Microsoft Graph > Delegated > User.ReadBasic.All
   - Microsoft Graph > Delegated > email
   - Microsoft Graph > Delegated > profile
   - Microsoft Graph > Delegated > openid
   ```

2. **Test Authentication Flow**
   ```bash
   # Restart your containers
   docker-compose -f docker-compose.optimized.windows.yml down
   docker-compose -f docker-compose.optimized.windows.yml up -d
   ```

3. **Check Logs**
   ```bash
   # Monitor authentication logs
   docker-compose -f docker-compose.optimized.windows.yml logs rag-fabric-app
   ```

## 🚨 **Common Issues and Solutions**

### Issue 1: "AADSTS650057: Invalid resource"
**Solution**: Ensure your app registration has the correct redirect URI configured in Azure AD.

### Issue 2: "AADSTS500113: No reply address registered"
**Solution**: Add `http://localhost:8501/callback` to the redirect URIs in your Azure AD app registration.

### Issue 3: "Access denied" after successful authentication
**Solution**: Check that the user's email is in the `AUTHORIZED_USERS` environment variable.

## 🔍 **Debugging Tips**

1. **Enable Debug Logging**
   ```python
   # Add to your .env file
   DEBUG=true
   ```

2. **Check Token Claims**
   ```python
   # The updated code will show token claims in logs
   # Look for user information in the JWT token
   ```

3. **Monitor Network Requests**
   - Use browser developer tools to see the OAuth flow
   - Check for any 401/403 errors in the Network tab

## ✅ **Expected Outcome**

After implementing these fixes, you should see:

```
✅ Authentication successful! Welcome, [User Name]!
🔄 Redirecting to main application...
```

Instead of the error message you were seeing before.

## 🆘 **If Issues Persist**

1. **Check Azure AD App Configuration**
   - Verify client ID, secret, and tenant ID
   - Ensure redirect URI matches exactly

2. **Review Environment Variables**
   - Confirm all Azure AD variables are set correctly
   - Check that `AUTHORIZED_USERS` includes your email

3. **Check Container Logs**
   ```bash
   docker-compose -f docker-compose.optimized.windows.yml logs rag-fabric-app
   ```

4. **Test with Direct Login**
   - Try the direct login option as a fallback
   - This bypasses OAuth for testing purposes 