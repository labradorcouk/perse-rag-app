# Fabric RAG Application

A comprehensive Retrieval-Augmented Generation (RAG) application built with Streamlit, Qdrant vector database, and Azure AD authentication. This application provides intelligent data querying, code generation, and analysis capabilities for Microsoft Fabric datasets.

## 🚀 Features

- **🔐 Azure AD Authentication**: Secure login with enterprise Azure Active Directory
- **🧠 RAG Capabilities**: Intelligent query processing with OpenAI GPT models
- **🔍 Vector Search**: Fast semantic search using Qdrant vector database
- **💻 SQL Editor**: Direct database querying with ODBC connectivity
- **📊 Data Analysis**: Dynamic code generation and execution
- **📈 Performance Monitoring**: Real-time system and application metrics
- **🐳 Docker Deployment**: Containerized deployment with optimized configurations
- **🔧 Diagnostics**: Comprehensive logging and error tracking

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Streamlit     │    │     Qdrant      │    │   Microsoft     │
│   Frontend      │◄──►│   Vector DB     │◄──►│   Fabric SQL    │
│                 │    │                 │    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Azure AD      │    │   OpenAI API    │    │   Performance   │
│ Authentication  │    │   (GPT Models)  │    │   Optimizer     │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## 📋 Prerequisites

- Docker and Docker Compose
- Azure AD application registration
- OpenAI API key
- Microsoft Fabric SQL Endpoint access

## 🛠️ Installation

### 1. Clone the Repository

```bash
git clone <repository-url>
cd fabric-rag
```

### 2. Environment Setup

Copy the example environment file and configure your settings:

```bash
cp .env.example.azure .env
```

Edit `.env` with your configuration:

```env
# Azure AD Configuration
AZURE_CLIENT_ID=your_client_id
AZURE_CLIENT_SECRET=your_client_secret
AZURE_TENANT_ID=your_tenant_id
AZURE_REDIRECT_URI=http://localhost:8501/callback
AZURE_SCOPE=https://analysis.windows.net/powerbi/api/.default

# OpenAI Configuration
OPENAI_API_KEY=your_openai_api_key
DEFAULT_MODEL=gpt-3.5-turbo
MAX_TOKENS=600
TEMPERATURE=0.2

# Authorized Users
AUTHORIZED_USERS=user1@domain.com,user2@domain.com

# Qdrant Configuration
QDRANT_HOST=qdrant
QDRANT_PORT=6333
QDRANT_URL=http://qdrant:6333

# Email Configuration (Optional)
EMAIL_USER=your_email@domain.com
ADMIN_EMAIL=admin@domain.com
```

### 3. Azure AD Setup

Follow the setup instructions in `setup_azure_ad.py` to configure Azure AD authentication:

```bash
python setup_azure_ad.py
```

### 4. Deploy with Docker

#### Windows Deployment (Recommended)

```bash
# Use the Windows-optimized deployment script
deploy_windows_optimized.bat
```

Or manually:

```bash
# Build and deploy
docker-compose -f docker-compose.optimized.windows.yml build --no-cache
docker-compose -f docker-compose.optimized.windows.yml up -d
```

#### Linux/CentOS Deployment

```bash
# Use the optimized deployment script
./deploy_optimized.sh
```

### 5. Access the Application

Open your browser and navigate to: `http://localhost:8501`

## 🔧 Configuration

### Docker Compose Files

- `docker-compose.yml`: Standard deployment
- `docker-compose.optimized.windows.yml`: Windows-optimized deployment
- `docker-compose.optimized.yml`: Linux-optimized deployment

### Dockerfiles

- `Dockerfile.docker`: Standard build
- `Dockerfile.docker.optimized`: Performance-optimized build
- `Dockerfile.docker.alternative`: Alternative ODBC build

## 📁 Project Structure

```
fabric-rag/
├── 📁 config/                 # Configuration files
│   ├── rag_tables_config.yaml
│   └── safe_builtins.py
├── 📁 utils/                  # Utility modules
│   ├── qdrant_utils.py       # Qdrant vector database utilities
│   ├── rag_utils.py          # RAG processing utilities
│   ├── performance_optimizer.py # Performance optimization
│   ├── diagnostics_logger.py # Logging and diagnostics
│   └── dataframe_corrector.py # DataFrame name correction
├── 📁 deployment-package/     # Deployment assets
├── 📄 rag_fabric_app.py      # Main Streamlit application
├── 📄 rag_fabric_app_docker.py # Docker-optimized app
├── 📄 auth_azure_docker.py   # Azure AD authentication
├── 📄 docker-compose.yml     # Docker Compose configuration
├── 📄 requirements.txt       # Python dependencies
└── 📄 README.md             # This file
```

## 🔍 Usage

### 1. Authentication

- Click the "Login with Azure AD" button
- Complete the OAuth flow
- Access is granted based on authorized users list

### 2. RAG Query Processing

1. **Select Vector Search Engine**: Choose between FAISS or Qdrant
2. **Enter Query**: Type your natural language question
3. **Review Results**: View retrieved documents and generated code
4. **Execute Analysis**: Run the generated code for data analysis
5. **Download Results**: Export findings as CSV

### 3. SQL Editor

1. **Connect to Database**: Use Azure AD token for authentication
2. **Write Queries**: Execute SQL against Microsoft Fabric
3. **View Results**: Interactive data exploration
4. **Export Data**: Download query results

### 4. Performance Monitoring

- **Real-time Metrics**: CPU, Memory, Disk usage
- **Cache Statistics**: Hit rates and optimization
- **System Health**: Container and service status

## 🛠️ Development

### Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Run the application
streamlit run rag_fabric_app.py
```

### Testing

```bash
# Test Qdrant connection
python test_qdrant_connection.py

# Test Azure authentication
python test_azure_auth.py

# Test DataFrame corrections
python test_dynamic_dataframe_fixes.py
```

### Adding New Features

1. **New Utilities**: Add to `utils/` directory
2. **Configuration**: Update `config/` files
3. **Dependencies**: Update `requirements.txt`
4. **Documentation**: Update this README

## 🔧 Troubleshooting

### Common Issues

#### Qdrant Connection Issues
```bash
# Check Qdrant health
curl http://localhost:6333/healthz

# View Qdrant logs
docker-compose logs qdrant
```

#### Authentication Problems
```bash
# Test Azure AD configuration
python test_azure_auth.py

# Check environment variables
python test_auth_fixes.py
```

#### Performance Issues
```bash
# Monitor system resources
docker stats

# Check application logs
docker-compose logs rag-fabric-app
```

### Debugging

1. **Enable Debug Logging**: Set `DEBUG=true` in environment
2. **Check Diagnostics**: Use the Diagnostics tab in the app
3. **Review Logs**: Check container logs for errors
4. **Test Components**: Use individual test scripts

## 📊 Performance Optimization

### System-Level Optimizations

- **Memory Management**: LRU caching for heavy components
- **I/O Optimization**: Batch operations and connection pooling
- **Resource Limits**: Docker container resource constraints
- **Background Processing**: Lazy loading and async operations

### Application-Level Optimizations

- **Model Caching**: Pre-downloaded embedding models
- **DataFrame Caching**: Intelligent caching of query results
- **Connection Pooling**: Reusable database connections
- **Memory Cleanup**: Automatic garbage collection

## 🔒 Security

### Authentication
- Azure AD OAuth 2.0 Authorization Code Flow
- Service Principal authentication for Docker deployments
- User validation against authorized users list

### Data Protection
- Environment variable configuration
- Secure token handling
- No hardcoded credentials

### Network Security
- Container isolation
- Internal service communication
- Health check endpoints

## 📈 Monitoring and Logging

### Diagnostics Dashboard
- Real-time system metrics
- Application performance monitoring
- Error tracking and analysis
- User activity logging

### Log Storage
- Centralized logging in Qdrant
- Structured log format
- Performance metrics tracking
- Error categorization

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Update documentation
6. Submit a pull request

## 📄 License

This project is licensed under the Apache License 2.0 - see the LICENSE file for details.

## 🆘 Support

For issues and questions:

1. Check the troubleshooting section
2. Review the diagnostics dashboard
3. Check container logs
4. Create an issue with detailed information

## 🔄 Version History

- **v1.0.0**: Initial release with basic RAG functionality
- **v1.1.0**: Added Azure AD authentication
- **v1.2.0**: Docker deployment support
- **v1.3.0**: Performance optimizations and monitoring
- **v1.4.0**: Windows deployment support and diagnostics

---

**Built with ❤️ for Microsoft Fabric data analysis** 