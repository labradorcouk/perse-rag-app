# DEVELOPMENT_GUIDE.md

## Development Setup and Guidelines

This guide provides comprehensive information for developers working on the Fabric RAG QA App, including setup instructions, architecture details, and contribution guidelines.

## 🚀 Development Environment Setup

### Prerequisites
- Python 3.8+
- Git
- Microsoft Fabric access
- OpenAI API key
- ODBC Driver 17/18 for SQL Server

### Local Development Setup

#### 1. Clone and Setup
```bash
# Clone the repository
git clone <repository-url>
cd fabric-rag-app

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

#### 2. Environment Configuration
```bash
# Copy environment template
cp .env.example .env

# Edit .env with your development credentials
OPENAI_API_KEY=your_openai_api_key
DEFAULT_MODEL=gpt-3.5-turbo
MAX_TOKENS=600
TEMPERATURE=0.2
APP_DEBUG=True  # Enable debug mode for development
```

#### 3. Azure Setup
- Ensure Azure AD account with Fabric access
- Install ODBC drivers matching your Python bitness
- Verify SQL endpoint permissions

#### 4. Run Development Server
```bash
streamlit run rag_fabric_app.py
```

## 🏗️ Architecture Overview

### Modular Design

The application follows a clean modular architecture with clear separation of concerns:

```
fabric-rag-app/
├── rag_fabric_app.py          # Main application (UI + orchestration)
├── auth.py                    # Authentication module
├── config/                    # Configuration files
│   ├── queries.py           # GraphQL queries
│   ├── table_schemas.py     # Table schemas
│   └── safe_builtins.py    # Safe builtins for code execution
├── utils/                    # Utility modules
│   ├── rag_utils.py         # RAG utility functions (RAGUtils class)
│   └── fine_tune_embedding_model.py # Fine-tuning utility
└── docs/                     # Documentation
```

### Core Components

#### 1. Main Application (`rag_fabric_app.py`)
- **Purpose**: Streamlit UI and application orchestration
- **Responsibilities**:
  - User interface and interaction
  - Authentication flow management
  - Data fetching and processing coordination
  - Code execution and result display
  - Error handling and user feedback

#### 2. Authentication Module (`auth.py`)
- **Purpose**: Azure AD authentication and session management
- **Responsibilities**:
  - Azure Interactive Browser Credential setup
  - Session state management
  - Authentication flow control
  - Token management and refresh

#### 3. RAGUtils Class (`utils/rag_utils.py`)
- **Purpose**: Centralized utility functions for RAG operations
- **Design**: Static methods for easy access and testing
- **Key Methods**:
  ```python
  class RAGUtils:
      @staticmethod
      def get_fabric_engine()  # Azure SQL connection
      @staticmethod
      def embed_question(question, model)  # Question embedding
      @staticmethod
      def vector_search(question_embedding, embeddings_matrix)  # FAISS search
      @staticmethod
      def validate_code_uses_actual_names(code, available_dfs, available_columns)  # Code validation
      @staticmethod
      def is_code_safe(code)  # Security validation
      @staticmethod
      def prepare_comprehensive_context(dfs, df1_context, user_question)  # Context preparation
      @staticmethod
      def create_intelligent_prompt(user_question, comprehensive_context)  # Prompt generation
  ```

#### 4. Configuration Files (`config/`)
- **queries.py**: GraphQL queries for Fabric data access
- **table_schemas.py**: Column type definitions for data processing
- **safe_builtins.py**: Safe Python builtins for code execution

## 🔧 Development Workflow

### Code Organization

#### Adding New Features
1. **Identify the appropriate module** for your feature
2. **Follow the modular structure** - utilities go in RAGUtils, config goes in config/
3. **Add proper error handling** and validation
4. **Update documentation** in the relevant docs files
5. **Add tests** if applicable

#### Example: Adding a New RAG Utility
```python
# In utils/rag_utils.py
class RAGUtils:
    @staticmethod
    def new_utility_function(param1, param2):
        """
        Description of what this function does.
        
        Args:
            param1: Description of param1
            param2: Description of param2
            
        Returns:
            Description of return value
        """
        try:
            # Implementation
            result = some_operation(param1, param2)
            return result
        except Exception as e:
            # Proper error handling
            raise RuntimeError(f"Error in new_utility_function: {e}")
```

#### Example: Adding New Configuration
```python
# In config/queries.py
QUERIES = {
    "existing_table": "existing_query",
    "new_table": """
        query($first: Int!) {
            newTable(first: $first) {
                items {
                    COLUMN1
                    COLUMN2
                    COLUMN3
                }
            }
        }
    """
}
```

### Testing Strategy

#### Manual Testing
1. **Unit Testing**: Test individual functions in isolation
2. **Integration Testing**: Test the full RAG pipeline
3. **UI Testing**: Test user interactions and flows
4. **Error Testing**: Test error conditions and edge cases

#### Test Examples
```python
# Test RAGUtils methods
def test_embed_question():
    model = RAGUtils.get_embedding_model()
    embedding = RAGUtils.embed_question("test question", model)
    assert isinstance(embedding, np.ndarray)
    assert embedding.shape[0] > 0

def test_validate_code_uses_actual_names():
    code = "df1['COLUMN'].mean()"
    available_dfs = ['df1']
    available_columns = ['COLUMN']
    is_valid, message = RAGUtils.validate_code_uses_actual_names(
        code, available_dfs, available_columns, "test"
    )
    assert is_valid == True
```

### Debugging

#### Debug Mode
Enable debug mode in `.env`:
```bash
APP_DEBUG=True
```

#### Logging
- **Audit Log**: `rag_audit_log.csv` tracks all RAG activities
- **Console Output**: Detailed error messages and stack traces
- **Streamlit Debug**: Use `st.write()` for debugging UI components

#### Common Debugging Techniques
```python
# Debug data flow
st.write("Debug: DataFrames loaded:", list(dfs.keys()))
st.write("Debug: df1 shape:", df1.shape if df1 is not None else "None")

# Debug embeddings
st.write("Debug: Embedding shape:", q_emb.shape)

# Debug code generation
st.write("Debug: Generated code:", pandas_code)
```

## 🔒 Security Considerations

### Code Execution Safety
- **Sandboxed Environment**: Limited builtins and resource constraints
- **Code Validation**: Check for forbidden keywords and invalid references
- **Input Validation**: Comprehensive validation of user inputs
- **Error Handling**: Graceful failure without data exposure

### Authentication Security
- **Azure AD Integration**: Secure authentication to Microsoft services
- **Token Management**: Automatic token refresh and caching
- **Session Management**: Proper session state handling

### Data Protection
- **Environment Variables**: Secure credential management
- **Audit Logging**: Track all activities for security and improvement
- **Error Handling**: Graceful failure without data exposure

## 📊 Performance Optimization

### Caching Strategy
- **Embeddings Cache**: Cache embeddings for repeated questions
- **Data Cache**: Cache frequently accessed data
- **Model Cache**: Cache loaded models to avoid reloading

### Memory Management
- **Sampling**: Use sampling for large datasets (> 500 rows)
- **Lazy Loading**: Load data only when needed
- **Cleanup**: Proper cleanup of large objects

### Query Optimization
- **Batch Processing**: Process data in batches
- **Index Usage**: Use FAISS indices for fast vector search
- **Parallel Processing**: Use concurrent operations where possible

## 🧪 Testing Guidelines

### Unit Tests
```python
import pytest
from utils.rag_utils import RAGUtils

def test_parse_embedding():
    # Test string embedding
    embedding_str = "[0.1, 0.2, 0.3]"
    result = RAGUtils.parse_embedding(embedding_str)
    assert isinstance(result, np.ndarray)
    assert len(result) == 3

def test_is_code_safe():
    # Test safe code
    safe_code = "df1['COLUMN'].mean()"
    is_safe, keyword = RAGUtils.is_code_safe(safe_code)
    assert is_safe == True

    # Test unsafe code
    unsafe_code = "import os; os.system('rm -rf /')"
    is_safe, keyword = RAGUtils.is_code_safe(unsafe_code)
    assert is_safe == False
```

### Integration Tests
```python
def test_full_rag_pipeline():
    # Test complete RAG pipeline
    question = "What is the average energy rating?"
    # ... implement full pipeline test
```

## 📝 Code Standards

### Python Style Guide
- **PEP 8**: Follow Python style guidelines
- **Docstrings**: Add comprehensive docstrings to all functions
- **Type Hints**: Use type hints where appropriate
- **Error Handling**: Include proper try/except blocks

### Documentation Standards
- **Function Documentation**: Document all functions with docstrings
- **Module Documentation**: Add module-level docstrings
- **README Updates**: Update documentation for new features
- **Code Comments**: Add comments for complex logic

### Git Workflow
1. **Feature Branches**: Create feature branches for new development
2. **Commit Messages**: Use descriptive commit messages
3. **Pull Requests**: Create PRs for code review
4. **Testing**: Ensure all tests pass before merging

## 🔄 Deployment

### Development Deployment
```bash
# Run locally
streamlit run rag_fabric_app.py

# Run with specific port
streamlit run rag_fabric_app.py --server.port 8502
```

### Production Deployment
See [DEPLOYMENT_PLAN.md](DEPLOYMENT_PLAN.md) for detailed production deployment instructions.

## 🐛 Common Development Issues

### Import Errors
- **Issue**: ModuleNotFoundError for config or utils
- **Solution**: Ensure __init__.py files exist in all directories

### Authentication Issues
- **Issue**: Azure authentication failures
- **Solution**: Check Azure AD setup and permissions

### Memory Issues
- **Issue**: Out of memory errors with large datasets
- **Solution**: Enable sampling and reduce batch sizes

### Code Generation Issues
- **Issue**: Generated code fails to execute
- **Solution**: Check data types and use the retry mechanism

## 🤝 Contributing

### Before Contributing
1. **Read Documentation**: Review all documentation in the docs/ folder
2. **Understand Architecture**: Familiarize yourself with the modular structure
3. **Test Locally**: Ensure your changes work in your development environment
4. **Follow Standards**: Adhere to code standards and documentation guidelines

### Contribution Process
1. **Fork Repository**: Create your own fork
2. **Create Feature Branch**: `git checkout -b feature/your-feature`
3. **Make Changes**: Implement your feature with proper testing
4. **Update Documentation**: Update relevant documentation files
5. **Submit Pull Request**: Create PR with detailed description

### Code Review Checklist
- [ ] Code follows PEP 8 style guidelines
- [ ] Functions have proper docstrings
- [ ] Error handling is implemented
- [ ] Tests are included (if applicable)
- [ ] Documentation is updated
- [ ] No security vulnerabilities introduced

## 📚 Additional Resources

### Documentation
- [Main Documentation](README.md)
- [Troubleshooting Guide](TROUBLESHOOTING.md)
- [Architecture Documentation](ARCHITECTURE_DIAGRAM.md)
- [Process Flow Documentation](PROCESS_FLOW_DIAGRAM.md)

### External Resources
- [Streamlit Documentation](https://docs.streamlit.io/)
- [Microsoft Fabric Documentation](https://learn.microsoft.com/en-us/fabric/)
- [OpenAI API Documentation](https://platform.openai.com/docs)
- [FAISS Documentation](https://github.com/facebookresearch/faiss)

---

**Last Updated**: July 2025  
**Version**: 2.1.0 