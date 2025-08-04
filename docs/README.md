# Fabric RAG (Retrieval-Augmented Generation) QA App - Complete Documentation

A comprehensive Streamlit-based application for Microsoft Fabric datasets featuring semantic search, SQL Editor, and AI-powered data analysis with modular architecture.

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [Architecture](#architecture)
- [Fine-Tuning Workflow](#fine-tuning-workflow)
- [Troubleshooting](#troubleshooting)
- [API Reference](#api-reference)

## 🎯 Overview

The Fabric RAG QA App is a sophisticated data analysis platform that combines:

- **Retrieval-Augmented Generation (RAG)**: Semantic search with AI-powered analysis
- **Microsoft Fabric Integration**: Direct access to Fabric datasets via SQL endpoint
- **Modular Architecture**: Clean separation of concerns with utility classes and config files
- **Fine-Tuning Capabilities**: Custom embedding model training for domain-specific data
- **Code Safety**: Sandboxed execution environment with comprehensive validation

## ✨ Features

### 🔍 RAG QA System
- **Semantic Search**: FAISS-based vector search for relevant data retrieval
- **AI-Powered Analysis**: OpenAI integration for intelligent data exploration
- **Code Generation**: Automatic Python code generation with safety validation
- **Visualization**: Matplotlib-based chart generation
- **Error Recovery**: Intelligent retry mechanisms with improved prompts
- **Audit Logging**: Comprehensive activity tracking for analysis and improvement

### 🗄️ SQL Editor
- **Direct Database Access**: Microsoft Fabric SQL endpoint integration
- **Azure Authentication**: Secure credential management via Interactive Browser Credential
- **CSV Export**: Data export capabilities for further analysis
- **Query Validation**: SQL syntax and safety checks
- **Real-time Results**: Immediate query execution and result display

### 🎛️ Fine-Tuning Workflow
- **Model Training**: Custom embedding model fine-tuning using exported Q&A pairs
- **CSV Export**: Q&A/code pair export from SQL Editor for training data
- **Domain Adaptation**: Specialized models for specific datasets and use cases
- **Model Selection**: Multiple embedding model support (default and fine-tuned)

### 🏗️ Modular Architecture
- **Utility Classes**: `RAGUtils` class with static methods for all core functions
- **Configuration Files**: Separate config files for queries, schemas, and safe builtins
- **Authentication Module**: Dedicated `auth.py` for Azure authentication
- **Clean Separation**: Clear separation between UI, business logic, and utilities

## 🚀 Installation

### Prerequisites
- Python 3.8+
- Microsoft Fabric access with proper permissions
- OpenAI API key
- ODBC Driver 17/18 for SQL Server
- Azure AD account with Fabric access

### Step-by-Step Setup

#### 1. Clone and Install
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

# Edit .env with your credentials
OPENAI_API_KEY=your_openai_api_key
DEFAULT_MODEL=gpt-3.5-turbo
MAX_TOKENS=600
TEMPERATURE=0.2
```

#### 3. Azure Setup
- Ensure you have Azure AD account with Fabric access
- Install ODBC drivers (17 or 18 for SQL Server)
- Verify SQL endpoint permissions

#### 4. Run Application
```bash
streamlit run rag_fabric_app.py
```

## ⚙️ Configuration

### Environment Variables

#### Required Variables
```bash
OPENAI_API_KEY=your_openai_api_key
DEFAULT_MODEL=gpt-3.5-turbo
MAX_TOKENS=600
TEMPERATURE=0.2
```

#### Optional Variables
```bash
APP_DEBUG=False
APP_ENV=development
MAX_SAMPLE_SIZE=1000
REPORT_TITLE=Energy Report QA Analysis
REPORT_FONT=Arial
REPORT_FONT_SIZE=16
```

### Project Structure
```
fabric-rag-app/
├── rag_fabric_app.py          # Main application
├── auth.py                    # Authentication module
├── requirements.txt           # Dependencies
├── .env                      # Environment variables
├── README.md                 # Main overview
├── docs/                     # Documentation
│   ├── README.md            # This file
│   ├── DEVELOPMENT_GUIDE.md # Development guide
│   ├── TROUBLESHOOTING.md   # Troubleshooting guide
│   ├── DEPLOYMENT_PLAN.md   # Deployment guide
│   ├── ARCHITECTURE_DIAGRAM.md # Architecture docs
│   └── PROCESS_FLOW_DIAGRAM.md # Process flow docs
├── config/                   # Configuration files
│   ├── queries.py           # GraphQL queries
│   ├── table_schemas.py     # Table schemas
│   └── safe_builtins.py    # Safe builtins for code execution
├── utils/                    # Utility modules
│   ├── rag_utils.py         # RAG utility functions
│   └── fine_tune_embedding_model.py # Fine-tuning utility
└── models/                   # Fine-tuned models
```

## 📖 Usage

### Basic Workflow

#### 1. Authentication
- Launch the application
- Sign in with your Azure credentials
- Verify connection to Fabric SQL endpoint

#### 2. Data Selection
- Choose relevant tables from the dropdown
- Set date range for data filtering
- Configure batch size for data fetching
- Enable auto-fetch for intelligent table selection

#### 3. Question Input
- Enter natural language questions about your data
- Reference specific columns or tables (e.g., df1, df2, column names)
- Use visualization keywords for charts (plot, chart, visualize, etc.)

#### 4. Analysis
- Review AI-generated reasoning and code
- Execute generated code safely
- View results as tables, charts, or summaries
- Download results as CSV files

### Advanced Features

#### Auto-Fetch Tables
- Automatically detects relevant tables based on question content
- Uses keyword matching for table selection
- Fetches data on-demand for better performance

#### Fine-Tuning Workflow
1. **Export Q&A Pairs**: Use SQL Editor to export relevant Q&A/code pairs as CSV
2. **Train Model**: Run fine-tuning utility with exported data
3. **Select Model**: Choose fine-tuned model from sidebar
4. **Test Results**: Compare performance between default and fine-tuned models

#### SQL Editor
- Direct access to Fabric SQL endpoint
- Execute custom SQL queries
- Export results as CSV files
- Real-time query validation and error handling

### Example Queries

#### Data Analysis
- "How do offices differ from restaurants in energy performance?"
- "What are the energy trends in Scottish homes?"
- "Compare domestic vs non-domestic energy ratings"

#### Visualizations
- "Visualize how different properties have different floor sizes"
- "Plot the distribution of energy ratings by property type"
- "Show the trend of building emissions over time"

#### Complex Analysis
- "Find properties with high energy consumption and suggest improvements"
- "Analyze the correlation between floor area and energy performance"
- "Identify outliers in the energy performance data"

## 🏗️ Architecture

### Modular Design

#### Core Components
- **Main App** (`rag_fabric_app.py`): Streamlit UI and orchestration
- **Authentication** (`auth.py`): Azure AD integration
- **Utilities** (`utils/rag_utils.py`): RAG utility functions as static methods
- **Configuration** (`config/`): Queries, schemas, and safe builtins

#### Key Classes and Functions

##### RAGUtils Class
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

##### Configuration Files
- **queries.py**: GraphQL queries for Fabric data access
- **table_schemas.py**: Column type definitions for data processing
- **safe_builtins.py**: Safe Python builtins for code execution

### Data Flow

#### RAG Process
1. **Question Embedding**: Convert user question to vector representation
2. **Semantic Search**: Use FAISS to find relevant data
3. **Context Preparation**: Create comprehensive context from multiple sources
4. **Code Generation**: Generate Python code with OpenAI
5. **Validation**: Check code safety and column references
6. **Execution**: Execute code in sandboxed environment
7. **Results**: Display tables, charts, and summaries

#### Security Measures
- **Code Validation**: Check for forbidden keywords and invalid references
- **Sandboxed Execution**: Limited builtins and resource constraints
- **Input Validation**: Comprehensive validation of user inputs
- **Audit Logging**: Track all activities for security and improvement

## 🔬 Fine-Tuning Workflow

### Overview
The fine-tuning workflow allows you to create custom embedding models specialized for your specific datasets and use cases.

### Step-by-Step Process

#### 1. Export Training Data
```bash
# Use SQL Editor to export Q&A/code pairs
# Save as CSV file with columns: question, code, result
```

#### 2. Run Fine-Tuning
```bash
python utils/fine_tune_embedding_model.py \
    --csv_path your_exported_data.csv \
    --base_model all-MiniLM-L6-v2 \
    --output_dir models/your_model_name
```

#### 3. Model Selection
- Fine-tuned models appear in the sidebar
- Select your custom model for RAG QA
- Compare results with default model

#### 4. Iterative Improvement
- Export more Q&A pairs based on performance
- Retrain model with expanded dataset
- Test and validate improvements

### Fine-Tuning Configuration
```python
# Training parameters
BATCH_SIZE = 16
LEARNING_RATE = 2e-5
NUM_EPOCHS = 3
WARMUP_STEPS = 100
```

## 🧑‍💻 Running and Fine-Tuning Deepseek Coder LLM Locally with GPU (CUDA)

### Prerequisites
- Windows 10/11 with NVIDIA GPU (e.g., RTX 4070)
- Python 3.8+
- CUDA Toolkit (11.8 or 12.1)
- cuDNN (matching your CUDA version)
- Latest NVIDIA drivers

### Environment Setup
1. Clone the repository and enter the directory:
   ```sh
   git clone <repository-url>
   cd fabric-rag-app
   ```
2. Create and activate a virtual environment:
   ```sh
   python -m venv venv
   venv\Scripts\activate  # (Windows)
   ```
3. Install PyTorch with CUDA support:
   ```sh
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
   ```
4. Install the rest of the dependencies:
   ```sh
   pip install -r requirements.txt
   ```

### Fine-Tuning Deepseek Coder LLM (1.3B) on GPU
1. Place your Q&A CSV (e.g., `epcNonDomesticScotlandQA.csv`) in the `downloads/` directory.
2. Ensure `MODEL_NAME = "deepseek-ai/deepseek-coder-1.3b-instruct"` and `USE_4BIT = False` in `utils/finetune_deepseekcode.py`.
3. Run the fine-tuning script:
   ```sh
   python utils/finetune_deepseekcode.py
   ```
   The fine-tuned model will be saved in `models/finetuned-deepseek-coder/`.

### Troubleshooting
- Check GPU availability:
  ```python
  import torch
  print(torch.cuda.is_available())
  print(torch.cuda.get_device_name(0))
  ```
- Out of memory? Use a smaller batch size or the 1.3B model.
- bitsandbytes errors? Ignore on Windows; quantization is not supported.

## 🐛 Troubleshooting

### Common Issues

#### ODBC/pyodbc Driver Errors
- **Symptom**: ImportError, driver not found, or authentication errors
- **Solution**: Ensure correct ODBC driver (17 or 18) is installed and matches Python bitness

#### Fine-Tuning Issues
- **Symptom**: Errors when running fine-tuning utility
- **Solution**: Ensure all dependencies are installed (`accelerate`, `datasets`, `torch`)

#### Environment Variable Issues
- **Symptom**: Authentication or API errors
- **Solution**: Double-check `.env` file for all required keys

#### NameError Issues
- **Symptom**: `NameError: name 'function_name' is not defined`
- **Solution**: All utility functions are now in `RAGUtils` class as static methods

### Debug Mode
```bash
# Enable debug mode in .env
APP_DEBUG=True
```

### Log Files
- **Audit Log**: `rag_audit_log.csv` - Tracks all RAG activities
- **Error Logs**: Check console output for detailed error messages

## 📚 API Reference

### RAGUtils Class Methods

#### Authentication & Connection
```python
RAGUtils.get_fabric_engine()  # Returns SQLAlchemy engine
RAGUtils.get_embedding_model(model_path=None)  # Returns SentenceTransformer
```

#### Embedding & Search
```python
RAGUtils.embed_question(question, model)  # Returns numpy array
RAGUtils.vector_search(question_embedding, embeddings_matrix, top_n=100)  # Returns indices, similarities
RAGUtils.parse_embedding(embedding_str)  # Converts string to numpy array
```

#### Data Fetching
```python
RAGUtils.fetch_embeddings(engine, date_range=None)  # Returns DataFrame
RAGUtils.fetch_raw_data(engine, keys, date_range=None)  # Returns DataFrame
RAGUtils.fetch_tables_for_question(user_question, selected_tables, batch_size, start_date, end_date, QUERIES, get_credential)  # Returns DataFrames
```

#### Code Generation & Validation
```python
RAGUtils.prepare_comprehensive_context(dfs, df1_context, user_question)  # Returns context string
RAGUtils.create_intelligent_prompt(user_question, comprehensive_context)  # Returns prompt string
RAGUtils.validate_code_uses_actual_names(code, available_dfs, available_columns, user_question)  # Returns (is_valid, message)
RAGUtils.is_code_safe(code)  # Returns (is_safe, keyword)
RAGUtils.clean_code(code)  # Returns cleaned code string
```

#### Utility Functions
```python
RAGUtils.process_table_types(df, table_name, table_schemas)  # Returns processed DataFrame
RAGUtils.summarize_dataframe(df)  # Returns summary string
RAGUtils.extract_keywords(question)  # Returns keyword list
RAGUtils.filter_df_by_keywords(df, keywords)  # Returns filtered DataFrame
RAGUtils.is_visualization_request(question)  # Returns boolean
```

### Configuration Files

#### queries.py
```python
QUERIES = {
    "epcNonDomesticScotlands": "GraphQL query string",
    "epcDomesticEngWales": "GraphQL query string",
    # ... more queries
}
```

#### table_schemas.py
```python
TABLE_SCHEMAS = {
    "table_name": {
        "numeric": ["column1", "column2"],
        "categorical": ["column3", "column4"],
        "datetime": ["column5"]
    }
}
```

#### safe_builtins.py
```python
SAFE_BUILTINS = {
    # Safe Python builtins for code execution
    'abs': abs,
    'all': all,
    'any': any,
    # ... more safe functions
}
```

## 🤝 Contributing

See [Development Guide](DEVELOPMENT_GUIDE.md) for detailed contribution guidelines.

## 📄 License

This project is licensed under the MIT License.

## 🆘 Support

- **Documentation**: Check this folder for detailed guides
- **Troubleshooting**: See [TROUBLESHOOTING.md](TROUBLESHOOTING.md) for common issues
- **Issues**: Create an issue in the repository for bugs or feature requests

---

**Version**: 2.1.0  
**Last Updated**: July 2025  
**Status**: Production Ready 