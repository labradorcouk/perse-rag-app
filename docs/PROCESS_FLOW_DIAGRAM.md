# Fabric RAG Application - Process Flow Diagram

## 🔄 Detailed Process Flow Overview

### **Main Application Flow**

```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                              APPLICATION STARTUP                                   │
├─────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                     │
│  1. Load Environment Variables                                                     │
│  2. Initialize Streamlit Session State                                            │
│  3. Setup Authentication (Azure AD)                                               │
│  4. Initialize Caching System                                                     │
│  5. Load Configuration Settings                                                   │
│                                                                                     │
└─────────────────────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                              SIDEBAR CONFIGURATION                                 │
├─────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                     │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐                │
│  │   Data Source   │    │   Batch Size    │    │   Date Range    │                │
│  │   Selection     │    │   Configuration │    │   Filter        │                │
│  │                 │    │                 │    │                 │                │
│  │ • Table Picker  │    │ • Min: 100      │    │ • Start Date    │                │
│  │ • Auto-Fetch    │    │ • Max: 5000     │    │ • End Date      │                │
│  │ • Manual Select │    │ • Default: 1000 │    │ • Apply Filter  │                │
│  └─────────────────┘    └─────────────────┘    └─────────────────┘                │
│                                                                                     │
└─────────────────────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                              NAVIGATION SELECTION                                  │
├─────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                     │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐                │
│  │   Data Preview  │    │   SQL Editor    │    │   RAG Analysis  │                │
│  │                 │    │                 │    │                 │                │
│  │ • Table Display │    │ • Query Builder │    │ • Question Input│                │
│  │ • Schema View   │    │ • Results Grid  │    │ • AI Analysis   │                │
│  │ • Data Stats    │    │ • Export Data   │    │ • Code Gen      │                │
│  └─────────────────┘    └─────────────────┘    └─────────────────┘                │
│                                                                                     │
└─────────────────────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                              DECISION POINT: NAVIGATION                            │
├─────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                     │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐                │
│  │   Data Preview  │    │   SQL Editor    │    │   RAG Analysis  │                │
│  │   Selected?     │    │   Selected?     │    │   Selected?     │                │
│  │                 │    │                 │    │                 │                │
│  │       YES       │    │       YES       │    │       YES       │                │
│  │                 │    │                 │    │                 │                │
│  └─────────────────┘    └─────────────────┘    └─────────────────┘                │
│           │                       │                       │                        │
│           ▼                       ▼                       ▼                        │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐                │
│  │   Preview Flow  │    │   SQL Editor    │    │   RAG Pipeline  │                │
│  │                 │    │   Flow          │    │   Flow          │                │
│  └─────────────────┘    └─────────────────┘    └─────────────────┘                │
└─────────────────────────────────────────────────────────────────────────────────────┘
```

## 📊 Data Preview Flow

```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                              DATA PREVIEW FLOW                                     │
├─────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                     │
│  1. Check if tables are selected                                                   │
│     │                                                                               │
│     ├─ NO: Show message "Please select tables to preview"                         │
│     │   └─ Return to sidebar                                                      │
│     │                                                                               │
│     └─ YES: Continue to data fetching                                             │
│                                                                                     │
│  2. Fetch data from Microsoft Fabric GraphQL API                                  │
│     │                                                                               │
│     ├─ SUCCESS: Store data in session state                                       │
│     │   └─ Continue to display                                                    │
│     │                                                                               │
│     └─ ERROR: Show error message and retry option                                 │
│                                                                                     │
│  3. Display data preview                                                           │
│     │                                                                               │
│     ├─ Show table selection                                                        │
│     ├─ Display schema information                                                  │
│     ├─ Show data statistics                                                        │
│     └─ Display sample data (first 100 rows)                                       │
│                                                                                     │
│  4. User interactions                                                              │
│     │                                                                               │
│     ├─ Change table selection                                                      │
│     ├─ Adjust batch size                                                           │
│     ├─ Apply date filters                                                          │
│     └─ Export data                                                                 │
│                                                                                     │
└─────────────────────────────────────────────────────────────────────────────────────┘
```

## 🔍 SQL Editor Flow

```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                              SQL EDITOR FLOW                                       │
├─────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                     │
│  1. Initialize SQL Editor                                                          │
│     │                                                                               │
│     ├─ Load connection settings                                                    │
│     ├─ Setup ODBC driver selection                                                │
│     └─ Initialize query history                                                    │
│                                                                                     │
│  2. User enters SQL query                                                          │
│     │                                                                               │
│     ├─ Validate SQL syntax                                                        │
│     │   ├─ VALID: Continue to execution                                           │
│     │   └─ INVALID: Show syntax error and highlight issues                        │
│     │                                                                               │
│     └─ Check for dangerous operations (DROP, DELETE, etc.)                        │
│         ├─ SAFE: Continue to execution                                             │
│         └─ DANGEROUS: Show warning and require confirmation                        │
│                                                                                     │
│  3. Execute SQL query                                                              │
│     │                                                                               │
│     ├─ Connect to Fabric SQL endpoint                                              │
│     │   ├─ SUCCESS: Execute query                                                  │
│     │   └─ ERROR: Show connection error and retry options                         │
│     │                                                                               │
│     ├─ Execute query                                                               │
│     │   ├─ SUCCESS: Display results                                                │
│     │   └─ ERROR: Show execution error and suggestions                            │
│     │                                                                               │
│     └─ Handle large result sets                                                   │
│         ├─ SMALL: Display full results                                            │
│         └─ LARGE: Show pagination and download options                            │
│                                                                                     │
│  4. Display results                                                               │
│     │                                                                               │
│     ├─ Show result count                                                          │
│     ├─ Display data grid                                                          │
│     ├─ Show execution time                                                        │
│     └─ Provide export options                                                     │
│                                                                                     │
└─────────────────────────────────────────────────────────────────────────────────────┘
```

## 🤖 RAG Analysis Flow

```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                              RAG ANALYSIS FLOW                                     │
├─────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                     │
│  1. User enters question                                                           │
│     │                                                                               │
│     ├─ Validate question format                                                    │
│     │   ├─ VALID: Continue to processing                                          │
│     │   └─ INVALID: Show format guidance                                          │
│     │                                                                               │
│     └─ Check question length                                                       │
│         ├─ ACCEPTABLE: Continue                                                   │
│         └─ TOO LONG: Suggest breaking into smaller questions                      │
│                                                                                     │
│  2. Intelligent table selection                                                    │
│     │                                                                               │
│     ├─ Analyze question keywords                                                  │
│     ├─ Match keywords to available tables                                         │
│     ├─ Auto-select relevant tables                                                │
│     └─ Allow manual override                                                      │
│                                                                                     │
│  3. Status bar updates: "Step 1: Embedding user question"                        │
│     │                                                                               │
│     ├─ Generate question embedding using OpenAI API                               │
│     │   ├─ SUCCESS: Continue to data fetching                                    │
│     │   └─ ERROR: Show embedding error and retry options                         │
│     │                                                                               │
│  4. Status bar updates: "Step 2: Connecting to Fabric SQL endpoint"               │
│     │                                                                               │
│     ├─ Establish connection to Fabric SQL                                         │
│     │   ├─ SUCCESS: Continue to vector search                                    │
│     │   └─ ERROR: Show connection error and authentication issues                 │
│     │                                                                               │
│  5. Status bar updates: "Step 3: Running semantic vector search (hybrid RAG)"    │
│     │                                                                               │
│     ├─ Fetch embeddings from database                                            │
│     │   ├─ SUCCESS: Perform vector search                                        │
│     │   └─ ERROR: Show data fetching error                                       │
│     │                                                                               │
│     ├─ Perform FAISS vector search                                               │
│     │   ├─ SUCCESS: Get top-N similar results                                    │
│     │   └─ ERROR: Show search error                                              │
│     │                                                                               │
│     ├─ Fetch raw data for top results                                            │
│     │   ├─ SUCCESS: Prepare context for LLM                                      │
│     │   └─ ERROR: Show data retrieval error                                      │
│     │                                                                               │
│  6. Status bar updates: "Step 4: Preparing LLM context and prompt"               │
│     │                                                                               │
│     ├─ Prepare schema information                                                │
│     ├─ Create sample data CSV                                                    │
│     ├─ Build context string                                                       │
│     └─ Construct intelligent prompt                                              │
│                                                                                     │
│  7. Status bar updates: "Step 5: Generating reasoning with LLM"                  │
│     │                                                                               │
│     ├─ Send reasoning prompt to OpenAI API                                       │
│     │   ├─ SUCCESS: Get AI reasoning                                             │
│     │   └─ ERROR: Show API error and retry options                               │
│     │                                                                               │
│  8. Status bar updates: "Step 6: Generating code with LLM"                       │
│     │                                                                               │
│     ├─ Send code generation prompt to OpenAI API                                 │
│     │   ├─ SUCCESS: Get generated code                                           │
│     │   └─ ERROR: Show code generation error                                     │
│     │                                                                               │
│  9. Status bar updates: "Step 7: Executing generated code"                       │
│     │                                                                               │
│     ├─ Validate generated code                                                   │
│     │   ├─ VALID: Execute code safely                                            │
│     │   └─ INVALID: Show validation error                                        │
│     │                                                                               │
│     ├─ Execute code in safe environment                                          │
│     │   ├─ SUCCESS: Display results and insights                                 │
│     │   └─ ERROR: Show execution error and debugging info                        │
│     │                                                                               │
│  10. Display results                                                              │
│      │                                                                              │
│      ├─ Show AI reasoning                                                        │
│      ├─ Display generated code                                                   │
│      ├─ Show execution results                                                   │
│      ├─ Provide visualizations                                                   │
│      └─ Offer export options                                                     │
│                                                                                     │
└─────────────────────────────────────────────────────────────────────────────────────┘
```

## 🔄 Error Handling Flow

```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                              ERROR HANDLING FLOW                                   │
├─────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                     │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐                │
│  │   Error Occurs  │───▶│ Error Detection │───▶│ Error Classification│              │
│  │                 │    │                 │    │                 │                │
│  └─────────────────┘    └─────────────────┘    └─────────────────┘                │
│           │                       │                       │                        │
│           ▼                       ▼                       ▼                        │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐                │
│  │ Authentication  │    │ Data Type       │    │ API Rate Limit  │                │
│  │ Error           │    │ Error           │    │ Error           │                │
│  │                 │    │                 │    │                 │                │
│  │ • Token Expired │    │ • Non-numeric   │    │ • OpenAI Quota  │                │
│  │ • Invalid Creds │    │ • Type Mismatch │    │ • Fabric Limits │                │
│  │ • Permission    │    │ • Conversion    │    │ • Backoff Logic │                │
│  └─────────────────┘    └─────────────────┘    └─────────────────┘                │
│           │                       │                       │                        │
│           ▼                       ▼                       ▼                        │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐                │
│  │ Connection      │    │ Memory          │    │ Code Execution  │                │
│  │ Error           │    │ Error           │    │ Error           │                │
│  │                 │    │                 │    │                 │                │
│  │ • Network Issue │    │ • Out of Memory │    │ • Syntax Error  │                │
│  │ • Timeout       │    │ • Large Dataset │    │ • Runtime Error │                │
│  │ • DNS Failure   │    │ • Memory Leak   │    │ • Logic Error   │                │
│  └─────────────────┘    └─────────────────┘    └─────────────────┘                │
│           │                       │                       │                        │
│           ▼                       ▼                       ▼                        │
│  ┌─────────────────────────────────────────────────────────────────────────────────┐ │
│  │                              ERROR RESPONSE STRATEGY                           │ │
│  │                                                                                 │ │
│  │ • Show User-Friendly Message    • Provide Retry Options    • Log Error Details│ │
│  │ • Suggest Solutions             • Offer Alternative Paths   • Monitor Patterns │ │
│  └─────────────────────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────────────────┘
```

## 🔄 Decision Tree Flow

```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                              DECISION TREE FLOW                                    │
├─────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                     │
│  User Question                                                                      │
│       │                                                                             │
│       ▼                                                                             │
│  ┌─────────────────┐                                                               │
│  │ Question Type?  │                                                               │
│  └─────────────────┘                                                               │
│       │                                                                             │
│       ├─ Data Analysis ──▶ Data Analysis Flow                                     │
│       │                                                                             │
│       ├─ Comparison ─────▶ Comparison Flow                                        │
│       │                                                                             │
│       ├─ Filtering ──────▶ Filtering Flow                                         │
│       │                                                                             │
│       ├─ Aggregation ────▶ Aggregation Flow                                       │
│       │                                                                             │
│       └─ Other ─────────▶ General Analysis Flow                                   │
│                                                                                     │
│  Data Analysis Flow                                                                │
│       │                                                                             │
│       ├─ Check data types                                                          │
│       ├─ Identify numeric columns                                                  │
│       ├─ Perform statistical analysis                                              │
│       └─ Generate visualizations                                                   │
│                                                                                     │
│  Comparison Flow                                                                   │
│       │                                                                             │
│       ├─ Group by categorical columns                                             │
│       ├─ Calculate metrics for each group                                         │
│       ├─ Perform statistical tests                                                │
│       └─ Create comparison visualizations                                         │
│                                                                                     │
│  Filtering Flow                                                                    │
│       │                                                                             │
│       ├─ Apply date filters                                                        │
│       ├─ Apply categorical filters                                                 │
│       ├─ Apply numeric range filters                                               │
│       └─ Combine multiple filters                                                  │
│                                                                                     │
│  Aggregation Flow                                                                  │
│       │                                                                             │
│       ├─ Group by relevant columns                                                │
│       ├─ Calculate aggregations (sum, mean, count)                                │
│       ├─ Handle missing values                                                     │
│       └─ Present aggregated results                                               │
│                                                                                     │
└─────────────────────────────────────────────────────────────────────────────────────┘
```

## 🔄 Memory Management Flow

```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                              MEMORY MANAGEMENT FLOW                                │
├─────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                     │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐                │
│  │   Memory Check  │───▶│ Memory Usage    │───▶│ Memory Action   │                │
│  │                 │    │ Assessment      │    │ Decision        │                │
│  └─────────────────┘    └─────────────────┘    └─────────────────┘                │
│           │                       │                       │                        │
│           ▼                       ▼                       ▼                        │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐                │
│  │ LOW MEMORY      │    │ MEDIUM MEMORY   │    │ HIGH MEMORY     │                │
│  │ (< 50%)         │    │ (50-80%)        │    │ (> 80%)        │                │
│  │                 │    │                 │    │                 │                │
│  │ • Normal ops    │    │ • Monitor       │    │ • Force GC      │                │
│  │ • Cache freely  │    │ • Limit cache   │    │ • Clear cache   │                │
│  │ • No warnings   │    │ • Show warnings │    │ • Show alerts   │                │
│  └─────────────────┘    └─────────────────┘    └─────────────────┘                │
│           │                       │                       │                        │
│           ▼                       ▼                       ▼                        │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐                │
│  │ Continue Normal │    │ Optimize Usage  │    │ Emergency Mode  │                │
│  │ Operations      │    │                 │    │                 │                │
│  │                 │    │ • Reduce batch  │    │ • Stop new ops  │                │
│  │ • Full caching  │    │ • Clear old     │    │ • Clear all     │                │
│  │ • Large batches │    │ • Monitor       │    │ • User warning  │                │
│  │ • No limits     │    │ • Alert user    │    │ • Restart if    │                │
│  └─────────────────┘    └─────────────────┘    └─────────────────┘                │
└─────────────────────────────────────────────────────────────────────────────────────┘
```

## 🔄 Performance Optimization Flow

```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                              PERFORMANCE OPTIMIZATION FLOW                         │
├─────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                     │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐                │
│  │ Performance     │───▶│ Bottleneck      │───▶│ Optimization    │                │
│  │ Monitoring      │    │ Identification   │    │ Strategy        │                │
│  └─────────────────┘    └─────────────────┘    └─────────────────┘                │
│           │                       │                       │                        │
│           ▼                       ▼                       ▼                        │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐                │
│  │ Vector Search   │    │ Data Fetching   │    │ LLM Processing  │                │
│  │ Slow            │    │ Slow            │    │ Slow            │                │
│  │                 │    │                 │    │                 │                │
│  │ • Reduce top_n  │    │ • Increase batch │   │ • Reduce tokens │                │
│  │ • Use FAISS     │    │ • Cache results │   │ • Stream output │                │
│  │ • GPU accel     │    │ • Parallel fetch│   │ • Async calls   │                │
│  └─────────────────┘    └─────────────────┘    └─────────────────┘                │
│           │                       │                       │                        │
│           ▼                       ▼                       ▼                        │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐                │
│  │ Memory Usage    │    │ Network Latency │    │ User Experience │                │
│  │ High            │    │ High            │    │ Poor            │                │
│  │                 │    │                 │    │                 │                │
│  │ • Clear cache   │    │ • Retry logic   │    │ • Progress bar  │                │
│  │ • Batch process │    │ • Connection    │    │ • Status updates│                │
│  │ • Stream data   │    │ pooling        │    │ • Error handling│                │
│  └─────────────────┘    └─────────────────┘    └─────────────────┘                │
└─────────────────────────────────────────────────────────────────────────────────────┘
```

Fine-Tuning & Domain Adaptation Flow:
- Export Q&A/code pairs from SQL Editor as CSV.
- Fine-tune embedding model using the utility script and local CSV file.
- No direct SQL access in fine-tuning utility.
- Dependencies: accelerate, datasets (see requirements.txt).

Troubleshooting:
See TROUBLESHOOTING.md for a list of known issues, ODBC/pyodbc driver troubleshooting, and solutions for common errors.

## 🚀 What's New in v2.1
- Model, table, and date selection are now in the main RAG QA tab (not the sidebar).
- Data preview and RAG QA support pagination for large tables.
- Data fetching and preview steps show progress bars.
- Automatic sampling is used for large datasets to improve performance.
- LLM context always includes unique values for categorical columns and guidance for substring/case-insensitive filtering.
- See TROUBLESHOOTING.md for new error handling and performance tips.

---

**Version**: 2.0.0  
**Last Updated**: December 2024  
**Status**: Production Ready 