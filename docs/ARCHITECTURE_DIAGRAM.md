# Fabric RAG Application - Architecture Diagram

## 🏗️ System Architecture Overview

### **High-Level Architecture**

```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                              USER INTERFACE LAYER                                  │
├─────────────────────────────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐  ┌───────────────┐ │
│  │   Streamlit UI  │  │   SQL Editor    │  │   Data Preview  │  │ Status Bar    │ │
│  │                 │  │                 │  │                 │  │               │ │
│  │ • Question Input│  │ • Query Builder │  │ • Table Display │  │ • Progress    │ │
│  │ • Results Display│  │ • Syntax Check │  │ • Schema View   │  │ • Step Status │ │
│  │ • Code Execution│  │ • Results Grid  │  │ • Data Stats    │  │ • Error Info  │ │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘  └───────────────┘ │
└─────────────────────────────────────────────────────────────────────────────────────┘
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                            APPLICATION LOGIC LAYER                                 │
├─────────────────────────────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐  ┌───────────────┐ │
│  │   RAG Pipeline  │  │ FAISS Vector    │  │ Intelligent     │  │ Data Type     │ │
│  │                 │  │ Search Engine   │  │ Table Selection │  │ Handler       │ │
│  │ • Hybrid Search │  │ • Index Mgmt    │  │ • Keyword Match │  │ • Type Detect │ │
│  │ • Context Gen   │  │ • Similarity    │  │ • Auto-Fetch    │  │ • Safe Ops    │ │
│  │ • Code Gen      │  │ • Memory Opt    │  │ • Performance   │  │ • Conversion  │ │
│  │ • Error Handle  │  │ • GPU Support   │  │ • User Control  │  │ • Validation  │ │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘  └───────────────┘ │
└─────────────────────────────────────────────────────────────────────────────────────┘
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                              DATA ACCESS LAYER                                     │
├─────────────────────────────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐  ┌───────────────┐ │
│  │ Microsoft Fabric│  │   Azure AD      │  │   OpenAI API    │  │ Local Cache   │ │
│  │   GraphQL API   │  │ Authentication  │  │                 │  │               │ │
│  │                 │  │                 │  │ • GPT-3.5-turbo │  │ • Embeddings  │ │
│  │ • Data Fetch    │  │ • Token Mgmt    │  │ • Code Gen      │  │ • Results     │ │
│  │ • Schema Query  │  │ • Permissions   │  │ • Reasoning     │  │ • Metadata    │ │
│  │ • Batch Process │  │ • Security      │  │ • Context       │  │ • TTL Mgmt    │ │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘  └───────────────┘ │
└─────────────────────────────────────────────────────────────────────────────────────┘
```

## 🔄 Data Flow Architecture

### **Primary Data Flow**

```
User Question
     │
     ▼
┌─────────────────┐
│  Streamlit UI   │
│  (Input Handler) │
└─────────────────┘
     │
     ▼
┌─────────────────┐
│ Intelligent     │
│ Table Selection │
│ • Keyword Match │
│ • Auto-Fetch    │
└─────────────────┘
     │
     ▼
┌─────────────────┐
│ Microsoft Fabric│
│ GraphQL API     │
│ • Data Fetch    │
│ • Schema Query  │
└─────────────────┘
     │
     ▼
┌─────────────────┐
│ FAISS Vector    │
│ Search Engine   │
│ • Embedding Gen │
│ • Similarity    │
│ • Top-N Results │
└─────────────────┘
     │
     ▼
┌─────────────────┐
│ OpenAI API      │
│ • Context Prep  │
│ • Reasoning Gen │
│ • Code Gen      │
└─────────────────┘
     │
     ▼
┌─────────────────┐
│ Safe Code       │
│ Execution       │
│ • Validation    │
│ • Error Handle  │
│ • Results       │
└─────────────────┘
     │
     ▼
┌─────────────────┐
│ Results Display │
│ • Visualizations│
│ • Insights      │
│ • Code Output   │
└─────────────────┘
```

## 🧠 Component Interaction Diagram

### **RAG Pipeline Components**

```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                              RAG PIPELINE ARCHITECTURE                            │
├─────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                     │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐                │
│  │   User Input    │───▶│ Question Parser │───▶│ Keyword Extractor│                │
│  │                 │    │                 │    │                 │                │
│  └─────────────────┘    └─────────────────┘    └─────────────────┘                │
│           │                       │                       │                        │
│           ▼                       ▼                       ▼                        │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐                │
│  │ Table Selector  │    │ Embedding Gen   │    │ Context Builder │                │
│  │                 │    │                 │    │                 │                │
│  │ • Auto-Detect   │    │ • OpenAI API    │    │ • Schema Info   │                │
│  │ • Manual Override│   │ • Local Model   │    │ • Sample Data   │                │
│  │ • Performance   │    │ • Vector Store  │    │ • Query Context │                │
│  └─────────────────┘    └─────────────────┘    └─────────────────┘                │
│           │                       │                       │                        │
│           ▼                       ▼                       ▼                        │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐                │
│  │ Data Fetcher    │    │ FAISS Search    │    │ LLM Processor   │                │
│  │                 │    │                 │    │                 │                │
│  │ • GraphQL API   │    │ • Index Query   │    │ • Reasoning Gen │                │
│  │ • Batch Process │    │ • Similarity    │    │ • Code Gen      │                │
│  │ • Cache Mgmt    │    │ • Top-N Results │    │ • Error Handle  │                │
│  └─────────────────┘    └─────────────────┘    └─────────────────┘                │
│           │                       │                       │                        │
│           ▼                       ▼                       ▼                        │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐                │
│  │ Data Processor  │    │ Result Ranker   │    │ Code Executor   │                │
│  │                 │    │                 │    │                 │                │
│  │ • Type Handler  │    │ • Relevance     │    │ • Safe Exec     │                │
│  │ • Validation    │    │ • Confidence    │    │ • Error Catch   │                │
│  │ • Conversion    │    │ • Diversity     │    │ • Output Format │                │
│  └─────────────────┘    └─────────────────┘    └─────────────────┘                │
│           │                       │                       │                        │
│           ▼                       ▼                       ▼                        │
│  ┌─────────────────────────────────────────────────────────────────────────────────┐ │
│  │                              RESULTS AGGREGATOR                               │ │
│  │                                                                                 │ │
│  │ • Combine Results    • Format Output    • Generate Insights    • Error Report │ │
│  └─────────────────────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────────────────┘
```

## 🔐 Security Architecture

### **Authentication and Authorization Flow**

```
┌─────────────────┐
│   User Session  │
└─────────────────┘
         │
         ▼
┌─────────────────┐
│ Azure AD Auth   │
│ • Interactive   │
│ • Browser Login │
│ • Token Mgmt    │
└─────────────────┘
         │
         ▼
┌─────────────────┐
│ Token Validator │
│ • Expiry Check  │
│ • Scope Verify  │
│ • Refresh Logic │
└─────────────────┘
         │
         ▼
┌─────────────────┐
│ Fabric Access   │
│ • GraphQL Auth  │
│ • SQL Auth      │
│ • Permission    │
└─────────────────┘
         │
         ▼
┌─────────────────┐
│ API Rate Limit  │
│ • OpenAI Quota  │
│ • Fabric Limits │
│ • Backoff Logic │
└─────────────────┘
```

## 💾 Data Storage Architecture

### **Caching and Storage Strategy**

```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                              STORAGE ARCHITECTURE                                 │
├─────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                     │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐                │
│  │   Memory Cache  │    │   Disk Cache    │    │   Vector Store  │                │
│  │                 │    │                 │    │                 │                │
│  │ • Session Data  │    │ • Embeddings    │    │ • FAISS Index   │                │
│  │ • User State    │    │ • Query Results │    │ • Similarity    │                │
│  │ • Temp Results  │    │ • Metadata      │    │ • Search Cache  │                │
│  │ • TTL: 1 hour  │    │ • TTL: 24 hours│    │ • Persistent    │                │
│  └─────────────────┘    └─────────────────┘    └─────────────────┘                │
│           │                       │                       │                        │
│           ▼                       ▼                       ▼                        │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐                │
│  │   Cache Manager │    │   Storage API   │    │   Index Manager │                │
│  │                 │    │                 │    │                 │                │
│  │ • LRU Policy   │    │ • File System   │    │ • Build Index   │                │
│  │ • Eviction     │    │ • Compression   │    │ • Update Index  │                │
│  │ • Monitoring   │    │ • Backup        │    │ • Optimize      │                │
│  └─────────────────┘    └─────────────────┘    └─────────────────┘                │
│           │                       │                       │                        │
│           ▼                       ▼                       ▼                        │
│  ┌─────────────────────────────────────────────────────────────────────────────────┐ │
│  │                              DATA LIFECYCLE MANAGER                           │ │
│  │                                                                                 │ │
│  │ • Cache Invalidation    • Storage Cleanup    • Index Maintenance    • Backup  │ │
│  └─────────────────────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────────────────┘
```

## ⚡ Performance Architecture

### **Scalability and Optimization**

```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                              PERFORMANCE ARCHITECTURE                             │
├─────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                     │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐                │
│  │   Load Balancer │    │   Cache Layer   │    │   Batch Process │                │
│  │                 │    │                 │    │                 │                │
│  │ • Request Dist  │    │ • Redis/Memory  │    │ • Chunk Process │                │
│  │ • Health Check  │    │ • TTL Mgmt      │    │ • Parallel Exec │                │
│  │ • Failover      │    │ • Hit Ratio     │    │ • Memory Opt    │                │
│  └─────────────────┘    └─────────────────┘    └─────────────────┘                │
│           │                       │                       │                        │
│           ▼                       ▼                       ▼                        │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐                │
│  │   Worker Pool   │    │   Async Queue   │    │   Resource Mgmt │                │
│  │                 │    │                 │    │                 │                │
│  │ • Thread Pool   │    │ • Task Queue    │    │ • Memory Limit  │                │
│  │ • Process Pool  │    │ • Priority      │    │ • CPU Monitor   │                │
│  │ • Auto Scale    │    │ • Retry Logic   │    │ • GC Control    │                │
│  └─────────────────┘    └─────────────────┘    └─────────────────┘                │
│           │                       │                       │                        │
│           ▼                       ▼                       ▼                        │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐                │
│  │   Monitoring    │    │   Optimization  │    │   Auto Scaling  │                │
│  │                 │    │                 │    │                 │                │
│  │ • Metrics Coll  │    │ • Query Opt     │    │ • Load Based    │                │
│  │ • Alert System  │    │ • Index Opt     │    │ • Resource Based│                │
│  │ • Performance   │    │ • Cache Opt     │    │ • Demand Based  │                │
│  └─────────────────┘    └─────────────────┘    └─────────────────┘                │
└─────────────────────────────────────────────────────────────────────────────────────┘
```

## 🔧 Configuration Architecture

### **Environment and Settings Management**

```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                              CONFIGURATION ARCHITECTURE                           │
├─────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                     │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐                │
│  │   Environment   │    │   Feature Flags │    │   User Settings │                │
│  │   Variables     │    │                 │    │                 │                │
│  │                 │    │                 │    │                 │                │
│  │ • API Keys      │    │ • Auto-Fetch    │    │ • Batch Size    │                │
│  │ • Endpoints     │    │ • FAISS Enable  │    │ • Date Range    │                │
│  │ • Credentials   │    │ • Debug Mode    │    │ • Table Select  │                │
│  └─────────────────┘    └─────────────────┘    └─────────────────┘                │
│           │                       │                       │                        │
│           ▼                       ▼                       ▼                        │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐                │
│  │   Config Loader │    │   Validator     │    │   Defaults      │                │
│  │                 │    │                 │    │                 │                │
│  │ • .env File     │    │ • Type Check    │    │ • Fallback      │                │
│  │ • Env Vars      │    │ • Required      │    │ • Sensible      │                │
│  │ • Secrets Mgmt  │    │ • Format        │    │ • User Friendly │                │
│  └─────────────────┘    └─────────────────┘    └─────────────────┘                │
│           │                       │                       │                        │
│           ▼                       ▼                       ▼                        │
│  ┌─────────────────────────────────────────────────────────────────────────────────┐ │
│  │                              CONFIGURATION MANAGER                            │ │
│  │                                                                                 │ │
│  │ • Hot Reload    • Validation    • Defaults    • Environment Specific          │ │
│  └─────────────────────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────────────────┘
```

Fine-Tuning & Domain Adaptation:
- Q&A/code pairs are exported from the SQL Editor as CSV.
- Fine-tuning is performed using the utility script and local CSV files (no direct SQL access).
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