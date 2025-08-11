# MongoDB Schema Integration Guide

## 🎯 Overview

This guide explains how to use the new MongoDB Schema Integration system to prevent parsing issues and optimize token usage in RAG applications.

## 🚀 Why Use Schema Configuration?

### **Before (Current Issues)**
- ❌ JSON parsing errors due to complex MongoDB document structures
- ❌ Token limit exceeded errors from large, unstructured data
- ❌ Inconsistent column selection across collections
- ❌ Hard-coded optimization logic in application code
- ❌ Difficult to maintain and extend

### **After (With Schema Configuration)**
- ✅ No more parsing issues - schema is predefined
- ✅ Optimized token usage - only essential data included
- ✅ Consistent data handling - standardized across collections
- ✅ Easy maintenance - add new collections via config
- ✅ Future-proof - schema evolves without code changes

## 📁 File Structure

```
config/
├── mongodb_schema_config.yaml          # Schema configuration
└── rag_tables_config.yaml             # Existing table config

utils/
├── mongodb_schema_manager.py          # Schema management utility
├── mongodb_utils.py                   # Existing MongoDB utilities
└── tests/
    └── test_mongodb_schema.py         # Test script
```

## ⚙️ Configuration

### **Schema Configuration File** (`config/mongodb_schema_config.yaml`)

```yaml
mongodb_collections:
  ecoesTechDetailsWithEmbedding:
    display_name: "ECoes Tech Details with Embedding"
    description: "MPAN and technical details with vector embeddings"
    
    schema:
      _id_oid:
        type: "string"
        description: "MongoDB Object ID as string"
        include_in_context: true
        max_length: 50
        
      type:
        type: "string"
        description: "Record type (e.g., MPAN, error type)"
        include_in_context: true
        max_length: 100
        
      Results:
        type: "complex"
        description: "Complex nested results structure"
        include_in_context: false  # Exclude from context due to size
        
      embedding:
        type: "vector"
        description: "Vector embedding for similarity search"
        include_in_context: false  # Never include in context
    
    context_optimization:
      max_rows: 10
      essential_columns: ["_id_oid", "type", "value"]
      exclude_columns: ["embedding", "Results", "salt"]
      sample_strategy: "random"
```

### **Key Configuration Options**

| Option | Description | Example |
|--------|-------------|---------|
| `include_in_context` | Whether to include column in LLM context | `true`/`false` |
| `max_length` | Maximum characters for this column | `50`, `100`, `200` |
| `type` | Data type for validation | `string`, `complex`, `vector` |
| `max_rows` | Maximum rows in context | `10`, `25`, `50` |
| `essential_columns` | Columns that must be included | `["_id_oid", "type"]` |
| `exclude_columns` | Columns to always exclude | `["embedding", "salt"]` |

## 🔧 Usage

### **1. Initialize Schema Manager**

```python
from utils.mongodb_schema_manager import MongoDBSchemaManager

# Initialize with default config path
schema_manager = MongoDBSchemaManager()

# Or specify custom config path
schema_manager = MongoDBSchemaManager("path/to/config.yaml")
```

### **2. Get Collection Schema**

```python
# Get full schema for a collection
schema = schema_manager.get_collection_schema("ecoesTechDetailsWithEmbedding")

# Get optimization settings
essential_cols = schema_manager.get_essential_columns("ecoesTechDetailsWithEmbedding")
exclude_cols = schema_manager.get_exclude_columns("ecoesTechDetailsWithEmbedding")
max_rows = schema_manager.get_max_context_rows("ecoesTechDetailsWithEmbedding")
```

### **3. Optimize DataFrame for Context**

```python
# Optimize DataFrame using schema configuration
df_optimized = schema_manager.optimize_dataframe_for_context(
    df, 
    "ecoesTechDetailsWithEmbedding"
)

print(f"Original: {df.shape}")
print(f"Optimized: {df_optimized.shape}")
```

### **4. Check Column Inclusion**

```python
# Check if a column should be included
should_include = schema_manager.should_include_column_in_context(
    "ecoesTechDetailsWithEmbedding", 
    "Results"
)

# Get maximum length for a column
max_length = schema_manager.get_column_max_length(
    "ecoesTechDetailsWithEmbedding", 
    "Results"
)
```

## 🧪 Testing

### **Run Test Script**

```bash
cd utils/tests
python test_mongodb_schema.py
```

### **Expected Output**

```
🧪 Testing MongoDB Schema Manager
==================================================

📋 Collection: ecoesTechDetailsWithEmbedding
------------------------------
✅ Schema found with 12 fields
📝 Display name: ECoes Tech Details with Embedding
📄 Description: MPAN and technical details with vector embeddings

🔧 Optimization Settings:
   Essential columns: ['_id_oid', 'type', 'value']
   Exclude columns: ['embedding', 'Results', 'salt']
   Max context rows: 10
   Max field length: 50

✅ Schema validation: Valid

🔄 Testing DataFrame Optimization
------------------------------
📊 Original DataFrame: 3 rows × 7 columns
   Columns: ['_id_oid', 'type', 'value', 'Results', 'embedding', 'createdAt_date', 'salt']

🎯 Optimized DataFrame: 3 rows × 3 columns
   Columns: ['_id_oid', 'type', 'value']

📋 Sample optimized data:
  _id_oid   type  value
0     123   MPAN  1580001422451
1     456   MPAN  1580001422452
```

## 🔄 Integration with Existing Code

### **Replace Current Optimization Logic**

**Before (Hard-coded):**
```python
# Current approach in rag_fabric_app_docker.py
max_context_rows = 10
context_columns = ['_id_oid', 'value']
df1_context_sample = df1_context.sample(n=max_context_rows, random_state=42)
df1_context_sample = df1_context_sample[context_columns]
```

**After (Schema-driven):**
```python
# New approach using schema manager
from utils.mongodb_schema_manager import MongoDBSchemaManager

schema_manager = MongoDBSchemaManager()
df1_context_sample = schema_manager.optimize_dataframe_for_context(
    df1_context, 
    collection_name
)
```

### **Benefits of Integration**

1. **Automatic Optimization**: No more manual column selection
2. **Consistent Behavior**: Same logic across all collections
3. **Easy Maintenance**: Add new collections via config only
4. **Better Performance**: Optimized for token usage
5. **Error Prevention**: No more parsing issues

## 📝 Adding New Collections

### **1. Add Schema to Config**

```yaml
mongodb_collections:
  newCollection:
    display_name: "New Collection"
    description: "Description of new collection"
    
    schema:
      field1:
        type: "string"
        include_in_context: true
        max_length: 100
        
      field2:
        type: "complex"
        include_in_context: false
        max_length: 50
    
    context_optimization:
      max_rows: 15
      essential_columns: ["field1"]
      exclude_columns: ["field2"]
```

### **2. Use in Application**

```python
# Automatically works with new collection
df_optimized = schema_manager.optimize_dataframe_for_context(
    df, 
    "newCollection"
)
```

## 🚨 Troubleshooting

### **Common Issues**

1. **Schema Not Found**
   - Check config file path
   - Verify collection name spelling
   - Ensure YAML syntax is correct

2. **Columns Missing**
   - Check `essential_columns` in config
   - Verify column names match exactly
   - Check for typos in column names

3. **Token Issues Persist**
   - Reduce `max_rows` in config
   - Set more columns to `include_in_context: false`
   - Reduce `max_length` values

### **Validation**

```python
# Validate collection schema
validation = schema_manager.validate_collection_schema("collectionName")
if not validation['valid']:
    print(f"Schema validation failed: {validation['error']}")
if validation['warnings']:
    print(f"Warnings: {validation['warnings']}")
```

## 🔮 Future Enhancements

1. **Dynamic Schema Discovery**: Auto-detect collection schemas
2. **Performance Metrics**: Track token usage and optimization
3. **Schema Versioning**: Handle schema evolution over time
4. **Advanced Filtering**: Complex column selection rules
5. **Integration with RAG**: Direct integration with RAG pipeline

## 📚 Additional Resources

- [MongoDB Atlas Documentation](https://docs.atlas.mongodb.com/)
- [YAML Configuration Guide](https://yaml.org/spec/)
- [Pandas DataFrame Optimization](https://pandas.pydata.org/docs/user_guide/performance.html)
- [RAG Token Management](https://platform.openai.com/docs/guides/token-usage)

---

**🎉 This system transforms MongoDB integration from a source of errors to a robust, maintainable solution!** 