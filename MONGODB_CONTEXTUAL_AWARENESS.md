# MongoDB Contextual Awareness & Semantic Search

## 🎯 Overview

The enhanced MongoDB Schema Manager now provides **contextual awareness** and **semantic search capabilities** that make MongoDB integration much more intelligent and user-friendly. This system can understand vague user questions and automatically enhance them using business context and data dictionary information.

## 🚀 Key Features

### **1. Business Context Understanding**
- **Domain Knowledge**: Understands the business domain (e.g., "Energy and Utilities")
- **Purpose Awareness**: Knows what the data is used for (e.g., "MPAN management and validation")
- **Common Queries**: Recognizes typical user questions and patterns
- **Key Entities**: Identifies important business concepts (e.g., "MPAN", "meter", "supplier")

### **2. Semantic Search Enhancement**
- **Query Expansion**: Automatically adds relevant synonyms and business terms
- **Intent Detection**: Recognizes what users are trying to accomplish
- **Column Relevance**: Knows which fields are most important for different types of queries
- **Search Strategy**: Chooses between vector search and semantic search based on query content

### **3. Data Dictionary Intelligence**
- **Business Meaning**: Each field has a clear business purpose explanation
- **Search Relevance**: Fields are ranked by their importance for search
- **Semantic Keywords**: Fields are tagged with relevant business terms
- **Validation Patterns**: Knows expected data formats and patterns

## 📁 Enhanced Configuration Structure

### **Business Context Section**
```yaml
business_context:
  domain: "Energy and Utilities"
  purpose: "MPAN (Meter Point Administration Number) management and technical validation"
  common_queries: [
    "Find MPAN records with specific errors",
    "Analyze error patterns in MPAN data",
    "Identify common technical issues"
  ]
  key_entities: ["MPAN", "meter", "supplier", "postcode", "error"]
```

### **Enhanced Data Dictionary**
```yaml
data_dictionary:
  type:
    type: "string"
    description: "Record type (e.g., MPAN, error type)"
    business_meaning: "The type of record or error being reported"
    include_in_context: true
    max_length: 100
    search_relevance: "high"
    semantic_keywords: ["MPAN", "error", "validation", "record type", "issue type"]
    example_values: ["MPAN", "validation_error", "system_error", "data_error"]
```

### **Query Enhancement Configuration**
```yaml
query_enhancement:
  enable_semantic_expansion: true
  business_aliases: {
    "MPAN": ["meter point", "meter number", "energy meter", "electricity meter"],
    "error": ["issue", "problem", "failure", "validation error", "technical issue"]
  }
  common_question_patterns: {
    "find_errors": ["error", "issue", "problem", "failure", "wrong", "invalid"],
    "analyze_patterns": ["pattern", "trend", "common", "frequent", "analysis"]
  }
```

## 🔧 Usage Examples

### **1. Basic Query Enhancement**
```python
from utils.mongodb_schema_manager import MongoDBSchemaManager

schema_manager = MongoDBSchemaManager()
collection_name = "ecoesTechDetailsWithEmbedding"

# Enhance a vague user query
user_query = "Show me problems with meters"
enhanced_info = schema_manager.enhance_user_query(collection_name, user_query)

print(f"Original: {enhanced_info['original_query']}")
print(f"Enhanced: {enhanced_info['enhanced_query']}")
print(f"Detected intent: {enhanced_info['detected_intent']}")
print(f"Business domain: {enhanced_info['business_domain']}")
```

**Output:**
```
Original: Show me problems with meters
Enhanced: Show me problems with meters meter point meter number energy meter electricity meter
Detected intent: ['find_errors']
Business domain: Energy and Utilities
```

### **2. Semantic Search Strategy**
```python
# Get search optimization settings
search_settings = schema_manager.get_search_optimization_settings(collection_name)
business_keywords = schema_manager.get_business_keywords(collection_name)
semantic_boost_fields = schema_manager.get_semantic_boost_fields(collection_name)

print(f"Business keywords: {business_keywords}")
print(f"Semantic boost fields: {semantic_boost_fields}")
```

**Output:**
```
Business keywords: ['MPAN', 'meter', 'supplier', 'postcode', 'error', 'validation']
Semantic boost fields: ['type', 'value']
```

### **3. Column Relevance Analysis**
```python
# Check which columns are most relevant for a query
relevant_columns = enhanced_info['relevant_columns']
for col in relevant_columns:
    print(f"Column: {col['name']}")
    print(f"  Relevance: {col['relevance']}")
    print(f"  Business meaning: {col['business_meaning']}")
    print(f"  Keywords: {col['keywords']}")
```

## 🧪 Testing the Enhanced Features

### **Run Enhanced Test Script**
```bash
cd utils/tests
python test_enhanced_mongodb_schema.py
```

### **Expected Output Highlights**
```
Testing Query Enhancement and Contextual Awareness
--------------------------------------------------

Query: What are the top 5 most common errors in mpan type records?
   Business domain: Energy and Utilities
   Purpose: MPAN (Meter Point Administration Number) management and technical validation
   Detected intent: ['find_errors', 'analyze_patterns']
   Semantic expansions: ['meter point', 'meter number', 'energy meter', 'electricity meter']
   Search strategy: semantic_search
   Relevant columns:
     - type: high relevance
       Business meaning: The type of record or error being reported
       Keywords: ['MPAN', 'error', 'validation', 'record type', 'issue type']
```

## 💡 Real-World Benefits

### **Before (Basic Schema)**
- ❌ User asks: "Show me meter problems"
- ❌ System doesn't understand "meter problems" = "MPAN errors"
- ❌ Search returns irrelevant results
- ❌ User gets frustrated with poor results

### **After (Enhanced Schema)**
- ✅ User asks: "Show me meter problems"
- ✅ System understands: "meter problems" = "MPAN errors" + "validation issues"
- ✅ Automatically expands query with business terms
- ✅ Search strategy switches to semantic search
- ✅ Results are highly relevant and contextual

## 🔄 Integration with RAG Application

### **1. Query Enhancement in RAG Pipeline**
```python
# In your RAG application
if vector_search_engine == "MongoDB":
    # Enhance user query before searching
    enhanced_query_info = schema_manager.enhance_user_query(
        collection_name, 
        user_question
    )
    
    # Use enhanced query for better search results
    search_results = mongodb_index.search(
        enhanced_query_info['enhanced_query'],
        limit=top_n,
        score_threshold=0.01
    )
    
    # Log enhancement details for debugging
    st.info(f"Query enhanced from '{enhanced_query_info['original_query']}' to '{enhanced_query_info['enhanced_query']}'")
    st.info(f"Detected intent: {enhanced_query_info['detected_intent']}")
```

### **2. Contextual Column Selection**
```python
# Use relevant columns for context generation
relevant_cols = enhanced_query_info['relevant_columns']
if relevant_cols:
    # Prioritize high-relevance columns
    high_relevance_cols = [col['name'] for col in relevant_cols if col['relevance'] == 'high']
    if high_relevance_cols:
        st.info(f"Focusing on high-relevance columns: {high_relevance_cols}")
```

## 📝 Adding New Collections with Context

### **1. Define Business Context**
```yaml
mongodb_collections:
  customerData:
    business_context:
      domain: "Customer Relationship Management"
      purpose: "Customer data analysis and segmentation"
      common_queries: [
        "Find high-value customers",
        "Analyze customer behavior patterns",
        "Identify customer segments"
      ]
      key_entities: ["customer", "segment", "value", "behavior"]
```

### **2. Define Data Dictionary**
```yaml
    data_dictionary:
      customer_id:
        business_meaning: "Unique customer identifier"
        search_relevance: "high"
        semantic_keywords: ["customer", "client", "user", "ID"]
        
      customer_value:
        business_meaning: "Customer lifetime value or revenue"
        search_relevance: "high"
        semantic_keywords: ["value", "revenue", "worth", "profit"]
```

### **3. Define Query Enhancement**
```yaml
    query_enhancement:
      business_aliases: {
        "customer": ["client", "user", "buyer"],
        "value": ["worth", "revenue", "profit", "income"]
      }
      common_question_patterns: {
        "find_valuable": ["valuable", "high-value", "premium", "top"],
        "analyze_behavior": ["behavior", "pattern", "trend", "habit"]
      }
```

## 🚨 Troubleshooting

### **Common Issues**

1. **Schema Not Found**
   - Check collection name spelling
   - Verify YAML syntax
   - Ensure all required sections are present

2. **Query Enhancement Not Working**
   - Check `enable_semantic_expansion: true`
   - Verify business aliases are defined
   - Check common question patterns

3. **Column Relevance Issues**
   - Verify `search_relevance` values
   - Check `semantic_keywords` are defined
   - Ensure `include_in_context` is set correctly

### **Validation**
```python
# Validate enhanced schema
validation = schema_manager.validate_collection_schema(collection_name)
if not validation['valid']:
    print(f"Schema validation failed: {validation['error']}")
if validation['warnings']:
    print(f"Warnings: {validation['warnings']}")
```

## 🔮 Future Enhancements

1. **Dynamic Learning**: Learn from user queries to improve patterns
2. **Multi-language Support**: Support for different business languages
3. **Advanced Intent Recognition**: More sophisticated intent detection
4. **Performance Metrics**: Track query enhancement effectiveness
5. **Integration with LLM**: Use LLM for query understanding

## 📚 Additional Resources

- [MongoDB Atlas Vector Search](https://docs.atlas.mongodb.com/atlas-vector-search/)
- [Semantic Search Best Practices](https://www.elastic.co/guide/en/elasticsearch/reference/current/semantic-search.html)
- [Business Intelligence Query Patterns](https://www.databricks.com/blog/2020/01/30/query-patterns-for-business-intelligence.html)

---

**🎉 This enhanced system transforms MongoDB from a simple data store into an intelligent, contextually aware search engine that understands your business domain!** 