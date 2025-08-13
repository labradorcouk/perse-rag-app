# MongoDB Enhanced Q&A Pattern Matching & Intent Detection

## 🎯 Overview

We've implemented a **hybrid approach** for enhanced contextual awareness in MongoDB collections:

1. **Core Q&A patterns in YAML config** - Fast, always available, version controlled
2. **Extended Q&A patterns in MongoDB collections** - Dynamic, scalable, user-contributable
3. **Advanced intent detection** - Multi-layered understanding of user queries
4. **Pattern matching with confidence scoring** - Precise query-to-pattern matching

## 🚀 **Why This Hybrid Approach?**

### **YAML Config (Core Patterns)**
- ✅ **Instant access** - No database queries needed
- ✅ **Version controlled** - Changes tracked in git
- ✅ **Fast performance** - No network latency
- ✅ **Consistent** - Same across all environments
- ✅ **Easy maintenance** - Simple YAML editing

### **MongoDB Collections (Extended Patterns)**
- ✅ **Dynamic updates** - No app restart required
- ✅ **Scalable** - Handle thousands of patterns
- ✅ **User contributions** - Allow users to add patterns
- ✅ **Analytics** - Track usage and effectiveness
- ✅ **A/B testing** - Test different pattern versions

## 🔧 **Configuration Structure**

### **1. Core Q&A Patterns (YAML)**
```yaml
question_answer_patterns:
  core_patterns:
    - question: "What are the most common errors in MPAN records?"
      answer_intent: "find_errors"
      business_entities: ["MPAN", "error", "validation"]
      expected_columns: ["type", "value", "Results"]
      search_strategy: "error_analysis"
      sample_queries: [
        "show me common MPAN errors",
        "what errors occur most frequently",
        "find the top error types"
      ]
  
  intent_categories:
    find_errors:
      description: "Questions about finding and identifying errors"
      keywords: ["error", "issue", "problem", "failure", "wrong", "invalid"]
      business_context: "Error identification and troubleshooting"
      expected_output: "List of errors with counts and details"
```

### **2. Extended Q&A Collections (MongoDB)**
```yaml
mongodb_qa_collections:
  extended_qa_patterns:
    collection_name: "extended_qa_patterns"
    schema:
      collection_name: "string"          # Which collection this applies to
      question_pattern: "string"         # The question pattern
      answer_intent: "string"            # Detected intent category
      business_entities: ["array"]       # Business entities mentioned
      expected_columns: ["array"]        # Columns to include in response
      search_strategy: "string"          # Recommended search approach
      confidence_score: "float"          # Pattern confidence (0.0-1.0)
      usage_count: "integer"            # How many times used
      success_rate: "float"              # Success rate (0.0-1.0)
      tags: ["array"]                   # Categorization tags
```

## 🎮 **How It Works**

### **1. Query Processing Pipeline**
```
User Query → Pattern Matching → Intent Detection → Column Selection → Search Strategy
     ↓              ↓              ↓              ↓              ↓
  "Show me    →  Q&A Pattern  →  find_errors  →  [type,value] →  error_analysis
   MPAN errors"    Match: 0.95     Intent        Columns         Strategy
```

### **2. Pattern Matching Algorithm**
- **Exact Match**: 0.95 confidence (pattern found in query)
- **Sample Query Match**: 0.90 confidence (matches example queries)
- **Business Entity Match**: 0.40 weight (entities found in query)
- **Keyword Match**: 0.60 weight (intent keywords found)
- **Combined Score**: Weighted average with 0.70 minimum threshold

### **3. Intent Detection Layers**
1. **Q&A Pattern Matching** - Primary intent detection
2. **Intent Categories** - Keyword-based classification
3. **Semantic Expansion** - Business term expansion
4. **Column Relevance** - Field importance analysis

## 📁 **File Structure**

```
├── config/
│   ├── mongodb_schema_config.yaml          # Core Q&A patterns
│   └── mongodb_qa_collections.yaml         # Extended Q&A collections
├── utils/
│   ├── mongodb_schema_manager.py           # Enhanced schema manager
│   └── tests/
│       └── test_enhanced_qa_patterns.py    # Q&A pattern tests
└── MONGODB_QA_PATTERNS.md                  # This documentation
```

## 🧪 **Testing the Enhanced Features**

### **Run Enhanced Q&A Pattern Tests**
```bash
cd utils/tests
python test_enhanced_qa_patterns.py
```

### **Expected Output**
```
Testing Enhanced Q&A Pattern Matching and Intent Detection
==========================================================

Collection: ecoesTechDetailsWithEmbedding
----------------------------------------
Found 5 Q&A patterns
Found 5 intent categories

Q&A Patterns:
--------------------
1. Question: What are the most common errors in MPAN records?
   Intent: find_errors
   Business Entities: ['MPAN', 'error', 'validation']
   Expected Columns: ['type', 'value', 'Results']
   Search Strategy: error_analysis
   Sample Queries: 3 examples

Testing Q&A Pattern Matching
----------------------------------------

Query: What are the most common errors in MPAN records?
--------------------------------------------------
✅ Pattern Match Found!
   Question Pattern: What are the most common errors in MPAN records?
   Answer Intent: find_errors
   Match Score: 0.95
   Business Entities: ['MPAN', 'error', 'validation']
   Expected Columns: ['type', 'value', 'Results']
   Search Strategy: error_analysis

Enhanced Query Info:
   Original Query: What are the most common errors in MPAN records?
   Enhanced Query: What are the most common errors in MPAN records?
   Business Domain: Energy and Utilities
   Purpose: MPAN (Meter Point Administration Number) management and technical validation
   Detected Intent: ['find_errors']
   Semantic Expansions: []
   Search Strategy: error_analysis
   Confidence Score: 0.95
   Q&A Pattern Match: ✅
     Pattern: What are the most common errors in MPAN records?
     Intent: find_errors
   Relevant Columns:
     - type: high relevance
       Business meaning: Expected column for find_errors intent
       Keywords: []
```

## 💡 **Real-World Examples**

### **Example 1: Error Analysis Query**
**User Query**: "Show me common MPAN errors"

**Pattern Matching**:
- ✅ **Exact Pattern**: "What are the most common errors in MPAN records?"
- ✅ **Intent**: `find_errors`
- ✅ **Confidence**: 0.95
- ✅ **Expected Columns**: `['type', 'value', 'Results']`
- ✅ **Search Strategy**: `error_analysis`

**Enhanced Query**: "Show me common MPAN errors"
**Business Context**: Energy and Utilities - MPAN management and validation
**Relevant Columns**: type, value, Results (high relevance)

### **Example 2: Pattern Analysis Query**
**User Query**: "Analyze error patterns in MPAN data"

**Pattern Matching**:
- ✅ **Pattern**: "Analyze error patterns in MPAN data"
- ✅ **Intent**: `analyze_patterns`
- ✅ **Confidence**: 0.95
- ✅ **Expected Columns**: `['type', 'value', 'createdAt_date']`
- ✅ **Search Strategy**: `pattern_analysis`

**Enhanced Query**: "Analyze error patterns in MPAN data"
**Business Context**: Pattern recognition and trend analysis
**Relevant Columns**: type, value, createdAt_date (for temporal analysis)

### **Example 3: Location-Based Query**
**User Query**: "Find MPANs in specific area"

**Pattern Matching**:
- ✅ **Sample Query**: "Find MPAN records by postcode location"
- ✅ **Intent**: `location_based_search`
- ✅ **Confidence**: 0.90
- ✅ **Expected Columns**: `['value', 'Results']`
- ✅ **Search Strategy**: `location_search`

**Enhanced Query**: "Find MPANs in specific area"
**Business Context**: Geographic data analysis and location-based insights
**Relevant Columns**: value, Results (for location data)

## 🔄 **Adding New Q&A Patterns**

### **1. Add to YAML Config (Core Patterns)**
```yaml
mongodb_collections:
  your_collection:
    question_answer_patterns:
      core_patterns:
        - question: "Your new question pattern"
          answer_intent: "your_intent_name"
          business_entities: ["entity1", "entity2"]
          expected_columns: ["col1", "col2"]
          search_strategy: "your_strategy"
          sample_queries: [
            "example query 1",
            "example query 2"
          ]
      
      intent_categories:
        your_intent_name:
          description: "Description of this intent"
          keywords: ["keyword1", "keyword2"]
          business_context: "Business context explanation"
          expected_output: "What users expect to see"
```

### **2. Add to MongoDB Collection (Extended Patterns)**
```python
# Example: Adding a new pattern to MongoDB
new_pattern = {
    "collection_name": "your_collection",
    "question_pattern": "Your new question pattern",
    "answer_intent": "your_intent_name",
    "business_entities": ["entity1", "entity2"],
    "expected_columns": ["col1", "col2"],
    "search_strategy": "your_strategy",
    "confidence_score": 0.85,
    "tags": ["tag1", "tag2"],
    "created_by": "user_or_system",
    "created_at": datetime.now(),
    "is_active": True
}

# Insert into MongoDB collection
db.extended_qa_patterns.insert_one(new_pattern)
```

## 🚨 **Troubleshooting**

### **Common Issues**

1. **Pattern Not Found**
   - Check pattern spelling and syntax
   - Verify confidence threshold (default: 0.70)
   - Check business entities and keywords

2. **Intent Detection Not Working**
   - Verify `enable_intent_detection: true`
   - Check intent categories configuration
   - Ensure keywords are properly defined

3. **Column Selection Issues**
   - Verify `expected_columns` in patterns
   - Check column names exist in collection
   - Ensure `include_in_context` is set correctly

### **Validation**
```python
# Validate enhanced schema
validation = schema_manager.validate_collection_schema(collection_name)
if not validation['valid']:
    print(f"Schema validation failed: {validation['error']}")
if validation['warnings']:
    print(f"Warnings: {validation['warnings']}")

# Check Q&A patterns
qa_patterns = schema_manager.get_qa_patterns(collection_name)
print(f"Found {len(qa_patterns)} Q&A patterns")

# Check intent categories
intent_categories = schema_manager.get_intent_categories(collection_name)
print(f"Found {len(intent_categories)} intent categories")
```

## 🔮 **Future Enhancements**

1. **Dynamic Learning**: Learn from user queries to improve patterns
2. **Pattern Evolution**: Automatically evolve patterns based on usage
3. **User Feedback Integration**: Use feedback to improve pattern matching
4. **A/B Testing**: Test different pattern versions for effectiveness
5. **Performance Metrics**: Track pattern matching accuracy and speed
6. **Multi-language Support**: Support for different business languages

## 🎉 **Summary**

The enhanced Q&A pattern matching system provides:

- ✅ **Fast Pattern Matching** - Core patterns in YAML for instant access
- ✅ **Scalable Learning** - Extended patterns in MongoDB for growth
- ✅ **Confidence Scoring** - Precise matching with confidence levels
- ✅ **Intent Detection** - Multi-layered understanding of user goals
- ✅ **Column Optimization** - Smart column selection for each intent
- ✅ **Search Strategy** - Recommended approaches for different query types
- ✅ **Business Context** - Deep understanding of business domain
- ✅ **Dynamic Updates** - Add new patterns without app restart

This hybrid approach gives you the **best of both worlds**: fast, reliable core patterns and scalable, dynamic extended patterns that can grow with your business needs.

Your MongoDB integration is now **intelligent, contextually aware, and pattern-driven**, making it much more powerful than a simple data store. Users can ask questions in natural language, and the system will automatically understand their intent and provide highly relevant, optimized results! 🎯 