# MongoDB Vector Search Integration

This document describes the MongoDB Atlas Vector Search integration for the RAG application.

## Overview

The MongoDB integration provides a third vector search engine option alongside FAISS and Qdrant. It uses MongoDB Atlas Vector Search to perform semantic similarity searches on vector embeddings stored in MongoDB collections.

## Features

- ✅ **Vector Search**: Semantic similarity search using MongoDB Atlas
- ✅ **Multiple Collections**: Support for multiple table collections
- ✅ **Environment Configuration**: Configurable via environment variables
- ✅ **Error Handling**: Comprehensive error handling and debugging
- ✅ **Fallback Support**: Graceful fallback to other vector search engines
- ✅ **Performance Optimization**: Caching and optimization features

## Environment Variables

### Required
- `MONGODB_URI`: MongoDB Atlas connection string

### Optional
- `MONGODB_DB_NAME`: Database name (default: "perse-data-network")
- `MONGODB_COLLECTION_NAME`: Default collection name (default: "addressMatches")
- `MONGODB_INDEX_NAME`: Vector search index name (default: "vector_index")

## Configuration Example

```bash
# Required
export MONGODB_URI="mongodb+srv://username:password@cluster.mongodb.net/?retryWrites=true&w=majority"

# Optional
export MONGODB_DB_NAME="perse-data-network"
export MONGODB_COLLECTION_NAME="addressMatches"
export MONGODB_INDEX_NAME="vector_index"
```

## Usage

### In the Application

1. **Select MongoDB**: Choose "MongoDB" as the vector search engine in the UI
2. **Configure Collections**: Ensure your MongoDB collections are properly indexed
3. **Run Queries**: The application will automatically use MongoDB for vector search

### Programmatic Usage

```python
from utils.mongodb_utils import MongoDBIndex

# Create MongoDB index
mongodb_index = MongoDBIndex(
    collection_name="your_collection",
    embedding_model=your_embedding_model
)

# Perform search
results = mongodb_index.search(
    query="your search query",
    limit=10,
    score_threshold=0.01
)
```

## MongoDB Atlas Setup

### 1. Create Vector Search Index

In MongoDB Atlas, create a vector search index:

```json
{
  "mappings": {
    "dynamic": true,
    "fields": {
      "embedding": {
        "dimensions": 1536,
        "similarity": "cosine",
        "type": "knnVector"
      }
    }
  }
}
```

### 2. Document Structure

Documents should contain:
- `embedding`: Vector embedding (array of floats)
- `text`: Original text content
- `metadata`: Additional metadata

Example document:
```json
{
  "_id": ObjectId("..."),
  "embedding": [0.1, 0.2, 0.3, ...],
  "text": "Energy efficiency rating A",
  "metadata": {
    "table": "epc_domestic_scotland",
    "column": "current_energy_rating"
  }
}
```

## Integration with RAG Pipeline

### 1. Vector Search Engine Selection

The application now supports three vector search engines:
- **FAISS**: Local vector search
- **Qdrant**: Vector database
- **MongoDB**: MongoDB Atlas Vector Search

### 2. Search Flow

1. User selects "MongoDB" as vector search engine
2. Application connects to MongoDB Atlas
3. Performs semantic search on collections
4. Retrieves relevant documents
5. Converts to DataFrame format
6. Integrates with existing RAG pipeline

### 3. Error Handling

- Connection failures fall back to other engines
- Detailed error messages and debugging info
- Graceful degradation

## Testing

Run the MongoDB integration tests:

```bash
python utils/tests/test_mongodb_integration.py
```

This will test:
- Environment configuration
- MongoDB connection
- Vector search functionality
- Error handling

## Dependencies

Required packages:
- `pymongo`: MongoDB Python driver
- `langchain-mongodb`: LangChain MongoDB integration
- `langchain-openai`: OpenAI embeddings (fallback)

Install with:
```bash
pip install pymongo langchain-mongodb langchain-openai
```

## Troubleshooting

### Common Issues

1. **Connection Failed**
   - Check `MONGODB_URI` environment variable
   - Verify MongoDB Atlas credentials
   - Ensure network connectivity

2. **Index Not Found**
   - Create vector search index in MongoDB Atlas
   - Check index name configuration

3. **No Results**
   - Verify documents contain embeddings
   - Check score threshold settings
   - Ensure proper document structure

### Debug Information

The application provides detailed debug information:
- Connection status
- Collection information
- Search results count
- Error details

## Performance Considerations

- **Caching**: Results are cached for 30 minutes
- **Batch Processing**: Multiple collections processed efficiently
- **Connection Pooling**: MongoDB connection reuse
- **Lazy Loading**: Connections established only when needed

## Security

- **Environment Variables**: Sensitive data stored in environment variables
- **Connection String**: MongoDB URI should be properly secured
- **Access Control**: MongoDB Atlas access controls apply

## Future Enhancements

- [ ] Support for custom embedding models
- [ ] Advanced filtering options
- [ ] Real-time indexing
- [ ] Performance monitoring
- [ ] Multi-region support 