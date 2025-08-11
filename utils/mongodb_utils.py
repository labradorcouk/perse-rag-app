import os
from dotenv import load_dotenv
from typing import List, Dict, Any, Optional
import pandas as pd
from urllib.parse import quote_plus
import yaml

load_dotenv()

# MongoDB configuration
MONGODB_URI = os.getenv("MONGODB_URI")
MONGODB_DB_NAME = os.getenv("MONGODB_DB_NAME", "perse-data-network")
MONGODB_INDEX_NAME = os.getenv("MONGODB_INDEX_NAME", "vector_index")

class MongoDBIndex:
    """
    MongoDB Atlas Vector Search integration for RAG application.
    Provides vector search capabilities using MongoDB Atlas.
    """
    
    def __init__(self, collection_name: str, embedding_model=None, db_name: str = None):
        """
        Initialize MongoDB vector search index.
        
        Args:
            collection_name: Name of the MongoDB collection (should match table name from config)
            embedding_model: Embedding model for generating vectors
            db_name: Database name (optional, uses environment variable if not provided)
        """
        self.collection_name = collection_name
        self.embedding_model = embedding_model
        self.db_name = db_name or MONGODB_DB_NAME
        self._vector_store = None
        self._mongodb_client = None
        
        # Validate required environment variables
        if not MONGODB_URI:
            raise ValueError("MONGODB_URI environment variable is required")
    
    def _get_mongodb_client(self):
        """Get MongoDB client for raw data operations."""
        try:
            from pymongo import MongoClient
            if self._mongodb_client is None:
                self._mongodb_client = MongoClient(MONGODB_URI)
            return self._mongodb_client
        except ImportError as e:
            raise ImportError(f"Required package not installed: {e}. Please install pymongo")
        except Exception as e:
            raise ConnectionError(f"Failed to connect to MongoDB: {str(e)}")
    
    def _get_mongodb_connection(self):
        """Get MongoDB Atlas Vector Search connection."""
        try:
            from langchain_mongodb import MongoDBAtlasVectorSearch
            from langchain_openai import OpenAIEmbeddings
            
            # Create embeddings model
            if hasattr(self.embedding_model, 'embed'):
                # Use the provided embedding model
                embeddings_model = self.embedding_model
            else:
                # Fallback to OpenAI embeddings
                openai_api_key = os.getenv("OPENAI_API_KEY")
                if not openai_api_key:
                    raise ValueError("OPENAI_API_KEY environment variable is required for OpenAI embeddings")
                embeddings_model = OpenAIEmbeddings(
                    disallowed_special=(), 
                    openai_api_key=openai_api_key
                )
            
            # Create vector store
            vector_store = MongoDBAtlasVectorSearch.from_connection_string(
                MONGODB_URI,
                f"{self.db_name}.{self.collection_name}",
                embeddings_model,
                index_name=MONGODB_INDEX_NAME,
            )
            
            return vector_store
            
        except ImportError as e:
            raise ImportError(f"Required packages not installed: {e}. Please install langchain-mongodb and langchain-openai")
        except Exception as e:
            raise ConnectionError(f"Failed to connect to MongoDB Atlas: {str(e)}")
    
    @property
    def vector_store(self):
        """Lazy property to get the MongoDB vector store."""
        if self._vector_store is None:
            self._vector_store = self._get_mongodb_connection()
        return self._vector_store
    
    def get_raw_data(self, limit: int = 5000, filter_query: Dict[str, Any] = None) -> pd.DataFrame:
        """
        Get raw data from MongoDB collection.
        
        Args:
            limit: Maximum number of documents to retrieve
            filter_query: MongoDB filter query
            
        Returns:
            DataFrame containing raw data from MongoDB
        """
        try:
            client = self._get_mongodb_client()
            db = client[self.db_name]
            collection = db[self.collection_name]
            
            # Build query
            query = filter_query or {}
            
            # Get documents
            cursor = collection.find(query).limit(limit)
            documents = list(cursor)
            
            if not documents:
                return pd.DataFrame()
            
            # Convert to DataFrame
            df = pd.DataFrame(documents)
            
            # Remove MongoDB-specific fields if present
            if '_id' in df.columns:
                df = df.drop('_id', axis=1)
            
            return df
            
        except Exception as e:
            raise Exception(f"Failed to retrieve raw data from MongoDB: {str(e)}")
    
    def search(self, query: str, limit: int = 5, score_threshold: float = 0.01, 
               pre_filter: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """
        Search for similar documents in MongoDB Atlas.
        
        Args:
            query: Search query
            limit: Maximum number of results to return (max 100 for MongoDB Atlas)
            score_threshold: Minimum similarity score threshold
            pre_filter: MongoDB filter to apply before vector search
            
        Returns:
            List of search results with payload and metadata
        """
        try:
            # MongoDB Atlas has limits on search parameters
            # Limit the number of results to a reasonable value
            search_limit = min(limit, 100)  # MongoDB Atlas typically limits to 100
            
            print(f"🔍 Debug: Starting search for query: '{query}' with limit: {search_limit}")
            
            # First, try vector search
            try:
                print("🔍 Debug: Attempting vector search...")
                
                # Create retriever with appropriate parameters
                retriever = self.vector_store.as_retriever(
                    search_type="similarity",
                    search_kwargs={
                        "k": search_limit,
                        "score_threshold": score_threshold,
                        "pre_filter": pre_filter
                    }
                )
                
                print("🔍 Debug: Retriever created successfully")
                
                # Get relevant documents
                docs = retriever.get_relevant_documents(query)
                
                print(f"🔍 Debug: Vector search returned {len(docs)} documents")
                
                # Convert to standardized format
                results = []
                for doc in docs:
                    try:
                        # Handle different payload formats
                        if hasattr(doc, 'page_content'):
                            payload = doc.page_content
                        elif isinstance(doc, dict):
                            payload = str(doc)
                        else:
                            payload = str(doc)
                        
                        # Try to parse as JSON if it looks like JSON
                        if payload.startswith('{') and payload.endswith('}'):
                            try:
                                import json
                                parsed_payload = json.loads(payload)
                                # Convert back to string for consistency
                                payload = json.dumps(parsed_payload, default=str)
                            except json.JSONDecodeError:
                                # If JSON parsing fails, use as-is
                                pass
                        
                        result = {
                            "payload": payload,
                            "metadata": getattr(doc, 'metadata', {}),
                            "score": getattr(doc, 'metadata', {}).get('score', 0.0)
                        }
                        results.append(result)
                    except Exception as e:
                        print(f"Warning: Could not process document: {e}")
                        continue
                
                if results:
                    print(f"🔍 Debug: Vector search successful, returning {len(results)} results")
                    return results
                else:
                    print("🔍 Debug: Vector search returned no results, trying fallback...")
                    
            except Exception as vector_error:
                print(f"❌ Vector search failed: {str(vector_error)}")
                print("Falling back to text search...")
            
            # Fallback: If vector search fails or returns no results, try text search
            try:
                print("🔍 Debug: Attempting text search fallback...")
                client = self._get_mongodb_client()
                db = client[self.db_name]
                collection = db[self.collection_name]
                
                # Try text search using MongoDB's text search capabilities
                # First, check if there's a text index
                indexes = list(collection.list_indexes())
                has_text_index = any(idx.get('key', {}).get('$**', None) == 'text' for idx in indexes)
                
                print(f"🔍 Debug: Has text index: {has_text_index}")
                
                if has_text_index:
                    # Use text search
                    print("🔍 Debug: Using text search with text index...")
                    search_results = list(collection.find(
                        {"$text": {"$search": query}},
                        {"score": {"$meta": "textScore"}}
                    ).sort([("score", {"$meta": "textScore"})]).limit(search_limit))
                else:
                    # Use simple text matching
                    print("🔍 Debug: Using regex text matching...")
                    search_results = list(collection.find(
                        {"$or": [
                            {"type": {"$regex": query, "$options": "i"}},
                            {"value": {"$regex": query, "$options": "i"}},
                            {"Results": {"$regex": query, "$options": "i"}}
                        ]}
                    ).limit(search_limit))
                
                print(f"🔍 Debug: Text search returned {len(search_results)} documents")
                
                # Convert to standardized format
                results = []
                for doc in search_results:
                    # Convert document to string representation for payload
                    doc_copy = doc.copy()
                    if '_id' in doc_copy:
                        doc_copy['_id'] = str(doc_copy['_id'])
                    
                    result = {
                        "payload": str(doc_copy),
                        "metadata": {"source": "text_search", "score": doc.get('score', 1.0)},
                        "score": doc.get('score', 1.0)
                    }
                    results.append(result)
                
                if results:
                    print(f"🔍 Debug: Text search successful, returning {len(results)} results")
                    return results
                else:
                    print("🔍 Debug: Text search returned no results, using sample documents...")
                
            except Exception as text_error:
                print(f"❌ Text search also failed: {str(text_error)}")
                
            # Final fallback: return sample documents
            print("🔍 Debug: Using sample documents as final fallback...")
            sample_docs = self.get_sample_documents(limit=search_limit)
            results = []
            for doc in sample_docs:
                result = {
                    "payload": str(doc),
                    "metadata": {"source": "sample_fallback"},
                    "score": 1.0
                }
                results.append(result)
            
            print(f"🔍 Debug: Sample fallback returning {len(results)} results")
            return results
            
        except Exception as e:
            raise Exception(f"Failed to search MongoDB Atlas: {str(e)}")
    
    def get_collection_info(self) -> Dict[str, Any]:
        """
        Get information about the MongoDB collection.
        
        Returns:
            Dictionary with collection information
        """
        try:
            client = self._get_mongodb_client()
            db = client[self.db_name]
            collection = db[self.collection_name]
            
            # Get collection stats
            stats = db.command("collstats", self.collection_name)
            
            # Get sample document to understand structure
            sample_doc = collection.find_one()
            
            info = {
                "collection_name": self.collection_name,
                "database_name": self.db_name,
                "document_count": stats.get("count", 0),
                "size_bytes": stats.get("size", 0),
                "avg_document_size": stats.get("avgObjSize", 0),
                "has_sample_document": sample_doc is not None,
                "sample_fields": list(sample_doc.keys()) if sample_doc else []
            }
            
            return info
            
        except Exception as e:
            raise Exception(f"Failed to get collection info: {str(e)}")
    
    def test_connection(self) -> bool:
        """
        Test MongoDB connection.
        
        Returns:
            True if connection is successful
        """
        try:
            client = self._get_mongodb_client()
            # Ping the database
            client.admin.command('ping')
            return True
        except Exception as e:
            print(f"MongoDB connection test failed: {str(e)}")
            # Add more detailed error information
            if "DNS" in str(e):
                print(f"DNS resolution error - check if the cluster name is correct")
                print(f"Current URI: {MONGODB_URI}")
            elif "Authentication" in str(e):
                print(f"Authentication error - check username and password")
            elif "Connection" in str(e):
                print(f"Connection error - check network connectivity and firewall")
            return False
    
    def get_sample_documents(self, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Get sample documents from the collection.
        
        Args:
            limit: Number of sample documents to retrieve
            
        Returns:
            List of sample documents
        """
        try:
            client = self._get_mongodb_client()
            db = client[self.db_name]
            collection = db[self.collection_name]
            
            # Get sample documents
            sample_docs = list(collection.find().limit(limit))
            
            # Convert ObjectId to string for JSON serialization
            for doc in sample_docs:
                if '_id' in doc:
                    doc['_id'] = str(doc['_id'])
            
            return sample_docs
            
        except Exception as e:
            raise Exception(f"Failed to get sample documents: {str(e)}")
    
    def create_index(self, index_config: Dict[str, Any] = None):
        """
        Create vector index on the collection.
        
        Args:
            index_config: Index configuration
        """
        try:
            client = self._get_mongodb_client()
            db = client[self.db_name]
            collection = db[self.collection_name]
            
            # Default index configuration
            default_config = {
                "mappings": {
                    "dynamic": True,
                    "fields": {
                        "embedding": {
                            "dimensions": 384,
                            "similarity": "cosine",
                            "type": "knnVector"
                        }
                    }
                }
            }
            
            config = index_config or default_config
            
            # Create index
            db.command({
                "createIndexes": self.collection_name,
                "indexes": [{
                    "key": {"embedding": "vector"},
                    "name": MONGODB_INDEX_NAME,
                    "numDimensions": 384,
                    "similarity": "cosine"
                }]
            })
            
            print(f"✅ Vector index created on collection: {self.collection_name}")
            
        except Exception as e:
            raise Exception(f"Failed to create index: {str(e)}")
    
    def insert_documents(self, documents: List[Dict[str, Any]]):
        """
        Insert documents into the collection.
        
        Args:
            documents: List of documents to insert
        """
        try:
            client = self._get_mongodb_client()
            db = client[self.db_name]
            collection = db[self.collection_name]
            
            # Insert documents
            result = collection.insert_many(documents)
            
            print(f"✅ Inserted {len(result.inserted_ids)} documents into collection: {self.collection_name}")
            
            return result.inserted_ids
            
        except Exception as e:
            raise Exception(f"Failed to insert documents: {str(e)}")
    
    def close(self):
        """Close MongoDB connection."""
        if self._mongodb_client:
            self._mongodb_client.close() 