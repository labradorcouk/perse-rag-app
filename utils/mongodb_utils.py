import os
from dotenv import load_dotenv
from typing import List, Dict, Any, Optional
import pandas as pd
from urllib.parse import quote_plus

load_dotenv()

# MongoDB configuration
MONGODB_URI = os.getenv("MONGODB_URI")
MONGODB_DB_NAME = os.getenv("MONGODB_DB_NAME", "perse-data-network")
MONGODB_COLLECTION_NAME = os.getenv("MONGODB_COLLECTION_NAME", "addressMatches")
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
            collection_name: Name of the MongoDB collection
            embedding_model: Embedding model for generating vectors
            db_name: Database name (optional, uses environment variable if not provided)
        """
        self.collection_name = collection_name
        self.embedding_model = embedding_model
        self.db_name = db_name or MONGODB_DB_NAME
        self._vector_store = None
        
        # Validate required environment variables
        if not MONGODB_URI:
            raise ValueError("MONGODB_URI environment variable is required")
    
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
    
    def search(self, query: str, limit: int = 5, score_threshold: float = 0.01, 
               pre_filter: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """
        Search for similar documents in MongoDB Atlas.
        
        Args:
            query: Search query
            limit: Maximum number of results to return
            score_threshold: Minimum similarity score threshold
            pre_filter: MongoDB filter to apply before vector search
            
        Returns:
            List of search results with payload and metadata
        """
        try:
            # Create retriever
            retriever = self.vector_store.as_retriever(
                search_type="similarity",
                search_kwargs={
                    "k": limit,
                    "score_threshold": score_threshold,
                    "pre_filter": pre_filter or {}
                },
            )
            
            # Perform search
            docs = retriever.get_relevant_documents(query)
            
            # Convert to standardized format
            results = []
            for doc in docs:
                result = {
                    'payload': doc.page_content,
                    'metadata': doc.metadata,
                    'score': getattr(doc, 'score', None)
                }
                results.append(result)
            
            return results
            
        except Exception as e:
            raise Exception(f"Search failed: {str(e)}")
    
    def get_collection_info(self) -> Dict[str, Any]:
        """
        Get information about the MongoDB collection.
        
        Returns:
            Dictionary with collection information
        """
        try:
            from pymongo import MongoClient
            
            # Parse connection string to get client
            client = MongoClient(MONGODB_URI)
            db = client[self.db_name]
            collection = db[self.collection_name]
            
            # Get collection stats
            stats = db.command("collstats", self.collection_name)
            
            return {
                'collection_name': self.collection_name,
                'database_name': self.db_name,
                'document_count': stats.get('count', 0),
                'size_bytes': stats.get('size', 0),
                'avg_document_size': stats.get('avgObjSize', 0),
                'index_count': stats.get('nindexes', 0)
            }
            
        except Exception as e:
            return {
                'collection_name': self.collection_name,
                'database_name': self.db_name,
                'error': str(e)
            }
    
    def test_connection(self) -> bool:
        """
        Test MongoDB connection.
        
        Returns:
            True if connection is successful, False otherwise
        """
        try:
            # Try to get collection info
            info = self.get_collection_info()
            return 'error' not in info
        except Exception:
            return False
    
    def get_sample_documents(self, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Get sample documents from the collection.
        
        Args:
            limit: Number of sample documents to return
            
        Returns:
            List of sample documents
        """
        try:
            from pymongo import MongoClient
            
            client = MongoClient(MONGODB_URI)
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
        Create vector search index (if not exists).
        Note: This requires admin privileges and should be done manually in MongoDB Atlas.
        
        Args:
            index_config: Index configuration
        """
        try:
            from pymongo import MongoClient
            
            client = MongoClient(MONGODB_URI)
            db = client[self.db_name]
            collection = db[self.collection_name]
            
            # Default index configuration for vector search
            if not index_config:
                index_config = {
                    "mappings": {
                        "dynamic": True,
                        "fields": {
                            "embedding": {
                                "dimensions": 1536,  # OpenAI embedding dimensions
                                "similarity": "cosine",
                                "type": "knnVector"
                            }
                        }
                    }
                }
            
            # Create index
            collection.create_index(index_config)
            print(f"Vector search index created for collection '{self.collection_name}'")
            
        except Exception as e:
            raise Exception(f"Failed to create index: {str(e)}")
    
    def insert_documents(self, documents: List[Dict[str, Any]]):
        """
        Insert documents into the collection.
        
        Args:
            documents: List of documents to insert
        """
        try:
            from pymongo import MongoClient
            
            client = MongoClient(MONGODB_URI)
            db = client[self.db_name]
            collection = db[self.collection_name]
            
            # Insert documents
            result = collection.insert_many(documents)
            print(f"Inserted {len(result.inserted_ids)} documents into '{self.collection_name}'")
            
        except Exception as e:
            raise Exception(f"Failed to insert documents: {str(e)}") 