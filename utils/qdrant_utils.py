import os
from dotenv import load_dotenv
from qdrant_client import QdrantClient, models
from qdrant_client.http.models import Distance, VectorParams
from fastembed.embedding import DefaultEmbedding

load_dotenv()

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")


class QdrantIndex:
    def __init__(self, collection_name, embedding_model=None):
        self.collection_name = collection_name
        self.embedding_model = embedding_model if embedding_model else DefaultEmbedding()
        # Don't connect immediately - make it lazy
        self._client = None

    def _get_qdrant_client(self):
        """Get Qdrant client with fallback connection methods for Docker networking"""
        import streamlit as st
        
        # If we already have a client, return it
        if self._client is not None:
            return self._client
        
        # Try different connection methods (same as test script)
        connection_methods = [
            # Method 1: Environment variables (this worked in our test)
            lambda: QdrantClient(url=QDRANT_URL, prefer_grpc=True) if QDRANT_URL else None,
            # Method 2: Docker service name
            lambda: QdrantClient("qdrant", port=6333, prefer_grpc=True),
            # Method 3: localhost
            lambda: QdrantClient("localhost", port=6333, prefer_grpc=True),
            # Method 4: host.docker.internal (for Docker Desktop)
            lambda: QdrantClient("host.docker.internal", port=6333, prefer_grpc=True),
        ]
        
        for i, method in enumerate(connection_methods):
            try:
                client = method()
                if client:
                    # Test the connection
                    client.get_collections()
                    self._client = client
                    return client
            except Exception as e:
                continue
        
        # If all methods fail, raise an error
        raise ConnectionError("Could not connect to Qdrant using any available method")

    @property
    def client(self):
        """Lazy property to get the Qdrant client"""
        if self._client is None:
            self._client = self._get_qdrant_client()
        return self._client

    def create_collection(self, vector_size, distance=Distance.COSINE):
        self.client.recreate_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(size=vector_size, distance=distance),
        )
        print(f"Collection '{self.collection_name}' created successfully.")

    def upsert_documents(self, documents):
        self.client.upsert(
            collection_name=self.collection_name,
            points=documents,
            wait=True
        )
        print(f"Upserted {len(documents)} documents into '{self.collection_name}'.")

    def search(self, query, limit=5, with_payload=True, score_threshold=None):
        # Handle different embedding model types
        if hasattr(self.embedding_model, 'embed'):
            # DefaultEmbedding from fastembed
            query_embedding_generator = self.embedding_model.embed(query)
            query_embedding = list(query_embedding_generator)[0]  # Get first (and only) embedding
        elif hasattr(self.embedding_model, 'encode'):
            # SentenceTransformer
            query_embedding = self.embedding_model.encode([query])[0]
        else:
            raise ValueError(f"Unsupported embedding model type: {type(self.embedding_model)}. Expected DefaultEmbedding or SentenceTransformer.")
        
        search_result = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_embedding,
            limit=limit,
            with_payload=with_payload,
            score_threshold=score_threshold
        )
        return search_result 