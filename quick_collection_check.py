#!/usr/bin/env python3
"""
Quick script to check Qdrant collection sizes.

Simple utility to get collection counts and estimated sizes.
"""

import os
from dotenv import load_dotenv
from qdrant_client import QdrantClient

load_dotenv()

def format_size(size_bytes):
    """Format bytes into human readable size."""
    if size_bytes == 0:
        return "0B"
    
    size_names = ["B", "KB", "MB", "GB", "TB"]
    i = 0
    while size_bytes >= 1024 and i < len(size_names) - 1:
        size_bytes /= 1024.0
        i += 1
    
    return f"{size_bytes:.2f} {size_names[i]}"

def main():
    """Quick collection size check."""
    print("🔍 Quick Qdrant Collection Check")
    print("=" * 40)
    
    try:
        # Initialize client
        client = QdrantClient(
            url=os.getenv("QDRANT_URL"),
            api_key=os.getenv("QDRANT_API_KEY"),
            prefer_grpc=True,
        )
        
        # Get collections
        collections = client.get_collections()
        print(f"📁 Found {len(collections.collections)} collections:\n")
        
        total_count = 0
        total_estimated_size = 0
        
        for collection in collections.collections:
            collection_name = collection.name
            
            try:
                # Get count
                count = client.count(collection_name)
                vector_count = count.count
                total_count += vector_count
                
                # Get collection info for vector size
                collection_info = client.get_collection(collection_name)
                vector_size = collection_info.config.params.vectors.size
                
                # Estimate size (rough calculation)
                vector_bytes = vector_size * 4  # 4 bytes per float32
                metadata_bytes = 100  # Rough estimate
                payload_bytes = 500   # Rough estimate
                bytes_per_vector = vector_bytes + metadata_bytes + payload_bytes
                estimated_size = vector_count * bytes_per_vector
                total_estimated_size += estimated_size
                
                print(f"📊 {collection_name}:")
                print(f"   📈 Vectors: {vector_count:,}")
                print(f"   🔢 Dimension: {vector_size}")
                print(f"   💾 Estimated: {format_size(estimated_size)}")
                print()
                
            except Exception as e:
                print(f"❌ Error getting info for {collection_name}: {e}\n")
        
        # Summary
        print("📊 Summary:")
        print(f"   📁 Collections: {len(collections.collections)}")
        print(f"   📈 Total vectors: {total_count:,}")
        print(f"   💾 Total estimated size: {format_size(total_estimated_size)}")
        
    except Exception as e:
        print(f"❌ Error connecting to Qdrant: {e}")
        print("\n💡 Make sure your .env file has:")
        print("   QDRANT_URL=your_qdrant_url")
        print("   QDRANT_API_KEY=your_api_key")

if __name__ == "__main__":
    main() 