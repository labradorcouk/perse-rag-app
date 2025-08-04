#!/usr/bin/env python3
"""
Utility script to check Qdrant collection sizes and disk usage.

This script provides various methods to check collection sizes and storage usage.
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv
from qdrant_client import QdrantClient
import json
import requests

load_dotenv()

def get_qdrant_client():
    """Initialize Qdrant client from environment variables."""
    return QdrantClient(
        url=os.getenv("QDRANT_URL"),
        api_key=os.getenv("QDRANT_API_KEY"),
        prefer_grpc=True,
    )

def check_collections_via_api():
    """Check collections using Qdrant REST API (if available)."""
    try:
        qdrant_url = os.getenv("QDRANT_URL")
        api_key = os.getenv("QDRANT_API_KEY")
        
        if not qdrant_url:
            print("❌ QDRANT_URL not found in environment variables")
            return None
            
        # Remove trailing slash if present
        qdrant_url = qdrant_url.rstrip('/')
        
        headers = {}
        if api_key:
            headers["api-key"] = api_key
            
        # Get collections info
        response = requests.get(f"{qdrant_url}/collections", headers=headers)
        if response.status_code == 200:
            collections = response.json()
            return collections
        else:
            print(f"❌ Failed to get collections via API: {response.status_code}")
            return None
            
    except Exception as e:
        print(f"❌ Error checking collections via API: {e}")
        return None

def check_collection_info_via_api(collection_name):
    """Get detailed collection info via REST API."""
    try:
        qdrant_url = os.getenv("QDRANT_URL").rstrip('/')
        api_key = os.getenv("QDRANT_API_KEY")
        
        headers = {}
        if api_key:
            headers["api-key"] = api_key
            
        # Get collection info
        response = requests.get(f"{qdrant_url}/collections/{collection_name}", headers=headers)
        if response.status_code == 200:
            return response.json()
        else:
            print(f"❌ Failed to get collection info for {collection_name}: {response.status_code}")
            return None
            
    except Exception as e:
        print(f"❌ Error getting collection info for {collection_name}: {e}")
        return None

def check_collections_via_client():
    """Check collections using Qdrant Python client."""
    try:
        client = get_qdrant_client()
        collections = client.get_collections()
        return collections
    except Exception as e:
        print(f"❌ Error checking collections via client: {e}")
        return None

def get_collection_stats(client, collection_name):
    """Get collection statistics."""
    try:
        # Get collection info
        collection_info = client.get_collection(collection_name)
        
        # Get collection count
        collection_count = client.count(collection_name)
        
        return {
            "name": collection_name,
            "info": collection_info,
            "count": collection_count
        }
    except Exception as e:
        print(f"❌ Error getting stats for collection {collection_name}: {e}")
        return None

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

def estimate_collection_size(collection_stats):
    """Estimate collection size based on vector dimensions and count."""
    try:
        count = collection_stats["count"]["count"]
        vector_size = collection_stats["info"]["config"]["params"]["vectors"]["size"]
        
        # Estimate size per vector (vector data + metadata + payload)
        # This is a rough estimate - actual size depends on payload complexity
        vector_bytes = vector_size * 4  # 4 bytes per float32
        metadata_bytes = 100  # Rough estimate for metadata
        payload_bytes = 500   # Rough estimate for payload
        
        total_bytes_per_vector = vector_bytes + metadata_bytes + payload_bytes
        total_size = count * total_bytes_per_vector
        
        return {
            "count": count,
            "vector_size": vector_size,
            "estimated_size_bytes": total_size,
            "estimated_size_formatted": format_size(total_size)
        }
    except Exception as e:
        print(f"❌ Error estimating collection size: {e}")
        return None

def main():
    """Main function to check Qdrant collection sizes."""
    print("🔍 Checking Qdrant Collection Sizes...")
    print("=" * 50)
    
    # Method 1: Check via Python client
    print("\n📊 Method 1: Using Qdrant Python Client")
    print("-" * 30)
    
    client = get_qdrant_client()
    if client:
        try:
            collections = client.get_collections()
            print(f"✅ Found {len(collections.collections)} collections:")
            
            total_estimated_size = 0
            
            for collection in collections.collections:
                collection_name = collection.name
                print(f"\n📁 Collection: {collection_name}")
                
                # Get detailed stats
                stats = get_collection_stats(client, collection_name)
                if stats:
                    print(f"   📈 Count: {stats['count']['count']:,} vectors")
                    
                    # Estimate size
                    size_estimate = estimate_collection_size(stats)
                    if size_estimate:
                        print(f"   💾 Estimated size: {size_estimate['estimated_size_formatted']}")
                        print(f"   🔢 Vector dimension: {size_estimate['vector_size']}")
                        total_estimated_size += size_estimate['estimated_size_bytes']
                    
                    # Show collection config
                    config = stats['info']['config']
                    print(f"   ⚙️  Distance: {config['params']['vectors']['distance']}")
                    print(f"   🔧 On disk: {config['params']['vectors']['on_disk']}")
                    
        except Exception as e:
            print(f"❌ Error getting collections: {e}")
    
    # Method 2: Check via REST API
    print("\n📊 Method 2: Using Qdrant REST API")
    print("-" * 30)
    
    collections_api = check_collections_via_api()
    if collections_api:
        print(f"✅ Found {len(collections_api['collections'])} collections via API:")
        
        for collection in collections_api['collections']:
            collection_name = collection['name']
            print(f"\n📁 Collection: {collection_name}")
            
            # Get detailed info via API
            collection_info = check_collection_info_via_api(collection_name)
            if collection_info:
                print(f"   📈 Status: {collection_info['result']['status']}")
                print(f"   💾 Optimizer status: {collection_info['result']['optimizer_status']}")
                
                # Try to get count via API
                try:
                    count_response = requests.get(
                        f"{os.getenv('QDRANT_URL').rstrip('/')}/collections/{collection_name}/points/count",
                        headers={"api-key": os.getenv("QDRANT_API_KEY")} if os.getenv("QDRANT_API_KEY") else {}
                    )
                    if count_response.status_code == 200:
                        count_data = count_response.json()
                        print(f"   📊 Count: {count_data['result']['count']:,} vectors")
                except Exception as e:
                    print(f"   ❌ Could not get count via API: {e}")
    
    # Method 3: Direct disk usage (if accessible)
    print("\n📊 Method 3: Direct Disk Usage Check")
    print("-" * 30)
    print("ℹ️  To check actual disk usage, you can:")
    print("   1. SSH into your Qdrant server")
    print("   2. Navigate to Qdrant storage directory")
    print("   3. Use 'du -sh' command on collection folders")
    print("   4. Example: du -sh /path/to/qdrant/storage/collections/*")
    
    # Summary
    if 'total_estimated_size' in locals() and total_estimated_size > 0:
        print(f"\n📊 Summary:")
        print(f"   💾 Total estimated size: {format_size(total_estimated_size)}")
        print(f"   📁 Total collections: {len(collections.collections)}")
    
    print("\n💡 Tips for monitoring Qdrant storage:")
    print("   • Use 'du -sh' on collection directories for exact disk usage")
    print("   • Monitor collection growth over time")
    print("   • Consider using on_disk=true for large collections")
    print("   • Use collection snapshots for backup and size estimation")

if __name__ == "__main__":
    main() 