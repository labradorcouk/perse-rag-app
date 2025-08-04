#!/usr/bin/env python3
"""
Qdrant Data Migration Script
This script migrates collections from local Qdrant to Azure Qdrant
"""

import os
import json
import requests
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
import time

def migrate_collections(local_url, azure_url, collections_to_migrate=None):
    """
    Migrate collections from local Qdrant to Azure Qdrant
    
    Args:
        local_url: URL of local Qdrant instance
        azure_url: URL of Azure Qdrant instance
        collections_to_migrate: List of collection names to migrate (None = all)
    """
    
    # Connect to local Qdrant
    local_client = QdrantClient(local_url)
    
    # Connect to Azure Qdrant
    azure_client = QdrantClient(azure_url)
    
    # Get all collections from local Qdrant
    local_collections = local_client.get_collections()
    print(f"Found {len(local_collections.collections)} collections in local Qdrant")
    
    # Filter collections if specified
    if collections_to_migrate:
        local_collections.collections = [
            col for col in local_collections.collections 
            if col.name in collections_to_migrate
        ]
        print(f"Migrating {len(local_collections.collections)} specified collections")
    
    for collection_info in local_collections.collections:
        collection_name = collection_info.name
        print(f"\nMigrating collection: {collection_name}")
        
        try:
            # Get collection info from local
            local_collection = local_client.get_collection(collection_name)
            print(f"  Collection config: {local_collection.config}")
            
            # Create collection in Azure (if it doesn't exist)
            try:
                azure_client.get_collection(collection_name)
                print(f"  Collection {collection_name} already exists in Azure")
            except:
                print(f"  Creating collection {collection_name} in Azure...")
                azure_client.create_collection(
                    collection_name=collection_name,
                    vectors_config=VectorParams(
                        size=local_collection.config.params.vectors.size,
                        distance=Distance.COSINE
                    )
                )
                print(f"  Collection {collection_name} created successfully")
            
            # Migrate points in batches
            batch_size = 1000
            offset = 0
            total_points = 0
            
            while True:
                # Get points from local Qdrant
                points = local_client.scroll(
                    collection_name=collection_name,
                    limit=batch_size,
                    offset=offset,
                    with_payload=True,
                    with_vectors=True
                )
                
                if not points[0]:  # No more points
                    break
                
                # Prepare points for Azure
                azure_points = []
                for point in points[0]:
                    azure_points.append({
                        'id': point.id,
                        'vector': point.vector,
                        'payload': point.payload
                    })
                
                # Upload to Azure
                azure_client.upsert(
                    collection_name=collection_name,
                    points=azure_points
                )
                
                total_points += len(azure_points)
                offset += batch_size
                print(f"  Migrated {total_points} points so far...")
                
                # Small delay to avoid overwhelming the service
                time.sleep(0.1)
            
            print(f"  ✅ Successfully migrated {total_points} points for collection {collection_name}")
            
        except Exception as e:
            print(f"  ❌ Error migrating collection {collection_name}: {e}")
            continue
    
    print("\nMigration completed!")

def export_collection_snapshot(local_url, collection_name, snapshot_path):
    """
    Export a collection snapshot from local Qdrant
    """
    try:
        # Create snapshot
        snapshot_info = requests.post(
            f"{local_url}/collections/{collection_name}/snapshots"
        ).json()
        
        snapshot_name = snapshot_info['result']['name']
        
        # Download snapshot
        snapshot_data = requests.get(
            f"{local_url}/collections/{collection_name}/snapshots/{snapshot_name}"
        ).content
        
        # Save to file
        with open(snapshot_path, 'wb') as f:
            f.write(snapshot_data)
        
        print(f"Snapshot saved to: {snapshot_path}")
        return snapshot_path
        
    except Exception as e:
        print(f"Error creating snapshot: {e}")
        return None

def import_collection_snapshot(azure_url, collection_name, snapshot_path):
    """
    Import a collection snapshot to Azure Qdrant
    """
    try:
        # Upload snapshot
        with open(snapshot_path, 'rb') as f:
            snapshot_data = f.read()
        
        # Upload to Azure
        response = requests.put(
            f"{azure_url}/collections/{collection_name}/snapshots",
            data=snapshot_data,
            headers={'Content-Type': 'application/octet-stream'}
        )
        
        if response.status_code == 200:
            print(f"Snapshot imported successfully to collection {collection_name}")
            return True
        else:
            print(f"Error importing snapshot: {response.text}")
            return False
            
    except Exception as e:
        print(f"Error importing snapshot: {e}")
        return False

if __name__ == "__main__":
    # Configuration
    LOCAL_QDRANT_URL = "http://localhost:6333"
    AZURE_QDRANT_URL = os.getenv("AZURE_QDRANT_URL", "http://your-azure-qdrant-ip:6333")
    
    # Collections to migrate (None = all collections)
    COLLECTIONS_TO_MIGRATE = [
        "epc_non_domestic_scotland",
        "epc_domestic_scotland",
        "verisk_edition_18"
    ]
    
    print("Qdrant Data Migration Tool")
    print("=" * 40)
    
    # Check if Azure URL is configured
    if AZURE_QDRANT_URL == "http://your-azure-qdrant-ip:6333":
        print("Please set the AZURE_QDRANT_URL environment variable")
        print("Example: export AZURE_QDRANT_URL=http://your-ip:6333")
        exit(1)
    
    print(f"Local Qdrant: {LOCAL_QDRANT_URL}")
    print(f"Azure Qdrant: {AZURE_QDRANT_URL}")
    print(f"Collections to migrate: {COLLECTIONS_TO_MIGRATE}")
    
    # Confirm before proceeding
    response = input("\nProceed with migration? (y/n): ")
    if response.lower() != 'y':
        print("Migration cancelled.")
        exit(0)
    
    # Start migration
    migrate_collections(
        local_url=LOCAL_QDRANT_URL,
        azure_url=AZURE_QDRANT_URL,
        collections_to_migrate=COLLECTIONS_TO_MIGRATE
    ) 