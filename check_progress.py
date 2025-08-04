#!/usr/bin/env python3
"""
Check Progress and Resume Script for Vector Index Population
"""

import json
import os
from qdrant_client import QdrantClient

def check_progress(offset_file="resume_offsets.json"):
    """Check the current progress for all collections"""
    try:
        with open(offset_file, 'r') as f:
            offsets = json.load(f)
        
        print("Current Progress:")
        print("=" * 50)
        for collection, offset in offsets.items():
            status = "COMPLETED" if offset == -1 else "IN PROGRESS"
            print(f"Collection: {collection}")
            print(f"  Status: {status}")
            if offset >= 0:
                print(f"  Last processed offset: {offset}")
                print(f"  Estimated batches processed: {offset // 1000}")
            print()
        
        return offsets
    except FileNotFoundError:
        print("No progress file found. Starting from beginning.")
        return {}

def check_qdrant_collections():
    """Check the current state of Qdrant collections"""
    QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
    QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "")
    
    try:
        client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
        collections = client.get_collections()
        
        print("Qdrant Collections Status:")
        print("=" * 50)
        for collection in collections.collections:
            try:
                info = client.get_collection(collection.name)
                print(f"Collection: {collection.name}")
                print(f"  Points: {info.points_count}")
                print(f"  Status: {info.status}")
                print()
            except Exception as e:
                print(f"Collection: {collection.name}")
                print(f"  Error: {e}")
                print()
    except Exception as e:
        print(f"Error connecting to Qdrant: {e}")

def reset_progress(collection_name=None, offset_file="resume_offsets.json"):
    """Reset progress for a specific collection or all collections"""
    try:
        with open(offset_file, 'r') as f:
            offsets = json.load(f)
    except FileNotFoundError:
        offsets = {}
    
    if collection_name:
        if collection_name in offsets:
            del offsets[collection_name]
            print(f"Reset progress for collection: {collection_name}")
        else:
            print(f"Collection {collection_name} not found in progress file")
    else:
        offsets = {}
        print("Reset progress for all collections")
    
    with open(offset_file, 'w') as f:
        json.dump(offsets, f, indent=2)

def set_offset(collection_name, offset, offset_file="resume_offsets.json"):
    """Set a specific offset for a collection"""
    try:
        with open(offset_file, 'r') as f:
            offsets = json.load(f)
    except FileNotFoundError:
        offsets = {}
    
    offsets[collection_name] = offset
    
    with open(offset_file, 'w') as f:
        json.dump(offsets, f, indent=2)
    
    print(f"Set offset for {collection_name} to {offset}")

def mark_complete(collection_name, offset_file="resume_offsets.json"):
    """Mark a collection as complete"""
    try:
        with open(offset_file, 'r') as f:
            offsets = json.load(f)
    except FileNotFoundError:
        offsets = {}
    
    offsets[collection_name] = -1
    
    with open(offset_file, 'w') as f:
        json.dump(offsets, f, indent=2)
    
    print(f"Marked collection {collection_name} as complete")

def show_usage():
    """Show usage information"""
    print("Usage:")
    print("  python check_progress.py check                    - Check current progress")
    print("  python check_progress.py reset [collection]      - Reset progress for all or specific collection")
    print("  python check_progress.py set <collection> <offset> - Set specific offset for collection")
    print("  python check_progress.py complete <collection>   - Mark collection as complete")
    print("  python check_progress.py resume <collection>     - Resume specific collection")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        command = sys.argv[1]
        
        if command == "check":
            check_progress()
            check_qdrant_collections()
        
        elif command == "reset":
            collection = sys.argv[2] if len(sys.argv) > 2 else None
            reset_progress(collection)
        
        elif command == "set":
            if len(sys.argv) >= 4:
                collection = sys.argv[2]
                offset = int(sys.argv[3])
                set_offset(collection, offset)
            else:
                print("Usage: python check_progress.py set <collection_name> <offset>")
        
        elif command == "complete":
            if len(sys.argv) >= 3:
                collection = sys.argv[2]
                mark_complete(collection)
            else:
                print("Usage: python check_progress.py complete <collection_name>")
        
        elif command == "resume":
            if len(sys.argv) >= 3:
                collection = sys.argv[2]
                print(f"To resume collection '{collection}', run:")
                print(f"python -m utils.vector_index_populator --table {collection}")
            else:
                print("Usage: python check_progress.py resume <collection_name>")
        
        else:
            print("Unknown command.")
            show_usage()
    else:
        print("Checking progress...")
        check_progress()
        check_qdrant_collections() 