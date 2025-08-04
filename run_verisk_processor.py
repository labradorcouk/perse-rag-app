#!/usr/bin/env python3
"""
Script to run the Verisk data processor.

This script processes Verisk data from GPKG and CSV files and saves it to Qdrant.
"""

import sys
import os
import argparse
from pathlib import Path

# Add the utils directory to the Python path
sys.path.append(str(Path(__file__).parent / "utils"))

from verisk_qdrant_processor import VeriskQdrantProcessor

def main():
    """
    Main function to run the Verisk data processor.
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Process Verisk data and save to Qdrant')
    parser.add_argument('--chunk-size', type=int, default=100000, 
                       help='Number of rows to process in each chunk (default: 100000)')
    parser.add_argument('--max-chunks', type=int, default=None,
                       help='Maximum number of chunks to process (for testing, default: None)')
    parser.add_argument('--test-mode', action='store_true',
                       help='Run in test mode with limited chunks')
    parser.add_argument('--skip-geometry', action='store_true',
                       help='Skip geometry processing (useful if geometry data is problematic)')
    parser.add_argument('--qdrant-batch-size', type=int, default=50,
                       help='Batch size for Qdrant upserting (default: 50, reduce for network issues)')
    parser.add_argument('--qdrant-timeout', type=int, default=60,
                       help='Timeout in seconds for Qdrant operations (default: 60)')
    
    args = parser.parse_args()
    
    # Set max_chunks for test mode
    if args.test_mode and args.max_chunks is None:
        args.max_chunks = 5
    
    print("🚀 Starting Verisk Data Processor...")
    print(f"📊 Chunk size: {args.chunk_size:,}")
    if args.max_chunks:
        print(f"🔬 Max chunks: {args.max_chunks} (test mode)")
    else:
        print("🔬 Processing all chunks")
    if args.skip_geometry:
        print("⚠️  Geometry processing will be skipped")
    print(f"📦 Qdrant batch size: {args.qdrant_batch_size}")
    print(f"⏱️  Qdrant timeout: {args.qdrant_timeout}s")
    
    # Check if downloads folder exists
    downloads_folder = Path("downloads")
    if not downloads_folder.exists():
        print(f"❌ Downloads folder not found: {downloads_folder}")
        print("Please create the downloads folder and place your files there:")
        print("  - edition_18_0_new_format.gpkg")
        print("  - UKBuildings_edition_18_abc_link_file.csv")
        return
    
    # Check if required files exist
    gpkg_file = downloads_folder / "edition_18_0_new_format.gpkg"
    csv_file = downloads_folder / "UKBuildings_edition_18_abc_link_file.csv"
    
    if not gpkg_file.exists():
        print(f"❌ GPKG file not found: {gpkg_file}")
        return
        
    if not csv_file.exists():
        print(f"❌ CSV file not found: {csv_file}")
        return
    
    print("✅ Required files found!")
    print(f"📁 GPKG file: {gpkg_file}")
    print(f"📁 CSV file: {csv_file}")
    
    try:
        # Initialize processor
        processor = VeriskQdrantProcessor(
            downloads_folder="downloads",
            qdrant_batch_size=args.qdrant_batch_size,
            qdrant_timeout=args.qdrant_timeout
        )
        
        # Process and save to Qdrant
        print("\n🔄 Processing data...")
        stats = processor.process_and_save(
            chunk_size=args.chunk_size,
            max_chunks=args.max_chunks,
            skip_geometry=args.skip_geometry
        )
        
        print("\n✅ Processing completed successfully!")
        print("📊 Statistics:")
        for key, value in stats.items():
            print(f"   {key}: {value}")
            
    except Exception as e:
        print(f"\n❌ Error during processing: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print("\n🎉 Verisk data has been successfully processed and saved to Qdrant!")
    print(f"📚 Collection name: {stats.get('collection_name', 'verisk_edition_18_raw')}")

if __name__ == "__main__":
    main() 