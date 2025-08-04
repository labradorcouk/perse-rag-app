# Verisk Data Processor

This utility processes Verisk data from GPKG and CSV files and saves it to Qdrant for vector search.

## Overview

The Verisk Data Processor:

1. **Reads GPKG file in chunks** and extracts the `edition_18_0_new_format` table (memory efficient for large files)
2. **Reads CSV file** (`UKBuildings_edition_18_abc_link_file`)
3. **Joins the tables** on `upn` and `verisk_premise_id`
4. **Processes geometry data** and converts to WKT and GeoJSON formats
5. **Creates embeddings** from text representations of the data
6. **Saves to Qdrant** for vector search capabilities

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements_verisk.txt
```

### 2. Prepare Files

Create a `downloads` folder in your project root and place the following files:

```
downloads/
├── edition_18_0_new_format.gpkg
└── UKBuildings_edition_18_abc_link_file.csv
```

### 3. Configure Environment Variables

Ensure your `.env` file contains:

```env
QDRANT_URL=your_qdrant_url
QDRANT_API_KEY=your_qdrant_api_key
```

## Usage

### Option 1: Run the Script (Recommended)

#### Basic Usage
```bash
python run_verisk_processor.py
```

#### With Custom Chunk Size
```bash
python run_verisk_processor.py --chunk-size 50000
```

#### Test Mode (Process Limited Chunks)
```bash
python run_verisk_processor.py --test-mode
```

#### Custom Test Configuration
```bash
python run_verisk_processor.py --chunk-size 25000 --max-chunks 10
```

### Option 2: Use the Processor Class

```python
from utils.verisk_qdrant_processor import VeriskQdrantProcessor

# Initialize processor
processor = VeriskQdrantProcessor(downloads_folder="downloads")

# Process and save to Qdrant with chunked processing
stats = processor.process_and_save(
    chunk_size=100000,  # Process 100k rows at a time
    max_chunks=None     # Process all chunks (set to number for testing)
)

print(f"Processing time: {stats['processing_time_seconds']:.2f} seconds")
print(f"Collection name: {stats['collection_name']}")
```

## Features

### 🔍 **Memory-Efficient Chunked Processing**
- **Chunked reading**: Processes large GPKG files in configurable chunks
- **Memory management**: Prevents memory overflow with large datasets (82M+ records)
- **Progress tracking**: Detailed logging of chunk processing progress
- **Resumable processing**: Can process specific chunks for testing

### 📊 **Smart Text Generation**
- **Property details**: Includes bedroom count, reception rooms, habitable rooms
- **Address information**: Building name, number, street, locality, town, postcode
- **Energy information**: Energy ratings and efficiency data
- **Identifiers**: UPRN and UPN linking information

### 🚀 **Efficient Processing**
- **Batch processing**: Processes data in configurable batches
- **Progress tracking**: Detailed logging of processing progress
- **Memory efficient**: Handles large datasets without memory issues
- **Error recovery**: Continues processing even if individual chunks fail

### 🔧 **Configurable Options**
- **Chunk size**: Configurable number of rows per chunk (default: 100,000)
- **Max chunks**: Limit processing for testing (default: None = all chunks)
- **Vector size**: Configurable embedding vector size (default: 768)
- **Batch sizes**: Configurable processing and upsert batch sizes
- **Collection name**: Customizable Qdrant collection name

## Data Flow

```
GPKG File → Read in Chunks → Process Geometry → Join with CSV → Create Embeddings → Save to Qdrant
```

## Collection Structure

The Qdrant collection `verisk_edition_18_raw` contains:

- **Vector embeddings**: Generated from text representations
- **Payload data**: All original fields from the joined dataset
- **Metadata**: Processing timestamps and statistics

## Performance

### Memory Usage
- **Chunked processing**: Only loads configurable number of rows at once
- **Efficient joins**: CSV file loaded once, joined with each chunk
- **Batch upserts**: Documents sent to Qdrant in batches

### Processing Speed
- **Chunk size 100k**: ~2-5 minutes per chunk (depending on data complexity)
- **Total processing**: ~27-68 hours for 82M records (820 chunks)
- **Memory usage**: ~2-4GB RAM per chunk (configurable)

### Recommended Settings
- **Production**: `--chunk-size 100000` (default)
- **Testing**: `--test-mode` or `--max-chunks 5`
- **Memory constrained**: `--chunk-size 50000`

## Example Query

After processing, you can query the data using the RAG app:

```python
# Example: Find properties with specific characteristics
query = "Find properties with 3 bedrooms in London with energy rating A"
```

## Troubleshooting

### Common Issues

1. **Memory errors**: Reduce chunk size with `--chunk-size 50000`
2. **File not found**: Ensure files are in the `downloads` folder with correct names
3. **Qdrant connection**: Check your Qdrant URL and API key
4. **Geometry errors**: Ensure GPKG file contains valid geometry data

### Testing

For testing with large files:

```bash
# Test with first 5 chunks (500k records)
python run_verisk_processor.py --test-mode

# Test with custom configuration
python run_verisk_processor.py --chunk-size 25000 --max-chunks 10
```

### Logging

The processor provides detailed logging:

```python
import logging
logging.basicConfig(level=logging.INFO)
```

## Integration

This processor integrates seamlessly with the existing RAG system:

1. **Same Qdrant instance**: Uses the same Qdrant database
2. **Consistent embeddings**: Uses the same embedding model
3. **Unified search**: Can be queried through the RAG app
4. **Collection name**: `verisk_edition_18_raw`

## Support

For issues or questions:
1. Check the logs for detailed error messages
2. Verify file formats and data integrity
3. Ensure all dependencies are installed correctly
4. Use test mode for initial validation 