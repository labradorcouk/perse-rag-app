import pandas as pd
import geopandas as gpd
import sqlite3
import os
import time
from pathlib import Path
from typing import Optional, Dict, Any
import logging
from qdrant_client import QdrantClient, models
from qdrant_client.http.models import Distance, VectorParams
from fastembed.embedding import DefaultEmbedding
from sentence_transformers import SentenceTransformer
import numpy as np
from dotenv import load_dotenv

load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VeriskQdrantProcessor:
    """
    Utility for processing Verisk data and saving it to Qdrant database.
    
    This processor:
    1. Reads GPKG file and extracts edition_18_0_new_format table
    2. Reads CSV file (ukbuildings_edition_18_abc_link_file)
    3. Joins the tables on upn and verisk_premise_id
    4. Processes geometry data and creates embeddings
    5. Saves the combined data to Qdrant
    """
    
    def __init__(self, downloads_folder: str = "downloads", qdrant_batch_size: int = 50, qdrant_timeout: int = 60):
        """
        Initialize the processor.
        
        Args:
            downloads_folder: Path to the downloads folder containing the files
            qdrant_batch_size: Batch size for Qdrant upserting (smaller for network issues)
            qdrant_timeout: Timeout in seconds for Qdrant operations
        """
        self.downloads_folder = Path(downloads_folder)
        self.qdrant_client = QdrantClient(
            url=os.getenv("QDRANT_URL"),
            api_key=os.getenv("QDRANT_API_KEY"),
            prefer_grpc=True,
        )
        self.embedding_model = DefaultEmbedding()
        self.collection_name = "verisk_edition_18_raw"
        self.qdrant_batch_size = qdrant_batch_size
        self.qdrant_timeout = qdrant_timeout
        
    def read_gpkg_table_chunked(self, gpkg_file: str, table_name: str = "edition_18_0_new_format", chunk_size: int = 100000) -> pd.DataFrame:
        """
        Read a specific table from a GPKG file in chunks for memory efficiency.
        
        Args:
            gpkg_file: Name of the GPKG file
            table_name: Name of the table to extract
            chunk_size: Number of rows to read in each chunk
            
        Returns:
            DataFrame containing the table data (first chunk for preview)
        """
        gpkg_path = self.downloads_folder / gpkg_file
        
        if not gpkg_path.exists():
            raise FileNotFoundError(f"GPKG file not found: {gpkg_path}")
            
        logger.info(f"Reading table '{table_name}' from {gpkg_file} in chunks of {chunk_size}")
        
        try:
            # Use geopandas to read in chunks
            chunks = []
            total_rows = 0
            
            # Read the first chunk to get column information
            first_chunk = gpd.read_file(gpkg_path, layer=table_name, rows=chunk_size)
            chunks.append(first_chunk)
            total_rows += len(first_chunk)
            
            # Continue reading chunks
            offset = chunk_size
            while True:
                try:
                    chunk = gpd.read_file(gpkg_path, layer=table_name, rows=chunk_size, skip=offset)
                    if len(chunk) == 0:
                        break
                    chunks.append(chunk)
                    total_rows += len(chunk)
                    offset += chunk_size
                    
                    if total_rows % (chunk_size * 10) == 0:
                        logger.info(f"Read {total_rows:,} rows so far...")
                        
                except Exception as e:
                    logger.warning(f"Error reading chunk at offset {offset}: {e}")
                    break
            
            logger.info(f"Successfully read {total_rows:,} total rows from {table_name}")
            
            # Return the first chunk for preview and processing
            return first_chunk
            
        except Exception as e:
            logger.error(f"Error reading GPKG table: {e}")
            raise
    
    def read_csv_file(self, csv_file: str) -> pd.DataFrame:
        """
        Read a CSV file.
        
        Args:
            csv_file: Name of the CSV file
            
        Returns:
            DataFrame containing the CSV data
        """
        csv_path = self.downloads_folder / csv_file
        
        if not csv_path.exists():
            raise FileNotFoundError(f"CSV file not found: {csv_path}")
            
        logger.info(f"Reading CSV file: {csv_file}")
        
        try:
            df = pd.read_csv(csv_path)
            logger.info(f"Successfully read {len(df)} rows from {csv_file}")
            return df
        except Exception as e:
            logger.error(f"Error reading CSV file: {e}")
            raise
    
    def process_geometry_data(self, gdf: gpd.GeoDataFrame) -> pd.DataFrame:
        """
        Process geometry data and convert to appropriate formats.
        
        Args:
            gdf: GeoDataFrame with geometry column
            
        Returns:
            DataFrame with processed geometry data
        """
        logger.info("Processing geometry data...")
        
        # Convert to regular DataFrame
        df = pd.DataFrame(gdf)
        
        # Convert geometry to WKT (Well-Known Text) format
        if 'geometry' in df.columns:
            df['geom_wkt'] = df['geometry'].astype(str)
            
        # Convert geometry to GeoJSON format
        if 'geometry' in df.columns:
            df['geom_geojson'] = df['geometry'].apply(lambda geom: geom.__geo_interface__ if geom else None)
            
        # Drop the original geometry column to avoid serialization issues
        if 'geometry' in df.columns:
            df = df.drop(columns=['geometry'])
            
        logger.info("Geometry processing completed")
        return df
    
    def join_tables(self, new_format_df: pd.DataFrame, abc_link_df: pd.DataFrame) -> pd.DataFrame:
        """
        Join the new format table with the ABC link file.
        
        Args:
            new_format_df: DataFrame from edition_18_0_new_format (main table)
            abc_link_df: DataFrame from ukbuildings_edition_18_abc_link_file (additional columns)
            
        Returns:
            Joined DataFrame with all records from main table
        """
        logger.info("Joining tables...")
        
        # Ensure we have the required columns
        if 'verisk_premise_id' not in new_format_df.columns:
            raise ValueError("Column 'verisk_premise_id' not found in new format table")
        if 'upn' not in abc_link_df.columns:
            raise ValueError("Column 'upn' not found in ABC link file")
            
        # Perform LEFT JOIN to preserve all records from main table
        # The CSV link file provides additional columns only when available
        joined_df = new_format_df.merge(
            abc_link_df,
            left_on='verisk_premise_id',
            right_on='upn',
            how='left'  # Keep ALL records from main table
        )
        
        # Rename columns for clarity
        if 'uprn' in joined_df.columns:
            joined_df = joined_df.rename(columns={'uprn': 'uprn_link'})
        if 'upn' in joined_df.columns:
            joined_df = joined_df.rename(columns={'upn': 'upn_link'})
        
        # Log join statistics
        total_main_records = len(new_format_df)
        matched_records = len(joined_df[joined_df['upn_link'].notna()])
        unmatched_records = total_main_records - matched_records
        
        logger.info(f"Join completed:")
        logger.info(f"  - Total records from main table: {total_main_records:,}")
        logger.info(f"  - Records with matching link data: {matched_records:,}")
        logger.info(f"  - Records without link data: {unmatched_records:,}")
        logger.info(f"  - Final joined records: {len(joined_df):,}")
        
        return joined_df
    
    def create_text_for_embedding(self, row: pd.Series) -> str:
        """
        Create a text representation of a row for embedding generation.
        
        Args:
            row: DataFrame row
            
        Returns:
            Text string for embedding
        """
        text_parts = []
        
        # Add key identifiers (may be missing due to left join)
        if pd.notna(row.get('uprn_link')):
            text_parts.append(f"UPRN: {row['uprn_link']}")
        if pd.notna(row.get('upn_link')):
            text_parts.append(f"UPN: {row['upn_link']}")
        if pd.notna(row.get('verisk_premise_id')):
            text_parts.append(f"Verisk Premise ID: {row['verisk_premise_id']}")
            
        # Add property details
        if pd.notna(row.get('premise_type')):
            text_parts.append(f"Property type: {row['premise_type']}")
        if pd.notna(row.get('bedroom_count')):
            text_parts.append(f"Bedrooms: {row['bedroom_count']}")
        if pd.notna(row.get('reception_room_count')):
            text_parts.append(f"Reception rooms: {row['reception_room_count']}")
        if pd.notna(row.get('habitable_rooms')):
            text_parts.append(f"Habitable rooms: {row['habitable_rooms']}")
            
        # Add address information
        address_parts = []
        for field in ['building_name', 'building_number', 'street_name', 'locality', 'town', 'postcode']:
            if pd.notna(row.get(field)):
                address_parts.append(str(row[field]))
        
        if address_parts:
            text_parts.append(f"Address: {' '.join(address_parts)}")
            
        # Add energy information
        if pd.notna(row.get('energy_rating')):
            text_parts.append(f"Energy rating: {row['energy_rating']}")
            
        return " | ".join(text_parts)
    
    def prepare_qdrant_documents(self, df: pd.DataFrame, batch_size: int = 1000, start_id: int = 0) -> list:
        """
        Prepare documents for Qdrant insertion.
        
        Args:
            df: DataFrame to process
            batch_size: Number of documents to process in each batch
            start_id: Starting ID for document IDs in the batch
            
        Returns:
            List of documents ready for Qdrant
        """
        logger.info("Preparing documents for Qdrant...")
        
        documents = []
        total_rows = len(df)
        
        for i in range(0, total_rows, batch_size):
            batch_df = df.iloc[i:i + batch_size]
            
            for batch_idx, (df_idx, row) in enumerate(batch_df.iterrows()):
                try:
                    # Create text for embedding
                    text = self.create_text_for_embedding(row)
                    
                    # Generate embedding using the correct method
                    embedding_generator = self.embedding_model.embed(text)
                    # Convert generator to list, then to numpy array
                    embedding = list(embedding_generator)[0]  # Get first (and only) embedding
                    
                    # Prepare payload (exclude embedding and raw geometry bytes)
                    payload = row.to_dict()
                    
                    # Remove raw geometry bytes that can't be serialized to JSON
                    if 'geom' in payload:
                        del payload['geom']  # Remove raw BSON bytes
                    
                    # Clean payload for JSON serialization
                    payload = self.clean_payload_for_json(payload)
                    
                    # Create document with unique ID
                    document = {
                        "id": start_id + i + batch_idx,  # Use global ID counter
                        "vector": embedding.tolist(),
                        "payload": payload
                    }
                    
                    documents.append(document)
                    
                except Exception as e:
                    logger.warning(f"Error processing row {df_idx}: {e}")
                    continue
                    
            logger.info(f"Processed batch {i//batch_size + 1}/{(total_rows + batch_size - 1)//batch_size}")
            
        logger.info(f"Prepared {len(documents)} documents for Qdrant")
        return documents
    
    def create_qdrant_collection(self, vector_size: int = None):
        """
        Create the Qdrant collection with the correct vector size.
        
        Args:
            vector_size: Size of the embedding vectors (if None, will be detected)
        """
        logger.info(f"Creating Qdrant collection: {self.collection_name}")
        
        # If vector_size not provided, detect it from the embedding model
        if vector_size is None:
            # Generate a test embedding to get the dimension
            test_embedding = list(self.embedding_model.embed("test"))[0]
            vector_size = len(test_embedding)
            logger.info(f"Detected embedding dimension: {vector_size}")
        
        try:
            self.qdrant_client.recreate_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
            )
            logger.info(f"Collection '{self.collection_name}' created successfully with dimension {vector_size}")
        except Exception as e:
            logger.error(f"Error creating collection: {e}")
            raise
    
    def upsert_to_qdrant(self, documents: list, batch_size: int = None, max_retries: int = 3):
        """
        Upsert documents to Qdrant with retry logic and smaller batch sizes.
        
        Args:
            documents: List of documents to upsert
            batch_size: Number of documents to upsert in each batch (uses instance default if None)
            max_retries: Maximum number of retries for failed batches
        """
        if batch_size is None:
            batch_size = self.qdrant_batch_size
            
        logger.info(f"Upserting {len(documents)} documents to Qdrant...")
        
        total_documents = len(documents)
        
        for i in range(0, total_documents, batch_size):
            batch = documents[i:i + batch_size]
            batch_num = i//batch_size + 1
            total_batches = (total_documents + batch_size - 1)//batch_size
            
            # Retry logic for each batch
            for retry_attempt in range(max_retries + 1):
                try:
                    # Convert documents to proper Qdrant PointStruct format
                    points = []
                    for doc in batch:
                        try:
                            point = models.PointStruct(
                                id=doc["id"],
                                vector=doc["vector"],
                                payload=doc["payload"]
                            )
                            points.append(point)
                        except Exception as doc_error:
                            logger.warning(f"Error creating PointStruct for document {doc.get('id', 'unknown')}: {doc_error}")
                            # Log problematic payload keys
                            if 'payload' in doc:
                                problematic_keys = [k for k, v in doc['payload'].items() if isinstance(v, bytes)]
                                if problematic_keys:
                                    logger.warning(f"Problematic bytes keys in payload: {problematic_keys}")
                            continue
                    
                    if points:
                        # Use longer timeout and wait for completion
                        self.qdrant_client.upsert(
                            collection_name=self.collection_name,
                            points=points,
                            wait=True,
                            timeout=self.qdrant_timeout
                        )
                        logger.info(f"Upserted batch {batch_num}/{total_batches} (attempt {retry_attempt + 1})")
                        break  # Success, exit retry loop
                    else:
                        logger.warning(f"No valid points to upsert in batch {batch_num}")
                        break
                        
                except Exception as e:
                    if retry_attempt < max_retries:
                        wait_time = (2 ** retry_attempt) * 5  # Exponential backoff: 5s, 10s, 20s
                        logger.warning(f"Error upserting batch {batch_num} (attempt {retry_attempt + 1}/{max_retries + 1}): {e}")
                        logger.info(f"Retrying in {wait_time} seconds...")
                        time.sleep(wait_time)
                    else:
                        logger.error(f"Failed to upsert batch {batch_num} after {max_retries + 1} attempts: {e}")
                        # Log more details about the error
                        if "Deadline Exceeded" in str(e):
                            logger.error("Qdrant timeout error - consider reducing batch size or checking network connection")
                        elif "Not supported json value" in str(e):
                            logger.error("JSON serialization error - check for unsupported data types in payload")
                        raise
                
        logger.info("Upsert completed successfully")
    
    def process_gpkg_in_chunks(self, gpkg_file: str, table_name: str = "edition_18_0_new_format", 
                              chunk_size: int = 100000, max_chunks: int = None, 
                              skip_geometry: bool = False) -> pd.DataFrame:
        """
        Process GPKG file in chunks and join with CSV data efficiently.
        
        Args:
            gpkg_file: Name of the GPKG file
            table_name: Name of the table to extract
            chunk_size: Number of rows to process in each chunk
            max_chunks: Maximum number of chunks to process (for testing)
            skip_geometry: Skip geometry processing if True
            
        Returns:
            DataFrame containing the joined data
        """
        gpkg_path = self.downloads_folder / gpkg_file
        
        if not gpkg_path.exists():
            raise FileNotFoundError(f"GPKG file not found: {gpkg_path}")
            
        logger.info(f"Processing GPKG file in chunks of {chunk_size}")
        if skip_geometry:
            logger.info("Geometry processing will be skipped")
        
        # Read CSV file once (it's smaller)
        csv_file = "UKBuildings_edition_18_abc_link_file.csv"
        abc_link_df = self.read_csv_file(csv_file)
        logger.info(f"Loaded {len(abc_link_df):,} rows from CSV file")
        
        # Create Qdrant collection
        self.create_qdrant_collection()
        
        total_processed = 0
        total_documents = 0
        chunk_count = 0
        global_document_id = 0  # Global counter for unique document IDs
        
        try:
            # Use SQL query to read chunks efficiently
            import sqlite3
            
            # Connect to the GPKG file (it's a SQLite database)
            conn = sqlite3.connect(gpkg_path)
            
            # Get total row count
            total_rows_query = f"SELECT COUNT(*) FROM '{table_name}'"
            total_rows = conn.execute(total_rows_query).fetchone()[0]
            logger.info(f"Total rows in GPKG file: {total_rows:,}")
            
            # Process in chunks using SQL LIMIT and OFFSET
            for offset in range(0, total_rows, chunk_size):
                if max_chunks and chunk_count >= max_chunks:
                    logger.info(f"Reached maximum chunks limit ({max_chunks})")
                    break
                    
                chunk_count += 1
                logger.info(f"Processing chunk {chunk_count}: reading rows {offset:,}-{min(offset + chunk_size, total_rows):,}")
                
                # Read chunk using SQL
                chunk_query = f"""
                SELECT * FROM '{table_name}' 
                LIMIT {chunk_size} OFFSET {offset}
                """
                
                try:
                    chunk_df = pd.read_sql_query(chunk_query, conn)
                    
                    if len(chunk_df) == 0:
                        logger.info("No more rows to process")
                        break
                    
                    logger.info(f"Read {len(chunk_df):,} rows from chunk {chunk_count}")
                    
                    # Convert to GeoDataFrame if geometry column exists and not skipping geometry
                    if 'geom' in chunk_df.columns and not skip_geometry:
                        # Convert WKB geometry to GeoDataFrame with robust error handling
                        import shapely.wkb
                        import shapely.geometry
                        import struct
                        
                        def robust_parse_geometry(wkb_data):
                            """Robustly parse WKB geometry with support for BSON format and coordinate transformation"""
                            try:
                                if wkb_data is None or pd.isna(wkb_data):
                                    return None
                                
                                # Handle BSON format (like in the source code)
                                if isinstance(wkb_data, bytes) and len(wkb_data) > 40:
                                    try:
                                        # Skip first 40 bytes (BSON header) and extract WKB
                                        wkb_bytes = wkb_data[40:]
                                        
                                        # Parse the WKB geometry
                                        geometry = shapely.wkb.loads(wkb_bytes)
                                        
                                        # Transform coordinates from EPSG:27700 (British National Grid) to EPSG:4326 (WGS84)
                                        from pyproj import Transformer
                                        from shapely.ops import transform
                                        
                                        transformer = Transformer.from_crs("EPSG:27700", "EPSG:4326", always_xy=True)
                                        transformed_geometry = transform(transformer.transform, geometry)
                                        
                                        return transformed_geometry
                                        
                                    except Exception as bson_e:
                                        logger.debug(f"BSON parsing failed: {bson_e}")
                                
                                # Try standard WKB parsing first
                                try:
                                    return shapely.wkb.loads(wkb_data)
                                except Exception as e:
                                    logger.debug(f"Standard WKB parsing failed: {e}")
                                
                                # Try parsing as hex string
                                if isinstance(wkb_data, str) and wkb_data.startswith('0x'):
                                    try:
                                        return shapely.wkb.loads(wkb_data, hex=True)
                                    except Exception as e:
                                        logger.debug(f"Hex WKB parsing failed: {e}")
                                
                                # Try parsing as bytes
                                if isinstance(wkb_data, bytes):
                                    try:
                                        return shapely.wkb.loads(wkb_data)
                                    except Exception as e:
                                        logger.debug(f"Bytes WKB parsing failed: {e}")
                                
                                # Custom WKB type handling for type 80 and other custom types
                                if isinstance(wkb_data, bytes) and len(wkb_data) >= 5:
                                    try:
                                        # Parse WKB header manually
                                        byte_order = struct.unpack('<B', wkb_data[0:1])[0]
                                        is_little_endian = byte_order == 1
                                        
                                        # Read geometry type
                                        if is_little_endian:
                                            geom_type = struct.unpack('<I', wkb_data[1:5])[0]
                                        else:
                                            geom_type = struct.unpack('>I', wkb_data[1:5])[0]
                                        
                                        logger.debug(f"Detected geometry type: {geom_type}")
                                        
                                        # Handle custom geometry types by converting to known types
                                        if geom_type == 80:  # Custom type 80
                                            logger.warning(f"Custom geometry type 80 detected, attempting to extract coordinates")
                                            
                                            # Try to extract coordinates from the WKB data
                                            try:
                                                coord_bytes = wkb_data[5:]
                                                
                                                # Try different coordinate extraction strategies
                                                if len(coord_bytes) >= 16:  # Minimum for 2D point
                                                    # Strategy 1: Try to extract as 2D point
                                                    try:
                                                        if is_little_endian:
                                                            x = struct.unpack('<d', coord_bytes[0:8])[0]
                                                            y = struct.unpack('<d', coord_bytes[8:16])[0]
                                                        else:
                                                            x = struct.unpack('>d', coord_bytes[0:8])[0]
                                                            y = struct.unpack('>d', coord_bytes[8:16])[0]
                                                        
                                                        # Validate coordinates (reasonable bounds)
                                                        if -180 <= x <= 180 and -90 <= y <= 90:
                                                            return shapely.geometry.Point(x, y)
                                                        else:
                                                            logger.warning(f"Extracted coordinates out of bounds: {x}, {y}")
                                                    except Exception as point_e:
                                                        logger.debug(f"2D point extraction failed: {point_e}")
                                                
                                                # Strategy 2: Try to extract as 3D point
                                                if len(coord_bytes) >= 24:  # Minimum for 3D point
                                                    try:
                                                        if is_little_endian:
                                                            x = struct.unpack('<d', coord_bytes[0:8])[0]
                                                            y = struct.unpack('<d', coord_bytes[8:16])[0]
                                                            z = struct.unpack('<d', coord_bytes[16:24])[0]
                                                        else:
                                                            x = struct.unpack('>d', coord_bytes[0:8])[0]
                                                            y = struct.unpack('>d', coord_bytes[8:16])[0]
                                                            z = struct.unpack('>d', coord_bytes[16:24])[0]
                                                        
                                                        # Validate coordinates
                                                        if -180 <= x <= 180 and -90 <= y <= 90:
                                                            return shapely.geometry.Point(x, y)
                                                        else:
                                                            logger.warning(f"Extracted 3D coordinates out of bounds: {x}, {y}, {z}")
                                                    except Exception as point3d_e:
                                                        logger.debug(f"3D point extraction failed: {point3d_e}")
                                                
                                                # Strategy 3: Try to find any valid coordinate pattern
                                                # Look for double precision numbers in the byte stream
                                                for i in range(0, len(coord_bytes) - 15, 8):
                                                    try:
                                                        if is_little_endian:
                                                            x = struct.unpack('<d', coord_bytes[i:i+8])[0]
                                                            y = struct.unpack('<d', coord_bytes[i+8:i+16])[0]
                                                        else:
                                                            x = struct.unpack('>d', coord_bytes[i:i+8])[0]
                                                            y = struct.unpack('>d', coord_bytes[i+8:i+16])[0]
                                                        
                                                        # Check if these look like valid coordinates
                                                        if -180 <= x <= 180 and -90 <= y <= 90:
                                                            logger.info(f"Found valid coordinates at offset {i}: {x}, {y}")
                                                            return shapely.geometry.Point(x, y)
                                                    except Exception as pattern_e:
                                                        continue
                                                        
                                            except Exception as coord_e:
                                                logger.warning(f"Could not extract coordinates from custom geometry: {coord_e}")
                                        
                                        # For other custom types, try standard parsing again
                                        else:
                                            return shapely.wkb.loads(wkb_data)
                                            
                                    except Exception as custom_e:
                                        logger.warning(f"Custom geometry parsing failed: {custom_e}")
                                
                                # If all else fails, create a default geometry
                                logger.warning(f"All geometry parsing methods failed, creating default geometry")
                                return shapely.geometry.Point(0, 0)
                                
                            except Exception as e:
                                logger.warning(f"Geometry parsing completely failed: {e}")
                                return None
                        
                        # Apply robust parsing to geometry column
                        chunk_df['geometry'] = chunk_df['geom'].apply(robust_parse_geometry)
                        
                        # Log BSON parsing statistics
                        bson_success = chunk_df['geometry'].notna().sum()
                        bson_total = len(chunk_df)
                        logger.info(f"BSON geometry parsing: {bson_success}/{bson_total} successful")
                        
                        # Remove rows with completely invalid geometry
                        valid_geometry_mask = chunk_df['geometry'].notna()
                        invalid_count = (~valid_geometry_mask).sum()
                        if invalid_count > 0:
                            logger.warning(f"Skipping {invalid_count} rows with invalid geometry")
                        
                        chunk_df = chunk_df[valid_geometry_mask]
                        
                        # Additional validation: check for reasonable geometries (but allow transformed coordinates)
                        if len(chunk_df) > 0:
                            # Filter out geometries that are clearly invalid
                            def is_valid_geometry(geom):
                                if geom is None:
                                    return False
                                try:
                                    # Check if geometry has valid bounds
                                    if hasattr(geom, 'bounds'):
                                        bounds = geom.bounds
                                        # Allow coordinates in WGS84 range (transformed from EPSG:27700)
                                        if bounds[0] >= -10 and bounds[2] <= 10 and bounds[1] >= 49 and bounds[3] <= 61:
                                            return True  # Valid UK coordinates in WGS84
                                        elif bounds[0] == bounds[2] and bounds[1] == bounds[3]:
                                            return False  # Point geometry with same coordinates
                                    return True
                                except Exception:
                                    return False
                            
                            # Apply additional validation
                            valid_geom_mask = chunk_df['geometry'].apply(is_valid_geometry)
                            additional_invalid = (~valid_geom_mask).sum()
                            if additional_invalid > 0:
                                logger.warning(f"Removing {additional_invalid} additional rows with invalid geometries")
                            
                            chunk_df = chunk_df[valid_geom_mask]
                        
                        if len(chunk_df) > 0:
                            chunk_gdf = gpd.GeoDataFrame(chunk_df, geometry='geometry')
                            logger.info(f"Successfully created GeoDataFrame with {len(chunk_df)} valid geometries")
                        else:
                            # If no valid geometry, treat as regular DataFrame
                            chunk_gdf = chunk_df
                            logger.warning("No valid geometry found in chunk, treating as regular DataFrame")
                    else:
                        chunk_gdf = chunk_df
                    
                    # Process geometry data for this chunk (if not skipping)
                    if not skip_geometry:
                        processed_chunk = self.process_geometry_data(chunk_gdf)
                    else:
                        processed_chunk = chunk_gdf
                    
                    # Join with CSV data
                    joined_chunk = self.join_tables(processed_chunk, abc_link_df)
                    
                    if len(joined_chunk) > 0:
                        # Prepare documents for this chunk with unique IDs
                        documents = self.prepare_qdrant_documents(joined_chunk, batch_size=1000, start_id=global_document_id)
                        
                        # Update global document ID counter
                        global_document_id += len(documents)
                        
                        # Upsert to Qdrant with smaller batch size and retry logic
                        try:
                            self.upsert_to_qdrant(documents, batch_size=None, max_retries=3)
                            
                            total_processed += len(joined_chunk)
                            total_documents += len(documents)
                            
                            logger.info(f"Chunk {chunk_count} completed: {len(joined_chunk):,} joined rows, {len(documents):,} documents (ID range: {global_document_id - len(documents)}-{global_document_id - 1})")
                            
                        except Exception as upsert_error:
                            logger.error(f"Failed to upsert chunk {chunk_count} after retries: {upsert_error}")
                            # Continue processing next chunk instead of stopping
                            logger.info(f"Continuing with next chunk...")
                            continue
                    else:
                        logger.info(f"Chunk {chunk_count}: No matching records found")
                    
                    total_processed += len(chunk_df)
                    
                except Exception as e:
                    logger.error(f"Error processing chunk {chunk_count} at offset {offset}: {e}")
                    continue
            
            # Close connection
            conn.close()
            
            logger.info(f"Completed processing: {total_processed:,} total rows, {total_documents:,} documents created")
            
            return pd.DataFrame()  # Return empty DataFrame since we've already processed everything
            
        except Exception as e:
            logger.error(f"Error during chunked processing: {e}")
            raise
    
    def process_and_save(self, 
                        gpkg_file: str = "edition_18_0_new_format.gpkg",
                        csv_file: str = "UKBuildings_edition_18_abc_link_file.csv",
                        table_name: str = "edition_18_0_new_format",
                        vector_size: int = None,
                        chunk_size: int = 100000,
                        max_chunks: int = None,
                        skip_geometry: bool = False) -> Dict[str, Any]:
        """
        Main method to process Verisk data and save to Qdrant using chunked processing.
        
        Args:
            gpkg_file: Name of the GPKG file
            csv_file: Name of the CSV file
            table_name: Name of the table in GPKG file
            vector_size: Size of embedding vectors (if None, will be detected)
            chunk_size: Number of rows to process in each chunk
            max_chunks: Maximum number of chunks to process (for testing)
            skip_geometry: Skip geometry processing if True
            
        Returns:
            Dictionary with processing statistics
        """
        start_time = time.time()
        
        try:
            logger.info("Starting Verisk data processing with chunked approach...")
            
            # Process GPKG file in chunks
            self.process_gpkg_in_chunks(
                gpkg_file=gpkg_file,
                table_name=table_name,
                chunk_size=chunk_size,
                max_chunks=max_chunks,
                skip_geometry=skip_geometry
            )
            
            end_time = time.time()
            processing_time = end_time - start_time
            
            stats = {
                "processing_time_seconds": processing_time,
                "collection_name": self.collection_name,
                "chunk_size": chunk_size,
                "max_chunks": max_chunks,
                "skip_geometry": skip_geometry
            }
            
            logger.info(f"Processing completed successfully in {processing_time:.2f} seconds")
            logger.info(f"Statistics: {stats}")
            
            return stats
            
        except Exception as e:
            logger.error(f"Error during processing: {e}")
            raise

    def clean_payload_for_json(self, payload: dict) -> dict:
        """
        Clean payload to ensure it can be serialized to JSON for Qdrant.
        
        Args:
            payload: Raw payload dictionary
            
        Returns:
            Cleaned payload dictionary
        """
        cleaned_payload = {}
        
        for key, value in payload.items():
            try:
                # Handle different data types
                if isinstance(value, np.integer):
                    cleaned_payload[key] = int(value)
                elif isinstance(value, np.floating):
                    cleaned_payload[key] = float(value)
                elif isinstance(value, np.ndarray):
                    cleaned_payload[key] = value.tolist()
                elif pd.isna(value):
                    cleaned_payload[key] = None
                elif isinstance(value, bytes):
                    # Skip bytes data that can't be serialized
                    cleaned_payload[key] = None
                elif hasattr(value, '__geo_interface__'):
                    # Convert geometry objects to GeoJSON
                    try:
                        cleaned_payload[key] = value.__geo_interface__
                    except Exception:
                        cleaned_payload[key] = None
                elif isinstance(value, (list, tuple)):
                    # Recursively clean lists/tuples
                    try:
                        cleaned_payload[key] = [self.clean_payload_for_json({f"item_{i}": item})[f"item_{i}"] 
                                              for i, item in enumerate(value)]
                    except Exception:
                        cleaned_payload[key] = None
                elif isinstance(value, dict):
                    # Recursively clean dictionaries
                    try:
                        cleaned_payload[key] = self.clean_payload_for_json(value)
                    except Exception:
                        cleaned_payload[key] = None
                elif isinstance(value, (str, int, float, bool)) or value is None:
                    # These types are safe for JSON
                    cleaned_payload[key] = value
                else:
                    # Convert to string for other types
                    try:
                        cleaned_payload[key] = str(value)
                    except Exception:
                        cleaned_payload[key] = None
                        
            except Exception as e:
                logger.debug(f"Error cleaning payload key '{key}': {e}")
                cleaned_payload[key] = None
                
        return cleaned_payload

def main():
    """
    Main function to run the Verisk data processor.
    """
    try:
        processor = VeriskQdrantProcessor()
        stats = processor.process_and_save()
        print(f"Processing completed successfully!")
        print(f"Statistics: {stats}")
    except Exception as e:
        print(f"Error during processing: {e}")
        raise

if __name__ == "__main__":
    main() 