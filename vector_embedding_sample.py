from pyspark.sql import functions as F
from pyspark.sql.types import StructType, StructField, ArrayType, FloatType, StringType
from sentence_transformers import SentenceTransformer

# Initialize constants
BATCH_SIZE = 50000  # Adjust based on your cluster capacity
excluded_columns = {"lmk_key", "building_reference_number", "uprn", "uprn_source", "report_type"}

# Function to process data in batches
def process_data_in_batches(source_table, target_path, selected_columns):
    print(f"Processing table: {source_table}")
    
    # Get total row count
    total_rows = spark.sql(f"SELECT COUNT(*) AS cnt FROM {source_table}").first().cnt
    print(f"Total rows: {total_rows:,}")
    
    # Process in batches
    for start in range(0, total_rows, BATCH_SIZE):
        end = start + BATCH_SIZE
        print(f"Processing batch: {start}-{end} ({min(end, total_rows)-start} rows)")
        
        # Load batch of data
        batch_df = spark.sql(f"""
            SELECT * 
            FROM {source_table}
            ORDER BY monotonically_increasing_id()
            LIMIT {BATCH_SIZE}
            OFFSET {start}
        """)
        
        # Get columns for concatenation
        columns_to_concat = [col_name for col_name in batch_df.columns 
                            if col_name not in excluded_columns]
        
        # Create combined text
        batch_combined = batch_df.withColumn(
            "combined_text",
            F.concat_ws(" | ", *columns_to_concat)
        )
        
        # Define schema
        input_fields = batch_combined.schema.fields
        output_schema = StructType(
            input_fields + [StructField("embedding", ArrayType(FloatType()), nullable=True)]
        )
        
        # Generate embeddings
        embedded_batch = batch_combined.mapInPandas(embed_map_in_pandas, schema=output_schema)
        
        # Convert and select columns
        embedded_batch = embedded_batch.withColumn("embedding", F.col("embedding").cast("string"))
        final_batch = embedded_batch.select(*selected_columns)
        
        
        print(f"Completed batch: {start}-{end}")

# Optimized embedding function with model caching
def embed_map_in_pandas(iterator):
    # Load model once per worker process
    if not hasattr(embed_map_in_pandas, "model"):
        embed_map_in_pandas.model = SentenceTransformer('all-MiniLM-L6-v2')
    model = embed_map_in_pandas.model
    
    for pdf in iterator:
        texts = pdf["combined_text"].tolist()
        embeddings = model.encode(texts).tolist()
        pdf["embedding"] = embeddings
        yield pdf

# Process EPC England & Wales data
process_data_in_batches(
    source_table="LH_external_datasets.epc.epcDomesticEngWales",
    target_path=table_path + 'epcDomesticEngWalesVE',
    selected_columns=["lmk_key", "uprn", "building_reference_number", "postcode_trim", "embedding"]
)

# Process EPC Scotland data
process_data_in_batches(
    source_table="LH_external_datasets.epc.epcDomesticScotland",
    target_path=table_path + 'epcDomesticScotlandVE',
    selected_columns=["building_reference_number", "postcode_trim", "embedding"]
)

# Process Verisk data
process_data_in_batches(
    source_table="LH_external_datasets.verisk.v_edition_18_joined",
    target_path=table_path + 'v_edition_18_joinedVE',
    selected_columns=["uprn_link", "uprn", "embedding"]
)