from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
import pandas as pd
from tqdm import tqdm

# Initialize
model = SentenceTransformer('BAAI/bge-large-en-v1.5')
client = QdrantClient(host="localhost", port=6333)

# Load data from Fabric (pseudo-code)
df = spark.read.format("delta").load("abfss://...").limit(10_000_000).toPandas()

# Create collection
client.create_collection(
    collection_name="epc_data",
    vectors_config=VectorParams(size=1024, distance=Distance.COSINE)
)

# Batch indexing
batch_size = 1000
for i in tqdm(range(0, len(df), batch_size)):
    batch = df.iloc[i:i+batch_size]
    texts = batch.apply(lambda row: format_row(row), axis=1).tolist()
    embeddings = model.encode(texts).tolist()
    
    client.upsert(
        collection_name="epc_data",
        points=[
            PointStruct(
                id=idx + i,
                vector=emb,
                payload=row.to_dict()
            )
            for idx, (_, row, emb) in enumerate(zip(batch.iterrows(), embeddings))
        ]
    )