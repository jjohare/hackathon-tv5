#!/usr/bin/env python3
"""
Load synthetic media embeddings into Milvus vector database.
Supports batch loading with HNSW indexing for fast similarity search.
"""

from pymilvus import (
    connections, Collection, FieldSchema, CollectionSchema, DataType,
    utility
)
import pyarrow.parquet as pq
import numpy as np
from tqdm import tqdm
import argparse
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_milvus_collection(collection_name="media_embeddings", drop_existing=False):
    """Create Milvus collection for media embeddings."""

    connections.connect(host='localhost', port='19530')
    logger.info("Connected to Milvus")

    # Drop existing collection if requested
    if drop_existing and utility.has_collection(collection_name):
        logger.warning(f"Dropping existing collection: {collection_name}")
        utility.drop_collection(collection_name)

    # Define schema
    fields = [
        FieldSchema(name="content_id", dtype=DataType.INT64, is_primary=True, auto_id=False),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=1024),
        FieldSchema(name="genre", dtype=DataType.VARCHAR, max_length=100),
        FieldSchema(name="mood", dtype=DataType.VARCHAR, max_length=50),
        FieldSchema(name="popularity", dtype=DataType.FLOAT),
        FieldSchema(name="release_year", dtype=DataType.INT32),
    ]

    schema = CollectionSchema(fields, description="TV5 Media Embeddings with multimodal features")

    collection = Collection(name=collection_name, schema=schema)
    logger.info(f"Created collection: {collection_name}")

    # Create HNSW index for fast similarity search
    logger.info("Creating HNSW index...")
    index_params = {
        "index_type": "HNSW",
        "metric_type": "COSINE",
        "params": {
            "M": 16,              # Number of connections per layer
            "efConstruction": 200  # Size of dynamic candidate list for construction
        }
    }

    collection.create_index(field_name="embedding", index_params=index_params)
    logger.info("Index created successfully")

    return collection


def load_data_to_milvus(parquet_path, collection, batch_size=10000):
    """Load embeddings from parquet to Milvus."""

    logger.info(f"Loading data from {parquet_path}")

    try:
        table = pq.read_table(parquet_path)
        df = table.to_pandas()

        # Ensure required columns exist
        required_cols = ['content_id', 'embedding', 'genre', 'mood', 'popularity', 'release_year']
        missing_cols = [col for col in required_cols if col not in df.columns]

        if missing_cols:
            logger.error(f"Missing columns: {missing_cols}")
            return 0

        total_inserted = 0

        for i in tqdm(range(0, len(df), batch_size), desc="Loading batches"):
            batch = df.iloc[i:i+batch_size]

            # Prepare entities
            entities = [
                batch['content_id'].tolist(),
                batch['embedding'].tolist(),
                batch['genre'].tolist(),
                batch['mood'].tolist(),
                batch['popularity'].tolist(),
                batch['release_year'].tolist(),
            ]

            # Insert batch
            collection.insert(entities)
            total_inserted += len(batch)

        # Flush to ensure data is persisted
        collection.flush()
        logger.info(f"Inserted {total_inserted} records from {parquet_path.name}")

        return total_inserted

    except Exception as e:
        logger.error(f"Error loading data from {parquet_path}: {e}")
        return 0


def load_collection_to_memory(collection):
    """Load collection into memory for querying."""
    logger.info("Loading collection into memory...")
    collection.load()
    logger.info("Collection loaded successfully")


def verify_data(collection):
    """Verify loaded data."""
    logger.info("Verifying data...")

    # Get collection stats
    stats = collection.num_entities
    logger.info(f"Total entities in collection: {stats}")

    # Perform a sample query
    search_params = {"metric_type": "COSINE", "params": {"ef": 50}}

    # Create a random query vector
    query_vectors = [np.random.random((1024,)).tolist()]

    results = collection.search(
        data=query_vectors,
        anns_field="embedding",
        param=search_params,
        limit=5,
        output_fields=["genre", "popularity", "release_year"]
    )

    logger.info("Sample search results:")
    for hits in results:
        for hit in hits:
            logger.info(f"  ID: {hit.id}, Distance: {hit.distance:.4f}, Genre: {hit.entity.get('genre')}")


def main():
    parser = argparse.ArgumentParser(description="Load synthetic media embeddings to Milvus")
    parser.add_argument("--data-dir", type=str,
                        default="/home/devuser/workspace/hackathon-tv5/data/embedded",
                        help="Directory with parquet files containing embeddings")
    parser.add_argument("--collection-name", type=str, default="media_embeddings",
                        help="Milvus collection name")
    parser.add_argument("--batch-size", type=int, default=10000,
                        help="Batch size for insertion")
    parser.add_argument("--drop-existing", action="store_true",
                        help="Drop existing collection before creating new one")
    parser.add_argument("--milvus-host", type=str, default="localhost",
                        help="Milvus host")
    parser.add_argument("--milvus-port", type=int, default=19530,
                        help="Milvus port")

    args = parser.parse_args()

    # Connect and create collection
    connections.connect(host=args.milvus_host, port=args.milvus_port)
    collection = create_milvus_collection(args.collection_name, args.drop_existing)

    # Find all parquet files
    data_dir = Path(args.data_dir)
    parquet_files = sorted(data_dir.glob("**/*.parquet"))

    logger.info(f"Found {len(parquet_files)} parquet files to load")

    if not parquet_files:
        logger.error(f"No parquet files found in {data_dir}")
        logger.info("Please run generate_embeddings.py first")
        return

    # Load all batches
    total_records = 0
    for parquet_file in parquet_files:
        records = load_data_to_milvus(parquet_file, collection, args.batch_size)
        total_records += records

    logger.info(f"Total records loaded: {total_records}")

    # Load collection to memory
    load_collection_to_memory(collection)

    # Verify data
    verify_data(collection)

    logger.info("Data loading complete!")


if __name__ == "__main__":
    main()
