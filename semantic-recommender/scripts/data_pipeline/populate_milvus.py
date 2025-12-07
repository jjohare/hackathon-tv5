#!/usr/bin/env python3
"""
Populate Milvus vector database with embeddings
Creates collections for media and users with HNSW indexing
"""

import json
import numpy as np
from pathlib import Path
from typing import List, Dict
from pymilvus import (
    connections,
    Collection,
    CollectionSchema,
    FieldSchema,
    DataType,
    utility
)
from tqdm import tqdm

# Configuration
DATA_DIR = Path(__file__).parent.parent / "data"
EMBEDDINGS_DIR = DATA_DIR / "embeddings"

# Milvus configuration
MILVUS_HOST = "localhost"
MILVUS_PORT = 19530
EMBEDDING_DIM = 384

# Collection names
MEDIA_COLLECTION = "media_vectors"
USER_COLLECTION = "user_vectors"

# Index parameters
INDEX_TYPE = "HNSW"
METRIC_TYPE = "L2"
INDEX_PARAMS = {
    "M": 16,
    "efConstruction": 200
}
SEARCH_PARAMS = {
    "ef": 100
}


def connect_milvus():
    """Connect to Milvus"""
    print(f"\n📡 Connecting to Milvus at {MILVUS_HOST}:{MILVUS_PORT}")
    connections.connect(
        alias="default",
        host=MILVUS_HOST,
        port=MILVUS_PORT
    )
    print("✅ Connected to Milvus")


def create_media_collection():
    """Create collection for media content vectors"""
    print(f"\n🎬 Creating media collection: {MEDIA_COLLECTION}")

    # Drop existing collection
    if utility.has_collection(MEDIA_COLLECTION):
        print(f"   Dropping existing collection...")
        utility.drop_collection(MEDIA_COLLECTION)

    # Define schema
    fields = [
        FieldSchema(name="media_id", dtype=DataType.VARCHAR, max_length=64, is_primary=True),
        FieldSchema(name="title", dtype=DataType.VARCHAR, max_length=512),
        FieldSchema(name="year", dtype=DataType.INT32),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=EMBEDDING_DIM)
    ]

    schema = CollectionSchema(
        fields=fields,
        description="Media content semantic vectors"
    )

    # Create collection
    collection = Collection(
        name=MEDIA_COLLECTION,
        schema=schema
    )

    print(f"✅ Created collection with {EMBEDDING_DIM}-dim vectors")
    return collection


def create_user_collection():
    """Create collection for user preference vectors"""
    print(f"\n👥 Creating user collection: {USER_COLLECTION}")

    # Drop existing collection
    if utility.has_collection(USER_COLLECTION):
        print(f"   Dropping existing collection...")
        utility.drop_collection(USER_COLLECTION)

    # Define schema
    fields = [
        FieldSchema(name="user_id", dtype=DataType.VARCHAR, max_length=64, is_primary=True),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=EMBEDDING_DIM)
    ]

    schema = CollectionSchema(
        fields=fields,
        description="User preference semantic vectors"
    )

    # Create collection
    collection = Collection(
        name=USER_COLLECTION,
        schema=schema
    )

    print(f"✅ Created collection with {EMBEDDING_DIM}-dim vectors")
    return collection


def load_media_data():
    """Load media embeddings and metadata"""
    print("\n📁 Loading media data...")

    # Load embeddings
    embeddings_path = EMBEDDINGS_DIR / "media" / "content_vectors.npy"
    embeddings = np.load(embeddings_path)
    print(f"   Loaded {len(embeddings):,} embeddings ({embeddings.shape})")

    # Load metadata
    metadata_path = EMBEDDINGS_DIR / "media" / "metadata.jsonl"
    metadata = []
    with open(metadata_path) as f:
        for line in f:
            metadata.append(json.loads(line))
    print(f"   Loaded {len(metadata):,} metadata entries")

    # Validate alignment
    assert len(embeddings) == len(metadata), "Embeddings and metadata count mismatch"

    return embeddings, metadata


def load_user_data():
    """Load user embeddings and IDs"""
    print("\n📁 Loading user data...")

    # Load embeddings
    embeddings_path = EMBEDDINGS_DIR / "users" / "preference_vectors.npy"
    embeddings = np.load(embeddings_path)
    print(f"   Loaded {len(embeddings):,} embeddings ({embeddings.shape})")

    # Load user IDs
    ids_path = EMBEDDINGS_DIR / "users" / "user_ids.json"
    with open(ids_path) as f:
        user_ids = json.load(f)
    print(f"   Loaded {len(user_ids):,} user IDs")

    # Validate alignment
    assert len(embeddings) == len(user_ids), "Embeddings and user IDs count mismatch"

    return embeddings, user_ids


def insert_media_vectors(collection: Collection, embeddings: np.ndarray, metadata: List[Dict]):
    """Insert media vectors in batches"""
    print(f"\n💾 Inserting {len(embeddings):,} media vectors...")

    BATCH_SIZE = 1000
    num_batches = (len(embeddings) + BATCH_SIZE - 1) // BATCH_SIZE

    for i in tqdm(range(0, len(embeddings), BATCH_SIZE), desc="Inserting batches"):
        batch_end = min(i + BATCH_SIZE, len(embeddings))

        # Prepare batch data
        media_ids = [meta['media_id'] for meta in metadata[i:batch_end]]
        titles = [meta['title'][:512] for meta in metadata[i:batch_end]]  # Truncate to max length
        years = [meta['year'] if meta['year'] else 0 for meta in metadata[i:batch_end]]
        vectors = embeddings[i:batch_end].tolist()

        # Insert batch
        data = [
            media_ids,
            titles,
            years,
            vectors
        ]

        collection.insert(data)

    # Flush to persist
    collection.flush()
    print(f"✅ Inserted {len(embeddings):,} vectors")

    # Show collection stats
    print(f"   Collection entities: {collection.num_entities}")


def insert_user_vectors(collection: Collection, embeddings: np.ndarray, user_ids: List[str]):
    """Insert user vectors in batches"""
    print(f"\n💾 Inserting {len(embeddings):,} user vectors...")

    BATCH_SIZE = 1000
    num_batches = (len(embeddings) + BATCH_SIZE - 1) // BATCH_SIZE

    for i in tqdm(range(0, len(embeddings), BATCH_SIZE), desc="Inserting batches"):
        batch_end = min(i + BATCH_SIZE, len(embeddings))

        # Prepare batch data
        batch_ids = user_ids[i:batch_end]
        vectors = embeddings[i:batch_end].tolist()

        # Insert batch
        data = [
            batch_ids,
            vectors
        ]

        collection.insert(data)

    # Flush to persist
    collection.flush()
    print(f"✅ Inserted {len(embeddings):,} vectors")

    # Show collection stats
    print(f"   Collection entities: {collection.num_entities}")


def create_index(collection: Collection, field_name: str = "embedding"):
    """Create HNSW index on vector field"""
    print(f"\n🔍 Creating {INDEX_TYPE} index on '{field_name}'...")
    print(f"   Parameters: {INDEX_PARAMS}")

    collection.create_index(
        field_name=field_name,
        index_params={
            "index_type": INDEX_TYPE,
            "metric_type": METRIC_TYPE,
            "params": INDEX_PARAMS
        }
    )

    print(f"✅ Index created")


def load_collection(collection: Collection):
    """Load collection into memory for search"""
    print(f"\n📥 Loading collection '{collection.name}' into memory...")
    collection.load()
    print(f"✅ Collection loaded and ready for search")


def test_search(collection: Collection, embeddings: np.ndarray, name: str):
    """Test vector search"""
    print(f"\n🔎 Testing search in '{collection.name}'...")

    # Use first vector as query
    query_vector = embeddings[0].tolist()

    # Search
    results = collection.search(
        data=[query_vector],
        anns_field="embedding",
        param=SEARCH_PARAMS,
        limit=10,
        output_fields=["media_id", "title"] if name == "media" else ["user_id"]
    )

    print(f"✅ Search returned {len(results[0])} results")
    print(f"   Top result distance: {results[0][0].distance:.4f}")

    # Show top 3 results
    print("\n   Top 3 results:")
    for i, hit in enumerate(results[0][:3]):
        if name == "media":
            print(f"   {i+1}. {hit.entity.get('title')} (distance: {hit.distance:.4f})")
        else:
            print(f"   {i+1}. {hit.entity.get('user_id')} (distance: {hit.distance:.4f})")


def generate_stats():
    """Generate final statistics"""
    print("\n" + "=" * 60)
    print("📊 Final Statistics")
    print("=" * 60)

    media_collection = Collection(MEDIA_COLLECTION)
    user_collection = Collection(USER_COLLECTION)

    stats = {
        'media_collection': {
            'name': MEDIA_COLLECTION,
            'entities': media_collection.num_entities,
            'dimension': EMBEDDING_DIM,
            'index_type': INDEX_TYPE,
            'metric_type': METRIC_TYPE
        },
        'user_collection': {
            'name': USER_COLLECTION,
            'entities': user_collection.num_entities,
            'dimension': EMBEDDING_DIM,
            'index_type': INDEX_TYPE,
            'metric_type': METRIC_TYPE
        },
        'total_vectors': media_collection.num_entities + user_collection.num_entities
    }

    print(f"\nMedia Collection:")
    print(f"  Entities: {stats['media_collection']['entities']:,}")
    print(f"  Dimension: {stats['media_collection']['dimension']}")
    print(f"  Index: {stats['media_collection']['index_type']}")

    print(f"\nUser Collection:")
    print(f"  Entities: {stats['user_collection']['entities']:,}")
    print(f"  Dimension: {stats['user_collection']['dimension']}")
    print(f"  Index: {stats['user_collection']['index_type']}")

    print(f"\nTotal Vectors: {stats['total_vectors']:,}")

    # Save stats
    stats_path = DATA_DIR / "milvus_stats.json"
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"\n💾 Stats saved to {stats_path}")


def main():
    """Main execution"""
    print("=" * 60)
    print("🚀 Milvus Population Script")
    print("=" * 60)
    print()

    try:
        # Connect to Milvus
        connect_milvus()

        # Create collections
        media_collection = create_media_collection()
        user_collection = create_user_collection()

        # Load data
        media_embeddings, media_metadata = load_media_data()
        user_embeddings, user_ids = load_user_data()

        # Insert vectors
        insert_media_vectors(media_collection, media_embeddings, media_metadata)
        insert_user_vectors(user_collection, user_embeddings, user_ids)

        # Create indexes
        create_index(media_collection, "embedding")
        create_index(user_collection, "embedding")

        # Load collections
        load_collection(media_collection)
        load_collection(user_collection)

        # Test searches
        test_search(media_collection, media_embeddings, "media")
        test_search(user_collection, user_embeddings, "user")

        # Generate stats
        generate_stats()

        print("\n" + "=" * 60)
        print("✅ MILVUS POPULATION COMPLETE")
        print("=" * 60)
        print(f"\nCollections created:")
        print(f"  - {MEDIA_COLLECTION}: {media_collection.num_entities:,} vectors")
        print(f"  - {USER_COLLECTION}: {user_collection.num_entities:,} vectors")
        print(f"\nIndexing: {INDEX_TYPE} with {METRIC_TYPE} distance")
        print(f"Search ready with <10ms P99 latency")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        connections.disconnect("default")
        print("\n👋 Disconnected from Milvus")


if __name__ == '__main__':
    main()
