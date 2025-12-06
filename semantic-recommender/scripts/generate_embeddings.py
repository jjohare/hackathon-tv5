#!/usr/bin/env python3
"""
Generate embeddings using SBERT on A100 GPU
Creates 384-dim vectors for media content and user preferences
"""

import json
import numpy as np
import torch
from pathlib import Path
from typing import List, Dict
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# Configuration
DATA_DIR = Path(__file__).parent.parent / "data"
PROCESSED_DIR = DATA_DIR / "processed"
EMBEDDINGS_DIR = DATA_DIR / "embeddings"

# Ensure output directories
for subdir in ["media", "users"]:
    (EMBEDDINGS_DIR / subdir).mkdir(parents=True, exist_ok=True)

# Model configuration
MODEL_NAME = 'paraphrase-multilingual-MiniLM-L12-v2'  # 384-dim, multilingual
EMBEDDING_DIM = 384
BATCH_SIZE = 512  # Optimized for A100


def check_gpu():
    """Check GPU availability and print info"""
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
        print(f"✅ GPU Available: {torch.cuda.get_device_name(0)}")
        print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        print(f"   CUDA Version: {torch.version.cuda}")
        return device
    else:
        print("⚠️  No GPU detected, using CPU (will be slower)")
        return torch.device('cpu')


def load_model(device):
    """Load SBERT model"""
    print(f"\n📦 Loading model: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)
    model.to(device)
    print(f"✅ Model loaded on {device}")
    print(f"   Embedding dimension: {model.get_sentence_embedding_dimension()}")
    return model


def create_media_text(movie: Dict, genome_data: Dict) -> str:
    """Create rich text representation for embedding"""
    text_parts = []

    # Title
    text_parts.append(movie['metadata']['title'])

    # Year
    if movie['metadata']['year']:
        text_parts.append(f"({movie['metadata']['year']})")

    # Genres
    if movie['classification']['genres']:
        text_parts.append(f"Genres: {', '.join(movie['classification']['genres'])}")

    # Genome tags (top 10)
    ml_id = str(movie['identifiers']['movielens_id'])
    if ml_id in genome_data:
        genome = genome_data[ml_id]
        top_tags = sorted(genome.items(), key=lambda x: x[1], reverse=True)[:10]
        if top_tags:
            tag_names = [tag for tag, _ in top_tags]
            text_parts.append(f"Themes: {', '.join(tag_names)}")

    return ". ".join(text_parts)


def generate_media_embeddings(model, device):
    """Generate embeddings for all movies"""
    print("\n" + "=" * 60)
    print("🎬 Generating Media Embeddings")
    print("=" * 60)

    # Load movies
    movies_path = PROCESSED_DIR / "media" / "movies.jsonl"
    print(f"\n📁 Loading movies from {movies_path}")
    movies = []
    with open(movies_path) as f:
        for line in f:
            movies.append(json.loads(line))
    print(f"✅ Loaded {len(movies):,} movies")

    # Load genome data
    genome_path = PROCESSED_DIR / "media" / "genome_scores.json"
    print(f"📁 Loading genome data from {genome_path}")
    with open(genome_path) as f:
        genome_data = json.load(f)
    print(f"✅ Loaded genome for {len(genome_data):,} movies")

    # Create text representations
    print(f"\n🔤 Creating text representations...")
    texts = []
    metadata = []
    for movie in movies:
        text = create_media_text(movie, genome_data)
        texts.append(text)
        metadata.append({
            'media_id': movie['media_id'],
            'title': movie['metadata']['title'],
            'year': movie['metadata']['year']
        })

    # Generate embeddings in batches
    print(f"\n🚀 Generating embeddings (batch_size={BATCH_SIZE})...")
    embeddings = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        convert_to_numpy=True,
        device=device,
        normalize_embeddings=True  # L2 normalization
    )

    # Validate
    print(f"\n✓ Generated {len(embeddings)} embeddings")
    print(f"  Shape: {embeddings.shape}")
    print(f"  Dtype: {embeddings.dtype}")
    print(f"  Normalized: {np.allclose(np.linalg.norm(embeddings, axis=1), 1.0)}")

    # Save embeddings
    embeddings_path = EMBEDDINGS_DIR / "media" / "content_vectors.npy"
    np.save(embeddings_path, embeddings)
    print(f"💾 Saved embeddings to {embeddings_path}")

    # Save metadata
    metadata_path = EMBEDDINGS_DIR / "media" / "metadata.jsonl"
    with open(metadata_path, 'w') as f:
        for meta in metadata:
            f.write(json.dumps(meta) + '\n')
    print(f"💾 Saved metadata to {metadata_path}")

    return embeddings, metadata


def generate_user_embeddings(model, device, media_embeddings, media_metadata):
    """Generate user preference embeddings from rating history"""
    print("\n" + "=" * 60)
    print("👥 Generating User Embeddings")
    print("=" * 60)

    # Create media ID to index mapping
    media_id_to_idx = {meta['media_id']: idx for idx, meta in enumerate(media_metadata)}

    # Load ratings and aggregate by user
    print("\n📁 Loading ratings...")
    user_ratings = {}
    ratings_path = PROCESSED_DIR / "interactions" / "ratings.jsonl"

    with open(ratings_path) as f:
        for line in tqdm(f, desc="Processing ratings"):
            interaction = json.loads(line)
            uid = interaction['user_id']
            mid = interaction['media_id']
            rating = interaction['rating']

            if uid not in user_ratings:
                user_ratings[uid] = []

            # Only include if we have embedding for this movie
            if mid in media_id_to_idx:
                user_ratings[uid].append((mid, rating))

    print(f"✅ Loaded ratings for {len(user_ratings):,} users")

    # Generate user embeddings as weighted averages
    print("\n🧮 Computing weighted averages...")
    user_embeddings = []
    user_ids = []
    skipped = 0

    for user_id, ratings in tqdm(user_ratings.items(), desc="Generating user embeddings"):
        # Collect vectors and weights
        vectors = []
        weights = []

        for media_id, rating in ratings:
            idx = media_id_to_idx[media_id]
            vectors.append(media_embeddings[idx])
            # Normalize rating from 1-5 to 0-1
            weights.append((rating - 1) / 4.0)

        if not vectors:
            skipped += 1
            continue

        # Weighted average
        vectors = np.array(vectors)
        weights = np.array(weights)
        weights = weights / weights.sum()  # Normalize weights

        user_embedding = np.average(vectors, axis=0, weights=weights)

        # Normalize to unit length
        user_embedding = user_embedding / np.linalg.norm(user_embedding)

        user_embeddings.append(user_embedding)
        user_ids.append(user_id)

    user_embeddings = np.array(user_embeddings)

    print(f"\n✓ Generated {len(user_embeddings)} user embeddings")
    print(f"  Skipped {skipped} users (no valid ratings)")
    print(f"  Shape: {user_embeddings.shape}")
    print(f"  Normalized: {np.allclose(np.linalg.norm(user_embeddings, axis=1), 1.0)}")

    # Save
    embeddings_path = EMBEDDINGS_DIR / "users" / "preference_vectors.npy"
    np.save(embeddings_path, user_embeddings)
    print(f"💾 Saved embeddings to {embeddings_path}")

    ids_path = EMBEDDINGS_DIR / "users" / "user_ids.json"
    with open(ids_path, 'w') as f:
        json.dump(user_ids, f)
    print(f"💾 Saved user IDs to {ids_path}")

    return user_embeddings, user_ids


def generate_summary_stats(media_embeddings, user_embeddings):
    """Generate summary statistics"""
    print("\n" + "=" * 60)
    print("📊 Summary Statistics")
    print("=" * 60)

    stats = {
        'media_embeddings': {
            'count': len(media_embeddings),
            'dimension': media_embeddings.shape[1],
            'dtype': str(media_embeddings.dtype),
            'normalized': bool(np.allclose(np.linalg.norm(media_embeddings, axis=1), 1.0)),
            'mean_norm': float(np.mean(np.linalg.norm(media_embeddings, axis=1))),
            'std_norm': float(np.std(np.linalg.norm(media_embeddings, axis=1)))
        },
        'user_embeddings': {
            'count': len(user_embeddings),
            'dimension': user_embeddings.shape[1],
            'dtype': str(user_embeddings.dtype),
            'normalized': bool(np.allclose(np.linalg.norm(user_embeddings, axis=1), 1.0)),
            'mean_norm': float(np.mean(np.linalg.norm(user_embeddings, axis=1))),
            'std_norm': float(np.std(np.linalg.norm(user_embeddings, axis=1)))
        },
        'total_vectors': len(media_embeddings) + len(user_embeddings),
        'total_size_mb': (media_embeddings.nbytes + user_embeddings.nbytes) / 1e6
    }

    print(f"\nMedia Embeddings:")
    print(f"  Count: {stats['media_embeddings']['count']:,}")
    print(f"  Dimension: {stats['media_embeddings']['dimension']}")
    print(f"  Normalized: {stats['media_embeddings']['normalized']}")

    print(f"\nUser Embeddings:")
    print(f"  Count: {stats['user_embeddings']['count']:,}")
    print(f"  Dimension: {stats['user_embeddings']['dimension']}")
    print(f"  Normalized: {stats['user_embeddings']['normalized']}")

    print(f"\nTotal:")
    print(f"  Vectors: {stats['total_vectors']:,}")
    print(f"  Size: {stats['total_size_mb']:.2f} MB")

    # Save stats
    stats_path = EMBEDDINGS_DIR / "embedding_stats.json"
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"\n💾 Stats saved to {stats_path}")


def main():
    """Main execution"""
    print("=" * 60)
    print("🚀 A100 GPU Embedding Generator")
    print("=" * 60)
    print()

    # Check GPU
    device = check_gpu()

    # Load model
    model = load_model(device)

    # Generate media embeddings
    media_embeddings, media_metadata = generate_media_embeddings(model, device)

    # Generate user embeddings
    user_embeddings, user_ids = generate_user_embeddings(
        model, device, media_embeddings, media_metadata
    )

    # Generate stats
    generate_summary_stats(media_embeddings, user_embeddings)

    print("\n" + "=" * 60)
    print("✅ EMBEDDING GENERATION COMPLETE")
    print("=" * 60)
    print(f"\nTotal vectors generated: {len(media_embeddings) + len(user_embeddings):,}")
    print(f"Embedding dimension: {EMBEDDING_DIM}")
    print(f"Model: {MODEL_NAME}")


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
