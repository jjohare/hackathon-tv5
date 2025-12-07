#!/usr/bin/env python3
"""
Complex Query Demonstration - TMDB 1.3M Dataset
Showcases semantic search with diverse, complex natural language queries
"""

import sys
import time
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from utils.trt_inference import TensorRTEncoder
import numpy as np
import json

def load_dataset():
    """Load TMDB dataset embeddings and metadata"""
    data_dir = Path(__file__).parent.parent / "data" / "embeddings" / "tmdb"

    print("Loading TMDB dataset...")
    start = time.time()

    # Load embeddings
    embeddings_path = data_dir / "content_vectors.npy"
    embeddings = np.load(str(embeddings_path))
    print(f"   Loaded embeddings: {embeddings.shape}")

    # Load metadata (JSONL format)
    metadata_path = data_dir / "metadata.jsonl"
    metadata = []
    with open(metadata_path, 'r') as f:
        for line in f:
            metadata.append(json.loads(line))
    print(f"   Loaded metadata: {len(metadata):,} records")

    elapsed = time.time() - start
    print(f"✅ Dataset loaded in {elapsed:.2f}s")
    print(f"   Total movies: {len(embeddings):,}")
    print(f"   Embedding dimension: {embeddings.shape[1]}\n")

    return embeddings, metadata

def run_query(encoder, embeddings, metadata, query, top_k=5):
    """Execute semantic search query"""
    start = time.time()

    # Encode query
    query_embedding = encoder.encode(query)

    # Convert to numpy if it's a tensor
    if hasattr(query_embedding, 'cpu'):
        query_embedding = query_embedding.cpu().numpy()

    # Flatten to 1D if needed
    if len(query_embedding.shape) > 1:
        query_embedding = query_embedding.flatten()

    # Compute cosine similarities
    query_norm = np.linalg.norm(query_embedding)
    embedding_norms = np.linalg.norm(embeddings, axis=1)

    # Avoid division by zero
    similarities = np.dot(embeddings, query_embedding) / (embedding_norms * query_norm + 1e-8)

    # Get top-k results
    top_indices = np.argsort(similarities)[-top_k:][::-1]
    top_scores = similarities[top_indices]

    elapsed = time.time() - start

    return top_indices, top_scores, elapsed * 1000  # Convert to ms

def main():
    print("="*70)
    print("COMPLEX QUERY DEMONSTRATION - TMDB 1.3M DATASET")
    print("="*70)
    print()

    # Initialize encoder
    print("Initializing TensorRT encoder...")
    engine_path = Path(__file__).parent.parent / "data" / "models" / "minilm_l12_v2_fp16.plan"
    encoder = TensorRTEncoder(str(engine_path))
    print("✅ Encoder ready\n")

    # Load dataset
    embeddings, metadata = load_dataset()

    # Define complex queries
    queries = [
        {
            "category": "Multi-Genre Complex",
            "query": "mind-bending psychological thriller with time travel and multiple timelines",
            "description": "Tests genre blending + concept matching"
        },
        {
            "category": "Emotional Tone + Setting",
            "query": "heartwarming story about found family in a small coastal town",
            "description": "Tests emotional understanding + location"
        },
        {
            "category": "Visual Style",
            "query": "visually stunning cyberpunk noir with neon-lit rain-soaked streets",
            "description": "Tests aesthetic/cinematography matching"
        },
        {
            "category": "Character-Driven",
            "query": "complex anti-hero struggling with moral ambiguity and redemption",
            "description": "Tests character archetype understanding"
        },
        {
            "category": "Reference-Based",
            "query": "like Inception meets The Matrix but with more emotional depth",
            "description": "Tests comparative reasoning"
        },
        {
            "category": "Mood + Pacing",
            "query": "slow-burn atmospheric horror that builds dread without jump scares",
            "description": "Tests pacing and mood understanding"
        },
        {
            "category": "Social Commentary",
            "query": "satirical science fiction exploring class inequality and corporate dystopia",
            "description": "Tests thematic depth"
        },
        {
            "category": "Era-Specific",
            "query": "1980s coming-of-age adventure with Spielberg-style wonder and nostalgia",
            "description": "Tests temporal + stylistic matching"
        },
        {
            "category": "Target Audience",
            "query": "intelligent thriller that respects audience intelligence without exposition dumps",
            "description": "Tests narrative sophistication"
        },
        {
            "category": "Narrative Structure",
            "query": "non-linear storytelling with unreliable narrator and plot twists",
            "description": "Tests structural understanding"
        },
        {
            "category": "Cultural Specific",
            "query": "Japanese animation exploring existential themes with beautiful hand-drawn art",
            "description": "Tests cultural + medium awareness"
        },
        {
            "category": "Intensity + Scale",
            "query": "epic space opera with massive battles and political intrigue",
            "description": "Tests scope and genre conventions"
        }
    ]

    print("="*70)
    print("RUNNING COMPLEX QUERIES")
    print("="*70)
    print()

    total_queries = len(queries)
    total_time = 0

    for i, query_spec in enumerate(queries, 1):
        print(f"[{i}/{total_queries}] Category: {query_spec['category']}")
        print(f"    Query: '{query_spec['query']}'")
        print(f"    Tests: {query_spec['description']}")

        top_indices, top_scores, latency = run_query(
            encoder, embeddings, metadata, query_spec['query'], top_k=5
        )

        total_time += latency

        print(f"    Search time: {latency:.2f}ms")
        print(f"    Top 5 results:")

        for rank, (idx, score) in enumerate(zip(top_indices, top_scores), 1):
            movie = metadata[idx]
            title = movie.get('title', 'Unknown')
            year = movie.get('release_date', '')[:4] if movie.get('release_date') else 'N/A'

            print(f"      {rank}. {title} ({year}) - Score: {score:.3f}")

        print()

    print("="*70)
    print("PERFORMANCE SUMMARY")
    print("="*70)
    print(f"   Total queries: {total_queries}")
    print(f"   Dataset size: {len(embeddings):,} movies")
    print(f"   Total search time: {total_time:.2f}ms")
    print(f"   Average latency: {total_time/total_queries:.2f}ms per query")
    print(f"   Throughput: {1000/(total_time/total_queries):.1f} QPS")
    print()
    print("✅ Complex query demonstration complete!")
    print()

if __name__ == "__main__":
    main()
