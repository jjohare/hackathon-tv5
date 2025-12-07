#!/usr/bin/env python3
"""
Generate embeddings from rich text using TensorRT acceleration.
Reuses existing stage3_gpu_embeddings.py infrastructure.
"""

import sys
from pathlib import Path

# Add utils to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import numpy as np
import time
from tqdm import tqdm
from utils.trt_inference import TensorRTEncoder

def embed_rich_text(
    input_file: str,
    output_dir: str,
    engine_path: str,
    batch_size: int = 64
):
    """
    Generate embeddings from rich text using TensorRT.

    Args:
        input_file: Path to rich text JSONL
        output_dir: Output directory for embeddings
        engine_path: Path to TensorRT engine file
        batch_size: Batch size for encoding
    """

    print(f"Loading rich text from {input_file}...")
    with open(input_file, 'r') as f:
        movies = [json.loads(line) for line in f]

    print(f"   Loaded {len(movies):,} movies")

    # Initialize TensorRT encoder
    print(f"\nInitializing TensorRT encoder from {engine_path}...")
    encoder = TensorRTEncoder(engine_path)
    print(f"✅ TensorRT encoder ready")

    # Prepare output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Extract texts
    texts = [movie['text'] for movie in movies]
    print(f"\nGenerating embeddings for {len(texts):,} texts...")
    print(f"   Batch size: {batch_size}")

    start_time = time.time()

    # Batch encoding
    all_embeddings = []
    num_batches = (len(texts) + batch_size - 1) // batch_size

    for i in tqdm(range(0, len(texts), batch_size), desc="Encoding", total=num_batches):
        batch_texts = texts[i:i + batch_size]
        batch_embeddings = encoder.encode_batch(batch_texts)

        # Convert to numpy
        if hasattr(batch_embeddings, 'cpu'):
            batch_embeddings = batch_embeddings.cpu().numpy()

        all_embeddings.append(batch_embeddings)

    # Concatenate all embeddings
    all_embeddings = np.vstack(all_embeddings)

    elapsed = time.time() - start_time
    throughput = len(texts) / elapsed

    print(f"\n   Embedding shape: {all_embeddings.shape}")
    print(f"   Time taken: {elapsed:.2f}s")
    print(f"   Throughput: {throughput:.1f} movies/sec")

    # Save embeddings
    embeddings_file = output_path / "content_vectors.npy"
    print(f"\nSaving embeddings to {embeddings_file}...")
    np.save(str(embeddings_file), all_embeddings)

    # Save metadata
    metadata_file = output_path / "metadata.jsonl"
    print(f"Saving metadata to {metadata_file}...")
    with open(metadata_file, 'w') as f:
        for movie in movies:
            # Keep only essential metadata
            metadata = {
                'tmdb_id': movie['tmdb_id'],
                'title': movie['title'],
                'release_date': movie.get('release_date'),
                'vote_average': movie.get('vote_average'),
                'vote_count': movie.get('vote_count')
            }
            f.write(json.dumps(metadata) + '\n')

    # Save summary
    summary = {
        'total_movies': len(movies),
        'embedding_dim': all_embeddings.shape[1],
        'embedding_shape': list(all_embeddings.shape),
        'batch_size': batch_size,
        'processing_time_sec': elapsed,
        'throughput_movies_per_sec': throughput,
        'embeddings_file': str(embeddings_file),
        'metadata_file': str(metadata_file),
        'source_file': input_file
    }

    summary_file = output_path / "summary.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)

    print("\n" + "="*70)
    print("EMBEDDING GENERATION COMPLETE")
    print("="*70)
    print(f"   Movies processed: {len(movies):,}")
    print(f"   Embedding dimension: {all_embeddings.shape[1]}")
    print(f"   Processing time: {elapsed:.2f}s")
    print(f"   Throughput: {throughput:.1f} movies/sec")
    print(f"\n✅ Embeddings saved to {output_dir}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Embed rich text with TensorRT")
    parser.add_argument(
        '--input',
        default="../../data/processed/demo_subset_50k_rich_text.jsonl",
        help="Input rich text JSONL file"
    )
    parser.add_argument(
        '--output',
        default="../../data/embeddings/tmdb_semantic_demo",
        help="Output directory for embeddings"
    )
    parser.add_argument(
        '--engine',
        default="../../data/models/minilm_l12_v2_fp16.plan",
        help="TensorRT engine file"
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=64,
        help="Batch size for encoding"
    )

    args = parser.parse_args()

    embed_rich_text(
        input_file=args.input,
        output_dir=args.output,
        engine_path=args.engine,
        batch_size=args.batch_size
    )
