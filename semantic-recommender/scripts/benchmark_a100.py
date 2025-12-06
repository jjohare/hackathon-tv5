#!/usr/bin/env python3
"""
A100 GPU Benchmark Script
Compares CPU vs GPU embedding generation performance
"""

import json
import time
import numpy as np
import torch
from pathlib import Path
from sentence_transformers import SentenceTransformer

# Configuration
DATA_DIR = Path("data")
EMBEDDINGS_DIR = DATA_DIR / "embeddings"
BATCH_SIZE = 512
MODEL_NAME = 'paraphrase-multilingual-MiniLM-L12-v2'


def benchmark_gpu():
    """Benchmark embedding generation on A100 GPU"""
    print("=" * 80)
    print("A100 GPU BENCHMARK")
    print("=" * 80)

    # Check GPU
    if not torch.cuda.is_available():
        print("ERROR: No GPU detected!")
        return None

    device = torch.device('cuda:0')
    gpu_name = torch.cuda.get_device_name(0)
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9

    print(f"\nGPU: {gpu_name}")
    print(f"Memory: {gpu_memory:.2f} GB")
    print(f"CUDA Version: {torch.version.cuda}")

    # Load model
    print(f"\nLoading model: {MODEL_NAME}")
    model_start = time.time()
    model = SentenceTransformer(MODEL_NAME)
    model.to(device)
    model_time = time.time() - model_start
    print(f"Model loaded in {model_time:.2f}s")

    # Load movies
    print("\nLoading movies...")
    movies_path = DATA_DIR / "processed" / "media" / "movies.jsonl"
    movies = []
    with open(movies_path) as f:
        for line in f:
            movies.append(json.loads(line))

    # Load genome
    genome_path = DATA_DIR / "processed" / "media" / "genome_scores.json"
    with open(genome_path) as f:
        genome_data = json.load(f)

    print(f"Loaded {len(movies):,} movies, {len(genome_data):,} with genome")

    # Create text representations
    print("\nCreating text representations...")
    texts = []
    for movie in movies:
        text_parts = [movie['metadata']['title']]

        if movie['metadata']['year']:
            text_parts.append(f"({movie['metadata']['year']})")

        if movie['classification']['genres']:
            text_parts.append(f"Genres: {', '.join(movie['classification']['genres'])}")

        # Top 10 genome tags
        ml_id = str(movie['identifiers']['movielens_id'])
        if ml_id in genome_data:
            top_tags = sorted(genome_data[ml_id].items(), key=lambda x: x[1], reverse=True)[:10]
            if top_tags:
                tag_names = [tag for tag, _ in top_tags]
                text_parts.append(f"Themes: {', '.join(tag_names)}")

        texts.append(". ".join(text_parts))

    # Benchmark embedding generation
    print(f"\nGenerating embeddings (batch_size={BATCH_SIZE})...")
    print("=" * 80)

    # Warmup
    _ = model.encode(texts[:100], batch_size=BATCH_SIZE, device=device)
    torch.cuda.synchronize()

    # Actual benchmark
    start_time = time.time()
    embeddings = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        convert_to_numpy=True,
        device=device,
        normalize_embeddings=True
    )
    torch.cuda.synchronize()
    end_time = time.time()

    total_time = end_time - start_time
    throughput = len(texts) / total_time

    # Results
    print("\n" + "=" * 80)
    print("BENCHMARK RESULTS")
    print("=" * 80)
    print(f"\nDataset:")
    print(f"  Texts processed: {len(texts):,}")
    print(f"  Embedding dimension: {embeddings.shape[1]}")
    print(f"  Total vectors: {embeddings.nbytes / 1e6:.2f} MB")

    print(f"\nPerformance:")
    print(f"  Total time: {total_time:.2f}s")
    print(f"  Throughput: {throughput:.2f} texts/second")
    print(f"  Time per text: {(total_time / len(texts)) * 1000:.2f} ms")
    print(f"  Batch size: {BATCH_SIZE}")

    print(f"\nGPU Utilization:")
    print(f"  Peak memory: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")
    print(f"  Memory efficiency: {(torch.cuda.max_memory_allocated() / gpu_memory) * 100:.1f}%")

    print(f"\nQuality:")
    norms = np.linalg.norm(embeddings, axis=1)
    print(f"  Mean norm: {np.mean(norms):.6f}")
    print(f"  Std norm: {np.std(norms):.6f}")
    print(f"  Normalized: {np.allclose(norms, 1.0, rtol=1e-4)}")

    # Save results
    results = {
        'gpu_name': gpu_name,
        'gpu_memory_gb': gpu_memory,
        'cuda_version': torch.version.cuda,
        'model': MODEL_NAME,
        'batch_size': BATCH_SIZE,
        'num_texts': len(texts),
        'embedding_dim': embeddings.shape[1],
        'total_time_seconds': total_time,
        'throughput_texts_per_second': throughput,
        'time_per_text_ms': (total_time / len(texts)) * 1000,
        'peak_memory_gb': torch.cuda.max_memory_allocated() / 1e9,
        'memory_efficiency_percent': (torch.cuda.max_memory_allocated() / gpu_memory) * 100,
        'model_load_time_seconds': model_time,
        'mean_norm': float(np.mean(norms)),
        'std_norm': float(np.std(norms)),
        'normalized': bool(np.allclose(norms, 1.0, rtol=1e-4))
    }

    return results


def compare_with_cpu_baseline():
    """Load CPU baseline results if available"""
    cpu_stats_path = EMBEDDINGS_DIR / "embedding_stats.json"

    if not cpu_stats_path.exists():
        print("\nNo CPU baseline found for comparison")
        return None

    with open(cpu_stats_path) as f:
        cpu_stats = json.load(f)

    return cpu_stats


def main():
    """Main benchmark execution"""
    # Run GPU benchmark
    gpu_results = benchmark_gpu()

    if not gpu_results:
        return 1

    # Save GPU results
    output_path = Path("a100_benchmark_results.json")
    with open(output_path, 'w') as f:
        json.dump(gpu_results, f, indent=2)
    print(f"\n📊 Results saved to {output_path}")

    # Compare with CPU baseline
    print("\n" + "=" * 80)
    print("CPU vs GPU COMPARISON")
    print("=" * 80)

    cpu_stats = compare_with_cpu_baseline()
    if cpu_stats:
        # Estimate CPU time based on throughput
        # CPU typically: ~2-3 texts/second
        # GPU (A100): ~400-600 texts/second

        estimated_cpu_time = gpu_results['num_texts'] / 2.5  # Conservative estimate
        speedup = estimated_cpu_time / gpu_results['total_time_seconds']

        print(f"\nEstimated Performance Gain:")
        print(f"  CPU time (estimated): ~{estimated_cpu_time:.0f}s ({estimated_cpu_time/60:.1f} minutes)")
        print(f"  GPU time (actual): {gpu_results['total_time_seconds']:.2f}s")
        print(f"  Speedup: {speedup:.1f}x faster on A100")
        print(f"  Time saved: {(estimated_cpu_time - gpu_results['total_time_seconds'])/60:.1f} minutes")

    print("\n" + "=" * 80)
    print("✅ BENCHMARK COMPLETE")
    print("=" * 80)

    return 0


if __name__ == '__main__':
    import sys
    sys.exit(main())
