#!/usr/bin/env python3
"""
Benchmark Optimized Hyper-Personalization V2
=============================================

Compares baseline vs optimized performance:
- Iteration 1: GPU-native cache
- Iteration 2: FP16 mixed precision

Author: Claude Sonnet 4.5
Date: December 7, 2025
"""

import sys
import time
import json
from pathlib import Path

import torch
import numpy as np

# Import both versions
sys.path.insert(0, str(Path(__file__).parent.parent))
from scripts.gpu_hyper_personalization import GPUHyperPersonalizationSystem as BaselineSystem
from scripts.gpu_hyper_personalization_v2 import GPUHyperPersonalizationSystem as OptimizedSystem


def benchmark_comparison():
    """Compare baseline vs optimized performance."""

    print("=" * 80)
    print("OPTIMIZATION BENCHMARK: Baseline vs V2")
    print("=" * 80)
    print()

    embeddings_path = "data/embeddings/media/content_vectors.npy"
    metadata_path = "data/embeddings/media/metadata.jsonl"

    # Test queries
    test_queries = [
        "sci-fi movies with time travel",
        "romantic comedies",
        "action thrillers",
        "psychological horror",
        "family animated movies"
    ]

    user_id = "benchmark_user_001"
    context = {
        'time_of_day': 'evening',
        'genre': 'sci-fi',
        'social': 'solo'
    }

    # Warm up GPU
    print("[Warming up GPU...]")
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        _ = torch.zeros(1000, 1000, device='cuda')
        torch.cuda.synchronize()
    print()

    # Benchmark Baseline
    print("=" * 80)
    print("BASELINE SYSTEM")
    print("=" * 80)
    baseline_system = BaselineSystem(
        item_embeddings_path=embeddings_path,
        metadata_path=metadata_path
    )

    baseline_times = []
    for query in test_queries:
        _, _, timings = baseline_system.personalized_search(
            user_id=user_id,
            query=query,
            top_k=10,
            context=context
        )
        baseline_times.append(timings['total'])
        print(f"Query: '{query[:30]}...' - {timings['total']:.2f}ms")

    baseline_mean = np.mean(baseline_times[1:])  # Skip cold start
    print(f"\nBaseline Mean (warm): {baseline_mean:.2f}ms")
    print()

    # Benchmark Optimized
    print("=" * 80)
    print("OPTIMIZED SYSTEM V2")
    print("=" * 80)
    optimized_system = OptimizedSystem(
        item_embeddings_path=embeddings_path,
        metadata_path=metadata_path
    )

    optimized_times = []
    for query in test_queries:
        _, _, timings = optimized_system.personalized_search(
            user_id=user_id,
            query=query,
            top_k=10,
            context=context
        )
        optimized_times.append(timings['total'])
        print(f"Query: '{query[:30]}...' - {timings['total']:.2f}ms")

    optimized_mean = np.mean(optimized_times[1:])  # Skip cold start
    print(f"\nOptimized Mean (warm): {optimized_mean:.2f}ms")
    print()

    # Comparison
    print("=" * 80)
    print("PERFORMANCE COMPARISON")
    print("=" * 80)
    speedup = baseline_mean / optimized_mean
    improvement_pct = ((baseline_mean - optimized_mean) / baseline_mean) * 100

    print(f"Baseline:     {baseline_mean:.2f}ms")
    print(f"Optimized:    {optimized_mean:.2f}ms")
    print(f"Speedup:      {speedup:.2f}×")
    print(f"Improvement:  {improvement_pct:.1f}% faster")
    print()

    # Save results
    results = {
        "baseline": {
            "mean_ms": baseline_mean,
            "all_queries_ms": baseline_times
        },
        "optimized": {
            "mean_ms": optimized_mean,
            "all_queries_ms": optimized_times
        },
        "comparison": {
            "speedup": speedup,
            "improvement_pct": improvement_pct
        }
    }

    output_file = "docs/OPTIMIZATION_V2_RESULTS.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to: {output_file}")


if __name__ == "__main__":
    benchmark_comparison()
