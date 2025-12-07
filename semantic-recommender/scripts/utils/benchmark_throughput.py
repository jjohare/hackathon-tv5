#!/usr/bin/env python3
"""
Throughput Benchmark for Integrated System

Benchmarks:
- GPU semantic similarity
- Ontology reasoning
- Hybrid scoring
- End-to-end query processing
"""

import sys
import time
import json
import argparse
import numpy as np
from pathlib import Path
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.gpu_ontology_reasoning import GPUOntologyReasoner


def benchmark_throughput(num_queries=1000, batch_size=32):
    """Benchmark end-to-end throughput"""

    print("=" * 80)
    print("THROUGHPUT BENCHMARK - Integrated System")
    print("=" * 80)

    # Initialize reasoner
    reasoner = GPUOntologyReasoner()

    # Select random query movies
    query_ids = np.random.choice(reasoner.media_ids, size=num_queries, replace=True)

    print(f"\nBenchmarking {num_queries} queries...")
    print(f"Batch size: {batch_size}")

    # Warmup
    print("\n🔥 Warming up...")
    for _ in range(10):
        query_id = reasoner.media_ids[0]
        _ = reasoner.hybrid_recommend(query_id, top_k=10, semantic_candidates=100)

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    # Benchmark
    print("\n⚡ Running benchmark...")

    timings = {
        'semantic': [],
        'ontology': [],
        'total': []
    }

    start_total = time.time()

    for i in range(0, num_queries, batch_size):
        batch = query_ids[i:i+batch_size]

        for query_id in batch:
            start = time.time()

            # GPU semantic similarity
            semantic_start = time.time()
            semantic_results, gpu_time = reasoner.gpu_semantic_similarity(query_id, top_k=100)
            timings['semantic'].append(gpu_time)

            # Ontology reasoning
            onto_start = time.time()
            for candidate_id, _ in semantic_results[:10]:
                _ = reasoner.ontology_similarity(query_id, candidate_id)
                _ = reasoner.genre_similarity(query_id, candidate_id)
            onto_time = (time.time() - onto_start) * 1000
            timings['ontology'].append(onto_time)

            total_time = (time.time() - start) * 1000
            timings['total'].append(total_time)

    elapsed_total = time.time() - start_total

    # Results
    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)

    qps = num_queries / elapsed_total

    print(f"\n📊 Throughput:")
    print(f"   Total time: {elapsed_total:.2f}s")
    print(f"   Queries: {num_queries}")
    print(f"   QPS: {qps:.2f}")

    print(f"\n⏱️  Latency (ms):")
    for name, times in timings.items():
        mean = np.mean(times)
        p50 = np.percentile(times, 50)
        p95 = np.percentile(times, 95)
        p99 = np.percentile(times, 99)

        print(f"   {name.capitalize():12} | Mean: {mean:6.2f} | p50: {p50:6.2f} | p95: {p95:6.2f} | p99: {p99:6.2f}")

    # GPU utilization
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated(0) / 1e9
        reserved = torch.cuda.memory_reserved(0) / 1e9

        print(f"\n💾 GPU Memory:")
        print(f"   Allocated: {allocated:.2f} GB")
        print(f"   Reserved: {reserved:.2f} GB")

    # Target comparison
    print(f"\n🎯 Target Comparison:")
    print(f"   Target QPS: 1000")
    print(f"   Achieved QPS: {qps:.2f}")
    print(f"   Target met: {'✅ YES' if qps >= 1000 else '❌ NO'}")

    if qps < 1000:
        print(f"\n💡 To reach 1000 QPS:")
        print(f"   - Current: {qps:.2f} QPS")
        print(f"   - Need: {1000/qps:.2f}x speedup")
        print(f"   - Suggestions:")
        print(f"     • Batch processing: ~2-3x speedup")
        print(f"     • TensorRT encoding: ~3-5x speedup")
        print(f"     • FAISS GPU search: ~3-5x speedup")
        print(f"     • Multi-GPU: ~2-3x speedup per GPU")

    return {
        'qps': qps,
        'timings': {k: {
            'mean': float(np.mean(v)),
            'p50': float(np.percentile(v, 50)),
            'p95': float(np.percentile(v, 95)),
            'p99': float(np.percentile(v, 99))
        } for k, v in timings.items()}
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--queries', type=int, default=1000, help='Number of queries')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    args = parser.parse_args()

    results = benchmark_throughput(args.queries, args.batch_size)

    # Save results
    output_file = Path(__file__).parent.parent.parent / "results/throughput_benchmark.json"
    output_file.parent.mkdir(exist_ok=True)

    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n💾 Results saved to: {output_file}")
