#!/usr/bin/env python3
"""
Comprehensive GPU Hyper-Personalization Benchmark

Measures:
- Queries Per Second (QPS)
- Encoding latency (mean, p95, p99)
- Total recommendation latency
- GPU memory usage and utilization
- Throughput scaling with batch size

Targets (RTX A6000):
- QPS: >1000 (vs 70 CPU baseline)
- Encoding: <1ms mean
- Total latency: <2ms
- GPU utilization: >80%

Usage:
    # PyTorch baseline
    python scripts/benchmark_gpu_hyper_personalization.py --mode pytorch

    # TensorRT (when available)
    python scripts/benchmark_gpu_hyper_personalization.py --mode tensorrt --engine data/models/minilm_l12_v2_fp16.plan

    # Full comparison
    python scripts/benchmark_gpu_hyper_personalization.py --mode both
"""

import sys
import json
import time
import argparse
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict
import statistics

import torch
import numpy as np
from tqdm import tqdm

# Add utils to path
sys.path.insert(0, str(Path(__file__).parent / "utils"))

from gpu_hyper_personalization import GPUHyperPersonalization


class GPUBenchmark:
    """Comprehensive GPU performance benchmark"""

    def __init__(
        self,
        system: GPUHyperPersonalization,
        num_users: int = 1000,
        num_queries: int = 100
    ):
        self.system = system
        self.num_users = num_users
        self.num_queries = num_queries
        self.device = system.device

        # Test data
        self.test_queries = self._generate_test_queries()
        self.test_users = [f"user_bench_{i:06d}" for i in range(num_users)]

        # Metrics storage
        self.metrics = {
            'encoding_latency_ms': [],
            'user_fusion_ms': [],
            'similarity_ms': [],
            'attention_ms': [],
            'total_ms': [],
            'qps': [],
            'gpu_memory_mb': [],
            'gpu_utilization_pct': []
        }

    def _generate_test_queries(self) -> List[str]:
        """Generate diverse test queries"""
        query_templates = [
            "action movies with explosions",
            "romantic comedies",
            "sci-fi space exploration",
            "psychological thrillers",
            "family-friendly animation",
            "historical war dramas",
            "horror movies with jump scares",
            "indie art house films",
            "superhero origin stories",
            "mystery detective series",
            "time travel paradox",
            "dystopian future society",
            "heist movies with twists",
            "coming of age stories",
            "martial arts action",
            "western gunfighters",
            "zombie apocalypse",
            "fantasy epic adventures",
            "courtroom legal dramas",
            "sports underdog stories"
        ]

        # Repeat to get desired count
        queries = []
        while len(queries) < self.num_queries:
            queries.extend(query_templates)

        return queries[:self.num_queries]

    def benchmark_encoding_latency(self, num_iterations: int = 100) -> Dict:
        """
        Benchmark pure encoding latency

        Measures:
        - Mean encoding time
        - p50, p95, p99 latencies
        - Throughput (QPS)
        """
        print(f"\n{'='*80}")
        print(f"Encoding Latency Benchmark ({num_iterations} iterations)")
        print(f"{'='*80}\n")

        latencies = []

        # Warmup
        for _ in range(10):
            _ = self.system.model.encode(
                self.test_queries[0],
                convert_to_tensor=True,
                device=self.device
            )

        # Benchmark
        torch.cuda.synchronize()

        for i in tqdm(range(num_iterations), desc="Encoding"):
            query = self.test_queries[i % len(self.test_queries)]

            start = time.perf_counter()
            _ = self.system.model.encode(
                query,
                convert_to_tensor=True,
                device=self.device
            )
            torch.cuda.synchronize()
            end = time.perf_counter()

            latency_ms = (end - start) * 1000
            latencies.append(latency_ms)

        # Calculate statistics
        latencies.sort()
        results = {
            'mean_ms': statistics.mean(latencies),
            'median_ms': statistics.median(latencies),
            'p95_ms': latencies[int(0.95 * len(latencies))],
            'p99_ms': latencies[int(0.99 * len(latencies))],
            'min_ms': min(latencies),
            'max_ms': max(latencies),
            'qps': 1000 / statistics.mean(latencies),
            'encoder_type': self.system.encoder_type
        }

        print(f"\nEncoding Results ({self.system.encoder_type}):")
        print(f"  Mean:   {results['mean_ms']:.3f} ms")
        print(f"  Median: {results['median_ms']:.3f} ms")
        print(f"  P95:    {results['p95_ms']:.3f} ms")
        print(f"  P99:    {results['p99_ms']:.3f} ms")
        print(f"  QPS:    {results['qps']:.1f}")

        return results

    def benchmark_end_to_end(self, num_iterations: int = 100) -> Dict:
        """
        Benchmark end-to-end recommendation pipeline

        Measures:
        - Query encoding
        - User fusion
        - GPU similarity search
        - Attention reranking
        - Total latency
        """
        print(f"\n{'='*80}")
        print(f"End-to-End Pipeline Benchmark ({num_iterations} iterations)")
        print(f"{'='*80}\n")

        # Metrics by stage
        stage_metrics = defaultdict(list)

        # Warmup
        for _ in range(10):
            _ = self.system.personalized_search(
                user_id=self.test_users[0],
                query=self.test_queries[0],
                top_k=10,
                context={
                    'time_of_day': [0.3, 0.4, 0.3],
                    'genre_prefs': [0.5, 0.3, 0.2],
                    'social_signal': [1.0, 0.0]
                }
            )

        # Benchmark
        torch.cuda.synchronize()

        for i in tqdm(range(num_iterations), desc="End-to-end"):
            user_id = self.test_users[i % len(self.test_users)]
            query = self.test_queries[i % len(self.test_queries)]

            # Context variation
            context = {
                'time_of_day': [np.random.random(), np.random.random(), np.random.random()],
                'genre_prefs': [np.random.random(), np.random.random(), np.random.random()],
                'social_signal': [1.0, 0.0]
            }

            # Execute
            result = self.system.personalized_search(
                user_id=user_id,
                query=query,
                top_k=10,
                context=context
            )

            # Collect metrics
            timing = result['timing']
            stage_metrics['encoding'].append(timing['query_encoding_ms'])
            stage_metrics['user_fusion'].append(timing['user_fusion_ms'])
            stage_metrics['similarity'].append(timing['gpu_similarity_ms'])
            stage_metrics['attention'].append(timing['attention_rerank_ms'])
            stage_metrics['total'].append(timing['total_ms'])

        # Calculate statistics
        results = {}
        for stage, values in stage_metrics.items():
            values.sort()
            results[stage] = {
                'mean_ms': statistics.mean(values),
                'median_ms': statistics.median(values),
                'p95_ms': values[int(0.95 * len(values))],
                'p99_ms': values[int(0.99 * len(values))],
                'min_ms': min(values),
                'max_ms': max(values)
            }

        # Overall QPS
        total_qps = 1000 / results['total']['mean_ms']
        results['qps'] = total_qps

        print(f"\nEnd-to-End Results:")
        print(f"  Total QPS: {total_qps:.1f}")
        print(f"\n  Stage Breakdown (mean / p95 / p99):")
        print(f"    Encoding:   {results['encoding']['mean_ms']:.3f} / {results['encoding']['p95_ms']:.3f} / {results['encoding']['p99_ms']:.3f} ms")
        print(f"    User Fusion: {results['user_fusion']['mean_ms']:.3f} / {results['user_fusion']['p95_ms']:.3f} / {results['user_fusion']['p99_ms']:.3f} ms")
        print(f"    Similarity:  {results['similarity']['mean_ms']:.3f} / {results['similarity']['p95_ms']:.3f} / {results['similarity']['p99_ms']:.3f} ms")
        print(f"    Attention:   {results['attention']['mean_ms']:.3f} / {results['attention']['p95_ms']:.3f} / {results['attention']['p99_ms']:.3f} ms")
        print(f"    Total:       {results['total']['mean_ms']:.3f} / {results['total']['p95_ms']:.3f} / {results['total']['p99_ms']:.3f} ms")

        return results

    def benchmark_throughput_scaling(self, batch_sizes: List[int] = None) -> Dict:
        """
        Benchmark throughput scaling with batch size

        Tests concurrent query processing
        """
        if batch_sizes is None:
            batch_sizes = [1, 4, 8, 16, 32, 64, 128]

        print(f"\n{'='*80}")
        print(f"Throughput Scaling Benchmark")
        print(f"{'='*80}\n")

        results = {}

        for batch_size in batch_sizes:
            print(f"\nBatch size: {batch_size}")

            # Create batch
            queries = self.test_queries[:batch_size]
            users = self.test_users[:batch_size]

            # Warmup
            for _ in range(5):
                for user, query in zip(users, queries):
                    _ = self.system.personalized_search(user, query, top_k=10)

            # Benchmark
            torch.cuda.synchronize()
            start = time.perf_counter()

            num_batches = max(10, 100 // batch_size)
            for _ in range(num_batches):
                for user, query in zip(users, queries):
                    _ = self.system.personalized_search(user, query, top_k=10)

            torch.cuda.synchronize()
            end = time.perf_counter()

            total_queries = batch_size * num_batches
            elapsed = end - start
            qps = total_queries / elapsed
            latency_ms = (elapsed / total_queries) * 1000

            results[batch_size] = {
                'qps': qps,
                'latency_ms': latency_ms,
                'num_queries': total_queries
            }

            print(f"  QPS: {qps:.1f}")
            print(f"  Latency: {latency_ms:.3f} ms")

        return results

    def measure_gpu_utilization(self) -> Dict:
        """
        Measure GPU memory usage and utilization

        Returns:
            Dict with GPU metrics
        """
        print(f"\n{'='*80}")
        print(f"GPU Resource Utilization")
        print(f"{'='*80}\n")

        if not torch.cuda.is_available():
            print("CUDA not available")
            return {}

        device_props = torch.cuda.get_device_properties(self.device)
        mem_allocated = torch.cuda.memory_allocated(self.device) / (1024**3)
        mem_reserved = torch.cuda.memory_reserved(self.device) / (1024**3)
        mem_total = device_props.total_memory / (1024**3)

        results = {
            'device_name': device_props.name,
            'cuda_version': torch.version.cuda,
            'memory_allocated_gb': mem_allocated,
            'memory_reserved_gb': mem_reserved,
            'memory_total_gb': mem_total,
            'memory_utilization_pct': (mem_allocated / mem_total) * 100,
            'compute_capability': f"{device_props.major}.{device_props.minor}",
            'multiprocessor_count': device_props.multi_processor_count
        }

        print(f"Device: {results['device_name']}")
        print(f"Compute Capability: {results['compute_capability']}")
        print(f"Memory Allocated: {mem_allocated:.2f} / {mem_total:.2f} GB ({results['memory_utilization_pct']:.1f}%)")
        print(f"Multiprocessors: {results['multiprocessor_count']}")

        return results

    def run_full_benchmark(self) -> Dict:
        """Run complete benchmark suite"""
        print(f"\n{'#'*80}")
        print(f"# GPU Hyper-Personalization Benchmark Suite")
        print(f"# Device: {torch.cuda.get_device_name(self.device)}")
        print(f"# Encoder: {self.system.encoder_type}")
        print(f"{'#'*80}")

        results = {
            'system_info': {
                'device': str(self.device),
                'encoder_type': self.system.encoder_type,
                'num_media_items': len(self.system.media_ids),
                'num_test_users': self.num_users,
                'num_test_queries': self.num_queries
            },
            'gpu_utilization': self.measure_gpu_utilization(),
            'encoding_latency': self.benchmark_encoding_latency(num_iterations=100),
            'end_to_end': self.benchmark_end_to_end(num_iterations=100),
            'throughput_scaling': self.benchmark_throughput_scaling()
        }

        # Summary
        print(f"\n{'#'*80}")
        print(f"# BENCHMARK SUMMARY")
        print(f"{'#'*80}")
        print(f"\nEncoding Performance:")
        print(f"  Mean Latency: {results['encoding_latency']['mean_ms']:.3f} ms")
        print(f"  P99 Latency:  {results['encoding_latency']['p99_ms']:.3f} ms")
        print(f"  Max QPS:      {results['encoding_latency']['qps']:.1f}")

        print(f"\nEnd-to-End Performance:")
        print(f"  Mean Latency: {results['end_to_end']['total']['mean_ms']:.3f} ms")
        print(f"  P99 Latency:  {results['end_to_end']['total']['p99_ms']:.3f} ms")
        print(f"  QPS:          {results['end_to_end']['qps']:.1f}")

        print(f"\nGPU Utilization:")
        print(f"  Memory:       {results['gpu_utilization']['memory_utilization_pct']:.1f}%")
        print(f"  Allocated:    {results['gpu_utilization']['memory_allocated_gb']:.2f} GB")

        # Compare to targets
        print(f"\n{'='*80}")
        print(f"Target Achievement (RTX A6000):")
        print(f"{'='*80}")

        encoding_target = 1.0  # <1ms
        total_target = 2.0     # <2ms
        qps_target = 1000      # >1000 QPS

        encoding_pass = results['encoding_latency']['mean_ms'] < encoding_target
        total_pass = results['end_to_end']['total']['mean_ms'] < total_target
        qps_pass = results['end_to_end']['qps'] > qps_target

        print(f"  Encoding < 1ms:     {'✅ PASS' if encoding_pass else '❌ FAIL'} ({results['encoding_latency']['mean_ms']:.3f} ms)")
        print(f"  Total < 2ms:        {'✅ PASS' if total_pass else '❌ FAIL'} ({results['end_to_end']['total']['mean_ms']:.3f} ms)")
        print(f"  QPS > 1000:         {'✅ PASS' if qps_pass else '❌ FAIL'} ({results['end_to_end']['qps']:.1f})")

        overall_pass = encoding_pass and total_pass and qps_pass
        print(f"\n  Overall:            {'✅ ALL TARGETS MET' if overall_pass else '⚠️  PARTIAL SUCCESS'}")

        return results


def main():
    parser = argparse.ArgumentParser(description="GPU Hyper-Personalization Benchmark")
    parser.add_argument('--mode', choices=['pytorch', 'tensorrt', 'both'], default='pytorch',
                       help='Benchmark mode')
    parser.add_argument('--engine', type=str, default=None,
                       help='TensorRT engine path (for tensorrt mode)')
    parser.add_argument('--num-users', type=int, default=1000,
                       help='Number of test users')
    parser.add_argument('--num-queries', type=int, default=100,
                       help='Number of test queries')
    parser.add_argument('--output', type=str, default=None,
                       help='Output JSON file for results')
    args = parser.parse_args()

    results_all = {}

    # PyTorch benchmark
    if args.mode in ['pytorch', 'both']:
        print(f"\n{'#'*80}")
        print(f"# PYTORCH BENCHMARK")
        print(f"{'#'*80}")

        system_pytorch = GPUHyperPersonalization(use_tensorrt=False)
        benchmark_pytorch = GPUBenchmark(
            system_pytorch,
            num_users=args.num_users,
            num_queries=args.num_queries
        )

        results_all['pytorch'] = benchmark_pytorch.run_full_benchmark()

    # TensorRT benchmark
    if args.mode in ['tensorrt', 'both']:
        print(f"\n{'#'*80}")
        print(f"# TENSORRT BENCHMARK")
        print(f"{'#'*80}")

        if args.engine and Path(args.engine).exists():
            try:
                system_trt = GPUHyperPersonalization(use_tensorrt=True)

                if system_trt.encoder_type == 'tensorrt':
                    benchmark_trt = GPUBenchmark(
                        system_trt,
                        num_users=args.num_users,
                        num_queries=args.num_queries
                    )
                    results_all['tensorrt'] = benchmark_trt.run_full_benchmark()
                else:
                    print("⚠️  TensorRT engine failed to load, using PyTorch")
                    results_all['tensorrt'] = {'error': 'TensorRT engine not loaded'}

            except Exception as e:
                print(f"❌ TensorRT benchmark failed: {e}")
                results_all['tensorrt'] = {'error': str(e)}
        else:
            print(f"⚠️  TensorRT engine not found: {args.engine}")
            print(f"   Run PyTorch benchmark only")

    # Comparison
    if 'pytorch' in results_all and 'tensorrt' in results_all and 'error' not in results_all['tensorrt']:
        print(f"\n{'#'*80}")
        print(f"# PYTORCH vs TENSORRT COMPARISON")
        print(f"{'#'*80}")

        pt_qps = results_all['pytorch']['end_to_end']['qps']
        trt_qps = results_all['tensorrt']['end_to_end']['qps']
        speedup = trt_qps / pt_qps

        pt_enc = results_all['pytorch']['encoding_latency']['mean_ms']
        trt_enc = results_all['tensorrt']['encoding_latency']['mean_ms']
        enc_speedup = pt_enc / trt_enc

        print(f"\nQPS:")
        print(f"  PyTorch:   {pt_qps:.1f}")
        print(f"  TensorRT:  {trt_qps:.1f}")
        print(f"  Speedup:   {speedup:.2f}x")

        print(f"\nEncoding Latency:")
        print(f"  PyTorch:   {pt_enc:.3f} ms")
        print(f"  TensorRT:  {trt_enc:.3f} ms")
        print(f"  Speedup:   {enc_speedup:.2f}x")

    # Save results
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            json.dump(results_all, f, indent=2)

        print(f"\n✅ Results saved to {output_path}")


if __name__ == "__main__":
    main()
