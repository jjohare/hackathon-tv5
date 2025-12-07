#!/usr/bin/env python3
"""
Batch Processing Performance Test - 1000 QPS Target

Validates:
1. Concurrent query throughput meets 1000 QPS
2. Latency distribution (p50, p95, p99)
3. GPU memory efficiency during batch processing
4. Thread pool scaling behavior
"""

import asyncio
import time
import statistics
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Tuple
import sys
from pathlib import Path

import pytest
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

try:
    from scripts.utils.gpu_hyper_personalization import GPUHyperPersonalization
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False


def send_query(system: 'GPUHyperPersonalization', query: str, user_id: str) -> Dict:
    """Send a single query to the recommendation system"""
    try:
        result = system.personalized_search(
            user_id=user_id,
            query=query,
            top_k=10,
            context=None
        )
        return {
            'status': 'success',
            'latency_ms': result['timing']['total_ms'],
            'results_count': len(result['results'])
        }
    except Exception as e:
        return {
            'status': 'error',
            'error': str(e),
            'latency_ms': 0
        }


@pytest.mark.skipif(not GPU_AVAILABLE, reason="GPU not available")
class TestBatchPerformance:
    """Test suite for batch processing performance"""

    @classmethod
    def setup_class(cls):
        """Initialize recommendation system once for all tests"""
        cls.system = GPUHyperPersonalization(use_tensorrt=True)

    def test_batch_throughput_1000_qps(self):
        """Test 1000 QPS with batch processing"""

        # Test configuration
        target_qps = 1000
        test_duration_seconds = 2
        expected_queries = target_qps * test_duration_seconds

        queries = [
            "action movies",
            "sci-fi thriller",
            "romantic comedy",
            "horror film",
            "animated adventure"
        ] * (expected_queries // 5 + 1)

        queries = queries[:expected_queries]  # Exactly 2000 queries

        print(f"\n{'='*80}")
        print(f"TEST: Batch Throughput - {expected_queries} queries")
        print(f"{'='*80}\n")

        latencies = []
        errors = 0

        start = time.time()

        # Send all queries concurrently
        with ThreadPoolExecutor(max_workers=50) as executor:
            futures = [
                executor.submit(send_query, self.system, query, f"user_{i}")
                for i, query in enumerate(queries)
            ]

            for future in as_completed(futures):
                result = future.result()
                if result['status'] == 'success':
                    latencies.append(result['latency_ms'])
                else:
                    errors += 1

        elapsed = time.time() - start
        qps = len(queries) / elapsed

        # Assertions
        assert qps >= target_qps, f"QPS {qps:.1f} below target {target_qps}"
        assert errors == 0, f"Got {errors} errors during batch processing"
        assert len(latencies) == expected_queries, f"Expected {expected_queries} results, got {len(latencies)}"

        # Print results
        print(f"✅ Achieved {qps:.1f} QPS (target: {target_qps})")
        print(f"   Total queries: {len(queries)}")
        print(f"   Elapsed time: {elapsed:.2f}s")
        print(f"   Error rate: {errors / len(queries) * 100:.2f}%")
        print(f"\nLatency Distribution:")
        print(f"   P50: {np.percentile(latencies, 50):.2f}ms")
        print(f"   P95: {np.percentile(latencies, 95):.2f}ms")
        print(f"   P99: {np.percentile(latencies, 99):.2f}ms")
        print(f"   Mean: {statistics.mean(latencies):.2f}ms")
        print(f"   Max: {max(latencies):.2f}ms")

    def test_latency_distribution(self):
        """Test latency stays within acceptable bounds"""

        num_queries = 100
        queries = ["thriller movies"] * num_queries

        latencies = []

        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [
                executor.submit(send_query, self.system, query, f"user_{i}")
                for i, query in enumerate(queries)
            ]

            for future in as_completed(futures):
                result = future.result()
                if result['status'] == 'success':
                    latencies.append(result['latency_ms'])

        # Latency requirements
        p95 = np.percentile(latencies, 95)
        p99 = np.percentile(latencies, 99)

        assert p95 < 100, f"P95 latency {p95:.2f}ms exceeds 100ms threshold"
        assert p99 < 200, f"P99 latency {p99:.2f}ms exceeds 200ms threshold"

        print(f"\n✅ Latency within acceptable bounds:")
        print(f"   P95: {p95:.2f}ms (< 100ms)")
        print(f"   P99: {p99:.2f}ms (< 200ms)")

    def test_memory_efficiency_under_load(self):
        """Test GPU memory doesn't grow excessively under load"""
        import torch

        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        torch.cuda.reset_peak_memory_stats()
        initial_memory = torch.cuda.memory_allocated() / (1024 ** 3)

        # Run sustained load
        queries = ["action thriller"] * 500

        with ThreadPoolExecutor(max_workers=20) as executor:
            futures = [
                executor.submit(send_query, self.system, query, f"user_{i}")
                for i, query in enumerate(queries)
            ]

            for future in as_completed(futures):
                future.result()

        final_memory = torch.cuda.memory_allocated() / (1024 ** 3)
        peak_memory = torch.cuda.max_memory_allocated() / (1024 ** 3)

        memory_growth = final_memory - initial_memory

        assert memory_growth < 1.0, f"Memory grew by {memory_growth:.2f} GB (should be < 1 GB)"

        print(f"\n✅ Memory efficiency validated:")
        print(f"   Initial: {initial_memory:.2f} GB")
        print(f"   Final: {final_memory:.2f} GB")
        print(f"   Peak: {peak_memory:.2f} GB")
        print(f"   Growth: {memory_growth:.2f} GB")

    def test_thread_pool_scaling(self):
        """Test performance with different thread pool sizes"""

        num_queries = 100
        queries = ["sci-fi movies"] * num_queries
        worker_counts = [1, 5, 10, 20, 50]

        results = {}

        print(f"\n{'='*80}")
        print("Thread Pool Scaling Test")
        print(f"{'='*80}\n")

        for workers in worker_counts:
            start = time.time()

            with ThreadPoolExecutor(max_workers=workers) as executor:
                futures = [
                    executor.submit(send_query, self.system, query, f"user_{i}")
                    for i, query in enumerate(queries)
                ]

                for future in as_completed(futures):
                    future.result()

            elapsed = time.time() - start
            qps = num_queries / elapsed
            results[workers] = qps

            print(f"Workers: {workers:2d} → {qps:7.1f} QPS ({elapsed:.2f}s)")

        # Verify scaling improves performance
        assert results[50] > results[1], "Scaling to 50 workers should improve QPS"
        assert results[20] > results[5], "Scaling to 20 workers should improve QPS"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
