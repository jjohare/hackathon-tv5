#!/usr/bin/env python3
"""
Test batch processing performance

Tests:
1. Single query (baseline)
2. Sequential queries (without batching)
3. Batch queries (with batching)
4. Concurrent batch queries (stress test)

Expected results:
- Batch processing: 1000+ QPS with TensorRT
- Sequential: ~40 QPS
- Improvement: 25x+ with batching
"""

import time
import json
import requests
import concurrent.futures
from typing import List

BASE_URL = "http://localhost:5000"


def test_single_query():
    """Test single query (baseline)"""
    print("\n" + "=" * 80)
    print("Test 1: Single Query (Baseline)")
    print("=" * 80)

    query = "action movies with sci-fi elements"

    start = time.time()
    response = requests.post(
        f"{BASE_URL}/api/query",
        json={"query": query, "limit": 5}
    )
    elapsed = (time.time() - start) * 1000

    result = response.json()
    print(f"Query: {query}")
    print(f"Results: {len(result.get('results', []))}")
    print(f"Time: {elapsed:.3f}ms")
    print(f"Encoding time: {result.get('performance', {}).get('encoding_time_ms', 0):.3f}ms")


def test_sequential_queries(num_queries: int = 100):
    """Test sequential queries without batching"""
    print("\n" + "=" * 80)
    print(f"Test 2: Sequential Queries (n={num_queries})")
    print("=" * 80)

    queries = [
        f"action movie {i}" for i in range(num_queries)
    ]

    start = time.time()
    for query in queries:
        response = requests.post(
            f"{BASE_URL}/api/query",
            json={"query": query, "limit": 5, "use_batch": False}
        )

    elapsed = (time.time() - start) * 1000
    qps = num_queries / (elapsed / 1000)

    print(f"Queries: {num_queries}")
    print(f"Total time: {elapsed:.3f}ms")
    print(f"Avg time per query: {elapsed / num_queries:.3f}ms")
    print(f"QPS: {qps:.2f}")


def test_batch_endpoint(num_queries: int = 100):
    """Test batch endpoint"""
    print("\n" + "=" * 80)
    print(f"Test 3: Batch Endpoint (n={num_queries})")
    print("=" * 80)

    queries = [
        f"action movie {i}" for i in range(num_queries)
    ]

    start = time.time()
    response = requests.post(
        f"{BASE_URL}/api/query/batch",
        json={"queries": queries, "limit": 5}
    )
    elapsed = (time.time() - start) * 1000

    result = response.json()
    batch_perf = result.get('batch_performance', {})

    print(f"Queries: {batch_perf.get('total_queries', 0)}")
    print(f"Total time: {batch_perf.get('total_time_ms', 0):.3f}ms")
    print(f"Avg time per query: {batch_perf.get('avg_time_per_query_ms', 0):.3f}ms")
    print(f"QPS: {batch_perf.get('qps', 0):.2f}")


def test_concurrent_batch_queries(num_workers: int = 10, queries_per_worker: int = 10):
    """Test concurrent batch queries (stress test)"""
    print("\n" + "=" * 80)
    print(f"Test 4: Concurrent Batch Queries (workers={num_workers}, queries/worker={queries_per_worker})")
    print("=" * 80)

    def worker(worker_id: int):
        """Worker function for concurrent requests"""
        queries = [
            f"worker{worker_id}_query{i}" for i in range(queries_per_worker)
        ]

        response = requests.post(
            f"{BASE_URL}/api/query/batch",
            json={"queries": queries, "limit": 5}
        )

        return response.json()

    start = time.time()

    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(worker, i) for i in range(num_workers)]
        results = [f.result() for f in concurrent.futures.as_completed(futures)]

    elapsed = (time.time() - start) * 1000
    total_queries = num_workers * queries_per_worker
    qps = total_queries / (elapsed / 1000)

    print(f"Workers: {num_workers}")
    print(f"Queries per worker: {queries_per_worker}")
    print(f"Total queries: {total_queries}")
    print(f"Total time: {elapsed:.3f}ms")
    print(f"Avg time per query: {elapsed / total_queries:.3f}ms")
    print(f"QPS: {qps:.2f}")


def test_batch_with_use_batch_flag(num_queries: int = 100):
    """Test individual queries with use_batch=True flag"""
    print("\n" + "=" * 80)
    print(f"Test 5: Individual Queries with Batch Flag (n={num_queries})")
    print("=" * 80)

    queries = [
        f"action movie {i}" for i in range(num_queries)
    ]

    # Send concurrent individual requests with use_batch=True
    def send_query(query):
        return requests.post(
            f"{BASE_URL}/api/query",
            json={"query": query, "limit": 5, "use_batch": True}
        )

    start = time.time()

    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(send_query, q) for q in queries]
        results = [f.result() for f in concurrent.futures.as_completed(futures)]

    elapsed = (time.time() - start) * 1000
    qps = num_queries / (elapsed / 1000)

    print(f"Queries: {num_queries}")
    print(f"Total time: {elapsed:.3f}ms")
    print(f"Avg time per query: {elapsed / num_queries:.3f}ms")
    print(f"QPS: {qps:.2f}")


def check_server_status():
    """Check if server is running and get status"""
    try:
        response = requests.get(f"{BASE_URL}/api/status")
        status = response.json()

        print("\n" + "=" * 80)
        print("Server Status")
        print("=" * 80)
        print(f"Backend: {status.get('backend')}")
        print(f"Device: {status.get('device')}")
        print(f"CUDA available: {status.get('cuda_available')}")
        print(f"Items loaded: {status.get('items_loaded')}")
        print(f"\nBatch Processor:")
        batch_info = status.get('batch_processor', {})
        print(f"  Enabled: {batch_info.get('enabled')}")
        print(f"  Max batch size: {batch_info.get('max_batch_size')}")
        print(f"  Max wait: {batch_info.get('max_wait_ms')}ms")
        print(f"  Queue size: {batch_info.get('queue_size')}")
        print(f"  Running: {batch_info.get('running')}")

        return True

    except Exception as e:
        print(f"\n❌ Server not available: {e}")
        print(f"Start server with: python scripts/server/query_interface.py")
        return False


def main():
    """Run all tests"""
    print("\n" + "=" * 80)
    print("Batch Processing Performance Tests")
    print("=" * 80)

    # Check server
    if not check_server_status():
        return

    # Run tests
    test_single_query()
    test_sequential_queries(num_queries=100)
    test_batch_endpoint(num_queries=100)
    test_batch_with_use_batch_flag(num_queries=100)
    test_concurrent_batch_queries(num_workers=10, queries_per_worker=10)

    print("\n" + "=" * 80)
    print("Tests Complete")
    print("=" * 80)


if __name__ == "__main__":
    main()
