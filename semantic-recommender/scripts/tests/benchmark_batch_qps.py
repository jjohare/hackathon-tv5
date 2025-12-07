#!/usr/bin/env python3
"""
Batch Processing QPS Benchmark

Measures actual QPS with concurrent load to verify 1000+ QPS target.
"""

import time
import json
import requests
import concurrent.futures
import argparse
from collections import defaultdict

BASE_URL = "http://localhost:5000"


def benchmark_qps(
    num_requests: int = 1000,
    num_workers: int = 20,
    batch_size: int = 32,
    use_batch_endpoint: bool = True
):
    """
    Benchmark QPS with concurrent load

    Args:
        num_requests: Total number of queries to send
        num_workers: Number of concurrent workers
        batch_size: Queries per batch (for batch endpoint)
        use_batch_endpoint: Use /api/query/batch vs /api/query with use_batch=True

    Returns:
        dict with performance metrics
    """
    print(f"\n{'=' * 80}")
    print(f"QPS Benchmark")
    print(f"{'=' * 80}")
    print(f"Total requests: {num_requests}")
    print(f"Workers: {num_workers}")
    print(f"Batch size: {batch_size}")
    print(f"Endpoint: {'batch' if use_batch_endpoint else 'single with batching'}")
    print(f"{'=' * 80}\n")

    # Generate queries
    queries = [f"query_{i}" for i in range(num_requests)]

    # Split into batches
    if use_batch_endpoint:
        batches = [
            queries[i:i + batch_size]
            for i in range(0, len(queries), batch_size)
        ]
    else:
        batches = [[q] for q in queries]  # Each request is single query

    results = {
        'total_queries': num_requests,
        'num_workers': num_workers,
        'batch_size': batch_size,
        'num_batches': len(batches),
        'successes': 0,
        'failures': 0,
        'errors': defaultdict(int)
    }

    def send_batch(batch_queries):
        """Send batch request"""
        try:
            if use_batch_endpoint:
                response = requests.post(
                    f"{BASE_URL}/api/query/batch",
                    json={"queries": batch_queries, "limit": 5},
                    timeout=10
                )
            else:
                # Send single query with use_batch=True
                response = requests.post(
                    f"{BASE_URL}/api/query",
                    json={"query": batch_queries[0], "limit": 5, "use_batch": True},
                    timeout=10
                )

            if response.status_code == 200:
                return {'success': True, 'count': len(batch_queries)}
            else:
                return {'success': False, 'error': f"HTTP {response.status_code}"}

        except Exception as e:
            return {'success': False, 'error': str(e)}

    # Run benchmark
    print("Starting benchmark...")
    start_time = time.time()

    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(send_batch, batch) for batch in batches]

        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            if result['success']:
                results['successes'] += result['count']
            else:
                results['failures'] += 1
                results['errors'][result['error']] += 1

    elapsed = time.time() - start_time

    # Calculate metrics
    results['total_time_s'] = elapsed
    results['avg_time_per_query_ms'] = (elapsed / num_requests) * 1000
    results['qps'] = num_requests / elapsed

    # Print results
    print(f"\n{'=' * 80}")
    print(f"Results")
    print(f"{'=' * 80}")
    print(f"Total queries: {results['total_queries']}")
    print(f"Successful: {results['successes']}")
    print(f"Failed: {results['failures']}")
    print(f"Total time: {results['total_time_s']:.3f}s")
    print(f"Avg time per query: {results['avg_time_per_query_ms']:.3f}ms")
    print(f"QPS: {results['qps']:.2f}")
    print(f"{'=' * 80}")

    if results['failures'] > 0:
        print(f"\n⚠️  Errors:")
        for error, count in results['errors'].items():
            print(f"  {error}: {count}")

    # Check if target achieved
    if results['qps'] >= 1000:
        print(f"\n✅ Target achieved: {results['qps']:.2f} QPS >= 1000 QPS")
    else:
        print(f"\n❌ Target not achieved: {results['qps']:.2f} QPS < 1000 QPS")
        print(f"   Improvement needed: {1000 - results['qps']:.2f} QPS")

    return results


def main():
    parser = argparse.ArgumentParser(description='Batch Processing QPS Benchmark')
    parser.add_argument('--requests', type=int, default=1000, help='Total number of requests')
    parser.add_argument('--workers', type=int, default=20, help='Number of concurrent workers')
    parser.add_argument('--batch-size', type=int, default=32, help='Queries per batch')
    parser.add_argument('--mode', choices=['batch', 'single'], default='batch',
                       help='Use batch endpoint or single with batching')
    args = parser.parse_args()

    # Check server
    try:
        response = requests.get(f"{BASE_URL}/api/status", timeout=5)
        status = response.json()
        print(f"\nServer Status:")
        print(f"  Backend: {status.get('backend')}")
        print(f"  Device: {status.get('device')}")
        print(f"  Batch processor: {status.get('batch_processor', {}).get('enabled')}")
    except Exception as e:
        print(f"\n❌ Server not available: {e}")
        print(f"Start server with: python scripts/server/query_interface.py")
        return

    # Run benchmark
    benchmark_qps(
        num_requests=args.requests,
        num_workers=args.workers,
        batch_size=args.batch_size,
        use_batch_endpoint=(args.mode == 'batch')
    )


if __name__ == "__main__":
    main()
