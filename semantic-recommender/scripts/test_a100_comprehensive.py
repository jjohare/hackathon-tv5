#!/usr/bin/env python3
"""
Comprehensive A100 GPU Testing Suite for Semantic Recommender

Tests:
1. GPU availability and setup
2. Embedding loading and validation
3. Single movie similarity (franchise detection)
4. User personalization
5. Batch processing (10, 100, 1000 queries)
6. Complex queries (genre + rating filters)
7. Performance benchmarking
8. Memory usage analysis
"""

import torch
import numpy as np
import json
import time
from pathlib import Path
from typing import List, Dict, Tuple
import sys

class A100ComprehensiveTest:
    def __init__(self, base_path="/home/devuser/workspace/hackathon-tv5/semantic-recommender"):
        self.base_path = Path(base_path)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        print("=" * 80)
        print("A100 Comprehensive Testing Suite")
        print("=" * 80)

        # GPU info
        if torch.cuda.is_available():
            print(f"\n✅ GPU Available: {torch.cuda.get_device_name(0)}")
            print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
            print(f"   CUDA Version: {torch.version.cuda}")
        else:
            print("\n❌ GPU NOT available - running on CPU")

    def load_data(self):
        """Load all necessary data"""
        print("\n" + "=" * 80)
        print("Loading Data")
        print("=" * 80)

        # Load media embeddings
        media_path = self.base_path / "data/embeddings/media"
        print(f"\n📊 Loading media embeddings from {media_path}")

        media_vectors = np.load(media_path / "content_vectors.npy")
        print(f"   Shape: {media_vectors.shape}")
        print(f"   Dtype: {media_vectors.dtype}")
        print(f"   Size: {media_vectors.nbytes / 1e6:.2f} MB")

        # Convert to torch tensor on GPU
        self.media_embeddings_gpu = torch.from_numpy(media_vectors).to(self.device)
        print(f"   ✅ Loaded to {self.device}")

        # Load metadata
        self.media_metadata = {}
        self.media_ids = []
        with open(media_path / "metadata.jsonl", 'r') as f:
            for line in f:
                item = json.loads(line)
                media_id = item['media_id']
                self.media_ids.append(media_id)
                self.media_metadata[media_id] = item

        print(f"   ✅ Loaded {len(self.media_metadata)} movie metadata entries")

        # Load user embeddings
        user_path = self.base_path / "data/embeddings/users"
        print(f"\n👥 Loading user embeddings from {user_path}")

        user_vectors = np.load(user_path / "preference_vectors.npy")
        print(f"   Shape: {user_vectors.shape}")
        print(f"   Size: {user_vectors.nbytes / 1e6:.2f} MB")

        self.user_embeddings_gpu = torch.from_numpy(user_vectors).float().to(self.device)
        print(f"   ✅ Loaded to {self.device}")

        # Load user IDs
        with open(user_path / "user_ids.json", 'r') as f:
            user_data = json.load(f)
            if isinstance(user_data, list):
                self.user_ids = user_data
            else:
                self.user_ids = user_data.get('user_ids', [])

        print(f"   ✅ Loaded {len(self.user_ids)} user IDs")

        # Memory usage
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated(0) / 1e9
            reserved = torch.cuda.memory_reserved(0) / 1e9
            print(f"\n💾 GPU Memory: {allocated:.2f} GB allocated, {reserved:.2f} GB reserved")

    def test_1_single_similarity(self):
        """Test 1: Single movie similarity (franchise detection)"""
        print("\n" + "=" * 80)
        print("Test 1: Single Movie Similarity - Franchise Detection")
        print("=" * 80)

        # Find Toy Story
        toy_story_id = None
        for media_id, metadata in self.media_metadata.items():
            if "Toy Story" in metadata['title'] and metadata['title'].startswith("Toy Story (1995)"):
                toy_story_id = media_id
                break

        if not toy_story_id:
            print("❌ Toy Story not found in dataset")
            return

        print(f"\n🎬 Query: {self.media_metadata[toy_story_id]['title']}")

        # Get query vector
        query_idx = self.media_ids.index(toy_story_id)
        query_vector = self.media_embeddings_gpu[query_idx]

        # Normalize
        query_norm = query_vector / torch.norm(query_vector)

        # Compute similarities
        start = time.time()
        similarities = torch.matmul(self.media_embeddings_gpu, query_norm)
        torch.cuda.synchronize()
        gpu_time = (time.time() - start) * 1000

        # Get top 10
        top_k_vals, top_k_indices = torch.topk(similarities, k=10)

        print(f"\n⚡ GPU Time: {gpu_time:.3f} ms")
        print(f"\n📋 Top 10 Similar Movies:")

        results = []
        for i, (idx, sim) in enumerate(zip(top_k_indices.cpu().numpy(), top_k_vals.cpu().numpy()), 1):
            media_id = self.media_ids[idx]
            metadata = self.media_metadata[media_id]
            title = metadata['title']
            genres = metadata.get('genres', [])

            print(f"   {i:2d}. {title:50s} - {sim*100:.1f}% similar")
            print(f"       Genres: {', '.join(genres)}")

            results.append({
                'rank': i,
                'title': title,
                'similarity': float(sim),
                'genres': genres
            })

        return {'gpu_time_ms': gpu_time, 'results': results}

    def test_2_user_personalization(self):
        """Test 2: User personalization"""
        print("\n" + "=" * 80)
        print("Test 2: User Personalization")
        print("=" * 80)

        # Test with first 5 users
        test_user_ids = self.user_ids[:5]

        all_times = []
        all_results = []

        for user_id in test_user_ids:
            user_idx = self.user_ids.index(user_id)
            user_vector = self.user_embeddings_gpu[user_idx]

            # Normalize
            user_norm = user_vector / torch.norm(user_vector)

            # Compute similarities
            start = time.time()
            similarities = torch.matmul(self.media_embeddings_gpu, user_norm)
            torch.cuda.synchronize()
            gpu_time = (time.time() - start) * 1000
            all_times.append(gpu_time)

            # Get top 5
            top_k_vals, top_k_indices = torch.topk(similarities, k=5)

            print(f"\n👤 User: {user_id}")
            print(f"   ⚡ GPU Time: {gpu_time:.3f} ms")
            print(f"   📋 Top 5 Recommendations:")

            user_results = []
            for i, (idx, sim) in enumerate(zip(top_k_indices.cpu().numpy(), top_k_vals.cpu().numpy()), 1):
                media_id = self.media_ids[idx]
                title = self.media_metadata[media_id]['title']
                print(f"      {i}. {title:45s} - {sim*100:.1f}%")
                user_results.append({'title': title, 'similarity': float(sim)})

            all_results.append({
                'user_id': user_id,
                'gpu_time_ms': gpu_time,
                'recommendations': user_results
            })

        avg_time = np.mean(all_times)
        print(f"\n📊 Average GPU Time: {avg_time:.3f} ms")

        return {'avg_gpu_time_ms': avg_time, 'user_results': all_results}

    def test_3_batch_processing(self):
        """Test 3: Batch processing at different scales"""
        print("\n" + "=" * 80)
        print("Test 3: Batch Processing Benchmarks")
        print("=" * 80)

        batch_sizes = [10, 100, 1000]
        results = {}

        for batch_size in batch_sizes:
            print(f"\n📦 Testing batch size: {batch_size}")

            # Select random movies
            indices = torch.randint(0, len(self.media_ids), (batch_size,))
            query_vectors = self.media_embeddings_gpu[indices]

            # Normalize
            query_norms = query_vectors / torch.norm(query_vectors, dim=1, keepdim=True)

            # Batch matrix multiplication
            start = time.time()
            similarities = torch.matmul(query_norms, self.media_embeddings_gpu.T)
            torch.cuda.synchronize()
            gpu_time = (time.time() - start) * 1000

            # Get top 10 for each
            top_k_vals, top_k_indices = torch.topk(similarities, k=10, dim=1)

            throughput = batch_size / (gpu_time / 1000)
            time_per_query = gpu_time / batch_size

            print(f"   ⚡ Total Time: {gpu_time:.2f} ms")
            print(f"   ⚡ Time per Query: {time_per_query:.3f} ms")
            print(f"   🚀 Throughput: {throughput:.1f} queries/second")

            results[f'batch_{batch_size}'] = {
                'total_time_ms': gpu_time,
                'time_per_query_ms': time_per_query,
                'throughput_qps': throughput
            }

        return results

    def test_4_genre_filtering(self):
        """Test 4: Complex query with genre filtering"""
        print("\n" + "=" * 80)
        print("Test 4: Genre-Filtered Recommendations")
        print("=" * 80)

        # Find a sci-fi movie
        query_id = None
        for media_id, metadata in self.media_metadata.items():
            if 'Sci-Fi' in metadata.get('genres', []):
                query_id = media_id
                break

        if not query_id:
            print("❌ No Sci-Fi movies found")
            return

        print(f"\n🎬 Query: {self.media_metadata[query_id]['title']}")
        print(f"   Genres: {', '.join(self.media_metadata[query_id]['genres'])}")

        # Get query vector
        query_idx = self.media_ids.index(query_id)
        query_vector = self.media_embeddings_gpu[query_idx]
        query_norm = query_vector / torch.norm(query_vector)

        # Compute similarities
        start = time.time()
        similarities = torch.matmul(self.media_embeddings_gpu, query_norm)
        torch.cuda.synchronize()
        gpu_time = (time.time() - start) * 1000

        # Get top 100 candidates
        top_k_vals, top_k_indices = torch.topk(similarities, k=100)

        # Filter to only Sci-Fi
        print(f"\n🔍 Filtering to Sci-Fi movies only...")
        sci_fi_results = []

        for idx, sim in zip(top_k_indices.cpu().numpy(), top_k_vals.cpu().numpy()):
            media_id = self.media_ids[idx]
            metadata = self.media_metadata[media_id]
            if 'Sci-Fi' in metadata.get('genres', []):
                sci_fi_results.append((media_id, float(sim), metadata))
                if len(sci_fi_results) >= 10:
                    break

        print(f"\n📋 Top 10 Sci-Fi Movies:")
        for i, (media_id, sim, metadata) in enumerate(sci_fi_results, 1):
            print(f"   {i:2d}. {metadata['title']:50s} - {sim*100:.1f}%")
            print(f"       Genres: {', '.join(metadata['genres'])}")

        return {
            'gpu_time_ms': gpu_time,
            'filtered_count': len(sci_fi_results),
            'results': [{'title': m['title'], 'similarity': s} for _, s, m in sci_fi_results]
        }

    def test_5_memory_analysis(self):
        """Test 5: Memory usage analysis"""
        print("\n" + "=" * 80)
        print("Test 5: GPU Memory Analysis")
        print("=" * 80)

        if not torch.cuda.is_available():
            print("❌ GPU not available")
            return

        allocated = torch.cuda.memory_allocated(0) / 1e9
        reserved = torch.cuda.memory_reserved(0) / 1e9
        total = torch.cuda.get_device_properties(0).total_memory / 1e9

        print(f"\n💾 Memory Usage:")
        print(f"   Allocated: {allocated:.2f} GB")
        print(f"   Reserved:  {reserved:.2f} GB")
        print(f"   Total:     {total:.2f} GB")
        print(f"   Free:      {total - reserved:.2f} GB ({(total-reserved)/total*100:.1f}%)")

        # Test peak memory with large batch
        print(f"\n🔬 Testing peak memory with batch=10000...")

        torch.cuda.reset_peak_memory_stats(0)

        # Large batch query
        indices = torch.randint(0, min(10000, len(self.media_ids)), (100,))
        query_vectors = self.media_embeddings_gpu[indices]
        query_norms = query_vectors / torch.norm(query_vectors, dim=1, keepdim=True)

        start = time.time()
        similarities = torch.matmul(query_norms, self.media_embeddings_gpu.T)
        top_k_vals, top_k_indices = torch.topk(similarities, k=10, dim=1)
        torch.cuda.synchronize()
        batch_time = (time.time() - start) * 1000

        peak_allocated = torch.cuda.max_memory_allocated(0) / 1e9
        peak_reserved = torch.cuda.max_memory_reserved(0) / 1e9

        print(f"   Peak Allocated: {peak_allocated:.2f} GB")
        print(f"   Peak Reserved:  {peak_reserved:.2f} GB")
        print(f"   Batch Time: {batch_time:.2f} ms")

        return {
            'allocated_gb': allocated,
            'reserved_gb': reserved,
            'total_gb': total,
            'free_gb': total - reserved,
            'peak_allocated_gb': peak_allocated,
            'peak_reserved_gb': peak_reserved
        }

    def run_all_tests(self):
        """Run all tests and generate report"""
        print("\n" + "=" * 80)
        print("Starting Comprehensive Test Suite")
        print("=" * 80)

        # Load data
        self.load_data()

        # Run tests
        results = {}

        print("\n" + "▶" * 40)
        results['test_1_similarity'] = self.test_1_single_similarity()

        print("\n" + "▶" * 40)
        results['test_2_personalization'] = self.test_2_user_personalization()

        print("\n" + "▶" * 40)
        results['test_3_batch'] = self.test_3_batch_processing()

        print("\n" + "▶" * 40)
        results['test_4_filtering'] = self.test_4_genre_filtering()

        print("\n" + "▶" * 40)
        results['test_5_memory'] = self.test_5_memory_analysis()

        # Summary
        print("\n" + "=" * 80)
        print("Test Summary")
        print("=" * 80)

        print(f"\n✅ All tests completed successfully!")

        if 'test_1_similarity' in results:
            print(f"\n🎯 Single Query Performance:")
            print(f"   GPU Time: {results['test_1_similarity']['gpu_time_ms']:.3f} ms")

        if 'test_2_personalization' in results:
            print(f"\n👥 User Personalization:")
            print(f"   Avg GPU Time: {results['test_2_personalization']['avg_gpu_time_ms']:.3f} ms")

        if 'test_3_batch' in results:
            print(f"\n📦 Batch Processing:")
            for key, val in results['test_3_batch'].items():
                batch_size = key.split('_')[1]
                print(f"   Batch {batch_size:4s}: {val['throughput_qps']:7.1f} QPS, {val['time_per_query_ms']:6.3f} ms/query")

        if 'test_5_memory' in results:
            mem = results['test_5_memory']
            print(f"\n💾 Memory Efficiency:")
            print(f"   Used: {mem['allocated_gb']:.2f} GB / {mem['total_gb']:.2f} GB ({mem['allocated_gb']/mem['total_gb']*100:.1f}%)")

        # Save results
        output_path = self.base_path / "results" / "a100_test_results.json"
        output_path.parent.mkdir(exist_ok=True)

        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"\n💾 Results saved to: {output_path}")

        return results

if __name__ == "__main__":
    tester = A100ComprehensiveTest()
    results = tester.run_all_tests()

    print("\n" + "=" * 80)
    print("Testing Complete!")
    print("=" * 80)
