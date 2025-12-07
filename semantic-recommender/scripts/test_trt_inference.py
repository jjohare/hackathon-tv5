#!/usr/bin/env python3
"""
Test script for TensorRT inference integration

Tests:
1. Engine loading
2. Tokenization
3. Inference execution
4. Memory management
5. Fallback behavior
6. Performance comparison
"""

import sys
import time
from pathlib import Path

import torch
import numpy as np

# Add utils to path
sys.path.insert(0, str(Path(__file__).parent / "utils"))

from trt_inference import TensorRTEncoder


def test_fallback_mode():
    """Test fallback when TensorRT unavailable"""
    print("\n" + "=" * 80)
    print("TEST 1: Fallback Mode (No Engine)")
    print("=" * 80)

    # Initialize without engine (should use fallback)
    encoder = TensorRTEncoder(
        "nonexistent.plan",
        "sentence-transformers/all-MiniLM-L6-v2"
    )

    test_texts = [
        "The quick brown fox jumps over the lazy dog",
        "Machine learning is transforming the world"
    ]

    embeddings = encoder.encode(test_texts)

    print(f"✅ Fallback mode working")
    print(f"   Shape: {embeddings.shape}")
    print(f"   Device: {embeddings.device}")
    print(f"   Using TensorRT: {encoder.use_tensorrt}")

    assert embeddings.shape[0] == len(test_texts)
    assert embeddings.shape[1] > 0  # Has embedding dimension
    assert not encoder.use_tensorrt


def test_with_tensorrt_engine(engine_path: str):
    """Test with actual TensorRT engine"""
    print("\n" + "=" * 80)
    print("TEST 2: TensorRT Engine")
    print("=" * 80)

    encoder = TensorRTEncoder(
        engine_path,
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )

    test_texts = [
        "Science fiction movie with time travel",
        "Romantic comedy set in Paris",
        "Action thriller with car chases",
        "Documentary about nature"
    ]

    print(f"Encoding {len(test_texts)} texts...")
    start = time.time()
    embeddings = encoder.encode(test_texts)
    elapsed = time.time() - start

    print(f"✅ TensorRT inference successful")
    print(f"   Shape: {embeddings.shape}")
    print(f"   Device: {embeddings.device}")
    print(f"   Time: {elapsed*1000:.2f}ms ({elapsed*1000/len(test_texts):.2f}ms per text)")
    print(f"   Using TensorRT: {encoder.use_tensorrt}")

    assert embeddings.shape[0] == len(test_texts)
    assert embeddings.device.type == "cuda"
    assert encoder.use_tensorrt


def test_batch_processing(encoder):
    """Test batch processing"""
    print("\n" + "=" * 80)
    print("TEST 3: Batch Processing")
    print("=" * 80)

    # Generate test texts
    test_texts = [f"Test sentence number {i}" for i in range(100)]

    print(f"Encoding {len(test_texts)} texts with batch_size=32...")
    start = time.time()
    embeddings = encoder.encode(test_texts, batch_size=32)
    elapsed = time.time() - start

    print(f"✅ Batch processing successful")
    print(f"   Shape: {embeddings.shape}")
    print(f"   Time: {elapsed*1000:.2f}ms ({elapsed*1000/len(test_texts):.2f}ms per text)")

    assert embeddings.shape[0] == len(test_texts)


def test_normalization(encoder):
    """Test embedding normalization"""
    print("\n" + "=" * 80)
    print("TEST 4: Embedding Normalization")
    print("=" * 80)

    test_texts = ["Test normalization"]

    # Without normalization
    emb_raw = encoder.encode(test_texts, normalize_embeddings=False)
    norm_raw = torch.norm(emb_raw[0]).item()

    # With normalization
    emb_norm = encoder.encode(test_texts, normalize_embeddings=True)
    norm_normalized = torch.norm(emb_norm[0]).item()

    print(f"✅ Normalization working")
    print(f"   Raw norm: {norm_raw:.4f}")
    print(f"   Normalized norm: {norm_normalized:.4f}")

    assert abs(norm_normalized - 1.0) < 0.01  # Should be ~1.0


def test_memory_management(encoder):
    """Test GPU memory management"""
    print("\n" + "=" * 80)
    print("TEST 5: Memory Management")
    print("=" * 80)

    if not torch.cuda.is_available():
        print("⚠️  CUDA not available, skipping memory test")
        return

    # Get initial memory
    torch.cuda.synchronize()
    mem_before = torch.cuda.memory_allocated() / (1024 ** 2)

    # Encode many batches
    for _ in range(10):
        texts = [f"Memory test {i}" for i in range(50)]
        embeddings = encoder.encode(texts)

    torch.cuda.synchronize()
    mem_after = torch.cuda.memory_allocated() / (1024 ** 2)

    print(f"✅ Memory management test")
    print(f"   Memory before: {mem_before:.2f} MB")
    print(f"   Memory after: {mem_after:.2f} MB")
    print(f"   Memory growth: {mem_after - mem_before:.2f} MB")

    # Should not have massive memory leak
    assert (mem_after - mem_before) < 100  # Less than 100MB growth


def benchmark_performance(encoder):
    """Benchmark TensorRT vs standard model"""
    print("\n" + "=" * 80)
    print("BENCHMARK: Performance Comparison")
    print("=" * 80)

    test_texts = [f"Benchmark sentence {i}" for i in range(100)]

    # Warmup
    _ = encoder.encode(test_texts[:10])

    # TensorRT benchmark
    if encoder.use_tensorrt:
        torch.cuda.synchronize()
        start = time.time()
        for _ in range(10):
            _ = encoder.encode(test_texts)
        torch.cuda.synchronize()
        trt_time = (time.time() - start) / 10

        print(f"TensorRT: {trt_time*1000:.2f}ms ({trt_time*1000/len(test_texts):.2f}ms per text)")

    # Fallback benchmark
    if encoder.fallback_model is not None:
        torch.cuda.synchronize()
        start = time.time()
        for _ in range(10):
            _ = encoder.fallback_model.encode(
                test_texts,
                convert_to_tensor=True,
                device=str(encoder.device)
            )
        torch.cuda.synchronize()
        fallback_time = (time.time() - start) / 10

        print(f"Standard: {fallback_time*1000:.2f}ms ({fallback_time*1000/len(test_texts):.2f}ms per text)")

        if encoder.use_tensorrt:
            speedup = fallback_time / trt_time
            print(f"\n🚀 Speedup: {speedup:.2f}x faster with TensorRT")


def main():
    """Run all tests"""
    import argparse

    parser = argparse.ArgumentParser(description='Test TensorRT Inference')
    parser.add_argument('--engine', type=str, help='Path to .plan engine file (optional)')
    args = parser.parse_args()

    print("=" * 80)
    print("TensorRT Inference Test Suite")
    print("=" * 80)

    # Test 1: Fallback mode
    test_fallback_mode()

    # Test 2-6: With TensorRT engine (if provided)
    if args.engine and Path(args.engine).exists():
        encoder = TensorRTEncoder(
            args.engine,
            "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        )

        test_with_tensorrt_engine(args.engine)
        test_batch_processing(encoder)
        test_normalization(encoder)
        test_memory_management(encoder)
        benchmark_performance(encoder)
    else:
        print("\n⚠️  No TensorRT engine provided. Skipping TensorRT-specific tests.")
        print("   Run with: --engine /path/to/model.plan")

    print("\n" + "=" * 80)
    print("✅ All tests completed")
    print("=" * 80)


if __name__ == "__main__":
    main()
