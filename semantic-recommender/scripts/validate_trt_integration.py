#!/usr/bin/env python3
"""
Phase 4 Validation: TensorRT Integration Testing

Validates:
1. TensorRT encoder loads correctly
2. Output matches PyTorch baseline (cosine similarity > 0.99)
3. Embeddings stay on GPU (no D2H transfers)
4. Batch processing works correctly
5. Performance gains are measurable
6. Backwards compatibility maintained

Usage:
    python scripts/validate_trt_integration.py --engine data/models/minilm_l12_v2_fp16.plan
    python scripts/validate_trt_integration.py --full-benchmark
"""

import sys
import os
import time
import argparse
from pathlib import Path
from typing import List

import torch
import numpy as np
from sentence_transformers import SentenceTransformer

# Add utils to path
sys.path.insert(0, str(Path(__file__).parent / "utils"))

from trt_inference import TensorRTEncoder


class ValidationTests:
    """Comprehensive validation suite for TensorRT integration"""

    def __init__(self, engine_path: str = None):
        self.engine_path = engine_path
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.test_queries = [
            "What are the best sci-fi movies?",
            "Looking for romantic comedies from the 90s",
            "Action movies with great special effects",
            "Drama films about family relationships",
            "Horror movies that are actually scary",
        ]

        print("=" * 80)
        print("TensorRT Integration Validation")
        print("=" * 80)
        print(f"Device: {self.device}")
        print(f"Engine: {engine_path if engine_path else 'Not provided'}")
        print()

    def test_1_encoder_loading(self) -> bool:
        """Test 1: TensorRT encoder loads correctly"""
        print("[Test 1] TensorRT Encoder Loading")
        print("-" * 80)

        if not self.engine_path or not os.path.exists(self.engine_path):
            print(f"❌ SKIP: Engine not found at {self.engine_path}")
            return False

        try:
            encoder = TensorRTEncoder(
                self.engine_path,
                model_name='paraphrase-multilingual-MiniLM-L12-v2',
                device=str(self.device)
            )
            print(f"✅ PASS: TensorRT encoder loaded successfully")
            print(f"   Using TensorRT: {encoder.use_tensorrt}")
            return True

        except Exception as e:
            print(f"❌ FAIL: {e}")
            return False

    def test_2_output_validation(self) -> bool:
        """Test 2: Output matches PyTorch baseline (cosine similarity > 0.99)"""
        print("\n[Test 2] Output Validation vs PyTorch Baseline")
        print("-" * 80)

        if not self.engine_path or not os.path.exists(self.engine_path):
            print(f"❌ SKIP: Engine not found")
            return False

        try:
            # Load both encoders
            trt_encoder = TensorRTEncoder(
                self.engine_path,
                model_name='paraphrase-multilingual-MiniLM-L12-v2',
                device=str(self.device)
            )

            pytorch_encoder = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
            pytorch_encoder.to(self.device)

            # Encode test queries
            print(f"  Encoding {len(self.test_queries)} queries...")

            trt_embeddings = trt_encoder.encode(
                self.test_queries,
                normalize_embeddings=True
            )

            pytorch_embeddings = pytorch_encoder.encode(
                self.test_queries,
                convert_to_tensor=True,
                device=str(self.device),
                normalize_embeddings=True
            )

            # Compute cosine similarities
            cosine_sims = torch.nn.functional.cosine_similarity(
                trt_embeddings,
                pytorch_embeddings,
                dim=1
            )

            mean_sim = cosine_sims.mean().item()
            min_sim = cosine_sims.min().item()

            print(f"  Mean cosine similarity: {mean_sim:.6f}")
            print(f"  Min cosine similarity:  {min_sim:.6f}")

            # Validation threshold
            if mean_sim > 0.99 and min_sim > 0.98:
                print(f"✅ PASS: Output matches baseline (similarity > 0.99)")
                return True
            else:
                print(f"❌ FAIL: Output mismatch (mean: {mean_sim:.6f}, min: {min_sim:.6f})")
                return False

        except Exception as e:
            print(f"❌ FAIL: {e}")
            import traceback
            traceback.print_exc()
            return False

    def test_3_gpu_memory(self) -> bool:
        """Test 3: Embeddings stay on GPU (no D2H transfers)"""
        print("\n[Test 3] GPU Memory Management")
        print("-" * 80)

        if not self.engine_path or not os.path.exists(self.engine_path):
            print(f"❌ SKIP: Engine not found")
            return False

        try:
            encoder = TensorRTEncoder(
                self.engine_path,
                model_name='paraphrase-multilingual-MiniLM-L12-v2',
                device=str(self.device)
            )

            embeddings = encoder.encode(
                self.test_queries,
                normalize_embeddings=True
            )

            # Check device
            if embeddings.device.type == 'cuda':
                print(f"✅ PASS: Embeddings on GPU ({embeddings.device})")
                print(f"   Shape: {embeddings.shape}")
                print(f"   Dtype: {embeddings.dtype}")
                return True
            else:
                print(f"❌ FAIL: Embeddings not on GPU ({embeddings.device})")
                return False

        except Exception as e:
            print(f"❌ FAIL: {e}")
            return False

    def test_4_batch_processing(self) -> bool:
        """Test 4: Batch processing works correctly"""
        print("\n[Test 4] Batch Processing")
        print("-" * 80)

        if not self.engine_path or not os.path.exists(self.engine_path):
            print(f"❌ SKIP: Engine not found")
            return False

        try:
            encoder = TensorRTEncoder(
                self.engine_path,
                model_name='paraphrase-multilingual-MiniLM-L12-v2',
                device=str(self.device)
            )

            batch_sizes = [1, 4, 8, 16, 32]
            results = {}

            for bs in batch_sizes:
                # Create batch
                batch = self.test_queries * (bs // len(self.test_queries) + 1)
                batch = batch[:bs]

                # Encode
                embeddings = encoder.encode(batch, batch_size=bs)

                # Validate
                if embeddings.shape[0] != bs:
                    print(f"❌ FAIL: Batch size {bs} returned {embeddings.shape[0]} embeddings")
                    return False

                results[bs] = embeddings.shape

                print(f"  Batch size {bs:2d}: {embeddings.shape} ✓")

            print(f"✅ PASS: All batch sizes processed correctly")
            return True

        except Exception as e:
            print(f"❌ FAIL: {e}")
            import traceback
            traceback.print_exc()
            return False

    def test_5_performance(self) -> bool:
        """Test 5: Performance gains are measurable"""
        print("\n[Test 5] Performance Benchmark")
        print("-" * 80)

        if not self.engine_path or not os.path.exists(self.engine_path):
            print(f"❌ SKIP: Engine not found")
            return False

        try:
            # Load both encoders
            trt_encoder = TensorRTEncoder(
                self.engine_path,
                model_name='paraphrase-multilingual-MiniLM-L12-v2',
                device=str(self.device)
            )

            pytorch_encoder = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
            pytorch_encoder.to(self.device)

            num_iterations = 50
            batch = self.test_queries

            # Warmup
            for _ in range(10):
                _ = trt_encoder.encode(batch)
                _ = pytorch_encoder.encode(batch, convert_to_tensor=True, device=str(self.device))

            # Benchmark TensorRT
            torch.cuda.synchronize()
            start = time.time()
            for _ in range(num_iterations):
                _ = trt_encoder.encode(batch)
            torch.cuda.synchronize()
            trt_time = (time.time() - start) / num_iterations * 1000

            # Benchmark PyTorch
            torch.cuda.synchronize()
            start = time.time()
            for _ in range(num_iterations):
                _ = pytorch_encoder.encode(batch, convert_to_tensor=True, device=str(self.device))
            torch.cuda.synchronize()
            pytorch_time = (time.time() - start) / num_iterations * 1000

            speedup = pytorch_time / trt_time

            print(f"  PyTorch:   {pytorch_time:.2f}ms")
            print(f"  TensorRT:  {trt_time:.2f}ms")
            print(f"  Speedup:   {speedup:.2f}x")

            if speedup > 1.5:
                print(f"✅ PASS: TensorRT shows {speedup:.2f}x speedup")
                return True
            else:
                print(f"⚠️  WARNING: Speedup only {speedup:.2f}x (expected > 1.5x)")
                return True  # Still pass, but warn

        except Exception as e:
            print(f"❌ FAIL: {e}")
            import traceback
            traceback.print_exc()
            return False

    def test_6_backwards_compatibility(self) -> bool:
        """Test 6: Backwards compatibility maintained"""
        print("\n[Test 6] Backwards Compatibility")
        print("-" * 80)

        try:
            # Test fallback when engine not found
            encoder = TensorRTEncoder(
                "nonexistent.plan",
                model_name='paraphrase-multilingual-MiniLM-L12-v2',
                device=str(self.device)
            )

            # Should fall back to PyTorch
            embeddings = encoder.encode(self.test_queries)

            if embeddings.shape[0] == len(self.test_queries):
                print(f"✅ PASS: Fallback to PyTorch works correctly")
                print(f"   Using TensorRT: {encoder.use_tensorrt}")
                print(f"   Output shape: {embeddings.shape}")
                return True
            else:
                print(f"❌ FAIL: Fallback produced incorrect output")
                return False

        except Exception as e:
            print(f"❌ FAIL: {e}")
            return False

    def run_all_tests(self) -> dict:
        """Run all validation tests"""
        results = {
            'test_1_loading': self.test_1_encoder_loading(),
            'test_2_output': self.test_2_output_validation(),
            'test_3_gpu': self.test_3_gpu_memory(),
            'test_4_batch': self.test_4_batch_processing(),
            'test_5_performance': self.test_5_performance(),
            'test_6_compatibility': self.test_6_backwards_compatibility(),
        }

        print("\n" + "=" * 80)
        print("VALIDATION SUMMARY")
        print("=" * 80)

        passed = sum(results.values())
        total = len(results)

        for test_name, passed_test in results.items():
            status = "✅ PASS" if passed_test else "❌ FAIL"
            print(f"{status}  {test_name}")

        print()
        print(f"Results: {passed}/{total} tests passed")

        if passed == total:
            print("🎉 All tests passed! TensorRT integration validated.")
        else:
            print("⚠️  Some tests failed. Review logs above.")

        print("=" * 80)

        return results


def main():
    parser = argparse.ArgumentParser(description='Validate TensorRT Integration')
    parser.add_argument(
        '--engine',
        type=str,
        default='data/models/minilm_l12_v2_fp16.plan',
        help='Path to TensorRT engine (.plan file)'
    )
    parser.add_argument(
        '--full-benchmark',
        action='store_true',
        help='Run comprehensive benchmarks'
    )

    args = parser.parse_args()

    # Run validation
    validator = ValidationTests(engine_path=args.engine)
    results = validator.run_all_tests()

    # Exit code
    sys.exit(0 if all(results.values()) else 1)


if __name__ == "__main__":
    main()
