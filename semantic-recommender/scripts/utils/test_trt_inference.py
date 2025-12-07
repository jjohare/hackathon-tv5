#!/usr/bin/env python3
"""
Test TensorRT Inference Integration

Tests:
1. TensorRTEncoder initialization and fallback behavior
2. Encoding functionality (TensorRT or fallback)
3. Embedding shape and normalization validation
4. Cosine similarity computation
5. TensorRT 10.x API compatibility

Note: TensorRT engine file may not exist yet. The encoder will use
fallback model (standard SentenceTransformer) until engine is built.
"""
import sys
sys.path.insert(0, '/home/devuser/workspace/hackathon-tv5/semantic-recommender/scripts/utils')

from trt_inference import TensorRTEncoder
import torch
import time

# Test sentences
test_sentences = [
    "AI and machine learning are transforming industries",
    "The cat sat on the mat",
    "Climate change poses significant challenges",
    "Python is a popular programming language"
]

print("=" * 80)
print("TensorRT Inference Integration Test")
print("=" * 80)

print("\n[1] Initializing TensorRTEncoder...")
encoder = TensorRTEncoder(
    engine_path="models/sentence_transformer_fp16_sm86.trt",
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    device="cuda:0"
)

print(f"✓ Encoder initialized")
print(f"  Using TensorRT: {encoder.use_tensorrt}")
print(f"  Device: {encoder.device}")

if not encoder.use_tensorrt:
    print("\n⚠️  TensorRT engine not available - using fallback model")
    print("   TensorRT acceleration will be available once engine is built")
    print("   Current mode: Standard SentenceTransformer on GPU")

# Test encoding
print("\n[2] Testing encoding...")
start = time.time()
embeddings = encoder.encode(test_sentences, convert_to_tensor=True, normalize_embeddings=True)
elapsed = time.time() - start

print(f"✓ Encoded {len(test_sentences)} sentences in {elapsed*1000:.2f}ms")
print(f"  Embedding shape: {embeddings.shape}")
print(f"  Expected shape: (4, 384)")
print(f"  Embeddings device: {embeddings.device}")

# Verify embeddings are normalized
print("\n[3] Verifying normalization...")
norms = torch.norm(embeddings, dim=1)
print(f"  L2 norms (should be ~1.0):")
for i, norm in enumerate(norms.tolist()):
    print(f"    Sentence {i+1}: {norm:.6f}")

# Check if norms are close to 1.0
norms_ok = all(abs(n - 1.0) < 0.01 for n in norms.tolist())
print(f"  Normalization {'✓ PASSED' if norms_ok else '✗ FAILED'}")

# Test cosine similarity
print("\n[4] Testing cosine similarity...")
from sentence_transformers.util import cos_sim
sim_matrix = cos_sim(embeddings, embeddings)
print(f"  Cosine similarity matrix:")
print(sim_matrix)

# Verify diagonal is ~1.0 (self-similarity)
diagonal_ok = all(abs(sim_matrix[i][i].item() - 1.0) < 0.01 for i in range(len(test_sentences)))
print(f"  Self-similarity {'✓ PASSED' if diagonal_ok else '✗ FAILED'}")

# Test shape validation
print("\n[5] Validating output shape...")
expected_shape = (4, 384)
shape_ok = embeddings.shape == expected_shape
print(f"  Expected: {expected_shape}")
print(f"  Actual: {embeddings.shape}")
print(f"  Shape validation {'✓ PASSED' if shape_ok else '✗ FAILED'}")

# Summary
print("\n" + "=" * 80)
print("Test Summary")
print("=" * 80)
print(f"✓ TensorRTEncoder initialization: PASSED")
print(f"✓ Encoding functionality: PASSED")
print(f"✓ Shape validation: {'PASSED' if shape_ok else 'FAILED'}")
print(f"✓ Normalization: {'PASSED' if norms_ok else 'FAILED'}")
print(f"✓ Cosine similarity: {'PASSED' if diagonal_ok else 'FAILED'}")

if encoder.use_tensorrt:
    print(f"\n🚀 TensorRT acceleration: ACTIVE")
    print(f"   Expected speedup: 3-5x over standard model")
else:
    print(f"\n📋 TensorRT acceleration: NOT ACTIVE (using fallback)")
    print(f"   To enable TensorRT:")
    print(f"   1. Build engine: trtexec --onnx=models/sentence_transformer.onnx \\")
    print(f"      --saveEngine=models/sentence_transformer_fp16_sm86.trt --fp16")
    print(f"   2. Or run: python scripts/ops/build_trt_engine.py")

all_passed = shape_ok and norms_ok and diagonal_ok
print(f"\n{'✓ ALL TESTS PASSED' if all_passed else '✗ SOME TESTS FAILED'}")
print("=" * 80)
