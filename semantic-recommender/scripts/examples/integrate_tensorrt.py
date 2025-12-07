#!/usr/bin/env python3
"""
Example: Integrating TensorRT into GPU Hyper-Personalization

Shows how to replace SentenceTransformer with TensorRTEncoder
for 3-5x performance improvement.
"""

import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Original imports
from sentence_transformers import SentenceTransformer

# TensorRT import
from utils.trt_inference import TensorRTEncoder


class GPUHyperPersonalizationTensorRT:
    """
    GPU Hyper-Personalization with TensorRT acceleration

    Simple modification to existing system - just replace the encoder!
    """

    def __init__(
        self,
        base_path: str = ".",
        use_tensorrt: bool = True,
        engine_path: str = None
    ):
        self.base_path = Path(base_path)
        self.use_tensorrt = use_tensorrt

        print("=" * 80)
        print("GPU Hyper-Personalization with TensorRT")
        print("=" * 80)

        # Load semantic model
        print("\n[Semantic Encoder]")

        if use_tensorrt and engine_path:
            # TensorRT encoder
            self.model = TensorRTEncoder(
                engine_path=engine_path,
                model_name='sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2',
                max_seq_length=128,
                device='cuda'
            )
            print(f"Using TensorRT: {self.model.use_tensorrt}")
        else:
            # Standard encoder (fallback)
            self.model = SentenceTransformer(
                'paraphrase-multilingual-MiniLM-L12-v2'
            )
            print("Using standard SentenceTransformer")

        # Rest of initialization (embeddings, cache, attention) unchanged
        # ...

        print("\n" + "=" * 80)
        print("✅ System ready!")
        print("=" * 80 + "\n")

    def encode_query(self, query: str):
        """
        Encode query - works identically with TensorRT or standard model

        Args:
            query: Natural language query

        Returns:
            Query embedding (torch.Tensor on GPU)
        """
        # This API call is IDENTICAL for both TensorRT and standard
        embedding = self.model.encode(
            query,
            convert_to_tensor=True,
            device='cuda'
        )

        return embedding


def example_simple_replacement():
    """
    Example 1: Simple drop-in replacement
    """
    print("\n" + "=" * 80)
    print("EXAMPLE 1: Simple Drop-in Replacement")
    print("=" * 80 + "\n")

    # BEFORE: Standard model
    print("[Before] Using SentenceTransformer:")
    model_standard = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
    emb1 = model_standard.encode("Test query", convert_to_tensor=True)
    print(f"  Shape: {emb1.shape}")
    print(f"  Device: {emb1.device}")

    # AFTER: TensorRT model (with fallback)
    print("\n[After] Using TensorRTEncoder:")
    model_tensorrt = TensorRTEncoder(
        engine_path="models/encoder.plan",  # Will fallback if not found
        model_name='sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'
    )
    emb2 = model_tensorrt.encode("Test query", convert_to_tensor=True)
    print(f"  Shape: {emb2.shape}")
    print(f"  Device: {emb2.device}")
    print(f"  Using TensorRT: {model_tensorrt.use_tensorrt}")


def example_conditional_usage():
    """
    Example 2: Conditional TensorRT usage
    """
    print("\n" + "=" * 80)
    print("EXAMPLE 2: Conditional TensorRT Usage")
    print("=" * 80 + "\n")

    import os

    # Use TensorRT if engine exists, else fallback
    engine_path = "models/encoder.plan"

    if os.path.exists(engine_path):
        print(f"✅ Engine found: {engine_path}")
        print("Using TensorRT acceleration")
        model = TensorRTEncoder(engine_path, "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    else:
        print(f"⚠️  Engine not found: {engine_path}")
        print("Using standard model")
        model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

    # Rest of code is identical
    embeddings = model.encode(["Query 1", "Query 2"], convert_to_tensor=True)
    print(f"Embeddings shape: {embeddings.shape}")


def example_performance_comparison():
    """
    Example 3: Performance comparison
    """
    print("\n" + "=" * 80)
    print("EXAMPLE 3: Performance Comparison")
    print("=" * 80 + "\n")

    import time
    import torch

    # Test data
    queries = [f"Query about movie genre {i}" for i in range(100)]

    # Standard model
    print("Benchmarking SentenceTransformer...")
    model_standard = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
    model_standard.to('cuda')

    torch.cuda.synchronize()
    start = time.time()
    for _ in range(10):
        _ = model_standard.encode(queries, convert_to_tensor=True, device='cuda')
    torch.cuda.synchronize()
    time_standard = (time.time() - start) / 10

    print(f"Standard model: {time_standard*1000:.2f}ms")

    # TensorRT model (will use fallback if engine missing)
    print("\nBenchmarking TensorRTEncoder...")
    model_tensorrt = TensorRTEncoder(
        "models/encoder.plan",
        'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'
    )

    torch.cuda.synchronize()
    start = time.time()
    for _ in range(10):
        _ = model_tensorrt.encode(queries, convert_to_tensor=True)
    torch.cuda.synchronize()
    time_tensorrt = (time.time() - start) / 10

    print(f"TensorRT model: {time_tensorrt*1000:.2f}ms")

    if model_tensorrt.use_tensorrt:
        speedup = time_standard / time_tensorrt
        print(f"\n🚀 Speedup: {speedup:.2f}x faster")
    else:
        print("\n⚠️  TensorRT not available - using fallback")


def example_integration_pattern():
    """
    Example 4: Recommended integration pattern
    """
    print("\n" + "=" * 80)
    print("EXAMPLE 4: Recommended Integration Pattern")
    print("=" * 80 + "\n")

    def create_encoder(config: dict):
        """
        Factory function for creating encoder with TensorRT support

        Args:
            config: Configuration dict with 'tensorrt_engine' and 'model_name'

        Returns:
            Encoder instance (TensorRT or fallback)
        """
        engine_path = config.get('tensorrt_engine')
        model_name = config.get('model_name', 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')

        if engine_path:
            # Try TensorRT first
            encoder = TensorRTEncoder(engine_path, model_name)
            if encoder.use_tensorrt:
                print(f"✅ Using TensorRT engine: {engine_path}")
                return encoder
            else:
                print(f"⚠️  TensorRT failed, using fallback")
                return encoder  # Will use fallback internally
        else:
            # Standard model
            print("Using standard SentenceTransformer")
            return SentenceTransformer(model_name)

    # Usage
    config = {
        'tensorrt_engine': 'models/encoder.plan',  # Optional
        'model_name': 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'
    }

    encoder = create_encoder(config)
    embeddings = encoder.encode(["Test query"], convert_to_tensor=True)
    print(f"Encoder ready: {type(encoder).__name__}")


def main():
    """Run all examples"""
    print("=" * 80)
    print("TensorRT Integration Examples")
    print("=" * 80)

    # Example 1: Simple replacement
    example_simple_replacement()

    # Example 2: Conditional usage
    example_conditional_usage()

    # Example 3: Performance comparison
    example_performance_comparison()

    # Example 4: Integration pattern
    example_integration_pattern()

    print("\n" + "=" * 80)
    print("✅ Examples completed")
    print("=" * 80)


if __name__ == "__main__":
    main()
