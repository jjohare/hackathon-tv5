#!/usr/bin/env python3
"""
Quick ONNX Export Test

A minimal test script to verify ONNX export functionality without full benchmarking.
Useful for CI/CD pipelines and quick validation.
"""

import sys
from pathlib import Path

try:
    import torch
    import onnx
    import onnxruntime as ort
    from sentence_transformers import SentenceTransformer
    import numpy as np
except ImportError as e:
    print(f"Error: Missing required package - {e}")
    print("Install with: pip install -r scripts/requirements-onnx.txt")
    sys.exit(1)


def quick_test():
    """Run a quick ONNX export test."""
    print("="* 60)
    print("Quick ONNX Export Test")
    print("="* 60)

    # Check CUDA availability
    cuda_available = torch.cuda.is_available()
    print(f"\nCUDA Available: {cuda_available}")
    if cuda_available:
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Check ONNX Runtime providers
    print(f"\nONNX Runtime Providers: {ort.get_available_providers()}")

    # Load model
    print("\nLoading model...")
    model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
    print("✓ Model loaded")

    # Test encoding
    test_sentence = "This is a test sentence for ONNX export validation"
    print(f"\nTest sentence: '{test_sentence}'")

    print("\nEncoding with PyTorch...")
    with torch.no_grad():
        pytorch_embedding = model.encode([test_sentence], convert_to_numpy=True)
    print(f"✓ PyTorch embedding shape: {pytorch_embedding.shape}")

    # Export to ONNX (minimal)
    print("\nExporting to ONNX...")
    output_dir = Path("models")
    output_dir.mkdir(exist_ok=True)
    onnx_path = output_dir / "sbert_test.onnx"

    transformer = model[0].auto_model
    tokenizer = model.tokenizer

    dummy_input = tokenizer(
        test_sentence,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=128
    )

    torch.onnx.export(
        transformer,
        (dummy_input['input_ids'], dummy_input['attention_mask']),
        str(onnx_path),
        input_names=['input_ids', 'attention_mask'],
        output_names=['output'],
        opset_version=14,
        do_constant_folding=True
    )
    print(f"✓ ONNX model exported to: {onnx_path}")

    # Verify ONNX model
    print("\nVerifying ONNX model...")
    onnx_model = onnx.load(str(onnx_path))
    onnx.checker.check_model(onnx_model)
    print("✓ ONNX model is valid")

    # Quick inference test
    print("\nRunning ONNX inference...")
    session = ort.InferenceSession(
        str(onnx_path),
        providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
    )

    onnx_inputs = {
        'input_ids': dummy_input['input_ids'].numpy().astype(np.int64),
        'attention_mask': dummy_input['attention_mask'].numpy().astype(np.int64)
    }

    outputs = session.run(None, onnx_inputs)
    print(f"✓ ONNX inference successful, output shape: {outputs[0].shape}")

    # Cleanup test file
    print("\nCleaning up test files...")
    onnx_path.unlink()
    print("✓ Test files removed")

    print("\n"+ "="* 60)
    print("Quick Test PASSED ✓")
    print("="* 60)
    print("\nNext step: Run full export with:")
    print("  python scripts/export_model_to_onnx.py")


if __name__ == '__main__':
    try:
        quick_test()
    except Exception as e:
        print(f"\n❌ Test FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
