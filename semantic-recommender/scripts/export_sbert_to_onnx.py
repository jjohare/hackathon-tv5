#!/usr/bin/env python3
"""
Export SBERT Model to ONNX for TensorRT Optimization
=====================================================

Exports the sentence-transformer model to ONNX format for TensorRT optimization.

Expected Performance:
- Current: 11ms query encoding (88% of total latency)
- TensorRT: 0.5ms query encoding (22× faster)
- Overall: 11.42ms → <1ms total latency

Author: Claude Sonnet 4.5
Date: December 7, 2025
"""

import torch
import torch.onnx
from sentence_transformers import SentenceTransformer
import numpy as np
from pathlib import Path
import json


def export_sbert_to_onnx(
    model_name: str = 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2',
    output_dir: str = 'models/onnx',
    opset_version: int = 14
):
    """
    Export SBERT model to ONNX format.

    Args:
        model_name: HuggingFace model name
        output_dir: Output directory for ONNX model
        opset_version: ONNX opset version (14+ for TensorRT)
    """
    print("=" * 80)
    print("SBERT to ONNX Export for TensorRT Optimization")
    print("=" * 80)
    print()

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load model
    print(f"[1/5] Loading model: {model_name}")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SentenceTransformer(model_name, device=str(device))
    model.eval()

    # Get the underlying transformer model
    transformer = model[0].auto_model
    tokenizer = model[0].tokenizer

    print(f"  Model loaded on: {device}")
    print(f"  Max sequence length: {model.max_seq_length}")
    print()

    # Create dummy input
    print("[2/5] Creating dummy input for export")
    dummy_text = "This is a sample text for ONNX export"
    encoded = tokenizer(
        dummy_text,
        padding='max_length',
        truncation=True,
        max_length=model.max_seq_length,
        return_tensors='pt'
    )

    input_ids = encoded['input_ids'].to(device)
    attention_mask = encoded['attention_mask'].to(device)

    print(f"  Input shape: {input_ids.shape}")
    print(f"  Sequence length: {input_ids.shape[1]}")
    print()

    # Export to ONNX
    print("[3/5] Exporting transformer to ONNX")
    onnx_model_path = output_path / "sbert_transformer.onnx"

    torch.onnx.export(
        transformer,
        (input_ids, attention_mask),
        str(onnx_model_path),
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=['input_ids', 'attention_mask'],
        output_names=['last_hidden_state', 'pooler_output'],
        dynamic_axes={
            'input_ids': {0: 'batch_size', 1: 'sequence_length'},
            'attention_mask': {0: 'batch_size', 1: 'sequence_length'},
            'last_hidden_state': {0: 'batch_size', 1: 'sequence_length'},
            'pooler_output': {0: 'batch_size'}
        }
    )

    print(f"  ✅ Exported to: {onnx_model_path}")
    print()

    # Verify ONNX model
    print("[4/5] Verifying ONNX model")
    import onnx
    onnx_model = onnx.load(str(onnx_model_path))
    onnx.checker.check_model(onnx_model)
    print("  ✅ ONNX model is valid")
    print()

    # Save model config
    print("[5/5] Saving model configuration")
    config = {
        'model_name': model_name,
        'max_seq_length': model.max_seq_length,
        'embedding_dim': model.get_sentence_embedding_dimension(),
        'opset_version': opset_version,
        'input_names': ['input_ids', 'attention_mask'],
        'output_names': ['last_hidden_state', 'pooler_output'],
        'pooling_mode': 'mean',  # SBERT uses mean pooling
        'normalize': True
    }

    config_path = output_path / "model_config.json"
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)

    print(f"  ✅ Config saved to: {config_path}")
    print()

    # Test ONNX inference
    print("[BONUS] Testing ONNX inference with ONNX Runtime")
    try:
        import onnxruntime as ort

        # Create inference session
        session = ort.InferenceSession(
            str(onnx_model_path),
            providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
        )

        print(f"  Provider: {session.get_providers()[0]}")

        # Run inference
        outputs = session.run(
            None,
            {
                'input_ids': input_ids.cpu().numpy(),
                'attention_mask': attention_mask.cpu().numpy()
            }
        )

        print(f"  Output shape: {outputs[0].shape}")
        print(f"  ✅ ONNX Runtime inference successful")

    except ImportError:
        print("  ⚠️  ONNX Runtime not installed (optional)")

    print()
    print("=" * 80)
    print("✅ ONNX Export Complete")
    print("=" * 80)
    print()
    print("Next Steps:")
    print("1. Optimize ONNX model with TensorRT:")
    print(f"   trtexec --onnx={onnx_model_path} \\")
    print(f"           --saveEngine=models/tensorrt/sbert.trt \\")
    print(f"           --fp16 \\")
    print(f"           --workspace=4096")
    print()
    print("2. Benchmark TensorRT engine on A100")
    print("3. Integrate into hyper-personalization pipeline")
    print()
    print(f"Expected speedup: 22× faster (11ms → 0.5ms)")
    print(f"Expected total latency: 11.42ms → <1ms")

    return str(onnx_model_path), config


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Export SBERT to ONNX for TensorRT")
    parser.add_argument(
        '--model',
        default='sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2',
        help='HuggingFace model name'
    )
    parser.add_argument(
        '--output',
        default='models/onnx',
        help='Output directory'
    )
    parser.add_argument(
        '--opset',
        type=int,
        default=14,
        help='ONNX opset version'
    )

    args = parser.parse_args()

    export_sbert_to_onnx(
        model_name=args.model,
        output_dir=args.output,
        opset_version=args.opset
    )
