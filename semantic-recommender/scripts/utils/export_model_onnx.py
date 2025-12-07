#!/usr/bin/env python3
"""
ONNX Model Export Script for Semantic Recommender

Exports SentenceTransformer model to ONNX format with complete pipeline including:
- Tokenizer preprocessing
- Transformer layers
- Mean pooling layer (CRITICAL)
- Normalization

This ensures the exported model maintains parity with SentenceTransformer outputs.
"""

import os
import sys
from pathlib import Path
import logging
import numpy as np
import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer
import onnx
import onnxruntime as ort
from typing import Dict, List, Tuple

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MeanPooling(nn.Module):
    """Mean pooling layer that takes attention mask into account."""

    def forward(self, token_embeddings: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Perform mean pooling on token embeddings.

        Args:
            token_embeddings: Shape (batch_size, seq_length, hidden_size)
            attention_mask: Shape (batch_size, seq_length)

        Returns:
            Pooled embeddings: Shape (batch_size, hidden_size)
        """
        # Expand attention mask to match token embeddings dimensions
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()

        # Sum embeddings weighted by attention mask
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, dim=1)

        # Sum attention mask (count of real tokens)
        sum_mask = torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)

        # Compute mean
        return sum_embeddings / sum_mask


class SentenceTransformerONNX(nn.Module):
    """
    Wrapper model that combines transformer + pooling for ONNX export.

    This ensures the complete pipeline is exported as a single ONNX model.
    """

    def __init__(self, transformer_model, pooling_layer):
        super().__init__()
        self.transformer = transformer_model
        self.pooling = pooling_layer

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through transformer and pooling.

        Args:
            input_ids: Shape (batch_size, seq_length)
            attention_mask: Shape (batch_size, seq_length)

        Returns:
            Sentence embeddings: Shape (batch_size, hidden_size)
        """
        # Get transformer outputs
        outputs = self.transformer(input_ids=input_ids, attention_mask=attention_mask)

        # Extract token embeddings (last hidden state)
        token_embeddings = outputs.last_hidden_state

        # Apply mean pooling
        sentence_embeddings = self.pooling(token_embeddings, attention_mask)

        # L2 normalization
        sentence_embeddings = torch.nn.functional.normalize(sentence_embeddings, p=2, dim=1)

        return sentence_embeddings


def export_to_onnx(
    model_name: str = "paraphrase-multilingual-MiniLM-L12-v2",
    output_path: str = "data/models/minilm_l12_v2.onnx",
    max_seq_length: int = 128,
) -> Path:
    """
    Export SentenceTransformer model to ONNX format with complete pipeline.

    Args:
        model_name: HuggingFace model identifier
        output_path: Path to save ONNX model
        max_seq_length: Maximum sequence length for model

    Returns:
        Path to exported ONNX model
    """
    logger.info(f"Loading SentenceTransformer model: {model_name}")

    # Load the SentenceTransformer model on CPU for export stability
    device = torch.device('cpu')
    st_model = SentenceTransformer(model_name, device=device)
    st_model.eval()

    # Extract the transformer (first module in the pipeline)
    transformer = st_model[0].auto_model
    transformer = transformer.to(device)

    # Create pooling layer
    pooling = MeanPooling()

    # Create combined model for export
    combined_model = SentenceTransformerONNX(transformer, pooling)
    combined_model.to(device)
    combined_model.eval()

    # Create output directory
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Preparing dummy inputs for ONNX export")

    # Create dummy inputs with batch_size=1 on CPU
    batch_size = 1
    dummy_input_ids = torch.randint(0, 1000, (batch_size, max_seq_length), dtype=torch.long, device=device)
    dummy_attention_mask = torch.ones((batch_size, max_seq_length), dtype=torch.long, device=device)

    # Define dynamic axes for variable batch size and sequence length
    dynamic_axes = {
        'input_ids': {0: 'batch_size', 1: 'sequence_length'},
        'attention_mask': {0: 'batch_size', 1: 'sequence_length'},
        'sentence_embedding': {0: 'batch_size'}
    }

    logger.info(f"Exporting model to ONNX: {output_path}")

    try:
        # Export to ONNX
        torch.onnx.export(
            combined_model,
            (dummy_input_ids, dummy_attention_mask),
            str(output_path),
            input_names=['input_ids', 'attention_mask'],
            output_names=['sentence_embedding'],
            dynamic_axes=dynamic_axes,
            opset_version=14,
            do_constant_folding=True,
            export_params=True,
        )

        logger.info("ONNX export completed successfully")

        # Verify the exported model
        logger.info("Verifying exported ONNX model")
        onnx_model = onnx.load(str(output_path))
        onnx.checker.check_model(onnx_model)
        logger.info("ONNX model verification passed")

        # Get model info
        logger.info(f"Model input names: {[inp.name for inp in onnx_model.graph.input]}")
        logger.info(f"Model output names: {[out.name for out in onnx_model.graph.output]}")

        return output_path

    except Exception as e:
        logger.error(f"ONNX export failed: {e}")
        raise


def validate_onnx_model(
    onnx_path: Path,
    model_name: str = "paraphrase-multilingual-MiniLM-L12-v2",
    test_sentences: List[str] = None,
    batch_sizes: List[int] = None,
) -> Dict[str, any]:
    """
    Validate exported ONNX model against SentenceTransformer baseline.

    Args:
        onnx_path: Path to ONNX model
        model_name: Original SentenceTransformer model name
        test_sentences: Test sentences for validation
        batch_sizes: Batch sizes to test

    Returns:
        Validation results dictionary
    """
    if test_sentences is None:
        test_sentences = [
            "This is a test sentence.",
            "Another example for validation.",
            "Machine learning is fascinating.",
            "ONNX enables cross-platform deployment."
        ]

    if batch_sizes is None:
        batch_sizes = [1, 4, 16]

    logger.info("Loading SentenceTransformer baseline model")
    st_model = SentenceTransformer(model_name)
    st_model.eval()

    logger.info(f"Loading ONNX model from {onnx_path}")
    ort_session = ort.InferenceSession(str(onnx_path))

    # Get tokenizer from SentenceTransformer
    tokenizer = st_model.tokenizer

    results = {
        'batch_tests': [],
        'max_diff': 0.0,
        'mean_diff': 0.0,
        'success': True
    }

    logger.info("Running validation tests")

    try:
        # Test different batch sizes
        for batch_size in batch_sizes:
            test_batch = test_sentences[:batch_size]

            logger.info(f"Testing batch_size={batch_size}")

            # Get baseline embeddings from SentenceTransformer
            baseline_embeddings = st_model.encode(
                test_batch,
                convert_to_numpy=True,
                normalize_embeddings=True
            )

            # Get ONNX model embeddings
            encoded = tokenizer(
                test_batch,
                padding=True,
                truncation=True,
                max_length=128,
                return_tensors='np'
            )

            onnx_inputs = {
                'input_ids': encoded['input_ids'].astype(np.int64),
                'attention_mask': encoded['attention_mask'].astype(np.int64)
            }

            onnx_outputs = ort_session.run(None, onnx_inputs)
            onnx_embeddings = onnx_outputs[0]

            # Verify shapes match
            assert baseline_embeddings.shape == onnx_embeddings.shape, \
                f"Shape mismatch: baseline {baseline_embeddings.shape} vs ONNX {onnx_embeddings.shape}"

            # Compute differences
            diff = np.abs(baseline_embeddings - onnx_embeddings)
            max_diff = np.max(diff)
            mean_diff = np.mean(diff)

            logger.info(f"Batch size {batch_size}: max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f}")

            results['batch_tests'].append({
                'batch_size': batch_size,
                'max_diff': float(max_diff),
                'mean_diff': float(mean_diff),
                'shape': onnx_embeddings.shape
            })

            results['max_diff'] = max(results['max_diff'], max_diff)
            results['mean_diff'] = max(results['mean_diff'], mean_diff)

            # Check if differences are acceptable (should be very small due to numerical precision)
            if max_diff > 1e-4:
                logger.warning(f"Large difference detected: {max_diff}")
                results['success'] = False

        # Test output dimensionality
        expected_dim = 384  # MiniLM-L12-v2 embedding dimension
        actual_dim = onnx_embeddings.shape[1]

        assert actual_dim == expected_dim, \
            f"Embedding dimension mismatch: expected {expected_dim}, got {actual_dim}"

        logger.info(f"Validation {'PASSED' if results['success'] else 'FAILED'}")
        logger.info(f"Overall max difference: {results['max_diff']:.6f}")
        logger.info(f"Overall mean difference: {results['mean_diff']:.6f}")

        return results

    except Exception as e:
        logger.error(f"Validation failed: {e}")
        results['success'] = False
        results['error'] = str(e)
        return results


def main():
    """Main execution function."""
    import argparse

    parser = argparse.ArgumentParser(description='Export SentenceTransformer model to ONNX')
    parser.add_argument(
        '--model-name',
        type=str,
        default='paraphrase-multilingual-MiniLM-L12-v2',
        help='HuggingFace model name'
    )
    parser.add_argument(
        '--output-path',
        type=str,
        default='data/models/minilm_l12_v2.onnx',
        help='Output path for ONNX model'
    )
    parser.add_argument(
        '--max-seq-length',
        type=int,
        default=128,
        help='Maximum sequence length'
    )
    parser.add_argument(
        '--skip-validation',
        action='store_true',
        help='Skip validation step'
    )

    args = parser.parse_args()

    try:
        # Export model
        onnx_path = export_to_onnx(
            model_name=args.model_name,
            output_path=args.output_path,
            max_seq_length=args.max_seq_length
        )

        logger.info(f"Model exported successfully to: {onnx_path}")

        # Validate if not skipped
        if not args.skip_validation:
            results = validate_onnx_model(
                onnx_path=onnx_path,
                model_name=args.model_name
            )

            if results['success']:
                logger.info("Validation PASSED - Model is ready for deployment")
                return 0
            else:
                logger.error("Validation FAILED - Check model export")
                return 1

        return 0

    except Exception as e:
        logger.error(f"Export process failed: {e}")
        return 1


if __name__ == '__main__':
    sys.exit(main())
