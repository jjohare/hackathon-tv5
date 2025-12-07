#!/usr/bin/env python3
"""
Export Sentence Transformer to ONNX for TensorRT Optimization

This script exports the paraphrase-multilingual-MiniLM-L12-v2 model to ONNX format
with optimizations for TensorRT inference, validates accuracy, and benchmarks performance.

Requirements:
    - sentence-transformers>=2.2.0
    - torch>=2.0.0
    - onnx>=1.14.0
    - onnxruntime>=1.15.0
    - onnxoptimizer (optional)
"""

import argparse
import logging
import os
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import onnx
import onnxruntime as ort
import torch
from sentence_transformers import SentenceTransformer
from scipy.spatial.distance import cosine

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ONNXModelExporter:
    """Export and optimize Sentence Transformer models to ONNX format."""

    def __init__(
        self,
        model_name: str = 'paraphrase-multilingual-MiniLM-L12-v2',
        output_dir: str = 'models',
        use_fp16: bool = True
    ):
        """
        Initialize the ONNX exporter.

        Args:
            model_name: HuggingFace model identifier
            output_dir: Directory to save ONNX model
            use_fp16: Whether to use FP16 optimization
        """
        self.model_name = model_name
        self.output_dir = Path(output_dir)
        self.use_fp16 = use_fp16
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Model paths
        self.onnx_path = self.output_dir / 'sbert_optimized.onnx'
        self.onnx_quantized_path = self.output_dir / 'sbert_quantized.onnx'

        # Test samples for validation
        self.test_sentences = [
            "The cat sits on the mat",
            "A feline rests on a rug",
            "Dogs are playing in the park",
            "Machine learning is fascinating",
            "Deep learning models require GPUs"
        ]

        logger.info(f"Initialized exporter for {model_name}")
        logger.info(f"Output directory: {self.output_dir}")
        logger.info(f"FP16 optimization: {use_fp16}")

    def load_model(self) -> SentenceTransformer:
        """Load the Sentence Transformer model."""
        logger.info(f"Loading model: {self.model_name}")
        model = SentenceTransformer(self.model_name)
        model.eval()
        logger.info("Model loaded successfully")
        return model

    def export_to_onnx(self, model: SentenceTransformer) -> str:
        """
        Export the model to ONNX format.

        Args:
            model: SentenceTransformer model to export

        Returns:
            Path to exported ONNX model
        """
        logger.info("Starting ONNX export...")

        # Get the underlying transformer model
        transformer = model[0].auto_model
        tokenizer = model.tokenizer

        # Create dummy input
        dummy_input = tokenizer(
            "This is a sample sentence",
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=128
        )

        # Extract input tensors
        input_ids = dummy_input['input_ids']
        attention_mask = dummy_input['attention_mask']

        logger.info(f"Input shape: {input_ids.shape}")

        # Dynamic axes for batch size and sequence length
        dynamic_axes = {
            'input_ids': {0: 'batch_size', 1: 'sequence_length'},
            'attention_mask': {0: 'batch_size', 1: 'sequence_length'},
            'output': {0: 'batch_size'}
        }

        # Export to ONNX
        torch.onnx.export(
            transformer,
            (input_ids, attention_mask),
            str(self.onnx_path),
            input_names=['input_ids', 'attention_mask'],
            output_names=['output'],
            dynamic_axes=dynamic_axes,
            opset_version=14,
            do_constant_folding=True,
            export_params=True,
            verbose=False
        )

        logger.info(f"Model exported to: {self.onnx_path}")

        # Verify ONNX model
        onnx_model = onnx.load(str(self.onnx_path))
        onnx.checker.check_model(onnx_model)
        logger.info("ONNX model verification passed")

        return str(self.onnx_path)

    def optimize_onnx(self) -> str:
        """
        Apply optimizations to the ONNX model.

        Returns:
            Path to optimized ONNX model
        """
        logger.info("Optimizing ONNX model...")

        # Load the ONNX model
        model = onnx.load(str(self.onnx_path))

        # Apply basic optimizations using onnx.optimizer
        try:
            from onnx import optimizer

            # List of optimization passes
            passes = [
                'eliminate_deadend',
                'eliminate_identity',
                'eliminate_nop_dropout',
                'eliminate_nop_monotone_argmax',
                'eliminate_nop_pad',
                'extract_constant_to_initializer',
                'eliminate_unused_initializer',
                'fuse_add_bias_into_conv',
                'fuse_bn_into_conv',
                'fuse_consecutive_concats',
                'fuse_consecutive_reduce_unsqueeze',
                'fuse_consecutive_squeezes',
                'fuse_consecutive_transposes',
                'fuse_matmul_add_bias_into_gemm',
                'fuse_pad_into_conv',
                'fuse_transpose_into_gemm',
            ]

            optimized_model = optimizer.optimize(model, passes)
            logger.info(f"Applied {len(passes)} optimization passes")

        except ImportError:
            logger.warning("onnx.optimizer not available, using basic optimizations")
            optimized_model = model

        # Save optimized model
        onnx.save(optimized_model, str(self.onnx_path))
        logger.info(f"Optimized model saved to: {self.onnx_path}")

        # Get model size
        size_mb = os.path.getsize(self.onnx_path) / (1024 * 1024)
        logger.info(f"Model size: {size_mb:.2f} MB")

        return str(self.onnx_path)

    def mean_pooling(self, token_embeddings: np.ndarray, attention_mask: np.ndarray) -> np.ndarray:
        """
        Apply mean pooling to token embeddings.

        Args:
            token_embeddings: Token-level embeddings
            attention_mask: Attention mask

        Returns:
            Sentence embeddings
        """
        # Expand attention mask to match embedding dimensions
        input_mask_expanded = np.expand_dims(attention_mask, -1).astype(np.float32)
        input_mask_expanded = np.broadcast_to(
            input_mask_expanded,
            token_embeddings.shape
        )

        # Apply mean pooling
        sum_embeddings = np.sum(token_embeddings * input_mask_expanded, axis=1)
        sum_mask = np.clip(np.sum(input_mask_expanded, axis=1), a_min=1e-9, a_max=None)

        return sum_embeddings / sum_mask

    def encode_with_pytorch(
        self,
        model: SentenceTransformer,
        sentences: List[str]
    ) -> np.ndarray:
        """
        Encode sentences using PyTorch model.

        Args:
            model: SentenceTransformer model
            sentences: List of sentences to encode

        Returns:
            Sentence embeddings
        """
        with torch.no_grad():
            embeddings = model.encode(sentences, convert_to_numpy=True)
        return embeddings

    def encode_with_onnx(
        self,
        sentences: List[str],
        tokenizer
    ) -> np.ndarray:
        """
        Encode sentences using ONNX model.

        Args:
            sentences: List of sentences to encode
            tokenizer: Tokenizer for preprocessing

        Returns:
            Sentence embeddings
        """
        # Create ONNX runtime session
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        # Use CUDA if available
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']

        session = ort.InferenceSession(str(self.onnx_path), sess_options, providers=providers)

        # Tokenize input
        encoded = tokenizer(
            sentences,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors='np'
        )

        # Run inference
        onnx_inputs = {
            'input_ids': encoded['input_ids'].astype(np.int64),
            'attention_mask': encoded['attention_mask'].astype(np.int64)
        }

        outputs = session.run(None, onnx_inputs)

        # Apply mean pooling
        token_embeddings = outputs[0]
        sentence_embeddings = self.mean_pooling(
            token_embeddings,
            encoded['attention_mask']
        )

        # Normalize embeddings
        sentence_embeddings = sentence_embeddings / np.linalg.norm(
            sentence_embeddings,
            axis=1,
            keepdims=True
        )

        return sentence_embeddings

    def validate_accuracy(
        self,
        model: SentenceTransformer
    ) -> Dict[str, float]:
        """
        Validate ONNX model accuracy against PyTorch baseline.

        Args:
            model: PyTorch SentenceTransformer model

        Returns:
            Dictionary with validation metrics
        """
        logger.info("Validating ONNX model accuracy...")

        # Get PyTorch embeddings
        pytorch_embeddings = self.encode_with_pytorch(model, self.test_sentences)

        # Get ONNX embeddings
        onnx_embeddings = self.encode_with_onnx(self.test_sentences, model.tokenizer)

        # Calculate cosine similarities
        similarities = []
        for pt_emb, onnx_emb in zip(pytorch_embeddings, onnx_embeddings):
            similarity = 1 - cosine(pt_emb, onnx_emb)
            similarities.append(similarity)

        # Calculate metrics
        avg_similarity = np.mean(similarities)
        min_similarity = np.min(similarities)
        max_similarity = np.max(similarities)

        # Calculate MSE
        mse = np.mean((pytorch_embeddings - onnx_embeddings) ** 2)

        metrics = {
            'avg_cosine_similarity': avg_similarity,
            'min_cosine_similarity': min_similarity,
            'max_cosine_similarity': max_similarity,
            'mse': mse
        }

        logger.info("Validation Metrics:")
        logger.info(f"  Average Cosine Similarity: {avg_similarity:.6f}")
        logger.info(f"  Min Cosine Similarity: {min_similarity:.6f}")
        logger.info(f"  Max Cosine Similarity: {max_similarity:.6f}")
        logger.info(f"  MSE: {mse:.8f}")

        # Check if accuracy threshold is met
        if avg_similarity >= 0.999:
            logger.info("✓ Accuracy validation PASSED (similarity >= 0.999)")
        else:
            logger.warning(f"⚠ Accuracy validation FAILED (similarity {avg_similarity:.6f} < 0.999)")

        return metrics

    def benchmark_performance(
        self,
        model: SentenceTransformer,
        num_iterations: int = 100
    ) -> Dict[str, float]:
        """
        Benchmark PyTorch vs ONNX inference performance.

        Args:
            model: PyTorch SentenceTransformer model
            num_iterations: Number of benchmark iterations

        Returns:
            Dictionary with benchmark results
        """
        logger.info(f"Benchmarking performance ({num_iterations} iterations)...")

        # Warm-up
        logger.info("Running warm-up iterations...")
        for _ in range(10):
            _ = self.encode_with_pytorch(model, self.test_sentences)
            _ = self.encode_with_onnx(self.test_sentences, model.tokenizer)

        # Benchmark PyTorch
        logger.info("Benchmarking PyTorch...")
        pytorch_times = []
        for _ in range(num_iterations):
            start = time.perf_counter()
            _ = self.encode_with_pytorch(model, self.test_sentences)
            pytorch_times.append(time.perf_counter() - start)

        # Benchmark ONNX
        logger.info("Benchmarking ONNX...")
        onnx_times = []
        for _ in range(num_iterations):
            start = time.perf_counter()
            _ = self.encode_with_onnx(self.test_sentences, model.tokenizer)
            onnx_times.append(time.perf_counter() - start)

        # Calculate statistics
        pytorch_avg = np.mean(pytorch_times) * 1000  # Convert to ms
        pytorch_std = np.std(pytorch_times) * 1000
        onnx_avg = np.mean(onnx_times) * 1000
        onnx_std = np.std(onnx_times) * 1000
        speedup = pytorch_avg / onnx_avg

        results = {
            'pytorch_avg_ms': pytorch_avg,
            'pytorch_std_ms': pytorch_std,
            'onnx_avg_ms': onnx_avg,
            'onnx_std_ms': onnx_std,
            'speedup': speedup
        }

        logger.info("Performance Benchmark Results:")
        logger.info(f"  PyTorch: {pytorch_avg:.2f} ± {pytorch_std:.2f} ms")
        logger.info(f"  ONNX:    {onnx_avg:.2f} ± {onnx_std:.2f} ms")
        logger.info(f"  Speedup: {speedup:.2f}x")

        return results

    def export_and_validate(self) -> Tuple[Dict[str, float], Dict[str, float]]:
        """
        Complete export and validation pipeline.

        Returns:
            Tuple of (validation_metrics, benchmark_results)
        """
        logger.info("="* 70)
        logger.info("Starting ONNX Export Pipeline")
        logger.info("="* 70)

        # Load model
        model = self.load_model()

        # Export to ONNX
        self.export_to_onnx(model)

        # Optimize ONNX model
        self.optimize_onnx()

        # Validate accuracy
        validation_metrics = self.validate_accuracy(model)

        # Benchmark performance
        benchmark_results = self.benchmark_performance(model)

        logger.info("="* 70)
        logger.info("ONNX Export Pipeline Completed")
        logger.info("="* 70)
        logger.info(f"ONNX model saved to: {self.onnx_path}")

        return validation_metrics, benchmark_results


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Export Sentence Transformer to ONNX format'
    )
    parser.add_argument(
        '--model-name',
        type=str,
        default='paraphrase-multilingual-MiniLM-L12-v2',
        help='HuggingFace model name'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='models',
        help='Output directory for ONNX model'
    )
    parser.add_argument(
        '--fp16',
        action='store_true',
        default=True,
        help='Use FP16 optimization'
    )
    parser.add_argument(
        '--benchmark-iterations',
        type=int,
        default=100,
        help='Number of benchmark iterations'
    )

    args = parser.parse_args()

    # Create exporter
    exporter = ONNXModelExporter(
        model_name=args.model_name,
        output_dir=args.output_dir,
        use_fp16=args.fp16
    )

    # Run export and validation
    validation_metrics, benchmark_results = exporter.export_and_validate()

    # Save results to file
    results_file = exporter.output_dir / 'onnx_export_results.txt'
    with open(results_file, 'w') as f:
        f.write("ONNX Export Results\n")
        f.write("=" * 70 + "\n\n")

        f.write("Model Information:\n")
        f.write(f"  Model: {args.model_name}\n")
        f.write(f"  ONNX Path: {exporter.onnx_path}\n")
        f.write(f"  Model Size: {os.path.getsize(exporter.onnx_path) / (1024*1024):.2f} MB\n\n")

        f.write("Validation Metrics:\n")
        for key, value in validation_metrics.items():
            f.write(f"  {key}: {value:.6f}\n")
        f.write("\n")

        f.write("Benchmark Results:\n")
        for key, value in benchmark_results.items():
            f.write(f"  {key}: {value:.2f}\n")

    logger.info(f"Results saved to: {results_file}")


if __name__ == '__main__':
    main()
