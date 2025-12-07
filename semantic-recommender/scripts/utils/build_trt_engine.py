#!/usr/bin/env python3
"""
TensorRT Engine Builder for Semantic Recommender

Builds optimized TensorRT engines from ONNX models with:
- FP16 precision for 2-4x speedup
- GPU-specific optimization (sm_86 for RTX A6000)
- Dynamic batching support
- Memory optimization
"""

import os
import sys
import logging
import argparse
from pathlib import Path
import tensorrt as trt

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# TensorRT logger
TRT_LOGGER = trt.Logger(trt.Logger.INFO)


def build_engine(
    onnx_path: str,
    engine_path: str,
    precision: str = "fp16",
    compute_capability: str = "sm_86",
    max_batch_size: int = 32,
    max_seq_length: int = 128,
    workspace_size: int = 4,
) -> bool:
    """
    Build TensorRT engine from ONNX model.

    Args:
        onnx_path: Path to ONNX model
        engine_path: Path to save TensorRT engine
        precision: Precision mode (fp16 or fp32)
        compute_capability: GPU compute capability (sm_86 for RTX A6000)
        max_batch_size: Maximum batch size
        max_seq_length: Maximum sequence length
        workspace_size: Workspace size in GB

    Returns:
        True if successful, False otherwise
    """
    logger.info(f"Building TensorRT engine from {onnx_path}")
    logger.info(f"Target precision: {precision}")
    logger.info(f"Target GPU: {compute_capability}")
    logger.info(f"Max batch size: {max_batch_size}")
    logger.info(f"Max sequence length: {max_seq_length}")

    try:
        # Create builder and network
        builder = trt.Builder(TRT_LOGGER)
        network = builder.create_network(
            1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        )
        parser = trt.OnnxParser(network, TRT_LOGGER)

        # Parse ONNX model
        logger.info(f"Parsing ONNX model: {onnx_path}")
        with open(onnx_path, 'rb') as f:
            if not parser.parse(f.read()):
                logger.error("Failed to parse ONNX model")
                for error_idx in range(parser.num_errors):
                    logger.error(f"Parser error {error_idx}: {parser.get_error(error_idx)}")
                return False

        logger.info("ONNX model parsed successfully")
        logger.info(f"Network inputs: {[network.get_input(i).name for i in range(network.num_inputs)]}")
        logger.info(f"Network outputs: {[network.get_output(i).name for i in range(network.num_outputs)]}")

        # Create builder config
        config = builder.create_builder_config()

        # Set workspace size
        workspace_bytes = workspace_size * (1 << 30)  # GB to bytes
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_bytes)
        logger.info(f"Workspace size: {workspace_size}GB ({workspace_bytes} bytes)")

        # Enable FP16 precision if requested
        if precision.lower() == "fp16":
            if builder.platform_has_fast_fp16:
                config.set_flag(trt.BuilderFlag.FP16)
                logger.info("FP16 precision enabled")
            else:
                logger.warning("FP16 not supported on this platform, falling back to FP32")

        # Set optimization profiles for dynamic shapes
        profile = builder.create_optimization_profile()

        # Configure dynamic shapes for input_ids
        profile.set_shape(
            "input_ids",
            min=(1, 1),  # min batch=1, min seq_len=1
            opt=(max_batch_size // 2, max_seq_length // 2),  # optimal batch=16, seq_len=64
            max=(max_batch_size, max_seq_length)  # max batch=32, seq_len=128
        )

        # Configure dynamic shapes for attention_mask
        profile.set_shape(
            "attention_mask",
            min=(1, 1),
            opt=(max_batch_size // 2, max_seq_length // 2),
            max=(max_batch_size, max_seq_length)
        )

        config.add_optimization_profile(profile)
        logger.info("Optimization profile configured for dynamic shapes")

        # Build engine
        logger.info("Building TensorRT engine (this may take a few minutes)...")
        serialized_engine = builder.build_serialized_network(network, config)

        if serialized_engine is None:
            logger.error("Failed to build TensorRT engine")
            return False

        # Save engine to file
        output_path = Path(engine_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(engine_path, 'wb') as f:
            f.write(serialized_engine)

        logger.info(f"TensorRT engine saved to: {engine_path}")

        # Get file size
        file_size_mb = os.path.getsize(engine_path) / (1024 * 1024)
        logger.info(f"Engine file size: {file_size_mb:.2f} MB")

        return True

    except Exception as e:
        logger.error(f"Engine build failed: {e}", exc_info=True)
        return False


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description='Build TensorRT engine from ONNX model')
    parser.add_argument(
        '--onnx-path',
        type=str,
        required=True,
        help='Path to ONNX model'
    )
    parser.add_argument(
        '--engine-path',
        type=str,
        required=True,
        help='Output path for TensorRT engine'
    )
    parser.add_argument(
        '--precision',
        type=str,
        default='fp16',
        choices=['fp16', 'fp32'],
        help='Precision mode (fp16 or fp32)'
    )
    parser.add_argument(
        '--compute-capability',
        type=str,
        default='sm_86',
        help='GPU compute capability (e.g., sm_86 for RTX A6000)'
    )
    parser.add_argument(
        '--max-batch-size',
        type=int,
        default=32,
        help='Maximum batch size'
    )
    parser.add_argument(
        '--max-seq-length',
        type=int,
        default=128,
        help='Maximum sequence length'
    )
    parser.add_argument(
        '--workspace-size',
        type=int,
        default=4,
        help='Workspace size in GB'
    )

    args = parser.parse_args()

    # Verify ONNX file exists
    if not os.path.exists(args.onnx_path):
        logger.error(f"ONNX model not found: {args.onnx_path}")
        return 1

    # Build engine
    success = build_engine(
        onnx_path=args.onnx_path,
        engine_path=args.engine_path,
        precision=args.precision,
        compute_capability=args.compute_capability,
        max_batch_size=args.max_batch_size,
        max_seq_length=args.max_seq_length,
        workspace_size=args.workspace_size,
    )

    if success:
        logger.info("TensorRT engine build completed successfully")
        return 0
    else:
        logger.error("TensorRT engine build failed")
        return 1


if __name__ == '__main__':
    sys.exit(main())
