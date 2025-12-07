#!/usr/bin/env python3
"""
Build TensorRT engine from ONNX model for A100 GPU

Optimizations:
- FP16 precision for A100 (sm_80)
- Dynamic batch/sequence length support
- Optimized profiles for typical query patterns
- Fallback to trtexec if Python API unavailable
"""

import os
import sys
import json
import subprocess
import logging
from pathlib import Path
from typing import Optional, Dict, Tuple

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration
PROJECT_ROOT = Path(__file__).parent.parent.parent
MODELS_DIR = PROJECT_ROOT / "data" / "models"
ONNX_MODEL_PATH = MODELS_DIR / "minilm_l12_v2.onnx"
TRT_ENGINE_PATH = MODELS_DIR / "minilm_l12_v2_fp16.plan"

# Optimization profiles for dynamic shapes
# Format: (min, opt, max) for (batch_size, sequence_length)
PROFILE_CONFIG = {
    'min': (1, 1),      # Single token (edge case)
    'opt': (1, 32),     # Typical query length
    'max': (16, 128)    # Batch inference with long sequences
}

# TensorRT configuration
TRT_CONFIG = {
    'fp16': True,              # Enable FP16 for A100
    'workspace_size': 2048,    # 2GB workspace
    'verbose': True,
    'max_batch_size': 16
}


def check_dependencies() -> Tuple[bool, bool]:
    """Check if TensorRT Python API and trtexec are available"""
    has_python_api = False
    has_trtexec = False

    # Check Python API
    try:
        import tensorrt as trt
        has_python_api = True
        logger.info(f"✅ TensorRT Python API available: v{trt.__version__}")
    except ImportError:
        logger.warning("⚠️  TensorRT Python API not available")

    # Check trtexec
    result = subprocess.run(['which', 'trtexec'],
                          capture_output=True, text=True)
    if result.returncode == 0:
        has_trtexec = True
        trtexec_path = result.stdout.strip()
        logger.info(f"✅ trtexec available: {trtexec_path}")

        # Get version
        version_result = subprocess.run(
            ['trtexec', '--version'],
            capture_output=True, text=True
        )
        if version_result.returncode == 0:
            logger.info(f"   Version info: {version_result.stdout.strip()}")
    else:
        logger.warning("⚠️  trtexec not available")

    return has_python_api, has_trtexec


def build_engine_python_api(onnx_path: Path, engine_path: Path) -> bool:
    """Build TensorRT engine using Python API"""
    try:
        import tensorrt as trt

        logger.info("=" * 60)
        logger.info("Building TensorRT Engine (Python API)")
        logger.info("=" * 60)

        # Create builder and network
        TRT_LOGGER = trt.Logger(trt.Logger.VERBOSE if TRT_CONFIG['verbose'] else trt.Logger.INFO)
        builder = trt.Builder(TRT_LOGGER)
        network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        network = builder.create_network(network_flags)
        parser = trt.OnnxParser(network, TRT_LOGGER)

        # Parse ONNX model
        logger.info(f"📦 Parsing ONNX model: {onnx_path}")
        with open(onnx_path, 'rb') as f:
            if not parser.parse(f.read()):
                logger.error("Failed to parse ONNX model")
                for i in range(parser.num_errors):
                    logger.error(f"  Error {i}: {parser.get_error(i)}")
                return False

        logger.info("✅ ONNX model parsed successfully")
        logger.info(f"   Network inputs: {network.num_inputs}")
        logger.info(f"   Network outputs: {network.num_outputs}")

        # Print input details
        for i in range(network.num_inputs):
            input_tensor = network.get_input(i)
            logger.info(f"   Input {i}: {input_tensor.name} {input_tensor.shape} {input_tensor.dtype}")

        # Create builder config
        config = builder.create_builder_config()

        # Set workspace size
        config.set_memory_pool_limit(
            trt.MemoryPoolType.WORKSPACE,
            TRT_CONFIG['workspace_size'] * (1 << 20)  # Convert MB to bytes
        )

        # Enable FP16 for A100
        if TRT_CONFIG['fp16']:
            if builder.platform_has_fast_fp16:
                config.set_flag(trt.BuilderFlag.FP16)
                logger.info("✅ FP16 precision enabled (A100 optimized)")
            else:
                logger.warning("⚠️  FP16 not supported on this platform")

        # Create optimization profile for dynamic shapes
        profile = builder.create_optimization_profile()

        # Configure input_ids
        min_shape = (PROFILE_CONFIG['min'][0], PROFILE_CONFIG['min'][1])
        opt_shape = (PROFILE_CONFIG['opt'][0], PROFILE_CONFIG['opt'][1])
        max_shape = (PROFILE_CONFIG['max'][0], PROFILE_CONFIG['max'][1])

        profile.set_shape('input_ids', min_shape, opt_shape, max_shape)
        logger.info(f"   input_ids min: {min_shape}")
        logger.info(f"   input_ids opt: {opt_shape}")
        logger.info(f"   input_ids max: {max_shape}")

        # Configure attention_mask
        profile.set_shape('attention_mask', min_shape, opt_shape, max_shape)
        logger.info(f"   attention_mask min: {min_shape}")
        logger.info(f"   attention_mask opt: {opt_shape}")
        logger.info(f"   attention_mask max: {max_shape}")

        config.add_optimization_profile(profile)

        # Build engine
        logger.info("\n🔨 Building TensorRT engine...")
        logger.info("   This may take several minutes...")

        serialized_engine = builder.build_serialized_network(network, config)

        if serialized_engine is None:
            logger.error("❌ Failed to build engine")
            return False

        # Save engine
        logger.info(f"\n💾 Saving engine to: {engine_path}")
        with open(engine_path, 'wb') as f:
            f.write(serialized_engine)

        engine_size_mb = engine_path.stat().st_size / (1024 * 1024)
        logger.info(f"✅ Engine saved ({engine_size_mb:.2f} MB)")

        # Verify engine
        runtime = trt.Runtime(TRT_LOGGER)
        engine = runtime.deserialize_cuda_engine(serialized_engine)

        if engine is None:
            logger.error("❌ Failed to deserialize engine")
            return False

        logger.info("\n📊 Engine Information:")
        logger.info(f"   Inputs: {engine.num_io_tensors}")
        logger.info(f"   Max batch size: {engine.max_batch_size}")
        logger.info(f"   Device memory: {engine.device_memory_size / (1024*1024):.2f} MB")

        return True

    except Exception as e:
        logger.error(f"❌ Python API build failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def build_engine_trtexec(onnx_path: Path, engine_path: Path) -> bool:
    """Build TensorRT engine using trtexec command"""
    try:
        logger.info("=" * 60)
        logger.info("Building TensorRT Engine (trtexec)")
        logger.info("=" * 60)

        # Build trtexec command
        cmd = [
            'trtexec',
            f'--onnx={onnx_path}',
            f'--saveEngine={engine_path}',
        ]

        # Add FP16 flag
        if TRT_CONFIG['fp16']:
            cmd.append('--fp16')
            logger.info("✅ FP16 precision enabled")

        # Add workspace size
        cmd.append(f'--workspace={TRT_CONFIG["workspace_size"]}')

        # Add dynamic shapes
        min_shapes = f"input_ids:{PROFILE_CONFIG['min'][0]}x{PROFILE_CONFIG['min'][1]},"
        min_shapes += f"attention_mask:{PROFILE_CONFIG['min'][0]}x{PROFILE_CONFIG['min'][1]}"

        opt_shapes = f"input_ids:{PROFILE_CONFIG['opt'][0]}x{PROFILE_CONFIG['opt'][1]},"
        opt_shapes += f"attention_mask:{PROFILE_CONFIG['opt'][0]}x{PROFILE_CONFIG['opt'][1]}"

        max_shapes = f"input_ids:{PROFILE_CONFIG['max'][0]}x{PROFILE_CONFIG['max'][1]},"
        max_shapes += f"attention_mask:{PROFILE_CONFIG['max'][0]}x{PROFILE_CONFIG['max'][1]}"

        cmd.extend([
            f'--minShapes={min_shapes}',
            f'--optShapes={opt_shapes}',
            f'--maxShapes={max_shapes}'
        ])

        # Add verbose output
        if TRT_CONFIG['verbose']:
            cmd.append('--verbose')

        logger.info("🔨 Running trtexec...")
        logger.info(f"Command: {' '.join(cmd)}")
        logger.info("This may take several minutes...")

        # Run trtexec
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=False
        )

        # Print output
        if result.stdout:
            logger.info("\nOutput:")
            for line in result.stdout.split('\n'):
                if line.strip():
                    logger.info(f"  {line}")

        if result.stderr:
            logger.warning("\nWarnings/Errors:")
            for line in result.stderr.split('\n'):
                if line.strip():
                    logger.warning(f"  {line}")

        # Check result
        if result.returncode != 0:
            logger.error(f"❌ trtexec failed with exit code {result.returncode}")
            return False

        # Verify engine file
        if not engine_path.exists():
            logger.error("❌ Engine file not created")
            return False

        engine_size_mb = engine_path.stat().st_size / (1024 * 1024)
        logger.info(f"\n✅ Engine built successfully ({engine_size_mb:.2f} MB)")

        return True

    except Exception as e:
        logger.error(f"❌ trtexec build failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def validate_engine(engine_path: Path) -> Dict:
    """Validate and benchmark the built engine"""
    try:
        logger.info("\n" + "=" * 60)
        logger.info("Validating TensorRT Engine")
        logger.info("=" * 60)

        # Try Python API validation
        try:
            import tensorrt as trt

            TRT_LOGGER = trt.Logger(trt.Logger.INFO)
            runtime = trt.Runtime(TRT_LOGGER)

            with open(engine_path, 'rb') as f:
                engine = runtime.deserialize_cuda_engine(f.read())

            if engine is None:
                raise RuntimeError("Failed to deserialize engine")

            stats = {
                'valid': True,
                'num_io_tensors': engine.num_io_tensors,
                'device_memory_mb': engine.device_memory_size / (1024*1024),
                'file_size_mb': engine_path.stat().st_size / (1024*1024)
            }

            logger.info("✅ Engine validation successful")
            logger.info(f"   I/O tensors: {stats['num_io_tensors']}")
            logger.info(f"   Device memory: {stats['device_memory_mb']:.2f} MB")
            logger.info(f"   File size: {stats['file_size_mb']:.2f} MB")

            return stats

        except ImportError:
            logger.warning("⚠️  Cannot validate without TensorRT Python API")
            stats = {
                'valid': True,
                'file_size_mb': engine_path.stat().st_size / (1024*1024)
            }
            logger.info(f"✅ Engine file exists ({stats['file_size_mb']:.2f} MB)")
            return stats

    except Exception as e:
        logger.error(f"❌ Validation failed: {e}")
        return {'valid': False, 'error': str(e)}


def save_build_metadata(engine_path: Path, build_method: str, validation: Dict):
    """Save build metadata for tracking"""
    metadata = {
        'onnx_model': str(ONNX_MODEL_PATH.name),
        'engine_path': str(engine_path.name),
        'build_method': build_method,
        'profile_config': PROFILE_CONFIG,
        'trt_config': TRT_CONFIG,
        'validation': validation
    }

    metadata_path = engine_path.with_suffix('.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"\n💾 Metadata saved to: {metadata_path}")


def main():
    """Main execution"""
    logger.info("=" * 60)
    logger.info("🚀 TensorRT Engine Builder for A100 GPU")
    logger.info("=" * 60)
    logger.info(f"\nONNX Model: {ONNX_MODEL_PATH}")
    logger.info(f"Output Engine: {TRT_ENGINE_PATH}")
    logger.info(f"\nOptimization Profiles:")
    logger.info(f"  Min shape (batch, seq): {PROFILE_CONFIG['min']}")
    logger.info(f"  Opt shape (batch, seq): {PROFILE_CONFIG['opt']}")
    logger.info(f"  Max shape (batch, seq): {PROFILE_CONFIG['max']}")
    logger.info(f"\nTensorRT Config:")
    logger.info(f"  FP16: {TRT_CONFIG['fp16']}")
    logger.info(f"  Workspace: {TRT_CONFIG['workspace_size']} MB")
    logger.info("")

    # Ensure directories exist
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # Check if ONNX model exists
    if not ONNX_MODEL_PATH.exists():
        logger.error(f"❌ ONNX model not found: {ONNX_MODEL_PATH}")
        logger.error("   Run convert_to_onnx.py first")
        return 1

    logger.info(f"✅ ONNX model found ({ONNX_MODEL_PATH.stat().st_size / (1024*1024):.2f} MB)")

    # Check dependencies
    has_python_api, has_trtexec = check_dependencies()

    if not has_python_api and not has_trtexec:
        logger.error("❌ Neither TensorRT Python API nor trtexec available")
        logger.error("   Install TensorRT to continue")
        return 1

    # Try Python API first, fallback to trtexec
    build_success = False
    build_method = None

    if has_python_api:
        logger.info("\n🎯 Attempting build with Python API...")
        build_success = build_engine_python_api(ONNX_MODEL_PATH, TRT_ENGINE_PATH)
        build_method = 'python_api'

        if not build_success and has_trtexec:
            logger.warning("\n⚠️  Python API build failed, trying trtexec...")
            build_success = build_engine_trtexec(ONNX_MODEL_PATH, TRT_ENGINE_PATH)
            build_method = 'trtexec'

    elif has_trtexec:
        logger.info("\n🎯 Building with trtexec...")
        build_success = build_engine_trtexec(ONNX_MODEL_PATH, TRT_ENGINE_PATH)
        build_method = 'trtexec'

    if not build_success:
        logger.error("\n❌ ENGINE BUILD FAILED")
        return 1

    # Validate engine
    validation = validate_engine(TRT_ENGINE_PATH)

    # Save metadata
    save_build_metadata(TRT_ENGINE_PATH, build_method, validation)

    # Final summary
    logger.info("\n" + "=" * 60)
    logger.info("✅ ENGINE BUILD COMPLETE")
    logger.info("=" * 60)
    logger.info(f"\nEngine path: {TRT_ENGINE_PATH}")
    logger.info(f"Build method: {build_method}")
    logger.info(f"FP16 enabled: {TRT_CONFIG['fp16']}")
    logger.info(f"Ready for A100 inference")

    return 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        logger.warning("\n\n⚠️  Build interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
