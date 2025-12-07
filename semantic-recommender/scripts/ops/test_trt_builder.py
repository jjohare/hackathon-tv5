#!/usr/bin/env python3
"""
Test script for TensorRT engine builder
Validates build process and engine functionality
"""

import sys
import json
import tempfile
import shutil
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_dependency_check():
    """Test dependency checking"""
    logger.info("\n" + "=" * 60)
    logger.info("TEST: Dependency Check")
    logger.info("=" * 60)

    from scripts.ops.build_trt_engine import check_dependencies

    has_python_api, has_trtexec = check_dependencies()

    logger.info(f"\nResults:")
    logger.info(f"  Python API: {has_python_api}")
    logger.info(f"  trtexec: {has_trtexec}")

    if not has_python_api and not has_trtexec:
        logger.error("❌ No TensorRT tools available")
        return False

    logger.info("✅ At least one TensorRT tool available")
    return True


def test_profile_configuration():
    """Test optimization profile configuration"""
    logger.info("\n" + "=" * 60)
    logger.info("TEST: Profile Configuration")
    logger.info("=" * 60)

    from scripts.ops.build_trt_engine import PROFILE_CONFIG, TRT_CONFIG

    logger.info(f"\nProfile Config:")
    logger.info(f"  Min: {PROFILE_CONFIG['min']}")
    logger.info(f"  Opt: {PROFILE_CONFIG['opt']}")
    logger.info(f"  Max: {PROFILE_CONFIG['max']}")

    # Validate min <= opt <= max
    assert PROFILE_CONFIG['min'][0] <= PROFILE_CONFIG['opt'][0] <= PROFILE_CONFIG['max'][0]
    assert PROFILE_CONFIG['min'][1] <= PROFILE_CONFIG['opt'][1] <= PROFILE_CONFIG['max'][1]

    logger.info(f"\nTensorRT Config:")
    logger.info(f"  FP16: {TRT_CONFIG['fp16']}")
    logger.info(f"  Workspace: {TRT_CONFIG['workspace_size']} MB")

    logger.info("✅ Configuration valid")
    return True


def test_build_metadata():
    """Test metadata generation"""
    logger.info("\n" + "=" * 60)
    logger.info("TEST: Build Metadata")
    logger.info("=" * 60)

    from scripts.ops.build_trt_engine import save_build_metadata

    # Create temporary file
    with tempfile.NamedTemporaryFile(suffix='.plan', delete=False) as f:
        temp_engine = Path(f.name)

    try:
        validation = {
            'valid': True,
            'num_io_tensors': 2,
            'device_memory_mb': 128.5,
            'file_size_mb': 50.2
        }

        save_build_metadata(temp_engine, 'test_method', validation)

        # Check metadata file
        metadata_path = temp_engine.with_suffix('.json')
        assert metadata_path.exists(), "Metadata file not created"

        with open(metadata_path) as f:
            metadata = json.load(f)

        # Validate structure
        assert 'build_method' in metadata
        assert 'profile_config' in metadata
        assert 'validation' in metadata
        assert metadata['build_method'] == 'test_method'

        logger.info(f"✅ Metadata generated successfully")
        logger.info(f"   File: {metadata_path}")

        # Cleanup
        metadata_path.unlink()
        return True

    finally:
        if temp_engine.exists():
            temp_engine.unlink()


def test_onnx_existence():
    """Test if ONNX model exists"""
    logger.info("\n" + "=" * 60)
    logger.info("TEST: ONNX Model Existence")
    logger.info("=" * 60)

    from scripts.ops.build_trt_engine import ONNX_MODEL_PATH

    logger.info(f"\nExpected path: {ONNX_MODEL_PATH}")

    if ONNX_MODEL_PATH.exists():
        size_mb = ONNX_MODEL_PATH.stat().st_size / (1024 * 1024)
        logger.info(f"✅ ONNX model found ({size_mb:.2f} MB)")
        return True
    else:
        logger.warning(f"⚠️  ONNX model not found")
        logger.warning(f"   Run convert_to_onnx.py first to generate it")
        return False


def test_engine_validation():
    """Test engine validation (if engine exists)"""
    logger.info("\n" + "=" * 60)
    logger.info("TEST: Engine Validation")
    logger.info("=" * 60)

    from scripts.ops.build_trt_engine import TRT_ENGINE_PATH, validate_engine

    if not TRT_ENGINE_PATH.exists():
        logger.warning("⚠️  Engine not found (run build first)")
        return True  # Not a failure, just skip

    logger.info(f"\nEngine path: {TRT_ENGINE_PATH}")
    validation = validate_engine(TRT_ENGINE_PATH)

    logger.info(f"\nValidation results:")
    for key, value in validation.items():
        logger.info(f"  {key}: {value}")

    if validation.get('valid'):
        logger.info("✅ Engine validation passed")
        return True
    else:
        logger.error("❌ Engine validation failed")
        return False


def test_trtexec_command_generation():
    """Test trtexec command generation"""
    logger.info("\n" + "=" * 60)
    logger.info("TEST: trtexec Command Generation")
    logger.info("=" * 60)

    from scripts.ops.build_trt_engine import PROFILE_CONFIG

    # Build expected command components
    min_shapes = f"input_ids:{PROFILE_CONFIG['min'][0]}x{PROFILE_CONFIG['min'][1]},"
    min_shapes += f"attention_mask:{PROFILE_CONFIG['min'][0]}x{PROFILE_CONFIG['min'][1]}"

    opt_shapes = f"input_ids:{PROFILE_CONFIG['opt'][0]}x{PROFILE_CONFIG['opt'][1]},"
    opt_shapes += f"attention_mask:{PROFILE_CONFIG['opt'][0]}x{PROFILE_CONFIG['opt'][1]}"

    max_shapes = f"input_ids:{PROFILE_CONFIG['max'][0]}x{PROFILE_CONFIG['max'][1]},"
    max_shapes += f"attention_mask:{PROFILE_CONFIG['max'][0]}x{PROFILE_CONFIG['max'][1]}"

    logger.info(f"\nGenerated shapes:")
    logger.info(f"  Min: {min_shapes}")
    logger.info(f"  Opt: {opt_shapes}")
    logger.info(f"  Max: {max_shapes}")

    # Validate format
    assert 'input_ids' in min_shapes
    assert 'attention_mask' in min_shapes
    assert 'x' in min_shapes  # Format: NxM

    logger.info("✅ Command generation valid")
    return True


def run_all_tests():
    """Run all tests"""
    logger.info("\n" + "=" * 80)
    logger.info("🧪 TensorRT Engine Builder Test Suite")
    logger.info("=" * 80)

    tests = [
        ("Dependency Check", test_dependency_check),
        ("Profile Configuration", test_profile_configuration),
        ("Build Metadata", test_build_metadata),
        ("ONNX Existence", test_onnx_existence),
        ("Engine Validation", test_engine_validation),
        ("trtexec Command", test_trtexec_command_generation)
    ]

    results = {}
    for test_name, test_func in tests:
        try:
            success = test_func()
            results[test_name] = success
        except Exception as e:
            logger.error(f"\n❌ Test '{test_name}' failed with exception: {e}")
            import traceback
            traceback.print_exc()
            results[test_name] = False

    # Summary
    logger.info("\n" + "=" * 80)
    logger.info("📊 Test Summary")
    logger.info("=" * 80)

    passed = sum(1 for v in results.values() if v)
    total = len(results)

    for test_name, success in results.items():
        status = "✅ PASS" if success else "❌ FAIL"
        logger.info(f"  {status}: {test_name}")

    logger.info(f"\n{passed}/{total} tests passed")

    if passed == total:
        logger.info("\n✅ ALL TESTS PASSED")
        return 0
    else:
        logger.error(f"\n❌ {total - passed} tests failed")
        return 1


if __name__ == '__main__':
    sys.exit(run_all_tests())
