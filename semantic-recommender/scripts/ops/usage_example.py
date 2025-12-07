#!/usr/bin/env python3
"""
Example usage of TensorRT engine builder
Demonstrates basic build and validation workflow
"""

from pathlib import Path
import sys

# Add project to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.ops.build_trt_engine import (
    check_dependencies,
    PROFILE_CONFIG,
    TRT_CONFIG,
    ONNX_MODEL_PATH,
    TRT_ENGINE_PATH
)


def example_basic_usage():
    """Basic usage example"""
    print("=" * 60)
    print("Basic Usage Example")
    print("=" * 60)
    
    # 1. Check dependencies
    print("\n1. Checking dependencies...")
    has_python_api, has_trtexec = check_dependencies()
    
    if not has_python_api and not has_trtexec:
        print("❌ No TensorRT tools available")
        print("   Install TensorRT or use NVIDIA container")
        return
    
    # 2. Check ONNX model
    print("\n2. Checking ONNX model...")
    if ONNX_MODEL_PATH.exists():
        print(f"✅ Found: {ONNX_MODEL_PATH}")
    else:
        print(f"❌ Not found: {ONNX_MODEL_PATH}")
        print("   Run: python scripts/ops/convert_to_onnx.py")
        return
    
    # 3. Show configuration
    print("\n3. Configuration:")
    print(f"   Min shape: {PROFILE_CONFIG['min']}")
    print(f"   Opt shape: {PROFILE_CONFIG['opt']}")
    print(f"   Max shape: {PROFILE_CONFIG['max']}")
    print(f"   FP16: {TRT_CONFIG['fp16']}")
    
    # 4. Build engine
    print("\n4. To build engine:")
    print("   python scripts/ops/build_trt_engine.py")
    
    # 5. Expected output
    print("\n5. Expected output:")
    print(f"   {TRT_ENGINE_PATH}")


def example_custom_config():
    """Example with custom configuration"""
    print("\n" + "=" * 60)
    print("Custom Configuration Example")
    print("=" * 60)
    
    # Custom profiles for different use cases
    configs = {
        'single_query': {
            'min': (1, 1),
            'opt': (1, 16),
            'max': (1, 64)
        },
        'batch_inference': {
            'min': (1, 1),
            'opt': (8, 32),
            'max': (32, 128)
        },
        'streaming': {
            'min': (1, 1),
            'opt': (4, 24),
            'max': (8, 96)
        }
    }
    
    print("\nExample configurations:")
    for name, config in configs.items():
        print(f"\n{name}:")
        print(f"  Min: {config['min']}")
        print(f"  Opt: {config['opt']}")
        print(f"  Max: {config['max']}")
    
    print("\nTo use custom config:")
    print("  1. Edit PROFILE_CONFIG in build_trt_engine.py")
    print("  2. Run: python scripts/ops/build_trt_engine.py")


def example_validation():
    """Example validation workflow"""
    print("\n" + "=" * 60)
    print("Validation Example")
    print("=" * 60)
    
    print("\n1. Run test suite:")
    print("   python scripts/ops/test_trt_builder.py")
    
    print("\n2. Check engine file:")
    print("   ls -lh data/models/*.plan")
    
    print("\n3. View metadata:")
    print("   cat data/models/minilm_l12_v2_fp16.json")
    
    print("\n4. Verify GPU compatibility:")
    print("   nvidia-smi --query-gpu=name,compute_cap --format=csv")


if __name__ == '__main__':
    example_basic_usage()
    example_custom_config()
    example_validation()
    
    print("\n" + "=" * 60)
    print("✅ Examples Complete")
    print("=" * 60)
    print("\nFor full documentation, see:")
    print("  - README_TRT.md (usage guide)")
    print("  - PHASE2_SUMMARY.md (implementation details)")
    print("  - QUICK_START.md (quick reference)")
