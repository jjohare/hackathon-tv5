# TensorRT Engine Build - Quick Start

## One-Line Build

```bash
./scripts/ops/build_all_optimized.sh
```

This runs the complete pipeline:
1. Converts PyTorch → ONNX
2. Builds TensorRT FP16 engine
3. Validates outputs

## Individual Steps

### Step 1: Convert to ONNX
```bash
python scripts/ops/convert_to_onnx.py
```
**Output:** `data/models/minilm_l12_v2.onnx`

### Step 2: Build TensorRT Engine
```bash
python scripts/ops/build_trt_engine.py
```
**Output:** `data/models/minilm_l12_v2_fp16.plan`

### Step 3: Validate
```bash
python scripts/ops/test_trt_builder.py
```

## Verify Installation

### Check TensorRT
```bash
python3 -c "import tensorrt; print(tensorrt.__version__)"
```

### Check trtexec
```bash
which trtexec
trtexec --version
```

## Expected Results

### File Sizes
- ONNX: ~45 MB
- TensorRT Engine: ~25-30 MB
- Metadata: ~1 KB

### Performance (A100)
- Latency: ~1.5ms per query
- Throughput: ~700 queries/sec
- Speedup: 2x vs ONNX Runtime

## Troubleshooting

### Error: TensorRT not found
```bash
# Install via NVIDIA container
docker pull nvcr.io/nvidia/pytorch:24.01-py3
```

### Error: ONNX model not found
```bash
# Run Phase 1 first
python scripts/ops/convert_to_onnx.py
```

## Configuration

Located in `build_trt_engine.py`:

```python
# Optimization profiles
PROFILE_CONFIG = {
    'min': (1, 1),      # Min batch/seq
    'opt': (1, 32),     # Optimal for queries
    'max': (16, 128)    # Max batch/seq
}

# TensorRT settings
TRT_CONFIG = {
    'fp16': True,              # A100 FP16
    'workspace_size': 2048,    # 2GB
}
```

## Output Files

```
data/models/
├── minilm_l12_v2.onnx          # ONNX model (Phase 1)
├── minilm_l12_v2_fp16.plan     # TensorRT engine (Phase 2)
└── minilm_l12_v2_fp16.json     # Build metadata
```

## Next Steps

1. Deploy to A100 VM
2. Run inference benchmarks
3. Integrate into recommendation service

## Documentation

- **Full Guide:** `README_TRT.md`
- **Implementation:** `PHASE2_SUMMARY.md`
- **Code:** `build_trt_engine.py`
