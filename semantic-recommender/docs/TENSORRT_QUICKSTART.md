# TensorRT Quickstart Guide

**Goal:** Accelerate hyper-personalization from 119 QPS → >1000 QPS

## Current Status

✅ **PyTorch Baseline Benchmark Complete**
- QPS: 119.3 (target: >1000)
- Encoding: 7.53ms (target: <1ms)
- Total: 8.38ms (target: <2ms)

❌ **TensorRT Not Yet Installed**
- Expected speedup: 7.5x (encoding) → 500-667 QPS
- With INT8: 12-18x → >1000 QPS

---

## Step 1: Install TensorRT (5 minutes)

### Option A: PIP Install (Easiest)

```bash
# Activate venv
source venv/bin/activate

# Install TensorRT and PyCUDA
pip install tensorrt pycuda nvidia-tensorrt

# Verify installation
python -c "import tensorrt as trt; print(f'TensorRT: {trt.__version__}')"
python -c "import pycuda.driver as cuda; print('PyCUDA: OK')"
```

### Option B: NVIDIA Container (Recommended for Production)

```bash
# Pull NVIDIA TensorRT container
docker pull nvcr.io/nvidia/tensorrt:24.10-py3

# Run with GPU access
docker run --gpus all -it --rm \
  -v $(pwd):/workspace \
  nvcr.io/nvidia/tensorrt:24.10-py3
```

---

## Step 2: Build TensorRT Engine (5-10 minutes)

### FP16 Engine (Recommended First)

```bash
# Build FP16 TensorRT engine
python scripts/build_trt_engine.py \
  --model paraphrase-multilingual-MiniLM-L12-v2 \
  --precision fp16 \
  --max-batch-size 32 \
  --output data/models/minilm_l12_v2_fp16.plan

# Expected output:
# ✅ TensorRT engine saved: data/models/minilm_l12_v2_fp16.plan
#    Size: ~50-100 MB
#    Precision: FP16
```

**Build Time:** 2-5 minutes (one-time, cached for inference)

### INT8 Engine (If FP16 Insufficient)

```bash
# Generate calibration dataset (100-1000 sample texts)
python -c "
with open('data/calibration_texts.txt', 'w') as f:
    texts = [
        'action movies with explosions',
        'romantic comedies',
        'sci-fi space exploration',
        # ... add 100+ diverse queries
    ]
    f.write('\n'.join(texts))
"

# Build INT8 engine with calibration
python scripts/build_trt_engine.py \
  --model paraphrase-multilingual-MiniLM-L12-v2 \
  --precision int8 \
  --calibration-data data/calibration_texts.txt \
  --output data/models/minilm_l12_v2_int8.plan
```

---

## Step 3: Benchmark TensorRT (2 minutes)

### Run Benchmark

```bash
# Benchmark TensorRT FP16
python scripts/benchmark_gpu_hyper_personalization.py \
  --mode tensorrt \
  --engine data/models/minilm_l12_v2_fp16.plan \
  --num-users 1000 \
  --num-queries 100 \
  --output data/benchmark_results_tensorrt_fp16.json
```

### Expected Results (FP16)

```
Encoding Performance:
  Mean Latency: 0.8-1.0 ms  (vs 7.53ms PyTorch → 7.5x faster)
  P99 Latency:  1.2-1.5 ms
  Max QPS:      1000-1250   (vs 133 PyTorch → 7.5x improvement)

End-to-End Performance:
  Mean Latency: 1.5-2.0 ms  (vs 8.38ms PyTorch → 4.2x faster)
  P99 Latency:  2.5-3.0 ms
  QPS:          500-667     (vs 119 PyTorch → 4.2-5.6x improvement)

Target Achievement:
  Encoding < 1ms:     ✅ PASS (0.8-1.0 ms)
  Total < 2ms:        ⚠️  MARGINAL (1.5-2.0 ms)
  QPS > 1000:         ⚠️  PARTIAL (500-667, need INT8 for >1000)
```

### Expected Results (INT8)

```
Encoding Performance:
  Mean Latency: 0.4-0.6 ms  (vs 7.53ms PyTorch → 12-18x faster)
  Max QPS:      1667-2500   (vs 133 PyTorch → 12-18x improvement)

End-to-End Performance:
  Mean Latency: 1.0-1.5 ms  (vs 8.38ms PyTorch → 5.6-8.4x faster)
  QPS:          667-1000    (vs 119 PyTorch → 5.6-8.4x improvement)

Target Achievement:
  Encoding < 1ms:     ✅ PASS (0.4-0.6 ms)
  Total < 2ms:        ✅ PASS (1.0-1.5 ms)
  QPS > 1000:         ✅ PASS (667-1000+)
```

---

## Step 4: Compare Results (1 minute)

```bash
# Generate comparison report
python scripts/benchmark_gpu_hyper_personalization.py \
  --mode both \
  --engine data/models/minilm_l12_v2_fp16.plan \
  --output data/benchmark_comparison.json
```

**Expected Output:**

```
PYTORCH vs TENSORRT COMPARISON

QPS:
  PyTorch:   119.3
  TensorRT:  500-667
  Speedup:   4.2-5.6x

Encoding Latency:
  PyTorch:   7.53 ms
  TensorRT:  0.8-1.0 ms
  Speedup:   7.5-9.4x
```

---

## Troubleshooting

### TensorRT Import Error

```bash
# Error: No module named 'tensorrt'
pip install tensorrt pycuda nvidia-tensorrt

# Verify CUDA is available
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
```

### Engine Build Fails

```bash
# Error: ONNX export failed
# Solution: Install compatible ONNX version
pip install onnx>=1.14.0 onnxruntime-gpu

# Error: Out of memory during build
# Solution: Reduce max batch size
python scripts/build_trt_engine.py --max-batch-size 16 ...
```

### Slower Than Expected

```bash
# Verify FP16 is enabled
python -c "
import tensorrt as trt
print(f'TensorRT version: {trt.__version__}')
# Should be 8.6+ for RTX A6000 Ampere optimizations
"

# Check GPU is being used
nvidia-smi  # Should show GPU utilization during benchmark
```

---

## Next Steps After TensorRT

### If Targets Met (>1000 QPS, <2ms)

1. ✅ Deploy to production
2. Load testing with real traffic
3. Monitor GPU utilization (target: >80%)
4. Scale horizontally if needed

### If Targets Not Met (500-1000 QPS)

1. Try INT8 quantization (Step 2)
2. Batch processing (process 16-32 queries at once)
3. Multi-stream inference (parallel CUDA streams)
4. Profile with `nsys` to find remaining bottlenecks

### Production Deployment

```python
# Update gpu_hyper_personalization.py to use TensorRT by default
system = GPUHyperPersonalization(
    use_tensorrt=True,
    engine_path="data/models/minilm_l12_v2_fp16.plan"
)

# Serve via REST API
@app.post("/recommend")
def recommend(query: str, user_id: str):
    result = system.personalized_search(
        user_id=user_id,
        query=query,
        top_k=10
    )
    return result['results']
```

---

## Performance Roadmap

| Stage | QPS | Latency | Status |
|-------|-----|---------|--------|
| **CPU Baseline** | 70 | 14ms | ✅ Baseline |
| **PyTorch GPU** | 119 | 8.38ms | ✅ Current |
| **TensorRT FP16** | 500-667 | 1.5-2.0ms | 🎯 Next |
| **TensorRT INT8** | 667-1000 | 1.0-1.5ms | 🎯 Target |
| **Batching** | 1000-2000 | 0.5-1.0ms | 🚀 Future |
| **Multi-Stream** | 2000-3000 | <0.5ms | 🚀 Scale |

---

## Files Reference

- **Benchmark Script:** `scripts/benchmark_gpu_hyper_personalization.py`
- **Engine Builder:** `scripts/build_trt_engine.py`
- **TRT Inference:** `scripts/utils/trt_inference.py`
- **Results:** `data/benchmark_results_*.json`
- **This Guide:** `docs/TENSORRT_QUICKSTART.md`

---

**Estimated Time to >1000 QPS:** 15-30 minutes (install + build + benchmark)
