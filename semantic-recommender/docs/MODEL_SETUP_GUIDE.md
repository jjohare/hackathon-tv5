# Model Setup Guide - TensorRT Engine Creation

**Purpose**: Build the TensorRT FP16 engine required for GPU-accelerated semantic search

**Time Required**: ~5 minutes on RTX A6000 or similar GPU

---

## Prerequisites

### System Requirements
- **GPU**: NVIDIA GPU with Compute Capability 7.0+ (Volta or newer)
- **CUDA**: CUDA Toolkit 11.0+ (tested with CUDA 13.0)
- **TensorRT**: TensorRT 10.x
- **Python**: 3.10+
- **VRAM**: 6 GB minimum

### Python Dependencies
```bash
pip install torch>=2.0.0
pip install transformers>=4.30.0
pip install sentence-transformers>=2.2.0
pip install tensorrt>=10.0.0
pip install onnx>=1.14.0
pip install pycuda>=2025.1
```

---

## Step 1: Export ONNX Model

The first step is to export the sentence transformer model to ONNX format.

```bash
cd semantic-recommender

# Create output directory
mkdir -p models

# Export model (creates ~500MB files)
python scripts/ops/export_model_onnx.py \
  --model-name paraphrase-multilingual-MiniLM-L12-v2 \
  --output models/sentence_transformer.onnx
```

**Expected Output**:
```
✅ Model exported successfully
   ONNX model: models/sentence_transformer.onnx (1.6 MB)
   ONNX data: models/sentence_transformer.onnx.data (448 MB)
```

**What This Does**:
- Downloads the pre-trained model from HuggingFace
- Exports to ONNX format for hardware optimisation
- Creates two files: `.onnx` (graph) and `.onnx.data` (weights)

---

## Step 2: Build TensorRT Engine

Convert the ONNX model to an optimised TensorRT engine.

```bash
# Build TensorRT FP16 engine (takes ~5 minutes)
python scripts/ops/build_trt_engine.py \
  --onnx models/sentence_transformer.onnx \
  --output models/sentence_transformer_fp16_sm86.trt \
  --fp16 \
  --batch-size 32
```

**Expected Output**:
```
🔧 Building TensorRT engine...
   Input: models/sentence_transformer.onnx
   Output: models/sentence_transformer_fp16_sm86.trt
   Precision: FP16
   Max batch size: 32

⏳ This may take several minutes...

✅ TensorRT engine built successfully
   Engine file: models/sentence_transformer_fp16_sm86.trt (226 MB)
   Precision: FP16
   Max batch size: 32
   Compute capability: 8.6 (SM86)
```

**What This Does**:
- Optimizes the ONNX model for your specific GPU
- Applies FP16 quantization (2x speed, 2x memory reduction)
- Creates GPU-specific optimizations
- Sets maximum batch size for concurrent processing

---

## Step 3: Create Symlink for Pipeline

The data pipeline expects the TensorRT engine at a specific path.

```bash
# Create directory
mkdir -p data/models

# Create symlink
ln -sf ../../models/sentence_transformer_fp16_sm86.trt \
       data/models/minilm_l12_v2_fp16.plan

# Verify
ls -lh data/models/minilm_l12_v2_fp16.plan
```

**Expected Output**:
```
lrwxrwxrwx ... data/models/minilm_l12_v2_fp16.plan -> ../../models/sentence_transformer_fp16_sm86.trt
```

---

## Verification

Test that the TensorRT engine works correctly:

```bash
python scripts/ops/test_trt_builder.py
```

**Expected Output**:
```
🧪 Testing TensorRT Engine

✅ Loading engine: models/sentence_transformer_fp16_sm86.trt
✅ Initializing execution context
✅ Encoding test query: "action thriller"
✅ Output shape: (1, 384)
✅ Encoding batch of 5 queries
✅ Output shape: (5, 384)

📊 Performance Test (100 queries):
   Mean latency: 2.3ms
   Throughput: 434 QPS

✅ All tests passed!
```

---

## File Structure

After setup, you should have:

```
semantic-recommender/
├── models/
│   ├── sentence_transformer.onnx           (1.6 MB)
│   ├── sentence_transformer.onnx.data      (448 MB)
│   └── sentence_transformer_fp16_sm86.trt  (226 MB)
├── data/
│   └── models/
│       └── minilm_l12_v2_fp16.plan → ../../models/sentence_transformer_fp16_sm86.trt
```

**Total Disk Usage**: ~676 MB

---

## Troubleshooting

### Issue: "CUDA out of memory"
**Solution**: Reduce batch size
```bash
python scripts/ops/build_trt_engine.py \
  --onnx models/sentence_transformer.onnx \
  --output models/sentence_transformer_fp16_sm86.trt \
  --fp16 \
  --batch-size 16  # Reduced from 32
```

### Issue: "TensorRT not found"
**Solution**: Install TensorRT from NVIDIA
```bash
# Option 1: pip (may have compatibility issues)
pip install tensorrt

# Option 2: Use NVIDIA NGC container (recommended)
docker pull nvcr.io/nvidia/tensorrt:24.10-py3
```

### Issue: "Compute capability mismatch"
**Problem**: Engine built for different GPU architecture

**Solution**: Check your GPU's compute capability
```bash
nvidia-smi --query-gpu=compute_cap --format=csv
```

Then rebuild with correct architecture:
- SM75: RTX 2080, Titan RTX (Turing)
- SM80: A100 (Ampere)
- SM86: RTX 3090, A6000 (Ampere)
- SM89: RTX 4090, L40S (Ada Lovelace)

### Issue: "Engine file not found during pipeline"
**Solution**: Verify symlink
```bash
# Check symlink exists and points to correct file
ls -lh data/models/minilm_l12_v2_fp16.plan

# If broken, recreate
rm data/models/minilm_l12_v2_fp16.plan
ln -sf ../../models/sentence_transformer_fp16_sm86.trt \
       data/models/minilm_l12_v2_fp16.plan
```

---

## Performance Comparison

### PyTorch (Fallback)
- Encoding latency: ~35ms per query
- Throughput: ~28 QPS
- Memory: 2.1 GB VRAM

### TensorRT FP16 (optimised)
- Encoding latency: ~2.3ms per query
- Throughput: ~434 QPS
- Memory: 1.1 GB VRAM

**Speedup**: 15.2x faster, 48% memory reduction

---

## Alternative: Download Pre-built Engine

If you have the same GPU architecture (SM86), you can download a pre-built engine:

**Note**: Pre-built engines are GPU-specific and may not work on different hardware.

### Recommended Approach
Always build the engine on your target GPU for optimal performance and compatibility.

---

## Next Steps

After setup:

1. **Verify installation**:
   ```bash
   python scripts/ops/test_trt_builder.py
   ```

2. **Run TMDB pipeline**:
   ```bash
   cd scripts/data_pipeline
   python run_tmdb_pipeline.py
   ```

3. **Test recommendations**:
   ```bash
   python scripts/test_recommendations.py
   ```

---

## Security Note

- Model files (`*.onnx`, `*.trt`) are excluded from git via `.gitignore`
- Total size: ~676 MB (too large for GitHub without Git LFS)
- Build locally or use external storage for distribution

---

**Setup Time**: ~5 minutes
**Disk Space**: ~676 MB
**GPU Required**: Yes (NVIDIA with TensorRT support)
