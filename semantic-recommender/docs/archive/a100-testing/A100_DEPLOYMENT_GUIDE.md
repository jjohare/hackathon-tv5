# A100 GPU Deployment and Testing Guide

**Date:** 2025-12-06
**Version:** 1.0
**System:** Semantic Recommender with GPU Acceleration

---

## Prerequisites

### Local System
- Package ready: `/tmp/semantic-recommender-deploy.tar.gz` (422 MB)
- Contents:
  - All Python scripts (14 files)
  - Media embeddings (92 MB + 4.3 MB metadata)
  - User embeddings (351 MB + 2 MB IDs)
  - Comprehensive test suite

### A100 VM
- Instance: `semantics-testbed-a100`
- Zone: `us-central1-a`
- GPU: NVIDIA A100-SXM4-40GB (42 GB memory)
- CUDA: 12.8
- OS: Debian/Ubuntu with Python 3.11

---

## Deployment Steps

### Step 1: Transfer Package

```bash
# From local machine (with gcloud configured)
gcloud compute scp /tmp/semantic-recommender-deploy.tar.gz \
  semantics-testbed-a100:/home/devuser/ \
  --zone us-central1-a
```

**Expected:**
- Transfer time: ~2-3 minutes (422 MB)
- Destination: `/home/devuser/semantic-recommender-deploy.tar.gz`

### Step 2: SSH to A100 VM

```bash
gcloud compute ssh semantics-testbed-a100 --zone us-central1-a
```

### Step 3: Extract and Setup

```bash
# Extract package
cd /home/devuser
tar -xzf semantic-recommender-deploy.tar.gz

# Create directory structure
mkdir -p semantic-recommender
cd semantic-recommender
mv ../scripts ./
mv ../data ./

# Verify extraction
ls -lh data/embeddings/media/
ls -lh data/embeddings/users/
ls -lh scripts/
```

**Expected:**
```
data/embeddings/media/content_vectors.npy (92 MB)
data/embeddings/media/metadata.jsonl (4.3 MB)
data/embeddings/users/preference_vectors.npy (351 MB)
data/embeddings/users/user_ids.json (2 MB)
scripts/test_a100_comprehensive.py (13 KB)
scripts/gpu_recommend.py (12 KB)
scripts/run_recommendations.py (12 KB)
```

### Step 4: Install Dependencies

```bash
# Install pip (if needed)
curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
python3 get-pip.py

# Install PyTorch with CUDA 12.1 support
pip install torch --index-url https://download.pytorch.org/whl/cu121

# Install other dependencies
pip install numpy

# Verify PyTorch GPU
python3 -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"
```

**Expected Output:**
```
PyTorch: 2.9.1+cu121
CUDA: True
GPU: NVIDIA A100-SXM4-40GB
```

---

## Running Tests

### Test Suite Overview

The comprehensive test suite (`test_a100_comprehensive.py`) includes:

1. **Test 1:** Single movie similarity (franchise detection)
2. **Test 2:** User personalization (5 users)
3. **Test 3:** Batch processing (10, 100, 1000 queries)
4. **Test 4:** Genre-filtered recommendations
5. **Test 5:** GPU memory analysis

### Execute Test Suite

```bash
cd /home/devuser/semantic-recommender
python3 scripts/test_a100_comprehensive.py
```

**Expected Runtime:** 2-5 minutes

### Test Details

#### Test 1: Single Movie Similarity
**Purpose:** Validate semantic understanding and franchise detection

**Example:**
```
Query: Toy Story (1995)
Expected Top Results:
  1. Toy Story 2 (1999)           - 94.0% similar
  2. Toy Story 3 (2010)           - 91.0% similar
  3. Toy Story 4 (2019)           - 90.2% similar
  4. Antz (1998)                  - 88.4% similar
  5. Monsters, Inc. (2001)        - 87.5% similar
```

**Success Criteria:**
- ✅ Franchise films ranked at top
- ✅ GPU time < 1ms
- ✅ Similarity scores > 90%

#### Test 2: User Personalization
**Purpose:** Validate user taste modeling

**Example:**
```
User: user_00000001
Expected: Drama, international cinema, arthouse films
  1. 2046 (2004)                  - 88.4%
  2. Nostalghia (1983)            - 87.7%
  3. Turin Horse, The (2011)      - 87.2%
```

**Success Criteria:**
- ✅ Genre alignment with user history
- ✅ Diverse time periods
- ✅ GPU time < 2ms per user

#### Test 3: Batch Processing
**Purpose:** Measure throughput at scale

**Expected Performance:**

| Batch Size | Total Time | Time/Query | Throughput |
|-----------|-----------|-----------|-----------|
| 10 | ~5 ms | 0.5 ms | 2,000 QPS |
| 100 | ~30 ms | 0.3 ms | 3,333 QPS |
| 1000 | ~200 ms | 0.2 ms | 5,000 QPS |

**Success Criteria:**
- ✅ Throughput > 1,000 QPS for batch=100
- ✅ Linear scaling up to batch=1000
- ✅ No OOM errors

#### Test 4: Genre Filtering
**Purpose:** Complex multi-constraint queries

**Example:**
```
Query: Star Wars (Sci-Fi)
Filter: Sci-Fi genre only
Expected: Star Trek, Alien, Blade Runner, etc.
```

**Success Criteria:**
- ✅ All results match filter constraint
- ✅ Semantic relevance maintained
- ✅ GPU time < 1ms (filtering done CPU-side)

#### Test 5: Memory Analysis
**Purpose:** GPU memory efficiency validation

**Expected:**
```
Memory Usage:
  Allocated: ~0.5 GB (embeddings)
  Reserved:  ~1.0 GB (PyTorch overhead)
  Total:     42.0 GB
  Free:      ~41 GB (97% available)

Peak Memory (batch=100):
  Peak Allocated: ~1.5 GB
  Peak Reserved:  ~2.0 GB
```

**Success Criteria:**
- ✅ Base usage < 2 GB
- ✅ Peak usage < 5 GB
- ✅ Headroom > 90%

---

## Expected Results Summary

### Performance Targets

| Metric | CPU Baseline | GPU Target | GPU Achieved |
|--------|-------------|-----------|--------------|
| **Single Query** | 27 ms | <1 ms | TBD |
| **User Rec** | 76 ms | <2 ms | TBD |
| **Batch 100** | 2.6 sec | <30 ms | TBD |
| **Throughput** | 38 QPS | >1,000 QPS | TBD |
| **Memory** | 8 GB RAM | <2 GB VRAM | TBD |

### Quality Targets

| Test | Target | Status |
|------|--------|--------|
| Franchise Detection | Top-3 same franchise | TBD |
| Genre Alignment | 80%+ genre match | TBD |
| Semantic Coherence | 85%+ similarity | TBD |
| Filter Accuracy | 100% constraint match | TBD |

---

## Troubleshooting

### Issue: PyTorch Not Found

```bash
# Reinstall with correct CUDA version
pip uninstall torch
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

### Issue: CUDA Out of Memory

**Symptoms:** `RuntimeError: CUDA out of memory`

**Solutions:**
1. Reduce batch size in tests
2. Clear GPU cache:
   ```python
   torch.cuda.empty_cache()
   ```
3. Check for memory leaks:
   ```python
   torch.cuda.reset_peak_memory_stats()
   ```

### Issue: Slow Performance

**Check:**
1. GPU actually being used:
   ```python
   print(torch.cuda.is_available())
   ```
2. Tensors on GPU:
   ```python
   print(embeddings.device)  # Should be "cuda:0"
   ```
3. CUDA synchronization:
   ```python
   torch.cuda.synchronize()  # Before timing measurements
   ```

### Issue: Incorrect Results

**Verify:**
1. Embedding normalization:
   ```python
   norms = torch.norm(embeddings, dim=1)
   print(norms.mean(), norms.std())  # Should be 1.0, ~0.0
   ```
2. Metadata loading:
   ```bash
   wc -l data/embeddings/media/metadata.jsonl  # Should match vector count
   ```

---

## Performance Profiling

### Detailed GPU Profiling

```python
import torch.profiler as profiler

with profiler.profile(
    activities=[
        profiler.ProfilerActivity.CPU,
        profiler.ProfilerActivity.CUDA,
    ],
    record_shapes=True,
    with_stack=True
) as prof:
    # Run recommendation
    similarities = torch.matmul(query, embeddings.T)
    top_k = torch.topk(similarities, k=10)

print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
```

### CUDA Event Timing

```python
start_event = torch.cuda.Event(enable_timing=True)
end_event = torch.cuda.Event(enable_timing=True)

start_event.record()

# Your GPU operation
similarities = torch.matmul(query, embeddings.T)

end_event.record()
torch.cuda.synchronize()

elapsed_ms = start_event.elapsed_time(end_event)
print(f"GPU Time: {elapsed_ms:.3f} ms")
```

---

## Benchmark Output Format

Results saved to: `results/a100_test_results.json`

**Structure:**
```json
{
  "test_1_similarity": {
    "gpu_time_ms": 0.523,
    "results": [
      {"rank": 1, "title": "Toy Story 2 (1999)", "similarity": 0.94},
      ...
    ]
  },
  "test_2_personalization": {
    "avg_gpu_time_ms": 1.234,
    "user_results": [...]
  },
  "test_3_batch": {
    "batch_10": {
      "total_time_ms": 5.2,
      "time_per_query_ms": 0.52,
      "throughput_qps": 1923.1
    },
    "batch_100": {...},
    "batch_1000": {...}
  },
  "test_4_filtering": {...},
  "test_5_memory": {
    "allocated_gb": 0.52,
    "reserved_gb": 1.02,
    "total_gb": 42.0,
    "free_gb": 40.98,
    "peak_allocated_gb": 1.48,
    "peak_reserved_gb": 2.01
  }
}
```

---

## Alternative: Run Individual Tests

If you want to run specific tests instead of the full suite:

### Test 1: Basic GPU Recommendation

```bash
python3 scripts/gpu_recommend.py
```

### Test 2: CPU Baseline (for comparison)

```bash
python3 scripts/run_recommendations.py
```

### Test 3: Embedding Benchmark

```bash
python3 scripts/benchmark_a100.py
```

---

## Next Steps After Testing

### 1. Document Results

Create `docs/A100_TEST_RESULTS.md` with:
- Performance metrics (actual vs. targets)
- Quality validation (franchise detection, genre alignment)
- Memory efficiency analysis
- Comparison vs. CPU baseline

### 2. Generate Performance Report

```bash
# After tests complete
cat results/a100_test_results.json | python3 -m json.tool > docs/a100_results_formatted.json
```

### 3. Push to GitHub

```bash
git add results/a100_test_results.json
git add docs/A100_TEST_RESULTS.md
git commit -m "test: A100 GPU comprehensive test results

- Single query: X.XX ms (XXx faster than CPU)
- Batch 100: XX.X ms (XXx faster)
- Throughput: X,XXX QPS (XXx improvement)
- Memory: X.XX GB used (XX% of available)
- Quality: All tests passed"

git push
```

---

## Advanced: Custom CUDA Kernels

For even better performance, integrate custom CUDA kernels:

### Compile Custom Kernels

```bash
cd /home/devuser/semantic-recommender
mkdir -p cuda/bin

# Compile tensor core kernel
nvcc -O3 -use_fast_math -arch=sm_80 \
  -o cuda/bin/semantic_similarity_fp16_tensor_cores.ptx \
  --ptx src/cuda/kernels/semantic_similarity_fp16_tensor_cores.cu
```

**Expected Improvement:**
- Single query: 0.5 ms → 0.1 ms (5x faster)
- Batch 100: 30 ms → 5 ms (6x faster)
- Throughput: 3,333 QPS → 20,000 QPS (6x improvement)

---

## Cost Analysis

### GCP A100 Pricing

- **On-Demand:** ~$3.67/hour
- **Preemptible:** ~$1.10/hour
- **Committed Use (1 year):** ~$2.20/hour

### Testing Cost Estimate

- Deployment + Setup: 15 minutes = $0.92
- Test Suite Execution: 5 minutes = $0.31
- Total: ~$1.23 per test run

### Production Cost (Monthly)

**Scenario: 100M queries/month**

| Configuration | QPS | Cost/Month | Cost/Query |
|--------------|-----|-----------|-----------|
| CPU (16 cores) | 608 | $1,408 | $0.000014 |
| 1x A100 | 10,000 | $1,584 | $0.000016 |
| 4x A100 | 40,000 | $6,336 | $0.000063 |

**ROI:** A100 pays for itself at >50M queries/month due to higher throughput.

---

## Reference

- **Ontology Integration:** See `docs/ONTOLOGY_INTEGRATION_PLAN.md`
- **System Status:** See `docs/SYSTEM_STATUS.md`
- **Production Deployment:** See `docs/PRODUCTION_DEPLOYMENT_PLAN.md`
- **Previous Benchmark:** See `docs/A100_GPU_BENCHMARK_REPORT.md` (embedding generation)

---

**Document Version:** 1.0
**Author:** Semantic Recommender Team
**Date:** 2025-12-06
**Status:** Deployment Ready
