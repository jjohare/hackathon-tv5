# A100 GPU Testing - Quick Start

**Ready for deployment and comprehensive testing on GCP A100 GPU**

---

## 🚀 Quick Deploy and Test

### Option 1: Automated Deployment (Recommended)

```bash
# Single command deployment and testing
cd /home/devuser/workspace/hackathon-tv5/semantic-recommender
./scripts/deploy_and_test_a100.sh
```

**What it does:**
1. ✅ Transfers 422 MB package to A100 VM
2. ✅ Installs PyTorch with CUDA 12.1
3. ✅ Runs 5 comprehensive tests
4. ✅ Downloads results locally
5. ✅ Generates JSON report

**Expected runtime:** 5-10 minutes

---

### Option 2: Manual Deployment

#### Step 1: Package is Ready

```bash
ls -lh /tmp/semantic-recommender-deploy.tar.gz
# Output: 422M (scripts + embeddings)
```

#### Step 2: Transfer to A100

```bash
gcloud compute scp /tmp/semantic-recommender-deploy.tar.gz \
  semantics-testbed-a100:/home/devuser/ \
  --zone us-central1-a
```

#### Step 3: SSH and Setup

```bash
gcloud compute ssh semantics-testbed-a100 --zone us-central1-a

# On A100 VM:
cd /home/devuser
tar -xzf semantic-recommender-deploy.tar.gz
cd semantic-recommender
pip install --user torch --index-url https://download.pytorch.org/whl/cu121
pip install --user numpy
```

#### Step 4: Run Tests

```bash
python3 scripts/test_a100_comprehensive.py
```

#### Step 5: Download Results

```bash
# From local machine:
gcloud compute scp semantics-testbed-a100:/home/devuser/semantic-recommender/results/a100_test_results.json ./results/ --zone us-central1-a
```

---

## 📊 What Gets Tested

### Test 1: Single Movie Similarity
**Query:** Toy Story (1995)
**Expected:** Toy Story 2 at 94% similarity
**Performance Target:** <1ms on GPU (vs 27ms CPU)

### Test 2: User Personalization
**Users Tested:** 5 different profiles
**Expected:** Genre-aligned recommendations
**Performance Target:** <2ms per user (vs 81ms CPU)

### Test 3: Batch Processing
**Batch Sizes:** 10, 100, 1000
**Performance Targets:**
- Batch 10: ~5ms (2,000 QPS)
- Batch 100: ~30ms (3,333 QPS)
- Batch 1000: ~200ms (5,000 QPS)

### Test 4: Genre Filtering
**Query:** Sci-Fi movie
**Filter:** Only Sci-Fi results
**Expected:** 100% constraint compliance

### Test 5: Memory Analysis
**Expected Usage:** 1-2 GB / 42 GB (5%)
**Peak Usage:** <5 GB
**Headroom:** 95%+

---

## 📈 Expected Performance

| Metric | CPU Baseline | A100 Target | Speedup |
|--------|-------------|------------|---------|
| Single Query | 27 ms | 0.5 ms | **54x** |
| User Rec | 81 ms | 1.5 ms | **54x** |
| Batch 100 | 2,730 ms | 30 ms | **91x** |
| Throughput | 37 QPS | 10,000 QPS | **270x** |

See `docs/EXPECTED_A100_RESULTS.md` for detailed predictions.

---

## 📁 Results Location

After testing:

**On A100 VM:**
- `/home/devuser/semantic-recommender/results/a100_test_results.json`
- `/home/devuser/semantic-recommender/results/test_output.log`

**Downloaded Locally:**
- `./results/a100_test_results.json`
- `./results/test_output.log`

---

## 🔍 Validate Results

```bash
# View JSON results
cat results/a100_test_results.json | python3 -m json.tool

# Key metrics to check:
# - test_1_similarity.gpu_time_ms < 1.0
# - test_3_batch.batch_100.throughput_qps > 1000
# - test_5_memory.allocated_gb < 2.0
```

---

## 📚 Documentation

- **Deployment Guide:** `docs/A100_DEPLOYMENT_GUIDE.md`
- **Expected Results:** `docs/EXPECTED_A100_RESULTS.md`
- **Ontology Integration:** `docs/ONTOLOGY_INTEGRATION_PLAN.md`
- **System Status:** `docs/SYSTEM_STATUS.md`

---

## ✅ Success Criteria

### Performance
- [ ] Single query < 1ms
- [ ] Batch 100 throughput > 1,000 QPS
- [ ] Memory usage < 5 GB
- [ ] No CUDA errors

### Quality
- [ ] Franchise detection: Top-3 same series
- [ ] Genre alignment: 80%+ match
- [ ] User personalization working
- [ ] Filter constraints: 100% compliance

### Scalability
- [ ] Batch scaling linear up to 100
- [ ] Memory headroom > 90%
- [ ] No OOM errors
- [ ] Stable performance across runs

---

## 🐛 Troubleshooting

**PyTorch not found:**
```bash
pip install --user torch --index-url https://download.pytorch.org/whl/cu121
```

**CUDA out of memory:**
- Check GPU usage: `nvidia-smi`
- Clear cache: `python3 -c "import torch; torch.cuda.empty_cache()"`

**Slow performance:**
- Verify GPU usage: `print(torch.cuda.is_available())`
- Check tensor device: `print(embeddings.device)`

---

## 🎯 Next Steps

1. **Run Tests:** Execute deployment script
2. **Validate Results:** Compare actual vs predicted
3. **Document Findings:** Create `A100_TEST_RESULTS.md`
4. **Optimize:** Integrate custom CUDA kernels (6x improvement)
5. **Scale:** Multi-GPU deployment
6. **Integrate:** Add ontology reasoning (see ONTOLOGY_INTEGRATION_PLAN.md)

---

**Status:** 🟢 Ready for Testing
**Package:** ✅ Built (422 MB)
**Scripts:** ✅ Complete
**Documentation:** ✅ Comprehensive
**GPU:** ⏳ Waiting for deployment

**Execute:** `./scripts/deploy_and_test_a100.sh`
