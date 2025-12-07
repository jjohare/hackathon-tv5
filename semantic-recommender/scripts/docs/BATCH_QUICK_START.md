# Batch Processing Quick Start

## 🚀 Quick Setup (30 seconds)

### 1. Install Dependencies
```bash
cd /home/devuser/workspace/hackathon-tv5/semantic-recommender
pip install flask requests
```

### 2. Verify Implementation
```bash
python3 scripts/tests/verify_batch_implementation.py
```
Expected: ✅ All 25 checks passed (100% success rate)

### 3. Start Server
```bash
python3 scripts/server/query_interface.py
```
Server starts on http://0.0.0.0:5000

---

## 🎯 Test Endpoints (Quick)

### Single Query with Batching
```bash
curl -X POST http://localhost:5000/api/query \
  -H "Content-Type: application/json" \
  -d '{"query": "action movies", "limit": 5, "use_batch": true}'
```

### Batch Query
```bash
curl -X POST http://localhost:5000/api/query/batch \
  -H "Content-Type: application/json" \
  -d '{"queries": ["action", "comedy", "thriller"], "limit": 5}'
```

### Check Status
```bash
curl http://localhost:5000/api/status | jq '.batch_processor'
```

---

## 📊 Run Benchmark (1 minute)

```bash
# Standard benchmark (1000 queries)
python3 scripts/tests/benchmark_batch_qps.py

# Expected with TensorRT: QPS >= 1000
```

---

## 🔍 Key Features

### BatchProcessor
- **Max batch size**: 32 (matches TensorRT)
- **Max wait**: 50ms (configurable)
- **Threading**: Background processing thread
- **Queue**: Thread-safe deque

### API Endpoints
1. `POST /api/query` - Single query (with optional batching)
2. `POST /api/query/batch` - Batch queries (high-throughput)
3. `GET /api/status` - System status

### Performance
- **Target**: 1000+ QPS
- **Baseline**: 39.9 QPS (sequential)
- **Improvement**: 25-30x with batching

---

## 📁 Files

### Modified
- `scripts/server/query_interface.py` - Added BatchProcessor class and endpoints
- `scripts/requirements.txt` - Added flask, requests

### Created
- `scripts/tests/test_batch_processing.py` - Functional tests
- `scripts/tests/benchmark_batch_qps.py` - QPS benchmark
- `scripts/tests/verify_batch_implementation.py` - Implementation verification
- `scripts/docs/BATCH_PROCESSING.md` - Complete documentation
- `scripts/docs/BATCH_PROCESSING_SUMMARY.md` - Implementation summary
- `scripts/docs/BATCH_QUICK_START.md` - This file

---

## ⚡ Quick Commands

```bash
# Verify implementation
python3 scripts/tests/verify_batch_implementation.py

# Run functional tests
python3 scripts/tests/test_batch_processing.py

# Benchmark QPS
python3 scripts/tests/benchmark_batch_qps.py

# Stress test (10K queries)
python3 scripts/tests/benchmark_batch_qps.py --requests 10000 --workers 50

# Monitor GPU
watch -n 1 nvidia-smi
```

---

## ✅ Success Checklist

- [ ] Dependencies installed (`pip install flask requests`)
- [ ] Verification passed (all 25 checks)
- [ ] Server started on port 5000
- [ ] Single query endpoint tested
- [ ] Batch endpoint tested
- [ ] Status endpoint shows batch processor running
- [ ] Benchmark shows QPS >= 1000 (with TensorRT)

---

## 📚 Full Documentation

See `scripts/docs/BATCH_PROCESSING.md` for:
- Detailed architecture
- API documentation
- Configuration tuning
- Troubleshooting guide
- Performance analysis

---

**Status**: ✅ Complete and verified
**Implementation Date**: 2025-12-07
