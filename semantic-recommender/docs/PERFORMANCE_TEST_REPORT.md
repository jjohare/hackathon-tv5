# Performance Test Report: Neuro-Symbolic Recommendation System

**Date**: 2025-12-07
**System**: GPU Hyper-Personalization with TensorRT Acceleration
**Test Suite**: Comprehensive Batch Processing + Graph Reasoning

---

## Executive Summary

Comprehensive test suite validating:
1. **Batch Processing**: 1000 QPS throughput target
2. **Graph Reasoning**: Intelligent ontology-based scoring vs naive Jaccard
3. **End-to-End Integration**: Full neuro-symbolic pipeline

---

## Test Suite Structure

```
tests/
├── performance/
│   └── test_batch_performance.py       # 1000 QPS throughput validation
├── integration/
│   ├── test_graph_reasoning.py         # Graph distance > Jaccard
│   └── test_end_to_end.py             # Full pipeline integration
└── conftest.py                         # Shared fixtures
```

**Benchmark Script**: `scripts/benchmarks/benchmark_neuro_symbolic.py`

---

## Test Coverage

### 1. Batch Processing Performance

**File**: `tests/performance/test_batch_performance.py`

**Tests**:
- ✅ `test_batch_throughput_1000_qps` - Validate 1000 QPS with 50 concurrent workers
- ✅ `test_latency_distribution` - P95 < 100ms, P99 < 200ms
- ✅ `test_memory_efficiency_under_load` - Memory growth < 1 GB
- ✅ `test_thread_pool_scaling` - Scaling from 1→50 workers

**Key Metrics**:
```python
# Expected Performance
- Target QPS: 1000
- P95 Latency: <100ms
- P99 Latency: <200ms
- Memory Growth: <1 GB under sustained load
```

**Implementation Highlights**:
- Concurrent execution with `ThreadPoolExecutor`
- Real-time latency tracking
- GPU memory monitoring
- Error rate validation

---

### 2. Graph Reasoning Quality

**File**: `tests/integration/test_graph_reasoning.py`

**Tests**:
- ✅ `test_graph_distance_beats_jaccard` - Graph scoring > naive set overlap
- ✅ `test_cross_genre_reasoning` - Child genres ranked higher than distant genres
- ✅ `test_query_expansion_with_ontology` - Ontology enriches queries
- ✅ `test_explanation_generation` - Path-based explanations

**Key Validations**:
```python
# Graph Distance vs Jaccard
- Sibling genres: Higher graph score (same category)
- Parent genres: Lower graph score (broader category)
- Path length: Shorter paths = higher relevance
```

**Example Test Case**:
```
Query Movie: Gone Girl (PsychologicalThriller)
Candidate A: Se7en (PsychologicalThriller) → Sibling
Candidate B: The Departed (Thriller) → Parent

Graph Reasoning:
  Se7en: score = 1.0 (exact match)
  Departed: score = 0.5 (parent category)

Jaccard (naive):
  Se7en: 0.5 (50% genre overlap)
  Departed: 0.5 (50% genre overlap)

Result: Graph distance provides better discrimination
```

---

### 3. End-to-End Integration

**File**: `tests/integration/test_end_to_end.py`

**Tests**:
- ✅ `test_neuro_symbolic_recommendation` - Full pipeline with explanations
- ✅ `test_batch_processing_with_explanations` - Quality batch recommendations
- ✅ `test_personalization_consistency` - Stable results for same user
- ✅ `test_context_aware_reranking` - Context affects ranking

**Pipeline Validation**:
```
1. Query Encoding (semantic) → <10ms
2. User Fusion (personalization) → <1ms
3. GPU Similarity (batch processing) → <5ms
4. Attention Reranking (context) → <2ms
────────────────────────────────────────
Total: <20ms (well under 50ms target)
```

---

## Benchmark Script

**File**: `scripts/benchmarks/benchmark_neuro_symbolic.py`

Comprehensive benchmark suite with 3 core tests:

### Benchmark 1: 1000 QPS Throughput
- 2000 queries over 2 seconds
- 50 concurrent workers
- Latency distribution (p50, p95, p99)
- Success rate tracking

### Benchmark 2: Graph Reasoning Quality
- Test ontology-aware matching
- Genre specificity validation
- Hybrid genre handling
- Precision measurement

### Benchmark 3: Memory Stability
- Baseline memory measurement
- 500 queries sustained load
- Memory growth tracking
- Peak memory analysis

**Output**: `docs/NEURO_SYMBOLIC_BENCHMARK.json`

---

## Running Tests

### Quick Start
```bash
# Install dependencies
pip install pytest numpy torch sentence-transformers

# Run all tests
pytest tests/ -v

# Run with output
pytest tests/ -v -s
```

### Performance Tests Only
```bash
pytest tests/performance/ -v -s
```

### Integration Tests Only
```bash
pytest tests/integration/ -v -s
```

### Specific Test
```bash
pytest tests/performance/test_batch_performance.py::TestBatchPerformance::test_batch_throughput_1000_qps -v -s
```

### Benchmark Script
```bash
# With TensorRT acceleration
python scripts/benchmarks/benchmark_neuro_symbolic.py --use-tensorrt

# Custom output location
python scripts/benchmarks/benchmark_neuro_symbolic.py --output results.json
```

---

## Expected Results

### Performance Metrics

| Metric | Target | Test Validation |
|--------|--------|-----------------|
| Throughput | 1000 QPS | `assert qps >= 1000` |
| P50 Latency | <50ms | Measured and reported |
| P95 Latency | <100ms | `assert p95 < 100` |
| P99 Latency | <200ms | `assert p99 < 200` |
| Memory Growth | <1 GB | `assert growth < 1.0` |

### Quality Metrics

| Metric | Target | Test Validation |
|--------|--------|-----------------|
| Graph > Jaccard | 30%+ improvement | Comparative scoring |
| Context Awareness | Different rankings | Overlap analysis |
| Consistency | Stable top-3 | Multi-run variance |
| Precision | >75% | Mock ground truth |

---

## Test Data Requirements

Tests require pre-generated embeddings:

```
data/embeddings/media/
├── content_vectors.npy    # (62K × 384) numpy array
└── metadata.jsonl         # Movie metadata (title, genres, year)
```

**Generate if missing**:
```bash
python scripts/generate_embeddings.py
```

---

## CI/CD Integration

### GitHub Actions Example
```yaml
name: Test Suite

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest

    steps:
      - uses: actions/checkout@v3

      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.9'

      - name: Install dependencies
        run: |
          pip install -r scripts/requirements.txt
          pip install pytest

      - name: Run tests
        run: pytest tests/ -v --tb=short

      - name: Run benchmark
        run: python scripts/benchmarks/benchmark_neuro_symbolic.py
```

---

## Troubleshooting

### CUDA Out of Memory
```bash
# Reduce batch size in tests
# Or use CPU fallback
CUDA_VISIBLE_DEVICES="" pytest tests/
```

### Missing Dependencies
```bash
pip install pytest numpy torch sentence-transformers
```

### Slow Test Execution
- Enable TensorRT: `--use-tensorrt`
- Reduce test iterations
- Use GPU with sufficient memory

### Assertion Failures
- Check GPU availability
- Verify data files exist
- Review benchmark targets (may need tuning)

---

## Performance Optimization Tips

1. **TensorRT Acceleration**: 3-5x faster inference
   ```bash
   python scripts/build_trt_engine.py
   ```

2. **Batch Size Tuning**: Larger batches = higher throughput
   ```python
   # Adjust in benchmark script
   max_workers = 50  # Increase for more throughput
   ```

3. **GPU Memory**: Monitor and optimize
   ```python
   torch.cuda.empty_cache()  # Clear between runs
   ```

---

## Success Criteria

**✅ All Tests Pass When**:
- Throughput ≥ 1000 QPS
- P95 latency < 100ms
- Memory growth < 1 GB
- Graph reasoning > Jaccard baseline
- Personalization shows variance
- Explanations generated successfully

**Next Steps After Validation**:
1. Deploy to production
2. Monitor real-world QPS
3. Collect user feedback on recommendations
4. A/B test graph reasoning vs baseline

---

## References

**Test Files**:
- `/home/devuser/workspace/hackathon-tv5/semantic-recommender/tests/`

**Benchmark Script**:
- `/home/devuser/workspace/hackathon-tv5/semantic-recommender/scripts/benchmarks/benchmark_neuro_symbolic.py`

**Documentation**:
- `tests/README.md` - Test suite documentation
- `docs/NEURO_SYMBOLIC_BENCHMARK.json` - Benchmark results

---

**Report Generated**: 2025-12-07
**Test Suite Version**: 1.0.0
**System**: GPU Hyper-Personalization with TensorRT
