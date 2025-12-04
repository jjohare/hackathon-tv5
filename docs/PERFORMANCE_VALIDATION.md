# Performance Validation Infrastructure

Complete performance testing system for validating the hybrid Milvus + Neo4j + PostgreSQL architecture on 100x T4 GPUs.

## Quick Reference

### Run Load Test
```bash
./scripts/load-test.sh
```

### Run Benchmarks
```bash
cargo bench
```

### Compare Architectures
```bash
./scripts/generate-comparison.sh
```

### Analyze Results
```bash
python3 scripts/analyze-results.py results/load-test-*.json
```

## Performance Targets

| Metric | Target | Stretch | Must-Pass |
|--------|--------|---------|-----------|
| p99 Latency | < 10ms | < 8.7ms | < 15ms |
| Sustained QPS | ≥ 7,000 | ≥ 10,000 | ≥ 5,000 |
| Error Rate | < 1% | < 0.1% | < 2% |
| GPU Utilization | > 60% | > 75% | > 40% |
| Speedup vs Neo4j | 2x | 3x | 1.5x |

## Components

### 1. Load Generator (`src/bin/load-generator.rs`)
Generates realistic load with configurable QPS, duration, and query profiles.

**Features:**
- Precise rate limiting per worker
- Realistic query distribution (simple, hybrid, complex)
- Per-request latency tracking
- Success/failure logging
- Progress reporting

**Usage:**
```bash
cargo run --release --bin load-generator -- \
  --qps 7000 \
  --duration 600 \
  --workers 100 \
  --profile mixed \
  --output results/test.json
```

### 2. Benchmark Suite (`benches/performance_validation.rs`)
Criterion.rs-based micro-benchmarks for component isolation.

**Benchmarks:**
1. `vector_search_scaling` - GPU scaling (1-100 GPUs)
2. `dataset_scaling` - Data size impact (100K-10M vectors)
3. `hybrid_vs_neo4j` - Direct architecture comparison
4. `query_complexity` - k-NN parameter scaling
5. `concurrent_throughput` - Parallel query handling
6. `graph_depth` - Graph traversal performance
7. `p99_latency_validation` - High-sample p99 measurement

**Usage:**
```bash
# Run all benchmarks
cargo bench

# Specific benchmark
cargo bench -- vector_search_scaling

# Compare baselines
STORAGE_MODE=hybrid cargo bench -- --save-baseline hybrid
STORAGE_MODE=neo4j_only cargo bench -- --save-baseline neo4j
```

### 3. Analysis Tools (`scripts/analyze-results.py`)
Python-based analysis with matplotlib visualizations.

**Outputs:**
- Latency distribution histogram
- Latency over time series
- Cumulative distribution function (CDF)
- Latency by query type box plots
- QPS over time tracking
- Error rate monitoring
- Markdown performance report

**Requirements:**
```bash
pip install numpy matplotlib
```

### 4. Grafana Dashboard (`grafana/performance-dashboard.json`)
Real-time monitoring with 8 panels:
1. Queries Per Second (with alerts)
2. Latency Percentiles (p50, p95, p99)
3. GPU Utilization Heatmap (100x T4)
4. Error Rate (with alerts)
5. Resource Usage (CPU, RAM, Network)
6. Query Type Breakdown (pie chart)
7. Connection Pool Health
8. Cache Hit Rate

## Test Scenarios

### Scenario 1: Vector Search Only
**Objective:** Milvus GPU acceleration baseline
```bash
cargo bench -- vector_search_scaling
```

### Scenario 2: Hybrid Search
**Objective:** Cross-system coordination overhead
```bash
./scripts/load-test.sh
```

### Scenario 3: Policy-Based Ranking
**Objective:** Full stack validation
```bash
TARGET_QPS=5000 PROFILE=complex ./scripts/load-test.sh
```

### Scenario 4: Sustained Load
**Objective:** 10-minute stability test
```bash
DURATION_SECONDS=600 ./scripts/load-test.sh
```

### Scenario 5: Burst Load
**Objective:** Spike handling
```bash
# Script cycles: 7K → 15K → 7K QPS
./scripts/burst-test.sh
```

### Scenario 6: Baseline Comparison
**Objective:** Neo4j vs Hybrid
```bash
./scripts/generate-comparison.sh
```

## Expected Results

### Hybrid Architecture Benefits
- **Vector Search:** 3-5x faster (GPU acceleration)
- **Sustained QPS:** 2-3x higher (parallel specialization)
- **GPU Utilization:** 4-6x better (dedicated vector ops)
- **Coordination Overhead:** < 2ms per query

### Sample Output
```
=== Load Test Results ===
Total Requests:     4,200,000
Successful:         4,194,900
Actual QPS:         7,001.5

Latency Distribution:
  p50:    3.24ms
  p95:    6.78ms
  p99:    8.92ms
  max:   12.45ms

=== Target Validation ===
✓ p99 latency: 8.92ms (target: <10ms)
✓ Sustained QPS: 7,001 (target: ≥7,000)
✓ Error rate: 0.12% (target: <1%)

🎉 ALL TARGETS MET!
```

## Metrics Collection

### Prometheus Setup
```bash
./scripts/start-prometheus.sh
```

**Scrape Targets:**
- Application: `localhost:9090`
- Node Exporter: `localhost:9100`
- DCGM GPU: `localhost:9400`
- Milvus: `localhost:9091`
- Neo4j: `localhost:2004`

### Grafana Import
1. Access: `http://localhost:3000`
2. Import: `grafana/performance-dashboard.json`
3. Select Prometheus data source
4. View real-time metrics

## Troubleshooting

### QPS Below Target
**Symptoms:** Actual QPS < 7,000
**Checks:**
- GPU utilization (should be > 60%)
- Connection pool size (increase workers)
- Network latency (check cross-system calls)
- Query distribution (validate mix)

**Actions:**
```bash
# Check GPU usage
nvidia-smi dmon -s u

# Profile queries
cargo flamegraph --bin load-generator

# Increase parallelism
WORKERS=200 ./scripts/load-test.sh
```

### High Latency
**Symptoms:** p99 > 10ms
**Checks:**
- Individual component latency (Milvus, Neo4j, AgentDB)
- Query plan optimization
- Cache hit rates
- Resource contention

**Actions:**
```bash
# Profile components
cargo bench -- p99_latency_validation --profile-time=60

# Check cache
curl http://localhost:9090/metrics | grep cache_hit_rate

# Optimize queries
./scripts/analyze-query-plans.sh
```

### High Error Rate
**Symptoms:** Errors > 1%
**Checks:**
- Connection pool exhaustion
- Timeout configurations
- Error types (network, timeout, server)
- Resource limits (memory, FDs)

**Actions:**
```bash
# Check errors
tail -f logs/coordinator.log | grep ERROR

# Increase timeouts
export QUERY_TIMEOUT_MS=5000

# Scale connection pool
export MAX_CONNECTIONS=500
```

## File Organization

```
hackathon-tv5/
├── docs/
│   ├── performance-test-plan.md       # Detailed test strategy
│   └── PERFORMANCE_VALIDATION.md      # This file
├── benches/
│   └── performance_validation.rs      # Criterion benchmarks
├── src/
│   └── bin/
│       └── load-generator.rs          # Load testing tool
├── scripts/
│   ├── load-test.sh                   # Main load test script
│   ├── analyze-results.py             # Python analysis
│   ├── generate-comparison.sh         # Baseline comparison
│   └── start-prometheus.sh            # Metrics collection
├── grafana/
│   └── performance-dashboard.json     # Real-time dashboard
├── results/
│   ├── load-test-*.json               # Test outputs
│   ├── latency-analysis.png           # Visualizations
│   └── performance-report.md          # Generated reports
└── config/
    └── prometheus.yml                 # Metrics config
```

## CI/CD Integration

### GitHub Actions
```yaml
name: Performance Validation
on: [push, pull_request]

jobs:
  benchmark:
    runs-on: [self-hosted, gpu]
    steps:
      - uses: actions/checkout@v4
      - uses: dtolnay/rust-toolchain@stable

      - name: Run benchmarks
        run: cargo bench

      - name: Load test
        run: DURATION_SECONDS=60 ./scripts/load-test.sh

      - name: Compare baselines
        run: ./scripts/generate-comparison.sh

      - name: Upload results
        uses: actions/upload-artifact@v4
        with:
          name: performance-results
          path: results/
```

## Production Deployment

### Pre-Deployment Checklist
- [ ] All targets met in load tests
- [ ] Baseline comparison shows ≥1.5x speedup
- [ ] No memory leaks detected (10+ minute tests)
- [ ] Error rate < 0.1% under sustained load
- [ ] Burst load handled gracefully
- [ ] Grafana dashboards configured
- [ ] Alerting rules defined
- [ ] Runbooks documented

### Monitoring Setup
1. Deploy Prometheus with 15-day retention
2. Configure Grafana with performance dashboard
3. Set up alerts for:
   - QPS < 7,000 for 5 minutes
   - p99 latency > 10ms for 5 minutes
   - Error rate > 1% for 1 minute
   - GPU utilization < 40% for 10 minutes

### Capacity Planning
- Current: 7,000 QPS @ 100x T4 GPUs
- Scaling: Linear up to 200 GPUs (~14K QPS)
- Buffer: Target 70% capacity in production
- Production target: ~5,000 QPS (30% headroom)

## References

- [Test Plan](performance-test-plan.md) - Comprehensive testing strategy
- [Grafana Dashboard](../grafana/performance-dashboard.json) - Real-time monitoring
- [Load Generator](../src/bin/load-generator.rs) - Source code
- [Benchmarks](../benches/performance_validation.rs) - Micro-benchmarks
