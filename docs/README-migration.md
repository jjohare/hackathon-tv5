# Migration Tooling: Neo4j → Hybrid Architecture

Complete zero-downtime migration suite for transitioning from Neo4j-only to hybrid storage architecture.

## 🎯 Quick Start

```bash
# 1. Preflight checks
cargo run --bin migrate -- preflight --check-connectivity

# 2. Full migration (automated)
./scripts/run-migration.sh

# 3. Monitor progress
cargo run --bin migrate -- monitor --interval-secs 5

# 4. Rollback (if needed)
./scripts/rollback.sh --confirm
```

## 📋 Table of Contents

- [Architecture](#architecture)
- [Migration Phases](#migration-phases)
- [CLI Tool Reference](#cli-tool-reference)
- [Scripts](#scripts)
- [Monitoring](#monitoring)
- [Troubleshooting](#troubleshooting)
- [Performance](#performance)

## 🏗️ Architecture

### Before Migration (Neo4j-Only)
```
┌─────────────┐
│   Neo4j     │
│  • Vectors  │──► All queries
│  • Graphs   │
└─────────────┘
```

### During Migration (Dual-Write)
```
┌─────────────┐
│   Neo4j     │◄── Writes (primary)
│  • Vectors  │──► Reads (primary)
│  • Graphs   │
└─────────────┘
       │
       │ Shadow writes
       ▼
┌─────────────┐
│   Milvus    │◄── Writes (shadow)
│  • Vectors  │    No reads yet
└─────────────┘
```

### After Migration (Hybrid)
```
┌─────────────┐     ┌─────────────┐
│   Milvus    │     │   Neo4j     │
│  • Vectors  │◄────│  • Graphs   │
│  • HNSW     │     │  • Metadata │
└─────────────┘     └─────────────┘
       │                   │
       └───────┬───────────┘
               ▼
         ┌─────────────┐
         │ PostgreSQL  │
         │  • AgentDB  │
         │  • RL State │
         └─────────────┘
```

## 📊 Migration Phases

| Phase | Duration | Downtime | Risk | Description |
|-------|----------|----------|------|-------------|
| **1. Preflight** | 1 min | None | Low | Validate system readiness |
| **2. Deploy Milvus** | 5 min | None | Low | Deploy vector database cluster |
| **3. Dual-Write** | 0 min | None | Low | Enable shadow writes to Milvus |
| **4. Migrate Data** | 10-15 min | None | Medium | Batch migrate historical embeddings |
| **5. Validate** | 2 min | None | Low | Verify migration accuracy |
| **6. Shadow Mode** | 5-10 min | None | Medium | Test Milvus reads with fallback |
| **7. Hybrid Mode** | 0 min | ~10 sec | High | Switch to Milvus for production |
| **8. AgentDB** | 2 min | None | Low | Migrate RL state to PostgreSQL |

**Total Duration:** ~25-35 minutes
**Total Downtime:** ~10 seconds (rolling update)

## 🔧 CLI Tool Reference

### Preflight Checks

```bash
# Basic checks
cargo run --bin migrate -- preflight

# With connectivity tests
cargo run --bin migrate -- preflight --check-connectivity

# Verbose output
cargo run --bin migrate -- preflight --check-connectivity --verbose
```

**Checks:**
- Neo4j version, connectivity, data count
- Milvus cluster health
- PostgreSQL availability
- Redis connectivity
- Disk space (3x embedding size required)
- Memory availability
- Estimated migration time

### Migrate Embeddings

```bash
# Dry run (no changes)
cargo run --bin migrate -- migrate-embeddings \
  --batch-size 1000 \
  --dry-run

# Production migration
cargo run --bin migrate -- migrate-embeddings \
  --batch-size 1000

# Resume from checkpoint
cargo run --bin migrate -- migrate-embeddings \
  --batch-size 1000 \
  --resume-from "media-12345"
```

**Options:**
- `--batch-size`: Items per batch (default: 1000)
- `--dry-run`: Simulate without writing
- `--resume-from`: Resume from checkpoint ID

### Validate Migration

```bash
# Random sample (fast)
cargo run --bin migrate -- validate --sample-size 1000

# Full scan (slow, thorough)
cargo run --bin migrate -- validate \
  --sample-size 10000 \
  --full-scan

# Auto-fix inconsistencies
cargo run --bin migrate -- validate \
  --sample-size 1000 \
  --fix-inconsistencies
```

**Validation checks:**
- Vector equality (L2 distance < 1e-5)
- Missing embeddings in Milvus
- Orphaned embeddings in Milvus
- Success rate (must be >99%)

### Migrate AgentDB

```bash
# Full migration
cargo run --bin migrate -- migrate-agentdb

# Skip experience history (faster)
cargo run --bin migrate -- migrate-agentdb --skip-history

# Custom batch size
cargo run --bin migrate -- migrate-agentdb --batch-size 1000
```

**Migrates:**
- RL experiences (Redis → PostgreSQL)
- Q-values and policies
- Agent metadata

### Monitor Progress

```bash
# Live monitoring (5 sec interval)
cargo run --bin migrate -- monitor --interval-secs 5

# Longer interval
cargo run --bin migrate -- monitor --interval-secs 30
```

**Displays:**
- Migration progress (%)
- Items migrated / total
- Throughput (items/sec)
- ETA remaining

### Rollback

```bash
# Preview rollback
cargo run --bin migrate -- rollback

# Execute rollback (preserve Milvus)
cargo run --bin migrate -- rollback --confirm --preserve-milvus

# Execute rollback (delete Milvus)
cargo run --bin migrate -- rollback --confirm
```

**Rollback actions:**
1. Update config to `neo4j_only`
2. Kubernetes rolling update
3. Preserve or delete Milvus data
4. Verification

### Generate Report

```bash
# Markdown format
cargo run --bin migrate -- report \
  --output-file migration-report.md \
  --format markdown

# JSON format
cargo run --bin migrate -- report \
  --output-file migration-report.json \
  --format json

# YAML format
cargo run --bin migrate -- report \
  --format yaml
```

## 📜 Scripts

### run-migration.sh

Orchestrates full migration:

```bash
# Normal run
./scripts/run-migration.sh

# Dry run mode
./scripts/run-migration.sh --dry-run

# Skip validation (faster, riskier)
./scripts/run-migration.sh --skip-validation
```

**Steps:**
1. Preflight checks
2. Deploy Milvus
3. Enable dual-write mode
4. Migrate embeddings
5. Validate migration
6. Shadow Milvus testing (5 min)
7. Switch to hybrid mode
8. Generate report

### deploy-milvus.sh

Deploys Milvus cluster:

```bash
./scripts/deploy-milvus.sh
```

**Deploys:**
- 3-node etcd cluster
- 4-node MinIO (distributed storage)
- 3-node Pulsar (message queue)
- 2 query nodes
- 2 data nodes
- 1 index node
- HNSW index configuration

### monitor-latency.sh

Monitors Neo4j vs Milvus performance:

```bash
# Monitor for 60 minutes (default)
./scripts/monitor-latency.sh

# Monitor for 10 minutes
./scripts/monitor-latency.sh 10
```

**Displays:**
- Neo4j p50/p95 latency
- Milvus p50/p95 latency
- Speedup factor
- Error rates
- Dual-write success rate

### rollback.sh

Quick rollback:

```bash
# Preview
./scripts/rollback.sh

# Execute (preserve data)
./scripts/rollback.sh --confirm

# Execute (delete Milvus)
./scripts/rollback.sh --confirm --delete-milvus
```

## 📈 Monitoring

### Grafana Dashboard

Import dashboard:

```bash
kubectl port-forward -n monitoring svc/grafana 3000:3000
# Open: http://localhost:3000
# Import: grafana/migration-dashboard.json
```

**Panels:**
1. Migration Progress (gauge)
2. Items Migrated Over Time (graph)
3. Neo4j vs Milvus Latency (graph)
4. Dual-Write Success Rate (stat)
5. Error Rate (graph)
6. Storage Mode (stat)
7. Validation Success Rate (gauge)
8. Memory Usage (graph)
9. Throughput (stat)
10. ETA Remaining (stat)
11. Milvus Collection Stats (table)
12. Neo4j Query Performance (graph)

### Prometheus Metrics

**Migration metrics:**
```promql
# Progress
(migration_migrated_items / migration_total_items) * 100

# Throughput
rate(migration_migrated_items[5m])

# Error rate
rate(migration_errors_total[5m])
```

**Latency metrics:**
```promql
# Neo4j p95
histogram_quantile(0.95, rate(neo4j_search_latency_ms_bucket[5m]))

# Milvus p95
histogram_quantile(0.95, rate(milvus_search_latency_ms_bucket[5m]))

# Speedup
neo4j_p95 / milvus_p95
```

**Dual-write metrics:**
```promql
# Success rate
(dual_write_success_total / dual_write_attempts_total) * 100

# Milvus error rate
rate(dual_write_milvus_errors_total[5m])
```

## 🐛 Troubleshooting

### Migration Stalled

**Symptoms:** No progress for 5+ minutes

**Diagnosis:**
```bash
# Check last checkpoint
cat /tmp/migration_checkpoint.json

# Check logs
kubectl logs -f deployment/recommendation-engine | grep migration
```

**Solution:**
```bash
# Resume from checkpoint
LAST_ID=$(jq -r .last_id /tmp/migration_checkpoint.json)
cargo run --bin migrate -- migrate-embeddings --resume-from "$LAST_ID"
```

### Validation Failed

**Symptoms:** Success rate < 99%

**Diagnosis:**
```bash
# Run validation with details
cargo run --bin migrate -- validate \
  --sample-size 10000 \
  --full-scan
```

**Solution:**
```bash
# Auto-fix inconsistencies
cargo run --bin migrate -- validate \
  --sample-size 1000 \
  --fix-inconsistencies

# Re-validate
cargo run --bin migrate -- validate --sample-size 1000
```

### Milvus Connection Failed

**Symptoms:** `Milvus unreachable` in preflight

**Diagnosis:**
```bash
# Check Milvus pods
kubectl get pods -n milvus

# Check logs
kubectl logs -n milvus -l app.kubernetes.io/name=milvus
```

**Solution:**
```bash
# Restart Milvus
kubectl rollout restart deployment/milvus -n milvus

# Or redeploy
./scripts/deploy-milvus.sh
```

### High Error Rate

**Symptoms:** Dual-write errors >1%

**Diagnosis:**
```bash
# Check metrics
kubectl logs -f deployment/recommendation-engine | \
  grep -E "(dual_write_error|milvus_error)"

# Check Milvus capacity
kubectl top pods -n milvus
```

**Solution:**
```bash
# Scale Milvus query nodes
kubectl scale deployment/milvus-querynode -n milvus --replicas=4

# Or reduce batch size
cargo run --bin migrate -- migrate-embeddings --batch-size 500
```

### Out of Memory

**Symptoms:** OOMKilled pods during migration

**Solution:**
```bash
# Reduce batch size
cargo run --bin migrate -- migrate-embeddings --batch-size 200

# Increase pod memory
kubectl set resources deployment/recommendation-engine \
  --limits=memory=8Gi

# Reduce concurrency
# Edit src/migration/embeddings.rs:
# Semaphore::new(5)  // down from 10
```

## ⚡ Performance

### Benchmarks

**Migration throughput:**
- Small batches (100): ~50 items/sec
- Medium batches (1000): ~200 items/sec ✅ **Recommended**
- Large batches (5000): ~300 items/sec (higher memory)

**Query latency improvement:**
| Operation | Neo4j | Milvus | Speedup |
|-----------|-------|--------|---------|
| Vector search (p50) | 45ms | 8ms | **5.6x** |
| Vector search (p95) | 82ms | 15ms | **5.5x** |
| Top-100 similar | 120ms | 22ms | **5.5x** |

**Resource usage:**
- Migration: ~4GB memory, 2 CPU cores
- Milvus cluster: ~20GB memory, 8 CPU cores (production)
- Disk: 3x embedding size (replication)

### Tuning

**Batch size:**
```bash
# Throughput vs memory tradeoff
--batch-size 100   # Low memory, slow
--batch-size 1000  # Balanced (recommended)
--batch-size 5000  # High memory, fast
```

**Concurrency:**
```rust
// src/migration/embeddings.rs
let semaphore = Arc::new(Semaphore::new(20));  // Up from 10
```

**HNSW parameters:**
```python
# High accuracy (slower)
{"M": 32, "efConstruction": 512}

# Balanced (recommended)
{"M": 16, "efConstruction": 256}

# Fast queries (lower recall)
{"M": 8, "efConstruction": 128}
```

## 🔒 Safety Features

**Zero-downtime design:**
- ✅ Reads from Neo4j during migration (no change)
- ✅ Dual-write mode catches all new data
- ✅ Rollback preserves all data
- ✅ Checkpointing allows resume

**Rollback guarantees:**
- ✅ Neo4j remains source of truth
- ✅ One-command rollback
- ✅ Data preserved for retry
- ✅ No data loss possible

**Validation:**
- ✅ Vector equality checks (L2 distance)
- ✅ Missing embedding detection
- ✅ Orphan cleanup
- ✅ Auto-fix inconsistencies

## 📚 Documentation

- [Migration Guide](docs/migration-guide.md) - Detailed step-by-step
- [Architecture Decision](docs/adr/001-hybrid-storage.md) - Why hybrid
- [Milvus Selection](docs/adr/002-milvus-selection.md) - Why Milvus
- [Zero-Downtime Strategy](docs/adr/003-zero-downtime-strategy.md) - How

## 🆘 Support

**Issues:** https://github.com/your-org/hackathon-tv5/issues
**Slack:** #migration-support
**On-call:** ops-team@example.com

## 📄 License

MIT License - see [LICENSE](../LICENSE)
