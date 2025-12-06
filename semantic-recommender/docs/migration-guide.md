# Migration Guide: Neo4j-Only → Hybrid Architecture

## Overview

This guide covers the zero-downtime migration from Neo4j-only storage to a hybrid architecture with:
- **Milvus** for vector embeddings (150x faster search)
- **Neo4j** for graph relationships
- **PostgreSQL + AgentDB** for reinforcement learning state

## Prerequisites

- Kubernetes cluster with 3+ nodes
- 50GB+ available storage
- Neo4j 5.15+ running
- PostgreSQL 15+ available
- Redis 7+ available
- `kubectl` access to cluster
- Rust toolchain installed

## Migration Phases

### Phase 1: Preflight Checks

Validate system readiness:

```bash
cargo run --bin migrate -- preflight --check-connectivity
```

This checks:
- ✅ Neo4j connectivity and data inventory
- ✅ Milvus cluster health
- ✅ PostgreSQL availability
- ✅ Redis connectivity
- ✅ Disk space (requires 3x embedding data size)
- ✅ Memory availability

**Expected output:**
```
✅ READY FOR MIGRATION
  • 150,000 media items
  • 500 GB disk available (150 GB required)
  • Estimated time: 12 minutes
```

### Phase 2: Deploy Milvus

Deploy Milvus cluster with HNSW indexing:

```bash
./scripts/deploy-milvus.sh
```

This deploys:
- Milvus 2.3+ with 3-node etcd
- MinIO for object storage (4 replicas)
- Pulsar for message queue
- 2 query nodes, 2 data nodes, 1 index node

**Verification:**
```bash
kubectl get pods -n milvus
# All pods should be Running

kubectl exec -n milvus deploy/milvus -- \
  curl -s localhost:9091/healthz
# Should return: OK
```

### Phase 3: Dual-Write Mode

Enable writes to both Neo4j (primary) and Milvus (shadow):

```bash
kubectl set env deployment/recommendation-engine \
  STORAGE_MODE=dual_write

kubectl rollout status deployment/recommendation-engine
```

**What happens:**
- All new embeddings written to both systems
- Reads still from Neo4j (no behavior change)
- Milvus errors logged but don't fail requests
- Metrics track dual-write success rate

**Monitor:**
```bash
kubectl logs -f deployment/recommendation-engine | grep "dual_write"
```

### Phase 4: Migrate Historical Data

Batch migrate existing embeddings:

```bash
# Dry run first
cargo run --bin migrate -- migrate-embeddings \
  --batch-size 1000 \
  --dry-run

# Actual migration
cargo run --bin migrate -- migrate-embeddings \
  --batch-size 1000
```

**Progress monitoring:**
```bash
# In separate terminal
cargo run --bin migrate -- monitor --interval-secs 5
```

**Checkpointing:**
- Saves progress to `/tmp/migration_checkpoint.json`
- Resume on failure: `--resume-from <last_id>`

### Phase 5: Validation

Verify migration completeness:

```bash
# Random sample
cargo run --bin migrate -- validate --sample-size 1000

# Full scan (slow)
cargo run --bin migrate -- validate --sample-size 10000 --full-scan

# Fix inconsistencies
cargo run --bin migrate -- validate \
  --sample-size 1000 \
  --fix-inconsistencies
```

**Success criteria:**
- ✅ 99%+ success rate
- ✅ Max vector L2 distance < 1e-5
- ✅ No missing embeddings

### Phase 6: Shadow Milvus Mode

Test Milvus reads with Neo4j fallback:

```bash
kubectl set env deployment/recommendation-engine \
  STORAGE_MODE=shadow_milvus

kubectl rollout status deployment/recommendation-engine
```

**Monitor performance:**
```bash
./scripts/monitor-latency.sh 10  # 10 minutes
```

**Expected results:**
- Milvus p95 latency: **5-15ms**
- Neo4j p95 latency: **40-80ms**
- **3-10x speedup** on vector search

### Phase 7: Full Hybrid Mode

Switch to Milvus as primary for reads:

```bash
kubectl set env deployment/recommendation-engine \
  STORAGE_MODE=hybrid

kubectl rollout status deployment/recommendation-engine
```

**What changes:**
- All vector searches use Milvus HNSW index
- Neo4j used only for graph traversals
- No fallback (faster, but requires Milvus uptime)

### Phase 8: Migrate AgentDB

Move RL state from Redis to PostgreSQL:

```bash
cargo run --bin migrate -- migrate-agentdb
```

This migrates:
- RL experiences (state, action, reward tuples)
- Q-values and policies
- Agent metadata

**Verification:**
```bash
psql $DATABASE_URL -c "SELECT COUNT(*) FROM rl_experiences;"
# Should match Redis key count
```

### Phase 9: Generate Report

```bash
cargo run --bin migrate -- report \
  --output-file migration-report.md \
  --format markdown

cat migration-report.md
```

## Rollback Procedure

### Emergency Rollback

If critical issues arise:

```bash
# Immediate rollback
./scripts/rollback.sh --confirm

# Or using migration tool
cargo run --bin migrate -- rollback --confirm --preserve-milvus
```

**What happens:**
1. Updates config to `STORAGE_MODE=neo4j_only`
2. Kubernetes rolling update to old configuration
3. Milvus data preserved for debugging
4. **Zero data loss** (Neo4j remained source of truth)

### Delete Milvus Data

If abandoning migration:

```bash
cargo run --bin migrate -- rollback --confirm
# Drops Milvus collection permanently
```

## Monitoring

### Grafana Dashboard

Import dashboard:

```bash
kubectl create configmap migration-dashboard \
  --from-file=grafana/migration-dashboard.json \
  -n monitoring

# Port-forward Grafana
kubectl port-forward -n monitoring svc/grafana 3000:3000
# Open: http://localhost:3000/d/migration-progress
```

**Key metrics:**
- Migration progress (%)
- Throughput (items/sec)
- Neo4j vs Milvus latency comparison
- Dual-write success rate
- Error rates

### CLI Monitoring

```bash
# Live progress
cargo run --bin migrate -- monitor --interval-secs 5

# Latency comparison
./scripts/monitor-latency.sh 30  # 30 minutes
```

## Troubleshooting

### Milvus Connection Failed

```bash
# Check Milvus pods
kubectl get pods -n milvus

# Check logs
kubectl logs -n milvus deploy/milvus

# Test connectivity
kubectl run -it --rm debug --image=alpine --restart=Never -- sh
apk add curl
curl http://milvus.milvus.svc.cluster.local:19530/healthz
```

### Migration Stalled

```bash
# Check last checkpoint
cat /tmp/migration_checkpoint.json

# Resume from checkpoint
cargo run --bin migrate -- migrate-embeddings \
  --resume-from <last_id>
```

### Validation Failed

```bash
# Re-run with auto-fix
cargo run --bin migrate -- validate \
  --sample-size 1000 \
  --fix-inconsistencies

# Check specific item
kubectl exec -n milvus deploy/milvus -- \
  python3 -c "
from pymilvus import connections, Collection
connections.connect(host='localhost', port='19530')
collection = Collection('media_embeddings')
results = collection.query(expr='id == \"<item_id>\"')
print(results)
"
```

### High Error Rate

```bash
# Check dual-write metrics
kubectl logs -f deployment/recommendation-engine | \
  grep -E "(dual_write|milvus_error)"

# Investigate Milvus capacity
kubectl top pods -n milvus
```

## Performance Tuning

### Batch Size

```bash
# Small batches (safer, slower)
--batch-size 100

# Large batches (faster, more memory)
--batch-size 5000
```

**Recommended:** 1000 items/batch (balanced)

### Concurrency

Edit `src/migration/embeddings.rs`:

```rust
// Increase from 10 to 20 concurrent batches
let semaphore = Arc::new(Semaphore::new(20));
```

### Milvus Index Parameters

Optimize HNSW for your workload:

```python
# Higher M = better recall, slower build
{"M": 32, "efConstruction": 512}  # High accuracy

# Lower M = faster, less accurate
{"M": 8, "efConstruction": 128}   # Fast queries
```

## Post-Migration

### Cleanup

```bash
# Remove dual-write code after 1 week
git rm src/storage/dual_write.rs

# Archive Neo4j embeddings (optional)
MATCH (m:MediaContent) REMOVE m.embedding
```

### Optimize Milvus

```bash
# Compact segments
kubectl exec -n milvus deploy/milvus -- \
  python3 -c "
from pymilvus import Collection
collection = Collection('media_embeddings')
collection.compact()
"

# Rebuild index
collection.release()
collection.drop_index()
collection.create_index(...)
collection.load()
```

## Architecture Decision Records

See `docs/adr/`:
- `001-hybrid-storage.md` - Why hybrid architecture
- `002-milvus-selection.md` - Why Milvus over alternatives
- `003-zero-downtime-strategy.md` - Migration approach

## Support

- **Issues:** https://github.com/your-org/hackathon-tv5/issues
- **Slack:** #migration-support
- **On-call:** ops-team@example.com
