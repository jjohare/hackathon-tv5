# AgentDB Deployment Guide

## Architecture

AgentDB provides multi-modal memory for AI agents:
- **Episodic**: State-action-reward trajectories
- **Semantic**: Learned patterns and preferences
- **Procedural**: RL policies (Q-values, Thompson Sampling)

## Performance Targets

- Policy lookups: <5ms (cached)
- Episode recording: Async (non-blocking)
- PostgreSQL fallback: <20ms

## Deployment Steps

### 1. PostgreSQL with pgvector

```bash
kubectl apply -f k8s/agentdb/postgres-statefulset.yaml
```

Wait for PostgreSQL to be ready:
```bash
kubectl wait --for=condition=ready pod -l app=agentdb-postgres -n media-gateway --timeout=300s
```

### 2. Initialize Schema

```bash
kubectl create configmap agentdb-schema \
  --from-file=sql/agentdb-schema.sql \
  -n media-gateway

kubectl apply -f k8s/agentdb/pgvector-init.yaml
```

### 3. Redis Cache Cluster

```bash
kubectl apply -f k8s/agentdb/redis-statefulset.yaml
```

### 4. Update Application Configuration

```bash
kubectl apply -f k8s/agentdb/secrets.yaml
```

Update recommendation engine deployment:
```yaml
env:
- name: AGENTDB_POSTGRES_URL
  valueFrom:
    secretKeyRef:
      name: agentdb-secrets
      key: postgres-url
- name: AGENTDB_REDIS_URL
  valueFrom:
    secretKeyRef:
      name: agentdb-secrets
      key: redis-url
```

## Performance Testing

```bash
cargo test --release --package media-recommendation-engine \
  --lib agentdb::tests::test_policy_lookup_latency

cargo test --release --package media-recommendation-engine \
  --lib agentdb::tests::test_batch_episode_insert
```

## Monitoring

Key metrics:
- `agentdb_policy_lookup_duration_ms` (p50, p95, p99)
- `agentdb_cache_hit_rate` (target: >70%)
- `agentdb_episodes_flushed_total`
- `agentdb_policy_updates_total`

## Scaling

- PostgreSQL: Vertical scaling (increase resources)
- Redis: Horizontal scaling (increase replicas)
- Connection pooling: Adjust `max_connections` parameter

## Cost Estimates

**Development (single instance)**:
- PostgreSQL: $50-100/month
- Redis: $30-50/month

**Production (100x T4 deployment)**:
- PostgreSQL: $1,440/month (r6i.4xlarge)
- Redis: $900/month (cache.r6g.2xlarge × 3)
- Total: $3,140/month (with high availability)
