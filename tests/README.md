# Hybrid Architecture Test Suite

Comprehensive testing for the Milvus + Neo4j + PostgreSQL + AgentDB hybrid storage architecture.

## Test Categories

### 1. Integration Tests (`hybrid_integration_tests.rs`)
End-to-end tests validating the complete hybrid architecture:

- **Latency Tests**: p99 < 10ms target validation
- **Accuracy Tests**: Vector search similarity ranking
- **Graph Enrichment**: Neo4j relationship integration
- **AgentDB Policy**: User preference-based re-ranking
- **Cache Effectiveness**: Redis cache hit performance
- **Concurrent Operations**: Multi-threaded read/write safety
- **Filter Pushdown**: Query optimization validation

**Run all integration tests:**
```bash
cargo test --test hybrid_integration_tests
```

**Run specific test:**
```bash
cargo test --test hybrid_integration_tests test_hybrid_search_latency_p99_under_10ms
```

### 2. Chaos Tests (`chaos_tests.rs`)
Failure injection and resilience testing:

- **Milvus Failure**: Fallback to Neo4j graceful degradation
- **Neo4j Failure**: Vector-only mode operation
- **Redis Failure**: Cache-less operation
- **Network Partition**: Shard isolation handling
- **Cascading Failures**: Multi-component failure recovery
- **Slow Component Timeout**: Latency spike handling
- **Data Corruption**: Inconsistency detection

**Run chaos tests (manual only):**
```bash
cargo test --test chaos_tests -- --ignored
```

### 3. Load Tests (`load_tests.rs`)
High-scale performance validation:

- **7000 QPS Sustained Load**: 60-second sustained throughput
- **Burst Traffic**: 10,000 concurrent request handling
- **Gradual Load Increase**: 1K → 7K QPS ramp-up
- **Mixed Read/Write**: 90% read, 10% write workload

**Run load tests (manual only - requires resources):**
```bash
# Full 7000 QPS test
cargo test --test load_tests test_sustained_load_7000_qps -- --ignored --nocapture

# Burst test
cargo test --test load_tests test_burst_traffic_handling -- --ignored
```

### 4. Performance Benchmarks (`benches/hybrid_benchmarks.rs`)
Criterion-based micro-benchmarks:

- **Vector Search Scaling**: 1K → 1M dataset performance
- **Hybrid vs Vector-Only**: Architecture comparison
- **Cache Effectiveness**: Cold vs warm cache
- **Batch Ingestion**: Single vs batch write performance
- **Filter Strategies**: Filter pushdown optimization
- **Concurrent Queries**: Multi-threaded scaling
- **AgentDB Re-ranking**: Policy-based overhead
- **Embedding Dimensions**: 128 → 1024 dimension impact

**Run all benchmarks:**
```bash
cargo bench --bench hybrid_benchmarks
```

**Run specific benchmark group:**
```bash
cargo bench --bench hybrid_benchmarks vector_search_scaling
```

## Test Infrastructure

### Docker Compose Setup
Start all test dependencies:
```bash
docker-compose -f tests/docker-compose.test.yml up -d
```

Services included:
- **Milvus**: v2.4.0 with etcd + MinIO
- **Neo4j**: 5.15 with 2GB heap
- **PostgreSQL**: pg16 with pgvector extension
- **Redis**: 7-alpine for caching

### Environment Variables
```bash
export MILVUS_ENDPOINT=localhost:19530
export NEO4J_URI=bolt://localhost:7687
export NEO4J_USER=neo4j
export NEO4J_PASSWORD=testpassword
export POSTGRES_URL=postgres://postgres:testpassword@localhost:5432/test
export REDIS_URL=redis://localhost:6379
```

## CI/CD Integration

### GitHub Actions Workflows (`.github/workflows/test-hybrid.yml`)

**Automated on every PR:**
- Integration tests (30 min timeout)
- Performance benchmarks with result comments
- Security scanning (Cargo Audit + Trivy)

**Automated on main branch push:**
- Chaos tests (destructive, main only)
- Full load tests on 16-core runners
- SLA validation (p99 < 100ms, QPS >= 6500)

**Benchmark result comments on PRs:**
```
## Benchmark Results

vector_search_scaling/1000    time: 8.23ms
vector_search_scaling/10000   time: 9.45ms
hybrid_vs_vector_only/hybrid  time: 12.1ms
hybrid_vs_vector_only/vector  time: 8.8ms
cache_effectiveness/cold      time: 15.3ms
cache_effectiveness/warm      time: 2.1ms
```

## Performance SLAs

### Latency Targets
- **p50**: < 5ms
- **p99**: < 10ms (strict target)
- **p999**: < 50ms
- **max**: < 100ms

### Throughput Targets
- **Sustained QPS**: 7,000
- **Burst QPS**: 10,000
- **Error Rate**: < 1%

### Resilience Targets
- **Single Component Failure**: 100% availability (degraded mode)
- **Recovery Time**: < 10 seconds
- **Data Consistency**: 100% (no data loss)

## Test Data Generators

### Media Generator (`tests/fixtures/media_generator.rs`)
```rust
// Generate 1000 realistic media items
let media = media_generator::generate_media_content(1000);

// Generate similar item (90% similarity)
let similar = media_generator::generate_similar_media(&base_item, 0.9);
```

### Query Generator (`tests/fixtures/query_generator.rs`)
```rust
// Random query with filters/user_id
let query = query_generator::create_random_query();

// Query without filters
let query = query_generator::create_query_without_filter();
```

### User Generator (`tests/fixtures/user_generator.rs`)
```rust
// SciFi lover policy
let policy = user_generator::generate_scifi_lover_policy("user_123");

// Random diverse policies
let policies = user_generator::generate_diverse_policies(100);
```

## Running Tests Locally

### Quick Integration Tests
```bash
# Start services
docker-compose -f tests/docker-compose.test.yml up -d

# Wait for services
sleep 20

# Run tests
cargo test --test hybrid_integration_tests --release

# Stop services
docker-compose -f tests/docker-compose.test.yml down -v
```

### Full Test Suite
```bash
# All tests (except manual chaos/load)
cargo test --release

# Benchmarks
cargo bench

# Chaos tests (manual)
cargo test --test chaos_tests -- --ignored

# Load tests (manual)
cargo test --test load_tests -- --ignored --nocapture
```

## Troubleshooting

### Services Not Starting
```bash
# Check service logs
docker-compose -f tests/docker-compose.test.yml logs milvus
docker-compose -f tests/docker-compose.test.yml logs neo4j

# Restart services
docker-compose -f tests/docker-compose.test.yml restart
```

### Test Timeouts
```bash
# Increase test timeout
RUST_TEST_THREADS=1 cargo test --test hybrid_integration_tests -- --test-threads=1

# Run tests sequentially
cargo test -- --test-threads=1
```

### Memory Issues (Load Tests)
```bash
# Monitor resource usage
docker stats

# Clean up volumes
docker-compose -f tests/docker-compose.test.yml down -v
docker system prune -a
```

## Contributing

When adding new tests:

1. **Integration Tests**: Add to `hybrid_integration_tests.rs`
2. **Chaos Tests**: Add to `chaos_tests.rs` with `#[ignore]`
3. **Load Tests**: Add to `load_tests.rs` with `#[ignore]`
4. **Benchmarks**: Add to `benches/hybrid_benchmarks.rs`

All tests must:
- Be self-contained (no external dependencies)
- Use test fixtures from `tests/fixtures/`
- Clean up resources after completion
- Document expected behavior and assertions
- Include performance targets in comments

## References

- [Milvus Documentation](https://milvus.io/docs)
- [Neo4j Cypher Manual](https://neo4j.com/docs/cypher-manual/current/)
- [pgvector Guide](https://github.com/pgvector/pgvector)
- [Criterion.rs Benchmarking](https://bheisler.github.io/criterion.rs/book/)
- [Testcontainers Rust](https://docs.rs/testcontainers/latest/testcontainers/)
