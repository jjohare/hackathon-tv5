# Project Structure - Quick Reference

## Primary Directories at a Glance

```
semantic-recommender/
├── 📊 benches/              Benchmark Suite (7 files)
├── ⚙️  config/               Configuration (YAML configs)
├── 📦 crates/               Rust Workspace Crates (6 crates)
├── 📁 data/                 Datasets & Embeddings (9.2GB)
├── 📐 design/               Design Documentation (2.5M)
├── 🐳 docker/               Docker Configurations
├── 📚 docs/                 Public Documentation (1.2M)
├── 📝 examples/             Example Applications (2 files)
├── 📊 grafana/              Monitoring Dashboards
├── ☸️  k8s/                  Kubernetes Manifests
├── 🔧 kernels/              Legacy CUDA Kernels
├── 🔌 proto/                Protocol Buffers
├── 📈 results/              Benchmark Results
├── 🧠 scripts/              Operational Scripts (8 subdirectories)
│   ├── benchmarks/          Test runners
│   ├── data_pipeline/       Data processing
│   ├── ops/                 Operations
│   ├── server/              Server implementations
│   └── utils/               Utilities
├── 🦀 src/                  Main Source Code (2.2M)
│   ├── api/                 REST/GraphQL API
│   ├── bin/                 Binary targets
│   ├── commands/            CLI commands
│   ├── cuda/                CUDA Implementation
│   │   ├── kernels/         GPU Kernels (PRIMARY)
│   │   ├── examples/        Executable demos
│   │   ├── benchmarks/      Microbenchmarks
│   │   └── Makefile.a100    A100 build config
│   ├── mcp/                 MCP Server
│   ├── migration/           Data migration
│   ├── rust/                Core Rust implementation
│   │   ├── gpu_engine/      GPU execution layer
│   │   ├── semantic_search/ Search engine
│   │   ├── storage/         Multi-storage layer
│   │   ├── ontology/        Ontology processing
│   │   └── distributed/     Distributed computing
│   └── utils/               Helper utilities
├── 🧪 tests/                Test Suite (18+ test files)
└── 📋 [Root Config Files]
```

---

## Quick Stats

| Metric | Count |
|--------|-------|
| Total Directories | 158 |
| Total Files | 458 |
| CUDA Kernel Files | 8+ |
| Python Scripts | 25+ |
| Documentation Files | 67 |
| Test Files | 18+ |
| Rust Source Files | 100+ |
| Configuration Files | 15+ |

---

## Key File Locations

### CUDA Development
- **Kernels**: `src/cuda/kernels/`
- **Build (A100)**: `src/cuda/kernels/Makefile.a100`
- **Build (T4)**: `src/cuda/kernels/Makefile`
- **Examples**: `src/cuda/examples/`

### Data Pipeline
- **Generation**: `scripts/data_pipeline/generate_*.py`
- **Population**: `scripts/data_pipeline/populate_*.py`
- **Validation**: `scripts/data_pipeline/validate_data.py`
- **Raw Data**: `data/raw/ml-25m/` (MovieLens)

### Server & API
- **REST API**: `src/api/lib.rs`
- **GraphQL**: `src/api/graphql.rs`
- **MCP Server**: `scripts/server/mcp_server.py`
- **GPU Service**: `scripts/server/gpu_recommend.py`

### Testing
- **Integration Tests**: `tests/*.rs`
- **Test Fixtures**: `tests/fixtures/`
- **Test Runner**: `tests/run_tests.sh`
- **CUDA Tests**: `tests/cuda_integration_test.rs`

### Deployment
- **Docker**: `docker/Dockerfile.a100`
- **Kubernetes**: `k8s/milvus/`, `k8s/agentdb/`
- **Compose**: `docker/docker-compose.yml`
- **Operations**: `scripts/ops/*.sh`

---

## Core Modules (src/rust/)

| Module | Purpose | Key Files |
|--------|---------|-----------|
| **gpu_engine** | GPU execution layer | engine.rs, hybrid_sssp.rs, adaptive_sssp.rs |
| **semantic_search** | Search & recommendation | unified_engine.rs, ranking.rs, explanation.rs |
| **storage** | Multi-backend storage | hybrid_coordinator.rs, milvus_client.rs, postgres_store.rs |
| **ontology** | Ontology processing | loader.rs, reasoner.rs, validator.rs |
| **distributed** | Distributed computing | query_router.rs, result_aggregator.rs, shard_manager.rs |
| **agentdb** | AgentDB integration | coordinator.rs, integration.rs |
| **adaptive_sssp** | Pathfinding algorithms | gpu_dijkstra.rs, landmark_apsp.rs |
| **models** | Data structures | embeddings.rs, recommendation.rs, ontology.rs |

---

## Data Organization

```
data/
├── raw/              MovieLens datasets, raw inputs
├── processed/        Cleaned, formatted data
├── embeddings/       Vector embeddings (indexed)
├── synthetic/        Generated test data
└── ontologies/       Ontology definitions
```

---

## Storage Systems Integration

| System | Config | Population Script | Use Case |
|--------|--------|-------------------|----------|
| **Milvus** | `k8s/milvus/` | `populate_milvus.py` | Vector search |
| **Neo4j** | `k8s/agentdb/` | `populate_neo4j.py` | Graph queries |
| **PostgreSQL** | `sql/agentdb-schema.sql` | `populate_agentdb.py` | Relational data |
| **Redis** | `src/rust/storage/redis_cache.rs` | - | Caching layer |

---

## Documentation Map

| Section | Location | Contents |
|---------|----------|----------|
| **Architecture** | `docs/architecture/` | System design, diagrams |
| **Guides** | `design/guides/` | Implementation tutorials |
| **API Reference** | `docs/reference/API.md` | REST/GraphQL endpoints |
| **Reports** | `docs/reports/` | Benchmarks, status |
| **CUDA Docs** | `src/cuda/README.md` | Kernel documentation |
| **Ontology** | `design/ontology/` | Ontology definitions |

---

## Common Tasks & Locations

### Compile CUDA Kernels
```bash
cd src/cuda/kernels
make -f Makefile.a100  # For A100
make                   # For T4
```

### Run Data Pipeline
```bash
python scripts/data_pipeline/generate_embeddings.py
python scripts/data_pipeline/populate_milvus.py
python scripts/data_pipeline/populate_neo4j.py
```

### Run Tests
```bash
cargo test -p semantic-recommender --lib
cd tests && ./run_tests.sh
```

### Deploy with Kubernetes
```bash
kubectl apply -f k8s/agentdb/
kubectl apply -f k8s/milvus/
```

### Run Benchmarks
```bash
python scripts/benchmarks/benchmark_a100.py
python scripts/benchmarks/test_a100_comprehensive.py
```

---

## Development Workflow

1. **Develop**: Edit files in appropriate subdirectories
2. **Build**: Use Makefiles for CUDA, Cargo for Rust
3. **Test**: Run integration tests in `tests/`
4. **Benchmark**: Execute scripts in `scripts/benchmarks/`
5. **Deploy**: Use Docker or Kubernetes configs
6. **Document**: Update docs/ and design/ as needed

---

## Environment Variables

Configured in `/design/.env` and `.env` files:
- Database connection strings
- API keys and secrets
- GPU configuration
- Performance tuning

---

**Quick Navigation**: Use this guide to locate any component in the project structure. For detailed information, see PROJECT_STRUCTURE_FINAL.md.
