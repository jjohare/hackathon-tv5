# Semantic Recommender Repository Reorganization Plan

**Date**: 2025-12-07
**Status**: Planning Phase
**Queen Coordinator**: Strategic Oversight & Swarm Coordination

## Mission Objective

Transform the hackathon-era repository structure into a production-ready organisation with:

1. **Single Source of Truth** - All documentation consolidated in `/docs`
2. **Clear Separation** - Production code vs experimental variants
3. **Logical Grouping** - Scripts organized by function
4. **Updated References** - All build files reflect new structure

## Strategic analysis

### Current State Assessment

**Documentation Chaos** (48+ markdown files scattered):
- Root level: `README.md`, `ARCHITECTURE.md`, `PERFORMANCE.md`, `CONTRIBUTING.md`, etc.
- `/docs`: Partially organized (8 files + subdirs)
- `/docs/reports`: 8 test/benchmark reports
- `/src/docs`: Duplicate documentation (3 files)
- `/design`: Massive archive (100+ files in `/archive/2025-12-04/`)
- Scattered READMEs in every subdirectory

**CUDA Kernel Fragmentation**:
- `/src/cuda/kernels/` - Production kernels (5 files)
- `/src/cuda/kernels/variants/` - Experimental algorithms (10 variants)
- `/src/cuda/examples/` - Test programs (4 files)
- `/src/cuda/benchmarks/` - Benchmarking code (1 file)
- `/kernels/` - Orphaned kernel copy (1 file)
- `/tests/` - CUDA test file (1 file)

**Scripts Disorganization** (21 files, no grouping):
```
scripts/
├── mcp_server.py
├── generate_embeddings.py
├── benchmark_a100.py
├── gpu_ontology_reasoning.py
└── ... (17 more ungrouped files)
```

**Build File Dependencies**:
- `build.rs` - Hardcoded paths to `/src/cuda/kernels/`
- `Cargo.toml` - Workspace configuration
- `Makefile` - Build automation
- Multiple Python scripts with hardcoded paths

## Reorganization Strategy

### Phase 1: Documentation Consolidation Swarm (3 Workers)

**Worker 1: Root Documentation**
- Move top-level `.md` files → `/docs/`
- Keep only `README.md` and `LICENSE` in root
- Update all internal links

**Worker 2: Source Documentation**
- Merge `/src/docs/` → `/docs/reference/`
- Consolidate duplicate content
- Remove empty READMEs

**Worker 3: Design Archive**
- Keep `/design/archive/` as-is (historical)
- Move active `/design/guides/` → `/docs/guides/`
- Move `/design/docs/` → `/docs/architecture/`
- Move `/design/ontology/` → `/docs/ontology/`

**Target Structure**:
```
docs/
├── README.md (index/navigation)
├── QUICKSTART.md
├── ARCHITECTURE.md
├── PERFORMANCE.md
├── CONTRIBUTING.md
├── API.md
├── architecture/
│   ├── system-architecture.md
│   ├── deployment-guide.md
│   └── diagrams/
├── guides/
│   ├── gpu-setup-guide.md
│   ├── cuda-optimization-strategies.md
│   ├── deployment-guide.md
│   └── ontology-reasoning-guide.md
├── ontology/
│   ├── ONTOLOGY_SOURCES.md
│   ├── ONTOLOGY_INTEGRATION_PLAN.md
│   ├── ONTOLOGY_REASONING_CAPABILITIES.md
│   └── GPU_ONTOLOGY_REASONING.md
├── reference/
│   ├── API_REFERENCE.md
│   ├── GETTING_STARTED.md
│   └── PERFORMANCE.md
└── reports/
    ├── A100_TEST_RESULTS.md
    ├── A100_DEPLOYMENT_GUIDE.md
    ├── DATA_PIPELINE_COMPLETE.md
    └── SYSTEM_STATUS.md
```

### Phase 2: CUDA Kernel Reorganization Swarm (2 Workers)

**Worker 1: Kernel Consolidation**
```
src/cuda/
├── kernels/
│   ├── production/          # Core production kernels
│   │   ├── semantic_similarity.cu
│   │   ├── graph_search.cu
│   │   ├── ontology_reasoning.cu
│   │   ├── hybrid_sssp.cu
│   │   └── *.cuh headers
│   └── experimental/        # Variant algorithms
│       ├── benchmark_algorithms.cu
│       ├── hybrid_index.cu
│       ├── lsh_gpu.cu
│       ├── memory_layout.cu
│       ├── product_quantization.cu
│       ├── semantic_similarity_fp16.cu
│       ├── semantic_similarity_fp16_tensor_cores.cu
│       ├── semantic_similarity_tf32.cu
│       ├── sorted_similarity.cu
│       └── unified_pipeline.cu
├── examples/                # Validation programs
│   ├── t4_validation.cu
│   ├── phase2_benchmark.cu
│   └── graph_search_example.cu
├── tests/                   # CUDA-specific tests
│   └── test_benchmark_algorithms.cu
└── README.md
```

**Worker 2: Cleanup & Deduplication**
- Delete `/kernels/semantic_similarity_tf32.cu` (duplicate)
- Move `/tests/test_benchmark_algorithms.cu` → `/src/cuda/tests/`
- Update Makefile for new structure

### Phase 3: Scripts Reorganization Swarm (3 Workers)

**Worker 1: Functional Grouping**
```
scripts/
├── deployment/
│   ├── deploy_and_test_a100.sh
│   └── run_all.sh
├── benchmarking/
│   ├── benchmark_a100.py
│   ├── benchmark_hyper_personalization.py
│   └── test_a100_comprehensive.py
├── data/
│   ├── parse_movielens.py
│   ├── generate_embeddings.py
│   ├── generate_platform_data.py
│   ├── generate_user_profiles.py
│   └── validate_data.py
├── population/
│   ├── populate_milvus.py
│   ├── populate_neo4j.py
│   └── populate_agentdb.py
├── inference/
│   ├── gpu_recommend.py
│   ├── gpu_ontology_reasoning.py
│   ├── gpu_hyper_personalization.py
│   └── run_recommendations.py
├── mcp/
│   ├── mcp_server.py
│   └── mcp_server_http.py
├── utils/
│   ├── convert_ascii_to_mermaid.py
│   └── rebuild_architecture_clean.py
└── README.md
```

**Worker 2: Path Updates**
- Update all import statements
- Fix hardcoded data paths
- Ensure relative imports work

**Worker 3: Documentation**
- Create `/scripts/README.md` with directory guide
- Add usage examples to each subdirectory

### Phase 4: Reference Update Swarm (2 Workers)

**Worker 1: Build System Updates**
- `build.rs`: Update CUDA kernel paths
  ```rust
  // OLD
  "src/cuda/kernels/unified_pipeline.cu"
  "src/cuda/kernels/semantic_similarity_fp16_tensor_cores.cu"

  // NEW
  "src/cuda/kernels/experimental/unified_pipeline.cu"
  "src/cuda/kernels/production/semantic_similarity.cu"
  ```
- `Cargo.toml`: Verify workspace paths
- `Makefile`: Update all source references

**Worker 2: Documentation Links**
- Update all cross-references in `/docs`
- Fix relative links after moves
- Validate all hyperlinks work

## Git Safety Protocol

### ALL operations use `git mv` for tracking:
```bash
# Example
git mv src/docs/API_REFERENCE.md docs/reference/API_REFERENCE.md
git mv src/cuda/kernels/variants/ src/cuda/kernels/experimental/
```

### Commit strategy:
1. Phase 1 commits: Documentation consolidation
2. Phase 2 commits: CUDA reorganization
3. Phase 3 commits: Scripts grouping
4. Phase 4 commits: Build file updates

## Success Criteria

✅ **Zero broken imports** - All code compiles
✅ **Git tracking complete** - All moves visible in history
✅ **Documentation navigable** - Clear `/docs` structure
✅ **Logical organisation** - Grouped by function
✅ **Build automation works** - `cargo build` succeeds
✅ **Tests pass** - No regressions introduced

## Risk Mitigation

**Rollback Plan**: Each phase creates a git commit, allowing atomic rollback
**Testing Gates**: Compile & test after each phase
**Validation**: Automated link checker + import validator
**Communication**: Memory namespace `hive/reorganization/*` for coordination

## Coordination Protocol

**Queen Coordinator** writes strategic directives to:
- `hive/reorganization/queen-status`
- `hive/reorganization/phase-gates`
- `hive/reorganization/validation-results`

**Worker Swarms** report progress to:
- `hive/reorganization/documentation-swarm/*`
- `hive/reorganization/cuda-swarm/*`
- `hive/reorganization/scripts-swarm/*`
- `hive/reorganization/reference-swarm/*`

## Timeline

| Phase | Duration | Completion Check |
|-------|----------|------------------|
| Phase 1 | 15 min | Documentation navigable |
| Phase 2 | 10 min | CUDA builds successfully |
| Phase 3 | 10 min | Scripts runnable |
| Phase 4 | 10 min | All tests pass |
| Validation | 5 min | Report generation |

**Total**: ~50 minutes for complete reorganization

---

**Generated by**: Queen Coordinator
**Hive Intelligence**: 4 specialised swarms (10 workers)
**Coordination**: `hive/reorganization/*` memory namespace
