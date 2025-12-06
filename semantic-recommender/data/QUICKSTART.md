# Data Generation Quickstart

## Prerequisites

1. **GCP A100 VM** (recommended) or local machine with GPU
2. **Python 3.9+**
3. **MovieLens datasets downloaded** (already in `data/raw/`)

## Installation

```bash
cd /path/to/semantic-recommender

# Install dependencies
pip install -r scripts/requirements.txt

# For GPU support
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

## Quick Test (Small Dataset)

Use `ml-latest-small` for quick testing:

```bash
# Modify scripts to use ml-latest-small instead of ml-25m
# Edit scripts/parse_movielens.py line 15:
# RAW_DIR = DATA_DIR / "raw" / "ml-latest-small"

# Run pipeline
python3 scripts/parse_movielens.py
python3 scripts/generate_user_profiles.py
python3 scripts/generate_platform_data.py
python3 scripts/generate_embeddings.py
```

**Expected time**: ~5 minutes on CPU, ~2 minutes on GPU

## Full Production Run (25M Dataset)

```bash
# Use default ml-25m configuration
# Ensure databases are running first:
docker-compose up -d milvus neo4j postgresql redis

# Run complete pipeline
chmod +x scripts/run_all.sh
./scripts/run_all.sh
```

**Expected time on A100**: ~4.5 hours total
- Phase 1 (Parse): 30 min
- Phase 2 (Synthetic): 75 min
- Phase 3 (Embeddings): 8 min
- Phase 4 (Databases): 2-3 hours
- Phase 5 (Validation): 30 min

## Step-by-Step Execution

### Phase 1: Parse MovieLens (30 min)

```bash
python3 scripts/parse_movielens.py
```

**Output**:
- `data/processed/media/movies.jsonl` (62,423 movies)
- `data/processed/media/genome_scores.json` (13,816 movies with tags)
- `data/processed/interactions/ratings.jsonl` (25M ratings)
- `data/processed/interactions/tags.jsonl` (1M tags)

### Phase 2: Generate Synthetic Data (75 min)

```bash
# User demographics
python3 scripts/generate_user_profiles.py

# Platform availability
python3 scripts/generate_platform_data.py
```

**Output**:
- `data/synthetic/users/demographics.jsonl` (162K users)
- `data/synthetic/platforms/availability.jsonl` (62K movies)

### Phase 3: Generate Embeddings on A100 (8 min)

```bash
python3 scripts/generate_embeddings.py
```

**Output**:
- `data/embeddings/media/content_vectors.npy` (62K × 384)
- `data/embeddings/users/preference_vectors.npy` (162K × 384)

**GPU Performance**:
- Batch size: 512
- Model: paraphrase-multilingual-MiniLM-L12-v2
- FP32 precision (can use FP16 for 2x speedup)

### Phase 4: Populate Databases (2-3 hours)

```bash
# Start databases first
docker-compose up -d milvus neo4j postgresql redis

# Populate Milvus (vector search)
python3 scripts/populate_milvus.py

# Populate Neo4j (knowledge graph)
python3 scripts/populate_neo4j.py

# Populate AgentDB (RL policies)
python3 scripts/populate_agentdb.py
```

**Output**:
- Milvus: 224K vectors with HNSW indexing
- Neo4j: 500K nodes, 30M relationships
- AgentDB: 162K policies, 25M episodes

### Phase 5: Validate (30 min)

```bash
# Comprehensive validation
python3 scripts/validate_data.py
```

**Checks**:
- Data integrity and alignment
- Database connectivity and performance
- Embedding normalization and dimensionality
- Benchmark query performance (<100ms Milvus, <1s Neo4j)

## Verify Output

```bash
# Check processed data
ls -lh data/processed/media/
ls -lh data/processed/interactions/

# Check synthetic data
ls -lh data/synthetic/users/
ls -lh data/synthetic/platforms/

# Check embeddings
ls -lh data/embeddings/media/
ls -lh data/embeddings/users/

# View stats
cat data/processed/processing_stats.json
cat data/embeddings/embedding_stats.json
```

## Troubleshooting

### Out of Memory (GPU)

Reduce batch size in `scripts/generate_embeddings.py`:
```python
BATCH_SIZE = 256  # Instead of 512
```

### Pandas Chunking Errors

For large files, increase chunk size or available RAM.

### Missing Dependencies

```bash
pip install -r scripts/requirements.txt --upgrade
```

## Next Steps

After data generation is complete:

1. **Start Databases**: `docker-compose up -d`
2. **Populate Databases**: Run populate_*.py scripts
3. **Test API**: `cd src/api && cargo run`
4. **Query System**: Try search API endpoints

## Performance Benchmarks

**Local Development (ml-latest-small)**:
- Parse: 10 seconds
- Synthetic: 30 seconds
- Embeddings (GPU): 5 seconds
- **Total**: ~1 minute

**A100 Production (ml-25m)**:
- Parse: 30 minutes
- Synthetic: 75 minutes
- Embeddings (A100): 8 minutes
- Database Population: 2-3 hours
- **Total**: ~4.5 hours

## File Sizes

```
data/raw/ml-25m/           1.1 GB
data/processed/            2.0 GB
data/synthetic/            500 MB
data/embeddings/           200 MB
---
Total:                     3.8 GB
```

## Contact

Issues: https://github.com/jjohare/hackathon-tv5/issues

---

**Last Updated**: 2025-12-06
