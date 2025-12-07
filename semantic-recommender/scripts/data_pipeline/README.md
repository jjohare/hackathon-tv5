# TMDB Dataset Processing Pipeline

Complete GPU-accelerated pipeline for processing 1.3M TMDB movies with semantic embeddings.

**⚠️ Data Quality Notice**: Current dataset contains **movie titles only** (no plot overviews/descriptions). See [DATA_QUALITY_REPORT.md](../../docs/DATA_QUALITY_REPORT.md) for details and enrichment path.

## Overview

This pipeline transforms the TMDB Movies Dataset (1,334,069 movies) into production-ready semantic vectors for the TV5 Media Recommendation System. Infrastructure is proven at scale; semantic depth requires metadata enrichment.

### Pipeline Architecture

```
TMDB CSV (930k × 24 columns)
    ↓
[Stage 1: Ingestion] → movies_clean.jsonl (JSONL format)
    ↓
[Stage 2: Ontology Mapping] → genome_scores.json (semantic tags)
    ↓
[Stage 3: GPU Embeddings] → content_vectors.npy (384-dim vectors)
```

### Performance Targets (A100 GPU)

| Stage | Task | Duration | Throughput |
|-------|------|----------|------------|
| 1 | CSV parsing & cleaning | ~60s | 15,500 movies/sec |
| 2 | Ontology mapping | ~120s | 7,750 movies/sec |
| 3 | GPU embedding generation | ~15min | 1,000 movies/sec |
| **Total** | **Complete pipeline** | **~17min** | **910 movies/sec** |

## Quick Start

### Prerequisites

1. **Download TMDB dataset**:
```bash
cd /home/devuser/workspace/hackathon-tv5/semantic-recommender
python scripts/download_tmdb_dataset.py
```

2. **Verify TensorRT engine exists**:
```bash
ls -lh data/models/minilm_l12_v2_fp16.plan
# Should show ~50MB file
```

### Run Complete Pipeline

```bash
# Run all 3 stages
cd scripts/data_pipeline
python run_tmdb_pipeline.py

# Resume from last checkpoint (skip completed stages)
python run_tmdb_pipeline.py --resume

# Run specific stages only
python run_tmdb_pipeline.py --stages 1 2
```

### Run Individual Stages

```bash
# Stage 1: Ingestion
python stage1_ingest_tmdb.py

# Stage 2: Ontology mapping
python stage2_ontology_mapping.py

# Stage 3: GPU embeddings
python stage3_gpu_embeddings.py --batch-size 32
```

## Stage Details

### Stage 1: TMDB Ingestion

**Purpose**: Parse TMDB CSV, clean missing values, map IMDB IDs to MovieLens IDs

**Input**: `data/raw/tmdb/TMDB_movie_dataset_v11.csv` (930k rows × 24 columns)

**Output**: `data/processed/tmdb/movies_clean.jsonl`

**Features**:
- Handles JSON fields (genres, keywords, production_companies)
- Extracts year from release_date
- Maps IMDB IDs to MovieLens IDs for overlap detection
- Cleans missing/invalid data
- Progress tracking with tqdm

**Key Functions**:
```python
parse_json_field(value)       # Parse JSON arrays from CSV
extract_year(release_date)    # Extract year from various date formats
clean_movie_row(row)          # Transform row to clean dictionary
```

**Actual Output** (verified 2025-12-07):
```json
{
  "tmdb_id": "27205",
  "imdb_id": "tt1375666",
  "ml_id": "ml_79132",
  "title": "Inception",
  "year": 2010,
  "genres": []  // Empty - source data limitation
}
```

**⚠️ Known Limitation**:
- NO `overview` field (source CSV doesn't contain plot descriptions)
- NO `keywords` field (source CSV doesn't contain semantic tags)
- Empty `genres` arrays (field present but unpopulated in source)
- This limits embeddings to title-only matching

### Stage 2: Ontology Mapping

**Purpose**: Map TMDB keywords → MovieLens genome tags using semantic similarity

**Input**:
- `data/processed/tmdb/movies_clean.jsonl`
- `data/processed/media/genome_scores.json` (MovieLens genome scores)

**Output**: `data/processed/tmdb/genome_scores.json`

**Features**:
- Loads MovieLens genome vocabulary (1,128 tags)
- Maps TMDB keywords to genome tags (fuzzy matching)
- Extracts themes from overview text (NER/semantic parsing)
- Reuses MovieLens scores for overlap movies
- Generates scores for TMDB-only movies

**Mapping Strategy**:
1. **MovieLens overlap**: Use existing genome scores (direct copy)
2. **Keyword matching**: Map TMDB keywords → genome tags (confidence: 0.7)
3. **Theme extraction**: Extract themes from overview (confidence: 0.5)
4. **Genre mapping**: Direct genre → tag mapping (confidence: 0.9)

**Example Mappings**:
```python
'revenge' → ['vengeance', 'revenge']
'action' → ['action', 'violence']
'romantic' → ['romantic', 'romance']
```

**Expected Output**:
```json
{
  "123456": {
    "action": 0.9,
    "violence": 0.7,
    "revenge": 0.7,
    "dark": 0.5
  }
}
```

### Stage 3: GPU Embedding Generation

**Purpose**: Generate 384-dim semantic vectors using TensorRT-accelerated model

**Input**:
- `data/processed/tmdb/movies_clean.jsonl`
- `data/models/minilm_l12_v2_fp16.plan` (TensorRT engine)

**Output**:
- `data/embeddings/tmdb/content_vectors.npy` (930k × 384, ~1.4GB)
- `data/embeddings/tmdb/metadata.jsonl`

**Features**:
- TensorRT GPU acceleration (3-5x faster than PyTorch)
- Batch processing (configurable batch_size)
- Checkpointing every 10k movies (resumable)
- Progress tracking with ETA
- Graceful fallback to PyTorch if TensorRT unavailable

**GPU Optimization**:
- Uses FP16 precision (2x memory reduction)
- Zero-copy CUDA memory management
- Batch size: 32 (optimal for A100)
- Expected throughput: 1,000 movies/second

**Text Source** (actual implementation):
```python
# Current reality: Title only (overview field doesn't exist)
text = movie['title']  # e.g., "Inception"

# Desired (requires TMDB API enrichment):
# text = f"{title}. {overview}. {keywords}. Starring {cast}"
```

**Impact on Embeddings**:
- Current: "Inception" (1 token) → 384-dim vector
- Expected with overviews: "Inception. A thief steals secrets through dreams..." (50+ tokens)
- Similarity scores: Current 0.26-0.31 → Expected 0.70-0.90 with enrichment

**Checkpointing**:
```python
# Automatic checkpoint every 10k movies
checkpoint.npz:
  - embeddings: np.ndarray (current progress)
  - processed_count: int (resume index)
```

## Output Files

### Stage 1 Output

**File**: `data/processed/tmdb/movies_clean.jsonl`

**Format**: JSONL (newline-delimited JSON)

**Size**: ~50 MB

**Fields**:
```
tmdb_id, imdb_id, ml_id, title, original_title, overview,
year, release_date, genres, keywords, production_companies,
original_language, popularity, vote_average, vote_count,
runtime, budget, revenue, adult
```

### Stage 2 Output

**File**: `data/processed/tmdb/genome_scores.json`

**Format**: JSON

**Size**: ~100 MB

**Structure**:
```json
{
  "tmdb_id": {
    "tag_name": score,  // 0.0 - 1.0
    ...
  }
}
```

### Stage 3 Output

**File**: `data/embeddings/tmdb/content_vectors.npy`

**Format**: NumPy array (float32)

**Size**: ~1.4 GB

**Shape**: (930000, 384)

**Metadata**: `data/embeddings/tmdb/metadata.jsonl`

## Error Handling

### Checkpointing (Stage 3)

If Stage 3 fails or is interrupted:

1. Checkpoint saved every 10k movies
2. Resume automatically on restart:
```bash
python stage3_gpu_embeddings.py
# Detects checkpoint and resumes from last saved index
```

3. Manual checkpoint management:
```bash
# View checkpoint
python -c "import numpy as np; ck = np.load('data/embeddings/tmdb/checkpoint.npz'); print(f\"Processed: {ck['processed_count']}\")"

# Delete checkpoint (start fresh)
rm data/embeddings/tmdb/checkpoint.npz
```

### Validation

Each stage validates its output:
- **Stage 1**: JSON parsing, required fields, sample records
- **Stage 2**: Tag coverage, score distribution, MovieLens overlap
- **Stage 3**: Array shape, NaN/Inf checks, file size

### Common Issues

**1. TMDB dataset not found**:
```bash
python scripts/download_tmdb_dataset.py
```

**2. TensorRT engine missing**:
```bash
# Falls back to PyTorch (slower but functional)
# Or rebuild engine:
python scripts/build_trt_engine.py
```

**3. GPU out of memory**:
```bash
# Reduce batch size
python stage3_gpu_embeddings.py --batch-size 16
```

**4. MovieLens genome scores missing**:
```bash
# Process MovieLens data first
python scripts/data_pipeline/parse_movielens.py
```

## Integration with Existing System

### Data Schema Compatibility

TMDB movies follow the same schema as MovieLens:

```python
# Both use unified schema
MediaAsset {
  media_id: "tmdb_123456" or "ml_1",
  identifiers: {imdb_id, tmdb_id, ml_id},
  metadata: {title, year, genres, ...},
  genome: {tag_scores},
  embeddings: 384-dim vector
}
```

### Hybrid Recommendations

Use TMDB embeddings alongside MovieLens:

```python
from scripts.utils.gpu_hyper_personalization import HyperPersonalizationEngine

engine = HyperPersonalizationEngine()

# Load TMDB embeddings
tmdb_embeddings = np.load("data/embeddings/tmdb/content_vectors.npy")

# Query both datasets
results = engine.hybrid_recommend(
  user_id="user_123",
  include_tmdb=True  # Search across 930k TMDB + 62k MovieLens
)
```

### Ontology Reasoning

TMDB genome scores enable ontology-based recommendations:

```python
from scripts.utils.gpu_ontology_reasoning import GPUOntologyReasoner

reasoner = GPUOntologyReasoner()

# Find similar movies using hybrid semantic + ontology
results = reasoner.hybrid_recommend(
  query_id="tmdb_123456",
  top_k=10,
  semantic_candidates=100  # Filter-then-boost strategy
)
```

## Performance Benchmarking

### Stage 1: Ingestion

Expected metrics:
```
Total rows: 930,000
Valid movies: ~925,000
MovieLens matches: ~15,000 (1.6%)
Processing time: ~60 seconds
Throughput: 15,500 movies/second
```

### Stage 2: Ontology Mapping

Expected metrics:
```
Total movies: 925,000
MovieLens overlap: 15,000 (reused scores)
TMDB-only: 910,000 (generated scores)
Avg tags per movie: 5-10
Processing time: ~120 seconds
Throughput: 7,750 movies/second
```

### Stage 3: GPU Embeddings

Expected metrics (A100 GPU):
```
Total movies: 925,000
Embedding dimension: 384
Batch size: 32
GPU time: ~900 seconds (15 minutes)
Throughput: 1,000 movies/second
Output size: 1.4 GB
```

### Memory Usage

| Component | Size |
|-----------|------|
| TMDB CSV (raw) | ~500 MB |
| movies_clean.jsonl | ~50 MB |
| genome_scores.json | ~100 MB |
| content_vectors.npy | ~1.4 GB |
| GPU memory (peak) | ~2 GB |

## Advanced Usage

### Custom Batch Size

```bash
# Larger batch for more GPU memory
python stage3_gpu_embeddings.py --batch-size 64

# Smaller batch for limited memory
python stage3_gpu_embeddings.py --batch-size 16
```

### Custom Checkpoint Interval

```bash
# Checkpoint every 5k movies (more frequent)
python stage3_gpu_embeddings.py --checkpoint-interval 5000

# Checkpoint every 20k movies (less overhead)
python stage3_gpu_embeddings.py --checkpoint-interval 20000
```

### Resumable Pipeline

```bash
# Skip completed stages automatically
python run_tmdb_pipeline.py --resume

# Example: If Stage 1 & 2 complete, only runs Stage 3
```

### Parallel Processing

For multiple datasets:
```bash
# Terminal 1: Process TMDB
python run_tmdb_pipeline.py

# Terminal 2: Process MovieLens (simultaneously)
python scripts/data_pipeline/parse_movielens.py
```

## Monitoring and Logging

### Real-time Progress

All stages use `tqdm` for progress bars:
```
Processing movies: 45%|████▌     | 419k/930k [00:27<00:33, 15.3k movies/s]
```

### Log Output

Logs written to stdout with timestamps:
```
2025-12-07 16:30:45 - INFO - ✅ Loaded 1,128 genome tags
2025-12-07 16:30:50 - INFO - 📊 Processing 930,000 movies
2025-12-07 16:31:45 - INFO - ✅ Stage 1 complete!
```

### Pipeline Report

Final report saved to `data/processed/tmdb/pipeline_report.txt`:
```
TMDB DATASET PROCESSING PIPELINE - FINAL REPORT
================================================================
Total Time: 1,020 seconds (17.0 minutes)

Stage Results:
  Stage 1: ✅ COMPLETED - movies_clean.jsonl (48.3 MB)
  Stage 2: ✅ COMPLETED - genome_scores.json (102.1 MB)
  Stage 3: ✅ COMPLETED - content_vectors.npy (1,398.5 MB)
================================================================
```

## Troubleshooting

### Issue: "TMDB dataset not found"

**Solution**:
```bash
python scripts/download_tmdb_dataset.py
```

### Issue: "TensorRT engine not found"

**Solution**:
```bash
# Stage 3 will fallback to PyTorch (slower)
# Or rebuild TensorRT engine:
python scripts/build_trt_engine.py \
  --model "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2" \
  --output data/models/minilm_l12_v2_fp16.plan
```

### Issue: "MovieLens genome scores not found"

**Solution**:
```bash
# Process MovieLens data first
cd scripts/data_pipeline
python parse_movielens.py
python generate_embeddings.py
```

### Issue: GPU out of memory

**Solution**:
```bash
# Reduce batch size
python stage3_gpu_embeddings.py --batch-size 16

# Or clear GPU cache
python -c "import torch; torch.cuda.empty_cache()"
```

### Issue: Checkpoint corruption

**Solution**:
```bash
# Delete checkpoint and restart
rm data/embeddings/tmdb/checkpoint.npz
python stage3_gpu_embeddings.py
```

## Contributing

### Code Style

- Follow existing code patterns
- Use type hints
- Add docstrings
- Include error handling
- Progress bars for long operations

### Testing

```bash
# Test Stage 1
python stage1_ingest_tmdb.py

# Test Stage 2
python stage2_ontology_mapping.py

# Test Stage 3 (small batch)
python stage3_gpu_embeddings.py --batch-size 4

# Test complete pipeline
python run_tmdb_pipeline.py --stages 1 2 3
```

## References

- **TMDB Dataset**: https://www.kaggle.com/datasets/asaniczka/tmdb-movies-dataset-2023-930k-movies
- **MovieLens Genome**: https://grouplens.org/datasets/movielens/25m/
- **TensorRT**: https://developer.nvidia.com/tensorrt
- **Sentence Transformers**: https://www.sbert.net/

---

**Last Updated**: 2025-12-07
**Pipeline Version**: 1.0.0
**Author**: TV5 Development Team
