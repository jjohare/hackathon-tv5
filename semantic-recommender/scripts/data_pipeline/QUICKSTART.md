# TMDB Pipeline - Quick Start Guide

**Process 930k TMDB movies in 17 minutes on A100 GPU**

## Prerequisites (One-time setup)

```bash
# 1. Download TMDB dataset
cd /home/devuser/workspace/hackathon-tv5/semantic-recommender
python scripts/download_tmdb_dataset.py

# 2. Verify TensorRT engine exists
ls -lh data/models/minilm_l12_v2_fp16.plan
# Should show ~50MB file
```

## Run Complete Pipeline

```bash
# Go to pipeline directory
cd scripts/data_pipeline

# Run all 3 stages (takes ~17 minutes)
python run_tmdb_pipeline.py
```

**Expected output:**
```
🎬 TMDB Dataset Processing Pipeline
====================================================================
RUNNING STAGE 1
====================================================================
✅ Loaded 15,000 MovieLens IMDB mappings
Processing movies: 100%|██████████| 930k/930k [00:60<00:00, 15.5k movies/s]
✅ Stage 1 completed successfully

====================================================================
RUNNING STAGE 2
====================================================================
✅ Loaded 1,128 genome tags
Mapping to genome: 100%|██████████| 930k/930k [02:00<00:00, 7.75k movies/s]
✅ Stage 2 completed successfully

====================================================================
RUNNING STAGE 3
====================================================================
✅ Encoder initialized (Using TensorRT: True)
Generating embeddings: 100%|██████████| 930k/930k [15:00<00:00, 1.03k movies/s]
✅ Stage 3 completed successfully

✅ PIPELINE COMPLETED SUCCESSFULLY
Total Time: 1020 seconds (17.0 minutes)
```

## Output Files

After completion:
```
data/processed/tmdb/movies_clean.jsonl       (50 MB)
data/processed/tmdb/genome_scores.json       (100 MB)
data/embeddings/tmdb/content_vectors.npy     (1.4 GB)
data/embeddings/tmdb/metadata.jsonl          (10 MB)
```

## Resume from Checkpoint

If interrupted:
```bash
# Automatically resumes from last checkpoint
python run_tmdb_pipeline.py --resume
```

## Run Individual Stages

```bash
# Stage 1 only (60 seconds)
python stage1_ingest_tmdb.py

# Stage 2 only (120 seconds)
python stage2_ontology_mapping.py

# Stage 3 only (15 minutes)
python stage3_gpu_embeddings.py
```

## Troubleshooting

### Issue: "TMDB dataset not found"
```bash
python scripts/download_tmdb_dataset.py
```

### Issue: "GPU out of memory"
```bash
# Reduce batch size
python stage3_gpu_embeddings.py --batch-size 16
```

### Issue: Checkpoint corruption
```bash
# Delete and restart
rm data/embeddings/tmdb/checkpoint.npz
python stage3_gpu_embeddings.py
```

## Verification

```bash
# Check output files exist
ls -lh data/processed/tmdb/*.jsonl
ls -lh data/embeddings/tmdb/*.npy

# Test embeddings load correctly
python -c "import numpy as np; e = np.load('data/embeddings/tmdb/content_vectors.npy'); print(f'Shape: {e.shape}')"
# Expected: Shape: (930000, 384)
```

## Performance Tuning

```bash
# Larger batch (if you have >16GB GPU memory)
python stage3_gpu_embeddings.py --batch-size 64

# More frequent checkpoints (if unstable)
python stage3_gpu_embeddings.py --checkpoint-interval 5000
```

## Next Steps

After pipeline completes:

1. **Integrate with recommendations**:
   ```python
   from scripts.utils.gpu_ontology_reasoning import GPUOntologyReasoner

   reasoner = GPUOntologyReasoner()
   results = reasoner.hybrid_recommend("tmdb_123456", top_k=10)
   ```

2. **Query hybrid dataset**:
   ```python
   # Search across 930k TMDB + 62k MovieLens = 992k total
   results = search_combined_dataset("action thriller")
   ```

3. **Build recommendation API**:
   ```bash
   cd scripts/server
   python recommendation_server.py
   ```

## Help

```bash
# Pipeline help
python run_tmdb_pipeline.py --help

# Stage-specific help
python stage1_ingest_tmdb.py --help
python stage2_ontology_mapping.py --help
python stage3_gpu_embeddings.py --help
```

## Documentation

- **Full Guide**: `scripts/data_pipeline/README.md`
- **Implementation Report**: `scripts/docs/TMDB_PIPELINE_IMPLEMENTATION.md`
- **Architecture Design**: See repository docs

---

**Ready to run?**
```bash
cd scripts/data_pipeline && python run_tmdb_pipeline.py
```
