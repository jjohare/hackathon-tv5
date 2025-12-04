# Synthetic Dataset Generation Pipeline

Complete implementation for generating 100M media items with embeddings using NVIDIA DataDesigner.

## Architecture

### Data Scale
- **Media Items**: 100,000,000
- **Users**: 10,000,000
- **Interactions**: 1,000,000,000
- **Embeddings**: 1024-dimensional multimodal vectors

### Technology Stack
- **Generation**: NVIDIA DataDesigner
- **Embeddings**: CLIP (visual) + Sentence-BERT (text) + Synthetic audio
- **Vector DB**: Milvus with HNSW indexing
- **Relational DB**: PostgreSQL with JSONB support
- **Format**: Parquet with Snappy compression

## Files Created

### Configuration
```
/home/devuser/workspace/hackathon-tv5/config/datadesigner/media_dataset.yaml
```
Complete DataDesigner configuration with 5 generation stages:
1. Taxonomy generation (GMC-O ontology)
2. Media content generation (100M records)
3. User generation (10M records)
4. Interaction generation (1B records)
5. Quality validation

### Scripts

#### 1. Embedding Generation
```bash
/home/devuser/workspace/hackathon-tv5/scripts/generate_embeddings.py
```
Generates 1024-dimensional embeddings:
- Text features (384 dims): Title + Description via Sentence-BERT
- Visual features (512 dims): Synthetic via CLIP text prompts
- Audio features (128 dims): Mood-based synthetic vectors

**Usage:**
```bash
python scripts/generate_embeddings.py \
    --input-dir /data/synthetic/tv5_media/media \
    --output-dir data/embedded \
    --num-workers 8
```

#### 2. DataDesigner Execution
```bash
/home/devuser/workspace/hackathon-tv5/scripts/run_datadesigner.sh
```
Orchestrates complete pipeline:
- Parallel batch generation (8 concurrent jobs)
- 100 batches of 1M media items each
- Streaming interaction generation
- Automatic quality validation

**Usage:**
```bash
./scripts/run_datadesigner.sh
```

#### 3. Milvus Loader
```bash
/home/devuser/workspace/hackathon-tv5/scripts/load_to_milvus.py
```
Loads embeddings to Milvus vector database:
- HNSW indexing (M=16, efConstruction=200)
- Cosine similarity metric
- Batch insertion (10k records/batch)

**Usage:**
```bash
python scripts/load_to_milvus.py \
    --data-dir data/embedded \
    --collection-name media_embeddings \
    --batch-size 10000
```

#### 4. PostgreSQL Loader
```bash
/home/devuser/workspace/hackathon-tv5/scripts/load_to_postgres.py
```
Loads metadata to PostgreSQL:
- Media content with JSONB cultural context
- User demographics and psychographics
- Interaction history with timestamps
- Optimized indexes for common queries

**Usage:**
```bash
python scripts/load_to_postgres.py \
    --data-dir /data/synthetic/tv5_media \
    --db-host localhost \
    --db-name tv5_media
```

#### 5. Dataset Validator
```bash
/home/devuser/workspace/hackathon-tv5/scripts/validate_dataset.py
```
Validates dataset quality:
- Zipf distribution compliance (popularity)
- Temporal pattern realism (daily peaks)
- Cultural diversity (Shannon entropy)
- Embedding consistency (no NaN/Inf)

**Usage:**
```bash
python scripts/validate_dataset.py --data-dir data/embedded
```

## Embedding Structure

### 1024-Dimensional Vector Breakdown

```
[0:384]    Text Features
           - Semantic meaning from title + description
           - Sentence-BERT (all-mpnet-base-v2)

[384:896]  Visual Features
           - Synthetic visual characteristics from CLIP
           - Generated from genre + mood + cultural context

[896:1024] Audio Features
           - Mood-based synthetic audio profiles
           - Valence-Arousal-Dominance model
```

## Data Schema

### Media Content
```sql
content_id BIGINT PRIMARY KEY
title VARCHAR(255)
description TEXT
genre VARCHAR(100)          -- Hierarchical (e.g., "Action > Martial Arts")
mood VARCHAR(50)            -- Uplifting, Melancholic, Tense, Romantic, Comedic
cultural_context JSONB      -- {language, region, era, themes}
popularity FLOAT            -- Zipf distributed (0.0-1.0)
release_year INTEGER        -- Weighted historical (1920-2024)
embedding FLOAT_VECTOR(1024)
```

### Users
```sql
user_id UUID PRIMARY KEY
demographics JSONB          -- {age, gender, location, languages}
psychographic_state FLOAT[] -- [valence, arousal, dominance]
taste_clusters VARCHAR[]    -- ["mainstream", "arthouse", ...]
```

### Interactions
```sql
user_id UUID
content_id BIGINT
interaction_type VARCHAR    -- view, like, share, save
watch_time FLOAT           -- Beta distributed
timestamp TIMESTAMP        -- Realistic patterns (daily peaks, weekends)
```

## Generation Parameters

### Popularity Distribution
- **Model**: Zipf's Law
- **Alpha**: 1.5
- **Effect**: 80/20 rule - small portion of content gets most views

### Temporal Patterns
- **Daily Peaks**: 8AM, 12PM, 8PM
- **Weekly**: Weekday vs Weekend patterns
- **Seasonal**: Holiday spikes

### Cultural Distribution
- **Languages**: French, English, Arabic, Spanish, Mandarin
- **Regions**: Europe, Africa, Asia, Americas
- **Balance**: Shannon entropy > 1.0

### Mood Distribution
```
Uplifting:    20%
Melancholic:  15%
Tense:        25%
Romantic:     20%
Comedic:      20%
```

## Quality Metrics

### Validation Checks
1. **Distribution Validation**
   - Popularity follows Zipf (alpha ≈ 1.5)
   - Temporal patterns realistic
   - Cultural diversity maintained

2. **Consistency Validation**
   - User interactions consistent
   - Content metadata complete
   - No impossible combinations

3. **Embedding Validation**
   - All 1024 dimensions present
   - No NaN or Infinity values
   - Reasonable value ranges

## Performance Optimization

### Parallel Generation
- 8 concurrent DataDesigner jobs
- 100 batches × 1M records = 100M total
- Estimated time: 24-48 hours (depends on API rate limits)

### Storage Optimization
- Parquet format with Snappy compression
- Partitioning by release_year and genre
- Expected size: ~2TB for full dataset

### Query Optimization
- Milvus HNSW index for vector search (<100ms at scale)
- PostgreSQL indexes on genre, mood, year, popularity
- Hybrid search: Vector + metadata filtering

## Usage Example

### End-to-End Pipeline

```bash
# 1. Generate synthetic data with DataDesigner
./scripts/run_datadesigner.sh

# 2. Generate embeddings
python scripts/generate_embeddings.py

# 3. Load to Milvus
python scripts/load_to_milvus.py

# 4. Load to PostgreSQL
python scripts/load_to_postgres.py

# 5. Validate dataset
python scripts/validate_dataset.py

# 6. Query example
python -c "
from pymilvus import connections, Collection

connections.connect(host='localhost', port='19530')
collection = Collection('media_embeddings')
collection.load()

# Search similar content
results = collection.search(
    data=[query_embedding],
    anns_field='embedding',
    param={'metric_type': 'COSINE', 'params': {'ef': 50}},
    limit=10,
    expr='release_year >= 2020 and popularity > 0.5'
)
"
```

## Testing with Sample Data

All scripts include sample data generation for testing without DataDesigner:

```bash
# Generate sample embeddings
python scripts/generate_embeddings.py --input-dir data/raw

# This will automatically create 100 sample records if no data exists
```

## Dependencies

```bash
# Python packages
pip install torch transformers sentence-transformers
pip install pymilvus psycopg2-binary pyarrow pandas numpy tqdm

# External services
# Milvus: docker run -d --name milvus -p 19530:19530 milvusdb/milvus
# PostgreSQL: docker run -d --name postgres -p 5432:5432 postgres:15
```

## Next Steps

1. Configure DataDesigner API keys in environment
2. Adjust batch sizes based on available compute
3. Monitor generation progress in logs
4. Validate sample batches before full run
5. Set up monitoring for Milvus/PostgreSQL

## References

- NVIDIA DataDesigner: https://github.com/NVIDIA/synthetic-data-generator
- GMC-O Ontology: See `/docs/gmc-o-ontology.md`
- Milvus Documentation: https://milvus.io/docs
- CLIP Model: https://github.com/openai/CLIP
