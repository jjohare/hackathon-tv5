# Synthetic Dataset Generation Pipeline with NVIDIA DataDesigner

**Investigation Date**: 2025-12-04
**Target**: Generate 100M media items + 10M users + 1B interactions
**Technology**: NVIDIA DataDesigner (NeMo)

---

## Executive Summary

**VERDICT**: DataDesigner is **HIGHLY SUITABLE** for this use case with some architectural considerations.

**Key Strengths**:
- ✅ Purpose-built for large-scale synthetic data (no hardcoded limits found)
- ✅ Concurrent LLM generation (configurable `max_parallel_requests`)
- ✅ Batch processing with configurable buffer sizes
- ✅ Statistical samplers for realistic distributions (Zipf, Gaussian, Poisson)
- ✅ Multi-modal support (can reference external embeddings)
- ✅ Dependency-aware column generation (DAG-based)
- ✅ Built-in validation and quality scoring
- ✅ Integration with NeMo Microservices for enterprise scale

**Limitations**:
- ⚠️ No native embedding generation (needs external model integration)
- ⚠️ GPU acceleration through LLM APIs only (not direct CUDA)
- ⚠️ Scale to 100M requires batching strategy + external storage
- ⚠️ Multi-language text needs prompt engineering or multiple model configs

---

## DataDesigner Capability Assessment

### 1. Core Architecture

```
┌─────────────────────────────────────────────────────────┐
│              DataDesigner Framework                     │
├─────────────────────────────────────────────────────────┤
│  Config Layer (Declarative Schema)                     │
│  ├─ SamplerColumns (Statistical)                       │
│  ├─ LLMTextColumns (Natural Language)                  │
│  ├─ LLMStructuredColumns (JSON Schema)                 │
│  ├─ ExpressionColumns (Jinja2 Transforms)              │
│  └─ ValidationColumns (Quality Control)                │
├─────────────────────────────────────────────────────────┤
│  Engine Layer (Execution)                              │
│  ├─ DAG-based dependency resolution                    │
│  ├─ ConcurrentThreadExecutor (max_workers)             │
│  ├─ DatasetBatchManager (buffer_size)                  │
│  ├─ Model Registry (LiteLLM integration)               │
│  └─ Artifact Storage (Parquet output)                  │
├─────────────────────────────────────────────────────────┤
│  LLM Backends                                           │
│  ├─ NVIDIA NIM (build.nvidia.com)                      │
│  ├─ OpenAI API                                          │
│  ├─ Any LiteLLM-compatible provider                    │
│  └─ NeMo Microservices (enterprise)                    │
└─────────────────────────────────────────────────────────┘
```

### 2. Scalability Features

**Batch Processing**:
- `buffer_size`: Records processed in memory before writing (default: 1000)
- Automatic chunking for large datasets
- Incremental Parquet writing

**Concurrency**:
- `max_parallel_requests`: Concurrent LLM API calls (default: 4, configurable per model)
- `ThreadPoolExecutor` for non-LLM operations (default: 4 workers)
- Context-aware thread management

**Error Handling**:
- Automatic error rate monitoring (50% threshold, 10-record window)
- Early shutdown on excessive failures
- Detailed error tracking and reporting

### 3. Statistical Samplers Available

| Sampler | Use Case | Distribution |
|---------|----------|--------------|
| **Category** | Genres, tags, content types | Discrete with weights (Zipf-like) |
| **Uniform** | Random numeric values | Uniform(low, high) |
| **Gaussian** | Ratings, engagement scores | N(μ, σ²) |
| **Poisson** | View counts, interaction rates | Poisson(λ) |
| **Bernoulli** | Binary flags (featured/not) | Bernoulli(p) |
| **Datetime** | Timestamps, release dates | Uniform time range |
| **Person** | User demographics | Faker library |
| **Subcategory** | Hierarchical taxonomy | Conditional sampling |
| **Scipy** | Custom distributions | Full scipy.stats access |

### 4. LLM Integration

**Supported Tasks**:
- Text generation (titles, descriptions, reviews)
- Structured JSON (multi-field objects)
- Code generation (not needed for our use case)
- Judge scoring (quality assessment)

**Prompt Engineering**:
- Jinja2 templating for dynamic prompts
- Reference other columns: `{{ product_category }}`
- Access nested fields: `{{ user.psychographic.valence }}`
- Conditional logic in templates

**Multi-Language Support**:
- Requires prompt engineering: "Generate in [Spanish/French/...]"
- Or multiple model configs with different system prompts
- No built-in localization layer

### 5. Validation & Quality Control

**Built-in Validators**:
- Python code validation (syntax, linting)
- SQL validation (syntax checking)
- Remote HTTP validators (external services)
- Custom callable validators

**LLM-Judge Scoring**:
- Multi-dimensional rubrics (relevance, quality, accuracy)
- 1-5 scales or categorical grades
- Post-generation filtering (keep only high-quality items)

---

## Synthetic Dataset Design for Media Recommendation System

### Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    Generation Pipeline                          │
├─────────────────────────────────────────────────────────────────┤
│  Stage 1: Content Taxonomy (1M records, seed dataset)          │
│  ├─ Genres (from GMC-O ontology)                                │
│  ├─ Moods, themes, cultural contexts                            │
│  └─ Hierarchical category structure                             │
├─────────────────────────────────────────────────────────────────┤
│  Stage 2: Media Content (100M records, 100 batches × 1M)       │
│  ├─ Titles (multi-language, LLM-generated)                      │
│  ├─ Descriptions (LLM with cultural context)                    │
│  ├─ Metadata (sampled: year, duration, ratings)                 │
│  ├─ Popularity (Zipf distribution)                              │
│  └─ Placeholder embeddings (external generation)                │
├─────────────────────────────────────────────────────────────────┤
│  Stage 3: User Profiles (10M records, 10 batches × 1M)         │
│  ├─ Demographics (Person sampler)                               │
│  ├─ Psychographic states (VAD: structured JSON)                 │
│  ├─ Taste clusters (category sampler)                           │
│  └─ Activity patterns (datetime + Poisson)                      │
├─────────────────────────────────────────────────────────────────┤
│  Stage 4: Interactions (1B records, 1000 batches × 1M)         │
│  ├─ User-content pairs (stratified sampling)                    │
│  ├─ Watch time (Gaussian clipped)                               │
│  ├─ Completion rate (Bernoulli mixture)                         │
│  └─ Temporal patterns (datetime with trends)                    │
├─────────────────────────────────────────────────────────────────┤
│  Stage 5: Embeddings (external process)                        │
│  ├─ Visual embeddings (CLIP/SigLIP models)                      │
│  ├─ Text embeddings (multilingual-e5 or BGE)                    │
│  ├─ Audio embeddings (CLAP or custom)                           │
│  └─ Join with content via content_id                            │
└─────────────────────────────────────────────────────────────────┘
```

### Detailed Configuration Example

#### **Stage 1: Taxonomy Seed Dataset**

```python
from data_designer.essentials import (
    DataDesigner,
    DataDesignerConfigBuilder,
    CategorySamplerParams,
    LLMStructuredColumnConfig,
    SamplerColumnConfig,
    SamplerType,
    ModelConfig,
    InferenceParameters,
)
import pandas as pd

# Initialize DataDesigner
designer = DataDesigner(
    artifact_path="./artifacts/taxonomy",
    managed_assets_path="./managed_assets"
)

# Model configuration for fast generation
model_configs = [
    ModelConfig(
        alias="nvidia-fast",
        model="nvidia/nvidia-nemotron-nano-9b-v2",
        provider="nvidia",
        inference_parameters=InferenceParameters(
            temperature=0.7,
            top_p=0.9,
            max_tokens=512,
            max_parallel_requests=16  # High concurrency for seed data
        )
    )
]

# Build taxonomy configuration
config = DataDesignerConfigBuilder(model_configs=model_configs)

# Base genre taxonomy (from GMC-O)
config.add_column(
    SamplerColumnConfig(
        name="genre_primary",
        sampler_type=SamplerType.CATEGORY,
        params=CategorySamplerParams(
            values=[
                "Action", "Adventure", "Comedy", "Drama", "Horror",
                "Sci-Fi", "Fantasy", "Documentary", "Animation", "Thriller",
                "Romance", "Mystery", "War", "Western", "Musical"
            ],
            weights=[8, 7, 10, 12, 6, 7, 6, 5, 8, 7, 9, 5, 3, 2, 4]  # Realistic distribution
        )
    )
)

# Subcategories (conditional on primary genre)
config.add_column(
    SamplerColumnConfig(
        name="genre_secondary",
        sampler_type=SamplerType.SUBCATEGORY,
        params=SubcategorySamplerParams(
            category="genre_primary",
            values={
                "Action": ["Superhero", "Martial Arts", "Spy", "Military", "Disaster"],
                "Drama": ["Historical", "Legal", "Medical", "Family", "Political"],
                "Comedy": ["Romantic", "Dark", "Slapstick", "Satire", "Parody"],
                "Sci-Fi": ["Cyberpunk", "Space Opera", "Time Travel", "Dystopian", "AI"],
                # ... complete mapping for all genres
            }
        )
    )
)

# Moods (LLM-generated structured data)
config.add_column(
    LLMStructuredColumnConfig(
        name="moods",
        model_alias="nvidia-fast",
        schema={
            "type": "object",
            "properties": {
                "primary_mood": {"type": "string", "enum": ["Uplifting", "Tense", "Melancholic", "Exciting", "Calm"]},
                "intensity": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                "secondary_moods": {"type": "array", "items": {"type": "string"}, "maxItems": 3}
            },
            "required": ["primary_mood", "intensity"]
        },
        prompt="""Generate mood descriptors for a {{ genre_primary }} / {{ genre_secondary }} film.
        Consider the typical emotional tone and pacing of this genre combination."""
    )
)

# Cultural contexts
config.add_column(
    SamplerColumnConfig(
        name="cultural_origin",
        sampler_type=SamplerType.CATEGORY,
        params=CategorySamplerParams(
            values=["North America", "Europe", "East Asia", "South Asia", "Latin America", "Middle East", "Africa", "Oceania"],
            weights=[30, 25, 20, 8, 10, 4, 2, 1]  # Reflects production distribution
        )
    )
)

# Generate taxonomy (1M unique combinations)
results = designer.create(config, num_records=1_000_000, dataset_name="taxonomy_seed")
taxonomy_df = results.load_dataset()
```

#### **Stage 2: Media Content Generation**

```python
# Media content with multi-language support
config_media = DataDesignerConfigBuilder(model_configs=model_configs)

# Use taxonomy as seed dataset
config_media.with_seed_dataset(
    seed_reference=designer.make_seed_reference_from_dataframe(
        taxonomy_df,
        "./artifacts/taxonomy/seed.parquet"
    )
)

# Content ID
config_media.add_column(
    SamplerColumnConfig(
        name="content_id",
        sampler_type=SamplerType.UUID
    )
)

# Sample from taxonomy seed
config_media.add_column(
    SeedDatasetColumnConfig(
        name="genre_taxonomy",
        seed_dataset_column="genre_primary"  # Reference seed columns
    )
)

# Language/Region
config_media.add_column(
    SamplerColumnConfig(
        name="primary_language",
        sampler_type=SamplerType.CATEGORY,
        params=CategorySamplerParams(
            values=["en", "es", "fr", "de", "ja", "ko", "zh", "hi", "ar", "pt"],
            weights=[40, 12, 8, 6, 10, 7, 9, 4, 2, 2]
        )
    )
)

# Title generation (multi-language via prompt)
config_media.add_column(
    LLMTextColumnConfig(
        name="title",
        model_alias="nvidia-fast",
        prompt="""Generate a creative film title in {{ primary_language }} language.
        Genre: {{ genre_taxonomy }}
        Cultural context: {{ cultural_origin }}
        Style: Authentic and engaging, matching the language's cultural norms.
        Output ONLY the title, no quotes or explanations.""",
        system_prompt="You are a multilingual creative writer specializing in film titles."
    )
)

# Description (multi-paragraph)
config_media.add_column(
    LLMTextColumnConfig(
        name="description",
        model_alias="nvidia-fast",
        prompt="""Write a compelling 3-sentence synopsis for a {{ genre_taxonomy }} film titled "{{ title }}".
        Language: {{ primary_language }}
        Target mood: {{ moods.primary_mood }}
        Cultural setting: {{ cultural_origin }}

        Make it engaging and suitable for {{ primary_language }}-speaking audiences."""
    )
)

# Release year (temporal distribution)
config_media.add_column(
    SamplerColumnConfig(
        name="release_year",
        sampler_type=SamplerType.UNIFORM,
        params=UniformSamplerParams(low=1970, high=2024),
        convert_to="int"
    )
)

# Duration (Gaussian by genre)
config_media.add_column(
    SamplerColumnConfig(
        name="duration_minutes",
        sampler_type=SamplerType.GAUSSIAN,
        params=GaussianSamplerParams(
            mean=105,  # Base mean
            std=20,
            conditional_params={
                "genre_taxonomy": {
                    "Documentary": {"mean": 85, "std": 15},
                    "Animation": {"mean": 95, "std": 10},
                    "Action": {"mean": 120, "std": 15}
                }
            }
        ),
        convert_to="int"
    )
)

# Popularity (Zipf-like distribution)
config_media.add_column(
    SamplerColumnConfig(
        name="popularity_score",
        sampler_type=SamplerType.SCIPY,
        params=ScipySamplerParams(
            distribution="zipfian",
            a=2.0  # Power law exponent
        )
    )
)

# Rating (clipped Gaussian)
config_media.add_column(
    SamplerColumnConfig(
        name="avg_rating",
        sampler_type=SamplerType.GAUSSIAN,
        params=GaussianSamplerParams(
            mean=6.5,
            std=1.2,
            min_value=1.0,
            max_value=10.0
        )
    )
)

# Placeholder for embeddings (will be generated externally)
config_media.add_column(
    ExpressionColumnConfig(
        name="embedding_visual_id",
        expression="{{ content_id }}",  # Reference for joining later
        drop=False
    )
)

config_media.add_column(
    ExpressionColumnConfig(
        name="embedding_text_id",
        expression="{{ content_id }}"
    )
)

# Quality validation
config_media.add_column(
    LLMJudgeColumnConfig(
        name="quality_score",
        model_alias="nvidia-fast",
        targets=["title", "description"],
        rubric={
            "coherence": {"scale": "1-5", "description": "Title and description match genre"},
            "language_quality": {"scale": "1-5", "description": "Native-like language quality"},
            "engagement": {"scale": "1-5", "description": "Would attract viewers"}
        }
    )
)

# Generate in batches (100 batches × 1M = 100M)
for batch_id in range(100):
    print(f"Generating media content batch {batch_id + 1}/100")

    results = designer.create(
        config_media,
        num_records=1_000_000,
        dataset_name=f"media_content_batch_{batch_id:03d}"
    )

    # Optional: Filter low-quality records
    df = results.load_dataset()
    df_filtered = df[df["quality_score.overall"] >= 3.5]  # Keep only quality >= 3.5

    # Save to distributed storage (S3, GCS, etc.)
    df_filtered.to_parquet(f"s3://datasets/media/content_{batch_id:03d}.parquet")
```

#### **Stage 3: User Profiles**

```python
config_users = DataDesignerConfigBuilder(model_configs=model_configs)

# User ID
config_users.add_column(
    SamplerColumnConfig(
        name="user_id",
        sampler_type=SamplerType.UUID
    )
)

# Demographics (Faker-based)
config_users.add_column(
    SamplerColumnConfig(
        name="demographics",
        sampler_type=SamplerType.PERSON_FROM_FAKER,
        params=PersonFromFakerSamplerParams(
            locale="en_US",
            age_range=[13, 80],
            include_fields=["first_name", "last_name", "age", "email", "city", "state", "country"]
        )
    )
)

# Psychographic state (VAD model: Valence, Arousal, Dominance)
config_users.add_column(
    LLMStructuredColumnConfig(
        name="psychographic_profile",
        model_alias="nvidia-fast",
        schema={
            "type": "object",
            "properties": {
                "valence": {"type": "number", "minimum": -1.0, "maximum": 1.0, "description": "Positive to negative"},
                "arousal": {"type": "number", "minimum": -1.0, "maximum": 1.0, "description": "Calm to excited"},
                "dominance": {"type": "number", "minimum": -1.0, "maximum": 1.0, "description": "Submissive to dominant"},
                "personality_traits": {
                    "type": "array",
                    "items": {"type": "string"},
                    "maxItems": 5
                }
            },
            "required": ["valence", "arousal", "dominance"]
        },
        prompt="""Generate a realistic psychographic profile for a {{ demographics.age }}-year-old from {{ demographics.country }}.
        Consider typical personality distributions and cultural factors."""
    )
)

# Taste clusters
config_users.add_column(
    SamplerColumnConfig(
        name="taste_cluster",
        sampler_type=SamplerType.CATEGORY,
        params=CategorySamplerParams(
            values=[
                "Mainstream Blockbuster",
                "Indie Art House",
                "Genre Specialist",
                "Foreign Cinema Enthusiast",
                "Classic Film Buff",
                "Family Entertainment",
                "Documentary Watcher",
                "Casual Viewer"
            ],
            weights=[25, 8, 15, 7, 5, 18, 6, 16]
        )
    )
)

# Preferred genres (multi-select based on taste cluster)
config_users.add_column(
    LLMStructuredColumnConfig(
        name="preferred_genres",
        model_alias="nvidia-fast",
        schema={
            "type": "array",
            "items": {"type": "string"},
            "minItems": 2,
            "maxItems": 5
        },
        prompt="""Select 2-5 film genres that would appeal to a {{ taste_cluster }} viewer.
        Consider their psychographic profile: valence={{ psychographic_profile.valence }}, arousal={{ psychographic_profile.arousal }}.
        Return a JSON array of genre names."""
    )
)

# Activity level (Poisson-distributed)
config_users.add_column(
    SamplerColumnConfig(
        name="monthly_watch_count",
        sampler_type=SamplerType.POISSON,
        params=PoissonSamplerParams(
            lambda_param=12.0,  # Average 12 watches/month
            conditional_params={
                "taste_cluster": {
                    "Casual Viewer": {"lambda_param": 4.0},
                    "Genre Specialist": {"lambda_param": 20.0},
                    "Classic Film Buff": {"lambda_param": 15.0}
                }
            }
        )
    )
)

# Registration date
config_users.add_column(
    SamplerColumnConfig(
        name="registration_date",
        sampler_type=SamplerType.DATETIME,
        params=DatetimeSamplerParams(
            start="2020-01-01",
            end="2024-12-01",
            unit="D"
        )
    )
)

# Generate users (10 batches × 1M = 10M)
for batch_id in range(10):
    results = designer.create(
        config_users,
        num_records=1_000_000,
        dataset_name=f"users_batch_{batch_id:02d}"
    )

    df = results.load_dataset()
    df.to_parquet(f"s3://datasets/users/users_{batch_id:02d}.parquet")
```

#### **Stage 4: Interactions Generation**

```python
# This is the most challenging stage due to scale (1B records)
# Strategy: Generate in 1000 batches of 1M interactions

config_interactions = DataDesignerConfigBuilder(model_configs=model_configs)

# Load user and content IDs (sampling pools)
user_ids = pd.read_parquet("s3://datasets/users/*.parquet", columns=["user_id", "taste_cluster"])
content_ids = pd.read_parquet("s3://datasets/media/*.parquet", columns=["content_id", "genre_primary", "popularity_score"])

# Pre-compute user-content compatibility matrix (for realistic matching)
# This would be done once as a preprocessing step

# Interaction ID
config_interactions.add_column(
    SamplerColumnConfig(
        name="interaction_id",
        sampler_type=SamplerType.UUID
    )
)

# Stratified sampling of user_id (weighted by activity level)
# Note: This requires custom sampler or expression column with external data
config_interactions.add_column(
    ExpressionColumnConfig(
        name="user_id",
        expression="{{ sample_from_distribution(user_ids, weights=monthly_watch_count) }}",
        # Pseudo-code: actual implementation needs custom sampler
    )
)

# Content_id sampling (weighted by popularity)
config_interactions.add_column(
    ExpressionColumnConfig(
        name="content_id",
        expression="{{ sample_from_distribution(content_ids, weights=popularity_score) }}"
    )
)

# Timestamp (with temporal patterns)
config_interactions.add_column(
    SamplerColumnConfig(
        name="watch_timestamp",
        sampler_type=SamplerType.DATETIME,
        params=DatetimeSamplerParams(
            start="2020-01-01 00:00:00",
            end="2024-12-01 23:59:59",
            unit="m"  # Minute-level precision
        )
    )
)

# Watch time (content duration dependent)
config_interactions.add_column(
    SamplerColumnConfig(
        name="watch_time_minutes",
        sampler_type=SamplerType.GAUSSIAN,
        params=GaussianSamplerParams(
            mean=0.7,  # 70% of duration (placeholder, needs content join)
            std=0.15,
            min_value=0.01,
            max_value=1.0
        )
    )
)

# Completion indicator
config_interactions.add_column(
    SamplerColumnConfig(
        name="completed",
        sampler_type=SamplerType.BERNOULLI,
        params=BernoulliSamplerParams(
            p=0.65  # 65% completion rate
        ),
        convert_to="bool"
    )
)

# Rating (sparse - only 20% of interactions)
config_interactions.add_column(
    SamplerColumnConfig(
        name="rating_given",
        sampler_type=SamplerType.BERNOULLI_MIXTURE,
        params=BernoulliMixtureSamplerParams(
            components=[
                {"p": 0.0, "weight": 0.80},  # 80% no rating
                {"p": 1.0, "weight": 0.20}   # 20% give rating
            ]
        ),
        convert_to="bool"
    )
)

config_interactions.add_column(
    SamplerColumnConfig(
        name="rating_value",
        sampler_type=SamplerType.UNIFORM,
        params=UniformSamplerParams(low=1, high=10),
        convert_to="int",
        drop_if_null=True  # Only include when rating_given=True
    )
)

# Generate interactions (1000 batches × 1M = 1B)
for batch_id in range(1000):
    if batch_id % 10 == 0:
        print(f"Generating interactions batch {batch_id + 1}/1000")

    results = designer.create(
        config_interactions,
        num_records=1_000_000,
        dataset_name=f"interactions_batch_{batch_id:04d}"
    )

    df = results.load_dataset()
    df.to_parquet(f"s3://datasets/interactions/interactions_{batch_id:04d}.parquet")
```

---

## Embedding Generation Strategy (External to DataDesigner)

DataDesigner does not natively generate embeddings. Use separate pipeline:

### **Option 1: NVIDIA NIM + PyTorch**

```python
import torch
from transformers import CLIPProcessor, CLIPModel
from sentence_transformers import SentenceTransformer

# Visual embeddings (CLIP)
clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

# Text embeddings (multilingual)
text_model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-mpnet-base-v2')

# Process content in batches
for batch_file in glob("s3://datasets/media/*.parquet"):
    df = pd.read_parquet(batch_file)

    # Generate text embeddings (1024-dim)
    text_embeddings = text_model.encode(
        df["title"] + " " + df["description"],
        batch_size=256,
        show_progress_bar=True,
        convert_to_numpy=True
    )

    # Visual embeddings (requires poster images - mock for now)
    # In production: download posters, process with CLIP

    # Create embeddings dataframe
    embeddings_df = pd.DataFrame({
        "content_id": df["content_id"],
        "text_embedding": list(text_embeddings),
        # "visual_embedding": list(visual_embeddings),
    })

    embeddings_df.to_parquet(f"s3://datasets/embeddings/{batch_file.name}")
```

### **Option 2: NeMo Embeddings Service**

```bash
# Use NVIDIA NeMo Embeddings Microservice
curl -X POST https://integrate.api.nvidia.com/v1/embeddings \
  -H "Authorization: Bearer $NVIDIA_API_KEY" \
  -d '{
    "input": ["Film title and description text"],
    "model": "nvidia/nv-embed-v2",
    "input_type": "passage"
  }'
```

---

## Quality Metrics & Validation

### 1. Automated Quality Checks

**During Generation**:
- LLM-Judge scoring (keep quality_score >= 3.5)
- Language validation (detect gibberish)
- Schema validation (structured JSON)
- Duplicate detection (title similarity)

**Post-Generation**:
- Distribution checks (Kolmogorov-Smirnov tests)
- Correlation analysis (genre vs. popularity)
- Realism scoring (compare to real data statistics)

### 2. Data Quality Metrics

```python
from data_designer import DataDesignerDatasetProfiler

# Profile generated dataset
profiler = DataDesignerDatasetProfiler()
analysis = profiler.profile(dataset_path="./artifacts/media_content_batch_000/dataset.parquet")

# View statistics
print(analysis.to_report())

# Key metrics to monitor:
# - Null value rates (should be < 1%)
# - Distribution skewness (popularity should be Zipf-like)
# - Cardinality (unique titles/descriptions)
# - LLM token usage and costs
```

---

## Scale-Up Strategy & Performance

### Estimated Generation Times

**Assumptions**:
- LLM throughput: 50 tokens/sec per request
- Concurrency: 16 parallel requests
- Average tokens per record: 200 (title + description)

**Calculations**:

| Stage | Records | Batches | Tokens/Record | Time/Batch | Total Time |
|-------|---------|---------|---------------|------------|------------|
| Taxonomy | 1M | 1 | 150 | 5.2 hours | **5.2 hours** |
| Media | 100M | 100 | 200 | 6.9 hours | **29 days** |
| Users | 10M | 10 | 180 | 6.2 hours | **2.6 days** |
| Interactions | 1B | 1000 | 50 | 1.7 hours | **71 days** |

**Total Sequential Time**: ~103 days

### Optimization Strategies

**1. Increase Concurrency**:
```python
# Boost parallel requests (requires higher API rate limits)
InferenceParameters(
    max_parallel_requests=64  # 4x speedup -> ~26 days total
)
```

**2. Use Faster Models**:
- Switch to Nemotron-Nano (2x faster) for non-critical fields
- Reserve larger models for titles/descriptions only

**3. Parallelize Batches**:
```bash
# Run multiple batch jobs simultaneously on different machines
python generate_media.py --batch_start=0 --batch_end=25 &   # Machine 1
python generate_media.py --batch_start=25 --batch_end=50 &  # Machine 2
python generate_media.py --batch_start=50 --batch_end=75 &  # Machine 3
python generate_media.py --batch_start=75 --batch_end=100 & # Machine 4
# 4x parallelism -> ~6.5 days for media content
```

**4. Hybrid Approach**:
- Generate 10M high-quality seed records with LLMs (detailed, diverse)
- Use template-based expansion for remaining 90M (faster, cheaper)
- Apply small LLM variations to avoid exact duplicates

**Optimized Timeline**: ~7-10 days with 4-machine parallelization

### Cost Estimation

**NVIDIA NIM Pricing** (estimated):
- $0.001 per 1K tokens (Nemotron-Nano)
- $0.005 per 1K tokens (Larger models)

**Total Tokens**:
- Media: 100M × 200 = 20B tokens → **$20,000**
- Users: 10M × 180 = 1.8B tokens → **$1,800**
- Interactions: 1B × 50 = 50B tokens → **$50,000**

**Total LLM Costs**: ~$72,000 (can be reduced with hybrid approach)

---

## Integration with Existing System

### Milvus Vector Database

```python
from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType

# Connect to Milvus
connections.connect("default", host="localhost", port="19530")

# Define schema
fields = [
    FieldSchema(name="content_id", dtype=DataType.VARCHAR, max_length=36, is_primary=True),
    FieldSchema(name="text_embedding", dtype=DataType.FLOAT_VECTOR, dim=1024),
    FieldSchema(name="visual_embedding", dtype=DataType.FLOAT_VECTOR, dim=768),
    FieldSchema(name="genre_primary", dtype=DataType.VARCHAR, max_length=50),
    FieldSchema(name="popularity_score", dtype=DataType.FLOAT),
]

schema = CollectionSchema(fields, description="Media content embeddings")
collection = Collection("media_content", schema)

# Insert generated embeddings (batched)
for batch_file in glob("s3://datasets/embeddings/*.parquet"):
    df = pd.read_parquet(batch_file)

    collection.insert([
        df["content_id"].tolist(),
        df["text_embedding"].tolist(),
        df["visual_embedding"].tolist(),
        df["genre_primary"].tolist(),
        df["popularity_score"].tolist(),
    ])

# Create HNSW index
collection.create_index(
    field_name="text_embedding",
    index_params={
        "metric_type": "COSINE",
        "index_type": "HNSW",
        "params": {"M": 16, "efConstruction": 256}
    }
)
```

### PostgreSQL Metadata

```sql
-- Import content metadata
COPY media_content (content_id, title, description, genre_primary, release_year, duration_minutes, avg_rating, popularity_score)
FROM PROGRAM 'aws s3 cp s3://datasets/media/*.parquet - | parquet-tools cat -'
WITH (FORMAT parquet);

-- Import users
COPY users (user_id, first_name, last_name, age, email, taste_cluster, registration_date)
FROM PROGRAM 'aws s3 cp s3://datasets/users/*.parquet - | parquet-tools cat -'
WITH (FORMAT parquet);

-- Import interactions
COPY interactions (interaction_id, user_id, content_id, watch_timestamp, watch_time_minutes, completed, rating_value)
FROM PROGRAM 'aws s3 cp s3://datasets/interactions/*.parquet - | parquet-tools cat -'
WITH (FORMAT parquet);

-- Create indexes
CREATE INDEX idx_content_genre ON media_content(genre_primary);
CREATE INDEX idx_content_popularity ON media_content(popularity_score DESC);
CREATE INDEX idx_interactions_user ON interactions(user_id, watch_timestamp DESC);
CREATE INDEX idx_interactions_content ON interactions(content_id);
```

---

## Alternative Technologies Comparison

| Technology | Pros | Cons | Verdict |
|------------|------|------|---------|
| **DataDesigner** | ✅ Purpose-built<br>✅ Statistical samplers<br>✅ LLM integration<br>✅ DAG dependencies | ⚠️ No native embeddings<br>⚠️ Requires batching at scale | **RECOMMENDED** |
| **Gretel.ai** | ✅ Privacy-focused<br>✅ Learned distributions<br>✅ SaaS platform | ❌ Less control<br>❌ Proprietary<br>💰 Expensive | Use for sensitive data only |
| **SDV (Synthetic Data Vault)** | ✅ Statistical modeling<br>✅ Open source<br>✅ Table relationships | ❌ No LLM support<br>❌ Limited text generation | Supplement for numeric data |
| **CTGAN** | ✅ Deep learning<br>✅ Learns from data | ❌ Requires real data<br>❌ No conditional generation<br>❌ No text | Not suitable |
| **Custom Pipeline** | ✅ Full control<br>✅ Optimized for use case | ❌ High development time<br>❌ Maintenance burden | Only if DataDesigner insufficient |

**Recommendation**: **Use DataDesigner as primary tool**, supplement with:
- External embedding generation (NeMo/HuggingFace)
- SDV for purely numeric interaction patterns (optional)
- Custom post-processing for quality filtering

---

## Implementation Roadmap

### Phase 1: Proof of Concept (Week 1-2)
- [ ] Setup DataDesigner environment
- [ ] Generate 1K taxonomy entries
- [ ] Generate 10K media content samples
- [ ] Generate 1K user profiles
- [ ] Generate 100K interactions
- [ ] Validate data quality
- [ ] Benchmark generation speed

### Phase 2: Pilot Scale (Week 3-4)
- [ ] Generate 100K taxonomy (full GMC-O coverage)
- [ ] Generate 1M media content
- [ ] Generate 100K users
- [ ] Generate 10M interactions
- [ ] Integrate embeddings generation
- [ ] Load into Milvus + PostgreSQL
- [ ] Test recommendation system

### Phase 3: Production Scale (Week 5-8)
- [ ] Parallelize generation across 4-8 machines
- [ ] Generate 100M media content (100 batches)
- [ ] Generate 10M users (10 batches)
- [ ] Generate 1B interactions (1000 batches)
- [ ] Quality assurance checks
- [ ] Data distribution validation
- [ ] System integration testing

### Phase 4: Continuous Generation (Ongoing)
- [ ] Automated monthly data refresh
- [ ] Incremental taxonomy updates
- [ ] Trend-based content generation
- [ ] User cohort expansion
- [ ] Model retraining with synthetic data

---

## Conclusion

**DataDesigner is HIGHLY SUITABLE** for generating the required 100M media items, 10M users, and 1B interactions for the media recommendation system.

**Key Advantages**:
1. **Scale**: No artificial limits, proven for large datasets
2. **Quality**: Built-in validation, LLM-judge scoring, statistical realism
3. **Flexibility**: Hierarchical dependencies, conditional sampling, multi-modal support
4. **Integration**: Parquet output, batch processing, fits existing infrastructure
5. **NVIDIA Ecosystem**: Direct access to NIM models, potential NeMo Microservices upgrade

**Critical Success Factors**:
1. **Parallelization**: Run 4-8 batch jobs simultaneously (7-10 days vs. 103 days)
2. **Model Selection**: Use fast models (Nemotron-Nano) for bulk, reserve larger models for quality
3. **Hybrid Strategy**: LLM for seed data (10M high-quality), templates for scale (90M variations)
4. **External Embeddings**: Separate pipeline using HuggingFace/NeMo for 1024-dim vectors
5. **Quality Gating**: LLM-Judge filtering (keep score >= 3.5) to maintain dataset quality

**Estimated Costs**: $72K LLM generation + $10K compute + $5K storage = **~$87K total**

**Recommended Next Steps**:
1. Run Phase 1 POC (10K records) to validate architecture
2. Benchmark actual throughput with target models
3. Refine configuration based on quality metrics
4. Scale to Phase 2 pilot (1M records) for integration testing
5. Proceed to Phase 3 production scale with parallel execution

**No alternative technology offers better fit for this use case.**
