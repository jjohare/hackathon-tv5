# Data Generation Plan for A100 GPU Deployment

**Target**: Production-ready dataset for TV5 Media Recommendation System
**Hardware**: GCP A100 VM (40GB GPU)
**Timeline**: Optimized for batch processing
**Output**: Multi-database population (Milvus, Neo4j, PostgreSQL, Redis)

---

## Executive Summary

Transform 25M MovieLens ratings + synthetic data into a comprehensive multi-modal recommendation dataset optimized for GPU-accelerated semantic search.

**Key Metrics**:
- 62,000 movies with 384-dim embeddings
- 162,000 user profiles with preference vectors
- 25M interactions for learning
- 15M semantic tag scores (knowledge graph)
- 1M synthetic subtitle segments

---

## Phase 1: MovieLens Data Parsing & Normalization

### 1.1 Parse Raw CSV Files

**Input**: `data/raw/ml-25m/*.csv`
**Output**: `data/processed/{media,users,interactions}/`

```python
# Script: scripts/parse_movielens.py

import pandas as pd
import json
from pathlib import Path

def parse_movies():
    """Parse movies.csv into structured media assets"""
    df = pd.read_csv('data/raw/ml-25m/movies.csv')

    media_assets = []
    for _, row in df.iterrows():
        # Extract year from title
        title, year = extract_year(row['title'])

        asset = {
            'media_id': f"ml_{row['movieId']}",
            'identifiers': {
                'movielens_id': row['movieId'],
                'internal_id': f"media_{row['movieId']:08d}"
            },
            'metadata': {
                'title': title,
                'year': year or 0,
                'language': 'en',  # Default, enhance later
                'country': ['US']   # Default
            },
            'classification': {
                'genres': row['genres'].split('|'),
                'themes': [],  # Will populate from genome
                'moods': [],
                'tags': []
            }
        }
        media_assets.append(asset)

    # Save as JSONL for streaming
    with open('data/processed/media/movies.jsonl', 'w') as f:
        for asset in media_assets:
            f.write(json.dumps(asset) + '\n')

    print(f"Processed {len(media_assets)} movies")

def parse_genome_tags():
    """Parse genome tags and scores for semantic enrichment"""
    tags_df = pd.read_csv('data/raw/ml-25m/genome-tags.csv')
    scores_df = pd.read_csv('data/raw/ml-25m/genome-scores.csv')

    # Create tag lookup
    tag_map = dict(zip(tags_df['tagId'], tags_df['tag']))

    # Group scores by movie
    movie_genomes = {}
    for movie_id, group in scores_df.groupby('movieId'):
        genome = {
            tag_map[row['tagId']]: row['relevance']
            for _, row in group.iterrows()
            if row['relevance'] > 0.5  # Only strong associations
        }
        movie_genomes[movie_id] = genome

    # Save as JSON
    with open('data/processed/media/genome_scores.json', 'w') as f:
        json.dump(movie_genomes, f)

    print(f"Processed genome for {len(movie_genomes)} movies")

def parse_interactions():
    """Parse ratings into interaction events"""
    # Process in chunks for memory efficiency
    chunksize = 1_000_000
    chunks = pd.read_csv('data/raw/ml-25m/ratings.csv', chunksize=chunksize)

    output_path = Path('data/processed/interactions/ratings.jsonl')
    with open(output_path, 'w') as f:
        for chunk_idx, df in enumerate(chunks):
            for _, row in df.iterrows():
                interaction = {
                    'interaction_id': f"int_{row.name}",
                    'user_id': f"user_{row['userId']:08d}",
                    'media_id': f"ml_{row['movieId']}",
                    'timestamp': row['timestamp'],
                    'interaction_type': 'Rate',
                    'rating': row['rating'],
                    'context': {
                        'device': 'unknown',
                        'time_of_day': classify_time(row['timestamp'])
                    }
                }
                f.write(json.dumps(interaction) + '\n')

            print(f"Processed chunk {chunk_idx+1} ({len(df)} ratings)")

    print("All interactions processed")

def parse_tags():
    """Parse user-applied tags"""
    df = pd.read_csv('data/raw/ml-25m/tags.csv')

    tag_interactions = []
    for _, row in df.iterrows():
        interaction = {
            'interaction_id': f"tag_{row.name}",
            'user_id': f"user_{row['userId']:08d}",
            'media_id': f"ml_{row['movieId']}",
            'timestamp': row['timestamp'],
            'interaction_type': 'Tag',
            'tag': row['tag']
        }
        tag_interactions.append(interaction)

    with open('data/processed/interactions/tags.jsonl', 'w') as f:
        for interaction in tag_interactions:
            f.write(json.dumps(interaction) + '\n')

    print(f"Processed {len(tag_interactions)} tag applications")

if __name__ == '__main__':
    parse_movies()
    parse_genome_tags()
    parse_interactions()
    parse_tags()
```

**Execution**:
```bash
cd semantic-recommender
python scripts/parse_movielens.py
```

**Output Files**:
- `data/processed/media/movies.jsonl` (62,424 lines)
- `data/processed/media/genome_scores.json` (13,816 movies with semantic tags)
- `data/processed/interactions/ratings.jsonl` (25M lines)
- `data/processed/interactions/tags.jsonl` (1M lines)

---

## Phase 2: Synthetic Data Generation

### 2.1 Generate User Demographics

**Goal**: Enhance 162K anonymous users with realistic profiles

```python
# Script: scripts/generate_user_profiles.py

from faker import Faker
import random
import json

fake = Faker(['en_US', 'fr_FR', 'es_ES', 'de_DE', 'it_IT'])

def generate_user_profile(user_id: str) -> dict:
    """Generate synthetic demographic data"""

    # Determine user archetype (influences preferences)
    archetype = random.choice([
        'cinephile', 'casual', 'family', 'young_adult',
        'senior', 'international', 'genre_specialist'
    ])

    profile = {
        'user_id': user_id,
        'demographics': {
            'age_range': random.choice(['18-24', '25-34', '35-44', '45-54', '55-64', '65+']),
            'location': {
                'country': random.choice(['US', 'FR', 'DE', 'ES', 'IT', 'CA', 'UK']),
                'timezone': random.choice(['UTC-8', 'UTC-5', 'UTC+1', 'UTC+2'])
            },
            'language_preferences': random.sample(
                ['en', 'fr', 'es', 'de', 'it'],
                k=random.randint(1, 3)
            ),
            'cultural_context': get_cultural_context(archetype)
        },
        'preferences': {
            'viewing_time_patterns': {
                'weekday_evening': random.uniform(0.3, 0.9),
                'weekend_afternoon': random.uniform(0.2, 0.8),
                'late_night': random.uniform(0.1, 0.6)
            },
            'device_preferences': {
                'tv': random.uniform(0.4, 0.9),
                'mobile': random.uniform(0.2, 0.7),
                'laptop': random.uniform(0.3, 0.8)
            }
        },
        'archetype': archetype
    }

    return profile

def get_cultural_context(archetype):
    contexts = {
        'cinephile': ['art_house', 'film_festivals', 'criterion'],
        'casual': ['mainstream', 'popular_culture'],
        'family': ['family_friendly', 'animated', 'educational'],
        'young_adult': ['trending', 'social_media', 'binge'],
        'senior': ['classic', 'documentary', 'historical'],
        'international': ['world_cinema', 'multilingual', 'festivals'],
        'genre_specialist': ['horror', 'scifi', 'noir', 'western']
    }
    return contexts.get(archetype, ['general'])

# Generate for all 162K users
user_ids = [f"user_{i:08d}" for i in range(1, 162001)]
profiles = [generate_user_profile(uid) for uid in user_ids]

with open('data/synthetic/users/demographics.jsonl', 'w') as f:
    for profile in profiles:
        f.write(json.dumps(profile) + '\n')
```

### 2.2 Generate Platform Availability

```python
# Script: scripts/generate_platform_data.py

def generate_platform_availability(movie_ids: list) -> dict:
    """Simulate platform availability for all movies"""

    platforms = [
        {'id': 'tv5monde', 'name': 'TV5MONDE', 'type': 'free'},
        {'id': 'netflix', 'name': 'Netflix', 'type': 'subscription'},
        {'id': 'amazon', 'name': 'Amazon Prime', 'type': 'subscription'},
        {'id': 'mubi', 'name': 'MUBI', 'type': 'subscription'},
        {'id': 'criterion', 'name': 'Criterion Channel', 'type': 'subscription'}
    ]

    availability = {}
    for movie_id in movie_ids:
        # 60% chance of being on at least one platform
        if random.random() < 0.6:
            num_platforms = random.randint(1, 3)
            selected_platforms = random.sample(platforms, k=num_platforms)

            availability[movie_id] = [
                {
                    'platform_id': p['id'],
                    'name': p['name'],
                    'availability_type': p['type'],
                    'region': random.choice(['US', 'FR', 'EU', 'Global']),
                    'quality': random.sample(['HD', '4K'], k=random.randint(1, 2)),
                    'subtitles': random.sample(['en', 'fr', 'es', 'de'], k=random.randint(1, 3))
                }
                for p in selected_platforms
            ]

    return availability
```

### 2.3 Generate Contextual Scenarios

```python
# Script: scripts/generate_contexts.py

def generate_viewing_contexts(count: int = 100000) -> list:
    """Generate diverse viewing context scenarios"""

    contexts = []
    for i in range(count):
        context = {
            'context_id': f"ctx_{i:08d}",
            'temporal': {
                'season': random.choice(['spring', 'summer', 'fall', 'winter']),
                'day_of_week': random.choice(['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']),
                'time_of_day': random.choice(['morning', 'afternoon', 'evening', 'night', 'late_night']),
                'is_holiday': random.choice([True, False], p=[0.1, 0.9])
            },
            'social': {
                'viewing_mode': random.choice(['alone', 'family', 'friends', 'date', 'party']),
                'group_size': random.randint(1, 6)
            },
            'intent': {
                'mood': random.choice([
                    'relaxed', 'energetic', 'melancholic', 'curious',
                    'romantic', 'adventurous', 'intellectual'
                ]),
                'occasion': random.choice([
                    'casual', 'special_event', 'education', 'background', 'focused'
                ]),
                'time_budget': random.choice(['short', 'medium', 'long', 'binge'])
            }
        }
        contexts.append(context)

    return contexts
```

### 2.4 Generate Subtitle Content (Simulated)

```python
# Script: scripts/generate_subtitle_data.py

def generate_subtitle_features(movie_id: str, genres: list) -> dict:
    """Generate realistic subtitle metadata without full transcripts"""

    # Genre-based vocabulary and sentiment
    genre_profiles = {
        'Drama': {'complexity': 0.75, 'sentiment_variance': 0.8},
        'Comedy': {'complexity': 0.55, 'sentiment_variance': 0.6},
        'Documentary': {'complexity': 0.85, 'sentiment_variance': 0.4},
        'Action': {'complexity': 0.45, 'sentiment_variance': 0.7},
        'Sci-Fi': {'complexity': 0.8, 'sentiment_variance': 0.6}
    }

    # Derive features from genres
    avg_complexity = np.mean([
        genre_profiles.get(g, {'complexity': 0.6})['complexity']
        for g in genres
    ])

    subtitle_data = {
        'media_id': movie_id,
        'language': 'en',
        'structured_content': {
            'vocabulary_complexity': avg_complexity,
            'dialogue_density': random.uniform(0.3, 0.9),
            'avg_sentence_length': random.randint(8, 25),
            'unique_word_count': random.randint(800, 3000),
            'key_topics': random.sample([
                'love', 'conflict', 'discovery', 'loss', 'triumph',
                'betrayal', 'redemption', 'mystery', 'science', 'history'
            ], k=random.randint(2, 5))
        },
        'sentiment_summary': {
            'overall_sentiment': random.uniform(-0.3, 0.8),
            'sentiment_arc': [
                random.uniform(-0.5, 0.8) for _ in range(10)  # 10-point arc
            ]
        }
    }

    return subtitle_data
```

### 2.5 Generate Trend Data

```python
# Script: scripts/generate_trends.py

import pandas as pd
from datetime import datetime, timedelta

def generate_trend_timeseries(movie_ids: list, start_date: str, end_date: str):
    """Generate popularity trends over time"""

    dates = pd.date_range(start=start_date, end=end_date, freq='D')

    trends = []
    for movie_id in random.sample(movie_ids, k=min(5000, len(movie_ids))):
        # Simulate trend pattern
        base_popularity = random.uniform(0.1, 0.8)
        volatility = random.uniform(0.05, 0.3)

        for date in dates:
            # Add seasonality and noise
            day_of_year = date.timetuple().tm_yday
            seasonal_factor = 0.2 * np.sin(2 * np.pi * day_of_year / 365)
            noise = random.gauss(0, volatility)

            popularity = max(0, min(1, base_popularity + seasonal_factor + noise))

            trends.append({
                'media_id': movie_id,
                'date': date.isoformat(),
                'popularity_score': popularity,
                'view_count_estimate': int(popularity * random.randint(100, 10000)),
                'search_volume': int(popularity * random.randint(50, 5000))
            })

    return pd.DataFrame(trends)
```

---

## Phase 3: Embedding Generation on A100

### 3.1 Setup Embedding Model

```python
# Script: scripts/generate_embeddings.py

from sentence_transformers import SentenceTransformer
import torch
import numpy as np

# Use A100 GPU
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Load multilingual SBERT model (384-dim)
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
model.to(device)

# Alternatively, for better quality (768-dim):
# model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
```

### 3.2 Generate Media Embeddings

```python
def generate_media_embeddings(batch_size: int = 512):
    """Generate embeddings for all movies using genome + metadata"""

    # Load processed data
    with open('data/processed/media/movies.jsonl') as f:
        movies = [json.loads(line) for line in f]

    with open('data/processed/media/genome_scores.json') as f:
        genome_data = json.load(f)

    embeddings = []
    metadata = []

    for i in range(0, len(movies), batch_size):
        batch = movies[i:i+batch_size]

        # Create rich text representations
        texts = []
        for movie in batch:
            # Combine title, genres, and top genome tags
            ml_id = str(movie['identifiers']['movielens_id'])
            genome = genome_data.get(ml_id, {})
            top_tags = sorted(genome.items(), key=lambda x: x[1], reverse=True)[:10]

            text = f"{movie['metadata']['title']}. "
            text += f"Genres: {', '.join(movie['classification']['genres'])}. "
            if top_tags:
                text += f"Themes: {', '.join([tag for tag, _ in top_tags])}."

            texts.append(text)

        # Generate embeddings on GPU
        batch_embeddings = model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            device=device
        )

        embeddings.extend(batch_embeddings)
        metadata.extend([{
            'media_id': m['media_id'],
            'title': m['metadata']['title']
        } for m in batch])

        print(f"Processed {min(i+batch_size, len(movies))}/{len(movies)} movies")

    # Save embeddings
    np.save('data/embeddings/media/content_vectors.npy', np.array(embeddings))
    with open('data/embeddings/media/metadata.jsonl', 'w') as f:
        for meta in metadata:
            f.write(json.dumps(meta) + '\n')

    print(f"Generated {len(embeddings)} embeddings of dimension {embeddings[0].shape[0]}")
```

### 3.3 Generate User Embeddings

```python
def generate_user_embeddings():
    """Generate user preference embeddings from rating history"""

    # Load user ratings
    user_ratings = {}
    with open('data/processed/interactions/ratings.jsonl') as f:
        for line in f:
            interaction = json.loads(line)
            uid = interaction['user_id']
            mid = interaction['media_id']
            rating = interaction['rating']

            if uid not in user_ratings:
                user_ratings[uid] = []
            user_ratings[uid].append((mid, rating))

    # Load media embeddings
    media_embeddings = np.load('data/embeddings/media/content_vectors.npy')
    with open('data/embeddings/media/metadata.jsonl') as f:
        media_meta = {json.loads(line)['media_id']: idx
                      for idx, line in enumerate(f)}

    # Generate user embeddings as weighted average
    user_embeddings = []
    user_ids = []

    for user_id, ratings in user_ratings.items():
        # Weight by rating (normalize to 0-1)
        weighted_vecs = []
        weights = []

        for media_id, rating in ratings:
            if media_id in media_meta:
                idx = media_meta[media_id]
                weighted_vecs.append(media_embeddings[idx])
                weights.append((rating - 1) / 4)  # Normalize 1-5 to 0-1

        if weighted_vecs:
            # Weighted average
            weights = np.array(weights) / sum(weights)
            user_embedding = np.average(weighted_vecs, axis=0, weights=weights)

            # Normalize
            user_embedding = user_embedding / np.linalg.norm(user_embedding)

            user_embeddings.append(user_embedding)
            user_ids.append(user_id)

    # Save
    np.save('data/embeddings/users/preference_vectors.npy', np.array(user_embeddings))
    with open('data/embeddings/users/user_ids.json', 'w') as f:
        json.dump(user_ids, f)

    print(f"Generated embeddings for {len(user_ids)} users")
```

**GPU Performance Estimation**:
- A100 40GB GPU
- Batch size: 512
- 62K movies: ~2-3 minutes
- 162K users: ~5 minutes (compute-bound, not inference)

---

## Phase 4: Database Population

### 4.1 Populate Milvus (Vector Database)

```python
# Script: scripts/populate_milvus.py

from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection
import numpy as np
import json

# Connect to Milvus
connections.connect("default", host="localhost", port="19530")

def create_media_collection():
    """Create and populate media embeddings collection"""

    # Define schema
    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="media_id", dtype=DataType.VARCHAR, max_length=100),
        FieldSchema(name="title", dtype=DataType.VARCHAR, max_length=500),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=384)
    ]
    schema = CollectionSchema(fields, description="Media content embeddings")
    collection = Collection("media_embeddings", schema)

    # Load data
    embeddings = np.load('data/embeddings/media/content_vectors.npy')
    with open('data/embeddings/media/metadata.jsonl') as f:
        metadata = [json.loads(line) for line in f]

    # Insert in batches
    batch_size = 10000
    for i in range(0, len(embeddings), batch_size):
        batch_meta = metadata[i:i+batch_size]
        batch_emb = embeddings[i:i+batch_size]

        entities = [
            [m['media_id'] for m in batch_meta],
            [m['title'] for m in batch_meta],
            batch_emb.tolist()
        ]

        collection.insert(entities)
        print(f"Inserted {min(i+batch_size, len(embeddings))}/{len(embeddings)}")

    # Create HNSW index for fast search
    index_params = {
        "metric_type": "COSINE",
        "index_type": "HNSW",
        "params": {"M": 16, "efConstruction": 200}
    }
    collection.create_index("embedding", index_params)
    collection.load()

    print("Media collection created and indexed")

def create_user_collection():
    """Similar for user embeddings"""
    # ... (similar structure)
```

**Execution**:
```bash
python scripts/populate_milvus.py
```

### 4.2 Populate Neo4j (Knowledge Graph)

```python
# Script: scripts/populate_neo4j.py

from neo4j import GraphDatabase
import json

driver = GraphDatabase.driver("bolt://localhost:7687", auth=("neo4j", "password"))

def create_nodes_and_relationships():
    """Populate Neo4j with movies, users, genres, and relationships"""

    with driver.session() as session:
        # Create movie nodes
        with open('data/processed/media/movies.jsonl') as f:
            for line in f:
                movie = json.loads(line)
                session.run("""
                    CREATE (m:Movie {
                        media_id: $media_id,
                        title: $title,
                        year: $year,
                        genres: $genres
                    })
                """, media_id=movie['media_id'],
                     title=movie['metadata']['title'],
                     year=movie['metadata']['year'],
                     genres=movie['classification']['genres'])

        # Create genre nodes and relationships
        session.run("""
            MATCH (m:Movie)
            UNWIND m.genres as genre
            MERGE (g:Genre {name: genre})
            WITH m, g
            CREATE (m)-[:HAS_GENRE]->(g)
        """)

        # Create user nodes and RATED relationships
        with open('data/processed/interactions/ratings.jsonl') as f:
            batch = []
            for i, line in enumerate(f):
                interaction = json.loads(line)
                batch.append(interaction)

                if len(batch) >= 10000:
                    session.run("""
                        UNWIND $batch as rating
                        MERGE (u:User {user_id: rating.user_id})
                        WITH u, rating
                        MATCH (m:Movie {media_id: rating.media_id})
                        CREATE (u)-[:RATED {
                            score: rating.rating,
                            timestamp: rating.timestamp
                        }]->(m)
                    """, batch=batch)
                    batch = []
                    print(f"Processed {i} ratings")

            # Insert remaining
            if batch:
                session.run(..., batch=batch)

        # Create similarity relationships based on embeddings
        # (Compute cosine similarity threshold > 0.8)
        # ... (compute and insert SIMILAR_TO edges)

    print("Neo4j population complete")
```

### 4.3 Populate PostgreSQL (AgentDB)

```python
# Script: scripts/populate_agentdb.py

import psycopg2
import json

conn = psycopg2.connect("postgresql://localhost/agentdb")
cur = conn.cursor()

def create_tables():
    """Create AgentDB schema"""
    cur.execute("""
        CREATE TABLE IF NOT EXISTS policies (
            user_id VARCHAR(50) PRIMARY KEY,
            state_hash VARCHAR(64),
            q_values JSONB,
            exploration_rate FLOAT,
            total_visits INT,
            updated_at TIMESTAMP DEFAULT NOW()
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS episodes (
            episode_id SERIAL PRIMARY KEY,
            user_id VARCHAR(50),
            state JSONB,
            action VARCHAR(100),
            reward FLOAT,
            next_state JSONB,
            timestamp TIMESTAMP
        )
    """)

    conn.commit()

def populate_policies():
    """Initialize user policies from rating history"""

    # Load user demographics
    with open('data/synthetic/users/demographics.jsonl') as f:
        users = [json.loads(line) for line in f]

    for user in users:
        # Initialize policy based on archetype
        initial_q_values = initialize_q_values(user['archetype'])

        cur.execute("""
            INSERT INTO policies (user_id, q_values, exploration_rate, total_visits)
            VALUES (%s, %s, %s, %s)
        """, (user['user_id'], json.dumps(initial_q_values), 0.1, 0))

    conn.commit()
    print(f"Initialized policies for {len(users)} users")

def populate_episodes():
    """Convert interactions to RL episodes"""
    # ... (transform ratings to state-action-reward tuples)
```

### 4.4 Populate Redis (Cache)

```python
# Script: scripts/populate_redis.py

import redis
import json

r = redis.Redis(host='localhost', port=6379, decode_responses=True)

def cache_platform_availability():
    """Pre-cache platform data"""
    with open('data/synthetic/platforms/availability.jsonl') as f:
        for line in f:
            data = json.loads(line)
            media_id = data['media_id']
            r.setex(
                f"platform:{media_id}",
                86400,  # 24-hour TTL
                json.dumps(data['platforms'])
            )

    print("Platform data cached")

def cache_trend_data():
    """Pre-cache trending content"""
    # ... (cache current trends with 15-min TTL)
```

---

## Phase 5: Validation & Quality Assurance

### 5.1 Data Quality Checks

```python
# Script: scripts/validate_data.py

def validate_embeddings():
    """Check embedding quality"""
    embeddings = np.load('data/embeddings/media/content_vectors.npy')

    checks = {
        'shape': embeddings.shape,
        'dimension': embeddings.shape[1],
        'normalized': np.allclose(np.linalg.norm(embeddings, axis=1), 1.0),
        'no_nans': not np.isnan(embeddings).any(),
        'no_zeros': not np.all(embeddings == 0, axis=1).any()
    }

    print("Embedding validation:", checks)
    return all(checks.values())

def validate_database_counts():
    """Check record counts across databases"""
    counts = {
        'milvus_media': collection.num_entities,
        'neo4j_movies': session.run("MATCH (m:Movie) RETURN count(m)").single()[0],
        'postgres_policies': cur.execute("SELECT COUNT(*) FROM policies").fetchone()[0]
    }

    print("Database counts:", counts)
    return counts
```

### 5.2 Benchmark Queries

```python
# Script: scripts/benchmark_queries.py

import time

def benchmark_vector_search():
    """Test Milvus query performance"""
    collection.load()

    # Random query vector
    query_vec = embeddings[0]

    start = time.time()
    results = collection.search(
        data=[query_vec],
        anns_field="embedding",
        param={"metric_type": "COSINE", "params": {"ef": 64}},
        limit=10
    )
    elapsed = (time.time() - start) * 1000

    print(f"Vector search: {elapsed:.2f}ms")
    return elapsed < 10  # Should be < 10ms on A100

def benchmark_graph_query():
    """Test Neo4j traversal performance"""
    query = """
        MATCH (u:User {user_id: $user_id})-[r:RATED]->(m:Movie)-[:HAS_GENRE]->(g:Genre)
        RETURN m, g
        LIMIT 10
    """
    # ... test performance
```

---

## Execution Timeline on A100

**Total Estimated Time**: 4-6 hours

| Phase | Task | Estimated Time |
|-------|------|---------------|
| 1 | Parse MovieLens CSVs | 30 min |
| 2 | Generate synthetic users | 45 min |
| 2 | Generate contexts & platforms | 30 min |
| 3 | Generate media embeddings (A100) | 3 min |
| 3 | Generate user embeddings | 5 min |
| 4 | Populate Milvus | 15 min |
| 4 | Populate Neo4j | 90 min |
| 4 | Populate PostgreSQL | 60 min |
| 4 | Populate Redis | 5 min |
| 5 | Validation & benchmarking | 30 min |

**Total**: ~4 hours 33 minutes

---

## Deployment Checklist

- [ ] GCP A100 VM provisioned
- [ ] Docker containers running (Milvus, Neo4j, PostgreSQL, Redis)
- [ ] Python environment with dependencies
- [ ] All scripts in `scripts/` directory
- [ ] Data directory structure created
- [ ] MovieLens datasets downloaded
- [ ] Execute Phase 1 (parsing)
- [ ] Execute Phase 2 (synthetic generation)
- [ ] Execute Phase 3 (embeddings on A100)
- [ ] Execute Phase 4 (database population)
- [ ] Execute Phase 5 (validation)
- [ ] Run benchmark queries
- [ ] Document performance metrics
- [ ] Create database backups

---

## Output Deliverables

1. **Databases**:
   - Milvus: 62K media + 162K user vectors
   - Neo4j: 200K+ nodes, 30M+ relationships
   - PostgreSQL: 162K policies, 25M episodes
   - Redis: Pre-cached platform/trend data

2. **Embeddings**:
   - `data/embeddings/media/content_vectors.npy` (62K × 384)
   - `data/embeddings/users/preference_vectors.npy` (162K × 384)

3. **Processed Data**:
   - JSONL files for all entities
   - Parquet files for analytics

4. **Documentation**:
   - Data schema definitions
   - API examples for each database
   - Performance benchmarks
   - Quality metrics

---

**Created**: 2025-12-06
**Author**: Data Engineering Team
**Version**: 1.0
