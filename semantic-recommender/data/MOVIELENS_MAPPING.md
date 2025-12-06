# MovieLens to System Schema Mapping

**Purpose**: Map MovieLens 25M dataset fields to our internal data schema

---

## Dataset Overview

**Downloaded**:
- `ml-latest-small/` (1MB) - Development/testing
- `ml-25m/` (1.1GB) - Production dataset

**Statistics**:
- 25,000,095 ratings
- 1,093,360 tag applications
- 62,423 movies
- 162,541 users
- 1,128 genome tags
- 15,584,448 genome scores (tag relevance)

---

## File-by-File Mapping

### 1. movies.csv → Media Assets

**Source Schema**:
```csv
movieId,title,genres
1,Toy Story (1995),Adventure|Animation|Children|Comedy|Fantasy
```

**Target Schema** (`data/processed/media/movies.jsonl`):
```json
{
  "media_id": "ml_1",
  "identifiers": {
    "movielens_id": 1,
    "internal_id": "media_00000001"
  },
  "metadata": {
    "title": "Toy Story",
    "original_title": "Toy Story",
    "year": 1995,
    "language": "en",
    "country": ["US"]
  },
  "classification": {
    "genres": ["Adventure", "Animation", "Children", "Comedy", "Fantasy"],
    "themes": [],
    "moods": [],
    "tags": []
  }
}
```

**Transformation**:
- Extract year from title using regex: `(.+) \((\d{4})\)`
- Split genres on pipe delimiter
- Prefix movieId with "ml_"
- Add defaults: language="en", country=["US"]

**Code**:
```python
def parse_title_year(title: str) -> tuple:
    import re
    match = re.match(r'(.+) \((\d{4})\)', title)
    if match:
        return match.group(1), int(match.group(2))
    return title, None
```

---

### 2. ratings.csv → User Interactions

**Source Schema**:
```csv
userId,movieId,rating,timestamp
1,296,5.0,1147880044
```

**Target Schema** (`data/processed/interactions/ratings.jsonl`):
```json
{
  "interaction_id": "int_0000000001",
  "user_id": "user_00000001",
  "media_id": "ml_296",
  "timestamp": 1147880044,
  "iso_timestamp": "2006-05-17T16:27:24Z",
  "interaction_type": "Rate",
  "rating": 5.0,
  "context": {
    "device": "unknown",
    "time_of_day": "afternoon",
    "day_of_week": "Wednesday"
  },
  "feedback": {
    "explicit_rating": 5.0
  }
}
```

**Transformation**:
- Zero-pad user/movie IDs
- Convert Unix timestamp to ISO 8601
- Classify time_of_day from timestamp
- Default context values

**Code**:
```python
from datetime import datetime

def classify_time_of_day(timestamp: int) -> str:
    dt = datetime.fromtimestamp(timestamp)
    hour = dt.hour
    if 6 <= hour < 12:
        return "morning"
    elif 12 <= hour < 17:
        return "afternoon"
    elif 17 <= hour < 21:
        return "evening"
    elif 21 <= hour < 24:
        return "night"
    else:
        return "late_night"

def format_interaction(row):
    return {
        "interaction_id": f"int_{row.name:010d}",
        "user_id": f"user_{row['userId']:08d}",
        "media_id": f"ml_{row['movieId']}",
        "timestamp": row['timestamp'],
        "iso_timestamp": datetime.fromtimestamp(row['timestamp']).isoformat(),
        "interaction_type": "Rate",
        "rating": row['rating'],
        "context": {
            "time_of_day": classify_time_of_day(row['timestamp'])
        }
    }
```

---

### 3. tags.csv → Tag Interactions

**Source Schema**:
```csv
userId,movieId,tag,timestamp
15,339,sandra 'boring' bullock,1138537770
```

**Target Schema** (`data/processed/interactions/tags.jsonl`):
```json
{
  "interaction_id": "tag_0000000001",
  "user_id": "user_00000015",
  "media_id": "ml_339",
  "timestamp": 1138537770,
  "interaction_type": "Tag",
  "tag": "sandra 'boring' bullock",
  "tag_normalized": "sandra_boring_bullock"
}
```

**Transformation**:
- Normalize tags (lowercase, replace spaces/special chars)
- Format as tag interaction event

---

### 4. genome-tags.csv → Semantic Tag Vocabulary

**Source Schema**:
```csv
tagId,tag
1,007
2,007 (series)
3,18th century
```

**Target Use**: Build semantic tag dictionary for genome scores

**Storage**: In-memory lookup table
```python
tag_lookup = {
    1: "007",
    2: "007 (series)",
    3: "18th century",
    # ... 1,128 tags total
}
```

---

### 5. genome-scores.csv → Semantic Media Enrichment

**Source Schema**:
```csv
movieId,tagId,relevance
1,1,0.02875
1,2,0.02375
```

**Target Schema** (`data/processed/media/genome_scores.json`):
```json
{
  "1": {
    "007": 0.02875,
    "007 (series)": 0.02375,
    "action": 0.78125,
    "adventure": 0.82500,
    "animation": 0.95000
  }
}
```

**Transformation**:
- Group by movieId
- Map tagId → tag name using genome-tags
- Filter relevance > 0.5 (strong associations only)
- Store as movie_id → {tag: score} mapping

**Usage**:
1. Enrich movie metadata with themes
2. Generate text descriptions for embeddings
3. Build knowledge graph relationships

**Code**:
```python
import pandas as pd

def process_genome_scores():
    tags_df = pd.read_csv('genome-tags.csv')
    scores_df = pd.read_csv('genome-scores.csv')

    tag_map = dict(zip(tags_df['tagId'], tags_df['tag']))

    genomes = {}
    for movie_id, group in scores_df.groupby('movieId'):
        genome = {
            tag_map[row['tagId']]: row['relevance']
            for _, row in group.iterrows()
            if row['relevance'] > 0.5  # Threshold
        }
        if genome:
            genomes[str(movie_id)] = genome

    return genomes
```

---

### 6. links.csv → External Identifiers

**Source Schema**:
```csv
movieId,imdbId,tmdbId
1,0114709,862
```

**Target Enhancement**: Add to media assets
```json
{
  "media_id": "ml_1",
  "identifiers": {
    "movielens_id": 1,
    "imdb_id": "tt0114709",
    "tmdb_id": 862,
    "internal_id": "media_00000001"
  }
}
```

**Transformation**:
- Prefix IMDB ID with "tt"
- Keep TMDB ID as integer

**Usage**: Enable cross-referencing with external APIs (TMDB, IMDB)

---

## Data Enrichment Strategy

### Genome-Based Text Generation

**Purpose**: Create rich text for embedding generation

**Input**: Movie + Genome Scores
```json
{
  "title": "Toy Story",
  "genres": ["Animation", "Comedy"],
  "genome": {
    "pixar": 0.98,
    "toys": 0.95,
    "friendship": 0.89,
    "imaginative": 0.87
  }
}
```

**Output Text**:
```
Toy Story. Genres: Animation, Comedy.
Themes: pixar, toys, friendship, imaginative, heartwarming, family.
A classic animated film about toys that come to life, exploring themes
of friendship and imagination with Pixar's signature storytelling.
```

**Code**:
```python
def generate_embedding_text(movie, genome):
    text = f"{movie['title']}. "
    text += f"Genres: {', '.join(movie['genres'])}. "

    # Top 10 genome tags
    top_tags = sorted(genome.items(), key=lambda x: x[1], reverse=True)[:10]
    if top_tags:
        text += f"Themes: {', '.join([tag for tag, _ in top_tags])}. "

    # Could add synopsis if available
    return text
```

---

## User Profile Enrichment

**Raw Data**: userId from ratings
**Enhanced Profile**:

```json
{
  "user_id": "user_00000123",
  "rating_stats": {
    "total_ratings": 247,
    "avg_rating": 3.8,
    "rating_variance": 1.2,
    "genres_rated": {
      "Drama": 89,
      "Comedy": 65,
      "Action": 43
    }
  },
  "preferences": {
    "genre_weights": {
      "Drama": 0.85,
      "Comedy": 0.72,
      "Action": 0.65
    },
    "tag_affinity": {
      "independent film": 0.91,
      "emotional": 0.87,
      "thought-provoking": 0.82
    }
  },
  "viewing_patterns": {
    "most_active_hour": 21,
    "most_active_day": "Saturday",
    "binge_probability": 0.65
  }
}
```

**Derivation**:
1. Aggregate all user ratings
2. Calculate genre distribution
3. Compute weighted tag affinity from rated movies' genomes
4. Extract temporal patterns from timestamps
5. Generate synthetic demographics

---

## Knowledge Graph Construction

### Neo4j Import Queries

**Movies**:
```cypher
CREATE (m:Movie {
  media_id: "ml_1",
  title: "Toy Story",
  year: 1995
})
```

**Genres**:
```cypher
MATCH (m:Movie {media_id: "ml_1"})
UNWIND ["Animation", "Comedy"] as genre
MERGE (g:Genre {name: genre})
CREATE (m)-[:HAS_GENRE]->(g)
```

**Themes** (from genome):
```cypher
MATCH (m:Movie {media_id: "ml_1"})
UNWIND ["pixar", "toys", "friendship"] as theme
MERGE (t:Theme {name: theme})
CREATE (m)-[:HAS_THEME {relevance: 0.95}]->(t)
```

**User Ratings**:
```cypher
MERGE (u:User {user_id: "user_00000001"})
WITH u
MATCH (m:Movie {media_id: "ml_296"})
CREATE (u)-[:RATED {score: 5.0, timestamp: 1147880044}]->(m)
```

**Similarity** (compute offline):
```cypher
MATCH (m1:Movie), (m2:Movie)
WHERE m1.media_id < m2.media_id  // Avoid duplicates
  AND cosine_similarity(m1.embedding, m2.embedding) > 0.8
CREATE (m1)-[:SIMILAR_TO {score: 0.85}]->(m2)
```

---

## Data Quality Checks

### Validation Queries

**Check movie count**:
```python
assert len(movies) == 62423
```

**Check rating range**:
```python
assert ratings['rating'].min() == 0.5
assert ratings['rating'].max() == 5.0
```

**Check genome coverage**:
```python
genome_movie_ids = set(genome_scores['movieId'].unique())
all_movie_ids = set(movies['movieId'])
coverage = len(genome_movie_ids) / len(all_movie_ids)
assert coverage > 0.20  # At least 20% have genome data
```

**Check timestamp validity**:
```python
from datetime import datetime
min_date = datetime.fromtimestamp(ratings['timestamp'].min())
max_date = datetime.fromtimestamp(ratings['timestamp'].max())
assert min_date.year >= 1990  # Reasonable range
assert max_date.year <= 2025
```

---

## Summary

| Source File | Records | Target Output | Purpose |
|-------------|---------|---------------|---------|
| movies.csv | 62,423 | media/movies.jsonl | Base metadata |
| ratings.csv | 25M | interactions/ratings.jsonl | User behavior |
| tags.csv | 1M | interactions/tags.jsonl | User tagging |
| genome-tags.csv | 1,128 | In-memory lookup | Tag vocabulary |
| genome-scores.csv | 15.5M | media/genome_scores.json | Semantic features |
| links.csv | 62,423 | Merged into movies | External IDs |

**Total Processed Output**: ~2GB
**Embedding Output**: ~50GB (after vectorization)
**Database Size**: ~100GB (fully populated)

---

**Created**: 2025-12-06
**Version**: 1.0
