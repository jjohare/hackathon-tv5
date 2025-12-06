# Data Directory Structure

## Overview
Comprehensive data organization for the TV5 Media Recommendation System supporting multi-modal semantic search with GPU acceleration.

## Directory Structure

```
data/
├── raw/                    # Original source data (gitignored)
│   ├── users/             # User profile data
│   ├── media/             # Media content metadata
│   ├── interactions/      # User-media interactions
│   ├── context/           # Contextual information
│   ├── platforms/         # Platform availability data
│   ├── trends/            # Temporal trends data
│   ├── subtitles/         # Subtitle/transcript data
│   └── ontology/          # Ontology/taxonomy data
│
├── processed/             # Cleaned & normalized data (gitignored)
│   ├── users/             # Processed user data
│   ├── media/             # Processed media metadata
│   ├── interactions/      # Normalized interactions
│   ├── context/           # Structured context
│   ├── platforms/         # Platform mappings
│   ├── trends/            # Time-series data
│   ├── subtitles/         # Parsed subtitle data
│   └── ontology/          # Knowledge graph data
│
├── synthetic/             # Generated training data
│   ├── users/             # Synthetic user profiles
│   ├── media/             # Synthetic media metadata
│   ├── interactions/      # Generated interactions
│   ├── context/           # Simulated context
│   ├── platforms/         # Platform simulation
│   ├── trends/            # Trend generation
│   ├── subtitles/         # Synthetic subtitles
│   └── ontology/          # Generated relationships
│
└── embeddings/            # Pre-computed embeddings (gitignored)
    ├── media/             # Media content embeddings
    ├── users/             # User preference embeddings
    ├── context/           # Contextual embeddings
    └── subtitles/         # Subtitle embeddings
```

## Data Categories

### 1. Users (Personal Information & Context)
**Purpose**: User profiles, preferences, viewing history, demographic information

**Raw Data Sources**:
- MovieLens ratings.csv (userId, rating patterns)
- MovieLens tags.csv (user tagging behavior)
- Synthetic user demographics
- Synthetic viewing contexts

**Schema**:
```
User {
    user_id: String,
    demographics: {
        age_range: String,
        location: GeoPoint,
        language_preferences: Vec<String>,
        cultural_context: Vec<String>
    },
    preferences: {
        genre_weights: HashMap<String, f32>,
        director_affinity: HashMap<String, f32>,
        actor_affinity: HashMap<String, f32>,
        mood_preferences: Vec<String>,
        viewing_time_patterns: TimePatterns
    },
    history: {
        watched: Vec<MediaId>,
        ratings: HashMap<MediaId, f32>,
        tags: Vec<Tag>,
        watch_duration: HashMap<MediaId, Duration>
    },
    context: {
        current_mood: String,
        viewing_occasion: String,
        device_type: String,
        time_of_day: String,
        social_context: String  // alone, family, friends
    },
    learning_policy: {
        agent_id: String,
        q_values: HashMap<State, HashMap<Action, f32>>,
        exploration_rate: f32,
        reward_history: Vec<f32>
    }
}
```

### 2. Media Assets (Global Media Metadata)
**Purpose**: Comprehensive media content information with multi-modal features

**Raw Data Sources**:
- MovieLens movies.csv (titles, genres)
- MovieLens links.csv (IMDB/TMDB IDs)
- MovieLens genome-tags.csv + genome-scores.csv (semantic tags)
- Synthetic extended metadata

**Schema**:
```
MediaAsset {
    media_id: String,
    identifiers: {
        imdb_id: String,
        tmdb_id: String,
        internal_id: String
    },
    metadata: {
        title: String,
        original_title: String,
        year: u32,
        duration_minutes: u32,
        language: String,
        country: Vec<String>,
        content_rating: String
    },
    creators: {
        directors: Vec<Person>,
        writers: Vec<Person>,
        actors: Vec<Person>,
        producers: Vec<Person>
    },
    classification: {
        genres: Vec<String>,
        themes: Vec<String>,
        moods: Vec<String>,
        tags: Vec<Tag>,
        cultural_context: Vec<String>
    },
    genome: {
        tag_scores: HashMap<String, f32>,  // 1,128 genome tags
        semantic_clusters: Vec<String>
    },
    visual: {
        cinematography_style: Vec<String>,
        color_palette: Vec<String>,
        visual_tone: String
    },
    audio: {
        soundtrack_style: Vec<String>,
        dialogue_density: f32,
        music_prominence: f32
    },
    quality_metrics: {
        imdb_rating: f32,
        tmdb_rating: f32,
        critical_score: f32,
        audience_score: f32
    }
}
```

### 3. Platform Availability
**Purpose**: Where and how content is accessible

**Schema**:
```
PlatformAvailability {
    media_id: String,
    platforms: Vec<Platform>,
    updated_at: DateTime
}

Platform {
    platform_id: String,
    name: String,        // TV5Monde, Netflix, Amazon, etc.
    region: String,
    availability_type: String,  // free, subscription, rental, purchase
    price: Option<f32>,
    quality: Vec<String>,       // SD, HD, 4K
    subtitles: Vec<String>,     // Available subtitle languages
    audio_tracks: Vec<String>   // Available audio languages
}
```

### 4. Interactions (User-Media Events)
**Purpose**: Temporal interaction data for learning

**Raw Data Sources**:
- MovieLens ratings.csv (25M ratings)
- MovieLens tags.csv (user-applied tags)
- Synthetic interaction events

**Schema**:
```
Interaction {
    interaction_id: String,
    user_id: String,
    media_id: String,
    timestamp: DateTime,
    interaction_type: InteractionType,
    rating: Option<f32>,
    watch_duration: Option<Duration>,
    completion_rate: Option<f32>,
    context: {
        device: String,
        location: String,
        time_of_day: String,
        social: String
    },
    feedback: {
        explicit_rating: Option<f32>,
        implicit_signals: HashMap<String, f32>  // pause_count, rewind, skip
    }
}

InteractionType = View | Rate | Tag | Search | Favorite | Share | Skip
```

### 5. Contextual Information
**Purpose**: Situational and environmental factors

**Schema**:
```
Context {
    context_id: String,
    temporal: {
        season: String,
        month: String,
        day_of_week: String,
        time_of_day: String,
        is_holiday: bool,
        cultural_events: Vec<String>
    },
    environmental: {
        weather: Option<String>,
        location_type: String  // home, commute, travel
    },
    social: {
        viewing_mode: String,  // alone, family, friends, date
        group_size: u32,
        group_demographics: Vec<String>
    },
    intent: {
        mood: String,
        occasion: String,  // relax, learn, entertain_guests
        time_budget: Duration,
        content_intensity_pref: String  // light, medium, intense
    }
}
```

### 6. Current Trends
**Purpose**: Temporal patterns and popularity dynamics

**Schema**:
```
TrendData {
    trend_id: String,
    time_window: TimeRange,
    scope: String,  // global, regional, platform-specific
    trending_content: Vec<TrendingItem>,
    trending_genres: HashMap<String, f32>,
    trending_themes: HashMap<String, f32>,
    viral_moments: Vec<ViralEvent>,
    seasonal_patterns: HashMap<String, f32>
}

TrendingItem {
    media_id: String,
    trend_score: f32,
    velocity: f32,        // Rate of growth
    peak_timestamp: DateTime,
    demographic_breakdown: HashMap<String, f32>
}
```

### 7. Subtitle Analysis
**Purpose**: Semantic content from dialogue and text

**Schema**:
```
SubtitleData {
    media_id: String,
    language: String,
    transcript: String,
    structured_content: {
        dialogue_segments: Vec<DialogueSegment>,
        key_phrases: Vec<String>,
        topics: Vec<Topic>,
        sentiment_arc: Vec<SentimentPoint>,
        vocabulary_complexity: f32,
        cultural_references: Vec<String>
    },
    embeddings: {
        full_transcript_embedding: Vec<f32>,
        segment_embeddings: Vec<Vec<f32>>,
        topic_embeddings: HashMap<String, Vec<f32>>
    }
}

DialogueSegment {
    start_time: Duration,
    end_time: Duration,
    speaker: Option<String>,
    text: String,
    sentiment: f32,
    topics: Vec<String>
}
```

### 8. Ontology/Knowledge Graph
**Purpose**: Semantic relationships and reasoning

**Schema**:
```
OntologyNode {
    node_id: String,
    node_type: String,  // Movie, Genre, Theme, Person, Concept
    label: String,
    properties: HashMap<String, Value>,
    embeddings: Vec<f32>
}

OntologyRelationship {
    source_id: String,
    target_id: String,
    relationship_type: String,  // SIMILAR_TO, INFLUENCED_BY, PART_OF, HAS_THEME
    strength: f32,
    properties: HashMap<String, Value>
}
```

## Data Sources

### MovieLens 25M Dataset
**Downloaded**: ml-25m/ (1.1GB)
- 25M ratings from 162,000 users on 62,000 movies
- 1M tag applications
- 15M genome scores (1,128 semantic tags × 13,816 movies)
- Links to IMDB and TMDB

**Files**:
- `movies.csv`: movieId, title, genres
- `ratings.csv`: userId, movieId, rating, timestamp
- `tags.csv`: userId, movieId, tag, timestamp
- `links.csv`: movieId, imdbId, tmdbId
- `genome-tags.csv`: tagId, tag (1,128 semantic tags)
- `genome-scores.csv`: movieId, tagId, relevance (0-1)

### MovieLens Latest-Small
**Downloaded**: ml-latest-small/ (1MB)
- 100K ratings
- 9,000 movies
- Quick development/testing

### Synthetic Data Requirements
**To be generated**:
1. Extended user demographics (200K profiles)
2. Cultural context markers
3. Platform availability (simulated)
4. Viewing context scenarios
5. Subtitle content (simulated)
6. Trend timeseries data
7. Learning policy states

## Database Distribution

### Milvus (Vector Database)
**Embeddings**: 384-dim SBERT-compatible
- Media content embeddings (62K vectors)
- User preference embeddings (162K vectors)
- Subtitle segment embeddings (1M vectors)
- Contextual embeddings (100K vectors)

**Collections**:
- `media_embeddings`: Full media semantic vectors
- `user_embeddings`: User preference vectors
- `subtitle_embeddings`: Dialogue segment vectors
- `context_embeddings`: Situational vectors

### Neo4j (Graph Database)
**Nodes**: 500K+
- Media nodes (62K)
- User nodes (162K)
- Genre nodes (50)
- Theme nodes (1,128 from genome)
- Person nodes (directors, actors)

**Relationships**: 10M+
- RATED (25M interactions)
- SIMILAR_TO (computed similarities)
- HAS_GENRE (multi-label)
- HAS_THEME (from genome scores)
- INFLUENCED_BY (temporal patterns)
- WATCHED_BY (viewing history)

### PostgreSQL (AgentDB)
**Tables**:
- `policies`: User learning policies (162K rows)
- `episodes`: Interaction episodes (25M rows)
- `states`: State-action-reward tuples (100M rows)
- `metrics`: Performance metrics (time-series)

### Redis (Cache Layer)
**Keys**: TTL-based caching
- Query results (5-min TTL)
- User embeddings (1-hour TTL)
- Trend data (15-min TTL)
- Platform availability (1-day TTL)

## Data Generation Pipeline

### Phase 1: MovieLens Integration
1. Parse MovieLens CSVs
2. Generate embeddings from genome scores
3. Create synthetic user demographics
4. Map to internal schema

### Phase 2: Synthetic Enhancement
1. Generate cultural context markers
2. Create platform availability data
3. Simulate viewing contexts
4. Generate subtitle content
5. Create trend patterns

### Phase 3: Knowledge Graph Construction
1. Build Neo4j schema
2. Import nodes and relationships
3. Compute similarity edges
4. Generate OWL ontology

### Phase 4: Embedding Generation
1. Batch process content → embeddings
2. User preference → embeddings
3. Context → embeddings
4. Store in Milvus

### Phase 5: Learning Data
1. Generate AgentDB policies
2. Create episode trajectories
3. Initialize Q-values
4. Populate PostgreSQL

## File Formats

- **Raw**: CSV (MovieLens), JSON (metadata)
- **Processed**: Parquet (columnar), JSONL (streaming)
- **Embeddings**: NPY (NumPy), HDF5 (large arrays)
- **Database Dumps**: SQL, Neo4j Cypher, Milvus backups

## Data Quality Metrics

- Completeness: % of required fields populated
- Accuracy: Validation against known sources
- Consistency: Cross-database integrity
- Coverage: % of content with embeddings
- Freshness: Time since last update

---

**Last Updated**: 2025-12-06
**Total Data Size**: ~2GB raw + ~10GB processed + ~50GB embeddings
