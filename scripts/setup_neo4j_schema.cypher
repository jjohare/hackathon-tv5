// Neo4j Schema Setup for GMC-O (Generic Media Content Ontology)
// This script creates constraints, indexes, and full-text search capabilities
// for efficient ontology querying and recommendation generation.

// ==============================================================================
// CONSTRAINTS - Ensure data integrity and uniqueness
// ==============================================================================

// Media entity constraints
CREATE CONSTRAINT media_id_unique IF NOT EXISTS
FOR (m:Media) REQUIRE m.id IS UNIQUE;

CREATE CONSTRAINT media_title_exists IF NOT EXISTS
FOR (m:Media) REQUIRE m.title IS NOT NULL;

// Genre constraints
CREATE CONSTRAINT genre_name_unique IF NOT EXISTS
FOR (g:Genre) REQUIRE g.name IS UNIQUE;

// Mood constraints
CREATE CONSTRAINT mood_name_unique IF NOT EXISTS
FOR (m:Mood) REQUIRE m.name IS UNIQUE;

// User profile constraints
CREATE CONSTRAINT user_id_unique IF NOT EXISTS
FOR (u:User) REQUIRE u.id IS UNIQUE;

// Cultural context constraints
CREATE CONSTRAINT cultural_context_unique IF NOT EXISTS
FOR (c:CulturalContext) REQUIRE c.region IS UNIQUE;

// Tag constraints
CREATE CONSTRAINT tag_name_unique IF NOT EXISTS
FOR (t:Tag) REQUIRE t.name IS UNIQUE;

// Theme constraints
CREATE CONSTRAINT theme_name_unique IF NOT EXISTS
FOR (t:Theme) REQUIRE t.name IS UNIQUE;

// ==============================================================================
// INDEXES - Optimize query performance
// ==============================================================================

// Media indexes for filtering and sorting
CREATE INDEX media_title IF NOT EXISTS
FOR (m:Media) ON (m.title);

CREATE INDEX media_type IF NOT EXISTS
FOR (m:Media) ON (m.media_type);

CREATE INDEX media_duration IF NOT EXISTS
FOR (m:Media) ON (m.duration_seconds);

CREATE INDEX media_format IF NOT EXISTS
FOR (m:Media) ON (m.format);

CREATE INDEX media_resolution IF NOT EXISTS
FOR (m:Media) ON (m.resolution);

CREATE INDEX media_created_at IF NOT EXISTS
FOR (m:Media) ON (m.created_at);

CREATE INDEX media_updated_at IF NOT EXISTS
FOR (m:Media) ON (m.updated_at);

// Genre indexes
CREATE INDEX genre_name IF NOT EXISTS
FOR (g:Genre) ON (g.name);

// Mood indexes for emotional search
CREATE INDEX mood_valence IF NOT EXISTS
FOR (m:Mood) ON (m.valence);

CREATE INDEX mood_arousal IF NOT EXISTS
FOR (m:Mood) ON (m.arousal);

CREATE INDEX mood_dominance IF NOT EXISTS
FOR (m:Mood) ON (m.dominance);

// User profile indexes
CREATE INDEX user_last_active IF NOT EXISTS
FOR (u:User) ON (u.last_active);

CREATE INDEX user_created_at IF NOT EXISTS
FOR (u:User) ON (u.created_at);

// Cultural context indexes
CREATE INDEX cultural_language IF NOT EXISTS
FOR (c:CulturalContext) ON (c.language);

CREATE INDEX cultural_region IF NOT EXISTS
FOR (c:CulturalContext) ON (c.region);

// Interaction record indexes (if stored as nodes)
CREATE INDEX interaction_timestamp IF NOT EXISTS
FOR (i:Interaction) ON (i.timestamp);

CREATE INDEX interaction_type IF NOT EXISTS
FOR (i:Interaction) ON (i.interaction_type);

// ==============================================================================
// FULL-TEXT SEARCH INDEXES - Enable semantic search
// ==============================================================================

// Media content search across multiple text fields
CREATE FULLTEXT INDEX media_search IF NOT EXISTS
FOR (m:Media) ON EACH [m.title, m.themes, m.semantic_tags];

// Genre search including characteristics
CREATE FULLTEXT INDEX genre_search IF NOT EXISTS
FOR (g:Genre) ON EACH [g.name, g.characteristics];

// Tag semantic search
CREATE FULLTEXT INDEX tag_search IF NOT EXISTS
FOR (t:Tag) ON EACH [t.name, t.description];

// Theme search
CREATE FULLTEXT INDEX theme_search IF NOT EXISTS
FOR (t:Theme) ON EACH [t.name, t.description];

// Cultural context search
CREATE FULLTEXT INDEX cultural_search IF NOT EXISTS
FOR (c:CulturalContext) ON EACH [c.cultural_themes, c.region];

// ==============================================================================
// COMPOSITE INDEXES - Multi-property query optimization
// ==============================================================================

// Media filtering by type and duration
CREATE INDEX media_type_duration IF NOT EXISTS
FOR (m:Media) ON (m.media_type, m.duration_seconds);

// Mood dimensional search
CREATE INDEX mood_dimensions IF NOT EXISTS
FOR (m:Mood) ON (m.valence, m.arousal);

// ==============================================================================
// RELATIONSHIP INDEXES - Optimize traversal performance
// ==============================================================================

// Index on inferred relationships for reasoning queries
CREATE INDEX inferred_relationships IF NOT EXISTS
FOR ()-[r:SUBGENRE_OF]-() ON (r.inferred);

CREATE INDEX relationship_confidence IF NOT EXISTS
FOR ()-[r:SIMILAR_TO]-() ON (r.confidence);

// ==============================================================================
// STATISTICS AND VALIDATION
// ==============================================================================

// Verify schema setup
CALL db.indexes() YIELD name, type, state, populationPercent
RETURN name, type, state, populationPercent
ORDER BY name;

CALL db.constraints() YIELD name, type
RETURN name, type
ORDER BY name;

// ==============================================================================
// INITIALIZATION QUERIES - Set up version tracking
// ==============================================================================

// Create schema version node
MERGE (v:SchemaVersion {id: 'current'})
SET v.version = '2.0.0',
    v.updated_at = datetime(),
    v.description = 'GMC-O with GPU acceleration support';

// ==============================================================================
// PERFORMANCE HINTS
// ==============================================================================

// For bulk loading, consider temporarily disabling constraints:
// DROP CONSTRAINT <constraint_name>;
// ... perform bulk load ...
// CREATE CONSTRAINT <constraint_name> ...;

// For large datasets, use batched UNWIND queries with CALL IN TRANSACTIONS:
// CALL {
//   UNWIND $nodes AS node
//   MERGE (m:Media {id: node.id})
//   SET m += node
// } IN TRANSACTIONS OF 1000 ROWS;

// Monitor query performance:
// PROFILE MATCH (m:Media)-[:HAS_GENRE]->(g:Genre) RETURN m, g;
