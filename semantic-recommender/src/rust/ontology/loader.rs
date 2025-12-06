/// Neo4j Integration Module
///
/// Loads GMC-O ontology data from Neo4j graph database with efficient bulk loading,
/// RDF triple parsing, and incremental updates.

use super::types::*;
use super::reasoner::MediaOntology;
use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::io::BufReader;
use std::path::Path;
use neo4rs::{Graph, Query, ConfigBuilder, Node, Relation};
use tokio::time::{timeout, Duration};

const BATCH_SIZE: usize = 1000;
const MAX_RETRIES: u32 = 3;
const RETRY_DELAY_MS: u64 = 1000;
const QUERY_TIMEOUT_SECS: u64 = 30;

/// Configuration for Neo4j connection
#[derive(Debug, Clone)]
pub struct Neo4jConfig {
    pub uri: String,
    pub username: String,
    pub password: String,
    pub database: String,
    pub max_connections: u32,
    pub connection_timeout_secs: u64,
}

impl Default for Neo4jConfig {
    fn default() -> Self {
        Self {
            uri: "bolt://localhost:7687".to_string(),
            username: "neo4j".to_string(),
            password: "password".to_string(),
            database: "neo4j".to_string(),
            max_connections: 10,
            connection_timeout_secs: 10,
        }
    }
}

/// RDF Triple for bulk loading
#[derive(Debug, Clone)]
pub struct RdfTriple {
    pub subject: String,
    pub predicate: String,
    pub object: RdfObject,
}

#[derive(Debug, Clone)]
pub enum RdfObject {
    Uri(String),
    Literal(String, Option<String>), // value, datatype
}

/// Neo4j ontology loader with connection pooling and retry logic
#[derive(Clone)]
pub struct OntologyLoader {
    config: Neo4jConfig,
    graph: Graph,
}

impl OntologyLoader {
    /// Create new loader with configuration and connection pool
    pub async fn new(config: Neo4jConfig) -> OntologyResult<Self> {
        let neo4j_config = ConfigBuilder::default()
            .uri(&config.uri)
            .user(&config.username)
            .password(&config.password)
            .db(&config.database)
            .max_connections(config.max_connections)
            .build()
            .map_err(|e| OntologyError::InferenceFailure(format!("Config error: {}", e)))?;

        let graph = Graph::connect(neo4j_config)
            .await
            .map_err(|e| OntologyError::InferenceFailure(format!("Connection failed: {}", e)))?;

        Ok(Self { config, graph })
    }

    /// Execute query with timeout and retry logic
    async fn execute_with_retry<T, F>(&self, operation: F) -> OntologyResult<T>
    where
        F: Fn() -> futures::future::BoxFuture<'_, Result<T, neo4rs::Error>>,
    {
        for attempt in 0..MAX_RETRIES {
            match timeout(
                Duration::from_secs(QUERY_TIMEOUT_SECS),
                operation()
            ).await {
                Ok(Ok(result)) => return Ok(result),
                Ok(Err(e)) => {
                    if attempt == MAX_RETRIES - 1 {
                        return Err(OntologyError::InferenceFailure(format!("Query failed: {}", e)));
                    }
                    tokio::time::sleep(Duration::from_millis(RETRY_DELAY_MS)).await;
                }
                Err(_) => {
                    if attempt == MAX_RETRIES - 1 {
                        return Err(OntologyError::InferenceFailure("Query timeout".to_string()));
                    }
                    tokio::time::sleep(Duration::from_millis(RETRY_DELAY_MS)).await;
                }
            }
        }
        Err(OntologyError::InferenceFailure("Max retries exceeded".to_string()))
    }

    /// Setup Neo4j schema with constraints and indexes
    pub async fn setup_schema(&self) -> OntologyResult<()> {
        let schema_queries = vec![
            // Constraints for unique IDs
            "CREATE CONSTRAINT media_id_unique IF NOT EXISTS FOR (m:Media) REQUIRE m.id IS UNIQUE",
            "CREATE CONSTRAINT genre_name_unique IF NOT EXISTS FOR (g:Genre) REQUIRE g.name IS UNIQUE",
            "CREATE CONSTRAINT mood_name_unique IF NOT EXISTS FOR (m:Mood) REQUIRE m.name IS UNIQUE",
            "CREATE CONSTRAINT user_id_unique IF NOT EXISTS FOR (u:User) REQUIRE u.id IS UNIQUE",
            "CREATE CONSTRAINT cultural_context_unique IF NOT EXISTS FOR (c:CulturalContext) REQUIRE c.region IS UNIQUE",

            // Indexes on frequently queried properties
            "CREATE INDEX media_title IF NOT EXISTS FOR (m:Media) ON (m.title)",
            "CREATE INDEX media_type IF NOT EXISTS FOR (m:Media) ON (m.media_type)",
            "CREATE INDEX genre_name IF NOT EXISTS FOR (g:Genre) ON (g.name)",
            "CREATE INDEX mood_valence IF NOT EXISTS FOR (m:Mood) ON (m.valence)",
            "CREATE INDEX user_last_active IF NOT EXISTS FOR (u:User) ON (u.last_active)",

            // Full-text search indexes
            "CREATE FULLTEXT INDEX media_search IF NOT EXISTS FOR (m:Media) ON EACH [m.title, m.themes, m.semantic_tags]",
            "CREATE FULLTEXT INDEX genre_search IF NOT EXISTS FOR (g:Genre) ON EACH [g.name, g.characteristics]",
        ];

        for query_str in schema_queries {
            self.execute_with_retry(|| {
                Box::pin(async {
                    self.graph.run(Query::new(query_str.to_string())).await
                })
            }).await?;
        }

        Ok(())
    }

    /// Parse RDF triples from Turtle file
    pub fn parse_ttl_file<P: AsRef<Path>>(path: P) -> OntologyResult<Vec<RdfTriple>> {
        use rio_api::parser::TriplesParser;
        use rio_turtle::TurtleParser;

        let file = File::open(path)
            .map_err(|e| OntologyError::InferenceFailure(format!("File error: {}", e)))?;
        let reader = BufReader::new(file);

        let mut triples = Vec::new();

        TurtleParser::new(reader, None).parse_all(&mut |triple| {
            let subject = triple.subject.to_string();
            let predicate = triple.predicate.to_string();
            let object = match triple.object {
                rio_api::model::Term::NamedNode(node) => RdfObject::Uri(node.to_string()),
                rio_api::model::Term::Literal(lit) => {
                    let datatype = lit.datatype().map(|dt| dt.to_string());
                    RdfObject::Literal(lit.value().to_string(), datatype)
                }
                _ => RdfObject::Literal(triple.object.to_string(), None),
            };

            triples.push(RdfTriple { subject, predicate, object });
            Ok(())
        }).map_err(|e| OntologyError::InferenceFailure(format!("Parse error: {}", e)))?;

        Ok(triples)
    }

    /// Bulk load RDF triples using batched UNWIND queries
    pub async fn bulk_load_triples(&self, triples: Vec<RdfTriple>) -> OntologyResult<()> {
        let start = std::time::Instant::now();

        // Group triples by entity type
        let mut nodes_by_type: HashMap<String, Vec<HashMap<String, String>>> = HashMap::new();
        let mut relationships: Vec<(String, String, String, HashMap<String, String>)> = Vec::new();

        for triple in triples {
            match &triple.object {
                RdfObject::Uri(uri) => {
                    // This is a relationship
                    relationships.push((
                        triple.subject.clone(),
                        extract_label(&triple.predicate),
                        uri.clone(),
                        HashMap::new()
                    ));
                }
                RdfObject::Literal(value, datatype) => {
                    // This is a property
                    let entity_type = extract_type(&triple.subject);
                    let properties = nodes_by_type.entry(entity_type).or_insert_with(Vec::new);

                    if let Some(last_node) = properties.last_mut() {
                        if last_node.get("id") == Some(&triple.subject) {
                            last_node.insert(extract_label(&triple.predicate), value.clone());
                            continue;
                        }
                    }

                    let mut props = HashMap::new();
                    props.insert("id".to_string(), triple.subject.clone());
                    props.insert(extract_label(&triple.predicate), value.clone());
                    properties.push(props);
                }
            }
        }

        // Batch insert nodes
        for (node_type, nodes) in nodes_by_type {
            for chunk in nodes.chunks(BATCH_SIZE) {
                let query = format!(
                    r#"
                    UNWIND $nodes AS node
                    MERGE (n:{} {{id: node.id}})
                    SET n += node
                    "#,
                    node_type
                );

                let chunk_data: Vec<_> = chunk.iter().cloned().collect();

                self.execute_with_retry(|| {
                    let q = Query::new(query.clone())
                        .param("nodes", chunk_data.clone());
                    Box::pin(async move {
                        self.graph.run(q).await
                    })
                }).await?;
            }
        }

        // Batch insert relationships
        for chunk in relationships.chunks(BATCH_SIZE) {
            let query = r#"
                UNWIND $rels AS rel
                MATCH (a {id: rel.source})
                MATCH (b {id: rel.target})
                CALL apoc.merge.relationship(a, rel.type, {}, rel.props, b) YIELD rel as r
                RETURN count(r)
            "#;

            let chunk_data: Vec<_> = chunk.iter().map(|(src, rel_type, tgt, props)| {
                serde_json::json!({
                    "source": src,
                    "type": rel_type,
                    "target": tgt,
                    "props": props
                })
            }).collect();

            self.execute_with_retry(|| {
                let q = Query::new(query.to_string())
                    .param("rels", chunk_data.clone());
                Box::pin(async move {
                    self.graph.run(q).await
                })
            }).await?;
        }

        let elapsed = start.elapsed();
        println!("Loaded {} triples in {:.2}s", triples.len(), elapsed.as_secs_f32());

        Ok(())
    }

    /// Incremental update using MERGE strategy
    pub async fn incremental_update(&self, entity_id: &str, properties: HashMap<String, String>) -> OntologyResult<()> {
        let query = r#"
            MERGE (n:Media {id: $id})
            SET n += $props
            SET n.updated_at = datetime()
            RETURN n
        "#;

        self.execute_with_retry(|| {
            let q = Query::new(query.to_string())
                .param("id", entity_id.to_string())
                .param("props", properties.clone());
            Box::pin(async move {
                self.graph.run(q).await
            })
        }).await?;

        Ok(())
    }

    /// Load complete ontology from Neo4j
    pub async fn load_ontology(&self) -> OntologyResult<MediaOntology> {
        let mut ontology = MediaOntology::default();

        ontology.media = self.load_media_entities().await?;
        ontology.genre_hierarchy = self.load_genre_hierarchy().await?;
        ontology.mood_relations = self.load_mood_relations().await?;
        ontology.cultural_contexts = self.load_cultural_contexts().await?;
        ontology.media_relations = self.load_media_relations().await?;
        ontology.disjoint_genres = self.load_disjoint_genres().await?;
        ontology.equivalent_genres = self.load_equivalent_genres().await?;
        ontology.tag_hierarchy = self.load_tag_hierarchy().await?;

        Ok(ontology)
    }

    /// Load all media entities
    async fn load_media_entities(&self) -> OntologyResult<HashMap<String, MediaEntity>> {
        let query = r#"
            MATCH (m:Media)
            OPTIONAL MATCH (m)-[:HAS_GENRE]->(g:Genre)
            OPTIONAL MATCH (m)-[:HAS_MOOD]->(mood:Mood)
            OPTIONAL MATCH (m)-[:HAS_THEME]->(theme:Theme)
            RETURN m,
                   collect(DISTINCT g.name) as genres,
                   collect(DISTINCT mood.name) as moods,
                   collect(DISTINCT theme.name) as themes
        "#;

        let mut media = HashMap::new();

        self.execute_with_retry(|| {
            Box::pin(async {
                let mut result = self.graph.execute(Query::new(query.to_string())).await?;

                while let Some(row) = result.next().await? {
                    let node: Node = row.get("m")?;
                    let genres: Vec<String> = row.get("genres")?;
                    let moods: Vec<String> = row.get("moods")?;
                    let themes: Vec<String> = row.get("themes")?;

                    let id: String = node.get("id")?;
                    let title: String = node.get("title")?;
                    let media_type_str: String = node.get("media_type")?;

                    let media_type = match media_type_str.as_str() {
                        "Video" => MediaType::Video,
                        "Audio" => MediaType::Audio,
                        "Image" => MediaType::Image,
                        "Text" => MediaType::Text,
                        "Interactive" => MediaType::Interactive,
                        _ => MediaType::Mixed,
                    };

                    let entity = MediaEntity {
                        id: id.clone(),
                        title,
                        media_type,
                        genres,
                        moods,
                        themes,
                        cultural_context: node.get("cultural_context").unwrap_or_default(),
                        technical_metadata: TechnicalMetadata {
                            duration_seconds: node.get("duration_seconds").ok(),
                            resolution: node.get("resolution").ok(),
                            format: node.get("format").unwrap_or_else(|_| "unknown".to_string()),
                            bitrate: node.get("bitrate").ok(),
                            file_size_bytes: node.get("file_size_bytes").ok(),
                        },
                        semantic_tags: node.get("semantic_tags").unwrap_or_default(),
                    };

                    media.insert(id, entity);
                }

                Ok(media)
            })
        }).await
    }

    /// Load genre hierarchy
    async fn load_genre_hierarchy(&self) -> OntologyResult<HashMap<String, HashSet<String>>> {
        let query = r#"
            MATCH (child:Genre)-[:SUBGENRE_OF]->(parent:Genre)
            RETURN child.name as child, collect(parent.name) as parents
        "#;

        let mut hierarchy = HashMap::new();

        self.execute_with_retry(|| {
            Box::pin(async {
                let mut result = self.graph.execute(Query::new(query.to_string())).await?;

                while let Some(row) = result.next().await? {
                    let child: String = row.get("child")?;
                    let parents: Vec<String> = row.get("parents")?;
                    hierarchy.insert(child, parents.into_iter().collect());
                }

                Ok(hierarchy)
            })
        }).await
    }

    /// Load mood relations and definitions
    async fn load_mood_relations(&self) -> OntologyResult<HashMap<String, Mood>> {
        let query = r#"
            MATCH (m:Mood)
            OPTIONAL MATCH (m)-[:RELATED_TO]->(related:Mood)
            RETURN m.name as name, m.valence as valence, m.arousal as arousal,
                   m.dominance as dominance, collect(related.name) as related_moods
        "#;

        let mut moods = HashMap::new();

        self.execute_with_retry(|| {
            Box::pin(async {
                let mut result = self.graph.execute(Query::new(query.to_string())).await?;

                while let Some(row) = result.next().await? {
                    let name: String = row.get("name")?;
                    let mood = Mood {
                        name: name.clone(),
                        valence: row.get("valence")?,
                        arousal: row.get("arousal")?,
                        dominance: row.get("dominance")?,
                        related_moods: row.get("related_moods")?,
                    };
                    moods.insert(name, mood);
                }

                Ok(moods)
            })
        }).await
    }

    /// Load cultural contexts
    async fn load_cultural_contexts(&self) -> OntologyResult<HashMap<String, CulturalContext>> {
        let query = r#"
            MATCH (c:CulturalContext)
            OPTIONAL MATCH (c)-[:HAS_THEME]->(theme:Theme)
            OPTIONAL MATCH (c)-[:HAS_TABOO]->(taboo:Taboo)
            RETURN c, collect(DISTINCT theme.name) as themes, collect(DISTINCT taboo.name) as taboos
        "#;

        let mut contexts = HashMap::new();

        self.execute_with_retry(|| {
            Box::pin(async {
                let mut result = self.graph.execute(Query::new(query.to_string())).await?;

                while let Some(row) = result.next().await? {
                    let node: Node = row.get("c")?;
                    let themes: Vec<String> = row.get("themes")?;
                    let taboos: Vec<String> = row.get("taboos")?;

                    let region: String = node.get("region")?;
                    let context = CulturalContext {
                        region: region.clone(),
                        language: node.get("language")?,
                        cultural_themes: themes,
                        taboos,
                        preferences: node.get("preferences").unwrap_or_default(),
                    };
                    contexts.insert(region, context);
                }

                Ok(contexts)
            })
        }).await
    }

    /// Load media relationships
    async fn load_media_relations(&self) -> OntologyResult<HashMap<String, Vec<(String, MediaRelation)>>> {
        let query = r#"
            MATCH (m1:Media)-[r]->(m2:Media)
            WHERE type(r) IN ['SEQUEL_OF', 'PREQUEL_OF', 'SIMILAR_TO', 'RELATED_TO',
                              'PART_OF', 'BASED_ON', 'REMAKE_OF', 'INSPIRED_BY']
            RETURN m1.id as source, type(r) as rel_type, m2.id as target
        "#;

        let mut relations: HashMap<String, Vec<(String, MediaRelation)>> = HashMap::new();

        self.execute_with_retry(|| {
            Box::pin(async {
                let mut result = self.graph.execute(Query::new(query.to_string())).await?;

                while let Some(row) = result.next().await? {
                    let source: String = row.get("source")?;
                    let target: String = row.get("target")?;
                    let rel_type_str: String = row.get("rel_type")?;

                    let rel_type = match rel_type_str.as_str() {
                        "SEQUEL_OF" => MediaRelation::SequelOf,
                        "PREQUEL_OF" => MediaRelation::PrequelOf,
                        "SIMILAR_TO" => MediaRelation::SimilarTo,
                        "RELATED_TO" => MediaRelation::RelatedTo,
                        "PART_OF" => MediaRelation::PartOf,
                        "BASED_ON" => MediaRelation::BasedOn,
                        "REMAKE_OF" => MediaRelation::RemakeOf,
                        "INSPIRED_BY" => MediaRelation::InspiredBy,
                        _ => continue,
                    };

                    relations.entry(source).or_insert_with(Vec::new).push((target, rel_type));
                }

                Ok(relations)
            })
        }).await
    }

    /// Load disjoint genre sets
    async fn load_disjoint_genres(&self) -> OntologyResult<Vec<HashSet<String>>> {
        let query = r#"
            MATCH (g1:Genre)-[:DISJOINT_WITH]-(g2:Genre)
            RETURN collect(DISTINCT [g1.name, g2.name]) as disjoint_pairs
        "#;

        self.execute_with_retry(|| {
            Box::pin(async {
                let mut result = self.graph.execute(Query::new(query.to_string())).await?;

                if let Some(row) = result.next().await? {
                    let pairs: Vec<Vec<String>> = row.get("disjoint_pairs")?;

                    // Group pairs into maximal disjoint sets
                    let mut disjoint_sets: Vec<HashSet<String>> = Vec::new();
                    for pair in pairs {
                        if pair.len() != 2 {
                            continue;
                        }

                        let mut found = false;
                        for set in &mut disjoint_sets {
                            if set.contains(&pair[0]) || set.contains(&pair[1]) {
                                set.insert(pair[0].clone());
                                set.insert(pair[1].clone());
                                found = true;
                                break;
                            }
                        }

                        if !found {
                            let mut new_set = HashSet::new();
                            new_set.insert(pair[0].clone());
                            new_set.insert(pair[1].clone());
                            disjoint_sets.push(new_set);
                        }
                    }

                    Ok(disjoint_sets)
                } else {
                    Ok(Vec::new())
                }
            })
        }).await
    }

    /// Load equivalent genres
    async fn load_equivalent_genres(&self) -> OntologyResult<HashMap<String, HashSet<String>>> {
        let query = r#"
            MATCH (g1:Genre)-[:EQUIVALENT_TO]-(g2:Genre)
            RETURN g1.name as genre, collect(g2.name) as equivalents
        "#;

        let mut equivalents = HashMap::new();

        self.execute_with_retry(|| {
            Box::pin(async {
                let mut result = self.graph.execute(Query::new(query.to_string())).await?;

                while let Some(row) = result.next().await? {
                    let genre: String = row.get("genre")?;
                    let equivs: Vec<String> = row.get("equivalents")?;
                    equivalents.insert(genre, equivs.into_iter().collect());
                }

                Ok(equivalents)
            })
        }).await
    }

    /// Load semantic tag hierarchy
    async fn load_tag_hierarchy(&self) -> OntologyResult<HashMap<String, HashSet<String>>> {
        let query = r#"
            MATCH (child:Tag)-[:SUBTAG_OF]->(parent:Tag)
            RETURN child.name as child, collect(parent.name) as parents
        "#;

        let mut hierarchy = HashMap::new();

        self.execute_with_retry(|| {
            Box::pin(async {
                let mut result = self.graph.execute(Query::new(query.to_string())).await?;

                while let Some(row) = result.next().await? {
                    let child: String = row.get("child")?;
                    let parents: Vec<String> = row.get("parents")?;
                    hierarchy.insert(child, parents.into_iter().collect());
                }

                Ok(hierarchy)
            })
        }).await
    }

    /// Store inferred axioms back to Neo4j
    pub async fn store_inferred_axioms(&self, axioms: &[super::reasoner::InferredMediaAxiom]) -> OntologyResult<()> {
        for chunk in axioms.chunks(BATCH_SIZE) {
            for axiom in chunk {
                let query = match axiom.axiom_type {
                    super::reasoner::MediaAxiomType::SubGenreOf => {
                        if let Some(ref parent) = axiom.object {
                            format!(
                                r#"
                                MATCH (child:Genre {{name: $child}}), (parent:Genre {{name: $parent}})
                                MERGE (child)-[r:SUBGENRE_OF]->(parent)
                                SET r.inferred = true, r.confidence = $confidence, r.reasoning = $reasoning
                                "#
                            )
                        } else {
                            continue;
                        }
                    }
                    super::reasoner::MediaAxiomType::SimilarMood => {
                        if let Some(ref mood2) = axiom.object {
                            format!(
                                r#"
                                MATCH (m1:Mood {{name: $subject}}), (m2:Mood {{name: $object}})
                                MERGE (m1)-[r:SIMILAR_TO]->(m2)
                                SET r.confidence = $confidence, r.reasoning = $reasoning
                                "#
                            )
                        } else {
                            continue;
                        }
                    }
                    _ => continue,
                };

                self.execute_with_retry(|| {
                    let q = Query::new(query.clone())
                        .param("subject", axiom.subject.clone())
                        .param("object", axiom.object.clone().unwrap_or_default())
                        .param("confidence", axiom.confidence as f64)
                        .param("reasoning", axiom.reasoning.clone());
                    Box::pin(async move {
                        self.graph.run(q).await
                    })
                }).await?;
            }
        }

        Ok(())
    }
}

/// Helper functions for RDF parsing
fn extract_label(uri: &str) -> String {
    uri.rsplit('#').next()
        .or_else(|| uri.rsplit('/').next())
        .unwrap_or(uri)
        .to_string()
}

fn extract_type(uri: &str) -> String {
    // Extract type from URI patterns like http://...#MediaType
    if uri.contains("Genre") {
        "Genre".to_string()
    } else if uri.contains("Mood") {
        "Mood".to_string()
    } else if uri.contains("Media") {
        "Media".to_string()
    } else if uri.contains("User") {
        "User".to_string()
    } else {
        "Entity".to_string()
    }
}

/// Helper function to create test ontology without Neo4j
pub fn create_test_ontology() -> MediaOntology {
    let mut ontology = MediaOntology::default();

    // Sample genre hierarchy
    ontology.genre_hierarchy.insert(
        "Thriller".to_string(),
        vec!["Drama".to_string()].into_iter().collect()
    );
    ontology.genre_hierarchy.insert(
        "PsychologicalThriller".to_string(),
        vec!["Thriller".to_string()].into_iter().collect()
    );

    // Disjoint genres
    ontology.disjoint_genres.push(
        vec!["Comedy".to_string(), "Horror".to_string()].into_iter().collect()
    );

    // Sample moods
    ontology.mood_relations.insert("Tense".to_string(), Mood {
        name: "Tense".to_string(),
        valence: -0.3,
        arousal: 0.8,
        dominance: 0.4,
        related_moods: vec!["Anxious".to_string()],
    });

    ontology
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extract_label() {
        assert_eq!(extract_label("http://example.org/ontology#Genre"), "Genre");
        assert_eq!(extract_label("http://example.org/ontology/Mood"), "Mood");
    }

    #[test]
    fn test_extract_type() {
        assert_eq!(extract_type("http://example.org#GenreRock"), "Genre");
        assert_eq!(extract_type("http://example.org#MoodHappy"), "Mood");
    }
}
