/// Neo4j Graph Client
///
/// Provides graph traversal and relationship enrichment for semantic search results.

use std::collections::HashMap;
use std::sync::Arc;
use neo4rs::{Graph, Query, Node, ConfigBuilder};
use serde::{Deserialize, Serialize};
use super::{StorageError, StorageResult};

#[derive(Debug, Clone)]
pub struct Neo4jConfig {
    pub uri: String,
    pub username: String,
    pub password: String,
    pub database: String,
    pub max_connections: usize,
    pub timeout_secs: u64,
}

impl Default for Neo4jConfig {
    fn default() -> Self {
        Self {
            uri: "bolt://localhost:7687".to_string(),
            username: "neo4j".to_string(),
            password: "password".to_string(),
            database: "neo4j".to_string(),
            max_connections: 32,
            timeout_secs: 10,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphEnrichment {
    pub content_id: String,
    pub genres: Vec<String>,
    pub themes: Vec<String>,
    pub moods: Vec<String>,
    pub relationships: Vec<ContentRelationship>,
    pub cultural_context: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContentRelationship {
    pub related_id: String,
    pub relationship_type: String,
    pub strength: f32,
}

/// Neo4j client optimized for graph traversal
pub struct Neo4jClient {
    graph: Arc<Graph>,
    config: Neo4jConfig,
}

impl Neo4jClient {
    pub async fn new(config: Neo4jConfig) -> StorageResult<Self> {
        let neo4j_config = ConfigBuilder::default()
            .uri(&config.uri)
            .user(&config.username)
            .password(&config.password)
            .db(&config.database)
            .max_connections(config.max_connections)
            .build()
            .map_err(|e| StorageError::Neo4j(format!("Config error: {}", e)))?;

        let graph = Graph::connect(neo4j_config)
            .await
            .map_err(|e| StorageError::Neo4j(format!("Connection failed: {}", e)))?;

        Ok(Self {
            graph: Arc::new(graph),
            config,
        })
    }

    /// Enrich content IDs with graph relationships
    pub async fn enrich_batch(
        &self,
        content_ids: &[String],
    ) -> StorageResult<HashMap<String, GraphEnrichment>> {
        if content_ids.is_empty() {
            return Ok(HashMap::new());
        }

        let query = Query::new(
            r#"
            MATCH (m:MediaContent) WHERE m.id IN $ids
            OPTIONAL MATCH (m)-[:BELONGS_TO]->(g:Genre)
            OPTIONAL MATCH (m)-[:HAS_THEME]->(t:Theme)
            OPTIONAL MATCH (m)-[:HAS_MOOD]->(mood:Mood)
            OPTIONAL MATCH (m)-[r:SIMILAR_TO|RELATED_TO]->(related:MediaContent)
            OPTIONAL MATCH (m)-[:IN_CONTEXT]->(c:CulturalContext)
            RETURN m.id as id,
                   collect(DISTINCT g.name) as genres,
                   collect(DISTINCT t.name) as themes,
                   collect(DISTINCT mood.name) as moods,
                   collect(DISTINCT {id: related.id, type: type(r), strength: r.strength}) as relationships,
                   c.region as cultural_context
            "#
        )
        .param("ids", content_ids.to_vec());

        let mut result = self.graph
            .execute(query)
            .await
            .map_err(|e| StorageError::Neo4j(format!("Query failed: {}", e)))?;

        let mut enrichments = HashMap::new();

        while let Some(row) = result.next().await.map_err(|e| StorageError::Neo4j(format!("Row fetch failed: {}", e)))? {
            let content_id: String = row.get("id")
                .map_err(|e| StorageError::Neo4j(format!("Missing id: {}", e)))?;

            let genres: Vec<String> = row.get("genres").unwrap_or_default();
            let themes: Vec<String> = row.get("themes").unwrap_or_default();
            let moods: Vec<String> = row.get("moods").unwrap_or_default();
            let cultural_context: Option<String> = row.get("cultural_context").ok();

            // Parse relationships
            let relationships_raw: Vec<HashMap<String, serde_json::Value>> =
                row.get("relationships").unwrap_or_default();

            let relationships: Vec<ContentRelationship> = relationships_raw
                .into_iter()
                .filter_map(|rel| {
                    Some(ContentRelationship {
                        related_id: rel.get("id")?.as_str()?.to_string(),
                        relationship_type: rel.get("type")?.as_str()?.to_string(),
                        strength: rel.get("strength")?.as_f64()? as f32,
                    })
                })
                .collect();

            enrichments.insert(content_id.clone(), GraphEnrichment {
                content_id,
                genres,
                themes,
                moods,
                relationships,
                cultural_context,
            });
        }

        Ok(enrichments)
    }

    /// Store new media content with relationships
    pub async fn store_content(
        &self,
        content_id: &str,
        title: &str,
        genres: &[String],
        themes: &[String],
    ) -> StorageResult<()> {
        let query = Query::new(
            r#"
            MERGE (m:MediaContent {id: $id})
            SET m.title = $title, m.updated_at = datetime()
            WITH m
            UNWIND $genres AS genre_name
            MERGE (g:Genre {name: genre_name})
            MERGE (m)-[:BELONGS_TO]->(g)
            WITH m
            UNWIND $themes AS theme_name
            MERGE (t:Theme {name: theme_name})
            MERGE (m)-[:HAS_THEME]->(t)
            "#
        )
        .param("id", content_id)
        .param("title", title)
        .param("genres", genres.to_vec())
        .param("themes", themes.to_vec());

        self.graph
            .run(query)
            .await
            .map_err(|e| StorageError::Neo4j(format!("Store failed: {}", e)))?;

        Ok(())
    }

    /// Create relationship between content items
    pub async fn create_relationship(
        &self,
        source_id: &str,
        target_id: &str,
        relationship_type: &str,
        strength: f32,
    ) -> StorageResult<()> {
        let query = Query::new(
            format!(
                r#"
                MATCH (a:MediaContent {{id: $source_id}})
                MATCH (b:MediaContent {{id: $target_id}})
                MERGE (a)-[r:{}]->(b)
                SET r.strength = $strength, r.updated_at = datetime()
                "#,
                relationship_type
            )
        )
        .param("source_id", source_id)
        .param("target_id", target_id)
        .param("strength", strength as f64);

        self.graph
            .run(query)
            .await
            .map_err(|e| StorageError::Neo4j(format!("Relationship creation failed: {}", e)))?;

        Ok(())
    }

    /// Find similar content using graph traversal
    pub async fn find_similar_by_graph(
        &self,
        content_id: &str,
        max_depth: usize,
        limit: usize,
    ) -> StorageResult<Vec<(String, f32)>> {
        let query = Query::new(
            format!(
                r#"
                MATCH (m:MediaContent {{id: $id}})
                MATCH path = (m)-[:SIMILAR_TO|RELATED_TO|BELONGS_TO*1..{}]-(similar:MediaContent)
                WHERE similar.id <> $id
                WITH similar, length(path) as depth, count(*) as connections
                RETURN similar.id as id, (1.0 / depth) * connections as score
                ORDER BY score DESC
                LIMIT $limit
                "#,
                max_depth
            )
        )
        .param("id", content_id)
        .param("limit", limit as i64);

        let mut result = self.graph
            .execute(query)
            .await
            .map_err(|e| StorageError::Neo4j(format!("Similar search failed: {}", e)))?;

        let mut similar = Vec::new();

        while let Some(row) = result.next().await.map_err(|e| StorageError::Neo4j(format!("Row fetch failed: {}", e)))? {
            let id: String = row.get("id").map_err(|e| StorageError::Neo4j(format!("Missing id: {}", e)))?;
            let score: f64 = row.get("score").map_err(|e| StorageError::Neo4j(format!("Missing score: {}", e)))?;
            similar.push((id, score as f32));
        }

        Ok(similar)
    }

    /// Health check
    pub async fn health_check(&self) -> StorageResult<bool> {
        let query = Query::new("RETURN 1 as health");

        match self.graph.execute(query).await {
            Ok(_) => Ok(true),
            Err(e) => {
                log::error!("Neo4j health check failed: {}", e);
                Ok(false)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_neo4j_config_default() {
        let config = Neo4jConfig::default();
        assert_eq!(config.uri, "bolt://localhost:7687");
        assert_eq!(config.max_connections, 32);
    }
}
