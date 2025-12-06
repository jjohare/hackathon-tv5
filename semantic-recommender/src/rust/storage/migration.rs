/// Migration Module
///
/// Handles migration from Neo4j-only to hybrid Milvus+Neo4j+PostgreSQL architecture.

use std::collections::HashMap;
use std::sync::Arc;
use neo4rs::{Graph, Query};
use serde::{Deserialize, Serialize};

use super::{
    MilvusClient, Neo4jClient, AgentDBCoordinator,
    StorageError, StorageResult,
};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MigrationStats {
    pub embeddings_migrated: u64,
    pub relationships_preserved: u64,
    pub policies_created: u64,
    pub errors: u64,
    pub duration_secs: f64,
}

#[derive(Debug, Clone)]
pub struct MigrationConfig {
    pub batch_size: usize,
    pub parallel_workers: usize,
    pub verify_data: bool,
    pub dry_run: bool,
}

impl Default for MigrationConfig {
    fn default() -> Self {
        Self {
            batch_size: 1000,
            parallel_workers: 4,
            verify_data: true,
            dry_run: false,
        }
    }
}

/// Hybrid migration coordinator
pub struct HybridMigration {
    source_neo4j: Arc<Graph>,
    target_milvus: Arc<MilvusClient>,
    target_neo4j: Arc<Neo4jClient>,
    target_agentdb: Arc<AgentDBCoordinator>,
    config: MigrationConfig,
}

impl HybridMigration {
    pub fn new(
        source_neo4j: Arc<Graph>,
        target_milvus: Arc<MilvusClient>,
        target_neo4j: Arc<Neo4jClient>,
        target_agentdb: Arc<AgentDBCoordinator>,
        config: MigrationConfig,
    ) -> Self {
        Self {
            source_neo4j,
            target_milvus,
            target_neo4j,
            target_agentdb,
            config,
        }
    }

    /// Migrate embeddings from Neo4j to Milvus
    pub async fn migrate_embeddings(&self) -> StorageResult<MigrationStats> {
        let start = std::time::Instant::now();
        log::info!("Starting embedding migration with batch_size={}", self.config.batch_size);

        let mut stats = MigrationStats {
            embeddings_migrated: 0,
            relationships_preserved: 0,
            policies_created: 0,
            errors: 0,
            duration_secs: 0.0,
        };

        // Query all media content with embeddings from Neo4j
        let query = Query::new(
            r#"
            MATCH (m:MediaContent)
            WHERE m.embedding IS NOT NULL
            RETURN m.id as id,
                   m.title as title,
                   m.embedding as embedding,
                   m.metadata as metadata,
                   m.created_at as created_at
            "#
        );

        let mut result = self.source_neo4j
            .execute(query)
            .await
            .map_err(|e| StorageError::Neo4j(format!("Query failed: {}", e)))?;

        let mut batch_vectors = Vec::new();
        let mut batch_metadata = Vec::new();
        let mut batch_count = 0;

        while let Some(row) = result.next()
            .await
            .map_err(|e| StorageError::Neo4j(format!("Row fetch failed: {}", e)))?
        {
            match self.extract_embedding_data(&row) {
                Ok((id, embedding, metadata)) => {
                    if self.config.dry_run {
                        log::debug!("Dry run: would migrate content {} with embedding dim={}", id, embedding.len());
                        stats.embeddings_migrated += 1;
                        continue;
                    }

                    batch_vectors.push(embedding);
                    batch_metadata.push(metadata);
                    batch_count += 1;

                    // Process batch when full
                    if batch_count >= self.config.batch_size {
                        match self.insert_batch(&batch_vectors, &batch_metadata).await {
                            Ok(count) => {
                                stats.embeddings_migrated += count;
                                log::info!("Migrated batch of {} embeddings", count);
                            }
                            Err(e) => {
                                log::error!("Batch insert failed: {}", e);
                                stats.errors += batch_count as u64;
                            }
                        }

                        batch_vectors.clear();
                        batch_metadata.clear();
                        batch_count = 0;
                    }
                }
                Err(e) => {
                    log::error!("Failed to extract embedding data: {}", e);
                    stats.errors += 1;
                }
            }
        }

        // Process final batch
        if !batch_vectors.is_empty() && !self.config.dry_run {
            match self.insert_batch(&batch_vectors, &batch_metadata).await {
                Ok(count) => {
                    stats.embeddings_migrated += count;
                    log::info!("Migrated final batch of {} embeddings", count);
                }
                Err(e) => {
                    log::error!("Final batch insert failed: {}", e);
                    stats.errors += batch_count as u64;
                }
            }
        }

        stats.duration_secs = start.elapsed().as_secs_f64();

        log::info!(
            "Migration complete: migrated={}, errors={}, duration={:.2}s",
            stats.embeddings_migrated,
            stats.errors,
            stats.duration_secs
        );

        Ok(stats)
    }

    /// Preserve graph relationships (keep in Neo4j)
    pub async fn preserve_relationships(&self) -> StorageResult<u64> {
        log::info!("Verifying relationship preservation in Neo4j");

        let query = Query::new(
            r#"
            MATCH (m1:MediaContent)-[r]->(m2:MediaContent)
            WHERE type(r) IN ['SIMILAR_TO', 'RELATED_TO', 'SEQUEL_OF', 'BELONGS_TO']
            RETURN count(r) as relationship_count
            "#
        );

        let mut result = self.source_neo4j
            .execute(query)
            .await
            .map_err(|e| StorageError::Neo4j(format!("Query failed: {}", e)))?;

        if let Some(row) = result.next()
            .await
            .map_err(|e| StorageError::Neo4j(format!("Row fetch failed: {}", e)))?
        {
            let count: i64 = row.get("relationship_count")
                .map_err(|e| StorageError::Neo4j(format!("Get count failed: {}", e)))?;

            log::info!("Preserved {} relationships in Neo4j", count);
            Ok(count as u64)
        } else {
            Ok(0)
        }
    }

    /// Migrate user interaction data to AgentDB policies
    pub async fn migrate_user_policies(&self) -> StorageResult<u64> {
        log::info!("Migrating user interaction history to AgentDB policies");

        let query = Query::new(
            r#"
            MATCH (u:User)-[interaction:VIEWED|LIKED|RATED]->(m:MediaContent)
            MATCH (m)-[:BELONGS_TO]->(g:Genre)
            WITH u.id as user_id,
                 collect(DISTINCT g.name) as genres,
                 count(interaction) as interaction_count,
                 avg(CASE WHEN type(interaction) = 'RATED' THEN interaction.rating ELSE 5.0 END) as avg_rating
            RETURN user_id, genres, interaction_count, avg_rating
            "#
        );

        let mut result = self.source_neo4j
            .execute(query)
            .await
            .map_err(|e| StorageError::Neo4j(format!("Query failed: {}", e)))?;

        let mut policies_created = 0u64;

        while let Some(row) = result.next()
            .await
            .map_err(|e| StorageError::Neo4j(format!("Row fetch failed: {}", e)))?
        {
            if self.config.dry_run {
                policies_created += 1;
                continue;
            }

            let user_id: String = row.get("user_id")
                .map_err(|e| StorageError::Neo4j(format!("Get user_id failed: {}", e)))?;

            let genres: Vec<String> = row.get("genres").unwrap_or_default();
            let interaction_count: i64 = row.get("interaction_count").unwrap_or(0);
            let avg_rating: f64 = row.get("avg_rating").unwrap_or(3.0);

            // Build preference map from genre interactions
            let mut preferences = HashMap::new();
            for genre in genres {
                let preference_score = (avg_rating / 5.0) as f32;
                preferences.insert(genre, preference_score);
            }

            let policy = super::postgres_store::UserPolicy {
                user_id: user_id.clone(),
                context: "default".to_string(),
                preferences,
                constraints: Vec::new(),
                learning_rate: 0.1,
                exploration_rate: 0.15,
            };

            match self.target_agentdb.update_policy(&policy).await {
                Ok(_) => {
                    policies_created += 1;
                    log::debug!("Created policy for user {}", user_id);
                }
                Err(e) => {
                    log::error!("Failed to create policy for user {}: {}", user_id, e);
                }
            }
        }

        log::info!("Created {} user policies in AgentDB", policies_created);
        Ok(policies_created)
    }

    /// Verify migration data integrity
    pub async fn verify_migration(&self) -> StorageResult<bool> {
        log::info!("Verifying migration data integrity");

        // Check embedding count in Milvus
        let milvus_metrics = self.target_milvus.get_metrics();
        let milvus_count = milvus_metrics.get("total_inserts").copied().unwrap_or(0);

        // Check content count in Neo4j
        let query = Query::new("MATCH (m:MediaContent) RETURN count(m) as count");

        let mut result = self.source_neo4j
            .execute(query)
            .await
            .map_err(|e| StorageError::Neo4j(format!("Query failed: {}", e)))?;

        let neo4j_count = if let Some(row) = result.next().await.ok().flatten() {
            row.get::<i64>("count").unwrap_or(0) as u64
        } else {
            0
        };

        let verified = milvus_count == neo4j_count;

        log::info!(
            "Verification: Milvus={}, Neo4j={}, Match={}",
            milvus_count,
            neo4j_count,
            verified
        );

        Ok(verified)
    }

    /// Run complete migration
    pub async fn run_full_migration(&self) -> StorageResult<MigrationStats> {
        log::info!("Starting full hybrid migration");

        let start = std::time::Instant::now();

        // Step 1: Migrate embeddings
        let mut stats = self.migrate_embeddings().await?;

        // Step 2: Preserve relationships
        stats.relationships_preserved = self.preserve_relationships().await?;

        // Step 3: Migrate user policies
        stats.policies_created = self.migrate_user_policies().await?;

        // Step 4: Verify if enabled
        if self.config.verify_data && !self.config.dry_run {
            let verified = self.verify_migration().await?;
            if !verified {
                log::warn!("Migration verification failed - data mismatch detected");
            }
        }

        stats.duration_secs = start.elapsed().as_secs_f64();

        log::info!("Full migration complete: {:?}", stats);

        Ok(stats)
    }

    async fn insert_batch(
        &self,
        vectors: &[Vec<f32>],
        metadata: &[HashMap<String, String>],
    ) -> StorageResult<u64> {
        self.target_milvus
            .insert("media_embeddings", vectors.to_vec(), metadata.to_vec())
            .await
            .map(|ids| ids.len() as u64)
    }

    fn extract_embedding_data(
        &self,
        row: &neo4rs::Row,
    ) -> StorageResult<(String, Vec<f32>, HashMap<String, String>)> {
        let id: String = row.get("id")
            .map_err(|e| StorageError::Neo4j(format!("Get id failed: {}", e)))?;

        let title: String = row.get("title").unwrap_or_else(|_| "Unknown".to_string());

        let embedding: Vec<f64> = row.get("embedding")
            .map_err(|e| StorageError::Neo4j(format!("Get embedding failed: {}", e)))?;

        let embedding_f32: Vec<f32> = embedding.into_iter().map(|v| v as f32).collect();

        let mut metadata = HashMap::new();
        metadata.insert("id".to_string(), id.clone());
        metadata.insert("title".to_string(), title);

        if let Ok(created_at) = row.get::<chrono::DateTime<chrono::Utc>>("created_at") {
            metadata.insert("created_at".to_string(), created_at.to_rfc3339());
        }

        Ok((id, embedding_f32, metadata))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_migration_config_default() {
        let config = MigrationConfig::default();
        assert_eq!(config.batch_size, 1000);
        assert_eq!(config.parallel_workers, 4);
        assert!(config.verify_data);
        assert!(!config.dry_run);
    }
}
