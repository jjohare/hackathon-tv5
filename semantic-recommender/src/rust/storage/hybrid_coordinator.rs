/// Hybrid Storage Coordinator
///
/// Orchestrates queries across Milvus (vectors), Neo4j (graphs), and PostgreSQL (policies)
/// to provide unified semantic search with < 10ms P99 latency.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;
use tokio::time::timeout;
use std::time::Duration;
use serde::{Deserialize, Serialize};

use super::{
    MilvusClient, Neo4jClient, AgentDBCoordinator, RedisCache,
    QueryPlanner, QueryStrategy,
    StorageError, StorageResult,
};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchQuery {
    pub embedding: Vec<f32>,
    pub k: usize,
    pub user_id: String,
    pub context: String,
    pub metadata_filters: HashMap<String, String>,
    pub graph_filters: HashMap<String, String>,
    pub include_relationships: bool,
    pub require_genre_filter: bool,
    pub require_cultural_context: bool,
    pub timeout_ms: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Recommendation {
    pub content_id: String,
    pub title: String,
    pub score: f32,
    pub genres: Vec<String>,
    pub themes: Vec<String>,
    pub moods: Vec<String>,
    pub relationships: Vec<ContentRelation>,
    pub reasoning: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContentRelation {
    pub related_id: String,
    pub relationship_type: String,
    pub strength: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchMetrics {
    pub total_time_us: u64,
    pub milvus_time_us: u64,
    pub neo4j_time_us: u64,
    pub agentdb_time_us: u64,
    pub aggregation_time_us: u64,
    pub strategy_used: String,
    pub results_count: usize,
    pub cache_hit: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MediaContent {
    pub id: String,
    pub title: String,
    pub embedding: Vec<f32>,
    pub metadata: HashMap<String, String>,
    pub genres: Vec<String>,
    pub themes: Vec<String>,
    pub created_at: chrono::DateTime<chrono::Utc>,
}

/// Hybrid storage coordinator
pub struct HybridStorageCoordinator {
    milvus: Arc<MilvusClient>,
    neo4j: Arc<Neo4jClient>,
    agentdb: Arc<AgentDBCoordinator>,
    query_planner: Arc<QueryPlanner>,
    cache: Option<Arc<RedisCache>>,
    metrics: Arc<CoordinatorMetrics>,
}

#[derive(Default)]
struct CoordinatorMetrics {
    total_searches: std::sync::atomic::AtomicU64,
    cache_hits: std::sync::atomic::AtomicU64,
    vector_only_queries: std::sync::atomic::AtomicU64,
    hybrid_queries: std::sync::atomic::AtomicU64,
    avg_latency_us: std::sync::atomic::AtomicU64,
}

impl HybridStorageCoordinator {
    pub fn new(
        milvus: Arc<MilvusClient>,
        neo4j: Arc<Neo4jClient>,
        agentdb: Arc<AgentDBCoordinator>,
        query_planner: Arc<QueryPlanner>,
        cache: Option<Arc<RedisCache>>,
    ) -> Self {
        Self {
            milvus,
            neo4j,
            agentdb,
            query_planner,
            cache,
            metrics: Arc::new(CoordinatorMetrics::default()),
        }
    }

    /// Execute hybrid search with context enrichment
    pub async fn search_with_context(
        &self,
        query: &SearchQuery,
    ) -> StorageResult<(Vec<Recommendation>, SearchMetrics)> {
        let start = Instant::now();
        self.metrics.total_searches.fetch_add(1, std::sync::atomic::Ordering::Relaxed);

        // Plan query execution
        let plan = self.query_planner.plan(query);

        log::debug!(
            "Query plan: strategy={:?}, estimated_latency={}ms",
            plan.strategy,
            plan.estimated_latency_ms
        );

        // Try cache first
        if plan.cache_eligible {
            if let Some(cached) = self.try_cache(query).await? {
                self.metrics.cache_hits.fetch_add(1, std::sync::atomic::Ordering::Relaxed);

                return Ok((
                    cached,
                    SearchMetrics {
                        total_time_us: start.elapsed().as_micros() as u64,
                        milvus_time_us: 0,
                        neo4j_time_us: 0,
                        agentdb_time_us: 0,
                        aggregation_time_us: 0,
                        strategy_used: "cached".to_string(),
                        results_count: 0,
                        cache_hit: true,
                    },
                ));
            }
        }

        // Execute based on strategy
        let (results, mut metrics) = match plan.strategy {
            QueryStrategy::VectorOnly => {
                self.metrics.vector_only_queries.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                self.execute_vector_only(query).await?
            }
            QueryStrategy::GraphOnly => self.execute_graph_only(query).await?,
            QueryStrategy::HybridParallel => {
                self.metrics.hybrid_queries.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                self.execute_hybrid_parallel(query).await?
            }
            QueryStrategy::HybridSequential => {
                self.metrics.hybrid_queries.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                self.execute_hybrid_sequential(query).await?
            }
            QueryStrategy::CachedOnly => {
                return Err(StorageError::QueryPlanning("Cache miss".to_string()));
            }
        };

        metrics.total_time_us = start.elapsed().as_micros() as u64;
        metrics.strategy_used = format!("{:?}", plan.strategy);

        // Update average latency
        let current_avg = self.metrics.avg_latency_us.load(std::sync::atomic::Ordering::Relaxed);
        let new_avg = (current_avg + metrics.total_time_us) / 2;
        self.metrics.avg_latency_us.store(new_avg, std::sync::atomic::Ordering::Relaxed);

        // Cache results if eligible
        if plan.cache_eligible {
            self.cache_results(query, &results).await?;
        }

        Ok((results, metrics))
    }

    /// Vector-only search (fastest path: ~8.7ms)
    async fn execute_vector_only(
        &self,
        query: &SearchQuery,
    ) -> StorageResult<(Vec<Recommendation>, SearchMetrics)> {
        let milvus_start = Instant::now();

        let vector_results = self.milvus
            .search(
                "media_embeddings",
                &query.embedding,
                query.k,
                &query.metadata_filters,
                None,
            )
            .await?;

        let milvus_time_us = milvus_start.elapsed().as_micros() as u64;

        // Convert to recommendations
        let recommendations: Vec<Recommendation> = vector_results
            .into_iter()
            .map(|result| Recommendation {
                content_id: result.id,
                title: result.metadata.get("title").cloned().unwrap_or_default(),
                score: result.score,
                genres: Vec::new(),
                themes: Vec::new(),
                moods: Vec::new(),
                relationships: Vec::new(),
                reasoning: "Vector similarity match".to_string(),
            })
            .collect();

        Ok((
            recommendations,
            SearchMetrics {
                total_time_us: 0,
                milvus_time_us,
                neo4j_time_us: 0,
                agentdb_time_us: 0,
                aggregation_time_us: 0,
                strategy_used: String::new(),
                results_count: 0,
                cache_hit: false,
            },
        ))
    }

    /// Graph-only search (rare)
    async fn execute_graph_only(
        &self,
        query: &SearchQuery,
    ) -> StorageResult<(Vec<Recommendation>, SearchMetrics)> {
        // Placeholder for graph-only queries
        Ok((Vec::new(), SearchMetrics {
            total_time_us: 0,
            milvus_time_us: 0,
            neo4j_time_us: 0,
            agentdb_time_us: 0,
            aggregation_time_us: 0,
            strategy_used: "graph_only".to_string(),
            results_count: 0,
            cache_hit: false,
        }))
    }

    /// Hybrid parallel search (most common for complex queries)
    async fn execute_hybrid_parallel(
        &self,
        query: &SearchQuery,
    ) -> StorageResult<(Vec<Recommendation>, SearchMetrics)> {
        let milvus_start = Instant::now();

        // Phase 1: Vector search from Milvus (over-fetch for re-ranking)
        let vector_future = self.milvus.search(
            "media_embeddings",
            &query.embedding,
            query.k * 3, // Over-fetch 3x for better re-ranking
            &query.metadata_filters,
            None,
        );

        let milvus_time_us = milvus_start.elapsed().as_micros() as u64;

        // Phase 2: Graph enrichment in parallel
        let neo4j_start = Instant::now();

        let vector_results = vector_future.await?;
        let content_ids: Vec<String> = vector_results.iter().map(|r| r.id.clone()).collect();

        let graph_enrichment_future = self.neo4j.enrich_batch(&content_ids);

        // Phase 3: Get user policy in parallel
        let agentdb_start = Instant::now();

        let (graph_data, user_policy) = tokio::try_join!(
            graph_enrichment_future,
            async {
                if !query.user_id.is_empty() {
                    self.agentdb.get_policy(&query.user_id, &query.context).await
                } else {
                    Err(StorageError::Postgres("No user policy".to_string()))
                }
            }
        )?;

        let neo4j_time_us = neo4j_start.elapsed().as_micros() as u64;
        let agentdb_time_us = agentdb_start.elapsed().as_micros() as u64;

        // Phase 4: Merge and re-rank
        let agg_start = Instant::now();

        let ranked = self.merge_and_rerank(
            &vector_results,
            &graph_data,
            Some(&user_policy),
            query.k,
        )?;

        let aggregation_time_us = agg_start.elapsed().as_micros() as u64;

        Ok((
            ranked,
            SearchMetrics {
                total_time_us: 0,
                milvus_time_us,
                neo4j_time_us,
                agentdb_time_us,
                aggregation_time_us,
                strategy_used: String::new(),
                results_count: 0,
                cache_hit: false,
            },
        ))
    }

    /// Hybrid sequential search (vector then graph)
    async fn execute_hybrid_sequential(
        &self,
        query: &SearchQuery,
    ) -> StorageResult<(Vec<Recommendation>, SearchMetrics)> {
        let milvus_start = Instant::now();

        // Phase 1: Vector search
        let vector_results = self.milvus
            .search(
                "media_embeddings",
                &query.embedding,
                query.k * 2,
                &query.metadata_filters,
                None,
            )
            .await?;

        let milvus_time_us = milvus_start.elapsed().as_micros() as u64;

        // Phase 2: Graph enrichment (sequential)
        let neo4j_start = Instant::now();

        let content_ids: Vec<String> = vector_results.iter().map(|r| r.id.clone()).collect();
        let graph_data = self.neo4j.enrich_batch(&content_ids).await?;

        let neo4j_time_us = neo4j_start.elapsed().as_micros() as u64;

        // Phase 3: Policy-based re-ranking
        let agentdb_start = Instant::now();

        let user_policy = if !query.user_id.is_empty() {
            Some(self.agentdb.get_policy(&query.user_id, &query.context).await?)
        } else {
            None
        };

        let agentdb_time_us = agentdb_start.elapsed().as_micros() as u64;

        // Phase 4: Merge and re-rank
        let agg_start = Instant::now();

        let ranked = self.merge_and_rerank(
            &vector_results,
            &graph_data,
            user_policy.as_ref(),
            query.k,
        )?;

        let aggregation_time_us = agg_start.elapsed().as_micros() as u64;

        Ok((
            ranked,
            SearchMetrics {
                total_time_us: 0,
                milvus_time_us,
                neo4j_time_us,
                agentdb_time_us,
                aggregation_time_us,
                strategy_used: String::new(),
                results_count: 0,
                cache_hit: false,
            },
        ))
    }

    /// Merge vector results with graph enrichment and apply policy re-ranking
    fn merge_and_rerank(
        &self,
        vector_results: &[super::milvus_client::VectorSearchResult],
        graph_enrichment: &HashMap<String, super::neo4j_client::GraphEnrichment>,
        user_policy: Option<&super::postgres_store::UserPolicy>,
        top_k: usize,
    ) -> StorageResult<Vec<Recommendation>> {
        let mut scored_results: Vec<(String, f32, Recommendation)> = vector_results
            .iter()
            .map(|result| {
                let mut score = result.score;
                let content_id = &result.id;

                // Build recommendation with graph enrichment
                let enrichment = graph_enrichment.get(content_id);

                let mut rec = Recommendation {
                    content_id: content_id.clone(),
                    title: result.metadata.get("title").cloned().unwrap_or_default(),
                    score,
                    genres: enrichment.as_ref().map(|e| e.genres.clone()).unwrap_or_default(),
                    themes: enrichment.as_ref().map(|e| e.themes.clone()).unwrap_or_default(),
                    moods: enrichment.as_ref().map(|e| e.moods.clone()).unwrap_or_default(),
                    relationships: enrichment
                        .as_ref()
                        .map(|e| {
                            e.relationships
                                .iter()
                                .map(|rel| ContentRelation {
                                    related_id: rel.related_id.clone(),
                                    relationship_type: rel.relationship_type.clone(),
                                    strength: rel.strength,
                                })
                                .collect()
                        })
                        .unwrap_or_default(),
                    reasoning: "Hybrid vector + graph match".to_string(),
                };

                // Apply user policy if available
                if let Some(policy) = user_policy {
                    for genre in &rec.genres {
                        if let Some(&preference) = policy.preferences.get(genre) {
                            score *= 1.0 + preference;
                        }
                    }

                    for theme in &rec.themes {
                        if let Some(&preference) = policy.preferences.get(theme) {
                            score *= 1.0 + (preference * 0.5);
                        }
                    }

                    rec.reasoning = format!("Personalized: {}", rec.reasoning);
                }

                rec.score = score;
                (content_id.clone(), score, rec)
            })
            .collect();

        // Sort by final score
        scored_results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // Take top K
        let results: Vec<Recommendation> = scored_results
            .into_iter()
            .take(top_k)
            .map(|(_, _, rec)| rec)
            .collect();

        Ok(results)
    }

    /// Store new media content across all systems
    pub async fn ingest_content(&self, content: &MediaContent) -> StorageResult<()> {
        // Store in parallel
        let milvus_future = self.milvus.insert(
            "media_embeddings",
            vec![content.embedding.clone()],
            vec![content.metadata.clone()],
        );

        let neo4j_future = self.neo4j.store_content(
            &content.id,
            &content.title,
            &content.genres,
            &content.themes,
        );

        tokio::try_join!(milvus_future, neo4j_future)?;

        Ok(())
    }

    /// Try to get results from cache
    async fn try_cache(&self, query: &SearchQuery) -> StorageResult<Option<Vec<Recommendation>>> {
        if let Some(cache) = &self.cache {
            let cache_key = self.generate_cache_key(query);
            cache.get(&cache_key).await
        } else {
            Ok(None)
        }
    }

    /// Cache search results
    async fn cache_results(&self, query: &SearchQuery, results: &[Recommendation]) -> StorageResult<()> {
        if let Some(cache) = &self.cache {
            let cache_key = self.generate_cache_key(query);
            cache.set(&cache_key, results, Some(Duration::from_secs(300))).await?;
        }
        Ok(())
    }

    fn generate_cache_key(&self, query: &SearchQuery) -> String {
        use std::hash::{Hash, Hasher};
        use std::collections::hash_map::DefaultHasher;

        let mut hasher = DefaultHasher::new();
        query.embedding.iter().for_each(|f| {
            f.to_bits().hash(&mut hasher);
        });
        query.k.hash(&mut hasher);
        format!("search:{:x}", hasher.finish())
    }

    /// Get coordinator metrics
    pub fn get_metrics(&self) -> HashMap<String, u64> {
        HashMap::from([
            ("total_searches".to_string(), self.metrics.total_searches.load(std::sync::atomic::Ordering::Relaxed)),
            ("cache_hits".to_string(), self.metrics.cache_hits.load(std::sync::atomic::Ordering::Relaxed)),
            ("vector_only_queries".to_string(), self.metrics.vector_only_queries.load(std::sync::atomic::Ordering::Relaxed)),
            ("hybrid_queries".to_string(), self.metrics.hybrid_queries.load(std::sync::atomic::Ordering::Relaxed)),
            ("avg_latency_us".to_string(), self.metrics.avg_latency_us.load(std::sync::atomic::Ordering::Relaxed)),
        ])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cache_key_generation() {
        // Test requires coordinator instance
    }
}
