use std::sync::Arc;
use anyhow::{Result, Context};
use tokio::sync::RwLock;

use crate::gpu_engine::unified_gpu::GPUPipeline;
use crate::adaptive_sssp::{
    AdaptiveSsspEngine, AdaptiveSsspConfig, AlgorithmMode, SsspMetrics,
};

/// Viewing context for personalized recommendations
#[derive(Debug, Clone)]
pub struct ViewingContext {
    pub time_of_day: String,
    pub device: String,
    pub viewing_history: Vec<String>,
    pub preferences: Vec<String>,
}

/// Recommendation with metadata
#[derive(Debug, Clone)]
pub struct Recommendation {
    pub content_id: String,
    pub title: String,
    pub similarity_score: f32,
    pub policy_score: f32,
    pub final_score: f32,
    pub metadata: serde_json::Value,
}

/// Policy for re-ranking recommendations
#[derive(Debug, Clone)]
pub struct RecommendationPolicy {
    pub diversity_weight: f32,
    pub recency_weight: f32,
    pub popularity_weight: f32,
    pub personalization_weight: f32,
}

/// Unified recommendation engine with all GPU optimizations
pub struct RecommendationEngine {
    gpu_pipeline: Arc<GPUPipeline>,
    embeddings: Arc<RwLock<Vec<f32>>>,
    metadata: Arc<RwLock<Vec<ContentMetadata>>>,
    embedding_dim: usize,
    adaptive_sssp: Arc<RwLock<AdaptiveSsspEngine>>,
}

#[derive(Debug, Clone)]
struct ContentMetadata {
    content_id: String,
    title: String,
    genre: Vec<String>,
    release_date: String,
    popularity_score: f32,
    metadata: serde_json::Value,
}

impl RecommendationEngine {
    /// Create new recommendation engine with GPU acceleration
    pub async fn new(
        embeddings: Vec<f32>,
        embedding_dim: usize,
        metadata: Vec<ContentMetadata>,
    ) -> Result<Self> {
        let gpu_pipeline = Arc::new(
            GPUPipeline::new(&embeddings, embedding_dim)
                .context("Failed to initialize GPU pipeline")?
        );

        // Initialize adaptive SSSP with default configuration
        let num_nodes = metadata.len();
        let mut adaptive_sssp = AdaptiveSsspEngine::new(AdaptiveSsspConfig::default());
        adaptive_sssp.update_graph_stats(num_nodes, num_nodes * 10); // Estimate edges

        Ok(Self {
            gpu_pipeline,
            embeddings: Arc::new(RwLock::new(embeddings)),
            metadata: Arc::new(RwLock::new(metadata)),
            embedding_dim,
            adaptive_sssp: Arc::new(RwLock::new(adaptive_sssp)),
        })
    }

    /// Create engine with custom SSSP configuration
    pub async fn with_sssp_config(
        embeddings: Vec<f32>,
        embedding_dim: usize,
        metadata: Vec<ContentMetadata>,
        sssp_config: AdaptiveSsspConfig,
    ) -> Result<Self> {
        let gpu_pipeline = Arc::new(
            GPUPipeline::new(&embeddings, embedding_dim)
                .context("Failed to initialize GPU pipeline")?
        );

        let num_nodes = metadata.len();
        let mut adaptive_sssp = AdaptiveSsspEngine::new(sssp_config);
        adaptive_sssp.update_graph_stats(num_nodes, num_nodes * 10);

        Ok(Self {
            gpu_pipeline,
            embeddings: Arc::new(RwLock::new(embeddings)),
            metadata: Arc::new(RwLock::new(metadata)),
            embedding_dim,
            adaptive_sssp: Arc::new(RwLock::new(adaptive_sssp)),
        })
    }

    /// Get current adaptive SSSP metrics
    pub async fn get_sssp_metrics(&self) -> Option<SsspMetrics> {
        let sssp = self.adaptive_sssp.read().await;
        sssp.last_metrics()
    }

    /// Get selected algorithm mode
    pub async fn get_algorithm_mode(&self) -> AlgorithmMode {
        let sssp = self.adaptive_sssp.read().await;
        sssp.select_algorithm()
    }

    /// Set algorithm mode (override auto-selection)
    pub async fn set_algorithm_mode(&self, mode: AlgorithmMode) -> Result<()> {
        let mut sssp = self.adaptive_sssp.write().await;
        let mut config = AdaptiveSsspConfig::default();
        config.mode = mode;
        *sssp = AdaptiveSsspEngine::new(config);
        Ok(())
    }

    /// Get personalized recommendations for user
    ///
    /// # Arguments
    /// * `user_id` - User identifier
    /// * `context` - Viewing context (time, device, history)
    /// * `k` - Number of recommendations to return
    ///
    /// # Returns
    /// Ranked list of recommendations with scores
    pub async fn recommend(
        &self,
        user_id: &str,
        context: &ViewingContext,
        k: usize,
    ) -> Result<Vec<Recommendation>> {
        // Get user embedding (from AgentDB or cache)
        let user_embedding = self.get_user_embedding(user_id, context).await?;

        // GPU-accelerated similarity search (ALL 3 PHASES)
        // Over-fetch for re-ranking
        let candidate_k = k * 3;
        let (candidate_ids, similarities) = self.gpu_pipeline
            .search_knn(&user_embedding, candidate_k)
            .context("GPU search failed")?;

        // Enrich with metadata
        let metadata = self.metadata.read().await;
        let mut enriched: Vec<_> = candidate_ids
            .into_iter()
            .zip(similarities.into_iter())
            .filter_map(|(idx, sim)| {
                let meta = metadata.get(idx as usize)?;
                Some((meta.clone(), sim))
            })
            .collect();

        // Get personalized policy
        let policy = self.get_policy(user_id, context).await?;

        // Policy-based re-ranking
        let ranked = self.rerank_with_policy(enriched, &policy, context).await?;

        // Take top-k
        Ok(ranked.into_iter().take(k).collect())
    }

    /// Batch recommendations for multiple users
    pub async fn batch_recommend(
        &self,
        user_ids: &[String],
        contexts: &[ViewingContext],
        k: usize,
    ) -> Result<Vec<Vec<Recommendation>>> {
        let mut results = Vec::with_capacity(user_ids.len());

        for (user_id, context) in user_ids.iter().zip(contexts.iter()) {
            let recs = self.recommend(user_id, context, k).await?;
            results.push(recs);
        }

        Ok(results)
    }

    /// Get similar content (content-to-content)
    pub async fn similar_content(
        &self,
        content_id: &str,
        k: usize,
    ) -> Result<Vec<Recommendation>> {
        // Get content embedding
        let metadata = self.metadata.read().await;
        let content_idx = metadata
            .iter()
            .position(|m| m.content_id == content_id)
            .context("Content not found")?;

        let embeddings = self.embeddings.read().await;
        let content_embedding: Vec<f32> = embeddings
            .iter()
            .skip(content_idx * self.embedding_dim)
            .take(self.embedding_dim)
            .copied()
            .collect();

        // GPU search
        let (candidate_ids, similarities) = self.gpu_pipeline
            .search_knn(&content_embedding, k + 1)?; // +1 to exclude self

        // Convert to recommendations
        let mut results = Vec::new();
        for (idx, sim) in candidate_ids.into_iter().zip(similarities.into_iter()) {
            if idx as usize == content_idx {
                continue; // Skip self
            }

            let meta = metadata.get(idx as usize).context("Invalid index")?;
            results.push(Recommendation {
                content_id: meta.content_id.clone(),
                title: meta.title.clone(),
                similarity_score: sim,
                policy_score: 1.0,
                final_score: sim,
                metadata: meta.metadata.clone(),
            });
        }

        Ok(results.into_iter().take(k).collect())
    }

    /// Hybrid search: semantic + keyword filter
    pub async fn hybrid_search(
        &self,
        query_text: &str,
        filters: &SearchFilters,
        k: usize,
    ) -> Result<Vec<Recommendation>> {
        // Convert query to embedding (would use text encoder in production)
        let query_embedding = self.encode_text(query_text).await?;

        // GPU-accelerated semantic search
        let (candidate_ids, similarities) = self.gpu_pipeline
            .search_knn(&query_embedding, k * 5)?; // Over-fetch

        // Apply filters
        let metadata = self.metadata.read().await;
        let filtered: Vec<_> = candidate_ids
            .into_iter()
            .zip(similarities.into_iter())
            .filter_map(|(idx, sim)| {
                let meta = metadata.get(idx as usize)?;
                if self.matches_filters(meta, filters) {
                    Some((meta.clone(), sim))
                } else {
                    None
                }
            })
            .take(k)
            .collect();

        // Convert to recommendations
        Ok(filtered
            .into_iter()
            .map(|(meta, sim)| Recommendation {
                content_id: meta.content_id.clone(),
                title: meta.title.clone(),
                similarity_score: sim,
                policy_score: 1.0,
                final_score: sim,
                metadata: meta.metadata.clone(),
            })
            .collect())
    }

    async fn get_user_embedding(
        &self,
        user_id: &str,
        context: &ViewingContext,
    ) -> Result<Vec<f32>> {
        // In production: query AgentDB for user embedding
        // For now: average of viewing history embeddings
        let embeddings = self.embeddings.read().await;
        let metadata = self.metadata.read().await;

        let mut user_embedding = vec![0.0f32; self.embedding_dim];
        let mut count = 0;

        for content_id in &context.viewing_history {
            if let Some(idx) = metadata.iter().position(|m| &m.content_id == content_id) {
                for (i, &val) in embeddings
                    .iter()
                    .skip(idx * self.embedding_dim)
                    .take(self.embedding_dim)
                    .enumerate()
                {
                    user_embedding[i] += val;
                }
                count += 1;
            }
        }

        if count > 0 {
            for val in &mut user_embedding {
                *val /= count as f32;
            }
        }

        Ok(user_embedding)
    }

    async fn get_policy(
        &self,
        user_id: &str,
        context: &ViewingContext,
    ) -> Result<RecommendationPolicy> {
        // In production: retrieve from AgentDB based on user preferences
        Ok(RecommendationPolicy {
            diversity_weight: 0.3,
            recency_weight: 0.2,
            popularity_weight: 0.2,
            personalization_weight: 0.3,
        })
    }

    async fn rerank_with_policy(
        &self,
        candidates: Vec<(ContentMetadata, f32)>,
        policy: &RecommendationPolicy,
        context: &ViewingContext,
    ) -> Result<Vec<Recommendation>> {
        let mut ranked: Vec<_> = candidates
            .into_iter()
            .map(|(meta, sim)| {
                // Compute policy score
                let policy_score =
                    policy.personalization_weight * sim +
                    policy.popularity_weight * meta.popularity_score +
                    policy.recency_weight * self.recency_score(&meta) +
                    policy.diversity_weight * self.diversity_score(&meta, context);

                let final_score = 0.7 * sim + 0.3 * policy_score;

                Recommendation {
                    content_id: meta.content_id.clone(),
                    title: meta.title.clone(),
                    similarity_score: sim,
                    policy_score,
                    final_score,
                    metadata: meta.metadata.clone(),
                }
            })
            .collect();

        // Sort by final score
        ranked.sort_by(|a, b| {
            b.final_score.partial_cmp(&a.final_score).unwrap()
        });

        Ok(ranked)
    }

    fn recency_score(&self, meta: &ContentMetadata) -> f32 {
        // Simple recency: 1.0 for new, decays over time
        0.8 // Placeholder
    }

    fn diversity_score(&self, meta: &ContentMetadata, context: &ViewingContext) -> f32 {
        // Check if genres different from recent history
        let history_genres: std::collections::HashSet<_> =
            context.preferences.iter().collect();

        let new_genres = meta.genre
            .iter()
            .filter(|g| !history_genres.contains(g))
            .count();

        (new_genres as f32) / (meta.genre.len() as f32).max(1.0)
    }

    async fn encode_text(&self, text: &str) -> Result<Vec<f32>> {
        // Placeholder: would use sentence transformer in production
        Ok(vec![0.0f32; self.embedding_dim])
    }

    fn matches_filters(&self, meta: &ContentMetadata, filters: &SearchFilters) -> bool {
        if let Some(ref genres) = filters.genres {
            if !meta.genre.iter().any(|g| genres.contains(g)) {
                return false;
            }
        }

        if let Some(min_popularity) = filters.min_popularity {
            if meta.popularity_score < min_popularity {
                return false;
            }
        }

        true
    }
}

#[derive(Debug, Clone, Default)]
pub struct SearchFilters {
    pub genres: Option<Vec<String>>,
    pub min_popularity: Option<f32>,
    pub release_year_min: Option<i32>,
    pub release_year_max: Option<i32>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_engine_creation() {
        let embeddings = vec![0.1f32; 1024 * 100]; // 100 items
        let metadata = vec![
            ContentMetadata {
                content_id: "test1".to_string(),
                title: "Test Content".to_string(),
                genre: vec!["action".to_string()],
                release_date: "2024-01-01".to_string(),
                popularity_score: 0.8,
                metadata: serde_json::json!({}),
            }
        ];

        let engine = RecommendationEngine::new(embeddings, 1024, metadata).await;
        assert!(engine.is_ok());
    }
}
