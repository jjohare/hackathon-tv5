// Semantic Search and Recommendation Module
// High-level recommendation engine with GPU-accelerated semantic search,
// OWL reasoning, and explainable path discovery

pub mod cache;
pub mod explanation;
pub mod path_discovery;
pub mod ranking;
pub mod recommendation;

pub use cache::{QueryCache, CacheEntry, CacheConfig};
pub use explanation::{Explanation, ExplanationGenerator, ExplanationPath};
pub use path_discovery::{SemanticPathDiscovery, SemanticPath, PathNode};
pub use ranking::{RankingEngine, ScoringConfig, RankedResult};
pub use recommendation::{RecommendationEngine, Recommendation, SimilarContent, Context};

use std::sync::Arc;
use anyhow::Result;

/// Re-exports from related modules
pub type UserId = String;
pub type ContentId = String;

/// Configuration for the semantic search system
#[derive(Debug, Clone)]
pub struct SemanticSearchConfig {
    pub vector_dimension: usize,
    pub max_cache_entries: usize,
    pub cache_ttl_seconds: u64,
    pub similarity_threshold: f32,
    pub max_path_depth: usize,
    pub batch_size: usize,
}

impl Default for SemanticSearchConfig {
    fn default() -> Self {
        Self {
            vector_dimension: 768,
            max_cache_entries: 10000,
            cache_ttl_seconds: 3600,
            similarity_threshold: 0.7,
            max_path_depth: 5,
            batch_size: 32,
        }
    }
}

/// Main semantic search system coordinator
pub struct SemanticSearchSystem {
    config: SemanticSearchConfig,
    recommendation_engine: Arc<RecommendationEngine>,
    path_discovery: Arc<SemanticPathDiscovery>,
    explanation_generator: Arc<ExplanationGenerator>,
    ranking_engine: Arc<RankingEngine>,
}

impl SemanticSearchSystem {
    pub fn new(config: SemanticSearchConfig) -> Result<Self> {
        let cache = Arc::new(QueryCache::new(
            config.max_cache_entries,
            config.cache_ttl_seconds,
        )?);

        let ranking_engine = Arc::new(RankingEngine::new(ScoringConfig::default())?);
        let explanation_generator = Arc::new(ExplanationGenerator::new()?);
        let path_discovery = Arc::new(SemanticPathDiscovery::new(config.max_path_depth)?);

        // Note: GPU engine, ontology, and vector DB would be injected in production
        let recommendation_engine = Arc::new(RecommendationEngine::new(
            cache.clone(),
            ranking_engine.clone(),
        )?);

        Ok(Self {
            config,
            recommendation_engine,
            path_discovery,
            explanation_generator,
            ranking_engine,
        })
    }

    /// Get personalized recommendations for a user
    pub async fn get_recommendations(
        &self,
        user_id: UserId,
        context: Context,
        limit: usize,
    ) -> Result<Vec<Recommendation>> {
        self.recommendation_engine
            .recommend(user_id, context, limit)
            .await
    }

    /// Find semantically similar content
    pub async fn find_similar(
        &self,
        content_id: ContentId,
        k: usize,
    ) -> Result<Vec<SimilarContent>> {
        self.recommendation_engine
            .find_similar(content_id, k)
            .await
    }

    /// Discover semantic path between two content items
    pub async fn discover_path(
        &self,
        from: ContentId,
        to: ContentId,
    ) -> Result<Vec<SemanticPath>> {
        self.path_discovery
            .find_paths(from, to)
            .await
    }

    /// Get explanation for a recommendation
    pub async fn explain_recommendation(
        &self,
        user_id: UserId,
        content_id: ContentId,
    ) -> Result<Explanation> {
        self.explanation_generator
            .generate_explanation(user_id, content_id)
            .await
    }

    /// Clear all caches
    pub async fn clear_caches(&self) -> Result<()> {
        self.recommendation_engine.clear_cache().await
    }

    /// Get cache statistics
    pub async fn cache_stats(&self) -> Result<serde_json::Value> {
        self.recommendation_engine.cache_stats().await
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_semantic_search_system_creation() {
        let config = SemanticSearchConfig::default();
        let system = SemanticSearchSystem::new(config);
        assert!(system.is_ok());
    }

    #[tokio::test]
    async fn test_default_config() {
        let config = SemanticSearchConfig::default();
        assert_eq!(config.vector_dimension, 768);
        assert_eq!(config.max_cache_entries, 10000);
        assert!(config.similarity_threshold > 0.0);
    }
}
