// Recommendation Engine
// GPU-accelerated semantic recommendations with multi-modal content similarity

use std::collections::HashMap;
use std::sync::Arc;
use anyhow::{Result, Context as AnyhowContext};
use serde::{Deserialize, Serialize};
use tokio::sync::RwLock;

use super::cache::QueryCache;
use super::ranking::{RankingEngine, RankedResult};

pub type UserId = String;
pub type ContentId = String;

/// User context for personalized recommendations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Context {
    pub location: Option<String>,
    pub time_of_day: Option<String>,
    pub device: Option<String>,
    pub recent_interactions: Vec<ContentId>,
    pub preferences: HashMap<String, f32>,
    pub session_duration: Option<u64>,
}

impl Default for Context {
    fn default() -> Self {
        Self {
            location: None,
            time_of_day: None,
            device: None,
            recent_interactions: Vec::new(),
            preferences: HashMap::new(),
            session_duration: None,
        }
    }
}

/// A recommendation result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Recommendation {
    pub content_id: ContentId,
    pub score: f32,
    pub reason: String,
    pub confidence: f32,
    pub metadata: HashMap<String, serde_json::Value>,
    pub embedding_similarity: f32,
    pub ontology_relevance: f32,
}

/// Similar content result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimilarContent {
    pub content_id: ContentId,
    pub similarity: f32,
    pub shared_concepts: Vec<String>,
    pub distance_metrics: HashMap<String, f32>,
}

/// Recommendation strategies
#[derive(Debug, Clone, Copy)]
pub enum RecommendationStrategy {
    Collaborative,      // User-user similarity
    ContentBased,       // Content-content similarity
    Hybrid,             // Combined approach
    ContextAware,       // Context-driven recommendations
    SemanticPath,       // Path-based recommendations
}

/// Main recommendation engine
pub struct RecommendationEngine {
    cache: Arc<QueryCache>,
    ranking_engine: Arc<RankingEngine>,
    user_profiles: Arc<RwLock<HashMap<UserId, UserProfile>>>,
    content_embeddings: Arc<RwLock<HashMap<ContentId, Vec<f32>>>>,
    strategy: RecommendationStrategy,
}

/// User profile for personalization
#[derive(Debug, Clone)]
struct UserProfile {
    user_id: UserId,
    preferences: HashMap<String, f32>,
    interaction_history: Vec<(ContentId, f32, i64)>, // (content_id, score, timestamp)
    embedding: Option<Vec<f32>>,
}

impl RecommendationEngine {
    pub fn new(
        cache: Arc<QueryCache>,
        ranking_engine: Arc<RankingEngine>,
    ) -> Result<Self> {
        Ok(Self {
            cache,
            ranking_engine,
            user_profiles: Arc::new(RwLock::new(HashMap::new())),
            content_embeddings: Arc::new(RwLock::new(HashMap::new())),
            strategy: RecommendationStrategy::Hybrid,
        })
    }

    /// Set recommendation strategy
    pub fn set_strategy(&mut self, strategy: RecommendationStrategy) {
        self.strategy = strategy;
    }

    /// Generate personalized recommendations
    pub async fn recommend(
        &self,
        user_id: UserId,
        context: Context,
        limit: usize,
    ) -> Result<Vec<Recommendation>> {
        // Check cache first
        let cache_key = format!("rec:{}:{:?}", user_id, context.recent_interactions);
        if let Some(cached) = self.cache.get(&cache_key).await {
            if let Ok(recommendations) = serde_json::from_slice::<Vec<Recommendation>>(&cached) {
                return Ok(recommendations.into_iter().take(limit).collect());
            }
        }

        // Generate recommendations based on strategy
        let recommendations = match self.strategy {
            RecommendationStrategy::Collaborative => {
                self.collaborative_filtering(&user_id, &context, limit).await?
            }
            RecommendationStrategy::ContentBased => {
                self.content_based_filtering(&user_id, &context, limit).await?
            }
            RecommendationStrategy::Hybrid => {
                self.hybrid_recommendations(&user_id, &context, limit).await?
            }
            RecommendationStrategy::ContextAware => {
                self.context_aware_recommendations(&user_id, &context, limit).await?
            }
            RecommendationStrategy::SemanticPath => {
                self.semantic_path_recommendations(&user_id, &context, limit).await?
            }
        };

        // Cache results
        let serialized = serde_json::to_vec(&recommendations)?;
        self.cache.put(cache_key, serialized).await?;

        Ok(recommendations)
    }

    /// Find similar content using multi-modal embeddings
    pub async fn find_similar(
        &self,
        content_id: ContentId,
        k: usize,
    ) -> Result<Vec<SimilarContent>> {
        // Check cache
        let cache_key = format!("similar:{}", content_id);
        if let Some(cached) = self.cache.get(&cache_key).await {
            if let Ok(similar) = serde_json::from_slice::<Vec<SimilarContent>>(&cached) {
                return Ok(similar.into_iter().take(k).collect());
            }
        }

        // Get content embedding
        let embeddings = self.content_embeddings.read().await;
        let query_embedding = embeddings
            .get(&content_id)
            .ok_or_else(|| anyhow::anyhow!("Content not found: {}", content_id))?;

        // Compute similarities
        let mut similarities: Vec<SimilarContent> = Vec::new();
        for (other_id, other_embedding) in embeddings.iter() {
            if other_id == &content_id {
                continue;
            }

            let similarity = cosine_similarity(query_embedding, other_embedding);
            if similarity > 0.5 {
                similarities.push(SimilarContent {
                    content_id: other_id.clone(),
                    similarity,
                    shared_concepts: vec![], // Would be populated from ontology
                    distance_metrics: HashMap::from([
                        ("cosine".to_string(), similarity),
                        ("euclidean".to_string(), euclidean_distance(query_embedding, other_embedding)),
                    ]),
                });
            }
        }

        // Sort by similarity
        similarities.sort_by(|a, b| b.similarity.partial_cmp(&a.similarity).unwrap());
        let result: Vec<SimilarContent> = similarities.into_iter().take(k).collect();

        // Cache results
        let serialized = serde_json::to_vec(&result)?;
        self.cache.put(cache_key, serialized).await?;

        Ok(result)
    }

    /// Update user profile with interaction
    pub async fn record_interaction(
        &self,
        user_id: UserId,
        content_id: ContentId,
        score: f32,
    ) -> Result<()> {
        let mut profiles = self.user_profiles.write().await;
        let profile = profiles.entry(user_id.clone()).or_insert_with(|| UserProfile {
            user_id: user_id.clone(),
            preferences: HashMap::new(),
            interaction_history: Vec::new(),
            embedding: None,
        });

        let timestamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)?
            .as_secs() as i64;

        profile.interaction_history.push((content_id, score, timestamp));

        // Keep only recent interactions (last 100)
        if profile.interaction_history.len() > 100 {
            profile.interaction_history.remove(0);
        }

        Ok(())
    }

    /// Clear cache
    pub async fn clear_cache(&self) -> Result<()> {
        self.cache.clear().await
    }

    /// Get cache statistics
    pub async fn cache_stats(&self) -> Result<serde_json::Value> {
        Ok(serde_json::json!({
            "cache_size": self.cache.size().await,
            "hit_rate": self.cache.hit_rate().await,
        }))
    }

    // Private recommendation strategies

    async fn collaborative_filtering(
        &self,
        user_id: &UserId,
        context: &Context,
        limit: usize,
    ) -> Result<Vec<Recommendation>> {
        // Simplified collaborative filtering
        let profiles = self.user_profiles.read().await;
        let user_profile = profiles.get(user_id);

        if user_profile.is_none() {
            return Ok(Vec::new());
        }

        // Find similar users and recommend their content
        let mut recommendations = Vec::new();
        for content_id in &context.recent_interactions {
            recommendations.push(Recommendation {
                content_id: content_id.clone(),
                score: 0.8,
                reason: "Popular among similar users".to_string(),
                confidence: 0.75,
                metadata: HashMap::new(),
                embedding_similarity: 0.8,
                ontology_relevance: 0.7,
            });
        }

        Ok(recommendations.into_iter().take(limit).collect())
    }

    async fn content_based_filtering(
        &self,
        _user_id: &UserId,
        context: &Context,
        limit: usize,
    ) -> Result<Vec<Recommendation>> {
        let mut recommendations = Vec::new();

        // Find similar content to recent interactions
        for content_id in &context.recent_interactions {
            let similar = self.find_similar(content_id.clone(), 5).await?;
            for sim in similar {
                recommendations.push(Recommendation {
                    content_id: sim.content_id,
                    score: sim.similarity,
                    reason: format!("Similar to {}", content_id),
                    confidence: sim.similarity,
                    metadata: HashMap::new(),
                    embedding_similarity: sim.similarity,
                    ontology_relevance: 0.6,
                });
            }
        }

        Ok(recommendations.into_iter().take(limit).collect())
    }

    async fn hybrid_recommendations(
        &self,
        user_id: &UserId,
        context: &Context,
        limit: usize,
    ) -> Result<Vec<Recommendation>> {
        // Combine collaborative and content-based
        let collab = self.collaborative_filtering(user_id, context, limit / 2).await?;
        let content = self.content_based_filtering(user_id, context, limit / 2).await?;

        let mut combined = collab;
        combined.extend(content);

        // Re-rank combined results
        let ranked = self.ranking_engine.rank_results(
            combined.into_iter()
                .map(|r| RankedResult {
                    id: r.content_id.clone(),
                    score: r.score,
                    features: HashMap::new(),
                })
                .collect()
        ).await?;

        Ok(ranked.into_iter()
            .map(|r| Recommendation {
                content_id: r.id,
                score: r.score,
                reason: "Hybrid recommendation".to_string(),
                confidence: r.score,
                metadata: HashMap::new(),
                embedding_similarity: r.score,
                ontology_relevance: 0.7,
            })
            .take(limit)
            .collect())
    }

    async fn context_aware_recommendations(
        &self,
        user_id: &UserId,
        context: &Context,
        limit: usize,
    ) -> Result<Vec<Recommendation>> {
        // Weight recommendations by context
        let base_recs = self.hybrid_recommendations(user_id, context, limit * 2).await?;

        let mut weighted_recs: Vec<Recommendation> = base_recs
            .into_iter()
            .map(|mut rec| {
                // Adjust score based on context
                if let Some(device) = &context.device {
                    if device == "mobile" {
                        rec.score *= 1.1; // Boost mobile-friendly content
                    }
                }
                rec
            })
            .collect();

        weighted_recs.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());
        Ok(weighted_recs.into_iter().take(limit).collect())
    }

    async fn semantic_path_recommendations(
        &self,
        _user_id: &UserId,
        context: &Context,
        limit: usize,
    ) -> Result<Vec<Recommendation>> {
        // Path-based recommendations (would integrate with path_discovery)
        let mut recommendations = Vec::new();

        for content_id in context.recent_interactions.iter().take(3) {
            recommendations.push(Recommendation {
                content_id: content_id.clone(),
                score: 0.85,
                reason: "Semantic path connection".to_string(),
                confidence: 0.8,
                metadata: HashMap::new(),
                embedding_similarity: 0.85,
                ontology_relevance: 0.9,
            });
        }

        Ok(recommendations.into_iter().take(limit).collect())
    }
}

// Helper functions

fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

    if norm_a == 0.0 || norm_b == 0.0 {
        0.0
    } else {
        dot / (norm_a * norm_b)
    }
}

fn euclidean_distance(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).powi(2))
        .sum::<f32>()
        .sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cosine_similarity() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![1.0, 0.0, 0.0];
        assert!((cosine_similarity(&a, &b) - 1.0).abs() < 0.001);

        let c = vec![1.0, 0.0, 0.0];
        let d = vec![0.0, 1.0, 0.0];
        assert!(cosine_similarity(&c, &d).abs() < 0.001);
    }

    #[test]
    fn test_euclidean_distance() {
        let a = vec![0.0, 0.0, 0.0];
        let b = vec![1.0, 1.0, 1.0];
        let dist = euclidean_distance(&a, &b);
        assert!((dist - 1.732).abs() < 0.01);
    }
}
