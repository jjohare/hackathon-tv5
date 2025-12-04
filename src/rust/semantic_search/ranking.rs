// Result Ranking and Scoring
// Multi-factor ranking engine for recommendation results

use std::collections::HashMap;
use std::sync::Arc;
use anyhow::Result;
use serde::{Deserialize, Serialize};
use tokio::sync::RwLock;

/// Configuration for scoring
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScoringConfig {
    pub weights: ScoringWeights,
    pub normalization: NormalizationMethod,
    pub diversity_factor: f32,
    pub freshness_decay: f32,
}

/// Weights for different scoring factors
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScoringWeights {
    pub semantic_similarity: f32,
    pub popularity: f32,
    pub freshness: f32,
    pub user_preference: f32,
    pub diversity: f32,
    pub contextual_relevance: f32,
}

impl Default for ScoringWeights {
    fn default() -> Self {
        Self {
            semantic_similarity: 0.35,
            popularity: 0.15,
            freshness: 0.10,
            user_preference: 0.25,
            diversity: 0.10,
            contextual_relevance: 0.05,
        }
    }
}

impl Default for ScoringConfig {
    fn default() -> Self {
        Self {
            weights: ScoringWeights::default(),
            normalization: NormalizationMethod::MinMax,
            diversity_factor: 0.2,
            freshness_decay: 0.1,
        }
    }
}

/// Normalization methods
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum NormalizationMethod {
    MinMax,
    ZScore,
    Softmax,
    None,
}

/// A result to be ranked
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RankedResult {
    pub id: String,
    pub score: f32,
    pub features: HashMap<String, f32>,
}

/// Detailed scoring breakdown
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScoringBreakdown {
    pub semantic_similarity: f32,
    pub popularity: f32,
    pub freshness: f32,
    pub user_preference: f32,
    pub diversity: f32,
    pub contextual_relevance: f32,
    pub final_score: f32,
}

/// Main ranking engine
pub struct RankingEngine {
    config: Arc<RwLock<ScoringConfig>>,
    feature_stats: Arc<RwLock<FeatureStatistics>>,
}

/// Statistics for feature normalization
#[derive(Debug)]
struct FeatureStatistics {
    means: HashMap<String, f32>,
    std_devs: HashMap<String, f32>,
    min_values: HashMap<String, f32>,
    max_values: HashMap<String, f32>,
}

impl FeatureStatistics {
    fn new() -> Self {
        Self {
            means: HashMap::new(),
            std_devs: HashMap::new(),
            min_values: HashMap::new(),
            max_values: HashMap::new(),
        }
    }

    fn update(&mut self, results: &[RankedResult]) {
        for result in results {
            for (feature_name, &value) in &result.features {
                // Update min/max
                self.min_values
                    .entry(feature_name.clone())
                    .and_modify(|min| *min = min.min(value))
                    .or_insert(value);

                self.max_values
                    .entry(feature_name.clone())
                    .and_modify(|max| *max = max.max(value))
                    .or_insert(value);
            }
        }

        // Update means and standard deviations
        for result in results {
            for (feature_name, &value) in &result.features {
                self.means
                    .entry(feature_name.clone())
                    .and_modify(|mean| *mean = (*mean + value) / 2.0)
                    .or_insert(value);
            }
        }
    }

    fn normalize(&self, feature: &str, value: f32, method: NormalizationMethod) -> f32 {
        match method {
            NormalizationMethod::MinMax => {
                let min = self.min_values.get(feature).copied().unwrap_or(0.0);
                let max = self.max_values.get(feature).copied().unwrap_or(1.0);
                if max - min > 0.0 {
                    (value - min) / (max - min)
                } else {
                    0.5
                }
            }
            NormalizationMethod::ZScore => {
                let mean = self.means.get(feature).copied().unwrap_or(0.0);
                let std_dev = self.std_devs.get(feature).copied().unwrap_or(1.0);
                if std_dev > 0.0 {
                    (value - mean) / std_dev
                } else {
                    0.0
                }
            }
            NormalizationMethod::Softmax => {
                // Single-value softmax (identity)
                value
            }
            NormalizationMethod::None => value,
        }
    }
}

impl RankingEngine {
    pub fn new(config: ScoringConfig) -> Result<Self> {
        Ok(Self {
            config: Arc::new(RwLock::new(config)),
            feature_stats: Arc::new(RwLock::new(FeatureStatistics::new())),
        })
    }

    /// Rank a list of results
    pub async fn rank_results(&self, mut results: Vec<RankedResult>) -> Result<Vec<RankedResult>> {
        let config = self.config.read().await;
        let mut stats = self.feature_stats.write().await;

        // Update statistics
        stats.update(&results);

        // Score each result
        for result in &mut results {
            result.score = self.calculate_score(result, &config, &stats).await?;
        }

        // Apply diversity penalty
        self.apply_diversity_penalty(&mut results, config.diversity_factor).await?;

        // Sort by score
        results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());

        Ok(results)
    }

    /// Get detailed scoring breakdown for a result
    pub async fn get_scoring_breakdown(
        &self,
        result: &RankedResult,
    ) -> Result<ScoringBreakdown> {
        let config = self.config.read().await;
        let stats = self.feature_stats.read().await;

        let semantic_similarity = self.get_normalized_feature(
            result,
            "semantic_similarity",
            &config,
            &stats,
        );
        let popularity = self.get_normalized_feature(
            result,
            "popularity",
            &config,
            &stats,
        );
        let freshness = self.get_normalized_feature(
            result,
            "freshness",
            &config,
            &stats,
        );
        let user_preference = self.get_normalized_feature(
            result,
            "user_preference",
            &config,
            &stats,
        );
        let diversity = self.get_normalized_feature(
            result,
            "diversity",
            &config,
            &stats,
        );
        let contextual_relevance = self.get_normalized_feature(
            result,
            "contextual_relevance",
            &config,
            &stats,
        );

        let final_score = semantic_similarity * config.weights.semantic_similarity
            + popularity * config.weights.popularity
            + freshness * config.weights.freshness
            + user_preference * config.weights.user_preference
            + diversity * config.weights.diversity
            + contextual_relevance * config.weights.contextual_relevance;

        Ok(ScoringBreakdown {
            semantic_similarity,
            popularity,
            freshness,
            user_preference,
            diversity,
            contextual_relevance,
            final_score,
        })
    }

    /// Update scoring configuration
    pub async fn update_config(&self, config: ScoringConfig) -> Result<()> {
        let mut current_config = self.config.write().await;
        *current_config = config;
        Ok(())
    }

    /// Get current configuration
    pub async fn get_config(&self) -> Result<ScoringConfig> {
        let config = self.config.read().await;
        Ok(config.clone())
    }

    // Private helper methods

    async fn calculate_score(
        &self,
        result: &RankedResult,
        config: &ScoringConfig,
        stats: &FeatureStatistics,
    ) -> Result<f32> {
        let semantic_similarity = self.get_normalized_feature(
            result,
            "semantic_similarity",
            config,
            stats,
        );
        let popularity = self.get_normalized_feature(
            result,
            "popularity",
            config,
            stats,
        );
        let freshness = self.get_normalized_feature(
            result,
            "freshness",
            config,
            stats,
        );
        let user_preference = self.get_normalized_feature(
            result,
            "user_preference",
            config,
            stats,
        );
        let diversity = self.get_normalized_feature(
            result,
            "diversity",
            config,
            stats,
        );
        let contextual_relevance = self.get_normalized_feature(
            result,
            "contextual_relevance",
            config,
            stats,
        );

        let score = semantic_similarity * config.weights.semantic_similarity
            + popularity * config.weights.popularity
            + freshness * config.weights.freshness
            + user_preference * config.weights.user_preference
            + diversity * config.weights.diversity
            + contextual_relevance * config.weights.contextual_relevance;

        Ok(score)
    }

    fn get_normalized_feature(
        &self,
        result: &RankedResult,
        feature_name: &str,
        config: &ScoringConfig,
        stats: &FeatureStatistics,
    ) -> f32 {
        let raw_value = result.features.get(feature_name).copied().unwrap_or(0.0);
        stats.normalize(feature_name, raw_value, config.normalization)
    }

    async fn apply_diversity_penalty(
        &self,
        results: &mut [RankedResult],
        diversity_factor: f32,
    ) -> Result<()> {
        // Simple diversity: penalize results too similar to higher-ranked ones
        let mut seen_features: Vec<HashMap<String, f32>> = Vec::new();

        for result in results.iter_mut() {
            let mut similarity_penalty = 0.0;

            for seen in &seen_features {
                let similarity = calculate_feature_similarity(&result.features, seen);
                if similarity > 0.8 {
                    similarity_penalty += diversity_factor * (similarity - 0.8);
                }
            }

            result.score *= 1.0 - similarity_penalty;
            seen_features.push(result.features.clone());
        }

        Ok(())
    }
}

// Helper functions

fn calculate_feature_similarity(
    features1: &HashMap<String, f32>,
    features2: &HashMap<String, f32>,
) -> f32 {
    let mut dot_product = 0.0;
    let mut norm1 = 0.0;
    let mut norm2 = 0.0;

    for (key, &val1) in features1 {
        if let Some(&val2) = features2.get(key) {
            dot_product += val1 * val2;
        }
        norm1 += val1 * val1;
    }

    for &val2 in features2.values() {
        norm2 += val2 * val2;
    }

    if norm1 > 0.0 && norm2 > 0.0 {
        dot_product / (norm1.sqrt() * norm2.sqrt())
    } else {
        0.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_ranking_engine_creation() {
        let config = ScoringConfig::default();
        let engine = RankingEngine::new(config);
        assert!(engine.is_ok());
    }

    #[tokio::test]
    async fn test_rank_results() {
        let config = ScoringConfig::default();
        let engine = RankingEngine::new(config).unwrap();

        let results = vec![
            RankedResult {
                id: "1".to_string(),
                score: 0.0,
                features: HashMap::from([
                    ("semantic_similarity".to_string(), 0.9),
                    ("popularity".to_string(), 0.7),
                ]),
            },
            RankedResult {
                id: "2".to_string(),
                score: 0.0,
                features: HashMap::from([
                    ("semantic_similarity".to_string(), 0.6),
                    ("popularity".to_string(), 0.9),
                ]),
            },
        ];

        let ranked = engine.rank_results(results).await;
        assert!(ranked.is_ok());
        let ranked = ranked.unwrap();
        assert_eq!(ranked.len(), 2);
        assert!(ranked[0].score > 0.0);
    }

    #[test]
    fn test_feature_similarity() {
        let features1 = HashMap::from([
            ("a".to_string(), 1.0),
            ("b".to_string(), 0.5),
        ]);
        let features2 = HashMap::from([
            ("a".to_string(), 1.0),
            ("b".to_string(), 0.5),
        ]);
        let similarity = calculate_feature_similarity(&features1, &features2);
        assert!((similarity - 1.0).abs() < 0.001);
    }
}
