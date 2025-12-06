/// Recommendation results and scoring
///
/// Represents recommendation outputs with:
/// - Scored content items
/// - Semantic paths (explanation trails)
/// - Confidence metrics
/// - Ranking factors

use serde::{Deserialize, Serialize};
use crate::models::{MediaContent, ContentId};

/// Recommendation result with score and explanation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Recommendation {
    /// Recommended content
    pub content: MediaContent,

    /// Overall score (0.0-1.0)
    pub score: RecommendationScore,

    /// Human-readable explanation
    pub explanation: String,

    /// Semantic path from user to content
    pub semantic_path: Option<SemanticPath>,

    /// Ranking factors breakdown
    pub ranking_factors: RankingFactors,

    /// Position in result list (1-indexed)
    pub rank: usize,

    /// Timestamp when generated
    pub generated_at: chrono::DateTime<chrono::Utc>,
}

impl Recommendation {
    /// Create new recommendation
    pub fn new(content: MediaContent, score: f32, explanation: impl Into<String>) -> Self {
        Self {
            content,
            score: RecommendationScore::new(score),
            explanation: explanation.into(),
            semantic_path: None,
            ranking_factors: RankingFactors::default(),
            rank: 0,
            generated_at: chrono::Utc::now(),
        }
    }

    /// Set semantic path
    pub fn with_path(mut self, path: SemanticPath) -> Self {
        self.semantic_path = Some(path);
        self
    }

    /// Set ranking factors
    pub fn with_factors(mut self, factors: RankingFactors) -> Self {
        self.ranking_factors = factors;
        self
    }

    /// Set rank position
    pub fn with_rank(mut self, rank: usize) -> Self {
        self.rank = rank;
        self
    }

    /// Check if high-confidence recommendation
    pub fn is_high_confidence(&self) -> bool {
        self.score.confidence > 0.8
    }
}

/// Recommendation score with components
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct RecommendationScore {
    /// Overall score (0.0-1.0)
    pub total: f32,

    /// Relevance score (semantic match)
    pub relevance: f32,

    /// Personalization score (user fit)
    pub personalization: f32,

    /// Quality score (content quality)
    pub quality: f32,

    /// Diversity score (serendipity)
    pub diversity: f32,

    /// Confidence (certainty of prediction)
    pub confidence: f32,
}

impl RecommendationScore {
    /// Create new score with total
    pub fn new(total: f32) -> Self {
        Self {
            total: total.clamp(0.0, 1.0),
            relevance: 0.0,
            personalization: 0.0,
            quality: 0.0,
            diversity: 0.0,
            confidence: 1.0,
        }
    }

    /// Create from components with weighted average
    pub fn from_components(
        relevance: f32,
        personalization: f32,
        quality: f32,
        diversity: f32,
    ) -> Self {
        let total = 0.4 * relevance + 0.4 * personalization + 0.15 * quality + 0.05 * diversity;

        Self {
            total: total.clamp(0.0, 1.0),
            relevance,
            personalization,
            quality,
            diversity,
            confidence: 1.0,
        }
    }

    /// Set confidence level
    pub fn with_confidence(mut self, confidence: f32) -> Self {
        self.confidence = confidence.clamp(0.0, 1.0);
        self
    }
}

/// Semantic path from user to recommended content
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SemanticPath {
    /// Path nodes (content IDs or concept URIs)
    pub nodes: Vec<PathNode>,

    /// Path length (number of hops)
    pub length: usize,

    /// Path strength (weakest link)
    pub strength: f32,

    /// Path type
    pub path_type: PathType,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PathNode {
    /// Node identifier
    pub id: String,

    /// Node type
    pub node_type: NodeType,

    /// Label for display
    pub label: String,

    /// Edge weight to next node
    pub edge_weight: Option<f32>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum NodeType {
    /// User profile
    User,
    /// Media content
    Content,
    /// Genre concept
    Genre,
    /// Visual aesthetic
    Aesthetic,
    /// Narrative structure
    Narrative,
    /// Mood/theme
    Mood,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PathType {
    /// Direct content similarity
    DirectSimilarity,
    /// Genre-based path
    GenrePath,
    /// Aesthetic similarity path
    AestheticPath,
    /// Narrative structure path
    NarrativePath,
    /// Collaborative filtering path
    CollaborativePath,
    /// Hybrid multi-hop path
    HybridPath,
}

impl SemanticPath {
    /// Create new semantic path
    pub fn new(nodes: Vec<PathNode>, path_type: PathType) -> Self {
        let length = nodes.len().saturating_sub(1);

        // Compute path strength (minimum edge weight)
        let strength = nodes
            .iter()
            .filter_map(|n| n.edge_weight)
            .min_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap_or(1.0);

        Self {
            nodes,
            length,
            strength,
            path_type,
        }
    }

    /// Generate human-readable explanation
    pub fn explain(&self) -> String {
        if self.nodes.len() < 2 {
            return "Direct recommendation".to_string();
        }

        let mut explanation = String::from("Because ");

        for (i, node) in self.nodes.iter().enumerate() {
            if i > 0 {
                explanation.push_str(" → ");
            }
            explanation.push_str(&node.label);
        }

        explanation
    }

    /// Check if path is strong (high confidence)
    pub fn is_strong(&self) -> bool {
        self.strength > 0.7
    }
}

/// Ranking factors breakdown
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RankingFactors {
    /// Vector similarity score
    pub vector_similarity: f32,

    /// Graph distance score
    pub graph_distance: f32,

    /// Matrix factorization score
    pub mf_score: f32,

    /// Neural network score
    pub neural_score: f32,

    /// LLM reranking score
    pub llm_score: Option<f32>,

    /// RL policy score (AgentDB)
    pub rl_score: f32,

    /// Temporal boost (trending)
    pub temporal_boost: f32,

    /// Cultural relevance boost
    pub cultural_boost: f32,

    /// Explanation reasons
    pub reasons: Vec<ExplanationReason>,
}

impl Default for RankingFactors {
    fn default() -> Self {
        Self {
            vector_similarity: 0.0,
            graph_distance: 0.0,
            mf_score: 0.0,
            neural_score: 0.0,
            llm_score: None,
            rl_score: 0.0,
            temporal_boost: 0.0,
            cultural_boost: 0.0,
            reasons: Vec::new(),
        }
    }
}

impl RankingFactors {
    /// Add explanation reason
    pub fn add_reason(&mut self, reason: ExplanationReason) {
        self.reasons.push(reason);
    }

    /// Get top reasons (sorted by importance)
    pub fn top_reasons(&self, n: usize) -> Vec<&ExplanationReason> {
        let mut sorted = self.reasons.iter().collect::<Vec<_>>();
        sorted.sort_by(|a, b| b.importance.partial_cmp(&a.importance).unwrap());
        sorted.into_iter().take(n).collect()
    }
}

/// Explanation reason for recommendation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExplanationReason {
    /// Reason type
    pub reason_type: ReasonType,

    /// Human-readable explanation
    pub explanation: String,

    /// Importance score (0.0-1.0)
    pub importance: f32,

    /// Supporting evidence (optional)
    pub evidence: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ReasonType {
    /// Similar to watched content
    SimilarTo,
    /// Matches user preferences
    MatchesPreferences,
    /// Popular in taste cluster
    PopularInCluster,
    /// Trending globally
    Trending,
    /// Culturally relevant
    CulturallyRelevant,
    /// Fills diversity gap
    DiversityPick,
    /// High critical acclaim
    CriticalAcclaim,
    /// Award-winning
    AwardWinning,
    /// Matches current mood
    MatchesMood,
    /// Recommended by similar users
    CollaborativeFiltering,
}

impl ExplanationReason {
    /// Create new reason
    pub fn new(
        reason_type: ReasonType,
        explanation: impl Into<String>,
        importance: f32,
    ) -> Self {
        Self {
            reason_type,
            explanation: explanation.into(),
            importance: importance.clamp(0.0, 1.0),
            evidence: None,
        }
    }

    /// Add supporting evidence
    pub fn with_evidence(mut self, evidence: impl Into<String>) -> Self {
        self.evidence = Some(evidence.into());
        self
    }
}

/// Batch recommendation result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecommendationBatch {
    /// User ID
    pub user_id: String,

    /// Recommendations (sorted by score)
    pub recommendations: Vec<Recommendation>,

    /// Total candidates considered
    pub candidates_count: usize,

    /// Processing time (milliseconds)
    pub processing_time_ms: f32,

    /// Cache hit
    pub cache_hit: bool,

    /// Metadata
    pub metadata: BatchMetadata,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchMetadata {
    /// Request timestamp
    pub requested_at: chrono::DateTime<chrono::Utc>,

    /// Query parameters
    pub query_params: std::collections::HashMap<String, String>,

    /// Algorithm version
    pub algorithm_version: String,

    /// Model versions used
    pub model_versions: Vec<String>,
}

impl RecommendationBatch {
    /// Create new batch
    pub fn new(user_id: impl Into<String>, recommendations: Vec<Recommendation>) -> Self {
        Self {
            user_id: user_id.into(),
            recommendations,
            candidates_count: 0,
            processing_time_ms: 0.0,
            cache_hit: false,
            metadata: BatchMetadata {
                requested_at: chrono::Utc::now(),
                query_params: std::collections::HashMap::new(),
                algorithm_version: "v1.0.0".to_string(),
                model_versions: Vec::new(),
            },
        }
    }

    /// Get top N recommendations
    pub fn top_n(&self, n: usize) -> &[Recommendation] {
        &self.recommendations[..n.min(self.recommendations.len())]
    }

    /// Filter by minimum score
    pub fn filter_by_score(&self, min_score: f32) -> Vec<&Recommendation> {
        self.recommendations
            .iter()
            .filter(|r| r.score.total >= min_score)
            .collect()
    }

    /// Get average score
    pub fn avg_score(&self) -> f32 {
        if self.recommendations.is_empty() {
            return 0.0;
        }

        let sum: f32 = self.recommendations.iter().map(|r| r.score.total).sum();
        sum / self.recommendations.len() as f32
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::{ContentId, ContentType};

    #[test]
    fn test_recommendation_score() {
        let score = RecommendationScore::from_components(0.9, 0.8, 0.7, 0.5);
        assert!(score.total > 0.0 && score.total <= 1.0);
    }

    #[test]
    fn test_semantic_path() {
        let nodes = vec![
            PathNode {
                id: "user1".to_string(),
                node_type: NodeType::User,
                label: "User".to_string(),
                edge_weight: Some(0.9),
            },
            PathNode {
                id: "genre:scifi".to_string(),
                node_type: NodeType::Genre,
                label: "Sci-Fi".to_string(),
                edge_weight: Some(0.8),
            },
            PathNode {
                id: "film123".to_string(),
                node_type: NodeType::Content,
                label: "Inception".to_string(),
                edge_weight: None,
            },
        ];

        let path = SemanticPath::new(nodes, PathType::GenrePath);
        assert_eq!(path.length, 2);
        assert_eq!(path.strength, 0.8); // Minimum edge weight
    }

    #[test]
    fn test_recommendation_batch() {
        let content = MediaContent::new(
            ContentId::new("film123"),
            ContentType::Film,
            "Test Film".to_string(),
        );

        let rec = Recommendation::new(content, 0.85, "Test recommendation");
        let batch = RecommendationBatch::new("user1", vec![rec]);

        assert_eq!(batch.recommendations.len(), 1);
        assert_eq!(batch.top_n(1).len(), 1);
    }
}
