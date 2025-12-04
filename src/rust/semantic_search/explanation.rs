// Explanation Generation
// Generate human-readable explanations for recommendations using semantic paths

use std::collections::HashMap;
use std::sync::Arc;
use anyhow::Result;
use serde::{Deserialize, Serialize};
use tokio::sync::RwLock;

use super::path_discovery::{SemanticPath, PathType};

pub type UserId = String;
pub type ContentId = String;

/// An explanation for a recommendation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Explanation {
    pub recommendation_id: String,
    pub primary_reason: String,
    pub supporting_reasons: Vec<String>,
    pub confidence: f32,
    pub paths: Vec<ExplanationPath>,
    pub visualization_data: Option<VisualizationData>,
}

/// A path used in an explanation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExplanationPath {
    pub path_type: PathType,
    pub description: String,
    pub steps: Vec<String>,
    pub relevance: f32,
}

/// Data for visualizing explanations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VisualizationData {
    pub node_positions: HashMap<String, (f32, f32)>,
    pub edge_weights: Vec<(String, String, f32)>,
    pub highlight_nodes: Vec<String>,
}

/// Explanation templates
#[derive(Debug, Clone)]
enum ExplanationTemplate {
    DirectSimilarity,
    ConceptualConnection,
    UserPreference,
    PopularChoice,
    SemanticPath,
    Contextual,
}

impl ExplanationTemplate {
    fn render(&self, context: &ExplanationContext) -> String {
        match self {
            ExplanationTemplate::DirectSimilarity => {
                format!(
                    "This content is {} similar to '{}' which you recently enjoyed.",
                    similarity_level(context.similarity_score),
                    context.reference_content.as_ref().unwrap_or(&"previous content".to_string())
                )
            }
            ExplanationTemplate::ConceptualConnection => {
                format!(
                    "This content relates to {} concepts you're interested in, including {}.",
                    context.shared_concepts.len(),
                    context.shared_concepts.get(0).unwrap_or(&"various topics".to_string())
                )
            }
            ExplanationTemplate::UserPreference => {
                format!(
                    "Based on your preferences for {}, this content matches your interests.",
                    context.user_preferences.join(", ")
                )
            }
            ExplanationTemplate::PopularChoice => {
                "This content is trending among users with similar interests.".to_string()
            }
            ExplanationTemplate::SemanticPath => {
                format!(
                    "This content connects to your interests through {}.",
                    context.path_description.as_ref().unwrap_or(&"related topics".to_string())
                )
            }
            ExplanationTemplate::Contextual => {
                format!(
                    "Recommended based on your current context: {}.",
                    context.context_factors.join(", ")
                )
            }
        }
    }
}

/// Context for generating explanations
#[derive(Debug, Clone, Default)]
struct ExplanationContext {
    similarity_score: f32,
    reference_content: Option<String>,
    shared_concepts: Vec<String>,
    user_preferences: Vec<String>,
    path_description: Option<String>,
    context_factors: Vec<String>,
}

/// Main explanation generator
pub struct ExplanationGenerator {
    user_history: Arc<RwLock<HashMap<UserId, Vec<ContentId>>>>,
    content_metadata: Arc<RwLock<HashMap<ContentId, ContentMetadata>>>,
}

#[derive(Debug, Clone)]
struct ContentMetadata {
    title: String,
    concepts: Vec<String>,
    popularity: f32,
}

impl ExplanationGenerator {
    pub fn new() -> Result<Self> {
        Ok(Self {
            user_history: Arc::new(RwLock::new(HashMap::new())),
            content_metadata: Arc::new(RwLock::new(HashMap::new())),
        })
    }

    /// Generate explanation for a recommendation
    pub async fn generate_explanation(
        &self,
        user_id: UserId,
        content_id: ContentId,
    ) -> Result<Explanation> {
        let history = self.user_history.read().await;
        let metadata = self.content_metadata.read().await;

        let user_interactions = history.get(&user_id).cloned().unwrap_or_default();
        let content_meta = metadata.get(&content_id);

        // Determine primary explanation template
        let (template, context) = self.select_explanation_template(
            &user_id,
            &content_id,
            &user_interactions,
            content_meta,
        ).await?;

        let primary_reason = template.render(&context);
        let supporting_reasons = self.generate_supporting_reasons(&context).await?;

        // Generate explanation paths
        let paths = self.generate_explanation_paths(&context).await?;

        // Calculate confidence based on evidence
        let confidence = self.calculate_confidence(&context, &paths).await?;

        Ok(Explanation {
            recommendation_id: format!("rec-{}-{}", user_id, content_id),
            primary_reason,
            supporting_reasons,
            confidence,
            paths,
            visualization_data: None, // Could be generated for UI
        })
    }

    /// Generate explanation from semantic paths
    pub async fn explain_from_paths(
        &self,
        paths: Vec<SemanticPath>,
    ) -> Result<Explanation> {
        let mut explanation_paths = Vec::new();

        for path in &paths {
            let steps: Vec<String> = path
                .nodes
                .iter()
                .map(|node| node.content_id.clone())
                .collect();

            explanation_paths.push(ExplanationPath {
                path_type: path.path_type,
                description: path.explanation.clone(),
                steps,
                relevance: path.total_score,
            });
        }

        let primary_reason = if let Some(best_path) = paths.first() {
            format!(
                "Connected through {} path: {}",
                format!("{:?}", best_path.path_type).to_lowercase(),
                best_path.explanation
            )
        } else {
            "Recommendation based on semantic analysis".to_string()
        };

        Ok(Explanation {
            recommendation_id: "path-based".to_string(),
            primary_reason,
            supporting_reasons: vec![],
            confidence: paths.first().map(|p| p.total_score).unwrap_or(0.5),
            paths: explanation_paths,
            visualization_data: None,
        })
    }

    /// Update user history for better explanations
    pub async fn record_interaction(
        &self,
        user_id: UserId,
        content_id: ContentId,
    ) -> Result<()> {
        let mut history = self.user_history.write().await;
        history
            .entry(user_id)
            .or_insert_with(Vec::new)
            .push(content_id);
        Ok(())
    }

    /// Add content metadata
    pub async fn add_content_metadata(
        &self,
        content_id: ContentId,
        title: String,
        concepts: Vec<String>,
        popularity: f32,
    ) -> Result<()> {
        let mut metadata = self.content_metadata.write().await;
        metadata.insert(
            content_id,
            ContentMetadata {
                title,
                concepts,
                popularity,
            },
        );
        Ok(())
    }

    // Private helper methods

    async fn select_explanation_template(
        &self,
        _user_id: &UserId,
        _content_id: &ContentId,
        user_interactions: &[ContentId],
        content_meta: Option<&ContentMetadata>,
    ) -> Result<(ExplanationTemplate, ExplanationContext)> {
        let mut context = ExplanationContext::default();

        // Check if similar to recent interactions
        if !user_interactions.is_empty() {
            context.reference_content = user_interactions.last().cloned();
            context.similarity_score = 0.8;
            return Ok((ExplanationTemplate::DirectSimilarity, context));
        }

        // Check conceptual connections
        if let Some(meta) = content_meta {
            if !meta.concepts.is_empty() {
                context.shared_concepts = meta.concepts.clone();
                return Ok((ExplanationTemplate::ConceptualConnection, context));
            }

            // Check popularity
            if meta.popularity > 0.7 {
                return Ok((ExplanationTemplate::PopularChoice, context));
            }
        }

        // Default to user preference
        Ok((ExplanationTemplate::UserPreference, context))
    }

    async fn generate_supporting_reasons(
        &self,
        context: &ExplanationContext,
    ) -> Result<Vec<String>> {
        let mut reasons = Vec::new();

        if context.similarity_score > 0.8 {
            reasons.push("High similarity score".to_string());
        }

        if !context.shared_concepts.is_empty() {
            reasons.push(format!(
                "Shares {} key concepts",
                context.shared_concepts.len()
            ));
        }

        if !context.user_preferences.is_empty() {
            reasons.push("Matches your stated preferences".to_string());
        }

        Ok(reasons)
    }

    async fn generate_explanation_paths(
        &self,
        context: &ExplanationContext,
    ) -> Result<Vec<ExplanationPath>> {
        let mut paths = Vec::new();

        if let Some(ref_content) = &context.reference_content {
            paths.push(ExplanationPath {
                path_type: PathType::Direct,
                description: format!("Similar to '{}'", ref_content),
                steps: vec![ref_content.clone()],
                relevance: context.similarity_score,
            });
        }

        if !context.shared_concepts.is_empty() {
            paths.push(ExplanationPath {
                path_type: PathType::Conceptual,
                description: "Shared concepts".to_string(),
                steps: context.shared_concepts.clone(),
                relevance: 0.75,
            });
        }

        Ok(paths)
    }

    async fn calculate_confidence(
        &self,
        context: &ExplanationContext,
        paths: &[ExplanationPath],
    ) -> Result<f32> {
        let mut confidence = 0.5_f32;

        // Boost confidence based on evidence
        if context.similarity_score > 0.8 {
            confidence += 0.2;
        }

        if !paths.is_empty() {
            confidence += 0.15;
        }

        if !context.shared_concepts.is_empty() {
            confidence += 0.1;
        }

        Ok(confidence.min(1.0_f32))
    }
}

// Helper functions

fn similarity_level(score: f32) -> &'static str {
    if score > 0.9 {
        "very"
    } else if score > 0.7 {
        "highly"
    } else if score > 0.5 {
        "moderately"
    } else {
        "somewhat"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_explanation_generator_creation() {
        let generator = ExplanationGenerator::new();
        assert!(generator.is_ok());
    }

    #[tokio::test]
    async fn test_record_interaction() {
        let generator = ExplanationGenerator::new().unwrap();
        let result = generator
            .record_interaction("user1".to_string(), "content1".to_string())
            .await;
        assert!(result.is_ok());
    }

    #[test]
    fn test_similarity_level() {
        assert_eq!(similarity_level(0.95), "very");
        assert_eq!(similarity_level(0.75), "highly");
        assert_eq!(similarity_level(0.6), "moderately");
        assert_eq!(similarity_level(0.4), "somewhat");
    }

    #[test]
    fn test_template_rendering() {
        let template = ExplanationTemplate::DirectSimilarity;
        let context = ExplanationContext {
            similarity_score: 0.85,
            reference_content: Some("Test Content".to_string()),
            ..Default::default()
        };
        let explanation = template.render(&context);
        assert!(explanation.contains("highly"));
        assert!(explanation.contains("Test Content"));
    }
}
