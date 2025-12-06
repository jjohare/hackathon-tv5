use crate::error::ApiError;
use crate::models::*;
use crate::integration::{AppState, EmbeddingService};
use dashmap::DashMap;
use std::sync::Arc;

/// High-performance recommendation engine
pub struct RecommendationEngine {
    /// In-memory cache for fast lookups
    cache: Arc<DashMap<String, Vec<MediaItem>>>,

    /// Mock data for demonstration
    mock_data: Vec<MediaItem>,

    /// Optional GPU-accelerated state (None = CPU fallback mode)
    gpu_state: Option<Arc<AppState>>,

    /// Embedding service for text-to-vector conversion
    embedding_service: Option<Arc<EmbeddingService>>,
}

impl RecommendationEngine {
    /// Create new recommendation engine without GPU acceleration
    pub async fn new() -> Self {
        Self {
            cache: Arc::new(DashMap::new()),
            mock_data: Self::generate_mock_data(),
            gpu_state: None,
            embedding_service: None,
        }
    }

    /// Create recommendation engine with GPU acceleration
    pub async fn with_gpu(state: Arc<AppState>) -> Self {
        // Clone embedding service from app state
        let embedding_service = Some(state.embedding_service.clone());

        Self {
            cache: Arc::new(DashMap::new()),
            mock_data: Self::generate_mock_data(),
            gpu_state: Some(state),
            embedding_service,
        }
    }

    /// Check if GPU acceleration is available
    pub async fn has_gpu(&self) -> bool {
        if let Some(state) = &self.gpu_state {
            state.gpu_available().await
        } else {
            false
        }
    }

    /// Generate embedding from text query
    /// This demonstrates the text-to-vector conversion bridge
    fn generate_embedding(&self, text: &str) -> Vec<f32> {
        if let Some(emb_service) = &self.embedding_service {
            tracing::debug!("Generating embedding for query: {}", text);
            let embedding = emb_service.embed_text(text);
            tracing::debug!("Generated {}-dimensional embedding", embedding.len());
            embedding
        } else {
            // Fallback: empty embedding
            tracing::warn!("No embedding service available, using empty embedding");
            vec![]
        }
    }

    /// Search for media content
    pub async fn search(&self, req: &MediaSearchRequest) -> Result<MediaSearchResponse, ApiError> {
        let start = std::time::Instant::now();

        // BRIDGE DEMONSTRATION: Convert text query to embedding
        let query_embedding = self.generate_embedding(&req.query);
        if !query_embedding.is_empty() {
            tracing::info!(
                "🔗 BRIDGE: Converted '{}' to {}-dim vector for semantic search",
                req.query,
                query_embedding.len()
            );
            // In production, this embedding would be sent to:
            // - Milvus for vector similarity search
            // - GPU kernels for accelerated computation
            // - HybridStorageCoordinator for multi-backend queries
        }

        // Check cache first
        if let Some(cached) = self.cache.get(&req.query) {
            let results = cached.clone();
            let query_time_ms = start.elapsed().as_millis() as u64;

            return Ok(MediaSearchResponse {
                results: results.into_iter().take(req.limit.unwrap_or(10)).collect(),
                total: cached.len(),
                query_time_ms,
                pagination: PaginationInfo {
                    current_page: 1,
                    total_pages: (cached.len() as f64 / req.limit.unwrap_or(10) as f64).ceil() as usize,
                    items_per_page: req.limit.unwrap_or(10),
                    total_items: cached.len(),
                },
            });
        }

        // Simulate semantic search (in production, this would use GPU-accelerated vector search)
        let mut results = self.mock_data.clone();

        // Apply filters
        if let Some(filters) = &req.filters {
            results = Self::apply_filters(results, filters);
        }

        // Sort by similarity score
        results.sort_by(|a, b| b.similarity_score.partial_cmp(&a.similarity_score).unwrap());

        // Cache results
        self.cache.insert(req.query.clone(), results.clone());

        let total = results.len();
        let limit = req.limit.unwrap_or(10);
        let offset = req.offset.unwrap_or(0);

        let paginated_results: Vec<_> = results.into_iter().skip(offset).take(limit).collect();

        let query_time_ms = start.elapsed().as_millis() as u64;

        Ok(MediaSearchResponse {
            results: paginated_results,
            total,
            query_time_ms,
            pagination: PaginationInfo {
                current_page: (offset / limit) + 1,
                total_pages: (total as f64 / limit as f64).ceil() as usize,
                items_per_page: limit,
                total_items: total,
            },
        })
    }

    /// Get personalized recommendations
    pub async fn recommend(
        &self,
        user_id: &str,
        params: &RecommendationParams,
    ) -> Result<RecommendationResponse, ApiError> {
        // Simulate personalized recommendations
        let mut recommendations: Vec<_> = self.mock_data
            .iter()
            .take(params.limit.unwrap_or(10))
            .map(|item| Recommendation {
                item: item.clone(),
                score: 0.85 + (rand::random::<f32>() * 0.15),
                reasoning: if params.explain.unwrap_or(false) {
                    Some(format!("Recommended based on your interest in {}", item.metadata.genres.join(", ")))
                } else {
                    None
                },
                influenced_by: Some(vec!["film_123".to_string(), "film_456".to_string()]),
            })
            .collect();

        recommendations.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());

        Ok(RecommendationResponse {
            user_id: user_id.to_string(),
            recommendations,
            generated_at: chrono::Utc::now().to_rfc3339(),
            model_version: "v2.3.0".to_string(),
        })
    }

    fn apply_filters(mut items: Vec<MediaItem>, filters: &SearchFilters) -> Vec<MediaItem> {
        if let Some(genres) = &filters.genres {
            items.retain(|item| {
                item.metadata.genres.iter().any(|g| genres.contains(g))
            });
        }

        if let Some(min_rating) = filters.min_rating {
            items.retain(|item| {
                item.metadata.rating.map_or(false, |r| r >= min_rating)
            });
        }

        if let Some(language) = &filters.language {
            items.retain(|item| &item.metadata.language == language);
        }

        if let Some((start, end)) = filters.year_range {
            items.retain(|item| item.metadata.year >= start && item.metadata.year <= end);
        }

        items
    }

    fn generate_mock_data() -> Vec<MediaItem> {
        vec![
            MediaItem {
                id: "film_123".to_string(),
                title: "À bout de souffle".to_string(),
                similarity_score: 0.92,
                explanation: "Classic French New Wave noir with existential undertones and philosophical dialogue".to_string(),
                metadata: MediaMetadata {
                    genres: vec!["drama".to_string(), "crime".to_string(), "noir".to_string()],
                    year: 1960,
                    language: "fr".to_string(),
                    rating: Some(8.2),
                    duration_minutes: Some(90),
                    director: Some("Jean-Luc Godard".to_string()),
                    cast: Some(vec!["Jean-Paul Belmondo".to_string(), "Jean Seberg".to_string()]),
                },
            },
            MediaItem {
                id: "film_456".to_string(),
                title: "Le Samouraï".to_string(),
                similarity_score: 0.88,
                explanation: "Stylized French neo-noir about a contract killer, featuring minimalist storytelling".to_string(),
                metadata: MediaMetadata {
                    genres: vec!["thriller".to_string(), "crime".to_string(), "noir".to_string()],
                    year: 1967,
                    language: "fr".to_string(),
                    rating: Some(8.0),
                    duration_minutes: Some(105),
                    director: Some("Jean-Pierre Melville".to_string()),
                    cast: Some(vec!["Alain Delon".to_string()]),
                },
            },
            MediaItem {
                id: "doc_789".to_string(),
                title: "Our Planet".to_string(),
                similarity_score: 0.85,
                explanation: "Comprehensive documentary series exploring climate impact on global ecosystems".to_string(),
                metadata: MediaMetadata {
                    genres: vec!["documentary".to_string(), "nature".to_string()],
                    year: 2019,
                    language: "en".to_string(),
                    rating: Some(9.3),
                    duration_minutes: Some(480),
                    director: None,
                    cast: Some(vec!["David Attenborough".to_string()]),
                },
            },
        ]
    }
}
