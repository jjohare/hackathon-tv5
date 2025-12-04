use serde::{Deserialize, Serialize};
use utoipa::ToSchema;

#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
pub struct MediaSearchRequest {
    /// Natural language search query
    #[schema(example = "French noir films with existential themes")]
    pub query: String,

    /// Optional filters for search refinement
    pub filters: Option<SearchFilters>,

    /// Maximum number of results to return (default: 10)
    #[schema(minimum = 1, maximum = 100)]
    pub limit: Option<usize>,

    /// Offset for pagination (default: 0)
    #[schema(minimum = 0)]
    pub offset: Option<usize>,
}

#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
pub struct SearchFilters {
    /// Filter by genres
    #[schema(example = json!(["drama", "thriller"]))]
    pub genres: Option<Vec<String>>,

    /// Minimum rating (0.0-10.0)
    #[schema(minimum = 0.0, maximum = 10.0)]
    pub min_rating: Option<f32>,

    /// Filter by language code (ISO 639-1)
    #[schema(example = "fr")]
    pub language: Option<String>,

    /// Filter by year range
    pub year_range: Option<(i32, i32)>,

    /// Filter by content type
    #[schema(example = "movie")]
    pub content_type: Option<ContentType>,
}

#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
#[serde(rename_all = "lowercase")]
pub enum ContentType {
    Movie,
    Series,
    Documentary,
    Short,
}

#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
pub struct MediaSearchResponse {
    /// List of search results
    pub results: Vec<MediaItem>,

    /// Total number of matching items
    pub total: usize,

    /// Query execution time in milliseconds
    pub query_time_ms: u64,

    /// Pagination information
    pub pagination: PaginationInfo,
}

#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
pub struct MediaItem {
    /// Unique media identifier
    #[schema(example = "film_123")]
    pub id: String,

    /// Media title
    #[schema(example = "À bout de souffle")]
    pub title: String,

    /// Semantic similarity score (0.0-1.0)
    #[schema(minimum = 0.0, maximum = 1.0)]
    pub similarity_score: f32,

    /// Human-readable explanation of why this was recommended
    #[schema(example = "Classic French New Wave noir with existential undertones")]
    pub explanation: String,

    /// Additional metadata
    pub metadata: MediaMetadata,
}

#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
pub struct MediaMetadata {
    pub genres: Vec<String>,
    pub year: i32,
    pub language: String,
    pub rating: Option<f32>,
    pub duration_minutes: Option<i32>,
    pub director: Option<String>,
    pub cast: Option<Vec<String>>,
}

#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
pub struct PaginationInfo {
    pub current_page: usize,
    pub total_pages: usize,
    pub items_per_page: usize,
    pub total_items: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
pub struct RecommendationRequest {
    /// User identifier
    pub user_id: String,

    /// Number of recommendations to return
    #[schema(minimum = 1, maximum = 50)]
    pub limit: Option<usize>,

    /// Include explanation for each recommendation
    pub include_explanation: Option<bool>,
}

#[derive(Debug, Clone, Deserialize, ToSchema)]
pub struct RecommendationParams {
    /// Number of recommendations (default: 10)
    #[schema(minimum = 1, maximum = 50)]
    pub limit: Option<usize>,

    /// Include detailed explanations
    pub explain: Option<bool>,

    /// Filter by content type
    pub content_type: Option<ContentType>,
}

#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
pub struct RecommendationResponse {
    /// User identifier
    pub user_id: String,

    /// List of personalized recommendations
    pub recommendations: Vec<Recommendation>,

    /// Timestamp when recommendations were generated (ISO 8601)
    pub generated_at: String,

    /// Model version used for recommendations
    pub model_version: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
pub struct Recommendation {
    /// Media item
    pub item: MediaItem,

    /// Recommendation score (0.0-1.0)
    #[schema(minimum = 0.0, maximum = 1.0)]
    pub score: f32,

    /// Why this was recommended
    pub reasoning: Option<String>,

    /// Related items that influenced this recommendation
    pub influenced_by: Option<Vec<String>>,
}

#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
pub struct HealthResponse {
    pub status: String,
    pub version: String,
    pub timestamp: chrono::DateTime<chrono::Utc>,
}
