pub mod error;
// pub mod graphql;  // Disabled due to dependency conflicts
pub mod hateoas;
pub mod jsonld;
pub mod mcp;
pub mod models;
pub mod recommendation;

// Re-export from parent for convenience
pub use crate::integration;

use axum::{
    extract::{Path, Query, State},
    http::StatusCode,
    response::Json,
    routing::{get, post},
    Router,
};
use std::sync::Arc;
use tower_http::cors::CorsLayer;
use tower_http::trace::TraceLayer;
use utoipa::OpenApi;
use utoipa_swagger_ui::SwaggerUi;

use error::{ApiError, ErrorResponse};
// GraphQL disabled due to version conflicts
// use graphql::create_graphql_handler;
use hateoas::HATEOASResponse;
use jsonld::{add_json_ld_context, JsonLdContext};
use mcp::{get_mcp_manifest, MCPManifest, MCPTool};
use models::*;
use recommendation::RecommendationEngine;

#[derive(OpenApi)]
#[openapi(
    paths(
        search_media,
        get_recommendations,
        health_check,
        metrics_endpoint,
        mcp::get_mcp_manifest
    ),
    components(schemas(
        MediaSearchRequest,
        MediaSearchResponse,
        MediaItem,
        RecommendationRequest,
        RecommendationResponse,
        Recommendation,
        MCPManifest,
        MCPTool,
        ErrorResponse,
        HATEOASResponse<MediaSearchResponse>,
        JsonLdContext,
        integration::health::HealthResponse,
        integration::health::ComponentHealth,
        integration::health::GpuInfo,
        integration::metrics::MetricsSnapshot
    )),
    tags(
        (name = "media", description = "Media discovery and search operations"),
        (name = "recommendations", description = "Personalized content recommendations"),
        (name = "agent", description = "AI agent integration (MCP protocol)")
    ),
    info(
        title = "TV5 Media Gateway API",
        version = "1.0.0",
        description = "GPU-accelerated semantic media recommendation system with AI agent integration",
        contact(
            name = "Media Gateway Team",
            email = "api@tv5monde.com"
        ),
        license(
            name = "MIT",
            url = "https://opensource.org/licenses/MIT"
        )
    ),
    servers(
        (url = "http://localhost:3000", description = "Local development server"),
        (url = "https://api.tv5monde.com", description = "Production API")
    )
)]
pub struct ApiDoc;

pub async fn create_app() -> Router {
    // Try to initialize with GPU acceleration, fall back to CPU if unavailable
    let engine = match integration::AppState::new().await {
        Ok(state) => {
            tracing::info!("GPU acceleration enabled");
            let state = Arc::new(state);
            Arc::new(RecommendationEngine::with_gpu(state).await)
        }
        Err(e) => {
            tracing::warn!("GPU initialization failed, using CPU fallback: {}", e);
            Arc::new(RecommendationEngine::new().await)
        }
    };

    Router::new()
        .route("/api/v1/search", post(search_media))
        .route("/api/v1/recommendations/:user_id", get(get_recommendations))
        .route("/api/v1/mcp/manifest", get(get_mcp_manifest))
        .route("/health", get(health_check))
        .route("/metrics", get(metrics_endpoint))
        // GraphQL endpoint temporarily disabled - use REST API instead
        // .route("/graphql", get(create_graphql_handler).post(create_graphql_handler))
        .merge(SwaggerUi::new("/swagger-ui").url("/api-docs/openapi.json", ApiDoc::openapi()))
        .layer(CorsLayer::permissive())
        .layer(TraceLayer::new_for_http())
        .with_state(engine)
}

#[utoipa::path(
    post,
    path = "/api/v1/search",
    request_body = MediaSearchRequest,
    responses(
        (status = 200, description = "Search results with HATEOAS links", body = HATEOASResponse<MediaSearchResponse>),
        (status = 400, description = "Invalid request", body = ErrorResponse),
        (status = 429, description = "Rate limit exceeded", body = ErrorResponse),
        (status = 500, description = "Internal server error", body = ErrorResponse)
    ),
    tag = "media"
)]
pub async fn search_media(
    State(engine): State<Arc<RecommendationEngine>>,
    Json(req): Json<MediaSearchRequest>,
) -> Result<Json<HATEOASResponse<MediaSearchResponse>>, ApiError> {
    tracing::info!("Search request: query={}, limit={}", req.query, req.limit.unwrap_or(10));

    let results = engine.search(&req).await?;

    let response = HATEOASResponse::new(results, "/api/v1/search")
        .add_action("/api/v1/search/refine", "refine", "POST", "Refine search results")
        .add_action("/api/v1/recommendations", "recommendations", "GET", "Get personalized recommendations");

    Ok(Json(response))
}

#[utoipa::path(
    get,
    path = "/api/v1/recommendations/{user_id}",
    params(
        ("user_id" = String, Path, description = "User identifier")
    ),
    responses(
        (status = 200, description = "Personalized recommendations with JSON-LD context", body = JsonLdContext),
        (status = 404, description = "User not found", body = ErrorResponse),
        (status = 500, description = "Internal server error", body = ErrorResponse)
    ),
    tag = "recommendations"
)]
pub async fn get_recommendations(
    State(engine): State<Arc<RecommendationEngine>>,
    Path(user_id): Path<String>,
    Query(params): Query<RecommendationParams>,
) -> Result<Json<JsonLdContext>, ApiError> {
    tracing::info!("Recommendations request: user_id={}, limit={}", user_id, params.limit.unwrap_or(10));

    let recommendations = engine.recommend(&user_id, &params).await?;

    Ok(Json(add_json_ld_context(
        recommendations,
        "RecommendationList",
    )))
}

#[utoipa::path(
    get,
    path = "/health",
    responses(
        (status = 200, description = "Service is healthy with component details", body = integration::health::HealthResponse)
    ),
    tag = "agent"
)]
pub async fn health_check(
    State(engine): State<Arc<RecommendationEngine>>,
) -> Json<integration::health::HealthResponse> {
    use integration::health::{HealthResponse as IntegrationHealth, ComponentHealth, GpuInfo};

    let (gpu_available, gpu_info) = if let Some(state) = &engine.gpu_state {
        let available = state.gpu_available().await;
        let info = if available {
            state.gpu_info().await.ok().map(GpuInfo::from_device_info)
        } else {
            None
        };
        (available, info)
    } else {
        (false, None)
    };

    let gpu_status = if gpu_available {
        "GPU acceleration enabled".to_string()
    } else {
        "CPU fallback mode".to_string()
    };

    let components = ComponentHealth::new(gpu_available, gpu_status);

    Json(IntegrationHealth {
        status: components.overall_status().to_string(),
        version: env!("CARGO_PKG_VERSION").to_string(),
        timestamp: chrono::Utc::now(),
        components,
        gpu_info,
    })
}

/// Metrics endpoint for performance monitoring
#[utoipa::path(
    get,
    path = "/metrics",
    responses(
        (status = 200, description = "Current performance metrics", body = integration::metrics::MetricsSnapshot)
    ),
    tag = "agent"
)]
pub async fn metrics_endpoint(
    State(engine): State<Arc<RecommendationEngine>>,
) -> Result<Json<integration::metrics::MetricsSnapshot>, ApiError> {
    if let Some(state) = &engine.gpu_state {
        let metrics = state.metrics.snapshot();
        Ok(Json(metrics))
    } else {
        // Return empty metrics if GPU state not available
        Ok(Json(integration::metrics::MetricsSnapshot {
            total_requests: 0,
            successful_requests: 0,
            failed_requests: 0,
            avg_latency_ms: 0.0,
            success_rate: 0.0,
            cache_hits: 0,
            cache_misses: 0,
            cache_hit_rate: 0.0,
        }))
    }
}
