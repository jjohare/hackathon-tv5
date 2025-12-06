/// Integration module for connecting API layer to GPU-accelerated recommendation engine
///
/// This module provides the bridge between the Axum HTTP API and the underlying
/// GPU-powered semantic search and recommendation system.

pub mod app_state;
pub mod embedding_service;
pub mod health;
pub mod metrics;
pub mod stub_gpu;

#[cfg(test)]
mod tests;

pub use app_state::AppState;
pub use embedding_service::EmbeddingService;
pub use health::{HealthResponse, ComponentHealth, GpuInfo};
pub use metrics::Metrics;
