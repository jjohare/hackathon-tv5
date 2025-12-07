//! Error types for attention reranking

use thiserror::Error;

/// Errors that can occur during attention operations
#[derive(Debug, Error)]
pub enum AttentionError {
    /// Tensor shape mismatch
    #[error("Tensor shape mismatch: expected {expected:?}, got {actual:?}")]
    ShapeMismatch {
        expected: Vec<i64>,
        actual: Vec<i64>,
    },

    /// Invalid embedding dimension
    #[error("Invalid embedding dimension: {0}")]
    InvalidEmbedDim(i64),

    /// Device error
    #[error("Device error: {0}")]
    DeviceError(String),

    /// Model I/O error
    #[error("Model I/O error: {0}")]
    IoError(#[from] std::io::Error),

    /// Serialization error
    #[error("Serialization error: {0}")]
    SerializationError(#[from] serde_json::Error),

    /// Invalid context features
    #[error("Invalid context features: {0}")]
    InvalidContext(String),

    /// Tensor conversion error
    #[error("Tensor conversion error: {0}")]
    TensorError(String),

    /// Generic attention error
    #[error("Attention error: {0}")]
    Other(String),
}

impl From<tch::TchError> for AttentionError {
    fn from(err: tch::TchError) -> Self {
        AttentionError::TensorError(err.to_string())
    }
}

/// Result type for attention operations
pub type AttentionResult<T> = Result<T, AttentionError>;
