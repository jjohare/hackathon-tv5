use thiserror::Error;

#[derive(Debug, Error)]
pub enum HyperPersonalizationError {
    #[error("Model error: {0}")]
    Model(String),

    #[error("GPU error: {0}")]
    Gpu(String),

    #[error("User not found: {0}")]
    UserNotFound(String),

    #[error("Item not found: {0}")]
    ItemNotFound(usize),

    #[error("Configuration error: {0}")]
    Config(String),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Other error: {0}")]
    Other(String),
}
