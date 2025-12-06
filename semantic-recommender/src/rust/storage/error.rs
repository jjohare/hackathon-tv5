/// Storage subsystem errors
use thiserror::Error;

#[derive(Error, Debug)]
pub enum StorageError {
    #[error("Milvus error: {0}")]
    Milvus(String),

    #[error("Neo4j error: {0}")]
    Neo4j(String),

    #[error("PostgreSQL error: {0}")]
    Postgres(String),

    #[error("Redis cache error: {0}")]
    Redis(String),

    #[error("Query planning error: {0}")]
    QueryPlanning(String),

    #[error("AgentDB error: {0}")]
    AgentDB(String),

    #[error("Serialization error: {0}")]
    Serialization(#[from] serde_json::Error),

    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    #[error("General error: {0}")]
    Other(String),
}

/// Convenience result type for storage operations
pub type StorageResult<T> = Result<T, StorageError>;
