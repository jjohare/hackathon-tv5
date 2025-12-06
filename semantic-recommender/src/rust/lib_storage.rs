// Add this to lib.rs to include storage module

/// Hybrid storage system (Milvus + Neo4j + PostgreSQL)
#[cfg(feature = "storage")]
pub mod storage;

// Also add to prelude:
#[cfg(feature = "storage")]
pub use crate::storage::{
    HybridStorageCoordinator,
    SearchQuery,
    Recommendation,
    MilvusClient,
    Neo4jClient,
    AgentDBCoordinator,
    QueryPlanner,
};
