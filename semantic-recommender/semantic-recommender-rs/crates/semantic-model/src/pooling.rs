//! Pooling strategies for sentence embeddings

use serde::{Deserialize, Serialize};

/// Pooling strategy for combining token embeddings into sentence embeddings
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PoolingStrategy {
    /// Mean pooling: average all token embeddings (weighted by attention mask)
    Mean,

    /// Max pooling: take element-wise maximum across all token embeddings
    Max,

    /// CLS pooling: use only the [CLS] token embedding
    Cls,
}

impl Default for PoolingStrategy {
    fn default() -> Self {
        Self::Mean
    }
}

impl std::fmt::Display for PoolingStrategy {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Mean => write!(f, "mean"),
            Self::Max => write!(f, "max"),
            Self::Cls => write!(f, "cls"),
        }
    }
}

impl std::str::FromStr for PoolingStrategy {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "mean" => Ok(Self::Mean),
            "max" => Ok(Self::Max),
            "cls" => Ok(Self::Cls),
            _ => Err(format!("Unknown pooling strategy: {}", s)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pooling_strategy_display() {
        assert_eq!(PoolingStrategy::Mean.to_string(), "mean");
        assert_eq!(PoolingStrategy::Max.to_string(), "max");
        assert_eq!(PoolingStrategy::Cls.to_string(), "cls");
    }

    #[test]
    fn test_pooling_strategy_from_str() {
        use std::str::FromStr;

        assert_eq!(PoolingStrategy::from_str("mean").unwrap(), PoolingStrategy::Mean);
        assert_eq!(PoolingStrategy::from_str("Mean").unwrap(), PoolingStrategy::Mean);
        assert_eq!(PoolingStrategy::from_str("max").unwrap(), PoolingStrategy::Max);
        assert_eq!(PoolingStrategy::from_str("cls").unwrap(), PoolingStrategy::Cls);
        assert!(PoolingStrategy::from_str("invalid").is_err());
    }

    #[test]
    fn test_default() {
        assert_eq!(PoolingStrategy::default(), PoolingStrategy::Mean);
    }
}
