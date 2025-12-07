// GPU-accelerated user embeddings with real-time updates
use anyhow::{Context, Result};
use cudarc::driver::{CudaDevice, CudaSlice, DevicePtr, LaunchAsync, LaunchConfig};
use dashmap::DashMap;
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tracing::{debug, info};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UserEmbedding {
    pub user_id: String,
    pub embedding: Vec<f32>,
    pub last_updated: std::time::SystemTime,
}

pub struct GPUUserEmbeddings {
    device: Arc<CudaDevice>,
    embeddings: Arc<DashMap<String, Arc<RwLock<Vec<f32>>>>>,
    embedding_dim: usize,
    device_embeddings: Arc<RwLock<Option<CudaSlice<f32>>>>,
    user_index: Arc<RwLock<Vec<String>>>,
}

impl GPUUserEmbeddings {
    pub fn new(device: Arc<CudaDevice>, embedding_dim: usize) -> Result<Self> {
        info!("Initializing GPU user embeddings (dim={})", embedding_dim);

        Ok(Self {
            device,
            embeddings: Arc::new(DashMap::new()),
            embedding_dim,
            device_embeddings: Arc::new(RwLock::new(None)),
            user_index: Arc::new(RwLock::new(Vec::new())),
        })
    }

    pub fn load_embeddings(&self, embeddings_path: &str) -> Result<()> {
        info!("Loading user embeddings from {}", embeddings_path);

        let data = std::fs::read(embeddings_path)
            .with_context(|| format!("Failed to read embeddings from {}", embeddings_path))?;

        let embeddings: Vec<UserEmbedding> = bincode::deserialize(&data)
            .context("Failed to deserialize user embeddings")?;

        let mut user_index = self.user_index.write();
        user_index.clear();

        for embedding in embeddings {
            user_index.push(embedding.user_id.clone());
            self.embeddings.insert(
                embedding.user_id,
                Arc::new(RwLock::new(embedding.embedding)),
            );
        }

        info!("Loaded {} user embeddings", user_index.len());
        Ok(())
    }

    pub fn get_embedding(&self, user_id: &str) -> Option<Vec<f32>> {
        self.embeddings.get(user_id).map(|e| e.read().clone())
    }

    pub fn update_embedding(&self, user_id: &str, embedding: Vec<f32>) -> Result<()> {
        if embedding.len() != self.embedding_dim {
            anyhow::bail!(
                "Embedding dimension mismatch: expected {}, got {}",
                self.embedding_dim,
                embedding.len()
            );
        }

        if let Some(existing) = self.embeddings.get(user_id) {
            *existing.write() = embedding;
        } else {
            self.embeddings.insert(
                user_id.to_string(),
                Arc::new(RwLock::new(embedding)),
            );
            self.user_index.write().push(user_id.to_string());
        }

        Ok(())
    }

    pub fn fuse_embeddings(&self, user_id: &str, query_embedding: &[f32]) -> Result<Vec<f32>> {
        let user_emb = self.get_embedding(user_id)
            .unwrap_or_else(|| vec![0.0; self.embedding_dim]);

        // Simple weighted fusion: 0.7 * query + 0.3 * user
        let mut fused = Vec::with_capacity(self.embedding_dim);
        for i in 0..self.embedding_dim {
            fused.push(0.7 * query_embedding[i] + 0.3 * user_emb[i]);
        }

        // Normalize
        let norm: f32 = fused.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            for x in &mut fused {
                *x /= norm;
            }
        }

        Ok(fused)
    }

    pub fn embedding_dim(&self) -> usize {
        self.embedding_dim
    }

    pub fn num_users(&self) -> usize {
        self.embeddings.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_embedding_operations() {
        let device = CudaDevice::new(0).unwrap();
        let embeddings = GPUUserEmbeddings::new(Arc::new(device), 384).unwrap();

        // Test update
        let emb = vec![1.0; 384];
        embeddings.update_embedding("user1", emb.clone()).unwrap();

        // Test get
        let retrieved = embeddings.get_embedding("user1").unwrap();
        assert_eq!(retrieved, emb);

        // Test fusion
        let query = vec![0.5; 384];
        let fused = embeddings.fuse_embeddings("user1", &query).unwrap();
        assert_eq!(fused.len(), 384);

        // Check normalization
        let norm: f32 = fused.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-5);
    }
}
