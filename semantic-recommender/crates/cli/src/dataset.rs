// Dataset loading utilities

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::path::Path;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Movie {
    pub id: String,
    pub title: String,
    pub description: Option<String>,
    pub genres: Vec<String>,
    pub year: Option<i32>,
    pub rating: Option<f32>,
}

pub struct MovieDataset {
    pub movies: Vec<Movie>,
    pub embeddings: Option<Vec<Vec<f32>>>,
}

impl MovieDataset {
    pub async fn load_csv(path: &Path) -> Result<Self> {
        let mut reader = csv::ReaderBuilder::new()
            .has_headers(true)
            .from_path(path)
            .context("Failed to open CSV file")?;

        let mut movies = Vec::new();

        for result in reader.deserialize() {
            let movie: Movie = result.context("Failed to parse movie record")?;
            movies.push(movie);
        }

        Ok(Self {
            movies,
            embeddings: None,
        })
    }

    pub fn len(&self) -> usize {
        self.movies.len()
    }

    pub fn is_empty(&self) -> bool {
        self.movies.is_empty()
    }

    pub async fn compute_embeddings(&mut self) -> Result<()> {
        // TODO: Actual embedding computation
        // For now, generate random embeddings
        let embedding_dim = 768;
        let mut embeddings = Vec::with_capacity(self.movies.len());

        for _ in 0..self.movies.len() {
            let embedding: Vec<f32> = (0..embedding_dim)
                .map(|_| rand::random::<f32>())
                .collect();
            embeddings.push(embedding);
        }

        self.embeddings = Some(embeddings);
        Ok(())
    }

    pub async fn load_embeddings(&mut self, path: &Path) -> Result<()> {
        // TODO: Load from file
        self.compute_embeddings().await
    }
}
