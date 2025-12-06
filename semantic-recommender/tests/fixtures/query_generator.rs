use rand::Rng;
use std::collections::HashMap;

pub struct SearchQuery {
    pub embedding: Vec<f32>,
    pub user_id: Option<String>,
    pub k: usize,
    pub filters: HashMap<String, String>,
}

const GENRES: &[&str] = &["SciFi", "Action", "Romance", "Comedy", "Drama"];

pub fn create_random_query() -> SearchQuery {
    let mut rng = rand::thread_rng();

    // Generate random normalized embedding
    let embedding: Vec<f32> = (0..768)
        .map(|_| rng.gen_range(-1.0..1.0))
        .collect();

    let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
    let normalized_embedding: Vec<f32> = embedding.iter().map(|x| x / norm).collect();

    // Random user ID (50% chance)
    let user_id = if rng.gen_bool(0.5) {
        Some(format!("user_{}", rng.gen_range(0..1000)))
    } else {
        None
    };

    // Random filters (30% chance)
    let mut filters = HashMap::new();
    if rng.gen_bool(0.3) {
        let genre = GENRES[rng.gen_range(0..GENRES.len())];
        filters.insert("genre".to_string(), genre.to_string());
    }

    SearchQuery {
        embedding: normalized_embedding,
        user_id,
        k: rng.gen_range(10..50),
        filters,
    }
}

pub fn create_query_without_filter() -> SearchQuery {
    let mut query = create_random_query();
    query.filters.clear();
    query
}

pub fn create_query_without_user() -> SearchQuery {
    let mut query = create_random_query();
    query.user_id = None;
    query
}

pub fn create_query_with_embedding(embedding: Vec<f32>) -> SearchQuery {
    SearchQuery {
        embedding,
        user_id: None,
        k: 20,
        filters: HashMap::new(),
    }
}
