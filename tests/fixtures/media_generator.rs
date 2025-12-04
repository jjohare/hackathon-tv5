use rand::Rng;
use std::collections::HashMap;

pub struct MediaContent {
    pub id: String,
    pub title: String,
    pub embedding: Vec<f32>,
    pub genres: Vec<String>,
    pub metadata: HashMap<String, String>,
}

const GENRES: &[&str] = &["SciFi", "Action", "Romance", "Comedy", "Drama", "Horror", "Thriller", "Documentary"];
const YEARS: &[&str] = &["2018", "2019", "2020", "2021", "2022", "2023", "2024"];

pub fn generate_media_content(count: usize) -> Vec<MediaContent> {
    (0..count)
        .map(|i| generate_single_media(&format!("media_{}", i)))
        .collect()
}

pub fn generate_single_media(id: &str) -> MediaContent {
    let mut rng = rand::thread_rng();

    // Generate realistic embedding (768 dimensions, normalized)
    let embedding: Vec<f32> = (0..768)
        .map(|_| rng.gen_range(-1.0..1.0))
        .collect();

    // Normalize
    let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
    let normalized_embedding: Vec<f32> = embedding.iter().map(|x| x / norm).collect();

    // Random genres (1-3)
    let genre_count = rng.gen_range(1..=3);
    let mut genres = Vec::new();
    for _ in 0..genre_count {
        let genre = GENRES[rng.gen_range(0..GENRES.len())];
        if !genres.contains(&genre.to_string()) {
            genres.push(genre.to_string());
        }
    }

    let mut metadata = HashMap::new();
    metadata.insert("year".to_string(), YEARS[rng.gen_range(0..YEARS.len())].to_string());
    metadata.insert("rating".to_string(), format!("{:.1}", rng.gen_range(1.0..10.0)));
    metadata.insert("duration".to_string(), format!("{}", rng.gen_range(60..180)));

    MediaContent {
        id: id.to_string(),
        title: format!("Title {}", id),
        embedding: normalized_embedding,
        genres,
        metadata,
    }
}

pub fn generate_similar_media(base: &MediaContent, similarity: f32) -> MediaContent {
    let mut rng = rand::thread_rng();

    // Create similar embedding by interpolating
    let noise: Vec<f32> = (0..768).map(|_| rng.gen_range(-1.0..1.0)).collect();
    let mut similar_embedding: Vec<f32> = base.embedding
        .iter()
        .zip(noise.iter())
        .map(|(b, n)| b * similarity + n * (1.0 - similarity))
        .collect();

    // Normalize
    let norm: f32 = similar_embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
    similar_embedding = similar_embedding.iter().map(|x| x / norm).collect();

    MediaContent {
        id: format!("{}_similar", base.id),
        title: format!("{} Similar", base.title),
        embedding: similar_embedding,
        genres: base.genres.clone(),
        metadata: base.metadata.clone(),
    }
}
