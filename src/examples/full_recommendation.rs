/// Example: Complete Recommendation Pipeline
///
/// Demonstrates:
/// - User profile creation
/// - Content embeddings and fusion
/// - Semantic path discovery
/// - Recommendation generation with explanations
/// - Ranking factors and scoring

use hackathon_tv5::models::{
    UserProfile, UserId, Interaction, InteractionType, DeviceType,
    PsychographicState, StateType, ViewingContext, TimeOfDay, SocialSetting,
    AmbientConditions, MediaContent, ContentId, ContentType, Genre,
    Recommendation, RecommendationScore, SemanticPath, PathNode, NodeType,
    PathType, RankingFactors, ExplanationReason, ReasonType,
    RecommendationBatch,
};
use std::collections::HashMap;

fn main() {
    println!("=== Complete Recommendation Pipeline Example ===\n");

    // Step 1: Create user profile with history
    let user = create_sample_user();
    println!("1. User Profile:");
    println!("  User ID: {}", user.user_id.as_str());
    println!("  Watch history: {} items", user.watch_history.len());
    println!("  Avg completion rate: {:.1}%", user.avg_completion_rate() * 100.0);
    println!("  Current state: {:?}", user.current_state.as_ref().map(|s| s.state));
    println!("  Taste cluster: {:?}\n", user.taste_cluster.as_ref().map(|t| t.cluster_id));

    // Step 2: Get user context
    let context = get_viewing_context();
    println!("2. Viewing Context:");
    println!("  Time: {:?}", context.time_of_day);
    println!("  Device: {:?}", context.device);
    println!("  Setting: {:?}", context.social_setting);
    println!("  Light level: {:.1}%\n", context.ambient.light_level * 100.0);

    // Step 3: Create content catalog
    let catalog = create_sample_catalog();
    println!("3. Content Catalog:");
    println!("  Total items: {}\n", catalog.len());

    // Step 4: Generate candidate recommendations
    let candidates = generate_candidates(&user, &catalog, &context);
    println!("4. Candidate Generation:");
    println!("  Candidates: {} items\n", candidates.len());

    // Step 5: Rank and score
    let recommendations = rank_and_score(candidates, &user, &context);
    println!("5. Final Recommendations:");
    for (i, rec) in recommendations.iter().take(5).enumerate() {
        println!("\n  {}. {} (Score: {:.3})",
                 i + 1,
                 rec.content.metadata.title,
                 rec.score.total);
        println!("     Genres: {:?}", rec.content.genres);
        println!("     Explanation: {}", rec.explanation);

        // Show score breakdown
        println!("     Score breakdown:");
        println!("       Relevance: {:.3}", rec.score.relevance);
        println!("       Personalization: {:.3}", rec.score.personalization);
        println!("       Quality: {:.3}", rec.score.quality);
        println!("       Diversity: {:.3}", rec.score.diversity);

        // Show semantic path if available
        if let Some(path) = &rec.semantic_path {
            println!("     Semantic path: {}", path.explain());
            println!("       Length: {} hops, Strength: {:.2}",
                     path.length, path.strength);
        }

        // Show ranking factors
        if let Some(reason) = rec.ranking_factors.top_reasons(1).first() {
            println!("     Top reason: {:?} (importance: {:.2})",
                     reason.reason_type, reason.importance);
        }
    }

    // Step 6: Create recommendation batch
    let batch = RecommendationBatch::new(user.user_id.as_str(), recommendations);
    println!("\n6. Batch Statistics:");
    println!("  Total recommendations: {}", batch.recommendations.len());
    println!("  Average score: {:.3}", batch.avg_score());
    println!("  High confidence (>0.8): {}",
             batch.filter_by_score(0.8).len());

    // Step 7: Simulate user interaction and learning
    println!("\n7. User Interaction Simulation:");
    let selected = batch.top_n(3);
    for rec in selected {
        println!("  User watches: {}", rec.content.metadata.title);

        // Calculate reward based on completion
        let completion_rate = 0.85; // Simulated
        let reward = calculate_reward(completion_rate, rec.score.total);
        println!("    Completion: {:.1}%, Reward: {:.3}",
                 completion_rate * 100.0, reward);
    }
}

/// Create sample user with rich profile
fn create_sample_user() -> UserProfile {
    let mut user = UserProfile::new(UserId::new());

    // Add watch history
    let interactions = vec![
        ("Inception", InteractionType::Complete, 0.95),
        ("Interstellar", InteractionType::Complete, 0.90),
        ("The Matrix", InteractionType::Watch, 0.75),
        ("Blade Runner 2049", InteractionType::Complete, 0.88),
        ("Arrival", InteractionType::Skip, 0.35),
    ];

    for (title, interaction_type, completion) in interactions {
        user.add_interaction(Interaction {
            content_id: format!("film:{}", title.to_lowercase()),
            interaction_type,
            timestamp: chrono::Utc::now(),
            watch_duration: Some((completion * 7200.0) as u32),
            content_duration: Some(7200),
            watch_completion_rate: Some(completion),
            rating: None,
            device: DeviceType::TV,
            context: None,
        });
    }

    // Set psychographic state
    user.current_state = Some(PsychographicState::new(
        StateType::Relaxed,
        0.7,
    ));

    // Set taste cluster (sci-fi enthusiast)
    user.taste_cluster = Some(create_taste_cluster(1, "Sci-Fi Enthusiast"));

    // Update embedding based on history
    let content_embeddings: Vec<Vec<f32>> = (0..5)
        .map(|i| generate_scifi_embedding(i as f32 / 5.0))
        .collect();
    user.update_embedding(&content_embeddings);

    user
}

/// Get current viewing context
fn get_viewing_context() -> ViewingContext {
    ViewingContext {
        time_of_day: TimeOfDay::Evening,
        day_of_week: chrono::Weekday::Fri,
        device: DeviceType::TV,
        network_speed: Some(50.0), // Mbps
        location: Some("San Francisco".to_string()),
        social_setting: SocialSetting::Solo,
        ambient: AmbientConditions {
            light_level: 0.3, // Dim lighting
            noise_level: 0.2, // Quiet
        },
    }
}

/// Create sample content catalog
fn create_sample_catalog() -> Vec<MediaContent> {
    vec![
        create_film("Dune", vec![Genre::SciFi, Genre::Adventure], 0.8),
        create_film("The Prestige", vec![Genre::Thriller, Genre::Mystery], 0.75),
        create_film("Ex Machina", vec![Genre::SciFi, Genre::Thriller], 0.82),
        create_film("Annihilation", vec![Genre::SciFi, Genre::Horror], 0.78),
        create_film("Moon", vec![Genre::SciFi, Genre::Drama], 0.85),
        create_film("The Notebook", vec![Genre::Romance, Genre::Drama], 0.45),
        create_film("Fast & Furious", vec![Genre::Action], 0.40),
        create_film("Tenet", vec![Genre::SciFi, Genre::Action], 0.80),
    ]
}

/// Generate candidate recommendations
fn generate_candidates(
    user: &UserProfile,
    catalog: &[MediaContent],
    _context: &ViewingContext,
) -> Vec<MediaContent> {
    // Simple similarity-based filtering
    let mut candidates: Vec<_> = catalog
        .iter()
        .map(|content| {
            let similarity = cosine_similarity(&user.user_embedding, &content.unified_embedding);
            (content.clone(), similarity)
        })
        .filter(|(_, sim)| *sim > 0.5) // Threshold
        .collect();

    // Sort by similarity
    candidates.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    candidates.into_iter()
        .map(|(content, _)| content)
        .take(10)
        .collect()
}

/// Rank and score candidates
fn rank_and_score(
    candidates: Vec<MediaContent>,
    user: &UserProfile,
    context: &ViewingContext,
) -> Vec<Recommendation> {
    candidates
        .into_iter()
        .enumerate()
        .map(|(i, content)| {
            // Compute scores
            let relevance = cosine_similarity(&user.user_embedding, &content.unified_embedding);
            let personalization = compute_personalization(&content, user);
            let quality = content.metadata.ratings.imdb.unwrap_or(7.0) / 10.0;
            let diversity = 1.0 - (i as f32 / 10.0) * 0.2; // Decrease with rank

            let score = RecommendationScore::from_components(
                relevance,
                personalization,
                quality,
                diversity,
            ).with_confidence(0.85);

            // Create semantic path
            let path = create_semantic_path(&user, &content);

            // Create ranking factors
            let mut factors = RankingFactors::default();
            factors.vector_similarity = relevance;
            factors.neural_score = personalization;
            factors.mf_score = quality;
            factors.add_reason(ExplanationReason::new(
                ReasonType::SimilarTo,
                format!("Similar to films you've enjoyed like Inception"),
                0.9,
            ));

            if matches!(context.time_of_day, TimeOfDay::Evening) {
                factors.temporal_boost = 0.1;
                factors.add_reason(ExplanationReason::new(
                    ReasonType::MatchesMood,
                    "Perfect for evening relaxation".to_string(),
                    0.7,
                ));
            }

            // Generate explanation
            let explanation = generate_explanation(&content, &factors);

            Recommendation::new(content, score.total, explanation)
                .with_path(path)
                .with_factors(factors)
                .with_rank(i + 1)
        })
        .collect()
}

/// Create semantic path from user to content
fn create_semantic_path(user: &UserProfile, content: &MediaContent) -> SemanticPath {
    let nodes = vec![
        PathNode {
            id: user.user_id.as_str(),
            node_type: NodeType::User,
            label: "You".to_string(),
            edge_weight: Some(0.9),
        },
        PathNode {
            id: "genre:scifi".to_string(),
            node_type: NodeType::Genre,
            label: "Sci-Fi Genre".to_string(),
            edge_weight: Some(0.85),
        },
        PathNode {
            id: content.id.as_str().to_string(),
            node_type: NodeType::Content,
            label: content.metadata.title.clone(),
            edge_weight: None,
        },
    ];

    SemanticPath::new(nodes, PathType::GenrePath)
}

/// Compute personalization score
fn compute_personalization(content: &MediaContent, user: &UserProfile) -> f32 {
    let mut score = 0.5;

    // Check genre match
    for genre in &content.genres {
        if user.preferences.favorite_genres.contains(&format!("{:?}", genre)) {
            score += 0.2;
        }
    }

    // Check completion history
    if user.avg_completion_rate() > 0.8 {
        score += 0.1;
    }

    score.min(1.0)
}

/// Generate human-readable explanation
fn generate_explanation(content: &MediaContent, factors: &RankingFactors) -> String {
    let mut parts = Vec::new();

    if factors.vector_similarity > 0.7 {
        parts.push("Highly similar to your favorites");
    }

    if let Some(genre) = content.primary_genre() {
        parts.push(&format!("Acclaimed {:?} film", genre));
    }

    if factors.temporal_boost > 0.0 {
        parts.push("Perfect timing for this viewing");
    }

    if parts.is_empty() {
        "Recommended based on your taste".to_string()
    } else {
        parts.join(". ")
    }
}

/// Calculate reward for RL learning
fn calculate_reward(completion_rate: f32, predicted_score: f32) -> f32 {
    0.3 * (if completion_rate > 0.5 { 1.0 } else { 0.0 })
        + 0.5 * completion_rate
        + 0.2 * predicted_score
}

// Helper functions

fn create_film(title: &str, genres: Vec<Genre>, quality: f32) -> MediaContent {
    let mut content = MediaContent::new(
        ContentId::new(format!("film:{}", title.to_lowercase())),
        ContentType::Film,
        title.to_string(),
    );

    content.genres = genres;
    content.unified_embedding = generate_scifi_embedding(quality);
    content.metadata.ratings.imdb = Some((quality * 10.0) as f32);

    content
}

fn generate_scifi_embedding(weight: f32) -> Vec<f32> {
    let mut embedding = vec![0.0; 1024];

    // Fill with sci-fi pattern
    for (i, val) in embedding.iter_mut().enumerate() {
        *val = weight * (1.0 + (i as f32 * 0.01).sin()) * 0.1;
    }

    // Normalize
    let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
    for val in &mut embedding {
        *val /= norm;
    }

    embedding
}

fn create_taste_cluster(id: u32, label: &str) -> hackathon_tv5::models::TasteCluster {
    let mut cluster = hackathon_tv5::models::TasteCluster::new(id, vec![0.5; 1024]);
    cluster.characteristics = vec![label.to_string()];
    cluster.typical_genres = vec!["SciFi".to_string(), "Thriller".to_string()];
    cluster
}

fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() {
        return 0.0;
    }
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm_a == 0.0 || norm_b == 0.0 { 0.0 } else { dot / (norm_a * norm_b) }
}

// To run: cargo run --example full_recommendation
