/// Benchmark Suite for MediaReasoner Performance
///
/// Measures transitive closure performance on graphs of varying sizes,
/// cultural matching throughput, and constraint checking speed.

#![feature(test)]
extern crate test;

use test::Bencher;
use recommendation_engine::ontology::reasoner::*;
use recommendation_engine::ontology::types::*;
use std::collections::HashMap;

/// Create ontology with specified number of nodes in linear hierarchy
fn create_linear_hierarchy(size: usize) -> MediaOntology {
    let mut ontology = MediaOntology::default();

    for i in 1..size {
        let child = format!("Genre{}", i);
        let parent = format!("Genre{}", i - 1);

        ontology.genre_hierarchy.insert(
            child,
            vec![parent].into_iter().collect()
        );
    }

    ontology
}

/// Create ontology with branching hierarchy
fn create_branching_hierarchy(depth: usize, branching_factor: usize) -> MediaOntology {
    let mut ontology = MediaOntology::default();
    let mut node_id = 0;

    fn add_children(
        ontology: &mut MediaOntology,
        parent_id: usize,
        depth: usize,
        branching_factor: usize,
        current_id: &mut usize,
    ) {
        if depth == 0 {
            return;
        }

        for _ in 0..branching_factor {
            *current_id += 1;
            let child = format!("Genre{}", current_id);
            let parent = format!("Genre{}", parent_id);

            ontology.genre_hierarchy.insert(
                child,
                vec![parent].into_iter().collect()
            );

            add_children(ontology, *current_id, depth - 1, branching_factor, current_id);
        }
    }

    add_children(&mut ontology, 0, depth, branching_factor, &mut node_id);
    ontology
}

/// Create media entities for testing
fn create_test_media_batch(count: usize) -> Vec<MediaEntity> {
    (0..count).map(|i| MediaEntity {
        id: format!("media_{}", i),
        title: format!("Test Media {}", i),
        media_type: MediaType::Video,
        genres: vec![format!("Genre{}", i % 10)],
        moods: vec!["Tense".to_string()],
        themes: vec!["family".to_string()],
        cultural_context: vec!["en-US".to_string()],
        technical_metadata: TechnicalMetadata {
            duration_seconds: Some(7200.0),
            resolution: Some("1080p".to_string()),
            format: "mp4".to_string(),
            bitrate: Some(5000),
            file_size_bytes: Some(1_000_000_000),
        },
        semantic_tags: vec![],
    }).collect()
}

#[bench]
fn bench_transitive_closure_1k(b: &mut Bencher) {
    let ontology = create_linear_hierarchy(1_000);

    b.iter(|| {
        let mut reasoner = ProductionMediaReasoner::new();
        reasoner.compute_genre_closure(&ontology);
    });
}

#[bench]
fn bench_transitive_closure_10k(b: &mut Bencher) {
    let ontology = create_linear_hierarchy(10_000);

    b.iter(|| {
        let mut reasoner = ProductionMediaReasoner::new();
        reasoner.compute_genre_closure(&ontology);
    });
}

#[bench]
fn bench_transitive_closure_100k(b: &mut Bencher) {
    let ontology = create_linear_hierarchy(100_000);

    b.iter(|| {
        let mut reasoner = ProductionMediaReasoner::new();
        reasoner.compute_genre_closure(&ontology);
    });
}

#[bench]
fn bench_transitive_closure_branching_10k(b: &mut Bencher) {
    // Create tree with depth 10, branching factor 3 (~10K nodes)
    let ontology = create_branching_hierarchy(10, 3);

    b.iter(|| {
        let mut reasoner = ProductionMediaReasoner::new();
        reasoner.compute_genre_closure(&ontology);
    });
}

#[bench]
fn bench_subgenre_lookup_cached(b: &mut Bencher) {
    let ontology = create_linear_hierarchy(10_000);
    let mut reasoner = ProductionMediaReasoner::new();
    reasoner.compute_genre_closure(&ontology);

    b.iter(|| {
        // Lookup deep in hierarchy
        reasoner.is_subgenre_of("Genre9999", "Genre0", &ontology)
    });
}

#[bench]
fn bench_circular_dependency_detection(b: &mut Bencher) {
    let mut ontology = create_linear_hierarchy(1_000);

    // Add circular dependency
    ontology.genre_hierarchy.insert(
        "Genre0".to_string(),
        vec!["Genre999".to_string()].into_iter().collect()
    );

    b.iter(|| {
        let mut reasoner = ProductionMediaReasoner::new();
        reasoner.detect_circular_dependencies(&ontology)
    });
}

#[bench]
fn bench_cultural_context_matching_single(b: &mut Bencher) {
    let media = MediaEntity {
        id: "test".to_string(),
        title: "Test".to_string(),
        media_type: MediaType::Video,
        genres: vec!["Drama".to_string()],
        moods: vec!["Tense".to_string()],
        themes: vec!["family".to_string(), "tradition".to_string()],
        cultural_context: vec!["en-US".to_string()],
        technical_metadata: TechnicalMetadata {
            duration_seconds: Some(7200.0),
            resolution: Some("1080p".to_string()),
            format: "mp4".to_string(),
            bitrate: Some(5000),
            file_size_bytes: Some(1_000_000_000),
        },
        semantic_tags: vec![],
    };

    let context = CulturalContext {
        region: "US".to_string(),
        language: "en-US".to_string(),
        cultural_themes: vec!["family".to_string()],
        taboos: vec![],
        preferences: HashMap::new(),
    };

    let reasoner = ProductionMediaReasoner::new();

    b.iter(|| {
        reasoner.match_cultural_context(&media, &context)
    });
}

#[bench]
fn bench_cultural_context_matching_batch_1k(b: &mut Bencher) {
    let media_batch = create_test_media_batch(1_000);

    let context = CulturalContext {
        region: "US".to_string(),
        language: "en-US".to_string(),
        cultural_themes: vec!["family".to_string()],
        taboos: vec![],
        preferences: HashMap::new(),
    };

    let reasoner = ProductionMediaReasoner::new();

    b.iter(|| {
        let scores: Vec<f32> = media_batch.iter()
            .map(|m| reasoner.match_cultural_context(m, &context))
            .collect();
        scores
    });
}

#[bench]
fn bench_mood_similarity_calculation(b: &mut Bencher) {
    let mood_a = Mood {
        name: "Tense".to_string(),
        valence: -0.3,
        arousal: 0.8,
        dominance: 0.4,
        related_moods: vec![],
    };

    let mood_b = Mood {
        name: "Anxious".to_string(),
        valence: -0.4,
        arousal: 0.7,
        dominance: 0.3,
        related_moods: vec![],
    };

    let reasoner = ProductionMediaReasoner::new();

    b.iter(|| {
        reasoner.calculate_mood_similarity(&mood_a, &mood_b)
    });
}

#[bench]
fn bench_paradox_detection_single(b: &mut Bencher) {
    let media = MediaEntity {
        id: "paradox".to_string(),
        title: "Paradoxical".to_string(),
        media_type: MediaType::Video,
        genres: vec!["FamilyFriendly".to_string()],
        moods: vec![],
        themes: vec![],
        cultural_context: vec!["en-US".to_string()],
        technical_metadata: TechnicalMetadata {
            duration_seconds: Some(7200.0),
            resolution: Some("1080p".to_string()),
            format: "mp4".to_string(),
            bitrate: Some(5000),
            file_size_bytes: Some(1_000_000_000),
        },
        semantic_tags: vec!["Rated-R".to_string()],
    };

    let reasoner = ProductionMediaReasoner::new();

    b.iter(|| {
        reasoner.check_paradoxical_properties(&media)
    });
}

#[bench]
fn bench_disjoint_violation_check(b: &mut Bencher) {
    let mut ontology = MediaOntology::default();

    ontology.disjoint_genres.push(
        vec!["Comedy".to_string(), "Horror".to_string()].into_iter().collect()
    );

    let media = MediaEntity {
        id: "disjoint".to_string(),
        title: "Disjoint Test".to_string(),
        media_type: MediaType::Video,
        genres: vec!["Comedy".to_string(), "Horror".to_string()],
        moods: vec![],
        themes: vec![],
        cultural_context: vec!["en-US".to_string()],
        technical_metadata: TechnicalMetadata {
            duration_seconds: Some(7200.0),
            resolution: Some("1080p".to_string()),
            format: "mp4".to_string(),
            bitrate: Some(5000),
            file_size_bytes: Some(1_000_000_000),
        },
        semantic_tags: vec![],
    };

    let reasoner = ProductionMediaReasoner::new();

    b.iter(|| {
        reasoner.check_disjoint_violations(&media, &ontology)
    });
}

#[bench]
fn bench_full_axiom_inference(b: &mut Bencher) {
    let mut ontology = create_linear_hierarchy(100);

    // Add moods
    ontology.mood_relations.insert("Tense".to_string(), Mood {
        name: "Tense".to_string(),
        valence: -0.3,
        arousal: 0.8,
        dominance: 0.4,
        related_moods: vec![],
    });

    ontology.mood_relations.insert("Anxious".to_string(), Mood {
        name: "Anxious".to_string(),
        valence: -0.4,
        arousal: 0.7,
        dominance: 0.3,
        related_moods: vec![],
    });

    let reasoner = ProductionMediaReasoner::new();

    b.iter(|| {
        reasoner.infer_axioms(&ontology).unwrap()
    });
}

#[bench]
fn bench_comprehensive_constraint_check_1k_media(b: &mut Bencher) {
    let mut ontology = create_linear_hierarchy(100);

    // Add media entities
    let media_batch = create_test_media_batch(1_000);
    for media in media_batch {
        ontology.media.insert(media.id.clone(), media);
    }

    // Add disjoint genres
    ontology.disjoint_genres.push(
        vec!["Comedy".to_string(), "Horror".to_string()].into_iter().collect()
    );

    b.iter(|| {
        let mut reasoner = ProductionMediaReasoner::new();
        reasoner.check_all_constraints(&ontology)
    });
}

#[bench]
fn bench_recommendation_generation_100_media(b: &mut Bencher) {
    let mut ontology = MediaOntology::default();

    let media_batch = create_test_media_batch(100);
    for media in media_batch {
        ontology.media.insert(media.id.clone(), media);
    }

    let user = UserProfile {
        user_id: "test_user".to_string(),
        preferred_genres: {
            let mut map = HashMap::new();
            map.insert("Drama".to_string(), 0.9);
            map.insert("Action".to_string(), 0.7);
            map
        },
        preferred_moods: {
            let mut map = HashMap::new();
            map.insert("Tense".to_string(), 0.8);
            map
        },
        cultural_background: vec!["en-US".to_string()],
        language_preferences: vec!["en".to_string()],
        interaction_history: vec![],
        demographic: Demographic {
            age_range: Some(AgeRange::Adult),
            location: Some("US".to_string()),
            timezone: Some("America/New_York".to_string()),
        },
    };

    let context = DeliveryContext {
        device_type: DeviceType::TV,
        network_quality: NetworkQuality::Good,
        time_of_day: 20,
        location: Some("US".to_string()),
        social_context: SocialContext::WithFamily,
    };

    let reasoner = ProductionMediaReasoner::new();

    b.iter(|| {
        reasoner.recommend_for_user(&user, &context, &ontology)
    });
}
