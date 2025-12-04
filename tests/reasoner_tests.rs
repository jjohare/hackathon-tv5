/// Comprehensive Test Suite for MediaReasoner
///
/// Tests transitive closure, paradox detection, circular reasoning prevention,
/// and cultural context scoring accuracy.

use recommendation_engine::ontology::reasoner::*;
use recommendation_engine::ontology::types::*;
use std::collections::{HashMap, HashSet};

/// Helper function to create a large test ontology
fn create_large_ontology(num_nodes: usize) -> MediaOntology {
    let mut ontology = MediaOntology::default();

    // Create a deep hierarchy: Genre0 -> Genre1 -> ... -> GenreN
    for i in 1..num_nodes {
        let child = format!("Genre{}", i);
        let parent = format!("Genre{}", i - 1);

        ontology.genre_hierarchy.insert(
            child,
            vec![parent].into_iter().collect()
        );
    }

    // Add some branching
    for i in (0..num_nodes).step_by(10) {
        let branch = format!("BranchGenre{}", i);
        let parent = format!("Genre{}", i);

        ontology.genre_hierarchy.insert(
            branch,
            vec![parent].into_iter().collect()
        );
    }

    ontology
}

/// Helper to create test media with specific properties
fn create_test_media(
    id: &str,
    genres: Vec<&str>,
    tags: Vec<&str>,
) -> MediaEntity {
    MediaEntity {
        id: id.to_string(),
        title: format!("Test Media: {}", id),
        media_type: MediaType::Video,
        genres: genres.iter().map(|s| s.to_string()).collect(),
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
        semantic_tags: tags.iter().map(|s| s.to_string()).collect(),
    }
}

#[test]
fn test_transitive_closure_small() {
    let mut ontology = MediaOntology::default();

    // Create hierarchy: Action -> Thriller -> Drama
    ontology.genre_hierarchy.insert(
        "Thriller".to_string(),
        vec!["Drama".to_string()].into_iter().collect()
    );
    ontology.genre_hierarchy.insert(
        "Action".to_string(),
        vec!["Thriller".to_string()].into_iter().collect()
    );

    let mut reasoner = ProductionMediaReasoner::new();
    reasoner.compute_genre_closure(&ontology);

    // Action should have both Thriller and Drama as ancestors
    assert!(reasoner.is_subgenre_of("Action", "Thriller", &ontology));
    assert!(reasoner.is_subgenre_of("Action", "Drama", &ontology));
    assert!(reasoner.is_subgenre_of("Thriller", "Drama", &ontology));

    // Reverse should not be true
    assert!(!reasoner.is_subgenre_of("Drama", "Thriller", &ontology));
}

#[test]
fn test_transitive_closure_large_graph() {
    use std::time::Instant;

    let ontology = create_large_ontology(10_000);
    let mut reasoner = ProductionMediaReasoner::new();

    let start = Instant::now();
    reasoner.compute_genre_closure(&ontology);
    let duration = start.elapsed();

    println!("Transitive closure on 10K nodes took: {:?}", duration);

    // Should complete in under 50ms (target benchmark)
    assert!(duration.as_millis() < 50, "Transitive closure took {}ms, expected <50ms", duration.as_millis());

    // Verify correctness: Genre9999 should be subgenre of Genre0
    assert!(reasoner.is_subgenre_of("Genre9999", "Genre0", &ontology));
    assert!(reasoner.is_subgenre_of("Genre5000", "Genre0", &ontology));

    // Verify branch genres
    assert!(reasoner.is_subgenre_of("BranchGenre100", "Genre0", &ontology));
}

#[test]
fn test_circular_dependency_detection() {
    let mut ontology = MediaOntology::default();

    // Create circular dependency: A -> B -> C -> A
    ontology.genre_hierarchy.insert(
        "GenreA".to_string(),
        vec!["GenreB".to_string()].into_iter().collect()
    );
    ontology.genre_hierarchy.insert(
        "GenreB".to_string(),
        vec!["GenreC".to_string()].into_iter().collect()
    );
    ontology.genre_hierarchy.insert(
        "GenreC".to_string(),
        vec!["GenreA".to_string()].into_iter().collect()
    );

    let mut reasoner = ProductionMediaReasoner::new();
    let violations = reasoner.detect_circular_dependencies(&ontology);

    assert_eq!(violations.len(), 1);
    assert_eq!(violations[0].violation_type, ViolationType::CircularHierarchy);
    assert_eq!(violations[0].severity, ViolationSeverity::Critical);

    println!("Detected circular dependency: {}", violations[0].explanation);
}

#[test]
fn test_paradox_detection_family_friendly_rated_r() {
    let media = create_test_media(
        "paradox1",
        vec!["FamilyFriendly", "Animation"],
        vec!["Rated-R", "Violence"]
    );

    let reasoner = ProductionMediaReasoner::new();
    let violations = reasoner.check_paradoxical_properties(&media);

    assert!(!violations.is_empty(), "Should detect paradox");
    assert_eq!(violations[0].violation_type, ViolationType::ParadoxicalProperty);

    println!("Detected paradox: {}", violations[0].explanation);
}

#[test]
fn test_paradox_detection_educational_exploitation() {
    let media = create_test_media(
        "paradox2",
        vec!["Educational", "Documentary"],
        vec!["Exploitation", "Gratuitous"]
    );

    let reasoner = ProductionMediaReasoner::new();
    let violations = reasoner.check_paradoxical_properties(&media);

    assert!(!violations.is_empty(), "Should detect educational/exploitation paradox");
    println!("Detected paradox: {}", violations[0].explanation);
}

#[test]
fn test_disjoint_genre_violation() {
    let mut ontology = MediaOntology::default();

    // Define Comedy and Horror as disjoint
    ontology.disjoint_genres.push(
        vec!["Comedy".to_string(), "Horror".to_string()].into_iter().collect()
    );

    // Create media with both disjoint genres
    let media = create_test_media(
        "disjoint1",
        vec!["Comedy", "Horror"],
        vec![]
    );

    ontology.media.insert(media.id.clone(), media.clone());

    let reasoner = ProductionMediaReasoner::new();
    let violations = reasoner.check_disjoint_violations(&media, &ontology);

    assert_eq!(violations.len(), 1);
    assert_eq!(violations[0].violation_type, ViolationType::DisjointGenreConflict);
    assert_eq!(violations[0].severity, ViolationSeverity::Error);

    println!("Detected disjoint violation: {}", violations[0].explanation);
}

#[test]
fn test_disjoint_with_subgenres() {
    let mut ontology = MediaOntology::default();

    // Setup hierarchy: RomCom -> Comedy
    ontology.genre_hierarchy.insert(
        "RomCom".to_string(),
        vec!["Comedy".to_string()].into_iter().collect()
    );

    // Define Comedy and Horror as disjoint
    ontology.disjoint_genres.push(
        vec!["Comedy".to_string(), "Horror".to_string()].into_iter().collect()
    );

    // Media with RomCom (subgenre of Comedy) and Horror should violate
    let media = create_test_media(
        "disjoint2",
        vec!["RomCom", "Horror"],
        vec![]
    );

    ontology.media.insert(media.id.clone(), media.clone());

    let mut reasoner = ProductionMediaReasoner::new();
    reasoner.compute_genre_closure(&ontology);

    let violations = reasoner.check_disjoint_violations(&media, &ontology);

    assert!(!violations.is_empty(), "Should detect subgenre disjoint violation");
    println!("Detected subgenre disjoint: {}", violations[0].explanation);
}

#[test]
fn test_cultural_context_exact_match() {
    let media = MediaEntity {
        id: "cultural1".to_string(),
        title: "US Drama".to_string(),
        media_type: MediaType::Video,
        genres: vec!["Drama".to_string()],
        moods: vec![],
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
    let score = reasoner.match_cultural_context(&media, &context);

    assert!(score > 0.8, "Exact cultural match should score >0.8, got {}", score);
    println!("Cultural context score (exact match): {}", score);
}

#[test]
fn test_cultural_context_partial_language_match() {
    let media = MediaEntity {
        id: "cultural2".to_string(),
        title: "UK Drama".to_string(),
        media_type: MediaType::Video,
        genres: vec!["Drama".to_string()],
        moods: vec![],
        themes: vec![],
        cultural_context: vec!["en-GB".to_string()],
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
        cultural_themes: vec![],
        taboos: vec![],
        preferences: HashMap::new(),
    };

    let reasoner = ProductionMediaReasoner::new();
    let score = reasoner.match_cultural_context(&media, &context);

    // Should get partial credit for same language family
    assert!(score > 0.5 && score < 0.8, "Partial language match should score 0.5-0.8, got {}", score);
    println!("Cultural context score (partial language): {}", score);
}

#[test]
fn test_cultural_context_taboo_penalty() {
    let media = MediaEntity {
        id: "cultural3".to_string(),
        title: "Controversial Content".to_string(),
        media_type: MediaType::Video,
        genres: vec!["Drama".to_string()],
        moods: vec![],
        themes: vec!["violence".to_string(), "drugs".to_string()],
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
        cultural_themes: vec![],
        taboos: vec!["violence".to_string(), "drugs".to_string()],
        preferences: HashMap::new(),
    };

    let reasoner = ProductionMediaReasoner::new();
    let score = reasoner.match_cultural_context(&media, &context);

    // Taboo violations should significantly reduce score
    assert!(score < 0.3, "Taboo violations should result in low score, got {}", score);
    println!("Cultural context score (with taboos): {}", score);
}

#[test]
fn test_mood_similarity_vad_model() {
    let mut ontology = MediaOntology::default();

    // Similar moods with close VAD values
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

    // Very different mood
    ontology.mood_relations.insert("Joyful".to_string(), Mood {
        name: "Joyful".to_string(),
        valence: 0.9,
        arousal: 0.6,
        dominance: 0.7,
        related_moods: vec![],
    });

    let reasoner = ProductionMediaReasoner::new();

    let tense = ontology.mood_relations.get("Tense").unwrap();
    let anxious = ontology.mood_relations.get("Anxious").unwrap();
    let joyful = ontology.mood_relations.get("Joyful").unwrap();

    let similar_score = reasoner.calculate_mood_similarity(tense, anxious);
    let different_score = reasoner.calculate_mood_similarity(tense, joyful);

    assert!(similar_score > 0.7, "Similar moods should score >0.7, got {}", similar_score);
    assert!(different_score < 0.3, "Different moods should score <0.3, got {}", different_score);

    println!("Mood similarity (Tense-Anxious): {}", similar_score);
    println!("Mood similarity (Tense-Joyful): {}", different_score);
}

#[test]
fn test_comprehensive_constraint_checking() {
    let mut ontology = create_large_ontology(100);

    // Add circular dependency
    ontology.genre_hierarchy.insert(
        "Genre0".to_string(),
        vec!["Genre99".to_string()].into_iter().collect()
    );

    // Add disjoint genres
    ontology.disjoint_genres.push(
        vec!["Comedy".to_string(), "Horror".to_string()].into_iter().collect()
    );

    // Add media with violations
    let paradox_media = create_test_media(
        "paradox",
        vec!["FamilyFriendly"],
        vec!["Rated-R"]
    );
    ontology.media.insert(paradox_media.id.clone(), paradox_media);

    let disjoint_media = create_test_media(
        "disjoint",
        vec!["Comedy", "Horror"],
        vec![]
    );
    ontology.media.insert(disjoint_media.id.clone(), disjoint_media);

    let mut reasoner = ProductionMediaReasoner::new();
    let violations = reasoner.check_all_constraints(&ontology);

    println!("\nFound {} total violations:", violations.len());
    for (i, violation) in violations.iter().enumerate() {
        println!("{}. [{:?}] {:?}: {}",
            i + 1,
            violation.severity,
            violation.violation_type,
            violation.explanation
        );
    }

    // Should find circular, paradox, and disjoint violations
    assert!(!violations.is_empty(), "Should detect multiple violations");

    // Check for each type
    assert!(violations.iter().any(|v| v.violation_type == ViolationType::CircularHierarchy));
    assert!(violations.iter().any(|v| v.violation_type == ViolationType::ParadoxicalProperty));
    assert!(violations.iter().any(|v| v.violation_type == ViolationType::DisjointGenreConflict));
}

#[test]
fn test_performance_100k_nodes() {
    use std::time::Instant;

    println!("\nPerformance test with 100K nodes...");

    let ontology = create_large_ontology(100_000);
    let mut reasoner = ProductionMediaReasoner::new();

    let start = Instant::now();
    reasoner.compute_genre_closure(&ontology);
    let duration = start.elapsed();

    println!("Transitive closure on 100K nodes took: {:?}", duration);

    // Verify correctness on large graph
    assert!(reasoner.is_subgenre_of("Genre99999", "Genre0", &ontology));

    // Should handle large graphs efficiently (target: <500ms for 100K)
    assert!(duration.as_millis() < 500,
        "100K node closure took {}ms, expected <500ms",
        duration.as_millis()
    );
}

#[test]
fn test_inference_quality() {
    let mut ontology = MediaOntology::default();

    // Create realistic genre hierarchy
    ontology.genre_hierarchy.insert(
        "PsychologicalThriller".to_string(),
        vec!["Thriller".to_string()].into_iter().collect()
    );
    ontology.genre_hierarchy.insert(
        "Thriller".to_string(),
        vec!["Drama".to_string()].into_iter().collect()
    );
    ontology.genre_hierarchy.insert(
        "ActionThriller".to_string(),
        vec!["Thriller".to_string(), "Action".to_string()].into_iter().collect()
    );

    let reasoner = ProductionMediaReasoner::new();
    let inferred = reasoner.infer_axioms(&ontology).unwrap();

    println!("\nInferred {} axioms:", inferred.len());
    for axiom in &inferred {
        println!("  - {:?}: {} -> {:?} (confidence: {})",
            axiom.axiom_type,
            axiom.subject,
            axiom.object,
            axiom.confidence
        );
    }

    // Should infer transitive relationships
    assert!(inferred.iter().any(|a|
        a.subject == "PsychologicalThriller"
        && a.object.as_ref() == Some(&"Drama".to_string())
    ));

    // All inferred axioms should have valid confidence
    assert!(inferred.iter().all(|a| a.confidence >= 0.0 && a.confidence <= 1.0));
}
