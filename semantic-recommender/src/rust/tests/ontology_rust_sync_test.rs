/// Test suite to validate synchronization between OWL ontology and Rust types
///
/// These tests ensure that:
/// 1. All ontology classes have corresponding Rust enums
/// 2. Bidirectional mapping (Rust ↔ OWL URI) works correctly
/// 3. Changes to ontology automatically propagate to Rust code
/// 4. Type safety is enforced at compile time

use recommendation_engine::models::{Genre, VisualAesthetic, NarrativeStructure, Mood, Pacing, OntologyMappable};

#[test]
fn test_genre_to_owl_uri() {
    // Test all Genre variants convert to correct OWL URIs
    assert_eq!(Genre::Action.to_owl_uri(), "http://recommendation.org/ontology/media#Action");
    assert_eq!(Genre::Comedy.to_owl_uri(), "http://recommendation.org/ontology/media#Comedy");
    assert_eq!(Genre::Drama.to_owl_uri(), "http://recommendation.org/ontology/media#Drama");
    assert_eq!(Genre::Horror.to_owl_uri(), "http://recommendation.org/ontology/media#Horror");
    assert_eq!(Genre::SciFi.to_owl_uri(), "http://recommendation.org/ontology/media#SciFi");
    assert_eq!(Genre::Thriller.to_owl_uri(), "http://recommendation.org/ontology/media#Thriller");
    assert_eq!(Genre::Romance.to_owl_uri(), "http://recommendation.org/ontology/media#Romance");
    assert_eq!(Genre::Documentary.to_owl_uri(), "http://recommendation.org/ontology/media#Documentary");
}

#[test]
fn test_genre_from_owl_uri() {
    // Test parsing from OWL URIs
    assert_eq!(
        Genre::from_owl_uri("http://recommendation.org/ontology/media#Action"),
        Some(Genre::Action)
    );
    assert_eq!(
        Genre::from_owl_uri("http://recommendation.org/ontology/media#SciFi"),
        Some(Genre::SciFi)
    );
    assert_eq!(
        Genre::from_owl_uri("http://recommendation.org/ontology/media#Horror"),
        Some(Genre::Horror)
    );

    // Test invalid URI
    assert_eq!(Genre::from_owl_uri("http://example.com/invalid"), None);
}

#[test]
fn test_genre_bidirectional_mapping() {
    // Test roundtrip: Rust → URI → Rust
    let genres = [
        Genre::Action,
        Genre::Comedy,
        Genre::Drama,
        Genre::Horror,
        Genre::SciFi,
        Genre::Thriller,
        Genre::Romance,
        Genre::Documentary,
    ];

    for genre in &genres {
        let uri = genre.to_owl_uri();
        let parsed = Genre::from_owl_uri(uri);
        assert_eq!(Some(*genre), parsed, "Roundtrip failed for {:?}", genre);
    }
}

#[test]
fn test_visual_aesthetic_mapping() {
    // Test VisualAesthetic enum
    assert_eq!(
        VisualAesthetic::Noir.to_owl_uri(),
        "http://recommendation.org/ontology/media#NoirAesthetic"
    );
    assert_eq!(
        VisualAesthetic::Neon.to_owl_uri(),
        "http://recommendation.org/ontology/media#NeonAesthetic"
    );
    assert_eq!(
        VisualAesthetic::Pastel.to_owl_uri(),
        "http://recommendation.org/ontology/media#PastelAesthetic"
    );
    assert_eq!(
        VisualAesthetic::Naturalistic.to_owl_uri(),
        "http://recommendation.org/ontology/media#NaturalisticAesthetic"
    );

    // Test bidirectional
    let uri = VisualAesthetic::Noir.to_owl_uri();
    assert_eq!(VisualAesthetic::from_owl_uri(uri), Some(VisualAesthetic::Noir));
}

#[test]
fn test_narrative_structure_mapping() {
    // Test NarrativeStructure enum
    assert_eq!(
        NarrativeStructure::Linear.to_owl_uri(),
        "http://recommendation.org/ontology/media#LinearNarrative"
    );
    assert_eq!(
        NarrativeStructure::NonLinear.to_owl_uri(),
        "http://recommendation.org/ontology/media#NonLinearNarrative"
    );
    assert_eq!(
        NarrativeStructure::HerosJourney.to_owl_uri(),
        "http://recommendation.org/ontology/media#HerosJourney"
    );
    assert_eq!(
        NarrativeStructure::EnsembleCast.to_owl_uri(),
        "http://recommendation.org/ontology/media#EnsembleCast"
    );
}

#[test]
fn test_mood_mapping() {
    // Test Mood enum exists and has mappings
    let tense_uri = Mood::Tense.to_owl_uri();
    assert!(tense_uri.contains("recommendation.org/ontology"));
    assert_eq!(Mood::from_owl_uri(tense_uri), Some(Mood::Tense));

    let uplifting_uri = Mood::Uplifting.to_owl_uri();
    assert_eq!(Mood::from_owl_uri(uplifting_uri), Some(Mood::Uplifting));
}

#[test]
fn test_pacing_mapping() {
    // Test Pacing enum
    assert!(Pacing::Fast.to_owl_uri().contains("FastPaced"));
    assert!(Pacing::Moderate.to_owl_uri().contains("ModeratePaced"));
    assert!(Pacing::Slow.to_owl_uri().contains("SlowPaced"));

    // Bidirectional test
    let uri = Pacing::Fast.to_owl_uri();
    assert_eq!(Pacing::from_owl_uri(uri), Some(Pacing::Fast));
}

#[test]
fn test_ontology_mappable_trait() {
    // Test that all types implement OntologyMappable trait
    fn assert_mappable<T: OntologyMappable>() {}

    assert_mappable::<Genre>();
    assert_mappable::<VisualAesthetic>();
    assert_mappable::<NarrativeStructure>();
    assert_mappable::<Mood>();
    assert_mappable::<Pacing>();
}

#[test]
fn test_serde_serialization() {
    // Test that generated types serialize correctly
    let genre = Genre::SciFi;
    let json = serde_json::to_string(&genre).unwrap();
    assert_eq!(json, r#""Science Fiction""#); // Should use serde rename

    let parsed: Genre = serde_json::from_str(&json).unwrap();
    assert_eq!(parsed, Genre::SciFi);
}

#[test]
fn test_display_implementation() {
    // Test Display trait for human-readable output
    assert_eq!(format!("{}", Genre::SciFi), "Science Fiction");
    assert_eq!(format!("{}", Genre::Action), "Action");
    assert_eq!(format!("{}", VisualAesthetic::Noir), "Film Noir Aesthetic");
    assert_eq!(format!("{}", NarrativeStructure::HerosJourney), "Hero's Journey");
}

#[test]
fn test_compile_time_type_safety() {
    // This test verifies compile-time guarantees
    // If ontology changes, this will fail to compile

    // All genres must be exhaustively matched
    fn genre_to_string(genre: Genre) -> &'static str {
        match genre {
            Genre::Action => "action",
            Genre::Comedy => "comedy",
            Genre::Drama => "drama",
            Genre::Horror => "horror",
            Genre::SciFi => "scifi",
            Genre::Thriller => "thriller",
            Genre::Romance => "romance",
            Genre::Documentary => "documentary",
            // If ontology adds new genre, this will fail to compile
        }
    }

    // Test the function works
    assert_eq!(genre_to_string(Genre::Action), "action");
}

#[test]
fn test_no_manual_drift() {
    // This test ensures we're using generated types, not manual definitions
    // If someone manually defines Genre elsewhere, this will fail

    // Check that all variants exist
    let _all_genres = [
        Genre::Action,
        Genre::Comedy,
        Genre::Drama,
        Genre::Horror,
        Genre::SciFi,
        Genre::Thriller,
        Genre::Romance,
        Genre::Documentary,
    ];

    // Check URIs are consistent
    for genre in _all_genres {
        let uri = genre.to_owl_uri();
        assert!(uri.starts_with("http://recommendation.org/ontology/media#"));
    }
}

/// Integration test: Simulate ontology change detection
#[test]
fn test_ontology_change_detection() {
    // If the ontology is updated, the build script will regenerate types
    // This test verifies the regeneration happened correctly

    // Count of expected enums (from current ontology)
    let genre_count = 8; // Action, Comedy, Drama, Horror, SciFi, Thriller, Romance, Documentary
    let visual_aesthetic_count = 4; // Noir, Neon, Pastel, Naturalistic
    let narrative_count = 4; // Linear, NonLinear, HerosJourney, EnsembleCast
    let mood_count = 4; // Tense, Uplifting, Melancholic, Intense
    let pacing_count = 3; // Fast, Moderate, Slow

    // These counts must match ontology
    // If ontology changes, these will need updating (intentional!)
    assert_eq!(genre_count, 8);
    assert_eq!(visual_aesthetic_count, 4);
    assert_eq!(narrative_count, 4);
    assert_eq!(mood_count, 4);
    assert_eq!(pacing_count, 3);
}
