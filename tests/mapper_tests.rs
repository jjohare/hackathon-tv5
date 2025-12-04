use serde_json::json;

#[cfg(test)]
mod integration_tests {
    use super::*;
    use hackathon_tv5::pipeline::metadata_mapper::*;

    #[test]
    fn test_visual_analysis_noir_mapping() {
        let mapper = MetadataMapper::new("movie".to_string());

        let input = AnalysisInput {
            media_id: "123".to_string(),
            visual: Some(VisualAnalysis {
                color_palette: vec!["#000000".to_string(), "#1a1a1a".to_string()],
                contrast: 0.95,
                composition_style: Some("symmetrical".to_string()),
                motion_vectors: None,
                lighting: Some(LightingAnalysis {
                    key_light_ratio: 0.3,
                    shadow_depth: 0.85,
                    color_temperature: 2800,
                }),
            }),
            audio: None,
            text: None,
        };

        let triples = mapper.map_to_rdf(&input);

        // Should detect noir aesthetic
        assert!(triples.iter().any(|t|
            t.predicate == "media:hasVisualAesthetic" &&
            t.object == "media:Noir" &&
            t.confidence > 0.7
        ));

        // Should include contrast value
        assert!(triples.iter().any(|t|
            t.predicate == "media:visualContrast" &&
            t.object.contains("0.95")
        ));

        // Should detect low-key lighting
        assert!(triples.iter().any(|t|
            t.predicate == "aesthetic:moodLighting" &&
            t.object.contains("LowKey")
        ));
    }

    #[test]
    fn test_audio_analysis_dialogue_complexity() {
        let mapper = MetadataMapper::new("movie".to_string());

        let input = AnalysisInput {
            media_id: "456".to_string(),
            visual: None,
            audio: Some(AudioAnalysis {
                tempo: Some(120.0),
                key: Some("C major".to_string()),
                dialogue_complexity: Some(0.85),
                sound_design_score: Some(0.9),
                dynamic_range: Some(0.7),
            }),
            text: None,
        };

        let triples = mapper.map_to_rdf(&input);

        // Should map to mature audience
        assert!(triples.iter().any(|t|
            t.predicate == "ctx:audienceLevel" &&
            t.object == "ctx:Mature"
        ));

        // Should include music features
        assert!(triples.iter().any(|t|
            t.predicate == "media:hasMusicFeatures"
        ));

        // Should detect cinematic sound design
        assert!(triples.iter().any(|t|
            t.predicate == "aesthetic:soundDesign" &&
            t.object.contains("Cinematic")
        ));
    }

    #[test]
    fn test_text_analysis_themes_and_tropes() {
        let mapper = MetadataMapper::new("movie".to_string());

        let input = AnalysisInput {
            media_id: "789".to_string(),
            visual: None,
            audio: None,
            text: Some(TextAnalysis {
                themes: vec![
                    ThemeDetection {
                        name: "redemption".to_string(),
                        confidence: 0.92,
                    },
                    ThemeDetection {
                        name: "betrayal".to_string(),
                        confidence: 0.78,
                    },
                ],
                tropes: vec![
                    TropeDetection {
                        name: "hero's journey".to_string(),
                        confidence: 0.88,
                        timestamp: Some(120.0),
                    },
                ],
                emotional_arc: Some(EmotionalArc {
                    start_valence: 0.3,
                    peak_valence: 0.9,
                    end_valence: 0.8,
                    tension_points: vec![0.4, 0.7, 0.9, 0.6],
                }),
                narrative_structure: Some("three_act".to_string()),
            }),
        };

        let triples = mapper.map_to_rdf(&input);

        // Should include redemption theme
        assert!(triples.iter().any(|t|
            t.predicate == "media:hasTheme" &&
            t.object.contains("Redemption") &&
            t.confidence > 0.9
        ));

        // Should include hero's journey trope
        assert!(triples.iter().any(|t|
            t.predicate == "media:containsTrope" &&
            t.object.contains("HerosJourney")
        ));

        // Should detect hero journey narrative structure
        assert!(triples.iter().any(|t|
            t.predicate == "media:narrativeStructure" &&
            t.object == "media:HeroJourney"
        ));
    }

    #[test]
    fn test_motion_vector_to_genre() {
        let mapper = MetadataMapper::new("movie".to_string());

        let input = AnalysisInput {
            media_id: "321".to_string(),
            visual: Some(VisualAnalysis {
                color_palette: vec!["#ff0000".to_string()],
                contrast: 0.6,
                composition_style: None,
                motion_vectors: Some(vec![
                    MotionVector {
                        magnitude: 0.9,
                        direction: "forward".to_string(),
                        frequency: 0.8,
                    },
                    MotionVector {
                        magnitude: 0.85,
                        direction: "pan".to_string(),
                        frequency: 0.7,
                    },
                ]),
                lighting: None,
            }),
            audio: None,
            text: None,
        };

        let triples = mapper.map_to_rdf(&input);

        // High motion + high frequency should suggest Action
        assert!(triples.iter().any(|t|
            t.predicate == "media:suggestedGenre" &&
            t.object == "media:Action" &&
            t.confidence >= 0.7
        ));
    }

    #[test]
    fn test_conflict_resolution_same_predicate() {
        let mapper = MetadataMapper::new("movie".to_string());

        let input = AnalysisInput {
            media_id: "999".to_string(),
            visual: Some(VisualAnalysis {
                color_palette: vec!["#000000".to_string(), "#ff6b9d".to_string()],
                contrast: 0.7,
                composition_style: None,
                motion_vectors: None,
                lighting: None,
            }),
            audio: None,
            text: None,
        };

        let triples = mapper.map_to_rdf(&input);

        // Should resolve to highest confidence aesthetic
        let aesthetic_triples: Vec<_> = triples.iter()
            .filter(|t| t.predicate == "media:hasVisualAesthetic")
            .collect();

        // Should have exactly one primary aesthetic
        assert_eq!(aesthetic_triples.len(), 1);

        // May have alternative aesthetics
        let alternative_count = triples.iter()
            .filter(|t| t.predicate.contains("alternative"))
            .count();

        assert!(alternative_count <= 1);
    }

    #[test]
    fn test_turtle_serialization() {
        let mapper = MetadataMapper::new("movie".to_string());

        let triples = vec![
            RdfTriple {
                subject: "movie:123".to_string(),
                predicate: "media:hasVisualAesthetic".to_string(),
                object: "media:Noir".to_string(),
                confidence: 0.95,
                source: "visual".to_string(),
            },
            RdfTriple {
                subject: "movie:123".to_string(),
                predicate: "media:visualContrast".to_string(),
                object: "\"0.95\"^^xsd:decimal".to_string(),
                confidence: 1.0,
                source: "visual".to_string(),
            },
        ];

        let turtle = mapper.to_turtle(&triples);

        // Should contain prefix declarations
        assert!(turtle.contains("@prefix media:"));
        assert!(turtle.contains("@prefix xsd:"));

        // Should contain subject
        assert!(turtle.contains("movie:123"));

        // Should contain predicates and objects
        assert!(turtle.contains("media:hasVisualAesthetic"));
        assert!(turtle.contains("media:Noir"));
        assert!(turtle.contains("media:visualContrast"));
        assert!(turtle.contains("0.95"));
    }

    #[test]
    fn test_json_input_parsing() {
        let json_input = json!({
            "media_id": "test_001",
            "visual": {
                "color_palette": ["#000000", "#1a1a1a", "#2d2d2d"],
                "contrast": 0.88,
                "composition_style": "rule_of_thirds",
                "lighting": {
                    "key_light_ratio": 0.35,
                    "shadow_depth": 0.75,
                    "color_temperature": 2900
                }
            },
            "audio": {
                "tempo": 100.0,
                "key": "D minor",
                "dialogue_complexity": 0.72,
                "sound_design_score": 0.85
            },
            "text": {
                "themes": [
                    {"name": "justice", "confidence": 0.89},
                    {"name": "corruption", "confidence": 0.82}
                ],
                "tropes": [
                    {"name": "dark knight", "confidence": 0.91, "timestamp": 300.0}
                ],
                "emotional_arc": {
                    "start_valence": 0.4,
                    "peak_valence": 0.85,
                    "end_valence": 0.6,
                    "tension_points": [0.3, 0.6, 0.85, 0.7, 0.6]
                }
            }
        });

        let input: AnalysisInput = serde_json::from_value(json_input).unwrap();
        let mapper = MetadataMapper::new("movie".to_string());
        let triples = mapper.map_to_rdf(&input);

        // Should generate multiple triples from all modalities
        assert!(triples.len() >= 10);

        // Should include visual, audio, and text assertions
        assert!(triples.iter().any(|t| t.source == "visual_analysis"));
        assert!(triples.iter().any(|t| t.source == "audio_analysis"));
        assert!(triples.iter().any(|t| t.source == "theme_analysis"));
    }

    #[test]
    fn test_confidence_threshold_filtering() {
        let mapper = MetadataMapper::new("movie".to_string());

        let input = AnalysisInput {
            media_id: "low_conf".to_string(),
            visual: None,
            audio: None,
            text: Some(TextAnalysis {
                themes: vec![
                    ThemeDetection {
                        name: "high_confidence_theme".to_string(),
                        confidence: 0.92,
                    },
                    ThemeDetection {
                        name: "low_confidence_theme".to_string(),
                        confidence: 0.45,
                    },
                ],
                tropes: vec![],
                emotional_arc: None,
                narrative_structure: None,
            }),
        };

        let triples = mapper.map_to_rdf(&input);

        // Should include high confidence theme
        assert!(triples.iter().any(|t|
            t.object.contains("HighConfidenceTheme")
        ));

        // Should exclude low confidence theme
        assert!(!triples.iter().any(|t|
            t.object.contains("LowConfidenceTheme")
        ));
    }

    #[test]
    fn test_emotional_arc_narrative_structures() {
        let mapper = MetadataMapper::new("movie".to_string());

        // Test tragedy arc
        let tragedy_input = AnalysisInput {
            media_id: "tragedy".to_string(),
            visual: None,
            audio: None,
            text: Some(TextAnalysis {
                themes: vec![],
                tropes: vec![],
                emotional_arc: Some(EmotionalArc {
                    start_valence: 0.8,
                    peak_valence: 0.9,
                    end_valence: 0.2,
                    tension_points: vec![0.8, 0.6, 0.4, 0.2],
                }),
                narrative_structure: None,
            }),
        };

        let triples = mapper.map_to_rdf(&tragedy_input);
        assert!(triples.iter().any(|t|
            t.object == "media:Tragedy"
        ));

        // Test hero journey arc
        let hero_input = AnalysisInput {
            media_id: "hero".to_string(),
            visual: None,
            audio: None,
            text: Some(TextAnalysis {
                themes: vec![],
                tropes: vec![],
                emotional_arc: Some(EmotionalArc {
                    start_valence: 0.3,
                    peak_valence: 0.9,
                    end_valence: 0.85,
                    tension_points: vec![0.3, 0.5, 0.9, 0.85],
                }),
                narrative_structure: None,
            }),
        };

        let triples = mapper.map_to_rdf(&hero_input);
        assert!(triples.iter().any(|t|
            t.object == "media:HeroJourney"
        ));
    }
}
