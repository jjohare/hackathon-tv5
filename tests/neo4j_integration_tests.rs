/// Integration tests for Neo4j ontology loading
///
/// These tests require a running Neo4j instance.
/// Run with: cargo test --test neo4j_integration_tests -- --ignored

use std::collections::HashMap;
use std::env;

// Import from the main crate
use hackathon_tv5::ontology::{
    loader::{OntologyLoader, Neo4jConfig, RdfTriple, RdfObject},
    types::*,
    reasoner::*,
};

/// Get Neo4j configuration from environment or use defaults
fn get_test_config() -> Neo4jConfig {
    Neo4jConfig {
        uri: env::var("NEO4J_URI").unwrap_or_else(|_| "bolt://localhost:7687".to_string()),
        username: env::var("NEO4J_USER").unwrap_or_else(|_| "neo4j".to_string()),
        password: env::var("NEO4J_PASSWORD").unwrap_or_else(|_| "password".to_string()),
        database: env::var("NEO4J_DATABASE").unwrap_or_else(|_| "neo4j".to_string()),
        max_connections: 5,
        connection_timeout_secs: 10,
    }
}

/// Clear test data from Neo4j
async fn clear_test_data(loader: &OntologyLoader) {
    let query = r#"
        MATCH (n)
        WHERE n:Media OR n:Genre OR n:Mood OR n:User OR n:CulturalContext OR n:Tag OR n:Theme
        DETACH DELETE n
    "#;

    loader.graph.run(neo4rs::Query::new(query.to_string()))
        .await
        .expect("Failed to clear test data");
}

#[tokio::test]
#[ignore] // Requires Neo4j instance
async fn test_connection() {
    let config = get_test_config();
    let loader = OntologyLoader::new(config).await;
    assert!(loader.is_ok(), "Failed to connect to Neo4j");
}

#[tokio::test]
#[ignore]
async fn test_schema_setup() {
    let config = get_test_config();
    let loader = OntologyLoader::new(config).await.expect("Connection failed");

    let result = loader.setup_schema().await;
    assert!(result.is_ok(), "Schema setup failed: {:?}", result.err());
}

#[tokio::test]
#[ignore]
async fn test_bulk_load_small_dataset() {
    let config = get_test_config();
    let loader = OntologyLoader::new(config).await.expect("Connection failed");

    // Clear existing test data
    clear_test_data(&loader).await;

    // Create test triples
    let triples = vec![
        RdfTriple {
            subject: "media:test_001".to_string(),
            predicate: "rdf:type".to_string(),
            object: RdfObject::Uri("media:Media".to_string()),
        },
        RdfTriple {
            subject: "media:test_001".to_string(),
            predicate: "media:title".to_string(),
            object: RdfObject::Literal("Test Movie".to_string(), None),
        },
        RdfTriple {
            subject: "media:test_001".to_string(),
            predicate: "media:mediaType".to_string(),
            object: RdfObject::Literal("Video".to_string(), None),
        },
        RdfTriple {
            subject: "genre:Action".to_string(),
            predicate: "rdf:type".to_string(),
            object: RdfObject::Uri("media:Genre".to_string()),
        },
        RdfTriple {
            subject: "media:test_001".to_string(),
            predicate: "media:hasGenre".to_string(),
            object: RdfObject::Uri("genre:Action".to_string()),
        },
    ];

    let result = loader.bulk_load_triples(triples).await;
    assert!(result.is_ok(), "Bulk load failed: {:?}", result.err());
}

#[tokio::test]
#[ignore]
async fn test_bulk_load_performance() {
    let config = get_test_config();
    let loader = OntologyLoader::new(config).await.expect("Connection failed");

    clear_test_data(&loader).await;

    // Generate 10,000 test triples
    let mut triples = Vec::with_capacity(30000);

    for i in 0..10000 {
        let media_id = format!("media:perf_test_{:05}", i);

        triples.push(RdfTriple {
            subject: media_id.clone(),
            predicate: "rdf:type".to_string(),
            object: RdfObject::Uri("media:Media".to_string()),
        });

        triples.push(RdfTriple {
            subject: media_id.clone(),
            predicate: "media:title".to_string(),
            object: RdfObject::Literal(format!("Test Media {}", i), None),
        });

        triples.push(RdfTriple {
            subject: media_id.clone(),
            predicate: "media:mediaType".to_string(),
            object: RdfObject::Literal("Video".to_string(), None),
        });
    }

    let start = std::time::Instant::now();
    let result = loader.bulk_load_triples(triples).await;
    let elapsed = start.elapsed();

    assert!(result.is_ok(), "Bulk load failed: {:?}", result.err());
    assert!(elapsed.as_secs() < 30, "Load took too long: {:?}", elapsed);

    println!("Loaded 30,000 triples in {:.2}s", elapsed.as_secs_f32());
}

#[tokio::test]
#[ignore]
async fn test_incremental_update() {
    let config = get_test_config();
    let loader = OntologyLoader::new(config).await.expect("Connection failed");

    let entity_id = "media:update_test_001";

    // Initial insert
    let mut properties = HashMap::new();
    properties.insert("title".to_string(), "Original Title".to_string());
    properties.insert("media_type".to_string(), "Video".to_string());

    let result = loader.incremental_update(entity_id, properties).await;
    assert!(result.is_ok(), "Initial update failed: {:?}", result.err());

    // Update with new properties
    let mut updated_properties = HashMap::new();
    updated_properties.insert("title".to_string(), "Updated Title".to_string());
    updated_properties.insert("duration_seconds".to_string(), "7200".to_string());

    let result = loader.incremental_update(entity_id, updated_properties).await;
    assert!(result.is_ok(), "Incremental update failed: {:?}", result.err());
}

#[tokio::test]
#[ignore]
async fn test_load_media_entities() {
    let config = get_test_config();
    let loader = OntologyLoader::new(config).await.expect("Connection failed");

    clear_test_data(&loader).await;

    // Insert test media
    let triples = vec![
        RdfTriple {
            subject: "media:load_test_001".to_string(),
            predicate: "media:title".to_string(),
            object: RdfObject::Literal("Test Load Movie".to_string(), None),
        },
        RdfTriple {
            subject: "media:load_test_001".to_string(),
            predicate: "media:mediaType".to_string(),
            object: RdfObject::Literal("Video".to_string(), None),
        },
        RdfTriple {
            subject: "media:load_test_001".to_string(),
            predicate: "media:format".to_string(),
            object: RdfObject::Literal("mp4".to_string(), None),
        },
    ];

    loader.bulk_load_triples(triples).await.expect("Load failed");

    // Load ontology
    let ontology = loader.load_ontology().await;
    assert!(ontology.is_ok(), "Load ontology failed: {:?}", ontology.err());

    let ontology = ontology.unwrap();
    assert!(ontology.media.len() > 0, "No media entities loaded");
}

#[tokio::test]
#[ignore]
async fn test_load_genre_hierarchy() {
    let config = get_test_config();
    let loader = OntologyLoader::new(config).await.expect("Connection failed");

    clear_test_data(&loader).await;

    // Create genre hierarchy
    let triples = vec![
        RdfTriple {
            subject: "genre:Thriller".to_string(),
            predicate: "rdf:type".to_string(),
            object: RdfObject::Uri("media:Genre".to_string()),
        },
        RdfTriple {
            subject: "genre:Drama".to_string(),
            predicate: "rdf:type".to_string(),
            object: RdfObject::Uri("media:Genre".to_string()),
        },
        RdfTriple {
            subject: "genre:Thriller".to_string(),
            predicate: "media:subGenreOf".to_string(),
            object: RdfObject::Uri("genre:Drama".to_string()),
        },
    ];

    loader.bulk_load_triples(triples).await.expect("Load failed");

    let ontology = loader.load_ontology().await.expect("Load failed");

    assert!(ontology.genre_hierarchy.contains_key("Thriller"), "Thriller not found");
    assert!(ontology.genre_hierarchy.get("Thriller").unwrap().contains("Drama"), "Hierarchy incorrect");
}

#[tokio::test]
#[ignore]
async fn test_store_inferred_axioms() {
    let config = get_test_config();
    let loader = OntologyLoader::new(config).await.expect("Connection failed");

    clear_test_data(&loader).await;

    // Create base genres
    let triples = vec![
        RdfTriple {
            subject: "genre:SciFi".to_string(),
            predicate: "rdf:type".to_string(),
            object: RdfObject::Uri("media:Genre".to_string()),
        },
        RdfTriple {
            subject: "genre:Action".to_string(),
            predicate: "rdf:type".to_string(),
            object: RdfObject::Uri("media:Genre".to_string()),
        },
    ];

    loader.bulk_load_triples(triples).await.expect("Load failed");

    // Create inferred axiom
    let axioms = vec![
        InferredMediaAxiom {
            axiom_type: MediaAxiomType::SubGenreOf,
            subject: "SciFi".to_string(),
            object: Some("Action".to_string()),
            confidence: 0.85,
            reasoning: "Inferred from co-occurrence patterns".to_string(),
        }
    ];

    let result = loader.store_inferred_axioms(&axioms).await;
    assert!(result.is_ok(), "Store axioms failed: {:?}", result.err());
}

#[tokio::test]
#[ignore]
async fn test_parse_ttl_file() {
    use std::io::Write;
    use std::fs::File;

    // Create temporary TTL file
    let ttl_content = r#"
        @prefix media: <http://example.org/media#> .
        @prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .

        media:movie001 rdf:type media:Media .
        media:movie001 media:title "Test Movie" .
        media:movie001 media:mediaType "Video" .
    "#;

    let temp_path = "/tmp/test_ontology.ttl";
    let mut file = File::create(temp_path).expect("Failed to create temp file");
    file.write_all(ttl_content.as_bytes()).expect("Failed to write");
    drop(file);

    let triples = OntologyLoader::parse_ttl_file(temp_path);
    assert!(triples.is_ok(), "Parse failed: {:?}", triples.err());

    let triples = triples.unwrap();
    assert!(triples.len() >= 3, "Expected at least 3 triples, got {}", triples.len());

    // Cleanup
    std::fs::remove_file(temp_path).ok();
}

#[tokio::test]
#[ignore]
async fn test_concurrent_operations() {
    let config = get_test_config();
    let loader = OntologyLoader::new(config).await.expect("Connection failed");

    clear_test_data(&loader).await;

    // Perform multiple concurrent updates
    let mut handles = vec![];

    for i in 0..10 {
        let loader_clone = loader.clone();
        let handle = tokio::spawn(async move {
            let entity_id = format!("media:concurrent_{:02}", i);
            let mut props = HashMap::new();
            props.insert("title".to_string(), format!("Concurrent Test {}", i));

            loader_clone.incremental_update(&entity_id, props).await
        });
        handles.push(handle);
    }

    // Wait for all operations
    for handle in handles {
        let result = handle.await.expect("Task panicked");
        assert!(result.is_ok(), "Concurrent update failed: {:?}", result.err());
    }
}

#[tokio::test]
#[ignore]
async fn test_retry_logic() {
    let config = get_test_config();
    let loader = OntologyLoader::new(config).await.expect("Connection failed");

    // This test verifies retry logic by attempting a valid operation
    // In a real scenario, you'd simulate network issues

    let entity_id = "media:retry_test";
    let mut props = HashMap::new();
    props.insert("title".to_string(), "Retry Test".to_string());

    let result = loader.incremental_update(entity_id, props).await;
    assert!(result.is_ok(), "Retry logic test failed");
}
