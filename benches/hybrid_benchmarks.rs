use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId, Throughput};
use recommendation_engine::storage::{HybridStorageCoordinator, MilvusClient, Neo4jClient};
use tokio::runtime::Runtime;
use std::sync::Arc;

mod fixtures;
use fixtures::{media_generator, query_generator};

fn setup_runtime() -> Runtime {
    tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .unwrap()
}

fn benchmark_vector_search_scaling(c: &mut Criterion) {
    let rt = setup_runtime();
    let mut group = c.benchmark_group("vector_search_scaling");

    for size in [1_000, 10_000, 100_000, 1_000_000].iter() {
        let coordinator = rt.block_on(async {
            let env = setup_test_coordinator().await.unwrap();
            insert_test_data(&env.coordinator, *size).await.unwrap();
            Arc::new(env.coordinator)
        });

        group.throughput(Throughput::Elements(*size as u64));
        group.bench_with_input(
            BenchmarkId::from_parameter(size),
            size,
            |b, _| {
                b.to_async(&rt).iter(|| async {
                    let query = query_generator::create_random_query();
                    let _ = coordinator.search_with_context(black_box(&query)).await.unwrap();
                });
            }
        );
    }

    group.finish();
}

fn benchmark_hybrid_vs_vector_only(c: &mut Criterion) {
    let rt = setup_runtime();
    let mut group = c.benchmark_group("hybrid_vs_vector_only");

    let (hybrid_coord, vector_only_coord) = rt.block_on(async {
        // Setup hybrid coordinator
        let env_hybrid = setup_test_coordinator().await.unwrap();
        insert_test_data(&env_hybrid.coordinator, 10_000).await.unwrap();

        // Setup vector-only coordinator (no Neo4j/AgentDB)
        let env_vector = setup_vector_only_coordinator().await.unwrap();
        insert_test_data(&env_vector.coordinator, 10_000).await.unwrap();

        (Arc::new(env_hybrid.coordinator), Arc::new(env_vector.coordinator))
    });

    group.bench_function("hybrid", |b| {
        b.to_async(&rt).iter(|| async {
            let query = query_generator::create_random_query();
            let _ = hybrid_coord.search_with_context(black_box(&query)).await.unwrap();
        });
    });

    group.bench_function("vector_only", |b| {
        b.to_async(&rt).iter(|| async {
            let query = query_generator::create_random_query();
            let _ = vector_only_coord.search_with_context(black_box(&query)).await.unwrap();
        });
    });

    group.finish();
}

fn benchmark_cache_effectiveness(c: &mut Criterion) {
    let rt = setup_runtime();
    let mut group = c.benchmark_group("cache_effectiveness");

    let coordinator = rt.block_on(async {
        let env = setup_test_coordinator().await.unwrap();
        insert_test_data(&env.coordinator, 10_000).await.unwrap();
        Arc::new(env.coordinator)
    });

    let query = query_generator::create_random_query();

    // Cold cache
    group.bench_function("cold_cache", |b| {
        b.to_async(&rt).iter(|| async {
            coordinator.invalidate_cache().await.unwrap();
            let _ = coordinator.search_with_context(black_box(&query)).await.unwrap();
        });
    });

    // Warm cache
    rt.block_on(async {
        let _ = coordinator.search_with_context(&query).await.unwrap();
    });

    group.bench_function("warm_cache", |b| {
        b.to_async(&rt).iter(|| async {
            let _ = coordinator.search_with_context(black_box(&query)).await.unwrap();
        });
    });

    group.finish();
}

fn benchmark_batch_vs_single_ingest(c: &mut Criterion) {
    let rt = setup_runtime();
    let mut group = c.benchmark_group("batch_vs_single_ingest");

    for batch_size in [1, 10, 100, 1000].iter() {
        let coordinator = rt.block_on(async {
            let env = setup_test_coordinator().await.unwrap();
            Arc::new(env.coordinator)
        });

        let media_items = media_generator::generate_media_content(*batch_size);

        group.throughput(Throughput::Elements(*batch_size as u64));
        group.bench_with_input(
            BenchmarkId::from_parameter(batch_size),
            batch_size,
            |b, _| {
                b.to_async(&rt).iter(|| async {
                    for item in &media_items {
                        let _ = coordinator.ingest_content(black_box(item)).await.unwrap();
                    }
                });
            }
        );
    }

    // Batch ingest
    let coordinator = rt.block_on(async {
        let env = setup_test_coordinator().await.unwrap();
        Arc::new(env.coordinator)
    });

    let batch_items = media_generator::generate_media_content(1000);

    group.bench_function("batch_1000", |b| {
        b.to_async(&rt).iter(|| async {
            let _ = coordinator.ingest_batch(black_box(&batch_items)).await.unwrap();
        });
    });

    group.finish();
}

fn benchmark_filter_strategies(c: &mut Criterion) {
    let rt = setup_runtime();
    let mut group = c.benchmark_group("filter_strategies");

    let coordinator = rt.block_on(async {
        let env = setup_test_coordinator().await.unwrap();

        // Insert data with various genres
        let mut media_items = Vec::new();
        for i in 0..10_000 {
            let genre = match i % 5 {
                0 => "SciFi",
                1 => "Action",
                2 => "Romance",
                3 => "Comedy",
                _ => "Drama",
            };

            let mut item = media_generator::generate_single_media(&format!("movie_{}", i));
            item.genres = vec![genre.to_string()];
            media_items.push(item);
        }

        coordinator.ingest_batch(&media_items).await.unwrap();
        Arc::new(env.coordinator)
    });

    // No filter
    group.bench_function("no_filter", |b| {
        b.to_async(&rt).iter(|| async {
            let query = query_generator::create_query_without_filter();
            let _ = coordinator.search_with_context(black_box(&query)).await.unwrap();
        });
    });

    // Single filter (selective - 20% of data)
    group.bench_function("filter_selective", |b| {
        b.to_async(&rt).iter(|| async {
            let mut query = query_generator::create_random_query();
            query.filters.insert("genre".to_string(), "SciFi".to_string());
            let _ = coordinator.search_with_context(black_box(&query)).await.unwrap();
        });
    });

    // Multiple filters (very selective - 1% of data)
    group.bench_function("filter_multi", |b| {
        b.to_async(&rt).iter(|| async {
            let mut query = query_generator::create_random_query();
            query.filters.insert("genre".to_string(), "SciFi".to_string());
            query.filters.insert("year".to_string(), "2020".to_string());
            let _ = coordinator.search_with_context(black_box(&query)).await.unwrap();
        });
    });

    group.finish();
}

fn benchmark_concurrent_queries(c: &mut Criterion) {
    let rt = setup_runtime();
    let mut group = c.benchmark_group("concurrent_queries");

    let coordinator = rt.block_on(async {
        let env = setup_test_coordinator().await.unwrap();
        insert_test_data(&env.coordinator, 10_000).await.unwrap();
        Arc::new(env.coordinator)
    });

    for concurrency in [1, 10, 50, 100].iter() {
        group.bench_with_input(
            BenchmarkId::from_parameter(concurrency),
            concurrency,
            |b, &concurrency| {
                b.to_async(&rt).iter(|| async {
                    let mut handles = Vec::new();

                    for _ in 0..concurrency {
                        let coord = coordinator.clone();
                        handles.push(tokio::spawn(async move {
                            let query = query_generator::create_random_query();
                            coord.search_with_context(&query).await.unwrap()
                        }));
                    }

                    for handle in handles {
                        let _ = handle.await.unwrap();
                    }
                });
            }
        );
    }

    group.finish();
}

fn benchmark_agentdb_reranking(c: &mut Criterion) {
    let rt = setup_runtime();
    let mut group = c.benchmark_group("agentdb_reranking");

    let coordinator = rt.block_on(async {
        let env = setup_test_coordinator().await.unwrap();
        insert_test_data(&env.coordinator, 10_000).await.unwrap();

        // Setup user policy
        let policy = Policy {
            agent_id: "bench_user".to_string(),
            preferences: vec![
                ("genre_scifi".to_string(), 0.9),
                ("genre_action".to_string(), 0.7),
            ].into_iter().collect(),
        };
        env.coordinator.agentdb.set_policy("bench_user", &policy).await.unwrap();

        Arc::new(env.coordinator)
    });

    // Without reranking
    group.bench_function("no_reranking", |b| {
        b.to_async(&rt).iter(|| async {
            let query = query_generator::create_query_without_user();
            let _ = coordinator.search_with_context(black_box(&query)).await.unwrap();
        });
    });

    // With AgentDB reranking
    group.bench_function("with_reranking", |b| {
        b.to_async(&rt).iter(|| async {
            let mut query = query_generator::create_random_query();
            query.user_id = Some("bench_user".to_string());
            let _ = coordinator.search_with_context(black_box(&query)).await.unwrap();
        });
    });

    group.finish();
}

fn benchmark_embedding_dimensions(c: &mut Criterion) {
    let rt = setup_runtime();
    let mut group = c.benchmark_group("embedding_dimensions");

    for dim in [128, 256, 512, 768, 1024].iter() {
        let coordinator = rt.block_on(async {
            let env = setup_test_coordinator_with_dim(*dim).await.unwrap();

            let mut media_items = Vec::new();
            for i in 0..1000 {
                let mut item = media_generator::generate_single_media(&format!("movie_{}", i));
                item.embedding = vec![0.5; *dim];
                media_items.push(item);
            }

            coordinator.ingest_batch(&media_items).await.unwrap();
            Arc::new(env.coordinator)
        });

        group.bench_with_input(
            BenchmarkId::from_parameter(dim),
            dim,
            |b, &dim| {
                b.to_async(&rt).iter(|| async {
                    let mut query = query_generator::create_random_query();
                    query.embedding = vec![0.5; dim];
                    let _ = coordinator.search_with_context(black_box(&query)).await.unwrap();
                });
            }
        );
    }

    group.finish();
}

criterion_group!(
    benches,
    benchmark_vector_search_scaling,
    benchmark_hybrid_vs_vector_only,
    benchmark_cache_effectiveness,
    benchmark_batch_vs_single_ingest,
    benchmark_filter_strategies,
    benchmark_concurrent_queries,
    benchmark_agentdb_reranking,
    benchmark_embedding_dimensions
);

criterion_main!(benches);
