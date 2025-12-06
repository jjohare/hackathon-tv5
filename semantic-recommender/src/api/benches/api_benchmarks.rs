use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use media_gateway_api::{models::*, recommendation::RecommendationEngine};
use tokio::runtime::Runtime;

fn benchmark_search(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let engine = rt.block_on(async { RecommendationEngine::new().await });

    let mut group = c.benchmark_group("search_operations");
    group.throughput(Throughput::Elements(1));

    let test_queries = vec![
        "French noir films",
        "documentaries about climate",
        "romantic comedies",
        "action thriller with complex plot",
        "philosophical drama",
    ];

    for query in test_queries {
        group.bench_with_input(BenchmarkId::from_parameter(query), &query, |b, &query| {
            b.to_async(&rt).iter(|| async {
                let req = MediaSearchRequest {
                    query: query.to_string(),
                    filters: None,
                    limit: Some(10),
                    offset: None,
                };
                black_box(engine.search(&req).await.unwrap())
            });
        });
    }

    group.finish();
}

fn benchmark_recommendations(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let engine = rt.block_on(async { RecommendationEngine::new().await });

    let mut group = c.benchmark_group("recommendation_operations");
    group.throughput(Throughput::Elements(1));

    for limit in [5, 10, 25, 50] {
        group.bench_with_input(
            BenchmarkId::from_parameter(format!("limit_{}", limit)),
            &limit,
            |b, &limit| {
                b.to_async(&rt).iter(|| async {
                    let params = RecommendationParams {
                        limit: Some(limit),
                        explain: Some(true),
                        content_type: None,
                    };
                    black_box(engine.recommend("user_123", &params).await.unwrap())
                });
            },
        );
    }

    group.finish();
}

fn benchmark_concurrent_requests(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let engine = rt.block_on(async { RecommendationEngine::new().await });

    let mut group = c.benchmark_group("concurrent_operations");

    for concurrent in [10, 50, 100] {
        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{}_concurrent", concurrent)),
            &concurrent,
            |b, &concurrent| {
                b.to_async(&rt).iter(|| async {
                    let mut handles = vec![];

                    for i in 0..concurrent {
                        let engine = &engine;
                        let handle = tokio::spawn(async move {
                            let req = MediaSearchRequest {
                                query: format!("query_{}", i),
                                filters: None,
                                limit: Some(10),
                                offset: None,
                            };
                            engine.search(&req).await.unwrap()
                        });
                        handles.push(handle);
                    }

                    for handle in handles {
                        black_box(handle.await.unwrap());
                    }
                });
            },
        );
    }

    group.finish();
}

criterion_group!(
    benches,
    benchmark_search,
    benchmark_recommendations,
    benchmark_concurrent_requests
);
criterion_main!(benches);
