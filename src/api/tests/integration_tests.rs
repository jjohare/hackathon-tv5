use axum::{
    body::Body,
    http::{Request, StatusCode},
};
use serde_json::json;
use tower::ServiceExt;

#[tokio::test]
async fn test_health_endpoint() {
    let app = media_gateway_api::create_app().await;

    let response = app
        .oneshot(
            Request::builder()
                .uri("/health")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
}

#[tokio::test]
async fn test_search_endpoint() {
    let app = media_gateway_api::create_app().await;

    let request_body = json!({
        "query": "French films",
        "limit": 5
    });

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/api/v1/search")
                .header("content-type", "application/json")
                .body(Body::from(serde_json::to_string(&request_body).unwrap()))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
}

#[tokio::test]
async fn test_mcp_manifest() {
    let app = media_gateway_api::create_app().await;

    let response = app
        .oneshot(
            Request::builder()
                .uri("/api/v1/mcp/manifest")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);

    let body = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .unwrap();
    let manifest: serde_json::Value = serde_json::from_slice(&body).unwrap();

    assert_eq!(manifest["name"], "tv5-media-gateway");
    assert!(manifest["tools"].is_array());
}
