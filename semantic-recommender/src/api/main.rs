use media_gateway_api::create_app;
use std::net::SocketAddr;
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize tracing
    tracing_subscriber::registry()
        .with(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "media_gateway_api=info,tower_http=info".into()),
        )
        .with(tracing_subscriber::fmt::layer())
        .init();

    tracing::info!("🚀 Starting TV5 Media Gateway API v{}", env!("CARGO_PKG_VERSION"));
    tracing::info!("📊 Initializing GPU-accelerated recommendation engine...");

    let app = create_app().await;

    let addr = SocketAddr::from(([0, 0, 0, 0], 3000));
    tracing::info!("✅ API server ready");
    tracing::info!("🌐 Listening on {}", addr);
    tracing::info!("📖 OpenAPI docs: http://{}/swagger-ui", addr);
    tracing::info!("🔌 MCP manifest: http://{}/api/v1/mcp/manifest", addr);
    tracing::info!("💚 Health check: http://{}/health", addr);
    tracing::info!("📈 Metrics: http://{}/metrics", addr);

    let listener = tokio::net::TcpListener::bind(addr).await?;
    axum::serve(listener, app).await?;

    Ok(())
}

