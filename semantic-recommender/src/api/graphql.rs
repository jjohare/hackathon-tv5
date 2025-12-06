use async_graphql::{Context, EmptyMutation, EmptySubscription, Object, Result, Schema};
use async_graphql_axum::{GraphQL, GraphQLRequest, GraphQLResponse};
use axum::{extract::State, response::IntoResponse, Extension};
use std::sync::Arc;

use crate::models::*;
use crate::recommendation::RecommendationEngine;

// Simplified GraphQL handler - comment out for now until type issues resolved

pub struct QueryRoot;

#[Object]
impl QueryRoot {
    /// Search for media content using semantic similarity
    async fn search_media(
        &self,
        ctx: &Context<'_>,
        query: String,
        limit: Option<i32>,
        genres: Option<Vec<String>>,
        min_rating: Option<f32>,
        language: Option<String>,
    ) -> Result<MediaSearchResponse> {
        let engine = ctx.data::<Arc<RecommendationEngine>>()?;

        let request = MediaSearchRequest {
            query,
            filters: Some(SearchFilters {
                genres,
                min_rating,
                language,
                year_range: None,
                content_type: None,
            }),
            limit: limit.map(|l| l as usize),
            offset: None,
        };

        engine
            .search(&request)
            .await
            .map_err(|e| async_graphql::Error::new(e.to_string()))
    }

    /// Get personalized recommendations for a user
    async fn recommendations(
        &self,
        ctx: &Context<'_>,
        user_id: String,
        limit: Option<i32>,
        explain: Option<bool>,
    ) -> Result<RecommendationResponse> {
        let engine = ctx.data::<Arc<RecommendationEngine>>()?;

        let params = RecommendationParams {
            limit: limit.map(|l| l as usize),
            explain,
            content_type: None,
        };

        engine
            .recommend(&user_id, &params)
            .await
            .map_err(|e| async_graphql::Error::new(e.to_string()))
    }

    /// Get a specific media item by ID
    async fn media_item(&self, _ctx: &Context<'_>, id: String) -> Result<Option<MediaItem>> {
        // Mock implementation - in production, this would query the database
        if id == "film_123" {
            Ok(Some(MediaItem {
                id: "film_123".to_string(),
                title: "À bout de souffle".to_string(),
                similarity_score: 1.0,
                explanation: "Direct lookup".to_string(),
                metadata: MediaMetadata {
                    genres: vec!["drama".to_string(), "crime".to_string()],
                    year: 1960,
                    language: "fr".to_string(),
                    rating: Some(8.2),
                    duration_minutes: Some(90),
                    director: Some("Jean-Luc Godard".to_string()),
                    cast: Some(vec!["Jean-Paul Belmondo".to_string()]),
                },
            }))
        } else {
            Ok(None)
        }
    }
}

pub type MediaSchema = Schema<QueryRoot, EmptyMutation, EmptySubscription>;

pub fn create_schema(engine: Arc<RecommendationEngine>) -> MediaSchema {
    Schema::build(QueryRoot, EmptyMutation, EmptySubscription)
        .data(engine)
        .finish()
}

pub async fn create_graphql_handler(
    State(engine): State<Arc<RecommendationEngine>>,
    req: GraphQLRequest,
) -> GraphQLResponse {
    let schema = create_schema(engine);
    let response = schema.execute(req.into_inner()).await;
    GraphQLResponse::from(response)
}
