use axum::response::Json;
use serde::{Deserialize, Serialize};
use utoipa::ToSchema;

/// MCP (Model Context Protocol) manifest for AI agent integration
#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
pub struct MCPManifest {
    /// Service name
    pub name: String,

    /// API version
    pub version: String,

    /// Service description
    pub description: String,

    /// Available tools for AI agents
    pub tools: Vec<MCPTool>,

    /// Supported authentication methods
    pub auth: Vec<String>,

    /// Rate limit information
    pub rate_limits: RateLimitInfo,
}

#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
pub struct MCPTool {
    /// Tool identifier
    pub name: String,

    /// Human-readable description
    pub description: String,

    /// JSON Schema for input parameters
    pub input_schema: serde_json::Value,

    /// JSON Schema for output format
    pub output_schema: serde_json::Value,

    /// Example usage patterns
    pub examples: Vec<ToolExample>,

    /// Average response time in milliseconds
    pub avg_response_time_ms: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
pub struct ToolExample {
    /// Example input
    pub input: serde_json::Value,

    /// Expected output
    pub output: serde_json::Value,

    /// Description of what this example demonstrates
    pub description: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
pub struct RateLimitInfo {
    pub requests_per_minute: u32,
    pub requests_per_hour: u32,
    pub burst_size: u32,
}

pub fn generate_mcp_manifest() -> MCPManifest {
    MCPManifest {
        name: "tv5-media-gateway".to_string(),
        version: "1.0.0".to_string(),
        description: "GPU-accelerated semantic media recommendation with vector search".to_string(),
        tools: vec![
            MCPTool {
                name: "search_media".to_string(),
                description: "Search for media content using semantic similarity. Uses GPU-accelerated vector embeddings for natural language understanding.".to_string(),
                input_schema: serde_json::json!({
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Natural language search query",
                            "minLength": 1,
                            "maxLength": 1000
                        },
                        "filters": {
                            "type": "object",
                            "properties": {
                                "genres": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                    "description": "Filter by genre tags"
                                },
                                "min_rating": {
                                    "type": "number",
                                    "minimum": 0.0,
                                    "maximum": 10.0,
                                    "description": "Minimum content rating"
                                },
                                "language": {
                                    "type": "string",
                                    "pattern": "^[a-z]{2}$",
                                    "description": "ISO 639-1 language code"
                                },
                                "year_range": {
                                    "type": "array",
                                    "items": {"type": "integer"},
                                    "minItems": 2,
                                    "maxItems": 2,
                                    "description": "Year range [start, end]"
                                },
                                "content_type": {
                                    "type": "string",
                                    "enum": ["movie", "series", "documentary", "short"]
                                }
                            }
                        },
                        "limit": {
                            "type": "integer",
                            "minimum": 1,
                            "maximum": 100,
                            "default": 10,
                            "description": "Maximum number of results"
                        },
                        "offset": {
                            "type": "integer",
                            "minimum": 0,
                            "default": 0,
                            "description": "Pagination offset"
                        }
                    },
                    "required": ["query"]
                }),
                output_schema: serde_json::json!({
                    "type": "object",
                    "properties": {
                        "data": {
                            "type": "object",
                            "properties": {
                                "results": {
                                    "type": "array",
                                    "items": {
                                        "type": "object",
                                        "properties": {
                                            "id": {"type": "string"},
                                            "title": {"type": "string"},
                                            "similarity_score": {
                                                "type": "number",
                                                "minimum": 0.0,
                                                "maximum": 1.0
                                            },
                                            "explanation": {"type": "string"},
                                            "metadata": {
                                                "type": "object",
                                                "properties": {
                                                    "genres": {"type": "array", "items": {"type": "string"}},
                                                    "year": {"type": "integer"},
                                                    "language": {"type": "string"},
                                                    "rating": {"type": "number"}
                                                }
                                            }
                                        }
                                    }
                                },
                                "total": {"type": "integer"},
                                "query_time_ms": {"type": "integer"}
                            }
                        },
                        "_links": {
                            "type": "object",
                            "description": "HATEOAS navigation links"
                        }
                    }
                }),
                examples: vec![
                    ToolExample {
                        input: serde_json::json!({
                            "query": "French noir films with existential themes",
                            "filters": {
                                "language": "fr",
                                "min_rating": 7.0
                            },
                            "limit": 5
                        }),
                        output: serde_json::json!({
                            "data": {
                                "results": [
                                    {
                                        "id": "film_123",
                                        "title": "À bout de souffle",
                                        "similarity_score": 0.92,
                                        "explanation": "Classic French New Wave noir with existential undertones and philosophical dialogue",
                                        "metadata": {
                                            "genres": ["drama", "crime", "noir"],
                                            "year": 1960,
                                            "language": "fr",
                                            "rating": 8.2
                                        }
                                    }
                                ],
                                "total": 1,
                                "query_time_ms": 8
                            }
                        }),
                        description: "Semantic search for French noir films".to_string()
                    },
                    ToolExample {
                        input: serde_json::json!({
                            "query": "documentaries about climate change",
                            "filters": {
                                "content_type": "documentary",
                                "year_range": [2018, 2024]
                            },
                            "limit": 3
                        }),
                        output: serde_json::json!({
                            "data": {
                                "results": [
                                    {
                                        "id": "doc_456",
                                        "title": "Our Planet",
                                        "similarity_score": 0.88,
                                        "explanation": "Comprehensive documentary series exploring climate impact on ecosystems",
                                        "metadata": {
                                            "genres": ["documentary", "nature"],
                                            "year": 2019,
                                            "language": "en",
                                            "rating": 9.3
                                        }
                                    }
                                ],
                                "total": 1,
                                "query_time_ms": 6
                            }
                        }),
                        description: "Finding recent climate documentaries".to_string()
                    }
                ],
                avg_response_time_ms: 8,
            },
            MCPTool {
                name: "get_recommendations".to_string(),
                description: "Get personalized content recommendations based on user preferences and viewing history. Uses collaborative filtering and content-based algorithms.".to_string(),
                input_schema: serde_json::json!({
                    "type": "object",
                    "properties": {
                        "user_id": {
                            "type": "string",
                            "description": "User identifier",
                            "pattern": "^[a-zA-Z0-9_-]+$"
                        },
                        "limit": {
                            "type": "integer",
                            "minimum": 1,
                            "maximum": 50,
                            "default": 10,
                            "description": "Number of recommendations"
                        },
                        "explain": {
                            "type": "boolean",
                            "default": false,
                            "description": "Include reasoning for recommendations"
                        },
                        "content_type": {
                            "type": "string",
                            "enum": ["movie", "series", "documentary", "short"],
                            "description": "Filter by content type"
                        }
                    },
                    "required": ["user_id"]
                }),
                output_schema: serde_json::json!({
                    "type": "object",
                    "properties": {
                        "@context": {
                            "type": "object",
                            "description": "JSON-LD context for semantic web integration"
                        },
                        "@type": {"type": "string"},
                        "data": {
                            "type": "object",
                            "properties": {
                                "user_id": {"type": "string"},
                                "recommendations": {
                                    "type": "array",
                                    "items": {
                                        "type": "object",
                                        "properties": {
                                            "item": {"$ref": "#/components/schemas/MediaItem"},
                                            "score": {"type": "number"},
                                            "reasoning": {"type": "string"},
                                            "influenced_by": {"type": "array"}
                                        }
                                    }
                                },
                                "model_version": {"type": "string"}
                            }
                        }
                    }
                }),
                examples: vec![
                    ToolExample {
                        input: serde_json::json!({
                            "user_id": "user_789",
                            "limit": 3,
                            "explain": true
                        }),
                        output: serde_json::json!({
                            "@context": {
                                "@vocab": "https://schema.org/",
                                "tv5": "https://tv5monde.com/vocab/"
                            },
                            "@type": "RecommendationList",
                            "data": {
                                "user_id": "user_789",
                                "recommendations": [
                                    {
                                        "item": {
                                            "id": "film_999",
                                            "title": "La Haine",
                                            "similarity_score": 0.89
                                        },
                                        "score": 0.91,
                                        "reasoning": "Based on your interest in urban dramas and social commentary",
                                        "influenced_by": ["film_123", "film_456"]
                                    }
                                ],
                                "model_version": "v2.3.0"
                            }
                        }),
                        description: "Personalized recommendations with explanations".to_string()
                    }
                ],
                avg_response_time_ms: 12,
            }
        ],
        auth: vec!["bearer".to_string(), "api-key".to_string()],
        rate_limits: RateLimitInfo {
            requests_per_minute: 120,
            requests_per_hour: 7000,
            burst_size: 20,
        },
    }
}

/// Get MCP tool manifest for AI agent integration
#[utoipa::path(
    get,
    path = "/api/v1/mcp/manifest",
    responses(
        (status = 200, description = "MCP tool manifest", body = MCPManifest)
    ),
    tag = "agent"
)]
pub async fn get_mcp_manifest() -> Json<MCPManifest> {
    Json(generate_mcp_manifest())
}
