use serde::{Deserialize, Serialize};
use utoipa::ToSchema;

/// JSON-LD response with semantic web context
#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
pub struct JsonLdContext {
    /// JSON-LD context definition
    #[serde(rename = "@context")]
    pub context: serde_json::Value,

    /// Type definition for this resource
    #[serde(rename = "@type")]
    pub ld_type: String,

    /// Actual data payload
    #[serde(flatten)]
    pub data: serde_json::Value,
}

/// Add JSON-LD context to any serializable data
pub fn add_json_ld_context<T: Serialize>(data: T, ld_type: &str) -> JsonLdContext {
    JsonLdContext {
        context: create_tv5_context(),
        ld_type: ld_type.to_string(),
        data: serde_json::to_value(data).expect("Failed to serialize data"),
    }
}

/// Create TV5 Monde vocabulary context
fn create_tv5_context() -> serde_json::Value {
    serde_json::json!({
        "@vocab": "https://schema.org/",
        "tv5": "https://tv5monde.com/vocab/",
        "xsd": "http://www.w3.org/2001/XMLSchema#",

        // Custom TV5 vocabulary
        "similarity": {
            "@id": "tv5:similarityScore",
            "@type": "xsd:float"
        },
        "embedding": {
            "@id": "tv5:embedding",
            "@type": "@json"
        },
        "queryTime": {
            "@id": "tv5:queryTimeMs",
            "@type": "xsd:integer"
        },
        "recommendationScore": {
            "@id": "tv5:recommendationScore",
            "@type": "xsd:float"
        },

        // Schema.org extensions
        "MediaItem": "schema:CreativeWork",
        "genre": "schema:genre",
        "contentRating": "schema:contentRating",
        "inLanguage": "schema:inLanguage",
        "datePublished": {
            "@id": "schema:datePublished",
            "@type": "xsd:gYear"
        },
        "director": "schema:director",
        "actor": "schema:actor",
        "duration": {
            "@id": "schema:duration",
            "@type": "xsd:duration"
        },

        // Recommendation-specific terms
        "RecommendationList": "tv5:RecommendationList",
        "recommendedBy": "tv5:recommendedBy",
        "basedOn": "tv5:basedOn",
        "reasoning": "tv5:reasoning",
        "influencedBy": "tv5:influencedBy",

        // Pagination
        "pagination": "schema:ItemList",
        "currentPage": "tv5:currentPage",
        "totalPages": "tv5:totalPages",
        "itemsPerPage": "schema:numberOfItems",

        // Metadata
        "modelVersion": "tv5:modelVersion",
        "generatedAt": {
            "@id": "schema:dateCreated",
            "@type": "xsd:dateTime"
        }
    })
}

/// Create a collection with JSON-LD context
pub fn create_collection<T: Serialize>(
    items: Vec<T>,
    collection_type: &str,
    total_items: usize,
) -> JsonLdContext {
    let collection = serde_json::json!({
        "@type": "Collection",
        "collectionType": collection_type,
        "totalItems": total_items,
        "member": items
    });

    JsonLdContext {
        context: create_tv5_context(),
        ld_type: "Collection".to_string(),
        data: collection,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_json_ld_context_creation() {
        let data = serde_json::json!({
            "title": "Test Movie",
            "year": 2024
        });

        let ld = add_json_ld_context(data, "MediaItem");

        assert_eq!(ld.ld_type, "MediaItem");
        assert!(ld.context.get("@vocab").is_some());
        assert!(ld.context.get("tv5").is_some());
    }

    #[test]
    fn test_collection_creation() {
        let items = vec![
            serde_json::json!({"id": "1", "title": "Movie 1"}),
            serde_json::json!({"id": "2", "title": "Movie 2"}),
        ];

        let collection = create_collection(items, "movies", 2);

        assert_eq!(collection.ld_type, "Collection");
        assert!(collection.data.get("member").is_some());
    }
}
