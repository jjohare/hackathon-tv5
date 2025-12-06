use serde::{Deserialize, Serialize};
use utoipa::ToSchema;

/// HATEOAS response wrapper with hypermedia links
#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
pub struct HATEOASResponse<T> {
    /// Response data
    pub data: T,

    /// Hypermedia links for navigation
    #[serde(rename = "_links")]
    pub links: Links,

    /// Embedded related resources
    #[serde(rename = "_embedded", skip_serializing_if = "Option::is_none")]
    pub embedded: Option<serde_json::Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
pub struct Links {
    /// Link to current resource
    #[serde(rename = "self")]
    pub self_link: Link,

    /// Related resources
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub related: Vec<Link>,

    /// Available actions on this resource
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub actions: Vec<Link>,
}

#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
pub struct Link {
    /// URL for the link
    pub href: String,

    /// Relationship type
    pub rel: String,

    /// HTTP method (GET, POST, etc.)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub method: Option<String>,

    /// Human-readable title
    #[serde(skip_serializing_if = "Option::is_none")]
    pub title: Option<String>,

    /// Media type of the target resource
    #[serde(skip_serializing_if = "Option::is_none")]
    pub type_: Option<String>,

    /// Whether this action is idempotent
    #[serde(skip_serializing_if = "Option::is_none")]
    pub idempotent: Option<bool>,
}

impl<T> HATEOASResponse<T> {
    /// Create a new HATEOAS response
    pub fn new(data: T, self_href: &str) -> Self {
        HATEOASResponse {
            data,
            links: Links {
                self_link: Link {
                    href: self_href.to_string(),
                    rel: "self".to_string(),
                    method: Some("GET".to_string()),
                    title: None,
                    type_: Some("application/json".to_string()),
                    idempotent: Some(true),
                },
                related: vec![],
                actions: vec![],
            },
            embedded: None,
        }
    }

    /// Add an action link
    pub fn add_action(mut self, href: &str, rel: &str, method: &str, title: &str) -> Self {
        self.links.actions.push(Link {
            href: href.to_string(),
            rel: rel.to_string(),
            method: Some(method.to_string()),
            title: Some(title.to_string()),
            type_: Some("application/json".to_string()),
            idempotent: Some(matches!(method, "GET" | "PUT" | "DELETE")),
        });
        self
    }

    /// Add a related resource link
    pub fn add_related(mut self, href: &str, rel: &str, title: &str) -> Self {
        self.links.related.push(Link {
            href: href.to_string(),
            rel: rel.to_string(),
            method: Some("GET".to_string()),
            title: Some(title.to_string()),
            type_: Some("application/json".to_string()),
            idempotent: Some(true),
        });
        self
    }

    /// Add embedded resources
    pub fn with_embedded(mut self, embedded: serde_json::Value) -> Self {
        self.embedded = Some(embedded);
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hateoas_response_creation() {
        let data = vec!["item1", "item2"];
        let response = HATEOASResponse::new(data, "/api/v1/items")
            .add_action("/api/v1/items", "create", "POST", "Create new item")
            .add_related("/api/v1/categories", "categories", "Related categories");

        assert_eq!(response.links.self_link.href, "/api/v1/items");
        assert_eq!(response.links.actions.len(), 1);
        assert_eq!(response.links.related.len(), 1);
    }
}
