//! Text preprocessing utilities

use serde::{Deserialize, Serialize};

/// Configuration for text preprocessing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PreprocessingConfig {
    /// Convert to lowercase
    pub lowercase: bool,

    /// Remove extra whitespace
    pub normalize_whitespace: bool,

    /// Remove control characters
    pub remove_control_chars: bool,

    /// Maximum text length (characters)
    pub max_chars: Option<usize>,
}

impl Default for PreprocessingConfig {
    fn default() -> Self {
        Self {
            lowercase: false, // Preserve case for better semantic understanding
            normalize_whitespace: true,
            remove_control_chars: true,
            max_chars: Some(10000),
        }
    }
}

/// Text preprocessor for cleaning and normalizing input text
#[derive(Debug, Clone)]
pub struct TextPreprocessor {
    config: PreprocessingConfig,
}

impl Default for TextPreprocessor {
    fn default() -> Self {
        Self {
            config: PreprocessingConfig::default(),
        }
    }
}

impl TextPreprocessor {
    /// Create a new text preprocessor with custom configuration
    pub fn new(config: PreprocessingConfig) -> Self {
        Self { config }
    }

    /// Preprocess input text
    pub fn preprocess(&self, text: &str) -> String {
        let mut result = text.to_string();

        // Remove control characters
        if self.config.remove_control_chars {
            result = result
                .chars()
                .filter(|c| !c.is_control() || c.is_whitespace())
                .collect();
        }

        // Normalize whitespace
        if self.config.normalize_whitespace {
            result = result.split_whitespace().collect::<Vec<_>>().join(" ");
        }

        // Convert to lowercase
        if self.config.lowercase {
            result = result.to_lowercase();
        }

        // Truncate if needed
        if let Some(max_chars) = self.config.max_chars {
            if result.len() > max_chars {
                result.truncate(max_chars);
                // Ensure we don't cut in the middle of a Unicode character
                while !result.is_char_boundary(result.len()) {
                    result.pop();
                }
            }
        }

        result
    }

    /// Preprocess multiple texts
    pub fn preprocess_batch(&self, texts: &[&str]) -> Vec<String> {
        texts.iter().map(|text| self.preprocess(text)).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_normalize_whitespace() {
        let preprocessor = TextPreprocessor::default();
        assert_eq!(
            preprocessor.preprocess("  hello   world  "),
            "hello world"
        );
        assert_eq!(
            preprocessor.preprocess("hello\n\nworld"),
            "hello world"
        );
    }

    #[test]
    fn test_lowercase() {
        let config = PreprocessingConfig {
            lowercase: true,
            ..Default::default()
        };
        let preprocessor = TextPreprocessor::new(config);
        assert_eq!(preprocessor.preprocess("Hello World"), "hello world");
    }

    #[test]
    fn test_remove_control_chars() {
        let preprocessor = TextPreprocessor::default();
        let text_with_control = "hello\x00world\x01test";
        let processed = preprocessor.preprocess(text_with_control);
        assert!(!processed.contains('\x00'));
        assert!(!processed.contains('\x01'));
    }

    #[test]
    fn test_max_chars() {
        let config = PreprocessingConfig {
            max_chars: Some(10),
            ..Default::default()
        };
        let preprocessor = TextPreprocessor::new(config);
        assert_eq!(
            preprocessor.preprocess("This is a very long text").len(),
            10
        );
    }

    #[test]
    fn test_unicode_boundary() {
        let config = PreprocessingConfig {
            max_chars: Some(5),
            ..Default::default()
        };
        let preprocessor = TextPreprocessor::new(config);
        let result = preprocessor.preprocess("Hello 世界");
        assert!(result.is_char_boundary(result.len()));
    }

    #[test]
    fn test_batch_preprocess() {
        let preprocessor = TextPreprocessor::default();
        let texts = vec!["  hello  ", "  world  "];
        let processed = preprocessor.preprocess_batch(&texts);
        assert_eq!(processed, vec!["hello", "world"]);
    }
}
