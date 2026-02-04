//! Ollama provider configuration.

use std::time::Duration;

/// Configuration for the Ollama provider.
///
/// Use struct update syntax with [`Default`] for ergonomic construction:
///
/// ```rust
/// use llm_stack_ollama::OllamaConfig;
///
/// let config = OllamaConfig {
///     model: "llama3.2".into(),
///     ..Default::default()
/// };
/// ```
#[derive(Debug, Clone)]
pub struct OllamaConfig {
    /// Model identifier (e.g. `"llama3.2"`, `"mistral"`).
    pub model: String,
    /// Base URL for the Ollama API. Defaults to `http://localhost:11434`.
    pub base_url: String,
    /// Request timeout. `None` uses reqwest's default.
    pub timeout: Option<Duration>,
    /// Pre-configured HTTP client for connection pooling.
    /// When `None`, a new client is created.
    pub client: Option<reqwest::Client>,
}

impl Default for OllamaConfig {
    fn default() -> Self {
        Self {
            model: "llama3.2".into(),
            base_url: "http://localhost:11434".into(),
            timeout: None,
            client: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = OllamaConfig::default();
        assert_eq!(config.model, "llama3.2");
        assert_eq!(config.base_url, "http://localhost:11434");
        assert!(config.timeout.is_none());
        assert!(config.client.is_none());
    }

    #[test]
    fn test_debug_output() {
        let config = OllamaConfig::default();
        let debug = format!("{config:?}");
        assert!(debug.contains("llama3.2"));
        assert!(debug.contains("localhost:11434"));
    }

    #[test]
    fn test_config_override() {
        let config = OllamaConfig {
            model: "mistral".into(),
            base_url: "http://remote:11434".into(),
            ..Default::default()
        };
        assert_eq!(config.model, "mistral");
        assert_eq!(config.base_url, "http://remote:11434");
    }
}
