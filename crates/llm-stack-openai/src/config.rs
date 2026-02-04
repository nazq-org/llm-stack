//! `OpenAI` provider configuration.

use std::time::Duration;

/// Configuration for the `OpenAI` provider.
///
/// Use struct update syntax with [`Default`] for ergonomic construction:
///
/// ```rust
/// use llm_stack_openai::OpenAiConfig;
///
/// let config = OpenAiConfig {
///     api_key: "sk-...".into(),
///     model: "gpt-4o".into(),
///     ..Default::default()
/// };
/// ```
#[derive(Clone)]
pub struct OpenAiConfig {
    /// `OpenAI` API key. Required.
    pub api_key: String,
    /// Model identifier (e.g. `"gpt-4o"`, `"gpt-4o-mini"`).
    pub model: String,
    /// Base URL for the API. Override for proxies, Azure, or local servers.
    pub base_url: String,
    /// Optional organization ID for API requests.
    pub organization: Option<String>,
    /// Request timeout. `None` uses reqwest's default.
    pub timeout: Option<Duration>,
    /// Pre-configured HTTP client for connection pooling across providers.
    /// When `None`, a new client is created.
    pub client: Option<reqwest::Client>,
}

impl std::fmt::Debug for OpenAiConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OpenAiConfig")
            .field("api_key", &"[REDACTED]")
            .field("model", &self.model)
            .field("base_url", &self.base_url)
            .field("organization", &self.organization)
            .field("timeout", &self.timeout)
            .field("client", &self.client.as_ref().map(|_| "..."))
            .finish()
    }
}

impl Default for OpenAiConfig {
    fn default() -> Self {
        Self {
            api_key: String::new(),
            model: "gpt-4o".into(),
            base_url: "https://api.openai.com/v1".into(),
            organization: None,
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
        let config = OpenAiConfig::default();
        assert_eq!(config.model, "gpt-4o");
        assert_eq!(config.base_url, "https://api.openai.com/v1");
        assert!(config.api_key.is_empty());
        assert!(config.organization.is_none());
        assert!(config.timeout.is_none());
        assert!(config.client.is_none());
    }

    #[test]
    fn test_debug_redacts_api_key() {
        let config = OpenAiConfig {
            api_key: "sk-super-secret".into(),
            ..Default::default()
        };
        let debug_output = format!("{config:?}");
        assert!(!debug_output.contains("sk-super-secret"));
        assert!(debug_output.contains("[REDACTED]"));
    }

    #[test]
    fn test_config_override() {
        let config = OpenAiConfig {
            api_key: "test-key".into(),
            model: "gpt-4o-mini".into(),
            organization: Some("org-123".into()),
            ..Default::default()
        };
        assert_eq!(config.api_key, "test-key");
        assert_eq!(config.model, "gpt-4o-mini");
        assert_eq!(config.organization.as_deref(), Some("org-123"));
    }
}
