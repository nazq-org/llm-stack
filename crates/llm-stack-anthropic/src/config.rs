//! Anthropic provider configuration.

use std::time::Duration;

/// Configuration for the Anthropic provider.
///
/// Use struct update syntax with [`Default`] for ergonomic construction:
///
/// ```rust
/// use llm_stack_anthropic::AnthropicConfig;
///
/// let config = AnthropicConfig {
///     api_key: "sk-ant-...".into(),
///     model: "claude-sonnet-4-20250514".into(),
///     ..Default::default()
/// };
/// ```
#[derive(Clone)]
pub struct AnthropicConfig {
    /// Anthropic API key. Required.
    pub api_key: String,
    /// Model identifier (e.g. `"claude-sonnet-4-20250514"`).
    pub model: String,
    /// Base URL for the API. Override for proxies or testing.
    pub base_url: String,
    /// Default max tokens for responses when not specified in `ChatParams`.
    pub max_tokens: u32,
    /// Anthropic API version header.
    pub api_version: String,
    /// Request timeout. `None` uses reqwest's default.
    pub timeout: Option<Duration>,
    /// Pre-configured HTTP client for connection pooling across providers.
    /// When `None`, a new client is created.
    pub client: Option<reqwest::Client>,
}

impl std::fmt::Debug for AnthropicConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AnthropicConfig")
            .field("api_key", &"[REDACTED]")
            .field("model", &self.model)
            .field("base_url", &self.base_url)
            .field("max_tokens", &self.max_tokens)
            .field("api_version", &self.api_version)
            .field("timeout", &self.timeout)
            .field("client", &self.client.as_ref().map(|_| "..."))
            .finish()
    }
}

impl Default for AnthropicConfig {
    fn default() -> Self {
        Self {
            api_key: String::new(),
            model: "claude-sonnet-4-20250514".into(),
            base_url: "https://api.anthropic.com".into(),
            max_tokens: 4096,
            api_version: "2023-06-01".into(),
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
        let config = AnthropicConfig::default();
        assert_eq!(config.model, "claude-sonnet-4-20250514");
        assert_eq!(config.base_url, "https://api.anthropic.com");
        assert_eq!(config.max_tokens, 4096);
        assert_eq!(config.api_version, "2023-06-01");
        assert!(config.api_key.is_empty());
        assert!(config.timeout.is_none());
        assert!(config.client.is_none());
    }

    #[test]
    fn test_debug_redacts_api_key() {
        let config = AnthropicConfig {
            api_key: "sk-ant-super-secret".into(),
            ..Default::default()
        };
        let debug_output = format!("{config:?}");
        assert!(
            !debug_output.contains("sk-ant-super-secret"),
            "Debug output should not contain the API key"
        );
        assert!(debug_output.contains("[REDACTED]"));
    }

    #[test]
    fn test_config_override() {
        let config = AnthropicConfig {
            api_key: "test-key".into(),
            model: "claude-3-5-haiku-20241022".into(),
            max_tokens: 1024,
            ..Default::default()
        };
        assert_eq!(config.api_key, "test-key");
        assert_eq!(config.model, "claude-3-5-haiku-20241022");
        assert_eq!(config.max_tokens, 1024);
        assert_eq!(config.base_url, "https://api.anthropic.com");
    }
}
