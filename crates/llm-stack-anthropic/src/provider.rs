//! Anthropic `Provider` implementation.

use std::collections::HashSet;

use llm_stack_core::ChatResponse;
use llm_stack_core::error::LlmError;
use llm_stack_core::provider::{Capability, ChatParams, Provider, ProviderMetadata};
use llm_stack_core::stream::ChatStream;
use reqwest::header::{HeaderMap, HeaderValue};
use tracing::instrument;

use crate::config::AnthropicConfig;
use crate::convert;

/// Anthropic Claude provider implementing [`Provider`].
///
/// Supports the Anthropic Messages API with tool calling, extended
/// thinking, and streaming.
///
/// # Example
///
/// ```rust,no_run
/// use llm_stack_anthropic::{AnthropicConfig, AnthropicProvider};
/// use llm_stack_core::{ChatParams, ChatMessage, Provider};
///
/// # async fn example() -> Result<(), llm_stack_core::LlmError> {
/// let provider = AnthropicProvider::new(AnthropicConfig {
///     api_key: std::env::var("ANTHROPIC_API_KEY").unwrap(),
///     ..Default::default()
/// });
///
/// let response = provider.generate(&ChatParams {
///     messages: vec![ChatMessage::user("Hello!")],
///     ..Default::default()
/// }).await?;
/// # Ok(())
/// # }
/// ```
#[derive(Debug)]
pub struct AnthropicProvider {
    config: AnthropicConfig,
    client: reqwest::Client,
}

impl AnthropicProvider {
    /// Create a new Anthropic provider from configuration.
    ///
    /// If `config.client` is `Some`, that client is reused for connection
    /// pooling. Otherwise a new client is built with the configured timeout.
    pub fn new(config: AnthropicConfig) -> Self {
        let client = config.client.clone().unwrap_or_else(|| {
            let mut builder = reqwest::Client::builder();
            if let Some(timeout) = config.timeout {
                builder = builder.timeout(timeout);
            }
            builder.build().expect("failed to build HTTP client")
        });
        Self { config, client }
    }

    /// Build the default headers for Anthropic API requests.
    fn default_headers(&self) -> Result<HeaderMap, LlmError> {
        let mut headers = HeaderMap::new();
        headers.insert(
            "x-api-key",
            HeaderValue::from_str(&self.config.api_key)
                .map_err(|_| LlmError::Auth("API key contains invalid header characters".into()))?,
        );
        headers.insert(
            "anthropic-version",
            HeaderValue::from_str(&self.config.api_version).map_err(|_| {
                LlmError::InvalidRequest("API version contains invalid header characters".into())
            })?,
        );
        headers.insert("content-type", HeaderValue::from_static("application/json"));
        Ok(headers)
    }

    /// Build the full URL for the messages endpoint.
    fn messages_url(&self) -> String {
        let base = self.config.base_url.trim_end_matches('/');
        format!("{base}/v1/messages")
    }

    /// Send a request to the Anthropic Messages API and return the raw
    /// response after validating the HTTP status.
    async fn send_request(
        &self,
        params: &ChatParams,
        stream: bool,
    ) -> Result<reqwest::Response, LlmError> {
        let request_body = convert::build_request(params, &self.config, stream)?;

        let mut headers = self.default_headers()?;
        if let Some(extra) = &params.extra_headers {
            headers.extend(extra.iter().map(|(k, v)| (k.clone(), v.clone())));
        }

        let mut req = self
            .client
            .post(self.messages_url())
            .headers(headers)
            .json(&request_body);

        if let Some(timeout) = params.timeout {
            req = req.timeout(timeout);
        }

        let response = req.send().await.map_err(|e| {
            if e.is_timeout() {
                LlmError::Timeout {
                    elapsed_ms: params
                        .timeout
                        .or(self.config.timeout)
                        .map_or(0, |d| u64::try_from(d.as_millis()).unwrap_or(u64::MAX)),
                }
            } else {
                LlmError::Http {
                    status: e.status().map(|s| {
                        http::StatusCode::from_u16(s.as_u16())
                            .unwrap_or(http::StatusCode::INTERNAL_SERVER_ERROR)
                    }),
                    message: e.to_string(),
                    retryable: e.is_connect() || e.is_timeout(),
                }
            }
        })?;

        let status = response.status();
        if !status.is_success() {
            let body = response.text().await.unwrap_or_default();
            let http_status = http::StatusCode::from_u16(status.as_u16())
                .unwrap_or(http::StatusCode::INTERNAL_SERVER_ERROR);
            return Err(convert::convert_error(http_status, &body));
        }

        Ok(response)
    }
}

impl Provider for AnthropicProvider {
    #[instrument(skip_all, fields(model = %self.config.model))]
    async fn generate(&self, params: &ChatParams) -> Result<ChatResponse, LlmError> {
        let response = self.send_request(params, false).await?;

        let api_response: crate::types::Response =
            response
                .json()
                .await
                .map_err(|e| LlmError::ResponseFormat {
                    message: format!("Failed to parse Anthropic response: {e}"),
                    raw: String::new(),
                })?;

        Ok(convert::convert_response(api_response))
    }

    #[instrument(skip_all, fields(model = %self.config.model))]
    async fn stream(&self, params: &ChatParams) -> Result<ChatStream, LlmError> {
        let response = self.send_request(params, true).await?;
        Ok(crate::stream::into_stream(response))
    }

    fn metadata(&self) -> ProviderMetadata {
        let mut capabilities = HashSet::new();
        capabilities.insert(Capability::Tools);
        capabilities.insert(Capability::Vision);
        capabilities.insert(Capability::Reasoning);
        capabilities.insert(Capability::Caching);
        capabilities.insert(Capability::StructuredOutput);

        ProviderMetadata {
            name: "anthropic".into(),
            model: self.config.model.clone(),
            context_window: context_window_for_model(&self.config.model),
            capabilities,
        }
    }
}

/// Look up the context window size for known Anthropic models.
fn context_window_for_model(model: &str) -> u64 {
    if model.contains("claude") {
        200_000
    } else {
        // Conservative default for unknown models
        100_000
    }
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use super::*;

    #[test]
    fn test_metadata() {
        let provider = AnthropicProvider::new(AnthropicConfig {
            model: "claude-sonnet-4-20250514".into(),
            ..Default::default()
        });
        let meta = provider.metadata();

        assert_eq!(meta.name, "anthropic");
        assert_eq!(meta.model, "claude-sonnet-4-20250514");
        assert_eq!(meta.context_window, 200_000);
        assert!(meta.capabilities.contains(&Capability::Tools));
        assert!(meta.capabilities.contains(&Capability::Vision));
        assert!(meta.capabilities.contains(&Capability::Reasoning));
        assert!(meta.capabilities.contains(&Capability::Caching));
    }

    #[test]
    fn test_context_window_claude_3_5() {
        assert_eq!(
            context_window_for_model("claude-3-5-haiku-20241022"),
            200_000
        );
        assert_eq!(
            context_window_for_model("claude-3-5-sonnet-20241022"),
            200_000
        );
    }

    #[test]
    fn test_context_window_claude_4() {
        assert_eq!(
            context_window_for_model("claude-sonnet-4-20250514"),
            200_000
        );
        assert_eq!(context_window_for_model("claude-opus-4-20250514"), 200_000);
    }

    #[test]
    fn test_context_window_unknown() {
        assert_eq!(context_window_for_model("some-future-model"), 100_000);
    }

    #[test]
    fn test_messages_url() {
        let provider = AnthropicProvider::new(AnthropicConfig {
            base_url: "https://api.anthropic.com".into(),
            ..Default::default()
        });
        assert_eq!(
            provider.messages_url(),
            "https://api.anthropic.com/v1/messages"
        );
    }

    #[test]
    fn test_messages_url_custom_base() {
        let provider = AnthropicProvider::new(AnthropicConfig {
            base_url: "http://localhost:8080".into(),
            ..Default::default()
        });
        assert_eq!(provider.messages_url(), "http://localhost:8080/v1/messages");
    }

    #[test]
    fn test_messages_url_trailing_slash() {
        let provider = AnthropicProvider::new(AnthropicConfig {
            base_url: "https://proxy.example.com/".into(),
            ..Default::default()
        });
        assert_eq!(
            provider.messages_url(),
            "https://proxy.example.com/v1/messages"
        );
    }

    #[test]
    fn test_default_headers() {
        let provider = AnthropicProvider::new(AnthropicConfig {
            api_key: "sk-ant-test123".into(),
            api_version: "2023-06-01".into(),
            ..Default::default()
        });
        let headers = provider.default_headers().unwrap();

        assert_eq!(headers.get("x-api-key").unwrap(), "sk-ant-test123");
        assert_eq!(headers.get("anthropic-version").unwrap(), "2023-06-01");
        assert_eq!(headers.get("content-type").unwrap(), "application/json");
    }

    #[test]
    fn test_default_headers_invalid_api_key() {
        let provider = AnthropicProvider::new(AnthropicConfig {
            api_key: "invalid\nkey".into(),
            ..Default::default()
        });
        let err = provider.default_headers().unwrap_err();
        assert!(matches!(err, llm_stack_core::LlmError::Auth(_)));
    }

    #[test]
    fn test_new_with_custom_client() {
        let custom_client = reqwest::Client::builder()
            .timeout(Duration::from_secs(10))
            .build()
            .unwrap();

        let provider = AnthropicProvider::new(AnthropicConfig {
            client: Some(custom_client),
            ..Default::default()
        });

        // Should use the provided client (we can't easily assert identity,
        // but this verifies it doesn't panic)
        assert_eq!(provider.metadata().name, "anthropic");
    }

    #[test]
    fn test_new_with_timeout() {
        let provider = AnthropicProvider::new(AnthropicConfig {
            timeout: Some(Duration::from_secs(30)),
            ..Default::default()
        });
        assert_eq!(provider.metadata().name, "anthropic");
    }
}
