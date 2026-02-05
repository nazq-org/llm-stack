//! `OpenAI` `Provider` implementation.

use std::collections::HashSet;

use llm_stack::ChatResponse;
use llm_stack::error::LlmError;
use llm_stack::provider::{Capability, ChatParams, Provider, ProviderMetadata};
use llm_stack::stream::ChatStream;
use reqwest::header::{HeaderMap, HeaderValue};
use tracing::instrument;

use crate::config::OpenAiConfig;
use crate::convert;

/// `OpenAI` provider implementing [`Provider`].
///
/// Supports the `OpenAI` Chat Completions API with tool calling,
/// structured output, and streaming.
///
/// # Example
///
/// ```rust,no_run
/// use llm_stack_openai::{OpenAiConfig, OpenAiProvider};
/// use llm_stack::{ChatParams, ChatMessage, Provider};
///
/// # async fn example() -> Result<(), llm_stack::LlmError> {
/// let provider = OpenAiProvider::new(OpenAiConfig {
///     api_key: std::env::var("OPENAI_API_KEY").unwrap(),
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
pub struct OpenAiProvider {
    config: OpenAiConfig,
    client: reqwest::Client,
}

impl OpenAiProvider {
    /// Create a new `OpenAI` provider from configuration.
    ///
    /// If `config.client` is `Some`, that client is reused for connection
    /// pooling. Otherwise a new client is built with the configured timeout.
    pub fn new(config: OpenAiConfig) -> Self {
        let client = config.client.clone().unwrap_or_else(|| {
            let mut builder = reqwest::Client::builder();
            if let Some(timeout) = config.timeout {
                builder = builder.timeout(timeout);
            }
            builder.build().expect("failed to build HTTP client")
        });
        Self { config, client }
    }

    /// Build the default headers for `OpenAI` API requests.
    fn default_headers(&self) -> Result<HeaderMap, LlmError> {
        let mut headers = HeaderMap::new();

        let auth_value = format!("Bearer {}", self.config.api_key);
        headers.insert(
            "authorization",
            HeaderValue::from_str(&auth_value)
                .map_err(|_| LlmError::Auth("API key contains invalid header characters".into()))?,
        );
        headers.insert("content-type", HeaderValue::from_static("application/json"));

        if let Some(org) = &self.config.organization {
            headers.insert(
                "openai-organization",
                HeaderValue::from_str(org).map_err(|_| {
                    LlmError::InvalidRequest(
                        "Organization ID contains invalid header characters".into(),
                    )
                })?,
            );
        }

        Ok(headers)
    }

    /// Build the full URL for the chat completions endpoint.
    fn completions_url(&self) -> String {
        let base = self.config.base_url.trim_end_matches('/');
        format!("{base}/chat/completions")
    }

    /// Send a request to the `OpenAI` API and return the raw response.
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
            .post(self.completions_url())
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

impl Provider for OpenAiProvider {
    #[instrument(skip_all, fields(model = %self.config.model))]
    async fn generate(&self, params: &ChatParams) -> Result<ChatResponse, LlmError> {
        let response = self.send_request(params, false).await?;

        let body = response
            .text()
            .await
            .map_err(|e| LlmError::ResponseFormat {
                message: format!("Failed to read OpenAI response body: {e}"),
                raw: String::new(),
            })?;

        let api_response: crate::types::Response =
            serde_json::from_str(&body).map_err(|e| LlmError::ResponseFormat {
                message: format!("Failed to parse OpenAI response: {e}"),
                raw: body,
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
        capabilities.insert(Capability::StructuredOutput);

        // o-series models support reasoning
        if self.config.model.starts_with("o1")
            || self.config.model.starts_with("o3")
            || self.config.model.starts_with("o4")
        {
            capabilities.insert(Capability::Reasoning);
        }

        ProviderMetadata {
            name: "openai".into(),
            model: self.config.model.clone(),
            context_window: context_window_for_model(&self.config.model),
            capabilities,
        }
    }
}

/// Look up the context window size for known `OpenAI` models.
fn context_window_for_model(model: &str) -> u64 {
    if model.starts_with("gpt-4o") || model.starts_with("gpt-4.1") {
        128_000
    } else if model.starts_with("o1") || model.starts_with("o3") || model.starts_with("o4") {
        200_000
    } else if model.starts_with("gpt-4") {
        128_000
    } else if model.starts_with("gpt-3.5") {
        16_385
    } else {
        128_000
    }
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use super::*;

    #[test]
    fn test_metadata() {
        let provider = OpenAiProvider::new(OpenAiConfig {
            model: "gpt-4o".into(),
            ..Default::default()
        });
        let meta = provider.metadata();

        assert_eq!(meta.name, "openai");
        assert_eq!(meta.model, "gpt-4o");
        assert_eq!(meta.context_window, 128_000);
        assert!(meta.capabilities.contains(&Capability::Tools));
        assert!(meta.capabilities.contains(&Capability::Vision));
        assert!(meta.capabilities.contains(&Capability::StructuredOutput));
        assert!(!meta.capabilities.contains(&Capability::Reasoning));
    }

    #[test]
    fn test_metadata_reasoning_model() {
        let provider = OpenAiProvider::new(OpenAiConfig {
            model: "o1-mini".into(),
            ..Default::default()
        });
        let meta = provider.metadata();

        assert!(meta.capabilities.contains(&Capability::Reasoning));
        assert_eq!(meta.context_window, 200_000);
    }

    #[test]
    fn test_context_window_gpt4o() {
        assert_eq!(context_window_for_model("gpt-4o"), 128_000);
        assert_eq!(context_window_for_model("gpt-4o-mini"), 128_000);
    }

    #[test]
    fn test_context_window_gpt35() {
        assert_eq!(context_window_for_model("gpt-3.5-turbo"), 16_385);
    }

    #[test]
    fn test_context_window_unknown() {
        assert_eq!(context_window_for_model("some-future-model"), 128_000);
    }

    #[test]
    fn test_completions_url() {
        let provider = OpenAiProvider::new(OpenAiConfig {
            base_url: "https://api.openai.com/v1".into(),
            ..Default::default()
        });
        assert_eq!(
            provider.completions_url(),
            "https://api.openai.com/v1/chat/completions"
        );
    }

    #[test]
    fn test_completions_url_trailing_slash() {
        let provider = OpenAiProvider::new(OpenAiConfig {
            base_url: "https://proxy.example.com/v1/".into(),
            ..Default::default()
        });
        assert_eq!(
            provider.completions_url(),
            "https://proxy.example.com/v1/chat/completions"
        );
    }

    #[test]
    fn test_default_headers() {
        let provider = OpenAiProvider::new(OpenAiConfig {
            api_key: "sk-test123".into(),
            ..Default::default()
        });
        let headers = provider.default_headers().unwrap();

        assert_eq!(headers.get("authorization").unwrap(), "Bearer sk-test123");
        assert_eq!(headers.get("content-type").unwrap(), "application/json");
    }

    #[test]
    fn test_default_headers_with_org() {
        let provider = OpenAiProvider::new(OpenAiConfig {
            api_key: "sk-test123".into(),
            organization: Some("org-abc".into()),
            ..Default::default()
        });
        let headers = provider.default_headers().unwrap();

        assert_eq!(headers.get("openai-organization").unwrap(), "org-abc");
    }

    #[test]
    fn test_default_headers_invalid_key() {
        let provider = OpenAiProvider::new(OpenAiConfig {
            api_key: "invalid\nkey".into(),
            ..Default::default()
        });
        let err = provider.default_headers().unwrap_err();
        assert!(matches!(err, LlmError::Auth(_)));
    }

    #[test]
    fn test_new_with_custom_client() {
        let custom_client = reqwest::Client::builder()
            .timeout(Duration::from_secs(10))
            .build()
            .unwrap();

        let provider = OpenAiProvider::new(OpenAiConfig {
            client: Some(custom_client),
            ..Default::default()
        });
        assert_eq!(provider.metadata().name, "openai");
    }

    #[test]
    fn test_new_with_timeout() {
        let provider = OpenAiProvider::new(OpenAiConfig {
            timeout: Some(Duration::from_secs(30)),
            ..Default::default()
        });
        assert_eq!(provider.metadata().name, "openai");
    }
}
