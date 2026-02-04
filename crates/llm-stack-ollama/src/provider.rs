//! Ollama `Provider` implementation.

use std::collections::HashSet;

use llm_stack_core::ChatResponse;
use llm_stack_core::error::LlmError;
use llm_stack_core::provider::{Capability, ChatParams, Provider, ProviderMetadata};
use llm_stack_core::stream::ChatStream;
use tracing::instrument;

use crate::config::OllamaConfig;
use crate::convert;

/// Ollama provider implementing [`Provider`].
///
/// Connects to a locally running Ollama instance. No authentication
/// is required by default.
///
/// # Example
///
/// ```rust,no_run
/// use llm_stack_ollama::{OllamaConfig, OllamaProvider};
/// use llm_stack_core::{ChatParams, ChatMessage, Provider};
///
/// # async fn example() -> Result<(), llm_stack_core::LlmError> {
/// let provider = OllamaProvider::new(OllamaConfig::default());
///
/// let response = provider.generate(&ChatParams {
///     messages: vec![ChatMessage::user("Hello!")],
///     ..Default::default()
/// }).await?;
/// # Ok(())
/// # }
/// ```
#[derive(Debug)]
pub struct OllamaProvider {
    config: OllamaConfig,
    client: reqwest::Client,
}

impl OllamaProvider {
    /// Create a new Ollama provider from configuration.
    ///
    /// If `config.client` is `Some`, that client is reused for connection
    /// pooling. Otherwise a new client is built with the configured timeout.
    pub fn new(config: OllamaConfig) -> Self {
        let client = config.client.clone().unwrap_or_else(|| {
            let mut builder = reqwest::Client::builder();
            if let Some(timeout) = config.timeout {
                builder = builder.timeout(timeout);
            }
            builder.build().expect("failed to build HTTP client")
        });
        Self { config, client }
    }

    /// Build the full URL for the chat endpoint.
    fn chat_url(&self) -> String {
        let base = self.config.base_url.trim_end_matches('/');
        format!("{base}/api/chat")
    }

    /// Send a request to the Ollama API and return the raw response.
    async fn send_request(
        &self,
        params: &ChatParams,
        stream: bool,
    ) -> Result<reqwest::Response, LlmError> {
        let request_body = convert::build_request(params, &self.config, stream)?;

        let mut headers = reqwest::header::HeaderMap::new();
        headers.insert(
            "content-type",
            reqwest::header::HeaderValue::from_static("application/json"),
        );
        if let Some(extra) = &params.extra_headers {
            headers.extend(extra.iter().map(|(k, v)| (k.clone(), v.clone())));
        }

        let mut req = self
            .client
            .post(self.chat_url())
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

impl Provider for OllamaProvider {
    #[instrument(skip_all, fields(model = %self.config.model))]
    async fn generate(&self, params: &ChatParams) -> Result<ChatResponse, LlmError> {
        let response = self.send_request(params, false).await?;

        let body = response
            .text()
            .await
            .map_err(|e| LlmError::ResponseFormat {
                message: format!("Failed to read Ollama response body: {e}"),
                raw: String::new(),
            })?;

        let api_response: crate::types::Response =
            serde_json::from_str(&body).map_err(|e| LlmError::ResponseFormat {
                message: format!("Failed to parse Ollama response: {e}"),
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

        ProviderMetadata {
            name: "ollama".into(),
            model: self.config.model.clone(),
            context_window: context_window_for_model(&self.config.model),
            capabilities,
        }
    }
}

/// Look up the context window size for known Ollama models.
///
/// Defaults to 128K for unknown models as most modern models
/// support large contexts via Ollama's automatic context extension.
fn context_window_for_model(model: &str) -> u64 {
    if model.starts_with("mistral") || model.starts_with("mixtral") {
        32_000
    } else if model.starts_with("gemma") {
        8_192
    } else {
        // llama3, phi, qwen, deepseek, and most modern models default to 128K
        128_000
    }
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use super::*;

    #[test]
    fn test_metadata() {
        let provider = OllamaProvider::new(OllamaConfig {
            model: "llama3.2".into(),
            ..Default::default()
        });
        let meta = provider.metadata();

        assert_eq!(meta.name, "ollama");
        assert_eq!(meta.model, "llama3.2");
        assert_eq!(meta.context_window, 128_000);
        assert!(meta.capabilities.contains(&Capability::Tools));
        assert!(meta.capabilities.contains(&Capability::Vision));
    }

    #[test]
    fn test_metadata_mistral() {
        let provider = OllamaProvider::new(OllamaConfig {
            model: "mistral".into(),
            ..Default::default()
        });
        let meta = provider.metadata();
        assert_eq!(meta.context_window, 32_000);
    }

    #[test]
    fn test_context_window_gemma() {
        assert_eq!(context_window_for_model("gemma2"), 8_192);
    }

    #[test]
    fn test_context_window_unknown() {
        assert_eq!(context_window_for_model("some-custom-model"), 128_000);
    }

    #[test]
    fn test_chat_url() {
        let provider = OllamaProvider::new(OllamaConfig {
            base_url: "http://localhost:11434".into(),
            ..Default::default()
        });
        assert_eq!(provider.chat_url(), "http://localhost:11434/api/chat");
    }

    #[test]
    fn test_chat_url_trailing_slash() {
        let provider = OllamaProvider::new(OllamaConfig {
            base_url: "http://remote:11434/".into(),
            ..Default::default()
        });
        assert_eq!(provider.chat_url(), "http://remote:11434/api/chat");
    }

    #[test]
    fn test_new_with_custom_client() {
        let custom_client = reqwest::Client::builder()
            .timeout(Duration::from_secs(10))
            .build()
            .unwrap();

        let provider = OllamaProvider::new(OllamaConfig {
            client: Some(custom_client),
            ..Default::default()
        });
        assert_eq!(provider.metadata().name, "ollama");
    }

    #[test]
    fn test_new_with_timeout() {
        let provider = OllamaProvider::new(OllamaConfig {
            timeout: Some(Duration::from_secs(60)),
            ..Default::default()
        });
        assert_eq!(provider.metadata().name, "ollama");
    }
}
