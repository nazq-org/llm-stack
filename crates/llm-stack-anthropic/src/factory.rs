//! Factory for building Anthropic providers from configuration.

use llm_stack::registry::{ProviderConfig, ProviderFactory};
use llm_stack::{DynProvider, LlmError};

use crate::{AnthropicConfig, AnthropicProvider};

/// Factory for creating [`AnthropicProvider`] instances from configuration.
///
/// Register this factory with the global registry to enable config-driven
/// provider instantiation:
///
/// ```rust,no_run
/// use llm_stack::ProviderRegistry;
/// use llm_stack_anthropic::AnthropicFactory;
///
/// ProviderRegistry::global().register(Box::new(AnthropicFactory));
/// ```
///
/// # Configuration
///
/// | Field | Required | Description |
/// |-------|----------|-------------|
/// | `provider` | Yes | Must be `"anthropic"` |
/// | `api_key` | Yes | Anthropic API key |
/// | `model` | Yes | Model identifier (e.g., `"claude-sonnet-4-20250514"`) |
/// | `base_url` | No | Custom API endpoint |
/// | `timeout` | No | Request timeout |
/// | `extra.max_tokens` | No | Default max tokens (default: 4096) |
/// | `extra.api_version` | No | API version header |
#[derive(Debug, Clone, Copy, Default)]
pub struct AnthropicFactory;

impl ProviderFactory for AnthropicFactory {
    fn name(&self) -> &'static str {
        "anthropic"
    }

    fn build(&self, config: &ProviderConfig) -> Result<Box<dyn DynProvider>, LlmError> {
        let api_key = config.api_key.clone().ok_or_else(|| {
            LlmError::InvalidRequest("anthropic provider requires api_key".into())
        })?;

        if config.model.is_empty() {
            return Err(LlmError::InvalidRequest(
                "anthropic provider requires model".into(),
            ));
        }

        let mut anthropic_config = AnthropicConfig {
            api_key,
            model: config.model.clone(),
            client: config.client.clone(),
            ..Default::default()
        };

        if let Some(base_url) = &config.base_url {
            anthropic_config.base_url.clone_from(base_url);
        }

        if let Some(timeout) = config.timeout {
            anthropic_config.timeout = Some(timeout);
        }

        if let Some(max_tokens) = config.get_extra_i64("max_tokens") {
            anthropic_config.max_tokens =
                u32::try_from(max_tokens).unwrap_or(anthropic_config.max_tokens);
        }

        if let Some(api_version) = config.get_extra_str("api_version") {
            anthropic_config.api_version = api_version.to_string();
        }

        Ok(Box::new(AnthropicProvider::new(anthropic_config)))
    }
}

/// Registers the Anthropic factory with the global registry.
///
/// Call this once at application startup to enable config-driven
/// Anthropic provider creation.
pub fn register_global() {
    llm_stack::ProviderRegistry::global().register(Box::new(AnthropicFactory));
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[test]
    fn test_factory_name() {
        let factory = AnthropicFactory;
        assert_eq!(factory.name(), "anthropic");
    }

    #[test]
    fn test_factory_build_success() {
        let factory = AnthropicFactory;
        let config = ProviderConfig::new("anthropic", "claude-3")
            .api_key("sk-test")
            .timeout(Duration::from_secs(30))
            .extra("max_tokens", 2048i64);

        let provider = factory.build(&config).unwrap();
        assert_eq!(provider.metadata().name, "anthropic");
        assert_eq!(provider.metadata().model, "claude-3");
    }

    #[test]
    fn test_factory_missing_api_key() {
        let factory = AnthropicFactory;
        let config = ProviderConfig::new("anthropic", "claude-3");

        let err = factory.build(&config).err().unwrap();
        assert!(matches!(err, LlmError::InvalidRequest(_)));
    }

    #[test]
    fn test_factory_empty_model() {
        let factory = AnthropicFactory;
        let config = ProviderConfig::new("anthropic", "").api_key("sk-test");

        let err = factory.build(&config).err().unwrap();
        assert!(matches!(err, LlmError::InvalidRequest(_)));
    }
}
