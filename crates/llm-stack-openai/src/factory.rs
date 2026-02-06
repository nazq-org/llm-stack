//! Factory for building `OpenAI` providers from configuration.

use llm_stack::registry::{ProviderConfig, ProviderFactory};
use llm_stack::{DynProvider, LlmError};

use crate::{OpenAiConfig, OpenAiProvider};

/// Factory for creating [`OpenAiProvider`] instances from configuration.
///
/// Register this factory with the global registry to enable config-driven
/// provider instantiation:
///
/// ```rust,no_run
/// use llm_stack::ProviderRegistry;
/// use llm_stack_openai::OpenAiFactory;
///
/// ProviderRegistry::global().register(Box::new(OpenAiFactory));
/// ```
///
/// # Configuration
///
/// | Field | Required | Description |
/// |-------|----------|-------------|
/// | `provider` | Yes | Must be `"openai"` |
/// | `api_key` | Yes | OpenAI API key |
/// | `model` | Yes | Model identifier (e.g., `"gpt-4o"`) |
/// | `base_url` | No | Custom API endpoint |
/// | `timeout` | No | Request timeout |
/// | `extra.organization` | No | OpenAI organization ID |
#[derive(Debug, Clone, Copy, Default)]
pub struct OpenAiFactory;

impl ProviderFactory for OpenAiFactory {
    fn name(&self) -> &'static str {
        "openai"
    }

    fn build(&self, config: &ProviderConfig) -> Result<Box<dyn DynProvider>, LlmError> {
        let api_key = config
            .api_key
            .clone()
            .ok_or_else(|| LlmError::InvalidRequest("openai provider requires api_key".into()))?;

        if config.model.is_empty() {
            return Err(LlmError::InvalidRequest(
                "openai provider requires model".into(),
            ));
        }

        let mut openai_config = OpenAiConfig {
            api_key,
            model: config.model.clone(),
            client: config.client.clone(),
            ..Default::default()
        };

        if let Some(base_url) = &config.base_url {
            openai_config.base_url.clone_from(base_url);
        }

        if let Some(timeout) = config.timeout {
            openai_config.timeout = Some(timeout);
        }

        if let Some(organization) = config.get_extra_str("organization") {
            openai_config.organization = Some(organization.to_string());
        }

        Ok(Box::new(OpenAiProvider::new(openai_config)))
    }
}

/// Registers the `OpenAI` factory with the global registry.
///
/// Call this once at application startup to enable config-driven
/// `OpenAI` provider creation.
pub fn register_global() {
    llm_stack::ProviderRegistry::global().register(Box::new(OpenAiFactory));
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[test]
    fn test_factory_name() {
        let factory = OpenAiFactory;
        assert_eq!(factory.name(), "openai");
    }

    #[test]
    fn test_factory_build_success() {
        let factory = OpenAiFactory;
        let config = ProviderConfig::new("openai", "gpt-4o")
            .api_key("sk-test")
            .timeout(Duration::from_secs(30))
            .extra("organization", "org-123");

        let provider = factory.build(&config).unwrap();
        assert_eq!(provider.metadata().name, "openai");
        assert_eq!(provider.metadata().model, "gpt-4o");
    }

    #[test]
    fn test_factory_missing_api_key() {
        let factory = OpenAiFactory;
        let config = ProviderConfig::new("openai", "gpt-4o");

        let err = factory.build(&config).err().unwrap();
        assert!(matches!(err, LlmError::InvalidRequest(_)));
    }

    #[test]
    fn test_factory_empty_model() {
        let factory = OpenAiFactory;
        let config = ProviderConfig::new("openai", "").api_key("sk-test");

        let err = factory.build(&config).err().unwrap();
        assert!(matches!(err, LlmError::InvalidRequest(_)));
    }
}
