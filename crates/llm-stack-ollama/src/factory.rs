//! Factory for building Ollama providers from configuration.

use llm_stack::registry::{ProviderConfig, ProviderFactory};
use llm_stack::{DynProvider, LlmError};

use crate::{OllamaConfig, OllamaProvider};

/// Factory for creating [`OllamaProvider`] instances from configuration.
///
/// Register this factory with the global registry to enable config-driven
/// provider instantiation:
///
/// ```rust,no_run
/// use llm_stack::ProviderRegistry;
/// use llm_stack_ollama::OllamaFactory;
///
/// ProviderRegistry::global().register(Box::new(OllamaFactory));
/// ```
///
/// # Configuration
///
/// | Field | Required | Description |
/// |-------|----------|-------------|
/// | `provider` | Yes | Must be `"ollama"` |
/// | `api_key` | No | Not used (Ollama doesn't require auth) |
/// | `model` | Yes | Model identifier (e.g., `"llama3.2"`) |
/// | `base_url` | No | Custom API endpoint (default: `http://localhost:11434`) |
/// | `timeout` | No | Request timeout |
#[derive(Debug, Clone, Copy, Default)]
pub struct OllamaFactory;

impl ProviderFactory for OllamaFactory {
    fn name(&self) -> &'static str {
        "ollama"
    }

    fn build(&self, config: &ProviderConfig) -> Result<Box<dyn DynProvider>, LlmError> {
        if config.model.is_empty() {
            return Err(LlmError::InvalidRequest(
                "ollama provider requires model".into(),
            ));
        }

        let mut ollama_config = OllamaConfig {
            model: config.model.clone(),
            ..Default::default()
        };

        if let Some(base_url) = &config.base_url {
            ollama_config.base_url.clone_from(base_url);
        }

        if let Some(timeout) = config.timeout {
            ollama_config.timeout = Some(timeout);
        }

        Ok(Box::new(OllamaProvider::new(ollama_config)))
    }
}

/// Registers the Ollama factory with the global registry.
///
/// Call this once at application startup to enable config-driven
/// Ollama provider creation.
pub fn register_global() {
    llm_stack::ProviderRegistry::global().register(Box::new(OllamaFactory));
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[test]
    fn test_factory_name() {
        let factory = OllamaFactory;
        assert_eq!(factory.name(), "ollama");
    }

    #[test]
    fn test_factory_build_success() {
        let factory = OllamaFactory;
        let config = ProviderConfig::new("ollama", "llama3.2")
            .base_url("http://remote:11434")
            .timeout(Duration::from_secs(60));

        let provider = factory.build(&config).unwrap();
        assert_eq!(provider.metadata().name, "ollama");
        assert_eq!(provider.metadata().model, "llama3.2");
    }

    #[test]
    fn test_factory_no_api_key_required() {
        let factory = OllamaFactory;
        // Ollama doesn't require an API key
        let config = ProviderConfig::new("ollama", "mistral");

        let provider = factory.build(&config).unwrap();
        assert_eq!(provider.metadata().model, "mistral");
    }

    #[test]
    fn test_factory_empty_model() {
        let factory = OllamaFactory;
        let config = ProviderConfig::new("ollama", "");

        let err = factory.build(&config).err().unwrap();
        assert!(matches!(err, LlmError::InvalidRequest(_)));
    }
}
