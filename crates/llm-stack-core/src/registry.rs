//! Dynamic provider registry for configuration-driven provider instantiation.
//!
//! The registry allows providers to be registered by name and instantiated from
//! configuration at runtime. This enables:
//!
//! - Config-file driven provider selection
//! - Third-party provider registration
//! - Dynamic provider switching without code changes
//!
//! # Example
//!
//! ```rust,no_run
//! use llm_stack_core::registry::{ProviderRegistry, ProviderConfig};
//!
//! // Get the global registry (providers register themselves on startup)
//! let registry = ProviderRegistry::global();
//!
//! // Build a provider from config
//! let config = ProviderConfig {
//!     provider: "anthropic".into(),
//!     api_key: Some("sk-...".into()),
//!     model: "claude-sonnet-4-20250514".into(),
//!     ..Default::default()
//! };
//!
//! let provider = registry.build(&config).expect("provider registered");
//! ```
//!
//! # Registering providers
//!
//! Provider crates register their factory on initialization:
//!
//! ```rust,ignore
//! use llm_stack_core::registry::{ProviderRegistry, ProviderFactory, ProviderConfig};
//!
//! struct MyProviderFactory;
//!
//! impl ProviderFactory for MyProviderFactory {
//!     fn name(&self) -> &str { "my-provider" }
//!
//!     fn build(&self, config: &ProviderConfig) -> Result<Box<dyn DynProvider>, LlmError> {
//!         // Build and return provider
//!     }
//! }
//!
//! // Register on crate initialization
//! ProviderRegistry::global().register(Box::new(MyProviderFactory));
//! ```

use std::collections::HashMap;
use std::sync::{Arc, OnceLock, RwLock};
use std::time::Duration;

use crate::error::LlmError;
use crate::provider::DynProvider;

/// Configuration for building a provider from the registry.
///
/// This struct contains common configuration fields that work across
/// all providers. Provider-specific options go in the `extra` map.
#[derive(Debug, Clone, Default)]
pub struct ProviderConfig {
    /// Provider name (e.g., "anthropic", "openai", "ollama").
    pub provider: String,

    /// API key for authenticated providers.
    pub api_key: Option<String>,

    /// Model identifier (e.g., "claude-sonnet-4-20250514", "gpt-4o").
    pub model: String,

    /// Custom base URL for the API endpoint.
    pub base_url: Option<String>,

    /// Request timeout.
    pub timeout: Option<Duration>,

    /// Provider-specific configuration options.
    ///
    /// Use this for options that don't fit the common fields above.
    /// Each provider documents which keys it recognizes.
    pub extra: HashMap<String, serde_json::Value>,
}

impl ProviderConfig {
    /// Creates a new config with the given provider and model.
    pub fn new(provider: impl Into<String>, model: impl Into<String>) -> Self {
        Self {
            provider: provider.into(),
            model: model.into(),
            ..Default::default()
        }
    }

    /// Sets the API key.
    #[must_use]
    pub fn api_key(mut self, key: impl Into<String>) -> Self {
        self.api_key = Some(key.into());
        self
    }

    /// Sets the base URL.
    #[must_use]
    pub fn base_url(mut self, url: impl Into<String>) -> Self {
        self.base_url = Some(url.into());
        self
    }

    /// Sets the timeout.
    #[must_use]
    pub fn timeout(mut self, timeout: Duration) -> Self {
        self.timeout = Some(timeout);
        self
    }

    /// Adds a provider-specific extra option.
    #[must_use]
    pub fn extra(mut self, key: impl Into<String>, value: impl Into<serde_json::Value>) -> Self {
        self.extra.insert(key.into(), value.into());
        self
    }

    /// Gets a string value from extra options.
    pub fn get_extra_str(&self, key: &str) -> Option<&str> {
        self.extra.get(key).and_then(|v| v.as_str())
    }

    /// Gets a bool value from extra options.
    pub fn get_extra_bool(&self, key: &str) -> Option<bool> {
        self.extra.get(key).and_then(serde_json::Value::as_bool)
    }

    /// Gets an integer value from extra options.
    pub fn get_extra_i64(&self, key: &str) -> Option<i64> {
        self.extra.get(key).and_then(serde_json::Value::as_i64)
    }
}

/// Factory trait for creating providers from configuration.
///
/// Implement this trait to register a provider with the registry.
pub trait ProviderFactory: Send + Sync {
    /// Returns the provider name used for registration and lookup.
    ///
    /// This should be a lowercase identifier (e.g., "anthropic", "openai").
    fn name(&self) -> &str;

    /// Creates a provider instance from the given configuration.
    ///
    /// # Errors
    ///
    /// Returns an error if the configuration is invalid or missing
    /// required fields for this provider.
    fn build(&self, config: &ProviderConfig) -> Result<Box<dyn DynProvider>, LlmError>;
}

/// A registry of provider factories for dynamic provider instantiation.
///
/// The registry maintains a map of provider names to their factories,
/// allowing providers to be created from configuration at runtime.
///
/// # Thread Safety
///
/// The registry is thread-safe and can be accessed concurrently.
/// Registration and lookup use interior mutability via `RwLock`.
///
/// # Global vs Local Registries
///
/// Use [`ProviderRegistry::global()`] for the shared global registry,
/// or create local registries with [`ProviderRegistry::new()`] for
/// testing or isolated contexts.
pub struct ProviderRegistry {
    factories: RwLock<HashMap<String, Arc<dyn ProviderFactory>>>,
}

impl std::fmt::Debug for ProviderRegistry {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let factories = self
            .factories
            .read()
            .expect("provider registry lock poisoned");
        let names: Vec<_> = factories.keys().collect();
        f.debug_struct("ProviderRegistry")
            .field("providers", &names)
            .finish()
    }
}

impl Default for ProviderRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl ProviderRegistry {
    /// Creates a new empty registry.
    pub fn new() -> Self {
        Self {
            factories: RwLock::new(HashMap::new()),
        }
    }

    /// Returns the global shared registry.
    ///
    /// Provider crates should register their factories here on initialization.
    /// Application code can then build providers from configuration without
    /// knowing which providers are available at compile time.
    pub fn global() -> &'static Self {
        static GLOBAL: OnceLock<ProviderRegistry> = OnceLock::new();
        GLOBAL.get_or_init(ProviderRegistry::new)
    }

    /// Registers a provider factory.
    ///
    /// If a factory with the same name already exists, it is replaced.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// use llm_stack_core::registry::{ProviderRegistry, ProviderFactory};
    ///
    /// ProviderRegistry::global().register(Box::new(MyProviderFactory));
    /// ```
    pub fn register(&self, factory: Box<dyn ProviderFactory>) -> &Self {
        let name = factory.name().to_lowercase();
        let mut factories = self
            .factories
            .write()
            .expect("provider registry lock poisoned");
        factories.insert(name, Arc::from(factory));
        self
    }

    /// Registers a provider factory (chainable Arc version).
    ///
    /// Use this when you want to share the factory instance.
    pub fn register_shared(&self, factory: Arc<dyn ProviderFactory>) -> &Self {
        let name = factory.name().to_lowercase();
        let mut factories = self
            .factories
            .write()
            .expect("provider registry lock poisoned");
        factories.insert(name, factory);
        self
    }

    /// Unregisters a provider by name.
    ///
    /// Returns `true` if the provider was registered and removed.
    pub fn unregister(&self, name: &str) -> bool {
        let mut factories = self
            .factories
            .write()
            .expect("provider registry lock poisoned");
        factories.remove(&name.to_lowercase()).is_some()
    }

    /// Checks if a provider is registered.
    pub fn contains(&self, name: &str) -> bool {
        let factories = self
            .factories
            .read()
            .expect("provider registry lock poisoned");
        factories.contains_key(&name.to_lowercase())
    }

    /// Returns the names of all registered providers.
    pub fn providers(&self) -> Vec<String> {
        let factories = self
            .factories
            .read()
            .expect("provider registry lock poisoned");
        factories.keys().cloned().collect()
    }

    /// Builds a provider from configuration.
    ///
    /// Looks up the factory by `config.provider` and delegates to it.
    ///
    /// # Errors
    ///
    /// Returns [`LlmError::InvalidRequest`] if no factory is registered
    /// for the requested provider name.
    pub fn build(&self, config: &ProviderConfig) -> Result<Box<dyn DynProvider>, LlmError> {
        let name = config.provider.to_lowercase();
        let factories = self
            .factories
            .read()
            .expect("provider registry lock poisoned");

        let factory = factories.get(&name).ok_or_else(|| {
            let available: Vec<_> = factories.keys().cloned().collect();
            LlmError::InvalidRequest(format!(
                "unknown provider '{}'. Available: {:?}",
                config.provider, available
            ))
        })?;

        factory.build(config)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chat::{ChatResponse, ContentBlock, StopReason};
    use crate::provider::{ChatParams, Provider, ProviderMetadata};
    use crate::stream::ChatStream;
    use crate::usage::Usage;
    use std::collections::{HashMap, HashSet};

    struct TestProvider {
        model: String,
    }

    impl Provider for TestProvider {
        async fn generate(&self, _params: &ChatParams) -> Result<ChatResponse, LlmError> {
            Ok(ChatResponse {
                content: vec![ContentBlock::Text("test".into())],
                usage: Usage::default(),
                stop_reason: StopReason::EndTurn,
                model: self.model.clone(),
                metadata: HashMap::default(),
            })
        }

        async fn stream(&self, _params: &ChatParams) -> Result<ChatStream, LlmError> {
            Err(LlmError::InvalidRequest("not implemented".into()))
        }

        fn metadata(&self) -> ProviderMetadata {
            ProviderMetadata {
                name: "test".into(),
                model: self.model.clone(),
                context_window: 4096,
                capabilities: HashSet::new(),
            }
        }
    }

    struct TestFactory;

    impl ProviderFactory for TestFactory {
        fn name(&self) -> &'static str {
            "test"
        }

        fn build(&self, config: &ProviderConfig) -> Result<Box<dyn DynProvider>, LlmError> {
            Ok(Box::new(TestProvider {
                model: config.model.clone(),
            }))
        }
    }

    #[test]
    fn test_registry_register_and_build() {
        let registry = ProviderRegistry::new();
        registry.register(Box::new(TestFactory));

        assert!(registry.contains("test"));
        assert!(registry.contains("TEST")); // case insensitive

        let config = ProviderConfig::new("test", "test-model");
        let provider = registry.build(&config).unwrap();

        assert_eq!(provider.metadata().model, "test-model");
    }

    #[test]
    fn test_registry_unknown_provider() {
        let registry = ProviderRegistry::new();

        let config = ProviderConfig::new("unknown", "model");
        let result = registry.build(&config);

        assert!(result.is_err());
        let err = result.err().unwrap();
        assert!(matches!(err, LlmError::InvalidRequest(_)));
    }

    #[test]
    fn test_registry_unregister() {
        let registry = ProviderRegistry::new();
        registry.register(Box::new(TestFactory));

        assert!(registry.contains("test"));
        assert!(registry.unregister("test"));
        assert!(!registry.contains("test"));
        assert!(!registry.unregister("test")); // already removed
    }

    #[test]
    fn test_registry_providers_list() {
        let registry = ProviderRegistry::new();
        registry.register(Box::new(TestFactory));

        let providers = registry.providers();
        assert_eq!(providers, vec!["test"]);
    }

    #[test]
    fn test_provider_config_builder() {
        let config = ProviderConfig::new("anthropic", "claude-3")
            .api_key("sk-123")
            .base_url("https://custom.api")
            .timeout(Duration::from_secs(60))
            .extra("organization", "org-123");

        assert_eq!(config.provider, "anthropic");
        assert_eq!(config.model, "claude-3");
        assert_eq!(config.api_key, Some("sk-123".into()));
        assert_eq!(config.base_url, Some("https://custom.api".into()));
        assert_eq!(config.timeout, Some(Duration::from_secs(60)));
        assert_eq!(config.get_extra_str("organization"), Some("org-123"));
    }

    #[test]
    fn test_provider_config_extra_types() {
        let config = ProviderConfig::new("test", "model")
            .extra("flag", true)
            .extra("count", 42i64)
            .extra("name", "value");

        assert_eq!(config.get_extra_bool("flag"), Some(true));
        assert_eq!(config.get_extra_i64("count"), Some(42));
        assert_eq!(config.get_extra_str("name"), Some("value"));
        assert_eq!(config.get_extra_str("missing"), None);
    }

    #[tokio::test]
    async fn test_built_provider_works() {
        let registry = ProviderRegistry::new();
        registry.register(Box::new(TestFactory));

        let config = ProviderConfig::new("test", "my-model");
        let provider = registry.build(&config).unwrap();

        let response = provider
            .generate_boxed(&ChatParams::default())
            .await
            .unwrap();
        assert_eq!(response.model, "my-model");
    }

    #[test]
    fn test_registry_replace_factory() {
        struct AltFactory;
        impl ProviderFactory for AltFactory {
            fn name(&self) -> &'static str {
                "test"
            }
            fn build(&self, config: &ProviderConfig) -> Result<Box<dyn DynProvider>, LlmError> {
                Ok(Box::new(TestProvider {
                    model: format!("alt-{}", config.model),
                }))
            }
        }

        let registry = ProviderRegistry::new();
        registry.register(Box::new(TestFactory));
        registry.register(Box::new(AltFactory)); // replaces

        let config = ProviderConfig::new("test", "model");
        let provider = registry.build(&config).unwrap();

        assert_eq!(provider.metadata().model, "alt-model");
    }
}
