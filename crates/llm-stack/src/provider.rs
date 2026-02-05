//! Provider trait and request types.
//!
//! This module defines two core abstractions:
//!
//! - **[`Provider`]** — the trait every backend implements. It uses Rust
//!   2024's native async-fn-in-traits (AFIT), so implementations are
//!   straightforward `async fn`s with no macro overhead.
//!
//! - **[`DynProvider`]** — an object-safe mirror of `Provider` that uses
//!   boxed futures. A blanket `impl<T: Provider> DynProvider for T`
//!   bridges the two, so any concrete provider can be stored as
//!   `Box<dyn DynProvider>` or `Arc<dyn DynProvider>` with zero
//!   boilerplate.
//!
//! # When to use which
//!
//! | Situation | Use |
//! |-----------|-----|
//! | Generic code that knows the concrete type | `Provider` |
//! | Need to store providers in a collection or behind `dyn` | `DynProvider` |
//! | Implementing a new backend | `impl Provider for MyBackend` |
//!
//! # Request parameters
//!
//! All request configuration lives in [`ChatParams`]. It serializes
//! cleanly to JSON (for logging / replay) with the exception of
//! [`timeout`](ChatParams::timeout) and
//! [`extra_headers`](ChatParams::extra_headers), which are transport
//! concerns and are `#[serde(skip)]`'d.

use std::borrow::Cow;
use std::collections::{HashMap, HashSet};
use std::future::Future;
use std::pin::Pin;
use std::time::Duration;

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::chat::{ChatMessage, ChatResponse};
use crate::error::LlmError;
use crate::stream::ChatStream;

/// The core trait every LLM provider implements.
///
/// `Provider` uses native async-fn-in-traits (Rust 2024 edition).
/// Implementations are plain `async fn`s — no `#[async_trait]` needed.
///
/// Cross-cutting concerns like retries, rate-limiting, and logging are
/// handled by the interceptor system, keeping individual backends focused
/// on HTTP mapping.
///
/// # Object safety
///
/// `Provider` is **not** object-safe because AFIT returns `impl Future`.
/// When you need dynamic dispatch (e.g. `Box<dyn _>` or `Arc<dyn _>`),
/// use [`DynProvider`] instead — every `Provider` automatically
/// implements `DynProvider` via a blanket impl.
pub trait Provider: Send + Sync {
    /// Sends a chat completion request and returns the full response.
    fn generate(
        &self,
        params: &ChatParams,
    ) -> impl Future<Output = Result<ChatResponse, LlmError>> + Send;

    /// Sends a chat completion request and returns a stream of events.
    ///
    /// The returned [`ChatStream`] yields [`StreamEvent`](crate::StreamEvent)s
    /// as they arrive from the provider.
    fn stream(
        &self,
        params: &ChatParams,
    ) -> impl Future<Output = Result<ChatStream, LlmError>> + Send;

    /// Returns static metadata describing this provider instance.
    fn metadata(&self) -> ProviderMetadata;
}

/// Object-safe counterpart of [`Provider`] for dynamic dispatch.
///
/// You rarely implement this directly — the blanket
/// `impl<T: Provider> DynProvider for T` does it for you. Use this
/// when you need to erase the concrete provider type:
///
/// ```rust,no_run
/// use llm_stack::{DynProvider, ChatParams};
///
/// async fn ask(provider: &dyn DynProvider, question: &str) -> String {
///     let params = ChatParams {
///         messages: vec![llm_stack::ChatMessage::user(question)],
///         ..Default::default()
///     };
///     let resp = provider.generate_boxed(&params).await.unwrap();
///     format!("{resp:?}")
/// }
/// ```
pub trait DynProvider: Send + Sync {
    /// Boxed-future version of [`Provider::generate`].
    fn generate_boxed<'a>(
        &'a self,
        params: &'a ChatParams,
    ) -> Pin<Box<dyn Future<Output = Result<ChatResponse, LlmError>> + Send + 'a>>;

    /// Boxed-future version of [`Provider::stream`].
    fn stream_boxed<'a>(
        &'a self,
        params: &'a ChatParams,
    ) -> Pin<Box<dyn Future<Output = Result<ChatStream, LlmError>> + Send + 'a>>;

    /// Returns static metadata describing this provider instance.
    fn metadata(&self) -> ProviderMetadata;
}

impl<T: Provider> DynProvider for T {
    fn generate_boxed<'a>(
        &'a self,
        params: &'a ChatParams,
    ) -> Pin<Box<dyn Future<Output = Result<ChatResponse, LlmError>> + Send + 'a>> {
        Box::pin(self.generate(params))
    }

    fn stream_boxed<'a>(
        &'a self,
        params: &'a ChatParams,
    ) -> Pin<Box<dyn Future<Output = Result<ChatStream, LlmError>> + Send + 'a>> {
        Box::pin(self.stream(params))
    }

    fn metadata(&self) -> ProviderMetadata {
        Provider::metadata(self)
    }
}

/// Describes a provider instance: its name, model, and capabilities.
///
/// The `name` field uses [`Cow<'static, str>`] so that built-in
/// providers can use `"anthropic"` (zero-alloc) while dynamic or
/// user-created providers can use owned strings.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProviderMetadata {
    /// Human-readable provider name (e.g. `"anthropic"`, `"openai"`).
    pub name: Cow<'static, str>,
    /// The model identifier (e.g. `"claude-sonnet-4-20250514"`).
    pub model: String,
    /// Maximum context window size in tokens.
    pub context_window: u64,
    /// Feature flags indicating what this provider supports.
    pub capabilities: HashSet<Capability>,
}

/// A feature that a provider may or may not support.
///
/// Callers can inspect [`ProviderMetadata::capabilities`] to decide
/// whether to include tool definitions, request structured output, etc.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[non_exhaustive]
pub enum Capability {
    /// Function/tool calling.
    Tools,
    /// JSON Schema–constrained output.
    StructuredOutput,
    /// Extended chain-of-thought reasoning.
    Reasoning,
    /// Image (and potentially video) understanding.
    Vision,
    /// Prompt caching for reduced latency and cost.
    Caching,
}

/// Parameters for a chat completion request.
///
/// Most fields are optional — at minimum you need [`messages`](Self::messages).
/// Use struct-update syntax for concise construction:
///
/// ```rust
/// use llm_stack::{ChatParams, ChatMessage};
///
/// let params = ChatParams {
///     messages: vec![ChatMessage::user("Hello")],
///     max_tokens: Some(256),
///     temperature: Some(0.7),
///     ..Default::default()
/// };
/// ```
///
/// # Serialization
///
/// `ChatParams` implements `Serialize` / `Deserialize` for logging and
/// request replay. The [`timeout`](Self::timeout) and
/// [`extra_headers`](Self::extra_headers) fields are skipped during
/// serialization because they are transport-layer concerns, not part of
/// the logical request.
///
/// # Equality
///
/// `PartialEq` is structural. The `extra_headers` field uses
/// `http::HeaderMap`, whose equality comparison is order-sensitive for
/// multi-valued headers. In practice this only matters in tests via
/// [`MockProvider::recorded_calls`](crate::mock::MockProvider::recorded_calls).
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
pub struct ChatParams {
    /// The conversation history.
    pub messages: Vec<ChatMessage>,
    /// Tool definitions the model may invoke.
    pub tools: Option<Vec<ToolDefinition>>,
    /// Controls whether and how the model uses tools.
    pub tool_choice: Option<ToolChoice>,
    /// Sampling temperature (0.0 = deterministic, higher = more random).
    pub temperature: Option<f32>,
    /// Upper bound on generated tokens.
    pub max_tokens: Option<u32>,
    /// System prompt (used by providers that accept it separately from
    /// the message list).
    pub system: Option<String>,
    /// Token budget for chain-of-thought reasoning, if the provider
    /// supports [`Capability::Reasoning`].
    pub reasoning_budget: Option<u32>,
    /// JSON Schema that the model's output must conform to.
    pub structured_output: Option<JsonSchema>,
    /// Per-request timeout. Skipped during serialization.
    #[serde(skip)]
    pub timeout: Option<Duration>,
    /// Extra HTTP headers to send with this request. Skipped during
    /// serialization.
    #[serde(skip)]
    pub extra_headers: Option<http::HeaderMap>,
    /// Arbitrary key-value pairs forwarded to the provider. Useful for
    /// provider-specific features that don't have a dedicated field.
    pub metadata: HashMap<String, Value>,
}

/// Controls whether the model should use tools and, if so, which ones.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[non_exhaustive]
pub enum ToolChoice {
    /// The model decides whether to call a tool.
    Auto,
    /// The model must not call any tools.
    None,
    /// The model must call at least one tool.
    Required,
    /// The model must call this specific tool.
    Specific(String),
}

/// Retry behavior predicate type.
///
/// Receives the error message and returns `true` if the error is retryable.
pub type RetryPredicate = std::sync::Arc<dyn Fn(&str) -> bool + Send + Sync>;

/// Configuration for automatic retries when a tool execution fails.
///
/// When a tool handler returns an error and retry configuration is present,
/// the registry will automatically retry with exponential backoff.
///
/// # Example
///
/// ```rust
/// use llm_stack::ToolRetryConfig;
/// use std::time::Duration;
///
/// let config = ToolRetryConfig {
///     max_retries: 3,
///     initial_backoff: Duration::from_millis(100),
///     max_backoff: Duration::from_secs(5),
///     backoff_multiplier: 2.0,
///     jitter: 0.5,
///     retry_if: None, // Retry all errors
/// };
/// ```
#[derive(Clone)]
pub struct ToolRetryConfig {
    /// Maximum retry attempts (not counting initial try). Default: 3.
    pub max_retries: u32,
    /// Initial backoff duration before first retry. Default: 100ms.
    pub initial_backoff: Duration,
    /// Maximum backoff duration cap. Default: 5 seconds.
    pub max_backoff: Duration,
    /// Backoff multiplier for exponential growth. Default: 2.0.
    pub backoff_multiplier: f64,
    /// Jitter factor (0.0 to 1.0) applied to backoff. Default: 0.5.
    pub jitter: f64,
    /// Optional predicate to determine if an error is retryable.
    /// Receives the error message. If `None`, all errors are retried.
    pub retry_if: Option<RetryPredicate>,
}

impl Default for ToolRetryConfig {
    fn default() -> Self {
        Self {
            max_retries: 3,
            initial_backoff: Duration::from_millis(100),
            max_backoff: Duration::from_secs(5),
            backoff_multiplier: 2.0,
            jitter: 0.5,
            retry_if: None,
        }
    }
}

impl std::fmt::Debug for ToolRetryConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ToolRetryConfig")
            .field("max_retries", &self.max_retries)
            .field("initial_backoff", &self.initial_backoff)
            .field("max_backoff", &self.max_backoff)
            .field("backoff_multiplier", &self.backoff_multiplier)
            .field("jitter", &self.jitter)
            .field("has_retry_if", &self.retry_if.is_some())
            .finish()
    }
}

impl PartialEq for ToolRetryConfig {
    fn eq(&self, other: &Self) -> bool {
        self.max_retries == other.max_retries
            && self.initial_backoff == other.initial_backoff
            && self.max_backoff == other.max_backoff
            && self.backoff_multiplier == other.backoff_multiplier
            && self.jitter == other.jitter
            && self.retry_if.is_some() == other.retry_if.is_some()
    }
}

/// A tool the model can invoke during generation.
///
/// Providers translate this into their native tool format.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ToolDefinition {
    /// The tool's name, used to match [`ToolCall::name`](crate::ToolCall::name).
    pub name: String,
    /// Human-readable description shown to the model so it knows when
    /// to use this tool.
    pub description: String,
    /// JSON Schema describing the tool's expected input.
    pub parameters: JsonSchema,
    /// Optional retry configuration for this tool.
    /// When present, failed tool executions will be automatically retried.
    #[serde(skip)]
    pub retry: Option<ToolRetryConfig>,
}

/// A JSON Schema document used for structured output or tool parameters.
///
/// Wraps a [`serde_json::Value`] and provides validation via the
/// [`jsonschema`] crate. The inner value is private — use
/// [`as_value`](Self::as_value) for read access.
///
/// # Construction
///
/// ```rust
/// use llm_stack::JsonSchema;
///
/// // From a raw JSON value
/// let schema = JsonSchema::new(serde_json::json!({
///     "type": "object",
///     "properties": { "name": { "type": "string" } },
///     "required": ["name"]
/// }));
///
/// // From a Rust type that implements schemars::JsonSchema
/// // let schema = JsonSchema::from_type::<MyStruct>()?;
/// ```
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct JsonSchema(Value);

impl JsonSchema {
    /// Creates a schema from a raw JSON value.
    pub fn new(schema: Value) -> Self {
        Self(schema)
    }

    /// Returns a reference to the underlying JSON value.
    pub fn as_value(&self) -> &Value {
        &self.0
    }

    /// Derives a JSON Schema from a Rust type that implements
    /// [`schemars::JsonSchema`].
    ///
    /// Returns an error if the generated schema cannot be serialized to
    /// `serde_json::Value` (should not happen in practice).
    ///
    /// Requires the `schema` feature (enabled by default).
    #[cfg(feature = "schema")]
    pub fn from_type<T: schemars::JsonSchema>() -> Result<Self, serde_json::Error> {
        let schema = schemars::schema_for!(T);
        let value = serde_json::to_value(schema)?;
        Ok(Self(value))
    }

    /// Validates `value` against this schema.
    ///
    /// Requires the `schema` feature (enabled by default).
    ///
    /// Returns `Ok(())` if validation passes, or
    /// [`LlmError::SchemaValidation`] with details on failure. Returns
    /// [`LlmError::InvalidRequest`] if the schema itself is malformed.
    #[cfg(feature = "schema")]
    pub fn validate(&self, value: &Value) -> Result<(), LlmError> {
        let validator = jsonschema::validator_for(&self.0)
            .map_err(|e| LlmError::InvalidRequest(format!("invalid JSON schema: {e}")))?;
        let errors: Vec<String> = validator
            .iter_errors(value)
            .map(|e| e.to_string())
            .collect();
        if errors.is_empty() {
            Ok(())
        } else {
            Err(LlmError::SchemaValidation {
                message: errors.join("; "),
                schema: self.0.clone(),
                actual: value.clone(),
            })
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // --- Capability tests ---

    #[test]
    fn test_capability_hash_set() {
        let caps: HashSet<Capability> = HashSet::from([
            Capability::Tools,
            Capability::StructuredOutput,
            Capability::Reasoning,
            Capability::Vision,
            Capability::Caching,
        ]);
        assert_eq!(caps.len(), 5);
    }

    #[test]
    fn test_capability_copy() {
        let c = Capability::Tools;
        let c2 = c; // Copy
        assert_eq!(c, c2);
    }

    #[test]
    fn test_capability_serde_roundtrip() {
        let cap = Capability::Tools;
        let json = serde_json::to_string(&cap).unwrap();
        let back: Capability = serde_json::from_str(&json).unwrap();
        assert_eq!(cap, back);
    }

    // --- ProviderMetadata tests ---

    #[test]
    fn test_provider_metadata_clone_eq() {
        let m = ProviderMetadata {
            name: "mock".into(),
            model: "test-model".into(),
            context_window: 128_000,
            capabilities: HashSet::from([Capability::Tools]),
        };
        assert_eq!(m, m.clone());
    }

    #[test]
    fn test_provider_metadata_owned_name() {
        let name = String::from("custom-provider");
        let m = ProviderMetadata {
            name: Cow::Owned(name),
            model: "test".into(),
            context_window: 4096,
            capabilities: HashSet::new(),
        };
        assert_eq!(m.name, "custom-provider");
    }

    // --- ChatParams tests ---

    #[test]
    fn test_chat_params_defaults() {
        let p = ChatParams::default();
        assert!(p.messages.is_empty());
        assert!(p.tools.is_none());
        assert!(p.tool_choice.is_none());
        assert!(p.temperature.is_none());
        assert!(p.max_tokens.is_none());
        assert!(p.system.is_none());
        assert!(p.reasoning_budget.is_none());
        assert!(p.structured_output.is_none());
        assert!(p.timeout.is_none());
        assert!(p.extra_headers.is_none());
        assert!(p.metadata.is_empty());
    }

    #[test]
    fn test_chat_params_full() {
        let p = ChatParams {
            messages: vec![ChatMessage::user("hi")],
            tools: Some(vec![]),
            tool_choice: Some(ToolChoice::Auto),
            temperature: Some(0.7),
            max_tokens: Some(1024),
            system: Some("you are helpful".into()),
            reasoning_budget: Some(2048),
            structured_output: Some(JsonSchema::new(serde_json::json!({"type": "object"}))),
            timeout: Some(Duration::from_secs(30)),
            extra_headers: Some(http::HeaderMap::new()),
            metadata: HashMap::from([("key".into(), serde_json::json!("val"))]),
        };
        assert_eq!(p.messages.len(), 1);
        assert!(p.tools.is_some());
        assert_eq!(p.temperature, Some(0.7));
    }

    // --- ToolChoice tests ---

    #[test]
    fn test_tool_choice_all_variants() {
        let variants = [
            ToolChoice::Auto,
            ToolChoice::None,
            ToolChoice::Required,
            ToolChoice::Specific("my_tool".into()),
        ];
        for v in &variants {
            assert_eq!(*v, v.clone());
        }
    }

    #[test]
    fn test_tool_choice_serde_roundtrip() {
        let tc = ToolChoice::Specific("search".into());
        let json = serde_json::to_string(&tc).unwrap();
        let back: ToolChoice = serde_json::from_str(&json).unwrap();
        assert_eq!(tc, back);
    }

    // --- JsonSchema tests ---

    #[test]
    fn test_json_schema_from_raw() {
        let schema = JsonSchema::new(serde_json::json!({"type": "object"}));
        assert_eq!(*schema.as_value(), serde_json::json!({"type": "object"}));
    }

    #[cfg(feature = "schema")]
    #[test]
    fn test_json_schema_from_type_simple() {
        #[derive(schemars::JsonSchema)]
        struct Foo {
            #[allow(dead_code)]
            x: i32,
        }
        let schema = JsonSchema::from_type::<Foo>().unwrap();
        let props = schema
            .as_value()
            .get("properties")
            .expect("should have properties");
        assert!(props.get("x").is_some());
    }

    #[cfg(feature = "schema")]
    #[test]
    fn test_json_schema_validate_valid() {
        let schema = JsonSchema::new(serde_json::json!({
            "type": "object",
            "properties": {
                "x": {"type": "integer"}
            },
            "required": ["x"]
        }));
        assert!(schema.validate(&serde_json::json!({"x": 42})).is_ok());
    }

    #[cfg(feature = "schema")]
    #[test]
    fn test_json_schema_validate_missing_field() {
        let schema = JsonSchema::new(serde_json::json!({
            "type": "object",
            "properties": {
                "x": {"type": "integer"}
            },
            "required": ["x"]
        }));
        let result = schema.validate(&serde_json::json!({}));
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            LlmError::SchemaValidation { .. }
        ));
    }

    #[cfg(feature = "schema")]
    #[test]
    fn test_json_schema_validate_wrong_type() {
        let schema = JsonSchema::new(serde_json::json!({
            "type": "object",
            "properties": {
                "x": {"type": "integer"}
            },
            "required": ["x"]
        }));
        let result = schema.validate(&serde_json::json!({"x": "not a number"}));
        assert!(result.is_err());
    }

    #[cfg(feature = "schema")]
    #[test]
    fn test_json_schema_validate_invalid_schema() {
        let schema = JsonSchema::new(serde_json::json!({"type": "bogus_not_a_type"}));
        let result = schema.validate(&serde_json::json!(42));
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), LlmError::InvalidRequest(_)));
    }

    #[test]
    fn test_json_schema_clone_eq() {
        let s = JsonSchema::new(serde_json::json!({"type": "string"}));
        assert_eq!(s, s.clone());
    }

    #[test]
    fn test_json_schema_serde_roundtrip() {
        let s = JsonSchema::new(
            serde_json::json!({"type": "object", "properties": {"x": {"type": "integer"}}}),
        );
        let json = serde_json::to_string(&s).unwrap();
        let back: JsonSchema = serde_json::from_str(&json).unwrap();
        assert_eq!(s, back);
    }

    #[test]
    fn test_tool_definition_serde_roundtrip() {
        let td = ToolDefinition {
            name: "search".into(),
            description: "Search the web".into(),
            parameters: JsonSchema::new(serde_json::json!({"type": "object"})),
            retry: None,
        };
        let json = serde_json::to_string(&td).unwrap();
        let back: ToolDefinition = serde_json::from_str(&json).unwrap();
        assert_eq!(td, back);
    }

    #[test]
    fn test_provider_metadata_serde_roundtrip() {
        let m = ProviderMetadata {
            name: "anthropic".into(),
            model: "claude-sonnet-4".into(),
            context_window: 200_000,
            capabilities: HashSet::from([Capability::Tools, Capability::Vision]),
        };
        let json = serde_json::to_string(&m).unwrap();
        let back: ProviderMetadata = serde_json::from_str(&json).unwrap();
        assert_eq!(m, back);
    }

    #[test]
    fn test_chat_params_serde_roundtrip_with_metadata() {
        let p = ChatParams {
            messages: vec![ChatMessage::user("hi")],
            metadata: HashMap::from([
                ("provider_key".into(), serde_json::json!("abc123")),
                ("flags".into(), serde_json::json!({"stream": true})),
            ]),
            ..Default::default()
        };
        let json = serde_json::to_string(&p).unwrap();
        let back: ChatParams = serde_json::from_str(&json).unwrap();
        assert_eq!(back.metadata.len(), 2);
        assert_eq!(back.metadata["provider_key"], serde_json::json!("abc123"));
        assert_eq!(back.metadata["flags"], serde_json::json!({"stream": true}));
    }

    #[test]
    fn test_chat_params_serde_roundtrip_skips_timeout_and_headers() {
        let p = ChatParams {
            messages: vec![ChatMessage::user("hi")],
            temperature: Some(0.7),
            timeout: Some(Duration::from_secs(30)),
            extra_headers: Some(http::HeaderMap::new()),
            ..Default::default()
        };
        let json = serde_json::to_string(&p).unwrap();
        let back: ChatParams = serde_json::from_str(&json).unwrap();
        // timeout and extra_headers are skipped
        assert_eq!(back.timeout, None);
        assert_eq!(back.extra_headers, None);
        // other fields survive
        assert_eq!(back.messages.len(), 1);
        assert_eq!(back.temperature, Some(0.7));
    }
}
