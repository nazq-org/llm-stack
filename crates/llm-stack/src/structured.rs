//! Structured output — typed LLM responses with schema validation.
//!
//! This module provides high-level functions that combine schema derivation,
//! LLM generation, validation, and deserialization into a single call.
//!
//! # Non-streaming
//!
//! [`generate_object`] sends a request with a JSON Schema constraint and
//! returns a fully validated, deserialized `T`:
//!
//! ```rust,no_run
//! use llm_stack::structured::{generate_object, GenerateObjectConfig};
//! use llm_stack::{ChatMessage, ChatParams};
//! use serde::Deserialize;
//!
//! #[derive(Deserialize, schemars::JsonSchema)]
//! struct Person {
//!     name: String,
//!     age: u32,
//! }
//!
//! # async fn example(provider: &dyn llm_stack::DynProvider) -> Result<(), llm_stack::LlmError> {
//! let params = ChatParams {
//!     messages: vec![ChatMessage::user("Generate a person named Alice aged 30")],
//!     ..Default::default()
//! };
//!
//! let result = generate_object::<Person>(provider, params, GenerateObjectConfig::default()).await?;
//! assert_eq!(result.value.name, "Alice");
//! # Ok(())
//! # }
//! ```
//!
//! # Streaming
//!
//! [`stream_object_async`] yields [`PartialObject<T>`] events as JSON tokens
//! arrive. Each event carries the accumulated JSON so far, and the final
//! event includes the fully deserialized object.

use serde::de::DeserializeOwned;
use serde_json::Value;

use crate::chat::ChatResponse;
use crate::error::LlmError;
use crate::provider::{ChatParams, DynProvider, JsonSchema};
use crate::stream::{ChatStream, StreamEvent};
use crate::usage::Usage;

// ── GenerateObjectConfig ────────────────────────────────────────────

/// Configuration for [`generate_object`] and [`stream_object_async`].
#[derive(Debug, Clone)]
pub struct GenerateObjectConfig {
    /// Maximum number of attempts (initial + retries). Default: 1
    /// (single attempt, no retries). Set to 2 for one retry, etc.
    pub max_attempts: u32,
    /// Whether to include the schema in the system prompt as a fallback
    /// for providers that don't support native structured output.
    /// Default: false.
    pub system_prompt_fallback: bool,
}

impl Default for GenerateObjectConfig {
    fn default() -> Self {
        Self {
            max_attempts: 1,
            system_prompt_fallback: false,
        }
    }
}

// ── PartialObject ───────────────────────────────────────────────────

/// A partially-received structured object from a streaming response.
#[derive(Debug, Clone)]
pub struct PartialObject<T> {
    /// The raw JSON accumulated so far (may be incomplete).
    pub partial_json: String,
    /// `Some(T)` once the stream completes and the object passes
    /// validation + deserialization.
    pub complete: Option<T>,
    /// Accumulated usage across retries and the final successful call.
    pub usage: Usage,
}

// ── GenerateObjectResult ────────────────────────────────────────────

/// The result of a successful [`generate_object`] call.
#[derive(Debug)]
pub struct GenerateObjectResult<T> {
    /// The deserialized, validated object.
    pub value: T,
    /// The raw JSON string returned by the model.
    pub raw_json: String,
    /// Token usage for the successful attempt (and all retries).
    pub usage: Usage,
    /// How many attempts were made (1 = succeeded on first try).
    pub attempts: u32,
}

// ── generate_object ─────────────────────────────────────────────────

/// Generates a typed object from the LLM with schema validation.
///
/// 1. Derives a JSON Schema from `T` (via [`schemars`])
/// 2. Sets `structured_output` on `ChatParams`
/// 3. Calls the provider
/// 4. Parses the response text as JSON
/// 5. Validates against the schema
/// 6. Deserializes to `T`
///
/// Retries up to `config.max_attempts` times on parse/validation failures.
/// On retry, the model's invalid response and the validation error are
/// appended to the message history so the model can self-correct.
///
/// # Errors
///
/// Returns [`LlmError`] if:
/// - `max_attempts` is 0
/// - The schema cannot be derived from `T`
/// - The provider returns an error (propagated immediately, not retried)
/// - All attempts fail validation (returns the last validation error)
#[cfg(feature = "schema")]
pub async fn generate_object<T>(
    provider: &dyn DynProvider,
    mut params: ChatParams,
    config: GenerateObjectConfig,
) -> Result<GenerateObjectResult<T>, LlmError>
where
    T: DeserializeOwned + schemars::JsonSchema,
{
    if config.max_attempts == 0 {
        return Err(LlmError::InvalidRequest(
            "max_attempts must be at least 1".into(),
        ));
    }

    let schema = JsonSchema::from_type::<T>()
        .map_err(|e| LlmError::InvalidRequest(format!("failed to derive JSON schema: {e}")))?;

    params.structured_output = Some(schema.clone());

    if config.system_prompt_fallback {
        inject_schema_prompt(&mut params, &schema);
    }

    let mut total_usage = Usage::default();
    let mut last_error = None;

    for attempt in 1..=config.max_attempts {
        let response = provider.generate_boxed(&params).await?;
        total_usage += response.usage.clone();

        match extract_and_validate::<T>(&response, &schema) {
            Ok((value, raw_json)) => {
                return Ok(GenerateObjectResult {
                    value,
                    raw_json,
                    usage: total_usage,
                    attempts: attempt,
                });
            }
            Err(e) => {
                last_error = Some(e);
                if attempt < config.max_attempts {
                    append_retry_feedback(
                        &mut params,
                        &response,
                        last_error.as_ref().expect("set on previous line"),
                    );
                }
            }
        }
    }

    Err(last_error.expect("max_attempts >= 1 guarantees at least one iteration"))
}

/// Async streaming variant of [`generate_object`].
///
/// Awaits the provider's stream creation and returns the [`ChatStream`]
/// directly. Yields [`StreamEvent`]s which can be collected via
/// [`collect_stream_object`].
///
/// The `max_attempts` field in `config` is ignored — retry logic must
/// be implemented by the caller.
#[cfg(feature = "schema")]
pub async fn stream_object_async<T>(
    provider: &dyn DynProvider,
    mut params: ChatParams,
    config: GenerateObjectConfig,
) -> Result<ChatStream, LlmError>
where
    T: DeserializeOwned + schemars::JsonSchema,
{
    let schema = JsonSchema::from_type::<T>()
        .map_err(|e| LlmError::InvalidRequest(format!("failed to derive JSON schema: {e}")))?;

    params.structured_output = Some(schema.clone());

    if config.system_prompt_fallback {
        inject_schema_prompt(&mut params, &schema);
    }

    provider.stream_boxed(&params).await
}

/// Collects a [`ChatStream`] into a [`PartialObject<T>`].
///
/// Accumulates text deltas, then validates and deserializes the result
/// when the stream completes. Use this with [`stream_object_async`].
///
/// Unlike [`generate_object`], this function does not retry on
/// validation failures — errors are returned immediately.
#[cfg(feature = "schema")]
pub async fn collect_stream_object<T>(
    mut stream: ChatStream,
    schema: &JsonSchema,
) -> Result<PartialObject<T>, LlmError>
where
    T: DeserializeOwned,
{
    use futures::StreamExt;

    let mut json_buf = String::new();
    let mut usage = Usage::default();

    while let Some(event) = stream.next().await {
        match event? {
            StreamEvent::TextDelta(text) => json_buf.push_str(&text),
            StreamEvent::Usage(u) => usage += u,
            StreamEvent::Done { .. } => break,
            _ => {}
        }
    }

    if json_buf.is_empty() {
        return Err(LlmError::ResponseFormat {
            message: "model returned no text content for structured output".into(),
            raw: String::new(),
        });
    }

    // Parse, validate, deserialize
    let value: Value = serde_json::from_str(&json_buf).map_err(|e| LlmError::ResponseFormat {
        message: format!("invalid JSON in structured output: {e}"),
        raw: json_buf.clone(),
    })?;

    schema.validate(&value)?;

    let typed: T = serde_json::from_value(value).map_err(|e| LlmError::ResponseFormat {
        message: format!("failed to deserialize structured output: {e}"),
        raw: json_buf.clone(),
    })?;

    Ok(PartialObject {
        partial_json: json_buf,
        complete: Some(typed),
        usage,
    })
}

// ── Helpers ─────────────────────────────────────────────────────────

/// Extracts the text from a `ChatResponse`, parses it as JSON,
/// validates against the schema, and deserializes to `T`.
#[cfg(feature = "schema")]
fn extract_and_validate<T: DeserializeOwned>(
    response: &ChatResponse,
    schema: &JsonSchema,
) -> Result<(T, String), LlmError> {
    let raw_json = response.text().ok_or_else(|| LlmError::ResponseFormat {
        message: "model returned no text content for structured output".into(),
        raw: String::new(),
    })?;

    let value: Value = serde_json::from_str(raw_json).map_err(|e| LlmError::ResponseFormat {
        message: format!("invalid JSON in structured output: {e}"),
        raw: raw_json.to_string(),
    })?;

    schema.validate(&value)?;

    let typed: T = serde_json::from_value(value).map_err(|e| LlmError::ResponseFormat {
        message: format!("failed to deserialize structured output: {e}"),
        raw: raw_json.to_string(),
    })?;

    Ok((typed, raw_json.to_string()))
}

/// Injects a system prompt instructing the model to respond with JSON
/// matching the given schema.
fn inject_schema_prompt(params: &mut ChatParams, schema: &JsonSchema) {
    let schema_json = serde_json::to_string_pretty(schema.as_value())
        .expect("serializing Value to JSON cannot fail");

    let instruction = format!(
        "You must respond with valid JSON that conforms to this JSON Schema:\n\
         ```json\n{schema_json}\n```\n\
         Respond ONLY with the JSON object. No markdown, no explanation."
    );

    match &mut params.system {
        Some(existing) => {
            existing.push_str("\n\n");
            existing.push_str(&instruction);
        }
        None => params.system = Some(instruction),
    }
}

/// Appends the model's failed response and the validation error as
/// feedback messages, so the model can retry.
fn append_retry_feedback(params: &mut ChatParams, response: &ChatResponse, error: &LlmError) {
    use crate::chat::ChatMessage;

    // Add the model's response as an assistant message
    params
        .messages
        .push(ChatMessage::assistant(response.text().unwrap_or("")));

    // Add the error as a user message asking for correction
    params.messages.push(ChatMessage::user(format!(
        "Your response did not pass validation: {error}\n\
         Please try again with valid JSON that conforms to the schema."
    )));
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chat::{ChatMessage, ContentBlock, StopReason};
    use crate::test_helpers::{mock_for, sample_usage};
    use serde::Deserialize;
    use serde_json::json;
    use std::collections::HashMap;

    #[derive(Debug, Deserialize, PartialEq, schemars::JsonSchema)]
    struct Person {
        name: String,
        age: u32,
    }

    #[derive(Debug, Deserialize, PartialEq, schemars::JsonSchema)]
    struct Coord {
        x: f64,
        y: f64,
    }

    fn json_response(json_str: &str) -> ChatResponse {
        ChatResponse {
            content: vec![ContentBlock::Text(json_str.into())],
            usage: sample_usage(),
            stop_reason: StopReason::EndTurn,
            model: "test-model".into(),
            metadata: HashMap::new(),
        }
    }

    // ── generate_object tests ──────────────────────────────────────

    #[tokio::test]
    async fn test_generate_object_happy_path() {
        let mock = mock_for("test", "test-model");
        mock.queue_response(json_response(r#"{"name": "Alice", "age": 30}"#));

        let params = ChatParams {
            messages: vec![ChatMessage::user("Generate a person")],
            ..Default::default()
        };

        let result: GenerateObjectResult<Person> =
            generate_object(&mock, params, GenerateObjectConfig::default())
                .await
                .unwrap();

        assert_eq!(
            result.value,
            Person {
                name: "Alice".into(),
                age: 30
            }
        );
        assert_eq!(result.attempts, 1);
        assert_eq!(result.raw_json, r#"{"name": "Alice", "age": 30}"#);
    }

    #[tokio::test]
    async fn test_generate_object_invalid_json() {
        let mock = mock_for("test", "test-model");
        mock.queue_response(json_response("not valid json"));

        let params = ChatParams {
            messages: vec![ChatMessage::user("Generate a person")],
            ..Default::default()
        };

        let err = generate_object::<Person>(&mock, params, GenerateObjectConfig::default())
            .await
            .unwrap_err();

        assert!(matches!(err, LlmError::ResponseFormat { .. }));
    }

    #[tokio::test]
    async fn test_generate_object_schema_violation() {
        let mock = mock_for("test", "test-model");
        // Missing required field "age"
        mock.queue_response(json_response(r#"{"name": "Alice"}"#));

        let params = ChatParams {
            messages: vec![ChatMessage::user("Generate a person")],
            ..Default::default()
        };

        let err = generate_object::<Person>(&mock, params, GenerateObjectConfig::default())
            .await
            .unwrap_err();

        assert!(matches!(err, LlmError::SchemaValidation { .. }));
    }

    #[tokio::test]
    async fn test_generate_object_wrong_type() {
        let mock = mock_for("test", "test-model");
        // age is string instead of number
        mock.queue_response(json_response(r#"{"name": "Alice", "age": "thirty"}"#));

        let params = ChatParams {
            messages: vec![ChatMessage::user("Generate a person")],
            ..Default::default()
        };

        let err = generate_object::<Person>(&mock, params, GenerateObjectConfig::default())
            .await
            .unwrap_err();

        assert!(matches!(err, LlmError::SchemaValidation { .. }));
    }

    #[tokio::test]
    async fn test_generate_object_retry_succeeds_on_second_attempt() {
        let mock = mock_for("test", "test-model");
        // First attempt: invalid
        mock.queue_response(json_response(r#"{"name": "Alice"}"#));
        // Second attempt: valid
        mock.queue_response(json_response(r#"{"name": "Alice", "age": 30}"#));

        let params = ChatParams {
            messages: vec![ChatMessage::user("Generate a person")],
            ..Default::default()
        };
        let config = GenerateObjectConfig {
            max_attempts: 2,
            ..Default::default()
        };

        let result: GenerateObjectResult<Person> =
            generate_object(&mock, params, config).await.unwrap();

        assert_eq!(
            result.value,
            Person {
                name: "Alice".into(),
                age: 30
            }
        );
        assert_eq!(result.attempts, 2);
        // Usage should include both attempts
        assert_eq!(result.usage.input_tokens, 200);
    }

    #[tokio::test]
    async fn test_generate_object_retry_exhausted() {
        let mock = mock_for("test", "test-model");
        // All attempts fail
        mock.queue_response(json_response(r#"{"name": "Alice"}"#));
        mock.queue_response(json_response(r#"{"name": "Bob"}"#));
        mock.queue_response(json_response(r#"{"name": "Charlie"}"#));

        let params = ChatParams {
            messages: vec![ChatMessage::user("Generate a person")],
            ..Default::default()
        };
        let config = GenerateObjectConfig {
            max_attempts: 3,
            ..Default::default()
        };

        let err = generate_object::<Person>(&mock, params, config)
            .await
            .unwrap_err();

        assert!(matches!(err, LlmError::SchemaValidation { .. }));
    }

    #[tokio::test]
    async fn test_generate_object_no_text_content() {
        let mock = mock_for("test", "test-model");
        // Response with no text (empty content)
        mock.queue_response(ChatResponse {
            content: vec![],
            usage: sample_usage(),
            stop_reason: StopReason::EndTurn,
            model: "test-model".into(),
            metadata: HashMap::new(),
        });

        let params = ChatParams {
            messages: vec![ChatMessage::user("Generate a person")],
            ..Default::default()
        };

        let err = generate_object::<Person>(&mock, params, GenerateObjectConfig::default())
            .await
            .unwrap_err();

        assert!(matches!(err, LlmError::ResponseFormat { .. }));
    }

    #[tokio::test]
    async fn test_generate_object_sets_structured_output() {
        let mock = mock_for("test", "test-model");
        mock.queue_response(json_response(r#"{"x": 1.0, "y": 2.0}"#));

        let params = ChatParams {
            messages: vec![ChatMessage::user("Generate coords")],
            ..Default::default()
        };

        let _result: GenerateObjectResult<Coord> =
            generate_object(&mock, params, GenerateObjectConfig::default())
                .await
                .unwrap();

        // Verify the provider received structured_output
        let recorded = mock.recorded_calls();
        assert!(recorded[0].structured_output.is_some());
    }

    #[tokio::test]
    async fn test_generate_object_system_prompt_fallback() {
        let mock = mock_for("test", "test-model");
        mock.queue_response(json_response(r#"{"x": 1.0, "y": 2.0}"#));

        let params = ChatParams {
            messages: vec![ChatMessage::user("Generate coords")],
            ..Default::default()
        };
        let config = GenerateObjectConfig {
            system_prompt_fallback: true,
            ..Default::default()
        };

        let _result: GenerateObjectResult<Coord> =
            generate_object(&mock, params, config).await.unwrap();

        let recorded = mock.recorded_calls();
        assert!(recorded[0].system.is_some());
        assert!(recorded[0].system.as_ref().unwrap().contains("JSON Schema"));
    }

    #[tokio::test]
    async fn test_generate_object_system_prompt_appends() {
        let mock = mock_for("test", "test-model");
        mock.queue_response(json_response(r#"{"x": 1.0, "y": 2.0}"#));

        let params = ChatParams {
            messages: vec![ChatMessage::user("Generate coords")],
            system: Some("You are a helpful assistant.".into()),
            ..Default::default()
        };
        let config = GenerateObjectConfig {
            system_prompt_fallback: true,
            ..Default::default()
        };

        let _result: GenerateObjectResult<Coord> =
            generate_object(&mock, params, config).await.unwrap();

        let recorded = mock.recorded_calls();
        let system = recorded[0].system.as_ref().unwrap();
        assert!(system.starts_with("You are a helpful assistant."));
        assert!(system.contains("JSON Schema"));
    }

    #[tokio::test]
    async fn test_generate_object_retry_appends_feedback() {
        let mock = mock_for("test", "test-model");
        mock.queue_response(json_response(r#"{"name": "Alice"}"#));
        mock.queue_response(json_response(r#"{"name": "Alice", "age": 30}"#));

        let params = ChatParams {
            messages: vec![ChatMessage::user("Generate a person")],
            ..Default::default()
        };
        let config = GenerateObjectConfig {
            max_attempts: 2,
            ..Default::default()
        };

        let _result: GenerateObjectResult<Person> =
            generate_object(&mock, params, config).await.unwrap();

        // Second call should have feedback messages
        let recorded = mock.recorded_calls();
        assert!(recorded[1].messages.len() > 1);
        // Should contain the model's failed response + error feedback
        let last_user_msg = recorded[1]
            .messages
            .iter()
            .rfind(|m| m.role == crate::chat::ChatRole::User)
            .unwrap();
        let text = last_user_msg.content.iter().find_map(|b| match b {
            ContentBlock::Text(t) => Some(t.as_str()),
            _ => None,
        });
        assert!(text.unwrap().contains("did not pass validation"));
    }

    // ── collect_stream_object tests ────────────────────────────────

    #[tokio::test]
    async fn test_collect_stream_object_happy_path() {
        let schema = JsonSchema::from_type::<Person>().unwrap();
        let events = vec![
            Ok(StreamEvent::TextDelta(r#"{"name":"#.into())),
            Ok(StreamEvent::TextDelta(r#" "Alice", "age": 30}"#.into())),
            Ok(StreamEvent::Usage(sample_usage())),
            Ok(StreamEvent::Done {
                stop_reason: StopReason::EndTurn,
            }),
        ];
        let stream: ChatStream = Box::pin(futures::stream::iter(events));

        let result: PartialObject<Person> = collect_stream_object(stream, &schema).await.unwrap();

        assert_eq!(
            result.complete.unwrap(),
            Person {
                name: "Alice".into(),
                age: 30
            }
        );
        assert_eq!(result.partial_json, r#"{"name": "Alice", "age": 30}"#);
        assert_eq!(result.usage.input_tokens, 100);
    }

    #[tokio::test]
    async fn test_collect_stream_object_invalid_json() {
        let schema = JsonSchema::from_type::<Person>().unwrap();
        let events = vec![
            Ok(StreamEvent::TextDelta("not json".into())),
            Ok(StreamEvent::Done {
                stop_reason: StopReason::EndTurn,
            }),
        ];
        let stream: ChatStream = Box::pin(futures::stream::iter(events));

        let err = collect_stream_object::<Person>(stream, &schema)
            .await
            .unwrap_err();

        assert!(matches!(err, LlmError::ResponseFormat { .. }));
    }

    #[tokio::test]
    async fn test_collect_stream_object_schema_violation() {
        let schema = JsonSchema::from_type::<Person>().unwrap();
        let events = vec![
            Ok(StreamEvent::TextDelta(r#"{"name": "Alice"}"#.into())),
            Ok(StreamEvent::Done {
                stop_reason: StopReason::EndTurn,
            }),
        ];
        let stream: ChatStream = Box::pin(futures::stream::iter(events));

        let err = collect_stream_object::<Person>(stream, &schema)
            .await
            .unwrap_err();

        assert!(matches!(err, LlmError::SchemaValidation { .. }));
    }

    #[tokio::test]
    async fn test_collect_stream_object_empty_stream() {
        let schema = JsonSchema::from_type::<Person>().unwrap();
        let events = vec![Ok(StreamEvent::Done {
            stop_reason: StopReason::EndTurn,
        })];
        let stream: ChatStream = Box::pin(futures::stream::iter(events));

        let err = collect_stream_object::<Person>(stream, &schema)
            .await
            .unwrap_err();

        assert!(matches!(err, LlmError::ResponseFormat { .. }));
    }

    #[tokio::test]
    async fn test_collect_stream_object_mid_stream_error() {
        let schema = JsonSchema::from_type::<Person>().unwrap();
        let events = vec![
            Ok(StreamEvent::TextDelta(r#"{"name"#.into())),
            Err(LlmError::Http {
                status: None,
                message: "connection lost".into(),
                retryable: true,
            }),
        ];
        let stream: ChatStream = Box::pin(futures::stream::iter(events));

        let err = collect_stream_object::<Person>(stream, &schema)
            .await
            .unwrap_err();

        assert!(matches!(err, LlmError::Http { .. }));
    }

    // ── stream_object_async tests ──────────────────────────────────

    #[tokio::test]
    async fn test_stream_object_async_sets_structured_output() {
        let mock = mock_for("test", "test-model");
        mock.queue_stream(vec![
            StreamEvent::TextDelta(r#"{"x": 1.0, "y": 2.0}"#.into()),
            StreamEvent::Done {
                stop_reason: StopReason::EndTurn,
            },
        ]);

        let params = ChatParams {
            messages: vec![ChatMessage::user("Generate coords")],
            ..Default::default()
        };

        let _stream = stream_object_async::<Coord>(&mock, params, GenerateObjectConfig::default())
            .await
            .unwrap();

        let recorded = mock.recorded_calls();
        assert!(recorded[0].structured_output.is_some());
    }

    #[tokio::test]
    async fn test_stream_object_async_end_to_end() {
        let mock = mock_for("test", "test-model");
        mock.queue_stream(vec![
            StreamEvent::TextDelta(r#"{"x": 1.23"#.into()),
            StreamEvent::TextDelta(r#", "y": 4.56}"#.into()),
            StreamEvent::Done {
                stop_reason: StopReason::EndTurn,
            },
        ]);

        let params = ChatParams {
            messages: vec![ChatMessage::user("Generate coords")],
            ..Default::default()
        };

        let schema = JsonSchema::from_type::<Coord>().unwrap();
        let stream = stream_object_async::<Coord>(&mock, params, GenerateObjectConfig::default())
            .await
            .unwrap();

        let result: PartialObject<Coord> = collect_stream_object(stream, &schema).await.unwrap();

        let coord = result.complete.unwrap();
        assert!((coord.x - 1.23).abs() < 0.001);
        assert!((coord.y - 4.56).abs() < 0.001);
    }

    // ── Helper tests ───────────────────────────────────────────────

    #[test]
    fn test_inject_schema_prompt_new() {
        let schema = JsonSchema::new(json!({"type": "object"}));
        let mut params = ChatParams::default();
        inject_schema_prompt(&mut params, &schema);

        assert!(params.system.as_ref().unwrap().contains("JSON Schema"));
        assert!(params.system.as_ref().unwrap().contains("ONLY"));
    }

    #[test]
    fn test_inject_schema_prompt_appends() {
        let schema = JsonSchema::new(json!({"type": "object"}));
        let mut params = ChatParams {
            system: Some("Be helpful.".into()),
            ..Default::default()
        };
        inject_schema_prompt(&mut params, &schema);

        let system = params.system.unwrap();
        assert!(system.starts_with("Be helpful."));
        assert!(system.contains("JSON Schema"));
    }

    #[test]
    fn test_generate_object_config_default() {
        let config = GenerateObjectConfig::default();
        assert_eq!(config.max_attempts, 1);
        assert!(!config.system_prompt_fallback);
    }

    #[test]
    fn test_partial_object_debug() {
        let po: PartialObject<Person> = PartialObject {
            partial_json: "{}".into(),
            complete: None,
            usage: Usage::default(),
        };
        let debug = format!("{po:?}");
        assert!(debug.contains("PartialObject"));
    }

    #[test]
    fn test_generate_object_result_debug() {
        let result = GenerateObjectResult {
            value: Person {
                name: "Alice".into(),
                age: 30,
            },
            raw_json: "{}".into(),
            usage: Usage::default(),
            attempts: 1,
        };
        let debug = format!("{result:?}");
        assert!(debug.contains("GenerateObjectResult"));
    }

    #[tokio::test]
    async fn test_generate_object_zero_attempts_errors() {
        let mock = mock_for("test", "test-model");
        let params = ChatParams {
            messages: vec![ChatMessage::user("Generate a person")],
            ..Default::default()
        };
        let config = GenerateObjectConfig {
            max_attempts: 0,
            ..Default::default()
        };

        let err = generate_object::<Person>(&mock, params, config)
            .await
            .unwrap_err();

        assert!(matches!(err, LlmError::InvalidRequest(_)));
    }

    #[tokio::test]
    async fn test_generate_object_provider_error_propagates() {
        let mock = mock_for("test", "test-model");
        // First response fails validation, second is a provider error
        mock.queue_response(json_response(r#"{"name": "Alice"}"#));
        mock.queue_error(crate::mock::MockError::Http {
            status: Some(http::StatusCode::SERVICE_UNAVAILABLE),
            message: "service down".into(),
            retryable: true,
        });

        let params = ChatParams {
            messages: vec![ChatMessage::user("Generate a person")],
            ..Default::default()
        };
        let config = GenerateObjectConfig {
            max_attempts: 2,
            ..Default::default()
        };

        let err = generate_object::<Person>(&mock, params, config)
            .await
            .unwrap_err();

        // Provider errors propagate immediately, not wrapped
        assert!(matches!(err, LlmError::Http { .. }));
    }
}
