//! Pre-built helpers for testing code that uses `llm-core` types.
//!
//! Available when the `test-utils` feature is enabled, allowing
//! downstream crates to reuse these utilities in their own test
//! suites. Also compiled during `#[cfg(test)]` for this crate's
//! own tests. Provides sample responses, message shorthands, stream
//! collectors, and a quick [`MockProvider`] factory.

use std::collections::{HashMap, HashSet};

use futures::StreamExt;

use crate::chat::{ChatMessage, ChatResponse, ContentBlock, StopReason, ToolCall};
use crate::error::LlmError;
use crate::mock::MockProvider;
use crate::provider::{Capability, ProviderMetadata};
use crate::stream::{ChatStream, StreamEvent};
use crate::usage::Usage;

/// Builds a [`ChatResponse`] with a single text block and default usage.
pub fn sample_response(text: &str) -> ChatResponse {
    ChatResponse {
        content: vec![ContentBlock::Text(text.into())],
        usage: sample_usage(),
        stop_reason: StopReason::EndTurn,
        model: "test-model".into(),
        metadata: HashMap::new(),
    }
}

/// Builds a [`ChatResponse`] containing the given tool calls.
pub fn sample_tool_response(calls: Vec<ToolCall>) -> ChatResponse {
    ChatResponse {
        content: calls.into_iter().map(ContentBlock::ToolCall).collect(),
        usage: sample_usage(),
        stop_reason: StopReason::ToolUse,
        model: "test-model".into(),
        metadata: HashMap::new(),
    }
}

/// Returns a [`Usage`] with 100 input / 50 output tokens.
pub fn sample_usage() -> Usage {
    Usage {
        input_tokens: 100,
        output_tokens: 50,
        reasoning_tokens: None,
        cache_read_tokens: None,
        cache_write_tokens: None,
    }
}

/// Shorthand for [`ChatMessage::user`].
pub fn user_msg(text: &str) -> ChatMessage {
    ChatMessage::user(text)
}

/// Shorthand for [`ChatMessage::assistant`].
pub fn assistant_msg(text: &str) -> ChatMessage {
    ChatMessage::assistant(text)
}

/// Shorthand for [`ChatMessage::system`].
pub fn system_msg(text: &str) -> ChatMessage {
    ChatMessage::system(text)
}

/// Shorthand for [`ChatMessage::tool_result`].
pub fn tool_result_msg(tool_call_id: &str, content: &str) -> ChatMessage {
    ChatMessage::tool_result(tool_call_id, content)
}

/// Collect stream events, returning results including errors.
pub async fn collect_stream_results(stream: ChatStream) -> Vec<Result<StreamEvent, LlmError>> {
    stream.collect::<Vec<_>>().await
}

/// Collect stream events, panicking on any error.
/// Use `collect_stream_results` when testing error scenarios.
pub async fn collect_stream(stream: ChatStream) -> Vec<StreamEvent> {
    stream
        .collect::<Vec<_>>()
        .await
        .into_iter()
        .map(|r| r.expect("stream event should be Ok"))
        .collect()
}

/// Creates a [`MockProvider`] with the given name, model, and [`Capability::Tools`].
pub fn mock_for(provider_name: &str, model: &str) -> MockProvider {
    MockProvider::new(ProviderMetadata {
        name: provider_name.to_owned().into(),
        model: model.into(),
        context_window: 128_000,
        capabilities: HashSet::from([Capability::Tools]),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chat::ChatRole;

    #[test]
    fn test_sample_response_is_valid() {
        let r = sample_response("hello");
        assert_eq!(r.content, vec![ContentBlock::Text("hello".into())]);
        assert_eq!(r.stop_reason, StopReason::EndTurn);
    }

    #[test]
    fn test_sample_tool_response() {
        let calls = vec![ToolCall {
            id: "tc_1".into(),
            name: "search".into(),
            arguments: serde_json::json!({"q": "rust"}),
        }];
        let r = sample_tool_response(calls);
        assert_eq!(r.stop_reason, StopReason::ToolUse);
        assert!(!r.content.is_empty());
    }

    #[test]
    fn test_sample_usage_fields() {
        let u = sample_usage();
        assert!(u.input_tokens > 0);
        assert!(u.output_tokens > 0);
    }

    #[test]
    fn test_helper_messages() {
        assert_eq!(user_msg("hi").role, ChatRole::User);
        assert_eq!(assistant_msg("hello").role, ChatRole::Assistant);
        assert_eq!(system_msg("be nice").role, ChatRole::System);
        assert_eq!(tool_result_msg("tc_1", "42").role, ChatRole::Tool);
    }

    #[tokio::test]
    async fn test_collect_stream_happy() {
        let events = vec![
            Ok(StreamEvent::TextDelta("hello".into())),
            Ok(StreamEvent::Done {
                stop_reason: StopReason::EndTurn,
            }),
        ];
        let stream: ChatStream = Box::pin(futures::stream::iter(events));
        let collected = collect_stream(stream).await;
        assert_eq!(collected.len(), 2);
    }

    #[tokio::test]
    async fn test_collect_stream_empty() {
        let stream: ChatStream = Box::pin(futures::stream::iter(Vec::<
            Result<StreamEvent, LlmError>,
        >::new()));
        let collected = collect_stream(stream).await;
        assert!(collected.is_empty());
    }

    #[tokio::test]
    async fn test_collect_stream_results_with_errors() {
        let events = vec![
            Ok(StreamEvent::TextDelta("hello".into())),
            Err(LlmError::Http {
                status: Some(http::StatusCode::INTERNAL_SERVER_ERROR),
                message: "server error".into(),
                retryable: true,
            }),
        ];
        let stream: ChatStream = Box::pin(futures::stream::iter(events));
        let collected = collect_stream_results(stream).await;
        assert_eq!(collected.len(), 2);
        assert!(collected[0].is_ok());
        assert!(collected[1].is_err());
    }

    #[test]
    fn test_mock_for_helper() {
        let mock = mock_for("anthropic", "claude-sonnet-4");
        let meta = crate::provider::Provider::metadata(&mock);
        assert_eq!(meta.name, "anthropic");
        assert_eq!(meta.model, "claude-sonnet-4");
    }

    #[test]
    fn test_mock_for_custom_name() {
        let mock = mock_for("my-custom-provider", "gpt-4");
        let meta = crate::provider::Provider::metadata(&mock);
        assert_eq!(meta.name, "my-custom-provider");
    }
}
