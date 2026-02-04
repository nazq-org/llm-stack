//! Mock provider for testing.
//!
//! [`MockProvider`] is a queue-based fake that lets tests control
//! exactly what responses and errors a provider returns, without
//! touching the network. It implements [`Provider`],
//! so it works anywhere a real provider does — including through
//! [`DynProvider`](crate::DynProvider) via the blanket impl.
//!
//! # Usage
//!
//! ```rust,no_run
//! use llm_stack_core::mock::{MockProvider, MockError};
//! use llm_stack_core::{Provider, ChatParams, ChatResponse, ContentBlock, StopReason, Usage};
//! use std::collections::{HashMap, HashSet};
//!
//! # async fn example() {
//! let mock = MockProvider::new(llm_stack_core::ProviderMetadata {
//!     name: "test".into(),
//!     model: "test-model".into(),
//!     context_window: 4096,
//!     capabilities: HashSet::new(),
//! });
//!
//! mock.queue_response(ChatResponse {
//!     content: vec![ContentBlock::Text("Hello!".into())],
//!     usage: Usage::default(),
//!     stop_reason: StopReason::EndTurn,
//!     model: "test-model".into(),
//!     metadata: HashMap::new(),
//! });
//!
//! let resp = mock.generate(&ChatParams::default()).await.unwrap();
//! assert_eq!(mock.recorded_calls().len(), 1);
//! # }
//! ```
//!
//! # Why `MockError` instead of `LlmError`?
//!
//! [`LlmError`] contains `Box<dyn Error>` and is not
//! `Clone`, so it can't be stored in a queue. [`MockError`] mirrors the
//! common error variants in a cloneable form and converts to `LlmError`
//! at dequeue time.

use std::collections::VecDeque;
use std::fmt;
use std::sync::{Arc, Mutex};

use crate::chat::ChatResponse;
use crate::error::LlmError;
use crate::provider::{ChatParams, Provider, ProviderMetadata};
use crate::stream::{ChatStream, StreamEvent};

/// A queue-based mock provider for unit and integration tests.
///
/// Push responses with [`queue_response`](Self::queue_response) and
/// errors with [`queue_error`](Self::queue_error). Each call to
/// `generate` or `stream` pops from the front of the respective queue.
///
/// Every call records its [`ChatParams`] for later assertion via
/// [`recorded_calls`](Self::recorded_calls).
///
/// # Panics
///
/// [`generate`](Provider::generate) panics if the response queue is empty.
/// [`stream`](Provider::stream) panics if the stream queue is empty.
pub struct MockProvider {
    responses: Mutex<VecDeque<Result<ChatResponse, MockError>>>,
    stream_responses: Mutex<VecDeque<Result<Vec<StreamEvent>, MockError>>>,
    meta: ProviderMetadata,
    calls: Arc<Mutex<Vec<ChatParams>>>,
}

/// Cloneable error subset for mock queuing.
///
/// [`LlmError`] contains `Box<dyn Error>` and is not `Clone`, so it
/// can't be queued directly. This type mirrors the common error
/// variants. Use [`queue_error`](MockProvider::queue_error) to enqueue
/// one — it is converted to `LlmError` when dequeued.
#[derive(Debug, Clone)]
pub enum MockError {
    /// Maps to [`LlmError::Http`].
    Http {
        /// HTTP status code, if any.
        status: Option<http::StatusCode>,
        /// Error message.
        message: String,
        /// Whether the error is retryable.
        retryable: bool,
    },
    /// Maps to [`LlmError::Auth`].
    Auth(String),
    /// Maps to [`LlmError::InvalidRequest`].
    InvalidRequest(String),
    /// Maps to [`LlmError::Provider`].
    Provider {
        /// Provider error code.
        code: String,
        /// Error message.
        message: String,
        /// Whether the error is retryable.
        retryable: bool,
    },
    /// Maps to [`LlmError::Timeout`].
    Timeout {
        /// Elapsed milliseconds.
        elapsed_ms: u64,
    },
    /// Maps to [`LlmError::ResponseFormat`].
    ResponseFormat {
        /// What went wrong during parsing.
        message: String,
        /// The raw response body.
        raw: String,
    },
    /// Maps to [`LlmError::SchemaValidation`].
    SchemaValidation {
        /// Validation error messages.
        message: String,
        /// The schema that was violated.
        schema: serde_json::Value,
        /// The value that failed validation.
        actual: serde_json::Value,
    },
    /// Maps to [`LlmError::RetryExhausted`].
    RetryExhausted {
        /// How many attempts were made.
        attempts: u32,
        /// Description of the last error.
        last_error_message: String,
    },
}

impl MockError {
    fn into_llm_error(self) -> LlmError {
        match self {
            Self::Http {
                status,
                message,
                retryable,
            } => LlmError::Http {
                status,
                message,
                retryable,
            },
            Self::Auth(msg) => LlmError::Auth(msg),
            Self::InvalidRequest(msg) => LlmError::InvalidRequest(msg),
            Self::Provider {
                code,
                message,
                retryable,
            } => LlmError::Provider {
                code,
                message,
                retryable,
            },
            Self::Timeout { elapsed_ms } => LlmError::Timeout { elapsed_ms },
            Self::ResponseFormat { message, raw } => LlmError::ResponseFormat { message, raw },
            Self::SchemaValidation {
                message,
                schema,
                actual,
            } => LlmError::SchemaValidation {
                message,
                schema,
                actual,
            },
            Self::RetryExhausted {
                attempts,
                last_error_message,
            } => LlmError::RetryExhausted {
                attempts,
                last_error: Box::new(LlmError::InvalidRequest(last_error_message)),
            },
        }
    }
}

impl fmt::Debug for MockProvider {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let response_len = self.responses.lock().unwrap().len();
        let stream_len = self.stream_responses.lock().unwrap().len();
        let call_count = self.calls.lock().unwrap().len();
        f.debug_struct("MockProvider")
            .field("meta", &self.meta)
            .field("queued_responses", &response_len)
            .field("queued_streams", &stream_len)
            .field("recorded_calls", &call_count)
            .finish()
    }
}

impl MockProvider {
    /// Creates a new mock with the given metadata and empty queues.
    pub fn new(meta: ProviderMetadata) -> Self {
        Self {
            responses: Mutex::new(VecDeque::new()),
            stream_responses: Mutex::new(VecDeque::new()),
            meta,
            calls: Arc::new(Mutex::new(Vec::new())),
        }
    }

    /// Enqueues a successful response for the next `generate` call.
    pub fn queue_response(&self, response: ChatResponse) -> &Self {
        self.responses.lock().unwrap().push_back(Ok(response));
        self
    }

    /// Enqueues an error for the next `generate` call.
    pub fn queue_error(&self, error: MockError) -> &Self {
        self.responses.lock().unwrap().push_back(Err(error));
        self
    }

    /// Enqueues stream events for the next `stream` call.
    pub fn queue_stream(&self, events: Vec<StreamEvent>) -> &Self {
        self.stream_responses.lock().unwrap().push_back(Ok(events));
        self
    }

    /// Enqueues an error for the next `stream` call.
    ///
    /// The error is returned from `stream()` itself (before any events
    /// are yielded), simulating failures like authentication errors or
    /// network issues that prevent the stream from starting.
    pub fn queue_stream_error(&self, error: MockError) -> &Self {
        self.stream_responses.lock().unwrap().push_back(Err(error));
        self
    }

    /// Returns a clone of all `ChatParams` passed to `generate` or
    /// `stream`, in call order.
    pub fn recorded_calls(&self) -> Vec<ChatParams> {
        self.calls.lock().unwrap().clone()
    }

    fn record_call(&self, params: &ChatParams) {
        self.calls.lock().unwrap().push(params.clone());
    }
}

impl Provider for MockProvider {
    async fn generate(&self, params: &ChatParams) -> Result<ChatResponse, LlmError> {
        self.record_call(params);
        let result = self
            .responses
            .lock()
            .unwrap()
            .pop_front()
            .expect("MockProvider: no queued responses remaining");
        result.map_err(MockError::into_llm_error)
    }

    async fn stream(&self, params: &ChatParams) -> Result<ChatStream, LlmError> {
        self.record_call(params);
        let result = self
            .stream_responses
            .lock()
            .unwrap()
            .pop_front()
            .expect("MockProvider: no queued stream responses remaining");
        let events = result.map_err(MockError::into_llm_error)?;
        let stream = futures::stream::iter(events.into_iter().map(Ok));
        Ok(Box::pin(stream))
    }

    fn metadata(&self) -> ProviderMetadata {
        self.meta.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chat::{ContentBlock, StopReason};
    use crate::provider::{Capability, DynProvider};
    use crate::test_helpers::sample_response;
    use futures::StreamExt;
    use std::collections::HashSet;

    fn test_metadata() -> ProviderMetadata {
        ProviderMetadata {
            name: "mock".into(),
            model: "test-model".into(),
            context_window: 128_000,
            capabilities: HashSet::from([Capability::Tools, Capability::StructuredOutput]),
        }
    }

    #[tokio::test]
    async fn test_mock_generate_returns_queued() {
        let mock = MockProvider::new(test_metadata());
        let resp = sample_response("test");
        mock.queue_response(resp.clone());

        let result = mock.generate(&ChatParams::default()).await.unwrap();
        assert_eq!(result, resp);
    }

    #[tokio::test]
    async fn test_mock_generate_multiple_queued() {
        let mock = MockProvider::new(test_metadata());
        mock.queue_response(sample_response("first"));
        mock.queue_response(sample_response("second"));
        mock.queue_response(sample_response("third"));

        let r1 = mock.generate(&ChatParams::default()).await.unwrap();
        let r2 = mock.generate(&ChatParams::default()).await.unwrap();
        let r3 = mock.generate(&ChatParams::default()).await.unwrap();

        assert_eq!(r1.content, vec![ContentBlock::Text("first".into())]);
        assert_eq!(r2.content, vec![ContentBlock::Text("second".into())]);
        assert_eq!(r3.content, vec![ContentBlock::Text("third".into())]);
    }

    #[tokio::test]
    async fn test_mock_generate_error() {
        let mock = MockProvider::new(test_metadata());
        mock.queue_error(MockError::Auth("bad key".into()));

        let result = mock.generate(&ChatParams::default()).await;
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), LlmError::Auth(_)));
    }

    #[tokio::test]
    async fn test_mock_generate_mixed_queue() {
        let mock = MockProvider::new(test_metadata());
        mock.queue_response(sample_response("ok"));
        mock.queue_error(MockError::Timeout { elapsed_ms: 5000 });
        mock.queue_response(sample_response("ok again"));

        let r1 = mock.generate(&ChatParams::default()).await;
        let r2 = mock.generate(&ChatParams::default()).await;
        let r3 = mock.generate(&ChatParams::default()).await;

        assert!(r1.is_ok());
        assert!(r2.is_err());
        assert!(r3.is_ok());
    }

    #[tokio::test]
    #[should_panic(expected = "no queued responses")]
    async fn test_mock_generate_empty_queue_panics() {
        let mock = MockProvider::new(test_metadata());
        let _ = mock.generate(&ChatParams::default()).await;
    }

    #[tokio::test]
    async fn test_mock_stream_returns_events() {
        let mock = MockProvider::new(test_metadata());
        mock.queue_stream(vec![
            StreamEvent::TextDelta("hello".into()),
            StreamEvent::TextDelta(" world".into()),
            StreamEvent::Done {
                stop_reason: StopReason::EndTurn,
            },
        ]);

        let stream = mock.stream(&ChatParams::default()).await.unwrap();
        let events: Vec<_> = stream.collect().await;
        assert_eq!(events.len(), 3);
        assert!(events.iter().all(Result::is_ok));
    }

    #[tokio::test]
    async fn test_mock_stream_error() {
        let mock = MockProvider::new(test_metadata());
        mock.queue_stream_error(MockError::Auth("bad token".into()));

        let result = mock.stream(&ChatParams::default()).await;
        assert!(result.is_err());
        let err = result.err().unwrap();
        assert!(matches!(err, LlmError::Auth(_)));
    }

    #[tokio::test]
    async fn test_mock_stream_empty_events() {
        let mock = MockProvider::new(test_metadata());
        mock.queue_stream(vec![]);

        let stream = mock.stream(&ChatParams::default()).await.unwrap();
        let events: Vec<_> = stream.collect().await;
        assert!(events.is_empty());
    }

    #[tokio::test]
    async fn test_mock_records_calls() {
        let mock = MockProvider::new(test_metadata());
        mock.queue_response(sample_response("a"));
        mock.queue_response(sample_response("b"));
        mock.queue_response(sample_response("c"));

        let _ = mock.generate(&ChatParams::default()).await;
        let _ = mock.generate(&ChatParams::default()).await;
        let _ = mock.generate(&ChatParams::default()).await;

        assert_eq!(mock.recorded_calls().len(), 3);
    }

    #[tokio::test]
    async fn test_mock_records_params_accurately() {
        let mock = MockProvider::new(test_metadata());
        mock.queue_response(sample_response("ok"));

        let params = ChatParams {
            temperature: Some(0.5),
            system: Some("be nice".into()),
            ..Default::default()
        };
        let _ = mock.generate(&params).await;

        let recorded = mock.recorded_calls();
        assert_eq!(recorded[0].temperature, Some(0.5));
        assert_eq!(recorded[0].system, Some("be nice".into()));
    }

    #[test]
    fn test_mock_metadata_returns_configured() {
        let meta = test_metadata();
        let mock = MockProvider::new(meta.clone());
        assert_eq!(Provider::metadata(&mock), meta);
    }

    #[tokio::test]
    async fn test_mock_concurrent_access() {
        let mock = Arc::new(MockProvider::new(test_metadata()));
        for _ in 0..10 {
            mock.queue_response(sample_response("ok"));
        }

        let mut handles = Vec::new();
        for _ in 0..10 {
            let m = mock.clone();
            handles.push(tokio::spawn(async move {
                m.generate(&ChatParams::default()).await.unwrap()
            }));
        }

        for h in handles {
            h.await.unwrap();
        }

        assert_eq!(mock.recorded_calls().len(), 10);
    }

    // --- DynProvider tests (through mock) ---

    #[tokio::test]
    async fn test_dyn_provider_blanket_impl() {
        let mock = MockProvider::new(test_metadata());
        mock.queue_response(sample_response("hello"));

        let dyn_provider: &dyn DynProvider = &mock;
        let params = ChatParams::default();
        let result = dyn_provider.generate_boxed(&params).await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_dyn_provider_error_propagation() {
        let mock = MockProvider::new(test_metadata());
        mock.queue_error(MockError::Http {
            status: Some(http::StatusCode::TOO_MANY_REQUESTS),
            message: "rate limited".into(),
            retryable: true,
        });

        let dyn_provider: &dyn DynProvider = &mock;
        let result = dyn_provider.generate_boxed(&ChatParams::default()).await;
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), LlmError::Http { .. }));
    }

    #[tokio::test]
    async fn test_dyn_provider_stream_blanket() {
        let mock = MockProvider::new(test_metadata());
        mock.queue_stream(vec![
            StreamEvent::TextDelta("hi".into()),
            StreamEvent::Done {
                stop_reason: StopReason::EndTurn,
            },
        ]);

        let dyn_provider: &dyn DynProvider = &mock;
        let params = ChatParams::default();
        let stream = dyn_provider.stream_boxed(&params).await.unwrap();
        let events: Vec<_> = stream.collect().await;
        assert_eq!(events.len(), 2);
    }

    #[tokio::test]
    async fn test_dyn_provider_metadata_matches() {
        let mock = MockProvider::new(test_metadata());
        let dyn_provider: &dyn DynProvider = &mock;
        assert_eq!(Provider::metadata(&mock), dyn_provider.metadata());
    }

    #[tokio::test]
    async fn test_dyn_provider_boxed_storage() {
        let mock = MockProvider::new(test_metadata());
        mock.queue_response(sample_response("from box"));

        let boxed: Box<dyn DynProvider> = Box::new(mock);
        let result = boxed.generate_boxed(&ChatParams::default()).await.unwrap();
        assert_eq!(result.content, vec![ContentBlock::Text("from box".into())]);
    }

    #[test]
    fn test_mock_provider_debug() {
        let mock = MockProvider::new(test_metadata());
        mock.queue_response(sample_response("a"));
        mock.queue_stream(vec![StreamEvent::TextDelta("hi".into())]);

        let debug = format!("{mock:?}");
        assert!(debug.contains("MockProvider"));
        assert!(debug.contains("queued_responses: 1"));
        assert!(debug.contains("queued_streams: 1"));
        assert!(debug.contains("recorded_calls: 0"));
    }

    #[test]
    fn test_provider_is_object_safe() {
        let f1: fn(&dyn DynProvider) = |_| {};
        let f2: fn(Box<dyn DynProvider>) = |_| {};
        // Suppress unused variable warnings
        let _ = (f1, f2);
    }

    #[tokio::test]
    async fn test_mock_error_into_llm_error_all_variants() {
        let variants: Vec<(MockError, &str)> = vec![
            (MockError::InvalidRequest("bad".into()), "InvalidRequest"),
            (
                MockError::Provider {
                    code: "e1".into(),
                    message: "fail".into(),
                    retryable: false,
                },
                "Provider",
            ),
            (
                MockError::ResponseFormat {
                    message: "bad json".into(),
                    raw: "{}".into(),
                },
                "ResponseFormat",
            ),
            (
                MockError::SchemaValidation {
                    message: "missing field".into(),
                    schema: serde_json::json!({"type": "object"}),
                    actual: serde_json::json!(42),
                },
                "SchemaValidation",
            ),
            (
                MockError::RetryExhausted {
                    attempts: 3,
                    last_error_message: "timed out".into(),
                },
                "RetryExhausted",
            ),
        ];

        for (mock_err, label) in variants {
            let mock = MockProvider::new(test_metadata());
            mock.queue_error(mock_err);
            let result = mock.generate(&ChatParams::default()).await;
            assert!(result.is_err(), "{label} should produce error");
            let err = result.unwrap_err();
            let debug = format!("{err:?}");
            assert!(
                debug.contains(label),
                "expected {label} in error debug: {debug}"
            );
        }
    }
}
