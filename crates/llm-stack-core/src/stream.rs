//! Streaming response types.
//!
//! When a provider streams its response, it yields a sequence of
//! [`StreamEvent`]s through a [`ChatStream`]. Events arrive
//! incrementally — text deltas, tool-call fragments, and finally a
//! [`Done`](StreamEvent::Done) event with the stop reason.
//!
//! # Collecting a stream
//!
//! ```rust,no_run
//! use futures::StreamExt;
//! use llm_stack_core::{ChatStream, StreamEvent};
//!
//! async fn print_stream(mut stream: ChatStream) {
//!     while let Some(event) = stream.next().await {
//!         match event {
//!             Ok(StreamEvent::TextDelta(text)) => print!("{text}"),
//!             Ok(StreamEvent::Done { stop_reason }) => {
//!                 println!("\n[done: {stop_reason:?}]");
//!             }
//!             Err(e) => eprintln!("stream error: {e}"),
//!             _ => {} // handle other events as needed
//!         }
//!     }
//! }
//! ```
//!
//! # Tool-call reassembly
//!
//! Tool calls arrive in three phases:
//! 1. [`ToolCallStart`](StreamEvent::ToolCallStart) — announces the
//!    call's `id` and `name`.
//! 2. [`ToolCallDelta`](StreamEvent::ToolCallDelta) — one or more JSON
//!    argument fragments, streamed as they're generated.
//! 3. [`ToolCallComplete`](StreamEvent::ToolCallComplete) — the fully
//!    assembled [`ToolCall`] with parsed arguments.
//!
//! The `index` field on each event identifies which call it belongs to
//! when the model invokes multiple tools in parallel.

use std::pin::Pin;

use futures::Stream;
use serde::{Deserialize, Serialize};

use crate::chat::{StopReason, ToolCall};
use crate::error::LlmError;
use crate::usage::Usage;

/// A pinned, boxed, `Send` stream of [`StreamEvent`] results.
///
/// This type alias keeps signatures readable. Consume it with
/// [`StreamExt`](futures::StreamExt) from the `futures` crate.
pub type ChatStream = Pin<Box<dyn Stream<Item = Result<StreamEvent, LlmError>> + Send>>;

/// An incremental event emitted during a streaming response.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[non_exhaustive]
pub enum StreamEvent {
    /// A fragment of the model's text output.
    TextDelta(String),
    /// A fragment of the model's reasoning (chain-of-thought) output.
    ReasoningDelta(String),
    /// Announces that a new tool call has started.
    ToolCallStart {
        /// Zero-based index identifying this call when multiple tools
        /// are invoked in parallel.
        index: u32,
        /// Provider-assigned identifier linking start → deltas → complete.
        id: String,
        /// The name of the tool being called.
        name: String,
    },
    /// A JSON fragment of the tool call's arguments.
    ToolCallDelta {
        /// The tool-call index this delta belongs to.
        index: u32,
        /// A chunk of the JSON arguments string.
        json_chunk: String,
    },
    /// The fully assembled tool call, ready to execute.
    ToolCallComplete {
        /// The tool-call index this completion corresponds to.
        index: u32,
        /// The complete, parsed tool call.
        call: ToolCall,
    },
    /// Token usage information for the request so far.
    Usage(Usage),
    /// The stream has ended.
    Done {
        /// Why the model stopped generating.
        stop_reason: StopReason,
    },
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures::StreamExt;

    #[test]
    fn test_stream_event_text_delta_eq() {
        let a = StreamEvent::TextDelta("hello".into());
        let b = a.clone();
        assert_eq!(a, b);
    }

    #[test]
    fn test_stream_event_reasoning_delta_eq() {
        let a = StreamEvent::ReasoningDelta("step 1".into());
        assert_eq!(a, a.clone());
    }

    #[test]
    fn test_stream_event_tool_call_start() {
        let e = StreamEvent::ToolCallStart {
            index: 0,
            id: "tc_1".into(),
            name: "search".into(),
        };
        assert!(matches!(
            &e,
            StreamEvent::ToolCallStart { index: 0, id, name }
                if id == "tc_1" && name == "search"
        ));
    }

    #[test]
    fn test_stream_event_tool_call_delta() {
        let e = StreamEvent::ToolCallDelta {
            index: 0,
            json_chunk: r#"{"q":"#.into(),
        };
        assert_eq!(e, e.clone());
    }

    #[test]
    fn test_stream_event_tool_call_complete() {
        let call = ToolCall {
            id: "tc_1".into(),
            name: "search".into(),
            arguments: serde_json::json!({"q": "rust"}),
        };
        let e = StreamEvent::ToolCallComplete {
            index: 0,
            call: call.clone(),
        };
        assert!(matches!(
            &e,
            StreamEvent::ToolCallComplete { call: c, .. } if *c == call
        ));
    }

    #[test]
    fn test_stream_event_usage() {
        let e = StreamEvent::Usage(Usage {
            input_tokens: 100,
            output_tokens: 50,
            ..Usage::default()
        });
        assert_eq!(e, e.clone());
    }

    #[test]
    fn test_stream_event_done() {
        let e = StreamEvent::Done {
            stop_reason: StopReason::EndTurn,
        };
        assert!(matches!(
            &e,
            StreamEvent::Done { stop_reason } if *stop_reason == StopReason::EndTurn
        ));
    }

    #[tokio::test]
    async fn test_chat_stream_collect() {
        let events = vec![
            Ok(StreamEvent::TextDelta("hello ".into())),
            Ok(StreamEvent::TextDelta("world".into())),
            Ok(StreamEvent::Done {
                stop_reason: StopReason::EndTurn,
            }),
        ];
        let stream: ChatStream = Box::pin(futures::stream::iter(events));
        let collected: Vec<_> = stream.collect().await;
        assert_eq!(collected.len(), 3);
        assert!(collected.iter().all(Result::is_ok));
    }

    #[tokio::test]
    async fn test_chat_stream_error_mid_stream() {
        let events = vec![
            Ok(StreamEvent::TextDelta("hello".into())),
            Ok(StreamEvent::TextDelta(" world".into())),
            Err(LlmError::Http {
                status: Some(http::StatusCode::INTERNAL_SERVER_ERROR),
                message: "server error".into(),
                retryable: true,
            }),
        ];
        let stream: ChatStream = Box::pin(futures::stream::iter(events));
        let collected: Vec<_> = stream.collect().await;
        assert_eq!(collected.len(), 3);
        assert!(collected[0].is_ok());
        assert!(collected[1].is_ok());
        assert!(collected[2].is_err());
    }

    #[test]
    fn test_chat_stream_is_send() {
        fn assert_send<T: Send>() {}
        assert_send::<ChatStream>();
    }
}
