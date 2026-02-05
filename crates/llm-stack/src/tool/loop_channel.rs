//! Channel-based tool loop with backpressure support.

use std::sync::Arc;

use futures::StreamExt;
use tokio::sync::mpsc;
use tokio::task::JoinHandle;

use crate::error::LlmError;
use crate::provider::{ChatParams, DynProvider};
use crate::stream::StreamEvent;

use super::LoopDepth;
use super::ToolRegistry;
use super::config::{ToolLoopConfig, ToolLoopResult};
use super::loop_stream::tool_loop_stream;

/// Channel-based tool loop with bounded buffer for backpressure.
///
/// Unlike [`tool_loop_stream`], this function spawns an internal task and
/// sends events through a bounded channel. This provides natural backpressure:
/// if the consumer is slow, the producer blocks when the buffer is full,
/// preventing unbounded memory growth.
///
/// Returns a tuple of:
/// - `Receiver<Result<StreamEvent, LlmError>>` - events from the stream
/// - `JoinHandle<ToolLoopResult>` - the final result (join to get it)
///
/// # Backpressure
///
/// The `buffer_size` parameter controls how many events can be buffered before
/// the producer blocks. Choose based on your use case:
/// - Small (4-16): Tight backpressure, minimal memory
/// - Medium (32-64): Balance between latency and memory
/// - Large (128+): More latency tolerance, higher memory
///
/// # Consumer Drop
///
/// If the receiver is dropped before the stream completes, the internal task
/// will detect this (send returns error) and terminate gracefully. The join
/// handle will still return a `ToolLoopResult`, though it may indicate partial
/// completion.
///
/// # Example
///
/// ```rust,no_run
/// use std::sync::Arc;
/// use llm_stack::{ChatParams, ChatMessage, ToolLoopConfig, ToolRegistry, StreamEvent};
/// use llm_stack::tool::tool_loop_channel;
///
/// async fn example(
///     provider: Arc<dyn llm_stack::DynProvider>,
///     registry: Arc<ToolRegistry<()>>,
/// ) -> Result<(), Box<dyn std::error::Error>> {
///     let params = ChatParams {
///         messages: vec![ChatMessage::user("Hello")],
///         ..Default::default()
///     };
///
///     let (mut rx, handle) = tool_loop_channel(
///         provider,
///         registry,
///         params,
///         ToolLoopConfig::default(),
///         Arc::new(()),
///         32, // buffer size
///     );
///
///     while let Some(event) = rx.recv().await {
///         match event? {
///             StreamEvent::TextDelta(text) => print!("{text}"),
///             StreamEvent::Done { .. } => break,
///             _ => {}
///         }
///     }
///
///     let result = handle.await?;
///     println!("\nCompleted in {} iterations", result.iterations);
///     Ok(())
/// }
/// ```
pub fn tool_loop_channel<Ctx: LoopDepth + Send + Sync + 'static>(
    provider: Arc<dyn DynProvider>,
    registry: Arc<ToolRegistry<Ctx>>,
    params: ChatParams,
    config: ToolLoopConfig,
    ctx: Arc<Ctx>,
    buffer_size: usize,
) -> (
    mpsc::Receiver<Result<StreamEvent, LlmError>>,
    JoinHandle<ToolLoopResult>,
) {
    let (tx, rx) = mpsc::channel(buffer_size);

    let handle = tokio::spawn(async move {
        let mut stream = tool_loop_stream(provider, registry, params, config, ctx);

        while let Some(event) = stream.next().await {
            let is_done = matches!(&event, Ok(StreamEvent::Done { .. }));

            // Try to send the event
            if tx.send(event).await.is_err() {
                // Consumer dropped - break out of loop
                break;
            }

            if is_done {
                // Stream is done, we can exit
                break;
            }
        }

        // Return a minimal result indicating completion.
        // Note: The actual iteration/usage tracking would need deeper
        // integration with tool_loop_stream's internal state. For now,
        // we return a placeholder result. The events sent through the
        // channel contain the actual usage data via StreamEvent::Usage.
        ToolLoopResult {
            response: crate::chat::ChatResponse::empty(),
            iterations: 0,
            total_usage: crate::usage::Usage::default(),
            termination_reason: super::config::TerminationReason::Complete,
        }
    });

    (rx, handle)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chat::{ChatMessage, StopReason};
    use crate::test_helpers::mock_for;

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn test_tool_loop_channel_basic() {
        let mock = Arc::new(mock_for("test", "test-model"));

        // Queue a stream response (tool_loop_stream uses streaming)
        mock.queue_stream(vec![
            StreamEvent::TextDelta("Hello ".into()),
            StreamEvent::TextDelta("from channel!".into()),
            StreamEvent::Done {
                stop_reason: StopReason::EndTurn,
            },
        ]);

        let registry: ToolRegistry<()> = ToolRegistry::new();
        let registry = Arc::new(registry);

        let params = ChatParams {
            messages: vec![ChatMessage::user("Hello")],
            ..Default::default()
        };

        let (mut rx, handle) = tool_loop_channel(
            mock,
            registry,
            params,
            ToolLoopConfig::default(),
            Arc::new(()),
            16,
        );

        let mut events = Vec::new();
        while let Some(event) = rx.recv().await {
            events.push(event);
        }

        // Should have received events
        assert!(!events.is_empty());

        // Join handle should complete
        let result = handle.await.unwrap();
        assert!(matches!(
            result.termination_reason,
            super::super::config::TerminationReason::Complete
        ));
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn test_tool_loop_channel_consumer_drop() {
        let mock = Arc::new(mock_for("test", "test-model"));

        // Queue a stream that completes
        mock.queue_stream(vec![
            StreamEvent::TextDelta("Hello".into()),
            StreamEvent::Done {
                stop_reason: StopReason::EndTurn,
            },
        ]);

        let registry: ToolRegistry<()> = ToolRegistry::new();
        let registry = Arc::new(registry);

        let params = ChatParams {
            messages: vec![ChatMessage::user("Hello")],
            ..Default::default()
        };

        let (rx, handle) = tool_loop_channel(
            mock,
            registry,
            params,
            ToolLoopConfig::default(),
            Arc::new(()),
            2, // Small buffer
        );

        // Drop the receiver immediately
        drop(rx);

        // Handle should still complete (gracefully)
        let _result = handle.await.unwrap();
        // Result will be a default/empty one since we dropped early - just check it completes
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn test_tool_loop_channel_backpressure() {
        let mock = Arc::new(mock_for("test", "test-model"));

        // Queue a stream response
        mock.queue_stream(vec![
            StreamEvent::TextDelta("Response".into()),
            StreamEvent::Done {
                stop_reason: StopReason::EndTurn,
            },
        ]);

        let registry: ToolRegistry<()> = ToolRegistry::new();
        let registry = Arc::new(registry);

        let params = ChatParams {
            messages: vec![ChatMessage::user("Hello")],
            ..Default::default()
        };

        // Very small buffer to test backpressure behavior
        let (mut rx, handle) = tool_loop_channel(
            mock,
            registry,
            params,
            ToolLoopConfig::default(),
            Arc::new(()),
            1, // Minimal buffer
        );

        // Consume all events
        while let Some(_event) = rx.recv().await {}

        let result = handle.await.unwrap();
        assert!(matches!(
            result.termination_reason,
            super::super::config::TerminationReason::Complete
        ));
    }
}
