//! Resumable tool loop with caller-controlled iteration.
//!
//! Unlike [`tool_loop`](super::tool_loop) which runs to completion autonomously,
//! the resumable loop yields control to the caller after each tool execution
//! round. The caller can then decide to continue, inject messages, or stop.
//!
//! This enables orchestration patterns like:
//! - Multi-agent systems where a master inspects tool results between iterations
//! - External event injection (user follow-ups, worker completions)
//! - Context compaction between iterations
//! - Custom routing logic based on which tools were called
//!
//! # Example
//!
//! ```rust,no_run
//! use llm_stack::tool::{ToolLoopConfig, ToolRegistry, ToolLoopHandle, TurnResult, LoopCommand};
//! use llm_stack::{ChatParams, ChatMessage};
//!
//! # async fn example(provider: &dyn llm_stack::DynProvider) -> Result<(), llm_stack::LlmError> {
//! let registry: ToolRegistry<()> = ToolRegistry::new();
//! let params = ChatParams {
//!     messages: vec![ChatMessage::user("Hello")],
//!     ..Default::default()
//! };
//!
//! let mut handle = ToolLoopHandle::new(
//!     provider,
//!     &registry,
//!     params,
//!     ToolLoopConfig::default(),
//!     &(),
//! );
//!
//! loop {
//!     match handle.next_turn().await {
//!         TurnResult::Yielded(turn) => {
//!             // Text from this turn is directly available
//!             if let Some(text) = turn.assistant_text() {
//!                 println!("LLM said: {text}");
//!             }
//!             // Inspect results, decide what to do
//!             turn.continue_loop();
//!         }
//!         TurnResult::Completed(done) => {
//!             println!("Done: {:?}", done.response.text());
//!             break;
//!         }
//!         TurnResult::Error(err) => {
//!             eprintln!("Error: {}", err.error);
//!             break;
//!         }
//!     }
//! }
//! # Ok(())
//! # }
//! ```

use crate::chat::{ChatMessage, ChatResponse, ContentBlock, ToolCall, ToolResult};
use crate::error::LlmError;
use crate::provider::{ChatParams, DynProvider};
use crate::usage::Usage;

use super::LoopDepth;
use super::ToolRegistry;
use super::config::{LoopEvent, TerminationReason, ToolLoopConfig, ToolLoopResult};
use super::loop_core::{CompletedData, ErrorData, IterationOutcome, LoopCore};

// ── Shared macros for Yielded-like types ─────────────────────────────

/// Implements the common methods on a `Yielded`-like struct.
///
/// Both `Yielded` and `OwnedYielded` have identical field layouts and
/// method bodies. The only difference is the handle type they borrow.
/// This macro eliminates the duplication.
///
/// Expects the struct to have fields: `handle`, `assistant_content`,
/// `tool_calls`, `results`, `iteration`, `total_usage`.
macro_rules! impl_yielded_methods {
    ($yielded:ident < $($lt:lifetime),* >) => {
        impl<$($lt,)* Ctx: LoopDepth + Send + Sync + 'static> $yielded<$($lt,)* Ctx> {
            /// Continue with the given command.
            pub fn resume(self, command: LoopCommand) {
                self.handle.resume(command);
            }

            /// Convenience: continue to the next LLM iteration with no injected messages.
            pub fn continue_loop(self) {
                self.resume(LoopCommand::Continue);
            }

            /// Convenience: inject messages and continue.
            pub fn inject_and_continue(self, messages: Vec<ChatMessage>) {
                self.resume(LoopCommand::InjectMessages(messages));
            }

            /// Convenience: stop the loop.
            pub fn stop(self, reason: Option<String>) {
                self.resume(LoopCommand::Stop(reason));
            }

            /// Extract text from `assistant_content` blocks.
            ///
            /// Returns the LLM's "thinking aloud" text emitted alongside tool calls,
            /// or `None` if there were no text blocks.
            pub fn assistant_text(&self) -> Option<String> {
                let text: String = self
                    .assistant_content
                    .iter()
                    .filter_map(|block| match block {
                        ContentBlock::Text(t) => Some(t.as_str()),
                        _ => None,
                    })
                    .collect::<Vec<_>>()
                    .join("\n");
                if text.is_empty() { None } else { Some(text) }
            }

            /// Access the full message history (read-only).
            pub fn messages(&self) -> &[ChatMessage] {
                self.handle.messages()
            }

            /// Access the full message history (mutable, for context compaction).
            pub fn messages_mut(&mut self) -> &mut Vec<ChatMessage> {
                self.handle.messages_mut()
            }
        }
    };
}

pub(crate) use impl_yielded_methods;

/// Converts an [`IterationOutcome`] into a `TurnResult`-like enum.
///
/// Both `TurnResult` and `OwnedTurnResult` need identical destructuring
/// of `IterationOutcome` into their respective `Yielded`/`Completed`/`Error`
/// variants. This macro generates that conversion.
macro_rules! outcome_to_turn_result {
    ($outcome:expr, $handle:expr, $turn_ty:ident, $yielded_ty:ident) => {{
        // Drain buffered events before constructing the result.
        // This moves them into the turn variant so callers get events
        // co-located with the turn data — no separate drain step needed.
        let events = $handle.core.drain_events();
        match $outcome {
            IterationOutcome::ToolsExecuted {
                tool_calls,
                results,
                assistant_content,
                iteration,
                total_usage,
            } => $turn_ty::Yielded($yielded_ty {
                handle: $handle,
                tool_calls,
                results,
                assistant_content,
                iteration,
                total_usage,
                events,
            }),
            IterationOutcome::Completed(CompletedData {
                response,
                termination_reason,
                iterations,
                total_usage,
            }) => $turn_ty::Completed(Completed {
                response,
                termination_reason,
                iterations,
                total_usage,
                events,
            }),
            IterationOutcome::Error(ErrorData {
                error,
                iterations,
                total_usage,
            }) => $turn_ty::Error(TurnError {
                error,
                iterations,
                total_usage,
                events,
            }),
        }
    }};
}

pub(crate) use outcome_to_turn_result;

/// Result of one turn of the tool loop.
///
/// Match on this to determine what happened and what you can do next.
/// Each variant carries the data from the turn AND (for `Yielded`) a handle
/// scoped to valid operations for that state.
///
/// This follows the same pattern as [`std::collections::hash_map::Entry`] —
/// the variant gives you exactly the methods that make sense for that state.
#[must_use = "a TurnResult must be matched — Yielded requires resume() to continue"]
pub enum TurnResult<'a, 'h, Ctx: LoopDepth + Send + Sync + 'static> {
    /// Tools were executed. The caller MUST consume this via `resume()`,
    /// `continue_loop()`, `inject_and_continue()`, or `stop()`.
    ///
    /// While this variant exists, the `ToolLoopHandle` is mutably borrowed
    /// and cannot be used directly. Consuming the `Yielded` releases the
    /// borrow.
    Yielded(Yielded<'a, 'h, Ctx>),

    /// The loop completed (no tool calls, stop condition, max iterations, or timeout).
    Completed(Completed),

    /// An unrecoverable error occurred.
    Error(TurnError),
}

/// Handle returned when tools were executed. Borrows the [`ToolLoopHandle`]
/// mutably, so the caller cannot call `next_turn()` again until this is
/// consumed via `resume()`, `continue_loop()`, `inject_and_continue()`, or
/// `stop()`.
///
/// The text content the LLM produced alongside tool calls is available
/// directly via [`assistant_content`](Self::assistant_content) and
/// [`assistant_text()`](Self::assistant_text) — no need to scan
/// `messages()`.
#[must_use = "must call .resume(), .continue_loop(), .inject_and_continue(), or .stop() to continue"]
pub struct Yielded<'a, 'h, Ctx: LoopDepth + Send + Sync + 'static> {
    handle: &'h mut ToolLoopHandle<'a, Ctx>,

    /// The tool calls the LLM requested.
    pub tool_calls: Vec<ToolCall>,

    /// Results from executing those tool calls.
    pub results: Vec<ToolResult>,

    /// Text content from the LLM's response alongside the tool calls.
    ///
    /// This is the `other_content` from `partition_content()` — `Text`,
    /// `Reasoning`, `Image`, etc. — everything that isn't a `ToolCall` or
    /// `ToolResult`. Previously only accessible by scanning `messages()`.
    pub assistant_content: Vec<ContentBlock>,

    /// Current iteration number (1-indexed).
    pub iteration: u32,

    /// Accumulated usage across all iterations so far.
    pub total_usage: Usage,

    /// Lifecycle events from this turn (`IterationStart`, `ToolExecutionStart/End`, etc.).
    ///
    /// Pre-drained from the internal buffer — no need to call
    /// [`ToolLoopHandle::drain_events()`] separately.
    pub events: Vec<LoopEvent>,
}

impl_yielded_methods!(Yielded<'a, 'h>);

/// Terminal: the loop completed successfully.
pub struct Completed {
    /// The final LLM response. Use `.text()` to get the response text.
    pub response: ChatResponse,
    /// Why the loop terminated.
    pub termination_reason: TerminationReason,
    /// Total iterations performed.
    pub iterations: u32,
    /// Accumulated usage.
    pub total_usage: Usage,
    /// Lifecycle events from the final turn.
    pub events: Vec<LoopEvent>,
}

/// Terminal: the loop errored.
pub struct TurnError {
    /// The error.
    pub error: LlmError,
    /// Iterations completed before the error.
    pub iterations: u32,
    /// Usage accumulated before the error.
    pub total_usage: Usage,
    /// Lifecycle events from the final turn (may include `IterationStart`
    /// even though the turn errored).
    pub events: Vec<LoopEvent>,
}

/// Commands sent by the caller to control the resumable loop.
///
/// Passed to [`Yielded::resume()`] after receiving a
/// [`TurnResult::Yielded`].
#[derive(Debug)]
pub enum LoopCommand {
    /// Continue to the next LLM iteration normally.
    Continue,

    /// Inject additional messages before the next LLM call.
    ///
    /// The injected messages are appended after the tool results from
    /// the current round. Use this to provide additional context
    /// (e.g., worker agent results, user follow-ups).
    InjectMessages(Vec<ChatMessage>),

    /// Stop the loop immediately.
    ///
    /// Returns a `Completed` event with `TerminationReason::StopCondition`.
    Stop(Option<String>),
}

// ── ToolLoopHandle ──────────────────────────────────────────────────

/// Caller-driven resumable tool loop.
///
/// Unlike [`tool_loop`](super::tool_loop) which runs autonomously, this struct
/// gives the caller control between each tool execution round. Call
/// [`next_turn()`](Self::next_turn) to advance the loop, inspect the result,
/// then consume the [`Yielded`] handle to control what happens next.
///
/// # No spawning required
///
/// This is a direct state machine — no background tasks, no channels. The
/// caller drives it by calling `next_turn()` which performs one iteration
/// (LLM call + tool execution) and returns.
///
/// # Lifecycle
///
/// 1. Create with [`new()`](Self::new)
/// 2. Call [`next_turn()`](Self::next_turn) to get the first result
/// 3. If `Yielded`, consume via `resume()` / `continue_loop()` / etc., then
///    call `next_turn()` again
/// 4. Repeat until `Completed` or `Error`
/// 5. Optionally call [`into_result()`](Self::into_result) for a `ToolLoopResult`
pub struct ToolLoopHandle<'a, Ctx: LoopDepth + Send + Sync + 'static> {
    provider: &'a dyn DynProvider,
    registry: &'a ToolRegistry<Ctx>,
    core: LoopCore<Ctx>,
}

impl<'a, Ctx: LoopDepth + Send + Sync + 'static> ToolLoopHandle<'a, Ctx> {
    /// Create a new resumable tool loop.
    ///
    /// Does not start execution — call [`next_turn()`](Self::next_turn) to
    /// begin the first iteration.
    ///
    /// # Depth Tracking
    ///
    /// Same as [`tool_loop`](super::tool_loop) — if `Ctx` implements [`LoopDepth`],
    /// nested calls are tracked and `max_depth` is enforced. If the depth limit
    /// is already exceeded, the first call to `next_turn()` returns `Error`.
    pub fn new(
        provider: &'a dyn DynProvider,
        registry: &'a ToolRegistry<Ctx>,
        params: ChatParams,
        config: ToolLoopConfig,
        ctx: &Ctx,
    ) -> Self {
        Self {
            provider,
            registry,
            core: LoopCore::new(params, config, ctx),
        }
    }

    /// Advance the loop and return the result of this turn.
    ///
    /// Each call performs one iteration: LLM generation, tool execution (if
    /// applicable), and returns the result.
    ///
    /// Returns a [`TurnResult`] that must be matched:
    /// - [`TurnResult::Yielded`] — tools ran, consume via `resume()` /
    ///   `continue_loop()` / `inject_and_continue()` / `stop()` to continue
    /// - [`TurnResult::Completed`] — loop is done, read `.response`
    /// - [`TurnResult::Error`] — loop failed, read `.error`
    ///
    /// After `Completed` or `Error`, all subsequent calls return the same
    /// terminal result.
    pub async fn next_turn(&mut self) -> TurnResult<'a, '_, Ctx> {
        let outcome = self.core.do_iteration(self.provider, self.registry).await;
        outcome_to_turn_result!(outcome, self, TurnResult, Yielded)
    }

    /// Tell the loop how to proceed before the next [`next_turn()`](Self::next_turn) call.
    ///
    /// When using [`TurnResult::Yielded`], prefer the convenience methods on
    /// [`Yielded`] (`continue_loop()`, `inject_and_continue()`, `stop()`),
    /// which consume the yielded handle and call this internally.
    ///
    /// This method is useful when you need to set a command on the handle
    /// directly — for example, when driving the handle from an external
    /// event loop that receives the command asynchronously after the
    /// `Yielded` has already been consumed.
    ///
    /// Has no effect after `Completed` or `Error`.
    pub fn resume(&mut self, command: LoopCommand) {
        self.core.resume(command);
    }

    /// Get a snapshot of the current conversation messages.
    ///
    /// Useful for context window management or debugging.
    pub fn messages(&self) -> &[ChatMessage] {
        self.core.messages()
    }

    /// Get a mutable reference to the conversation messages.
    ///
    /// Allows direct manipulation of the message history between iterations
    /// (e.g., for context compaction/summarization).
    pub fn messages_mut(&mut self) -> &mut Vec<ChatMessage> {
        self.core.messages_mut()
    }

    /// Get the accumulated usage across all iterations so far.
    pub fn total_usage(&self) -> &Usage {
        self.core.total_usage()
    }

    /// Get the current iteration count.
    pub fn iterations(&self) -> u32 {
        self.core.iterations()
    }

    /// Whether the loop has finished (returned Completed or Error).
    pub fn is_finished(&self) -> bool {
        self.core.is_finished()
    }

    /// Drain any remaining buffered [`LoopEvent`]s.
    ///
    /// Most callers should use the `events` field on [`Yielded`], [`Completed`],
    /// or [`TurnError`] instead — those are pre-populated by [`next_turn()`](Self::next_turn).
    ///
    /// This method exists for edge cases where events may accumulate between
    /// turns (e.g., after calling [`resume()`](Self::resume) directly from an
    /// external event loop).
    pub fn drain_events(&mut self) -> Vec<LoopEvent> {
        self.core.drain_events()
    }

    /// Consume the handle and return a `ToolLoopResult`.
    ///
    /// If the loop hasn't completed yet, returns a result with current
    /// iteration count and `TerminationReason::Complete`.
    pub fn into_result(self) -> ToolLoopResult {
        self.core.into_result()
    }

    /// Convert this borrowed handle into an owned handle.
    ///
    /// The provider and registry must be provided as `Arc` since this
    /// handle only holds references. The loop state (iterations, messages,
    /// usage, etc.) is transferred as-is.
    pub fn into_owned(
        self,
        provider: std::sync::Arc<dyn DynProvider>,
        registry: std::sync::Arc<ToolRegistry<Ctx>>,
    ) -> super::OwnedToolLoopHandle<Ctx> {
        super::OwnedToolLoopHandle::from_core(provider, registry, self.core)
    }
}

impl<Ctx: LoopDepth + Send + Sync + 'static> std::fmt::Debug for ToolLoopHandle<'_, Ctx> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ToolLoopHandle")
            .field("core", &self.core)
            .finish_non_exhaustive()
    }
}
