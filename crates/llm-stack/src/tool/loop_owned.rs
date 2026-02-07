//! Arc-owned resumable tool loop that is `Send + 'static`.
//!
//! [`OwnedToolLoopHandle`] is identical in behavior to
//! [`ToolLoopHandle`](super::ToolLoopHandle) but owns its provider and
//! registry via `Arc`, making it safe to move into `tokio::spawn` or any
//! context requiring `Send + 'static`.
//!
//! # When to use
//!
//! Use `OwnedToolLoopHandle` when the loop must outlive its creator:
//! - Task agents spawned via `tokio::spawn`
//! - Holding the handle across an `await` point that requires `'static`
//! - Sending the handle to another thread
//!
//! Use [`ToolLoopHandle`](super::ToolLoopHandle) when the loop lives on the
//! caller's stack (e.g., a master orchestrator driving the loop directly).
//!
//! # Example
//!
//! ```rust,no_run
//! use std::sync::Arc;
//! use llm_stack::tool::{ToolLoopConfig, ToolRegistry, OwnedToolLoopHandle, OwnedTurnResult};
//! use llm_stack::{ChatParams, ChatMessage};
//!
//! # async fn example(provider: Arc<dyn llm_stack::DynProvider>) {
//! let registry = Arc::new(ToolRegistry::<()>::new());
//! let params = ChatParams {
//!     messages: vec![ChatMessage::user("Hello")],
//!     ..Default::default()
//! };
//!
//! let mut handle = OwnedToolLoopHandle::new(
//!     provider,
//!     registry,
//!     params,
//!     ToolLoopConfig::default(),
//!     &(),
//! );
//!
//! // Safe to spawn because OwnedToolLoopHandle is Send + 'static
//! tokio::spawn(async move {
//!     loop {
//!         match handle.next_turn().await {
//!             OwnedTurnResult::Yielded(turn) => turn.continue_loop(),
//!             OwnedTurnResult::Completed(done) => {
//!                 println!("Done: {:?}", done.response.text());
//!                 break;
//!             }
//!             OwnedTurnResult::Error(err) => {
//!                 eprintln!("Error: {}", err.error);
//!                 break;
//!             }
//!         }
//!     }
//! });
//! # }
//! ```

use std::sync::Arc;

use crate::chat::{ChatMessage, ContentBlock, ToolCall, ToolResult};
use crate::provider::{ChatParams, DynProvider};
use crate::usage::Usage;

use super::LoopDepth;
use super::ToolRegistry;
use super::config::{ToolLoopConfig, ToolLoopResult};
use super::loop_core::{CompletedData, ErrorData, IterationOutcome, LoopCore};
use super::loop_resumable::{Completed, LoopCommand, TurnError};

/// Result of one turn of the owned tool loop.
///
/// Same semantics as [`TurnResult`](super::TurnResult) but without the
/// provider lifetime parameter — this makes `OwnedToolLoopHandle` fully
/// `Send + 'static`.
#[must_use = "an OwnedTurnResult must be matched — Yielded requires resume() to continue"]
pub enum OwnedTurnResult<'h, Ctx: LoopDepth + Send + Sync + 'static> {
    /// Tools were executed. Consume via `resume()`, `continue_loop()`,
    /// `inject_and_continue()`, or `stop()`.
    Yielded(OwnedYielded<'h, Ctx>),

    /// The loop completed.
    Completed(Completed),

    /// An unrecoverable error occurred.
    Error(TurnError),
}

/// Handle returned when tools were executed on an [`OwnedToolLoopHandle`].
///
/// Same API as [`Yielded`](super::Yielded) but borrows an
/// `OwnedToolLoopHandle` instead of a `ToolLoopHandle`.
#[must_use = "must call .resume(), .continue_loop(), .inject_and_continue(), or .stop() to continue"]
pub struct OwnedYielded<'h, Ctx: LoopDepth + Send + Sync + 'static> {
    handle: &'h mut OwnedToolLoopHandle<Ctx>,

    /// The tool calls the LLM requested.
    pub tool_calls: Vec<ToolCall>,

    /// Results from executing those tool calls.
    pub results: Vec<ToolResult>,

    /// Text content from the LLM's response alongside the tool calls.
    pub assistant_content: Vec<ContentBlock>,

    /// Current iteration number (1-indexed).
    pub iteration: u32,

    /// Accumulated usage across all iterations so far.
    pub total_usage: Usage,
}

impl<Ctx: LoopDepth + Send + Sync + 'static> OwnedYielded<'_, Ctx> {
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

// ── OwnedToolLoopHandle ─────────────────────────────────────────────

/// Arc-owned resumable tool loop.
///
/// Identical in behavior to [`ToolLoopHandle`](super::ToolLoopHandle) but
/// owns provider and registry via `Arc`, making it `Send + 'static`.
///
/// # Lifecycle
///
/// Same as `ToolLoopHandle`:
///
/// 1. Create with [`new()`](Self::new)
/// 2. Call [`next_turn()`](Self::next_turn)
/// 3. If `Yielded`, consume via `resume()` / `continue_loop()` / etc.
/// 4. Repeat until `Completed` or `Error`
/// 5. Optionally call [`into_result()`](Self::into_result)
pub struct OwnedToolLoopHandle<Ctx: LoopDepth + Send + Sync + 'static> {
    provider: Arc<dyn DynProvider>,
    registry: Arc<ToolRegistry<Ctx>>,
    core: LoopCore<Ctx>,
}

impl<Ctx: LoopDepth + Send + Sync + 'static> OwnedToolLoopHandle<Ctx> {
    /// Create a new owned resumable tool loop.
    ///
    /// Takes `Arc`-wrapped provider and registry so the handle can be
    /// moved into `tokio::spawn`.
    pub fn new(
        provider: Arc<dyn DynProvider>,
        registry: Arc<ToolRegistry<Ctx>>,
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

    /// Internal constructor from a pre-built `LoopCore`.
    ///
    /// Used by `ToolLoopHandle::into_owned()`.
    pub(crate) fn from_core(
        provider: Arc<dyn DynProvider>,
        registry: Arc<ToolRegistry<Ctx>>,
        core: LoopCore<Ctx>,
    ) -> Self {
        Self {
            provider,
            registry,
            core,
        }
    }

    /// Advance the loop and return the result of this turn.
    ///
    /// Identical semantics to [`ToolLoopHandle::next_turn()`](super::ToolLoopHandle::next_turn).
    pub async fn next_turn(&mut self) -> OwnedTurnResult<'_, Ctx> {
        let outcome = self
            .core
            .do_iteration(&*self.provider, &self.registry)
            .await;
        match outcome {
            IterationOutcome::ToolsExecuted {
                tool_calls,
                results,
                assistant_content,
                iteration,
                total_usage,
            } => OwnedTurnResult::Yielded(OwnedYielded {
                handle: self,
                tool_calls,
                results,
                assistant_content,
                iteration,
                total_usage,
            }),
            IterationOutcome::Completed(CompletedData {
                response,
                termination_reason,
                iterations,
                total_usage,
            }) => OwnedTurnResult::Completed(Completed {
                response,
                termination_reason,
                iterations,
                total_usage,
            }),
            IterationOutcome::Error(ErrorData {
                error,
                iterations,
                total_usage,
            }) => OwnedTurnResult::Error(TurnError {
                error,
                iterations,
                total_usage,
            }),
        }
    }

    /// Tell the loop how to proceed before the next `next_turn()` call.
    pub fn resume(&mut self, command: LoopCommand) {
        self.core.resume(command);
    }

    /// Get a snapshot of the current conversation messages.
    pub fn messages(&self) -> &[ChatMessage] {
        self.core.messages()
    }

    /// Get a mutable reference to the conversation messages.
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

    /// Consume the handle and return a `ToolLoopResult`.
    pub fn into_result(self) -> ToolLoopResult {
        self.core.into_result()
    }
}

impl<Ctx: LoopDepth + Send + Sync + 'static> std::fmt::Debug for OwnedToolLoopHandle<Ctx> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OwnedToolLoopHandle")
            .field("core", &self.core)
            .finish_non_exhaustive()
    }
}
