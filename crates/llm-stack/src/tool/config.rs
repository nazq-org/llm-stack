//! Tool loop configuration and event types.

use std::pin::Pin;
use std::sync::Arc;
use std::time::Duration;

use futures::Stream;
use serde_json::Value;

use crate::chat::{ChatResponse, ToolCall, ToolResult};
use crate::error::LlmError;
use crate::usage::Usage;

/// Callback type for tool call approval.
pub type ToolApprovalFn = Arc<dyn Fn(&ToolCall) -> ToolApproval + Send + Sync>;

/// Callback type for stop conditions.
pub type StopConditionFn = Arc<dyn Fn(&StopContext) -> StopDecision + Send + Sync>;

/// A pinned, boxed, `Send` stream of [`LoopEvent`] results.
///
/// The unified event stream from [`tool_loop_stream`](super::tool_loop_stream).
/// Emits both LLM streaming events (text deltas, tool call fragments) and
/// loop-level events (iteration boundaries, tool execution progress).
/// Terminates with [`LoopEvent::Done`] carrying the final [`ToolLoopResult`].
pub type LoopStream = Pin<Box<dyn Stream<Item = Result<LoopEvent, LlmError>> + Send>>;

/// Context provided to stop condition callbacks.
///
/// Contains information about the current state of the tool loop
/// to help decide whether to stop early.
#[derive(Debug)]
pub struct StopContext<'a> {
    /// Current iteration number (1-indexed).
    pub iteration: u32,
    /// The response from this iteration.
    pub response: &'a ChatResponse,
    /// Accumulated usage across all iterations so far.
    pub total_usage: &'a Usage,
    /// Total number of tool calls executed so far (across all iterations).
    pub tool_calls_executed: usize,
    /// Tool results from the most recent execution (empty on first response).
    pub last_tool_results: &'a [ToolResult],
}

/// Decision returned by a stop condition callback.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StopDecision {
    /// Continue the tool loop normally.
    Continue,
    /// Stop the loop immediately, using the current response as final.
    Stop,
    /// Stop the loop with a reason (for observability/debugging).
    StopWithReason(String),
}

/// Configuration for detecting repeated tool calls (stuck agents).
///
/// When an agent repeatedly makes the same tool call with identical arguments,
/// it's usually stuck in a loop. This configuration detects that pattern and
/// takes action to break the cycle.
///
/// # Example
///
/// ```rust
/// use llm_stack::tool::{LoopDetectionConfig, LoopAction};
///
/// let config = LoopDetectionConfig {
///     threshold: 3,  // Trigger after 3 consecutive identical calls
///     action: LoopAction::InjectWarning,  // Tell the agent it's looping
/// };
/// ```
#[derive(Debug, Clone, Copy)]
pub struct LoopDetectionConfig {
    /// Number of consecutive identical tool calls before triggering.
    ///
    /// A tool call is "identical" if it has the same name and arguments
    /// (compared via JSON equality). Default: 3.
    pub threshold: u32,

    /// Action to take when a loop is detected.
    pub action: LoopAction,
}

impl Default for LoopDetectionConfig {
    fn default() -> Self {
        Self {
            threshold: 3,
            action: LoopAction::Warn,
        }
    }
}

/// Action to take when a tool call loop is detected.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LoopAction {
    /// Emit [`LoopEvent::LoopDetected`] and continue execution.
    ///
    /// Use this for monitoring/alerting without interrupting the agent.
    Warn,

    /// Stop the loop immediately with an error.
    ///
    /// Returns `LlmError::ToolExecution` describing the loop.
    Stop,

    /// Inject a warning message into the conversation and continue.
    ///
    /// Adds a system message like "You have called {tool} with identical
    /// arguments {n} times. Try a different approach." This often helps
    /// the agent break out of the loop.
    ///
    /// The warning fires at every multiple of `threshold` (3, 6, 9, …)
    /// until the agent changes its approach. This prevents infinite loops
    /// where the agent ignores the first warning.
    InjectWarning,
}

/// Unified event emitted during tool loop execution.
///
/// `LoopEvent` merges LLM streaming events (text deltas, tool call fragments)
/// with loop-level lifecycle events (iteration boundaries, tool execution
/// progress) into a single stream. This gives consumers a complete, ordered
/// view of everything happening inside the loop.
///
/// The stream terminates with [`Done`](Self::Done) carrying the final
/// [`ToolLoopResult`].
///
/// # Example
///
/// ```rust,no_run
/// use llm_stack::tool::{tool_loop_stream, ToolLoopConfig, LoopEvent};
/// use futures::StreamExt;
/// use std::sync::Arc;
///
/// # async fn example(
/// #     provider: Arc<dyn llm_stack::DynProvider>,
/// #     registry: Arc<llm_stack::ToolRegistry<()>>,
/// #     params: llm_stack::ChatParams,
/// # ) {
/// let mut stream = tool_loop_stream(provider, registry, params, ToolLoopConfig::default(), Arc::new(()));
/// while let Some(event) = stream.next().await {
///     match event.unwrap() {
///         LoopEvent::TextDelta(text) => print!("{text}"),
///         LoopEvent::IterationStart { iteration, .. } => {
///             println!("\n--- Iteration {iteration} ---");
///         }
///         LoopEvent::ToolExecutionStart { tool_name, .. } => {
///             println!("[calling {tool_name}...]");
///         }
///         LoopEvent::ToolExecutionEnd { tool_name, duration, .. } => {
///             println!("[{tool_name} completed in {duration:?}]");
///         }
///         LoopEvent::Done(result) => {
///             println!("\nDone: {:?}", result.termination_reason);
///             break;
///         }
///         _ => {}
///     }
/// }
/// # }
/// ```
#[derive(Debug, Clone)]
#[non_exhaustive]
pub enum LoopEvent {
    // ── LLM streaming (translated from provider StreamEvent) ────
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

    /// Token usage information for this LLM call.
    Usage(Usage),

    // ── Loop lifecycle ──────────────────────────────────────────
    /// A new iteration of the tool loop is starting.
    IterationStart {
        /// The iteration number (1-indexed).
        iteration: u32,
        /// Number of messages in the conversation so far.
        message_count: usize,
    },

    /// About to execute a tool.
    ///
    /// When `parallel_tool_execution` is true, events arrive in **completion
    /// order** (whichever tool finishes first), not the order the LLM listed
    /// the calls. Use `call_id` to correlate start/end pairs.
    ToolExecutionStart {
        /// The tool call ID from the LLM.
        call_id: String,
        /// Name of the tool being called.
        tool_name: String,
        /// Arguments passed to the tool.
        arguments: Value,
    },

    /// Tool execution completed.
    ///
    /// When `parallel_tool_execution` is true, events arrive in **completion
    /// order**. Use `call_id` to correlate with the corresponding
    /// [`ToolExecutionStart`](Self::ToolExecutionStart).
    ToolExecutionEnd {
        /// The tool call ID from the LLM.
        call_id: String,
        /// Name of the tool that was called.
        tool_name: String,
        /// The result from the tool.
        result: ToolResult,
        /// How long the tool took to execute.
        duration: Duration,
    },

    /// A tool call loop was detected.
    ///
    /// Emitted when the same tool is called with identical arguments
    /// for `threshold` consecutive times. Only emitted when
    /// [`LoopDetectionConfig`] is configured.
    LoopDetected {
        /// Name of the tool being called repeatedly.
        tool_name: String,
        /// Number of consecutive identical calls detected.
        consecutive_count: u32,
        /// The action being taken in response.
        action: LoopAction,
    },

    // ── Terminal ────────────────────────────────────────────────
    /// The loop has finished. Carries the final [`ToolLoopResult`]
    /// with the accumulated response, usage, iteration count, and
    /// termination reason.
    Done(ToolLoopResult),
}

/// Configuration for [`tool_loop`](super::tool_loop) and [`tool_loop_stream`](super::tool_loop_stream).
pub struct ToolLoopConfig {
    /// Maximum number of generate-execute iterations. Default: 10.
    pub max_iterations: u32,
    /// Whether to execute multiple tool calls in parallel. Default: true.
    pub parallel_tool_execution: bool,
    /// Optional callback to approve, deny, or modify each tool call
    /// before execution.
    ///
    /// Called once per tool call in the LLM response, **after** the response
    /// is assembled but **before** any tool is executed. Receives the
    /// [`ToolCall`](crate::chat::ToolCall) as parsed from the LLM output.
    /// Modified arguments are re-validated against the tool's schema.
    ///
    /// Panics in the callback propagate and terminate the loop.
    pub on_tool_call: Option<ToolApprovalFn>,
    /// Optional stop condition checked after each LLM response.
    ///
    /// Called **after** the LLM response is received but **before** tools
    /// are executed. If the callback returns [`StopDecision::Stop`] or
    /// [`StopDecision::StopWithReason`], the loop terminates immediately
    /// without executing the requested tool calls.
    ///
    /// Receives a [`StopContext`] with information about the current
    /// iteration and returns a [`StopDecision`]. Use this to implement:
    ///
    /// - `final_answer` tool patterns (stop when a specific tool is called)
    /// - Token budget enforcement
    /// - Total tool call limits
    /// - Content pattern matching
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// use llm_stack::tool::{ToolLoopConfig, StopDecision};
    /// use std::sync::Arc;
    ///
    /// let config = ToolLoopConfig {
    ///     stop_when: Some(Arc::new(|ctx| {
    ///         // Stop if we've executed 5 or more tool calls
    ///         if ctx.tool_calls_executed >= 5 {
    ///             StopDecision::StopWithReason("Tool call limit reached".into())
    ///         } else {
    ///             StopDecision::Continue
    ///         }
    ///     })),
    ///     ..Default::default()
    /// };
    /// ```
    pub stop_when: Option<StopConditionFn>,

    /// Optional loop detection to catch stuck agents.
    ///
    /// When enabled, tracks consecutive identical tool calls (same name
    /// and arguments) and takes action when the threshold is reached.
    ///
    /// # Example
    ///
    /// ```rust
    /// use llm_stack::tool::{ToolLoopConfig, LoopDetectionConfig, LoopAction};
    ///
    /// let config = ToolLoopConfig {
    ///     loop_detection: Some(LoopDetectionConfig {
    ///         threshold: 3,
    ///         action: LoopAction::InjectWarning,
    ///     }),
    ///     ..Default::default()
    /// };
    /// ```
    pub loop_detection: Option<LoopDetectionConfig>,

    /// Maximum wall-clock time for the entire tool loop.
    ///
    /// If exceeded, returns with [`TerminationReason::Timeout`].
    /// This is useful for enforcing time budgets in production systems.
    ///
    /// # Example
    ///
    /// ```rust
    /// use llm_stack::tool::ToolLoopConfig;
    /// use std::time::Duration;
    ///
    /// let config = ToolLoopConfig {
    ///     timeout: Some(Duration::from_secs(30)),
    ///     ..Default::default()
    /// };
    /// ```
    pub timeout: Option<Duration>,

    /// Maximum allowed nesting depth for recursive tool loops.
    ///
    /// When a tool calls `tool_loop` internally (e.g., spawning a sub-agent),
    /// the depth is tracked via the context's [`LoopDepth`](super::LoopDepth)
    /// implementation. If `ctx.loop_depth() >= max_depth` at entry,
    /// returns `Err(LlmError::MaxDepthExceeded)`.
    ///
    /// - `Some(n)`: Error if depth >= n
    /// - `None`: No limit (dangerous, use with caution)
    ///
    /// Default: `Some(3)` (allows master → worker → one more level)
    ///
    /// # Example
    ///
    /// ```rust
    /// use llm_stack::tool::ToolLoopConfig;
    ///
    /// // Master/Worker pattern: master=0, worker=1, no grandchildren
    /// let config = ToolLoopConfig {
    ///     max_depth: Some(2),
    ///     ..Default::default()
    /// };
    /// ```
    pub max_depth: Option<u32>,
}

impl Clone for ToolLoopConfig {
    fn clone(&self) -> Self {
        Self {
            max_iterations: self.max_iterations,
            parallel_tool_execution: self.parallel_tool_execution,
            on_tool_call: self.on_tool_call.clone(),
            stop_when: self.stop_when.clone(),
            loop_detection: self.loop_detection,
            timeout: self.timeout,
            max_depth: self.max_depth,
        }
    }
}

impl Default for ToolLoopConfig {
    fn default() -> Self {
        Self {
            max_iterations: 10,
            parallel_tool_execution: true,
            on_tool_call: None,
            stop_when: None,
            loop_detection: None,
            timeout: None,
            max_depth: Some(3),
        }
    }
}

impl std::fmt::Debug for ToolLoopConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ToolLoopConfig")
            .field("max_iterations", &self.max_iterations)
            .field("parallel_tool_execution", &self.parallel_tool_execution)
            .field("has_on_tool_call", &self.on_tool_call.is_some())
            .field("has_stop_when", &self.stop_when.is_some())
            .field("loop_detection", &self.loop_detection)
            .field("timeout", &self.timeout)
            .field("max_depth", &self.max_depth)
            .finish()
    }
}

/// Result of approving a tool call before execution.
#[derive(Debug, Clone)]
pub enum ToolApproval {
    /// Allow the tool call to proceed as-is.
    Approve,
    /// Deny the tool call. The reason is sent back to the LLM as an
    /// error tool result.
    Deny(String),
    /// Modify the tool call arguments before execution.
    Modify(Value),
}

/// The result of a completed tool loop.
#[derive(Debug, Clone)]
pub struct ToolLoopResult {
    /// The final response from the LLM (after all tool iterations).
    pub response: ChatResponse,
    /// How many generate-execute iterations were performed.
    pub iterations: u32,
    /// Accumulated usage across all iterations.
    pub total_usage: Usage,
    /// Why the loop terminated.
    ///
    /// This provides observability into the loop's completion reason,
    /// useful for debugging and monitoring agent behavior.
    pub termination_reason: TerminationReason,
}

/// Why a tool loop terminated.
///
/// Used for observability and debugging. Each variant captures specific
/// information about why the loop ended.
///
/// # Example
///
/// ```rust,no_run
/// use llm_stack::tool::TerminationReason;
/// use std::time::Duration;
///
/// # fn check_result(reason: TerminationReason) {
/// match reason {
///     TerminationReason::Complete => println!("Task completed naturally"),
///     TerminationReason::StopCondition { reason } => {
///         println!("Custom stop: {}", reason.as_deref().unwrap_or("no reason"));
///     }
///     TerminationReason::MaxIterations { limit } => {
///         println!("Hit iteration limit: {limit}");
///     }
///     TerminationReason::LoopDetected { tool_name, count } => {
///         println!("Stuck calling {tool_name} {count} times");
///     }
///     TerminationReason::Timeout { limit } => {
///         println!("Exceeded timeout: {limit:?}");
///     }
/// }
/// # }
/// ```
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TerminationReason {
    /// LLM returned a response with no tool calls (natural completion).
    Complete,

    /// Custom stop condition returned [`StopDecision::Stop`] or
    /// [`StopDecision::StopWithReason`].
    StopCondition {
        /// The reason provided via [`StopDecision::StopWithReason`], if any.
        reason: Option<String>,
    },

    /// Hit the `max_iterations` limit.
    MaxIterations {
        /// The configured limit that was reached.
        limit: u32,
    },

    /// Loop detection triggered with [`LoopAction::Stop`].
    LoopDetected {
        /// Name of the tool being called repeatedly.
        tool_name: String,
        /// Number of consecutive identical calls.
        count: u32,
    },

    /// Wall-clock timeout exceeded.
    Timeout {
        /// The configured timeout that was exceeded.
        limit: Duration,
    },
}
