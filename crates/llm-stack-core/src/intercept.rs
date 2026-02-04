// Interceptor methods are chainable builders, not pure functions
#![allow(clippy::must_use_candidate)]
// Explicit casts for duration conversions are clearer than try_into
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_sign_loss)]
#![allow(clippy::cast_precision_loss)]
#![allow(clippy::cast_possible_wrap)]
// Lifetime names are kept explicit for clarity in trait impls
#![allow(clippy::needless_lifetimes)]

//! Unified interceptor system for LLM calls and tool executions.
//!
//! This module provides a composable middleware-like system that works across
//! different domains. The core abstraction is the [`Interceptor`] trait, which
//! wraps operations and can inspect, modify, or short-circuit them.
//!
//! # Architecture
//!
//! ```text
//! InterceptorStack::new()
//!     .with(Logging::default())      // outermost: sees request first
//!     .with(Retry::default())        // retries wrap everything inside
//!     .with(Timeout::new(30s))       // timeout on inner operation
//!     .execute(&input, operation)
//! ```
//!
//! # Domains
//!
//! The system supports multiple domains via marker types:
//!
//! - [`ToolExec<Ctx>`] - Tool executions (integrated with [`ToolRegistry`](crate::ToolRegistry))
//! - [`LlmCall`] - LLM provider requests (reserved for future use, not yet integrated)
//!
//! # Generic Interceptors
//!
//! Interceptors like [`Retry`], [`Timeout`], and [`Logging`] are generic over
//! any domain that implements the required behavior traits ([`Retryable`],
//! [`Timeoutable`], [`Loggable`]).
//!
//! # Example
//!
//! ```rust,ignore
//! use llm_stack_core::ToolRegistry;
//! use llm_stack_core::intercept::{InterceptorStack, Retry, Timeout, ToolExec};
//! use std::time::Duration;
//!
//! let registry: ToolRegistry<()> = ToolRegistry::new()
//!     .with_interceptors(
//!         InterceptorStack::<ToolExec<()>>::new()
//!             .with(Retry::default())
//!             .with(Timeout::new(Duration::from_secs(30)))
//!     );
//! ```

use std::future::Future;
use std::marker::PhantomData;
use std::pin::Pin;
use std::sync::Arc;

// Re-export core types at module level for convenience
pub use behavior::{Loggable, Outcome, Retryable, Timeoutable};
pub use domain::{LlmCall, ToolExec, ToolRequest, ToolResponse};

// Note: Interceptable, Interceptor, InterceptorStack, Next, Operation, FnOperation
// are defined at module root and don't need re-exporting

/// An operation that can be intercepted.
///
/// This trait defines the input and output types for an interceptable operation.
/// Implement this for marker types that represent different domains (e.g., LLM calls,
/// tool executions).
pub trait Interceptable: Send + Sync + 'static {
    /// Input to the operation.
    type Input: Send;

    /// Output from the operation.
    type Output: Send;
}

/// Wraps an interceptable operation.
///
/// Interceptors form a chain. Each interceptor receives the input and a [`Next`]
/// handle. It can:
/// - Pass through: call `next.run(input).await`
/// - Modify input: transform input, then call `next.run(&modified).await`
/// - Short-circuit: return early without calling `next`
/// - Retry: call `next.clone().run(input).await` multiple times
/// - Wrap output: call `next`, then transform the result
///
/// # Implementing
///
/// ```rust,ignore
/// use llm_stack_core::intercept::{Interceptor, Interceptable, Next};
/// use std::future::Future;
/// use std::pin::Pin;
///
/// struct MyInterceptor;
///
/// impl<T: Interceptable> Interceptor<T> for MyInterceptor
/// where
///     T::Input: Sync,
/// {
///     fn intercept<'a>(
///         &'a self,
///         input: &'a T::Input,
///         next: Next<'a, T>,
///     ) -> Pin<Box<dyn Future<Output = T::Output> + Send + 'a>> {
///         Box::pin(async move {
///             // Do something before
///             let result = next.run(input).await;
///             // Do something after
///             result
///         })
///     }
/// }
/// ```
pub trait Interceptor<T: Interceptable>: Send + Sync {
    /// Intercept the operation.
    ///
    /// Call `next.run(input)` to continue the chain, or return early to short-circuit.
    fn intercept<'a>(
        &'a self,
        input: &'a T::Input,
        next: Next<'a, T>,
    ) -> Pin<Box<dyn Future<Output = T::Output> + Send + 'a>>;
}

/// Handle to invoke the next interceptor in the chain (or the final operation).
///
/// `Next` is [`Clone`], which is essential for retry interceptors that need to
/// call the chain multiple times. Cloning is cheap - it only clones references.
pub struct Next<'a, T: Interceptable> {
    interceptors: &'a [Arc<dyn Interceptor<T>>],
    operation: &'a dyn Operation<T>,
}

impl<T: Interceptable> Clone for Next<'_, T> {
    fn clone(&self) -> Self {
        *self
    }
}

// Copy is valid: both fields are references
impl<T: Interceptable> Copy for Next<'_, T> {}

impl<T: Interceptable> Next<'_, T>
where
    T::Input: Sync,
{
    /// Run the operation through the remaining chain.
    ///
    /// This consumes `self`, but since `Next` is `Copy`, you can call it multiple
    /// times by copying first (e.g., for retry logic).
    pub async fn run(self, input: &T::Input) -> T::Output {
        if let Some((first, rest)) = self.interceptors.split_first() {
            let next = Next {
                interceptors: rest,
                operation: self.operation,
            };
            first.intercept(input, next).await
        } else {
            self.operation.execute(input).await
        }
    }
}

/// The final operation to execute after all interceptors.
///
/// This is object-safe to allow storing different operation types.
pub trait Operation<T: Interceptable>: Send + Sync {
    /// Execute the operation.
    fn execute<'a>(
        &'a self,
        input: &'a T::Input,
    ) -> Pin<Box<dyn Future<Output = T::Output> + Send + 'a>>
    where
        T::Input: Sync;
}

/// Wrap a closure as an [`Operation`].
pub struct FnOperation<T, F>
where
    T: Interceptable,
    F: Fn(&T::Input) -> Pin<Box<dyn Future<Output = T::Output> + Send + '_>> + Send + Sync,
{
    f: F,
    _marker: PhantomData<T>,
}

impl<T, F> FnOperation<T, F>
where
    T: Interceptable,
    F: Fn(&T::Input) -> Pin<Box<dyn Future<Output = T::Output> + Send + '_>> + Send + Sync,
{
    /// Create a new operation from a closure.
    pub fn new(f: F) -> Self {
        Self {
            f,
            _marker: PhantomData,
        }
    }
}

impl<T, F> Operation<T> for FnOperation<T, F>
where
    T: Interceptable,
    F: Fn(&T::Input) -> Pin<Box<dyn Future<Output = T::Output> + Send + '_>> + Send + Sync,
{
    fn execute<'a>(
        &'a self,
        input: &'a T::Input,
    ) -> Pin<Box<dyn Future<Output = T::Output> + Send + 'a>>
    where
        T::Input: Sync,
    {
        (self.f)(input)
    }
}

/// A composable stack of interceptors.
///
/// Interceptors are executed in the order they are added:
/// - First added = outermost = sees request first, sees response last
/// - Last added = innermost = sees request last, sees response first
///
/// # Example
///
/// ```rust,ignore
/// use llm_stack_core::intercept::{InterceptorStack, Retry, ToolExec};
///
/// let stack = InterceptorStack::<ToolExec<()>>::new()
///     .with(Retry::default());
/// ```
pub struct InterceptorStack<T: Interceptable> {
    layers: Vec<Arc<dyn Interceptor<T>>>,
}

impl<T: Interceptable> Clone for InterceptorStack<T> {
    fn clone(&self) -> Self {
        Self {
            layers: self.layers.clone(),
        }
    }
}

impl<T: Interceptable> InterceptorStack<T> {
    /// Create an empty interceptor stack.
    pub fn new() -> Self {
        Self { layers: Vec::new() }
    }

    /// Add an interceptor to the stack.
    ///
    /// Interceptors are executed in the order added (first = outermost).
    #[must_use]
    pub fn with<I: Interceptor<T> + 'static>(mut self, interceptor: I) -> Self {
        self.layers.push(Arc::new(interceptor));
        self
    }

    /// Add a shared interceptor instance.
    ///
    /// Useful when the same interceptor instance needs to be used across
    /// multiple stacks (e.g., for shared metrics collection).
    #[must_use]
    pub fn with_shared(mut self, interceptor: Arc<dyn Interceptor<T>>) -> Self {
        self.layers.push(interceptor);
        self
    }

    /// Check if the stack has any interceptors.
    pub fn is_empty(&self) -> bool {
        self.layers.is_empty()
    }

    /// Get the number of interceptors in the stack.
    pub fn len(&self) -> usize {
        self.layers.len()
    }

    /// Execute an operation through the interceptor stack.
    pub async fn execute<'a, O>(&'a self, input: &'a T::Input, operation: &'a O) -> T::Output
    where
        T::Input: Sync,
        O: Operation<T>,
    {
        let next = Next {
            interceptors: &self.layers,
            operation,
        };
        next.run(input).await
    }

    /// Execute with a closure as the operation.
    pub async fn execute_fn<'a, F>(&'a self, input: &'a T::Input, f: F) -> T::Output
    where
        T::Input: Sync,
        F: Fn(&T::Input) -> Pin<Box<dyn Future<Output = T::Output> + Send + '_>> + Send + Sync,
    {
        let op = FnOperation::<T, F>::new(f);
        self.execute(input, &op).await
    }
}

impl<T: Interceptable> Default for InterceptorStack<T> {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Domain markers
// ============================================================================

/// Domain-specific marker types and their Interceptable implementations.
pub mod domain {
    use super::Interceptable;
    use crate::ChatResponse;
    use crate::error::LlmError;
    use crate::provider::ChatParams;
    use serde_json::Value;
    use std::marker::PhantomData;

    /// Marker for LLM provider calls.
    ///
    /// - Input: [`ChatParams`]
    /// - Output: `Result<ChatResponse, LlmError>`
    ///
    /// # Status: Reserved
    ///
    /// This domain marker is defined for future use but **not yet integrated**.
    /// Unlike [`ToolExec`] (which is wired into [`ToolRegistry`](crate::ToolRegistry)),
    /// there is currently no `Provider` wrapper that executes through an
    /// `InterceptorStack<LlmCall>`.
    ///
    /// The marker exists so that:
    /// 1. Generic interceptors (`Retry`, `Timeout`, `Logging`) already work with it
    /// 2. Future provider-level interception can be added without breaking changes
    ///
    /// To use `LlmCall` today, you would need to build your own wrapper that
    /// implements `Provider` and delegates through an `InterceptorStack<LlmCall>`.
    pub struct LlmCall;

    impl Interceptable for LlmCall {
        type Input = ChatParams;
        type Output = Result<ChatResponse, LlmError>;
    }

    /// Marker for tool executions.
    ///
    /// The `Ctx` type parameter matches the context type used by `ToolRegistry<Ctx>`.
    ///
    /// - Input: [`ToolRequest`]
    /// - Output: [`ToolResponse`]
    pub struct ToolExec<Ctx = ()>(PhantomData<fn() -> Ctx>);

    impl<Ctx: Send + Sync + 'static> Interceptable for ToolExec<Ctx> {
        type Input = ToolRequest;
        type Output = ToolResponse;
    }

    /// Input for tool execution.
    #[derive(Debug, Clone)]
    pub struct ToolRequest {
        /// Name of the tool being called.
        pub name: String,

        /// Unique ID of this tool call.
        pub call_id: String,

        /// Arguments passed to the tool (JSON).
        pub arguments: Value,
    }

    /// Output from tool execution.
    #[derive(Debug, Clone)]
    pub struct ToolResponse {
        /// The tool's output content.
        pub content: String,

        /// Whether the execution resulted in an error.
        pub is_error: bool,
    }

    impl ToolResponse {
        /// Create a successful response.
        pub fn success(content: impl Into<String>) -> Self {
            Self {
                content: content.into(),
                is_error: false,
            }
        }

        /// Create an error response.
        pub fn error(content: impl Into<String>) -> Self {
            Self {
                content: content.into(),
                is_error: true,
            }
        }
    }
}

// ============================================================================
// Behavior traits
// ============================================================================

/// Behavior traits that interceptors can require.
pub mod behavior {
    use crate::ChatResponse;
    use crate::error::LlmError;
    use crate::provider::ChatParams;
    use std::time::Duration;

    use super::domain::{ToolRequest, ToolResponse};

    /// Output that can indicate whether retry is appropriate.
    pub trait Retryable {
        /// Returns true if the operation should be retried.
        fn should_retry(&self) -> bool;
    }

    impl Retryable for Result<ChatResponse, LlmError> {
        fn should_retry(&self) -> bool {
            match self {
                Ok(_) => false,
                Err(e) => e.is_retryable(),
            }
        }
    }

    impl Retryable for ToolResponse {
        fn should_retry(&self) -> bool {
            // By default, tool errors are not retried
            // Users can implement custom retry logic via interceptors
            false
        }
    }

    /// Output that can represent a timeout.
    pub trait Timeoutable: Sized {
        /// Create a timeout error.
        fn timeout_error(duration: Duration) -> Self;
    }

    impl Timeoutable for Result<ChatResponse, LlmError> {
        fn timeout_error(duration: Duration) -> Self {
            Err(LlmError::Timeout {
                elapsed_ms: duration.as_millis() as u64,
            })
        }
    }

    impl Timeoutable for ToolResponse {
        fn timeout_error(duration: Duration) -> Self {
            ToolResponse {
                content: format!("Tool execution timed out after {duration:?}"),
                is_error: true,
            }
        }
    }

    /// Input that can describe itself for logging.
    pub trait Loggable {
        /// Return a description of the operation for logging.
        fn log_description(&self) -> String;
    }

    impl Loggable for ChatParams {
        fn log_description(&self) -> String {
            let tool_count = self.tools.as_ref().map_or(0, Vec::len);
            format!(
                "LLM request: {} messages, {} tools",
                self.messages.len(),
                tool_count
            )
        }
    }

    impl Loggable for ToolRequest {
        fn log_description(&self) -> String {
            format!("Tool call: {} ({})", self.name, self.call_id)
        }
    }

    /// Output that can report success/failure for logging.
    ///
    /// Separate from `Retryable` because logging success != retry decision.
    /// A successful response might still be retryable (e.g., partial results),
    /// and a failed response might not be retryable (e.g., auth error).
    pub trait Outcome {
        /// Returns true if the operation succeeded.
        fn is_success(&self) -> bool;
    }

    impl Outcome for Result<ChatResponse, LlmError> {
        fn is_success(&self) -> bool {
            self.is_ok()
        }
    }

    impl Outcome for ToolResponse {
        fn is_success(&self) -> bool {
            !self.is_error
        }
    }
}

// ============================================================================
// Built-in interceptors
// ============================================================================

/// Built-in interceptors for common cross-cutting concerns.
pub mod interceptors {
    #[cfg(feature = "tracing")]
    use super::behavior::{Loggable, Outcome};
    use super::behavior::{Retryable, Timeoutable};
    use super::{Interceptable, Interceptor, Next};
    use std::future::Future;
    use std::pin::Pin;
    use std::time::Duration;

    /// Retry interceptor with exponential backoff.
    ///
    /// Retries the operation when the output indicates failure via [`Retryable`].
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// use llm_stack_core::intercept::{InterceptorStack, Retry, ToolExec};
    /// use std::time::Duration;
    ///
    /// let stack = InterceptorStack::<ToolExec<()>>::new()
    ///     .with(Retry::new(3, Duration::from_millis(100)));
    /// ```
    #[derive(Debug, Clone)]
    pub struct Retry {
        /// Maximum number of attempts (including the first).
        pub max_attempts: u32,

        /// Initial delay before first retry.
        pub initial_delay: Duration,

        /// Maximum delay between retries.
        pub max_delay: Duration,

        /// Multiplier for exponential backoff.
        pub multiplier: f64,
    }

    impl Default for Retry {
        fn default() -> Self {
            Self {
                max_attempts: 3,
                initial_delay: Duration::from_millis(500),
                max_delay: Duration::from_secs(30),
                multiplier: 2.0,
            }
        }
    }

    impl Retry {
        /// Create a retry interceptor with the given attempts and initial delay.
        pub fn new(max_attempts: u32, initial_delay: Duration) -> Self {
            Self {
                max_attempts,
                initial_delay,
                ..Default::default()
            }
        }

        fn delay_for_attempt(&self, attempt: u32) -> Duration {
            let delay_ms = self.initial_delay.as_millis() as f64
                * self.multiplier.powi(attempt.saturating_sub(1) as i32);
            let delay = Duration::from_millis(delay_ms as u64);
            std::cmp::min(delay, self.max_delay)
        }
    }

    impl<T> Interceptor<T> for Retry
    where
        T: Interceptable,
        T::Input: Sync,
        T::Output: Retryable,
    {
        fn intercept<'a>(
            &'a self,
            input: &'a T::Input,
            next: Next<'a, T>,
        ) -> Pin<Box<dyn Future<Output = T::Output> + Send + 'a>> {
            Box::pin(async move {
                let mut last_result: Option<T::Output> = None;

                for attempt in 1..=self.max_attempts {
                    let result = next.run(input).await;

                    if !result.should_retry() || attempt == self.max_attempts {
                        return result;
                    }

                    // Sleep before retry
                    let delay = self.delay_for_attempt(attempt);
                    tokio::time::sleep(delay).await;

                    last_result = Some(result);
                }

                // Should not reach here, but return last result if we do
                last_result.expect("at least one attempt should have been made")
            })
        }
    }

    /// Timeout interceptor.
    ///
    /// Wraps the operation with a timeout. If the timeout expires, returns
    /// an error via [`Timeoutable`].
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// use llm_stack_core::intercept::{InterceptorStack, Timeout, ToolExec};
    /// use std::time::Duration;
    ///
    /// let stack = InterceptorStack::<ToolExec<()>>::new()
    ///     .with(Timeout::new(Duration::from_secs(30)));
    /// ```
    #[derive(Debug, Clone)]
    pub struct Timeout {
        /// Maximum duration for the operation.
        pub duration: Duration,
    }

    impl Timeout {
        /// Create a timeout interceptor with the given duration.
        pub fn new(duration: Duration) -> Self {
            Self { duration }
        }
    }

    impl<T> Interceptor<T> for Timeout
    where
        T: Interceptable,
        T::Input: Sync,
        T::Output: Timeoutable,
    {
        fn intercept<'a>(
            &'a self,
            input: &'a T::Input,
            next: Next<'a, T>,
        ) -> Pin<Box<dyn Future<Output = T::Output> + Send + 'a>> {
            let duration = self.duration;
            Box::pin(async move {
                match tokio::time::timeout(duration, next.run(input)).await {
                    Ok(result) => result,
                    Err(_) => T::Output::timeout_error(duration),
                }
            })
        }
    }

    /// Pass-through interceptor that does nothing.
    ///
    /// Useful for testing and as a placeholder.
    #[derive(Debug, Clone, Default)]
    pub struct NoOp;

    impl<T> Interceptor<T> for NoOp
    where
        T: Interceptable,
        T::Input: Sync,
    {
        fn intercept<'a>(
            &'a self,
            input: &'a T::Input,
            next: Next<'a, T>,
        ) -> Pin<Box<dyn Future<Output = T::Output> + Send + 'a>> {
            Box::pin(next.run(input))
        }
    }

    /// Logging interceptor using tracing.
    ///
    /// Logs operation start/completion with configurable verbosity.
    /// Requires the `tracing` feature.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// use llm_stack_core::intercept::{InterceptorStack, Logging, LogLevel, ToolExec};
    ///
    /// let stack = InterceptorStack::<ToolExec<()>>::new()
    ///     .with(Logging::new(LogLevel::Debug));
    /// ```
    #[cfg(feature = "tracing")]
    #[derive(Debug, Clone)]
    pub struct Logging {
        /// Verbosity level for log output.
        pub level: LogLevel,
    }

    /// Verbosity level for logging interceptor.
    #[cfg(feature = "tracing")]
    #[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
    pub enum LogLevel {
        /// Basic logging: operation type and duration.
        #[default]
        Info,
        /// Verbose logging: includes success/failure status.
        Debug,
        /// Trace logging: includes input descriptions.
        Trace,
    }

    #[cfg(feature = "tracing")]
    impl Default for Logging {
        fn default() -> Self {
            Self {
                level: LogLevel::Info,
            }
        }
    }

    #[cfg(feature = "tracing")]
    impl Logging {
        /// Create a logging interceptor with the given level.
        pub fn new(level: LogLevel) -> Self {
            Self { level }
        }
    }

    #[cfg(feature = "tracing")]
    impl<T> Interceptor<T> for Logging
    where
        T: Interceptable,
        T::Input: Sync + Loggable,
        T::Output: Outcome,
    {
        fn intercept<'a>(
            &'a self,
            input: &'a T::Input,
            next: Next<'a, T>,
        ) -> Pin<Box<dyn Future<Output = T::Output> + Send + 'a>> {
            let description = input.log_description();
            let level = self.level;

            Box::pin(async move {
                let start = std::time::Instant::now();

                if level == LogLevel::Trace {
                    tracing::debug!(description = %description, "operation starting");
                }

                let result = next.run(input).await;
                let duration = start.elapsed();
                let success = result.is_success();

                match level {
                    LogLevel::Info => {
                        tracing::info!(
                            duration_ms = duration.as_millis() as u64,
                            "operation completed"
                        );
                    }
                    LogLevel::Debug | LogLevel::Trace => {
                        tracing::debug!(
                            duration_ms = duration.as_millis() as u64,
                            success,
                            "operation completed"
                        );
                    }
                }

                result
            })
        }
    }
}

// Re-export interceptors at module level
#[cfg(feature = "tracing")]
pub use interceptors::{LogLevel, Logging};
pub use interceptors::{NoOp, Retry, Timeout};

// ============================================================================
// Domain-specific interceptors
// ============================================================================

/// Tool-specific interceptors.
pub mod tool_interceptors {
    use super::{
        Interceptor, Next,
        domain::{ToolExec, ToolRequest, ToolResponse},
    };
    use serde_json::Value;
    use std::future::Future;
    use std::pin::Pin;

    /// Decision returned by an approval check function.
    #[derive(Debug, Clone)]
    pub enum ApprovalDecision {
        /// Allow the tool call to proceed.
        Allow,
        /// Deny the tool call with an error message.
        Deny(String),
        /// Modify the tool call arguments before proceeding.
        Modify(Value),
    }

    /// Approval gate interceptor for tool calls.
    ///
    /// Runs a check function before each tool execution. The function can:
    /// - Allow the call to proceed unchanged
    /// - Deny the call with an error message
    /// - Modify the arguments before proceeding
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// use llm_stack_core::intercept::{InterceptorStack, ToolExec, Approval, ApprovalDecision};
    ///
    /// let stack = InterceptorStack::<ToolExec<()>>::new()
    ///     .with(Approval::new(|req| {
    ///         if req.name == "delete_file" {
    ///             ApprovalDecision::Deny("Destructive operations not allowed".into())
    ///         } else {
    ///             ApprovalDecision::Allow
    ///         }
    ///     }));
    /// ```
    pub struct Approval<F> {
        check: F,
    }

    impl<F> Approval<F>
    where
        F: Fn(&ToolRequest) -> ApprovalDecision + Send + Sync,
    {
        /// Create an approval interceptor with the given check function.
        pub fn new(check: F) -> Self {
            Self { check }
        }
    }

    impl<F> std::fmt::Debug for Approval<F> {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            f.debug_struct("Approval").finish_non_exhaustive()
        }
    }

    impl<Ctx, F> Interceptor<ToolExec<Ctx>> for Approval<F>
    where
        Ctx: Send + Sync + 'static,
        F: Fn(&ToolRequest) -> ApprovalDecision + Send + Sync,
    {
        fn intercept<'a>(
            &'a self,
            input: &'a ToolRequest,
            next: Next<'a, ToolExec<Ctx>>,
        ) -> Pin<Box<dyn Future<Output = ToolResponse> + Send + 'a>> {
            Box::pin(async move {
                match (self.check)(input) {
                    ApprovalDecision::Allow => next.run(input).await,
                    ApprovalDecision::Deny(reason) => ToolResponse {
                        content: reason,
                        is_error: true,
                    },
                    ApprovalDecision::Modify(new_args) => {
                        let modified = ToolRequest {
                            name: input.name.clone(),
                            call_id: input.call_id.clone(),
                            arguments: new_args,
                        };
                        next.run(&modified).await
                    }
                }
            })
        }
    }
}

pub use tool_interceptors::{Approval, ApprovalDecision};

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU32, Ordering};
    use std::time::Duration;

    // Simple test domain
    struct TestOp;

    impl Interceptable for TestOp {
        type Input = String;
        type Output = Result<String, String>;
    }

    impl behavior::Retryable for Result<String, String> {
        fn should_retry(&self) -> bool {
            self.is_err()
        }
    }

    impl behavior::Timeoutable for Result<String, String> {
        fn timeout_error(duration: Duration) -> Self {
            Err(format!("timeout after {duration:?}"))
        }
    }

    struct EchoOp;

    impl Operation<TestOp> for EchoOp {
        fn execute<'a>(
            &'a self,
            input: &'a String,
        ) -> Pin<Box<dyn Future<Output = Result<String, String>> + Send + 'a>> {
            Box::pin(async move { Ok(format!("echo: {input}")) })
        }
    }

    struct FailOp {
        failures: AtomicU32,
        max_failures: u32,
    }

    impl FailOp {
        fn new(max_failures: u32) -> Self {
            Self {
                failures: AtomicU32::new(0),
                max_failures,
            }
        }
    }

    impl Operation<TestOp> for FailOp {
        fn execute<'a>(
            &'a self,
            input: &'a String,
        ) -> Pin<Box<dyn Future<Output = Result<String, String>> + Send + 'a>> {
            Box::pin(async move {
                let count = self.failures.fetch_add(1, Ordering::SeqCst);
                if count < self.max_failures {
                    let failure_num = count + 1;
                    Err(format!("failure {failure_num}"))
                } else {
                    Ok(format!("success after {count} failures: {input}"))
                }
            })
        }
    }

    #[tokio::test]
    async fn empty_stack_passthrough() {
        let stack = InterceptorStack::<TestOp>::new();
        let input = "hello".to_string();
        let result = stack.execute(&input, &EchoOp).await;
        assert_eq!(result, Ok("echo: hello".to_string()));
    }

    #[tokio::test]
    async fn noop_interceptor_passthrough() {
        let stack = InterceptorStack::<TestOp>::new().with(NoOp);
        let input = "test".to_string();
        let result = stack.execute(&input, &EchoOp).await;
        assert_eq!(result, Ok("echo: test".to_string()));
    }

    #[tokio::test]
    async fn multiple_noop_interceptors() {
        let stack = InterceptorStack::<TestOp>::new()
            .with(NoOp)
            .with(NoOp)
            .with(NoOp);
        let input = "multi".to_string();
        let result = stack.execute(&input, &EchoOp).await;
        assert_eq!(result, Ok("echo: multi".to_string()));
    }

    #[tokio::test]
    async fn retry_succeeds_after_failures() {
        let stack = InterceptorStack::<TestOp>::new().with(Retry::new(3, Duration::from_millis(1)));

        let op = FailOp::new(2); // Fail twice, then succeed
        let input = "retry-test".to_string();
        let result = stack.execute(&input, &op).await;

        assert!(result.is_ok());
        assert!(result.unwrap().contains("success after 2 failures"));
    }

    #[tokio::test]
    async fn retry_exhausted() {
        let stack = InterceptorStack::<TestOp>::new().with(Retry::new(2, Duration::from_millis(1)));

        let op = FailOp::new(10); // Always fail
        let input = "exhaust".to_string();
        let result = stack.execute(&input, &op).await;

        assert!(result.is_err());
        assert!(result.unwrap_err().contains("failure"));
    }

    #[tokio::test]
    async fn timeout_success() {
        let stack = InterceptorStack::<TestOp>::new().with(Timeout::new(Duration::from_secs(1)));
        let input = "fast".to_string();
        let result = stack.execute(&input, &EchoOp).await;
        assert_eq!(result, Ok("echo: fast".to_string()));
    }

    #[tokio::test]
    async fn timeout_expires() {
        struct SlowOp;

        impl Operation<TestOp> for SlowOp {
            fn execute<'a>(
                &'a self,
                _input: &'a String,
            ) -> Pin<Box<dyn Future<Output = Result<String, String>> + Send + 'a>> {
                Box::pin(async {
                    tokio::time::sleep(Duration::from_secs(10)).await;
                    Ok("should not reach".to_string())
                })
            }
        }

        let stack = InterceptorStack::<TestOp>::new().with(Timeout::new(Duration::from_millis(10)));
        let input = "slow".to_string();
        let result = stack.execute(&input, &SlowOp).await;

        assert!(result.is_err());
        assert!(result.unwrap_err().contains("timeout"));
    }

    #[tokio::test]
    async fn interceptor_ordering() {
        use std::sync::Mutex;

        struct RecordingInterceptor {
            name: &'static str,
            log: Arc<Mutex<Vec<String>>>,
        }

        impl Interceptor<TestOp> for RecordingInterceptor {
            fn intercept<'a>(
                &'a self,
                input: &'a String,
                next: Next<'a, TestOp>,
            ) -> Pin<Box<dyn Future<Output = Result<String, String>> + Send + 'a>> {
                let name = self.name;
                let log = Arc::clone(&self.log);
                Box::pin(async move {
                    log.lock().unwrap().push(format!("{name}-before"));
                    let result = next.run(input).await;
                    log.lock().unwrap().push(format!("{name}-after"));
                    result
                })
            }
        }

        let log = Arc::new(Mutex::new(Vec::new()));

        let stack = InterceptorStack::<TestOp>::new()
            .with(RecordingInterceptor {
                name: "A",
                log: Arc::clone(&log),
            })
            .with(RecordingInterceptor {
                name: "B",
                log: Arc::clone(&log),
            });

        let input = "order".to_string();
        let _ = stack.execute(&input, &EchoOp).await;

        let recorded = log.lock().unwrap().clone();
        assert_eq!(recorded, vec!["A-before", "B-before", "B-after", "A-after"]);
    }

    #[tokio::test]
    async fn short_circuit_interceptor() {
        struct ShortCircuit;

        impl Interceptor<TestOp> for ShortCircuit {
            fn intercept<'a>(
                &'a self,
                _input: &'a String,
                _next: Next<'a, TestOp>,
            ) -> Pin<Box<dyn Future<Output = Result<String, String>> + Send + 'a>> {
                Box::pin(async { Err("short-circuited".to_string()) })
            }
        }

        let stack = InterceptorStack::<TestOp>::new()
            .with(ShortCircuit)
            .with(NoOp); // This should never run

        let input = "blocked".to_string();
        let result = stack.execute(&input, &EchoOp).await;

        assert_eq!(result, Err("short-circuited".to_string()));
    }

    #[tokio::test]
    async fn execute_with_closure() {
        let stack = InterceptorStack::<TestOp>::new().with(NoOp);

        let input = "closure-test".to_string();
        let result = stack
            .execute_fn(&input, |i| Box::pin(async move { Ok(format!("fn: {i}")) }))
            .await;

        assert_eq!(result, Ok("fn: closure-test".to_string()));
    }

    #[tokio::test]
    async fn next_is_copy() {
        // Test that Next can be used multiple times (for retry)
        struct MultiCallInterceptor {
            calls: AtomicU32,
        }

        impl Interceptor<TestOp> for MultiCallInterceptor {
            fn intercept<'a>(
                &'a self,
                input: &'a String,
                next: Next<'a, TestOp>,
            ) -> Pin<Box<dyn Future<Output = Result<String, String>> + Send + 'a>> {
                Box::pin(async move {
                    // Call next twice to verify Copy works
                    let _ = next.run(input).await;
                    self.calls.fetch_add(1, Ordering::SeqCst);
                    next.run(input).await
                })
            }
        }

        let interceptor = MultiCallInterceptor {
            calls: AtomicU32::new(0),
        };

        let stack = InterceptorStack::<TestOp>::new().with(interceptor);
        let input = "copy-test".to_string();
        let result = stack.execute(&input, &EchoOp).await;

        assert_eq!(result, Ok("echo: copy-test".to_string()));
    }

    #[tokio::test]
    async fn shared_interceptor() {
        let shared: Arc<dyn Interceptor<TestOp>> = Arc::new(NoOp);

        let stack1 = InterceptorStack::<TestOp>::new().with_shared(Arc::clone(&shared));

        let stack2 = InterceptorStack::<TestOp>::new().with_shared(Arc::clone(&shared));

        let input = "shared".to_string();
        let r1 = stack1.execute(&input, &EchoOp).await;
        let r2 = stack2.execute(&input, &EchoOp).await;

        assert_eq!(r1, Ok("echo: shared".to_string()));
        assert_eq!(r2, Ok("echo: shared".to_string()));
    }

    #[test]
    fn stack_len_and_is_empty() {
        let empty: InterceptorStack<TestOp> = InterceptorStack::new();
        assert!(empty.is_empty());
        assert_eq!(empty.len(), 0);

        let one = InterceptorStack::<TestOp>::new().with(NoOp);
        assert!(!one.is_empty());
        assert_eq!(one.len(), 1);

        let two = InterceptorStack::<TestOp>::new().with(NoOp).with(NoOp);
        assert_eq!(two.len(), 2);
    }

    // =========================================================================
    // Approval interceptor tests
    // =========================================================================

    mod approval_tests {
        use super::*;
        use crate::intercept::domain::{ToolExec, ToolRequest, ToolResponse};
        use crate::intercept::tool_interceptors::{Approval, ApprovalDecision};
        use serde_json::json;

        struct EchoToolOp;

        impl Operation<ToolExec<()>> for EchoToolOp {
            fn execute<'a>(
                &'a self,
                input: &'a ToolRequest,
            ) -> Pin<Box<dyn Future<Output = ToolResponse> + Send + 'a>> {
                Box::pin(async move {
                    ToolResponse {
                        content: format!("executed: {} with {:?}", input.name, input.arguments),
                        is_error: false,
                    }
                })
            }
        }

        #[tokio::test]
        async fn approval_allow() {
            let stack = InterceptorStack::<ToolExec<()>>::new()
                .with(Approval::new(|_| ApprovalDecision::Allow));

            let input = ToolRequest {
                name: "test_tool".into(),
                call_id: "call_1".into(),
                arguments: json!({"x": 1}),
            };

            let result = stack.execute(&input, &EchoToolOp).await;
            assert!(!result.is_error);
            assert!(result.content.contains("test_tool"));
        }

        #[tokio::test]
        async fn approval_deny() {
            let stack = InterceptorStack::<ToolExec<()>>::new().with(Approval::new(|req| {
                if req.name == "dangerous" {
                    ApprovalDecision::Deny("Not allowed".into())
                } else {
                    ApprovalDecision::Allow
                }
            }));

            let input = ToolRequest {
                name: "dangerous".into(),
                call_id: "call_2".into(),
                arguments: json!({}),
            };

            let result = stack.execute(&input, &EchoToolOp).await;
            assert!(result.is_error);
            assert_eq!(result.content, "Not allowed");
        }

        #[tokio::test]
        async fn approval_modify() {
            let stack = InterceptorStack::<ToolExec<()>>::new().with(Approval::new(|req| {
                // Always add a "modified" field
                let mut args = req.arguments.clone();
                args["modified"] = json!(true);
                ApprovalDecision::Modify(args)
            }));

            let input = ToolRequest {
                name: "my_tool".into(),
                call_id: "call_3".into(),
                arguments: json!({"original": "value"}),
            };

            let result = stack.execute(&input, &EchoToolOp).await;
            assert!(!result.is_error);
            assert!(result.content.contains("modified"));
            assert!(result.content.contains("true"));
        }

        #[tokio::test]
        async fn approval_debug() {
            let approval = Approval::new(|_: &ToolRequest| ApprovalDecision::Allow);
            let debug_str = format!("{approval:?}");
            assert!(debug_str.contains("Approval"));
        }
    }
}
