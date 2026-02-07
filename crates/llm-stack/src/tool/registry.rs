//! Tool registry for managing and executing tools.

use std::collections::HashMap;
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use std::time::Duration;

use rand::Rng;

use super::ToolHandler;
use crate::chat::{ToolCall, ToolResult};
use crate::intercept::domain::{ToolExec, ToolRequest, ToolResponse};
use crate::intercept::{InterceptorStack, Operation};
use crate::provider::{ToolDefinition, ToolRetryConfig};

/// A registry of tool handlers, indexed by name.
///
/// Generic over context type `Ctx` which is passed to tool handlers on
/// execution. Default is `()` for backwards compatibility.
///
/// Provides validation of tool call arguments against their schemas
/// and parallel execution of multiple tool calls.
///
/// # Interceptors
///
/// Tool execution can be wrapped with interceptors for cross-cutting concerns
/// like logging, approval gates, or rate limiting:
///
/// ```rust,ignore
/// use llm_stack::{ToolRegistry, tool_fn};
/// use llm_stack::intercept::{InterceptorStack, ToolExec, Approval, ApprovalDecision};
///
/// let mut registry: ToolRegistry<()> = ToolRegistry::new()
///     .with_interceptors(
///         InterceptorStack::<ToolExec<()>>::new()
///             .with(Approval::new(|req| {
///                 if req.name.starts_with("dangerous_") {
///                     ApprovalDecision::Deny("Not allowed".into())
///                 } else {
///                     ApprovalDecision::Allow
///                 }
///             }))
///     );
/// ```
pub struct ToolRegistry<Ctx = ()>
where
    Ctx: Send + Sync + 'static,
{
    pub(crate) handlers: HashMap<String, Arc<dyn ToolHandler<Ctx>>>,
    interceptors: InterceptorStack<ToolExec<Ctx>>,
}

impl<Ctx> Default for ToolRegistry<Ctx>
where
    Ctx: Send + Sync + 'static,
{
    fn default() -> Self {
        Self {
            handlers: HashMap::new(),
            interceptors: InterceptorStack::new(),
        }
    }
}

impl<Ctx> Clone for ToolRegistry<Ctx>
where
    Ctx: Send + Sync + 'static,
{
    /// Clone the registry.
    ///
    /// This is cheap — it clones `Arc` pointers to handlers, not the
    /// handlers themselves.
    fn clone(&self) -> Self {
        Self {
            handlers: self.handlers.clone(),
            interceptors: self.interceptors.clone(),
        }
    }
}

impl<Ctx> std::fmt::Debug for ToolRegistry<Ctx>
where
    Ctx: Send + Sync + 'static,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ToolRegistry")
            .field("tools", &self.handlers.keys().collect::<Vec<_>>())
            .field("interceptors", &self.interceptors.len())
            .finish()
    }
}

impl<Ctx: Send + Sync + 'static> ToolRegistry<Ctx> {
    /// Creates an empty registry.
    pub fn new() -> Self {
        Self::default()
    }

    /// Registers a tool handler.
    ///
    /// If a handler with the same name already exists, it is replaced.
    pub fn register(&mut self, handler: impl ToolHandler<Ctx> + 'static) -> &mut Self {
        let name = handler.definition().name.clone();
        self.handlers.insert(name, Arc::new(handler));
        self
    }

    /// Registers a shared tool handler.
    pub fn register_shared(&mut self, handler: Arc<dyn ToolHandler<Ctx>>) -> &mut Self {
        let name = handler.definition().name.clone();
        self.handlers.insert(name, handler);
        self
    }

    /// Returns the handler for the given tool name.
    pub fn get(&self, name: &str) -> Option<&Arc<dyn ToolHandler<Ctx>>> {
        self.handlers.get(name)
    }

    /// Returns whether a tool with the given name is registered.
    pub fn contains(&self, name: &str) -> bool {
        self.handlers.contains_key(name)
    }

    /// Returns the definitions of all registered tools.
    ///
    /// Pass this to [`ChatParams::tools`](crate::provider::ChatParams::tools) to tell the model which
    /// tools are available.
    pub fn definitions(&self) -> Vec<ToolDefinition> {
        self.handlers.values().map(|h| h.definition()).collect()
    }

    /// Returns the number of registered tools.
    pub fn len(&self) -> usize {
        self.handlers.len()
    }

    /// Returns true if no tools are registered.
    pub fn is_empty(&self) -> bool {
        self.handlers.is_empty()
    }

    /// Returns a new registry excluding the named tools.
    ///
    /// Useful for creating scoped registries in Master/Worker patterns
    /// where workers should not have access to certain tools (e.g., `spawn_task`).
    ///
    /// # Example
    ///
    /// ```rust
    /// use llm_stack::ToolRegistry;
    ///
    /// let master_registry: ToolRegistry<()> = ToolRegistry::new();
    /// // ... register tools ...
    ///
    /// // Workers can't spawn or use admin tools
    /// let worker_registry = master_registry.without(["spawn_task", "admin_tool"]);
    /// ```
    #[must_use]
    pub fn without<'a>(&self, names: impl IntoIterator<Item = &'a str>) -> Self {
        use std::collections::HashSet;
        let exclude: HashSet<&str> = names.into_iter().collect();
        let mut new = Self {
            handlers: HashMap::new(),
            interceptors: self.interceptors.clone(),
        };
        for (name, handler) in &self.handlers {
            if !exclude.contains(name.as_str()) {
                new.handlers.insert(name.clone(), Arc::clone(handler));
            }
        }
        new
    }

    /// Returns a new registry with only the named tools.
    ///
    /// Useful for creating minimal registries with specific capabilities.
    ///
    /// # Example
    ///
    /// ```rust
    /// use llm_stack::ToolRegistry;
    ///
    /// let full_registry: ToolRegistry<()> = ToolRegistry::new();
    /// // ... register tools ...
    ///
    /// // Read-only registry with just search tools
    /// let search_registry = full_registry.only(["search_docs", "search_web"]);
    /// ```
    #[must_use]
    pub fn only<'a>(&self, names: impl IntoIterator<Item = &'a str>) -> Self {
        use std::collections::HashSet;
        let include: HashSet<&str> = names.into_iter().collect();
        let mut new = Self {
            handlers: HashMap::new(),
            interceptors: self.interceptors.clone(),
        };
        for (name, handler) in &self.handlers {
            if include.contains(name.as_str()) {
                new.handlers.insert(name.clone(), Arc::clone(handler));
            }
        }
        new
    }

    /// Sets the interceptor stack for all tool executions.
    ///
    /// Interceptors run in the order added (first = outermost). They can
    /// inspect, modify, or block tool calls before they reach the handler.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// use llm_stack::{ToolRegistry, tool_fn};
    /// use llm_stack::intercept::{InterceptorStack, ToolExec, Approval, ApprovalDecision, Retry};
    ///
    /// let registry: ToolRegistry<()> = ToolRegistry::new()
    ///     .with_interceptors(
    ///         InterceptorStack::<ToolExec<()>>::new()
    ///             .with(Approval::new(|req| {
    ///                 if req.name == "dangerous" {
    ///                     ApprovalDecision::Deny("Not allowed".into())
    ///                 } else {
    ///                     ApprovalDecision::Allow
    ///                 }
    ///             }))
    ///             .with(Retry::default())
    ///     );
    /// ```
    #[must_use]
    pub fn with_interceptors(mut self, interceptors: InterceptorStack<ToolExec<Ctx>>) -> Self {
        self.interceptors = interceptors;
        self
    }

    /// Executes a single tool call with schema validation and optional retry.
    ///
    /// 1. Looks up the handler by [`ToolCall::name`]
    /// 2. Validates arguments against the tool's parameter schema
    /// 3. Runs the call through interceptors (if any)
    /// 4. Invokes the handler with the provided context
    /// 5. If the tool has retry configuration and execution fails,
    ///    retries with exponential backoff
    ///
    /// Returns a [`ToolResult`] (always succeeds at the outer level).
    /// Execution errors are captured in `ToolResult::is_error`.
    pub async fn execute(&self, call: &ToolCall, ctx: &Ctx) -> ToolResult {
        self.execute_inner(&call.name, &call.id, call.arguments.clone(), ctx)
            .await
    }

    /// Executes a tool by name with the given arguments.
    ///
    /// This is a lower-level method used internally when the tool call
    /// components are already separated (e.g., from `execute_with_events`).
    /// Accepts owned arguments to avoid an extra deep clone of `serde_json::Value`.
    pub(crate) async fn execute_by_name(
        &self,
        name: &str,
        call_id: &str,
        arguments: serde_json::Value,
        ctx: &Ctx,
    ) -> ToolResult {
        self.execute_inner(name, call_id, arguments, ctx).await
    }

    /// Shared implementation for `execute` and `execute_by_name`.
    async fn execute_inner(
        &self,
        name: &str,
        call_id: &str,
        arguments: serde_json::Value,
        ctx: &Ctx,
    ) -> ToolResult {
        let Some(handler) = self.handlers.get(name) else {
            return ToolResult {
                tool_call_id: call_id.to_string(),
                content: format!("Unknown tool: {name}"),
                is_error: true,
            };
        };

        // Validate arguments against schema
        #[cfg(feature = "schema")]
        {
            let definition = handler.definition();
            if let Err(e) = definition.parameters.validate(&arguments) {
                return ToolResult {
                    tool_call_id: call_id.to_string(),
                    content: format!("Invalid arguments for tool '{name}': {e}"),
                    is_error: true,
                };
            }
        }

        let request = ToolRequest {
            name: name.to_string(),
            call_id: call_id.to_string(),
            arguments,
        };

        let operation = ToolHandlerOperation {
            handler: handler.clone(),
            ctx,
            retry_config: handler.definition().retry,
        };

        let response = self.interceptors.execute(&request, &operation).await;

        ToolResult {
            tool_call_id: request.call_id,
            content: response.content,
            is_error: response.is_error,
        }
    }

    /// Executes multiple tool calls, preserving order.
    ///
    /// When `parallel` is true, all calls run concurrently via
    /// `futures::future::join_all`. When false, they run sequentially.
    pub async fn execute_all(
        &self,
        calls: &[ToolCall],
        ctx: &Ctx,
        parallel: bool,
    ) -> Vec<ToolResult> {
        if !parallel || calls.len() <= 1 {
            let mut results = Vec::with_capacity(calls.len());
            for call in calls {
                results.push(self.execute(call, ctx).await);
            }
            return results;
        }

        // Parallel execution using join_all (no spawn needed)
        let futures: Vec<_> = calls.iter().map(|call| self.execute(call, ctx)).collect();
        futures::future::join_all(futures).await
    }
}

/// Computes backoff duration with exponential growth and jitter.
///
/// Formula: `min(initial * multiplier^attempt, max) * random(1-jitter, 1)`
fn compute_backoff(config: &ToolRetryConfig, attempt: u32) -> Duration {
    // Safe to cast: attempt is bounded by max_retries which is u32,
    // and reasonable values are << i32::MAX
    #[allow(clippy::cast_possible_wrap)]
    let base =
        config.initial_backoff.as_secs_f64() * config.backoff_multiplier.powi(attempt as i32);
    let capped = base.min(config.max_backoff.as_secs_f64());

    // Apply jitter: random value in range [1-jitter, 1]
    let jitter_factor = if config.jitter > 0.0 {
        let min_factor = 1.0 - config.jitter;
        let mut rng = rand::rng();
        rng.random_range(min_factor..=1.0)
    } else {
        1.0
    };

    Duration::from_secs_f64(capped * jitter_factor)
}

/// Wraps a tool handler as an [`Operation`] for the interceptor stack.
///
/// This struct captures the handler, context, and retry config so that
/// the interceptor stack can execute the tool.
struct ToolHandlerOperation<'a, Ctx: Send + Sync + 'static> {
    handler: Arc<dyn ToolHandler<Ctx>>,
    ctx: &'a Ctx,
    retry_config: Option<ToolRetryConfig>,
}

impl<Ctx: Send + Sync + 'static> Operation<ToolExec<Ctx>> for ToolHandlerOperation<'_, Ctx> {
    fn execute<'b>(
        &'b self,
        input: &'b ToolRequest,
    ) -> Pin<Box<dyn Future<Output = ToolResponse> + Send + 'b>>
    where
        ToolRequest: Sync,
    {
        Box::pin(async move {
            match &self.retry_config {
                Some(config) => execute_with_retry(&self.handler, input, self.ctx, config).await,
                None => execute_once(&self.handler, input, self.ctx).await,
            }
        })
    }
}

/// Executes a tool once without retry.
async fn execute_once<Ctx: Send + Sync + 'static>(
    handler: &Arc<dyn ToolHandler<Ctx>>,
    request: &ToolRequest,
    ctx: &Ctx,
) -> ToolResponse {
    match handler.execute(request.arguments.clone(), ctx).await {
        Ok(output) => ToolResponse {
            content: output.content,
            is_error: false,
        },
        Err(e) => ToolResponse {
            content: e.message,
            is_error: true,
        },
    }
}

/// Executes a tool with retry logic.
async fn execute_with_retry<Ctx: Send + Sync + 'static>(
    handler: &Arc<dyn ToolHandler<Ctx>>,
    request: &ToolRequest,
    ctx: &Ctx,
    config: &ToolRetryConfig,
) -> ToolResponse {
    let mut attempt = 0u32;

    loop {
        match handler.execute(request.arguments.clone(), ctx).await {
            Ok(output) => {
                return ToolResponse {
                    content: output.content,
                    is_error: false,
                };
            }
            Err(e) => {
                let error_msg = e.message;

                // Check if we should retry this error
                let should_retry = config
                    .retry_if
                    .as_ref()
                    .is_none_or(|predicate| predicate(&error_msg));

                if !should_retry || attempt >= config.max_retries {
                    return ToolResponse {
                        content: error_msg,
                        is_error: true,
                    };
                }

                // Calculate backoff with jitter
                let backoff = compute_backoff(config, attempt);
                tokio::time::sleep(backoff).await;

                attempt += 1;
            }
        }
    }
}
