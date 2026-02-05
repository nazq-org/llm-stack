//! Tool handler trait and implementations.

use std::future::Future;
use std::marker::PhantomData;
use std::pin::Pin;

use serde_json::Value;

use super::{ToolError, ToolOutput};
use crate::provider::ToolDefinition;

/// A single tool that can be invoked by the LLM.
///
/// Implement this trait for tools that need complex state or lifetime
/// management. For simple tools, use [`super::tool_fn`] to wrap a closure.
///
/// The trait is generic over a context type `Ctx` which is passed to
/// `execute()`. This allows tools to access shared state like database
/// connections or user identity without closure capture. The default
/// context type is `()` for backwards compatibility.
///
/// The trait is object-safe (uses boxed futures) so handlers can be
/// stored as `Arc<dyn ToolHandler<Ctx>>`.
///
/// # Example with Context
///
/// ```rust
/// use llm_stack::tool::{ToolHandler, ToolOutput, ToolError};
/// use llm_stack::{ToolDefinition, JsonSchema};
/// use serde_json::{json, Value};
/// use std::future::Future;
/// use std::pin::Pin;
///
/// struct AppContext {
///     user_id: String,
/// }
///
/// struct UserInfoTool;
///
/// impl ToolHandler<AppContext> for UserInfoTool {
///     fn definition(&self) -> ToolDefinition {
///         ToolDefinition {
///             name: "get_user_info".into(),
///             description: "Get current user info".into(),
///             parameters: JsonSchema::new(json!({"type": "object"})),
///             retry: None,
///         }
///     }
///
///     fn execute<'a>(
///         &'a self,
///         _input: Value,
///         ctx: &'a AppContext,
///     ) -> Pin<Box<dyn Future<Output = Result<ToolOutput, ToolError>> + Send + 'a>> {
///         Box::pin(async move {
///             Ok(ToolOutput::new(format!("User: {}", ctx.user_id)))
///         })
///     }
/// }
/// ```
pub trait ToolHandler<Ctx = ()>: Send + Sync {
    /// Returns the tool's definition (name, description, parameter schema).
    fn definition(&self) -> ToolDefinition;

    /// Executes the tool with the given JSON arguments and context.
    ///
    /// Returns a [`ToolOutput`] containing the content for the LLM and
    /// optional metadata for application use. Providers expect tool results
    /// as text content — callers should `serde_json::to_string()` if they
    /// have structured data.
    fn execute<'a>(
        &'a self,
        input: Value,
        ctx: &'a Ctx,
    ) -> Pin<Box<dyn Future<Output = Result<ToolOutput, ToolError>> + Send + 'a>>;
}

/// A tool handler backed by an async closure.
///
/// Created via [`super::tool_fn`] or [`super::tool_fn_with_ctx`].
pub struct FnToolHandler<Ctx, F> {
    pub(crate) definition: ToolDefinition,
    pub(crate) handler: F,
    pub(crate) _ctx: PhantomData<fn(&Ctx)>,
}

impl<Ctx, F> std::fmt::Debug for FnToolHandler<Ctx, F> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FnToolHandler")
            .field("name", &self.definition.name)
            .finish_non_exhaustive()
    }
}

impl<Ctx, F, Fut, O> ToolHandler<Ctx> for FnToolHandler<Ctx, F>
where
    Ctx: Send + Sync + 'static,
    F: for<'c> Fn(Value, &'c Ctx) -> Fut + Send + Sync,
    Fut: Future<Output = Result<O, ToolError>> + Send + 'static,
    O: Into<ToolOutput> + Send + 'static,
{
    fn definition(&self) -> ToolDefinition {
        self.definition.clone()
    }

    fn execute<'a>(
        &'a self,
        input: Value,
        ctx: &'a Ctx,
    ) -> Pin<Box<dyn Future<Output = Result<ToolOutput, ToolError>> + Send + 'a>> {
        let fut = (self.handler)(input, ctx);
        Box::pin(async move { fut.await.map(Into::into) })
    }
}

/// A tool handler without context, created by [`super::tool_fn`].
pub struct NoCtxToolHandler<F> {
    pub(crate) definition: ToolDefinition,
    pub(crate) handler: F,
}

impl<F> std::fmt::Debug for NoCtxToolHandler<F> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("NoCtxToolHandler")
            .field("name", &self.definition.name)
            .finish_non_exhaustive()
    }
}

impl<F, Fut, O> ToolHandler<()> for NoCtxToolHandler<F>
where
    F: Fn(Value) -> Fut + Send + Sync,
    Fut: Future<Output = Result<O, ToolError>> + Send + 'static,
    O: Into<ToolOutput> + Send + 'static,
{
    fn definition(&self) -> ToolDefinition {
        self.definition.clone()
    }

    fn execute<'a>(
        &'a self,
        input: Value,
        _ctx: &'a (),
    ) -> Pin<Box<dyn Future<Output = Result<ToolOutput, ToolError>> + Send + 'a>> {
        let fut = (self.handler)(input);
        Box::pin(async move { fut.await.map(Into::into) })
    }
}
