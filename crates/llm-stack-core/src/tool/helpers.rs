//! Helper functions for creating tool handlers.

use std::future::Future;
use std::marker::PhantomData;

use serde_json::Value;

use super::{FnToolHandler, NoCtxToolHandler, ToolError, ToolOutput};
use crate::provider::ToolDefinition;

/// Creates a [`ToolHandler<()>`](super::ToolHandler) from a closure (no context).
///
/// The closure receives the tool's JSON arguments and returns a
/// `Result<impl Into<ToolOutput>, ToolError>`. Returning `Result<String, ToolError>`
/// also works via the `From<String>` impl on `ToolOutput`.
///
/// For tools that need shared context, use [`tool_fn_with_ctx`] instead.
///
/// # Example
///
/// ```rust
/// use llm_stack_core::tool::tool_fn;
/// use llm_stack_core::{JsonSchema, ToolDefinition};
/// use serde_json::{json, Value};
///
/// let handler = tool_fn(
///     ToolDefinition {
///         name: "add".into(),
///         description: "Add two numbers".into(),
///         parameters: JsonSchema::new(json!({
///             "type": "object",
///             "properties": {
///                 "a": { "type": "number" },
///                 "b": { "type": "number" }
///             },
///             "required": ["a", "b"]
///         })),
///         retry: None,
///     },
///     |input: Value| async move {
///         let a = input["a"].as_f64().unwrap_or(0.0);
///         let b = input["b"].as_f64().unwrap_or(0.0);
///         Ok(format!("{}", a + b))
///     },
/// );
/// ```
pub fn tool_fn<F, Fut, O>(definition: ToolDefinition, handler: F) -> NoCtxToolHandler<F>
where
    F: Fn(Value) -> Fut + Send + Sync + 'static,
    Fut: Future<Output = Result<O, ToolError>> + Send + 'static,
    O: Into<ToolOutput> + Send + 'static,
{
    NoCtxToolHandler {
        definition,
        handler,
    }
}

/// Creates a [`ToolHandler<Ctx>`](super::ToolHandler) from a closure that receives context.
///
/// The closure receives the tool's JSON arguments and a reference to the
/// context, and returns a `Result<impl Into<ToolOutput>, ToolError>`.
///
/// # Example
///
/// ```rust
/// use llm_stack_core::tool::{tool_fn_with_ctx, ToolOutput};
/// use llm_stack_core::{JsonSchema, ToolDefinition};
/// use serde_json::{json, Value};
///
/// struct AppContext {
///     db_url: String,
/// }
///
/// let handler = tool_fn_with_ctx(
///     ToolDefinition {
///         name: "lookup_user".into(),
///         description: "Look up a user by ID".into(),
///         parameters: JsonSchema::new(json!({
///             "type": "object",
///             "properties": {
///                 "user_id": { "type": "string" }
///             },
///             "required": ["user_id"]
///         })),
///         retry: None,
///     },
///     |input: Value, ctx: &AppContext| {
///         let user_id = input["user_id"].as_str().unwrap_or("").to_string();
///         let db_url = ctx.db_url.clone();
///         async move {
///             // Use db_url and user_id in async work...
///             Ok(ToolOutput::new(format!("Found user {} in {}", user_id, db_url)))
///         }
///     },
/// );
/// ```
pub fn tool_fn_with_ctx<Ctx, F, Fut, O>(
    definition: ToolDefinition,
    handler: F,
) -> FnToolHandler<Ctx, F>
where
    Ctx: Send + Sync + 'static,
    F: for<'c> Fn(Value, &'c Ctx) -> Fut + Send + Sync + 'static,
    Fut: Future<Output = Result<O, ToolError>> + Send + 'static,
    O: Into<ToolOutput> + Send + 'static,
{
    FnToolHandler {
        definition,
        handler,
        _ctx: PhantomData,
    }
}
