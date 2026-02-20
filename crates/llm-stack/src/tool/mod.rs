//! Tool execution engine.
//!
//! This module provides the runtime layer for executing tools that LLMs
//! invoke during generation. It builds on the foundational types from
//! [`chat`](crate::chat) ([`ToolCall`](crate::chat::ToolCall), [`ToolResult`](crate::chat::ToolResult)) and
//! [`provider`](crate::provider) ([`ToolDefinition`](crate::provider::ToolDefinition), [`JsonSchema`](crate::JsonSchema)).
//!
//! # Architecture
//!
//! ```text
//!   ToolHandler        — defines a single tool (schema + execute fn)
//!       │
//!   ToolRegistry       — stores handlers by name, validates & dispatches
//!       │
//!   tool_loop()           — automates generate → execute → feedback cycle
//!   tool_loop_stream()    — unified LoopEvent stream (LLM deltas + loop lifecycle)
//!   ToolLoopHandle        — caller-driven resumable variant (borrowed refs)
//!   OwnedToolLoopHandle   — caller-driven resumable variant (Arc, Send + 'static)
//! ```
//!
//! # Example
//!
//! ```rust,no_run
//! use llm_stack::tool::{ToolRegistry, tool_fn, ToolLoopConfig, tool_loop};
//! use llm_stack::{ChatParams, ChatMessage, JsonSchema, ToolDefinition};
//! use serde_json::{json, Value};
//!
//! # async fn example(provider: &dyn llm_stack::DynProvider) -> Result<(), llm_stack::LlmError> {
//! let mut registry: ToolRegistry<()> = ToolRegistry::new();
//! registry.register(tool_fn(
//!     ToolDefinition {
//!         name: "add".into(),
//!         description: "Add two numbers".into(),
//!         parameters: JsonSchema::new(json!({
//!             "type": "object",
//!             "properties": {
//!                 "a": {"type": "number"},
//!                 "b": {"type": "number"}
//!             },
//!             "required": ["a", "b"]
//!         })),
//!         retry: None,
//!     },
//!     |input: Value| async move {
//!         let a = input["a"].as_f64().unwrap_or(0.0);
//!         let b = input["b"].as_f64().unwrap_or(0.0);
//!         Ok(format!("{}", a + b))
//!     },
//! ));
//!
//! let params = ChatParams {
//!     messages: vec![ChatMessage::user("What is 2 + 3?")],
//!     tools: Some(registry.definitions()),
//!     ..Default::default()
//! };
//!
//! let result = tool_loop(provider, &registry, params, ToolLoopConfig::default(), &()).await?;
//! println!("Final answer: {:?}", result.response.text());
//! # Ok(())
//! # }
//! ```
//!
//! # Using Context
//!
//! Tools often need access to shared state like database connections, user identity,
//! or configuration. Use [`tool_fn_with_ctx`] to create tools that receive context:
//!
//! ```rust,no_run
//! use llm_stack::tool::{tool_fn_with_ctx, ToolRegistry, ToolError, ToolOutput, tool_loop, ToolLoopConfig, LoopContext};
//! use llm_stack::{ToolDefinition, JsonSchema, ChatParams, ChatMessage};
//! use serde_json::{json, Value};
//!
//! #[derive(Clone)]
//! struct AppState {
//!     user_id: String,
//!     api_key: String,
//! }
//!
//! type AppCtx = LoopContext<AppState>;
//!
//! # async fn example(provider: &dyn llm_stack::DynProvider) -> Result<(), llm_stack::LlmError> {
//! let handler = tool_fn_with_ctx(
//!     ToolDefinition {
//!         name: "get_user_data".into(),
//!         description: "Fetch data for the current user".into(),
//!         parameters: JsonSchema::new(json!({"type": "object"})),
//!         retry: None,
//!     },
//!     |_input: Value, ctx: &AppCtx| {
//!         // Clone data from context before the async block
//!         let user_id = ctx.state.user_id.clone();
//!         async move {
//!             Ok(ToolOutput::new(format!("Data for user: {}", user_id)))
//!         }
//!     },
//! );
//!
//! let mut registry: ToolRegistry<AppCtx> = ToolRegistry::new();
//! registry.register(handler);
//!
//! let ctx = LoopContext::new(AppState {
//!     user_id: "user123".into(),
//!     api_key: "secret".into(),
//! });
//!
//! let params = ChatParams {
//!     messages: vec![ChatMessage::user("Get my data")],
//!     tools: Some(registry.definitions()),
//!     ..Default::default()
//! };
//!
//! let result = tool_loop(provider, &registry, params, ToolLoopConfig::default(), &ctx).await?;
//! # Ok(())
//! # }
//! ```
//!
//! **Note on lifetimes**: The closure passed to `tool_fn_with_ctx` uses higher-ranked
//! trait bounds (`for<'c> Fn(Value, &'c Ctx) -> Fut`). This means the future returned
//! by your closure must be `'static` — it cannot borrow from the context reference.
//! Clone any data you need from the context before creating the async block.

mod approval;
pub mod cache;
mod cacher;
mod config;
mod depth;
mod error;
mod execution;
mod extractor;
mod handler;
mod helpers;
pub(crate) mod loop_core;
mod loop_detection;
mod loop_owned;
mod loop_resumable;
mod loop_stream;
mod loop_sync;
mod output;
mod processor;
mod registry;

// Re-export all public types
pub use cacher::{CachedResult, ToolResultCacher};
pub use config::{
    LoopAction, LoopDetectionConfig, LoopEvent, LoopStream, ObservationMaskingConfig,
    StopConditionFn, StopContext, StopDecision, TerminationReason, ToolApproval, ToolApprovalFn,
    ToolLoopConfig, ToolLoopResult,
};
pub use depth::{LoopContext, LoopDepth};
pub use error::ToolError;
pub use extractor::{ExtractedResult, ToolResultExtractor};
pub use handler::{FnToolHandler, NoCtxToolHandler, ToolHandler};
pub use helpers::{tool_fn, tool_fn_with_ctx};
pub use loop_owned::{OwnedToolLoopHandle, OwnedTurnResult, OwnedYielded};
pub use loop_resumable::{Completed, LoopCommand, ToolLoopHandle, TurnError, TurnResult, Yielded};
pub use loop_stream::tool_loop_stream;
pub use loop_sync::tool_loop;
pub use output::ToolOutput;
pub use processor::{ProcessedResult, ToolResultProcessor};
pub use registry::ToolRegistry;

#[cfg(test)]
mod tests;
