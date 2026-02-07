//! # llm-stack
//!
//! Provider-agnostic types and traits for interacting with large language models.
//!
//! This crate defines the shared vocabulary that every LLM provider implementation
//! speaks: messages, responses, tool calls, streaming events, usage tracking, and
//! errors. It intentionally contains **zero** provider-specific code — concrete
//! providers live in sibling crates and implement [`Provider`] (or its
//! object-safe counterpart [`DynProvider`]).
//!
//! # Provider Crates
//!
//! Official provider implementations:
//!
//! | Crate | Provider | Features |
//! |-------|----------|----------|
//! | [`llm-stack-anthropic`](https://docs.rs/llm-stack-anthropic) | Claude (Anthropic) | Streaming, tools, vision, caching |
//! | [`llm-stack-openai`](https://docs.rs/llm-stack-openai) | GPT (`OpenAI`) | Streaming, tools, structured output |
//! | [`llm-stack-ollama`](https://docs.rs/llm-stack-ollama) | Ollama (local) | Streaming, tools |
//!
//! # Architecture
//!
//! ```text
//!  ┌───────────────────┐ ┌───────────────────┐ ┌───────────────────┐
//!  │ llm-stack-anthropic│ │  llm-stack-openai │ │  llm-stack-ollama │
//!  └─────────┬─────────┘ └─────────┬─────────┘ └─────────┬─────────┘
//!            │                     │                     │
//!            └──────────┬──────────┴──────────┬──────────┘
//!                       │                     │
//!                       ▼                     ▼
//!              ┌─────────────────────────────────────┐
//!              │             llm-stack               │  ← you are here
//!              │  (Provider trait, ChatParams, etc.) │
//!              └─────────────────────────────────────┘
//! ```
//!
//! # Quick start
//!
//! ```rust,no_run
//! use llm_stack::{ChatMessage, ChatParams, Provider};
//!
//! # async fn example(provider: impl Provider) -> Result<(), llm_stack::LlmError> {
//! let params = ChatParams {
//!     messages: vec![ChatMessage::user("Explain ownership in Rust")],
//!     max_tokens: Some(1024),
//!     ..Default::default()
//! };
//!
//! let response = provider.generate(&params).await?;
//! # Ok(())
//! # }
//! ```
//!
//! # Modules
//!
//! | Module | Purpose |
//! |--------|---------|
//! | [`chat`] | Messages, content blocks, tool calls, and responses |
//! | [`context`] | Token-budgeted conversation history management |
//! | [`error`] | Unified [`LlmError`] across all providers |
//! | [`intercept`] | Unified interceptor system for LLM calls and tool executions |
//! | [`provider`] | The [`Provider`] trait and request parameters |
//! | [`stream`] | Server-sent event types and the [`ChatStream`] alias |
//! | [`structured`] | Typed LLM responses with schema validation (feature-gated) |
//! | [`tool`] | Tool execution engine with registry and approval hooks |
//! | [`registry`] | Dynamic provider instantiation from configuration |
//! | [`usage`] | Token counts and cost tracking |

#![warn(missing_docs)]

pub mod chat;
pub mod context;
pub mod error;
pub mod intercept;
pub mod provider;
pub mod registry;
pub mod stream;
pub mod structured;
pub mod tool;
pub mod usage;

pub mod mcp;

#[cfg(any(test, feature = "test-utils"))]
pub mod mock;

#[cfg(any(test, feature = "test-utils"))]
pub mod test_helpers;

pub use chat::{
    ChatMessage, ChatResponse, ChatRole, ContentBlock, ImageSource, StopReason, ToolCall,
    ToolResult,
};
pub use error::LlmError;
pub use provider::{
    Capability, ChatParams, DynProvider, JsonSchema, Provider, ProviderMetadata, RetryPredicate,
    ToolChoice, ToolDefinition, ToolRetryConfig,
};
pub use stream::{ChatStream, StreamEvent};
pub use tool::{
    Completed, FnToolHandler, LoopAction, LoopCommand, LoopContext, LoopDepth, LoopDetectionConfig,
    LoopEvent, LoopStream, NoCtxToolHandler, OwnedToolLoopHandle, OwnedTurnResult, OwnedYielded,
    StopConditionFn, StopContext, StopDecision, TerminationReason, ToolApproval, ToolError,
    ToolHandler, ToolLoopConfig, ToolLoopHandle, ToolLoopResult, ToolOutput, ToolRegistry,
    TurnError, TurnResult, Yielded, tool_fn, tool_fn_with_ctx,
};
pub use usage::{Cost, ModelPricing, Usage, UsageTracker};

pub use context::{ContextWindow, estimate_message_tokens, estimate_tokens};
pub use registry::{ProviderConfig, ProviderFactory, ProviderRegistry};

pub use mcp::{McpError, McpRegistryExt, McpService};

#[cfg(feature = "schema")]
pub use structured::{
    GenerateObjectConfig, GenerateObjectResult, PartialObject, collect_stream_object,
    generate_object, stream_object_async,
};

#[cfg(any(test, feature = "test-utils"))]
pub use mock::{MockError, MockProvider};
