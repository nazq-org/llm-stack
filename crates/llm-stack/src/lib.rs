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
//!  ┌─────────────────────┐ ┌───────────────────┐ ┌───────────────────┐
//!  │ llm-stack-anthropic │ │  llm-stack-openai  │ │  llm-stack-ollama │
//!  └──────────┬──────────┘ └─────────┬─────────┘ └─────────┬─────────┘
//!             │                      │                     │
//!             └───────────┬──────────┴──────────┬──────────┘
//!                         │                     │
//!                         ▼                     ▼
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

// ── Core re-exports ────────────────────────────────────────────────
//
// Only the types that appear in nearly every program are re-exported
// at the crate root. Everything else lives in its submodule:
//
//   llm_stack::tool::*        — tool execution, loop handles, events
//   llm_stack::provider::*    — capabilities, metadata, retry config
//   llm_stack::chat::*        — StopReason, ImageSource, ChatRole
//   llm_stack::stream::*      — ChatStream, StreamEvent
//   llm_stack::usage::*       — Cost, ModelPricing, UsageTracker
//   llm_stack::context::*     — ContextWindow, token estimation
//   llm_stack::registry::*    — ProviderRegistry, ProviderFactory
//   llm_stack::mcp::*         — McpService, McpRegistryExt
//   llm_stack::structured::*  — generate_object, stream_object_async
//   llm_stack::mock::*        — MockProvider (test-utils feature)

pub use chat::{ChatMessage, ChatResponse, ContentBlock, ToolCall, ToolResult};
pub use error::LlmError;
pub use provider::{ChatParams, DynProvider, JsonSchema, Provider, ToolChoice, ToolDefinition};
pub use registry::ProviderRegistry;
pub use stream::{ChatStream, StreamEvent};
pub use tool::{ToolHandler, ToolLoopConfig, ToolRegistry};
pub use usage::Usage;
