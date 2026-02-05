//! Anthropic Claude provider for the llm-stack SDK.
//!
//! This crate implements [`Provider`](llm_stack_core::Provider) for Anthropic's
//! Messages API, supporting both non-streaming and streaming generation
//! with full tool-calling and extended thinking support.
//!
//! # Quick start
//!
//! ```rust,no_run
//! use llm_stack_anthropic::{AnthropicConfig, AnthropicProvider};
//! use llm_stack_core::{ChatMessage, ChatParams, Provider};
//!
//! # async fn example() -> Result<(), llm_stack_core::LlmError> {
//! let provider = AnthropicProvider::new(AnthropicConfig {
//!     api_key: std::env::var("ANTHROPIC_API_KEY").unwrap(),
//!     ..Default::default()
//! });
//!
//! let params = ChatParams {
//!     messages: vec![ChatMessage::user("Hello!")],
//!     ..Default::default()
//! };
//!
//! let response = provider.generate(&params).await?;
//! println!("{}", response.text().unwrap_or("no text"));
//! # Ok(())
//! # }
//! ```

#![warn(missing_docs)]

mod config;
mod convert;
mod factory;
mod provider;
mod stream;
mod types;

pub use config::AnthropicConfig;
pub use factory::{AnthropicFactory, register_global};
pub use provider::AnthropicProvider;
