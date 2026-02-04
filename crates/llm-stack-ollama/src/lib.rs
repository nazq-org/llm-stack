//! Ollama provider for the `llm_rs` SDK.
//!
//! This crate implements [`Provider`](llm_stack_core::Provider) for Ollama's
//! Chat API, supporting both non-streaming and streaming generation
//! with tool calling.
//!
//! Ollama runs locally and requires no authentication by default.
//!
//! # Quick start
//!
//! ```rust,no_run
//! use llm_stack_ollama::{OllamaConfig, OllamaProvider};
//! use llm_stack_core::{ChatMessage, ChatParams, Provider};
//!
//! # async fn example() -> Result<(), llm_stack_core::LlmError> {
//! let provider = OllamaProvider::new(OllamaConfig::default());
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

pub use config::OllamaConfig;
pub use factory::{OllamaFactory, register_global};
pub use provider::OllamaProvider;
