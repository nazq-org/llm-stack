# llm-stack

<div align="center">

**Production-ready Rust SDK for LLM providers**

[![CI](https://github.com/nazq/llm-stack/actions/workflows/ci.yml/badge.svg)](https://github.com/nazq/llm-stack/actions/workflows/ci.yml)
[![Crates.io](https://img.shields.io/crates/v/llm-stack.svg)](https://crates.io/crates/llm-stack)
[![Documentation](https://docs.rs/llm-stack/badge.svg)](https://docs.rs/llm-stack)
[![Rust Version](https://img.shields.io/badge/rust-1.85+-blue.svg)](https://www.rust-lang.org)
[![License](https://img.shields.io/badge/license-Apache--2.0-green.svg)](LICENSE)

[Quick Start](#quick-start) •
[Features](#features) •
[Documentation](#documentation) •
[Examples](#examples) •
[Contributing](#contributing)

</div>

---

## Overview

llm-stack is a unified Rust interface for building LLM-powered applications. Write against one set of types, swap providers without changing application code.

```rust
use llm_stack::{ChatMessage, ChatParams, Provider};
use llm_stack_anthropic::AnthropicProvider;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let provider = AnthropicProvider::from_env()?;

    let response = provider.generate(&ChatParams {
        messages: vec![ChatMessage::user("What is Rust's ownership model?")],
        max_tokens: Some(1024),
        ..Default::default()
    }).await?;

    println!("{}", response.text().unwrap_or_default());
    Ok(())
}
```

### Why llm-stack?

- **🔌 Provider Agnostic** — Same code works with Anthropic, OpenAI, Ollama, or any custom provider
- **🛠️ Batteries Included** — Tool execution, structured output, streaming, retry logic out of the box
- **🎯 Type Safe** — `generate_object::<T>()` returns `T`, not `serde_json::Value`
- **⚡ Production Ready** — Comprehensive error handling, token tracking, cost calculation
- **🧪 Testable** — `MockProvider` for unit tests, no network mocks needed
- **🦀 Rust Native** — Async/await, strong typing, zero-cost abstractions

---

## Quick Start

Add to your `Cargo.toml`:

```toml
[dependencies]
llm-stack = "0.1"
llm-stack-anthropic = "0.1"  # or llm-stack-openai, llm-stack-ollama
tokio = { version = "1", features = ["rt-multi-thread", "macros"] }
```

Set your API key:

```bash
export ANTHROPIC_API_KEY="sk-ant-..."
# or OPENAI_API_KEY for OpenAI
```

Run:

```rust
use llm_stack::{ChatMessage, ChatParams, Provider};
use llm_stack_anthropic::AnthropicProvider;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let provider = AnthropicProvider::from_env()?;

    let response = provider.generate(&ChatParams {
        messages: vec![ChatMessage::user("Hello!")],
        ..Default::default()
    }).await?;

    println!("{}", response.text().unwrap_or_default());
    println!("Tokens: {} in, {} out",
        response.usage.input_tokens,
        response.usage.output_tokens);
    Ok(())
}
```

---

## Features

### Core Capabilities

| Feature | Description |
|---------|-------------|
| **Unified Provider Trait** | Two methods define a provider: `generate()` + `stream()` |
| **Streaming** | First-class async streaming with `StreamEvent` types |
| **Tool Execution** | Register handlers, validate inputs, execute in agentic loops |
| **Resumable Tool Loop** | Caller-driven iteration with inspect/inject/stop between rounds |
| **Structured Output** | `generate_object::<T>()` with JSON Schema validation |
| **Interceptors** | Composable retry, timeout, logging, approval gates |
| **Usage Tracking** | Token counts, cost calculation in microdollars |
| **Context Management** | Token budget tracking, message truncation |

### Provider Support

| Provider | Crate | Models |
|----------|-------|--------|
| **Anthropic** | `llm-stack-anthropic` | Claude 3.5 Sonnet, Claude 3 Opus/Haiku |
| **OpenAI** | `llm-stack-openai` | GPT-4o, GPT-4 Turbo, GPT-3.5 |
| **Ollama** | `llm-stack-ollama` | Llama 3, Mistral, CodeLlama, any local model |

### Tool Execution Engine

Build agentic applications with the tool loop:

```rust
use llm_stack::{
    ChatParams, ChatMessage, ToolRegistry, ToolLoopConfig,
    tool::{tool_fn, tool_loop},
    ToolDefinition, JsonSchema,
};
use serde_json::json;

// Define a tool
let mut registry: ToolRegistry<()> = ToolRegistry::new();
registry.register(tool_fn(
    ToolDefinition {
        name: "get_weather".into(),
        description: "Get current weather for a city".into(),
        parameters: JsonSchema::new(json!({
            "type": "object",
            "properties": {
                "city": { "type": "string" }
            },
            "required": ["city"]
        })),
        ..Default::default()
    },
    |input| async move {
        let city = input["city"].as_str().unwrap_or("unknown");
        Ok(format!("Weather in {city}: 72°F, sunny"))
    },
));

// Run the agentic loop
let result = tool_loop(
    &provider,
    &registry,
    ChatParams {
        messages: vec![ChatMessage::user("What's the weather in Tokyo?")],
        tools: Some(registry.definitions()),
        ..Default::default()
    },
    ToolLoopConfig::default(),
    &(),
).await?;

println!("Final answer: {}", result.response.text().unwrap_or_default());
```

### Resumable Tool Loop

For orchestration patterns that need control between iterations (multi-agent, event injection, context compaction):

```rust
use llm_stack::{ToolLoopHandle, LoopEvent, LoopCommand};

let mut handle = ToolLoopHandle::new(
    &provider, &registry, params, ToolLoopConfig::default(), &(),
);

loop {
    match handle.next_event().await {
        LoopEvent::ToolsExecuted { results, .. } => {
            // Inspect results, inject messages, or stop
            handle.resume(LoopCommand::Continue);
        }
        LoopEvent::Completed { .. } | LoopEvent::Error { .. } => break,
    }
}
```

See [Tool documentation](docs/tools.md#resumable-tool-loop) for the full API.

### Structured Output

Get typed responses with schema validation:

```rust
use llm_stack::structured::generate_object;
use serde::Deserialize;

#[derive(Deserialize, schemars::JsonSchema)]
struct MovieReview {
    title: String,
    rating: u8,
    summary: String,
}

let review: MovieReview = generate_object(
    &provider,
    &ChatParams {
        messages: vec![ChatMessage::user("Review the movie Inception")],
        ..Default::default()
    },
).await?;

println!("{}: {}/10 - {}", review.title, review.rating, review.summary);
```

### Interceptors

Add cross-cutting concerns without modifying provider code:

```rust
use llm_stack::intercept::{InterceptorStack, Retry, Timeout, Logging};
use std::time::Duration;

let registry = ToolRegistry::new()
    .with_interceptors(
        InterceptorStack::new()
            .with(Retry::new(3, Duration::from_millis(100)))
            .with(Timeout::new(Duration::from_secs(30)))
            .with(Logging::new())
    );
```

---

## Documentation

| Guide | Description |
|-------|-------------|
| [Quick Start](docs/quickstart.md) | Get up and running in 5 minutes |
| [Architecture](docs/ARCH.md) | Design principles and module overview |
| [Providers](docs/providers.md) | Provider configuration and selection |
| [Tools](docs/tools.md) | Tool execution and agentic loops |
| [Structured Output](docs/structured-output.md) | Type-safe LLM responses |
| [Interceptors](docs/interceptors.md) | Retry, timeout, logging, approval |
| [Context Window](docs/context-window.md) | Token management and truncation |
| [Migration Guide](docs/migration-from-llm.md) | Coming from the `llm` crate? |

### API Reference

```bash
cargo doc --open
```

Or view on [docs.rs](https://docs.rs/llm-stack).

---

## Examples

### Streaming

```rust
use futures::StreamExt;
use llm_stack::stream::StreamEvent;

let mut stream = provider.stream(&params).await?;

while let Some(event) = stream.next().await {
    match event? {
        StreamEvent::TextDelta(text) => print!("{text}"),
        StreamEvent::Done { stop_reason } => break,
        _ => {}
    }
}
```

### Multi-Provider Setup

```rust
use llm_stack::{ProviderRegistry, ProviderConfig};

let registry = ProviderRegistry::new()
    .register("claude", AnthropicProvider::from_env()?)
    .register("gpt4", OpenAiProvider::from_env()?)
    .register("local", OllamaProvider::new("http://localhost:11434"));

// Select at runtime
let provider = registry.get("claude")?;
```

### Testing with MockProvider

```rust
use llm_stack::test_helpers::mock_for;

#[tokio::test]
async fn test_my_agent() {
    let mock = mock_for("test", "mock-model");
    mock.queue_response(ChatResponse {
        content: vec![ContentBlock::Text("Hello!".into())],
        ..Default::default()
    });

    let response = mock.generate(&params).await.unwrap();
    assert_eq!(response.text(), Some("Hello!"));
}
```

---

## Crate Map

| Crate | Purpose |
|-------|---------|
| [`llm-stack`](crates/llm-stack) | Traits, types, errors, streaming, tools, interceptors |
| [`llm-stack-anthropic`](crates/llm-stack-anthropic) | Anthropic Claude provider |
| [`llm-stack-openai`](crates/llm-stack-openai) | OpenAI GPT provider |
| [`llm-stack-ollama`](crates/llm-stack-ollama) | Ollama local provider |

---

## Development

### Prerequisites

- **Rust 1.85+** (2024 edition)
- **[just](https://github.com/casey/just)** — Command runner

### Commands

```bash
just gate      # Full CI check: fmt + clippy + test + doc
just test      # Run all tests
just clippy    # Lint with warnings as errors
just doc       # Build documentation
just fcheck    # Quick feedback: fmt + check
```

### Running Tests

```bash
# All tests
just test

# Specific test
just test-one test_tool_loop

# With output
just test-verbose
```

---

## Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

### Quick Checklist

- [ ] `just gate` passes (fmt, clippy, tests, docs)
- [ ] New features have tests
- [ ] Public APIs have documentation
- [ ] Commit messages follow [Conventional Commits](https://www.conventionalcommits.org/)

---

## License

Licensed under the Apache License, Version 2.0. See [LICENSE](LICENSE) for details.

---

## Acknowledgments

Built on the shoulders of giants:

- [Tokio](https://tokio.rs/) — Async runtime
- [reqwest](https://github.com/seanmonstar/reqwest) — HTTP client
- [serde](https://serde.rs/) — Serialization
- [jsonschema](https://github.com/Stranger6667/jsonschema-rs) — JSON Schema validation

---

<div align="center">

**[⬆ Back to Top](#llm-stack)**

</div>
