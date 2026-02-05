# Quick Start

Get up and running with llm-stack in 5 minutes.

## Installation

Add llm-stack to your `Cargo.toml`:

```toml
[dependencies]
llm-stack = "0.1"
llm-stack-anthropic = "0.1"  # or llm-stack-openai, llm-stack-ollama
tokio = { version = "1", features = ["rt-multi-thread", "macros"] }
```

## Your First Request

Set your API key and make a simple request:

```bash
export ANTHROPIC_API_KEY="sk-ant-..."
```

```rust
use llm_stack::{ChatMessage, ChatParams, Provider};
use llm_stack_anthropic::AnthropicProvider;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create a provider from environment variables
    let provider = AnthropicProvider::from_env()?;

    // Build request parameters
    let params = ChatParams {
        messages: vec![
            ChatMessage::system("You are a helpful assistant."),
            ChatMessage::user("What is Rust's ownership model?"),
        ],
        max_tokens: Some(1024),
        ..Default::default()
    };

    // Make the request
    let response = provider.generate(&params).await?;

    // Print the response
    println!("{}", response.text().unwrap_or_default());
    println!("Tokens: {} in, {} out",
        response.usage.input_tokens,
        response.usage.output_tokens);

    Ok(())
}
```

---

## Core Concepts

### Messages and Roles

A conversation is a list of `ChatMessage`s. Each message has a role and one or more content blocks.

```rust
use llm_stack::{ChatMessage, ChatRole, ContentBlock};

// Simple text messages (most common)
let system = ChatMessage::system("You are a helpful assistant.");
let user = ChatMessage::user("Explain async/await in Rust.");
let assistant = ChatMessage::assistant("Async/await is...");

// Multi-block messages (text + images)
let multimodal = ChatMessage {
    role: ChatRole::User,
    content: vec![
        ContentBlock::Text("Describe this image.".into()),
        ContentBlock::Image {
            media_type: "image/png".into(),
            data: llm_stack::ImageSource::Base64("...".into()),
        },
    ],
};
```

### Request Parameters

All request configuration goes into `ChatParams`. Only `messages` is required.

```rust
use llm_stack::{ChatParams, ChatMessage};

let params = ChatParams {
    messages: vec![ChatMessage::user("Hello!")],

    // Optional parameters
    max_tokens: Some(1024),        // Limit response length
    temperature: Some(0.7),        // Creativity (0.0 = deterministic)
    system: Some("Be concise.".into()), // System prompt (alternative to system message)
    stop_sequences: Some(vec!["END".into()]), // Stop on these strings

    ..Default::default()
};
```

### Responses

`ChatResponse` contains the model's output, token usage, and stop reason.

```rust
use llm_stack::{ChatResponse, ContentBlock, StopReason};

fn handle_response(response: &ChatResponse) {
    // Get text content (most common)
    if let Some(text) = response.text() {
        println!("Response: {text}");
    }

    // Or iterate all content blocks
    for block in &response.content {
        match block {
            ContentBlock::Text(text) => println!("Text: {text}"),
            ContentBlock::ToolCall(call) => println!("Tool: {}", call.name),
            _ => {}
        }
    }

    // Check token usage
    println!("Tokens: {} in, {} out",
        response.usage.input_tokens,
        response.usage.output_tokens);

    // Check why the model stopped
    match response.stop_reason {
        StopReason::EndTurn => println!("Model finished normally"),
        StopReason::ToolUse => println!("Model wants to call tools"),
        StopReason::MaxTokens => println!("Hit token limit"),
        StopReason::StopSequence => println!("Hit stop sequence"),
        _ => {}
    }
}
```

---

## Streaming

For real-time output, use `stream()` instead of `generate()`:

```rust
use futures::StreamExt;
use llm_stack::{ChatParams, ChatMessage, Provider, StreamEvent};
use llm_stack_anthropic::AnthropicProvider;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let provider = AnthropicProvider::from_env()?;

    let params = ChatParams {
        messages: vec![ChatMessage::user("Write a haiku about Rust.")],
        ..Default::default()
    };

    let mut stream = provider.stream(&params).await?;

    while let Some(event) = stream.next().await {
        match event? {
            StreamEvent::TextDelta(text) => print!("{text}"),
            StreamEvent::Done { stop_reason } => {
                println!("\n[Done: {stop_reason:?}]");
                break;
            }
            StreamEvent::Usage(usage) => {
                println!("\n[Tokens: {}]", usage.input_tokens + usage.output_tokens);
            }
            _ => {}
        }
    }

    Ok(())
}
```

---

## Error Handling

All errors are `LlmError`. Check `is_retryable()` to know if you should retry:

```rust
use llm_stack::LlmError;

fn handle_error(err: &LlmError) {
    if err.is_retryable() {
        println!("Retryable error, backing off: {err}");
        // Implement exponential backoff here
    } else {
        println!("Fatal error: {err}");
    }

    // Or match specific variants
    match err {
        LlmError::Http { status, message, .. } => {
            println!("HTTP {status:?}: {message}");
        }
        LlmError::Auth(msg) => {
            println!("Authentication failed: {msg}");
        }
        LlmError::Timeout { elapsed_ms } => {
            println!("Timed out after {elapsed_ms}ms");
        }
        _ => println!("Error: {err}"),
    }
}
```

---

## Testing with MockProvider

Test LLM-dependent code without network calls:

```rust
use llm_stack::test_helpers::mock_for;
use llm_stack::{ChatParams, ChatMessage, ChatResponse, ContentBlock, Provider};

#[tokio::test]
async fn test_my_feature() {
    // Create a mock provider
    let mock = mock_for("test", "test-model");

    // Queue expected responses
    mock.queue_response(ChatResponse {
        content: vec![ContentBlock::Text("42".into())],
        ..Default::default()
    });

    // Your code under test
    let params = ChatParams {
        messages: vec![ChatMessage::user("What is the answer?")],
        ..Default::default()
    };

    let response = mock.generate(&params).await.unwrap();
    assert_eq!(response.text(), Some("42"));
}
```

---

## Switching Providers

The same code works with any provider:

```rust
use llm_stack::{ChatParams, ChatMessage, Provider};

// Works with any provider!
async fn ask_question(provider: &impl Provider) -> Result<String, llm_stack::LlmError> {
    let params = ChatParams {
        messages: vec![ChatMessage::user("What is 2+2?")],
        ..Default::default()
    };

    let response = provider.generate(&params).await?;
    Ok(response.text().unwrap_or_default().to_string())
}

// Use with different providers
use llm_stack_anthropic::AnthropicProvider;
use llm_stack_openai::OpenAiProvider;
use llm_stack_ollama::OllamaProvider;

let claude = AnthropicProvider::from_env()?;
let gpt = OpenAiProvider::from_env()?;
let local = OllamaProvider::new("http://localhost:11434", "llama3");

let answer = ask_question(&claude).await?;
```

---

## What's Next?

Now that you have the basics:

| Guide | Learn about |
|-------|-------------|
| [Tools](tools.md) | Build agentic applications with tool execution |
| [Structured Output](structured-output.md) | Get typed Rust structs from LLMs |
| [Interceptors](interceptors.md) | Add retry, logging, approval gates |
| [Providers](providers.md) | Configure Anthropic, OpenAI, Ollama |
| [Context Window](context-window.md) | Manage token budgets |
| [Architecture](ARCH.md) | Understand the design |
