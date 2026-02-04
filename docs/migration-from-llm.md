# Migration from the `llm` Crate

This guide helps you migrate from the [`llm`](https://crates.io/crates/llm) crate (by graniet) to `llm-stack`. Both provide unified interfaces for LLM providers, but `llm-stack` has a different API design.

## Key differences

| Aspect | `llm` crate | `llm-stack` |
|--------|------------|----------|
| Provider trait | `LLMProvider` (combines chat + completion + embedding) | `Provider` (chat-focused, single responsibility) |
| Configuration | Builder pattern with `LLMBuilder` | Config structs per provider |
| Messages | `ChatMessage` with `message_type` enum | `ChatMessage` with `content: Vec<ContentBlock>` |
| Streaming | Returns `Stream<Item = Result<String, _>>` | Returns `ChatStream` of `StreamEvent` |
| Tools | `Tool` + `FunctionTool` structs | `ToolDefinition` + `ToolHandler` trait |
| Errors | `LLMError` enum | `LlmError` enum (more variants) |
| Middleware | `resilient_llm` / `validated_llm` wrappers | Unified `Interceptor` system |

## Provider setup

### Before (`llm`)

```rust
use llm::builder::{LLMBackend, LLMBuilder};

let provider = LLMBuilder::new()
    .backend(LLMBackend::Anthropic)
    .api_key(std::env::var("ANTHROPIC_API_KEY")?)
    .model("claude-sonnet-4-20250514")
    .temperature(0.7)
    .max_tokens(1024)
    .build()?;
```

### After (`llm-stack`)

```rust
use llm_anthropic::{AnthropicProvider, AnthropicConfig};

let provider = AnthropicProvider::new(AnthropicConfig {
    api_key: std::env::var("ANTHROPIC_API_KEY")?,
    model: "claude-sonnet-4-20250514".into(),
    ..Default::default()
});

// Temperature and max_tokens go in ChatParams, not provider config
```

## Messages

### Before (`llm`)

```rust
use llm::chat::{ChatMessage, ChatRole, MessageType};

let messages = vec![
    ChatMessage {
        role: ChatRole::User,
        message_type: MessageType::Text,
        content: "Hello!".to_string(),
    },
];

// Image message
let image_msg = ChatMessage {
    role: ChatRole::User,
    message_type: MessageType::Image((ImageMime::PNG, image_bytes)),
    content: "What's in this image?".to_string(),
};
```

### After (`llm-stack`)

```rust
use llm_stack_core::{ChatMessage, ChatRole, ContentBlock, ImageSource};

let messages = vec![
    ChatMessage::user("Hello!"),
];

// Image message (multi-block)
let image_msg = ChatMessage {
    role: ChatRole::User,
    content: vec![
        ContentBlock::Text("What's in this image?".into()),
        ContentBlock::Image {
            media_type: "image/png".into(),
            data: ImageSource::Base64(base64_data),
        },
    ],
};
```

### Message helpers

| `llm` | `llm-stack` |
|-------|----------|
| `ChatMessage { role: ChatRole::User, .. }` | `ChatMessage::user("text")` |
| `ChatMessage { role: ChatRole::Assistant, .. }` | `ChatMessage::assistant("text")` |
| N/A (no system role) | `ChatMessage::system("text")` |

## Making requests

### Before (`llm`)

```rust
use llm::chat::ChatProvider;

// Basic chat
let response = provider.chat(&messages).await?;
println!("{}", response.text().unwrap_or_default());

// With tools
let response = provider.chat_with_tools(&messages, Some(&tools)).await?;

// Streaming
let stream = provider.chat_stream(&messages).await?;
while let Some(chunk) = stream.next().await {
    print!("{}", chunk?);
}
```

### After (`llm-stack`)

```rust
use llm_stack_core::{Provider, ChatParams, StreamEvent};
use futures::StreamExt;

// Basic chat
let params = ChatParams {
    messages,
    max_tokens: Some(1024),
    temperature: Some(0.7),
    ..Default::default()
};
let response = provider.generate(&params).await?;
println!("{}", response.text().unwrap_or_default());

// With tools
let params = ChatParams {
    messages,
    tools: vec![tool_definition],
    ..Default::default()
};
let response = provider.generate(&params).await?;

// Streaming
let mut stream = provider.stream(&params).await?;
while let Some(event) = stream.next().await {
    match event? {
        StreamEvent::TextDelta(text) => print!("{text}"),
        StreamEvent::Done { .. } => break,
        _ => {}
    }
}
```

## Tool definitions

### Before (`llm`)

```rust
use llm::chat::{Tool, FunctionTool, ParametersSchema, ParameterProperty};
use std::collections::HashMap;

let tool = Tool {
    tool_type: "function".to_string(),
    function: FunctionTool {
        name: "get_weather".to_string(),
        description: "Get weather for a location".to_string(),
        parameters: serde_json::json!({
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "City name"
                }
            },
            "required": ["location"]
        }),
    },
};
```

### After (`llm-stack`)

```rust
use llm_stack_core::ToolDefinition;
use serde_json::json;

let tool = ToolDefinition::new(
    "get_weather",
    "Get weather for a location",
    json!({
        "type": "object",
        "properties": {
            "location": {
                "type": "string",
                "description": "City name"
            }
        },
        "required": ["location"]
    }),
);
```

## Tool execution

### Before (`llm`)

Manual tool execution loop:

```rust
// Check for tool calls in response
if let Some(tool_calls) = response.tool_calls() {
    for tc in tool_calls {
        let args: serde_json::Value = serde_json::from_str(&tc.function.arguments)?;
        let result = execute_tool(&tc.function.name, args)?;
        // Manually add tool result to messages and call again
    }
}
```

### After (`llm-stack`)

Automatic tool loop with `ToolRegistry` and `tool_fn`:

```rust
use llm_stack_core::{ToolRegistry, tool_fn, tool_loop, ToolLoopConfig, ToolDefinition, JsonSchema};
use serde_json::json;

let weather_tool = tool_fn(
    ToolDefinition {
        name: "get_weather".into(),
        description: "Get weather for a location".into(),
        parameters: JsonSchema::new(json!({
            "type": "object",
            "properties": { "location": { "type": "string" } },
            "required": ["location"]
        })),
        retry: None,
    },
    |args| async move {
        let location = args["location"].as_str().unwrap_or("unknown");
        Ok(format!("Sunny, 72°F in {location}"))
    },
);

let mut registry: ToolRegistry<()> = ToolRegistry::new();
registry.register(weather_tool);

let result = tool_loop(provider, &registry, params, ToolLoopConfig::default(), &()).await?;
println!("{}", result.response.text().unwrap_or_default());
```

## Streaming responses

### Before (`llm`)

```rust
use llm::chat::StreamChunk;

let stream = provider.chat_stream_with_tools(&messages, Some(&tools)).await?;
while let Some(chunk) = stream.next().await {
    match chunk? {
        StreamChunk::Text(text) => print!("{text}"),
        StreamChunk::ToolUseStart { id, name, .. } => println!("Calling {name}..."),
        StreamChunk::ToolUseComplete { tool_call, .. } => { /* handle */ }
        StreamChunk::Done { stop_reason } => break,
        _ => {}
    }
}
```

### After (`llm-stack`)

```rust
use llm_stack_core::StreamEvent;

let mut stream = provider.stream(&params).await?;
while let Some(event) = stream.next().await {
    match event? {
        StreamEvent::TextDelta(text) => print!("{text}"),
        StreamEvent::ToolCallStart { name, .. } => println!("Calling {name}..."),
        StreamEvent::ToolCallComplete(tool_call) => { /* handle */ }
        StreamEvent::Done { stop_reason } => break,
        _ => {}
    }
}
```

## Error handling

### Before (`llm`)

```rust
use llm::error::LLMError;

match result {
    Err(LLMError::HttpError(e)) => { /* network error */ }
    Err(LLMError::AuthError(e)) => { /* bad API key */ }
    Err(LLMError::ProviderError(e)) => { /* provider rejected request */ }
    Err(LLMError::RetryExceeded { attempts, last_error }) => { /* retries exhausted */ }
    _ => {}
}
```

### After (`llm-stack`)

```rust
use llm_stack_core::LlmError;

match result {
    Err(LlmError::Http { status, message, retryable }) => {
        if retryable { /* can retry */ }
    }
    Err(LlmError::Authentication(e)) => { /* bad API key */ }
    Err(LlmError::Provider { code, message, retryable }) => { /* provider error */ }
    Err(LlmError::RetryExhausted { attempts, last_error }) => { /* retries exhausted */ }
    Err(LlmError::Timeout { elapsed_ms }) => { /* request timed out */ }
    _ => {}
}
```

Key addition: `retryable` flag tells you if an error is transient.

## Resilience / Retry

### Before (`llm`)

```rust
use llm::resilient_llm::ResilientLLM;

let resilient = ResilientLLM::new(
    provider,
    3,     // max retries
    1000,  // initial delay ms
    2.0,   // backoff multiplier
);
```

### After (`llm-stack`)

```rust
use llm_stack_core::intercept::{InterceptorStack, Retry, LlmCall};
use std::time::Duration;

// Using the interceptor system for LLM calls
let interceptors = InterceptorStack::<LlmCall>::new()
    .with(Retry {
        max_attempts: 3,
        initial_delay: Duration::from_millis(1000),
        multiplier: 2.0,
        ..Default::default()
    });

// Or for tool execution with ToolRegistry
use llm_stack_core::ToolRegistry;
use llm_stack_core::intercept::{InterceptorStack, Retry, ToolExec};

let registry: ToolRegistry<()> = ToolRegistry::new()
    .with_interceptors(
        InterceptorStack::<ToolExec<()>>::new()
            .with(Retry::default())
    );
```

## Usage / Token tracking

### Before (`llm`)

```rust
if let Some(usage) = response.usage() {
    println!("Prompt: {} tokens", usage.prompt_tokens);
    println!("Completion: {} tokens", usage.completion_tokens);
    println!("Total: {} tokens", usage.total_tokens);
}
```

### After (`llm-stack`)

```rust
let usage = response.usage;
println!("Input: {} tokens", usage.input_tokens);
println!("Output: {} tokens", usage.output_tokens);
// Total is computed: usage.input_tokens + usage.output_tokens

// Extended tracking
if let Some(reasoning) = usage.reasoning_tokens {
    println!("Reasoning: {} tokens", reasoning);
}
if let Some(cache_read) = usage.cache_read_tokens {
    println!("Cache read: {} tokens", cache_read);
}
```

## Structured output

### Before (`llm`)

```rust
use llm::chat::StructuredOutputFormat;

let format = StructuredOutputFormat {
    name: "Person".to_string(),
    description: Some("A person".to_string()),
    schema: Some(serde_json::json!({
        "type": "object",
        "properties": {
            "name": { "type": "string" },
            "age": { "type": "integer" }
        }
    })),
    strict: Some(true),
};

// Pass to provider somehow (provider-specific)
```

### After (`llm-stack`)

```rust
use llm_stack_core::structured::{generate_object, GenerateObjectConfig};
use serde::Deserialize;
use schemars::JsonSchema;

#[derive(Deserialize, JsonSchema)]
struct Person {
    name: String,
    age: u32,
}

let result = generate_object::<Person>(
    provider,
    params,
    GenerateObjectConfig::default(),
).await?;

let person: Person = result.value;
println!("Name: {}, Age: {}", person.name, person.age);
```

## Memory / Context management

### Before (`llm`)

```rust
use llm::memory::{SlidingWindowMemory, TrimStrategy};

let memory = SlidingWindowMemory::new(10, TrimStrategy::OldestFirst);
// Use with ChatWithMemory wrapper
```

### After (`llm-stack`)

```rust
use llm_stack_core::context::ContextWindow;

let mut window = ContextWindow::new(128_000, 4_000);

// Add messages with token counts
window.push(ChatMessage::user("Hello"), 5);
window.push(ChatMessage::assistant("Hi there!"), 8);

// Protect recent messages
window.protect_recent(2);

// Compact when needed
if window.needs_compaction(0.8) {
    let old = window.compact();
    // Summarize old messages...
}

// Get messages for request
let params = ChatParams {
    messages: window.messages_owned(),
    ..Default::default()
};
```

## Quick reference

| `llm` crate | `llm-stack` equivalent |
|-------------|---------------------|
| `LLMProvider` | `Provider` |
| `ChatProvider::chat()` | `Provider::generate()` |
| `ChatProvider::chat_stream()` | `Provider::stream()` |
| `ChatMessage` | `ChatMessage` (different structure) |
| `ChatRole::User/Assistant` | `ChatRole::User/Assistant/System` |
| `LLMError` | `LlmError` |
| `Tool` | `ToolDefinition` |
| `ToolCall` | `ToolCall` (same concept) |
| `StreamChunk` | `StreamEvent` |
| `ResilientLLM` | `Retry` interceptor |
| `ValidatedLLM` | `generate_object` with schema validation |
| `SlidingWindowMemory` | `ContextWindow` |
| `LLMBackend::OpenAI` | `OpenAiProvider` |
| `LLMBackend::Anthropic` | `AnthropicProvider` |
| `LLMBackend::Ollama` | `OllamaProvider` |

## Feature comparison

| Feature | `llm` | `llm-stack` |
|---------|-------|----------|
| Chat | ✅ | ✅ |
| Streaming | ✅ | ✅ |
| Tools | ✅ | ✅ (with auto-loop) |
| Structured output | ✅ (manual) | ✅ (derive-based) |
| Retry/backoff | ✅ | ✅ (interceptors) |
| Logging | ✅ | ✅ (interceptors) |
| Approval gates | ❌ | ✅ (interceptors) |
| Context window | ✅ | ✅ |
| Embeddings | ✅ | ❌ (not yet) |
| Text completion | ✅ | ❌ (chat only) |
| Speech (TTS/STT) | ✅ | ❌ (not yet) |

## Getting help

- Check the [docs/](.) folder for detailed guides on each feature
- See [quickstart.md](quickstart.md) for basic usage
- File issues at the project repository
