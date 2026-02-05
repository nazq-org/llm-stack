# Providers

Providers are the bridge between the SDK and LLM APIs. Each provider implements the `Provider` trait, giving you a consistent interface regardless of which service you use.

## Available providers

| Crate | Provider | Models |
|-------|----------|--------|
| `llm-stack-anthropic` | Anthropic Messages API | Claude 3.5, Claude 3 Opus/Sonnet/Haiku |
| `llm-stack-openai` | OpenAI Chat Completions | GPT-4o, GPT-4 Turbo, GPT-3.5 |
| `llm-stack-ollama` | Ollama (local) | Llama 3, Mistral, Qwen, any Ollama model |

## Creating a provider

Each provider has a config struct with sensible defaults.

### Anthropic

```rust
use llm_anthropic::{AnthropicProvider, AnthropicConfig};

let provider = AnthropicProvider::new(AnthropicConfig {
    api_key: std::env::var("ANTHROPIC_API_KEY")?,
    model: "claude-sonnet-4-20250514".into(),
    ..Default::default()
});
```

### OpenAI

```rust
use llm_openai::{OpenAiProvider, OpenAiConfig};

let provider = OpenAiProvider::new(OpenAiConfig {
    api_key: std::env::var("OPENAI_API_KEY")?,
    model: "gpt-4o".into(),
    ..Default::default()
});
```

### Ollama

```rust
use llm_ollama::{OllamaProvider, OllamaConfig};

let provider = OllamaProvider::new(OllamaConfig {
    base_url: "http://localhost:11434".parse()?,
    model: "llama3.2".into(),
    ..Default::default()
});
```

## Using a provider

All providers implement `Provider`, so the usage is identical:

```rust
use llm_stack::{ChatMessage, ChatParams, Provider};

let params = ChatParams {
    messages: vec![
        ChatMessage::system("You are a helpful assistant."),
        ChatMessage::user("Explain async/await in Rust."),
    ],
    max_tokens: Some(1024),
    ..Default::default()
};

// Non-streaming
let response = provider.generate(&params).await?;
println!("{}", response.text().unwrap_or_default());

// Streaming
use futures::StreamExt;

let mut stream = provider.stream(&params).await?;
while let Some(event) = stream.next().await {
    match event? {
        StreamEvent::TextDelta(text) => print!("{text}"),
        StreamEvent::Done { .. } => break,
        _ => {}
    }
}
```

## Provider capabilities

Check what a provider supports before using advanced features:

```rust
use llm_stack::Capability;

let meta = provider.metadata();
println!("Model: {}", meta.model);
println!("Context window: {} tokens", meta.context_window);

if meta.capabilities.contains(&Capability::Tools) {
    println!("Supports tool use");
}
if meta.capabilities.contains(&Capability::Vision) {
    println!("Supports images");
}
if meta.capabilities.contains(&Capability::StructuredOutput) {
    println!("Supports JSON schema output");
}
```

Available capabilities:
- `Tools` — Function calling / tool use
- `Vision` — Image inputs
- `Reasoning` — Extended thinking (Claude)
- `Caching` — Prompt caching
- `StructuredOutput` — JSON schema responses

## Configuration options

### Common options

All providers support:

```rust
// Timeout for API calls
config.timeout = Some(Duration::from_secs(120));
```

### Anthropic-specific

```rust
AnthropicConfig {
    api_key: "...".into(),
    model: "claude-sonnet-4-20250514".into(),
    base_url: None,  // Custom endpoint (AWS Bedrock, etc.)
    timeout: Some(Duration::from_secs(60)),
    max_retries: 0,  // Application handles retries via interceptors
}
```

### OpenAI-specific

```rust
OpenAiConfig {
    api_key: "...".into(),
    model: "gpt-4o".into(),
    base_url: None,  // Custom endpoint (Azure, local proxy)
    organization: None,
    timeout: Some(Duration::from_secs(60)),
}
```

### Ollama-specific

```rust
OllamaConfig {
    base_url: "http://localhost:11434".parse()?,
    model: "llama3.2".into(),
    timeout: Some(Duration::from_secs(300)),  // Local models can be slow
}
```

## Dynamic provider selection

Use `DynProvider` to work with providers as trait objects:

```rust
use llm_stack::DynProvider;

fn get_provider(name: &str) -> Box<dyn DynProvider> {
    match name {
        "anthropic" => Box::new(AnthropicProvider::new(/* ... */)),
        "openai" => Box::new(OpenAiProvider::new(/* ... */)),
        "ollama" => Box::new(OllamaProvider::new(/* ... */)),
        _ => panic!("Unknown provider"),
    }
}

let provider = get_provider("anthropic");
let response = provider.generate_boxed(&params).await?;
```

## Provider registry

For config-driven provider selection, use `ProviderRegistry`. This is ideal when the provider choice comes from configuration files, environment variables, or user input.

### Registration

Provider crates include factories that you register at startup:

```rust
use llm_anthropic::register_global as register_anthropic;
use llm_openai::register_global as register_openai;
use llm_ollama::register_global as register_ollama;

// Call once at application startup
register_anthropic();
register_openai();
register_ollama();
```

### Building from config

```rust
use llm_stack::{ProviderRegistry, ProviderConfig};

let config = ProviderConfig::new("anthropic", "claude-sonnet-4-20250514")
    .api_key(std::env::var("ANTHROPIC_API_KEY")?)
    .timeout(Duration::from_secs(60));

let provider = ProviderRegistry::global().build(&config)?;
let response = provider.generate_boxed(&params).await?;
```

### Configuration struct

`ProviderConfig` has common fields plus an `extra` map for provider-specific options:

```rust
use llm_stack::ProviderConfig;
use std::time::Duration;

let config = ProviderConfig {
    provider: "openai".into(),
    api_key: Some("sk-...".into()),
    model: "gpt-4o".into(),
    base_url: None,  // Use default
    timeout: Some(Duration::from_secs(30)),
    extra: Default::default(),
};

// Or use the builder pattern:
let config = ProviderConfig::new("openai", "gpt-4o")
    .api_key("sk-...")
    .base_url("https://custom.endpoint/v1")
    .timeout(Duration::from_secs(30))
    .extra("organization", "org-123");  // Provider-specific
```

### Provider-specific extra fields

| Provider | Extra Field | Description |
|----------|-------------|-------------|
| `anthropic` | `max_tokens` | Default max tokens (default: 4096) |
| `anthropic` | `api_version` | API version header |
| `openai` | `organization` | OpenAI organization ID |

### Registering third-party providers

Third-party crates can implement `ProviderFactory` to integrate with the registry:

```rust
use llm_stack::{ProviderFactory, ProviderConfig, DynProvider, LlmError};

pub struct MyProviderFactory;

impl ProviderFactory for MyProviderFactory {
    fn name(&self) -> &str {
        "my-provider"
    }

    fn build(&self, config: &ProviderConfig) -> Result<Box<dyn DynProvider>, LlmError> {
        let api_key = config.api_key.clone().ok_or_else(|| {
            LlmError::InvalidRequest("my-provider requires api_key".into())
        })?;

        // Read provider-specific options from config.extra
        let custom_option = config.get_extra_str("custom_option");

        Ok(Box::new(MyProvider::new(api_key, config.model.clone())))
    }
}

// Register at startup
llm_stack::ProviderRegistry::global().register(Box::new(MyProviderFactory));
```

### Loading from TOML/JSON

A typical pattern is loading provider config from a file:

```rust
use serde::Deserialize;
use llm_stack::{ProviderConfig, ProviderRegistry};
use std::collections::HashMap;

#[derive(Deserialize)]
struct AppConfig {
    provider: String,
    api_key: Option<String>,
    model: String,
    #[serde(default)]
    extra: HashMap<String, serde_json::Value>,
}

fn load_provider(app_config: &AppConfig) -> Result<Box<dyn llm_stack::DynProvider>, llm_stack::LlmError> {
    let config = ProviderConfig {
        provider: app_config.provider.clone(),
        api_key: app_config.api_key.clone(),
        model: app_config.model.clone(),
        base_url: None,
        timeout: None,
        extra: app_config.extra.clone(),
    };

    ProviderRegistry::global().build(&config)
}
```

## Error handling

Provider errors are normalized to `LlmError`:

```rust
use llm_stack::LlmError;

match provider.generate(&params).await {
    Ok(response) => { /* ... */ }
    Err(LlmError::Http { status, message, retryable }) => {
        if retryable {
            // Rate limit, server error — retry with backoff
        } else {
            // Client error — fix the request
        }
    }
    Err(LlmError::Timeout { elapsed_ms }) => {
        // Request timed out
    }
    Err(LlmError::InvalidRequest(msg)) => {
        // Bad parameters
    }
    Err(e) => {
        // Other errors
    }
}
```

The `retryable` flag indicates whether the error is transient. Use the interceptor system to handle retries automatically (see [interceptors.md](interceptors.md)).

## Custom providers

Implement `Provider` to add support for other LLM services. With Rust 2024's async-fn-in-traits, no macro is needed:

```rust
use llm_stack::{Provider, ChatParams, ChatResponse, ChatStream, ProviderMetadata, LlmError, Capability};
use std::collections::HashSet;

struct MyProvider { /* ... */ }

impl Provider for MyProvider {
    async fn generate(&self, params: &ChatParams) -> Result<ChatResponse, LlmError> {
        // Call your API here
        todo!()
    }

    async fn stream(&self, params: &ChatParams) -> Result<ChatStream, LlmError> {
        // Return a stream of events
        todo!()
    }

    fn metadata(&self) -> ProviderMetadata {
        ProviderMetadata {
            name: "my-provider".into(),
            model: "my-model".into(),
            context_window: 128_000,
            capabilities: HashSet::from([Capability::Tools]),
        }
    }
}
```

Your custom provider automatically works with the interceptor system and tool execution engine.
