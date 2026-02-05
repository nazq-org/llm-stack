# Structured Output

Get typed, validated responses from LLMs. Instead of parsing free-form text, define a Rust struct and let the SDK handle JSON generation, validation, and deserialization.

## Quick example

```rust
use llm_stack::{ChatMessage, ChatParams};
use llm_stack::structured::{generate_object, GenerateObjectConfig};
use serde::Deserialize;
use schemars::JsonSchema;

#[derive(Debug, Deserialize, JsonSchema)]
struct Person {
    name: String,
    age: u32,
    interests: Vec<String>,
}

let params = ChatParams {
    messages: vec![ChatMessage::user("Generate a fictional person")],
    ..Default::default()
};

let result = generate_object::<Person>(provider, params, GenerateObjectConfig::default()).await?;

println!("Name: {}", result.value.name);
println!("Age: {}", result.value.age);
println!("Interests: {:?}", result.value.interests);
```

## How it works

1. **Schema derivation** — The SDK derives a JSON Schema from your Rust type using `schemars`
2. **Provider hint** — If the provider supports structured output, the schema is sent as a constraint
3. **Fallback prompt** — Otherwise, the schema is injected into the system prompt
4. **Validation** — The response is validated against the schema
5. **Retry** — If validation fails, the SDK retries with error feedback
6. **Deserialization** — The validated JSON is deserialized into your type

## Defining schemas

Use `serde` and `schemars` derives:

```rust
use serde::Deserialize;
use schemars::JsonSchema;

#[derive(Deserialize, JsonSchema)]
struct MovieReview {
    title: String,
    rating: f32,
    summary: String,
    pros: Vec<String>,
    cons: Vec<String>,
}
```

### Field constraints

Add constraints with `schemars` attributes:

```rust
#[derive(Deserialize, JsonSchema)]
struct Product {
    #[schemars(length(min = 1, max = 100))]
    name: String,

    #[schemars(range(min = 0.0))]
    price: f64,

    #[schemars(length(min = 1))]
    tags: Vec<String>,
}
```

### Enums

Enums work naturally:

```rust
#[derive(Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
enum Sentiment {
    Positive,
    Neutral,
    Negative,
}

#[derive(Deserialize, JsonSchema)]
struct Analysis {
    sentiment: Sentiment,
    confidence: f32,
}
```

### Optional fields

Use `Option` for fields that might not be present:

```rust
#[derive(Deserialize, JsonSchema)]
struct Contact {
    name: String,
    email: String,
    phone: Option<String>,  // Model can omit this
}
```

### Nested types

Compose complex structures:

```rust
#[derive(Deserialize, JsonSchema)]
struct Address {
    street: String,
    city: String,
    country: String,
}

#[derive(Deserialize, JsonSchema)]
struct Company {
    name: String,
    address: Address,
    employees: Vec<Person>,
}
```

## Configuration

Control retry behavior and schema handling:

```rust
use llm_stack::structured::GenerateObjectConfig;

let config = GenerateObjectConfig {
    max_attempts: 3,             // Retry up to 3 times on validation failure
    system_prompt_fallback: true, // Inject schema into system prompt if provider doesn't support structured output
};
```

### Retry behavior

When validation fails, the SDK:
1. Sends the error message back to the model
2. Includes the original response and validation errors
3. Asks the model to try again

```rust
let config = GenerateObjectConfig {
    max_attempts: 5,  // More retries for complex schemas
    ..Default::default()
};
```

### System prompt fallback

For providers without native structured output support:

```rust
let config = GenerateObjectConfig {
    system_prompt_fallback: true,  // Default
    ..Default::default()
};
```

This injects instructions like:
```
You must respond with JSON matching this schema:
{"type": "object", "properties": {...}}
Output ONLY valid JSON, no other text.
```

## Results

`generate_object` returns detailed results:

```rust
let result = generate_object::<Person>(provider, params, config).await?;

// The deserialized value
let person: Person = result.value;

// How many attempts it took
println!("Attempts: {}", result.attempts);

// Raw JSON string (for debugging)
println!("Raw: {}", result.raw_json);

// Total token usage across all attempts
println!("Tokens: {}", result.usage.input_tokens + result.usage.output_tokens);
```

## Streaming

For long outputs, stream partial objects:

```rust
use llm_stack::structured::{stream_object_async, collect_stream_object, PartialObject};
use llm_stack::provider::JsonSchema;
use futures::StreamExt;

let schema = JsonSchema::from_type::<Person>()?;
let stream = stream_object_async::<Person>(provider, params, config).await?;

let partial: PartialObject<Person> = collect_stream_object(stream, &schema).await?;

if let Some(person) = partial.complete {
    println!("Complete: {:?}", person);
} else {
    println!("Partial JSON: {}", partial.raw_json);
}
```

Streaming is useful for:
- Showing progress on long generations
- Partial results if the stream is interrupted
- Real-time UI updates

## Error handling

```rust
use llm_stack::LlmError;

match generate_object::<Person>(provider, params, config).await {
    Ok(result) => println!("{:?}", result.value),

    Err(LlmError::SchemaValidation { message, raw_json }) => {
        // JSON was valid but didn't match schema
        println!("Validation failed: {message}");
        println!("Got: {raw_json}");
    }

    Err(LlmError::ResponseFormat { message, raw_body }) => {
        // Response wasn't valid JSON at all
        println!("Not JSON: {message}");
    }

    Err(e) => {
        // Network error, provider error, etc.
        println!("Error: {e}");
    }
}
```

## Best practices

### 1. Keep schemas simple

Models work better with flat, straightforward structures:

```rust
// Good — clear, flat structure
#[derive(Deserialize, JsonSchema)]
struct ExtractedData {
    name: String,
    date: String,
    amount: f64,
}

// Avoid — deeply nested, complex relationships
#[derive(Deserialize, JsonSchema)]
struct ComplexData {
    entities: Vec<Entity>,
    relationships: Vec<Relationship>,
    metadata: Metadata,
    // ...
}
```

### 2. Use descriptive field names

Field names help the model understand what to generate:

```rust
#[derive(Deserialize, JsonSchema)]
struct Recipe {
    recipe_name: String,           // Clear
    cooking_time_minutes: u32,     // Includes units
    ingredient_list: Vec<String>,  // Descriptive
}
```

### 3. Add descriptions

Use doc comments for extra guidance:

```rust
#[derive(Deserialize, JsonSchema)]
struct Task {
    /// A short, actionable title
    title: String,

    /// Priority from 1 (lowest) to 5 (highest)
    priority: u8,

    /// Estimated time to complete in minutes
    estimated_minutes: Option<u32>,
}
```

### 4. Validate early

Check provider capabilities before relying on structured output:

```rust
use llm_stack::Capability;

let meta = provider.metadata();
if !meta.capabilities.contains(&Capability::StructuredOutput) {
    // Will use system prompt fallback — may be less reliable
    println!("Warning: Using prompt-based structured output");
}
```

### 5. Handle partial results

For important data, validate the result even after successful generation:

```rust
let result = generate_object::<Order>(provider, params, config).await?;

// Additional business logic validation
if result.value.total < 0.0 {
    return Err("Invalid order total".into());
}
if result.value.items.is_empty() {
    return Err("Order must have items".into());
}
```
