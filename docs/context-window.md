# Context Window Management

Long conversations eventually exceed the model's context window. The `ContextWindow` type helps you track token usage and compact old messages before hitting limits.

## Quick example

```rust
use llm_stack_core::context::ContextWindow;
use llm_stack_core::ChatMessage;

// 128K context, reserve 4K for output
let mut window = ContextWindow::new(128_000, 4_000);

// Add messages with token counts (from provider usage)
window.push(ChatMessage::system("You are helpful."), 10);
window.push(ChatMessage::user("Hello!"), 5);
window.push(ChatMessage::assistant("Hi there!"), 8);

// Check remaining space
println!("Available: {} tokens", window.available());

// Protect recent messages from compaction
window.protect_recent(2);

// When approaching capacity, compact
if window.needs_compaction(0.8) {
    let old = window.compact();
    // Summarize old messages, then add the summary
    window.push(ChatMessage::system("Previous context: ..."), summary_tokens);
}
```

## Core concepts

### Token budget

The context window is divided into:
- **Input budget**: `max_tokens - reserved_for_output`
- **Reserved for output**: Space the model needs to generate a response

```rust
// 8K context, 2K reserved for output = 6K for input
let window = ContextWindow::new(8000, 2000);
assert_eq!(window.input_budget(), 6000);
assert_eq!(window.available(), 6000);  // Nothing pushed yet
```

### Token tracking

Track each message's token count as you add it:

```rust
// After a provider call, use actual usage
let response = provider.generate(&params).await?;
let tokens = response.usage.input_tokens as u32;
window.push(message, tokens);

// Or estimate before a call
use llm_stack_core::context::estimate_message_tokens;
let estimated = estimate_message_tokens(&message);
window.push(message, estimated);
```

### Compaction

When the window fills up, remove old messages:

```rust
// Check if 80% full
if window.needs_compaction(0.8) {
    // Remove compactable messages
    let removed = window.compact();

    // Summarize them (your LLM call)
    let summary = summarize_messages(&removed).await?;

    // Add the summary back
    window.push(ChatMessage::system(&summary), summary_tokens);
}
```

## API reference

### Creating a window

```rust
let window = ContextWindow::new(max_tokens, reserved_for_output);
```

- `max_tokens`: Model's full context window (e.g., 128000 for Claude)
- `reserved_for_output`: Tokens to reserve for the response (e.g., 4096)

Panics if `reserved_for_output >= max_tokens`.

### Adding messages

```rust
window.push(message, token_count);
```

New messages are compactable by default. Use `protect()` to preserve important messages.

### Checking capacity

```rust
// Tokens remaining for new content
let available = window.available();

// Current total tokens
let used = window.total_tokens();

// Check if above threshold (0.0 to 1.0)
if window.needs_compaction(0.8) {
    // 80% full, time to compact
}
```

### Protecting messages

Protect messages you don't want removed during compaction:

```rust
// Protect the last N messages
window.protect_recent(3);  // Keep last 3

// Protect a specific message by index
window.protect(0);  // Keep the system prompt

// Check protection status
if window.is_protected(0) {
    println!("Message 0 is protected");
}

// Remove protection
window.unprotect(0);
```

### Compacting

```rust
let removed: Vec<ChatMessage> = window.compact();
```

Returns all compactable messages in order. Protected messages stay in the window.

### Accessing messages

```rust
// Iterate without allocation
for msg in window.iter() {
    println!("{:?}", msg.role);
}

// Get as a Vec (allocates)
let messages: Vec<&ChatMessage> = window.messages();

// Get owned copies
let owned: Vec<ChatMessage> = window.messages_owned();

// Message count
let count = window.len();
let empty = window.is_empty();
```

### Token counts

```rust
// Get token count for a specific message
let tokens = window.token_count(index);

// Update after getting accurate count from provider
window.update_token_count(index, actual_tokens);
```

## Token estimation

When you don't have exact counts, use the built-in heuristics:

```rust
use llm_stack_core::context::{estimate_tokens, estimate_message_tokens};

// Estimate text tokens (~4 chars per token for English)
let tokens = estimate_tokens("Hello, how are you?");  // ~5 tokens

// Estimate a full message (includes overhead)
let msg = ChatMessage::user("What's the weather?");
let tokens = estimate_message_tokens(&msg);  // text + 4 overhead
```

These are rough estimates. For accuracy, use provider-reported token counts after calls.

### Estimation details

| Content type | Estimation |
|--------------|------------|
| Text | `len / 4` (ceiling) |
| Image | 85 tokens (low-res baseline) |
| Tool call | name tokens + args JSON tokens |
| Tool result | content tokens + 10 overhead |
| Message overhead | +4 tokens (role markers) |

## Common patterns

### Sliding window with system prompt

Keep the system prompt and recent messages:

```rust
let mut window = ContextWindow::new(128_000, 4_000);

// System prompt is always protected
window.push(ChatMessage::system("You are a helpful assistant."), 15);
window.protect(0);

// Add conversation messages
for turn in conversation {
    window.push(turn.user_msg, turn.user_tokens);
    window.push(turn.assistant_msg, turn.assistant_tokens);

    // Protect the last 2 exchanges (4 messages)
    window.protect_recent(4);

    // Compact when 80% full
    if window.needs_compaction(0.8) {
        let old = window.compact();
        let summary = summarize(&old).await?;
        window.push(ChatMessage::system(summary), estimate_tokens(&summary));
    }
}
```

### Pre-flight capacity check

Check if a new message fits before sending:

```rust
let new_msg = ChatMessage::user(&user_input);
let estimated = estimate_message_tokens(&new_msg);

if estimated > window.available() {
    // Compact first
    let old = window.compact();
    // ... summarize old messages
}

window.push(new_msg, estimated);
```

### Update with actual usage

Replace estimates with real counts:

```rust
// Push with estimate
let estimated = estimate_message_tokens(&user_msg);
let user_idx = window.len();
window.push(user_msg.clone(), estimated);

// Make the API call
let params = ChatParams {
    messages: window.messages_owned(),
    ..Default::default()
};
let response = provider.generate(&params).await?;

// Update with actual input tokens
// (includes all messages, so this is approximate for the last one)
window.update_token_count(user_idx, estimated);  // Keep estimate or refine

// Add response with actual output tokens
window.push(
    ChatMessage::assistant(response.text().unwrap_or_default()),
    response.usage.output_tokens as u32,
);
```

## Tips

1. **Reserve enough for output** — If your responses average 1K tokens, reserve at least 2K to be safe

2. **Protect strategically** — System prompts, important context, and recent exchanges should be protected

3. **Summarize thoughtfully** — Good summaries preserve key facts; bad ones lose context

4. **Update estimates** — Replace heuristic estimates with actual counts when available

5. **Monitor usage** — Log `window.total_tokens()` and `window.available()` to tune thresholds

6. **Handle edge cases** — What if `compact()` returns nothing? (All messages protected)

```rust
let removed = window.compact();
if removed.is_empty() && window.needs_compaction(0.9) {
    // Everything is protected but we're almost full
    // Consider unprotecting older messages or warning the user
}
```
