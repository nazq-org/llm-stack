# Contributing to llm-stack

Thank you for your interest in contributing to llm-stack! This document provides guidelines and information for contributors.

## Code of Conduct

Be respectful, inclusive, and constructive. We're all here to build great software together.

## Getting Started

### Prerequisites

- **Rust 1.85+** (2024 edition)
- **[just](https://github.com/casey/just)** — Command runner (`cargo install just`)

### Setup

```bash
# Clone the repository
git clone https://github.com/nazq/llm-stack.git
cd llm-stack

# Verify everything works
just gate
```

## Development Workflow

### 1. Create a Branch

```bash
git checkout -b feat/my-feature
# or
git checkout -b fix/bug-description
```

### 2. Make Changes

Write your code following our style guidelines (below).

### 3. Run Quality Checks

```bash
# Full CI check (required before PR)
just gate

# Quick feedback during development
just fcheck  # fmt + check
just test    # run tests
```

### 4. Commit

Use [Conventional Commits](https://www.conventionalcommits.org/) format:

```
feat: add support for streaming tool calls
fix: handle empty response from provider
docs: update quickstart guide
refactor: simplify tool registry internals
test: add coverage for retry interceptor
```

### 5. Submit Pull Request

- Ensure `just gate` passes
- Write a clear PR description
- Reference any related issues

## Code Style

### Rust Guidelines

- **Format**: Run `cargo fmt` (or `just fmt`)
- **Lints**: Zero warnings from `cargo clippy -- -D warnings`
- **Edition**: Rust 2024
- **Async**: Use `async`/`await`, not manual futures

### Documentation

- All public items must have doc comments
- Include examples in doc comments where helpful
- Run `just doc` to verify documentation builds

### Testing

- Write tests for new functionality
- Unit tests go in the same file as the code (`#[cfg(test)]` module)
- Integration tests go in `tests/` directory
- Use `MockProvider` from `test_helpers` for provider tests

Example test:

```rust
#[tokio::test]
async fn test_my_feature() {
    let mock = mock_for("test", "test-model");
    mock.queue_response(/* ... */);

    let result = my_function(&mock).await;
    assert!(result.is_ok());
}
```

### Error Handling

- Use `LlmError` for all error returns
- Include context in error messages
- Set `retryable` flag appropriately for HTTP/provider errors

## Architecture Guidelines

### Adding a New Provider

1. Create a new crate: `crates/llm-stack-{provider}/`
2. Implement the `Provider` trait
3. Add conversion logic for request/response types
4. Add streaming support via `stream()` method
5. Write integration tests
6. Document configuration in `docs/providers.md`

### Adding a New Interceptor

1. Implement the `Interceptor<T>` trait for your domain
2. Add behavior traits if needed (`Retryable`, `Timeoutable`, etc.)
3. Write tests covering success, failure, and edge cases
4. Document in `docs/interceptors.md`

### Modifying Core Types

Core types in `llm-stack-core` are public API. Changes require:

- Backwards compatibility (or clear migration path)
- Documentation updates
- Test coverage

## Pull Request Checklist

Before submitting:

- [ ] `just gate` passes (fmt, clippy, tests, docs)
- [ ] New code has tests
- [ ] Public APIs have doc comments
- [ ] `CHANGELOG.md` updated (if applicable)
- [ ] Commit messages follow Conventional Commits
- [ ] PR description explains the change

## Release Process

Releases are managed by maintainers:

1. Update version in `Cargo.toml` files
2. Update `CHANGELOG.md`
3. Create git tag: `v0.x.y`
4. Push tag to trigger release workflow
5. Publish to crates.io

## Getting Help

- **Questions**: Open a [Discussion](https://github.com/nazq/llm-stack/discussions)
- **Bugs**: Open an [Issue](https://github.com/nazq/llm-stack/issues)
- **Security**: Email maintainers directly (do not open public issues)

## License

By contributing, you agree that your contributions will be licensed under the Apache-2.0 license.
