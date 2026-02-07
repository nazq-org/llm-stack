# llm-stack development commands

set shell := ["bash", "-euo", "pipefail", "-c"]

# Unset CARGO_TARGET_DIR to avoid filling /tmp
_unset := "unset CARGO_TARGET_DIR &&"

# Default: run check, test, clippy, fmt
default: check test clippy fmt-check

# Compile check (all features)
check:
    {{_unset}} cargo check --all-features

# Run all tests
test:
    {{_unset}} cargo test --all-features

# Run a specific test by name
test-one NAME:
    {{_unset}} cargo test --all-features -- {{NAME}}

# Clippy with all features, deny warnings
clippy:
    {{_unset}} cargo clippy --all-features -- -D warnings

# Auto-fix clippy suggestions
clippy-fix:
    {{_unset}} cargo clippy --all-features --fix --allow-dirty

# Format all code
fmt:
    {{_unset}} cargo fmt

# Check formatting without modifying
fmt-check:
    {{_unset}} cargo fmt -- --check

# Format then compile check — fast feedback loop
fcheck: fmt clippy test

# Full CI-style gate: fmt + clippy + test + docs
gate: fmt-check clippy test doc

# Build release
build-release:
    {{_unset}} cargo build --release

# Clean build artifacts
clean:
    {{_unset}} cargo clean

# Run tests with output visible
test-verbose:
    {{_unset}} cargo test --all-features -- --nocapture

# Build and check documentation
doc:
    {{_unset}} RUSTDOCFLAGS="-D warnings" cargo doc --all-features --no-deps

# Open documentation in browser
doc-open:
    {{_unset}} RUSTDOCFLAGS="-D warnings" cargo doc --all-features --no-deps --open

# Run all integration tests (requires API keys: ANTHROPIC_API_KEY, OPENAI_API_KEY, Ollama)
# Tests within each binary run serially (--test-threads=1) to avoid API rate limits.
# Tool loop tests are provider-agnostic (in llm-stack core) and run against the first available provider.
integ:
    {{_unset}} cargo test --all-features -p llm-stack --test integration_tool_loop -- --nocapture --test-threads=1
    {{_unset}} cargo test --all-features -p llm-stack-anthropic --test integration -- --nocapture --test-threads=1
    {{_unset}} cargo test --all-features -p llm-stack-openai --test integration -- --nocapture --test-threads=1
    {{_unset}} cargo test --all-features -p llm-stack-ollama --test integration -- --nocapture --test-threads=1

# Run integration tests for a single provider (anthropic, openai, ollama)
integ-one PROVIDER:
    {{_unset}} cargo test --all-features -p llm-stack-{{PROVIDER}} --test 'integration*' -- --nocapture --test-threads=1

# Run only the provider-agnostic tool loop integration tests
integ-tool-loop:
    {{_unset}} cargo test --all-features -p llm-stack --test integration_tool_loop -- --nocapture --test-threads=1

# Coverage report (requires cargo-llvm-cov)
coverage:
    {{_unset}} cargo llvm-cov --all-features --html

# Coverage lcov output for CI
coverage-lcov:
    {{_unset}} cargo llvm-cov --all-features --lcov --output-path lcov.info
