# Changelog

All notable changes to this project will be documented in this file.

## [0.4.0] - 2026-02-07

### Features

- *(tool)* Unified LoopEvent stream and streaming-first LoopCore ([#16](https://github.com/nazq-org/llm-stack/pull/16))

### Miscellaneous

- Release v0.3.0 ([#14](https://github.com/nazq-org/llm-stack/pull/14))

### Refactor

- *(tool)* Post-review cleanup — dedup, correctness, and docs ([#17](https://github.com/nazq-org/llm-stack/pull/17))


## [0.3.0] - 2026-02-06

### Bug Fixes

- Update README resumable loop example and add Usage::total_tokens() ([#13](https://github.com/nazq-org/llm-stack/pull/13))

### Features

- *(registry)* Add shared HTTP client to ProviderConfig ([#10](https://github.com/nazq-org/llm-stack/pull/10))
- *(tool)* Add resumable tool loop and optimize hot-path allocations ([#9](https://github.com/nazq-org/llm-stack/pull/9))
- *(tool)* Replace LoopEvent with TurnResult API for resumable tool loop ([#12](https://github.com/nazq-org/llm-stack/pull/12))


## [0.2.2] - 2026-02-05

### Documentation

- Add provider crates section with links to docs.rs

### Features

- *(usage)* Skip serializing None fields in Usage struct ([#8](https://github.com/nazq-org/llm-stack/pull/8))


## [0.2.1] - 2026-02-05

### Miscellaneous

- Add docs.rs metadata configuration


## [0.2.0] - 2026-02-05

### Features

- Llm-stack SDK — unified Rust interface for LLM providers
- Rename llm-stack-core to llm-stack ([#4](https://github.com/nazq-org/llm-stack/pull/4))

