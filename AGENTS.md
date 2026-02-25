# AGENTS.md — Coding Agent Guide for vib3

## Project Overview

vib3 is a weight-indexed inference engine for Mixture-of-Experts (MoE) models,
written in Rust with optional CUDA support. It implements a three-tier page
buffer manager for NVMe-to-GPU streaming, predictive expert prefetching, and an
OpenAI-compatible HTTP API.

## Build Commands

```bash
# Build (default includes CUDA feature)
cargo build
cargo build --release

# Build without CUDA (CPU-only fallback)
cargo build --no-default-features

# Build a specific binary
cargo build --bin vib3
cargo build --bin vib3-serve
cargo build --bin vib3-convert
```

The build script (`build.rs`) compiles CUDA kernels via `nvcc` targeting sm_89,
sm_90, sm_100. It falls back gracefully if `nvcc` is not found.

## Test Commands

```bash
# Run all tests
cargo test

# Run a single test by name
cargo test test_page_id_roundtrip

# Run a single test with stdout visible
cargo test test_page_id_roundtrip -- --nocapture

# Run tests without CUDA feature (CPU-only)
cargo test --no-default-features

# Run ignored/slow tests (e.g. real model tests)
cargo test --release --no-default-features test_mixtral_cpu_decode_smoke -- --ignored --nocapture

# Run only integration tests
cargo test --test integration_test

# Run only unit tests in a specific module
cargo test --lib tiered_kv
```

Tests are self-contained: each test creates its own temp directory and model
file via `create_test_model()`. Tests requiring real model files skip
gracefully when files are absent.

## Lint / Format

```bash
# Format (standard rustfmt, no custom config)
cargo fmt
cargo fmt --check

# Lint (standard clippy, no custom config)
cargo clippy
cargo clippy --no-default-features
```

**Project invariant: zero warnings.** All code must compile with no warnings
under both `cargo build` and `cargo clippy`.

## Code Style Guidelines

### Naming Conventions

- `snake_case` for functions, variables, modules
- `CamelCase` for structs, enums, traits
- `SCREAMING_SNAKE_CASE` for constants
- Use full descriptive names, not abbreviations: `PageBufferManager`, not `PBM`

### File & Module Structure

- Every file begins with `//!` module-level doc comments explaining purpose,
  design rationale, and architectural context.
- Section headers within files use: `// --- Section Name ---`
- Source lives in `src/` (library) and `tools/src/bin/` (CLI binaries).
- Integration tests are in `tests/integration_test.rs` (single file, ~5k lines).
- Unit tests live inside source files in `#[cfg(test)]` modules at the bottom.

### Imports

- Group by crate path. Internal imports use `crate::module::submodule::Type`.
- Wildcard imports are used for the core types module: `use crate::core::types::*`.
- External crates are imported directly at the top.

### Types

- Heavy use of `#[repr(C)]` and `#[repr(C, packed)]` for zero-copy on-disk
  structures.
- Types that are read from disk derive `bytemuck::Pod + Zeroable`.
- Compile-time size assertions:
  `const _: () = assert!(std::mem::size_of::<Vib3Header>() == HEADER_SIZE);`
- Config structs derive `serde::Serialize/Deserialize` with `#[serde(default)]`.

### Error Handling

Two-tiered approach:

- **Library code** (`src/`): Uses `thiserror`-derived `Error` enum defined in
  `src/core/error.rs` with a `Result<T>` type alias. Error variants have
  helper methods like `is_transient()` and `is_config_error()`.
- **Binary/tool code** (`tools/`): Uses `anyhow::Result` for top-level
  error propagation.

Never use `.unwrap()` in library code except in tests. Prefer `?` propagation.

### Unsafe Code

- Minimal and concentrated in `compute/cuda_ffi.rs` and pointer ops in
  `buffer_manager.rs`.
- Every `unsafe` block MUST have a `// SAFETY:` comment justifying why it is
  sound.
- `unsafe impl Send/Sync` must have justification comments.

### Concurrency

- `parking_lot::Mutex` (not `std::sync::Mutex`) for sync contexts.
- `tokio::sync::Mutex` for async contexts.
- `DashMap` for lock-free concurrent hash maps.
- `Arc` for shared ownership across async tasks.
- `AtomicU64`, `AtomicU8` for hot-path counters.

### Floating Point Comparisons

In tests, use epsilon-based comparison:
```rust
assert!((actual - expected).abs() < 1e-5, "values differ: {actual} vs {expected}");
```

### Documentation

- Public types and methods get `///` doc comments.
- Complex algorithms get inline `//` comments explaining the "why".
- Module-level `//!` docs describe architecture and design decisions.

### Logging

Use the `tracing` crate: `tracing::info!`, `tracing::warn!`, `tracing::debug!`.

### CLI

CLI tools use `clap` with derive macros and a subcommands pattern.

### Feature Flags

- `default = ["cuda"]` — CUDA support via `cudarc`.
- `cuda` — Optional, gated with `#[cfg(feature = "cuda")]`.
- `python` — Optional Python bindings via `pyo3`.
- Every CUDA code path must have a CPU fallback.

## Architecture Overview

```
src/
  core/           # Types, config, errors (PageId, Tier, DType, etc.)
  storage/        # .vib3 format, buffer manager, io_uring NVMe reads
  index/          # Vector index, co-activation graph, domain classifier
  compute/        # CUDA FFI, kernel launchers (matmul, attention, etc.)
  runtime/        # Engine orchestrator, query planner, sampler, KV cache
  api/            # Axum HTTP server (OpenAI-compatible, SSE streaming)
  registry.rs     # Model download, local store, hardware detection
  validation.rs   # Reference model, output comparison
tools/src/bin/
  run.rs          # Main CLI (vib3 run, vib3 pull)
  serve.rs        # API server (vib3-serve)
  convert.rs      # Model converter (vib3-convert)
  bench.rs        # Benchmarks (vib3-bench)
  inspect.rs      # Model inspector (vib3-inspect)
```

## Key Design Principles

1. **Zero-copy where possible** — On-disk structures are `Pod`, read via mmap.
2. **CPU fallback always works** — Never assume CUDA is available.
3. **Tests are self-contained** — No shared mutable state between tests.
4. **Unsafe is minimal and documented** — Concentrated in FFI boundary.
5. **Zero compiler warnings** — Treat warnings as errors.
