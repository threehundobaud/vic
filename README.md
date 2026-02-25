# vib3 — Weight-Indexed Inference Engine for Mixture-of-Experts Models

**vib3** is a storage-engine-first inference runtime for large MoE language models. It treats expert weight matrices as **indexed database tables** — enabling page-level access with predictive prefetching across a three-tier storage hierarchy (VRAM → RAM → NVMe).

Run frontier-class 1T MoE models at interactive speeds on a single GPU.

## Core Insight

Every existing inference engine loads **entire expert weight blocks** (~22 MB each at INT4, ~84 MB at FP16) as monolithic tensors on every activation. With 384 experts per layer and 630 GB total, the model doesn't fit in any single GPU's VRAM.

vib3 builds **indexes over the weight matrices** and manages them as a three-tier storage hierarchy (VRAM → RAM → NVMe) with predictive prefetching. By predicting expert activations and pre-staging only the needed pages, vib3 transforms the performance envelope for single-GPU MoE inference.

## Architecture

```
┌─────────────────────────────────┐
│        Python API (PyO3)        │
├─────────────────────────────────┤
│      Runtime / Query Planner    │  ← Async, ownership-safe
├─────────────────────────────────┤
│    Page Buffer Manager          │  ← DashMap, lock-free hot path
├─────────────────────────────────┤
│    Storage Engine / io_uring    │  ← Kernel-bypass NVMe reads
├─────────────────────────────────┤
│    Vector Index                 │  ← Predictive page lookup
├─────────────────────────────────┤
│    CUDA Kernel Launcher         │  ← Thin unsafe FFI boundary
│    (cuda/*.cu → nvcc → .a)      │
└─────────────────────────────────┘
```

## Target

- **Model**: Kimi K2.5 (1T MoE, 384 experts, INT4 = 630 GB)
- **Hardware**: 1× 96 GB Blackwell + 256 GB RAM + Gen5 NVMe array
- **Target**: 50+ tokens/sec single-user decode

## Building

```bash
# Release build
cargo build --release

# Without CUDA (CPU-only fallback)
cargo build --release --no-default-features

# Run tests
cargo test

# Benchmark
cargo bench
```

## Quick Start

```bash
# Convert model to .vib3 format
cargo run --release --bin vib3-convert -- \
  --model ./Kimi-K2.5-INT4/ --output kimi-k2.5.vib3 --build-indexes

# Run inference server
cargo run --release --bin vib3-serve -- --model kimi-k2.5.vib3 --port 8080

# Benchmark
cargo run --release --bin vib3-bench -- --model kimi-k2.5.vib3 --runs 5

# Inspect model file
cargo run --release --bin vib3-inspect -- --model kimi-k2.5.vib3 --all
```

## Project Structure

```
vib3/
├── src/
│   ├── lib.rs              # Crate root
│   ├── core/               # Types, config, errors
│   │   ├── types.rs        # PageId, Tier, ExpertActivation, Stats
│   │   ├── config.rs       # ModelConfig, EngineConfig, BufferPoolConfig
│   │   └── error.rs        # Error types (thiserror)
│   ├── storage/            # Storage engine
│   │   ├── format.rs       # .vib3 file format (mmap, zero-copy)
│   │   ├── buffer_manager.rs  # Three-tier page pool (DashMap, async DMA)
│   │   └── io_engine.rs    # io_uring async NVMe reads
│   ├── index/              # Predictive indexing
│   │   ├── vector_index.rs # Embedding → expert → page predictions
│   │   ├── coactivation.rs # Expert co-activation graph
│   │   └── domain.rs       # Workload domain classifier
│   ├── compute/            # GPU compute (thin unsafe boundary)
│   │   ├── cuda_ffi.rs     # CUDA FFI wrappers
│   │   └── kernels.rs      # Kernel launchers
│   ├── runtime/            # Inference orchestration
│   │   ├── engine.rs       # Top-level engine
│   │   ├── query_planner.rs # Router → page plan resolution
│   │   └── generate.rs     # Generation loop
│   └── api/                # HTTP server
│       └── server.rs       # OpenAI-compatible API (axum)
├── cuda/src/               # CUDA kernel source (.cu)
│   └── kernels.cu          # Partial matmul, fused SwiGLU, router
├── tools/src/bin/          # CLI tools
├── models/kimi-k2.5/       # Model-specific config
├── build.rs                # CUDA compilation
└── Cargo.toml
```

## Why Rust

vib3 is a **storage engine** that happens to launch GPU kernels — not a GPU library that happens to manage storage. The critical code (buffer pool, page table, async I/O, prefetch scheduling) is all concurrent data structure management, which is exactly where Rust's ownership model prevents the bugs that would take weeks to debug in C++.

The CUDA boundary is thin (~200 lines of `unsafe`) — everything above it is safe Rust.

## License

Apache 2.0
