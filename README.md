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

- **Model**: Kimi K2.6 (1T MoE, 61 layers, 384 routed + 1 shared expert, top-8, MLA attention)
- **Reference hardware**: 1× **NVIDIA RTX PRO 6000 Blackwell Workstation** (96 GB GDDR7, compute capability **12.0 / SM120a**) + 256 GB RAM + Gen5 NVMe array on /data
- **Toolchain minimum**: CUDA 12.8+ (required for `sm_120a` block-scaled MMA used by the native NVFP4 path)
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
cargo run --release --bin vic-convert -- \
  --model ./Kimi-K2.5-INT4/ --output kimi-k2.5.vib3 --build-indexes

# Run inference server
cargo run --release --bin vic-serve -- --model kimi-k2.5.vib3 --port 8080

# Benchmark
cargo run --release --bin vic-bench -- --model kimi-k2.5.vib3 --runs 5

# Inspect model file
cargo run --release --bin vic-inspect -- --model kimi-k2.5.vib3 --all
```

## MLA Attention Backends

K2.6 uses DeepSeek-style Multi-head Latent Attention (MLA). vib3 has three
decode backends available, selected via env vars:

| backend | selector | status on sm_120a |
|---|---|---|
| **built-in** (9-kernel pipeline in `kernels.cu`) | (default, no flag) | works |
| **CUTLASS** (CUTLASS 4.2 `Sm100FmhaMlaKernelTmaWarpspecialized`) | `VIB3_CUTLASS_MLA=1` | can't fit smem on sm_120a; runtime falls back to built-in |
| **xqa K26** (forked flashinfer `mla_sm120.cu`, `third_party/xqa/`) | `VIB3_CUTLASS_MLA=1 VIB3_MLA_BACKEND=xqa` | works, numerics match built-in at first-token decode (bit-exact L2/max_abs across layers 0–60) |

The xqa K26 port lives in `third_party/xqa/mla_sm120_k26.cu` — a 64-head
FP16 derivation of flashinfer's xqa `mla_sm120.cu` whose upstream is hard-gated
to 128 heads × FP8 e4m3. The NOTICE, the delta vs upstream, and the debug
history are in `third_party/xqa/PORT_STATUS.md`.

### Decode attention wall time (K2.6 on RTX PRO 6000 Blackwell)

Per-step attention across 61 MLA layers, 30-token warm prompt, greedy decode,
steady-state step (post-paging-warmup):

| backend | attn/step (ms) | first-token numerics |
|---|---|---|
| built-in | 74.2 | reference |
| xqa K26 | 72.1 | matches reference to FP16 noise |

Attention is ~1% of K2.6 decode latency (MoE dominates at ~5–7s/step), so
the port is primarily a correctness landing that unblocks the `--chat` path
(previously hit a NaN mid-MLA on certain token sequences). Further attention
perf work is bounded by this ceiling — attention kernel speed does not
move end-to-end tok/s meaningfully on K2.6.

## Runtime Env Vars (Debug/Guardrails)

- `VIB3_NVFP4_DELTANET=1` — request NVFP4 path for DeltaNet projections.
- `VIB3_F32_DELTANET_PROJ=1` — enable experimental FP32-input DeltaNet projections (default OFF; may regress quality on some checkpoints).
- `VIB3_ENABLE_SHARED_NVFP4_PREASSEMBLY=1` — enable shared projection preassembly conversion to NVFP4 on qwen3.5 (default OFF; FP16 shared projections are used by default for correctness).
- `VIB3_ALLOW_UNSAFE_DELTANET_NVFP4=1` — explicit override for Qwen3.5 models; without this, DeltaNet NVFP4 is ignored due known severe quality regression.
- `VIB3_NVFP4_DELTANET_QKV=0|1`, `VIB3_NVFP4_DELTANET_Z=0|1`, `VIB3_NVFP4_DELTANET_OUT=0|1` — per-projection DeltaNet NVFP4 toggles (when enabled).
- `VIB3_NVFP4_DELTANET_SCALAR=1` — force scalar NVFP4 GEMV for DeltaNet (kernel-path isolation).
- `VIB3_ALLOW_UNSAFE_NVFP4_EXPERTS=1` — opt into NVFP4 expert fast path on Qwen3.5; default is safer FP16 expert path for serving quality.
- `VIB3_DIAG=1` — enable runtime diagnostics.
- `VIB3_DIAG_NVFP4_COMPARE=1` — log runtime FP16-vs-NVFP4 projection deltas (`NVFP4_COMPARE`).
- `VIB3_DIAG_NVFP4_WEIGHT_AUDIT=1` — log load-time FP16-vs-dequantized-NVFP4 weight deltas (`NVFP4_WEIGHT_AUDIT`).
- `VIB3_CUTLASS_MLA=1` — route MLA decode through `mla_cutlass_decode` instead of the built-in pipeline.
- `VIB3_MLA_BACKEND=xqa` — when `VIB3_CUTLASS_MLA=1` is also set, dispatch to the xqa K26 kernel (`third_party/xqa/mla_sm120_k26.cu`) instead of CUTLASS.

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
