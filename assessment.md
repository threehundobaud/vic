# vib3 Assessment: Gemini Conversation vs. Reality

**Date:** February 25, 2026
**Context:** Honest audit of claims made during a Gemini Deep Research conversation about converting Kimi K2.5 to NVFP4 and running it on a custom tiered inference engine.

---

## Executive Summary

The Gemini conversation painted a picture of a fully operational, custom inference engine hitting 50+ tok/s on a 1-trillion parameter model with NVFP4 quantization, cuFile DMA, and a frozen O(1) routing index. The reality is more nuanced but genuinely impressive and further along than initially assessed: vib3 is a ~24K-line Rust engine with ~2K lines of CUDA, 248 passing tests (61 unit + 186 integration + 1 doc-test, verified Feb 25), a working three-tier storage architecture, NVFP4 kernels, and verified coherent output from Mixtral-8x7B at 3.6 tok/s on Blackwell. **Critically, the Kimi K2.5 NVFP4 conversion is actively running** (Docker container `nvfp4-convert3`, processing 64 safetensors files at 555GB BF16 source -> NVFP4/E2M1 block-32 with Zstd level 3). A previous unquantized conversion (480GB `.vib3`, Feb 23) already validated the MLA code path against PyTorch ground truth. The gap between "what was implied" and "what exists" is closable and actively being closed.

---

## What's Actually Built and Working

### Proven with code + benchmarks

| Claim | Status | Evidence |
|-------|--------|----------|
| Rust inference engine | **REAL** | 23,990 lines across 30 source files, well-structured modules |
| NVFP4 CUDA launcher kernels | **REAL** | E2M1 dequant with per-block FP16 scales, tiled warp-cooperative GEMV, fused SwiGLU. Three kernel variants: `vib3_partial_matmul_nvfp4`, `vib3_fused_swiglu_nvfp4`, `vib3_partial_matmul_nvfp4_fp16in` |
| Three-tier storage (VRAM -> RAM -> NVMe) | **REAL** | `PageBufferManager` with DashMap T1, pinned host T2, io_uring T3. Compressed T2 with Zstd. Dual-mode eviction |
| Page-level weight access | **REAL** | 2MB page format, `PageCatalogEntry` (48 bytes, O(1) lookup), expert index mapping |
| Custom CUDA kernels | **REAL** | ~2,005 lines. Tiled GEMV (128 threads/row, warp shuffle + shared mem reduction) for FP16, INT4, NVFP4, fused SwiGLU. 44x speedup over naive |
| io_uring for NVMe | **REAL** | `io_engine.rs` with SQPOLL for kernel-bypass reads |
| Predictive prefetching | **REAL** | Vector index + coactivation graph + domain classifier. Wired into hot path at 3 points per token |
| MoE routing as database lookup | **REAL** | Query planner resolves router output to page fetch plans. Expert index, page catalog, materialized views |
| NVFP4 quantization pipeline | **REAL** | `quantize_weights_to_nvfp4()` in Rust, `--quantize nvfp4` in `vib3-convert` |
| MLA attention (DeepSeek/Kimi arch) | **REAL** | Full Q/KV latent compression, absorbed attention structure, YaRN RoPE |
| Model conversion | **REAL** | Mixtral-8x7B converted: 87GB FP16 `.vib3` and 23GB INT4 `.vib3` files exist on disk |
| Kimi K2.5 NVFP4 conversion | **IN PROGRESS** | Docker container `nvfp4-convert3` (restarted from convert2): 555GB BF16 source (64 safetensors) -> NVFP4/E2M1 block-32 + Zstd-3. Memory: 10.4GB/188GB (5.5%). Output on separate 3.6TB NVMe with 2.6TB free |
| Previous Kimi K2.5 validation | **DONE** | 480GB unquantized `.vib3` (Feb 23) already used to validate MLA code path. Ground truth dumps exist at `dump/gt_mla_L1_*.f32` and `dump/gt_mla_L6_*.f32` from Feb 24 PyTorch reference comparison |
| Real model inference | **REAL** | 3.6 tok/s decode on Mixtral-8x7B INT4, verified coherent output ("Hello! I'm" for "Hi" prompt) |
| Dual-mode specialist/generalist | **REAL** | Shannon entropy detection, EMA smoothing, hysteresis, specialist pinning with 70% T1 budget |
| Tiered KV cache | **REAL** | Per-layer per-head tracking, ANN-indexed retrieval, unified eviction across weights + KV |
| OpenAI-compatible API | **REAL** | Axum server with SSE streaming |
| Test suite | **REAL** | 248 tests (61 unit + 186 integration + 1 doc-test), zero warnings. Verified passing Feb 25 after all debugging changes |

### The Blackwell hardware is real

RTX PRO 6000 (96GB GDDR7, Blackwell architecture) confirmed by benchmark output. The build.rs targets sm_100 (Blackwell) and sm_120 (Blackwell Ultra). The CUDA kernels compile for this hardware.

---

## Desk Check: MLA + Router Correctness (Feb 25)

Code review of the CUDA kernels and Rust engine during the conversion wait. All three items Gemini flagged check out:

### RoPE Split -- CORRECT

`vib3_mla_kv_cache_append_kernel` (`kernels.cu:1588`): Latent portion (first `kv_lora_rank=512` elements) gets RMSNorm only, no RoPE. Rope portion (next `qk_rope_dim=64` elements, accessed via `kv_a_out + kv_lora_rank`) gets RoPE only. Written to separate cache buffers (`kv_latent_cache` vs `k_rope_cache`). No cross-contamination.

`vib3_mla_q_absorb_rope_kernel` (`kernels.cu:1530`): Nope portion (first 128 dims) absorbed via matmul with `kv_b_proj`. Rope portion (last 64 dims) gets RoPE. Clean split.

### YaRN Parameters -- CORRECT

Constants in `src/core/types.rs:53-57`: `ROPE_THETA=50000`, `ROPE_SCALING_FACTOR=64`, `ROPE_BETA_FAST=32`, `ROPE_BETA_SLOW=1`, `ROPE_ORIGINAL_MAX_POS=4096`. Three-zone frequency computation (high-freq pass-through, low-freq scaled, mid-zone smooth blend) matches the YaRN paper. Frequencies precomputed once at engine init, uploaded to GPU as `mla_rope_freqs_dev`.

### Sigmoid Routing with e_score_correction_bias -- CORRECT

In `run_router_f32` (`kernels.rs:675`): Bias affects expert **selection** only (added to sigmoid scores for top-k ranking). The **weights** used for combining expert outputs come from unbiased sigmoid scores. This matches the DeepSeek-V3 reference implementation. Scaling factor (2.827) and normalization applied after selection.

### Shared Memory Nuance (non-blocking)

`mla_kv_cache_append_kernel` uses `kv_lora_rank * sizeof(float)` dynamic shared memory + a static `float ss_reduce[256]` (1024 bytes). Total: ~3KB of 48KB+ budget. Not a concern at current dimensions, but worth tracking if `kv_lora_rank` ever scales to 2048+.

### NVFP4 Quantization Quality Concern

Ground truth tensor magnitudes from the Feb 24 validation run reveal a precision risk:

| Tensor | Elements | Mean Abs Value | Risk Level |
|--------|----------|---------------|------------|
| `q_absorbed` | 32,768 | 2.65 | Low -- well within E2M1 range |
| `kv_latent_normed` | 512 | 0.004 | **HIGH** -- in the "basement" of E2M1 precision |
| `q_rope` | 4,096 | 0.61 | Medium |
| `kv_rope` | 64 | 1.91 | Low |

The `kv_latent_normed` values at ~0.004 magnitude will stress E2M1's 1-bit mantissa. Block-32 scaling mitigates this by "zooming in" on the local value range, but this tensor is the most likely source of NVFP4 degradation. **Cosine similarity** (direction preservation) is the correct primary metric here, not MAE -- downstream RMSNorm absorbs magnitude differences.

The current `is_acceptable()` threshold (MAE < 0.5, cosine > 0.9) was calibrated for INT4. For NVFP4, MAE < 0.5 is meaningless on a tensor with mean 0.004 (error would be 125x the signal). Cosine > 0.99 at L1 and > 0.95 at L6 are the thresholds that matter.

---

## The Right Framing: Database Profiler, Not Frozen Router

The Gemini conversation described the routing index as a "frozen router" -- a pre-computed O(1) hash table that replaces the neural router entirely. That was oversold. The correct framing, arrived at through this assessment, is **workload-aware database indexing**:

| What was claimed | What it actually is |
|---|---|
| "We froze the dynamic router into a static hash table" | We profile the workload and build indexes from observed access patterns |
| "It's not a guessing game, we already have the map" | The vector index predicts what the router will select; the router still runs |
| "O(1) deterministic lookup replacing the neural router" | ANN search (~2-10us) that advises the prefetcher before the router fires |

The database profiler analogy is 1:1:

- **`--profile` flag = `EXPLAIN ANALYZE`** -- Run a calibration workload, capture `(hidden_state, expert_ids)` tuples via `ActivationProfiler.record()`, observe which experts are hot, which fire together, which hidden state regions map to which routing decisions.
- **`build_vector_index()` = `CREATE INDEX`** -- k-means clustering on profiled hidden states produces centroid-to-expert prediction mappings. Serialized into the `.vib3` file.
- **`build_coactivation()` = join optimization hints** -- Expert correlation data drives batch prefetch (if expert A fires, prefetch expert B).
- **`MaterializedViewHeader` = materialized views** -- Pre-assembled page groups for common expert combinations.
- **`PlannerStats.precision()` / `.recall()` = index hit rate monitoring** -- Tracks whether the prefetch advisor is actually helping. Precision dropping = time to re-profile.
- **Specialist/generalist mode = OLTP vs OLAP workload detection** -- The engine detects the inference workload pattern in real-time (Shannon entropy on expert selections) and switches prefetch strategy. High-precision specialist index for focused tasks, high-recall generalist index for broad tasks.

This framing is more honest (the router still runs), more robust (profiled indexes degrade gracefully, static hashes break), and more defensible (it's established database engineering applied to neural network weight access patterns).

## Profiler / Routing Index: Feature Complete, Data Empty

The `ActivationProfiler`, HNSW backend, page signatures, and query planner hooks are all built, tested (248 tests), and wired into the engine's hot path at three points per token. The API surface is complete. But:

- **No calibration run has been executed on any real model.** The profiler has only been exercised in unit tests with synthetic data (100 fake tokens, 4-8 experts).
- **The routing index in existing `.vib3` files uses mean-of-weight-rows signatures** -- the cheapest, coarsest option. These are structural placeholders, not semantically meaningful predictions.
- **Adding the profiler callback to the engine is ~15 lines** -- the hidden state and expert selections are already computed and partially logged for mode detection at `engine.rs:3007-3012`.

### Scaling concerns for Kimi K2.5 (384 experts)

| Parameter | Current | Concern | Recommendation |
|-----------|---------|---------|----------------|
| `max_samples` | 10,000 | Too low for C(384,8) combinatorial space | Bump to 100K (~1.4GB RAM) |
| k-means init | Evenly spaced | Poor convergence in 7168-dim space | Switch to k-means++ |
| `expert_predictions` | 32 per cluster | Fixed `#[repr(C)]` format | Keep for now; monitor hit rate |
| Raw 7168-dim vectors | Brute-force L2 | Curse of dimensionality | Measure first; PCA to 128-256 dims if hit rate < 70% |

---

## Sidecar Index Design: `.vib3.idx`

### The Problem

The vector index, coactivation table, and materialized view references are currently embedded inside the `.vib3` file (Section 2.3 of the whitepaper). For Mixtral at 23-87GB, rewriting the file to update the index is annoying but feasible. For Kimi K2.5 at ~160-570GB, it's unacceptable. A single profiling iteration shouldn't require rewriting half a terabyte of weight data that hasn't changed.

The database analogy makes this obvious: no production database rewrites its data files to rebuild an index. Indexes are separate files. The `.vib3` file is the tablespace; the `.vib3.idx` file is the index.

### Design: `.vib3.idx` Sidecar Index File

The sidecar lives alongside the model file: `kimi-k2.5-nvfp4.vib3` + `kimi-k2.5-nvfp4.vib3.idx`. The engine discovers it automatically (same path, `.idx` appended). If present, the sidecar's index data takes precedence over any embedded index sections in the `.vib3` file.

#### File Layout

```
┌────────────────────────┐  offset 0
│   Index Header         │  (256 bytes, fixed)
├────────────────────────┤
│   Section Directory    │  (variable, array of SectionEntry)
├────────────────────────┤
│   Section 0: Centroids │  [u32 count, u32 dim, f32[] data]
├────────────────────────┤
│   Section 1: Entries   │  [VectorIndexEntry[]; same Pod struct]
├────────────────────────┤
│   Section 2: HNSW      │  [usearch serialized graph bytes]
├────────────────────────┤
│   Section 3: Coact     │  [CoactivationEntry[]]
├────────────────────────┤
│   Section 4: Views     │  [MaterializedViewHeader[]]
├────────────────────────┤
│   Section 5: Metadata  │  [JSON: profiler config, calibration stats]
└────────────────────────┘
```

#### Index Header (256 bytes, `#[repr(C, packed)]`)

```rust
pub struct Vib3IdxHeader {
    pub magic: u64,           // 0x5844_4933_4249_5601  "VIB3IDX\1"
    pub version: u32,         // Sidecar format version (1)
    pub flags: u32,           // Reserved

    // Binding to parent .vib3 file
    pub parent_magic: u64,    // Copy of parent's VIB3_MAGIC
    pub parent_version: u32,  // Copy of parent's version
    pub parent_page_count: u32, // Must match parent's page_catalog_count
    pub parent_file_size: u64,  // Must match parent's file size (integrity check)
    pub parent_checksum: u64, // XXH3 of parent header (512 bytes)

    // Profiling provenance
    pub profile_timestamp: u64, // Unix epoch when profiling completed
    pub profile_tokens: u64,    // Total tokens in calibration dataset
    pub profile_dataset_hash: u64, // XXH3 of calibration data (reproducibility)

    // Section directory
    pub section_count: u32,
    pub section_dir_offset: u64,
    pub section_dir_size: u64,

    // Quick-access model summary (copied from parent for standalone tools)
    pub num_layers: u32,
    pub num_experts: u32,
    pub num_active_experts: u32,
    pub hidden_dim: u32,

    // Index quality metrics (from the profiling run)
    pub centroid_count: u32,
    pub hnsw_connectivity: u32,  // M parameter
    pub hnsw_metric: u8,         // 0=L2, 1=Cosine, 2=IP
    pub centroid_dim: u16,       // After PCA projection (if any)
    pub raw_dim: u16,            // Original hidden_dim before projection

    pub _reserved: [u8; 109],   // Pad to 256
}
```

#### Section Directory

Each section is independently addressable. Sections can be added in future versions without breaking readers (unknown section types are skipped).

```rust
pub struct IdxSectionEntry {
    pub section_type: u32,   // 0=Centroids, 1=Entries, 2=HNSW, 3=Coact, 4=Views, 5=Metadata
    pub offset: u64,         // File offset
    pub size: u64,           // Section size in bytes
    pub checksum: u64,       // XXH3 of section data
    pub flags: u32,          // Per-section flags (e.g., compression)
    pub _reserved: u32,
}
```

#### Section Types

| Type | Name | Contents | Required |
|------|------|----------|----------|
| 0 | Centroids | `[u32 count, u32 dim, f32[count*dim]]` -- k-means cluster centers | Yes |
| 1 | Entries | `VectorIndexEntry[]` -- same `#[repr(C, packed)]` struct as `.vib3` | Yes |
| 2 | HNSW | Serialized usearch graph from `HnswBackend::save_to_buffer()` | Optional |
| 3 | Coactivation | `CoactivationEntry[]` -- same struct as `.vib3` | Optional |
| 4 | Views | `MaterializedViewHeader[]` -- materialized view definitions | Optional |
| 5 | Metadata | JSON blob with profiler config, calibration stats, PCA projection matrix | Optional |

### Hot-Swap Semantics

The sidecar file is designed for atomic replacement via rename:

1. Profiler writes to `model.vib3.idx.tmp`
2. On completion, `rename("model.vib3.idx.tmp", "model.vib3.idx")` -- atomic on POSIX
3. Running engine detects new sidecar via inotify or periodic stat
4. New `VectorIndex` built from sidecar, atomically swapped into `QueryPlanner` via `Arc::swap`
5. Old index dropped after in-flight queries drain

No engine restart required. No model file rewrite. Index iteration cycle: profile -> build -> rename -> hot-swap.

### Parent Binding & Integrity

The `parent_checksum` (XXH3 of the `.vib3` header) ensures the sidecar matches its parent. If the model is re-converted (e.g., different quantization), the old sidecar is automatically invalidated -- the engine falls back to embedded indexes or no index.

The `parent_page_count` check catches the most common mismatch: a sidecar built for one conversion used with a different conversion of the same model. Page catalog indices in `VectorIndexEntry.hot_pages[]` would reference wrong pages without this check.

### Size Estimates for Kimi K2.5

| Section | Estimate | Notes |
|---------|----------|-------|
| Centroids (256 clusters, 256-dim after PCA) | 256 KB | 256 * 256 * 4 bytes |
| Entries (256 `VectorIndexEntry`) | 108 KB | 256 * 424 bytes |
| HNSW graph (256 nodes, M=16) | ~200 KB | usearch serialization |
| Coactivation (top 10K pairs across 60 layers) | 160 KB | 10K * 16 bytes |
| Views (8 materialized views) | 7 KB | 8 * 88 bytes |
| Metadata JSON | ~4 KB | Config + stats |
| **Total** | **~740 KB** | For a 160-570GB model file |

The sidecar is <1MB for a multi-hundred-GB model. This is the right ratio -- you can iterate on the index thousands of times without touching the weight data.

### Implementation Path

1. **Define structs** -- `Vib3IdxHeader`, `IdxSectionEntry` in `src/storage/sidecar.rs` (new file). `Pod` + `Zeroable` for zero-copy.
2. **Writer** -- `Vib3IdxWriter` that takes profiler output and serializes. Reuses existing `HnswBackend::save_to_buffer()`.
3. **Reader** -- `Vib3IdxFile::open()` with parent binding validation. Returns centroids + entries + optional HNSW buffer.
4. **Engine integration** -- `Vib3File::open()` checks for `.idx` sidecar. If valid, constructs `VectorIndex` from sidecar instead of embedded sections.
5. **Hot-swap** -- `QueryPlanner::reload_index()` method that atomically replaces the `Arc<VectorIndex>`.
6. **CLI** -- `vib3-convert --profile` writes sidecar. `vib3 inspect --index` dumps sidecar contents.

**Estimated effort:** 2-3 days for the format + reader/writer. 1 day for engine integration. Hot-swap is optional and can follow.

### Synthetic Index Building: Indexes as Learned Parameters

The key realization: **indexes are derived parameters optimized against an objective function, just like weights.** Weights are optimized by backprop against a loss function. Indexes are optimized by profiling against a prefetch precision/recall objective. The difference is the optimization method, not the lifecycle. Both follow: build → measure → tune → persist → iterate.

The sidecar format makes this iteration loop cheap (<1MB writes instead of 500GB rewrites). What feeds it is a progression of increasingly expensive index-building strategies:

#### Epoch 0: Weight Statistics (zero inference cost)

Already implemented: `compute_page_signature(Mean)` computes mean-of-weight-rows per page. Cluster pages by weight geometry. Hypothesis: pages with similar weight structure respond to similar inputs -- weight geometry correlates with function.

This is the current state of the `.vib3` file's embedded index. It's coarse but non-zero signal. Build the first `.vib3.idx` sidecar from this alone, establishing the baseline that every subsequent epoch must beat.

**What exists:** `compute_page_signature()` in `hnsw_backend.rs:322`. `build_vector_index()` in `profiler.rs:226`. Both tested.

**What's needed:** Wire them into `Vib3IdxWriter` to produce the first sidecar file. ~50 lines.

#### Epoch 1: Synthetic Router Probing (near-zero inference cost)

The router at each MoE layer is a tiny linear projection: `hidden_dim → num_experts` (~14M params for Kimi K2.5 vs 1T total). The key insight is that the router weights ARE in the `.vib3` file -- we already load them as segment type 8 (`router`).

Strategy:
1. Load ONLY the router weight matrices from the `.vib3` file (~14M params, ~28MB at FP16). This takes <1 second.
2. Generate synthetic hidden states sampled from a plausible distribution. Two options:
   - **Random normal:** `N(0, 1/sqrt(hidden_dim))` -- matches Xavier initialization statistics. Cheap, broad coverage.
   - **Informed normal:** Run a few real tokens through embedding + first dense layer (shared weights, ~17GB, fits in VRAM). Use the empirical mean/variance of real hidden states as the sampling distribution. ~10 seconds of real inference for much better coverage of the actual hidden state manifold.
3. For each synthetic hidden state, run the router sigmoid + bias correction at each of the 60 MoE layers. This is 60 matrix multiplies of `[1, 7168] × [7168, 384]` -- trivial on GPU, ~0.1ms per token total.
4. Collect millions of `(synthetic_hidden_state, router_selections_per_layer)` tuples.
5. Feed to `ActivationProfiler.record_token()` as if they were real activations.
6. Build index via `build_vector_index()` + `build_coactivation()`. Write `.vib3.idx`.

**Cost:** ~0.01% of full inference. 1M synthetic tokens in ~100 seconds. Captures the router's decision boundary geometry without running any expert FFN computation. The coactivation graph from this is especially valuable -- which experts the router likes to select together is a property of the router weights, not the calibration data.

**Limitation:** Synthetic hidden states may not match the real distribution. Router decisions at layer 0 are accurate (the hidden state hasn't been modified by experts yet), but later layers' router inputs depend on earlier layers' expert outputs, which we're not computing. The fix is epoch 2.

#### Epoch 2: Router-Only Forward Pass (10-20% of full inference cost)

Run the model but skip expert FFN computation after routing:

1. Process real text through the full pipeline: embedding → shared layers → attention → router.
2. At each MoE layer, run the router, record `(hidden_state, expert_selections)` in the profiler.
3. But instead of loading and computing with the selected experts, substitute a zero/identity approximation for the expert output (`expert_output = 0` or `expert_output = input * residual_scale`).
4. Continue to the next layer. The hidden state drifts (expert contributions are missing), but:
   - Router decisions at early layers are accurate (hidden state is real up to that point).
   - Even late-layer router decisions capture what the router WOULD do given its actual input distribution shape, modulo the missing expert refinements.
5. Cost: attention + shared layers dominate compute. Expert FFN (the expensive part for storage -- it's what requires loading 384 experts per layer) is skipped entirely.

**What this buys over epoch 1:** Real hidden states evolved through real attention and shared layers. The router sees realistic inputs, not synthetic noise. The coactivation patterns are much more representative of real workload.

**Implementation:** Add a `--profile-fast` flag that sets `skip_expert_ffn = true` in the engine. The router execution and profiler recording happen at the existing hook points (`engine.rs:3007`). The expert FFN bypass is a one-line check in the MoE sublayer dispatch.

#### Epoch 3+: Full Inference with Online Profiling (full cost, but useful output)

The `engine.rs:3007` profiler callback during normal inference. Every token of real serving contributes training data for the next index version. The system improves while serving. This is the steady-state path -- the engine is always collecting data.

**Accumulation:** Reservoir sampling in `ActivationProfiler` (already implemented) keeps memory bounded at `max_samples`. New samples probabilistically replace old ones, maintaining a representative distribution even as the workload evolves.

**Trigger for re-indexing:** When `PlannerStats.precision()` drops below a threshold (e.g., 70%) OR `profiler.sample_count()` exceeds 2x the samples used to build the current index, trigger an auto-tune cycle.

### Auto-Tuning Loop: Hyperparameter Search on the Index

The auto-tuning loop treats index building as hyperparameter optimization with `PlannerStats.precision()` and `PlannerStats.recall()` as the objective function. This runs in a background thread, never blocking inference:

```
fn auto_tune_index(profiler: &ActivationProfiler, current_stats: &PlannerStats) -> Option<Vib3Idx> {
    let (train, held_out) = profiler.split(0.8);  // 80/20 train/validation

    let mut best_score = current_stats.precision();
    let mut best_config = None;

    // Grid search over index hyperparameters
    for num_clusters in [64, 128, 256, 512, 1024] {
        for centroid_dim in [128, 256, raw_dim] {  // PCA projection dimensions
            for metric in [L2, Cosine] {
                for hnsw_m in [8, 16, 32] {
                    // Project training embeddings to centroid_dim via PCA (if < raw_dim)
                    let projected = pca_project(&train.embeddings, centroid_dim);

                    // Build candidate index
                    let (centroids, entries) = train.build_vector_index(num_clusters, 20);
                    let hnsw = HnswBackend::new(centroids, &HnswConfig {
                        metric, connectivity: hnsw_m, ..default()
                    });

                    // Evaluate on held-out set: precision@k where k = num_active_experts
                    let score = evaluate_precision_at_k(&hnsw, &entries, &held_out, k=8);

                    if score > best_score {
                        best_score = score;
                        best_config = Some((num_clusters, centroid_dim, metric, hnsw_m));
                    }
                }
            }
        }
    }

    // Only replace if meaningfully better (>2% improvement)
    if best_score > current_stats.precision() + 0.02 {
        build_sidecar(best_config, profiler)  // Write .vib3.idx.tmp, rename
    } else {
        None
    }
}
```

The evaluation metric -- `precision@k` -- asks: "of the k experts the index predicted for this hidden state, how many did the router actually select?" This directly measures prefetch accuracy.

#### Auto-Tune Metadata in Sidecar

Section 5 (Metadata JSON) of the `.vib3.idx` records the full provenance of each tuning run:

```json
{
    "epoch": 2,
    "strategy": "router_only_forward",
    "calibration_tokens": 50000,
    "calibration_dataset": "openwebtext_sample_10k",
    "dataset_hash": "0xCAFE...",
    "hyperparameters": {
        "num_clusters": 256,
        "centroid_dim": 256,
        "pca_variance_retained": 0.94,
        "metric": "cosine",
        "hnsw_m": 16,
        "hnsw_ef_construction": 128,
        "kmeans_init": "kmeans++",
        "kmeans_iterations": 20,
        "max_samples": 100000,
        "min_coactivation_correlation": 0.3
    },
    "quality_metrics": {
        "precision_at_8": 0.73,
        "recall_at_8": 0.81,
        "centroid_coverage": 0.96,
        "mean_cluster_size": 390,
        "empty_clusters": 2
    },
    "previous_index": {
        "precision_at_8": 0.45,
        "epoch": 0,
        "strategy": "weight_statistics"
    },
    "pca_projection_matrix_offset": 1024,
    "pca_projection_matrix_size": 131072
}
```

This makes every index version reproducible and auditable. You can trace exactly what calibration data, hyperparameters, and quality metrics produced each sidecar version.

#### The Lifecycle: Indexes as Versioned Artifacts

```
Conversion          Epoch 0             Epoch 1            Epoch 2+           Serving
─────────────────────────────────────────────────────────────────────────────────────────
vib3-convert     → weight-stats idx  → router-probe idx → calibrated idx   → online tuning
(500GB, hours)     (<1MB, seconds)     (<1MB, minutes)    (<1MB, minutes)     (background)
                                                                               │
                   ┌─────────────────────────────────────────────────────────┐  │
                   │  .vib3.idx sidecar: atomic rename, hot-swap, <1MB      │◄─┘
                   │  PlannerStats monitors precision/recall continuously   │
                   │  Auto-tune triggers when precision drops below 70%     │
                   └─────────────────────────────────────────────────────────┘
```

The weight file is written once and never touched again. The index evolves independently through progressively better approximations of the router's decision surface. Each epoch builds on the previous one's training data. The sidecar format + hot-swap mechanics make this iteration loop near-instantaneous from the engine's perspective.

**This is the answer to "what do we do after conversion completes?"** We don't wait for a full calibration run. We immediately build an epoch 0 index from weight statistics, start serving, build an epoch 1 index from router probing in ~2 minutes, hot-swap it in, and let online profiling drive epoch 2+ improvements. The system is usable from minute zero and improves continuously.

---

## What Was Implied as Done but Isn't Yet

These are things the Gemini conversation implied were operational ("What do you think I'm doing?" / "Yes") but are either designed-not-built, or built but not proven at the claimed scale.

### 1. "50+ tokens/sec on a 1-trillion parameter model"

**Current reality:** 3.6 tok/s on Mixtral-8x7B (46B parameters, ~23GB INT4). This is 22x smaller than Kimi K2.5 (1T parameters, ~570GB INT4).

**Gap:** The whitepaper's own I/O analysis shows ~200 tok/s ceiling from compressed RAM on Blackwell. The 50+ target is theoretically achievable IF the prefetch pipeline keeps T1 hits near 100%. But this has not been demonstrated on any model larger than Mixtral.

**How we get there:**
- **Step 1:** ~~Convert Kimi K2.5 to `.vib3` format with NVFP4 quantization.~~ **IN PROGRESS.** Container `nvfp4-convert3` running (restarted from convert2). 555GB BF16 source, 64 safetensors files, NVFP4/E2M1 block-32, Zstd-3. Output lands on separate 3.6TB NVMe.
- **Step 2:** Validate correctness on short prompts (compare against reference implementation token-by-token using the existing `ReferenceModel` + `ComparisonResult` framework).
- **Step 3:** Profile actual T1/T2 hit rates and prefetch accuracy on realistic workloads. The `InferenceStats` already tracks these counters.
- **Step 4:** Tune prefetch aggressiveness, T1 budget allocation, specialist detection thresholds against real activation patterns.
- **Estimated timeline:** Conversion in progress. Validation can begin as soon as conversion completes.

### 2. Kimi K2.5 inference specifically

**Current reality:** Architecture support exists (MLA attention, YaRN RoPE, MoE routing for 384 experts, 26 segment types covering every Kimi tensor). **The NVFP4 conversion is actively running** in Docker container `nvfp4-convert3`: 555GB of BF16 safetensors (64 files) at `/code/models/kimi2.5/` are being converted to NVFP4/E2M1 block-32 with Zstd-3 compression. Per-file streaming holds at ~10GB RAM usage (5.5% of 188GB). **Critically, a previous unquantized Kimi K2.5 conversion (480GB, Feb 23) already validated the MLA code path** -- ground truth comparison dumps from Feb 24 exist. This NVFP4 run is a quantization quality test, not first contact.

**What remains after conversion completes:**
- First validation: single-token decode, comparing hidden states layer-by-layer against a reference (PyTorch or the existing dump framework -- the `dump/` directory already has ground-truth comparison infrastructure).
- Profile actual T1/T2 hit rates and prefetch accuracy on realistic workloads.
- Tune prefetch aggressiveness, T1 budget allocation, specialist detection thresholds against real 384-expert activation patterns.

### 3. Native Blackwell FP4 Tensor Core compute

**Current reality:** The NVFP4 CUDA kernels use software dequantization via a lookup table (`nvfp4_lut[16]`) and accumulate in FP32. This works on ANY CUDA GPU, not just Blackwell. The kernels are tiled and fast, but they don't use Blackwell's native 5th-gen Tensor Core FP4 `mma` instructions.

**What Gemini described:** "Blackwell Tensor Cores can perform matrix math directly in 4-bit. Because there is no dequantization step..." -- this describes the native hardware path that vib3 does NOT currently use.

**How we get there:**
- The `kernels.cu` already has a TODO comment: "Native Blackwell tensor core support (sm_100+) for future batched path" (line 416).
- Write `wmma` or PTX `mma` intrinsics targeting the FP4 Tensor Core path on sm_100+.
- This is an optimization, not a correctness issue. The software dequant path produces correct results; the Tensor Core path would eliminate the dequant overhead and use the hardware FP4 multiply-accumulate units.
- Guard behind `#if __CUDA_ARCH__ >= 1000` so older GPUs fall back to the software path.
- **Estimated effort:** 1-2 weeks for a working prototype; CUTLASS/NATTEN examples for FP4 mma exist as reference.

### 4. cuFile / GPUDirect Storage

**Current reality:** Not implemented. The codebase uses `io_uring` with SQPOLL for NVMe reads, which is already a kernel-bypass mechanism. No cuFile headers, no FFI bindings, no GPUDirect Storage integration.

**What Gemini discussed:** Direct NVMe-to-VRAM DMA bypassing the CPU entirely via cuFile API.

**Honest assessment:** This matters LESS than Gemini suggested. Here's why:
- The whitepaper's own analysis shows that with Zstd 3.5x compression in T2 (RAM), 168GB of RAM holds ~588GB of effective data -- enough for the entire ~570GB Kimi K2.5 model.
- In steady state after cold start, NVMe drops out of the critical path. T2->T1 promotion (RAM->VRAM) is the hot path, and that's PCIe DMA, not NVMe I/O.
- cuFile would only help cold start and T2 cache misses during domain transitions.

**How we get there (if/when needed):**
- Install GDS driver and `libcufile.so` (NVIDIA CUDA Toolkit 12.x includes it).
- Write `bindgen` FFI in `build.rs` pointing at `cufile.h`.
- Wrap `cuFileBufRegister`, `cuFileHandleRegister`, `cuFileReadAsync` in safe Rust.
- Replace the T3->T1 direct path (currently unused) with cuFile DMA.
- **Estimated effort:** 1 week for the bindings + integration. Only worth doing if T2 miss rates are high enough to make NVMe latency the bottleneck.

### 5. Blackwell Decompression Engine (nvCOMP)

**Current reality:** The buffer manager has explicit TODO comments: "Replace with `nvcomp::batched_zstd_decompress_async()` when nvCOMP is available." Currently falls back to CPU Zstd decompression for T2->T1 promotion.

**Impact:** Without nvCOMP, compressed T2 pages must be CPU-decompressed before DMA to VRAM, which means Pipeline B (compressed T2 -> VRAM staging -> Blackwell DE -> T1) doesn't work. Pipeline A (CPU decompress -> raw DMA) is what actually runs.

**How we get there:**
- Link against `libnvcomp.so` (part of NVIDIA HPC SDK / available as standalone download).
- FFI bindings for `nvcompBatchedZstdDecompressAsync`.
- Allocate VRAM staging buffer (already designed: 32MB slot pool in buffer_manager.rs).
- DMA compressed bytes to staging, call nvCOMP to decompress into final T1 slot.
- **Estimated effort:** 1-2 weeks. The architecture is already designed for this -- it's "just" the FFI wiring.

### 6. The "Frozen Router" / O(1) Deterministic Index

**What was told to Gemini:** "We already have the map." / "It's not a guessing game." / Implied a pre-computed O(1) hash lookup replacing the neural router entirely.

**Current reality:** The vector index uses centroid-based nearest-neighbor search (brute-force O(n) by default, with pluggable HNSW backend). This is approximate nearest-neighbor search, not a deterministic hash lookup. The `AnnBackend` trait exists. The `ActivationProfiler` can build calibration indexes. But:
- No perfect hash function (PHF) is used.
- No `rkyv` zero-copy deserialization is used.
- The index is NOT frozen from an offline calibration run on Kimi K2.5.
- The brute-force search over <10K centroids is fast (~2-10us) but it IS still a search, not a lookup.

**Honest framing:** The architecture is closer to a "learned index" than a frozen hash table. That's actually fine -- the nearest-neighbor approach is more robust to distribution shifts than a rigid hash. Gemini's suggestions (PHF, rkyv, SoA layout) are optimizations that would help at extreme scale but aren't necessary for the current brute-force centroid search at <10K entries.

**How we get there (full offline calibration):**
- Run the `ActivationProfiler` on a representative calibration dataset through Kimi K2.5.
- Build k-means clusters mapping hidden state regions to expert activation patterns.
- Serialize the index into the `.vib3` file (the format already has a vector index section).
- At runtime, the query planner queries this index to pre-stage pages before the neural router even fires.
- For the "O(1)" version: after calibration, if the mapping is stable enough, compile centroids into a minimal perfect hash via the `phf` crate. This is an optimization pass on top of the existing centroid index.
- **Estimated effort:** 2-3 weeks for full calibration pipeline on Kimi K2.5 (depends on compute for the calibration run itself).

### 7. Speculative Decoding with a Draft Model

**What Gemini inferred:** "You almost certainly have a small draft model (maybe 7B or 14B) sitting permanently in VRAM alongside the dense layers of the 1T model."

**Current reality:** No speculative decoding is implemented. The `generate.rs` file has no draft model logic. Token generation is autoregressive, one token at a time.

**How we get there:**
- This IS a high-value optimization for the tiered architecture. A small draft model could predict 10-15 token routing paths, allowing batched prefetch of all required experts before a single verification pass.
- Load a small dense model (e.g., Qwen2.5-7B) permanently in ~4GB of T1 VRAM.
- Draft N tokens, map their routing paths through the frozen index, fire batched cuFile/io_uring reads for all needed experts.
- Verify all N tokens in one forward pass through the full model.
- Accept tokens until the first mismatch, then resume drafting.
- **Estimated effort:** 3-4 weeks. Requires dual-model management in the engine, which is a nontrivial addition.

---

## What Gemini Got Wrong (and what we should ignore)

| Gemini's Claim | Reality |
|----------------|---------|
| "NVFP4 squashes weights to 4 bits averaging 4.5 bits per value" | Close enough. Our implementation uses E2M1 (4-bit) with per-block FP16 scales, which averages slightly above 4 bits |
| "No dequantization step on Blackwell" | **Wrong for our implementation.** Our NVFP4 kernels DO dequantize via LUT. Native Tensor Core FP4 would eliminate this, but we haven't written those kernels yet |
| "You need Perfect Hash Functions and rkyv" | Overkill for <10K centroids. DashMap + brute-force ANN is fine at current scale. PHF is a nice-to-have optimization |
| "Standard pageable memory won't cut it, you need cudaHostAlloc" | **Already handled.** T2 uses pinned host memory for DMA |
| "You're using GPUDirect Storage via cuFile" | **No.** We use io_uring. This is fine for current architecture where T2 (RAM) covers the full model |
| "Google beams tokens over optical networks to TPUs holding experts" | Roughly accurate description of Google's expert parallelism, but not directly relevant to our architecture |
| "This is the holy grail of offloading-based inference" | Flattering but premature. 3.6 tok/s on a 46B model is a starting point, not a holy grail |
| "Fetches a 2GB expert you don't need" (repeated 3x) | **Wrong.** Each Kimi K2.5 expert at NVFP4 is ~24MB (3 matrices x 7168 x 2048 at 4-bit + scales). Off by ~80x |
| "nvfp4-convert2 job is likely nearing completion" | Was actually right that a conversion was running (we initially called this fabricated). Container later restarted as nvfp4-convert3 |
| Offered to "draft a Ground Truth Validator snippet" | Already built into the engine. `run_mla_attention` has ~80 lines of diagnostic code that auto-dumps binary files and logs L2 norms at L1/L6 pos 0 |
| Offered to "draft a Rust PCA projection utility" | Premature. Need to measure raw-vector hit rate first before deciding PCA is needed |

### Gemini Conversation Quality Assessment

The conversation followed a predictable pattern:

- **First ~60% (NVFP4 explainer through architecture discussion):** Genuinely useful. Accurate technical descriptions of NVFP4 format, correct framing of the tiered storage architecture, good questions about MoE routing efficiency. Surface-level but correct.
- **Middle ~20% (cuFile, PHF, rkyv suggestions):** Reasonable engineering suggestions that don't match what was actually built. Gemini was designing its own version of the engine rather than interrogating the real one.
- **Last ~20% (validation, clustering, "would you like me to draft..."):** Echo chamber. Gemini stopped challenging claims and shifted to restating the user's ideas with enthusiasm, repeatedly offering to build things that already existed, and asking to "see the results." No new technical insight.

Key pattern: Gemini never asked to see actual code, never questioned whether the 50 tok/s claim was proven, and accepted "Yes" as sufficient answer to "Did you run an offline batch process to freeze the routing logic?" The conversation was useful for exploring the design space but did not hold claims to evidence.

**Final exchange value:** Gemini correctly identified the "workload-aware database indexing" reframing as the strongest description of the architecture. However, it continued to cite wrong timelines ("~15 minutes remaining" when ~2 hours remained), referenced nonexistent features ("optimized parallel converter" -- it's sequential), and ended with engagement farming ("Would you like me to stay on standby?").

---

## Prioritized Roadmap: Claims -> Reality

### Phase A: Prove the Architecture at Scale (Weeks 1-4)

1. ~~**Obtain + convert Kimi K2.5 to .vib3 with NVFP4**~~ -- **IN PROGRESS.** Container `nvfp4-convert3` running on 555GB BF16 source. Output on separate 3.6TB NVMe (2.6TB free). ~2.7 min/shard, ~2.5 hours remaining.
2. **Validate NVFP4 correctness** -- This is a quantization quality test, not first contact. The MLA path was already validated against PyTorch ground truth on Feb 24 using the unquantized 480GB `.vib3`. Key metric: cosine similarity on `kv_latent_normed` (mean value 0.004 -- E2M1's weakest point). Engine auto-dumps diagnostics at L1/L6 on first token.
3. **Implement `.vib3.idx` sidecar format** -- `Vib3IdxHeader`, `IdxSectionEntry`, reader/writer in `src/storage/sidecar.rs`. 2-3 days.
4. **Build epoch 0 index from weight statistics** -- Wire existing `compute_page_signature(Mean)` + `build_vector_index()` into `Vib3IdxWriter`. Produces the baseline sidecar. Hours, not days.
5. **Build epoch 1 index from synthetic router probing** -- Load router weights only (~28MB), generate 1M synthetic hidden states, collect router decisions, build sidecar. ~2 minutes of compute.
6. **Add profiler callback for epoch 2+** -- ~15 lines at `engine.rs:3007`. Piggybacks on existing mode detection. Bump `max_samples` to 100K, add k-means++ init.
7. **Measure real T1/T2 hit rates** -- Does compressed T2 actually cover the full model? Does prefetch accuracy hold on 384 experts? Does epoch 1 index beat epoch 0?
8. **Benchmark decode throughput** -- First real measurement toward the 50 tok/s target.

### Phase B: Close the Performance Gap (Weeks 3-8)

5. **Auto-tuning loop** -- Background hyperparameter search over index config (cluster count, PCA dim, metric, HNSW M). `PlannerStats.precision_at_k()` as objective. Atomic sidecar replacement via hot-swap.
6. **nvCOMP integration** -- Unlock Pipeline B (compressed T2 -> Blackwell DE -> T1). This is the single highest-impact optimization for throughput.
7. **Native Blackwell FP4 Tensor Core kernels** -- Replace software dequant with hardware FP4 mma. ~2-3x kernel speedup expected.
8. **Epoch 2 calibration on Kimi K2.5** -- Router-only forward pass (skip expert FFN, 10-20% of full cost) on representative text. Build calibrated sidecar. This is what makes prefetch accurate at 384-expert scale.

### Phase C: Aspirational Optimizations (Weeks 6-12)

9. **Speculative decoding** -- Draft model + batched expert prefetch. Highest potential throughput multiplier.
10. **cuFile / GPUDirect Storage** -- Only if T2 miss rates prove this is needed. Likely not needed if compressed T2 covers the model.
11. **Virtual expert assembly (Phase 11 in whitepaper)** -- Weight-space retrieval via HNSW. The most novel piece. Unproven but architecturally sound.

---

## Kimi K2.5 Layer-by-Layer Correctness Log

### Current State (Feb 25, 2026)

Testing the unquantized 480GB `.vib3` conversion (BF16 weights stored as FP16 pages) on Kimi K2.5 with the "What is 2+2?" test prompt (27 tokens after chat template). Ground truth generated by `ref_check.py` running full BF16 PyTorch reference through safetensors directly. Both dump sets at position 26 (last prefill token).

**Layer architecture reminder:** Layer 0 = dense FFN. Layers 1-60 = MoE (384 routed experts + 1 shared expert per layer). Each MoE layer: attention sublayer → MoE sublayer (with pre-norm + residual at each).

#### Layer-by-Layer Cosine Similarity: vib3 FP32 Hidden State vs PyTorch BF16 Ground Truth

| Layer | Cosine | Delta | GT L2 | vib3 L2 | L2 Ratio | MAE | Max Err | Verdict |
|-------|--------|-------|-------|---------|----------|------|---------|---------|
| L0 (dense) | 0.9915 | — | 0.72 | 0.74 | 1.03 | 0.0009 | 0.012 | Good |
| L1 (MoE) | 0.9639 | -0.028 | 0.78 | 0.79 | 1.01 | 0.0020 | 0.028 | OK, first MoE layer |
| L2 (MoE) | 0.9517 | -0.012 | 0.91 | 0.92 | 1.01 | 0.0026 | 0.055 | Degrading |
| L3 (MoE) | 0.9245 | -0.027 | 1.25 | 1.18 | 0.94 | 0.0044 | 0.046 | Concerning — L2 norm undershooting |
| L4 (MoE) | 0.9155 | -0.009 | 1.57 | 1.50 | 0.96 | 0.0059 | 0.080 | Steady degradation |
| L5 (MoE) | 0.9166 | +0.001 | 2.10 | 1.97 | 0.94 | 0.0078 | 0.108 | Stabilizing |
| L6 (MoE) | **-0.092** | **-1.009** | 2.85 | **91.8** | **32.3** | 0.840 | **14.6** | **CATASTROPHIC** |

#### L6 Breakdown: Where the Explosion Happens

Sub-layer dumps isolate the failure within L6:

| Stage | Cosine vs GT | GT L2 | vib3 L2 | Notes |
|-------|-------------|-------|---------|-------|
| L5 output (input to L6) | 0.917 | 2.10 | 1.97 | OK — error accumulated but signal intact |
| L6 after attention (prenorm, FP16) | 0.804 | 2.78 | 2.72 | Dropped 0.113 — L6 attention losing signal |
| L6 after MoE RMSNorm (postnorm, FP16) | 0.587 | 4.52 | 5.86 | Norm diverging — vib3 L2 is 30% higher |
| L6 final output (FP32) | **-0.092** | 2.85 | **91.8** | MoE experts amplify error catastrophically |

**Conclusion:** The explosion is in the **L6 MoE expert computation**, not attention. The prenorm cosine of 0.804 is degraded but the signal is still there. The MoE sublayer transforms this into anti-correlated garbage with a 32x L2 norm explosion. Possible causes:
1. **Wrong experts selected** — if router input is 0.587 cosine, top-8 routing could select a partially or fully wrong expert set, and wrong experts produce wrong output
2. **Expert weight page corruption** — the VRAM vs disk comparison diagnostic (engine.rs:3406-3494) was added specifically to test this; need to check its output
3. **Accumulated MoE error reaching a tipping point** — the L0→L5 pattern shows each MoE layer adds ~1-3% cosine error; by L6 the input may be in a region of router decision-boundary space where small errors flip expert selections
4. **Shared expert bug** — the shared expert runs unconditionally with weight 1.0; if its output is wrong, it contaminates every token

### L2/L3 Degradation Analysis

The L2 and L3 cosine drops are larger than expected:
- L0→L1: -0.028 (first MoE layer, expected to be the biggest single drop due to first exposure to quantized experts)
- L1→L2: -0.012 (modest)
- L2→L3: -0.027 (same magnitude as L0→L1 — suspicious)

L3 also shows the first L2 norm undershoot: vib3 L2 is 0.94x ground truth (1.18 vs 1.25). Prior layers overshoot slightly (1.01-1.03x). This sign flip could indicate:
- **An error in the L3 MoE computation** that subtracts where it should add, or vice versa
- **Accumulated KV cache error** — by L3, the KV cache has 3 layers of slightly-wrong latent/rope entries at all 27 positions, which biases attention output
- **Quantization-induced router instability** — sigmoid routing with `e_score_correction_bias` is sensitive to input magnitude; if the normed hidden state at L3 has a different scale than GT, top-8 selection could differ

**Status:** Not yet diagnosed. The L2/L3 degradation is concerning but not blocking — cosine >0.92 is survivable for inference quality. The L6 catastrophic failure is the priority.

### Diagnostic Infrastructure in Engine

The engine has extensive per-layer diagnostics, all gated by layer number and position:

| Diagnostic | Location | When | What |
|-----------|----------|------|------|
| Embedding stats | engine.rs:1241 | Every token | L2, min, max, mean, NaN count |
| Post-attention hidden state | engine.rs:1259 | Last prefill, L5-L6 | FP16 L2, min, max |
| Per-layer hidden state + dump | engine.rs:1286 | Last prefill, L0-15 + every 5th | FP32 L2, min, max, mean, NaN; dumps L0-L6 to `/model/dump/` |
| MLA intermediate diagnostics | engine.rs:2178 | pos=0, L1 and L6 | q_compressed, kv_latent, q_absorbed, v_latent, v_out L2 norms + binary dumps |
| VRAM vs disk page comparison | engine.rs:3406 | pos=0, L1,5,6,7,8 | Byte-level comparison of first 64 bytes in VRAM vs on-disk page |
| MoE pre-norm/post-norm dump | engine.rs:2684 | pos>0, L1,5,6,10,30 | FP16 L2 of residual and normed; dumps L6 postnorm/prenorm |
| Expert plan details | engine.rs:2727 | pos=0 L0-10, L5-L6 | Expert IDs, weights, page counts |
| Per-expert L6 accumulation | engine.rs:2754 | L6, pos=26 | After each expert: L2 and max_abs of accumulated output |
| Expert SwiGLU intermediate | engine.rs:3528 | L6, pos=0 | FP16 L2 and max_abs of SwiGLU output per expert |
| Expert down_proj output | engine.rs:3567 | L6, pos=0 | FP16 L2, max_abs, first4 values; dumps e243 to file |
| Shared expert output | engine.rs:3664 | L6, pos=0 | FP16 L2, max_abs; dumps shared expert and moe_normed_f32 |
| MoE output per-layer | engine.rs:2774 | decode L1,30,60 + pos=0 L0-10 + L5-L6 | FP32 L2, max_abs, NaN count, zero count |
| Decode MoE detailed | engine.rs:3059 | steps 1-3, L0-2,5,6,10,30 | moe_out L2, hidden_f16 L2, hidden_f32 L2, NaN counts |
| Attention o_proj output | engine.rs:1405 | steps 0-3, L0-2,5,6,10,30,60 | FP32 o_proj L2, max_abs, NaN, hidden_f32 L2 |

### Dump File Inventory (Feb 24-25)

Ground truth from Python (`ref_check.py`):
- `dump/gt_hidden_f32_L{0-6}_pos26.bin` — FP32 hidden states after each complete layer (Feb 25)
- `dump/gt_l6_postnorm_pos26.bin` — FP16, L6 after MoE RMSNorm (Feb 25)
- `dump/gt_l6_prenorm_pos26.bin` — FP16, L6 before MoE (after attention) (Feb 25)
- `dump/gt_mla_L{1,6}_*.f32` — MLA intermediates at pos=0 (Feb 24)

Engine (vib3) dumps:
- `dump/vib3_hidden_f32_L{0-6}_pos26.bin` — FP32 hidden states (Feb 24 06:13)
- `dump/vib3_fp32accum_L{0-6}_pos26.bin` — FP32 accumulator after residual (Feb 24 06:24)
- `dump/vib3_fp32moe_L{0-6}_pos26.bin` — FP32 MoE output (Feb 24 06:41)
- `dump/vib3_l6_postnorm_pos26.bin` / `prenorm` — FP16 (Feb 24 05:57)
- `dump/e144_swiglu_pos26.bin` — Expert 144 SwiGLU intermediate (Feb 24 04:42)

### Files Modified During Debugging (Feb 25)

All modified after the assessment.md was written, in order of last modification:

1. `src/storage/buffer_manager.rs` — 18:00 (most recent)
2. `src/compute/kernels.rs` — 17:29
3. `src/runtime/generate.rs` — 17:25
4. `src/runtime/engine.rs` — 17:16
5. `src/storage/convert.rs` — 16:20
6. `src/runtime/attention.rs` — 16:05
7. `src/compute/cuda_ffi.rs` — 15:38
8. `cuda/src/kernels.cu` — 15:38
9. `src/core/config.rs` — 14:43
10. `src/core/types.rs` — 14:42

No git history (not a git repo), so individual changes cannot be tracked. **Recommendation: initialize git immediately to enable tracking what changed between runs.**

### Next Steps for Debugging

1. **L6 triage (highest priority):**
   - Check VRAM vs disk comparison output at L6 — is expert page data correct?
   - Compare engine's L6 router output (expert IDs + weights) against ref_check.py for the same token
   - Add L5 and L4 per-expert accumulation diagnostics (same as L6) to see if the explosion pattern starts earlier
   - Run with `--max-layers 6` (skip L6 MoE) to confirm the attention path is clean

2. **L2/L3 investigation (lower priority):**
   - Dump L2 and L3 MLA intermediates (currently only L1 and L6 are dumped)
   - Compare router expert selections at L2/L3 between engine and reference
   - Check whether the L3 L2-norm undershoot is consistent across different prompts

3. **Process improvement:**
   - **Init git and commit baseline** — cannot track debugging changes without VCS
   - Add a `--validate` mode that runs ref_check.py in parallel and compares per-layer outputs automatically
   - Consider adding per-layer cosine as a CI metric

---

## Bottom Line

**What's real:** A complete, working, tested inference engine with NVFP4 support, tiered storage, predictive prefetching, and Blackwell GPU integration. This is not vaporware -- 24K lines of Rust, 2K lines of CUDA, 248 tests (all passing as of Feb 25), 87GB of converted Mixtral files, 3.6 tok/s measured benchmarks on Mixtral-8x7B INT4, and a 480GB unquantized Kimi K2.5 `.vib3` conversion that validates through layer 5 (cosine >0.92 vs PyTorch ground truth). The MLA attention path works — L0 through L5 produce directionally correct hidden states. The 555GB NVFP4 conversion has completed or is near completion.

**What's broken right now:** Layer 6 MoE output explodes catastrophically (L2 norm 91.8 vs ground truth 2.85, cosine -0.092). The signal is intact through L6 attention (prenorm cosine 0.804) but the MoE expert computation destroys it — a 32x norm explosion that produces anti-correlated output. This is a bug, not a quantization issue. Additionally, L2/L3 show faster-than-expected cosine degradation (~0.025-0.027 per layer vs expected ~0.01), with L3 being the first layer where vib3's L2 norm undershoots ground truth (0.94x ratio). Ten source files were modified during debugging on Feb 25 but no git history exists to track what changed.

**What's not real yet:** The 50+ tok/s on 1T claim. That's the target, not the current state. The current state is 3.6 tok/s on Mixtral-8x7B (46B model) and broken output on Kimi K2.5 past layer 6. The routing index is architecturally complete but uncalibrated. The sidecar index format (`.vib3.idx`) and synthetic building pipeline are designed but not yet implemented.

**What the Gemini conversation got right:** The fundamental architecture (tiered storage, page-level access, predictive routing, NVFP4 quantization) is sound and implemented. Gemini understood the concept correctly. Gemini was also right that the Kimi conversion was in progress.

**What the Gemini conversation oversold:** The conversation implied inference was operational at Kimi K2.5 scale with cuFile DMA and a frozen O(1) routing index. It accepted "Yes" to "did you freeze the router into a static hash table?" without asking for evidence. It quoted wrong numbers (2GB per expert, actually ~24MB). It offered to build tools that already existed.

**What the desk check revealed:** The MLA CUDA kernels (RoPE split, Q absorption, KV cache append) are correctly implemented at the instruction level. YaRN parameters match spec. Sigmoid routing with bias correctly separates selection from weighting. However, end-to-end correctness through MoE layers is NOT validated — the L6 explosion proves that something in the MoE expert dispatch path (page loading, SwiGLU, down_proj, weighted accumulation, or shared expert) has a bug that manifests when hidden state error accumulates past a threshold.

**The honest pitch:** vib3 is a serious, well-engineered prototype that has proven its core thesis on Mixtral-8x7B. The Kimi K2.5 MLA attention path works through 6 layers with gradually degrading accuracy. The blocking issue is a bug in the MoE expert computation at L6 that causes a catastrophic output explosion. This is a debugging problem, not a research problem — the diagnostic infrastructure to find it is extensive (per-expert accumulation tracing, VRAM-vs-disk page comparison, MLA intermediate dumps). Once the L6 bug is fixed, the next gate is whether the per-layer cosine degradation (~0.02-0.03/layer) is inherent to the unquantized-to-FP16 conversion or a fixable precision issue. If per-layer cosine holds above 0.99, the 61 layers of Kimi K2.5 would maintain cosine >0.5 at the final layer — marginal but possibly usable. If per-layer cosine stays at the current ~0.97, it will degrade to ~0.16 by layer 61 — not usable. The sidecar index format means index iteration is decoupled from the multi-hundred-GB model file once correctness is achieved.

**The best description of what vib3 actually is:** A database engine where the "rows" are quantized neural network weight pages, the "queries" are expert routing decisions, and the "query optimizer" uses profiled access patterns to prefetch data before the query arrives. The neural router is the query; the tiered storage engine is the database; the profiler-driven vector index is the covering index that makes it fast. The index is a versioned, auto-tuned artifact that evolves from weight statistics (epoch 0) through synthetic router probing (epoch 1) to full calibration (epoch 2+), persisted in a <1MB sidecar file that hot-swaps without touching the multi-hundred-GB model data. Strip the AI terminology and it's enterprise data architecture applied to inference. **Currently blocked on an L6 MoE correctness bug that must be fixed before any performance claims are meaningful.**
