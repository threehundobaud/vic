# vib3: A Storage-Engine-First Inference Runtime for Trillion-Parameter Mixture-of-Experts Models

**Version 1.0 -- February 2026**

---

## Abstract

Mixture-of-Experts (MoE) models have emerged as the dominant architecture for frontier language models, achieving trillion-parameter scale while activating only a small fraction of parameters per token. Yet existing inference engines treat MoE weights the same way they treat dense model weights: as monolithic tensors that must be loaded entirely into accelerator memory. This is the equivalent of performing a full table scan for every database query.

vib3 is an inference runtime that inverts this assumption. It treats expert weight matrices as **indexed database tables**, enabling page-level random access with predictive prefetching across a three-tier storage hierarchy (GPU VRAM → system RAM → NVMe SSD). By building vector indexes over the weight space and fetching only the 2 MB pages needed per expert activation -- typically 4-22 MB per expert (a few hot pages) instead of 22-84 MB for the full expert -- vib3 targets trillion-parameter MoE models at interactive speeds on a single GPU.

**Beyond prefetching, vib3 is evolving toward a deeper thesis: inference as retrieval over indexed weight space.** Rather than using a learned router to select fixed experts and then loading their pages, the system uses an embedded HNSW vector index (via usearch) to retrieve the most relevant weight pages directly from the hidden state -- regardless of expert boundaries. This enables **virtual expert assembly**: composing ad-hoc computation from the top-k most relevant weight pages per token, bypassing the fixed expert partitioning entirely. The expert boundary becomes a storage convention, not a computational constraint. New capabilities can be added by inserting weight pages into the index without retraining a router.

**The key quantitative finding:** with Zstd compression at 3.5x ratio, 168 GB of system RAM holds ~588 GB of effective model data -- enough to cover the entire ~570 GB model. Combined with Blackwell's hardware Decompression Engine (600 GB/s), compressed pages flow from RAM to VRAM with near-zero decompression cost. NVMe drops out of the steady-state critical path. The I/O ceiling from compressed RAM alone is ~200 tok/s, making attention compute -- not storage -- the sole bottleneck. A tiered compression cascade (maximum compression on NVMe, GPU-friendly compression in RAM, raw in VRAM) further multiplies effective NVMe bandwidth for cold-start and cache-miss scenarios.

vib3 is implemented as ~18,900 lines of Rust (plus ~5,340 lines of integration tests and ~490 lines of CUDA kernels) with 248 tests (61 unit + 186 integration + 1 doc-test), zero warnings: a complete `.vib3` binary file format, three-tier page buffer manager with compressed storage, io_uring NVMe integration, predictive indexing (vector index wired into the hot path with pluggable ANN backend, coactivation graph, domain classifier), tiled warp-cooperative CUDA GEMV kernels achieving 3.6 tok/s on Mixtral-8x7B (single GPU, all weights in VRAM), INT4 quantization pipeline, entropy-based dual-mode adaptive caching, a unified tiered KV cache with sparse attention and ANN-indexed retrieval, a validation framework with deterministic reference model, and an OpenAI-compatible streaming API server.

This paper describes the architecture and its rationale, situates vib3 against existing approaches (BitNet.cpp, AirLLM, vLLM, llama.cpp, TensorRT-LLM, and the emerging tiered KV cache ecosystem including LMCache, NVIDIA Dynamo, and AWS HyperPod), examines the database storage engine analogy and where it leads, describes the implemented unified weight-and-KV tiered cache, and lays out a validation plan for the assumptions that determine whether this approach works.

---

## Table of Contents

1. [The Problem](#1-the-problem)
2. [Core Insight: Weights as Indexed Tables](#2-core-insight-weights-as-indexed-tables)
   - 2.4 [Virtual Experts: Weight-Space Retrieval as the Inference Mechanism](#24-virtual-experts-weight-space-retrieval-as-the-inference-mechanism)
3. [Architecture](#3-architecture)
4. [What We Built (Phases 1-10)](#4-what-we-built-phases-1-10)
5. [Comparative Analysis](#5-comparative-analysis)
6. [The Database Analogy: Where It Holds, Where It Doesn't, and Where It Leads](#6-the-database-analogy-where-it-holds-where-it-doesnt-and-where-it-leads)
7. [Failure Modes and Honest Risks](#7-failure-modes-and-honest-risks)
8. [The Unified Tiered Cache](#8-the-unified-tiered-cache)
   - 8.3 [Embedded Vector Index: usearch as the Routing Engine](#83-embedded-vector-index-usearch-as-the-routing-engine)
9. [What We Borrowed and Why](#9-what-we-borrowed-and-why)
10. [Validation Plan](#10-validation-plan)
11. [Reference System Architecture](#11-reference-system-architecture)
12. [Roadmap](#12-roadmap)
13. [Conclusion](#13-conclusion)

---

## 1. The Problem

### 1.1 MoE Models Are Sparse but Loaded Densely

Kimi K2.5 has 384 experts per MoE layer across 60 MoE layers. Each token activates 8 of 384 experts -- roughly 2% of expert parameters. At INT4 quantization with group-32 scales, the model is ~570 GB (384 × 60 × ~24 MB per expert ≈ 553 GB expert weights + ~17 GB shared weights). But the parameters touched per token are ~11.5 GB (8 experts × 60 layers × ~24 MB per expert at INT4 with scales).

Every existing inference engine -- vLLM, TensorRT-LLM, llama.cpp, Triton -- loads entire expert weight blocks when an expert is activated. For Kimi K2.5, each expert's weight block is approximately **24 MB at INT4 with group-32 scales** (or ~84 MB at FP16) across its three projection matrices (up_proj, gate_proj, down_proj): 3 matrices × 7168 × 2048 at 4-bit with FP16 per-group scales (group_size=32) ≈ 24 MB. Activating 8 experts at one layer means loading ~192 MB of weight data. Across all 60 MoE layers, that's ~11.5 GB per token -- manageable individually, but the problem is that with 384 experts per layer and varying activation patterns, the **total model size** (~570 GB) vastly exceeds VRAM. The key optimization opportunity is not avoiding per-expert loads (each is only 24 MB), but **predicting which experts will be needed and pre-staging them** across the storage tiers before they're requested.

This is the database equivalent of reading an entire table to answer a query that matches a few rows.

### 1.2 The Hardware Reality

A single NVIDIA Blackwell GPU has 96 GB of GDDR7 VRAM (RTX PRO 6000 workstation). The ~570 GB model does not fit. The standard solution is multi-GPU tensor parallelism -- 8 or more GPUs, with their associated cost, power, and interconnect complexity.

But consider the storage hierarchy available on a single workstation:

| Tier                 | Capacity | Bandwidth to GPU      | Latency |
| -------------------- | -------- | --------------------- | ------- |
| GPU VRAM (GDDR7)     | 96 GB    | ~1,792 GB/s internal  | ~ns     |
| System RAM (DDR5)    | 180 GB   | ~64 GB/s via PCIe 5.0 | ~100 ns |
| NVMe Gen5 (4x array) | 8+ TB    | ~28-56 GB/s           | ~5 us   |

The ~11.5 GB of active parameters per token could be loaded from NVMe in under 500ms, from RAM in under 200ms, and computed on in under 50ms. If the system can predict which parameters are needed and prefetch them before they're needed, single-GPU inference becomes viable.

### 1.3 Why This Hasn't Been Done

Three reasons:

1. **Dense models don't benefit.** Every parameter is used on every token. There's nothing to skip.
2. **Existing MoE engines were built for multi-GPU.** DeepSeek, Mixtral, and Switch Transformer deployments use expert parallelism across GPUs. The assumption is that you have enough aggregate VRAM.
3. **Page-level weight access requires custom kernels.** Standard GEMM libraries expect contiguous weight matrices. Computing with a subset of rows requires kernels that operate on page-level slices.

vib3 addresses all three: it targets MoE models specifically, it's designed for single-GPU with storage offloading, and it implements page-level partial matmul kernels.

---

## 2. Core Insight: Weights as Indexed Tables

### 2.1 The Analogy

| Database Concept    | vib3 Equivalent                                          |
| ------------------- | -------------------------------------------------------- |
| Table               | Expert weight matrix (up_proj, gate_proj, down_proj)     |
| Row                 | Contiguous slice of weight matrix rows                   |
| Page                | 2 MB aligned block of weight data                        |
| Page buffer pool    | Three-tier VRAM/RAM/NVMe buffer manager                  |
| B-tree index        | Expert index (expert_id -> page ranges)                  |
| Query               | Router output (which experts, which weights)             |
| Query optimizer     | Query planner (router output -> page fetch plan)         |
| Materialized view   | Pre-grouped pages for common workload domains            |
| Query result cache  | T1 VRAM page pool (recently-used pages stay hot)         |
| Predictive prefetch | Vector index (embedding -> predicted expert activations) |

### 2.2 What This Buys

A standard inference engine activates expert 42 at layer 5 and loads all ~24 MB (INT4 with group-32 scales) of that expert's weights as a monolithic block. vib3:

1. Looks up expert 42, layer 5 in the **expert index** -- finds it spans 3 segments x N pages.
2. The **query planner** determines which pages are needed based on the computation.
3. Checks the **page buffer pool** -- some pages may already be in VRAM (T1 hit) or RAM (T2 hit).
4. Issues **io_uring** reads for missing pages from NVMe (T3).
5. While waiting for I/O, the **vector index** predicts which experts will be needed at layer 6 and starts prefetching.
6. The **CUDA kernels** operate on page-level slices, not full matrices.

The key savings:

- Prediction-driven prefetch hides NVMe latency behind current-layer compute
- Pages shared across tokens stay in VRAM (buffer pool caching)
- Coactivation graph expands prefetch to likely-needed neighbors
- Sub-expert page-level access enables partial loading when only specific rows are needed

### 2.3 Where the Analogy Is Not Just Analogy

This isn't metaphorical. The `.vib3` file format is a literal database file:

```
File Header (512 bytes, fixed)
Model Metadata (JSON)
Page Catalog (48-byte fixed entries, O(1) lookup)
Expert Index (expert_id -> page ranges)
Vector Index (embedding -> expert predictions)
Coactivation Table (expert pair correlations)
Materialized View Catalog
Page Data (2 MB-aligned, hugepage-compatible)
```

Each `PageCatalogEntry` is 48 bytes and contains the page's physical location, logical identity (layer, expert, segment, page_idx), row/column ranges, mean activation, access frequency, and coactivation cluster assignment. This is a database page directory.

**Sidecar index files (`.vib3.idx`).** The vector index, coactivation table, and materialized view references are decoupled from the data file into a sidecar index file that lives alongside the model: `model.vib3` + `model.vib3.idx`. This follows standard database practice — no production database rewrites its data files to rebuild an index. The sidecar is a 256-byte header (with parent binding via XXH3 checksum, profiling provenance, and section directory) followed by independently addressable sections: centroids, vector index entries, serialized HNSW graph, coactivation entries, materialized view headers, and metadata JSON. For a 160-570GB model file, the sidecar is <1MB (~740KB for Kimi K2.5 with 256 clusters). This means the index can be iteratively rebuilt — from weight statistics, to synthetic router probing, to full calibration — without touching the weight data. The engine discovers the sidecar automatically, validates parent binding, and hot-swaps new indexes via atomic `rename()` + `Arc::swap` with no restart required. See Phase 11 in the Roadmap (Section 12) for the full sidecar format specification, synthetic building pipeline, and auto-tuning loop.

### 2.4 Virtual Experts: Weight-Space Retrieval as the Inference Mechanism

The database analogy leads to a deeper insight when taken to its logical conclusion. The standard MoE inference path is:

```
hidden_state → learned router → softmax/sigmoid → top-k expert IDs → load fixed expert weights → compute
```

This is query planning with a hardcoded optimizer: the router is a small neural network trained jointly with the experts, and it produces a fixed mapping from hidden state to expert ID. The expert boundaries are determined at training time and cannot change.

**Weight-indexed inference replaces the fixed router with vector similarity search over the weight space itself:**

```
hidden_state → HNSW search over page signatures → top-k most relevant weight pages → compute
```

The expert boundary disappears. Instead of selecting "expert 42" and loading all 12 of its pages, the system retrieves the 12 most relevant weight pages from _any_ expert based on their signature vectors. This is **virtual expert assembly** -- composing an ad-hoc expert from the weight pages most relevant to the current hidden state.

**What this enables:**

| Capability                | Fixed Router                                        | Weight-Space Retrieval                                  |
| ------------------------- | --------------------------------------------------- | ------------------------------------------------------- |
| Expert granularity        | Fixed at training time (e.g., 2048×7168 per expert) | Arbitrary -- can retrieve individual 2MB pages          |
| Adding new capability     | Retrain router + new expert                         | Insert new weight pages into HNSW index                 |
| Cross-expert composition  | Not possible -- each expert is monolithic           | Natural -- top-k pages can span multiple experts        |
| Domain-specific filtering | Separate classifier needed                          | Metadata filter on HNSW search: `WHERE domain = 'code'` |
| Scaling expert count      | Router grows O(hidden_dim × num_experts)            | HNSW search is O(log n) regardless of expert count      |

**Implementation path (Phase 11):**

1. **Page signatures.** During `vib3-convert`, compute a representative vector for each 2MB weight page. Three options under evaluation:
   - Mean of weight rows in the page (cheap, coarse)
   - First principal component via truncated SVD (moderate cost, better signal)
   - Activation-based: run calibration data through the model, record which hidden states cause each page to contribute most to the output (expensive, best signal)

2. **Embedded HNSW index.** Replace the brute-force `BruteForceBackend` with an HNSW backend (usearch, embedded in the binary -- no external server). The `AnnBackend` trait already supports this swap with zero changes to prediction or prefetch logic. usearch provides filtered search, quantized vectors (f16, i8), and serialization to/from the `.vib3` file.

3. **Page-level query planner.** Extend `QueryPlanner` with a new execution strategy: instead of `expert_id → pages_for_expert()`, use `hidden_state → hnsw.search_k(hidden_state, top_k_pages) → ResolvedPage list`. The expert boundary becomes optional.

4. **Scatter-gather compute kernel.** The retrieved pages may not be contiguous. A custom CUDA kernel computes partial matmul across non-contiguous pages and accumulates results weighted by relevance score.

**What's hard:**

- **Page signature quality is everything.** Bad signatures → wrong pages retrieved → garbage output. The calibration step determines whether this works.
- **Scatter-gather matmul is less GPU-friendly** than contiguous expert matrices. Non-contiguous pages break memory coalescing.
- **No end-to-end training signal.** The learned router was optimized jointly with the experts. ANN search over page signatures is a proxy for relevance -- plausible but unproven at scale.
- **Validation requires a real model.** This cannot be tested with synthetic weights. Mixtral-8x7B (open weights, Apache 2.0) is the first validation target.

**What exists nearby but doesn't do this:**

- **RAG** retrieves documents, not weights.
- **kNN-LM** (Khandelwal et al.) retrieves token representations from a datastore -- closer, but retrieves outputs, not weights.
- **Product Key Memory** (Lample et al. 2019) uses key-value lookup for memory layers -- architecturally similar but baked into training, not post-hoc.
- **RETRO** retrieves text chunks. Same retrieval machinery, different target.

No existing system treats the weight tensor pages themselves as a retrieval corpus with a production ANN index as the routing mechanism. This is the gap vib3 targets.

**External routing integration.** The filtered search capability (usearch's `filtered_search` with predicate callbacks) enables external routing signals to bias page retrieval at query time. A task-conditioned routing layer — such as Clank's Gearbox, which injects a learned task embedding into MoE router logits — can provide task metadata (e.g., `gear: "code"`) that narrows the HNSW search space to task-relevant pages. This turns unfiltered virtual expert assembly into **task-conditioned virtual expert assembly**: different virtual experts assembled for different task types, per token, without retraining the model or modifying the index. The integration is optional — without external signals, the search operates over the full index as described above. See the [Clank + vib3 Integration Spec](/code/clank-llm/docs/integration-spec.md) for the full design.

---

## 3. Architecture

### 3.1 Six-Layer Stack

```
+------------------------------------------+
|          Python API / CLI (PyO3)         |   User-facing
+------------------------------------------+
|        Runtime / Query Planner           |   Orchestration
+------------------------------------------+
|        Page Buffer Manager               |   Memory management
+------------------------------------------+
|     Storage Engine / io_uring            |   Disk I/O
+------------------------------------------+
|        Predictive Indexing               |   Speculation
+------------------------------------------+
|        CUDA Kernel Launcher              |   Compute
+------------------------------------------+
```

### 3.2 The Three-Tier Storage Hierarchy

```
T1: VRAM (74 GB usable)             T2: RAM (168 GB usable)      T3: NVMe (8+ TB)
+------------------+                +------------------+          +------------------+
| Hot pages:       |  <-- async --> | Warm pages:      | <-- io_  | Cold pages:      |
| lock-free lookup |  DMA+decomp   | COMPRESSED data  |  uring   | all model weights |
| DashMap + atomic |                | pinned host mem  |          | .vib3 file format |
+------------------+                +------------------+          +------------------+
       |                                   |                              |
       | ~1,792 GB/s internal              | ~64 GB/s PCIe 5.0           | ~14-22 GB/s
       |                                   |                              |
  [GPU compute]                    [Blackwell DE 600 GB/s]        [Gen5 + Gen4 NVMe]
  [specialist pins]                [~588 GB effective @ 3.5x]

Pipeline B (default):  T2 compressed → PCIe DMA → VRAM staging → Blackwell DE → T1
Pipeline A (fallback): T2 raw → PCIe DMA → T1 directly (no GPU decompression)
```

**T1 (VRAM):** Lock-free hot path. Page lookup is DashMap (concurrent hashmap) + atomic access tracking. A T1 hit costs one atomic read and a pointer return -- no mutex, no syscall. In specialist mode, hot expert clusters are pinned in T1 (~52-79 GB for 40-60 experts assuming consistent activation across layers -- see caveat in Section 3.6).

**T2 (RAM) — Compressed Mode (Pipeline B, default):** Stores Zstd-compressed pages directly from disk. **No CPU decompression.** T2→T1 promotion sends compressed data over PCIe to a VRAM staging buffer, where the Blackwell Decompression Engine (600 GB/s) decompresses into the final T1 slot. At 3.5x Zstd compression ratio, 168 GB of RAM holds ~588 GB of effective page data — **enough to cover the entire ~570 GB model**. Combined T1 (74 GB raw) + T2 (588 GB effective) exceeds the model size, so in steady state every page is either in T1 or T2. NVMe is only needed for cold start (initial T2 population). In practice, after cold start, NVMe drops out of the steady-state critical path for most tokens, though domain transitions and generalist-mode working set churn can trigger T2 evictions that require NVMe backfill.

**T2 (RAM) — Raw Mode (Pipeline A, fallback):** Pinned host memory with CPU-side Zstd/LZ4 decompression during T3→T2 load. Direct DMA to T1. Used when Blackwell DE is not available.

**T3 (NVMe):** The authoritative store. All pages live here permanently. Reads use `io_uring` with `SQPOLL` for kernel-bypass. Only needed for: initial cold start, and the ~7% of pages that don't fit in compressed T2.

#### Tiered Compression Cascade

Each tier's compression is tuned to its bottleneck. NVMe is bandwidth-starved (14.7 GB/s); maximize compression there. RAM needs GPU-friendly decompression; use moderate compression. VRAM is compute-ready; store raw.

```
T3 (NVMe):  Zstd level 15-19  (~5-6x ratio)   ← maximize effective NVMe bandwidth
    ↓  io_uring read → CPU partial-decompress to Zstd-1 level → store in T2
T2 (RAM):   Zstd level 1-3    (~3-3.5x ratio)  ← GPU-friendly, Blackwell DE compatible
    ↓  PCIe DMA compressed → VRAM staging → Blackwell DE (600 GB/s) → T1
T1 (VRAM):  Raw / decompressed                  ← compute-ready, zero overhead
```

The cascade creates a compression multiplier at each tier boundary:

| Tier Transition  | What Moves    | Compression | Effective Bandwidth                                 |
| ---------------- | ------------- | ----------- | --------------------------------------------------- |
| T3→T2 (NVMe→RAM) | Zstd-19 pages | ~5-6x       | 14.7 GB/s × 5.5 = **~81 GB/s effective**            |
| T2→T1 (RAM→VRAM) | Zstd-1 pages  | ~3.5x       | 64 GB/s PCIe, 600 GB/s DE = **~224 GB/s effective** |

With Zstd-19 on NVMe instead of Zstd-3, the ~570 GB model compresses to ~95-115 GB on disk (vs ~163 GB at 3.5x). Cold start drops from ~11s to **~6-8s**. Cache misses that reach NVMe see ~81 GB/s effective throughput from a single Gen5 drive — competitive with a 6-drive Gen4 RAID array.

The T3→T2 transition uses CPU decompression (Zstd-19 → Zstd-1 recompression, or Zstd-19 → raw if T2 is in raw mode). This is acceptable because T3→T2 is the cold path — it only fires on cold start or T2 misses, not on the hot T2→T1 promotion path. The CPU work is also naturally overlappable with GPU compute on other pages.

**Implementation note:** The `.vib3` file stores pages at the highest compression level. The `PageCatalogEntry.compression` field identifies the algorithm and level. T2 can store pages at a _different_ compression level than on disk — the buffer manager handles recompression during T3→T2 promotion. This is directly analogous to database systems that use different compression for hot (in-memory) vs cold (on-disk) data.

### 3.3 Predictive Indexing

Three subsystems work together to predict what pages will be needed. The vector index is wired into the engine's hot path and queried on every token — like a database index consulted on every query.

**Vector Index (active in hot path):** Maps regions of embedding space to predicted expert activations and specific hot pages. The vector index is used at three points during each token's generation:

1. **Pre-warm** (`predict_and_prewarm`): Before each MoE layer's router runs, the planner queries the vector index with the latest hidden state to predict which pages will be needed, and submits prefetch requests. This is the "index lookup before table scan" — pages start loading before the router even runs.
2. **Cross-layer prefetch** (`submit_cross_layer_prefetch`): After processing each layer, the planner uses the vector index to predict pages for layers 2-3 ahead, not just the immediate next layer. Priority decreases with distance.
3. **Speculative prefetch** (`submit_vector_prefetch`): After each token completes and the trajectory is updated, the vector index uses linear extrapolation of recent hidden states to predict pages for future tokens (3-token lookahead). This overlaps I/O with the next token's compute.

Uses centroid-based nearest-neighbor lookup. The default path is a brute-force O(n) L2 scan with no dynamic dispatch — the compiler inlines it directly. This is the right choice for the expert centroid use case (<10K centroids at 64-256 dims ≈ 2-10μs per search). For larger indexes (>50K centroids, or the KV cache ANN search in Section 8), a custom backend implementing the `AnnBackend` trait (usearch HNSW, Faiss IVF) can be plugged in via `VectorIndex::load_with_backend()` without changing any prediction or prefetch logic. The architecture uses an enum dispatch (`BruteForce` | `Custom(Box<dyn>)`) so the default path pays zero abstraction cost. Mode-aware: in Specialist mode, pages already pinned in T1 are skipped; confidence is scaled by the prefetch multiplier (0.5x specialist, 1.5x generalist).

**Coactivation Graph:** A graph of expert co-activation correlations. If expert A is active, the graph predicts which other experts are likely active at the same layer. Used to expand prefetch beyond the directly predicted set.

**Domain Classifier:** Classifies the current input into workload domains (code, math, creative writing) using cosine similarity against learned domain centroids. Each domain has a recommended materialized view -- a pre-grouped set of pages optimized for that domain's typical expert activation pattern.

**Trajectory Prediction:** The query planner tracks recent hidden states (sliding window of 32) and the vector index uses linear extrapolation to predict where future tokens' embeddings will land. This drives the speculative prefetch at the end of each token generation.

### 3.4 Page-Level Compute Kernels

Standard GEMM libraries (cuBLAS, CUTLASS) expect contiguous weight matrices. vib3's kernels operate on **page-level slices**: a 2 MB page covers a contiguous range of rows from a weight matrix. The partial matmul kernel computes `output[1, M_slice] = input[1, K] x weight[M_slice, K]^T` where `M_slice` is determined by the page's row range.

Supported dtypes: FP16, BF16, INT8, INT4 (with group-wise dequantization), NF4, FP8.

Fused operations: `partial_swiglu` computes `SiLU(input x up_page) * (input x gate_page)` in a single kernel launch, halving memory traffic for the MoE FFN.

**Tiled warp-cooperative GEMV (Phase 10).** The CUDA GEMV kernels use a tiled reduction strategy where 128 threads cooperate on each output row via warp shuffle + shared memory reduction. Each thread computes a partial dot product over K/128 elements, then reduces via `__shfl_down_sync` within warps (4 warps per row) and a shared memory inter-warp reduction. Two rows are computed per 256-thread block. This achieves ~44x speedup over the naive 1-thread-per-row implementation for Mixtral-8x7B INT4 experts (K=4096, M_slice~993 rows per page). All three GEMV variants (FP16, INT4 with group-wise dequant, fused SwiGLU FP16) use this tiling. INT4 SwiGLU is decomposed into two tiled INT4 matmuls + a lightweight `silu_mul` elementwise kernel.

### 3.5 Eviction Policy

Not LRU. Eviction score is:

```
score = predicted_reuse * 0.6 + recency * 0.4
```

Where `predicted_reuse` comes from the query planner's estimate of whether this page will be needed again soon, and `recency` is the normalized tick of last access. Pages with high predicted reuse survive eviction even if they haven't been accessed recently. Pinned pages (shared layers: attention weights, embeddings, router weights) are never evicted.

### 3.6 Dual-Mode Expert Activation: Generalist vs. Specialist

The engine detects and adapts to two distinct workload patterns at runtime:

**Mode A — Generalist:** Surface-level skimming across diverse topics. Expert activations are spread across many experts (200+ per window), producing high Shannon entropy. Working set is large, T1 hit rate is 60-70%, aggressive prediction and wider prefetch are needed. This is the safer default.

**Mode B — Specialist:** Deep domain focus (e.g., multi-file code refactoring, sustained mathematical reasoning). The same ~40-60 experts are activated repeatedly per layer, producing low entropy. Working set estimate: 40-60 experts × 60 MoE layers × 24 MB = ~58-86 GB at INT4 with scales, assuming the _same_ experts are hot across all layers. In practice, different layers have partially overlapping but not identical hot sets; the union of unique experts across layers could be 80-120, yielding 115-173 GB -- exceeding T1's 74 GB budget. The `SpecialistPinSet` budget (70% of T1 = ~52 GB) handles this gracefully by pinning as many as fit and leaving the rest for demand-loading from compressed T2. Strategy: pin the hot expert cluster in T1, suppress eviction, hold steady for maximum throughput.

**Detection mechanism:** The `ActivationModeDetector` measures Shannon entropy of expert activation frequencies over a sliding window of recent tokens (default: 128 tokens). For Kimi K2.5 (384 routed experts):

- Maximum entropy: log2(384) ≈ 8.58 bits (uniform activation)
- Specialist threshold: ~6.0 bits (70% of max, tunable)
- Specialist mode: 4-5 bits (40-60 experts dominate)
- Generalist mode: 7-8 bits (200+ experts active)

The detector uses EMA smoothing (alpha=0.9) to prevent mode oscillation, plus hysteresis (8 consecutive opposite-mode readings required before switching). Mode detection is computed every 16 tokens to amortize cost.

**Specialist pinning:** When the engine enters specialist mode, it identifies the top-K most frequently activated experts via `top_experts()` and pins their pages in T1 via `pin_expert_cluster()`. These pages are tracked separately in a `SpecialistPinSet` so they can be bulk-unpinned when switching back to generalist mode without disturbing shared-layer pins.

**Materialized views** are the pre-computed specialist profiles: a mapping from domain → expert set → page set. A `SpecialistProfile` contains the domain centroid, per-layer hot expert lists, total page count, and VRAM requirement.

---

## 4. What We Built (Phases 1-10)

### Phase 1: Core Types and File Format

- PageId addressing scheme (layer, expert, segment, page_idx)
- Three-tier storage hierarchy types
- `.vib3` binary file format with zero-copy mmap for metadata
- PageCatalogEntry (48 bytes, O(1) lookup)
- Expert index, vector index, coactivation table, materialized views

### Phase 2: Storage Engine

- Three-tier PageBufferManager with DashMap-based T1 lookup
- io_uring integration with SQPOLL for kernel-bypass NVMe reads
- Async DMA (T2->T1) overlapping with GPU compute
- Background prefetch and eviction worker tasks
- Prediction-aware eviction (not LRU)

### Phase 3: Compute and Runtime

- Page-level partial matmul kernels (FP16, BF16, INT4, INT8)
- Fused SwiGLU kernel
- RoPE, RMSNorm, multi-head attention with GQA
- KV cache management
- Query planner: router output -> page fetch plan
- Token generation loop with sampling (temperature, top-k, top-p)
- OpenAI-compatible API server with SSE streaming

### Phase 4: Predictive Indexing

- Vector index with centroid-based nearest-neighbor prediction
- Speculative prefetch with trajectory extrapolation
- Coactivation graph for neighbor expansion
- Domain classifier with materialized view recommendations

### Phase 5: Performance Optimizations

Borrowed conceptually from BitNet.cpp and AirLLM:

1. **Priority prefetch queue** -- BinaryHeap replacing FIFO channel; critical requests dequeued before speculative ones
2. **LZ4 compressed pages** -- 2-3x NVMe read throughput; decompress in T2 (CPU) while GPU computes
3. **LUT-based INT4 CPU kernels** -- eliminate multiply in inner loop; makes CPU fallback viable
4. **Embedding quantization** -- INT8 embedding tables with per-row scales; 50% size reduction
5. **Transport quantization** -- FP16->INT8 for NVMe transfer, dequantize during T2->T1 promotion
6. **Auto-profile hardware** -- benchmark RAM/NVMe/H2D bandwidth at startup; auto-size pools

Phase 5 state: 68 tests passing, zero warnings.

### Phase 6: Model Conversion Pipeline (see Section 12 for details)

The conversion pipeline is the bridge between HuggingFace model checkpoints and vib3's runtime. It implements the full quantization and indexing stack.

**INT4 Quantization Engine:**

- `quantize_weights_to_int4(f32, rows, cols)` -- group-wise quantization (default group_size=32, configurable) with per-group FP16 scales
- Quantization scheme: `q = clamp(round(val / scale) + 8, 0, 15)`, matching the LUT dequant in `cpu_matmul_int4`
- `quantize_fp16_to_int4()` / `quantize_bf16_to_int4()` -- convenience wrappers for direct safetensors dtype handling
- `convert_bf16_to_fp16()` / `convert_f32_to_fp16()` -- dtype conversion for shared layers

**HuggingFace Config Parser:**

- `ModelConfig::from_hf_config(json)` -- parses Mixtral, DeepSeek-V2/V3, Qwen2-MoE, Kimi architecture configs
- Handles MLA fields (`kv_lora_rank`, `q_lora_rank`, etc.), `first_k_dense_replace`, `moe_layer_frequency`
- Auto-detects `config.json` in model directories

**Full Converter (`vib3-convert`):**

- Detects safetensors tensor dtype (FP16/BF16/FP32) and converts appropriately
- Expert weights: quantized to INT4 with group-wise scales (default for real models)
- Shared weights: passed at source precision (BF16 for attention/norms, FP16 for embeddings)
- Page splitting with proper INT4 layout (nibbles + scales interleaved per page)
- `--compress zstd|lz4|none` and `--quantize int4|none` CLI flags
- Progress reporting with source→output size reduction stats

**Zstd Compression:**

- `CompressionMethod::Zstd { level }` added to `Vib3Writer` alongside LZ4
- Reader transparently decompresses Zstd pages via `zstd::bulk::decompress`
- On Blackwell, the nvCOMP pipeline will decompress Zstd at 600 GB/s via the hardware Decompression Engine (Phase 7)

**Expert Activation Profiler:**

- `ActivationProfiler` records expert activations during calibration with reservoir sampling (10K sample cap)
- Per-layer frequency histograms and coactivation pair counting
- `build_vector_index(num_clusters, iterations)` -- k-means clustering on hidden states → `VectorIndexEntry`
- `build_coactivation(min_correlation)` → `CoactivationEntry`

**Shared Layer Handling:**

- `num_segments` in `ExpertIndexEntry` now computed from actual distinct segments
- `pages_for_shared(layer)` and `pages_for_segment(segment)` methods for non-expert page lookup

Current state: **87 tests passing** (82 integration + 5 profiler unit), zero warnings.

### Phase 7: Compressed T2 + Dual-Mode Activation + Specialist Pinning + Engine Integration

Implemented Pipeline B infrastructure, adaptive expert cache management, and runtime engine integration:

- Compressed T2 storage: T2 stores Zstd-compressed pages directly from disk
- T2→T1 compressed promotion path via VRAM staging buffer (32 MB, slot-based pool)
- Dual-mode detection: Shannon entropy over sliding window with EMA smoothing and hysteresis
- Specialist pinning: `pin_expert_cluster()` / `unpin_expert_cluster()` with budget (70% of T1)
- Mode detection wired into engine with automatic Generalist↔Specialist transitions
- Mode-aware QueryPlanner adjusts prefetch aggressiveness per mode (0.5x specialist, 1.5x generalist)
- Vector index wired into three hot-path points: pre-warm, cross-layer prefetch, speculative prefetch
- Pluggable ANN backend via `AnnBackend` trait with enum dispatch (`BruteForce` | `Custom(Box<dyn>)`)

Phase 7 state: 114 tests passing, zero warnings.

### Phase 8: Tiered KV Cache + Unified Eviction

Extended the storage engine to manage KV cache pages alongside weight pages under a unified memory manager:

**8.1 — PageId Extension:** Extended `PageId` with `EXPERT_KV_CACHE = 0xFFFE` sentinel, K/V segment constants, `PageId::kv_cache()` constructor, and predicate methods (`is_kv_cache()`, `is_weight()`, `is_k_page()`, `is_v_page()`). KV page geometry: `head_dim=128 → 512 bytes/position → 4,096 positions per 2 MB page`. Extended `InferenceStats` and `StatsSnapshot` with 8 KV cache counters.

**8.2 — KvCacheConfig:** Configuration struct with T1/T2 position capacities, sparse attention toggle, top-k retrieval count, recent window size, landmark count, and unified pool budget fractions. Integrated into `EngineConfig` with serde defaults.

**8.3 — TieredKvCache:** Per-layer, per-head `HeadCache` with three-tier tracking (T1/T2/T3 `HashSet`s), K and V data vectors, and per-position metadata (tier, attention weight, last access tick). `LandmarkTracker` identifies high-attention positions for pinning. Append, advance, gather, and demote operations. T1→T2 demotion on capacity overflow (oldest non-landmark, non-recent-window positions first). T2→T3 demotion by lowest attention weight. 13 unit tests.

**8.4 — KvIndex:** ANN index over K vectors for one head at one layer. Incremental insert/remove, dot-product similarity search, top-k retrieval. Brute-force O(n) default — production path uses HNSW via the same `AnnBackend` trait used for expert prediction.

**8.5 — Sparse Attention:** `sparse_attention_head()` computes Q·K^T softmax → weighted V sum over a subset of positions retrieved by KvIndex. `multi_head_sparse_attention()` wraps with GQA-aware head mapping. `self_attention_tiered()` implements the full tiered attention layer: Q/K/V projection, RoPE, append to tiered cache, ANN-indexed position gathering, sparse attention over the retrieved subset.

**8.6 — Unified Eviction Policy:** `UnifiedEvictionPolicy` evaluates T1 memory pressure across weights and KV jointly. Four pressure levels (Low/Medium/High/Critical). `EvictionRecommendation` specifies how many weight pages and KV positions to evict, with a 60/40 KV-preferred split (weight misses have higher stall cost). KV budget fraction enforcement prevents either system from starving the other.

**8.7 — Engine Wiring:** The engine optionally creates `TieredKvCache` + `UnifiedEvictionPolicy` from config. `run_attention_layer()` dispatches to tiered or flat attention path. `generate_token()` advances tiered KV position and runs unified eviction checks. `prefill()` clears and advances the tiered cache.

**8.8 — Integration Tests:** 11 new tests covering engine with/without tiered KV, sparse attention kernel, multi-head sparse attention, KV index operations, multi-layer cache, eviction policy, PageId interop, and stats snapshots.

Phase 8 state: 161 tests (30 lib unit + 130 integration + 1 doc), zero warnings.

### Phase 9: MLA Attention + YaRN RoPE + Converter Fixes

Implemented Multi-head Latent Attention (MLA), the attention architecture used by DeepSeek-V3 and Kimi K2.5, replacing the GQA stub that was fundamentally wrong for these models.

**9.1 — MLA Attention (`attention.rs`):** Full MLA forward pass:

- **Q path:** `hidden → q_a_proj [q_lora_rank=1536, hidden_dim=7168] → RMSNorm → q_b_proj [num_heads*(nope+rope)=12288, q_lora_rank] → split(q_nope [128/head], q_rope [64/head]) → YaRN RoPE on q_rope`
- **KV path:** `hidden → kv_a_proj [kv_lora_rank+rope=576, hidden_dim] → split(kv_latent [512], k_rope [64]) → RMSNorm on latent, YaRN RoPE on k_rope → cache latent+rope`
- **Attention:** For each head, reconstruct k_nope/v from cached latent via `kv_b_proj [num_heads*(nope+v)=16384, kv_lora_rank=512]`, compose K=[k_nope, k_rope], standard scaled dot-product attention
- **O projection:** `concat(head_outputs) → o_proj [hidden_dim, num_heads*v_head_dim=8192]`
- Helper functions: `gemv_f16()`, `gemv_f16_slice()` (per-head sliced reconstruction), `rms_norm_f32()`, `rms_norm_f32_with_weight()`

**9.2 — MLA KV Cache:** `MlaKvCache` stores compressed latent vectors (kv_lora_rank=512) + RoPE component (qk_rope_head_dim=64) per position, shared across all heads. Memory comparison at 4K context:

- Standard GQA: 64 heads × 128 dim × 2 (K+V) × 4K × 4 bytes = **256 MB/layer**
- MLA: (512 + 64) × 4K × 4 bytes = **9.2 MB/layer** (~28x smaller)

`MlaKvCacheSet` maintains one `MlaKvCache` per layer. The engine creates this when `model_config.mla` is `Some`.

**9.3 — YaRN RoPE:** Full YaRN (Yet another RoPE scaling) implementation with three-zone frequency scaling:

- High-frequency dimensions (wavelength < original_max_pos/beta_slow): original theta=50000 frequencies, no interpolation
- Low-frequency dimensions (wavelength > original_max_pos/beta_fast): interpolated by factor=64
- Mid-range: smooth linear blend
- Parameters from Kimi K2.5: theta=50000, factor=64, beta_fast=32, beta_slow=1, original_max_pos=4096

This replaces the hardcoded `rope_base=10000` with the model's actual RoPE configuration, enabling correct position encoding at extended context lengths (262K).

**9.4 — Engine Dispatch:** `run_attention_layer()` now dispatches three-way: MLA → Tiered GQA → Flat GQA. `run_mla_attention()` loads segments 20-25 (q_a_proj, q_b_proj, kv_a_proj, kv_b_proj, q_a_layernorm, kv_a_layernorm) plus segment 5 (o_proj).

**9.5 — Converter Fixes:**

- **\_packed/\_scale pairing:** Two-pass conversion — first pass collects all `_scale` tensors into a HashMap, second pass pairs each `_packed` tensor with its corresponding scale. `combine_packed_and_scales()` produces the `[nibbles][scales]` layout expected by our INT4 matmul. Previously `_scale` tensors were silently dropped.
- **MLA norm segments:** Added segment 24 (q_a_layernorm) and segment 25 (kv_a_layernorm) for the RMSNorm weights applied to compressed Q and KV latents.
- **Segment convention now covers 26 segment types** (0-25), fully mapping every tensor in the Kimi K2.5 / DeepSeek-V3 architecture.

**9.6 — Future Optimization: Absorbed Attention.** The current CPU fallback reconstructs per-head k_nope and v from the cached latent at every position via kv_b_proj multiplication. DeepSeek-V2 describes an "absorbed attention" optimization that reformulates the attention computation to avoid this per-position reconstruction, computing attention directly in latent space. This is a GPU kernel optimization (fusing kv_b_proj into the attention kernel) and is not implemented in the CPU fallback. It would reduce MLA attention FLOPs by ~4x at long contexts.

Phase 9 state: 173 tests (30 lib unit + 142 integration + 1 doc), zero warnings. ~16,150 lines lib+tools, ~4,971 lines tests.

### Phase 10: Full GPU Kernel Dispatch + Tiled GEMV Optimization

Moved all inference computation to GPU and optimized the CUDA kernels from naive single-thread-per-row to tiled warp-cooperative reductions.

**10.1 — GPU Attention Projections:** Q/K/V/O projections now execute as GPU GEMV (`linear_projection`) on device weight tensors assembled via D2D copies from preloaded T1 pages. `try_gpu_attention_projection()` is a synchronous method (no await boundaries, avoiding Send issues with raw pointers) that: (a) launches 3 GEMV kernels for Q/K/V, (b) syncs and D2H the small projected vectors (Q=16KB, K=4KB, V=4KB), (c) runs RoPE + KV cache + multi-head attention on CPU (tiny for decode), (d) H2D attention output and launches O projection GEMV, (e) fused residual add on GPU. Falls back to CPU path if device tensors are unavailable.

**10.2 — Shared Tensor Device Cache:** `ensure_shared_tensor_device()` assembles multi-page shared tensors (attention norms, Q/K/V/O weights, router weights, MoE norms) directly on device via D2D copies from T1 page slots, avoiding the VRAM->host->VRAM roundtrip. Cache key is `(layer << 16 | segment)`. After preload, all `ensure` calls are cache hits (<1us).

**10.3 — Model Preloading:** All 13,915 pages (27.8 GB) loaded to T1 VRAM at 1,400 MB/s via mmap + H2D copies at startup (~20s). With full T1 residency, the I/O tier drops out entirely — every page is a T1 hit.

**10.4 — Tiled Warp-Cooperative GEMV Kernels:** Rewrote all three GEMV kernels (`partial_matmul_fp16`, `partial_matmul_int4`, `fused_swiglu_fp16`) from naive 1-thread-per-row to tiled reduction:

- **128 threads per output row** (4 warps cooperating via `__shfl_down_sync`)
- **2 rows per 256-thread block** (`ROWS_PER_BLOCK=2`)
- Each thread computes partial dot product over K/128 elements, then reduces via warp shuffle (intra-warp) + shared memory (inter-warp)
- INT4 kernel: each thread processes pairs of weights at stride, with per-group scale lookup
- Fused SwiGLU: dual accumulators (up + gate) with shared memory storing both partial sums

**10.5 — Measured Performance (Mixtral-8x7B INT4, RTX PRO 6000 Blackwell):**

| Metric                       | Before (naive GEMV) | After (tiled GEMV) | Speedup |
| ---------------------------- | ------------------- | ------------------ | ------- |
| Decode throughput            | 0.5 tok/s           | **3.6 tok/s**      | 7.2x    |
| Time to first token          | 23.8s               | **3.8s**           | 6.3x    |
| MoE per layer                | 48 ms               | 1.1 ms             | 44x     |
| Attention GEMV per layer     | 1.2 ms              | <0.1 ms            | >12x    |
| Full decode step (32 layers) | 2,465 ms            | ~256 ms            | 9.6x    |

Output verified coherent (greedy decode produces "Hello! I'm" for "Hi" prompt).

**10.6 — Profiling Findings:** The original 23ms/layer GEMV measurement was a measurement artifact: `cudaMemcpy` (used for D2D copies) synchronizes only the default stream, not the kernel stream. The GEMV `stream.synchronize()` was draining async MoE work from the previous layer. After isolating with a pre-sync, true GEMV was uniformly 1.2ms/layer (before optimization). The tiled kernels reduced this to <0.1ms/layer.

Phase 10 state: 248 tests (61 unit + 186 integration + 1 doc-test), zero clippy warnings (with CUDA features), zero formatting issues. ~18,900 lines lib+tools, ~5,340 lines tests, ~490 lines CUDA.

### Phase 6b: Validation Scaffolding (Partial)

Built a deterministic validation framework for measuring quantization error and correctness:

- `ReferenceModel`: deterministic weights generated from LCG PRNG seed, f32 forward pass as ground truth
- `compare_outputs()`: produces `ComparisonResult` with MAE, RMSE, max error, cosine similarity
- `QuantizationErrorTracker`: per-layer error accumulation, detects linear vs exponential growth patterns
- 8 unit tests covering determinism, different seeds, forward pass, comparison, and error tracking

This provides the scaffolding for Phase 10 validation: convert a real model, run inference, and compare against a reference implementation token-by-token.

---

## 5. Comparative Analysis

### 5.1 Landscape Overview

| System             | Core Strategy                             | Model Type   | Hardware Target | Speed                | Maturity   |
| ------------------ | ----------------------------------------- | ------------ | --------------- | -------------------- | ---------- |
| **vib3**           | Storage-engine page indexing              | MoE          | 1 GPU + NVMe    | 50+ tok/s (target)   | Prototype  |
| **BitNet.cpp**     | LUT-based ternary kernels                 | 1-bit native | CPU (ARM/x86)   | 5-7 tok/s (100B)     | Production |
| **AirLLM**         | Layer-wise disk streaming                 | Any dense    | Any GPU (4GB+)  | ~minutes/tok         | Production |
| **vLLM**           | PagedAttention + continuous batching      | Any          | Multi-GPU       | High throughput      | Production |
| **llama.cpp**      | Quantized CPU/GPU inference               | Dense        | CPU/GPU         | Good                 | Production |
| **TensorRT-LLM**   | Compiled graph + expert parallelism       | Any          | Multi-GPU       | Highest throughput   | Production |
| **DeepSeek Infra** | Expert parallelism + prefill-decode split | MoE          | Multi-node      | Production scale     | Production |
| **LMCache**        | Tiered KV cache (GPU/CPU/disk)            | Any          | Multi-GPU       | 3-10x TTFT reduction | Production |
| **NVIDIA Dynamo**  | Distributed KV cache management           | Any          | Multi-node      | Massive context      | Production |
| **AWS HyperPod**   | Auto-tiered KV (CPU mem/SSD)              | Any          | AWS managed     | Managed              | Production |

### 5.2 What Each System Optimizes

**vLLM** optimizes **serving throughput** via continuous batching and PagedAttention. It manages KV cache memory efficiently across concurrent requests. It does not optimize weight loading -- it assumes all weights fit in aggregate GPU memory.

**BitNet.cpp** optimizes **compute efficiency** by eliminating floating-point multiply entirely. Weights are ternary {-1, 0, 1}, so matrix multiplication becomes table lookup + accumulation. This is a hardware-level advantage that no software optimization on standard models can match. The limitation is that it requires purpose-trained 1-bit models.

**AirLLM** optimizes **accessibility** -- run any model on any GPU. Layer-by-layer disk streaming means VRAM only needs to hold one layer at a time. The cost is catastrophic latency: each token requires loading every layer from disk sequentially.

**TensorRT-LLM** optimizes **raw speed** via graph compilation, kernel fusion, and multi-GPU parallelism. It's the fastest option when you have the GPUs. It doesn't address the "model doesn't fit on available hardware" problem.

**vib3** optimizes **single-GPU MoE inference** by exploiting sparsity at the storage level. It's the only system that treats expert weights as an indexing/retrieval problem rather than a memory management problem.

**LMCache** optimizes **KV cache reuse across requests** by tiering KV data across GPU, CPU, and disk. It integrates with vLLM and SGLang to share cached context across requests (e.g., common system prompts), reducing TTFT by 3-10x. It solves the KV cache memory wall but does not address weight memory.

**NVIDIA Dynamo** optimizes **massive context windows** by managing KV cache across GPU memory and distributed storage. It handles the data movement scheduling between tiers. Like LMCache, it's focused exclusively on KV cache -- weight management is out of scope.

**AWS SageMaker HyperPod** offers **managed auto-tiered KV caching** from CPU memory to local SSD, abstracting the tiering decisions behind a managed service. Convenient but not customizable and tied to AWS infrastructure.

### 5.3 Fundamental Differences

**vib3 vs vLLM:** vLLM assumes weights fit in memory and optimizes KV cache. vib3 assumes weights don't fit and optimizes weight retrieval. These are complementary -- the ideal system would do both.

**vib3 vs BitNet.cpp:** BitNet changes the model architecture to make compute trivial. vib3 keeps the model architecture and makes storage intelligent. BitNet requires retraining; vib3 works with existing models.

**vib3 vs AirLLM:** Both stream weights from disk. AirLLM streams entire layers sequentially (simple, slow). vib3 streams individual pages predictively (complex, potentially fast). AirLLM's key insight -- compress for bandwidth, not memory -- directly influenced vib3's transport quantization.

**vib3 vs DeepSeek Infra:** DeepSeek distributes experts across many GPUs, each holding a subset of experts in VRAM. vib3 distributes experts across storage tiers on a single machine, paging them into one GPU on demand. DeepSeek is horizontal scaling; vib3 is vertical scaling.

**vib3 vs LMCache/Dynamo/HyperPod:** These systems solve KV cache tiering -- moving KV pairs between GPU, CPU, and disk. They do this well but treat KV cache as the _only_ thing that needs tiered management. They assume weights fit in GPU memory. vib3 solves both problems: weight tiering (Phases 1-7) and KV cache tiering (Phase 8), unified under one storage engine that manages the _global_ memory budget across weights, KV cache, and activations as one resource pool. The `UnifiedEvictionPolicy` makes cross-domain eviction tradeoffs that no existing system can make (Section 8).

### 5.4 What We Borrowed

| From             | Concept                                  | How We Applied It                                                              |
| ---------------- | ---------------------------------------- | ------------------------------------------------------------------------------ |
| BitNet.cpp       | LUT-based low-bit kernels                | LUT inner loop for INT4 CPU matmul -- eliminates per-element multiply          |
| BitNet.cpp       | Auto-tuned tiling                        | Tiled warp-cooperative CUDA GEMV kernels with 128 threads/row (Phase 10)       |
| BitNet.cpp       | Embedding quantization                   | INT8 embedding table with per-row FP16 scales                                  |
| AirLLM           | Compress for bandwidth, not memory       | LZ4 compressed pages on NVMe; transport quantization (FP16->INT8 for transfer) |
| AirLLM           | Graceful hardware degradation            | Auto-profile at startup, adapt pool sizes to available hardware                |
| vLLM             | PagedAttention concept                   | Inspiration for tiered KV cache (Section 8)                                    |
| LMCache/Dynamo   | Tiered KV management across GPU/CPU/disk | Validates the approach; we extend to unified weight+KV tiering                 |
| LMCache          | Cross-request KV sharing                 | Shared context pinning alongside shared weight pages                           |
| MCaM             | Workload-specific multi-tier eviction    | KV eviction needs different heuristics than weight eviction                    |
| Database systems | Buffer pool management                   | Three-tier page buffer with prediction-aware eviction                          |
| Database systems | Query planning                           | Router output -> page-level fetch plan with lookahead                          |
| Database systems | Materialized views                       | Pre-grouped pages for common workload domains                                  |
| Vector databases | ANN indexes                              | Vector index for embedding -> expert prediction; KV retrieval (Section 8.3)    |

---

## 6. The Database Analogy: Where It Holds, Where It Doesn't, and Where It Leads

The database framing is not a metaphor. vib3 is a specialized storage engine: it has a binary file format with a page catalog, an expert index for O(1) page lookup, a three-tier buffer pool with prediction-aware eviction, a query planner that translates router output into page-level fetch plans, materialized views for common workload patterns, and tiered compression across storage tiers. These are not borrowed _ideas_ from databases — they are database storage engine components, implemented.

The right comparison is not to PostgreSQL or MySQL (OLTP systems optimized for unpredictable read-write workloads with ACID guarantees). It is to **OLAP column stores** — systems like DuckDB, ClickHouse, Apache Parquet, and BigQuery that are read-heavy, column-oriented, operate on immutable data with predictable scan patterns, and use tiered compression to maximize I/O throughput. vib3 is closer to a column store than it is to a general-purpose database, and that's not a limitation — column stores are among the most performant storage engines ever built.

That said, the analogy has genuine structural limits and unexplored depth. Both are worth examining.

### 6.1 Structural Limits

**Sequential pipeline, not random queries.** A database query planner chooses between index scan, table scan, hash join, and merge join based on statistics and can reorder operations. A transformer processes layers sequentially, top to bottom, every time. The query planner cannot reorder layers or skip layers. The only degree of freedom is which pages to load within each layer — and even that is determined by the router, which depends on all previous layers.

**Atomic GPU compute, not streaming.** In a database, the storage engine can stream rows into a hash join incrementally. In GPU inference, a partial matmul kernel needs all its weight pages resident in VRAM before launch. Each expert computation is atomic: fetch all pages, compute, accumulate. The prefetch pipeline must be ahead of the compute pipeline by at least one expert — ideally by one full layer.

**Per-token query, not amortized index.** A database index amortizes its construction cost over millions of queries. The vector index in vib3 is queried once per token, 50+ times per second, and the query (the hidden state) is unique every time. The index must be extremely fast or it becomes overhead.

**No transactions.** Databases guarantee ACID properties. None applies to inference: no concurrent writes, no rollbacks, no isolation. Weight data is immutable. This eliminates WAL, MVCC, and lock management — a huge fraction of traditional database complexity.

### 6.2 Where the Analogy Is Genuine

These are not analogies. They are the same techniques applied to a different data domain.

| Database Component                      | vib3 Component                                  | Shared Principle                                      |
| --------------------------------------- | ----------------------------------------------- | ----------------------------------------------------- |
| Page catalog                            | `PageCatalogEntry` (48 bytes, O(1) lookup)      | Fixed-size directory for random page access           |
| Buffer pool manager                     | `PageBufferManager` (DashMap + three-tier)      | Hot/warm/cold tiering with eviction policy            |
| Prediction-aware eviction (2Q/ARC)      | `score = predicted_reuse * 0.6 + recency * 0.4` | Eviction that models future access, not just past     |
| Query planner                           | `QueryPlanner` (router → page fetch plan)       | Translate logical request into physical I/O plan      |
| Materialized views                      | `SpecialistProfile` (domain → expert → page)    | Pre-computed working sets for known workload patterns |
| Tiered compression (Oracle, SQL Server) | Compression cascade (Zstd-19 → Zstd-1 → raw)    | Different compression for hot vs cold storage tiers   |
| Column store page format                | `.vib3` page format (2 MB aligned, typed)       | Columnar layout for vectorized access                 |
| Statistics collector                    | `ActivationProfiler` + `PlannerStats`           | Workload statistics driving optimization decisions    |

The caching and locality patterns are identical in structure. MoE expert activations exhibit temporal locality (same experts fire for related tokens) and workload locality (code prompts activate different experts than math prompts). These are exactly the properties that make buffer pool management effective in databases — and they hold here for the same mathematical reasons.

### 6.3 Unexplored Depth: Where the Analogy Leads

The current query planner has exactly one strategy: prefetch predicted pages and hope. A real query optimizer has multiple execution strategies and selects between them based on cost estimation. vib3 already has the infrastructure for this — the three `PageMode` variants are embryonic execution strategies:

| Execution Strategy                                              | Database Analog                           | When to Use                             | Cost Model                           |
| --------------------------------------------------------------- | ----------------------------------------- | --------------------------------------- | ------------------------------------ |
| **Exact** (load all pages for active experts)                   | Full table scan                           | Low prediction confidence, first tokens | Cost = page_count × tier_latency     |
| **Predictive** (prefetch predicted pages, demand-load misses)   | Index scan with prefetch                  | Steady-state, >70% prediction accuracy  | Cost = miss_rate × miss_latency      |
| **Approximate** (load only top-K highest-activation pages)      | Lossy index scan                          | High system load, latency-sensitive     | Cost = accuracy_loss vs latency_gain |
| **Speculative** (prefetch multiple tokens ahead via trajectory) | Speculative execution / branch prediction | Stable hidden state trajectory          | Cost = wasted_io vs latency_hiding   |

A cost-based planner would maintain running statistics (hit rates, prediction accuracy, tier latencies, attention compute time) and select the strategy that minimizes expected per-token latency at each layer. In specialist mode, Exact is nearly free (everything is pinned). In generalist mode, Predictive with fallback to Exact on miss is optimal. Under memory pressure, Approximate trades accuracy for throughput — structured sparsity via page selection.

**This is genuine query planning, not an analogy.** The optimizer has a cost model, multiple execution strategies, workload statistics, and adaptive strategy selection. The fact that the "query" is a router output and the "table" is a weight matrix does not make it less of a query plan — it makes it a domain-specific query plan, the same way a geospatial database has domain-specific query plans for spatial joins.

**Other database techniques with direct applicability:**

- **Partition pruning.** In specialist mode, the mode detector identifies which "partitions" (expert clusters) are active. The planner skips page lookups for experts outside the active partition, reducing index traffic.
- **Adaptive indexing (database cracking).** Rather than building the vector index once during conversion, refine it online based on observed activation patterns. Centroids that produce poor predictions get split; unused centroids get merged.
- **Query result caching.** If the same expert-at-layer combination produces identical output for similar hidden states (within a quantization bucket), cache the intermediate result, not just the weight pages. This is memoization at the storage engine level.
- **Write-ahead of predictions.** Log predicted activations and compare against actual activations to train the predictor online — analogous to query plan feedback loops in adaptive query processing.

**Indexes as learned parameters.** The deepest extension of the analogy: indexes are derived parameters optimized against an objective function, just like weights. Model weights are optimized by backprop against a loss function. Database indexes are optimized by profiling against a query performance objective. The vib3 vector index sits at the intersection — it's optimized by profiling against a prefetch precision/recall objective, where `precision@k` measures "of the k experts the index predicted, how many did the router actually select?" The optimization method differs (profiling + hyperparameter search vs. gradient descent), but the lifecycle is identical: build → measure → tune → persist → iterate. The sidecar index format (`.vib3.idx`, Phase 11) makes this iteration loop practical: each index version is a <1MB file write instead of a 500GB model rewrite. The index progresses from weight statistics (seconds, zero inference cost) through synthetic router probing (minutes, ~0.01% cost) to full calibration (minutes, 10-20% cost) to continuous online improvement during serving. This is the same progression a production database follows: create a naive index at table load time, gather statistics under real workload, rebuild with better statistics, and continue adapting. The difference is that the "table" is 570GB of quantized neural network weights and the "queries" are hidden state vectors.

---

## 7. Failure Modes and Honest Risks

### 7.1 The KV Cache Is the Real Memory Wall

This is the biggest blind spot in the current design.

The entire three-tier storage architecture is designed for expert weight pages. But for long-context inference, the KV cache can rival the model weights in size. For Kimi K2.5 at full 262K context:

```
Naive KV cache (no MLA, standard multi-head attention):
  = 2 (K+V) × 61 layers × 64 kv_heads × 112 head_dim × 262,144 positions × 2 bytes/BF16
  = 2 × 61 × 64 × 112 × 262,144 × 2
  ~ 460 GB

With MLA (kv_lora_rank=512, compressing K+V into a single latent per layer):
  = 61 layers × 512 latent_dim × 262,144 positions × 2 bytes/BF16
  ~ 16 GB

With MLA at FP32 accumulation (common in practice):
  = 61 layers × 512 latent_dim × 262,144 positions × 4 bytes/f32
  ~ 32 GB (still larger than what VRAM can spare after weight pages)
```

MLA compresses the KV cache by ~14x vs naive (460 GB → 32 GB at FP32). Even so, 32 GB at full context is a significant fraction of the 96 GB VRAM budget -- VRAM that's also needed for weight pages, activations, and compute scratch. The current MLA implementation (`attention.rs:MlaKvCache`) stores compressed latent vectors (kv_lora_rank=512) + RoPE component (64) per position, achieving ~28x memory reduction vs per-head GQA storage. For non-MLA models, the flat `KvCache` stores per-head K/V as unbounded `Vec<Vec<f32>>` in host RAM.

Every production serving system (vLLM, TensorRT-LLM, SGLang) now spends most of its engineering budget on KV cache management. **Section 8 describes the implemented solution** — a tiered KV cache with ANN-indexed retrieval, unified with weight page management under a single eviction policy.

### 7.2 Prediction Accuracy May Be Insufficient

The vector index predicts expert activations from the embedding vector. But expert activations at layer 30 are a function of the hidden state AFTER processing layers 1-29 -- attention, MoE routing, residual connections, and normalization at every layer. Predicting layer 30 from the input embedding is like predicting the end of a chess game from the opening move.

The vector index is now wired into the hot path (Section 3.3) with three integration points: per-layer pre-warming, cross-layer prefetch, and post-token speculative prefetch. This makes the prediction accuracy question empirically testable. If prediction accuracy is below ~70%, the prefetch pipeline becomes net-negative:

- Wasted I/O bandwidth loading pages that aren't used
- Eviction of useful pages to make room for mispredicted ones
- Additional CPU overhead for vector index lookup and prefetch scheduling

The speculative prefetch uses linear extrapolation of hidden states (`vector_index.rs:320-334`). Transformer hidden states do not evolve linearly -- attention creates sharp, context-dependent discontinuities. The extrapolation will be noisy. However, even noisy predictions with 50% accuracy are valuable if the per-prediction cost (a centroid distance computation) is cheap relative to the I/O latency saved by a correct prediction.

**Mitigation:** The query planner tracks prediction precision and recall (`PlannerStats`). If accuracy drops below a threshold, the system could fall back to reactive loading (no speculation) or switch to `PageMode::Exact` which loads all pages for activated experts without prediction. The cross-layer prefetch deliberately uses lower priorities for further-ahead layers, so mispredictions for layer N+3 are cheaper than mispredictions for layer N+1.

### 7.3 Single-Stream, Single-User Architecture

The engine owns a single `hidden_state` buffer, a single position counter, and a single `KvCacheSet`. There's no request batching. The buffer pool manages pages for one inference stream.

For serving (the OpenAI-compatible API), concurrent requests would thrash the buffer pool -- each request activates different experts, evicting pages the other requests need. Production MoE serving (DeepSeek, Mixtral at scale) batches tokens across requests specifically to amortize expert loading: if 8 requests all need expert 42 at layer 5, you load it once and compute 8 tokens against it.

**This is likely the hardest engineering problem on the roadmap.** Multi-request scheduling that maximizes page reuse across concurrent requests is essentially a new kind of query scheduler.

### 7.4 Attention Dominates Latency for Long Contexts

The flat attention implementation is O(n^2) naive dot-product attention on CPU. The tiered KV cache (Section 8) provides sparse attention via ANN-indexed retrieval, reducing effective complexity to O(k) per head where k is the retrieval count. However, without Flash Attention CUDA kernels, even O(k) attention on CPU is slow. For a 262K context window with flat attention, the computation dominates any NVMe stall.

At 262K context with 64 heads and 112 head_dim, each attention layer computes:

```
Q * K^T: 2 × 64 heads × 112 head_dim × 262,144 positions = ~3.8 TFLOPs per layer (GEMV per position)
Full attention per layer: 2 × 64 × 262,144² × 112 / 1e12 ≈ ~985 TFLOPs per layer
Across 61 layers: ~60 PFLOPs per token
```

Note: With MLA (now implemented — Phase 9), attention operates on compressed latents (kv_lora_rank=512) which changes the compute profile significantly. The Q·K dot product is split: q_nope·k_nope (128 dims, reconstructed from latent) + q_rope·k_rope (64 dims, cached directly). The CPU fallback reconstructs k_nope/v per-position via kv_b_proj multiplication; the planned absorbed attention optimization (fusing kv_b_proj into the attention kernel) would eliminate this overhead on GPU.

On CPU, this takes minutes, not seconds. Even on an RTX PRO 6000 at ~1,000 TOPS INT4 (or ~250 TFLOPS at FP16), a single 262K-context token's naive attention takes ~30+ seconds at FP16. The expert page management becomes irrelevant if attention is the bottleneck. (Note: at practical context lengths of 4K-16K, attention is manageable -- Section 11.7 shows ~12-45 ms with Flash Attention.)

**Mitigation:** Flash attention on GPU is a known solution. The architecture supports it -- attention weight pages are managed by the same buffer pool. But it requires CUDA kernel implementation that doesn't exist yet.

### 7.5 MoE Correctness at Scale: The L6 Explosion

**Mixtral-8x7B is converted and producing coherent output at 3.6 tok/s** (Phase 10). The conversion pipeline and inference engine are validated end-to-end on this model.

**Kimi K2.5 (1T, 61 layers, 384 experts) is partially validated but has a blocking bug.** Layer-by-layer comparison against PyTorch BF16 ground truth on the unquantized 480GB `.vib3` conversion shows:

- **L0-L5:** Cosine similarity degrades gradually from 0.991 (L0) to 0.917 (L5). Per-layer delta ~0.01-0.03. The signal is intact but error accumulates. L3 shows the first L2-norm undershoot (vib3 is 0.94x ground truth) — the MoE experts are undershooting the reference at this layer.
- **L6:** Catastrophic failure. Cosine drops to -0.092. L2 norm explodes from 2.85 (GT) to 91.8 (vib3) — a 32x blowup. The explosion occurs in the MoE expert computation, not attention: the L6 prenorm (after attention, before MoE) has cosine 0.804 with signal still present; the MoE sublayer transforms this into anti-correlated garbage.

This is a bug, not a quantization issue (the model is unquantized BF16→FP16 — no INT4/NVFP4 in this conversion). Possible causes under investigation: wrong expert selection due to accumulated router input error, expert weight page corruption during buffer pool management, or a shared expert computation bug. Extensive per-expert diagnostic tracing is in place (VRAM-vs-disk comparison, per-expert accumulation logging, SwiGLU intermediate dumps). **This must be fixed before any performance claims on Kimi K2.5 are meaningful.**

The per-layer cosine degradation rate (~0.02-0.03/layer) is also a concern. At this rate, after 61 layers the cosine would be ~0.16 — not usable. The rate likely improves once the L6 bug is fixed (it may be poisoning later layers' KV caches), but even the L0-L5 pattern suggests precision work is needed — possibly higher-precision residual accumulation in the attention path, or BF16 weights instead of FP16 truncation during conversion.

### 7.6 Expert Activation Locality May Vary by Model

The design assumes expert activations exhibit temporal locality (same experts fire for related tokens) and domain locality (code prompts fire different experts than math). Research on MoE models shows this varies significantly:

- DeepSeek-V2 reports high expert locality within domains
- Mixtral-8x7B shows more uniform activation patterns
- Switch Transformer shows high specialization but not necessarily temporal locality

If the target model has low expert locality, the buffer pool's caching strategy degrades to near-random page replacement.

### 7.7 Tiered KV Cache Risks

Section 8 implements unified weight and KV cache management. The existing tiered KV cache ecosystem (LMCache, Dynamo, HyperPod, MCaM) has identified real challenges that vib3 inherits:

**Transfer latency vs compute overlap.** Moving KV pairs from CPU/NVMe back to GPU takes time. For weight pages, this latency can be hidden behind current-layer compute because expert activations are predicted ahead of time. For KV pairs, the "prediction" is the Q vector -- which isn't available until Q projection completes within the current layer. This means KV retrieval is inherently more latency-sensitive than weight prefetch. The ANN index lookup must be fast enough (~100us) to not add to the critical path.

**Privacy and multi-tenant isolation.** LMCache enables cross-request KV sharing (e.g., shared system prompts). This is a major throughput optimization but creates information leakage risk between tenants. vib3's unified buffer pool would need tenant-tagged pages with isolation enforcement, adding complexity to the eviction and sharing logic. (Not yet implemented -- the current `TieredKvCache` is single-request.)

**Eviction policy complexity.** The weight eviction score (`predicted_reuse * 0.6 + recency * 0.4`) was designed for weight pages where predicted reuse comes from the vector index. For KV pages, "predicted reuse" is harder to define: a KV pair from position 50 might be critical for every future attention computation, or it might be effectively ignored. The implemented `UnifiedEvictionPolicy` uses a pragmatic heuristic: recent-window positions and landmarks are protected; demotion is by age (T1→T2) and attention weight (T2→T3). This is simpler than learning-based policies (MCaM) but correct by construction -- empirical validation will determine if more sophistication is needed.

**Approximate attention accuracy.** Replacing full attention with ANN-retrieved sparse attention introduces approximation error. This is acceptable for long-context scenarios where most KV pairs contribute negligibly, but the accuracy/speed tradeoff must be validated per-model. Some attention heads are "retrieval heads" (sparse, position-dependent) while others are "summary heads" (dense, using many positions). A blanket top-k retrieval may degrade summary heads.

---

## 8. The Unified Tiered Cache

### 8.0 Prior Art: Tiered KV Caching Is Not New

We must be direct: tiered KV caching is an established technique with production implementations. Several systems already move KV pairs between GPU HBM, CPU DRAM, and NVMe/networked storage:

- **LMCache:** Open-source, integrates with vLLM and SGLang. Manages KV cache across GPU, CPU, and disk with cross-request sharing. Demonstrated 3-10x TTFT reduction on previously-seen contexts.
- **NVIDIA Dynamo:** Production KV cache manager for massive context windows. Moves data between GPU memory and distributed storage with intelligent scheduling.
- **AWS SageMaker HyperPod:** Managed auto-tiered KV caching from CPU memory to local SSD.
- **MCaM:** Multi-tier cache system designed for virtual assistant and multi-turn conversation scenarios.

These systems have identified and solved the KV cache memory wall. They work. The known challenges are real:

- **Transfer latency:** Moving KV pairs from CPU/disk back to GPU can negate the benefit if retrieval isn't ahead of compute.
- **Privacy/isolation:** Sharing KV caches across users/tenants risks information leakage. Multi-tenant serving needs isolation boundaries.
- **Eviction strategy:** Deciding which KV pairs to keep on GPU vs offload requires workload-aware policies. Simple LRU is insufficient; attention patterns are not recency-ordered.

**What none of these systems do is manage KV cache and model weights under the same storage engine.** They all assume weights are resident in GPU memory. For dense models on multi-GPU setups, this is reasonable. For MoE models that don't fit in VRAM, it isn't. vib3's contribution is not tiered KV caching per se -- that's solved -- but **unifying weight page management and KV page management in a single tiered storage engine** that makes global resource allocation decisions across both. This is now implemented.

### 8.1 Attention IS Retrieval

The core attention operation is:

```
scores = Q * K^T          -- find K vectors most similar to Q
weights = softmax(scores)  -- normalize similarity scores
output = weights * V       -- retrieve V values weighted by similarity
```

This is literally a vector similarity search followed by weighted retrieval. It is exactly what vector databases (Qdrant, Milvus, Faiss, usearch) do at their core. The only difference is that attention computes similarity against ALL stored vectors, while a vector database uses an approximate index to retrieve the top-k most relevant ones.

For short contexts, exhaustive attention is fine. For 262K contexts, the vast majority of KV pairs receive near-zero attention weight. They're loaded into VRAM, multiplied, and the results are thrown away. This is a full table scan when an index lookup would suffice.

### 8.2 Tiered KV Cache Architecture

The three-tier storage hierarchy now manages KV cache pages alongside weight pages:

```
                     Weight Pages                    KV Cache Pages
                    +-------------+                 +------------------+
T1 (VRAM):          | Hot experts  |                 | Recent window    |
                    | (predicted)  |                 | + landmarks      |
                    +-------------+                 +------------------+
                          |                               |
T2 (RAM):           | Warm experts |                 | Indexed KV       |
                    | (prefetched) |                 | (ANN over K vecs)|
                    +-------------+                 +------------------+
                          |                               |
T3 (NVMe):          | All experts  |                 | Cold KV          |
                    | (.vib3 file) |                 | (long context)   |
                    +-------------+                 +------------------+
```

**T1 KV Cache (implemented):** Three zones managed by `TieredKvCache`. (1) **Recent window** -- the last N positions (configurable, default 2048), always resident, never demoted. (2) **Landmarks** -- positions with the highest aggregate attention weight, tracked by `LandmarkTracker`, pinned in T1 regardless of age. (3) **ANN-retrieved** -- positions promoted on demand when the `KvIndex` identifies them as relevant to the current query. Configurable capacity via `KvCacheConfig.t1_positions`.

**T2 KV Cache (implemented):** Positions demoted from T1 when capacity is exceeded. Demotion order: oldest non-landmark, non-recent-window positions first. The `KvIndex` (brute-force dot-product search, upgradeable to HNSW via `AnnBackend` trait) indexes K vectors in T2, enabling O(k) retrieval instead of O(n) exhaustive scan. Configurable capacity via `KvCacheConfig.t2_positions`.

**T3 KV Cache (implemented):** Positions demoted from T2 by lowest attention weight. These are the coldest KV pairs -- positions that received negligible attention in recent tokens. T3 positions can be promoted back to T1 if the KvIndex identifies them as relevant.

### 8.3 Embedded Vector Index: usearch as the Routing Engine

Rather than depending on an external vector database server (Qdrant, Milvus), vib3 embeds the vector index directly into the binary via **usearch** -- a single-file C++ HNSW library with first-class Rust bindings (Apache 2.0). This eliminates network round-trips, deployment complexity, and serialization overhead. The index lives in-process, compiled into the same binary that runs inference.

**Why usearch, not Qdrant/Milvus/Faiss:**

| Criterion           | External DB (Qdrant)                 | Embedded (usearch)                                      |
| ------------------- | ------------------------------------ | ------------------------------------------------------- |
| Deployment          | Separate server process              | Single binary, zero config                              |
| Latency             | ~100-500μs (network + serialization) | ~2-10μs (in-process function call)                      |
| Filtered search     | Full payload filtering               | Predicate callbacks (`filtered_search`)                 |
| Persistence         | Managed by server                    | `save_to_buffer` / `load_from_buffer` into `.vib3` file |
| Vector quantization | Server-side                          | f16, i8 scalar quantization built-in                    |
| Thread safety       | Server handles concurrency           | `Send + Sync`, lockless reads                           |

**Three integration points using the existing `AnnBackend` trait:**

**1. Expert prediction (replacing brute-force).** The `HnswBackend` implements `AnnBackend` and plugs into `VectorIndex` via `SearchBackend::Custom`. For the expert centroid use case (<10K centroids), this is a marginal improvement over brute-force. The real value emerges at page-level indexing (Section 2.4): with ~45,000 weight pages for Kimi K2.5, HNSW reduces search from O(n) to O(log n) per query.

**2. KV cache retrieval at scale.** The `KvIndex` (`tiered_kv.rs`) supports incremental insert, remove, and top-k dot-product search over K vectors. It is wired into `self_attention_tiered()`: the attention layer queries KvIndex with the current Q vector to select which positions to attend to, then runs sparse attention over only the selected subset. The brute-force implementation is correct but O(n); the `HnswBackend` provides the upgrade path for 100K+ context without changing the attention pipeline.

**3. Page-level virtual expert routing (Section 2.4).** The most novel application: index individual weight pages by their signature vectors, then retrieve the top-k most relevant pages per hidden state. usearch's `filtered_search` enables domain-aware retrieval: "find nearest weight pages WHERE layer == 5 AND domain == 'code'" -- replacing the separate `DomainClassifier` with a metadata predicate on the same search.

**4. Coactivation as graph search.** The coactivation graph is currently a hand-rolled adjacency list. A graph-aware index could answer richer queries: "given experts {3, 17, 42} are active at layer 5, which experts are likely active at layers 6, 7, and 8?" Cross-layer coactivation prediction would significantly improve prefetch accuracy.

**Persistence:** The HNSW index is serialized via `save_to_buffer` and stored as a section in the `.vib3` file, loaded via `load_from_buffer` at model open time. The index can also be mmapped via `view_from_buffer` for zero-copy access, consistent with vib3's zero-copy design.

### 8.4 The Unified Page Manager

The key architectural insight: weight pages and KV cache pages share the same lifecycle management. Both need:

- Three-tier storage with promotion/demotion
- Predictive prefetch (weights: predict from embedding; KV: predict from Q similarity)
- Eviction based on predicted reuse
- Async DMA to overlap I/O with compute

`PageId` now addresses KV cache pages natively:

```rust
// Weight page
PageId { layer: 5, expert: 42, segment: 0 (up_proj), page_idx: 3 }

// KV cache page (implemented)
PageId { layer: 5, expert: 0xFFFE (KV_CACHE), segment: 0 (K), page_idx: 1024 }
```

The `EXPERT_KV_CACHE = 0xFFFE` sentinel distinguishes KV pages from weight pages (`expert < 0xFFFE`) and shared layers (`expert = 0xFFFF`). `PageId::kv_cache(layer, segment, block_idx)` constructs KV page IDs. Predicate methods `is_kv_cache()`, `is_weight()`, `is_k_page()`, `is_v_page()` enable type-safe dispatch. KV page geometry: at `head_dim=128`, each position occupies 512 bytes (128 dims x 4 bytes), so a 2 MB page holds 4,096 positions.

**What's novel here is not the tiering -- LMCache, Dynamo, and HyperPod already tier KV cache effectively.** The novel contribution is the **unified resource manager** that governs both weight pages and KV pages in a single system, enabling cross-domain tradeoffs:

- **Global eviction (implemented):** The `UnifiedEvictionPolicy` evaluates T1 memory pressure across both weight pages and KV positions, producing an `EvictionRecommendation` that specifies how many of each to evict. The policy prefers evicting KV over weights at a 60/40 ratio when both are cold, because weight cache misses have higher stall cost (NVMe round-trip vs KV recomputation that may be masked by sparse attention). Four pressure levels (Low/Medium/High/Critical) drive increasingly aggressive eviction.
- **Joint capacity planning (implemented):** `KvCacheConfig.t1_kv_fraction` and `t2_kv_fraction` set the budget split between weights and KV at startup. The `UnifiedEvictionPolicy` enforces these fractions, preventing either system from starving the other. Default: 15% of T1 and 10% of T2 reserved for KV.
- **Shared I/O scheduling:** Weight prefetch and KV prefetch compete for the same NVMe bandwidth. A unified scheduler can prioritize based on which stall (weight miss vs KV miss) is more expensive for the current token.
- **Coordinated prediction:** The vector index that predicts expert activations also knows the workload domain. KV eviction can be domain-aware: "this is a code completion task, keep KV for recent code context, aggressively evict KV from the earlier natural-language preamble."

This matters specifically for MoE models where VRAM is scarce. On a 96 GB GPU running a ~570 GB model, VRAM is a zero-sum game between weight pages, KV cache, and compute scratch. A unified manager prevents the pathological case where two independent systems each try to use 60 GB of VRAM.

### 8.5 Attention as a Query Plan

With tiered KV cache, the attention computation at each layer becomes a query plan. This is now implemented in `self_attention_tiered()`:

```
1. Q, K, V = project(hidden_state)             -- compute projections
2. Apply RoPE to Q and K                        -- positional encoding
3. cache.append_layer(layer, &k, &v)            -- store K,V in tiered cache
4. positions = kv_index.search(Q, top_k)        -- ANN search over K vectors
5. positions += recent_window + landmarks        -- always include these
6. K_subset, V_subset = cache.gather(positions)  -- fetch from whichever tier
7. output = sparse_attention(Q, K_subset, V_subset) -- attend only to retrieved subset
```

This transforms O(n) attention into O(k) retrieval + O(k) attention, where k is the number of retrieved positions (e.g., 512 out of 262,144). The accuracy loss from approximate retrieval is bounded by the ANN index's recall. The engine dispatches between this tiered path and the standard flat attention path based on `KvCacheConfig.enabled`.

Research validating retrieval-based attention:

- **InfLLM** (2024): Offloads distant KV to CPU, retrieves relevant blocks
- **Quest** (2024): Query-aware KV cache selection
- **MemoryFormer** (2024): External memory with learned retrieval
- **Landmark Attention** (2023): Selected landmark tokens for long-range retrieval

Production systems validating tiered KV management:

- **LMCache** (2024): GPU/CPU/disk KV tiering with cross-request sharing, 3-10x TTFT reduction
- **NVIDIA Dynamo** (2025): Distributed KV cache for massive context
- **AWS SageMaker HyperPod** (2025): Managed auto-tiered KV caching
- **MCaM** (2024): Multi-tier cache for multi-turn virtual assistant workloads

The research papers implement retrieval as a one-off attention optimization. The production systems implement tiering as a one-off KV memory optimization. Neither combines tiered KV with tiered weights under one storage engine, and neither uses ANN-indexed retrieval within the tiered architecture. vib3's contribution is the integration: retrieval-based attention as a natural consequence of the unified storage engine, not a separate system bolted on.

### 8.6 Implementation Details

The tiered KV cache is implemented across ~2,200 lines of new Rust code with 42 tests:

| Component                                                           | File                        | Lines | Tests |
| ------------------------------------------------------------------- | --------------------------- | ----- | ----- |
| `TieredKvCache`, `HeadCache`, `LandmarkTracker`, `PositionMeta`     | `runtime/tiered_kv.rs`      | ~700  | 13    |
| `KvIndex` (ANN search over K vectors)                               | `runtime/tiered_kv.rs`      | ~150  | 4     |
| `UnifiedEvictionPolicy`, `EvictionRecommendation`, `MemoryPressure` | `runtime/tiered_kv.rs`      | ~200  | 4     |
| `KvCacheConfig`                                                     | `core/config.rs`            | ~60   | -     |
| `PageId` KV extensions, KV geometry constants, KV stats             | `core/types.rs`             | ~120  | 10    |
| `sparse_attention_head()`, `multi_head_sparse_attention()`          | `compute/kernels.rs`        | ~150  | 2     |
| `self_attention_tiered()`                                           | `runtime/attention.rs`      | ~100  | -     |
| Engine wiring (tiered dispatch, eviction, advance)                  | `runtime/engine.rs`         | ~80   | 2     |
| Integration tests (Phase 8)                                         | `tests/integration_test.rs` | ~350  | 11    |

**Dual attention path:** The engine's `run_attention_layer()` dispatches between `self_attention_layer()` (flat KV cache, default) and `self_attention_tiered()` (tiered KV cache, opt-in via `KvCacheConfig.enabled`). This is a clean separation: existing workloads are unaffected, and tiered KV can be enabled per-model or per-request.

**What needs empirical validation:**

- Sparse attention accuracy vs full attention across different model architectures
- KvIndex retrieval quality at scale (brute-force is correct but O(n); HNSW upgrade needed for >16K context)
- Optimal T1/T2 KV budget fractions for different context lengths
- Landmark detection quality — whether aggregate attention weight correctly identifies important positions

---

## 9. What We Borrowed and Why

### 9.1 From BitNet.cpp

**LUT-based INT4 kernels:** BitNet's T-MAC methodology replaces multiply-accumulate with table-lookup-accumulate. For ternary weights (3 values), the LUT has 3 entries. We extended this to INT4 (16 entries per nibble). The LUT is precomputed per input element: `LUT[w] = input_val * dequant(w)`. The inner loop becomes `acc += LUT[packed & 0x0F]` -- no multiply. This makes the CPU fallback path viable for testing and for machines without GPUs.

**Embedding quantization:** BitNet's `--quant-embd` flag compresses embedding tables to FP16. We went further with INT8 + per-row FP16 scales, halving the embedding footprint while maintaining accuracy through per-row calibration.

**Configurable tiling:** BitNet's parallel kernel implementations with configurable tiling achieve 1.15-2.1x additional speedup. This informed the design of the tiled warp-cooperative CUDA GEMV kernels implemented in Phase 10: 128 threads per output row (4 warps cooperating via `__shfl_down_sync`), 2 rows per 256-thread block, achieving 44x speedup over the naive 1-thread-per-row implementation on Mixtral-8x7B INT4 experts.

### 9.2 From AirLLM

**Compress for bandwidth, not memory:** AirLLM's core insight is that when disk I/O is the bottleneck, compressing weights reduces loading time even if the compute operates on the decompressed data. We applied this at two levels:

- LZ4 compressed pages on NVMe (lossless, ~2-3x compression on INT4 weight data)
- Transport quantization: FP16 -> INT8 for NVMe transfer, dequantize during T2->T1 promotion (lossy but bounded error, halves transfer size)

**Hardware adaptation:** AirLLM runs on 4GB GPUs. This inspired the auto-profile hardware system: benchmark RAM, NVMe, and H2D bandwidth at startup, then auto-size tier capacities and prefetch depth based on measured performance.

### 9.3 From vLLM

**PagedAttention concept:** vLLM's insight that KV cache can be managed in non-contiguous pages (like virtual memory) directly inspired the tiered KV cache proposal. vLLM pages KV cache within VRAM; we propose paging it across VRAM, RAM, and NVMe.

### 9.4 From the Tiered KV Cache Ecosystem (LMCache, Dynamo, HyperPod, MCaM)

**Three-tier KV management is validated.** These production systems prove that tiering KV cache across GPU/CPU/disk works and delivers real benefits (3-10x TTFT reduction, massive context support). We don't need to re-prove this.

**Cross-request KV sharing.** LMCache's ability to share cached context (e.g., system prompts) across requests is directly applicable. In vib3's unified model, shared KV pages would be pinned in T1/T2 alongside shared weight pages (attention weights, embeddings, router weights), all managed by the same pinning logic.

**Eviction challenges are real.** MCaM's workload-specific multi-tier eviction and Dynamo's intelligent scheduling confirm that simple LRU is insufficient for KV tiering. vib3's prediction-aware eviction needs to handle KV pages differently than weight pages -- attention patterns don't have the same predictability as expert activation patterns.

**Privacy isolation.** LMCache's cross-request sharing creates information leakage risk. Any shared KV implementation needs tenant isolation, which adds a dimension to page identity (`PageId` would need a tenant/request field for KV pages).

### 9.5 From Database Systems

**Buffer pool management:** PostgreSQL, MySQL/InnoDB, and SQLite all manage page caches with eviction policies tuned to access patterns. vib3's prediction-aware eviction (score = predicted_reuse _ 0.6 + recency _ 0.4) is a simplified version of 2Q or ARC policies used in production databases.

**Query planning:** The concept of translating a high-level query (router output) into a physical execution plan (page fetch schedule) comes directly from database query compilation. The planner's stats tracking (precision, recall) mirrors database plan quality monitoring.

**Materialized views:** Pre-computed page groupings for common workloads are directly analogous to database materialized views that pre-join frequently-accessed tables.

### 9.6 From Vector Databases

**ANN indexing:** The proposal to use HNSW indexes for both expert prediction and KV cache retrieval comes from the vector database ecosystem (Qdrant, Faiss, Milvus). These systems have battle-tested implementations of incremental index construction, filtered search, and quantized vector storage.

---

## 10. Validation Plan

Before building more features, the core assumptions must be empirically validated. The following experiments are ordered by criticality.

### 10.1 Expert Activation Predictability

**Question:** Can expert activations be predicted from the input embedding with >70% accuracy?

**Method:**

1. Run Mixtral-8x7B (small enough to run on available hardware) on diverse prompts
2. Record (hidden_state_at_layer_0, expert_activations_at_each_layer) pairs
3. Build vector index from recorded data (centroid clustering)
4. Measure prediction precision and recall at each layer depth
5. Measure decay: how much does accuracy drop from layer 1 to layer 32?

**Success criteria:** >70% recall at layer depth 1-4, >50% at deeper layers. If accuracy at deeper layers is low, prefetch strategy should shift from embedding-based prediction to layer-by-layer reactive prefetch.

### 10.2 NVMe Page Read Latency vs Compute Time

**Question:** Can NVMe page reads be hidden behind GPU compute?

**Method:**

1. Benchmark 2 MB page read latency at queue depths 1, 4, 8, 16, 32 via io_uring on Gen5 NVMe
2. Measure GPU compute time for one expert (partial matmul + SwiGLU + accumulate) at various hidden_dim/expert_hidden_dim
3. Compare: if page_read_latency < compute_time, prefetch can be fully hidden

**Success criteria:** At queue depth >= 8, page read latency < expert compute time for Kimi K2.5 dimensions on Blackwell.

### 10.3 Buffer Pool Hit Rate Under Realistic Workloads

**Question:** What T1 hit rate does the buffer pool achieve with realistic token sequences?

**Method:**

1. Record expert activation traces from real conversations (multi-turn, mixed domains)
2. Simulate the buffer pool with realistic T1/T2 sizes
3. Measure T1 hit rate, T2 hit rate, T3 stall rate
4. Compare prediction-aware eviction vs LRU vs random eviction

**Success criteria:** T1 hit rate > 60% for single-domain conversations, combined T1+T2 hit rate > 85%.

### 10.4 Prediction Overhead vs Benefit

**Question:** Does the prefetch pipeline's overhead (vector index lookup + I/O scheduling + wasted reads) outweigh its benefit (reduced stalls)?

**Method:**

1. Run end-to-end inference with prefetch enabled vs disabled
2. Measure tokens/second, stall count, wasted I/O bytes
3. Sweep prediction confidence threshold to find optimal operating point

**Success criteria:** Prefetch-enabled achieves >1.5x tokens/second vs prefetch-disabled.

### 10.5 KV Cache Size at Target Context Lengths

**Question:** Does the KV cache fit in RAM at target context lengths with MLA?

**Method:**

1. Calculate exact KV cache size for Kimi K2.5 at 4K, 16K, 64K, 262K context
2. Account for MLA compression (kv_lora_rank=512 compresses KV to ~0.5 KB per position per layer)
3. Expected sizes: 4K = ~0.5 GB, 16K = ~2 GB, 64K = ~8 GB, 262K = ~32 GB
4. Compare to available RAM after T2 pool allocation (256 GB - T2 budget)
5. Determine at what context length KV cache + T2 pool exceeds RAM

**Success criteria:** KV cache fits in remaining RAM (after T2 allocation) at 64K context. At 262K, tiered KV cache (Section 8) is required.

### 10.6 End-to-End on a Real Model

**Question:** Does the complete pipeline (convert -> load -> prefill -> generate) work on a real MoE model?

**Method:**

1. Build safetensors-to-vib3 converter for Mixtral-8x7B
2. Convert, including building vector index and coactivation graph from profiling
3. Run generation and measure token quality (perplexity) vs reference implementation
4. Measure tokens/second vs llama.cpp on same hardware

**Success criteria:** Identical output tokens to reference. Tokens/second within 2x of llama.cpp for Mixtral on single GPU (note: Mixtral is small enough to fit in VRAM, so vib3's advantage only shows on larger models).

---

## 11. Reference System Architecture

This section provides the concrete hardware target and performance projections for the first deployment.

### 11.1 Target Hardware

| Tier       | Hardware                        | Specs                                                                           | Bandwidth                      |
| ---------- | ------------------------------- | ------------------------------------------------------------------------------- | ------------------------------ |
| T1 (VRAM)  | NVIDIA RTX PRO 6000 (Blackwell) | 96 GB GDDR7, 512-bit, 24,064 CUDA cores, 752 5th-gen tensor cores, 4 PFLOPS FP4 | 1,792 GB/s internal            |
| T2 (RAM)   | System DDR5                     | 180 GB                                                                          | ~64 GB/s to GPU (PCIe 5.0 x16) |
| T3a (NVMe) | Samsung 9100 Pro (Gen5)         | PCIe 5.0 x4, Samsung Presto controller, 236L V8 TLC                             | 14,700 MB/s seq read           |
| T3b (NVMe) | Samsung 990 Pro (Gen4)          | PCIe 4.0 x4                                                                     | ~7,450 MB/s seq read           |
| GPU TDP    |                                 | 600W (workstation edition)                                                      |                                |
| PCIe       |                                 | Gen 5 x16 = 64 GB/s bidirectional                                               |                                |

### 11.2 VRAM Budget Breakdown

```
Total VRAM:                                    96 GB
─ Shared weights (pinned, never evicted):
    Embeddings:     163,840 vocab × 7,168 dim × 2B (BF16)  =  2.19 GB
    LM head:        163,840 × 7,168 × 2B                   =  2.19 GB
    Router weights: 384 × 7,168 × 60 layers × 2B           =  0.33 GB
    Attention (MLA): 61 layers, BF16 projections            = ~11.50 GB
      q_a_proj: 61 × 7168 × 1536 × 2B                     =  1.30 GB
      q_b_proj: 61 × 1536 × (128+64)×64 × 2B              =  1.15 GB
      kv_a_proj_with_mqa: 61 × 7168 × (512+64) × 2B       =  0.50 GB
      kv_b_proj: 61 × 512 × (128+128)×64 × 2B             =  0.97 GB
      o_proj: 61 × 128×64 × 7168 × 2B                     =  7.16 GB
      (+ nope/rope components, layernorms)                  =  0.42 GB
    Layer norms:                                            =  0.01 GB
─ Compute scratch (activations, buffers):                   = ~3.00 GB
─ KV cache (at 16K context, MLA):
    61 layers × 16K × 512 latent × 4B                      = ~2.00 GB
─ nvCOMP decompression staging:                             = ~1.00 GB
                                                            ─────────
Pinned + scratch + KV:                                     ~22.22 GB
Available for expert page cache (T1):                      ~74 GB
```

At ~24 MB per expert, T1 holds ~3,100 expert instances (13% of 23,040 total). But only 480 are active per token. If the working set has locality (~60-80 unique experts firing repeatedly, consistent across layers), those experts across 60 layers = 3,600-4,800 instances at 86-115 GB — exceeding T1 even in the best case. However, the specialist pin budget (70% of T1 ≈ 52 GB) handles this by pinning as many as fit (~2,150 expert instances, or ~36 unique experts across all layers) and leaving the rest for demand-loading from compressed T2 at ~5 ms per miss. In practice, different layers have partially overlapping but not identical hot sets; the union of unique experts across layers is typically 100-200.

### 11.3 RAM Budget Breakdown

```
Total RAM:                                     180 GB
─ OS + system overhead:                        ~6 GB
─ T2 pinned page pool:                         ~168 GB
─ KV cache overflow (if >16K context):         ~6 GB reserve
                                               ────────
T2 page slots (raw mode):        168 GB / 2 MB = 84,000 pages = 29% of model
T2 effective (compressed, 3.5x): 168 GB × 3.5 = 588 GB = 103% of model (full coverage)
Combined T1 + T2 (raw mode):     74 + 168 = 242 GB = 42% of model
Combined T1 + T2 (compressed):   74 + 588 effective = ~116% coverage (headroom)
```

### 11.4 Per-Token Data Requirements

```
Expert size (INT4 + group-32 scales):
  up_proj:  2048 × 7168 at 4-bit + scales  ≈  8.1 MB
  gate_proj: 2048 × 7168 at 4-bit + scales ≈  8.1 MB
  down_proj: 7168 × 2048 at 4-bit + scales ≈  8.1 MB
  Total per expert:                         ≈ 24 MB
Pages per expert:    24 MB / 2 MB                                = 12 pages
Active per token:    8 experts × 60 MoE layers                   = 480 expert activations
Pages per token:     480 × 12                                    = 5,760 pages
Raw bytes per token: 5,760 × 2 MB                                = 11.5 GB
```

### 11.5 Compression Strategy: nvCOMP Is the Game-Changer

LZ4 provides ~2-2.5x compression at ~5 GB/s per CPU core decompression. But the RTX PRO 6000 (Blackwell) has a **hardware Decompression Engine (DE)** accessible through NVIDIA's nvCOMP library. Starting with nvCOMP 4.2, Blackwell's DE achieves up to **600 GB/s decompression throughput** with fused copy-decompress operations and overlap of decompress with compute.

This fundamentally changes the I/O pipeline architecture. There are three possible pipelines:

**Pipeline A: CPU Decompress (current, no nvCOMP)**

```
NVMe → [compressed] → CPU RAM (decompress on CPU @ ~5 GB/s/core) → [raw] → DMA to VRAM
Bottleneck: CPU decompression OR PCIe (carries raw data)
Effective throughput: min(NVMe × ratio, CPU_decomp, PCIe)
```

**Pipeline B: GPU Decompress via nvCOMP (Phase 7 target)**

```
NVMe → [compressed] → DMA compressed to VRAM staging → Blackwell DE @ 600 GB/s → VRAM compute
Bottleneck: NVMe (carries compressed data) or PCIe (carries compressed data)
Effective throughput: min(NVMe × ratio, PCIe × ratio, GPU_decomp)
```

**Pipeline C: T2 as Decompressed Cache (hybrid)**

```
First access:  NVMe → [compressed] → CPU RAM → decompress → store raw in T2
T2→T1 promo:  T2 [raw] → DMA to VRAM @ PCIe speed (no decompress needed)
Effective throughput for T2 hits: PCIe (64 GB/s raw)
```

**Pipeline B is the critical insight.** By sending compressed data over PCIe, we multiply the effective bandwidth of every link in the chain:

| Link                    | Raw Bandwidth   | With Zstd 3.5x          | With ANS ~4x            |
| ----------------------- | --------------- | ----------------------- | ----------------------- |
| Samsung 9100 Pro (Gen5) | 14.7 GB/s       | **51.5 GB/s effective** | **58.8 GB/s effective** |
| PCIe 5.0 x16            | 64 GB/s         | **224 GB/s effective**  | **256 GB/s effective**  |
| Blackwell DE decompress | 600 GB/s output | N/A (not bottleneck)    | N/A                     |

The NVMe bandwidth multiplier is the most impactful: a single Samsung 9100 Pro at 14.7 GB/s with Zstd 3.5x compression delivers **51.5 GB/s of raw weight data equivalent** through Pipeline B. This is competitive with a 4-drive Gen4 NVMe RAID array -- from a single drive.

**Quantitative pipeline comparison for a 90% hit-rate token (1.15 GB raw miss):**

| Pipeline                      | NVMe Read                 | Decompress                | PCIe Transfer            | Total Latency | tok/s Limit |
| ----------------------------- | ------------------------- | ------------------------- | ------------------------ | ------------- | ----------- |
| A-1c: CPU LZ4, 1 core (2.5x)  | 460 MB @ 14.7 = 31 ms     | 1.15 GB @ 5 GB/s = 230 ms | 1.15 GB @ 64 = 18 ms     | ~230 ms       | ~4          |
| A-4c: CPU LZ4, 4 cores (2.5x) | 31 ms                     | 1.15 GB @ 20 GB/s = 58 ms | 18 ms                    | ~58 ms        | ~17         |
| B: GPU Zstd (3.5x)            | 329 MB @ 14.7 = **22 ms** | 329 MB @ 600 = **0.5 ms** | 329 MB @ 64 = **5.1 ms** | **~22 ms**    | **~45**     |
| B: GPU ANS (4x)               | 288 MB @ 14.7 = 19.5 ms   | 288 MB @ 600 = 0.5 ms     | 288 MB @ 64 = 4.5 ms     | ~19.5 ms      | ~51         |
| C: T2 hit (raw)               | N/A                       | N/A                       | 1.15 GB @ 64 = 18 ms     | ~18 ms        | ~56         |

**Pipeline B delivers a ~10x improvement over Pipeline A-1c (single-core), and ~2.6x over Pipeline A-4c (4-core).** The bottleneck shifts cleanly to NVMe sequential read -- exactly where it should be, because that's the link with the most room for hardware improvement (Gen6 NVMe, multi-drive striping).

The GPU decompression step (0.5 ms) is negligible -- it's effectively free compared to the NVMe latency. This means higher compression ratios are pure win: every additional bit of compression translates directly into reduced NVMe read time with no decompression penalty.

**Phase 6 status:** Zstd compression is now implemented in the writer and reader. The `.vib3` file format supports `COMPRESSION_ZSTD = 2`. The current reader uses CPU-side `zstd::bulk::decompress` (Pipeline A/C). Phase 7 will add the nvCOMP GPU decompress path (Pipeline B).

**Tiered compression approach (updated):**

| Algorithm        | Ratio on INT4 | CPU Decompress   | GPU Decompress (Blackwell DE) | When to Use                                      |
| ---------------- | ------------- | ---------------- | ----------------------------- | ------------------------------------------------ |
| None             | 1x            | N/A              | N/A                           | Small pages (<64 bytes), already-quantized       |
| LZ4              | ~2-2.5x       | ~5 GB/s/core     | ~600 GB/s (nvCOMP)            | T2→T1 promotions (fast CPU path)                 |
| Zstd (level 1-3) | ~3-4x         | ~1.5-2 GB/s/core | ~600 GB/s (nvCOMP)            | **Default on-disk format** for T3→T1 via GPU     |
| ANS (nvCOMP)     | ~3-5x         | N/A (GPU only)   | ~600 GB/s (device API)        | Future: INT4 pages with non-uniform distribution |

**Recommended on-disk format (updated with compression cascade):**

- Expert pages on disk: Zstd level 15-19 (~5-6x ratio on INT4 data, maximize NVMe throughput)
- Shared pages on disk: Zstd level 15-19 (BF16 attention weights compress ~3-4x at high levels)
- T2 stores pages at Zstd level 1-3 (~3-3.5x, GPU-friendly for Blackwell DE)
- `PageCatalogEntry.compression` field: 0=none, 1=LZ4, 2=Zstd (implemented), 3=ANS (future)
- T3→T2 promotion: CPU decompress Zstd-19 → recompress Zstd-1 (or store raw in T2 raw mode)
- T2→T1 promotion: DMA compressed page to VRAM staging → nvCOMP Blackwell DE decompress
- T3→T1 fast path: DMA Zstd-19 to VRAM → Blackwell DE decompress (if nvCOMP supports high levels)

**Compression cascade math for the target system:**

| Metric                           | Zstd-3 only (current)        | Cascade: Zstd-19 disk / Zstd-1 T2 |
| -------------------------------- | ---------------------------- | --------------------------------- |
| On-disk model size               | ~163 GB                      | **~95-115 GB**                    |
| Cold start (single 9100 Pro)     | 163 GB / 14.7 = ~11s         | 95 GB / 14.7 = **~6.5s**          |
| NVMe effective throughput        | 14.7 × 3.5 = 51.5 GB/s       | 14.7 × 5.5 = **~81 GB/s**         |
| T2 effective capacity            | 168 GB × 3.5 = 588 GB (103%) | 168 GB × 3.5 = 588 GB (103%)      |
| Cache miss NVMe penalty (1 page) | 0.57 MB / 14.7 = 39 us       | 0.36 MB / 14.7 = **25 us**        |

The cascade does not change T2 effective capacity (T2 still stores Zstd-1 at ~3.5x). What it changes is NVMe performance: cold start, T2 miss latency, and the cost of backfilling T2 from disk. The CPU work for T3→T2 recompression (decompress Zstd-19, recompress Zstd-1) is ~1-2 GB/s per core — acceptable because T3→T2 is the cold path, not the hot path, and can be parallelized across cores and overlapped with GPU compute.

### 11.6 NVMe Drive Strategy

**Primary (Samsung 9100 Pro):** Store the .vib3 model file. At Zstd 3.5x compression, the ~570 GB model compresses to ~163 GB. Fits comfortably on a 1TB or 2TB 9100.

**Secondary (Samsung 990 Pro):** OS, workspace, KV cache overflow to disk, model conversion scratch. Not on the critical I/O path during inference.

If higher throughput is needed, the io_engine can stripe reads across both drives:

- `BufferPoolConfig.nvme_paths: vec!["/mnt/9100/model.vib3", "/mnt/990/model.vib3"]`
- io_engine dispatches reads round-robin or load-balanced
- Aggregate: ~22 GB/s sequential read

For v1, single-drive (9100 only) is simpler and sufficient.

### 11.7 Performance Projections

**Compute time per token (expert MoE):**

```
Per expert: 3 matmuls (up/gate/down) x hidden_dim(7168) x expert_intermediate(2048) x 2 FLOPs
  = 3 x 7168 x 2048 x 2 = ~88 MFLOPs per expert
480 active experts per token: 480 x 88 MFLOPs = ~42 GFLOPs
RTX PRO 6000 at INT4 tensor cores (~1000 TOPS): 42 GFLOPs / 1000 TOPS = 0.04 ms
```

Expert compute is effectively free at tensor-core throughput. **This is entirely I/O-bound.**

**Measured validation (Phase 10, Mixtral-8x7B INT4):** On the smaller Mixtral model (2 experts/layer x 32 layers, hidden_dim=4096), measured MoE compute is 1.1 ms/layer using tiled warp-cooperative GEMV (scalar CUDA cores, not tensor cores). The 0.04 ms projection above assumes tensor-core-accelerated INT4 WMMA/MMA instructions on Kimi K2.5's 480 active experts. Tensor core GEMV is a Phase 10 roadmap item (Section 12). The measured Mixtral result (35.2 ms total MoE across 32 layers, ~256 ms full decode step, 3.6 tok/s) validates that expert compute is fast relative to the full pipeline, with the remaining ~6.7 ms/layer overhead dominated by router dispatch, D2D synchronous `cudaMemcpy`, and query planner overhead rather than GEMV compute itself.

**NVMe read time per token -- comparing pipelines:**

The raw data miss per token at various hit rates (11.5 GB total per token):

| T1+T2 Hit Rate | Raw Miss | Pipeline A-4c (CPU LZ4 2.5x, 4 cores) | Pipeline B (GPU Zstd 3.5x) | Pipeline C (T2 raw hit)     |
| -------------- | -------- | ------------------------------------- | -------------------------- | --------------------------- |
| 80%            | 2.30 GB  | 920 MB NVMe + 115 ms CPU = ~62 ms     | 657 MB NVMe = **45 ms**    | N/A (all misses go to NVMe) |
| 85%            | 1.73 GB  | 690 MB NVMe + 86 ms CPU = ~47 ms      | 493 MB NVMe = **34 ms**    | N/A                         |
| 90%            | 1.15 GB  | 460 MB NVMe + 58 ms CPU = ~31 ms      | 329 MB NVMe = **22 ms**    | N/A                         |
| 95%            | 575 MB   | 230 MB NVMe + 29 ms CPU = ~16 ms      | 164 MB NVMe = **11 ms**    | N/A                         |

Pipeline A-4c uses 4 CPU cores for LZ4 decompression (~20 GB/s aggregate). With 1 core (~5 GB/s) the same decompress steps take 4x longer (see Section 11.5 table). Pipeline B sends compressed data directly to VRAM -- the Blackwell DE decompression (0.3-1.0 ms) is negligible and fully overlapped with the next NVMe read. Pipeline B's bottleneck is purely NVMe sequential read bandwidth.

**I/O-limited tok/s ceiling (Pipeline B, GPU Zstd 3.5x, single Samsung 9100):**

| T1+T2 Hit Rate | NVMe time | GPU decompress | tok/s (I/O only) |
| -------------- | --------- | -------------- | ---------------- |
| 80%            | 45 ms     | 1.1 ms         | ~22              |
| 85%            | 34 ms     | 0.8 ms         | ~29              |
| 90%            | 22 ms     | 0.5 ms         | **~45**          |
| 95%            | 11 ms     | 0.3 ms         | **~91**          |

**Attention time per token (with Flash Attention on Blackwell tensor cores):**

```
Standard attention:
  At 4K context:   ~0.1-0.3 ms/layer x 61 layers = ~6-18 ms
  At 16K context:  ~0.5-1 ms/layer x 61 layers   = ~30-60 ms
  At 64K context:  ~2-4 ms/layer x 61 layers      = ~120-240 ms

MLA-optimized Flash Attention (KV compressed by ~3x):
  At 4K context:   ~2-6 ms total
  At 16K context:  ~10-20 ms total
  At 64K context:  ~40-80 ms total
```

**Projected end-to-end tok/s (single user, Pipeline B with nvCOMP):**

```
4K context, 90% hit, standard attn:    1000 / max(22, 12) = ~45 tok/s
4K context, 95% hit, standard attn:    1000 / max(11, 12) = ~83 tok/s
16K context, 90% hit, standard attn:   1000 / max(22, 45) = ~22 tok/s
16K context, 90% hit, MLA Flash Attn:  1000 / max(22, 15) = ~45 tok/s
16K context, 95% hit, MLA Flash Attn:  1000 / max(11, 15) = ~67 tok/s
64K context, 90% hit, MLA Flash Attn:  1000 / max(22, 60) = ~17 tok/s
64K context, 95% hit, MLA Flash Attn:  1000 / max(11, 60) = ~17 tok/s
```

**Without nvCOMP (Pipeline A, CPU LZ4, current implementation):**

```
4K context, 90% hit, standard attn:    1000 / max(31, 12) = ~32 tok/s
16K context, 90% hit, standard attn:   1000 / max(31, 45) = ~22 tok/s
16K context, 90% hit, MLA Flash Attn:  1000 / max(31, 15) = ~32 tok/s
```

**nvCOMP delivers a ~1.4x improvement at 90% hit rate (32→45 tok/s at 4K) and ~1.4x at 16K with MLA (32→45 tok/s).** The improvement is larger at lower hit rates where more pages come from NVMe: at 80% hit rate, Pipeline B gives ~22 tok/s vs Pipeline A's ~16 tok/s (1.4x). At 95% hit rate, the I/O pipeline is so fast that attention dominates regardless of pipeline choice.

**Key insights:**

1. **At short contexts (4K), ~45 tok/s is achievable at 90% hit rate with Pipeline B (nvCOMP + Zstd).** At 95% hit rate, we reach 83 tok/s. Without nvCOMP, we fall to ~32 tok/s. The 50 tok/s target requires either 93%+ hit rate with Pipeline B, or ANS compression at 4x+.

2. **At 16K+ context, attention dominates.** MLA-optimized Flash Attention is non-negotiable. With MLA Flash Attention reducing attention to ~15 ms at 16K, the I/O pipeline becomes the bottleneck again -- and nvCOMP makes the difference between 32 and 45 tok/s.

3. **The crossover point is ~90% hit rate.** Below this, I/O dominates and nvCOMP is critical. Above this, the working set fits in T1+T2 and the I/O pipeline matters less. Achieving 90% hit rate requires good expert activation locality -- the validation experiment in Section 10.3.

4. **nvCOMP's value is highest with a single NVMe drive.** With a 4-drive Gen5 RAID array (~56 GB/s), Pipeline A with LZ4 would achieve ~56 × 2.5 / 1.15 = ~122 tok/s I/O ceiling at 90% hit rate -- nvCOMP becomes less critical. But a single-drive setup is the practical target for a workstation, and there nvCOMP is the difference between "works" and "doesn't."

5. **Compression ratio directly translates to tok/s** in Pipeline B because decompression is free (600 GB/s). Zstd 3.5x → 45 tok/s. If ANS achieves 4.5x on INT4 data → 58 tok/s. There's a direct incentive to maximize compression ratio with zero speed penalty.

### 11.8 What Must Change in the Codebase

1. ~~**Zstd compression in Vib3Writer**~~ -- **Done (Phase 6).** `CompressionMethod::Zstd { level }` implemented, reader handles transparent Zstd decompression.
2. ~~**INT4 quantization pipeline**~~ -- **Done (Phase 6).** `quantize_weights_to_int4` with configurable group size (default 32 for Kimi K2.5), integrated into `vib3-convert`.
3. ~~**HuggingFace config.json parser**~~ -- **Done (Phase 6).** `ModelConfig::from_hf_config()` supports Mixtral, DeepSeek, Qwen2-MoE.
4. ~~**Activation profiler**~~ -- **Done (Phase 6).** K-means clustering, coactivation counting, vector index building.
5. **nvCOMP integration** -- Add nvCOMP as a build dependency. Implement GPU-side decompression in the T3→T1 path (Pipeline B). This is the highest-priority remaining optimization -- it's the difference between ~36 and ~50 tok/s.
6. **Flash Attention CUDA kernels** -- Non-negotiable. The CPU attention path cannot work at any useful context length. Integrate FlashAttention-3 or implement MLA-aware fused attention. Required for 50 tok/s at 16K+ context.
7. ~~**Decompression staging buffer**~~ -- **Done (Phase 7).** `VramStagingBuffer` implemented in buffer_manager.rs. Configurable size (default 32 MB, ~16 pages in flight). Slot-based pool with acquire/release for concurrent decompress operations.
8. **io_engine multi-drive support** -- `BufferPoolConfig.nvme_paths` already accepts `Vec<String>`. Wire it through to the io_engine with per-drive fd management.
9. **Auto-profiler hardware detection** -- Detect the Blackwell DE capability, available VRAM, NVMe device count and bandwidth. Set tier sizes automatically.
10. **PCIe 5.0 x16 DMA optimization** -- The card supports 64 GB/s. Ensure DMA transfers use optimal chunk sizes and overlap with compute via separate CUDA streams.
11. ~~**Compressed T2 storage**~~ -- **Done (Phase 7).** `t2_compressed` flag in `BufferPoolConfig`. T2 stores Zstd-compressed pages directly from disk. T2→T1 promotion: compressed DMA to VRAM staging → GPU decompress (CPU fallback until nvCOMP integrated). `read_page_compressed_sync()` added to `Vib3File`.
12. ~~**Dual-mode expert activation**~~ -- **Done (Phase 7).** `ActivationMode` enum, `ActivationModeDetector` with Shannon entropy over sliding window, EMA smoothing, hysteresis. `ActivationModeConfig` with tunable parameters.
13. ~~**Specialist expert pinning**~~ -- **Done (Phase 7).** `pin_expert_cluster()` loads and pins entire expert working sets in T1. `unpin_expert_cluster()` bulk-unpins on mode transition. `SpecialistPinSet` tracks pinned pages separately from shared-layer pins. Pin budget defaults to 70% of T1 capacity.
14. ~~**Wire mode detection into runtime engine**~~ -- **Done (Phase 7).** `ActivationModeDetector` initialized in `Engine::new()` from `ActivationModeConfig`. Expert IDs collected across all MoE layers per token, fed to `record()`. Detection runs every `detect_interval` tokens. Automatic transitions: Generalist→Specialist calls `pin_expert_cluster()` with top-K hot experts; Specialist→Generalist calls `unpin_expert_cluster()`. `QueryPlanner::set_mode()` adjusts prefetch aggressiveness per mode.
15. ~~**MLA attention for Kimi K2.5 / DeepSeek-V3**~~ -- **Done (Phase 9).** `MlaKvCache` stores compressed latent+RoPE per position (~28x memory reduction). Full MLA forward pass: Q path (q_a_proj → RMSNorm → q_b_proj → split/YaRN-RoPE), KV path (kv_a_proj → split → RMSNorm/YaRN-RoPE → cache), per-head attention with kv_b_proj reconstruction, O projection. Engine dispatches MLA → Tiered GQA → Flat GQA.
16. ~~**YaRN RoPE scaling**~~ -- **Done (Phase 9).** Three-zone frequency scaling (theta=50000, factor=64) replacing hardcoded rope_base=10000.
17. ~~**Converter \_packed/\_scale pairing**~~ -- **Done (Phase 9).** Two-pass conversion pairs compressed-tensors `_packed` with `_scale` for correct INT4 group-wise dequantization. MLA norm segments (24, 25) added.
18. **Absorbed attention** -- Future GPU optimization: fuse kv_b_proj into the attention kernel to avoid per-position reconstruction. ~4x FLOPs reduction at long contexts.

---

## 12. Roadmap

_Target system: RTX PRO 6000 (96 GB) + 180 GB RAM + Samsung 9100 Pro + Samsung 990 Pro_

### Phase 6: Model Conversion Pipeline — COMPLETE

**Goal:** Build the full pipeline from HuggingFace safetensors to `.vib3` format.

**Delivered:**

- ✓ INT4 quantization engine (configurable group size, default 32 for Kimi K2.5, with per-group FP16 scales)
- ✓ BF16→FP16 and FP32→FP16 dtype conversion
- ✓ HuggingFace `config.json` auto-detection (Mixtral, DeepSeek-V2/V3, Qwen2-MoE, Kimi)
- ✓ Full `vib3-convert` with safetensors dtype detection, INT4 quantization, Zstd compression
- ✓ Zstd compression in `Vib3Writer` (alongside LZ4)
- ✓ Expert activation profiler with k-means vector index builder and coactivation graph
- ✓ Proper shared layer handling (num_segments, shared page lookup)
- ✓ 87 tests passing (82 integration + 5 unit), zero warnings (now 102 after Phase 7)

**Not yet done (deferred to Phase 6b):**

- Convert a real model (Mixtral-8x7B) and validate inference output against reference
- Profile expert activations on a calibration dataset
- Measure quantization error accumulation across layers
- `.vib3.idx` sidecar writer: serialize profiler output (centroids, vector index entries, HNSW graph, coactivation, metadata) into the sidecar format instead of embedding in the `.vib3` file (see Phase 11)

### Phase 7: Compressed T2 + Dual-Mode Activation + Specialist Pinning + Engine Integration — COMPLETE

**Goal:** Implement Pipeline B infrastructure, adaptive expert cache management, and wire mode detection into the runtime engine.

**Delivered:**

- ✓ Compressed T2 storage: `t2_compressed` in `BufferPoolConfig`, T2 stores Zstd pages directly
- ✓ T2→T1 compressed promotion path: DMA to VRAM staging → GPU decompress (CPU fallback)
- ✓ VRAM staging buffer: `VramStagingBuffer` with slot-based pool (default 32 MB)
- ✓ `read_page_compressed_sync()` for raw compressed reads from disk
- ✓ Dual-mode detection: `ActivationMode` enum, `ActivationModeDetector` with Shannon entropy
- ✓ EMA smoothing (alpha=0.9), hysteresis (8 consecutive), configurable threshold
- ✓ Specialist pinning: `pin_expert_cluster()`, `unpin_expert_cluster()`, `SpecialistPinSet`
- ✓ `SpecialistProfile` type for materialized expert working sets
- ✓ `ActivationModeConfig` with tunable parameters in `EngineConfig`
- ✓ `DecompressFailed` error variant, `memcpy_d2h` sync path
- ✓ **Mode detection wired into Engine**: `ActivationModeDetector` initialized from config, expert IDs collected across all MoE layers per token, fed to `record()`, `detect()` called every N tokens
- ✓ **Automatic mode transitions**: Generalist→Specialist triggers `pin_expert_cluster()` with top-K hot experts; Specialist→Generalist triggers `unpin_expert_cluster()`
- ✓ **Mode-aware QueryPlanner**: `set_mode()` adjusts `prefetch_multiplier` (0.5x specialist, 1.5x generalist), `submit_lookahead()` skips pinned pages and adjusts priority
- ✓ **Engine accessors**: `current_mode()`, `mode_detector()` for external inspection
- ✓ Whitepaper v0.5: fixed Pipeline A table labeling, attention FLOPs formula, T2 coverage claims, specialist VRAM math caveats
- ✓ 102 tests passing (98 prior + 4 new), zero warnings

**Not yet done (deferred to Phase 7b — CUDA-dependent):**

- nvCOMP integration (GPU-side Zstd decompression via Blackwell DE)
- Flash Attention CUDA kernels (FlashAttention-3 or MLA-aware fused attention)
- ~~Real partial matmul CUDA kernels for INT4 (tensor core-accelerated)~~ — **Done (Phase 10).** Tiled warp-cooperative GEMV with 128 threads/row, 44x speedup over naive for INT4 MoE experts.
- ~~Fused SwiGLU on GPU~~ — **Done (Phase 10).** Decomposed INT4 SwiGLU (2x tiled INT4 matmul + silu_mul) and tiled FP16 fused SwiGLU.
- Benchmark: nvCOMP decompress throughput on RTX PRO 6000 Blackwell DE
- ~~Benchmark: measure actual compute time per expert, per attention layer~~ — **Done (Phase 10).** MoE expert: 1.1ms/layer (2 experts), attention projection: <0.1ms/layer on Mixtral-8x7B.
- Validate that NVMe reads can be hidden behind GPU compute

### Phase 8: Tiered KV Cache + Unified Eviction — COMPLETE

**Goal:** Extend the storage engine to manage KV cache pages alongside weight pages under a unified memory manager.

**Delivered:**

- ✓ `PageId` extended with `EXPERT_KV_CACHE = 0xFFFE`, K/V segment constants, KV geometry (4,096 positions per 2 MB page)
- ✓ `KvCacheConfig` with T1/T2 capacities, sparse attention toggle, top-k, recent window, landmarks, unified pool fractions
- ✓ `TieredKvCache` with per-layer per-head `HeadCache`, three-tier tracking, `LandmarkTracker`, T1→T2→T3 demotion
- ✓ `KvIndex` with incremental insert/remove, brute-force dot-product search, `AnnBackend` upgrade path
- ✓ `sparse_attention_head()` and `multi_head_sparse_attention()` kernels with GQA support
- ✓ `self_attention_tiered()` — full tiered attention layer (project → RoPE → append → ANN gather → sparse attend)
- ✓ `UnifiedEvictionPolicy` with four pressure levels, 60/40 KV-preferred eviction split, budget fraction enforcement
- ✓ Engine wiring: dual attention path dispatch, tiered KV advance/clear in generate/prefill, eviction checks
- ✓ `InferenceStats` extended with 8 KV cache counters
- ✓ 42 new tests (13 unit + 11 integration + 18 across PageId, kernels, attention)
- ✓ 161 tests total (after Phase 6b validation tests + regression tests), zero warnings

**Not yet done (deferred to empirical validation):**

- Benchmark sparse attention accuracy vs full attention (perplexity delta)
- HNSW backend integration for KvIndex at scale (>16K positions)
- Multi-tenant KV isolation (tenant-tagged pages)

### Phase 6b: Validation Scaffolding — PARTIAL

**Goal:** Build deterministic reference model and error tracking for quantization validation.

**Delivered:**

- ✓ `ReferenceModel` with LCG PRNG deterministic weights, f32 forward pass ground truth
- ✓ `compare_outputs()` producing MAE, RMSE, max error, cosine similarity
- ✓ `QuantizationErrorTracker` with per-layer error accumulation, linear vs exponential growth detection
- ✓ 8 unit tests

**Not yet done:**

- Run `ReferenceModel` against engine output end-to-end
- Convert real model (Mixtral-8x7B) and validate against reference implementation

### Phase 9: MLA Attention + YaRN RoPE + Converter Fixes — COMPLETE

**Goal:** Implement the correct attention architecture for Kimi K2.5 / DeepSeek-V3 (MLA), replace hardcoded RoPE, fix converter for compressed-tensors format.

**Delivered:**

- ✓ `MlaKvCache` storing compressed latent (kv_lora_rank=512) + RoPE (64) per position (~28x memory reduction vs GQA)
- ✓ `MlaKvCacheSet` with per-layer cache, engine creation when `model_config.mla` is `Some`
- ✓ `mla_attention_layer()` — full MLA forward pass: Q path (q_a_proj → RMSNorm → q_b_proj → split/RoPE), KV path (kv_a_proj → split → RMSNorm/RoPE → cache), per-head attention with kv_b_proj reconstruction, O projection
- ✓ `YarnRopeConfig` + `apply_yarn_rope()` — three-zone frequency scaling (theta=50000, factor=64, beta_fast=32, beta_slow=1)
- ✓ Engine dispatch: MLA → Tiered GQA → Flat GQA in `run_attention_layer()`
- ✓ `run_mla_attention()` loads segments 20-25 (4 MLA projections + 2 layernorms + O proj)
- ✓ Converter: two-pass \_packed/\_scale pairing with `combine_packed_and_scales()` for compressed-tensors INT4
- ✓ Converter: segments 24 (q_a_layernorm) and 25 (kv_a_layernorm) for MLA norms
- ✓ 12 new tests (MLA cache, YaRN RoPE, MLA attention with/without weights, Kimi K2.5 dimensions)
- ✓ 173 tests total, zero warnings

**Not yet done (deferred to empirical validation):**

- Absorbed attention optimization (fuse kv_b_proj into attention kernel on GPU)
- Test converter on real Kimi K2.5 safetensors files
- Validate MLA output against reference implementation

### Phase 10: Full GPU Kernel Dispatch + Tiled GEMV — COMPLETE

**Goal:** Move all inference computation to GPU. Optimize CUDA GEMV kernels from naive to tiled warp-cooperative reduction. Validate end-to-end on Mixtral-8x7B.

**Delivered:**

- ✓ GPU attention projections: Q/K/V/O as GPU GEMV via `try_gpu_attention_projection()` (synchronous, no Send issues)
- ✓ `self_attention_projected()` — takes pre-projected Q/K/V f32 slices, does RoPE + KV cache + attention on CPU
- ✓ Shared tensor device cache with D2D assembly from T1 pages (no VRAM→host→VRAM roundtrip)
- ✓ Model preloading: 13,915 pages (27.8 GB) at 1,400 MB/s (~20s) for full T1 residency
- ✓ Embedding lookup, logits computation, RMSNorm, fused residual add on GPU
- ✓ All MoE sublayer kernels on GPU: partial_swiglu, partial_matmul, weighted_accumulate
- ✓ Tiled GEMV kernels: 128 threads/row, 2 rows/block, warp shuffle + shared memory reduction
- ✓ Three kernel variants optimized: `partial_matmul_fp16`, `partial_matmul_int4`, `fused_swiglu_fp16`
- ✓ Server Send fix: `tokio::spawn` for inference work in axum handler
- ✓ Measured: **3.6 tok/s, 3.8s TTFT** on Mixtral-8x7B INT4 (7.2x speedup from tiled GEMV alone)
- ✓ 248 tests (61 unit + 186 integration + 1 doc-test), zero clippy warnings, zero formatting issues

**Not yet done:**

- Pre-allocate projection scratch buffers (eliminate 128 cudaMalloc/cudaFree per token)
- Flash Attention CUDA kernels (required for 16K+ context)
- Tensor-core-accelerated GEMV (WMMA/MMA instructions for further speedup)

### Phase 11: Embedded HNSW Backend + Virtual Expert Assembly

**Goal:** Embed usearch HNSW into the binary as the production ANN backend, enabling page-level weight-space retrieval and virtual expert assembly.

**11.1 — HnswBackend (implementing):**

- `HnswBackend` struct wrapping `usearch::Index`, implementing `AnnBackend` trait
- `new(centroids, metric, connectivity, expansion)` — build HNSW from centroid vectors
- `from_index(usearch::Index)` — wrap pre-built index
- L2 and cosine distance metric support via `usearch::MetricKind`
- Centroid storage for the `centroid()` trait method (usearch doesn't expose stored vectors directly)
- `SearchBackend::Custom(Box::new(HnswBackend))` integration with `VectorIndex`
- Tests: construction, search accuracy vs brute-force, filtered search, serialization round-trip

**11.2 — Page-level signatures (planned):**

- `compute_page_signature(page_data, method) → Vec<f32>` in a new `src/index/page_signatures.rs`
- Methods: `Mean` (mean of weight rows), `Svd` (first principal component), `Activation` (calibration-based)
- During `vib3-convert`: compute signatures for all pages, build HNSW index, store in `.vib3` file
- Page signature dimension: 64-256 (configurable, trading index size vs accuracy)

**11.3 — Virtual expert query planner (planned):**

- New `PageMode::VirtualExpert` execution strategy in `QueryPlanner`
- `plan_layer_virtual()`: hidden_state → hnsw.search_k(hidden_state, top_k) → ResolvedPage list
- Relevance-weighted accumulation: each page's matmul output scaled by (1 / distance)
- Fallback: if HNSW index unavailable, use standard expert routing

**11.4 — Scatter-gather matmul kernel (planned):**

- CUDA kernel for matmul across non-contiguous weight pages
- Input: hidden_state, list of (page_ptr, row_range, relevance_weight) tuples
- Output: weighted sum of partial matmul results
- CPU fallback: sequential partial_matmul with accumulation

**11.5 — Validation on Mixtral-8x7B (planned):**

- Convert Mixtral-8x7B (open weights, Apache 2.0) with page signatures
- Compare output quality: fixed router vs virtual expert retrieval
- Measure: perplexity delta, tokens/second, HNSW search latency overhead
- Success criteria: perplexity within 5% of reference, HNSW overhead < 1ms/token

**Not yet done:**

- Page signature quality comparison (mean vs SVD vs activation-based)
- Optimal top-k page count per layer (hyperparameter sweep)
- Scatter-gather kernel GPU performance vs contiguous expert matmul

**11.6 — Sidecar Index File (`.vib3.idx`) (designed, not yet implemented):**

The vector index, coactivation table, and materialized view references are decoupled from the `.vib3` data file into a sidecar index file. No production database rewrites its tablespace to rebuild an index. The `.vib3` file is the tablespace; the `.vib3.idx` file is the index.

The sidecar lives alongside the model file: `kimi-k2.5-nvfp4.vib3` + `kimi-k2.5-nvfp4.vib3.idx`. The engine discovers it automatically (same path, `.idx` appended). If present, the sidecar's index data takes precedence over any embedded index sections.

File layout:

```
┌────────────────────────┐  offset 0
│   Vib3IdxHeader        │  256 bytes, fixed, #[repr(C, packed)]
├────────────────────────┤
│   Section Directory    │  IdxSectionEntry[] (32 bytes each)
├────────────────────────┤
│   Section 0: Centroids │  [u32 count, u32 dim, f32[] data]
├────────────────────────┤
│   Section 1: Entries   │  VectorIndexEntry[] (same Pod struct as .vib3)
├────────────────────────┤
│   Section 2: HNSW      │  usearch serialized graph (from save_to_buffer())
├────────────────────────┤
│   Section 3: Coact     │  CoactivationEntry[]
├────────────────────────┤
│   Section 4: Views     │  MaterializedViewHeader[]
├────────────────────────┤
│   Section 5: Metadata  │  JSON: profiler config, calibration stats, PCA matrix
└────────────────────────┘
```

The header contains parent binding fields: `parent_checksum` (XXH3 of the parent `.vib3` header), `parent_page_count`, and `parent_file_size`. If the model is re-converted (different quantization, different page layout), the old sidecar is automatically invalidated — the engine falls back to embedded indexes or no index.

Hot-swap semantics: profiler writes to `model.vib3.idx.tmp`, completes with `rename()` (atomic on POSIX), engine detects via inotify or periodic stat, builds new `VectorIndex` from sidecar, atomically swaps into `QueryPlanner` via `Arc::swap`. Old index dropped after in-flight queries drain. No engine restart required.

Size: <1MB for a 160-570GB model file. Estimated ~740KB for Kimi K2.5 with 256 clusters (256KB centroids + 108KB entries + 200KB HNSW + 160KB coactivation + 7KB views + 4KB metadata).

Implementation path:

1. Define `Vib3IdxHeader`, `IdxSectionEntry` structs in `src/storage/sidecar.rs`. `Pod` + `Zeroable` for zero-copy.
2. `Vib3IdxWriter` that takes profiler output and serializes. Reuses `HnswBackend::save_to_buffer()`.
3. `Vib3IdxFile::open()` with parent binding validation. Returns centroids + entries + optional HNSW buffer.
4. `Vib3File::open()` checks for `.idx` sidecar. If valid, constructs `VectorIndex` from sidecar instead of embedded sections.
5. `QueryPlanner::reload_index()` for atomic `Arc<VectorIndex>` swap.
6. CLI: `vib3-convert --profile` writes sidecar. `vib3 inspect --index` dumps sidecar contents.

Estimated effort: 2-3 days for format + reader/writer, 1 day for engine integration, hot-swap optional.

**11.7 — Synthetic Index Building: Indexes as Learned Parameters (designed, not yet implemented):**

Key insight: **indexes are derived parameters optimized against an objective function, just like weights.** Weights are optimized by backprop against a loss function. Indexes are optimized by profiling against a prefetch precision/recall objective. The difference is the optimization method, not the lifecycle. Both follow: build → measure → tune → persist → iterate. The sidecar format makes this iteration loop cheap (<1MB writes instead of 500GB rewrites).

**Epoch 0 — Weight statistics (zero inference cost, already implemented):**

`compute_page_signature(Mean)` in `hnsw_backend.rs` computes mean-of-weight-rows per page. Cluster pages by weight geometry. Hypothesis: pages with similar weight structure respond to similar inputs. This is the current state of embedded indexes — coarse but non-zero signal. Build the baseline `.vib3.idx` sidecar from this alone in seconds. Every subsequent epoch must beat it.

What exists: `compute_page_signature()` and `build_vector_index()`, both tested. What's needed: wire into `Vib3IdxWriter`. ~50 lines.

**Epoch 1 — Synthetic router probing (near-zero inference cost, planned):**

The router at each MoE layer is a tiny linear projection: `hidden_dim → num_experts` (~14M params for Kimi K2.5 vs 1T total). The router weights are in the `.vib3` file as segment type 8 (`router`).

1. Load ONLY the router weight matrices (~14M params, ~28MB at FP16). Takes <1 second.
2. Generate synthetic hidden states: `N(0, 1/sqrt(hidden_dim))` (Xavier-matched) or empirical distribution from a few real tokens through embedding + first dense layer (~17GB shared weights).
3. Run router sigmoid + bias correction at each of 60 MoE layers. 60 matrix multiplies of `[1, 7168] × [7168, 384]` — trivial on GPU, ~0.1ms per token total.
4. Collect millions of `(synthetic_hidden_state, router_selections_per_layer)` tuples.
5. Feed to `ActivationProfiler.record_token()` as synthetic activations.
6. Build index via `build_vector_index()` + `build_coactivation()`. Write `.vib3.idx`.

Cost: ~0.01% of full inference. 1M synthetic tokens in ~100 seconds. Captures the router's decision boundary geometry without running any expert FFN computation. The coactivation graph is especially valuable — which experts the router selects together is a property of the router weights, not the calibration data.

Limitation: Synthetic hidden states may not match the real distribution. Router decisions at layer 0 are accurate (hidden state unmodified by experts), but later layers' inputs depend on earlier layers' expert outputs, which are not computed. The fix is epoch 2.

**Epoch 2 — Router-only forward pass (10-20% of full inference cost, planned):**

Run the model but skip expert FFN computation after routing:

1. Process real text through embedding → shared layers → attention → router.
2. At each MoE layer, run the router, record `(hidden_state, expert_selections)` in the profiler.
3. Substitute a zero/identity approximation for the expert output instead of loading and computing with selected experts.
4. Continue to the next layer. The hidden state drifts, but router decisions at early layers are accurate and late-layer decisions capture the router's behavior given its actual input distribution shape.
5. Cost: attention + shared layers dominate. Expert FFN (the expensive part for storage) is skipped entirely.

What this buys over epoch 1: real hidden states evolved through real attention and shared layers. The router sees realistic inputs, not synthetic noise.

Implementation: `--profile-fast` flag that sets `skip_expert_ffn = true`. Router execution and profiler recording happen at existing hook points (`engine.rs:3007`). Expert FFN bypass is a one-line check in MoE sublayer dispatch.

**Epoch 3+ — Online profiling during serving (full cost, useful output):**

The `engine.rs:3007` profiler callback during normal inference. Every token of real serving contributes training data for the next index version. Reservoir sampling in `ActivationProfiler` (already implemented) keeps memory bounded at `max_samples`. The system improves while serving.

Trigger for re-indexing: `PlannerStats.precision()` drops below 70% OR `profiler.sample_count()` exceeds 2x the current index's training data.

**The index lifecycle:**

```
Conversion          Epoch 0             Epoch 1            Epoch 2+           Serving
─────────────────────────────────────────────────────────────────────────────────────────
vib3-convert     → weight-stats idx  → router-probe idx → calibrated idx   → online tuning
(500GB, hours)     (<1MB, seconds)     (<1MB, minutes)    (<1MB, minutes)     (background)
                                                                               │
                   ┌────────────────────────────────────────────────────────┐   │
                   │  .vib3.idx sidecar: atomic rename, hot-swap, <1MB     │◄──┘
                   │  PlannerStats monitors precision/recall continuously  │
                   │  Auto-tune triggers when precision drops below 70%    │
                   └────────────────────────────────────────────────────────┘
```

The weight file is written once and never touched again. The index evolves independently through progressively better approximations of the router's decision surface. The system is usable from minute zero (epoch 0 index from weight statistics) and improves continuously.

**11.8 — Auto-Tuning Loop (designed, not yet implemented):**

The auto-tuning loop treats index building as hyperparameter optimization with `PlannerStats.precision()` and `PlannerStats.recall()` as the objective function. Runs in a background thread, never blocking inference.

```rust
fn auto_tune_index(profiler: &ActivationProfiler, current_stats: &PlannerStats) -> Option<Vib3Idx> {
    let (train, held_out) = profiler.split(0.8);  // 80/20 train/validation

    let mut best_score = current_stats.precision();
    let mut best_config = None;

    // Grid search over index hyperparameters
    for num_clusters in [64, 128, 256, 512, 1024] {
        for centroid_dim in [128, 256, raw_dim] {  // PCA projection dimensions
            for metric in [L2, Cosine] {
                for hnsw_m in [8, 16, 32] {
                    let projected = pca_project(&train.embeddings, centroid_dim);
                    let (centroids, entries) = train.build_vector_index(num_clusters, 20);
                    let hnsw = HnswBackend::new(centroids, &HnswConfig {
                        metric, connectivity: hnsw_m, ..default()
                    });

                    // precision@k where k = num_active_experts (8 for Kimi)
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

The evaluation metric — `precision@k` — asks: "of the k experts the index predicted for this hidden state, how many did the router actually select?" This directly measures prefetch accuracy.

Auto-tune metadata in sidecar Section 5 (Metadata JSON) records full provenance: epoch, strategy, calibration tokens/dataset/hash, hyperparameters (clusters, PCA dim, variance retained, metric, HNSW M/ef, k-means init/iterations, max samples, min coactivation correlation), quality metrics (precision@k, recall@k, centroid coverage, cluster sizes, empty clusters), and previous index stats. Every index version is reproducible and auditable.

Trigger conditions: precision drops below 70%, OR sample count exceeds 2x current index's training data. Only replaces if >2% improvement. Atomic sidecar rename + hot-swap.

### Phase 12: Batched Multi-Request Serving

**Goal:** Serve multiple concurrent requests without buffer pool thrashing.

- Request scheduler that groups tokens by predicted expert overlap
- Shared expert loading: when multiple requests need the same expert, load once
- Per-request KV cache isolation with shared weight page pool
- Continuous batching inspired by vLLM's approach
- Benchmark: throughput scaling from 1 to 8 to 32 concurrent requests

### Phase 13: Production Hardening

- Integrate tiktoken/sentencepiece tokenizer (replace byte-level stub)
- Model registry with chunked resumable downloads
- GGUF import path (for compatibility with llama.cpp quantized models)
- Prometheus metrics export
- Docker image with CUDA + io_uring support
- Benchmark suite against vLLM, TensorRT-LLM, llama.cpp on target models

---

## 13. Conclusion

### 13.1 The Bet

vib3 makes a specific, testable bet: that the dominant bottleneck for single-GPU MoE inference at trillion-parameter scale is **weight retrieval, not compute**, and that database storage engine techniques -- page-level indexing, predictive prefetch, tiered caching, query planning -- can eliminate that bottleneck.

The hardware math supports this concretely. On the target system (RTX PRO 6000 + Samsung 9100 Pro), expert compute at tensor-core throughput takes 0.04 ms per token while I/O takes 20-40 ms. The system is I/O-bound by three orders of magnitude. **Measured validation on Mixtral-8x7B INT4** (Phase 10): with all weights preloaded to VRAM (100% T1 hit, zero I/O), the system achieves 3.6 tok/s (256 ms/token), with MoE expert compute at 1.1 ms/layer (35 ms total) and the remaining ~220 ms consumed by router dispatch, D2D copies, attention, and synchronization overhead. When I/O drops out, compute overhead dominates -- confirming that I/O optimization is the lever, not compute optimization (though tensor-core GEMV would further reduce the 35 ms MoE component). The question is whether the I/O pipeline can be made fast enough.

### 13.2 The Compressed RAM Insight

**The most important quantitative finding is that NVMe drops out of the steady-state critical path.**

With Zstd compression at 3.5x, 168 GB of system RAM holds ~588 GB of effective model data — enough to cover the entire ~570 GB model. Combined with Blackwell's hardware Decompression Engine (600 GB/s), compressed pages flow from RAM to VRAM with decompression cost under 1 ms per token. The I/O ceiling from compressed RAM alone is ~200 tok/s (limited by 64 GB/s PCIe ÷ compressed page size), making attention compute — not storage retrieval — the sole bottleneck in steady state.

This changes the performance model fundamentally. The system we designed to solve the NVMe-to-GPU bandwidth problem discovered that it can largely bypass NVMe entirely. The three-tier hierarchy GPU ← RAM ← NVMe remains essential for cold start, cache misses, and the ~7% of the model that doesn't fit in compressed T2 — but the hot path is GPU ← compressed RAM, and that path is fast.

The tiered compression cascade pushes this further. Maximum compression on NVMe (Zstd-19, ~5-6x) turns a single Gen5 drive into ~81 GB/s effective throughput for cold-start and miss scenarios. GPU-friendly compression in RAM (Zstd-1, ~3.5x) feeds the Blackwell DE at wire speed. Each tier's compression is tuned to its bottleneck, the same way production databases use different compression for hot and cold storage.

**This finding should be qualified:** the ~200 tok/s ceiling assumes 100% T2 hit for T1 misses. During cold start, domain transitions, or generalist-mode working set churn, T2 misses fall through to NVMe. The compression cascade mitigates this (~81 GB/s effective from NVMe vs 14.7 GB/s raw), but the system is not immune to cold-path stalls.

### 13.3 The Conditions

**With nvCOMP + Zstd (Pipeline B), 45+ tok/s is conditionally achievable at 90% hit rate, and 50+ tok/s at 93%+ hit rate.** A single Gen5 NVMe drive at 14.7 GB/s, carrying Zstd-compressed pages at 3.5x ratio, delivers 51.5 GB/s effective raw throughput when decompressed by the Blackwell hardware DE at 600 GB/s. At 90% buffer pool hit rate, this yields ~22 ms I/O latency per token — enough for ~45 tok/s at short contexts.

The conditions for success:

- **90%+ buffer pool hit rate** — requires expert activation locality. Validated by research on DeepSeek-V2 (high locality within domains) but needs empirical confirmation on Kimi K2.5.
- **nvCOMP Pipeline B** — sends compressed data over PCIe, decompresses on GPU. Without this, Pipeline A (CPU decompress) yields ~32 tok/s.
- **MLA-optimized Flash Attention** — at 16K+ context, attention (~30-60 ms standard) dominates I/O (~22 ms). MLA reduces this to ~15 ms, making I/O the bottleneck again and nvCOMP the deciding factor.
- **Expert activation prediction >70%** — for prefetch to help rather than hurt. Below this threshold, reactive loading (no speculation) is better.

The conditions for failure:

- KV cache memory dominates (mitigation: tiered KV cache now implemented, Section 8 — needs empirical accuracy validation)
- Expert activations are too unpredictable (mitigation: fall back to `PageMode::Exact`)
- Attention compute dominates latency (mitigation: Flash Attention + MLA — MLA now implemented, Flash Attention CUDA kernels still needed)
- INT4 quantization error degrades generation quality (mitigation: calibrated quantization, GPTQ/AWQ — validation framework built, Section 6b)

### 13.4 What Exists Today

vib3 is not a proposal. It is ~18,900 lines of Rust library + tools code (plus ~5,340 lines of integration tests and ~490 lines of CUDA kernels) with 248 tests (all passing), zero warnings, implementing a complete inference pipeline that produces coherent output at **3.6 tok/s on Mixtral-8x7B INT4** (single RTX PRO 6000 GPU). **Kimi K2.5 validation is in progress:** MLA attention is correct through layer 5 (cosine >0.92 vs PyTorch BF16 ground truth) but layer 6 MoE has a blocking bug (see Section 7.5).

| Component                                                                                                  | Status   | Lines  | Tests |
| ---------------------------------------------------------------------------------------------------------- | -------- | ------ | ----- |
| `.vib3` binary file format with page catalog, expert index, vector index, coactivation table               | Complete | ~985   | 8     |
| Three-tier `PageBufferManager` with compressed T2, specialist pinning, prediction-aware eviction           | Complete | ~1,390 | 12    |
| io_uring NVMe integration with SQPOLL kernel-bypass                                                        | Complete | ~270   | -     |
| Predictive indexing: vector index (pluggable ANN backend), coactivation graph, domain classifier, profiler | Complete | ~1,710 | 13    |
| Entropy-based dual-mode adaptive caching, wired into runtime engine                                        | Complete | ~500   | 8     |
| CUDA compute kernels: tiled GEMV (FP16/INT4), fused SwiGLU, RMSNorm, residual add, embedding, logits       | Complete | ~1,685 | 22    |
| CPU compute kernels: partial matmul, RoPE, YaRN RoPE, INT4 LUT, sparse attention                           | Complete | ~500   | -     |
| MLA attention (DeepSeek-V3/Kimi K2.5), GQA attention, flat and tiered KV cache paths                       | Complete | ~920   | 18    |
| Tiered KV cache: `TieredKvCache`, `KvIndex`, `UnifiedEvictionPolicy`, `LandmarkTracker`                    | Complete | ~1,350 | 21    |
| Runtime engine with GPU dispatch, MLA, query planner, mode-aware prefetch, shared tensor device cache      | Complete | ~2,100 | 14    |
| INT4 quantization pipeline, HF config parser, \_packed/\_scale pairing, full model converter               | Complete | ~1,270 | 14    |
| Validation framework: `ReferenceModel`, error tracking, comparison tools                                   | Complete | ~525   | 8     |
| OpenAI-compatible streaming API server                                                                     | Complete | ~365   | -     |
| CLI tools: run, bench, serve, inspect                                                                      | Complete | ~770   | -     |
| CUDA kernels (`kernels.cu`): tiled GEMV, fused SwiGLU, INT4 dequant, silu_mul, router GEMV, RMSNorm        | Complete | ~490   | -     |
| `.vib3.idx` sidecar index format, synthetic building pipeline (epochs 0-3+), auto-tuning loop             | Designed | -      | -     |

**The choice of Rust is deliberate and structural.** Zero-copy mmap for the `.vib3` file format. Lock-free DashMap for T1 page lookup (one atomic read per hit, no mutex). `io_uring` integration via the `io-uring` crate with `SQPOLL` for kernel-bypass NVMe reads. Unsafe FFI for CUDA interop with explicit lifetime management. No garbage collector pauses in the critical path. The buffer manager's hot path — T1 lookup, atomic access tracking, pointer return — is deterministic-latency, something that is structurally harder to achieve in Python or managed-memory languages. For a system where per-page management decisions happen at microsecond granularity, this matters.

### 13.5 The Dual-Mode Activation System

The dual-mode activation system addresses the largest unknown: expert locality. Rather than betting on one access pattern, the engine measures the Shannon entropy of expert activations over a sliding window and adapts. In specialist mode (low entropy), it pins hot expert clusters in T1 for 95%+ hit rates and ~67-91 tok/s. In generalist mode (high entropy), it accepts 60-70% hit rates and leans on aggressive prefetch from compressed T2. Mode transitions trigger automatic pin/unpin of expert clusters and adjustment of prefetch aggressiveness in the query planner. The hysteresis mechanism (8 consecutive opposite-mode readings) prevents oscillation.

### 13.6 The Storage Engine Is the Contribution

The unified tiered cache (Section 8) is now implemented, not proposed. Tiered KV caching is a solved problem (LMCache, Dynamo, HyperPod). What no existing system does is manage weight pages AND KV pages under one storage engine. vib3 now does this: `PageId` addresses both weight pages and KV pages, the `UnifiedEvictionPolicy` makes cross-domain eviction tradeoffs (preferring KV eviction over weight eviction when both are cold, because weight misses have higher stall cost), and the engine dispatches between flat and tiered attention paths based on configuration. For MoE models that don't fit in VRAM, this unification is the key architectural advantage: VRAM is a zero-sum game between hot expert pages and active KV pairs, and only a unified manager can make the right tradeoff.

### 13.7 Virtual Experts: The Deeper Thesis

The database analogy leads somewhere unexpected when followed to its conclusion. If expert weights are indexed database pages, and the router is a query optimizer, then the natural next step is to **replace the fixed router with vector similarity search over the weight space itself** (Section 2.4).

This is not just a performance optimization -- it is a different inference mechanism. Standard MoE inference selects fixed experts via a learned router. Weight-indexed inference retrieves the most relevant weight pages via HNSW search, regardless of expert boundaries. The expert partitioning becomes a storage convention, not a computational constraint.

The embedded HNSW backend (usearch, Phase 11) is the first concrete step. It implements the `AnnBackend` trait, plugs into the existing `VectorIndex` infrastructure, and enables three levels of retrieval:

1. Expert-level prediction (current: centroid-based, replacing brute-force with HNSW)
2. Page-level retrieval (next: individual weight pages indexed by signature vectors)
3. Virtual expert assembly (future: composing ad-hoc experts from the most relevant weight pages)

The research risk is clear: page signature quality determines whether virtual expert assembly produces coherent output. The learned router was optimized end-to-end with backpropagation. ANN search over page signatures is a proxy for relevance. Whether this proxy is good enough is an empirical question that requires validation on real models (Mixtral-8x7B first, then Kimi K2.5).

If it works, the implications are significant: models become extensible databases of weight pages. New capabilities (new languages, new domains, fine-tuned behaviors) can be added by inserting weight pages into the HNSW index -- no retraining, no router modification, no full model reload. The model becomes a living, editable index of computation.

### 13.8 The Architecture Is the Argument

More broadly, the database analogy is not a framing device — it is the architecture. vib3 is a specialized read-only indexed storage engine with a query planner, a three-tier buffer pool, tiered compression, materialized views, workload-adaptive cache management, a tiered KV cache with ANN-indexed retrieval, and now an embedded HNSW vector index for weight-space routing. The "database" is the model. The "queries" are hidden states. The "rows" are weight pages and KV pairs. The techniques — page-level indexing, prediction-aware eviction, cost-based execution strategy selection, compression cascades, ANN-indexed sparse retrieval, filtered search with metadata predicates — are database and vector-database storage engine techniques applied to a new domain. The novelty is not in any single technique but in recognizing that MoE inference is fundamentally a retrieval problem — for both weights and KV cache — and that the inference mechanism itself can be retrieval over indexed weight space. The sidecar index format (`.vib3.idx`) completes this picture: indexes are decoupled from data files, iteratively rebuilt through progressively better approximations (weight statistics → synthetic router probing → full calibration → online profiling), auto-tuned via hyperparameter search against a precision@k objective, and hot-swapped without engine restart — the same index lifecycle a production database follows, applied to neural network weight retrieval.

---

## References

### Systems

- vLLM: Kwon et al., "Efficient Memory Management for Large Language Model Serving with PagedAttention" (SOSP 2023)
- BitNet.cpp: Microsoft, "1-bit AI Infra: Fast and Lossless BitNet b1.58 Inference on CPUs" (2024)
- AirLLM: Li, "Scaling Large Language Models on Low-End Commodity Computers" (2023)
- TensorRT-LLM: NVIDIA (2024)
- llama.cpp: Gerganov et al. (2023)
- DeepSeek-V2: DeepSeek AI, "DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model" (2024)

### Models

- Kimi K2.5: Moonshot AI (2025)
- Mixtral-8x7B: Mistral AI (2024)
- Switch Transformer: Fedus et al. (2022)

### KV Cache and Long Context

- InfLLM: Xiao et al., "Efficient Streaming Language Models with Attention Sinks" (2024)
- Quest: Tang et al., "Query-Aware Sparsity in KV Cache for Efficient LLM Inference" (2024)
- MemoryFormer: Wu et al. (2024)
- Landmark Attention: Mohtashami and Jaggi (2023)

### Tiered KV Cache Systems

- LMCache: "LMCache: Reducing TTFT for Long-Context LLM Applications" (2024) -- https://github.com/LMCache/LMCache
- NVIDIA Dynamo: NVIDIA KV cache manager for massive context windows (2025)
- AWS SageMaker HyperPod: Managed auto-tiered KV caching (2025)
- MCaM: Multi-tier Cache Manager for multi-turn LLM serving (2024)

### Compression

- nvCOMP: NVIDIA, "High-speed GPU compression/decompression library" -- Blackwell DE: 600 GB/s decompression (2025)
- LZ4: Collet (2011)
- Zstandard: Collet and Kucherawy, RFC 8478 (2018)

### Quantization

- T-MAC: Wei et al., "Table-Lookup-based MAC for Low-Bit LLMs" (2024)
- GPTQ: Frantar et al. (2023)
- AWQ: Lin et al. (2024)

### Vector Indexing

- HNSW: Malkov and Yashunin, "Efficient and Robust Approximate Nearest Neighbor using Hierarchical Navigable Small World Graphs" (2018)
- Faiss: Johnson et al. (2019)
- Qdrant: Qdrant (2023)
- usearch: Vardanian (2023) — embedded in vib3 as `HnswBackend` (Apache 2.0, single-file C++ with Rust FFI)
- Product Key Memory: Lample et al., "Large Memory Layers with Product Keys" (NeurIPS 2019)
- kNN-LM: Khandelwal et al., "Generalization through Memorization: Nearest Neighbor Language Models" (ICLR 2020)

### Database Storage Engines

- PostgreSQL Buffer Manager: PostgreSQL Documentation
- InnoDB Buffer Pool: MySQL Documentation
- 2Q/ARC eviction: O'Neil et al. (1993), Megiddo and Modha (2003)
