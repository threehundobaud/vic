# vib3 roadmap

Target reference: **Kimi K2.6 (1T MoE, 60 layers, 384 experts, MLA)** running on a single
RTX PRO 6000 Blackwell (94 GB VRAM, 188 GB RAM, NVMe).

This document is the up-to-date single source of truth for what's done,
what's measured, and what to do next. Ordered by honest effort estimate.

---

## 1. Current state (2026-04-21)

### Correctness
- **Matches llama.cpp token-for-token** on the canned smoke prompt
  `"What is 2+2? Answer with one word."` → `"4.<|im_end|>..."`.
- Per-layer postlayer cosine vs llama.cpp at pos=0: **cos=1.0000 L0..L59**
  (last-position slice at L60 is a dump-shape artifact, not a correctness issue).
- Fixed in commit `8f663c7`:
  1. `expert_storage_layer` off-by-one — planner was adding
     `dense_layer_idx` on top of an already-absolute engine layer index,
     so every MoE layer read the *next* layer's expert weights.
  2. MLA attention `global_max` aliasing `scores[0]` in shared memory —
     overwrote position-0's score with the max, giving pos=0 an
     `exp(0)=1.0` weight regardless of its real logit whenever seq_len>1.

### Performance (warm, steady state)
Measured on same hardware, single-token decode, seq_len ≈ 14:

| | Cold first invocation | Warm (T1 hit ≥ 95%) |
|---|---|---|
| Decode | 0.3–0.5 tok/s | **4.6–6.7 tok/s** |
| TTFT | 70–97 s | 0.5–1.2 s |

Per-token breakdown at 6.7 tok/s peak (150 ms total):

| Stage | Time | % |
|---|---|---|
| Attention (60 × MLA) | 98 ms | 66% |
| MoE (60 × 8 routed + shared) | 50 ms | 33% |
| lm_head + logits + sample + sync | ~2 ms | 1% |

Baseline tok/s is competitive with llama.cpp on the same GPU for this
workload; the headroom above this requires kernel engineering, not
structural changes.

### Sampling quality (known issue, out of scope here)
Greedy decoding (`temperature=0`, no repetition penalty, no chat template)
collapses into `"maskedmaskedmasked..."` on most bare prompts. The instruction-
formatted canonical prompt still decodes correctly. This is a sampler /
prompt-template problem, **not** an engine correctness problem. Fix is one of:
- apply K2.6's Jinja chat template around user input
- turn on repetition penalty (already supported via the sampler)
- use `temperature > 0`

---

## 2. Investigated this session

### #10 Shared-expert / routed-expert stream overlap — null result
Moved the shared expert to a second CUDA stream (`shexp_stream`) with
events serializing the sigmoid-gated accumulate on the main stream.
Expected: ~24 ms/token saved by overlapping the ~0.4 ms/layer shared expert
with the routed experts.

**Measured: +5–10 ms regression.** Routed experts saturate SM occupancy during
their execution window, so kernels queued on a second stream are
scheduled sequentially after routed work anyway. The only net effect is
the cost of 240 CUDA event record/wait operations per token (~3–5 ms).

Reverted cleanly. Don't retry until upstream work (batched INT4 expert
kernel, CUDA graph, fully-resident moe_page_table) changes the SM
saturation profile.

---

## 3. Perf work — ordered by honest effort × expected gain

Effort sizes: **hour** = this or next session, **day** = one focused
session, **week** = multi-session project, **month** = whitepaper-scale.

### Hour-scoped — try these next

These are small, measurable, low-risk changes. Each should be A/B'd
against the 150 ms baseline with the same prompt sequence.

- **#1  MLA dual-norm fusion** — fuse `q_a_proj → rms_norm` into a single
  kernel so the 1536-dim intermediate stays in registers/shmem instead of
  round-tripping to VRAM. One new CUDA kernel + one call-site change in
  `run_mla_attention`. Projected gain: 3–5 ms/token. Risk: kernel bug at
  RMSNorm reduction. `cuda/src/kernels.cu` + engine.rs:4936.

- **#7a  Fused INT4 SwiGLU (up + gate + silu_mul)** — SHIPPED `fc23e50`.
  The kernel `vib3_fused_swiglu_int4_f32` already existed but wasn't
  wired up; `partial_swiglu_f32` still launched up+gate+silu_mul
  separately. Wired to the fused launcher. Measured: ~7 ms/token
  improvement (150 ms → 143 ms warm steady state), smaller than the
  10 ms projection because CUDA was already pipelining launches.

- **#28  Remove `ensure_shared_tensor_device` awaits from MLA hot path**
  — 4 awaits per attention layer × 60 = 240 async boundary crossings per
  token. They mostly hit the fast cache-hit path but each still has
  future polling overhead. Replace with a sync cache lookup; fall back
  to the async path only on miss. Projected gain: 2–4 ms/token. Risk: low.

- **#29  MLA kv_b_proj F32 upload once at model-load**, not lazily per-
  layer on first access. `kv_b_proj_f32_device` is populated inside
  `run_mla_attention`; move to pre-assembly at engine init. Projected
  gain: eliminates first-token-per-layer stalls; near-zero on truly warm
  runs. Risk: VRAM accounting.

### Day-scoped — next focused session

- **#7b  Port the NVFP4 `moe_experts_fused` batched kernel to INT4.**
  Currently 8 × 3 = 24 matmul launches per MoE layer → 1440/token.
  A single batched kernel launch iterating 8 experts internally collapses
  that to 60 launches. Projected gain: 10–15 ms/token. See
  `cuda/src/kernels.cu::vib3_launch_moe_experts_fused` for the NVFP4
  reference.

- **#9 Absorbed-attention fused kernel** — currently
  `mla_decode_attention` reads `kv_latent` separately from V reconstruction.
  Fuse scores + softmax + V-via-absorbed-latent into a single kernel so
  `v_latent` never hits VRAM. Projected gain: 10–20 ms/token attention.
  See whitepaper §9.6 "Future Optimization: Absorbed Attention".

- **#30  Predictive expert prefetch from the current token's router
  output** — while the current layer's MoE is computing, submit
  prefetch requests for the *next* layer's likely-hot experts based on
  activation history. Lookahead already exists in the planner
  (`planner.submit_lookahead`) but isn't hooked to a prefetch pipeline.
  Projected gain: 1–2% T1 hit improvement → ~5 ms/token.

### Week-scoped — real engineering, big levers

- **#31  Port vLLM's sm_100 CUTLASS MLA kernel** — SCAFFOLDING SHIPPED
  `a082735`, **partial knob reduction, still blocked on sm_120a smem**.
  Full CUTLASS 4.2.1 tree in `third_party/cutlass/` (gitignored, fetch
  via `scripts/fetch_cutlass.sh` which also applies
  `scripts/patch_cutlass.sh`). `cuda/src/cutlass_mla.cu` wraps
  `Sm100FmhaMlaKernelTmaWarpspecialized` with raw-pointer (no-torch)
  entry points, builds clean. `run_mla_attention` has a
  `VIB3_CUTLASS_MLA=1` gate with a sticky runtime fallback.
  
  Smem wall investigation results:
    upstream (StagesQK=12): 212 KB  ✗  (> 99 KB device opt-in)
    patched  (StagesQK=2):  129 KB  ✗  (still > 99 KB by 30 KB)
    StagesQK=1:             structurally impossible —
                            cutlass/gemm/collective/sm100_mma_warpspecialized.hpp:188
                            `static_assert(DispatchPolicy::Stages >= 2, ...)`
                            The TMA+UMMA warp-specialized pipeline is a
                            producer/consumer split; one stage can't
                            both fill and drain. Both CollectiveMmaQK
                            and CollectiveMmaPV hit the same assert.
    IsPersistent=false:     same 129 KB — scheduler state isn't the driver.
    cp.async vs TMA:        same 129 KB — load path doesn't change tile.
    kIs2Sm cluster=<_2,1,1>: already the minimum; switching to 1-SM
                            mode would double per-CTA smem (no multicast
                            sharing between CTAs). Going the wrong way.
  
  The 129 KB floor is dominated by the tile shape
  `<TileShapeH=128, TileShapeS=128, <TileShapeL=512, TileShapeR=64>>`
  which `static_asserts TileShapeH == 128`. Cutting H to 64 (what K2.6
  actually needs) would roughly halve smem but requires rewriting the
  kernel around a different MMA tile shape. That's kernel-author-scope
  work, not knob-tweaking.
  
  Two concrete unblock paths:
    1. Fork kernel/sm100_fmha_mla_tma_warpspecialized.hpp, drop the
       TileShapeH=128 assert, and re-thread 64-head MMA tiles through
       the collective. Est. 1-3 days + correctness debugging.
    2. Port xqa/mla_sm120.cu from flashinfer's archive
       (/code/.homecache/.../flashinfer/data/csrc/xqa/mla_sm120.cu) —
       NVIDIA-written specifically for sm_120, so smem already fits.
       4200 LOC across 8-10 files with coupling to an internal KV cache
       abstraction. Est. 2-4 days.
  
  Either way, the CUTLASS scaffolding (build infra, wrapper, FFI,
  runtime integration, fallback) stays as-is — drop the new kernel in,
  flip `VIB3_CUTLASS_MLA=1`, done.

- **#32  CUDA graph replay with dynamic MoE dispatch.**
  The `VIB3_CUDA_GRAPH=1` fast path exists but NaNs on K2.6 because
  captured graphs bake host-side expert indices. Fix requires a fully-
  GPU MoE path (`moe_experts_fused_gpu`), which currently only engages
  when the whole model is T1-resident. Needs either the compressed-T2
  path (#34) to enlarge effective T1 or a device-indirect expert
  dispatch that reads expert IDs from the router's device buffer.
  Projected gain: 30–40%.

- **#10b  Revisit stream overlap with paged-page prefetch overlap.**
  Overlap between routed-expert *weight prefetch* (PCIe DMA) and
  *compute* of earlier experts, not between compute and compute. This
  is different from the #10 we tried — the SMs aren't the contended
  resource, PCIe is.

### Month-scoped — whitepaper bets

- **#33  Blackwell Decompression Engine via nvCOMP.**
  Whitepaper §7 pipeline-B. Keeps T2 compressed → DMA → GPU decompress.
  1.5× effective T2 capacity, projected 45 tok/s at 90% hit vs ~6.7
  today. Needs nvCOMP integration, CUDA 12.4+.

- **#34  HNSW predictive expert prefetch.**
  Whitepaper §11.1–11.3. Embeds current hidden state, ANN-lookups
  likely-activated experts for layer N+2/N+3, prefetches their pages.
  Requires >70% recall to beat prefetch-overhead break-even.

- **#35  LoRA-lites for infrequent experts.**
  Whitepaper §11.9. Rank-16 delta weights keep cold experts warm at
  1/8 the bandwidth. Only pays off if we have the prefetch predictor
  (#34) to know which experts are cold.

- **#36  Probe-mined specialist distillation.**
  Whitepaper §2.8 / §11.10. Train a small model to mimic the MoE
  behavior for frequently-asked queries. Out of scope for the inference
  engine, but drives the product vision.

---

## 4. Correctness / tooling backlog

- **#24  Auto-pipe GGUF vocab+merges through vic-convert** so a fresh
  GGUF produces a ready-to-run `.vib3` + `tokenizer.json`. Currently
  the smoke script assumes a pre-existing tokenizer in
  `/data/huggingface/vib3-models/tokenizer.json`.

- **#12  Phase-11 HNSW virtual-expert retrieval on K2.6** — whitepaper
  design exists; implementation deferred until #34 proves the recall
  threshold.

- **Apply K2.6's chat template** in `run.rs` / `serve.rs` so bare user
  prompts don't greedy-decode into `<mask>` attractors.

- **Delete archived `kimi-k2.6.vib3.bad_int4_scales`** (483 GB) once
  confirmed unneeded — the hypothesis that created it was wrong (the
  file was fine; the engine bug was the off-by-one).

---

## 5. How to A/B measure any of the above

Standard measurement protocol — run the same sequence, look at the
warm-steady portion (token ≥ 7 of a 32+ token generation):

```bash
{ echo "Warmup hi."; echo "The capital of France is"; echo "Say hi."; } | \
  VIB3_DIAG=0 VIB3_LOG=vib3=info PATH=/usr/local/cuda/bin:$PATH \
  /code/vib3/target/release/vic run \
    /data/huggingface/vib3-models/kimi-k2.6.vib3 \
    --max-tokens 16 --temperature 0.0 --no-check 2>&1 | \
  grep -E "attn=.*moe=|tokens,"
```

Then compare `attn=X.Yms moe=X.Yms` against the 150 ms baseline on the
same prompt, same token index.

Correctness check (always do this before committing a perf change):
```bash
VIB3_DIAG=0 PATH=/usr/local/cuda/bin:$PATH \
  /code/vib3/target/release/vic run \
    /data/huggingface/vib3-models/kimi-k2.6.vib3 \
    --max-tokens 5 --temperature 0.0 --no-check \
    < <(echo "What is 2+2? Answer with one word.")
# Must produce: "4.<|im_end|><|im_end|><|im_end|>"
```

---

## 6. Reference timings

### commit fc23e50, 2026-04-21 — after #7a fused INT4 SwiGLU wired in

```
multi-prompt session, token 7-15 steady state:
  Total:              143 ms = 7.0 tok/s
  ├─ Attention:        90 ms  (60 layers × 1.5 ms MLA — 9 kernel launches each)
  ├─ MoE:              51 ms  (60 layers × 0.85 ms — 8 experts fused SwiGLU + 1 shared)
  ├─ lm_head + logits: 0.6 ms
  └─ GPU sync:         0.5 ms

best-token observed: 135 ms = 7.4 tok/s
worst (non-eviction) steady-state token: 148 ms = 6.8 tok/s
```

### commit 8f663c7, pre-fusion baseline — for comparison

```
multi-prompt session, token 7-15 steady state:
  Total:              150 ms = 6.7 tok/s
  ├─ Attention:        98 ms
  ├─ MoE:              50 ms  (24 launches per MoE layer — up+gate+silu_mul × 8 experts)
  ├─ lm_head + logits: 0.6 ms
  └─ GPU sync:         0.7 ms

diverse-prompt session: 4.6 tok/s average, T1 hit rate 95%

cold first invocation: 0.3 tok/s, TTFT 70-97 s
```

The compute floor is ~200 μs/layer attention if launch overhead went to
zero. That's the ceiling #31 (CUTLASS MLA port) + #32 (CUDA graph) are
aiming at: ~12 ms/token attention = ~60 tok/s.
