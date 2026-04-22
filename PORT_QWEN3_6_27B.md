# Qwen3.6-27B port — status & remaining work

## Bisection against HF reference (2026-04-22)

**HF reference top-1 on "The capital of France is":** token 11751 = `" Paris"` with logit 15.625.
**vib3 top-1:** token 180887 = `"asilan"` with logit 4.5.

**Layer-by-layer cosine vs HF reference** (tools/qwen36_hf_reference.py + qwen36_diff_layers.py):

| Layer | type | vib3 L2 | HF L2 | cos |
|---|---|---|---|---|
| L0 output | DeltaNet+FFN | 12.04 | 17.96 | **0.8422** |
| L1 output | DeltaNet+FFN | 13.41 | 28.16 | **0.7725** |
| L2 output | DeltaNet+FFN | 12.19 | 39.96 | **0.5840** |
| L3 output | GatedAttn+FFN | 37.95 | 46.46 | 0.9075 |
| L4 output | DeltaNet+FFN | 37.00 | 49.54 | 0.8679 |
| L5 output | DeltaNet+FFN | 39.02 | 53.51 | 0.8522 |
| ... (stays ~0.8 through L9) | | | | |

**Root cause: DeltaNet layers silently produce wrong output.** Evidence:
- Full-attention layer L3 sharply recovers cosine (0.58 → 0.91).
- Magnitude shrinkage is cumulative through L0→L2.
- `DELTANET_DIAG L1 pos=0`: `q_head_L2 first4=[0.0884, 0.0884, 0.0884, 0.0884]` — identical across heads (suspicious, should differ per head).
- `DELTANET_STEP L1 pos=0: step_out L2=0.0036` — tiny, suggests state-read-before-update at pos 0 returning near-zero state contribution.

**Top suspects** (ranked by likelihood, for next session):
1. **DeltaNet state update order** — classic bug: reading state before writing it on the first position gives ~0 output (since initial state is 0). HF does update-then-read.
2. **Per-head Q reshape** — `q_head_L2` being identical across 4 probed heads suggests Q is not being split per head correctly.
3. **`in_proj_a` / `in_proj_b` semantics** — Qwen3.6 has these split, shapes `[48, 5120]` each (projecting to per-head alpha/beta scalars). vib3's Qwen3.5 path may combine them differently.
4. **Per-head RMSNorm on DeltaNet intermediate** — norm weight shape `[128]` matches head_dim; the kernel stride for per-head norm needs verification.

**Tooling committed:**
- `tools/qwen36_hf_reference.py` — runs Qwen3.6-27B through HF transformers and dumps per-layer hidden states for tok0 and tok1 into `/tmp/hfref/`.
- `tools/qwen36_diff_layers.py` — compares vib3's `dump/vib3_postlayer_f32_L*_tok*.bin` against HF dumps, prints cos/L2/diff per layer.
- `VIB3_DIAG_ALL_LAYERS=1` env var forces vib3 to dump hidden state at every (layer, token) pair (default is sparse).

To reproduce the bisection from a fresh checkout:
```bash
# 1. vib3 dumps (all layers, tok 0..4)
rm -f dump/vib3_postlayer_*
VIB3_DIAG=1 VIB3_DIAG_ALL_LAYERS=1 VIB3_LOG=vib3=warn \
  ./target/release/vic run /data/huggingface/vib3-models/qwen3.6-27b.vib3 \
    --tokenizer /data/huggingface/vib3-models/qwen3.6-27b-hf/tokenizer.json \
    --max-tokens 1 --temperature 0.0 --no-check < <(echo "The capital of France is")

# 2. HF reference dumps (requires PyTorch + transformers in env)
uvx --with torch --with transformers --with accelerate --with safetensors --with numpy \
  python tools/qwen36_hf_reference.py

# 3. Diff
uvx --with numpy python tools/qwen36_diff_layers.py
```

## Unsloth Qwen3.6 comparison (2026-04-22)

From https://unsloth.ai/docs/models/qwen3.6:

| dimension | Unsloth | vib3 (Qwen3.6-27B today) |
|---|---|---|
| Backend | llama.cpp + Unsloth Studio | custom Rust + CUDA, sm_120a block-scaled MMA |
| Quants shipped | UD-Q2_K_XL, UD-Q4_K_XL (Dynamic 4-bit), Q4_K_M, Q6_K, BF16 | BF16 native converter, runtime NVFP4 FFN (drafted) |
| 27B footprint | 15 GB (3-bit) → 18 GB (4-bit) → 30 GB (8-bit) → 55 GB (BF16) | 39 GB (zstd-compressed BF16 .vib3), 52 GB raw BF16 |
| Perf claims | none published for 27B on Blackwell | n/a until correctness lands |
| Quality (KLD vs BF16) | "Pareto frontier, top-perf in 21/22 sizes" (35B-A3B chart) | not yet measurable |
| DeltaNet handling | "important layers upcasted" under Dynamic 2.0 | same shape support; correctness bug currently |
| Vision | skipped by default (separate mmproj) | skipped by design, text-only |
| Ollama compatible | no (mmproj split) | — |

**Unsloth's advantage right now** is breadth of quants (down to 3-bit) + mature llama.cpp tooling. They don't publish raw throughput numbers so no direct perf comparison possible.

**vib3's structural bet** is sm_120a block-scaled NVFP4 MMA that llama.cpp doesn't use + the page-indexed weight system for oversized models (K2.6-style). For Qwen3.6-27B dense (fits 96 GB BF16) that bet isn't tested yet — Unsloth's Q4_K_XL at 18 GB will be faster on smaller cards. vib3's lever is on >96 GB models (K2.6-style) and on Blackwell-specific MMA that llama.cpp kernels don't hit.

To move past "plumbing works" into "beats Unsloth on a meaningful axis":
1. Fix the DeltaNet correctness bug (above) to produce coherent output.
2. Wire the NVFP4 FFN conversion (drafted in PORT_QWEN3_6_27B §NVFP4 lever).
3. Bench both side-by-side on a long-context prompt where the sm_120a MMA beats llama.cpp's GPU kernels.

---



## What's landed

| Layer | State | Commit |
|---|---|---|
| `qwen36_27b` constants (types.rs) | shipped | `4cccf6e` |
| `ModelConfig::qwen36_27b()` builder | shipped | `4cccf6e` |
| HF `config.json` parser (architecture matcher, M-RoPE, YaRN, dense detection) | shipped + tested against real Qwen3.6-27B config | `9f9100a` |
| Inline parse test (`qwen36_parse_tests::parse_qwen36_27b_hf_config`) | green | `9f9100a` |
| Engine `run_dense_ffn_sublayer` covers every layer when `dense_layer_idx == num_layers` | confirmed (no engine code change needed) | — |
| `convert.rs` tensor-name map handles `model.layers.N.mlp.{up,gate,down}_proj` → segments 17/18/19 | confirmed | — |
| `tools/smoke_qwen36_27b.sh` end-to-end smoke harness | shipped | (this commit) |
| BF16-native generation (text-only) | code paths in place; **untested without weights** | — |

## Quickstart (when shards land)

```bash
hf download Qwen/Qwen3.6-27B \
  --local-dir /data/huggingface/vib3-models/qwen3.6-27b-hf
./tools/smoke_qwen36_27b.sh                 # build + convert + inspect + run
# or step by step:
STEP=convert ./tools/smoke_qwen36_27b.sh
STEP=run     ./tools/smoke_qwen36_27b.sh
```

## "Beat vLLM" lever — runtime BF16 → NVFP4 for the dense FFN

### Why
vLLM serves Qwen3.6-27B at BF16 (~56 GB on the 96 GB Blackwell). NVFP4 weight
storage cuts the FFN tile size 4× and feeds the sm_120a block-scaled MMA path
that vib3 already uses for `lm_head` and the Qwen3.5 MoE experts. Net: same
quality target, faster GEMV per FFN call.

### Surface area (3 changes, all reuse existing infrastructure)

#### 1. BF16 → NVFP4 conversion at first weight load

The existing `kernels::fp16_to_nvfp4_weight` (`src/compute/kernels.rs:1289`)
takes FP16 input. Two acceptable paths:

- **Convert on disk** — extend `vic-convert --quantize nvfp4` to handle BF16
  source for *only segments 17/18/19* (DeltaNet projections must stay BF16
  to avoid the regression that already gates `VIB3_ALLOW_UNSAFE_DELTANET_NVFP4`).
  Pros: zero runtime conversion cost. Cons: per-segment quant is new in the
  converter; needs a `--quantize-segments 17,18,19` style flag.
- **Convert at runtime on device** (recommended for first cut, mirrors lm_head):
  add a `bf16_to_fp16_device` kernel (host-side `convert_bf16_to_fp16` exists
  at `kernels.rs:4368`; promote to device), then chain the existing
  `fp16_to_nvfp4_weight`. Cache the NVFP4 buffer in
  `shared_tensor_cache_device` keyed by `(layer, segment)` exactly like
  `lm_head_nvfp4` is cached today.

Code site: `src/runtime/engine.rs::ensure_shared_tensor_device`
(around line 9380–9454). After `copy_from_host`, branch on
`segment ∈ {17, 18, 19} && self.model_config.architecture == "qwen3_6_dense"
&& env_flag("VIB3_DENSE_FFN_NVFP4")` and replace the cached buffer with the
NVFP4-converted version.

#### 2. Dense FFN dispatch picks NVFP4 dtype

`src/runtime/engine.rs::run_dense_ffn_sublayer` (line ~6281) calls
`kernels::partial_swiglu` and `kernels::partial_matmul` with the weight ptr
+ `DType`. Both already dispatch `DType::NVFP4` to the FP16-input NVFP4
GEMV (`vib3_launch_partial_matmul_nvfp4_fp16in`,
`src/compute/kernels.rs:112–130`). The change is a one-liner per call site:
pass `DType::NVFP4` instead of `DType::FP16` when the cached weight is NVFP4.

The NVFP4 dispatch keeps activations FP16 (the kernel quantizes per-tile
internally), so no per-call activation quant pass is needed — unlike the
lm_head path that pre-quantizes hidden_state to FP4.

#### 3. Gate behind an env var

`VIB3_DENSE_FFN_NVFP4=1` opt-in, mirroring `VIB3_ALLOW_UNSAFE_NVFP4_EXPERTS`.
Default OFF until quality is bench-validated against BF16 reference logits
on a few prompts (cosine ≥ 0.999, top-1 token match).

### Expected speedup

NVFP4 GEMV on sm_120a's block-scaled MMA on a (5120, 17408) FFN tile is
~3-4× the BF16 throughput in the existing `lm_head` and Qwen3.5 expert
benches. Per-layer FFN dominates dense decode — call it ~1.8× end-to-end on
text-only decode after Amdahl's law (attention + tokenization + sample
overhead).

## Other deferred items

- **YaRN frequency rescaling** (1M context). Fields parsed in `ModelConfig`
  (`rope_yarn_factor`, `rope_original_max_position`); kernel side needs a
  scaled freq table generation. Only matters past 262 k.
- **M-RoPE for VLM**. Sections `[11, 11, 10]` carried as metadata. For
  text-only inference T=H=W=position so it's standard partial RoPE. When
  vision tower lands, `apply_rope` needs a 3-axis variant.
- **Vision tower** (27-layer ViT, hidden 1152, patch 16, projects to 5120).
  Required for image/video inputs; pure-text inference doesn't need it.
- **MTP head** (`mtp_num_hidden_layers=1`). Same shape as Qwen3.5; speculative
  decode integration is a separate project.

## Risks / unknowns

- **Tensor naming under VLM safetensors**: confirmed `model.layers.N.mlp.*`
  for the text path against the published config layout. If the actual
  shards prefix with `text_model.layers.N.mlp.*` (some VLMs do), `convert.rs`
  needs an additional regex. Will surface immediately on the first
  `vic-convert` run.
- **MTP layers as additional model.layers**: the 27B may ship 64 + 1 = 65
  layers in the safetensors. The MTP layer is currently skipped by setting
  `num_hidden_layers = 64` from config; verify the tensor-name walker
  doesn't try to process layer 64.
- **Dense + DeltaNet end-to-end correctness**: the vib3 dense FFN path was
  designed for DeepSeek's pre-MoE first layer; running it for 64 hybrid
  layers in a row is new ground. Quality risk at first-light.
