# Qwen3.6-27B port — status & remaining work

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
