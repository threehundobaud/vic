# vib3 Refinement Log

Last updated: 2026-02-25

## Current State

Working tree clean. 6 commits on master. Release and debug binaries built.

### Git History
```
9226e41 Fix converter regression: INT4 quantizer wrote BF16 scales, all consumers expect FP16
625b518 Revert w1/w3 segment swap: reconverted model produces NaN
1035fbf Fix w1/w3 segment mapping in converter; remove engine call-site swap
3f9713b Update docs with bug fix status; gate diagnostics behind VIB3_DIAG=1
2c7e294 Fix two critical MoE bugs: router normalization and gate/up swap
3a119ec Initial commit: vib3 inference engine with INT4 FP16/BF16 scale fix
```

### Model Files on Disk (1.3TB used, 532GB free)
| File | Size | Status |
|------|------|--------|
| `mixtral-full-v2.vib3` | 23GB | Working. Converted Feb 12 with old code. THE reference model. |
| `mixtral-v3.vib3` | 23GB | Working. Reconverted Feb 25 with fixed converter (commit 9226e41). Verified identical output. |
| `mixtral-fp16.vib3` | 87GB | FP16 Mixtral. Has page loading errors at L5. Not usable. |
| `mixtral-full.vib3` | 23GB | Old conversion. Probably same as v2. Could delete. |
| `mixtral-reconv.vib3` | 23GB | Old reconversion attempt. Pre-fix. Could delete. |
| `mixtral.vib3` | 1.4GB | Tiny/partial test model. |

**Cleanup opportunity:** `mixtral-full.vib3`, `mixtral-reconv.vib3`, and `mixtral-fp16.vib3` are ~133GB of likely-dead models. Deleting them would free disk for Kimi or Qwen conversions.

---

## Bugs Fixed (4 total, all committed)

### Bug 1: FP16/BF16 Scale Mismatch in CUDA Kernels (commit 3a119ec)
INT4 CUDA kernels decoded quantization scales using `bf16_to_float()` but the working .vib3 file stored FP16 scales. Fixed by adding `fp16_scale_to_float()` in `kernels.cu` at 4 call sites.

### Bug 2: Softmax Router Top-K Normalization Missing (commit 2c7e294)
`run_router_f32()` in `kernels.rs` computed softmax over ALL experts, selected top-k, but didn't renormalize to sum to 1.0. For Mixtral top-2, weights summed to ~0.49 instead of 1.0. Fixed by adding normalization block after `indexed.truncate(top_k)`.

### Bug 3: SwiGLU Gate/Up Projection Swap (commit 2c7e294)
Mixtral's `w1 = gate_proj` (SiLU input), `w3 = up_proj`. But `convert.rs` maps `w1 -> segment 0 ("up")` and `w3 -> segment 1 ("gate")`. Fixed by swapping arguments at call site in `execute_expert()` (engine.rs). The naming in convert.rs remains wrong but is compensated by the engine swap.

### Bug 4: Converter Regression / INT4 BF16-vs-FP16 Scale Write (commit 9226e41)
`quantize_weights_to_int4()` wrote scales as BF16 (`f32_to_bf16`), but CUDA kernel reads FP16 (`fp16_scale_to_float`). Every other quantization path (NVFP4, NF4, INT8) already wrote FP16. The INT4 path was the sole outlier. BF16 bits interpreted as FP16 produce ~1000x scale errors, causing inf/NaN at the SwiGLU layer.

Fixed 8 sites across 4 files:
- INT4 quantizer write: `f32_to_bf16` -> `f16::from_f32` (kernels.rs:2493)
- CPU matmul read: `bf16_to_f32` -> `f16::from_bits().to_f32()` (kernels.rs:1478)
- CPU swiglu reads (2 sites): same change (kernels.rs:1674, 1688)
- `convert_int4_to_nvfp4` reads (2 sites): same change (kernels.rs:2684, 2695)
- Engine diagnostics: decode as FP16 instead of BF16 (engine.rs:3584)
- Test constants: BF16 `0x3f80`/`0x3f00` -> FP16 `0x3C00`/`0x3800` (integration_test.rs)

**This unblocks all new model conversions.** The reconverted Mixtral (`mixtral-v3.vib3`) produces identical output to the original Feb 12 model.

---

## Performance Benchmarks (Mixtral-8x7B INT4, RTX PRO 6000 Blackwell 96GB)

### Throughput
| Metric | Debug Build | Release Build |
|--------|------------|---------------|
| Decode tok/s | ~20.4 | ~21.5 |
| TTFT | ~1500ms | ~1489ms |
| T1 hit rate | 100% | 100% |
| Preload speed | 1155 MB/s | 1155 MB/s |
| Preload time | ~24s (28GB) | ~24s (28GB) |

Release build is only ~5% faster than debug. **The bottleneck is CUDA kernel execution, not Rust overhead.**

### GPU Utilization During Inference
| Metric | Value |
|--------|-------|
| SM Utilization | 91-94% |
| Memory Bandwidth Util | 15-20% |
| Power Draw | 261-293W (of 600W TDP) |
| GPU Temp | 41-44C |

**Analysis:** SM utilization is high because the GPU is constantly busy with kernel launches, but power draw is only ~45% of TDP. This means the individual kernels are too small to saturate the ALUs. Each expert matmul is `[1, 4096] x [14336, 4096]^T` (~58M FLOPs) -- tiny by GPU standards. With 192 kernel launches per token (2 experts x 3 matmuls x 32 layers), kernel launch overhead and small-workload inefficiency dominate.

This is the fundamental batch-size-1 MoE problem. Possible mitigations:
1. **Batched inference** -- multiple requests would saturate ALUs
2. **Kernel fusion** -- fuse router + matmul + SiLU + mul into fewer launches
3. **Persistent kernels** -- keep SMs occupied across layers
4. **cuBLAS/CUTLASS** -- use optimized GEMM implementations instead of hand-written kernels

### Verified Output Quality
| Prompt | Output |
|--------|--------|
| "What is 2+2?" | "The sum of 2 and 2 is 4." |
| "What is the capital of France?" | "The capital of France is Paris." |

Longer prompts (reasoning, code, creative, logic, factual) all produce coherent output. Full benchmark suite was interrupted mid-run -- needs completion next session.

---

## Known Issues (Not Yet Fixed)

### Converter: w1/w3 Segment Naming Still Wrong
`convert.rs` maps `w1 -> segment 0 ("up")` and `w3 -> segment 1 ("gate")`. This is semantically backwards for Mixtral (w1=gate_proj, w3=up_proj). The engine compensates with a call-site argument swap in `execute_expert()` at engine.rs:3610-3614.

**This breaks Kimi K2.5**, which uses `up_proj`/`gate_proj` naming directly. The unconditional swap applies SiLU to `up_proj` instead of `gate_proj` for Kimi. This is likely contributing to the L6 MoE explosion. The fix should either:
- Make the swap conditional on model architecture, OR
- Fix the converter mapping and remove the swap

### Kimi K2.5 L6 Explosion (Separate from Mixtral bugs)
Kimi MoE output explodes at L6 (cosine -0.092, L2 91.8 vs GT 2.85). The three Mixtral fixes don't fully apply (different code paths). The w1/w3 swap issue above likely contributes. Needs investigation with a valid Kimi .vib3 file (previous conversion failed due to disk space, and converter was broken).

### Kimi K2.5 Conversion Not Yet Attempted with Fixed Converter
The previous NVFP4 conversion attempt failed with "No space left on device" at shard 54/64. The converter regression (now fixed) would have also produced corrupt output. Need to:
1. Free disk space (delete dead model files)
2. Re-run conversion with fixed converter
3. Test inference

### FP16 Mixtral Model Has Page Loading Errors
`mixtral-fp16.vib3` (87GB) fails at L5 with page loading errors. Not investigated. Low priority since INT4 works fine.

### CPU Fallback Path Untested for INT4
The CPU fallback now reads FP16 scales (matching GPU path), but there's no integration test that exercises CPU INT4 matmul end-to-end. The `test_mixtral_cpu_decode_smoke` test is ignored.

### `dequantize_int4_to_f32()` is Dead Code
The function exists with a `scale_format` parameter but has zero callers. Could be removed or wired up as a test utility.

---

## Architecture Notes for Next Session

### Key Code Paths (modified files)
- `src/compute/kernels.rs` -- All kernel dispatch. Router at ~775. INT4 quantizer at ~2468. CPU fallbacks at ~1450, ~1620. `INT4_GROUP_SIZE = 32` at line 2416.
- `src/runtime/engine.rs` -- SwiGLU gate/up swap at ~3610. Diagnostics gated behind `VIB3_DIAG=1` (field at ~248, init at ~759).
- `src/storage/convert.rs` -- `classify_tensor()` at ~1615. w1/w3 mapping at ~1630. `combine_packed_and_scales()` at ~1389.
- `cuda/src/kernels.cu` -- `fp16_scale_to_float()` at ~37. INT4 dequant kernel at ~106. `vib3_silu_mul` at ~733.
- `tests/integration_test.rs` -- INT4 matmul tests at ~1709, ~1788 (updated FP16 scale constants).

### SiLU/Mul Kernel Argument Order (Don't Touch Without Tracing)
The C wrapper `vib3_launch_silu_mul(up_result, gate_result)` passes `up_result` as the FIRST arg to the kernel, where SiLU is applied. So `SiLU(up_result) * gate_result` -- the C naming is backwards. The Rust side in `kernels.rs` compensates by passing `gate_result` first and `up_result` second. This is correct but confusing.

### Two Quantization Paths in Converter
- **Path A (compressed-tensors passthrough):** Model has `_packed`/`_scale` tensor pairs. Scales are copied verbatim (FP16 from HuggingFace). Used for pre-quantized models.
- **Path B (quantize from scratch):** Model has raw BF16/FP16 weights. Converter quantizes to INT4 and writes FP16 scales (after Bug 4 fix). Used for Mixtral.

### Qwen3.5-397B-A17B as Alternative Target
Researched but not started. Key points:
- Half the params of Kimi (397B vs 1T), half activated (17B vs 32B)
- 75% DeltaNet layers (linear attention, no KV cache) -- needs new kernel
- 512 experts at 1024 intermediate dim (each ~4x smaller than Kimi's)
- Native MTP (multi-token prediction) for speculative decoding
- Would require: DeltaNet kernel, updated converter, possibly new routing logic

### Run Commands
```bash
# Inference (release)
echo "prompt" | RUST_LOG=error /home/brian/code/vib3/target/release/vib3 run \
  /home/brian/code/vib3/models/mixtral-v3.vib3 \
  --tokenizer /home/brian/code/vib3/models/mixtral/tokenizer.json \
  --max-tokens 100 --temperature 0.0 --no-check 2>&1

# With diagnostics
echo "Hi" | VIB3_DIAG=1 RUST_LOG=warn /home/brian/code/vib3/target/release/vib3 run \
  /home/brian/code/vib3/models/mixtral-v3.vib3 \
  --tokenizer /home/brian/code/vib3/models/mixtral/tokenizer.json \
  --max-tokens 1 --temperature 0.0 --no-check 2>&1

# Convert model
/home/brian/code/vib3/target/release/vib3-convert \
  --model /path/to/safetensors/dir \
  --output /path/to/output.vib3 \
  --quantize int4  # or nvfp4

# Tests
cd /home/brian/code/vib3 && cargo test

# GPU monitoring during inference
nvidia-smi dmon -s pu -d 1
```

---

## Next Steps (Priority Order)

1. **Complete benchmark suite** -- interrupted mid-run. Run diverse prompts on release build, collect tok/s for V3 vs V2, longer generations.
2. **Delete dead model files** -- `mixtral-full.vib3`, `mixtral-reconv.vib3`, `mixtral-fp16.vib3` = ~133GB.
3. **Fix w1/w3 conditional swap for Kimi** -- make the SwiGLU argument swap architecture-aware so it doesn't break Kimi K2.5.
4. **Re-attempt Kimi K2.5 NVFP4 conversion** -- with fixed converter and freed disk space.
5. **Investigate kernel fusion / performance** -- 91% SM util but only 45% power means small-kernel inefficiency. Low-hanging fruit may exist in reducing kernel launch count.
6. **Evaluate Qwen3.5** -- decide whether to pursue it alongside or instead of Kimi.
