# vib3 Throughput MVP (Rust/CUDA)

## Objective
Build a correctness-first, high-throughput serving path for vib3 by combining:
- **llama.cpp-style defaults**: one conservative, validated default path.
- **vLLM-style scheduling**: continuous batching, prefill/decode separation, KV-aware admission.

## Non-Goals (MVP)
- No full architecture rewrite.
- No speculative kernels enabled by default.
- No multi-GPU/tensor-parallel redesign in MVP.

## Success Criteria (MVP Exit)
1. **Correctness gate** (default config): no gibberish regressions on pinned smoke prompts for 35B and 122B.
2. **Stability gate**: no segfaults / CUDA launch failures in 100 short requests at `temperature=0`.
3. **Throughput uplift**: measurable tok/s and req/s improvement vs current baseline under mixed request lengths.
4. **Deterministic fallback**: when fast paths are disabled, default path remains healthy and coherent.

## Baseline Contract (Must Hold Before/After Each Phase)
- Conservative default execution path is always available.
- Risky/experimental paths are opt-in via env flags.
- Admission control is fail-closed on VRAM budget violations.
- New optimization paths require parity checks against reference/default outputs.

## Phased Plan

### Phase 0 — Measurement & Guardrails
**Deliverables**
- Standard benchmark harness (short, mixed, long prompts; 35B/122B).
- Baseline metrics capture: TTFT, tok/s, req/s, p95 latency, OOM/error counts.
- Minimal quality smoke suite (deterministic prompts + token sanity checks).

**Acceptance**
- Baseline report checked into repo.
- CI/local script can run smoke + perf quick pass.

### Phase 1 — Scheduler MVP (Highest ROI)
**Deliverables**
- Iteration-level continuous batching.
- Explicit split of prefill and decode lanes.
- Admission control based on KV budget + scratch budget + safety margin.
- Fairness policy: short-job bias with starvation cap.

**Acceptance**
- No quality regression vs baseline contract.
- Improved req/s on mixed-load benchmark.

### Phase 2 — KV & Memory Governor
**Deliverables**
- Paged KV with bounded fragmentation policy.
- Prefix reuse/prefix caching with explicit invalidation rules.
- VRAM governor model: `weights + kv + workspace + transfer buffers + reserve`.

**Acceptance**
- Eliminate launch-failure class caused by overcommit.
- Stable 100-request runs at configured concurrency.

### Phase 3 — CUDA Overlap Pipeline
**Deliverables**
- Stream topology: dedicated streams for H2D, compute, D2H.
- Double-buffer staging for expert/page transfers.
- Event-driven dependencies to overlap transfer and compute.

**Acceptance**
- Improved TTFT and decode tok/s under concurrent requests.
- No correctness drift under overlap-enabled path.

### Phase 4 — MoE Throughput Path
**Deliverables**
- Grouped expert GEMMs per step (launch amortization).
- Batch packing by active experts to increase tile utilization.
- Conservative cap knobs to prevent memory spikes.

**Acceptance**
- Throughput uplift on MoE-heavy prompts.
- Stable defaults remain coherent without fast-path flags.

### Phase 5 — Graph Capture (Safe Subset)
**Deliverables**
- Capture stable decode shapes where legal.
- Automatic fallback for dynamic/outlier shapes.

**Acceptance**
- Additional latency reduction without new failure modes.

## Work Breakdown (Concrete MVP Tasks)
1. Implement benchmark runner and reporting schema.
2. Add scheduler interfaces (`prefill_queue`, `decode_queue`, admission API).
3. Add KV budget estimator and guardrail enforcement.
4. Integrate continuous batching loop into runtime engine.
5. Add smoke-quality gate and default-vs-fast-path parity check.
6. Add CUDA stream/event overlap for transfer/compute.
7. Add grouped expert execution and packing heuristics.
8. Document runtime knobs and safe defaults in README.

## Risk Register
- **Risk:** Throughput optimizations silently degrade quality.
  - **Mitigation:** parity checks + smoke prompts as a hard gate.
- **Risk:** VRAM overcommit causes intermittent CUDA errors.
  - **Mitigation:** hard admission guard + reserve margin.
- **Risk:** Scheduler improves req/s but hurts p95 latency.
  - **Mitigation:** fairness constraints and queue aging caps.

## Recommended Default Runtime Policy
- Default: conservative kernels + guarded scheduler + strict memory admission.
- Opt-in: aggressive projection/preassembly/conversion or experimental fused kernels.
- On any parity failure: auto-disable offending fast path and continue serving.

## Tracking
This document is the source-of-truth MVP plan. Execution should be tracked by phase, with each phase marked complete only when its acceptance criteria pass on both 35B and 122B.

## Progress Board

| Phase | Status | Notes |
|---|---|---|
| Phase 0 — Measurement & Guardrails | In Progress (Blocked) | Harness + reports complete; fail-closed API guard added; final isolated 122B rerun completed under clean headroom with stable serving, but smoke correctness still fails (deterministic math wrong + weak coherence output). |
| Phase 1 — Scheduler MVP | Not Started | Pending Phase 0 baseline lock. |
| Phase 2 — KV & Memory Governor | Not Started | Pending scheduler interfaces. |
| Phase 3 — CUDA Overlap Pipeline | Not Started | Pending stable scheduler + governor. |
| Phase 4 — MoE Throughput Path | Not Started | Pending overlap plumbing. |
| Phase 5 — Graph Capture (Safe Subset) | Not Started | Final optimization pass after stability. |

## Phase 0 Attempt Log (Chronological)

- 2026-03-03: Added baseline harness + report workflow (`tools/phase0_eval.py`, reports templates).
- 2026-03-03: Initial runs invalidated by CPU-fallback error-text completions (`f16_to_f32 requires GPU`).
- 2026-03-04: API hardened to fail-closed (HTTP 500 JSON) instead of embedding internal errors in generated content.
- 2026-03-04: Added CPU fallbacks for key kernels to reduce hard failure surface in CPU mode.
- 2026-03-04: Enforced isolated model testing (no 35B+122B concurrent runs).
- 2026-03-04: 35B isolated smoke stabilized and returned coherent deterministic output (`Four`).
- 2026-03-04: 122B isolated quick runs remained below correctness gate (math wrong / degraded coherence).
- 2026-03-04: Sampler fix landed: repetition penalty was previously defined but not applied; token-history-aware penalty now active.
- 2026-03-04: 35B vs 122B decode divergence check completed on same prompt/settings.
  - Earliest divergence appears at step-0 logits (pre-sampling): 35B top-1=`Four` vs 122B top-1=`The`.
- 2026-03-04: Stability hardening landed for lm_head fast path.
  - NVFP4 lm_head path failures now fallback to FP16/paged lm_head instead of failing the request.
- 2026-03-04: Added `VIB3_DIAG_LMHEAD_PARITY=1` diagnostic to compare fast-path logits vs paged-lm_head reference at step 0.
  - Observed severe mismatch on 122B (`cosine≈0.0616`, `rel_l2≈1.422`, top1 mismatch), indicating lm_head fast-path parity issue.
- 2026-03-04: Changed default policy for Qwen3.5 to FP16 lm_head (NVFP4 lm_head now opt-in via env override).
  - Goal: prioritize correctness over throughput while Phase 0 gate is still failing.
- 2026-03-04: Added `LMHEAD_INPUT_PARITY` step-0 diagnostic (`VIB3_DIAG_LMHEAD_PARITY=1`) to verify fast lm_head path does not mutate lm_head input.
  - Clean isolated 122B run result: `LMHEAD_INPUT_PARITY cosine=1.0, rel_l2=0.0, max_abs=0.0` while `LMHEAD_PARITY` remained severely mismatched (`cosine≈0.0616`, top1 mismatch).
  - Conclusion: step-0 divergence is in projection/logits path parity, not upstream hidden-state corruption.
- 2026-03-04: Fixed policy bug in eager lm_head conversion at engine init.
  - Root cause: load-time conversion still enabled `nvfp4_fast` by default for Qwen3.5, bypassing the intended FP16 default.
  - Fix: eager conversion now uses the same default policy as decode (`qwen3_5_moe` defaults to FP16 unless `VIB3_FP16_LM_HEAD=0`).
- 2026-03-04: Added `VIB3_FORCE_PAGED_LM_HEAD=1` control path and ran A/B on identical prompt/settings.
  - Prompt: `What is 2+2? Answer with one word.` / `temperature=0` / `max_tokens=1`.
  - `fp16_shared_fast`: token `id=16` (`"1"`), `LMHEAD_PARITY cosine≈0.075, rel_l2≈1.373`.
  - `paged_forced`: token `id=59` (`"\\"`).
  - Conclusion: shared FP16 lm_head path still diverges materially from paged reference; mismatch is not limited to NVFP4.
- 2026-03-04: Added row-level lm_head parity + on-disk reference diagnostics.
  - `LMHEAD_INPUT_PARITY` remains perfect (`cos=1.0`), so lm_head input vector is stable.
  - For sampled rows, early rows matched (`0,1,16,17,59`), but rows `760` and `247804` were full mismatches (`exact_bits=0/3072`).
  - On-disk ground truth check: paged rows were exact (`paged_exact=3072/3072`), shared rows were not (`shared_exact=0/3072`) for both mismatched rows.
- 2026-03-04: Added `LMHEAD_GUARD` to skip direct device assembly of L0/S11 and force host-mediated path by default.
  - Guard confirmed active in logs, but mismatched rows persisted unchanged.
  - Interpretation: corruption is not limited to direct device assembly; issue remains in the shared contiguous lm_head path used by `fp16_shared_fast`.
- 2026-03-04: Implemented correctness-first default for FP16 lm_head decode path.
  - Change: `fp16_shared_fast` is now explicit opt-in via `VIB3_ALLOW_FP16_SHARED_FAST_LM_HEAD=1`.
  - Default behavior: FP16 lm_head decode uses trusted paged logits path even when shared contiguous lm_head is available.
  - Validation (122B, `:8132`, `VIB3_DIAG=1 VIB3_DIAG_LMHEAD_PARITY=1`):
    - Log confirms guard activation: `LMHEAD_GUARD: FP16 shared fast lm_head disabled by default; using paged lm_head...`
    - No `path=fp16_shared_fast` in step-0 parity logs under default settings.
    - Probe remained stable (`2+2?` one-token completion returned without request failure).
- 2026-03-04: Ran fresh post-mitigation Phase 0 short baselines.
  - 35B artifact: `reports/phase0_baseline_35b_postmit.json` (`runs=8`, `concurrency=2`) → smoke `all_passed=true` (`math_one_word=Four`, coherent short output present), perf scenarios `success=8/8` with no sample errors.
  - 122B artifact: `reports/phase0_baseline_122b_postmit.json` (`runs=2`, `concurrency=1`) → smoke `all_passed=false`, all perf scenarios `success=0/2` with repeated `HTTP 500` / `CUDA error: f16_to_f32 launch failed (err=2)`.
  - Correlated server log (`/tmp/vib3_phase0_122b_postmit.log`) repeatedly showed `cudaMalloc(1525678080 bytes) failed: out of memory` on `L0/S11` immediately before request failures.
- 2026-03-04: Identified concrete VRAM contention root cause for recent `L0/S11` allocation failures.
  - Live `nvidia-smi` during failures showed concurrent GPU consumers: two `vib3-serve` processes (`~70.1 GiB` + `~23.3 GiB`) plus a root-owned Python process (`~2.5 GiB`), leaving only `~1.2 GiB` free.
  - `L0/S11` requires ~`1.42 GiB` contiguous allocation (`1525678080` bytes), so allocation failure was expected under that state.
  - After killing duplicate `vib3-serve` processes, free VRAM returned to ~`94.7 GiB`; single 122B probe on `:8123` succeeded (`2+2?` → `2`) without `L0/S11` alloc errors.
  - Added runtime diagnostics so allocation failures now log both `need` and live `free` VRAM in MiB.

Detailed artifacts and metrics are tracked in `reports/phase0_baseline_summary.md`.

## Phase 0 Execution Commands

1) Run smoke + perf against local server:

```bash
python3 tools/phase0_eval.py \
  --base-url http://127.0.0.1:8122/v1 \
  --model default \
  --runs 24 \
  --concurrency 4 \
  --max-tokens 64 \
  --temperature 0 \
  --output reports/phase0_baseline_35b.json
```

2) Run a 122B baseline pass:

```bash
python3 tools/phase0_eval.py \
  --base-url http://127.0.0.1:8123/v1 \
  --model default \
  --runs 24 \
  --concurrency 2 \
  --max-tokens 64 \
  --temperature 0 \
  --output reports/phase0_baseline_122b.json
```

3) Record summary in:
- `reports/phase0_baseline_template.md`
- `reports/phase0_baseline_summary.md`
