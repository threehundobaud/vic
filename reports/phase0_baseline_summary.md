# Phase 0 Baseline Report (Captured)

## Run Metadata
- Date: 2026-03-03
- Commit: `2b5d853`
- GPU: Not available to vib3 during capture (`cudaSetDevice(0) failed: out of memory`)
- Driver/CUDA: N/A in captured logs (CUDA init failed)
- Model: 35B (`/code/qwen35-35b-a3b-fresh-fixed3.vib3`) and 122B (`/code/qwen35-122b.vib3`)
- Tokenizer: `/code/qwen35-official-tokenizer/tokenizer.json`
- vib3 flags: `VIB3_DIAG=0 RUST_LOG=info`
- Harness:
  - `python3 tools/phase0_eval.py --base-url http://127.0.0.1:8122/v1 --model default --runs 24 --concurrency 4 --max-tokens 64 --temperature 0 --output reports/phase0_baseline_35b.json`
  - `python3 tools/phase0_eval.py --base-url http://127.0.0.1:8123/v1 --model default --runs 8 --concurrency 1 --max-tokens 32 --temperature 0 --output reports/phase0_baseline_122b.json`

## Smoke Results
- 35B all_passed: `false`
  - `math_one_word`: failed, output=`[Error: CUDA error: f16_to_f32 requires GPU]`
  - `short_coherent`: marked pass by rule but output was same error string
- 122B all_passed: `false`
  - `math_one_word`: failed, output=`[Error: CUDA error: f16_to_f32 requires GPU]`
  - `short_coherent`: marked pass by rule but output was same error string

## Performance Summary (Diagnostic Only)

### 35B (`reports/phase0_baseline_35b.json`)
| Scenario | Runs | Concurrency | Success | Errors | req/s | tok/s | p95 latency (ms) | p95 TTFT (ms) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| short | 24 | 4 | 24 | 0 | 342.106 | 2052.639 | 15.46 | 15.43 |
| mixed | 24 | 4 | 24 | 0 | 339.076 | 2034.457 | 15.07 | 15.01 |
| long | 24 | 4 | 24 | 0 | 334.455 | 2006.727 | 14.61 | 14.55 |

### 122B (`reports/phase0_baseline_122b.json`)
| Scenario | Runs | Concurrency | Success | Errors | req/s | tok/s | p95 latency (ms) | p95 TTFT (ms) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| short | 8 | 1 | 8 | 0 | 93.769 | 562.612 | 14.38 | 14.20 |
| mixed | 8 | 1 | 8 | 0 | 104.859 | 629.155 | 14.30 | 14.12 |
| long | 8 | 1 | 8 | 0 | 108.589 | 651.535 | 12.38 | 12.27 |

## Observations
- Stability: server stayed up for harness requests in both runs.
- Quality: Phase 0 smoke failed for both models due to GPU-required conversion path surfacing in outputs.
- Throughput: collected values are not valid production throughput baselines because responses are error strings.
- Primary bottleneck hypothesis: GPU unavailability/overcommit causes CPU fallback + GPU-required op errors (`f16_to_f32 requires GPU`).

## Go/No-Go
- Phase 0 exit criteria satisfied: **No**
- Blockers:
  1. Runtime enters CPU fallback while model path still executes GPU-required operations.
  2. Smoke correctness gate fails (`math_one_word`).
  3. Need clean GPU-resident run before Phase 1 scheduler work can be meaningfully benchmarked.

## Post-Guard Update (2026-03-04)
- API behavior hardened: streaming failures now return HTTP 500 JSON errors instead of embedding `[Error: ...]` in assistant text.
- Harness smoke gate hardened: error-like outputs are auto-fail.
- Verification artifact: `reports/phase0_baseline_122b_postguard.json`
  - `success=0/2` across perf scenarios under current CPU-fallback state
  - `sample_errors` now correctly report `HTTPError 500` with `CUDA error: f16_to_f32 requires GPU`

## 122B Isolated Re-Run (2026-03-04, 35B stopped)
- Environment: 35B process stopped; stale GPU allocation removed before test; 122B served alone on `http://127.0.0.1:8132/v1`.
- Verification artifact: `reports/phase0_quick_122b_8132_solo.json`
- Smoke:
  - `all_passed=false`
  - `math_one_word`: failed, content echoed prompt-like text (`What is 2+2. Answer with one word.`)
  - `short_coherent`: technically passed rule, but content quality invalid (`The 1000000000000000000000`)
- Performance (diagnostic only due smoke failure):
  - `short`: `req/s=0.181`, `tok/s=2.196`, `p95 latency=6332.48ms`
  - `mixed`: `req/s=0.160`, `tok/s=1.517`, `p95 latency=7352.98ms`
  - `long`: `req/s=0.106`, `tok/s=1.668`, `p95 latency=10735.61ms`
- Conclusion: isolated 122B run no longer reproduces the previous mixed-load CUDA launch failure (`err=2`), but default-path output quality remains below Phase 0 correctness gate.

## 122B Final Isolated Re-Run (2026-03-04, clean headroom)
- Environment: local vib3 servers stopped, 122B served alone on `http://127.0.0.1:8137/v1`, GPU free memory ~94.7 GiB at launch.
- Verification artifact: `reports/phase0_quick_122b_8137_final.json`
- Smoke:
  - `all_passed=false`
  - `math_one_word`: failed, output=`The answer is 2+3.`
  - `short_coherent`: pass by current harness rule, but semantically weak output (`The 2015`).
- Performance:
  - `short`: `req/s=0.384`, `tok/s=3.651`, `p95 latency=2959.60ms`
  - `mixed`: `req/s=0.418`, `tok/s=2.089`, `p95 latency=2621.65ms`
  - `long`: `req/s=0.255`, `tok/s=1.104`, `p95 latency=4597.35ms`
- Conclusion: isolated serving is stable (no transport/internal errors in quick run), but correctness gate still fails on deterministic math and output quality remains below Phase 0 exit criteria.

## Decode Divergence Check (35B vs 122B, same prompt/settings)
- Prompt: `What is 2+2? Answer with one word.` with `temperature=0`, `max_tokens=12`, chat-completions.
- 35B reference run (`/tmp/vib3_cmp_35b.log`): first token `26108` (`"Four"`), response correctness pass.
- 122B run (`/tmp/vib3_cmp_122b.log`): first token `760` (`"The"`), then `" answer"`, `" is"`, yielding pattern `"The answer is ..."` and incorrect arithmetic variants.
- Earliest divergence is **pre-sampling at step 0 logits** (top-1 mismatch: 35B top-1=`Four`, 122B top-1=`The`).
- Interpretation: this is not primarily a sampler bug; the decode distribution is already skewed at logits step 0 for 122B.

## Stability Hardening Applied (2026-03-04)
- File: `src/runtime/engine.rs`
- Change: NVFP4 lm_head fast-path failures (e.g., `f16_to_f32 launch failed`) now auto-fallback to FP16/paged lm_head instead of failing the full request.
- Impact: reduces 500-error incidence under headroom pressure and keeps service available while preserving existing fallback behavior.

## 122B lm_head Parity Deep-Dive (2026-03-04, clean single-instance)
- Environment: all prior 122B servers killed, single process on `:8132`, diagnostics enabled (`VIB3_DIAG=1`, `VIB3_DIAG_LMHEAD_PARITY=1`).
- Probe prompt: `What is 2+2? Answer with one word.` with `temperature=0`, `max_tokens=12`.
- Response remained incorrect: `The answer is 2+3.`
- Step-0 diagnostics from `/tmp/vib3_122b_input_parity.log`:
  - `LMHEAD_INPUT_PARITY step=0: cosine=1.000000, rel_l2=0.000000, max_abs=0.000000`
  - `LMHEAD_PARITY step=0: cosine=0.061552, rel_l2=1.421995, max_abs=14.412472, top1_fast=760, top1_ref=247804`
- Interpretation: lm_head fast path does **not** mutate its input vector; mismatch source is in projection/logits computation parity (NVFP4 lm_head path vs reference), not upstream hidden-state corruption.

  ## lm_head Path Isolation Update (2026-03-04)
  - Code fix: eager lm_head conversion policy at engine init now honors Qwen3.5 FP16-default policy (`qwen3_5_moe` defaults to FP16 unless `VIB3_FP16_LM_HEAD=0`).
  - Added diagnostic control: `VIB3_FORCE_PAGED_LM_HEAD=1` to bypass shared/NVFP4 lm_head fast paths.
  - Post-fix confirmation:
    - Normal mode reports `LMHEAD_PARITY ... path=fp16_shared_fast` (no longer `nvfp4_fast` by default).
    - For prompt `What is 2+2? Answer with one word.` at `temperature=0`, `max_tokens=1`:
      - `fp16_shared_fast` output token: `id=16` (`"1"`), parity still poor (`cosine=0.075045`, `rel_l2=1.373427`).
      - `paged_forced` output token: `id=59` (`"\\"`).
  - Interpretation: mismatch persists after removing unintended NVFP4 default, and A/B token divergence confirms parity issue remains in shared FP16 lm_head fast path versus paged reference.

  ## lm_head Row/Disk Parity Update (2026-03-04)
  - Added row-level lm_head diagnostics at step 0:
    - `LMHEAD_WEIGHT_PARITY row=<id>` compares shared contiguous lm_head row bytes vs paged row bytes.
    - `LMHEAD_WEIGHT_PARITY_DISK row=<id>` compares both sources to on-disk page bytes (`read_page_sync`).
  - Clean short probe findings (`2+2?`, `max_tokens=1`):
    - Rows `0,1,16,17,59` matched exactly.
    - Rows `760` and `247804` were complete mismatches (`exact_bits=0/3072`).
    - Disk truth showed paged rows were exact (`paged_exact=3072/3072`) while shared rows were not (`shared_exact=0/3072`) for both mismatched rows.
  - Added `LMHEAD_GUARD` (skip direct device assembly for `L0/S11`, host-mediated default) and validated it was active in logs.
    - Result: mismatched rows persisted unchanged.
  - Interpretation: current evidence points to corruption/mapping error in the shared contiguous lm_head path itself (`fp16_shared_fast`), not in lm_head input vectors and not only in the direct device assembly fast path.

## FP16 lm_head Default Routing Mitigation (2026-03-04)
- Code change (`src/runtime/engine.rs`): FP16 shared lm_head fast path is now opt-in only.
  - New gate: `VIB3_ALLOW_FP16_SHARED_FAST_LM_HEAD=1`.
  - Default behavior: when NVFP4 is not used, decode falls back to paged lm_head logits instead of `fp16_shared_fast`.
- Validation run (122B, single instance on `:8132`, `VIB3_DIAG=1 VIB3_DIAG_LMHEAD_PARITY=1`):
  - Probe: `2+2?`, `max_tokens=1`, `temperature=0`.
  - Log evidence: `LMHEAD_GUARD: FP16 shared fast lm_head disabled by default; using paged lm_head...`
  - No `path=fp16_shared_fast` observed for the default run.
  - Request completed successfully (one-token response), confirming correctness-first routing does not break serving.

## Post-Mitigation Baseline Refresh (2026-03-04)
- 35B run (artifact: `reports/phase0_baseline_35b_postmit.json`)
  - Config: `runs=8`, `concurrency=2`, `max_tokens=64`, `temperature=0`.
  - Smoke: `all_passed=true` (`math_one_word=Four`, coherent short answer present).
  - Perf: short/mixed/long each `success=8/8`, no sample errors.
- 122B run (artifact: `reports/phase0_baseline_122b_postmit.json`)
  - Config: `runs=2`, `concurrency=1`, `max_tokens=16`, `temperature=0`.
  - Smoke: `all_passed=false` (both smoke cases returned HTTP 500).
  - Perf: short/mixed/long each `success=0/2`, `tok_s=0.0`.
  - Error signature: `CUDA error: f16_to_f32 launch failed (err=2)`.
  - Correlated server log (`/tmp/vib3_phase0_122b_postmit.log`) repeatedly showed
    `Failed to allocate device buffer for shared tensor L0 S11: cudaMalloc(1525678080 bytes) failed: out of memory`.
- Interpretation: lm_head routing guard improves correctness posture and 35B remains healthy, but 122B is still blocked by VRAM headroom / kernel launch instability in this environment.

## VRAM Allocation Root Cause Clarification (2026-03-04)
- During reported `L0/S11` allocation failures, live GPU state was overcommitted by concurrent processes:
  - `vib3-serve` (~70.1 GiB)
  - `vib3-serve` (~23.3 GiB)
  - root `python` (~2.5 GiB)
  - free VRAM observed: ~1.2 GiB
- `L0/S11` allocation request size is `1525678080` bytes (~1.42 GiB), so failure was expected under that headroom.
- After terminating duplicate `vib3-serve` processes, free VRAM returned to ~94.7 GiB, and a clean single 122B probe on `:8123` completed without the prior `L0/S11` allocation warning.
- Diagnostic improvement landed in `src/runtime/engine.rs`: shared tensor allocation failures now include `need=<MiB>` and live `free=<MiB>` to distinguish true model-fit issues from concurrent-process contention.
