#!/usr/bin/env bash
# Qwen3.6-27B (dense, hybrid DeltaNet + GatedAttention) end-to-end smoke
# test on an RTX PRO 6000 Blackwell host.
#
# Released 2026-04-21. 27.78B BF16 params, ~56 GB on disk; fits an RTX PRO
# 6000 (96 GB) comfortably. No official quants — community AWQ/GGUF will
# follow but for first-light we run BF16 directly. NVFP4 conversion of the
# FFN tensors is a follow-up (the lever for beating vLLM on throughput).
#
# Prereqs:
#   - CUDA 12.8+ (sm_120a) — same toolchain as K2.6.
#   - Qwen/Qwen3.6-27B safetensors shards under $QWEN36_HF_DIR.
#     Fetch with:
#       hf download Qwen/Qwen3.6-27B --local-dir "$QWEN36_HF_DIR"
#     ~56 GB across 15 shards + the tokenizer + config.json + chat template.
#   - ≥120 GB free under $VIB3_OUT_DIR for the converted .vib3.
#   - 96 GB VRAM + 64+ GB RAM.
#
# Steps:
#   1. cargo build --release (produces vic, vic-convert, vic-inspect).
#   2. vic-convert HF → .vib3 (--quantize none for BF16-native first light).
#   3. vic-inspect for sanity (segment + page coverage across 64 layers).
#   4. vic run with a canned prompt + --max-tokens 1.
#
# Each step is idempotent — re-run individually with STEP=build|convert|inspect|run.

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

: "${QWEN36_HF_DIR:=/data/huggingface/vib3-models/qwen3.6-27b-hf}"
: "${VIB3_OUT_DIR:=/data/huggingface/vib3-models}"
: "${QWEN36_VIB3:=$VIB3_OUT_DIR/qwen3.6-27b.vib3}"
: "${VIB3_HOME:=$VIB3_OUT_DIR/.vib3-home}"
: "${STEP:=all}"
: "${LOG_DIR:=$VIB3_OUT_DIR/smoke-logs}"
: "${PROMPT:=The capital of France is}"
: "${QUANTIZE:=none}"  # none = BF16-native; nvfp4 = runtime-quant FFN (faster but quality TBD)
mkdir -p "$LOG_DIR" "$VIB3_HOME/models"

BOLD="$(tput bold 2>/dev/null || true)"; RST="$(tput sgr0 2>/dev/null || true)"
section() { echo; echo "${BOLD}=== $* ===${RST}"; }

cmd_build() {
  section "Step 1: cargo build --release"
  if ! command -v nvcc >/dev/null 2>&1; then
    echo "FATAL: nvcc not in PATH. Install CUDA 12.8+ toolkit (sm_120a required)." >&2
    exit 1
  fi
  nvcc --version | head -n5
  nvidia-smi --query-gpu=name,memory.total,compute_cap,driver_version --format=csv || true
  time cargo build --release --features cuda 2>&1 | tee "$LOG_DIR/qwen36_build.log" | tail -n40
}

cmd_convert() {
  section "Step 2: vic-convert HF → .vib3"
  if [[ -f "$QWEN36_VIB3" ]]; then
    echo "Output already exists: $QWEN36_VIB3 ($(du -sh "$QWEN36_VIB3" | cut -f1))"
    echo "  delete it to force a re-convert, or set STEP=inspect"
    return 0
  fi
  [[ -d "$QWEN36_HF_DIR" ]] || { echo "FATAL: HF dir not found: $QWEN36_HF_DIR" >&2; exit 1; }
  shard_count=$(ls "$QWEN36_HF_DIR"/*.safetensors 2>/dev/null | wc -l)
  [[ $shard_count -ge 1 ]] || { echo "FATAL: no .safetensors shards in $QWEN36_HF_DIR" >&2; exit 1; }
  [[ -f "$QWEN36_HF_DIR/config.json" ]] || { echo "FATAL: missing config.json in $QWEN36_HF_DIR" >&2; exit 1; }
  echo "Found $shard_count shard(s); $(du -sh "$QWEN36_HF_DIR" | cut -f1) total."

  # Compress with zstd; expert-quant flag is no-op for dense (no routed experts)
  # but still gates per-tensor dtype handling for the dense FFN path.
  RUST_BACKTRACE=1 VIB3_LOG="${VIB3_LOG:-vib3=info}" \
    time ./target/release/vic-convert \
      --model "$QWEN36_HF_DIR" \
      --output "$QWEN36_VIB3" \
      --arch auto \
      --quantize "$QUANTIZE" \
      --compress zstd \
      --build-indexes \
    2>&1 | tee "$LOG_DIR/qwen36_convert.log" | tail -n60

  echo "Converted: $(du -sh "$QWEN36_VIB3" | cut -f1) at $QWEN36_VIB3"
}

cmd_inspect() {
  section "Step 3a: vic-inspect summary"
  ./target/release/vic-inspect "$QWEN36_VIB3" 2>&1 | tee "$LOG_DIR/qwen36_inspect.log" | head -n80

  section "Step 3b: dense-FFN segment coverage (17/18/19) across 64 layers"
  # All 64 layers are dense for Qwen3.6-27B → segments 17 (up), 18 (gate),
  # 19 (down) must each carry pages for every layer.
  for seg in 17 18 19; do
    miss=$(./target/release/vic-inspect "$QWEN36_VIB3" --segment "$seg" --pages 2>&1 \
      | awk '/^ *layer=/ {seen[$2]=1} END {for (l=0; l<64; l++) { k="layer="l; if (!(k in seen)) missing++ } print missing+0}')
    echo "  seg $seg layers missing: $miss (expected 0)"
    if [[ "$miss" != "0" ]]; then
      echo "FAIL: dense FFN seg $seg missing pages on $miss layers."
      exit 2
    fi
  done

  section "Step 3c: DeltaNet vs GatedAttention layer assignment (3:1 pattern)"
  # Layers 3,7,11,...,63 should be full attention; rest DeltaNet.
  ./target/release/vic-inspect "$QWEN36_VIB3" 2>&1 \
    | grep -E 'layer_is_attention|deltanet' | head -n3

  section "Step 3d: page decompression sample"
  ./target/release/vic-inspect "$QWEN36_VIB3" --verify --sample 32 2>&1 | tail -n20
}

cmd_run() {
  section "Step 4: first-decode smoke (one token, greedy)"
  ln -sfn "$QWEN36_VIB3" "$VIB3_HOME/models/qwen3.6-27b.vib3"
  echo "Prompt: $PROMPT"
  VIB3_HOME="$VIB3_HOME" VIB3_DIAG=1 VIB3_LOG=vib3=info \
    ./target/release/vic run qwen3.6-27b \
      --max-tokens 1 \
      --temperature 0.0 \
      --no-check \
    < <(echo "$PROMPT") \
    2>&1 | tee "$LOG_DIR/qwen36_run_first_token.log" | tail -n40
}

case "$STEP" in
  build)   cmd_build ;;
  convert) cmd_convert ;;
  inspect) cmd_inspect ;;
  run)     cmd_run ;;
  all)     cmd_build; cmd_convert; cmd_inspect; cmd_run ;;
  *)       echo "unknown STEP=$STEP (expected build|convert|inspect|run|all)" >&2; exit 1 ;;
esac

section "Done. Logs: $LOG_DIR"
