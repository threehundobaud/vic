#!/usr/bin/env bash
# Fetches NVIDIA CUTLASS 4.2.1 into third_party/cutlass/ so build.rs can compile
# the Blackwell MLA kernel (cuda/src/cutlass_mla.cu) against examples/77_blackwell_fmha.
# Run from the repo root. Safe to re-run — clones only if the directory is missing.

set -euo pipefail

CUTLASS_VERSION="${CUTLASS_VERSION:-v4.2.1}"
TARGET="${TARGET:-third_party/cutlass}"

if [ -d "$TARGET/include/cutlass" ] && [ -d "$TARGET/examples/77_blackwell_fmha" ]; then
    echo "CUTLASS already present at $TARGET (skipping clone)"
    exit 0
fi

mkdir -p "$(dirname "$TARGET")"
git clone --depth 1 --branch "$CUTLASS_VERSION" \
    https://github.com/NVIDIA/cutlass.git "$TARGET"

echo ""
echo "CUTLASS $CUTLASS_VERSION fetched into $TARGET"
echo ""
echo "Applying vib3-local patches..."
"$(dirname "$0")/patch_cutlass.sh"

echo ""
echo "build.rs will now compile cuda/src/cutlass_mla.cu on next \`cargo build\`."
