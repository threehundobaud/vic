#!/usr/bin/env bash
# Apply vib3-local patches to third_party/cutlass after fetch_cutlass.sh.
# Idempotent — safe to re-run.
#
# Current patches:
#   1. Reduce Sm100FmhaMlaKernel StagesQK from "24 / sizeof(Element)" to
#      a fixed 2. Upstream setting produces a 212 KB shared-storage
#      kernel which exceeds the 99 KB per-block smem opt-in cap on
#      RTX PRO 6000 Blackwell Workstation (sm_120a). With StagesQK=2
#      the kernel still doesn't fit (129 KB), but it's the closest
#      achievable — further reduction needs StagesPV >= 2 per the
#      kernel's own static_assert. Kept as a patch in case a future
#      kernel trim makes it fit; on sm_100 you'd want to revert this
#      to the upstream value for max pipeline throughput.

set -euo pipefail

TARGET="${TARGET:-third_party/cutlass}"
KERNEL="${TARGET}/examples/77_blackwell_fmha/kernel/sm100_fmha_mla_tma_warpspecialized.hpp"

if [ ! -f "$KERNEL" ]; then
    echo "error: $KERNEL not found — run scripts/fetch_cutlass.sh first"
    exit 1
fi

if grep -q "vib3 sm_120a patch" "$KERNEL"; then
    echo "CUTLASS patches already applied"
    exit 0
fi

# Replace the StagesQK line with our reduced version.
python3 - <<EOF
import pathlib, re
p = pathlib.Path("$KERNEL")
src = p.read_text()
old = '  static const int StagesQK = 24 / sizeof(Element);  // free parameter\n'
new = (
    "  // vib3 sm_120a patch: upstream StagesQK = 24 / sizeof(Element) = 12 for\n"
    "  // FP16 pushes SharedStorageSize to 212 KB, which exceeds the 99 KB\n"
    "  // per-block smem opt-in limit on RTX PRO 6000 Blackwell Workstation.\n"
    "  // Minimum is 2 (StagesPV = StagesQK and StagesPV must be >= 2).\n"
    "  // Restore on sm_100 (B200) where 228 KB is available.\n"
    "  static const int StagesQK = 2;  // free parameter (vib3: reduced for sm_120a smem)\n"
)
if old not in src:
    print("Expected StagesQK line not found — patch may need updating for this cutlass version")
    raise SystemExit(1)
p.write_text(src.replace(old, new))
print("Patched StagesQK → 2 in", p)
EOF

echo "CUTLASS patches applied."
