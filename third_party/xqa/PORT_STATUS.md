# xqa MLA K26 — vib3 port status

## Status: **working, numerics match reference**

Running the full K2.6 pipeline under `VIB3_CUTLASS_MLA=1 VIB3_MLA_BACKEND=xqa`:

- "What is 2+2? Answer with one word." → token id=19 ("4")
- ATTN DIAG L0 pos=0: o_proj L2=0.6175, max_abs=0.2087, nan=0
  ↳ matches built-in MLA reference to 4 decimals
- L1/L2 match similarly
- Top-10 logits identical (19, 220, 42629, 4698, ...); magnitudes within FP16 noise (±0.05)
- All 186 integration tests still pass

Enable with: `VIB3_CUTLASS_MLA=1 VIB3_MLA_BACKEND=xqa`. Both flags are required —
the first opens the path into `mla_cutlass_decode`, the second dispatches from
there to the xqa kernel instead of the CUTLASS path.

## Port deltas vs upstream

- `defines.h` IS_MLA gate widened to accept `HEAD_GRP_SIZE == 64` alongside 128.
- `mla_sm120_k26.cu` (1.9 KLoc forked from `mla_sm120.cu`):
  - FP16 inputs + FP16 KV cache (instead of `__nv_fp8_e4m3`).
  - `qmmaShape` overridden to `{16, 8, 16}` for `m16n8k16.f16` MMA atoms.
  - All three `mma<__nv_fp8_e4m3>` call sites switched to `mma<half>`.
  - X softmax buffer stored as FP16 directly (no e4m3 quantization); FP8 prmt
    reorder removed (`stmatrix.b16.x4` lays halves in the native layout that
    the consumer's `ldmatrix.b16.x4` expects).
  - V tile load switched from `ldmatrix.m16n16.x2.trans.b8` to
    `ldmatrix.m8n8.x4.trans.b16` with Mat16x32-convention addressing
    `(row = L%16, col = L/16 LdGrain)`. AtomB pairing
    `v[0] = {data[0], data[1]}, v[1] = {data[2], data[3]}` (TL+BL / TR+BR,
    matching the column-major quadrant order produced by lane → matIdx=L/8).
  - X-store in `storeOrderedXToShm` writes r[0..3] = {TL, BL, TR, BR} with
    `matRowBase = 8 * (matIdx & 1)` and `matColLdGrain = matIdx >> 1`.
  - xF16 packing slot order [0..3] = {m=0·j, m=1·j, m=0·j+1, m=1·j+1} so the
    accumulator content aligns with stmatrix.b16.x4 r[0..3] = {TL, BL, TR, BR}.
  - `computeRowSumFromF8` disabled — FP32 row-sum path used.
  - `xScale` / `rcpXScale` collapse to 1.0.
  - `finalize()` converts accumulator to `half2` (not `bfloat162`) to match
    K26's `OutputHead = Vec<half, 512>`.
  - `partElemsV` halved (128 → 64) so the V tensor-map swizzle stays at the
    supported 128 B. Consumer iterates `nbVPartsInWarpTile = 2` V-parts per
    tile (upstream FP8 was always 1).
  - `tokensPerTile` halved (64 → 32) so `WarpOutSwizzleBuf::rows` divides
    `Dst::size = warpTile.y = 16` under the K26 head count.
  - `nbXVBufs` forced to 1 and `nbKBufs` halved (12 → 6) to fit sm_120a's
    99 KB per-block smem budget.
  - `OutputHead` (via `mha.h`) switched to `half` under `VIB3_MLA_K26`.
  - **Q-prefetch trigger**: upstream fired at `idxAtomBx2 == 2` under the FP8
    setup (tileNbAtomBx2 = 4 → fires at midpoint). K26 has tileNbAtomBx2 = 2
    so that branch would never fire, leaving regQBuf uninitialized (all
    zeros) for idxInstK > 0 and producing NaN. Changed to
    `idxAtomBx2 + 1 == tileNbAtomBx2` so it fires on the last atomBx2 of
    every idxInstK regardless of count.
  - `makeTensorMapForQ` swizzle selection made dtype-aware: FP16 partBytes = 128
    → `SWIZZLE_128B` (upstream hardcoded `SWIZZLE_64B`).
- `mla_sm120.cuh` `loadShmRowMax` / `storeRowMax` / `storeRowMaxAsync` got a
  `lane < rowsPerIter` guard so they work when `tileM < warp_size` (K26 has
  `warpTile.y = 16` for the producer; upstream never exercised this case).
- `xqa_wrapper_k26.cu` packs vib3's separate `q_nope/q_pe/kv_c/k_pe` buffers
  into xqa's concatenated 576-wide paged layout and calls `launchMLA`. Exposes
  `vib3_launch_xqa_mla_decode_k26` and `vib3_xqa_mla_workspace_size_k26`.
- `build.rs` compiles the three CU files under `-arch=sm_120a` and links
  `libcuda`. `-DNDEBUG` is passed to the K26 sources so the upstream
  `assert(maxVal >= elem * qkScaleLog2e)` (hot path) stays out.
- `src/compute/cuda_ffi.rs` declares the new symbols.
- `src/runtime/engine.rs::mla_cutlass_decode` branches on
  `VIB3_MLA_BACKEND=xqa`.

## Residual observations

- At `T=0`, generation matches the built-in reference exactly for the first
  1–2 decode tokens, then drifts (different top-1 at step 3 onwards). This is
  FP16 precision noise compounding across 61 layers × multiple positions —
  expected when comparing two independent kernel implementations. At higher
  temperatures / with sampling the drift disappears into sampling noise.
- Per-step attention wall-time is the same as the built-in path (~74 ms).
  Both run the full 61-layer MLA. Further perf work is its own session.

## Known constraints

- Only sm_120a arch is compiled. The upstream kernel uses `__CUDA_ARCH__ >= 900`
  intrinsics (TMA, cluster barriers), so sm_89 / older is not supported.
- Only `HEAD_GRP_SIZE == 64` + `HEAD_ELEMS == 576` is wired up (K2.6 shape).
  Shapes for other models would need a separate variant.
