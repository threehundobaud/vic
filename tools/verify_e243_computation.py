#!/usr/bin/env python3
"""
Verify e243 computation at L6 pos=0: compare vib3 engine output vs Python ground truth.

Loads the engine's dumped FP32 moe_normed input and e243's INT4 weights from HF,
then computes SwiGLU + down_proj in FP32. Compares against engine's dumped
e243 intermediate and down_proj outputs.

This definitively answers: is the engine computing expert 243 correctly,
or is there a computation bug?

Usage: python3 tools/verify_e243_computation.py
"""

import json
import struct
import sys
import time

import numpy as np
import torch
from safetensors import safe_open

MODEL_DIR = "/code/models/kimi2.5"
DUMP_DIR = "/tmp/vib3_dump"

# Load model config
with open(f"{MODEL_DIR}/config.json") as f:
    raw_cfg = json.load(f)
cfg = raw_cfg.get("text_config", raw_cfg)

HIDDEN_SIZE = cfg["hidden_size"]  # 7168
MOE_INTERMEDIATE = cfg["moe_intermediate_size"]  # 2048

PREFIX = "language_model.model."
LAYER_IDX = 6

# Load weight map
with open(f"{MODEL_DIR}/model.safetensors.index.json") as f:
    weight_map = json.load(f)["weight_map"]


def load_tensor(key, dtype=torch.float32):
    shard = weight_map.get(key)
    if shard is None:
        raise KeyError(f"Key {key} not found in weight map")
    path = f"{MODEL_DIR}/{shard}"
    with safe_open(path, framework="pt") as f:
        return f.get_tensor(key).to(dtype)


def load_int4_weight(key_prefix, dtype=torch.float32):
    """Load and dequantize an INT4 compressed-tensors weight.

    Format: packed uint4 in int32 (8 nibbles per int32, little-endian byte order,
    low nibble first per byte) + BF16 group-32 scales.
    """
    packed_key = f"{key_prefix}.weight_packed"
    scale_key = f"{key_prefix}.weight_scale"
    shape_key = f"{key_prefix}.weight_shape"

    packed = load_tensor(packed_key, dtype=torch.int32)
    scale = load_tensor(scale_key, dtype=dtype)
    shape_tensor = load_tensor(shape_key, dtype=torch.int64)

    out_features = int(shape_tensor[0].item())
    in_features = int(shape_tensor[1].item())

    packed_bytes = packed.view(torch.uint8)
    rows = packed_bytes.shape[0]

    low_nibbles = (packed_bytes & 0x0F).to(dtype)
    high_nibbles = ((packed_bytes >> 4) & 0x0F).to(dtype)

    unpacked = torch.stack([low_nibbles, high_nibbles], dim=-1)
    unpacked = unpacked.reshape(rows, -1)
    unpacked = unpacked[:out_features, :in_features]
    unpacked = unpacked - 8.0

    n_groups = (in_features + 31) // 32
    scale_expanded = scale[:out_features, :n_groups]
    scale_expanded = scale_expanded.repeat_interleave(32, dim=1)[:, :in_features]

    return unpacked * scale_expanded


def cosine_similarity(a, b):
    """Compute cosine similarity between two 1D tensors."""
    return (torch.dot(a, b) / (a.norm() * b.norm())).item()


def main():
    print("=" * 70)
    print("E243 Computation Verification: Engine vs Python Ground Truth")
    print("=" * 70)

    # ========================================================================
    # Step 1: Load engine's dumped FP32 moe_normed input
    # ========================================================================
    moe_normed_path = f"{DUMP_DIR}/moe_normed_f32_L6_pos0.bin"
    moe_normed_data = np.fromfile(moe_normed_path, dtype=np.float32)
    assert len(moe_normed_data) == HIDDEN_SIZE, (
        f"Expected {HIDDEN_SIZE} floats, got {len(moe_normed_data)}"
    )
    moe_normed = torch.tensor(moe_normed_data, dtype=torch.float32).unsqueeze(
        0
    )  # [1, 7168]

    print(f"\nEngine moe_normed_f32 (L6 pos=0):")
    print(f"  Shape: {moe_normed.shape}")
    print(f"  L2: {moe_normed.norm().item():.6f}")
    print(f"  min: {moe_normed.min().item():.6f}, max: {moe_normed.max().item():.6f}")
    print(f"  first8: {[f'{v:.6f}' for v in moe_normed[0, :8].tolist()]}")
    print(f"  last4:  {[f'{v:.6f}' for v in moe_normed[0, -4:].tolist()]}")

    # ========================================================================
    # Step 2: Load engine's dumped e243 SwiGLU intermediate (FP16)
    # ========================================================================
    e243_swiglu_path = f"{DUMP_DIR}/e243_swiglu_L6_pos0.bin"
    e243_swiglu_data = np.fromfile(e243_swiglu_path, dtype=np.float16)
    assert len(e243_swiglu_data) == MOE_INTERMEDIATE, (
        f"Expected {MOE_INTERMEDIATE} halfs, got {len(e243_swiglu_data)}"
    )
    engine_swiglu = torch.tensor(e243_swiglu_data, dtype=torch.float32)

    print(f"\nEngine e243 SwiGLU intermediate (L6 pos=0):")
    print(f"  Shape: {engine_swiglu.shape}")
    print(f"  L2: {engine_swiglu.norm().item():.6f}")
    print(
        f"  min: {engine_swiglu.min().item():.6f}, max: {engine_swiglu.max().item():.6f}"
    )
    print(f"  first8: {[f'{v:.6f}' for v in engine_swiglu[:8].tolist()]}")

    # ========================================================================
    # Step 3: Load engine's dumped e243 down_proj output (FP16)
    # ========================================================================
    e243_down_path = f"{DUMP_DIR}/e243_downproj_L6_pos0.bin"
    e243_down_data = np.fromfile(e243_down_path, dtype=np.float16)
    assert len(e243_down_data) == HIDDEN_SIZE, (
        f"Expected {HIDDEN_SIZE} halfs, got {len(e243_down_data)}"
    )
    engine_down = torch.tensor(e243_down_data, dtype=torch.float32)

    print(f"\nEngine e243 down_proj output (L6 pos=0):")
    print(f"  Shape: {engine_down.shape}")
    print(f"  L2: {engine_down.norm().item():.6f}")
    print(f"  min: {engine_down.min().item():.6f}, max: {engine_down.max().item():.6f}")
    print(f"  first8: {[f'{v:.6f}' for v in engine_down[:8].tolist()]}")

    # ========================================================================
    # Step 4: Load engine's dumped shared expert output (FP16)
    # ========================================================================
    shared_path = f"{DUMP_DIR}/shared_expert_L6_pos0.bin"
    shared_data = np.fromfile(shared_path, dtype=np.float16)
    assert len(shared_data) == HIDDEN_SIZE, (
        f"Expected {HIDDEN_SIZE} halfs, got {len(shared_data)}"
    )
    engine_shared = torch.tensor(shared_data, dtype=torch.float32)

    print(f"\nEngine shared expert output (L6 pos=0):")
    print(f"  Shape: {engine_shared.shape}")
    print(f"  L2: {engine_shared.norm().item():.6f}")

    # ========================================================================
    # Step 5: Load e243 INT4 weights from HF and compute GT
    # ========================================================================
    print(f"\n{'=' * 50}")
    print("Computing Python ground truth for e243...")
    print(f"{'=' * 50}")

    lp = f"{PREFIX}layers.{LAYER_IDX}.mlp"
    ep = f"{lp}.experts.243"

    t0 = time.time()
    gate_w = load_int4_weight(f"{ep}.gate_proj", torch.float32)
    up_w = load_int4_weight(f"{ep}.up_proj", torch.float32)
    down_w = load_int4_weight(f"{ep}.down_proj", torch.float32)
    elapsed = time.time() - t0

    print(f"  Loaded e243 weights in {elapsed:.1f}s")
    print(
        f"  gate_proj: {gate_w.shape} (should be [{MOE_INTERMEDIATE}, {HIDDEN_SIZE}])"
    )
    print(f"  up_proj:   {up_w.shape}")
    print(
        f"  down_proj: {down_w.shape} (should be [{HIDDEN_SIZE}, {MOE_INTERMEDIATE}])"
    )

    # Weight statistics
    print(
        f"\n  gate_w stats: L2={gate_w.norm().item():.4f}, "
        f"max_abs={gate_w.abs().max().item():.6f}, "
        f"mean_abs={gate_w.abs().mean().item():.6f}"
    )
    print(
        f"  up_w stats:   L2={up_w.norm().item():.4f}, "
        f"max_abs={up_w.abs().max().item():.6f}, "
        f"mean_abs={up_w.abs().mean().item():.6f}"
    )
    print(
        f"  down_w stats: L2={down_w.norm().item():.4f}, "
        f"max_abs={down_w.abs().max().item():.6f}, "
        f"mean_abs={down_w.abs().mean().item():.6f}"
    )

    # Compute SwiGLU step by step
    gate_out = moe_normed @ gate_w.T  # [1, 2048]
    up_out = moe_normed @ up_w.T  # [1, 2048]
    gate_activated = torch.nn.functional.silu(gate_out)
    gt_swiglu = (gate_activated * up_out).squeeze(0)  # [2048]

    print(f"\n  Python GT computation:")
    print(
        f"    gate_proj output: L2={gate_out.norm().item():.6f}, "
        f"max_abs={gate_out.abs().max().item():.6f}"
    )
    print(
        f"    up_proj output:   L2={up_out.norm().item():.6f}, "
        f"max_abs={up_out.abs().max().item():.6f}"
    )
    print(f"    SiLU(gate):       L2={gate_activated.norm().item():.6f}")
    print(
        f"    SwiGLU inter:     L2={gt_swiglu.norm().item():.6f}, "
        f"max_abs={gt_swiglu.abs().max().item():.6f}"
    )
    print(f"    first8:           {[f'{v:.6f}' for v in gt_swiglu[:8].tolist()]}")

    # Compute down_proj
    gt_down = (gt_swiglu.unsqueeze(0) @ down_w.T).squeeze(0)  # [7168]

    print(
        f"    down_proj output: L2={gt_down.norm().item():.6f}, "
        f"max_abs={gt_down.abs().max().item():.6f}"
    )
    print(f"    first8:           {[f'{v:.6f}' for v in gt_down[:8].tolist()]}")

    # ========================================================================
    # Step 6: COMPARE engine vs GT
    # ========================================================================
    print(f"\n{'=' * 70}")
    print("COMPARISON: Engine vs Python Ground Truth")
    print(f"{'=' * 70}")

    # SwiGLU intermediate
    swiglu_cos = cosine_similarity(engine_swiglu, gt_swiglu)
    swiglu_diff_l2 = (engine_swiglu - gt_swiglu).norm().item()
    swiglu_ratio = engine_swiglu.norm().item() / max(gt_swiglu.norm().item(), 1e-10)

    print(f"\n  SwiGLU intermediate:")
    print(f"    Engine L2: {engine_swiglu.norm().item():.6f}")
    print(f"    GT     L2: {gt_swiglu.norm().item():.6f}")
    print(f"    L2 ratio:  {swiglu_ratio:.6f}")
    print(f"    Cosine:    {swiglu_cos:.8f}")
    print(f"    Diff L2:   {swiglu_diff_l2:.6f}")

    # Down proj
    down_cos = cosine_similarity(engine_down, gt_down)
    down_diff_l2 = (engine_down - gt_down).norm().item()
    down_ratio = engine_down.norm().item() / max(gt_down.norm().item(), 1e-10)

    print(f"\n  down_proj output:")
    print(f"    Engine L2: {engine_down.norm().item():.6f}")
    print(f"    GT     L2: {gt_down.norm().item():.6f}")
    print(f"    L2 ratio:  {down_ratio:.6f}")
    print(f"    Cosine:    {down_cos:.8f}")
    print(f"    Diff L2:   {down_diff_l2:.6f}")

    # Element-wise comparison for SwiGLU
    print(f"\n  Element-wise SwiGLU comparison (first 16):")
    for i in range(min(16, len(engine_swiglu))):
        e_val = engine_swiglu[i].item()
        g_val = gt_swiglu[i].item()
        diff = abs(e_val - g_val)
        rel = diff / max(abs(g_val), 1e-8)
        print(
            f"    [{i:3d}] engine={e_val:12.6f}  gt={g_val:12.6f}  "
            f"diff={diff:.6f}  rel={rel:.4f}"
        )

    # Element-wise comparison for down_proj
    print(f"\n  Element-wise down_proj comparison (first 16):")
    for i in range(min(16, len(engine_down))):
        e_val = engine_down[i].item()
        g_val = gt_down[i].item()
        diff = abs(e_val - g_val)
        rel = diff / max(abs(g_val), 1e-8)
        print(
            f"    [{i:3d}] engine={e_val:12.6f}  gt={g_val:12.6f}  "
            f"diff={diff:.6f}  rel={rel:.4f}"
        )

    # Find worst mismatches
    swiglu_abs_diff = (engine_swiglu - gt_swiglu).abs()
    worst_swiglu_idx = swiglu_abs_diff.argmax().item()
    print(f"\n  Worst SwiGLU mismatch at index {worst_swiglu_idx}:")
    print(
        f"    engine={engine_swiglu[worst_swiglu_idx].item():.6f}, "
        f"gt={gt_swiglu[worst_swiglu_idx].item():.6f}, "
        f"diff={swiglu_abs_diff[worst_swiglu_idx].item():.6f}"
    )

    down_abs_diff = (engine_down - gt_down).abs()
    worst_down_idx = down_abs_diff.argmax().item()
    print(f"\n  Worst down_proj mismatch at index {worst_down_idx}:")
    print(
        f"    engine={engine_down[worst_down_idx].item():.6f}, "
        f"gt={gt_down[worst_down_idx].item():.6f}, "
        f"diff={down_abs_diff[worst_down_idx].item():.6f}"
    )

    # ========================================================================
    # Step 7: Also verify shared expert
    # ========================================================================
    print(f"\n{'=' * 50}")
    print("Shared expert verification...")
    print(f"{'=' * 50}")

    t0 = time.time()
    shared_gate_w = load_tensor(f"{lp}.shared_experts.gate_proj.weight", torch.float32)
    shared_up_w = load_tensor(f"{lp}.shared_experts.up_proj.weight", torch.float32)
    shared_down_w = load_tensor(f"{lp}.shared_experts.down_proj.weight", torch.float32)

    shared_gate_out = moe_normed @ shared_gate_w.T
    shared_up_out = moe_normed @ shared_up_w.T
    shared_gate_activated = torch.nn.functional.silu(shared_gate_out)
    shared_swiglu = shared_gate_activated * shared_up_out
    gt_shared = (shared_swiglu @ shared_down_w.T).squeeze(0)

    elapsed = time.time() - t0
    print(f"  Computed in {elapsed:.1f}s")
    print(f"  GT     shared L2: {gt_shared.norm().item():.6f}")
    print(f"  Engine shared L2: {engine_shared.norm().item():.6f}")

    shared_cos = cosine_similarity(engine_shared, gt_shared)
    shared_diff_l2 = (engine_shared - gt_shared).norm().item()
    print(f"  Cosine: {shared_cos:.8f}")
    print(f"  Diff L2: {shared_diff_l2:.6f}")

    # ========================================================================
    # Step 8: VERDICT
    # ========================================================================
    print(f"\n{'=' * 70}")
    print("VERDICT")
    print(f"{'=' * 70}")

    swiglu_match = swiglu_cos > 0.99
    down_match = down_cos > 0.99
    shared_match = shared_cos > 0.99

    if swiglu_match and down_match:
        print(f"\n  ENGINE MATCHES GT for e243!")
        print(f"  SwiGLU cosine={swiglu_cos:.6f}, down_proj cosine={down_cos:.6f}")
        print(f"\n  => The MoE COMPUTATION is CORRECT.")
        print(f"  => The bug is UPSTREAM: attention produces wrong hidden states")
        print(f"     that cause wrong MoE input, activating e243 with huge output.")
        print(f"  => Focus investigation on attention layers 0-5.")

        if gt_swiglu.norm().item() > 100:
            print(
                f"\n  NOTE: GT also shows large SwiGLU L2={gt_swiglu.norm().item():.4f}"
            )
            print(f"  This confirms the INPUT is causing expert 243 to explode,")
            print(f"  not a computation bug.")
        else:
            print(f"\n  NOTE: GT SwiGLU L2={gt_swiglu.norm().item():.4f} is normal.")
            print(f"  Engine SwiGLU L2={engine_swiglu.norm().item():.4f}")
            print(
                f"  Both match but are small - the input doesn't trigger e243 explosion."
            )
    else:
        print(f"\n  ENGINE DOES NOT MATCH GT for e243!")
        print(f"  SwiGLU cosine={swiglu_cos:.6f}, down_proj cosine={down_cos:.6f}")
        print(f"\n  => There is a COMPUTATION BUG in the SwiGLU/down_proj path.")
        print(f"  => Check: INT4 dequantization, page assembly, kernel math,")
        print(f"     gate/up segment assignment, weight layout.")

        if not swiglu_match:
            print(f"\n  SwiGLU mismatch details:")
            print(
                f"    Engine L2={engine_swiglu.norm().item():.6f} vs GT L2={gt_swiglu.norm().item():.6f}"
            )
            if swiglu_ratio > 10 or swiglu_ratio < 0.1:
                print(
                    f"    Magnitude differs by {swiglu_ratio:.1f}x - likely wrong weights loaded"
                )
            elif swiglu_cos < 0:
                print(f"    Negative cosine - gate/up might be SWAPPED")
            elif swiglu_cos < 0.5:
                print(f"    Low cosine - wrong expert weights or wrong dequantization")

    print()


if __name__ == "__main__":
    main()
