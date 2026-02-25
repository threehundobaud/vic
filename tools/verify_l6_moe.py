#!/usr/bin/env python3
"""
Verify L6 MoE computation by comparing vib3 output with Python ground truth.

Uses vib3's dumped post-norm hidden state as input to isolate MoE from attention.
Computes per-expert SwiGLU + down_proj for the 8 selected experts at L6 pos=26.

Usage: python3 tools/verify_l6_moe.py
"""

import json
import struct
import sys
import time

import numpy as np
import torch
from safetensors import safe_open

MODEL_DIR = "/code/models/kimi2.5"
DUMP_DIR = "/code/vib3/dump"

# Load model config
with open(f"{MODEL_DIR}/config.json") as f:
    raw_cfg = json.load(f)
cfg = raw_cfg.get("text_config", raw_cfg)

HIDDEN_SIZE = cfg["hidden_size"]  # 7168
MOE_INTERMEDIATE = cfg["moe_intermediate_size"]  # 2048
N_ROUTED_EXPERTS = cfg["n_routed_experts"]  # 384
NUM_EXPERTS_PER_TOK = cfg["num_experts_per_tok"]  # 8
ROUTED_SCALING_FACTOR = cfg["routed_scaling_factor"]  # 2.827
NORM_TOPK_PROB = cfg["norm_topk_prob"]  # True

PREFIX = "language_model.model."
LAYER_IDX = 6

# Load weight map
with open(f"{MODEL_DIR}/model.safetensors.index.json") as f:
    weight_map = json.load(f)["weight_map"]


def get_shard_for_key(key):
    return weight_map.get(key)


def load_tensor(key, dtype=torch.float32):
    shard = get_shard_for_key(key)
    if shard is None:
        raise KeyError(f"Key {key} not found in weight map")
    path = f"{MODEL_DIR}/{shard}"
    with safe_open(path, framework="pt") as f:
        return f.get_tensor(key).to(dtype)


def load_int4_weight(key_prefix, dtype=torch.float32):
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


def load_e_score_bias():
    bias_path = f"{MODEL_DIR}/e_score_correction_bias.bin"
    try:
        with open(bias_path, "rb") as f:
            data = f.read()
        n_floats = len(data) // 4
        values = struct.unpack(f"<{n_floats}f", data)
        bias = torch.tensor(values, dtype=torch.float32).reshape(61, 384)
        return bias
    except FileNotFoundError:
        return None


def main():
    print("=" * 70)
    print("L6 MoE Verification: vib3 post-norm input -> Python GT computation")
    print("=" * 70)

    # Load vib3's dumped post-norm hidden state (FP16, 7168 elements)
    postnorm_path = f"{DUMP_DIR}/vib3_l6_postnorm_pos26.bin"
    postnorm_data = np.fromfile(postnorm_path, dtype=np.float16)
    assert len(postnorm_data) == HIDDEN_SIZE, (
        f"Expected {HIDDEN_SIZE}, got {len(postnorm_data)}"
    )
    postnorm = torch.tensor(postnorm_data, dtype=torch.float32).unsqueeze(
        0
    )  # [1, 7168]
    print(
        f"\nLoaded vib3 post-norm: L2={postnorm.norm().item():.4f}, shape={postnorm.shape}"
    )
    print(f"  first8: {[f'{v:.6f}' for v in postnorm[0, :8].tolist()]}")
    print(f"  min={postnorm.min().item():.6f}, max={postnorm.max().item():.6f}")

    # Also load vib3's e144 SwiGLU intermediate for comparison
    e144_swiglu_path = f"{DUMP_DIR}/vib3_e144_swiglu_pos26.bin"
    e144_swiglu_data = np.fromfile(e144_swiglu_path, dtype=np.float16)
    vib3_e144_swiglu = torch.tensor(e144_swiglu_data, dtype=torch.float32)
    print(
        f"\nvib3 e144 SwiGLU intermediate: L2={vib3_e144_swiglu.norm().item():.4f}, "
        f"first8={[f'{v:.6f}' for v in vib3_e144_swiglu[:8].tolist()]}"
    )

    # Load router weight
    lp = f"{PREFIX}layers.{LAYER_IDX}.mlp"
    router_weight = load_tensor(f"{lp}.gate.weight", torch.float32)
    print(f"\nRouter weight: {router_weight.shape}")

    # Load e_score_correction_bias
    bias = load_e_score_bias()
    layer_bias = bias[LAYER_IDX] if bias is not None else None

    # Compute routing
    logits = postnorm @ router_weight.T  # [1, 384]
    scores = torch.sigmoid(logits)

    selection_scores = scores.clone()
    if layer_bias is not None:
        selection_scores = selection_scores + layer_bias

    topk_weights, topk_indices = torch.topk(
        selection_scores, NUM_EXPERTS_PER_TOK, dim=-1
    )
    topk_weights = scores.gather(1, topk_indices)

    if NORM_TOPK_PROB:
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
    topk_weights = topk_weights * ROUTED_SCALING_FACTOR

    expert_ids = topk_indices[0].tolist()
    expert_weights = topk_weights[0].tolist()

    print(f"\nRouting (Python GT):")
    for i, (eid, ew) in enumerate(zip(expert_ids, expert_weights)):
        print(f"  [{i}] e{eid} weight={ew:.6f}")
    print(f"  Sum of weights: {sum(expert_weights):.6f}")

    print(f"\nRouting (vib3 logged):")
    vib3_routing = [
        (144, 1.0993),
        (243, 0.6920),
        (346, 0.2593),
        (375, 0.2021),
        (356, 0.2188),
        (116, 0.1404),
        (254, 0.0977),
        (341, 0.1175),
    ]
    for i, (eid, ew) in enumerate(vib3_routing):
        print(f"  [{i}] e{eid} weight={ew:.4f}")

    # Compute per-expert outputs
    print(f"\n{'=' * 50}")
    print("Per-Expert SwiGLU + down_proj computation")
    print(f"{'=' * 50}")

    total_output = torch.zeros(1, HIDDEN_SIZE, dtype=torch.float32)

    for rank, (expert_id, weight) in enumerate(zip(expert_ids, expert_weights)):
        t0 = time.time()
        ep = f"{lp}.experts.{expert_id}"

        gate_w = load_int4_weight(f"{ep}.gate_proj", torch.float32)
        up_w = load_int4_weight(f"{ep}.up_proj", torch.float32)
        down_w = load_int4_weight(f"{ep}.down_proj", torch.float32)

        # SwiGLU
        gate_out = postnorm @ gate_w.T
        up_out = postnorm @ up_w.T
        gate_activated = torch.nn.functional.silu(gate_out)
        swiglu_intermediate = gate_activated * up_out
        down_out = swiglu_intermediate @ down_w.T

        elapsed = time.time() - t0

        print(f"\n  Expert {expert_id} (weight={weight:.6f}, rank={rank}):")
        print(f"    gate_proj L2={gate_out.norm().item():.4f}")
        print(f"    up_proj   L2={up_out.norm().item():.4f}")
        print(
            f"    SwiGLU    L2={swiglu_intermediate.norm().item():.4f}, "
            f"max_abs={swiglu_intermediate.abs().max().item():.4f}"
        )
        print(
            f"    down_proj L2={down_out.norm().item():.4f}, "
            f"max_abs={down_out.abs().max().item():.4f}"
        )
        print(
            f"    first8 SwiGLU: {[f'{v:.6f}' for v in swiglu_intermediate[0, :8].tolist()]}"
        )
        print(f"    first8 down:   {[f'{v:.6f}' for v in down_out[0, :8].tolist()]}")
        print(f"    ({elapsed:.1f}s)")

        # Compare e144 SwiGLU intermediate with vib3
        if expert_id == 144:
            diff = (swiglu_intermediate.squeeze(0) - vib3_e144_swiglu).norm().item()
            print(
                f"    >>> e144 SwiGLU diff vs vib3: L2={diff:.6f} "
                f"(GT L2={swiglu_intermediate.norm().item():.4f}, "
                f"vib3 L2={vib3_e144_swiglu.norm().item():.4f})"
            )

        # Accumulate weighted output
        total_output += weight * down_out
        accum_l2 = total_output.norm().item()
        print(f"    Accumulated L2 after e{expert_id}: {accum_l2:.4f}")

    # Shared expert
    print(f"\n  Shared expert:")
    t0 = time.time()
    shared_gate_w = load_tensor(f"{lp}.shared_experts.gate_proj.weight", torch.float32)
    shared_up_w = load_tensor(f"{lp}.shared_experts.up_proj.weight", torch.float32)
    shared_down_w = load_tensor(f"{lp}.shared_experts.down_proj.weight", torch.float32)

    shared_gate_out = postnorm @ shared_gate_w.T
    shared_up_out = postnorm @ shared_up_w.T
    shared_gate_activated = torch.nn.functional.silu(shared_gate_out)
    shared_swiglu = shared_gate_activated * shared_up_out
    shared_down_out = shared_swiglu @ shared_down_w.T

    elapsed = time.time() - t0
    print(f"    gate_proj L2={shared_gate_out.norm().item():.4f}")
    print(f"    up_proj   L2={shared_up_out.norm().item():.4f}")
    print(f"    SwiGLU    L2={shared_swiglu.norm().item():.4f}")
    print(f"    down_proj L2={shared_down_out.norm().item():.4f}")
    print(f"    ({elapsed:.1f}s)")

    total_output += shared_down_out
    final_l2 = total_output.norm().item()
    print(f"\n  FINAL MoE output L2 (routed + shared): {final_l2:.4f}")
    print(f"  vib3 reported MoE output L6 L2: 91.5735")

    # Print comparison summary
    print(f"\n{'=' * 70}")
    print("COMPARISON SUMMARY")
    print(f"{'=' * 70}")
    print(f"  GT  MoE output L2: {final_l2:.4f}")
    print(f"  vib3 MoE output L2: 91.5735")
    routed_l2 = (total_output - shared_down_out).norm().item()
    print(f"  GT  routed-only L2: {routed_l2:.4f}")

    # vib3 logged per-expert accumulation:
    print(f"\n  vib3 L6 per-expert accumulation:")
    print(f"    e144*1.0993 -> L2=14.3121")
    print(f"    e243*0.6920 -> L2=76.7422  ← HUGE JUMP")
    print(f"    e346*0.2593 -> L2=88.0296")
    print(f"    final (routed) -> L2=87.9926")
    print(f"    final (with shared) -> L2=91.5735")


if __name__ == "__main__":
    main()
