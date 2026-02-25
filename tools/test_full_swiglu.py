#!/usr/bin/env python3
"""
Compute full SwiGLU reference for expert 144, layer 6, with all-0.01 input.
Also verifies expert output with a dumped hidden state if available.
"""

import json
import sys
import struct
import numpy as np
import torch
from safetensors import safe_open

MODEL_DIR = "/code/models/kimi2.5"
HIDDEN_SIZE = 7168
MOE_INTERMEDIATE = 2048
GROUP_SIZE = 32

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


def main():
    ep = "language_model.model.layers.6.mlp.experts.144"

    print("Loading expert 144 weights (INT4 dequantized)...")
    gate_w = load_int4_weight(f"{ep}.gate_proj", torch.float32)
    up_w = load_int4_weight(f"{ep}.up_proj", torch.float32)
    down_w = load_int4_weight(f"{ep}.down_proj", torch.float32)
    print(f"  gate_w: {gate_w.shape}, up_w: {up_w.shape}, down_w: {down_w.shape}")

    # Test 1: all-0.01 input
    print("\n=== Test 1: All-0.01 input ===")
    x = torch.full([1, HIDDEN_SIZE], 0.01, dtype=torch.float32)

    gate_out = x @ gate_w.T
    up_out = x @ up_w.T
    gate_act = torch.nn.functional.silu(gate_out)
    intermediate = gate_act * up_out
    down_out = intermediate @ down_w.T

    print(f"  gate_proj L2={gate_out.norm().item():.6f}")
    print(f"  up_proj   L2={up_out.norm().item():.6f}")
    print(f"  SiLU(gate) L2={gate_act.norm().item():.6f}")
    print(f"  SwiGLU intermediate L2={intermediate.norm().item():.6f}")
    print(f"  down_proj  L2={down_out.norm().item():.6f}")
    print(f"  intermediate first8={[f'{v:.8f}' for v in intermediate[0, :8].tolist()]}")
    print(f"  intermediate max_abs={intermediate.abs().max().item():.8f}")

    # Test 2: load dumped hidden state if available
    dump_files = []
    import glob

    for path in (
        sorted(glob.glob("/model/dump/l6_postnorm_pos*.bin"))
        + sorted(glob.glob("/code/vib3/dump/l6_postnorm_pos*.bin"))
        + sorted(glob.glob("/tmp/l6_postnorm_pos*.bin"))
    ):
        dump_files.append(path)

    if dump_files:
        for dump_path in dump_files:
            print(f"\n=== Test 2: With dumped hidden state from {dump_path} ===")
            raw = open(dump_path, "rb").read()
            # FP16 data
            f16_vals = np.frombuffer(raw, dtype=np.float16)
            x_dump = torch.from_numpy(f16_vals.astype(np.float32)).unsqueeze(0)
            print(f"  Hidden state shape={x_dump.shape}, L2={x_dump.norm().item():.4f}")
            print(f"  first8={[f'{v:.6f}' for v in x_dump[0, :8].tolist()]}")
            print(f"  min={x_dump.min().item():.6f}, max={x_dump.max().item():.6f}")

            gate_out = x_dump @ gate_w.T
            up_out = x_dump @ up_w.T
            gate_act = torch.nn.functional.silu(gate_out)
            intermediate = gate_act * up_out
            down_out = intermediate @ down_w.T

            print(f"  gate_proj L2={gate_out.norm().item():.4f}")
            print(f"  up_proj   L2={up_out.norm().item():.4f}")
            print(f"  SwiGLU intermediate L2={intermediate.norm().item():.4f}")
            print(f"  intermediate max_abs={intermediate.abs().max().item():.4f}")
            print(
                f"  intermediate first8={[f'{v:.6f}' for v in intermediate[0, :8].tolist()]}"
            )
            print(f"  down_proj  L2={down_out.norm().item():.4f}")
            print(f"  down_proj  max_abs={down_out.abs().max().item():.4f}")

            # Compare against vib3 dumped SwiGLU intermediate if available
            swiglu_pattern = dump_path.replace("l6_postnorm_pos", "e144_swiglu_pos")
            import os

            if os.path.exists(swiglu_pattern):
                print(f"\n  Comparing against vib3 SwiGLU dump: {swiglu_pattern}")
                raw_swiglu = open(swiglu_pattern, "rb").read()
                vib3_f16 = np.frombuffer(raw_swiglu, dtype=np.float16)
                vib3_inter = torch.from_numpy(vib3_f16.astype(np.float32)).unsqueeze(0)
                print(f"  vib3  SwiGLU L2={vib3_inter.norm().item():.4f}")
                print(f"  Python SwiGLU L2={intermediate.norm().item():.4f}")
                diff = (vib3_inter - intermediate).norm().item()
                print(f"  L2 difference={diff:.4f}")
                print(
                    f"  vib3  first8={[f'{v:.6f}' for v in vib3_inter[0, :8].tolist()]}"
                )
                print(
                    f"  Python first8={[f'{v:.6f}' for v in intermediate[0, :8].tolist()]}"
                )

                # Find top-5 most different elements
                abs_diff = (vib3_inter - intermediate).abs().squeeze()
                top5_idx = abs_diff.topk(5).indices
                for idx in top5_idx:
                    i = idx.item()
                    print(
                        f"  elem[{i}]: vib3={vib3_inter[0, i].item():.6f}, "
                        f"python={intermediate[0, i].item():.6f}, "
                        f"diff={abs_diff[i].item():.6f}"
                    )
    else:
        print("\nNo dump files found. Run vib3-serve first to generate dumps.")
        print("Looking in: /model/dump/, /code/vib3/dump/, /tmp/")

    # Test 3: What should the SwiGLU look like per-page?
    print("\n=== Per-page breakdown (all-0.01 input) ===")
    rows_per_page = [520, 520, 520, 488]
    row_start = 0
    for i, n_rows in enumerate(rows_per_page):
        page_gate = gate_out[0, row_start : row_start + n_rows]
        page_up = up_out[0, row_start : row_start + n_rows]
        page_silu = torch.nn.functional.silu(page_gate)
        page_inter = page_silu * page_up
        print(
            f"  Page {i} (rows {row_start}-{row_start + n_rows - 1}): "
            f"gate L2={page_gate.norm().item():.6f}, "
            f"up L2={page_up.norm().item():.6f}, "
            f"SwiGLU L2={page_inter.norm().item():.6f}"
        )
        row_start += n_rows


if __name__ == "__main__":
    main()
