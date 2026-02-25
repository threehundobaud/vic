#!/usr/bin/env python3
"""
Test INT4 matmul: compute gate_proj output for layer 6 expert 144
with a known input (all 0.01 FP16) and print results for comparison
against vib3 CUDA kernel output.

This validates the CUDA `vib3_partial_matmul_int4` kernel's correctness.

Usage: python3 tools/test_int4_matmul.py
"""

import json
import struct
import numpy as np
import torch
from safetensors import safe_open

MODEL_DIR = "/code/models/kimi2.5"

# Model dimensions
HIDDEN_SIZE = 7168
MOE_INTERMEDIATE = 2048
GROUP_SIZE = 32

# Target: layer 6, expert 144, gate_proj
LAYER = 6
EXPERT = 144

# Load weight map
with open(f"{MODEL_DIR}/model.safetensors.index.json") as f:
    weight_map = json.load(f)["weight_map"]


def load_int4_weight(key_prefix):
    """Load and dequantize INT4 weight to float32."""
    packed_key = f"{key_prefix}.weight_packed"
    scale_key = f"{key_prefix}.weight_scale"
    shape_key = f"{key_prefix}.weight_shape"

    shard = weight_map[packed_key]
    path = f"{MODEL_DIR}/{shard}"
    with safe_open(path, framework="pt") as f:
        packed = f.get_tensor(packed_key).to(torch.int32)
        scale = f.get_tensor(scale_key).to(torch.float32)
        shape_tensor = f.get_tensor(shape_key).to(torch.int64)

    out_features = int(shape_tensor[0].item())
    in_features = int(shape_tensor[1].item())
    print(f"  Weight shape: [{out_features}, {in_features}]")
    print(f"  Packed shape: {packed.shape}, Scale shape: {scale.shape}")

    # Unpack INT4 nibbles
    packed_bytes = packed.view(torch.uint8)  # [rows, cols/2]
    rows = packed_bytes.shape[0]

    low_nibbles = (packed_bytes & 0x0F).to(torch.float32)
    high_nibbles = ((packed_bytes >> 4) & 0x0F).to(torch.float32)

    # Interleave: low nibble first, then high nibble
    unpacked = torch.stack([low_nibbles, high_nibbles], dim=-1)
    unpacked = unpacked.reshape(rows, -1)[:out_features, :in_features]

    # Symmetric dequant: (nibble - 8) * scale
    unpacked = unpacked - 8.0

    n_groups = (in_features + GROUP_SIZE - 1) // GROUP_SIZE
    scale_expanded = scale[:out_features, :n_groups].repeat_interleave(
        GROUP_SIZE, dim=1
    )[:, :in_features]

    return unpacked * scale_expanded


def simulate_page_matmul(key_prefix, input_vec):
    """Simulate the page-based INT4 matmul the way vib3 does it.

    This loads the raw packed + scale data and does the matmul per-page
    to detect any page-boundary issues.
    """
    packed_key = f"{key_prefix}.weight_packed"
    scale_key = f"{key_prefix}.weight_scale"
    shape_key = f"{key_prefix}.weight_shape"

    shard = weight_map[packed_key]
    path = f"{MODEL_DIR}/{shard}"
    with safe_open(path, framework="pt") as f:
        packed = f.get_tensor(packed_key).to(torch.int32)
        scale_raw = f.get_tensor(scale_key)  # Keep original dtype (BF16)
        shape_tensor = f.get_tensor(shape_key).to(torch.int64)

    out_features = int(shape_tensor[0].item())
    in_features = int(shape_tensor[1].item())

    # Convert packed to raw bytes, matching what the converter sees
    packed_bytes = packed.view(torch.uint8).numpy()  # [rows, cols/2]
    rows = out_features
    packed_k = (in_features + 1) // 2  # = 3584
    num_groups = (in_features + GROUP_SIZE - 1) // GROUP_SIZE  # = 224

    # Convert BF16 scales to raw bytes
    # Can't use numpy directly for BF16, use torch's untyped_storage
    scale_bf16 = scale_raw.to(torch.bfloat16).contiguous()
    # Get raw bytes via memoryview of storage
    scale_raw_bytes = np.frombuffer(bytes(scale_bf16.untyped_storage()), dtype=np.uint8)
    scale_bytes = scale_raw_bytes.reshape(rows, num_groups, 2)  # [rows, num_groups, 2]

    # Compute page layout (matching convert.rs)
    bytes_per_row = packed_k + num_groups * 2  # 3584 + 448 = 4032
    PAGE_SIZE = 2 * 1024 * 1024  # 2MB
    rows_per_page = PAGE_SIZE // bytes_per_row  # = 520
    num_pages = (rows + rows_per_page - 1) // rows_per_page

    print(f"\n  Page layout: packed_k={packed_k}, num_groups={num_groups}")
    print(f"  bytes_per_row={bytes_per_row}, rows_per_page={rows_per_page}")
    print(f"  num_pages={num_pages}, total_rows={rows}")

    # Input as float32 for reference
    inp = input_vec.float().numpy()
    output = np.zeros(out_features, dtype=np.float32)

    # Vectorized dequantization of the full weight matrix
    # This matches the CUDA kernel's computation but is much faster
    low_nibbles = (packed_bytes & 0x0F).astype(np.float32) - 8.0
    high_nibbles = ((packed_bytes >> 4) & 0x0F).astype(np.float32) - 8.0
    # Interleave: [rows, packed_k, 2] -> [rows, in_features]
    full_weights = np.stack([low_nibbles, high_nibbles], axis=-1).reshape(rows, -1)[
        :, :in_features
    ]

    # Dequantize scales from BF16 bytes to float32
    # scale_bytes: [rows, num_groups, 2] raw bytes
    scale_u16 = scale_bytes[:, :, 0].astype(np.uint32) | (
        scale_bytes[:, :, 1].astype(np.uint32) << 8
    )
    # BF16 -> float32: shift left by 16
    scale_u32 = scale_u16.astype(np.uint32) << 16
    scale_f32 = scale_u32.view(np.float32)  # [rows, num_groups]
    # Expand scales to match weight columns
    scale_expanded = np.repeat(scale_f32, GROUP_SIZE, axis=1)[:, :in_features]

    # Apply scales
    full_weights_dequant = full_weights * scale_expanded

    for page_idx in range(num_pages):
        row_start = page_idx * rows_per_page
        row_count = min(rows_per_page, rows - row_start)

        print(
            f"\n  Page {page_idx}: rows [{row_start}, {row_start + row_count}), row_count={row_count}"
        )

        # Compute matmul for this page's rows (vectorized)
        page_weights = full_weights_dequant[row_start : row_start + row_count]
        page_result = page_weights @ inp  # [row_count]
        output[row_start : row_start + row_count] = page_result

        print(f"    Page {page_idx} output L2: {np.linalg.norm(page_result):.6f}")
        print(f"    Page {page_idx} first 3: {page_result[:3]}")

    # Verify a few rows element-by-element (just rows 0 and 520 to check page boundary)
    print("\n  Element-by-element verification of row 0 and row 520:")
    for check_row in [0, 520]:
        if check_row >= rows:
            continue
        acc = 0.0
        for k in range(0, in_features, 2):
            byte_idx = k // 2
            packed_byte = packed_bytes[check_row, byte_idx]
            w0 = int(packed_byte & 0x0F) - 8
            w1 = int((packed_byte >> 4) & 0x0F) - 8
            group = k // GROUP_SIZE
            bf16_raw = int(scale_bytes[check_row, group, 0]) | (
                int(scale_bytes[check_row, group, 1]) << 8
            )
            float_bits = bf16_raw << 16
            scale_f = struct.unpack("f", struct.pack("I", float_bits))[0]
            acc += inp[k] * w0 * scale_f
            if k + 1 < in_features:
                acc += inp[k + 1] * w1 * scale_f
        print(
            f"    Row {check_row}: element-wise={acc:.6f}, vectorized={output[check_row]:.6f}, diff={abs(acc - output[check_row]):.2e}"
        )

    return output


def main():
    print("=" * 70)
    print(f"INT4 Matmul Test: Layer {LAYER}, Expert {EXPERT}, gate_proj")
    print("=" * 70)

    # Create known input: all 0.01 in FP16
    input_fp16 = torch.full((HIDDEN_SIZE,), 0.01, dtype=torch.float16)
    input_f32 = input_fp16.float()
    print(f"\nInput: all 0.01 FP16, shape={input_fp16.shape}")
    print(f"  FP16 actual value: {input_fp16[0].item()}")
    print(f"  As float32: {input_f32[0].item()}")

    # Method 1: Simple dequantized matmul
    key_prefix = f"language_model.model.layers.{LAYER}.mlp.experts.{EXPERT}.gate_proj"
    print(f"\nLoading weights: {key_prefix}")
    gate_weight = load_int4_weight(key_prefix)

    result_simple = (input_f32 @ gate_weight.T).numpy()
    print(f"\nSimple matmul result:")
    print(f"  First 10: {result_simple[:10]}")
    print(f"  Last 10:  {result_simple[-10:]}")
    print(f"  L2 norm:  {np.linalg.norm(result_simple):.6f}")
    print(f"  Mean:     {np.mean(result_simple):.6f}")
    print(f"  Std:      {np.std(result_simple):.6f}")
    print(f"  Min:      {np.min(result_simple):.6f}")
    print(f"  Max:      {np.max(result_simple):.6f}")

    # Method 2: Page-based simulation (matching vib3's per-page kernel invocations)
    print(f"\nPage-based simulation:")
    result_paged = simulate_page_matmul(key_prefix, input_fp16)

    print(f"\nPaged matmul result:")
    print(f"  First 10: {result_paged[:10]}")
    print(f"  Last 10:  {result_paged[-10:]}")
    print(f"  L2 norm:  {np.linalg.norm(result_paged):.6f}")

    # Compare
    diff = np.abs(result_simple - result_paged)
    print(f"\nSimple vs Paged comparison:")
    print(f"  Max diff: {np.max(diff):.2e}")
    print(f"  Mean diff: {np.mean(diff):.2e}")

    # Also do up_proj and down_proj for completeness
    for proj in ["up_proj", "down_proj"]:
        key = f"language_model.model.layers.{LAYER}.mlp.experts.{EXPERT}.{proj}"
        print(f"\n--- {proj} ---")
        w = load_int4_weight(key)
        if proj == "down_proj":
            # down_proj: [7168, 2048], input is intermediate (2048)
            test_input = torch.full((MOE_INTERMEDIATE,), 0.01, dtype=torch.float32)
        else:
            test_input = input_f32
        result = (test_input @ w.T).numpy()
        print(f"  L2 norm: {np.linalg.norm(result):.6f}")
        print(f"  First 5: {result[:5]}")

    # Print expected CUDA output format for easy comparison
    print("\n" + "=" * 70)
    print("EXPECTED CUDA OUTPUT (for input = all 0.01 FP16):")
    print(f"  gate_proj L2: {np.linalg.norm(result_simple):.6f}")
    print(f"  gate_proj[0:10]: {result_simple[:10]}")
    print("=" * 70)


if __name__ == "__main__":
    main()
