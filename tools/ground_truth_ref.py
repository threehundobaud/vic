#!/usr/bin/env python3
"""
Ground truth reference: compute hidden states through Kimi K2.5 layers
to verify vib3 engine correctness.

Loads individual safetensors shards one at a time to minimize memory.
Computes on CPU in float32 for accuracy.

Usage: python3 tools/ground_truth_ref.py
"""

import json
import math
import struct
import sys
import time

import numpy as np
import torch
from safetensors import safe_open
from tokenizers import Tokenizer

MODEL_DIR = "/code/models/kimi2.5"
NUM_LAYERS_TO_COMPUTE = 7  # Compute through layers 0-6 (layer 6 is the focus)

# Load model config
with open(f"{MODEL_DIR}/config.json") as f:
    raw_cfg = json.load(f)
cfg = raw_cfg.get("text_config", raw_cfg)

HIDDEN_SIZE = cfg["hidden_size"]  # 7168
NUM_HEADS = cfg["num_attention_heads"]  # 64
NUM_KV_HEADS = cfg["num_key_value_heads"]  # 64
DENSE_INTERMEDIATE = cfg["intermediate_size"]  # 18432
MOE_INTERMEDIATE = cfg["moe_intermediate_size"]  # 2048
N_ROUTED_EXPERTS = cfg["n_routed_experts"]  # 384
NUM_EXPERTS_PER_TOK = cfg["num_experts_per_tok"]  # 8
ROUTED_SCALING_FACTOR = cfg["routed_scaling_factor"]  # 2.827
NORM_TOPK_PROB = cfg["norm_topk_prob"]  # True
SCORING_FUNC = cfg["scoring_func"]  # "sigmoid"
RMS_NORM_EPS = cfg.get("rms_norm_eps", 1e-5)
FIRST_K_DENSE = cfg["first_k_dense_replace"]  # 1
Q_LORA_RANK = cfg["q_lora_rank"]  # 1536
KV_LORA_RANK = cfg["kv_lora_rank"]  # 512
QK_ROPE_DIM = cfg["qk_rope_head_dim"]  # 64
QK_NOPE_DIM = cfg["qk_nope_head_dim"]  # 128
V_DIM = cfg["v_head_dim"]  # 128
ROPE_THETA = cfg.get("rope_theta", 50000.0)

# YaRN RoPE scaling config
rope_scaling = cfg.get("rope_scaling", {})
ROPE_FACTOR = rope_scaling.get("factor", 1.0)
ROPE_BETA_FAST = rope_scaling.get("beta_fast", 32.0)
ROPE_BETA_SLOW = rope_scaling.get("beta_slow", 1.0)
ROPE_ORIGINAL_MAX_POS = rope_scaling.get("original_max_position_embeddings", 4096)
ROPE_MSCALE = rope_scaling.get("mscale", 1.0)
ROPE_MSCALE_ALL_DIM = rope_scaling.get("mscale_all_dim", 0.0)

PREFIX = "language_model.model."

# Load weight map
with open(f"{MODEL_DIR}/model.safetensors.index.json") as f:
    weight_map = json.load(f)["weight_map"]


def get_shard_for_key(key):
    return weight_map.get(key)


def load_tensor(key, dtype=torch.float32):
    """Load a single tensor from the correct shard."""
    shard = get_shard_for_key(key)
    if shard is None:
        raise KeyError(f"Key {key} not found in weight map")
    path = f"{MODEL_DIR}/{shard}"
    with safe_open(path, framework="pt") as f:
        return f.get_tensor(key).to(dtype)


def load_int4_weight(key_prefix, dtype=torch.float32):
    """Load and dequantize an INT4 compressed-tensors weight.

    Format: packed uint4 in int32 (8 nibbles per int32, little-endian byte order,
    low nibble first per byte) + BF16 group-32 scales.
    Returns dequantized weight in specified dtype.
    """
    packed_key = f"{key_prefix}.weight_packed"
    scale_key = f"{key_prefix}.weight_scale"
    shape_key = f"{key_prefix}.weight_shape"

    packed = load_tensor(packed_key, dtype=torch.int32)  # [rows, cols/8]
    scale = load_tensor(scale_key, dtype=dtype)  # [rows, cols/32] (group_size=32)
    shape_tensor = load_tensor(shape_key, dtype=torch.int64)

    out_features = int(shape_tensor[0].item())
    in_features = int(shape_tensor[1].item())

    # Unpack: view as bytes, extract nibbles
    packed_bytes = packed.view(torch.uint8)  # [rows, cols/2]
    rows = packed_bytes.shape[0]

    low_nibbles = (packed_bytes & 0x0F).to(dtype)
    high_nibbles = ((packed_bytes >> 4) & 0x0F).to(dtype)

    # Interleave: low nibble first, high nibble second
    unpacked = torch.stack([low_nibbles, high_nibbles], dim=-1)  # [rows, cols/2, 2]
    unpacked = unpacked.reshape(rows, -1)  # [rows, cols]

    # Trim to actual shape
    unpacked = unpacked[:out_features, :in_features]

    # Symmetric dequant: (nibble - 8) * scale
    unpacked = unpacked - 8.0

    # Apply group-32 scales
    n_groups = (in_features + 31) // 32
    scale_expanded = scale[:out_features, :n_groups]
    scale_expanded = scale_expanded.repeat_interleave(32, dim=1)[:, :in_features]

    return unpacked * scale_expanded


def rms_norm(x, weight, eps=RMS_NORM_EPS):
    """RMSNorm: x * weight / sqrt(mean(x^2) + eps)"""
    # x: [seq_len, hidden_size]
    variance = x.pow(2).mean(-1, keepdim=True)
    x_normed = x * torch.rsqrt(variance + eps)
    return x_normed * weight


def swiglu_ffn(x, gate_weight, up_weight, down_weight):
    """SwiGLU FFN: down_proj(silu(gate_proj(x)) * up_proj(x))"""
    gate = torch.nn.functional.silu(x @ gate_weight.T)
    up = x @ up_weight.T
    return (gate * up) @ down_weight.T


def sigmoid_topk_route(hidden, router_weight, layer_idx, bias=None):
    """Route tokens to top-k experts using sigmoid scoring.

    Returns: (expert_indices, expert_weights) for each token
    expert_indices: [seq_len, num_experts_per_tok]
    expert_weights: [seq_len, num_experts_per_tok]
    """
    # Router logits
    logits = hidden @ router_weight.T  # [seq_len, n_routed_experts]
    scores = torch.sigmoid(logits)

    # For expert selection, add bias if available
    selection_scores = scores.clone()
    if bias is not None:
        selection_scores = selection_scores + bias

    # Top-k selection
    topk_weights, topk_indices = torch.topk(
        selection_scores, NUM_EXPERTS_PER_TOK, dim=-1
    )

    # Gather ORIGINAL sigmoid scores (not biased) for the selected experts
    topk_weights = scores.gather(1, topk_indices)

    # Normalize then scale
    if NORM_TOPK_PROB:
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
    topk_weights = topk_weights * ROUTED_SCALING_FACTOR

    return topk_indices, topk_weights


def swiglu_ffn_verbose(x, gate_weight, up_weight, down_weight, label=""):
    """SwiGLU FFN with per-stage diagnostics. x is [1, hidden_size]."""
    gate_out = x @ gate_weight.T  # [1, intermediate]
    up_out = x @ up_weight.T  # [1, intermediate]
    gate_activated = torch.nn.functional.silu(gate_out)
    intermediate = gate_activated * up_out  # SwiGLU
    down_out = intermediate @ down_weight.T  # [1, hidden_size]

    print(
        f"    {label} gate_proj L2={gate_out.norm().item():.4f}, "
        f"up_proj L2={up_out.norm().item():.4f}, "
        f"SwiGLU L2={intermediate.norm().item():.4f}, "
        f"down_proj L2={down_out.norm().item():.4f}"
    )
    return down_out


def compute_moe_layer(normed_hidden, layer_idx, shard_file, bias=None):
    """Compute MoE layer output for all tokens."""
    seq_len = normed_hidden.shape[0]
    last_tok = seq_len - 1
    output = torch.zeros_like(normed_hidden)
    verbose = layer_idx == 6  # Per-expert diagnostics at layer 6

    lp = f"{PREFIX}layers.{layer_idx}.mlp"

    with safe_open(f"{MODEL_DIR}/{shard_file}", framework="pt") as f:
        # Load router weight (BF16)
        router_weight = f.get_tensor(f"{lp}.gate.weight").to(torch.float32)

    # Route
    expert_indices, expert_weights = sigmoid_topk_route(
        normed_hidden, router_weight, layer_idx, bias
    )

    print(f"  Router: top expert indices[last]={expert_indices[last_tok].tolist()}")
    print(
        f"  Router: top expert weights[last]={[f'{w:.4f}' for w in expert_weights[last_tok].tolist()]}"
    )

    # Compute routed expert outputs
    # For each selected expert, load its weights and compute
    unique_experts = expert_indices.unique().tolist()
    print(f"  {len(unique_experts)} unique experts selected")

    for expert_id in unique_experts:
        expert_id = int(expert_id)
        ep = f"{lp}.experts.{expert_id}"

        # Load INT4 expert weights
        gate_w = load_int4_weight(f"{ep}.gate_proj", torch.float32)
        up_w = load_int4_weight(f"{ep}.up_proj", torch.float32)
        down_w = load_int4_weight(f"{ep}.down_proj", torch.float32)

        # Find which tokens use this expert
        mask = expert_indices == expert_id  # [seq_len, topk]

        for tok_idx in range(seq_len):
            positions = mask[tok_idx].nonzero(as_tuple=True)[0]
            if len(positions) == 0:
                continue

            x = normed_hidden[tok_idx : tok_idx + 1]  # [1, hidden_size]

            if verbose and tok_idx == last_tok:
                expert_out = swiglu_ffn_verbose(
                    x, gate_w, up_w, down_w, label=f"e{expert_id}"
                )
            else:
                expert_out = swiglu_ffn(x, gate_w, up_w, down_w)  # [1, hidden_size]

            for pos in positions:
                weight = expert_weights[tok_idx, pos].item()
                output[tok_idx] += weight * expert_out.squeeze(0)

                if verbose and tok_idx == last_tok:
                    accum_l2 = output[tok_idx].norm().item()
                    print(f"    e{expert_id} * {weight:.4f} -> accum L2={accum_l2:.4f}")

    # Always compute expert 144 at layer 6 for comparison with vib3
    if verbose:
        print(f"\n  --- Computing e144 for comparison with vib3 ---")
        ep144 = f"{lp}.experts.144"
        gate_w = load_int4_weight(f"{ep144}.gate_proj", torch.float32)
        up_w = load_int4_weight(f"{ep144}.up_proj", torch.float32)
        down_w = load_int4_weight(f"{ep144}.down_proj", torch.float32)
        x = normed_hidden[last_tok : last_tok + 1]
        e144_out = swiglu_ffn_verbose(x, gate_w, up_w, down_w, label="e144")
        # Also check the intermediate in detail
        gate_out = x @ gate_w.T
        up_out = x @ up_w.T
        gate_act = torch.nn.functional.silu(gate_out)
        intermediate = gate_act * up_out
        print(
            f"    e144 intermediate max_abs={intermediate.abs().max().item():.4f}, "
            f"first8={[f'{v:.6f}' for v in intermediate[0, :8].tolist()]}"
        )
        # Also check: what if gate/up were swapped?
        gate_act_swap = torch.nn.functional.silu(up_out)
        inter_swap = gate_act_swap * gate_out
        down_swap = inter_swap @ down_w.T
        print(
            f"    e144 SWAP TEST (SiLU(up)*gate): SwiGLU L2={inter_swap.norm().item():.4f}, "
            f"down L2={down_swap.norm().item():.4f}"
        )

    if verbose:
        # Print routed total before shared expert
        routed_l2 = output[last_tok].norm().item()
        print(f"\n  Routed experts total (last tok) L2={routed_l2:.4f}")

    # Shared expert
    shared_gate_w = load_tensor(f"{lp}.shared_experts.gate_proj.weight", torch.float32)
    shared_up_w = load_tensor(f"{lp}.shared_experts.up_proj.weight", torch.float32)
    shared_down_w = load_tensor(f"{lp}.shared_experts.down_proj.weight", torch.float32)

    if verbose:
        x = normed_hidden[last_tok : last_tok + 1]
        shared_out_verbose = swiglu_ffn_verbose(
            x, shared_gate_w, shared_up_w, shared_down_w, label="shared"
        )
        # Still compute full shared_out for all tokens
        shared_out = swiglu_ffn(
            normed_hidden, shared_gate_w, shared_up_w, shared_down_w
        )
    else:
        shared_out = swiglu_ffn(
            normed_hidden, shared_gate_w, shared_up_w, shared_down_w
        )

    output = output + shared_out

    return output


def compute_mla_attention_simple(hidden, layer_idx, shard_file, position, kv_cache):
    """Simplified MLA attention for single-step (no proper KV cache for sequence).

    For ground truth purposes, we compute prefill attention where each position
    attends to all previous positions + itself (causal).

    This is a simplified version — we compute the absorbed form.
    """
    seq_len = hidden.shape[0]
    lp = f"{PREFIX}layers.{layer_idx}"

    with safe_open(f"{MODEL_DIR}/{shard_file}", framework="pt") as f:
        # Load attention weights
        q_a_proj = f.get_tensor(f"{lp}.self_attn.q_a_proj.weight").to(
            torch.float32
        )  # [q_lora_rank, hidden]
        q_a_layernorm_w = f.get_tensor(f"{lp}.self_attn.q_a_layernorm.weight").to(
            torch.float32
        )
        q_b_proj = f.get_tensor(f"{lp}.self_attn.q_b_proj.weight").to(
            torch.float32
        )  # [num_heads*(qk_nope+qk_rope), q_lora_rank]

        kv_a_proj = f.get_tensor(f"{lp}.self_attn.kv_a_proj_with_mqa.weight").to(
            torch.float32
        )  # [kv_lora_rank+qk_rope, hidden]
        kv_a_layernorm_w = f.get_tensor(f"{lp}.self_attn.kv_a_layernorm.weight").to(
            torch.float32
        )
        kv_b_proj = f.get_tensor(f"{lp}.self_attn.kv_b_proj.weight").to(
            torch.float32
        )  # [num_heads*(qk_nope+v_dim), kv_lora_rank]

        o_proj = f.get_tensor(f"{lp}.self_attn.o_proj.weight").to(
            torch.float32
        )  # [hidden, num_heads*v_dim]

    # Q path: hidden -> q_a -> layernorm -> q_b -> split into nope + rope parts
    q_compressed = hidden @ q_a_proj.T  # [seq, q_lora_rank]
    q_compressed = rms_norm(q_compressed, q_a_layernorm_w, eps=1e-6)
    q_full = q_compressed @ q_b_proj.T  # [seq, num_heads*(qk_nope+qk_rope)]

    # Split Q into nope and rope parts per head
    q_full = q_full.view(seq_len, NUM_HEADS, QK_NOPE_DIM + QK_ROPE_DIM)
    q_nope = q_full[:, :, :QK_NOPE_DIM]  # [seq, heads, 128]
    q_rope = q_full[:, :, QK_NOPE_DIM:]  # [seq, heads, 64]

    # KV path: hidden -> kv_a -> split compressed_kv and k_rope
    kv_combined = hidden @ kv_a_proj.T  # [seq, kv_lora_rank + qk_rope]
    compressed_kv = kv_combined[:, :KV_LORA_RANK]  # [seq, 512]
    k_rope_raw = kv_combined[:, KV_LORA_RANK:]  # [seq, 64]

    # Layernorm on compressed_kv only
    compressed_kv = rms_norm(compressed_kv, kv_a_layernorm_w, eps=1e-6)

    # kv_b: [num_heads*(qk_nope+v_dim), kv_lora_rank]
    kv_full = compressed_kv @ kv_b_proj.T  # [seq, num_heads*(qk_nope+v_dim)]
    kv_full = kv_full.view(seq_len, NUM_HEADS, QK_NOPE_DIM + V_DIM)
    k_nope = kv_full[:, :, :QK_NOPE_DIM]  # [seq, heads, 128]
    v = kv_full[:, :, QK_NOPE_DIM:]  # [seq, heads, 128]

    # Apply RoPE to q_rope and k_rope
    # k_rope is shared across heads
    k_rope = k_rope_raw.unsqueeze(1).expand(-1, NUM_HEADS, -1)  # [seq, heads, 64]

    def yarn_find_correction_dim(
        num_rotations, dim, base=10000, max_position_embeddings=2048
    ):
        return (
            dim * math.log(max_position_embeddings / (num_rotations * 2 * math.pi))
        ) / (2 * math.log(base))

    def yarn_find_correction_range(
        low_rot, high_rot, dim, base=10000, max_position_embeddings=2048
    ):
        low = math.floor(
            yarn_find_correction_dim(low_rot, dim, base, max_position_embeddings)
        )
        high = math.ceil(
            yarn_find_correction_dim(high_rot, dim, base, max_position_embeddings)
        )
        return max(low, 0), min(high, dim - 1)

    def yarn_linear_ramp_mask(min_val, max_val, dim):
        if min_val == max_val:
            max_val += 0.001
        linear_func = (torch.arange(dim, dtype=torch.float32) - min_val) / (
            max_val - min_val
        )
        return torch.clamp(linear_func, 0, 1)

    def compute_yarn_inv_freq(dim):
        """Compute YaRN-scaled inverse frequencies matching DeepseekV3YarnRotaryEmbedding."""
        freq_extra = 1.0 / (
            ROPE_THETA ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim)
        )
        freq_inter = 1.0 / (
            ROPE_FACTOR
            * ROPE_THETA ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim)
        )

        low, high = yarn_find_correction_range(
            ROPE_BETA_FAST, ROPE_BETA_SLOW, dim, ROPE_THETA, ROPE_ORIGINAL_MAX_POS
        )
        inv_freq_mask = 1.0 - yarn_linear_ramp_mask(low, high, dim // 2)
        inv_freq = freq_inter * (1 - inv_freq_mask) + freq_extra * inv_freq_mask
        return inv_freq

    def apply_rope(x, seq_offset=0):
        """Apply YaRN rotary position embeddings (interleaved pairs)."""
        d = x.shape[-1]
        positions = torch.arange(
            seq_offset, seq_offset + x.shape[0], dtype=torch.float32
        )
        inv_freq = compute_yarn_inv_freq(d)
        angles = positions.unsqueeze(1) * inv_freq.unsqueeze(0)  # [seq, d/2]
        cos = torch.cos(angles)
        sin = torch.sin(angles)

        # Interleaved pairs: (x0, x1), (x2, x3), ...
        x_pairs = x.view(*x.shape[:-1], d // 2, 2)
        x0 = x_pairs[..., 0]
        x1 = x_pairs[..., 1]

        # Broadcast cos/sin to match head dimension
        while cos.dim() < x0.dim():
            cos = cos.unsqueeze(1)
            sin = sin.unsqueeze(1)

        out0 = x0 * cos - x1 * sin
        out1 = x0 * sin + x1 * cos

        out = torch.stack([out0, out1], dim=-1)
        return out.view(*x.shape)

    q_rope = apply_rope(q_rope)
    k_rope = apply_rope(k_rope)

    # Full Q key: [nope, rope]
    q = torch.cat([q_nope, q_rope], dim=-1)  # [seq, heads, 192]
    k = torch.cat([k_nope, k_rope], dim=-1)  # [seq, heads, 192]

    # Attention: Q @ K^T / sqrt(d) with YaRN mscale correction
    # DeepSeek-V3 modeling code applies mscale^2 to the softmax scale
    # mscale = 0.1 * ln(max_position_embeddings / original_max_position_embeddings) + 1.0
    #        = 0.1 * ln(64) + 1.0 = 1.4159 (when factor=64, mscale_all_dim=1.0)
    mscale = 0.1 * math.log(64.0) + 1.0  # = 1.4159
    scale = (1.0 / math.sqrt(QK_NOPE_DIM + QK_ROPE_DIM)) * (mscale * mscale)  # ~0.1448

    # [seq, heads, 192] x [seq, heads, 192]^T -> [heads, seq, seq]
    q_t = q.permute(1, 0, 2)  # [heads, seq, 192]
    k_t = k.permute(1, 0, 2)  # [heads, seq, 192]

    attn_weights = torch.bmm(q_t, k_t.transpose(1, 2)) * scale  # [heads, seq, seq]

    # Causal mask
    causal_mask = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool), diagonal=1)
    attn_weights.masked_fill_(causal_mask.unsqueeze(0), float("-inf"))

    attn_weights = torch.softmax(attn_weights, dim=-1)

    # Attend to values
    v_t = v.permute(1, 0, 2)  # [heads, seq, v_dim]
    attn_output = torch.bmm(attn_weights, v_t)  # [heads, seq, v_dim]

    # Reshape: [seq, heads*v_dim]
    attn_output = (
        attn_output.permute(1, 0, 2).contiguous().view(seq_len, NUM_HEADS * V_DIM)
    )

    # Output projection
    output = attn_output @ o_proj.T  # [seq, hidden_size]

    return output


def load_e_score_bias():
    """Load e_score_correction_bias if available."""
    bias_path = f"{MODEL_DIR}/e_score_correction_bias.bin"
    try:
        with open(bias_path, "rb") as f:
            data = f.read()
        # 61 layers × 384 experts × f32
        n_floats = len(data) // 4
        values = struct.unpack(f"<{n_floats}f", data)
        bias = torch.tensor(values, dtype=torch.float32).reshape(61, 384)
        print(f"Loaded e_score_correction_bias: {bias.shape}")
        return bias
    except FileNotFoundError:
        print("No e_score_correction_bias.bin found")
        return None


def main():
    print("=" * 70)
    print("Ground Truth Reference: Kimi K2.5 Hidden State Computation")
    print("=" * 70)

    # Tokenize
    tokenizer = Tokenizer.from_file(f"{MODEL_DIR}/tokenizer.json")
    template = (
        "<|im_system|>system<|im_middle|>"
        "You are Kimi, an AI assistant created by Moonshot AI."
        "<|im_end|>"
        "<|im_user|>user<|im_middle|>Hello<|im_end|>"
        "<|im_assistant|>assistant<|im_middle|>"
    )
    tokens = tokenizer.encode(template, add_special_tokens=False).ids
    print(f"\nTokens: {len(tokens)}")
    print(f"Token IDs: {tokens}")
    print(f"Last token (pos {len(tokens) - 1}): {tokens[-1]}")

    # Load e_score_correction_bias
    bias = load_e_score_bias()

    # Load embedding table
    print("\n--- Loading embeddings ---")
    t0 = time.time()
    embed_weight = load_tensor(f"{PREFIX}embed_tokens.weight", torch.float32)
    print(f"  Embedding table: {embed_weight.shape}, loaded in {time.time() - t0:.1f}s")

    # Embed tokens
    input_ids = torch.tensor(tokens, dtype=torch.long)
    hidden = embed_weight[input_ids]  # [seq_len, hidden_size]
    del embed_weight  # Free memory

    print(f"\n--- Embedding output ---")
    last_tok_idx = len(tokens) - 1
    h_last = hidden[last_tok_idx]
    print(f"  Token {last_tok_idx} (id={tokens[last_tok_idx]}):")
    print(f"    first8: [{', '.join(f'{v:.6f}' for v in h_last[:8].tolist())}]")
    print(f"    min={h_last.min().item():.6f}, max={h_last.max().item():.6f}")
    print(f"    mean={h_last.mean().item():.6f}, L2={h_last.norm().item():.4f}")

    # Process layers
    for layer_idx in range(NUM_LAYERS_TO_COMPUTE):
        print(f"\n{'=' * 50}")
        print(f"Layer {layer_idx}")
        print(f"{'=' * 50}")

        is_moe = layer_idx >= FIRST_K_DENSE
        shard_idx = layer_idx + 1  # Layer 0 -> shard 1, Layer 1 -> shard 2, etc.
        shard_file = f"model-{shard_idx:05d}-of-000064.safetensors"

        lp = f"{PREFIX}layers.{layer_idx}"

        # --- Attention sublayer ---
        print(f"  Loading attention norm...")
        attn_norm_w = load_tensor(f"{lp}.input_layernorm.weight", torch.float32)
        normed_for_attn = rms_norm(hidden, attn_norm_w)
        del attn_norm_w

        h_normed_last = normed_for_attn[last_tok_idx]
        print(f"  Normed (for attn) last token L2={h_normed_last.norm().item():.4f}")

        print(f"  Computing MLA attention...")
        t0 = time.time()
        attn_output = compute_mla_attention_simple(
            normed_for_attn, layer_idx, shard_file, 0, None
        )
        print(f"  Attention computed in {time.time() - t0:.1f}s")
        del normed_for_attn

        attn_last = attn_output[last_tok_idx]
        print(f"  Attention output last token L2={attn_last.norm().item():.4f}")

        # Residual connection
        hidden = hidden + attn_output
        del attn_output

        h_after_attn = hidden[last_tok_idx]
        print(f"  After attn residual last token L2={h_after_attn.norm().item():.4f}")

        # --- FFN/MoE sublayer ---
        print(f"  Loading FFN norm...")
        ffn_norm_w = load_tensor(f"{lp}.post_attention_layernorm.weight", torch.float32)
        normed_for_ffn = rms_norm(hidden, ffn_norm_w)
        del ffn_norm_w

        h_normed_ffn_last = normed_for_ffn[last_tok_idx]
        print(f"  Normed (for FFN) last token L2={h_normed_ffn_last.norm().item():.4f}")

        # Dump hidden states at layer 6 for comparison with vib3
        if layer_idx == 6:
            import os

            dump_dir = "/code/vib3/dump"
            os.makedirs(dump_dir, exist_ok=True)
            # Dump pre-norm (residual) hidden state
            pre_norm_f16 = hidden[last_tok_idx].to(torch.float16).numpy()
            pre_norm_f16.tofile(f"{dump_dir}/gt_l6_prenorm_pos{last_tok_idx}.bin")
            # Dump post-norm hidden state
            post_norm_f16 = normed_for_ffn[last_tok_idx].to(torch.float16).numpy()
            post_norm_f16.tofile(f"{dump_dir}/gt_l6_postnorm_pos{last_tok_idx}.bin")
            print(f"  DUMPED GT hidden states to {dump_dir}/gt_l6_*.bin")

        if is_moe:
            print(f"  Computing MoE (384 experts, top-8)...")
            t0 = time.time()
            layer_bias = bias[layer_idx] if bias is not None else None
            ffn_output = compute_moe_layer(
                normed_for_ffn, layer_idx, shard_file, layer_bias
            )
            print(f"  MoE computed in {time.time() - t0:.1f}s")
        else:
            print(f"  Computing dense FFN...")
            t0 = time.time()
            gate_w = load_tensor(f"{lp}.mlp.gate_proj.weight", torch.float32)
            up_w = load_tensor(f"{lp}.mlp.up_proj.weight", torch.float32)
            down_w = load_tensor(f"{lp}.mlp.down_proj.weight", torch.float32)
            ffn_output = swiglu_ffn(normed_for_ffn, gate_w, up_w, down_w)
            del gate_w, up_w, down_w
            print(f"  Dense FFN computed in {time.time() - t0:.1f}s")

        del normed_for_ffn

        ffn_last = ffn_output[last_tok_idx]
        print(f"  FFN/MoE output last token L2={ffn_last.norm().item():.4f}")

        # Residual connection
        hidden = hidden + ffn_output
        del ffn_output

        h_final = hidden[last_tok_idx]
        print(
            f"  *** After layer {layer_idx}: L2={h_final.norm().item():.4f}, "
            f"min={h_final.min().item():.4f}, max={h_final.max().item():.4f}, "
            f"mean={h_final.mean().item():.6f}"
        )

        # Dump FP32 hidden state for layers 0-6 for comparison with vib3
        if layer_idx <= 6:
            import os

            dump_dir = "/code/vib3/dump"
            os.makedirs(dump_dir, exist_ok=True)
            dump_path = f"{dump_dir}/gt_hidden_f32_L{layer_idx}_pos{last_tok_idx}.bin"
            h_final.numpy().astype(np.float32).tofile(dump_path)
            print(f"  DUMPED GT FP32 hidden state to {dump_path}")

    # Summary
    print(f"\n{'=' * 70}")
    print("SUMMARY: Hidden state L2 norms (last token) after each layer")
    print(f"{'=' * 70}")
    print("(Compare these values with vib3's PREFILL DIAG output)")
    print()
    print("vib3 reference values (from logs):")
    print("  Layer 0: L2=0.74")
    print("  Layer 1: L2=0.79")
    print("  Layer 2: L2=0.92")
    print("  Layer 30: L2=3044.15")
    print("  Layer 60: L2=4694.88")


if __name__ == "__main__":
    main()
