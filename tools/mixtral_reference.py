#!/usr/bin/env python3
"""
Run Mixtral-8x7B through HuggingFace transformers and extract per-layer hidden states.
Compares against vib3 engine output to identify where divergence occurs.

Uses the BF16 (unquantized) weights at /home/brian/code/vib3/models/mixtral/
"""

import json
import os
import sys
import torch
import numpy as np
from safetensors import safe_open
from tokenizers import Tokenizer

MODEL_DIR = "/home/brian/code/vib3/models/mixtral"

# Load config
with open(f"{MODEL_DIR}/config.json") as f:
    cfg = json.load(f)

HIDDEN_SIZE = cfg["hidden_size"]  # 4096
NUM_HEADS = cfg["num_attention_heads"]  # 32
NUM_KV_HEADS = cfg["num_key_value_heads"]  # 8
HEAD_DIM = HIDDEN_SIZE // NUM_HEADS  # 128
INTERMEDIATE_SIZE = cfg["intermediate_size"]  # 14336
NUM_EXPERTS = cfg["num_local_experts"]  # 8
TOP_K = cfg["num_experts_per_tok"]  # 2
NUM_LAYERS = cfg["num_hidden_layers"]  # 32
RMS_NORM_EPS = cfg.get("rms_norm_eps", 1e-5)
ROPE_THETA = cfg.get("rope_theta", 1000000.0)

print(
    f"Mixtral config: hidden={HIDDEN_SIZE}, layers={NUM_LAYERS}, heads={NUM_HEADS}, "
    f"kv_heads={NUM_KV_HEADS}, intermediate={INTERMEDIATE_SIZE}, experts={NUM_EXPERTS}"
)

# Build shard map
_tensor_shard_map = {}
for i in range(1, 20):
    path = f"{MODEL_DIR}/model-{i:05d}-of-00019.safetensors"
    try:
        with safe_open(path, framework="pt") as f:
            for key in f.keys():
                _tensor_shard_map[key] = path
    except:
        pass
print(f"Found {len(_tensor_shard_map)} tensors across shards")


def load_tensor(key, dtype=torch.float32):
    path = _tensor_shard_map[key]
    with safe_open(path, framework="pt") as f:
        return f.get_tensor(key).to(dtype)


def rms_norm(x, weight, eps=RMS_NORM_EPS):
    x_f32 = x.float()
    rms = torch.sqrt(torch.mean(x_f32**2, dim=-1, keepdim=True) + eps)
    return x_f32 / rms * weight.float()


def apply_rotary_pos_emb_batch(x, positions, head_dim, theta=ROPE_THETA):
    """Apply RoPE. x: [num_heads, seq_len, head_dim], positions: [seq_len]"""
    half_dim = head_dim // 2
    freqs = 1.0 / (theta ** (torch.arange(0, half_dim, dtype=torch.float32) / half_dim))
    # [seq_len, half_dim]
    angles = positions.float().unsqueeze(1) * freqs.unsqueeze(0)
    cos_vals = torch.cos(angles)  # [seq_len, half_dim]
    sin_vals = torch.sin(angles)

    x_f32 = x.float()
    x1 = x_f32[..., :half_dim]  # [num_heads, seq_len, half_dim]
    x2 = x_f32[..., half_dim:]
    out1 = x1 * cos_vals.unsqueeze(0) - x2 * sin_vals.unsqueeze(0)
    out2 = x2 * cos_vals.unsqueeze(0) + x1 * sin_vals.unsqueeze(0)
    return torch.cat([out1, out2], dim=-1)


# Tokenize
tokenizer = Tokenizer.from_file(f"{MODEL_DIR}/tokenizer.json")
prompt = "[INST] What is 2+2? [/INST]"
encoded = tokenizer.encode(prompt)
tokens = encoded.ids
print(f"Tokens ({len(tokens)}): {tokens}")

# Load embedding
embed_weight = load_tensor("model.embed_tokens.weight", dtype=torch.float32)
seq_len = len(tokens)

# Embed all tokens: [seq_len, hidden_size]
h = embed_weight[tokens]
print(f"Embedding: shape={h.shape}, L2(last)={h[-1].norm():.4f}")

positions = torch.arange(seq_len)
heads_per_kv = NUM_HEADS // NUM_KV_HEADS

# Process through all layers
for layer_idx in range(NUM_LAYERS):
    # === Attention sublayer ===
    ln1_w = load_tensor(
        f"model.layers.{layer_idx}.input_layernorm.weight", dtype=torch.float32
    )
    h_normed = rms_norm(h, ln1_w)  # [seq_len, hidden_size]

    q_proj = load_tensor(
        f"model.layers.{layer_idx}.self_attn.q_proj.weight", dtype=torch.float32
    )
    k_proj = load_tensor(
        f"model.layers.{layer_idx}.self_attn.k_proj.weight", dtype=torch.float32
    )
    v_proj = load_tensor(
        f"model.layers.{layer_idx}.self_attn.v_proj.weight", dtype=torch.float32
    )
    o_proj = load_tensor(
        f"model.layers.{layer_idx}.self_attn.o_proj.weight", dtype=torch.float32
    )

    # Projections: [seq_len, proj_dim]
    q = h_normed @ q_proj.T  # [seq_len, 4096]
    k = h_normed @ k_proj.T  # [seq_len, 1024]
    v = h_normed @ v_proj.T  # [seq_len, 1024]

    # Reshape to heads: [num_heads, seq_len, head_dim]
    q_heads = q.view(seq_len, NUM_HEADS, HEAD_DIM).transpose(0, 1)
    k_heads = k.view(seq_len, NUM_KV_HEADS, HEAD_DIM).transpose(0, 1)
    v_heads = v.view(seq_len, NUM_KV_HEADS, HEAD_DIM).transpose(0, 1)

    # Apply RoPE
    q_heads = apply_rotary_pos_emb_batch(q_heads, positions, HEAD_DIM)
    k_heads = apply_rotary_pos_emb_batch(k_heads, positions, HEAD_DIM)

    # GQA attention with causal mask
    attn_out = torch.zeros(NUM_HEADS, seq_len, HEAD_DIM)
    for h_idx in range(NUM_HEADS):
        kv_idx = h_idx // heads_per_kv
        # scores: [seq_len, seq_len]
        scores = (q_heads[h_idx] @ k_heads[kv_idx].T) / (HEAD_DIM**0.5)
        # Causal mask
        mask = torch.triu(torch.ones(seq_len, seq_len) * float("-inf"), diagonal=1)
        scores = scores + mask
        weights = torch.softmax(scores, dim=-1)
        attn_out[h_idx] = weights @ v_heads[kv_idx]

    # [seq_len, hidden_size]
    attn_flat = attn_out.transpose(0, 1).reshape(seq_len, -1)
    o_out = attn_flat @ o_proj.T

    h = h + o_out  # residual

    # Dump after-attention hidden state for layers 0-5
    if layer_idx <= 5:
        dump_dir = "/home/brian/code/vib3/dump"
        os.makedirs(dump_dir, exist_ok=True)
        np.array(h[-1].numpy(), dtype=np.float32).tofile(
            f"{dump_dir}/gt_mixtral_after_attn_f32_L{layer_idx}_lastpos.bin"
        )
        last_after_attn = h[-1]
        print(
            f"  After attn: L2(last)={last_after_attn.norm():.4f}, "
            f"first4=[{last_after_attn[0]:.6f},{last_after_attn[1]:.6f},{last_after_attn[2]:.6f},{last_after_attn[3]:.6f}]"
        )

    # === MoE sublayer ===
    ln2_w = load_tensor(
        f"model.layers.{layer_idx}.post_attention_layernorm.weight", dtype=torch.float32
    )
    h_moe_normed = rms_norm(h, ln2_w)

    # Router: [seq_len, num_experts]
    router_w = load_tensor(
        f"model.layers.{layer_idx}.block_sparse_moe.gate.weight", dtype=torch.float32
    )
    router_logits = h_moe_normed @ router_w.T
    router_probs = torch.softmax(router_logits, dim=-1)
    top_vals, top_idx = router_probs.topk(TOP_K, dim=-1)
    # Normalize
    top_vals = top_vals / top_vals.sum(dim=-1, keepdim=True)

    # MoE output: process each token
    moe_out = torch.zeros_like(h)
    for pos in range(seq_len):
        for i in range(TOP_K):
            eidx = int(top_idx[pos, i])
            weight = float(top_vals[pos, i])

            w1 = load_tensor(
                f"model.layers.{layer_idx}.block_sparse_moe.experts.{eidx}.w1.weight",
                dtype=torch.float32,
            )
            w2 = load_tensor(
                f"model.layers.{layer_idx}.block_sparse_moe.experts.{eidx}.w2.weight",
                dtype=torch.float32,
            )
            w3 = load_tensor(
                f"model.layers.{layer_idx}.block_sparse_moe.experts.{eidx}.w3.weight",
                dtype=torch.float32,
            )

            gate = w1 @ h_moe_normed[pos]
            up = w3 @ h_moe_normed[pos]
            inter = torch.nn.functional.silu(gate) * up
            down = w2 @ inter
            moe_out[pos] += weight * down

    h = h + moe_out

    # Print per-layer diagnostics for last token
    last_h = h[-1]
    print(
        f"Layer {layer_idx:2d} done: L2(last)={last_h.norm():.4f}, "
        f"max_abs={last_h.abs().max():.4f}, "
        f"min={last_h.min():.4f}, max={last_h.max():.4f}"
    )

    # Free expert weights
    del q_proj, k_proj, v_proj, o_proj, router_w

    # For first and last token, dump hidden state
    if layer_idx in [0, 1, 2, 5, 10, 15, 20, 25, 30, 31]:
        dump_dir = "/home/brian/code/vib3/dump"
        os.makedirs(dump_dir, exist_ok=True)
        # Save last token hidden state
        np.array(last_h.numpy(), dtype=np.float32).tofile(
            f"{dump_dir}/gt_mixtral_hidden_f32_L{layer_idx}_lastpos.bin"
        )
        # Save first token (BOS) hidden state
        np.array(h[0].numpy(), dtype=np.float32).tofile(
            f"{dump_dir}/gt_mixtral_hidden_f32_L{layer_idx}_pos0.bin"
        )

# Final norm + logit comparison
print(f"\n=== Final state ===")
final_norm_w = load_tensor("model.norm.weight", dtype=torch.float32)
h_final = rms_norm(h[-1:], final_norm_w)[0]
print(f"After final norm: L2={h_final.norm():.4f}")

# LM head
lm_head_w = load_tensor("lm_head.weight", dtype=torch.float32)
logits = lm_head_w @ h_final
print(f"Logits: L2={logits.norm():.4f}, min={logits.min():.4f}, max={logits.max():.4f}")

# Top-5 token predictions
top5_vals, top5_idx = logits.topk(5)
for i in range(5):
    tid = int(top5_idx[i])
    tok_str = tokenizer.decode([tid])
    print(
        f"  Top {i + 1}: token={tid} ({repr(tok_str)}), logit={float(top5_vals[i]):.4f}"
    )
