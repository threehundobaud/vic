#!/usr/bin/env python3
"""
Targeted check: Does Mixtral-8x7B L1 expert 3 produce large outputs for BOS token?
Uses the BF16 (unquantized) safetensors to compute ground truth through L0 and L1.

This answers: is the L1 explosion (L2=2797 in SwiGLU intermediate) a quantization
artifact, or does the model genuinely produce large outputs at L1?
"""

import json
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
RMS_NORM_EPS = cfg.get("rms_norm_eps", 1e-5)
ROPE_THETA = cfg.get("rope_theta", 1000000.0)

print(
    f"Mixtral config: hidden={HIDDEN_SIZE}, heads={NUM_HEADS}, kv_heads={NUM_KV_HEADS}, "
    f"head_dim={HEAD_DIM}, intermediate={INTERMEDIATE_SIZE}, experts={NUM_EXPERTS}, top_k={TOP_K}"
)

# Find which shard contains each tensor
# No index file, so scan all shards
_tensor_shard_map = {}
_shard_cache = {}


def build_shard_map():
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
    """RMSNorm in float32"""
    x_f32 = x.float()
    rms = torch.sqrt(torch.mean(x_f32**2) + eps)
    return (x_f32 / rms * weight.float()).to(x.dtype)


def apply_rotary_pos_emb(x, pos, head_dim, theta=ROPE_THETA):
    """Apply RoPE to a tensor of shape [num_heads, head_dim]"""
    half_dim = head_dim // 2
    freqs = 1.0 / (theta ** (torch.arange(0, half_dim, dtype=torch.float32) / half_dim))
    angles = pos * freqs  # [half_dim]
    cos_vals = torch.cos(angles)
    sin_vals = torch.sin(angles)

    x_f32 = x.float()
    x1 = x_f32[..., :half_dim]
    x2 = x_f32[..., half_dim:]
    out1 = x1 * cos_vals - x2 * sin_vals
    out2 = x2 * cos_vals + x1 * sin_vals
    return torch.cat([out1, out2], dim=-1).to(x.dtype)


print("Building shard map...")
build_shard_map()

# Load tokenizer
tokenizer = Tokenizer.from_file(f"{MODEL_DIR}/tokenizer.json")

# Tokenize with BOS
prompt = "What is 2+2?"
encoded = tokenizer.encode(prompt)
tokens = [1] + encoded.ids  # BOS=1 for Mistral
print(f"Tokens ({len(tokens)}): {tokens}")
print(f"  Decoded: {[tokenizer.decode([t]) for t in tokens]}")

# Load embedding
embed_weight = load_tensor("model.embed_tokens.weight", dtype=torch.float32)
print(f"Embedding shape: {embed_weight.shape}, dtype: {embed_weight.dtype}")

# ── Process BOS token (pos=0) through L0 and L1 ──
tok_id = tokens[0]  # BOS
print(f"\n=== Processing BOS token (id={tok_id}) at pos=0 ===")

h = embed_weight[tok_id].clone()  # [4096]
print(f"Embedding: L2={h.norm():.4f}, min={h.min():.6f}, max={h.max():.6f}")

# ── Layer 0: Attention ──
print(f"\n--- Layer 0: Attention ---")
ln1_w = load_tensor("model.layers.0.input_layernorm.weight", dtype=torch.float32)
h_normed = rms_norm(h, ln1_w)
print(f"After input_layernorm: L2={h_normed.norm():.4f}")

# GQA attention for single token at pos=0
q_proj = load_tensor("model.layers.0.self_attn.q_proj.weight", dtype=torch.float32)
k_proj = load_tensor("model.layers.0.self_attn.k_proj.weight", dtype=torch.float32)
v_proj = load_tensor("model.layers.0.self_attn.v_proj.weight", dtype=torch.float32)
o_proj = load_tensor("model.layers.0.self_attn.o_proj.weight", dtype=torch.float32)

q = q_proj @ h_normed  # [4096]
k = k_proj @ h_normed  # [1024] (8 kv heads * 128)
v = v_proj @ h_normed  # [1024]

# Reshape to heads
q_heads = q.view(NUM_HEADS, HEAD_DIM)  # [32, 128]
k_heads = k.view(NUM_KV_HEADS, HEAD_DIM)  # [8, 128]
v_heads = v.view(NUM_KV_HEADS, HEAD_DIM)  # [8, 128]

# Apply RoPE
q_heads = apply_rotary_pos_emb(q_heads, 0, HEAD_DIM)
k_heads = apply_rotary_pos_emb(k_heads, 0, HEAD_DIM)

# Self-attention at pos=0: single token, score = q·k / sqrt(d)
# With GQA: each query head group attends to one KV head
heads_per_kv = NUM_HEADS // NUM_KV_HEADS  # 4
attn_out = torch.zeros(NUM_HEADS, HEAD_DIM)
for h_idx in range(NUM_HEADS):
    kv_idx = h_idx // heads_per_kv
    score = (q_heads[h_idx] @ k_heads[kv_idx]) / (HEAD_DIM**0.5)
    attn_weight = torch.softmax(torch.tensor([score]), dim=0)  # single position = 1.0
    attn_out[h_idx] = attn_weight[0] * v_heads[kv_idx]

attn_flat = attn_out.reshape(-1)  # [4096]
o_out = o_proj @ attn_flat
print(f"Attention output: L2={o_out.norm():.4f}")

h = h + o_out  # residual
print(f"After L0 attention residual: L2={h.norm():.4f}")

# ── Layer 0: MoE ──
print(f"\n--- Layer 0: MoE ---")
ln2_w = load_tensor(
    "model.layers.0.post_attention_layernorm.weight", dtype=torch.float32
)
h_moe_normed = rms_norm(h, ln2_w)
print(f"After post_attn_layernorm: L2={h_moe_normed.norm():.4f}")

# Router
router_w = load_tensor(
    "model.layers.0.block_sparse_moe.gate.weight", dtype=torch.float32
)
router_logits = router_w @ h_moe_normed  # [8]
router_probs = torch.softmax(router_logits, dim=0)
top_vals, top_idx = router_probs.topk(TOP_K)
# Normalize top-k weights
top_vals = top_vals / top_vals.sum()
top_experts_str = ", ".join(
    [f"(e{int(top_idx[i])}, {float(top_vals[i]):.4f})" for i in range(TOP_K)]
)
print(f"L0 Router: top experts = [{top_experts_str}]")

# Compute expert outputs
moe_out = torch.zeros(HIDDEN_SIZE)
for i in range(TOP_K):
    eidx = int(top_idx[i])
    weight = float(top_vals[i])

    w1 = load_tensor(
        f"model.layers.0.block_sparse_moe.experts.{eidx}.w1.weight", dtype=torch.float32
    )
    w2 = load_tensor(
        f"model.layers.0.block_sparse_moe.experts.{eidx}.w2.weight", dtype=torch.float32
    )
    w3 = load_tensor(
        f"model.layers.0.block_sparse_moe.experts.{eidx}.w3.weight", dtype=torch.float32
    )

    gate = w1 @ h_moe_normed  # [14336]
    up = w3 @ h_moe_normed  # [14336]
    inter = torch.nn.functional.silu(gate) * up  # SwiGLU
    print(
        f"  L0 Expert {eidx} (w={weight:.4f}): gate L2={gate.norm():.4f}, up L2={up.norm():.4f}, "
        f"SwiGLU L2={inter.norm():.4f}, max_abs={inter.abs().max():.4f}"
    )

    down = w2 @ inter  # [4096]
    moe_out += weight * down

print(f"L0 MoE output: L2={moe_out.norm():.4f}")
h = h + moe_out
print(f"After L0 MoE residual: L2={h.norm():.4f}")

# ── Layer 1: Attention ──
print(f"\n--- Layer 1: Attention ---")
ln1_w = load_tensor("model.layers.1.input_layernorm.weight", dtype=torch.float32)
h_normed = rms_norm(h, ln1_w)
print(f"After input_layernorm: L2={h_normed.norm():.4f}")

q_proj = load_tensor("model.layers.1.self_attn.q_proj.weight", dtype=torch.float32)
k_proj = load_tensor("model.layers.1.self_attn.k_proj.weight", dtype=torch.float32)
v_proj = load_tensor("model.layers.1.self_attn.v_proj.weight", dtype=torch.float32)
o_proj = load_tensor("model.layers.1.self_attn.o_proj.weight", dtype=torch.float32)

q = q_proj @ h_normed
k = k_proj @ h_normed
v = v_proj @ h_normed

q_heads = q.view(NUM_HEADS, HEAD_DIM)
k_heads = k.view(NUM_KV_HEADS, HEAD_DIM)
v_heads = v.view(NUM_KV_HEADS, HEAD_DIM)

# At pos=0, we need to attend to BOS KV from L1's perspective
# But for pos=0, it's just self-attention to the single token
q_heads = apply_rotary_pos_emb(q_heads, 0, HEAD_DIM)
k_heads = apply_rotary_pos_emb(k_heads, 0, HEAD_DIM)

attn_out = torch.zeros(NUM_HEADS, HEAD_DIM)
for h_idx in range(NUM_HEADS):
    kv_idx = h_idx // heads_per_kv
    score = (q_heads[h_idx] @ k_heads[kv_idx]) / (HEAD_DIM**0.5)
    attn_weight = torch.softmax(torch.tensor([score]), dim=0)
    attn_out[h_idx] = attn_weight[0] * v_heads[kv_idx]

attn_flat = attn_out.reshape(-1)
o_out = o_proj @ attn_flat
print(f"Attention output: L2={o_out.norm():.4f}")

h = h + o_out
print(f"After L1 attention residual: L2={h.norm():.4f}")

# ── Layer 1: MoE ──
print(f"\n--- Layer 1: MoE ---")
ln2_w = load_tensor(
    "model.layers.1.post_attention_layernorm.weight", dtype=torch.float32
)
h_moe_normed = rms_norm(h, ln2_w)
print(f"After post_attn_layernorm: L2={h_moe_normed.norm():.4f}")

# Check hidden state going into MoE
print(f"  h (pre-norm) L2={h.norm():.4f}, max_abs={h.abs().max():.6f}")
print(
    f"  h_moe_normed L2={h_moe_normed.norm():.4f}, max_abs={h_moe_normed.abs().max():.6f}"
)

# Router
router_w = load_tensor(
    "model.layers.1.block_sparse_moe.gate.weight", dtype=torch.float32
)
router_logits = router_w @ h_moe_normed
router_probs = torch.softmax(router_logits, dim=0)
top_vals, top_idx = router_probs.topk(TOP_K)
top_vals = top_vals / top_vals.sum()
print(
    f"L1 Router logits: {[f'{float(router_logits[i]):.4f}' for i in range(NUM_EXPERTS)]}"
)
print(
    f"L1 Router probs: {[f'{float(router_probs[i]):.4f}' for i in range(NUM_EXPERTS)]}"
)
top_experts_str = ", ".join(
    [f"(e{int(top_idx[i])}, {float(top_vals[i]):.4f})" for i in range(TOP_K)]
)
print(f"L1 Router: top experts = [{top_experts_str}]")

# Compute ALL expert outputs for analysis
print(f"\nL1 Expert SwiGLU intermediates for ALL experts (not just top-k):")
for eidx in range(NUM_EXPERTS):
    w1 = load_tensor(
        f"model.layers.1.block_sparse_moe.experts.{eidx}.w1.weight", dtype=torch.float32
    )
    w3 = load_tensor(
        f"model.layers.1.block_sparse_moe.experts.{eidx}.w3.weight", dtype=torch.float32
    )

    gate = w1 @ h_moe_normed
    up = w3 @ h_moe_normed
    inter = torch.nn.functional.silu(gate) * up

    marker = " <<<" if eidx in [int(top_idx[j]) for j in range(TOP_K)] else ""
    print(
        f"  Expert {eidx}: SwiGLU L2={inter.norm():.4f}, max_abs={inter.abs().max():.4f}, "
        f"gate L2={gate.norm():.4f}, up L2={up.norm():.4f}{marker}"
    )

# Now compute full MoE output for top-k experts
moe_out = torch.zeros(HIDDEN_SIZE)
for i in range(TOP_K):
    eidx = int(top_idx[i])
    weight = float(top_vals[i])

    w1 = load_tensor(
        f"model.layers.1.block_sparse_moe.experts.{eidx}.w1.weight", dtype=torch.float32
    )
    w2 = load_tensor(
        f"model.layers.1.block_sparse_moe.experts.{eidx}.w2.weight", dtype=torch.float32
    )
    w3 = load_tensor(
        f"model.layers.1.block_sparse_moe.experts.{eidx}.w3.weight", dtype=torch.float32
    )

    gate = w1 @ h_moe_normed
    up = w3 @ h_moe_normed
    inter = torch.nn.functional.silu(gate) * up
    down = w2 @ inter

    print(f"\n  L1 Expert {eidx} (w={weight:.4f}):")
    print(f"    gate L2={gate.norm():.4f}, up L2={up.norm():.4f}")
    print(
        f"    SwiGLU intermediate: L2={inter.norm():.4f}, max_abs={inter.abs().max():.4f}"
    )
    print(f"    down_proj output: L2={down.norm():.4f}, max_abs={down.abs().max():.4f}")

    moe_out += weight * down

print(f"\nL1 MoE output: L2={moe_out.norm():.4f}, max_abs={moe_out.abs().max():.4f}")
h_after_l1 = h + moe_out
print(
    f"After L1 MoE residual: L2={h_after_l1.norm():.4f}, max_abs={h_after_l1.abs().max():.4f}"
)

# ── Compare with vib3 engine values ──
print(f"\n{'=' * 60}")
print(f"COMPARISON WITH VIB3 ENGINE (from logs):")
print(f"{'=' * 60}")
print(f"  vib3 MOE_NORMED_F32 L1 pos=0: L2=25.928232")
print(f"  vib3 HIDDEN_STATE_F32 L1 pos=0: L2=5.032650")
print(f"  vib3 Expert plan L1: e3=0.9524, e4=0.0143")
print(f"  vib3 SWIGLU_INTER L1 e3: L2=2797.127, max_abs=2786.0")
print(f"  vib3 SWIGLU_INTER L1 e4: L2=226.590, max_abs=208.875")
print(f"  vib3 MOE OUTPUT L1: L2=2565.14, max_abs=2537.81")
