#!/usr/bin/env python3
"""
Reference implementation: compute hidden states for Kimi K2.5 through first few layers.
Compares against vib3 engine hidden state dumps to find where computation diverges.

Uses safetensors directly to load only the weights we need (no full model loading).
"""

import torch
import numpy as np
import struct
from pathlib import Path
from safetensors import safe_open
from tokenizers import Tokenizer

MODEL_DIR = Path("/code/models/kimi2.5")
DUMP_DIR = Path("/model/dump")  # This is inside the Docker volume
EPS = 1e-5


_INDEX_CACHE = None


def load_tensor(name, dtype=torch.bfloat16):
    """Load a tensor by name from the safetensors shards."""
    global _INDEX_CACHE
    if _INDEX_CACHE is None:
        index_path = MODEL_DIR / "model.safetensors.index.json"
        import json

        with open(index_path) as f:
            _INDEX_CACHE = json.load(f)
    # Try exact name first, then with language_model.model. prefix
    full_name = name
    if full_name not in _INDEX_CACHE["weight_map"]:
        full_name = f"language_model.model.{name}"
    if full_name not in _INDEX_CACHE["weight_map"]:
        full_name = f"language_model.{name}"
    shard_file = _INDEX_CACHE["weight_map"][full_name]
    shard_path = MODEL_DIR / shard_file
    with safe_open(str(shard_path), framework="pt") as f:
        return f.get_tensor(full_name)


def rms_norm(x, weight, eps=EPS):
    """RMSNorm: x * weight / sqrt(mean(x^2) + eps)"""
    rms = torch.sqrt(torch.mean(x.float() ** 2) + eps)
    return (x.float() / rms * weight.float()).to(x.dtype)


def main():
    print("=== Kimi K2.5 Reference Hidden State Check ===")

    # 1. Tokenize
    tok = Tokenizer.from_file(str(MODEL_DIR / "tokenizer.json"))
    template = (
        "<|im_system|>system<|im_middle|>You are Kimi, an AI assistant created by Moonshot AI.<|im_end|>"
        "<|im_user|>user<|im_middle|>What is 2+2?<|im_end|>"
        "<|im_assistant|>assistant<|im_middle|>"
    )
    encoded = tok.encode(template)
    token_ids = encoded.ids
    print(f"Token count: {len(token_ids)}")
    print(f"Tokens: {token_ids}")

    # We'll focus on the last token (index 32) since that's what the engine logs
    last_tok = token_ids[-1]  # 163601
    print(f"Last token: {last_tok} = {repr(tok.decode([last_tok]))}")

    # 2. Load embedding table
    print("\nLoading embeddings...")
    embed_weight = load_tensor("embed_tokens.weight")
    print(f"Embed shape: {embed_weight.shape}, dtype: {embed_weight.dtype}")

    # Get embedding for last token
    hidden = embed_weight[last_tok].float()  # [7168]
    print(f"Embedding L2: {torch.norm(hidden).item():.4f}")
    print(f"Embedding first8: {hidden[:8].tolist()}")

    # For full comparison, we need ALL token positions' hidden states
    # But the engine processes token-by-token, using attention with KV cache.
    # For the first token (pos=0), attention with seq_len=1 is just:
    #   q * k^T * v (all from same position), which is just a linear transform.

    # Let's trace all tokens through layer 0 (dense layer) to build up KV cache,
    # then check the hidden state of the last token after all layers.

    # Actually, let's simplify: compare just the embedding + layer 0 output.
    # The engine dumps hidden_f32_L0_pos32.bin which should match.

    # 3. Load Layer 0 attention weights
    print("\nLoading Layer 0 weights...")

    # Attention norm (input_layernorm) - segment 6
    attn_norm = load_tensor("layers.0.input_layernorm.weight")
    print(f"Attn norm shape: {attn_norm.shape}")

    # MLA projections for layer 0
    # q_a_proj, q_b_proj, kv_a_proj_with_mqa, kv_b_proj, o_proj
    q_a_proj = load_tensor("layers.0.self_attn.q_a_proj.weight")
    q_b_proj = load_tensor("layers.0.self_attn.q_b_proj.weight")
    kv_a_proj = load_tensor("layers.0.self_attn.kv_a_proj_with_mqa.weight")
    o_proj = load_tensor("layers.0.self_attn.o_proj.weight")

    # q/kv layernorms
    q_a_layernorm = load_tensor("layers.0.self_attn.q_a_layernorm.weight")
    kv_a_layernorm = load_tensor("layers.0.self_attn.kv_a_layernorm.weight")

    print(f"q_a_proj: {q_a_proj.shape}")  # [1536, 7168]
    print(
        f"q_b_proj: {q_b_proj.shape}"
    )  # [num_heads*(nope+rope), 1536] = [128*192, 1536] = [24576, 1536]
    print(f"kv_a_proj: {kv_a_proj.shape}")  # [576, 7168] (512 + 64)
    print(f"o_proj: {o_proj.shape}")  # [7168, 128*128] = [7168, 16384]
    print(f"q_a_layernorm: {q_a_layernorm.shape}")
    print(f"kv_a_layernorm: {kv_a_layernorm.shape}")

    # Process all tokens through layer 0 to build up KV cache
    num_tokens = len(token_ids)
    hidden_dim = 7168
    num_heads = 64
    q_lora_rank = 1536
    kv_lora_rank = 512
    qk_rope_dim = 64
    qk_nope_dim = 128
    v_head_dim = 128
    q_head_dim = qk_nope_dim + qk_rope_dim  # 192
    kv_a_dim = kv_lora_rank + qk_rope_dim  # 576

    # Try to also load kv_b_proj for full attention
    kv_b_proj = load_tensor("layers.0.self_attn.kv_b_proj.weight")
    print(
        f"kv_b_proj: {kv_b_proj.shape}"
    )  # [(nope+v)*n_heads, kv_lora] = [(128+128)*128, 512] but might differ

    # Process all tokens sequentially (matching vib3 engine's token-by-token prefill)
    all_hidden = []  # Embedding for each token
    for tid in token_ids:
        all_hidden.append(embed_weight[tid].float())  # [7168]

    # Now process through layer 0 for the last token
    # But we need KV cache from all previous positions...
    # Let's do the full layer 0 processing for ALL tokens

    print(f"\n=== Processing {num_tokens} tokens through Layer 0 ===")

    # Build KV cache through all positions
    kv_latent_cache = []  # [seq_len, kv_lora_rank] - normalized latent
    kv_rope_cache = []  # [seq_len, qk_rope_dim] - RoPE'd key rope

    # RoPE frequencies (Yarn RoPE with Kimi K2.5 params)
    def compute_yarn_freqs(
        dim,
        max_pos=131072,
        base=10000.0,
        scaling_factor=40.0,
        beta_fast=32.0,
        beta_slow=1.0,
        mscale=0.707,
    ):
        """Compute YaRN RoPE frequencies."""
        # Base frequencies
        freqs = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))

        # YaRN interpolation (simplified)
        # Original context length assumed to be 4096, extended to max_pos
        low_dim = max(
            0, int(dim * np.log(1.0 / (2 * np.pi * beta_slow)) / (2 * np.log(base)))
        )
        high_dim = min(
            dim - 2,
            int(dim * np.log(1.0 / (2 * np.pi * beta_fast)) / (2 * np.log(base))),
        )

        for i in range(len(freqs)):
            d = i * 2
            if d < low_dim:
                freqs[i] /= scaling_factor
            elif d < high_dim:
                t = (d - low_dim) / max(1, high_dim - low_dim)
                freqs[i] = freqs[i] * (1 - t) + freqs[i] / scaling_factor * t

        return freqs

    def apply_rope(x, pos, freqs):
        """Apply RoPE to a vector x at position pos."""
        # x: [dim], freqs: [dim//2]
        dim = x.shape[0]
        angles = pos * freqs  # [dim//2]
        cos_a = torch.cos(angles)
        sin_a = torch.sin(angles)
        # Rotate pairs
        x_even = x[0::2]  # [dim//2]
        x_odd = x[1::2]  # [dim//2]
        out = torch.zeros_like(x)
        out[0::2] = x_even * cos_a - x_odd * sin_a
        out[1::2] = x_even * sin_a + x_odd * cos_a
        return out

    rope_freqs = compute_yarn_freqs(qk_rope_dim)

    # Process each token position through Layer 0
    hidden_states_after_layer0 = []

    for pos in range(num_tokens):
        h = all_hidden[pos].clone()  # [7168] in float32

        # Apply attention norm (RMSNorm with weight)
        h_normed = rms_norm(h.bfloat16(), attn_norm, eps=EPS).float()

        # Q path: h → q_a_proj → q_a_layernorm → q_b_proj → q_full
        q_compressed = h_normed.bfloat16() @ q_a_proj.t()  # [1536]
        q_normed = rms_norm(q_compressed, q_a_layernorm, eps=1e-6)
        q_full = q_normed @ q_b_proj.t()  # [num_heads * q_head_dim]
        q_full = q_full.float()

        # KV path: h → kv_a_proj → split (latent + rope)
        kv_a_out = (h_normed.bfloat16() @ kv_a_proj.t()).float()  # [kv_a_dim]
        kv_latent_raw = kv_a_out[:kv_lora_rank]  # [512]
        kv_rope_raw = kv_a_out[kv_lora_rank:]  # [64]

        # Normalize KV latent
        kv_latent_normed = rms_norm(
            kv_latent_raw.bfloat16(), kv_a_layernorm, eps=1e-6
        ).float()

        # Apply RoPE to KV rope
        kv_rope_roped = apply_rope(kv_rope_raw, pos, rope_freqs)

        # Store in cache
        kv_latent_cache.append(kv_latent_normed)
        kv_rope_cache.append(kv_rope_roped)

        # Attention computation
        # Q: reshape to [num_heads, q_head_dim]
        q_heads = q_full.view(num_heads, q_head_dim)  # [128, 192]
        q_nope = q_heads[:, :qk_nope_dim]  # [128, 128]
        q_rope_raw = q_heads[:, qk_nope_dim:]  # [128, 64]

        # Apply RoPE to Q rope
        q_rope = torch.zeros_like(q_rope_raw)
        for h_idx in range(num_heads):
            q_rope[h_idx] = apply_rope(q_rope_raw[h_idx], pos, rope_freqs)

        # Q absorption: q_nope → q_absorbed = q_nope × kv_b_proj_k
        # kv_b_proj: [(nope+v)*num_heads, kv_lora_rank]
        # But actually kv_b_proj has k and v parts interleaved per head
        # Shape should be [num_heads * (qk_nope_dim + v_head_dim), kv_lora_rank]
        # = [128 * 256, 512] = [32768, 512]
        kv_b = kv_b_proj.float()
        kv_b_per_head = kv_b.view(num_heads, qk_nope_dim + v_head_dim, kv_lora_rank)
        # k part: [128, 128, 512], v part: [128, 128, 512]
        kv_b_k = kv_b_per_head[:, :qk_nope_dim, :]  # [128, 128, 512]
        kv_b_v = kv_b_per_head[:, qk_nope_dim:, :]  # [128, 128, 512]

        # Absorbed Q: q_nope [128, 128] × kv_b_k [128, 128, 512] → [128, 512]
        # q_absorbed[h] = q_nope[h] @ kv_b_k[h]  → [512]
        q_absorbed = torch.einsum("hd,hdk->hk", q_nope, kv_b_k)  # [128, 512]

        # Compute attention scores
        seq_len = pos + 1
        scale = 1.0 / (q_head_dim**0.5)  # 1/sqrt(192)

        # Stack KV cache
        kv_lat = torch.stack(kv_latent_cache[:seq_len])  # [seq_len, 512]
        kv_rp = torch.stack(kv_rope_cache[:seq_len])  # [seq_len, 64]

        # Nope scores: q_absorbed[h] dot kv_latent[t] for each head h, position t
        # q_absorbed: [128, 512], kv_lat: [seq_len, 512]
        nope_scores = torch.einsum("hk,tk->ht", q_absorbed, kv_lat)  # [128, seq_len]

        # Rope scores: q_rope[h] dot kv_rope[t]
        rope_scores = torch.einsum("hd,td->ht", q_rope, kv_rp)  # [128, seq_len]

        # Total scores
        scores = (nope_scores + rope_scores) * scale  # [128, seq_len]

        # Causal mask: only attend to positions <= pos
        # (all positions in cache are <= pos, so no masking needed)
        attn_weights = torch.softmax(scores, dim=-1)  # [128, seq_len]

        # Weighted sum of latent values
        # v_latent = attn_weights × kv_latent
        v_latent = torch.einsum("ht,tk->hk", attn_weights, kv_lat)  # [128, 512]

        # V reconstruction: v_latent × kv_b_v
        # v_out[h] = v_latent[h] @ kv_b_v[h].T → [v_head_dim]
        v_out = torch.einsum("hk,hdk->hd", v_latent, kv_b_v)  # [128, 128]

        # Flatten and project through o_proj
        attn_output = v_out.reshape(-1)  # [16384]
        o_output = (attn_output.bfloat16() @ o_proj.t()).float()  # [7168]

        # Residual add
        h_after_attn = all_hidden[pos] + o_output

        if pos == num_tokens - 1:
            print(f"\nLayer 0, pos={pos} (last token):")
            print(f"  Embedding L2: {torch.norm(all_hidden[pos]).item():.4f}")
            print(f"  h_normed L2: {torch.norm(h_normed).item():.4f}")
            print(f"  q_compressed L2: {torch.norm(q_compressed.float()).item():.4f}")
            print(f"  kv_latent_normed L2: {torch.norm(kv_latent_normed).item():.4f}")
            print(f"  kv_rope (RoPE'd) L2: {torch.norm(kv_rope_roped).item():.4f}")
            print(f"  q_absorbed L2: {torch.norm(q_absorbed).item():.4f}")
            print(f"  q_rope L2: {torch.norm(q_rope).item():.4f}")
            print(f"  scores (last pos): mean={scores[:, -1].mean().item():.4f}")
            print(
                f"  attn_weights sum (should=1): {attn_weights.sum(dim=-1).mean().item():.4f}"
            )
            print(f"  v_latent L2: {torch.norm(v_latent).item():.4f}")
            print(f"  v_out L2: {torch.norm(v_out).item():.4f}")
            print(f"  o_output L2: {torch.norm(o_output).item():.4f}")
            print(f"  h_after_attn L2: {torch.norm(h_after_attn).item():.4f}")
            print(f"  h_after_attn first8: {h_after_attn[:8].tolist()}")

        # Store for next layer
        hidden_states_after_layer0.append(h_after_attn)

    # Now process the MLP (dense FFN) for layer 0
    print("\n=== Layer 0 Dense FFN ===")
    ffn_norm = load_tensor("layers.0.post_attention_layernorm.weight")

    # Dense FFN: up_proj, gate_proj, down_proj
    # For dense layer, these are shared (not MoE)
    up_proj = load_tensor("layers.0.mlp.up_proj.weight")
    gate_proj = load_tensor("layers.0.mlp.gate_proj.weight")
    down_proj = load_tensor("layers.0.mlp.down_proj.weight")

    print(f"FFN: up={up_proj.shape}, gate={gate_proj.shape}, down={down_proj.shape}")

    # Process last token through FFN
    last_h = hidden_states_after_layer0[-1]
    h_ffn_normed = rms_norm(last_h.bfloat16(), ffn_norm, eps=EPS).float()

    # SwiGLU: output = down(SiLU(gate(x)) * up(x))
    gate_out = (h_ffn_normed.bfloat16() @ gate_proj.t()).float()
    up_out = (h_ffn_normed.bfloat16() @ up_proj.t()).float()
    intermediate = torch.nn.functional.silu(gate_out) * up_out
    ffn_output = (intermediate.bfloat16() @ down_proj.t()).float()

    h_after_layer0 = last_h + ffn_output

    print(f"FFN normed L2: {torch.norm(h_ffn_normed).item():.4f}")
    print(f"gate_out L2: {torch.norm(gate_out).item():.4f}")
    print(f"up_out L2: {torch.norm(up_out).item():.4f}")
    print(f"intermediate L2: {torch.norm(intermediate).item():.4f}")
    print(f"ffn_output L2: {torch.norm(ffn_output).item():.4f}")
    print(f"h_after_layer0 L2: {torch.norm(h_after_layer0).item():.4f}")
    print(f"h_after_layer0 first8: {h_after_layer0[:8].tolist()}")
    print(f"h_after_layer0 min: {h_after_layer0.min().item():.6f}")
    print(f"h_after_layer0 max: {h_after_layer0.max().item():.6f}")

    print(f"\n=== ENGINE COMPARISON (from logs) ===")
    print(f"Engine PREFILL DIAG tok=32 layer=0: L2=0.73")
    print(
        f"Reference h_after_layer0:           L2={torch.norm(h_after_layer0).item():.4f}"
    )

    # Now do layer 1 (first MoE layer) - attention only first
    print(f"\n=== Layer 1 Attention (MoE layer) ===")

    # Need to process ALL tokens through layer 0 FFN and build layer 1 KV cache
    # Currently we only did the last token through FFN. Let's redo for all tokens.
    layer0_outputs = []
    for pos_idx in range(num_tokens):
        h = hidden_states_after_layer0[pos_idx]  # After attn residual add
        h_fn = rms_norm(h.bfloat16(), ffn_norm, eps=EPS).float()
        go = (h_fn.bfloat16() @ gate_proj.t()).float()
        uo = (h_fn.bfloat16() @ up_proj.t()).float()
        inter = torch.nn.functional.silu(go) * uo
        fout = (inter.bfloat16() @ down_proj.t()).float()
        layer0_outputs.append(h + fout)

    # Load layer 1 attention weights
    l1_attn_norm = load_tensor("layers.1.input_layernorm.weight")
    l1_q_a_proj = load_tensor("layers.1.self_attn.q_a_proj.weight")
    l1_q_b_proj = load_tensor("layers.1.self_attn.q_b_proj.weight")
    l1_kv_a_proj = load_tensor("layers.1.self_attn.kv_a_proj_with_mqa.weight")
    l1_o_proj = load_tensor("layers.1.self_attn.o_proj.weight")
    l1_q_a_layernorm = load_tensor("layers.1.self_attn.q_a_layernorm.weight")
    l1_kv_a_layernorm = load_tensor("layers.1.self_attn.kv_a_layernorm.weight")
    l1_kv_b_proj = load_tensor("layers.1.self_attn.kv_b_proj.weight")

    # Process all tokens through layer 1 attention
    l1_kv_latent_cache = []
    l1_kv_rope_cache = []
    l1_after_attn = []

    for pos_idx in range(num_tokens):
        h = layer0_outputs[pos_idx]
        h_normed_l1 = rms_norm(h.bfloat16(), l1_attn_norm, eps=EPS).float()

        # Q path
        q_comp = (h_normed_l1.bfloat16() @ l1_q_a_proj.t()).float()
        q_normed_l1 = rms_norm(q_comp.bfloat16(), l1_q_a_layernorm, eps=1e-6).float()
        q_full_l1 = (q_normed_l1.bfloat16() @ l1_q_b_proj.t()).float()

        # KV path
        kv_a_out_l1 = (h_normed_l1.bfloat16() @ l1_kv_a_proj.t()).float()
        kv_lat_raw = kv_a_out_l1[:kv_lora_rank]
        kv_rp_raw = kv_a_out_l1[kv_lora_rank:]

        kv_lat_normed = rms_norm(
            kv_lat_raw.bfloat16(), l1_kv_a_layernorm, eps=1e-6
        ).float()
        kv_rp_roped = apply_rope(kv_rp_raw, pos_idx, rope_freqs)

        l1_kv_latent_cache.append(kv_lat_normed)
        l1_kv_rope_cache.append(kv_rp_roped)

        # Attention
        q_heads_l1 = q_full_l1.view(num_heads, q_head_dim)
        q_nope_l1 = q_heads_l1[:, :qk_nope_dim]
        q_rope_raw_l1 = q_heads_l1[:, qk_nope_dim:]

        q_rope_l1 = torch.zeros_like(q_rope_raw_l1)
        for h_idx in range(num_heads):
            q_rope_l1[h_idx] = apply_rope(q_rope_raw_l1[h_idx], pos_idx, rope_freqs)

        kv_b_l1 = l1_kv_b_proj.float()
        kv_b_per_head_l1 = kv_b_l1.view(
            num_heads, qk_nope_dim + v_head_dim, kv_lora_rank
        )
        kv_b_k_l1 = kv_b_per_head_l1[:, :qk_nope_dim, :]
        kv_b_v_l1 = kv_b_per_head_l1[:, qk_nope_dim:, :]

        q_abs_l1 = torch.einsum("hd,hdk->hk", q_nope_l1, kv_b_k_l1)

        seq_len_l1 = pos_idx + 1
        scale_l1 = 1.0 / (q_head_dim**0.5)

        kv_lat_stack = torch.stack(l1_kv_latent_cache[:seq_len_l1])
        kv_rp_stack = torch.stack(l1_kv_rope_cache[:seq_len_l1])

        nope_scores_l1 = torch.einsum("hk,tk->ht", q_abs_l1, kv_lat_stack)
        rope_scores_l1 = torch.einsum("hd,td->ht", q_rope_l1, kv_rp_stack)
        scores_l1 = (nope_scores_l1 + rope_scores_l1) * scale_l1
        attn_w_l1 = torch.softmax(scores_l1, dim=-1)

        v_lat_l1 = torch.einsum("ht,tk->hk", attn_w_l1, kv_lat_stack)
        v_out_l1 = torch.einsum("hk,hdk->hd", v_lat_l1, kv_b_v_l1)

        attn_out_l1 = v_out_l1.reshape(-1)
        o_out_l1 = (attn_out_l1.bfloat16() @ l1_o_proj.t()).float()

        h_after_l1_attn = h + o_out_l1
        l1_after_attn.append(h_after_l1_attn)

        if pos_idx == num_tokens - 1:
            print(f"Layer 1 Attention, pos={pos_idx}:")
            print(f"  h_normed L2: {torch.norm(h_normed_l1).item():.4f}")
            print(f"  q_compressed L2: {torch.norm(q_comp).item():.4f}")
            print(f"  kv_latent_normed L2: {torch.norm(kv_lat_normed).item():.4f}")
            print(f"  q_absorbed L2: {torch.norm(q_abs_l1).item():.4f}")
            print(f"  o_output L2: {torch.norm(o_out_l1).item():.4f}")
            print(f"  h_after_l1_attn L2: {torch.norm(h_after_l1_attn).item():.4f}")

    # MoE for layer 1 would need quantized expert weights - skip for now
    # But we can check the pre-MoE norm values
    l1_moe_norm = load_tensor("layers.1.post_attention_layernorm.weight")
    last_h_l1 = l1_after_attn[-1]
    h_moe_normed = rms_norm(last_h_l1.bfloat16(), l1_moe_norm, eps=EPS).float()
    print(f"\nLayer 1 pre-MoE norm:")
    print(f"  h_before_norm L2: {torch.norm(last_h_l1).item():.4f}")
    print(f"  h_after_norm L2: {torch.norm(h_moe_normed).item():.4f}")
    print(f"  h_after_norm first8: {h_moe_normed[:8].tolist()}")

    print(f"\n=== ENGINE vs REFERENCE ===")
    print(f"Engine L0 L2: 0.73   vs Reference: {torch.norm(h_after_layer0).item():.4f}")
    print(f"Engine L1 (with MoE): 0.77")
    print(
        f"Reference L1 (after attn only, no MoE): {torch.norm(l1_after_attn[-1]).item():.4f}"
    )
    print(f"Engine MOE NORM DIAG L1 pos=1: pre_norm L2=4.79, post_norm L2=2.43")
    print(f"Reference pre-MoE norm input L2: {torch.norm(last_h_l1).item():.4f}")
    print(f"Reference post-MoE norm L2: {torch.norm(h_moe_normed).item():.4f}")


if __name__ == "__main__":
    main()
