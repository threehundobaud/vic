//! Attention implementations: GQA and Multi-head Latent Attention (MLA).
//!
//! This module provides two attention paths:
//!
//! 1. **GQA** (Grouped-Query Attention): Standard path for Mixtral-like models.
//!    Q, K, V are projected from hidden_state via fused QKV weight matrix.
//!    KV cache stores per-head K/V vectors (head_dim per head per position).
//!
//! 2. **MLA** (Multi-head Latent Attention): DeepSeek-V3/Kimi K2.5 path.
//!    Uses low-rank projections to compress K/V into a shared latent space.
//!    KV cache stores the compressed latent vector (kv_lora_rank per position,
//!    shared across all heads), drastically reducing cache memory.
//!
//! ## MLA Architecture (DeepSeek-V2/V3, Kimi K2.5)
//!
//! ```text
//! hidden_state [hidden_dim=7168]
//!     │
//!     ├──→ q_a_proj [q_lora_rank=1536, hidden_dim] ─→ RMSNorm ─→
//!     │      q_b_proj [num_heads*(nope+rope)=64*(128+64)=12288, q_lora_rank=1536]
//!     │        ├── q_nope [num_heads, qk_nope_head_dim=128]  (non-positional)
//!     │        └── q_rope [num_heads, qk_rope_head_dim=64]   (apply RoPE)
//!     │
//!     └──→ kv_a_proj_with_mqa [kv_lora_rank+rope=512+64=576, hidden_dim]
//!            ├── kv_latent [kv_lora_rank=512] ─→ RMSNorm ─→ KV cache ─→
//!            │      kv_b_proj [num_heads*(nope+v)=64*(128+128)=16384, kv_lora_rank=512]
//!            │        ├── k_nope [num_heads, qk_nope_head_dim=128]
//!            │        └── v [num_heads, v_head_dim=128]
//!            └── k_rope_shared [qk_rope_head_dim=64] (apply RoPE, shared across heads)
//! ```
//!
//! The key insight: instead of caching per-head K/V (which would be massive
//! for 64 heads), MLA caches only the compressed latent (512 dims) plus the
//! rope component (64 dims) = 576 dims per position. At decode time, k_nope
//! and v are reconstructed from the latent via kv_b_proj.

use crate::compute::kernels;
use crate::core::config::{MlaConfig, ModelConfig};
use crate::runtime::tiered_kv::TieredKvCache;
use half::f16;

// ═══════════════════════════════════════════════════════════════════════════
// GQA KV Cache (existing, for non-MLA models)
// ═══════════════════════════════════════════════════════════════════════════

/// KV cache for one layer (GQA path).
///
/// Stores key and value vectors for all past positions,
/// organized by KV head group.
pub struct KvCache {
    /// K cache: [num_kv_heads][positions * head_dim] in f32
    pub k: Vec<Vec<f32>>,
    /// V cache: [num_kv_heads][positions * head_dim] in f32
    pub v: Vec<Vec<f32>>,
    /// Current sequence length (number of positions stored)
    pub seq_len: usize,
    /// Head dimension
    pub head_dim: usize,
    /// Number of KV heads
    pub num_kv_heads: usize,
    /// Maximum sequence length
    pub max_seq_len: usize,
}

impl KvCache {
    pub fn new(num_kv_heads: usize, head_dim: usize, max_seq_len: usize) -> Self {
        let k = (0..num_kv_heads)
            .map(|_| Vec::with_capacity(max_seq_len * head_dim))
            .collect();
        let v = (0..num_kv_heads)
            .map(|_| Vec::with_capacity(max_seq_len * head_dim))
            .collect();
        Self {
            k,
            v,
            seq_len: 0,
            head_dim,
            num_kv_heads,
            max_seq_len,
        }
    }

    /// Append one position's K/V vectors to the cache.
    pub fn append(&mut self, k_heads: &[Vec<f32>], v_heads: &[Vec<f32>]) {
        assert_eq!(k_heads.len(), self.num_kv_heads);
        assert_eq!(v_heads.len(), self.num_kv_heads);

        for h in 0..self.num_kv_heads {
            self.k[h].extend_from_slice(&k_heads[h]);
            self.v[h].extend_from_slice(&v_heads[h]);
        }
        self.seq_len += 1;
    }

    /// Clear the cache (for new sequences).
    pub fn clear(&mut self) {
        for h in 0..self.num_kv_heads {
            self.k[h].clear();
            self.v[h].clear();
        }
        self.seq_len = 0;
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// MLA KV Cache
// ═══════════════════════════════════════════════════════════════════════════

/// KV cache for one layer using MLA (Multi-head Latent Attention).
///
/// Instead of storing per-head K/V (which would be 64 heads * head_dim * seq_len),
/// MLA stores only:
/// - `kv_latent`: [positions * kv_lora_rank] — compressed KV representation
/// - `k_rope`:    [positions * qk_rope_head_dim] — RoPE component (shared across heads)
///
/// At decode time, the full per-head k_nope and v are reconstructed from
/// kv_latent via kv_b_proj multiplication. This is the "absorbed attention"
/// optimization from DeepSeek-V2.
///
/// Memory comparison for Kimi K2.5 at 4K context:
/// - Standard GQA: 64 heads * 128 dim * 2 (K+V) * 4K * 4 bytes = 256 MB/layer
/// - MLA: (512 + 64) * 4K * 4 bytes = 9.2 MB/layer  (~28x smaller)
pub struct MlaKvCache {
    /// Compressed KV latent cache: [positions * kv_lora_rank] in f32.
    /// This is the output of kv_a_proj (first kv_lora_rank dims), after RMSNorm.
    pub kv_latent: Vec<f32>,

    /// RoPE key component cache: [positions * qk_rope_head_dim] in f32.
    /// This is the last qk_rope_head_dim dims of kv_a_proj output, after RoPE.
    pub k_rope: Vec<f32>,

    /// Current sequence length
    pub seq_len: usize,

    /// KV latent dimension (512 for Kimi K2.5)
    pub kv_lora_rank: usize,

    /// RoPE dimension (64 for Kimi K2.5)
    pub qk_rope_head_dim: usize,

    /// Maximum sequence length
    pub max_seq_len: usize,
}

impl MlaKvCache {
    pub fn new(kv_lora_rank: usize, qk_rope_head_dim: usize, max_seq_len: usize) -> Self {
        Self {
            kv_latent: Vec::with_capacity(max_seq_len * kv_lora_rank),
            k_rope: Vec::with_capacity(max_seq_len * qk_rope_head_dim),
            seq_len: 0,
            kv_lora_rank,
            qk_rope_head_dim,
            max_seq_len,
        }
    }

    /// Append one position's compressed KV to the cache.
    ///
    /// - `latent`: [kv_lora_rank] — compressed KV after RMSNorm
    /// - `rope`: [qk_rope_head_dim] — RoPE component after RoPE application
    pub fn append(&mut self, latent: &[f32], rope: &[f32]) {
        debug_assert_eq!(latent.len(), self.kv_lora_rank);
        debug_assert_eq!(rope.len(), self.qk_rope_head_dim);
        self.kv_latent.extend_from_slice(latent);
        self.k_rope.extend_from_slice(rope);
        self.seq_len += 1;
    }

    /// Clear the cache (for new sequences).
    pub fn clear(&mut self) {
        self.kv_latent.clear();
        self.k_rope.clear();
        self.seq_len = 0;
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Full KV Cache Sets
// ═══════════════════════════════════════════════════════════════════════════

/// Full set of KV caches (one per layer) — GQA path.
pub struct KvCacheSet {
    pub layers: Vec<KvCache>,
}

impl KvCacheSet {
    pub fn new(config: &ModelConfig) -> Self {
        let num_layers = config.num_layers as usize;
        let num_kv_heads = config.num_kv_heads as usize;
        let head_dim = if config.num_heads > 0 {
            config.hidden_dim as usize / config.num_heads as usize
        } else {
            128
        };
        let max_seq_len = config.max_seq_len as usize;

        let layers = (0..num_layers)
            .map(|_| KvCache::new(num_kv_heads, head_dim, max_seq_len))
            .collect();

        Self { layers }
    }

    pub fn clear(&mut self) {
        for layer in &mut self.layers {
            layer.clear();
        }
    }
}

/// Full set of MLA KV caches (one per layer).
pub struct MlaKvCacheSet {
    pub layers: Vec<MlaKvCache>,
}

impl MlaKvCacheSet {
    pub fn new(config: &ModelConfig, mla: &MlaConfig) -> Self {
        let num_layers = config.num_layers as usize;
        let max_seq_len = config.max_seq_len as usize;

        let layers = (0..num_layers)
            .map(|_| {
                MlaKvCache::new(
                    mla.kv_lora_rank as usize,
                    mla.qk_rope_head_dim as usize,
                    max_seq_len,
                )
            })
            .collect();

        Self { layers }
    }

    pub fn clear(&mut self) {
        for layer in &mut self.layers {
            layer.clear();
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// YaRN RoPE
// ═══════════════════════════════════════════════════════════════════════════

/// YaRN (Yet another RoPE scaling) configuration.
///
/// YaRN applies dimension-dependent scaling to extend context length beyond
/// the training window while preserving short-context quality. It divides
/// rope dimensions into three zones:
/// - Low frequency (below beta_fast): full interpolation (scale by factor)
/// - High frequency (above beta_slow): no interpolation (keep original freqs)
/// - Middle: smooth interpolation blend
///
/// Reference: https://arxiv.org/abs/2309.00071
pub struct YarnRopeConfig {
    pub base_theta: f32,
    pub scaling_factor: f32,
    pub beta_fast: f32,
    pub beta_slow: f32,
    pub original_max_pos: usize,
    pub rope_dim: usize,
}

impl YarnRopeConfig {
    /// Create from Kimi K2.5 config.
    pub fn from_kimi_k25(rope_dim: usize) -> Self {
        use crate::core::types::kimi_k25::*;
        Self {
            base_theta: ROPE_THETA as f32,
            scaling_factor: ROPE_SCALING_FACTOR as f32,
            beta_fast: ROPE_BETA_FAST as f32,
            beta_slow: ROPE_BETA_SLOW as f32,
            original_max_pos: ROPE_ORIGINAL_MAX_POS as usize,
            rope_dim,
        }
    }

    /// Compute the correction dimension index from a number of rotations.
    ///
    /// This matches `yarn_find_correction_dim()` from the DeepSeek-V3 reference:
    /// finds the dimension index whose wavelength corresponds to the given
    /// number of rotations within `original_max_position_embeddings`.
    fn correction_dim(&self, num_rotations: f32) -> f32 {
        (self.rope_dim as f32
            * (self.original_max_pos as f32 / (num_rotations * 2.0 * std::f32::consts::PI)).ln())
            / (2.0 * (self.base_theta as f64).ln() as f32)
    }

    /// Compute the dimension index range for the YaRN linear ramp.
    ///
    /// Returns `(low, high)` clamped to valid dimension pair indices.
    /// Matches `yarn_find_correction_range()` from the reference implementation.
    fn correction_range(&self) -> (usize, usize) {
        let half_dim = self.rope_dim / 2;
        let low = self.correction_dim(self.beta_fast).floor() as i32;
        let high = self.correction_dim(self.beta_slow).ceil() as i32;
        (low.max(0) as usize, (high as usize).min(half_dim - 1))
    }

    /// Compute the frequency for a given dimension pair index.
    ///
    /// Uses the index-based linear ramp mask from the DeepSeek-V3 reference
    /// implementation (`yarn_linear_ramp_mask` + frequency blending).
    ///
    /// - For dim indices below `low`: use original (extrapolated) frequency
    /// - For dim indices above `high`: use interpolated frequency (÷ factor)
    /// - For dim indices in [low, high]: linear blend in index space
    pub fn freq(&self, dim_idx: usize) -> f32 {
        let base_freq = 1.0
            / self
                .base_theta
                .powf(2.0 * dim_idx as f32 / self.rope_dim as f32);
        let interp_freq = base_freq / self.scaling_factor;

        let (low, high) = self.correction_range();

        // Compute the ramp mask value for this dimension index.
        // mask = 1.0 means use original freq, mask = 0.0 means use interpolated.
        // This matches: inv_freq_mask = 1.0 - yarn_linear_ramp_mask(low, high, dim//2)
        //               inv_freq = freq_inter * (1 - mask) + freq_extra * mask
        let mask = if low == high {
            // Prevent division by zero (matches reference: max += 0.001)
            let ramp = (dim_idx as f32 - low as f32) / (high as f32 + 0.001 - low as f32);
            1.0 - ramp.clamp(0.0, 1.0)
        } else {
            let ramp = (dim_idx as f32 - low as f32) / (high as f32 - low as f32);
            1.0 - ramp.clamp(0.0, 1.0)
        };

        // Blend: mask=1 → original freq, mask=0 → interpolated freq
        interp_freq * (1.0 - mask) + base_freq * mask
    }
}

/// Apply YaRN RoPE to a vector in-place.
///
/// This applies rotary position embeddings with YaRN-scaled frequencies.
/// Used for the rope dimensions of Q and K in MLA.
pub fn apply_yarn_rope(x: &mut [f32], position: usize, config: &YarnRopeConfig) {
    let half_dim = x.len() / 2;
    for i in 0..half_dim {
        let freq = config.freq(i);
        let theta = position as f32 * freq;
        let cos_t = theta.cos();
        let sin_t = theta.sin();

        let x0 = x[2 * i];
        let x1 = x[2 * i + 1];
        x[2 * i] = x0 * cos_t - x1 * sin_t;
        x[2 * i + 1] = x0 * sin_t + x1 * cos_t;
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// MLA Attention Forward Pass
// ═══════════════════════════════════════════════════════════════════════════

/// MLA weight matrices needed for one layer's attention.
///
/// These are loaded from shared pages with segments 20-23, 5, 6.
pub struct MlaWeights<'a> {
    /// q_a_proj: [q_lora_rank, hidden_dim] — segment 20
    pub q_a_proj: Option<&'a [f16]>,
    /// q_b_proj: [num_heads*(nope+rope), q_lora_rank] — segment 21
    pub q_b_proj: Option<&'a [f16]>,
    /// kv_a_proj_with_mqa: [kv_lora_rank + rope_dim, hidden_dim] — segment 22
    pub kv_a_proj: Option<&'a [f16]>,
    /// kv_b_proj: [num_heads*(nope+v), kv_lora_rank] — segment 23
    pub kv_b_proj: Option<&'a [f16]>,
    /// o_proj: [hidden_dim, num_heads * v_head_dim] — segment 5
    pub o_proj: Option<&'a [f16]>,
    /// q_norm: [q_lora_rank] — RMSNorm for Q latent (if available)
    pub q_norm: Option<&'a [f16]>,
    /// kv_norm: [kv_lora_rank] — RMSNorm for KV latent (if available)
    pub kv_norm: Option<&'a [f16]>,
}

/// Run MLA attention for one layer on CPU.
///
/// Steps:
/// 1. Q path: hidden → q_a_proj → RMSNorm → q_b_proj → split(q_nope, q_rope)
///    Apply YaRN RoPE to q_rope
/// 2. KV path: hidden → kv_a_proj → split(kv_latent, k_rope_shared)
///    Apply RMSNorm to kv_latent, apply YaRN RoPE to k_rope_shared
///    Cache kv_latent and k_rope_shared
/// 3. Reconstruct K/V from cache: kv_latent → kv_b_proj → split(k_nope, v)
///    Compose k = [k_nope, k_rope_shared]
/// 4. Attention: Q·K^T / sqrt(d) → softmax → V for each head
/// 5. O projection
/// `rms_norm_eps` is the model's epsilon (e.g. 1e-5 for Kimi K2.5).
pub fn mla_attention_layer(
    hidden_state: &[f16],
    weights: &MlaWeights,
    kv_cache: &mut MlaKvCache,
    position: usize,
    config: &ModelConfig,
    mla: &MlaConfig,
) -> Vec<f16> {
    let hidden_dim = config.hidden_dim as usize;
    let num_heads = config.num_heads as usize;
    let q_lora_rank = mla.q_lora_rank as usize;
    let kv_lora_rank = mla.kv_lora_rank as usize;
    let qk_rope_dim = mla.qk_rope_head_dim as usize;
    let qk_nope_dim = mla.qk_nope_head_dim as usize;
    let v_head_dim = mla.v_head_dim as usize;

    // Total Q head dim = nope + rope
    let q_head_dim = qk_nope_dim + qk_rope_dim;
    // Total K head dim for attention = nope + rope
    let k_head_dim = qk_nope_dim + qk_rope_dim;

    // YaRN RoPE config
    let yarn = YarnRopeConfig::from_kimi_k25(qk_rope_dim);

    // Convert input to f32
    let state_f32: Vec<f32> = hidden_state.iter().map(|v| v.to_f32()).collect();

    // ── Step 1: Q projection ──────────────────────────────────────────

    // q_a_proj: hidden_dim → q_lora_rank (low-rank compression)
    let q_compressed = if let Some(q_a) = weights.q_a_proj {
        gemv_f16(&state_f32, q_a, hidden_dim, q_lora_rank)
    } else {
        // Fallback: truncate hidden state
        state_f32[..q_lora_rank].to_vec()
    };

    // RMSNorm on q_compressed
    let q_normed = if let Some(q_norm_w) = weights.q_norm {
        rms_norm_f32_with_weight(&q_compressed, q_norm_w)
    } else {
        rms_norm_f32(&q_compressed)
    };

    // q_b_proj: q_lora_rank → num_heads * (nope + rope)
    let q_full = if let Some(q_b) = weights.q_b_proj {
        gemv_f16(&q_normed, q_b, q_lora_rank, num_heads * q_head_dim)
    } else {
        // Fallback: repeat q_normed across heads
        let mut fallback = vec![0.0f32; num_heads * q_head_dim];
        for h in 0..num_heads {
            for d in 0..q_head_dim.min(q_lora_rank) {
                fallback[h * q_head_dim + d] = q_normed[d % q_normed.len()];
            }
        }
        fallback
    };

    // Split Q into per-head nope and rope, apply RoPE to rope part
    let mut q_heads: Vec<Vec<f32>> = Vec::with_capacity(num_heads);
    for h in 0..num_heads {
        let head_start = h * q_head_dim;
        let q_nope = &q_full[head_start..head_start + qk_nope_dim];
        let mut q_rope = q_full[head_start + qk_nope_dim..head_start + q_head_dim].to_vec();

        // Apply YaRN RoPE to the rope component
        apply_yarn_rope(&mut q_rope, position, &yarn);

        // Compose: q = [q_nope, q_rope]
        let mut q_head = Vec::with_capacity(k_head_dim);
        q_head.extend_from_slice(q_nope);
        q_head.extend_from_slice(&q_rope);
        q_heads.push(q_head);
    }

    // ── Step 2: KV projection and caching ─────────────────────────────

    // kv_a_proj: hidden_dim → kv_lora_rank + qk_rope_dim
    let kv_a_dim = kv_lora_rank + qk_rope_dim;
    let kv_a_out = if let Some(kv_a) = weights.kv_a_proj {
        gemv_f16(&state_f32, kv_a, hidden_dim, kv_a_dim)
    } else {
        // Fallback
        let mut fallback = vec![0.0f32; kv_a_dim];
        for d in 0..kv_a_dim.min(hidden_dim) {
            fallback[d] = state_f32[d] * 0.5;
        }
        fallback
    };

    // Split: kv_latent [kv_lora_rank] and k_rope_shared [qk_rope_dim]
    let kv_latent_raw = &kv_a_out[..kv_lora_rank];
    let mut k_rope_shared = kv_a_out[kv_lora_rank..].to_vec();

    // RMSNorm on kv_latent
    let kv_latent_normed = if let Some(kv_norm_w) = weights.kv_norm {
        rms_norm_f32_with_weight(kv_latent_raw, kv_norm_w)
    } else {
        rms_norm_f32(kv_latent_raw)
    };

    // Apply YaRN RoPE to the shared K rope component
    apply_yarn_rope(&mut k_rope_shared, position, &yarn);

    // Cache: store compressed latent + rope'd K
    kv_cache.append(&kv_latent_normed, &k_rope_shared);

    // ── Step 3: Reconstruct K/V for all cached positions and attend ───

    // For each cached position, we need to:
    // 1. kv_b_proj: kv_latent → num_heads * (nope + v)
    // 2. Compose K = [k_nope, k_rope_cached]
    // 3. V = v from kv_b_proj output
    //
    // For the "absorbed" optimization (DeepSeek-V2 inference trick):
    // Instead of materializing per-head K/V for all positions, we can
    // reformulate attention to avoid the kv_b_proj multiplication at
    // every position. However, for correctness-first CPU fallback,
    // we do the explicit reconstruction.

    let seq_len = kv_cache.seq_len;

    // Reconstruct K and V for all cached positions
    // k_nope + v are from kv_b_proj; k_rope is cached directly
    let kv_b_out_dim = num_heads * (qk_nope_dim + v_head_dim);

    // Per-head attention
    let mut attn_output = vec![0.0f32; num_heads * v_head_dim];

    for (h, q_head) in q_heads.iter().enumerate().take(num_heads) {
        let scale = mla.softmax_scale;

        // Compute attention scores for all positions
        let mut scores = Vec::with_capacity(seq_len);

        for pos in 0..seq_len {
            // Reconstruct K for this position and head
            let latent_start = pos * kv_lora_rank;
            let latent = &kv_cache.kv_latent[latent_start..latent_start + kv_lora_rank];

            let rope_start = pos * qk_rope_dim;
            let k_rope_pos = &kv_cache.k_rope[rope_start..rope_start + qk_rope_dim];

            // kv_b_proj: latent → [num_heads * (nope + v)]
            // We only need head h's k_nope portion
            let k_nope = if let Some(kv_b) = weights.kv_b_proj {
                // Extract head h's k_nope from kv_b_proj output
                // kv_b_proj layout: for each head: [qk_nope_head_dim | v_head_dim]
                let head_offset = h * (qk_nope_dim + v_head_dim);
                gemv_f16_slice(
                    latent,
                    kv_b,
                    kv_lora_rank,
                    kv_b_out_dim,
                    head_offset,
                    qk_nope_dim,
                )
            } else {
                // Fallback
                latent[..qk_nope_dim.min(kv_lora_rank)].to_vec()
            };

            // Compose K = [k_nope, k_rope]
            // Dot product with Q = [q_nope, q_rope]
            let mut dot = 0.0f32;
            // Nope part
            for d in 0..qk_nope_dim.min(k_nope.len()) {
                dot += q_head[d] * k_nope[d];
            }
            // Rope part
            for d in 0..qk_rope_dim.min(k_rope_pos.len()) {
                dot += q_head[qk_nope_dim + d] * k_rope_pos[d];
            }

            scores.push(dot * scale);
        }

        // Softmax
        if scores.is_empty() {
            continue;
        }

        let max_score = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mut exp_scores: Vec<f32> = scores.iter().map(|s| (s - max_score).exp()).collect();
        let sum: f32 = exp_scores.iter().sum();
        if sum > 0.0 {
            for s in &mut exp_scores {
                *s /= sum;
            }
        }

        // Weighted sum of V across positions
        let head_out_start = h * v_head_dim;
        for (pos, &w) in exp_scores.iter().enumerate().take(seq_len) {
            if w < 1e-8 {
                continue; // Skip negligible weights
            }

            // Reconstruct V for this position and head
            let latent_start = pos * kv_lora_rank;
            let latent = &kv_cache.kv_latent[latent_start..latent_start + kv_lora_rank];

            let v_vec = if let Some(kv_b) = weights.kv_b_proj {
                let head_offset = h * (qk_nope_dim + v_head_dim) + qk_nope_dim;
                gemv_f16_slice(
                    latent,
                    kv_b,
                    kv_lora_rank,
                    kv_b_out_dim,
                    head_offset,
                    v_head_dim,
                )
            } else {
                let len = v_head_dim.min(kv_lora_rank);
                let mut fallback = vec![0.0f32; v_head_dim];
                fallback[..len].copy_from_slice(&latent[..len]);
                fallback
            };

            for d in 0..v_head_dim {
                attn_output[head_out_start + d] += w * v_vec[d];
            }
        }
    }

    // ── Step 5: O projection ──────────────────────────────────────────

    let o_dim = num_heads * v_head_dim;
    let output_f32 = if let Some(o_w) = weights.o_proj {
        gemv_f16(&attn_output, o_w, o_dim, hidden_dim)
    } else {
        // Fallback: truncate/pad to hidden_dim
        let mut out = vec![0.0f32; hidden_dim];
        let copy_len = attn_output.len().min(hidden_dim);
        out[..copy_len].copy_from_slice(&attn_output[..copy_len]);
        out
    };

    // Convert back to FP16
    output_f32.iter().map(|v| f16::from_f32(*v)).collect()
}

// ═══════════════════════════════════════════════════════════════════════════
// GQA Attention (existing, preserved)
// ═══════════════════════════════════════════════════════════════════════════

/// Run self-attention for one layer on CPU (GQA path).
///
/// Steps:
/// 1. Project hidden_state through Q, K, V weight matrices
/// 2. Split into heads
/// 3. Apply RoPE to Q and K
/// 4. Append K, V to cache
/// 5. Compute multi-head attention
/// 6. Project output through O_proj
pub fn self_attention_layer(
    hidden_state: &[f16],       // [hidden_dim]
    qkv_weight: Option<&[f16]>, // [3 * hidden_dim, hidden_dim] or None for fallback
    o_weight: Option<&[f16]>,   // [hidden_dim, hidden_dim] or None for fallback
    kv_cache: &mut KvCache,
    position: usize,
    config: &ModelConfig,
) -> Vec<f16> {
    let hidden_dim = config.hidden_dim as usize;
    let num_heads = config.num_heads as usize;
    let num_kv_heads = config.num_kv_heads as usize;
    let head_dim = hidden_dim / num_heads;
    let rope_base = config.rope_theta;

    // Convert hidden state to f32
    let state_f32: Vec<f32> = hidden_state.iter().map(|v| v.to_f32()).collect();

    // Project Q, K, V
    let (q_all, k_all, v_all) = if let Some(qkv_w) = qkv_weight {
        // Real projection: hidden_state x W_qkv^T
        let qkv_dim = hidden_dim + 2 * (num_kv_heads * head_dim);
        let mut qkv_out = vec![0.0f32; qkv_dim];

        for i in 0..qkv_dim {
            let mut acc = 0.0f32;
            for d in 0..hidden_dim {
                acc += state_f32[d] * qkv_w[i * hidden_dim + d].to_f32();
            }
            qkv_out[i] = acc;
        }

        let q = qkv_out[..hidden_dim].to_vec();
        let k = qkv_out[hidden_dim..hidden_dim + num_kv_heads * head_dim].to_vec();
        let v = qkv_out[hidden_dim + num_kv_heads * head_dim..].to_vec();
        (q, k, v)
    } else {
        // Fallback: use hidden state directly as Q, derive K/V from it
        let q = state_f32.clone();
        let kv_dim = num_kv_heads * head_dim;
        let k: Vec<f32> = (0..kv_dim)
            .map(|i| state_f32[i % hidden_dim] * 0.5)
            .collect();
        let v: Vec<f32> = (0..kv_dim)
            .map(|i| state_f32[(i + head_dim) % hidden_dim] * 0.5)
            .collect();
        (q, k, v)
    };

    // Split into heads and apply RoPE
    let mut q_heads: Vec<Vec<f32>> = Vec::with_capacity(num_heads);
    for h in 0..num_heads {
        let mut head = q_all[h * head_dim..(h + 1) * head_dim].to_vec();
        kernels::apply_rope(&mut head, position, head_dim, rope_base);
        q_heads.push(head);
    }

    let mut k_heads: Vec<Vec<f32>> = Vec::with_capacity(num_kv_heads);
    let mut v_heads: Vec<Vec<f32>> = Vec::with_capacity(num_kv_heads);
    for h in 0..num_kv_heads {
        let mut k_head = k_all[h * head_dim..(h + 1) * head_dim].to_vec();
        kernels::apply_rope(&mut k_head, position, head_dim, rope_base);
        k_heads.push(k_head);
        v_heads.push(v_all[h * head_dim..(h + 1) * head_dim].to_vec());
    }

    // Append to KV cache
    kv_cache.append(&k_heads, &v_heads);

    // Run multi-head attention
    let attn_out = kernels::multi_head_attention(
        &q_heads,
        &kv_cache.k,
        &kv_cache.v,
        num_heads,
        num_kv_heads,
        head_dim,
        kv_cache.seq_len,
    );

    // O projection
    let output_f32 = if let Some(o_w) = o_weight {
        let mut out = vec![0.0f32; hidden_dim];
        for i in 0..hidden_dim {
            let mut acc = 0.0f32;
            for d in 0..hidden_dim {
                acc += attn_out[d] * o_w[i * hidden_dim + d].to_f32();
            }
            out[i] = acc;
        }
        out
    } else {
        // Fallback: truncate/pad to hidden_dim
        let mut out = vec![0.0f32; hidden_dim];
        let copy_len = attn_out.len().min(hidden_dim);
        out[..copy_len].copy_from_slice(&attn_out[..copy_len]);
        out
    };

    // Convert back to FP16
    output_f32.iter().map(|v| f16::from_f32(*v)).collect()
}

/// Self-attention using the tiered KV cache (sparse retrieval path).
///
/// This is the "attention-as-query-plan" implementation for GQA models.
pub fn self_attention_tiered(
    hidden_state: &[f16],       // [hidden_dim]
    qkv_weight: Option<&[f16]>, // [3 * hidden_dim, hidden_dim] or None for fallback
    o_weight: Option<&[f16]>,   // [hidden_dim, hidden_dim] or None for fallback
    tiered_kv: &mut TieredKvCache,
    layer: usize,
    position: usize,
    config: &ModelConfig,
) -> Vec<f16> {
    let hidden_dim = config.hidden_dim as usize;
    let num_heads = config.num_heads as usize;
    let num_kv_heads = config.num_kv_heads as usize;
    let head_dim = hidden_dim / num_heads;
    let rope_base = config.rope_theta;

    // Convert hidden state to f32
    let state_f32: Vec<f32> = hidden_state.iter().map(|v| v.to_f32()).collect();

    // Project Q, K, V
    let (q_all, k_all, v_all) = if let Some(qkv_w) = qkv_weight {
        let qkv_dim = hidden_dim + 2 * (num_kv_heads * head_dim);
        let mut qkv_out = vec![0.0f32; qkv_dim];

        for i in 0..qkv_dim {
            let mut acc = 0.0f32;
            for d in 0..hidden_dim {
                acc += state_f32[d] * qkv_w[i * hidden_dim + d].to_f32();
            }
            qkv_out[i] = acc;
        }

        let q = qkv_out[..hidden_dim].to_vec();
        let k = qkv_out[hidden_dim..hidden_dim + num_kv_heads * head_dim].to_vec();
        let v = qkv_out[hidden_dim + num_kv_heads * head_dim..].to_vec();
        (q, k, v)
    } else {
        // Fallback: use hidden state directly as Q, derive K/V from it
        let q = state_f32.clone();
        let kv_dim = num_kv_heads * head_dim;
        let k: Vec<f32> = (0..kv_dim)
            .map(|i| state_f32[i % hidden_dim] * 0.5)
            .collect();
        let v: Vec<f32> = (0..kv_dim)
            .map(|i| state_f32[(i + head_dim) % hidden_dim] * 0.5)
            .collect();
        (q, k, v)
    };

    // Split into heads and apply RoPE
    let mut q_heads: Vec<Vec<f32>> = Vec::with_capacity(num_heads);
    for h in 0..num_heads {
        let mut head = q_all[h * head_dim..(h + 1) * head_dim].to_vec();
        kernels::apply_rope(&mut head, position, head_dim, rope_base);
        q_heads.push(head);
    }

    let mut k_heads: Vec<Vec<f32>> = Vec::with_capacity(num_kv_heads);
    let mut v_heads: Vec<Vec<f32>> = Vec::with_capacity(num_kv_heads);
    for h in 0..num_kv_heads {
        let mut k_head = k_all[h * head_dim..(h + 1) * head_dim].to_vec();
        kernels::apply_rope(&mut k_head, position, head_dim, rope_base);
        k_heads.push(k_head);
        v_heads.push(v_all[h * head_dim..(h + 1) * head_dim].to_vec());
    }

    // Append to tiered KV cache
    tiered_kv.append_layer(layer, &k_heads, &v_heads);

    // Gather attention positions and K/V vectors for each KV head
    let mut k_per_head: Vec<Vec<Vec<f32>>> = Vec::with_capacity(num_kv_heads);
    let mut v_per_head: Vec<Vec<Vec<f32>>> = Vec::with_capacity(num_kv_heads);

    for kv_head in 0..num_kv_heads {
        // Use Q from the first Q head mapped to this KV head for ANN search
        let q_head_idx = kv_head * (num_heads / num_kv_heads.max(1));
        let q_for_search = &q_heads[q_head_idx.min(num_heads - 1)];

        // Gather positions: recent window + landmarks + ANN-retrieved
        let positions = tiered_kv.gather_attention_positions(layer, kv_head, q_for_search);

        // Get K and V vectors for the gathered positions
        let k_vecs = tiered_kv.get_k_vectors(layer, kv_head, &positions);
        let v_vecs = tiered_kv.get_v_vectors(layer, kv_head, &positions);

        k_per_head.push(k_vecs);
        v_per_head.push(v_vecs);
    }

    // Compute sparse multi-head attention
    let attn_out = kernels::multi_head_sparse_attention(
        &q_heads,
        &k_per_head,
        &v_per_head,
        num_heads,
        num_kv_heads,
        head_dim,
    );

    // O projection
    let output_f32 = if let Some(o_w) = o_weight {
        let mut out = vec![0.0f32; hidden_dim];
        for i in 0..hidden_dim {
            let mut acc = 0.0f32;
            for d in 0..hidden_dim {
                acc += attn_out[d] * o_w[i * hidden_dim + d].to_f32();
            }
            out[i] = acc;
        }
        out
    } else {
        let mut out = vec![0.0f32; hidden_dim];
        let copy_len = attn_out.len().min(hidden_dim);
        out[..copy_len].copy_from_slice(&attn_out[..copy_len]);
        out
    };

    // Convert back to FP16
    output_f32.iter().map(|v| f16::from_f32(*v)).collect()
}

// ═══════════════════════════════════════════════════════════════════════════
// GQA Attention with Pre-projected Q/K/V (GPU projection path)
// ═══════════════════════════════════════════════════════════════════════════

/// Run self-attention for one layer given pre-projected Q, K, V vectors.
///
/// This variant is used when QKV projection has already been done on GPU.
/// The caller provides the projected Q/K/V as f32 slices (after D2H from GPU).
/// This function handles RoPE, KV cache update, multi-head attention, and
/// returns the concatenated attention output (before O projection).
///
/// The O projection is done by the caller on GPU — this function only returns
/// the concatenated head outputs [num_heads * head_dim] for the GPU O matmul.
///
/// Returns: `Vec<f32>` of length `num_heads * head_dim` (the attention output
/// before O projection).
pub fn self_attention_projected(
    q_proj: &[f32], // [hidden_dim] = [num_heads * head_dim]
    k_proj: &[f32], // [num_kv_heads * head_dim]
    v_proj: &[f32], // [num_kv_heads * head_dim]
    kv_cache: &mut KvCache,
    position: usize,
    config: &ModelConfig,
) -> Vec<f32> {
    let hidden_dim = config.hidden_dim as usize;
    let num_heads = config.num_heads as usize;
    let num_kv_heads = config.num_kv_heads as usize;
    let head_dim = hidden_dim / num_heads;
    let rope_base = config.rope_theta;

    // Split into heads and apply RoPE
    let mut q_heads: Vec<Vec<f32>> = Vec::with_capacity(num_heads);
    for h in 0..num_heads {
        let mut head = q_proj[h * head_dim..(h + 1) * head_dim].to_vec();
        kernels::apply_rope(&mut head, position, head_dim, rope_base);
        q_heads.push(head);
    }

    let mut k_heads: Vec<Vec<f32>> = Vec::with_capacity(num_kv_heads);
    let mut v_heads: Vec<Vec<f32>> = Vec::with_capacity(num_kv_heads);
    for h in 0..num_kv_heads {
        let mut k_head = k_proj[h * head_dim..(h + 1) * head_dim].to_vec();
        kernels::apply_rope(&mut k_head, position, head_dim, rope_base);
        k_heads.push(k_head);
        v_heads.push(v_proj[h * head_dim..(h + 1) * head_dim].to_vec());
    }

    // Append to KV cache
    kv_cache.append(&k_heads, &v_heads);

    // Run multi-head attention
    kernels::multi_head_attention(
        &q_heads,
        &kv_cache.k,
        &kv_cache.v,
        num_heads,
        num_kv_heads,
        head_dim,
        kv_cache.seq_len,
    )
}

// ═══════════════════════════════════════════════════════════════════════════
// Helper functions
// ═══════════════════════════════════════════════════════════════════════════

/// GEMV: output = input × weight^T, where weight is [out_dim, in_dim] in FP16.
/// input is f32, output is f32.
fn gemv_f16(input: &[f32], weight: &[f16], in_dim: usize, out_dim: usize) -> Vec<f32> {
    let mut output = vec![0.0f32; out_dim];
    for (row, out_val) in output.iter_mut().enumerate().take(out_dim) {
        let mut acc = 0.0f32;
        let row_start = row * in_dim;
        for col in 0..in_dim {
            acc += input[col] * weight[row_start + col].to_f32();
        }
        *out_val = acc;
    }
    output
}

/// GEMV for a slice of rows: compute rows [row_offset..row_offset+num_rows]
/// of the full weight matrix [total_out_dim, in_dim].
///
/// This avoids materializing the full kv_b_proj output when we only need
/// one head's k_nope or v slice.
fn gemv_f16_slice(
    input: &[f32],
    weight: &[f16],
    in_dim: usize,
    _total_out_dim: usize,
    row_offset: usize,
    num_rows: usize,
) -> Vec<f32> {
    let mut output = vec![0.0f32; num_rows];
    for (r, out_val) in output.iter_mut().enumerate().take(num_rows) {
        let row = row_offset + r;
        let mut acc = 0.0f32;
        let row_start = row * in_dim;
        for col in 0..in_dim {
            acc += input[col] * weight[row_start + col].to_f32();
        }
        *out_val = acc;
    }
    output
}

/// Public wrapper for `gemv_f16_slice`, used by engine GPU MLA path.
pub fn gemv_f16_slice_pub(
    input: &[f32],
    weight: &[f16],
    in_dim: usize,
    total_out_dim: usize,
    row_offset: usize,
    num_rows: usize,
) -> Vec<f32> {
    gemv_f16_slice(input, weight, in_dim, total_out_dim, row_offset, num_rows)
}

/// RMSNorm on f32 vector (no learned weight).
pub fn rms_norm_f32(x: &[f32]) -> Vec<f32> {
    let eps = 1e-5f32;
    let n = x.len();
    let sum_sq: f32 = x.iter().map(|v| v * v).sum();
    let rms = ((sum_sq / n as f32) + eps).sqrt();
    let inv_rms = 1.0 / rms;
    x.iter().map(|v| v * inv_rms).collect()
}

/// RMSNorm on f32 vector with learned FP16 weight.
pub fn rms_norm_f32_with_weight(x: &[f32], weight: &[f16]) -> Vec<f32> {
    let eps = 1e-5f32;
    let n = x.len();
    let sum_sq: f32 = x.iter().map(|v| v * v).sum();
    let rms = ((sum_sq / n as f32) + eps).sqrt();
    let inv_rms = 1.0 / rms;
    x.iter()
        .enumerate()
        .map(|(i, v)| {
            let w = if i < weight.len() {
                weight[i].to_f32()
            } else {
                1.0
            };
            v * inv_rms * w
        })
        .collect()
}
