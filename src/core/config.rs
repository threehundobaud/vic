//! Model configuration and metadata.

use crate::core::types::{DType, PAGE_SIZE};
use serde::{Deserialize, Serialize};

/// Complete model configuration.
///
/// Loaded from the .vib3 file header or from a config YAML during conversion.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ModelConfig {
    pub name: String,
    pub architecture: String,

    // Dimensions
    pub hidden_dim: u32,
    pub expert_hidden_dim: u32,
    pub num_layers: u32,
    pub num_moe_layers: u32,
    pub dense_layer_idx: u32,
    pub num_experts: u32,
    pub num_active_experts: u32,
    pub num_heads: u32,
    pub num_kv_heads: u32,
    pub max_seq_len: u32,
    pub vocab_size: u32,

    // Quantization
    pub expert_dtype: DType,
    pub shared_dtype: DType,

    // MLA configuration (optional — DeepSeek-V2/V3, Kimi K2.5)
    #[serde(default)]
    pub mla: Option<MlaConfig>,

    // DeltaNet configuration (optional — Qwen3.5 hybrid models)
    #[serde(default)]
    pub deltanet: Option<DeltaNetConfig>,

    // RoPE configuration
    /// Base frequency for Rotary Position Embeddings.
    /// - Mixtral: 1,000,000.0 (1M)
    /// - Llama 2: 10,000.0
    /// - Kimi K2.5: 50,000.0 (with YaRN scaling)
    ///   Default: 10000.0 (standard RoPE).
    #[serde(default = "default_rope_theta")]
    pub rope_theta: f32,

    /// RMSNorm epsilon for numerical stability.
    /// - Mixtral: 1e-5
    /// - Llama: 1e-5
    /// - Kimi K2.5: 1e-5
    ///   Default: 1e-5 (matches most models).
    #[serde(default = "default_rms_norm_eps")]
    pub rms_norm_eps: f32,

    // Router configuration
    /// Router scoring function: "softmax" (default) or "sigmoid".
    #[serde(default = "default_scoring_func")]
    pub scoring_func: String,

    /// Scaling factor for sigmoid router gating (DeepSeek-V3/Kimi K2.5).
    /// Applied as: weight = sigmoid(score) * routed_scaling_factor.
    /// Only used when scoring_func = "sigmoid".
    #[serde(default)]
    pub routed_scaling_factor: f32,

    /// Whether to normalize top-k routing weights to sum to 1.
    /// When true (default for DeepSeek-V3/Kimi K2.5), the weights are
    /// divided by their sum after applying the scoring function.
    /// Without this, sigmoid routing with scaling_factor > 1 causes
    /// hidden state overflow within a few layers.
    #[serde(default = "default_true")]
    pub norm_topk_prob: bool,

    /// Number of shared (unconditional) experts per MoE layer.
    /// These run on every token in addition to the routed experts.
    #[serde(default)]
    pub num_shared_experts: u32,

    /// Intermediate size for shared experts (may differ from moe_intermediate_size).
    #[serde(default)]
    pub shared_intermediate_size: u32,

    /// Intermediate size for dense (non-MoE) layers' FFN.
    /// For Kimi K2.5: 18432. Only used when dense_layer_idx > 0.
    #[serde(default)]
    pub dense_intermediate_size: u32,

    /// Explicit attention head dimension (when head_dim != hidden_dim / num_heads).
    /// Qwen3.5: head_dim=256 while hidden_dim/num_heads=3072/32=96.
    /// Default 0 means infer as hidden_dim / num_heads (standard transformers).
    #[serde(default)]
    pub attn_head_dim: u32,

    /// Number of expert groups for grouped top-k routing (DeepSeek-V3 field
    /// `n_group`). Experts are partitioned into `n_group` contiguous groups of
    /// `num_experts / n_group` each. When `> 1`, routing picks `topk_group`
    /// groups by aggregate group score and then top-k experts within those
    /// groups. When `1` (Kimi K2.5 / K2.6), routing degenerates to flat top-k.
    #[serde(default = "default_one_u32")]
    pub n_group: u32,

    /// Number of expert groups selected per token for grouped top-k routing
    /// (DeepSeek-V3 field `topk_group`). Only meaningful when `n_group > 1`.
    /// Default 1 matches the flat-top-k degenerate case.
    #[serde(default = "default_one_u32")]
    pub topk_group: u32,

    /// Number of native multi-token-prediction layers (DeepSeek-V3 field
    /// `num_nextn_predict_layers`). K2.5/K2.6 ship `0`. When `> 0`, the model
    /// carries MTP draft heads that can feed a speculative-decode front end.
    #[serde(default)]
    pub num_nextn_predict_layers: u32,

    /// M-RoPE interleaved section lengths (temporal, height, width), each in
    /// rotary-pair units. Present on Qwen-VL-family models (Qwen3.6, Qwen2.5-VL)
    /// that encode 3D positions. For text-only inference all three axes share
    /// the token position, so M-RoPE collapses to standard partial RoPE and
    /// this field is purely metadata — it becomes load-bearing when/if the
    /// engine adds a vision tower.
    #[serde(default)]
    pub mrope_sections: Option<Vec<u32>>,

    /// YaRN scaling factor for RoPE frequency extension. Qwen3.6-27B ships
    /// `factor=4.0` with `original_max_position_embeddings=262144` to reach
    /// ~1M context. `None` means no YaRN (native context only).
    #[serde(default)]
    pub rope_yarn_factor: Option<f32>,

    /// Native (pre-YaRN) max position count. Only meaningful when
    /// `rope_yarn_factor` is set.
    #[serde(default)]
    pub rope_original_max_position: Option<u32>,
}

impl ModelConfig {
    /// Post-deserialization fixup: fill in defaults that were missing in older
    /// .vib3 headers.  Call this after deserializing from the file header.
    pub fn fixup_defaults(&mut self) {
        // dense_intermediate_size was added after the first .vib3 conversion.
        // When it's 0 but the model has dense layers (dense_layer_idx > 0) and
        // uses MLA (Kimi K2.5 / DeepSeek-V3), we know the correct value.
        if self.dense_intermediate_size == 0 && self.dense_layer_idx > 0 && self.mla.is_some() {
            // Kimi K2.5 / DeepSeek-V3: intermediate_size = 18432
            self.dense_intermediate_size = 18432;
            tracing::info!(
                "ModelConfig fixup: dense_intermediate_size was 0, set to {}",
                self.dense_intermediate_size
            );
        }

        // softmax_scale was added to MlaConfig to fix the missing YaRN mscale
        // correction. Older .vib3 files have softmax_scale = 0.0 (serde default).
        // Fix up with the Kimi K2.5 values since that's the only MLA model.
        if let Some(ref mut mla) = self.mla {
            if mla.softmax_scale == 0.0 {
                // Apply YaRN mscale correction: mscale_all_dim=1.0, factor=64.0
                // for Kimi K2.5 / DeepSeek-V3 models.
                use crate::core::types::kimi_k25::{MSCALE_ALL_DIM, ROPE_SCALING_FACTOR};
                mla.softmax_scale = compute_mla_softmax_scale(
                    mla.qk_nope_head_dim,
                    mla.qk_rope_head_dim,
                    ROPE_SCALING_FACTOR,
                    MSCALE_ALL_DIM,
                );
                tracing::info!(
                    "ModelConfig fixup: MLA softmax_scale was 0, set to {:.6} (mscale={:.4}, base=1/sqrt({}))",
                    mla.softmax_scale,
                    yarn_get_mscale(ROPE_SCALING_FACTOR, MSCALE_ALL_DIM),
                    mla.qk_nope_head_dim + mla.qk_rope_head_dim,
                );
            }
        }
    }
}

fn default_rope_theta() -> f32 {
    10000.0
}

fn default_true() -> bool {
    true
}

fn default_one_u32() -> u32 {
    1
}

fn default_rms_norm_eps() -> f32 {
    1e-5
}

fn default_scoring_func() -> String {
    "softmax".to_string()
}

impl Default for ModelConfig {
    fn default() -> Self {
        Self {
            name: String::new(),
            architecture: String::new(),
            hidden_dim: 0,
            expert_hidden_dim: 0,
            num_layers: 0,
            num_moe_layers: 0,
            dense_layer_idx: 0,
            num_experts: 0,
            num_active_experts: 0,
            num_heads: 0,
            num_kv_heads: 0,
            max_seq_len: 0,
            vocab_size: 0,
            expert_dtype: DType::FP16,
            shared_dtype: DType::FP16,
            rope_theta: 10000.0,
            rms_norm_eps: 1e-5,
            mla: None,
            deltanet: None,
            scoring_func: "softmax".to_string(),
            routed_scaling_factor: 1.0,
            norm_topk_prob: true,
            num_shared_experts: 0,
            shared_intermediate_size: 0,
            dense_intermediate_size: 0,
            attn_head_dim: 0,
            n_group: 1,
            topk_group: 1,
            num_nextn_predict_layers: 0,
            mrope_sections: None,
            rope_yarn_factor: None,
            rope_original_max_position: None,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MlaConfig {
    pub kv_lora_rank: u32,
    pub q_lora_rank: u32,
    pub qk_rope_head_dim: u32,
    pub qk_nope_head_dim: u32,
    pub v_head_dim: u32,

    /// Pre-computed attention softmax scale, incorporating YaRN mscale correction.
    ///
    /// For models with YaRN RoPE scaling and `mscale_all_dim > 0`:
    ///   `mscale = 0.1 * mscale_all_dim * ln(rope_scaling_factor) + 1.0`
    ///   `softmax_scale = (1 / sqrt(qk_nope_head_dim + qk_rope_head_dim)) * mscale^2`
    ///
    /// Without YaRN mscale: `softmax_scale = 1 / sqrt(qk_nope_head_dim + qk_rope_head_dim)`
    ///
    /// For Kimi K2.5: mscale ≈ 1.4159, softmax_scale ≈ 0.1447 (vs 0.0722 without mscale).
    #[serde(default)]
    pub softmax_scale: f32,
}

/// Compute the YaRN mscale factor.
///
/// Matches the HuggingFace `yarn_get_mscale()` function from DeepSeek-V3 modeling code:
/// ```python
/// def yarn_get_mscale(scale=1, mscale=1):
///     if scale <= 1:
///         return 1.0
///     return 0.1 * mscale * math.log(scale) + 1.0
/// ```
pub fn yarn_get_mscale(scaling_factor: f64, mscale: f64) -> f64 {
    if scaling_factor <= 1.0 {
        1.0
    } else {
        0.1 * mscale * scaling_factor.ln() + 1.0
    }
}

/// Compute the MLA attention softmax scale with YaRN mscale correction.
///
/// This is the critical scaling factor applied to Q·K^T / scale in attention.
/// When `mscale_all_dim > 0`, the base 1/sqrt(d) scale is multiplied by mscale^2,
/// making attention logits ~2x larger for Kimi K2.5 (more focused attention).
pub fn compute_mla_softmax_scale(
    qk_nope_head_dim: u32,
    qk_rope_head_dim: u32,
    rope_scaling_factor: f64,
    mscale_all_dim: f64,
) -> f32 {
    let q_head_dim = (qk_nope_head_dim + qk_rope_head_dim) as f64;
    let base_scale = 1.0 / q_head_dim.sqrt();

    if mscale_all_dim > 0.0 {
        let mscale = yarn_get_mscale(rope_scaling_factor, mscale_all_dim);
        (base_scale * mscale * mscale) as f32
    } else {
        base_scale as f32
    }
}

impl ModelConfig {
    /// Bytes per expert (one layer, one expert, all segments).
    pub fn expert_size_bytes(&self) -> usize {
        // SwiGLU: up_proj + gate_proj + down_proj
        // up/gate: [expert_hidden_dim, hidden_dim]
        // down:    [hidden_dim, expert_hidden_dim]
        let up_gate_params = self.expert_hidden_dim as usize * self.hidden_dim as usize * 2; // up + gate
        let down_params = self.hidden_dim as usize * self.expert_hidden_dim as usize;
        self.expert_dtype.bytes_for(up_gate_params + down_params)
    }

    /// Total bytes for all expert weights.
    pub fn total_expert_bytes(&self) -> usize {
        self.expert_size_bytes() * self.num_experts as usize * self.num_moe_layers as usize
    }

    /// Effective attention head dimension.
    /// Returns attn_head_dim if explicitly set, otherwise hidden_dim / num_heads.
    pub fn effective_head_dim(&self) -> u32 {
        if self.attn_head_dim > 0 {
            self.attn_head_dim
        } else if self.num_heads > 0 {
            self.hidden_dim / self.num_heads
        } else {
            128 // fallback
        }
    }

    /// Number of 2MB pages per expert segment.
    pub fn pages_per_segment(&self) -> usize {
        let segment_params = self.expert_hidden_dim as usize * self.hidden_dim as usize;
        let segment_bytes = self.expert_dtype.bytes_for(segment_params);
        segment_bytes.div_ceil(PAGE_SIZE)
    }

    /// Number of 2MB pages per expert (all 3 segments).
    pub fn pages_per_expert(&self) -> usize {
        self.pages_per_segment() * 3 // up_proj + gate_proj + down_proj
    }

    /// Total number of expert pages in the model.
    pub fn total_expert_pages(&self) -> usize {
        self.pages_per_expert() * self.num_experts as usize * self.num_moe_layers as usize
    }

    /// Estimated total model size in bytes.
    pub fn estimated_total_bytes(&self) -> usize {
        // Expert weights + shared layers (rough estimate)
        let expert_bytes = self.total_expert_bytes();
        let shared_estimate = self.estimated_shared_bytes();
        expert_bytes + shared_estimate
    }

    /// Estimated shared layer size (attention, embeddings, norms, DeltaNet projections).
    ///
    /// This estimates the **device-side FP16** size of all shared tensors that
    /// get cached via `ensure_shared_tensor_device`.  It must be accurate enough
    /// that the VRAM budget calculation leaves room for the device cache.
    pub fn estimated_shared_bytes(&self) -> usize {
        let h = self.hidden_dim as usize;
        let head_dim = self.effective_head_dim() as usize;

        // Global: embeddings + lm_head + final_norm
        let embedding = self.vocab_size as usize * h;
        let lm_head = embedding;
        let global = (embedding + lm_head + h) * 2; // FP16

        // Per-layer common: norms + router + shared expert
        let norms_per_layer = h * 2; // attn_norm + ffn_norm
        let router_per_layer = self.num_experts as usize * h;
        let shared_expert_per_layer = if self.num_shared_experts > 0 {
            h * self.shared_intermediate_size as usize * 3 + h // up+gate+down + gate_scalar
        } else {
            0
        };
        let common_per_layer = (norms_per_layer + router_per_layer + shared_expert_per_layer) * 2;

        // Attention layers: Q(doubled for gating) + K + V + O + q_norm + k_norm
        let num_attn_layers;
        let attn_per_layer;
        let num_dn_layers;
        let dn_per_layer;

        if let Some(ref dn) = self.deltanet {
            let interval = dn.full_attention_interval as usize;
            num_attn_layers = if interval > 0 {
                self.num_layers as usize / interval
            } else {
                self.num_layers as usize
            };
            num_dn_layers = self.num_layers as usize - num_attn_layers;

            // Gated attention: Q proj outputs 2x (Q + gate), interleaved
            let q_size = h * (self.num_heads as usize * head_dim * 2);
            let k_size = h * (self.num_kv_heads as usize * head_dim);
            let v_size = k_size;
            let o_size = (self.num_heads as usize * head_dim) * h;
            let q_norm = self.num_heads as usize * head_dim;
            let k_norm = self.num_kv_heads as usize * head_dim;
            attn_per_layer = (q_size + k_size + v_size + o_size + q_norm + k_norm) * 2;

            // DeltaNet: in_proj_qkv + z + beta + alpha + conv1d + dt_bias + A_log + norm + out_proj
            let inner = dn.inner_dim as usize;
            let nkh = dn.num_key_heads as usize;
            let nvh = dn.num_value_heads as usize;
            let khd = dn.key_head_dim as usize;
            let vhd = dn.value_head_dim as usize;
            let qkv = h * (inner + nkh * khd + nvh * vhd);
            let z = h * inner;
            let beta = h * inner;
            let alpha = h * inner;
            let conv = nvh * vhd * dn.conv_kernel_size as usize;
            let dt = nvh;
            let a_log = nvh * khd;
            let norm = inner;
            let out = inner * h;
            dn_per_layer = (qkv + z + beta + alpha + conv + dt + a_log + norm + out) * 2;
        } else {
            num_attn_layers = self.num_layers as usize;
            num_dn_layers = 0;
            // Standard attention: Q + K + V + O
            let per_layer = h * h * 4;
            attn_per_layer = per_layer * 2;
            dn_per_layer = 0;
        }

        global
            + common_per_layer * self.num_layers as usize
            + attn_per_layer * num_attn_layers
            + dn_per_layer * num_dn_layers
    }

    /// Create config for Kimi K2.5 / K2.6.
    ///
    /// Values sourced from `moonshotai/Kimi-K2.6/config.json` on HuggingFace.
    /// K2.5 and K2.6 share the exact same architecture constants (hf model_type
    /// `kimi_k25`, text model_type `kimi_k2`, DeepseekV3ForCausalLM stack); only
    /// the weights differ. Use the `name` and `architecture` args to distinguish
    /// them at the registry/telemetry level.
    pub fn kimi_k25() -> Self {
        Self::kimi_k2_family("Kimi-K2.5", "kimi-k2.5")
    }

    pub fn kimi_k26() -> Self {
        Self::kimi_k2_family("Kimi-K2.6", "kimi-k2.6")
    }

    fn kimi_k2_family(name: &str, architecture: &str) -> Self {
        use crate::core::types::kimi_k25::*;
        Self {
            name: name.into(),
            architecture: architecture.into(),
            hidden_dim: HIDDEN_DIM,
            expert_hidden_dim: EXPERT_HIDDEN_DIM,
            num_layers: NUM_LAYERS,
            num_moe_layers: NUM_MOE_LAYERS,
            dense_layer_idx: DENSE_LAYERS,
            num_experts: NUM_EXPERTS,
            num_active_experts: NUM_ACTIVE,
            num_heads: NUM_HEADS,
            num_kv_heads: NUM_KV_HEADS,
            max_seq_len: MAX_SEQ_LEN,
            vocab_size: VOCAB_SIZE,
            rope_theta: ROPE_THETA as f32,
            rms_norm_eps: 1e-5, // Kimi K2 family text_config.rms_norm_eps
            expert_dtype: DType::INT4,
            shared_dtype: DType::BF16,
            mla: Some(MlaConfig {
                kv_lora_rank: KV_LORA_RANK,
                q_lora_rank: Q_LORA_RANK,
                qk_rope_head_dim: QK_ROPE_HEAD_DIM,
                qk_nope_head_dim: QK_NOPE_HEAD_DIM,
                v_head_dim: V_HEAD_DIM,
                softmax_scale: compute_mla_softmax_scale(
                    QK_NOPE_HEAD_DIM,
                    QK_ROPE_HEAD_DIM,
                    ROPE_SCALING_FACTOR,
                    MSCALE_ALL_DIM,
                ),
            }),
            deltanet: None,
            scoring_func: SCORING_FUNC.to_string(),
            routed_scaling_factor: ROUTED_SCALING_FACTOR as f32,
            norm_topk_prob: true,
            num_shared_experts: NUM_SHARED_EXPERTS,
            shared_intermediate_size: SHARED_INTERMEDIATE_SIZE,
            dense_intermediate_size: 18432, // Kimi K2 family dense-layer intermediate_size
            attn_head_dim: 0,               // Standard: head_dim = hidden_dim / num_heads
            n_group: N_GROUP,
            topk_group: TOPK_GROUP,
            num_nextn_predict_layers: NUM_NEXTN_PREDICT_LAYERS,
            mrope_sections: None,
            rope_yarn_factor: None,
            rope_original_max_position: None,
        }
    }

    /// Create config for Qwen3.5-122B-A10B.
    ///
    /// Hybrid architecture: 75% DeltaNet (linear attention) + 25% full attention,
    /// with MoE on every layer. Values from `Qwen/Qwen3.5-122B-A10B/config.json`.
    pub fn qwen35_122b() -> Self {
        use crate::core::types::qwen35_122b::*;

        // Build per-layer attention/deltanet map
        let layer_is_attention: Vec<bool> =
            (0..NUM_LAYERS).map(|i| is_attention_layer(i)).collect();

        Self {
            name: "Qwen3.5-122B-A10B".into(),
            architecture: "qwen3_5_moe".into(),
            hidden_dim: HIDDEN_DIM,
            expert_hidden_dim: EXPERT_INTERMEDIATE_SIZE,
            num_layers: NUM_LAYERS,
            num_moe_layers: NUM_LAYERS, // MoE on every layer
            dense_layer_idx: 0,         // no dense-only layers
            num_experts: NUM_EXPERTS,
            num_active_experts: NUM_ACTIVE_EXPERTS,
            num_heads: NUM_ATTN_HEADS,
            num_kv_heads: NUM_KV_HEADS,
            max_seq_len: MAX_SEQ_LEN,
            vocab_size: VOCAB_SIZE,
            rope_theta: ROPE_THETA as f32,
            rms_norm_eps: RMS_NORM_EPS as f32,
            expert_dtype: DType::INT4,
            shared_dtype: DType::BF16,
            mla: None,
            deltanet: Some(DeltaNetConfig {
                num_key_heads: DELTANET_NUM_KEY_HEADS,
                num_value_heads: DELTANET_NUM_VALUE_HEADS,
                key_head_dim: DELTANET_KEY_HEAD_DIM,
                value_head_dim: DELTANET_VALUE_HEAD_DIM,
                conv_kernel_size: DELTANET_CONV_KERNEL,
                inner_dim: DELTANET_INNER_DIM,
                full_attention_interval: FULL_ATTN_INTERVAL,
                layer_is_attention,
            }),
            scoring_func: SCORING_FUNC.to_string(),
            routed_scaling_factor: 1.0,
            norm_topk_prob: true, // softmax routing with normalized weights
            num_shared_experts: NUM_SHARED_EXPERTS,
            shared_intermediate_size: SHARED_EXPERT_INTERMEDIATE_SIZE,
            dense_intermediate_size: 0, // no dense layers
            attn_head_dim: HEAD_DIM,    // 256 (differs from hidden_dim/num_heads = 96)
            n_group: 1,
            topk_group: 1,
            num_nextn_predict_layers: 0,
            mrope_sections: None,
            rope_yarn_factor: None,
            rope_original_max_position: None,
        }
    }

    /// Create config for Qwen3.5-35B-A3B.
    ///
    /// Hybrid architecture: 75% DeltaNet (linear attention) + 25% full attention,
    /// with MoE on every layer. Same family as 122B but smaller dimensions.
    /// Values from `Qwen/Qwen3.5-35B-A3B/config.json` and GGUF metadata.
    pub fn qwen35_35b() -> Self {
        use crate::core::types::qwen35_35b::*;

        // Build per-layer attention/deltanet map
        let layer_is_attention: Vec<bool> =
            (0..NUM_LAYERS).map(|i| is_attention_layer(i)).collect();

        Self {
            name: "Qwen3.5-35B-A3B".into(),
            architecture: "qwen3_5_moe".into(),
            hidden_dim: HIDDEN_DIM,
            expert_hidden_dim: EXPERT_INTERMEDIATE_SIZE,
            num_layers: NUM_LAYERS,
            num_moe_layers: NUM_LAYERS, // MoE on every layer
            dense_layer_idx: 0,         // no dense-only layers
            num_experts: NUM_EXPERTS,
            num_active_experts: NUM_ACTIVE_EXPERTS,
            num_heads: NUM_ATTN_HEADS,
            num_kv_heads: NUM_KV_HEADS,
            max_seq_len: MAX_SEQ_LEN,
            vocab_size: VOCAB_SIZE,
            rope_theta: ROPE_THETA as f32,
            rms_norm_eps: RMS_NORM_EPS as f32,
            expert_dtype: DType::INT4,
            shared_dtype: DType::BF16,
            mla: None,
            deltanet: Some(DeltaNetConfig {
                num_key_heads: DELTANET_NUM_KEY_HEADS,
                num_value_heads: DELTANET_NUM_VALUE_HEADS,
                key_head_dim: DELTANET_KEY_HEAD_DIM,
                value_head_dim: DELTANET_VALUE_HEAD_DIM,
                conv_kernel_size: DELTANET_CONV_KERNEL,
                inner_dim: DELTANET_INNER_DIM,
                full_attention_interval: FULL_ATTN_INTERVAL,
                layer_is_attention,
            }),
            scoring_func: SCORING_FUNC.to_string(),
            routed_scaling_factor: 1.0,
            norm_topk_prob: true, // softmax routing with normalized weights
            num_shared_experts: NUM_SHARED_EXPERTS,
            shared_intermediate_size: SHARED_EXPERT_INTERMEDIATE_SIZE,
            dense_intermediate_size: 0, // no dense layers
            attn_head_dim: HEAD_DIM,    // 256 (differs from hidden_dim/num_heads = 128)
            n_group: 1,
            topk_group: 1,
            num_nextn_predict_layers: 0,
            mrope_sections: None,
            rope_yarn_factor: None,
            rope_original_max_position: None,
        }
    }

    /// Create config for Qwen3.6-27B (dense hybrid, VLM-capable).
    ///
    /// 64 layers of 3:1 DeltaNet → GatedAttention, no routed experts (each
    /// layer ends in a single FFN with intermediate_size=17408). Released
    /// 2026-04-21; positioned as flagship-tier coding in a 27 B dense model.
    ///
    /// M-RoPE interleaved `[11, 11, 10]` is carried as metadata; for
    /// text-only inference (T=H=W) it reduces to standard partial RoPE
    /// over the 64-dim rotary slice, so no new kernel is needed.
    ///
    /// NOTE: the dense + DeltaNet-hybrid shape is a new class for vib3.
    /// The runtime expert/routing code still needs a path that bypasses
    /// the MoE router for `num_experts == 0`. Until that lands the config
    /// parses but `Engine::new` will fail on dispatch.
    pub fn qwen36_27b() -> Self {
        use crate::core::types::qwen36_27b::*;

        let layer_is_attention: Vec<bool> =
            (0..NUM_LAYERS).map(is_attention_layer).collect();

        Self {
            name: "Qwen3.6-27B".into(),
            architecture: "qwen3_6_dense".into(),
            hidden_dim: HIDDEN_DIM,
            expert_hidden_dim: 0,        // dense — no routed experts
            num_layers: NUM_LAYERS,
            num_moe_layers: 0,           // all layers dense
            dense_layer_idx: NUM_LAYERS, // every layer is a dense FFN
            num_experts: 0,
            num_active_experts: 0,
            num_heads: NUM_ATTN_HEADS,
            num_kv_heads: NUM_KV_HEADS,
            max_seq_len: MAX_SEQ_LEN,
            vocab_size: VOCAB_SIZE,
            rope_theta: ROPE_THETA as f32,
            rms_norm_eps: RMS_NORM_EPS as f32,
            expert_dtype: DType::BF16, // BF16-only weights at release (no official quant)
            shared_dtype: DType::BF16,
            mla: None,
            deltanet: Some(DeltaNetConfig {
                num_key_heads: DELTANET_NUM_KEY_HEADS,
                num_value_heads: DELTANET_NUM_VALUE_HEADS,
                key_head_dim: DELTANET_KEY_HEAD_DIM,
                value_head_dim: DELTANET_VALUE_HEAD_DIM,
                conv_kernel_size: DELTANET_CONV_KERNEL,
                inner_dim: DELTANET_INNER_DIM,
                full_attention_interval: FULL_ATTN_INTERVAL,
                layer_is_attention,
            }),
            scoring_func: "softmax".to_string(),
            routed_scaling_factor: 1.0,
            norm_topk_prob: true,
            num_shared_experts: 0,
            shared_intermediate_size: 0,
            dense_intermediate_size: INTERMEDIATE_SIZE,
            attn_head_dim: HEAD_DIM, // 256 (differs from hidden_dim/num_heads = 213)
            n_group: 1,
            topk_group: 1,
            num_nextn_predict_layers: 0,
            mrope_sections: Some(MROPE_SECTIONS.to_vec()),
            rope_yarn_factor: Some(YARN_FACTOR as f32),
            rope_original_max_position: Some(MAX_SEQ_LEN),
        }
    }
}

impl ModelConfig {
    /// Parse a HuggingFace `config.json` into a `ModelConfig`.
    ///
    /// Supports multiple MoE architectures:
    /// - **Mixtral**: `model_type = "mixtral"` — 8 experts, top-2
    /// - **DeepSeek-V2/V3**: `model_type = "deepseek_v2"` — 160+ experts, top-6/8, MLA
    /// - **Kimi/Moonshot**: similar to DeepSeek V3 architecture
    /// - **Qwen2-MoE**: `model_type = "qwen2_moe"`
    ///
    /// Fields not present in HF config (like `expert_dtype`) default to BF16 for
    /// source weights — the converter will quantize to INT4 during conversion.
    pub fn from_hf_config(
        json: &serde_json::Value,
        name: &str,
    ) -> std::result::Result<Self, String> {
        let model_type = json
            .get("model_type")
            .and_then(|v| v.as_str())
            .unwrap_or("unknown");

        // Qwen3.6-27B ships under `model_type = "qwen3_5"` / `"qwen3_5_text"`.
        // Distinguish dense vs MoE by the presence of MoE field names — the
        // 27B dense carries none, the 35B-A3B MoE sibling does.
        let has_moe_fields = json.get("num_local_experts").is_some()
            || json.get("n_routed_experts").is_some()
            || json.get("num_experts").is_some();
        let is_qwen_dense = matches!(
            model_type,
            "qwen3_5" | "qwen3_5_text" | "qwen3_6" | "qwen3_6_dense"
        ) && !has_moe_fields;

        let hidden_dim = json
            .get("hidden_size")
            .and_then(|v| v.as_u64())
            .ok_or("Missing hidden_size")? as u32;

        let num_heads = json
            .get("num_attention_heads")
            .and_then(|v| v.as_u64())
            .ok_or("Missing num_attention_heads")? as u32;

        let num_kv_heads = json
            .get("num_key_value_heads")
            .and_then(|v| v.as_u64())
            .unwrap_or(num_heads as u64) as u32;

        let num_layers = json
            .get("num_hidden_layers")
            .and_then(|v| v.as_u64())
            .ok_or("Missing num_hidden_layers")? as u32;

        let vocab_size = json
            .get("vocab_size")
            .and_then(|v| v.as_u64())
            .unwrap_or(32000) as u32;

        let max_seq_len = json
            .get("max_position_embeddings")
            .and_then(|v| v.as_u64())
            .unwrap_or(4096) as u32;

        // MoE-specific fields. Default to 1 so historical non-MoE configs
        // (Llama-style dense) still load; dense-architecture detection below
        // zeroes this out for qwen3_6_dense etc.
        let num_experts = json
            .get("num_local_experts")
            .or_else(|| json.get("n_routed_experts"))
            .or_else(|| json.get("num_experts"))
            .and_then(|v| v.as_u64())
            .unwrap_or(1) as u32;

        let num_active_experts = json
            .get("num_experts_per_tok")
            .or_else(|| json.get("num_experts_per_token"))
            .or_else(|| json.get("topk"))
            .and_then(|v| v.as_u64())
            .unwrap_or(2) as u32;

        // Expert hidden dimension (intermediate_size for MoE experts)
        let expert_hidden_dim = json
            .get("moe_intermediate_size")
            .or_else(|| json.get("expert_intermediate_size"))
            .or_else(|| json.get("intermediate_size"))
            .and_then(|v| v.as_u64())
            .unwrap_or(hidden_dim as u64 * 4) as u32;

        // First k dense layers (DeepSeek-style: first_k_dense_replace).
        // For fully dense architectures (Qwen3.6-27B has no routed experts),
        // every layer is a dense FFN, so bump `dense_layer_idx` to cover them
        // all so the engine's `is_dense_ffn_layer = layer < dense_layer_idx`
        // guard fires on each layer.
        let dense_layer_idx = if is_qwen_dense {
            num_layers
        } else {
            json.get("first_k_dense_replace")
                .and_then(|v| v.as_u64())
                .unwrap_or(0) as u32
        };
        // Force `num_experts = 0` under dense mode so buffer allocation +
        // router dispatch treat this as no-MoE rather than 1-expert-MoE.
        let num_experts = if is_qwen_dense { 0 } else { num_experts };
        let num_active_experts = if is_qwen_dense { 0 } else { num_active_experts };

        // Number of MoE layers = total layers - dense layers at the start
        let num_moe_layers = if num_experts > 1 {
            // Some models specify which layers are MoE
            let moe_layer_freq = json
                .get("moe_layer_frequency")
                .or_else(|| json.get("decoder_sparse_step"))
                .and_then(|v| v.as_u64())
                .unwrap_or(1) as u32;

            if moe_layer_freq > 1 {
                // Not every layer is MoE (e.g., every Nth layer)
                (num_layers - dense_layer_idx).div_ceil(moe_layer_freq)
            } else {
                num_layers - dense_layer_idx
            }
        } else {
            0
        };

        // YaRN RoPE scaling parameters (for mscale attention correction)
        let rope_scaling_factor = json
            .get("rope_scaling")
            .and_then(|rs| rs.get("factor"))
            .and_then(|v| v.as_f64())
            .unwrap_or(1.0);
        let mscale_all_dim = json
            .get("rope_scaling")
            .and_then(|rs| rs.get("mscale_all_dim"))
            .and_then(|v| v.as_f64())
            .unwrap_or(0.0);

        // MLA configuration (DeepSeek-V2/V3 / Kimi)
        let mla = {
            let kv_lora_rank = json.get("kv_lora_rank").and_then(|v| v.as_u64());
            let q_lora_rank = json.get("q_lora_rank").and_then(|v| v.as_u64());
            let qk_rope_head_dim = json.get("qk_rope_head_dim").and_then(|v| v.as_u64());
            let qk_nope_head_dim = json.get("qk_nope_head_dim").and_then(|v| v.as_u64());
            let v_head_dim = json.get("v_head_dim").and_then(|v| v.as_u64());

            if let (Some(kv), Some(q), Some(rope), Some(nope), Some(v)) = (
                kv_lora_rank,
                q_lora_rank,
                qk_rope_head_dim,
                qk_nope_head_dim,
                v_head_dim,
            ) {
                Some(MlaConfig {
                    kv_lora_rank: kv as u32,
                    q_lora_rank: q as u32,
                    qk_rope_head_dim: rope as u32,
                    qk_nope_head_dim: nope as u32,
                    v_head_dim: v as u32,
                    softmax_scale: compute_mla_softmax_scale(
                        nope as u32,
                        rope as u32,
                        rope_scaling_factor,
                        mscale_all_dim,
                    ),
                })
            } else {
                None
            }
        };

        // Determine source dtype from torch_dtype
        let _torch_dtype = json
            .get("torch_dtype")
            .and_then(|v| v.as_str())
            .unwrap_or("bfloat16");

        // Architecture string. Qwen3.6-27B reuses the `qwen3_5` / `qwen3_5_text`
        // model_type class name (Qwen team shipped it under the 3.5 internal
        // class). `is_qwen_dense` was computed earlier in this function from
        // the presence/absence of MoE field names.
        let architecture = match model_type {
            "mixtral" => "mixtral",
            "deepseek_v2" | "deepseek_v3" => "deepseek-v2",
            "qwen2_moe" => "qwen2-moe",
            "qwen3_5_moe" | "qwen3_5_moe_text" => "qwen3_5_moe",
            // Qwen3.6 family: dense hybrid (DeltaNet + GatedAttention), no
            // routed experts. `qwen3_6_moe` = the 35B-A3B MoE sibling.
            "qwen3_6_moe" | "qwen3_6_moe_text" => "qwen3_6_moe",
            _ if is_qwen_dense => "qwen3_6_dense",
            _ => model_type,
        };

        // Router configuration
        let scoring_func = json
            .get("scoring_func")
            .and_then(|v| v.as_str())
            .unwrap_or("softmax")
            .to_string();

        let routed_scaling_factor = json
            .get("routed_scaling_factor")
            .and_then(|v| v.as_f64())
            .unwrap_or(1.0) as f32;

        let norm_topk_prob = json
            .get("norm_topk_prob")
            .and_then(|v| v.as_bool())
            .unwrap_or(true); // Default true (DeepSeek-V3/Kimi K2.5)

        // RoPE theta
        let rope_theta = json
            .get("rope_theta")
            .and_then(|v| v.as_f64())
            .unwrap_or(10000.0) as f32;

        // RMSNorm epsilon
        let rms_norm_eps = json
            .get("rms_norm_eps")
            .and_then(|v| v.as_f64())
            .unwrap_or(1e-5) as f32;

        // Shared experts
        let num_shared_experts = json
            .get("n_shared_experts")
            .or_else(|| json.get("num_shared_experts"))
            .and_then(|v| v.as_u64())
            .unwrap_or(0) as u32;

        // For MoE models (DeepSeek-V3 / Kimi K2.5), the shared expert uses the
        // same intermediate size as routed experts (moe_intermediate_size), NOT the
        // dense layer's intermediate_size. The actual shared expert weight shape is
        // [moe_intermediate_size * n_shared_experts, hidden_size].
        let shared_intermediate_size = if num_shared_experts > 0 && expert_hidden_dim > 0 {
            // shared expert intermediate = n_shared_experts × moe_intermediate_size
            num_shared_experts * expert_hidden_dim
        } else {
            json.get("intermediate_size")
                .and_then(|v| v.as_u64())
                .unwrap_or(expert_hidden_dim as u64 * 4) as u32
        };

        // Dense layer intermediate size (for layers before dense_layer_idx)
        let dense_intermediate_size = json
            .get("intermediate_size")
            .and_then(|v| v.as_u64())
            .unwrap_or(0) as u32;

        // DeltaNet / hybrid attention config (Qwen3.5 series)
        let deltanet = {
            let linear_num_key_heads = json.get("linear_num_key_heads").and_then(|v| v.as_u64());
            let linear_num_value_heads =
                json.get("linear_num_value_heads").and_then(|v| v.as_u64());
            let linear_key_head_dim = json.get("linear_key_head_dim").and_then(|v| v.as_u64());
            let linear_value_head_dim = json.get("linear_value_head_dim").and_then(|v| v.as_u64());
            let linear_conv_kernel_dim =
                json.get("linear_conv_kernel_dim").and_then(|v| v.as_u64());

            if let (Some(nk), Some(nv), Some(dk), Some(dv), Some(ck)) = (
                linear_num_key_heads,
                linear_num_value_heads,
                linear_key_head_dim,
                linear_value_head_dim,
                linear_conv_kernel_dim,
            ) {
                let full_attn_interval = json
                    .get("full_attention_interval")
                    .and_then(|v| v.as_u64())
                    .unwrap_or(4) as u32;

                let inner_dim = nv * dv;

                // Build layer_is_attention from layer_types array or interval
                let layer_is_attention: Vec<bool> =
                    if let Some(layer_types) = json.get("layer_types").and_then(|v| v.as_array()) {
                        layer_types
                            .iter()
                            .map(|v| v.as_str().unwrap_or("linear_attention") == "full_attention")
                            .collect()
                    } else {
                        (0..num_layers)
                            .map(|i| (i + 1) % full_attn_interval == 0)
                            .collect()
                    };

                Some(DeltaNetConfig {
                    num_key_heads: nk as u32,
                    num_value_heads: nv as u32,
                    key_head_dim: dk as u32,
                    value_head_dim: dv as u32,
                    conv_kernel_size: ck as u32,
                    inner_dim: inner_dim as u32,
                    full_attention_interval: full_attn_interval,
                    layer_is_attention,
                })
            } else {
                None
            }
        };

        // Qwen3.5 has shared_expert_intermediate_size as a dedicated field
        let shared_intermediate_size = json
            .get("shared_expert_intermediate_size")
            .and_then(|v| v.as_u64())
            .map(|v| v as u32)
            .unwrap_or(shared_intermediate_size);

        // Qwen3.5 also puts num_shared_experts = 1 implicitly via shared_expert_intermediate_size
        let num_shared_experts =
            if num_shared_experts == 0 && json.get("shared_expert_intermediate_size").is_some() {
                1
            } else {
                num_shared_experts
            };

        // Override RoPE theta from rope_parameters if present
        let rope_theta = json
            .get("rope_parameters")
            .and_then(|rp| rp.get("rope_theta"))
            .and_then(|v| v.as_f64())
            .unwrap_or(rope_theta as f64) as f32;

        // Explicit head_dim (when it differs from hidden_dim / num_heads).
        // Qwen3.5: head_dim=256 while hidden_dim=3072/32_heads=96.
        let attn_head_dim = json.get("head_dim").and_then(|v| v.as_u64()).unwrap_or(0) as u32;

        // Grouped top-k (DeepSeek-V3 / Kimi K2.5 / K2.6). n_group=1 on Kimi
        // degenerates to flat top-k. topk_group defaults to n_group.
        let n_group = json
            .get("n_group")
            .and_then(|v| v.as_u64())
            .unwrap_or(1) as u32;
        let topk_group = json
            .get("topk_group")
            .and_then(|v| v.as_u64())
            .unwrap_or(n_group as u64) as u32;

        // Native multi-token prediction layer count (DeepSeek-V3 family).
        // K2.5/K2.6 ship 0.
        let num_nextn_predict_layers = json
            .get("num_nextn_predict_layers")
            .and_then(|v| v.as_u64())
            .unwrap_or(0) as u32;

        // Qwen-VL family (Qwen3.6, Qwen2.5-VL) stores rotary/M-RoPE config
        // under `rope_parameters` (preferred) or legacy `rope_scaling`.
        //   rope_parameters: {
        //     rope_type: "mrope" | "yarn" | ...,
        //     mrope_section: [11, 11, 10],
        //     mrope_interleaved: true,
        //     factor: 4.0,                           // YaRN scale
        //     original_max_position_embeddings: 262144,
        //   }
        // For text-only inference M-RoPE degenerates to standard partial RoPE
        // (T=H=W=token position), so we just surface the metadata.
        let rope_params = json
            .get("rope_parameters")
            .or_else(|| json.get("rope_scaling"));
        let mrope_sections: Option<Vec<u32>> = rope_params
            .and_then(|rp| rp.get("mrope_section"))
            .and_then(|v| v.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|x| x.as_u64())
                    .map(|x| x as u32)
                    .collect()
            });
        let rope_yarn_factor: Option<f32> = rope_params.and_then(|rp| {
            let typ = rp
                .get("rope_type")
                .or_else(|| rp.get("type"))
                .and_then(|v| v.as_str());
            if typ == Some("yarn") {
                rp.get("factor").and_then(|v| v.as_f64()).map(|v| v as f32)
            } else {
                None
            }
        });
        let rope_original_max_position: Option<u32> = rope_params
            .and_then(|rp| rp.get("original_max_position_embeddings"))
            .and_then(|v| v.as_u64())
            .map(|v| v as u32);

        Ok(ModelConfig {
            name: name.to_string(),
            architecture: architecture.to_string(),
            hidden_dim,
            expert_hidden_dim,
            num_layers,
            num_moe_layers,
            dense_layer_idx,
            num_experts,
            num_active_experts,
            num_heads,
            num_kv_heads,
            max_seq_len,
            vocab_size,
            rope_theta,
            rms_norm_eps,
            expert_dtype: DType::INT4, // target: always quantize experts to INT4
            shared_dtype: DType::BF16, // shared layers stay at source precision
            mla,
            deltanet,
            scoring_func,
            routed_scaling_factor,
            norm_topk_prob,
            num_shared_experts,
            shared_intermediate_size,
            dense_intermediate_size,
            attn_head_dim,
            n_group,
            topk_group,
            num_nextn_predict_layers,
            mrope_sections,
            rope_yarn_factor,
            rope_original_max_position,
        })
    }

    /// Load a HuggingFace `config.json` from a file path.
    pub fn from_hf_config_path(path: &std::path::Path) -> std::result::Result<Self, String> {
        let data = std::fs::read_to_string(path)
            .map_err(|e| format!("Failed to read {}: {}", path.display(), e))?;
        let json: serde_json::Value = serde_json::from_str(&data)
            .map_err(|e| format!("Failed to parse config.json: {}", e))?;

        // Vision-language models (e.g., Kimi K2.5) nest the text model config
        // under "text_config". If present and the top-level JSON lacks "hidden_size",
        // unwrap it so the parser sees the actual model architecture fields.
        let effective_json = if json.get("hidden_size").is_none() {
            if let Some(text_cfg) = json.get("text_config") {
                text_cfg
            } else {
                &json
            }
        } else {
            &json
        };

        // Try to extract model name from _name_or_path or the parent directory
        let name = effective_json
            .get("_name_or_path")
            .or_else(|| json.get("_name_or_path"))
            .and_then(|v| v.as_str())
            .map(|s| s.to_string())
            .unwrap_or_else(|| {
                path.parent()
                    .and_then(|p| p.file_name())
                    .and_then(|n| n.to_str())
                    .unwrap_or("unknown")
                    .to_string()
            });

        Self::from_hf_config(effective_json, &name)
    }
}

/// DeltaNet (Gated Delta Rule) configuration for hybrid attention models.
///
/// Used by Qwen3.5 MoE series where 75% of layers use DeltaNet linear attention
/// instead of standard softmax attention. The DeltaNet layers maintain a recurrent
/// state S ∈ R^{head_v_dim × head_v_dim} per head, updated via the delta rule.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DeltaNetConfig {
    /// Number of key/query heads (groups). Qwen3.5-122B: 16.
    pub num_key_heads: u32,
    /// Number of value heads (determines state parallelism). Qwen3.5-122B: 64.
    pub num_value_heads: u32,
    /// Dimension per key head. Qwen3.5-122B: 128.
    pub key_head_dim: u32,
    /// Dimension per value head. Qwen3.5-122B: 128.
    pub value_head_dim: u32,
    /// Causal 1D depthwise convolution kernel size. Qwen3.5-122B: 4.
    pub conv_kernel_size: u32,
    /// Total inner dimension = num_value_heads × value_head_dim. Qwen3.5-122B: 8192.
    pub inner_dim: u32,
    /// Interval at which full attention layers appear (e.g., every 4th layer).
    pub full_attention_interval: u32,
    /// Per-layer type: true = full attention, false = DeltaNet.
    /// Length = num_layers. Derived from `layer_types` in HF config.
    pub layer_is_attention: Vec<bool>,
}

impl DeltaNetConfig {
    /// Recurrent state size per layer: head_v_dim × head_v_dim × num_value_heads.
    /// For Qwen3.5-122B: 128 × 128 × 64 = 1,048,576 floats = 4 MB (FP32).
    pub fn state_size_floats(&self) -> usize {
        self.value_head_dim as usize * self.value_head_dim as usize * self.num_value_heads as usize
    }

    /// Conv state size per layer: (conv_kernel_size - 1) × conv_channels.
    /// conv_channels = key_dim × 2 + value_dim (Q + K + V concatenated).
    /// For Qwen3.5-122B: 3 × (2048 + 2048 + 8192) = 3 × 12288 = 36864 floats.
    pub fn conv_state_size_floats(&self) -> usize {
        let key_dim = self.num_key_heads as usize * self.key_head_dim as usize;
        let conv_channels = key_dim * 2 + self.inner_dim as usize;
        (self.conv_kernel_size as usize - 1) * conv_channels
    }

    /// Total key dimension = num_key_heads × key_head_dim.
    pub fn key_dim(&self) -> u32 {
        self.num_key_heads * self.key_head_dim
    }

    /// Total QKV projection output dimension (for the joint projection).
    /// = key_dim * 2 + inner_dim (Q + K + V).
    pub fn qkv_dim(&self) -> u32 {
        self.key_dim() * 2 + self.inner_dim
    }
}

/// Configuration for the buffer pool / tier hierarchy.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BufferPoolConfig {
    /// T1 (VRAM) budget in bytes. 0 = auto-detect.
    pub t1_capacity: usize,
    /// T2 (RAM) budget in bytes. 0 = auto-detect.
    pub t2_capacity: usize,
    /// NVMe paths for T3 storage (can stripe across multiple drives).
    pub nvme_paths: Vec<String>,

    /// CUDA device index.
    pub cuda_device: i32,

    /// Start evicting when tier reaches this utilization.
    pub eviction_low_watermark: f32,
    /// Evict aggressively above this utilization.
    pub eviction_high_watermark: f32,

    /// Maximum concurrent prefetch transfers.
    pub max_inflight_transfers: usize,
    /// Prefetch queue depth.
    pub prefetch_queue_depth: usize,

    /// Whether T2 stores compressed pages (Pipeline B mode).
    ///
    /// When true, T2 (RAM) stores Zstd-compressed pages directly from disk.
    /// The T2→T1 promotion path sends compressed data over PCIe to a VRAM
    /// staging buffer, where the Blackwell Decompression Engine (600 GB/s)
    /// decompresses into the final T1 slot.
    ///
    /// This dramatically increases T2 effective capacity:
    /// - 168 GB raw RAM × 3.5x Zstd compression = ~588 GB effective
    /// - Combined T1 (74 GB) + T2 (588 GB) = 662 GB > ~570 GB model
    /// - The entire model fits in RAM+VRAM; NVMe drops out of steady-state
    pub t2_compressed: bool,

    /// Expected compression ratio for T2 compressed mode.
    /// Used for capacity planning. Default: 3.5 (Zstd on INT4 weight data).
    pub t2_compression_ratio: f32,

    /// Size of the VRAM staging buffer for compressed→decompressed transfers (bytes).
    /// This buffer receives compressed data from T2 via PCIe DMA, then the
    /// Blackwell DE decompresses into the final T1 slot.
    /// Default: 32 MB (enough for ~16 compressed pages in flight).
    pub vram_staging_size: usize,
}

impl Default for BufferPoolConfig {
    fn default() -> Self {
        Self {
            t1_capacity: 0,
            t2_capacity: 0,
            nvme_paths: vec![],
            cuda_device: 0,
            eviction_low_watermark: 0.80,
            eviction_high_watermark: 0.95,
            max_inflight_transfers: 8,
            prefetch_queue_depth: 64,
            t2_compressed: true, // Pipeline B is the default
            t2_compression_ratio: 3.5,
            vram_staging_size: 32 * 1024 * 1024, // 32 MB
        }
    }
}

/// Configuration for the activation mode detector.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ActivationModeConfig {
    /// Enable adaptive mode detection.
    pub enabled: bool,

    /// Sliding window size (in tokens) for entropy computation.
    pub window_size: usize,

    /// Entropy threshold (bits). Below → Specialist, above → Generalist.
    /// 0 = auto-compute as 70% of max entropy (log2(num_experts)).
    pub entropy_threshold: f32,

    /// EMA smoothing factor (0.0 = no smoothing, 1.0 = ignore new data).
    pub ema_alpha: f32,

    /// Hysteresis: consecutive opposite-mode detections before switching.
    pub hysteresis: u32,

    /// How often to recompute mode (every N tokens).
    pub detect_interval: usize,

    /// Maximum number of experts to pin in specialist mode.
    /// 0 = auto-compute based on VRAM budget.
    pub max_pinned_experts: usize,
}

impl Default for ActivationModeConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            window_size: 128,
            entropy_threshold: 0.0, // auto-compute
            ema_alpha: 0.9,
            hysteresis: 8,
            detect_interval: 16,
            max_pinned_experts: 0, // auto
        }
    }
}

/// Configuration for the tiered KV cache (Section 8).
///
/// The tiered KV cache stores K and V vectors across the same three-tier
/// hierarchy as weight pages (T1 VRAM → T2 RAM → T3 NVMe), managed by
/// the unified page buffer manager. This enables:
/// - Global eviction across weights and KV (one pool, one eviction policy)
/// - Sparse attention via ANN index over K vectors in T2
/// - Context lengths beyond VRAM capacity (offload cold KV to RAM/NVMe)
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct KvCacheConfig {
    /// Enable the tiered KV cache (vs the flat in-memory KvCacheSet).
    pub enabled: bool,

    /// Maximum positions to keep in T1 (VRAM) per KV head.
    /// These are the "hot" positions — recent tokens plus landmarks.
    /// Default: 4096 (sliding window).
    pub t1_positions: usize,

    /// Maximum positions to keep in T2 (RAM) per KV head.
    /// These are indexed by an ANN structure over K vectors.
    /// Default: 65536 (most of context).
    pub t2_positions: usize,

    /// Whether to use ANN-indexed sparse attention for T2 KV retrieval.
    /// When false, T2 KV is only accessed via full scan (slower but exact).
    pub sparse_attention: bool,

    /// Number of K vectors to retrieve via ANN search per query head.
    /// Only used when `sparse_attention = true`.
    /// Default: 256 (out of potentially 262K context positions).
    pub top_k_positions: usize,

    /// Number of recent positions to always include in attention (sliding window).
    /// These positions bypass ANN search and are always attended to.
    /// Default: 512.
    pub recent_window: usize,

    /// Number of "landmark" positions to pin in T1.
    /// Landmarks are positions with historically high aggregate attention weight.
    /// Default: 64.
    pub landmark_count: usize,

    /// Whether to share the same buffer pool as weight pages (unified eviction).
    /// When true, KV pages compete with weight pages for T1/T2 slots.
    /// When false, KV has a separate memory budget.
    pub unified_pool: bool,

    /// Fraction of T1 budget allocated to KV cache (when unified_pool = true).
    /// The remaining fraction is for weight pages.
    /// Default: 0.15 (15% of T1 for KV, 85% for weights).
    pub t1_kv_fraction: f32,

    /// Fraction of T2 budget allocated to KV cache (when unified_pool = true).
    /// Default: 0.10 (10% of T2 for KV, 90% for weights).
    pub t2_kv_fraction: f32,
}

impl Default for KvCacheConfig {
    fn default() -> Self {
        Self {
            enabled: false, // Opt-in: existing flat KV cache is the default
            t1_positions: 4096,
            t2_positions: 65536,
            sparse_attention: true,
            top_k_positions: 256,
            recent_window: 512,
            landmark_count: 64,
            unified_pool: true,
            t1_kv_fraction: 0.15,
            t2_kv_fraction: 0.10,
        }
    }
}

/// Configuration for gear-based task context integration (Clank Gearbox).
///
/// Controls how vib3 responds to external task signals. When a `TaskContext`
/// is provided with an inference request, these settings govern how
/// aggressively the engine uses the gear signal for cache management,
/// mode detection, and page retrieval filtering.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GearConfig {
    /// Enable gear-based task context processing.
    /// When false, task_context is accepted but ignored.
    pub enabled: bool,

    /// Default alpha for gear bias strength (0.0 = ignore, 1.0 = full).
    /// Can be overridden per-request via `TaskContext.alpha`.
    pub default_alpha: f32,

    /// Whether to override entropy-based mode detection when a gear signal
    /// is present. When true, the gear's expected mode (specialist/generalist)
    /// takes effect immediately without waiting for entropy warmup.
    pub override_mode_detection: bool,

    /// Whether to proactively warm the cache when a gear signal arrives.
    /// Requires specialist profiles to be loaded (Phase B).
    pub proactive_cache_warming: bool,

    /// Whether to filter HNSW search by gear domain tags (Phase C).
    pub filtered_hnsw: bool,

    /// Whether to tag cached pages with gear context for eviction (Phase D).
    pub gear_aware_eviction: bool,

    /// Gear-to-domain mapping (Phase C).
    /// Maps gear names to lists of domain tags for HNSW filtering.
    /// Example: { "code": ["code", "math", "logic"] }
    #[serde(default)]
    pub gear_domains: std::collections::HashMap<String, Vec<String>>,
}

impl Default for GearConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            default_alpha: 0.01, // Conservative — ~35% expert change at trained alphas
            override_mode_detection: true,
            proactive_cache_warming: false, // Phase B — not yet implemented
            filtered_hnsw: false,           // Phase C — not yet implemented
            gear_aware_eviction: false,     // Phase D — not yet implemented
            gear_domains: std::collections::HashMap::new(),
        }
    }
}

/// Top-level engine configuration.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EngineConfig {
    /// Path to the .vib3 model file.
    pub model_path: String,

    /// Path to the tokenizer.json file (or a directory containing it).
    /// If empty, the engine tries to find it adjacent to the model file,
    /// then falls back to the byte-level SimpleTokenizer.
    #[serde(default)]
    pub tokenizer_path: String,

    /// Buffer pool configuration.
    #[serde(default)]
    pub buffer_pool: BufferPoolConfig,

    /// Activation mode detection configuration.
    #[serde(default)]
    pub activation_mode: ActivationModeConfig,

    /// Tiered KV cache configuration (Section 8).
    #[serde(default)]
    pub kv_cache: KvCacheConfig,

    /// Gear-based task context configuration (Clank Gearbox integration).
    #[serde(default)]
    pub gear: GearConfig,

    /// Maximum sequence length.
    pub max_seq_len: usize,

    /// How many layers to prefetch ahead during generation.
    pub lookahead_layers: usize,

    /// How many tokens to speculatively prefetch.
    pub speculative_tokens: usize,

    /// Minimum confidence to issue a prefetch.
    pub prefetch_confidence_threshold: f32,

    /// Page loading mode.
    pub page_mode: PageMode,

    /// API server port (0 = disabled).
    pub api_port: u16,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq)]
pub enum PageMode {
    /// Load all pages for active experts (exact computation).
    Exact,
    /// Load predicted pages (may skip low-activation pages).
    Predictive,
    /// Load top-K pages only (structured sparsity, approximate).
    Approximate,
}

impl Default for EngineConfig {
    fn default() -> Self {
        Self {
            model_path: String::new(),
            tokenizer_path: String::new(),
            buffer_pool: BufferPoolConfig::default(),
            activation_mode: ActivationModeConfig::default(),
            kv_cache: KvCacheConfig::default(),
            gear: GearConfig::default(),
            max_seq_len: 262_144,
            lookahead_layers: 2,
            speculative_tokens: 4,
            prefetch_confidence_threshold: 0.1,
            page_mode: PageMode::Exact,
            api_port: 8080,
        }
    }
}

#[cfg(test)]
mod qwen36_parse_tests {
    use super::*;
    use std::io::Write;

    /// Parse the real Qwen3.6-27B HF config.json and verify the fields vib3
    /// lifts match the Qwen team's reference. The test is skipped when the
    /// cached config isn't on disk so CI doesn't hit the network.
    #[test]
    fn parse_qwen36_27b_hf_config() {
        let path = std::path::Path::new("/tmp/qwen36_config.json");
        if !path.exists() {
            // Write a minimal trimmed fixture inline so the test has signal
            // even without network / cache access.
            let fixture = r#"{
                "architectures": ["Qwen3_5ForConditionalGeneration"],
                "model_type": "qwen3_5",
                "text_config": {
                    "model_type": "qwen3_5_text",
                    "hidden_size": 5120,
                    "num_hidden_layers": 64,
                    "num_attention_heads": 24,
                    "num_key_value_heads": 4,
                    "head_dim": 256,
                    "intermediate_size": 17408,
                    "vocab_size": 248320,
                    "max_position_embeddings": 262144,
                    "rms_norm_eps": 1e-6,
                    "partial_rotary_factor": 0.25,
                    "full_attention_interval": 4,
                    "linear_num_key_heads": 16,
                    "linear_num_value_heads": 48,
                    "linear_key_head_dim": 128,
                    "linear_value_head_dim": 128,
                    "linear_conv_kernel_dim": 4,
                    "rope_parameters": {
                        "rope_theta": 10000000,
                        "rope_type": "default",
                        "mrope_section": [11, 11, 10],
                        "mrope_interleaved": true,
                        "partial_rotary_factor": 0.25
                    }
                }
            }"#;
            let mut f = std::fs::File::create(path).unwrap();
            f.write_all(fixture.as_bytes()).unwrap();
        }

        let cfg = ModelConfig::from_hf_config_path(path)
            .expect("Qwen3.6-27B config should parse");

        // Core dims
        assert_eq!(cfg.architecture, "qwen3_6_dense");
        assert_eq!(cfg.hidden_dim, 5120);
        assert_eq!(cfg.num_layers, 64);
        assert_eq!(cfg.num_heads, 24);
        assert_eq!(cfg.num_kv_heads, 4);
        assert_eq!(cfg.attn_head_dim, 256);
        assert_eq!(cfg.vocab_size, 248_320);

        // Dense classification: zero experts, every layer is dense FFN
        assert_eq!(cfg.num_experts, 0, "Qwen3.6-27B must be dense");
        assert_eq!(cfg.num_active_experts, 0);
        assert_eq!(cfg.num_moe_layers, 0);
        assert_eq!(cfg.dense_layer_idx, cfg.num_layers);
        assert_eq!(cfg.dense_intermediate_size, 17408);

        // RoPE
        assert_eq!(cfg.rope_theta, 10_000_000.0);

        // M-RoPE metadata surfaced
        assert_eq!(cfg.mrope_sections.as_deref(), Some(&[11u32, 11, 10][..]));

        // DeltaNet: 3:1 pattern
        let dn = cfg.deltanet.as_ref().expect("qwen3.6-27b carries DeltaNet config");
        assert_eq!(dn.num_key_heads, 16);
        assert_eq!(dn.num_value_heads, 48);
        assert_eq!(dn.full_attention_interval, 4);
        assert_eq!(dn.layer_is_attention.len(), 64);
        // layer 3, 7, 11, ..., 63 are full attention.
        for i in 0..64 {
            let expect_attn = (i + 1) % 4 == 0;
            assert_eq!(
                dn.layer_is_attention[i], expect_attn,
                "layer {i} should be {}",
                if expect_attn { "full" } else { "deltanet" }
            );
        }
    }
}
