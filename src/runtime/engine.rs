//! The vib3 inference engine — top-level orchestrator.

use crate::compute::cuda_ffi::{self, CudaStream, DeviceBuffer};
use crate::compute::kernels;
use crate::core::config::{EngineConfig, ModelConfig};
use crate::core::error::{Error, Result};
use crate::core::types::*;
use crate::index::domain::ActivationModeDetector;
use crate::index::vector_index::VectorIndex;
use crate::runtime::attention::{
    mla_attention_layer, rms_norm_f32, rms_norm_f32_with_weight,
    self_attention_layer, self_attention_projected, self_attention_tiered,
    KvCacheSet, MlaKvCacheSet, MlaWeights,
};
use crate::runtime::generate::{Sampler, SamplingParams, TokenizerWrapper};
use crate::runtime::query_planner::QueryPlanner;
use crate::runtime::tiered_kv::{TieredKvCache, UnifiedEvictionPolicy};
use crate::storage::buffer_manager::PageBufferManager;
use crate::storage::format::Vib3File;
use half::f16;
use std::sync::atomic::Ordering;
use std::sync::Arc;
use std::time::Instant;

/// Generation result.
#[derive(Debug)]
pub struct GenerateResult {
    pub text: String,
    pub tokens_generated: usize,
    pub prompt_tokens: usize,
    pub tokens_per_second: f64,
    pub time_to_first_token_ms: f64,
    pub total_time_ms: f64,
    pub stats: StatsSnapshot,
}

/// Pre-assembled weight storage — either a borrowed T1 page pointer (single-page)
/// or an owned contiguous DeviceBuffer (multi-page, assembled once at init).
enum PreAssembledWeight {
    /// Single-page tensor: raw pointer into T1 page slot. Zero-copy, zero extra VRAM.
    /// The T1 page is pinned for the lifetime of inference (model is fully resident).
    SinglePage { ptr: *const u8, size: usize },
    /// Multi-page tensor: contiguous DeviceBuffer assembled via D2D from T1 pages.
    /// Allocated once at startup, never freed during inference.
    Assembled(DeviceBuffer),
    /// NVFP4-converted weight: [FP4 data (M*K/2) | BF16 scales (M*(K/32)*2)] in one buffer.
    /// Converted from FP16 at preassembly time for ~4x bandwidth reduction.
    /// Uses MMA GEMV instead of scalar FP16 GEMV.
    Nvfp4 { buf: DeviceBuffer, m: usize, k: usize },
}

impl PreAssembledWeight {
    fn ptr(&self) -> *const u8 {
        match self {
            PreAssembledWeight::SinglePage { ptr, .. } => *ptr,
            PreAssembledWeight::Assembled(buf) => buf.as_ptr(),
            PreAssembledWeight::Nvfp4 { buf, .. } => buf.as_ptr(),
        }
    }

    fn size(&self) -> usize {
        match self {
            PreAssembledWeight::SinglePage { size, .. } => *size,
            PreAssembledWeight::Assembled(buf) => buf.size(),
            PreAssembledWeight::Nvfp4 { buf, .. } => buf.size(),
        }
    }

    fn is_nvfp4(&self) -> bool {
        matches!(self, PreAssembledWeight::Nvfp4 { .. })
    }

    /// For NVFP4 weights, return (m, k) dimensions.
    fn nvfp4_dims(&self) -> Option<(usize, usize)> {
        match self {
            PreAssembledWeight::Nvfp4 { m, k, .. } => Some((*m, *k)),
            _ => None,
        }
    }
}

/// The vib3 inference engine.
pub struct Engine {
    config: EngineConfig,
    #[allow(dead_code)]
    model_file: Arc<Vib3File>,
    buffer_mgr: Arc<PageBufferManager>,
    vector_index: Option<Arc<VectorIndex>>,
    planner: QueryPlanner,
    model_config: ModelConfig,
    stats: Arc<InferenceStats>,
    tokenizer: TokenizerWrapper,
    sampler: Sampler,
    stream: CudaStream,
    /// Separate stream for router GEMV to avoid flushing the main compute pipeline
    router_stream: CudaStream,
    /// Event for synchronizing router stream with compute stream
    router_event: cuda_ffi::CudaEvent,

    // ── GPU working buffers (DeviceBuffer: VRAM when CUDA is active, host otherwise) ──
    /// Hidden state buffer (FP16, [hidden_dim]) — on device.
    /// Working copy: populated from hidden_state_f32 at each sublayer entry.
    hidden_state: DeviceBuffer,
    /// FP32 hidden state accumulator ([hidden_dim] × 4 bytes = 28672 bytes).
    /// The authoritative hidden state across layers. Eliminates FP16 rounding
    /// in residual connections: each sublayer does hidden_state_f32 += f32(output_f16)
    /// instead of the old hidden_state_f16 = residual_f16 + output_f16.
    hidden_state_f32: DeviceBuffer,
    /// Expert output scratch buffer (FP16, [expert_hidden_dim]) — on device.
    expert_output_buf: DeviceBuffer,
    /// Expert output scratch buffer (FP32, [expert_hidden_dim]) — on device.
    /// Used for NVFP4 routed experts to eliminate FP16 truncation of SwiGLU intermediate.
    expert_output_f32_buf: DeviceBuffer,
    /// Accumulated layer output buffer (FP32, [hidden_dim]) — on device.
    /// FP32 to eliminate truncation errors when accumulating 8+ expert outputs.
    layer_output_buf: DeviceBuffer,
    /// Residual connection buffer (FP16, [hidden_dim]) — on device.
    /// Holds the pre-norm hidden_state for residual add.
    residual_buf: DeviceBuffer,
    /// Down-projection scratch buffer (FP16, [hidden_dim]) — on device.
    /// Pre-allocated to avoid per-call cudaMalloc in execute_expert.
    down_proj_buf: DeviceBuffer,
    /// Down-projection scratch buffer (FP32, [hidden_dim]) — on device.
    /// Used for NVFP4 routed experts to eliminate FP16 truncation of down_proj output.
    down_proj_f32_buf: DeviceBuffer,
    /// FP32 normalized hidden state for MoE sublayer (FP32, [hidden_dim]).
    /// Used as input to FP32-input expert matmuls and FP32-input router.
    moe_normed_f32: DeviceBuffer,

    // ── Pre-quantized activation buffers for MMA (VRAM) ──
    // Pre-quantized FP4 E2M1 + E8M0 scales for the MoE normalized hidden state.
    // Computed once per MoE layer before the expert loop, reused across all expert SwiGLU calls.
    /// Pre-quantized activation FP4 data ([hidden_dim/2] bytes, split-half format).
    preq_act_fp4: DeviceBuffer,
    /// Pre-quantized activation E8M0 scales ([hidden_dim/32] bytes).
    preq_act_scales: DeviceBuffer,
    /// Pre-quantized activation FP4 data for down_proj ([expert_hidden_dim/2]).
    preq_down_act_fp4: DeviceBuffer,
    /// Pre-quantized activation E8M0 scales for down_proj ([expert_hidden_dim/32]).
    preq_down_act_scales: DeviceBuffer,
    /// Pre-quantized activation FP4 data for inner_dim projections ([inner_dim/2]).
    /// Used when DeltaNet/GQA output projection weights are NVFP4.
    preq_inner_act_fp4: DeviceBuffer,
    /// Pre-quantized activation E8M0 scales for inner_dim ([inner_dim/32]).
    preq_inner_act_scales: DeviceBuffer,
    /// FP32 normalized hidden state for attention sublayer.
    /// Used as source for FP4 activation quantization when NVFP4 weights are used.
    attn_normed_f32: DeviceBuffer,
    /// FP32 scratch buffer for NVFP4 GEMV output. Sized to max(q_dim, qkv_dim, inner_dim, attn_dim) * 4.
    /// Used for projections where output is FP32 but downstream expects FP16.
    nvfp4_f32_scratch: DeviceBuffer,

    // ── Pre-allocated attention projection scratch buffers (VRAM) ──
    // These eliminate 4x cudaMalloc + cudaFree per layer (128 calls/token).
    /// Q projection scratch (FP16). For gated attention (Qwen3.5): [num_heads * 2 * head_dim] = [16384].
    /// For standard GQA: [hidden_dim].
    q_proj_dev: DeviceBuffer,
    /// K projection scratch (FP16, [kv_dim] = [num_kv_heads * head_dim]).
    k_proj_dev: DeviceBuffer,
    /// V projection scratch (FP16, [kv_dim]).
    v_proj_dev: DeviceBuffer,
    /// Attention output scratch for O projection input (FP16, [num_heads * head_dim]).
    /// For Qwen3.5: [8192] (differs from hidden_dim=3072).
    attn_dev: DeviceBuffer,
    /// Deinterleaved gate for gated attention (FP16, [num_heads * head_dim]).
    /// Only used by Qwen3.5. Holds sigmoid gate extracted from doubled Q projection.
    gated_attn_gate_dev: DeviceBuffer,

    // ── Pre-allocated SwiGLU temp buffers for INT4 decomposition (VRAM) ──
    // INT4 SwiGLU decomposes into 2 matmuls + silu_mul, needing temp space.
    // Sized to expert_hidden_dim (covers any per-page m_slice).
    /// SwiGLU up-projection temp (FP16, [expert_hidden_dim]).
    swiglu_up_tmp: DeviceBuffer,
    /// SwiGLU gate-projection temp (FP16, [expert_hidden_dim]).
    swiglu_gate_tmp: DeviceBuffer,

    // ── Pre-allocated router scores buffer (VRAM) ──
    /// Router output scores (f32, [num_experts]).
    router_scores_dev: DeviceBuffer,
    /// GPU top-k output: expert IDs (u16, [top_k]).
    router_topk_ids_dev: DeviceBuffer,
    /// GPU top-k output: routing weights (f32, [top_k]).
    router_topk_weights_dev: DeviceBuffer,

    /// Pre-allocated logits device buffer (FP32, [vocab_size]).
    /// Used by `compute_logits` to avoid per-token device_alloc/free.
    logits_dev: DeviceBuffer,

    /// NVFP4-converted lm_head weight, lazily populated on first token.
    /// Layout: [FP4 data (M*K/2) | BF16 scales (M*(K/32)*2)] where M=vocab_size, K=hidden_dim.
    lm_head_nvfp4: Option<DeviceBuffer>,

    // ── Pre-allocated MLA projection scratch buffers (VRAM) ──
    // Eliminates CPU GEMV for MLA attention projections (q_a, q_b, kv_a, o_proj).
    /// MLA q_compressed (FP16, [q_lora_rank=1536]).
    mla_q_compressed_dev: DeviceBuffer,
    /// MLA q_full (FP16, [num_heads*(nope+rope)=12288]).
    mla_q_full_dev: DeviceBuffer,
    /// MLA kv_a output (FP16, [kv_lora_rank+rope=576]).
    mla_kv_a_dev: DeviceBuffer,
    /// MLA attention output before O projection (FP16, [num_heads*v_head_dim=8192]).
    /// No longer used after F32 o_proj path change — V_out goes directly to o_proj.
    #[allow(dead_code)]
    mla_attn_out_dev: DeviceBuffer,
    /// MLA O projection output (FP32, [hidden_dim=7168]). FP32 to avoid truncation.
    mla_o_out_dev: DeviceBuffer,

    // ── Pre-allocated dense FFN scratch buffers (VRAM) ──
    // Eliminates CPU GEMV for dense layer 0 FFN.
    /// Dense FFN intermediate (FP16, [dense_intermediate_size=18432]).
    dense_ffn_gate_dev: DeviceBuffer,
    /// Dense FFN up-proj output (FP16, [dense_intermediate_size=18432]).
    #[allow(dead_code)]
    dense_ffn_up_dev: DeviceBuffer,
    /// Dense FFN down-proj output (FP16, [hidden_dim=7168]).
    dense_ffn_down_dev: DeviceBuffer,

    // ── Pre-allocated shared expert scratch buffers (VRAM) ──
    // Eliminates per-layer cudaMalloc/cudaFree in execute_shared_expert.
    /// Shared expert intermediate (FP16, [shared_intermediate_size]).
    shared_expert_inter_dev: DeviceBuffer,
    /// Shared expert down-proj output (FP16, [hidden_dim]).
    shared_expert_down_dev: DeviceBuffer,

    // ── Host staging buffers (for D2H transfers before CPU attention) ──
    /// Host staging buffer 1 — used to hold pre-norm hidden_state on host
    /// for residual connection around CPU attention.
    host_staging: Vec<u8>,
    /// Host staging buffer 2 — used to hold post-norm hidden_state on host
    /// for CPU attention input.
    host_staging2: Vec<u8>,

    /// Whether the engine is using real GPU compute (vs CPU fallback).
    #[allow(dead_code)]
    gpu_mode: bool,

    // ── GPU-resident KV cache for decode attention ──
    // Keeps K/V in VRAM, eliminating the GPU→CPU→GPU round-trip per layer.
    // Layout per layer: [num_kv_heads * max_kv_len * head_dim] in FP16.
    // Only used when CUDA feature is active; #[allow(dead_code)] for CPU-only builds.
    #[allow(dead_code)]
    gpu_kv_k: Vec<DeviceBuffer>,
    #[allow(dead_code)]
    gpu_kv_v: Vec<DeviceBuffer>,
    #[allow(dead_code)]
    max_kv_len: usize,

    // ── DeltaNet (Gated Delta Rule) state for Qwen3.5 hybrid models ──
    // Per-layer recurrent state S ∈ R^{num_v_heads × v_head_dim × v_head_dim} (FP32).
    // Only allocated for DeltaNet layers (36 of 48 for Qwen3.5-122B).
    dn_recurrent_state: Vec<DeviceBuffer>,
    // Per-layer conv1d state: [(kernel_size-1) × qkv_dim] FP32.
    dn_conv_state: Vec<DeviceBuffer>,
    // Scratch buffers for DeltaNet computation (allocated once, reused per layer):
    /// QKV projection output (FP16, [qkv_dim=12288]).
    dn_qkv_proj_dev: DeviceBuffer,
    /// QKV projection converted to FP32 (FP32, [qkv_dim=12288]).
    dn_qkv_f32_dev: DeviceBuffer,
    /// Conv1d output (FP32, [qkv_dim=12288]).
    dn_conv_out_dev: DeviceBuffer,
    /// Z gate projection output (FP16, [inner_dim=8192]).
    dn_z_proj_dev: DeviceBuffer,
    /// Z gate converted to FP32 (FP32, [inner_dim=8192]).
    dn_z_f32_dev: DeviceBuffer,
    /// Alpha projection output (FP16, [num_v_heads=64]).
    dn_alpha_proj_dev: DeviceBuffer,
    /// Alpha F32 (FP32, [num_v_heads=64]).
    dn_alpha_f32_dev: DeviceBuffer,
    /// Beta projection output (FP16, [num_v_heads=64]).
    dn_beta_proj_dev: DeviceBuffer,
    /// Beta F32 after sigmoid (FP32, [num_v_heads=64]).
    dn_beta_f32_dev: DeviceBuffer,
    /// Gate output (FP32, [num_v_heads=64]).
    dn_gate_dev: DeviceBuffer,
    /// Q after L2 norm (FP32, [num_key_heads × key_head_dim = 2048]).
    dn_q_norm_dev: DeviceBuffer,
    /// K after L2 norm (FP32, [num_key_heads × key_head_dim = 2048]).
    dn_k_norm_dev: DeviceBuffer,
    /// Q after tiled repeat 16→64 heads (FP32, [num_v_heads × key_head_dim = 8192]).
    dn_q_rep_dev: DeviceBuffer,
    /// K after tiled repeat 16→64 heads (FP32, [num_v_heads × key_head_dim = 8192]).
    dn_k_rep_dev: DeviceBuffer,
    /// DeltaNet step output (FP32, [inner_dim=8192]).
    dn_step_out_dev: DeviceBuffer,
    /// Gated RMSNorm output (FP32, [inner_dim=8192]).
    dn_norm_out_dev: DeviceBuffer,
    /// O-projection output (FP32, [hidden_dim=3072]).
    dn_o_out_dev: DeviceBuffer,
    /// Small buffer for per-head scalars: A values (FP32, [num_v_heads]).
    /// Loaded once per layer from segment 36.
    dn_a_f32_dev: DeviceBuffer,
    /// Small buffer for dt_bias (FP32, [num_v_heads]).
    /// Loaded once per layer from segment 35.
    dn_dt_bias_f32_dev: DeviceBuffer,
    /// Small buffer for norm weight (FP32, [value_head_dim=128]).
    /// Loaded once per layer from segment 37.
    dn_norm_weight_f32_dev: DeviceBuffer,

    // ── Pre-allocated DeltaNet weight staging buffers ──
    // Eliminates per-layer cudaMalloc/cudaFree overhead (~30ms/token).
    // One buffer per DeltaNet weight segment, reused across all layers.
    // Populated via D2D copy from T1 pages at the start of each layer.
    /// seg 30: QKV weight [qkv_dim × hidden_dim] (BF16)
    dn_w_qkv: DeviceBuffer,
    /// seg 31: Z gate weight [inner_dim × hidden_dim] (BF16)
    dn_w_z: DeviceBuffer,
    /// seg 32: Beta weight [num_v_heads × hidden_dim] (BF16)
    dn_w_beta: DeviceBuffer,
    /// seg 33: Alpha weight [num_v_heads × hidden_dim] (BF16)
    dn_w_alpha: DeviceBuffer,
    /// seg 34: Conv1d weight [qkv_dim × conv_kernel] (BF16)
    dn_w_conv: DeviceBuffer,
    /// seg 35: dt_bias [num_v_heads] (BF16)
    dn_w_dt_bias: DeviceBuffer,
    /// seg 36: A_log [num_v_heads × key_head_dim] (BF16)
    dn_w_a_log: DeviceBuffer,
    /// seg 37: Norm weight [value_head_dim] (BF16)
    dn_w_norm: DeviceBuffer,
    /// seg 38: Out projection [inner_dim × hidden_dim] (BF16)
    dn_w_out: DeviceBuffer,
    /// Pre-allocated conv weight FP32 buffer (avoids cudaMalloc per layer)
    dn_conv_w_f32_dev: DeviceBuffer,

    // ── Pre-allocated GQA attention weight staging buffers ──
    // Eliminates 12× cudaMalloc(156MB) + cudaFree per token.
    // One buffer per weight type, reused across all GQA layers.
    /// seg 4: Q weight [q_dim × hidden_dim] (FP16) — ~96MB for Qwen3.5
    gqa_w_q: DeviceBuffer,
    /// seg 5: O projection [hidden_dim × attn_dim] (FP16) — ~48MB for Qwen3.5
    gqa_w_o: DeviceBuffer,
    /// seg 12: K weight [kv_dim × hidden_dim] (FP16) — ~6MB
    gqa_w_k: DeviceBuffer,
    /// seg 13: V weight [kv_dim × hidden_dim] (FP16) — ~6MB
    gqa_w_v: DeviceBuffer,
    /// seg 27: q_norm [head_dim] (FP16) — 512B
    gqa_w_qnorm: DeviceBuffer,
    /// seg 28: k_norm [head_dim] (FP16) — 512B
    gqa_w_knorm: DeviceBuffer,
    /// seg 6: attention pre-norm weight [hidden_dim] (FP16) — ~6KB
    attn_norm_w: DeviceBuffer,

    /// KV cache for attention layers (flat, in-memory — used when tiered KV is disabled).
    kv_cache: KvCacheSet,
    /// MLA KV cache (for MLA-based models like Kimi K2.5 / DeepSeek-V3).
    /// Stores compressed latent vectors instead of per-head K/V.
    mla_kv_cache: Option<MlaKvCacheSet>,
    /// Tiered KV cache (Section 8 — used when kv_cache config is enabled).
    tiered_kv: Option<TieredKvCache>,
    /// Unified eviction policy (coordinates weight + KV eviction).
    eviction_policy: Option<UnifiedEvictionPolicy>,
    /// Current position in sequence (for RoPE)
    position: usize,
    /// Background worker handles
    _worker_handles: Vec<tokio::task::JoinHandle<()>>,

    // ── Activation mode detection ────────────────────────────────────
    /// Detects Generalist vs Specialist mode from expert activation entropy.
    mode_detector: Option<ActivationModeDetector>,
    /// Current activation mode (cached from last detection).
    current_mode: ActivationMode,
    /// Experts activated during the current token (collected across layers,
    /// fed to mode_detector.record() after all MoE layers complete).
    token_expert_ids: Vec<u16>,
    /// Decode step counter (for detect_interval gating).
    decode_step: u64,

    /// Cache for assembled shared tensors (host-side).
    /// Key: (layer << 16 | segment). Value: reassembled tensor bytes.
    /// Shared tensors are static during inference, so we load them once.
    shared_tensor_cache: std::collections::HashMap<u32, Vec<u8>>,

    /// Cache for kv_b_proj converted to F32 (avoids per-token F16→F32 conversion).
    /// Key: layer index. Value: kv_b_proj as contiguous F32 array.
    /// Size per layer: num_heads*(nope+v)*kv_lora_rank*4 = 64*256*512*4 = 33.6 MB.
    kv_b_proj_f32_cache: std::collections::HashMap<u16, Vec<f32>>,

    /// GPU-resident kv_b_proj F32 cache for MLA V reconstruction on GPU.
    /// Key: layer index. Value: DeviceBuffer holding kv_b_proj as F32 on VRAM.
    kv_b_proj_f32_device: std::collections::HashMap<u16, DeviceBuffer>,

    /// GPU-resident MLA KV cache: per-layer latent + rope on VRAM.
    /// Each entry: latent DeviceBuffer [max_kv_len * kv_lora_rank] F32,
    ///             rope DeviceBuffer [max_kv_len * qk_rope_dim] F32.
    /// Only populated when MLA + GPU are active.
    mla_gpu_kv_latent: Vec<DeviceBuffer>,
    mla_gpu_kv_rope: Vec<DeviceBuffer>,

    /// MLA GPU scratch buffers for absorbed attention (F32).
    /// q_absorbed: [num_heads * kv_lora_rank] = [64 * 512] = 32768 floats
    mla_q_absorbed_dev: DeviceBuffer,
    /// q_rope_f32: [num_heads * qk_rope_dim] = [64 * 64] = 4096 floats
    mla_q_rope_f32_dev: DeviceBuffer,
    /// v_latent_out: [num_heads * kv_lora_rank] = [64 * 512] = 32768 floats
    mla_v_latent_dev: DeviceBuffer,
    /// v_out_f32: [num_heads * v_head_dim] = [64 * 128] = 8192 floats
    mla_v_out_f32_dev: DeviceBuffer,

    /// Precomputed YaRN RoPE frequencies on GPU: [qk_rope_dim/2] F32.
    /// Computed once at startup, reused for every layer/token.
    mla_rope_freqs_dev: Option<DeviceBuffer>,

    /// GPU-resident kv_norm weights per layer for the fused KV cache append kernel.
    /// Key: layer index. Value: DeviceBuffer holding kv_norm weight as FP16.
    #[allow(dead_code)]
    mla_kv_norm_weight_dev: std::collections::HashMap<u16, DeviceBuffer>,

    /// FP16 scratch buffer for v_out conversion (F32→FP16).
    /// No longer used after F32 o_proj path change, kept for potential future use.
    #[allow(dead_code)]
    mla_v_out_f16_dev: DeviceBuffer,

    /// Cache for shared tensors in device memory (VRAM).
    /// Used when gpu_mode is true — avoids repeated H2D copies for
    /// embeddings, norms, QKV projections, router weights, lm_head, etc.
    shared_tensor_cache_device: std::collections::HashMap<u32, DeviceBuffer>,

    /// Pre-assembled shared weight tensors — eliminates ALL per-token D2D copies.
    ///
    /// Populated once at startup (after model preload). For single-page tensors
    /// (norms, biases), stores the T1 page pointer directly (zero-copy).
    /// For multi-page tensors (Q/K/V/O projections, DeltaNet QKV/Z/Out),
    /// assembles pages into a contiguous DeviceBuffer once and caches permanently.
    ///
    /// Key: `(layer as u32) << 16 | segment as u32`.
    /// Value: `(device_ptr, size_bytes)`.
    pre_assembled_weights: std::collections::HashMap<u32, PreAssembledWeight>,

    // ── Task context (Gearbox integration — Phase A/B) ─────────────
    /// Current task context from Clank's Gearbox or an external classifier.
    /// When Some, drives mode detection override, cache warming, HNSW filtering,
    /// and eviction policy adjustments. When None, falls back to entropy-based
    /// mode detection and unfiltered search.
    task_context: Option<TaskContext>,

    /// Pre-computed gear profiles mapping gear names to specialist profiles.
    /// Each profile contains per-layer hot expert IDs that should be pinned
    /// when that gear is active. Loaded from gear_profiles.json (generated
    /// by Gearbox training analysis).
    gear_profiles: std::collections::HashMap<String, SpecialistProfile>,

    /// Per-layer e_score_correction_bias for sigmoid routing (DeepSeek-V3/Kimi K2.5).
    /// Layout: [num_layers][num_experts] F32. Added to sigmoid scores before top-k
    /// selection but NOT used in final expert weights.
    /// Loaded from external binary file at startup.
    e_score_correction_bias: Option<Vec<Vec<f32>>>,

    /// Enable verbose per-layer diagnostics (MoE norms, SwiGLU intermediates,
    /// VRAM-vs-disk comparisons, MLA dumps, etc.). Set via `VIB3_DIAG=1` env var.
    /// Off by default to avoid overhead in normal inference.
    diag_enabled: bool,

    /// NVFP4-converted shared expert weights for MMA GEMV fast path.
    /// Key: `(layer as u32) << 16 | segment as u32` for segments 14/15/16.
    /// Value: DeviceBuffer containing [FP4 data | BF16 scales] in MMA-ready format.
    /// Populated lazily on first use via `fp16_to_nvfp4_weight()` runtime conversion.
    #[allow(dead_code)]
    shared_expert_nvfp4: std::collections::HashMap<u32, DeviceBuffer>,

    /// FP32 intermediate buffer for shared expert SwiGLU (MMA outputs FP32).
    /// Size: shared_intermediate_size * 4 bytes.
    #[allow(dead_code)]
    shared_expert_swiglu_f32: DeviceBuffer,

    /// FP32 output buffer for shared expert down_proj (MMA outputs FP32).
    /// Size: hidden_dim * 4 bytes.
    #[allow(dead_code)]
    shared_expert_down_f32: DeviceBuffer,

    /// Pre-quantized FP4 buffer for shared expert mid activation (after SwiGLU).
    /// Size: shared_intermediate_size / 2 bytes (FP4 data) + shared_intermediate_size / 32 bytes (E8M0 scales).
    #[allow(dead_code)]
    shared_expert_mid_fp4: DeviceBuffer,
    #[allow(dead_code)]
    shared_expert_mid_scales: DeviceBuffer,

    /// Device-side page pointer table for GPU-only MoE dispatch (zero host sync).
    /// Layout: [num_moe_layers * num_experts * 3] u64 page pointers.
    /// Index: moe_layer * num_experts * 3 + expert_id * 3 + segment (0=up, 1=gate, 2=down).
    /// Built once at init after preload when model is fully resident.
    /// None when model is not fully resident or expert dtype is not NVFP4.
    moe_page_table_dev: Option<DeviceBuffer>,

    /// Device-side position scalar (int32). Written via H2D memcpy before each
    /// decode step. Kernels that need position (RoPE, KV append, decode attention)
    /// read from this pointer, enabling CUDA graph capture with static arguments.
    d_position: DeviceBuffer,

    /// Cached CUDA graph executable for the 48-layer decode forward pass.
    /// Captured on the second decode step (after warmup), replayed on all subsequent steps.
    /// None before capture or if graph is not supported.
    cuda_graph_exec: Option<*mut std::ffi::c_void>,

    /// Raw CUDA graph handle (kept alive for the lifetime of the exec).
    cuda_graph: Option<*mut std::ffi::c_void>,

    /// True while CUDA graph capture is in progress. Used to suppress operations
    /// that are illegal during capture (e.g. cudaFree from HashMap::retain).
    capturing_graph: bool,
}

// SAFETY: Engine is only accessed through a tokio::sync::Mutex in the API server,
// ensuring single-threaded access. The raw pointers within (via CudaStream, buffer
// manager page pointers) are allocated/freed on the same thread.
unsafe impl Send for Engine {}
unsafe impl Sync for Engine {}

impl Engine {
    fn maybe_audit_nvfp4_weight_pair(
        &self,
        layer: u16,
        segment: u16,
        m: usize,
        k: usize,
        fp16_buf: &DeviceBuffer,
        nvfp4_buf: &DeviceBuffer,
    ) -> Result<()> {
        let enabled = self.diag_enabled
            && std::env::var("VIB3_DIAG_NVFP4_WEIGHT_AUDIT").map_or(false, |v| v == "1");
        if !enabled || layer != 0 || !(segment == 30 || segment == 31 || segment == 38) {
            return Ok(());
        }
        if k == 0 || m == 0 || !k.is_multiple_of(32) {
            return Ok(());
        }

        let fp16_bytes = m * k * 2;
        let packed_k = k / 2;
        let num_groups = k / 32;
        let data_bytes = m * packed_k;
        let scale_bytes = m * num_groups * 2;
        let expected_nvfp4 = data_bytes + scale_bytes;
        if fp16_buf.size() < fp16_bytes || nvfp4_buf.size() < expected_nvfp4 {
            tracing::warn!(
                "NVFP4_WEIGHT_AUDIT skip L{} seg{}: fp16_size={} nvfp4_size={} expected_nvfp4={}",
                layer,
                segment,
                fp16_buf.size(),
                nvfp4_buf.size(),
                expected_nvfp4,
            );
            return Ok(());
        }

        self.stream.synchronize()?;
        let mut fp16_host = vec![0u8; fp16_bytes];
        let mut nvfp4_host = vec![0u8; expected_nvfp4];
        fp16_buf.copy_to_host(&mut fp16_host)?;
        nvfp4_buf.copy_to_host(&mut nvfp4_host)?;

        let fp16_vals =
            unsafe { std::slice::from_raw_parts(fp16_host.as_ptr() as *const f16, m * k) };
        let (data_region, scale_region) = nvfp4_host.split_at(data_bytes);

        const FP4_LUT: [f32; 16] = [
            0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0,
            -6.0,
        ];

        let mut dot = 0.0f64;
        let mut n_ref = 0.0f64;
        let mut n_q = 0.0f64;
        let mut diff2 = 0.0f64;
        let mut abs_sum = 0.0f64;
        let mut max_abs = 0.0f32;
        let mut count = 0usize;

        for row in 0..m {
            let row_data = &data_region[row * packed_k..(row + 1) * packed_k];
            for group in 0..num_groups {
                let sb = &scale_region[(row * num_groups + group) * 2..][..2];
                let scale_bits = u16::from_le_bytes([sb[0], sb[1]]);
                let e8m0 = ((scale_bits >> 7) & 0xFF) as u32;
                let scale = f32::from_bits(e8m0 << 23);
                let gbyte = group * 16;
                let base_col = group * 32;

                for j in 0..16 {
                    let b = row_data[gbyte + j];
                    let n0 = (b & 0x0F) as usize;
                    let n1 = (b >> 4) as usize;

                    let c0 = base_col + j;
                    let c1 = base_col + j + 16;

                    let q0 = FP4_LUT[n0] * scale;
                    let r0 = fp16_vals[row * k + c0].to_f32();
                    let d0 = r0 - q0;
                    dot += (r0 as f64) * (q0 as f64);
                    n_ref += (r0 as f64) * (r0 as f64);
                    n_q += (q0 as f64) * (q0 as f64);
                    diff2 += (d0 as f64) * (d0 as f64);
                    abs_sum += d0.abs() as f64;
                    max_abs = max_abs.max(d0.abs());
                    count += 1;

                    let q1 = FP4_LUT[n1] * scale;
                    let r1 = fp16_vals[row * k + c1].to_f32();
                    let d1 = r1 - q1;
                    dot += (r1 as f64) * (q1 as f64);
                    n_ref += (r1 as f64) * (r1 as f64);
                    n_q += (q1 as f64) * (q1 as f64);
                    diff2 += (d1 as f64) * (d1 as f64);
                    abs_sum += d1.abs() as f64;
                    max_abs = max_abs.max(d1.abs());
                    count += 1;
                }
            }
        }

        if count == 0 {
            return Ok(());
        }

        let cosine = if n_ref > 0.0 && n_q > 0.0 {
            dot / (n_ref.sqrt() * n_q.sqrt())
        } else {
            0.0
        };
        let rel_l2 = if n_ref > 0.0 {
            diff2.sqrt() / n_ref.sqrt()
        } else {
            0.0
        };
        let mean_abs = abs_sum / count as f64;

        tracing::info!(
            "NVFP4_WEIGHT_AUDIT L{} seg{} (M={} K={}): cosine={:.8} rel_l2={:.8} mean_abs={:.8} max_abs={:.6}",
            layer,
            segment,
            m,
            k,
            cosine,
            rel_l2,
            mean_abs,
            max_abs,
        );

        Ok(())
    }

    fn log_projection_compare(
        &self,
        label: &str,
        layer_idx: u16,
        a_ptr: *const u8,
        b_ptr: *const u8,
        len: usize,
    ) -> Result<()> {
        if len == 0 {
            return Ok(());
        }
        self.stream.synchronize()?;
        let mut a_bytes = vec![0u8; len * 4];
        let mut b_bytes = vec![0u8; len * 4];
        crate::compute::cuda_ffi::memcpy_d2h(a_bytes.as_mut_ptr(), a_ptr, a_bytes.len())?;
        crate::compute::cuda_ffi::memcpy_d2h(b_bytes.as_mut_ptr(), b_ptr, b_bytes.len())?;

        let a = unsafe { std::slice::from_raw_parts(a_bytes.as_ptr() as *const f32, len) };
        let b = unsafe { std::slice::from_raw_parts(b_bytes.as_ptr() as *const f32, len) };

        let mut dot = 0.0f64;
        let mut na = 0.0f64;
        let mut nb = 0.0f64;
        let mut diff2 = 0.0f64;
        let mut max_abs = 0.0f32;
        for i in 0..len {
            let av = a[i] as f64;
            let bv = b[i] as f64;
            let d = (av - bv) as f32;
            dot += av * bv;
            na += av * av;
            nb += bv * bv;
            diff2 += (d as f64) * (d as f64);
            max_abs = max_abs.max(d.abs());
        }

        let cosine = if na > 0.0 && nb > 0.0 {
            dot / (na.sqrt() * nb.sqrt())
        } else {
            0.0
        };
        let rel_l2 = if na > 0.0 {
            diff2.sqrt() / na.sqrt()
        } else {
            0.0
        };

        tracing::info!(
            "NVFP4_COMPARE {} L{} pos={}: cosine={:.8}, rel_l2={:.8}, max_abs={:.6}",
            label,
            layer_idx,
            self.position,
            cosine,
            rel_l2,
            max_abs,
        );
        Ok(())
    }

    fn normalize_prompt_for_model(model_arch: &str, prompt: &str) -> String {
        // If the prompt already contains a chat template, preserve it as-is.
        if prompt.contains("<|im_start|>") || prompt.contains("<|im_end|>") {
            return prompt.to_string();
        }
        // Mixtral/Mistral instruct checkpoints typically expect [INST] wrappers.
        if model_arch.eq_ignore_ascii_case("mixtral")
            && !prompt.contains("[INST]")
            && !prompt.contains("[/INST]")
        {
            return format!("[INST] {} [/INST]", prompt.trim());
        }
        prompt.to_string()
    }

    /// Open a .vib3 model file and create an engine.
    pub async fn from_path(path: &str) -> Result<Self> {
        let config = EngineConfig {
            model_path: path.to_string(),
            ..Default::default()
        };
        Self::new(config).await
    }

    /// Create engine with full configuration.
    pub async fn new(config: EngineConfig) -> Result<Self> {
        let mut config = config;

        tracing::info!("Opening model: {}", config.model_path);
        let model_file = Arc::new(Vib3File::open(&config.model_path)?);
        let model_config = model_file.model_config().clone();

        tracing::info!(
            "Model: {} -- {} experts x {} layers, {} pages",
            model_config.name,
            model_config.num_experts,
            model_config.num_moe_layers,
            model_file.page_count(),
        );

        // Auto-configure buffer pool for CPU-only mode.
        //
        // When no real GPU is present and t1_capacity is 0, we need to
        // allocate system RAM as "T1" (since device_alloc falls back to
        // host allocation anyway). Without this, the buffer manager would
        // have zero T1 slots and all page loads would fail.
        //
        // Strategy: probe the CUDA device first. If it's a CPU fallback,
        // auto-split available RAM between T1 and T2. On a real GPU,
        // leave the defaults (auto-detect from CUDA free memory).
        let device = crate::compute::cuda_ffi::CudaDevice::new(config.buffer_pool.cuda_device)?;
        if !device.is_real_cuda() && config.buffer_pool.t1_capacity == 0 {
            // CPU-only mode: auto-configure from system RAM.
            // Model size estimate from page catalog.
            let model_pages = model_file.page_count();
            let model_bytes = model_pages * PAGE_SIZE;

            // Use at most 80% of detected system RAM, but never more than
            // 2x the model size (no point allocating 150 GB for a 50 MB model).
            let available_ram = (device.free_mem() as f64 * 0.8) as usize;
            let reasonable_budget = (model_bytes * 2).max(256 * 1024 * 1024); // At least 256 MB
            let budget = available_ram.min(reasonable_budget);

            // T1 gets enough to hold the whole model (or the full budget if
            // the model is small). T2 gets a matching amount for spillover.
            // In CPU mode, T1/T2 distinction is mostly about the eviction
            // policy — T1 pages are the "hot" set, T2 is the cold spillover.
            let t1_budget = model_bytes.min(budget);
            let t2_budget = budget.saturating_sub(t1_budget).max(PAGE_SIZE * 16);

            // Disable compressed T2 in CPU-only mode — there's no Blackwell
            // Decompression Engine, so compressed T2 would require CPU-side
            // Zstd decompression on every T2→T1 promotion (expensive).
            config.buffer_pool.t2_compressed = false;

            config.buffer_pool.t1_capacity = t1_budget;
            config.buffer_pool.t2_capacity = t2_budget;

            tracing::info!(
                "CPU-only mode: auto-configured T1={} MB, T2={} MB (model={} MB, {} pages)",
                t1_budget / (1024 * 1024),
                t2_budget / (1024 * 1024),
                model_bytes / (1024 * 1024),
                model_pages,
            );
        } else if device.is_real_cuda() && config.buffer_pool.t1_capacity == 0 {
            // Real GPU detected — T1 uses VRAM (cudaMalloc) for weight pages.
            // Working buffers (hidden_state, etc.) are also in VRAM, so all
            // pointers passed to GPU kernels are device pointers.
            //
            // Reserve ~2 GB of VRAM for working buffers, CUDA context, and
            // shared tensor caches. The rest goes to T1 for weight pages.
            let model_pages = model_file.page_count();
            let model_bytes = model_pages * PAGE_SIZE;
            let vram_free = device.free_mem();

            // Reserve VRAM for non-T1 uses:
            // - GPU KV cache: num_layers * num_kv_heads * max_kv_len * head_dim * 2 (FP16) * 2 (K+V)
            //   For Kimi K2.5: 61 * 128 * 4096 * 56 * 2 * 2 = ~6.8 GB
            // - Working buffers: ~5 * hidden_bytes + expert_bytes ≈ 100 KB for Mixtral
            // - Shared tensor cache (embeddings, norms, QKV, lm_head): ~2 GB for large models
            // - CUDA context/overhead: ~500 MB
            let hidden_dim_est = model_config.hidden_dim as usize;
            let num_heads_est = model_config.num_heads as usize;
            let num_kv_heads_est = model_config.num_kv_heads as usize;
            let head_dim_est = if num_heads_est > 0 { hidden_dim_est / num_heads_est } else { 128 };
            let max_kv_est: usize = 4096;
            let kv_cache_bytes = model_config.num_layers as usize
                * num_kv_heads_est * max_kv_est * head_dim_est
                * std::mem::size_of::<f16>() * 2; // K + V
            let shared_tensor_reserve = model_config.estimated_shared_bytes().min(4 * 1024 * 1024 * 1024); // cap at 4 GB
            let vram_reserve = kv_cache_bytes + shared_tensor_reserve + 512 * 1024 * 1024; // KV + shared + 512 MB overhead
            tracing::info!(
                "VRAM reserve: KV cache={} MB, shared tensors={} MB, overhead=512 MB, total={} MB",
                kv_cache_bytes / (1024 * 1024),
                shared_tensor_reserve / (1024 * 1024),
                vram_reserve / (1024 * 1024),
            );
            let t1_budget = if vram_free > vram_reserve {
                let available = vram_free - vram_reserve;
                // Don't allocate more than the model needs
                available.min(model_bytes)
            } else {
                // Very little VRAM — use what we can
                (vram_free / 2).max(PAGE_SIZE * 16)
            };

            // T2 uses system RAM (pinned host memory)
            let system_ram = detect_system_ram();
            let t2_budget = if config.buffer_pool.t2_capacity == 0 {
                // Use up to 50% of system RAM for T2 spillover, but not
                // more than the model size (pages not fitting in T1).
                let remaining_model = model_bytes.saturating_sub(t1_budget);
                let max_t2 = (system_ram as f64 * 0.5) as usize;
                remaining_model.min(max_t2).max(PAGE_SIZE * 16)
            } else {
                config.buffer_pool.t2_capacity
            };

            // Disable compressed T2 for now (simplicity; enable later for
            // Blackwell decompression engine).
            config.buffer_pool.t2_compressed = false;

            config.buffer_pool.t1_capacity = t1_budget;
            config.buffer_pool.t2_capacity = t2_budget;

            tracing::info!(
                "GPU mode: T1={} MB (VRAM), T2={} MB (pinned host), model={} MB ({} pages), VRAM free={} MB",
                t1_budget / (1024 * 1024),
                t2_budget / (1024 * 1024),
                model_bytes / (1024 * 1024),
                model_pages,
                vram_free / (1024 * 1024),
            );
        }

        // Initialize buffer pool
        let buffer_mgr = Arc::new(PageBufferManager::new(
            config.buffer_pool.clone(),
            model_file.clone(),
        ));
        buffer_mgr.initialize().await?;

        // Preload entire model into T1 when it fits (eliminates I/O stalls
        // during inference — all pages are in VRAM before the first token).
        if buffer_mgr.model_fits_in_t1() {
            match buffer_mgr.preload_all().await {
                Ok(loaded) => {
                    tracing::info!("Model fully resident in T1: {} pages preloaded", loaded);
                }
                Err(e) => {
                    tracing::warn!("Preload failed (will load on-demand): {}", e);
                }
            }
        }

        // Load vector index if available
        let vector_index = if model_file.has_vector_index() {
            match VectorIndex::load(&model_file) {
                Ok(idx) => {
                    tracing::info!("Vector index loaded");
                    Some(Arc::new(idx))
                }
                Err(e) => {
                    tracing::warn!("Failed to load vector index: {e}");
                    None
                }
            }
        } else {
            tracing::info!("No vector index (model converted without profiling)");
            None
        };

        let stats = buffer_mgr.stats.clone();

        let planner = QueryPlanner::new(
            buffer_mgr.clone(),
            vector_index.clone(),
            model_file.clone(),
            model_config.clone(),
        );

        // Create compute stream for kernel dispatch.
        // When a real GPU is available, use a real CUDA stream so that
        // kernel launches (matmul, SwiGLU, RMSNorm, etc.) execute on the GPU.
        // Working buffers are allocated as DeviceBuffers (VRAM when CUDA is
        // active), so all pointers passed to kernels are device pointers.
        let gpu_mode = device.is_real_cuda();
        let stream = if gpu_mode {
            CudaStream::new(&device)?
        } else {
            CudaStream::cpu_only()
        };
        let router_stream = if gpu_mode {
            CudaStream::new(&device)?
        } else {
            CudaStream::cpu_only()
        };
        let router_event = cuda_ffi::CudaEvent::new()?;

        // Allocate working buffers as DeviceBuffers (VRAM on GPU, host on CPU).
        let hidden_dim = model_config.hidden_dim as usize;
        let vocab_size = model_config.vocab_size as usize;
        let expert_hidden = model_config.expert_hidden_dim as usize;
        let hidden_bytes = hidden_dim * std::mem::size_of::<f16>();
        let expert_bytes = expert_hidden * std::mem::size_of::<f16>();

        // Pre-compute sizes for attention projection scratch buffers
        let num_heads = model_config.num_heads as usize;
        let num_kv_heads = model_config.num_kv_heads as usize;
        let head_dim = model_config.effective_head_dim() as usize;
        // Gated attention (Qwen3.5): Q projection outputs 2x (Q + gate interleaved)
        // q_proj_dev must hold num_heads * 2 * head_dim FP16 elements
        let is_gated_attn = model_config.deltanet.is_some();
        let q_proj_elems = if is_gated_attn {
            num_heads * 2 * head_dim // 32 * 2 * 256 = 16384
        } else {
            hidden_dim // standard: hidden_dim = num_heads * head_dim
        };
        let q_proj_bytes = q_proj_elems * 2; // FP16
        let kv_proj_bytes = num_kv_heads * head_dim * 2; // FP16
        // attn_dev holds attention output: num_heads * head_dim (may differ from hidden_dim)
        let attn_out_elems = num_heads * head_dim; // 32 * 256 = 8192 for Qwen3.5
        let attn_out_bytes = attn_out_elems * 2; // FP16
        // gate_buf for deinterleaved attention gate (Qwen3.5 only)
        let gate_buf_bytes = if is_gated_attn { attn_out_elems * 2 } else { 64 };
        let router_scores_bytes = (model_config.num_experts as usize).max(1) * 4; // f32
        let top_k = model_config.num_active_experts as usize;
        let router_topk_ids_bytes = top_k.max(1) * 2; // u16
        let router_topk_weights_bytes = top_k.max(1) * 4; // f32

        // Pre-compute MLA projection scratch buffer sizes
        let mla_q_compressed_bytes;
        let mla_q_full_bytes;
        let mla_kv_a_bytes;
        let mla_attn_out_bytes;
        let mla_o_out_bytes;
        let mla_q_absorbed_bytes;
        let mla_q_rope_f32_bytes;
        let mla_v_latent_bytes;
        let mla_v_out_f32_bytes;
        let mla_v_out_f16_bytes;
        let mut mla_rope_freqs_dev: Option<DeviceBuffer> = None;
        if let Some(ref mla) = model_config.mla {
            mla_q_compressed_bytes = mla.q_lora_rank as usize * 2;
            mla_q_full_bytes = num_heads * (mla.qk_nope_head_dim as usize + mla.qk_rope_head_dim as usize) * 2;
            mla_kv_a_bytes = (mla.kv_lora_rank as usize + mla.qk_rope_head_dim as usize) * 2;
            mla_attn_out_bytes = num_heads * mla.v_head_dim as usize * 2;
            mla_o_out_bytes = hidden_dim * 4; // FP32 to avoid truncation in o_proj
            // F32 GPU scratch buffers for absorbed MLA attention
            mla_q_absorbed_bytes = num_heads * mla.kv_lora_rank as usize * 4; // F32
            mla_q_rope_f32_bytes = num_heads * mla.qk_rope_head_dim as usize * 4;
            mla_v_latent_bytes = num_heads * mla.kv_lora_rank as usize * 4;
            mla_v_out_f32_bytes = num_heads * mla.v_head_dim as usize * 4;
            mla_v_out_f16_bytes = num_heads * mla.v_head_dim as usize * 2;

            // Precompute YaRN RoPE frequencies and upload to GPU
            let qk_rope_dim = mla.qk_rope_head_dim as usize;
            let yarn = crate::runtime::attention::YarnRopeConfig::from_kimi_k25(qk_rope_dim);
            let half_rope = qk_rope_dim / 2;
            let freqs: Vec<f32> = (0..half_rope).map(|i| yarn.freq(i)).collect();
            if gpu_mode {
                let freq_bytes = half_rope * 4;
                let dev_buf = DeviceBuffer::new(freq_bytes)?;
                let src = unsafe {
                    std::slice::from_raw_parts(freqs.as_ptr() as *const u8, freq_bytes)
                };
                dev_buf.copy_from_host(src)?;
                tracing::info!("MLA RoPE frequencies uploaded to GPU: {} values", half_rope);
                mla_rope_freqs_dev = Some(dev_buf);
            }
        } else {
            mla_q_compressed_bytes = 64;
            mla_q_full_bytes = 64;
            mla_kv_a_bytes = 64;
            mla_attn_out_bytes = 64;
            mla_o_out_bytes = 64;
            mla_q_absorbed_bytes = 64;
            mla_q_rope_f32_bytes = 64;
            mla_v_latent_bytes = 64;
            mla_v_out_f32_bytes = 64;
            mla_v_out_f16_bytes = 64;
        }

        // Pre-compute dense FFN scratch buffer sizes
        let dense_inter = model_config.dense_intermediate_size as usize;
        let dense_ffn_inter_bytes = dense_inter.max(1) * 2; // FP16
        // Shared expert scratch
        let shared_inter = model_config.shared_intermediate_size as usize;
        let shared_expert_inter_bytes = shared_inter.max(1) * 2;

        // Start background prefetch/eviction workers
        let worker_handles = buffer_mgr.start_workers();

        let tokenizer = TokenizerWrapper::load(
            &config.tokenizer_path,
            &config.model_path,
            model_config.vocab_size,
        );
        let sampler = Sampler::new(42);

        // Initialize KV cache (flat, in-memory — always available as fallback)
        let kv_cache = KvCacheSet::new(&model_config);

        // Initialize MLA KV cache if model uses MLA
        let mla_kv_cache = if let Some(ref mla) = model_config.mla {
            tracing::info!(
                "MLA attention enabled: kv_lora_rank={}, q_lora_rank={}, rope_dim={}, nope_dim={}, v_dim={}",
                mla.kv_lora_rank,
                mla.q_lora_rank,
                mla.qk_rope_head_dim,
                mla.qk_nope_head_dim,
                mla.v_head_dim,
            );
            Some(MlaKvCacheSet::new(&model_config, mla))
        } else {
            None
        };

        // Initialize tiered KV cache (Section 8) if enabled
        let head_dim = model_config.effective_head_dim() as usize;
        let (tiered_kv, eviction_policy) = if config.kv_cache.enabled {
            let tkv = TieredKvCache::new(
                model_config.num_layers as usize,
                model_config.num_kv_heads as usize,
                head_dim,
                config.kv_cache.clone(),
            );
            let policy = UnifiedEvictionPolicy::new(
                config.kv_cache.t1_kv_fraction,
                config.kv_cache.t2_kv_fraction,
            );
            tracing::info!(
                "Tiered KV cache enabled: T1={} positions, T2={} positions, sparse={}, top_k={}, landmarks={}",
                config.kv_cache.t1_positions,
                config.kv_cache.t2_positions,
                config.kv_cache.sparse_attention,
                config.kv_cache.top_k_positions,
                config.kv_cache.landmark_count,
            );
            (Some(tkv), Some(policy))
        } else {
            tracing::info!("Using flat in-memory KV cache (tiered KV disabled)");
            (None, None)
        };

        // Initialize activation mode detector
        let mode_detector = if config.activation_mode.enabled {
            let mut detector = ActivationModeDetector::new(
                model_config.num_experts as usize,
                config.activation_mode.window_size,
            );
            // Apply config overrides
            if config.activation_mode.entropy_threshold > 0.0 {
                detector.set_threshold(config.activation_mode.entropy_threshold);
            }
            detector.set_ema_alpha(config.activation_mode.ema_alpha);
            detector.set_hysteresis(config.activation_mode.hysteresis);
            tracing::info!(
                "Activation mode detector enabled (window={}, threshold={:.1}, hysteresis={}, interval={})",
                config.activation_mode.window_size,
                detector.current_entropy(), // will show initial neutral value
                config.activation_mode.hysteresis,
                config.activation_mode.detect_interval,
            );
            Some(detector)
        } else {
            tracing::info!("Activation mode detection disabled");
            None
        };

        // Allocate GPU-resident KV cache for decode attention.
        // For Mixtral: 8 kv_heads * 4096 max_seq * 128 head_dim * 2 bytes = 8 MB per layer
        // 32 layers = 256 MB total — trivially fits in 96 GB VRAM.
        let max_kv_len: usize = 4096;
        let kv_buf_size = num_kv_heads * max_kv_len * head_dim * std::mem::size_of::<f16>();
        let num_layers = model_config.num_layers as usize;
        let use_mla = model_config.mla.is_some();
        let mut gpu_kv_k = Vec::with_capacity(num_layers);
        let mut gpu_kv_v = Vec::with_capacity(num_layers);
        if !use_mla {
            // Only allocate GQA KV cache for non-MLA models.
            // MLA models use compressed latent KV cache (mla_gpu_kv_latent/rope) instead.
            for _ in 0..num_layers {
                gpu_kv_k.push(DeviceBuffer::new(kv_buf_size)?);
                gpu_kv_v.push(DeviceBuffer::new(kv_buf_size)?);
            }
        }

        // Allocate GPU-resident MLA KV cache (latent + rope per layer).
        // For Kimi K2.5: kv_lora_rank=512, rope_dim=64, max_kv_len=4096
        // Per layer: latent = 4096 * 512 * 4 = 8 MB, rope = 4096 * 64 * 4 = 1 MB
        // 61 layers = ~549 MB total — fits easily in 96 GB VRAM.
        let mut mla_gpu_kv_latent = Vec::with_capacity(num_layers);
        let mut mla_gpu_kv_rope = Vec::with_capacity(num_layers);
        if let Some(ref mla) = model_config.mla {
            let latent_buf_size = max_kv_len * mla.kv_lora_rank as usize * 4; // F32
            let rope_buf_size = max_kv_len * mla.qk_rope_head_dim as usize * 4; // F32
            for _ in 0..num_layers {
                mla_gpu_kv_latent.push(DeviceBuffer::new(latent_buf_size)?);
                mla_gpu_kv_rope.push(DeviceBuffer::new(rope_buf_size)?);
            }
            if gpu_mode {
                tracing::info!(
                    "GPU MLA KV cache: {} MB for {} layers (latent={} B, rope={} B per layer)",
                    (latent_buf_size + rope_buf_size) * num_layers / (1024 * 1024),
                    num_layers,
                    latent_buf_size,
                    rope_buf_size,
                );
            }
        }

        if gpu_mode {
            if use_mla {
                tracing::info!(
                    "GPU mode: working buffers in VRAM (hidden={} B, expert={} B, GQA KV cache skipped — MLA active)",
                    hidden_bytes,
                    expert_bytes,
                );
            } else {
                tracing::info!(
                    "GPU mode: working buffers in VRAM (hidden={} B, expert={} B, GPU KV cache={} MB for {} layers)",
                    hidden_bytes,
                    expert_bytes,
                    (kv_buf_size * 2 * num_layers) / (1024 * 1024),
                    num_layers,
                );
            }
        } else {
            tracing::info!("CPU mode: working buffers in host memory");
        }

        // ── Allocate DeltaNet recurrent + conv1d state (per DeltaNet layer) ──
        let mut dn_recurrent_state = Vec::new();
        let mut dn_conv_state = Vec::new();
        if let Some(ref dn) = model_config.deltanet {
            let num_dn_layers = dn.layer_is_attention.iter().filter(|&&is_attn| !is_attn).count();
            let state_bytes = dn.state_size_floats() * 4; // FP32
            let conv_bytes = dn.conv_state_size_floats() * 4; // FP32
            for _ in 0..num_dn_layers {
                let s = DeviceBuffer::new(state_bytes)?;
                s.zero();
                dn_recurrent_state.push(s);
                let c = DeviceBuffer::new(conv_bytes)?;
                c.zero();
                dn_conv_state.push(c);
            }
            tracing::info!(
                "DeltaNet state allocated: {} layers × ({} MB recurrent + {} KB conv) = {} MB total",
                num_dn_layers,
                state_bytes / (1024 * 1024),
                conv_bytes / 1024,
                (state_bytes + conv_bytes) * num_dn_layers / (1024 * 1024),
            );
        }

        // DeltaNet scratch buffer sizes (0 if no DeltaNet)
        let dn_qkv_dim = model_config.deltanet.as_ref().map_or(0, |d| d.qkv_dim() as usize);
        let dn_inner_dim = model_config.deltanet.as_ref().map_or(0, |d| d.inner_dim as usize);
        let dn_num_v_heads = model_config.deltanet.as_ref().map_or(0, |d| d.num_value_heads as usize);
        let dn_key_dim = model_config.deltanet.as_ref().map_or(0, |d| d.key_dim() as usize);
        let dn_v_head_dim = model_config.deltanet.as_ref().map_or(0, |d| d.value_head_dim as usize);
        // Minimum 1 byte for DeviceBuffer (avoid zero-size allocations)
        let dn_min = |sz: usize| if sz == 0 { 1 } else { sz };

        let mut engine = Self {
            config,
            model_file,
            buffer_mgr,
            vector_index,
            planner,
            model_config,
            stats,
            tokenizer,
            sampler,
            stream,
            router_stream,
            router_event,
            hidden_state: DeviceBuffer::new(hidden_bytes)?,
            hidden_state_f32: DeviceBuffer::new(hidden_dim * 4)?, // FP32 accumulator
            expert_output_buf: DeviceBuffer::new(expert_bytes)?,
            expert_output_f32_buf: DeviceBuffer::new(expert_hidden * 4)?, // FP32 SwiGLU intermediate
            layer_output_buf: DeviceBuffer::new(hidden_dim * 4)?, // FP32 for precision
            moe_normed_f32: DeviceBuffer::new(hidden_dim * 4)?, // FP32 normalized hidden state for MoE
            preq_act_fp4: DeviceBuffer::new(hidden_dim / 2)?, // FP4 split-half for hidden_dim
            preq_act_scales: DeviceBuffer::new(hidden_dim / 32)?, // E8M0 scales
            preq_down_act_fp4: DeviceBuffer::new(expert_hidden.max(1) / 2)?, // FP4 for expert_hidden_dim
            preq_down_act_scales: DeviceBuffer::new(expert_hidden.max(1) / 32)?, // E8M0 scales
            preq_inner_act_fp4: DeviceBuffer::new(dn_inner_dim.max(1) / 2)?, // FP4 for inner_dim (8192)
            preq_inner_act_scales: DeviceBuffer::new(dn_inner_dim.max(1) / 32)?, // E8M0 scales
            attn_normed_f32: DeviceBuffer::new(hidden_dim * 4)?, // FP32 normalized state for NVFP4 GEMV
            nvfp4_f32_scratch: DeviceBuffer::new({
                let max_proj_dim = q_proj_elems.max(dn_qkv_dim).max(dn_inner_dim).max(attn_out_elems);
                // Extra space for batched GQA Q+K+V output: q_dim + kv_dim + kv_dim (all FP32)
                let kv_elems = kv_proj_bytes / 2;
                let batched_qkv = q_proj_elems + kv_elems * 2;
                max_proj_dim.max(batched_qkv).max(1) * 4
            })?, // FP32 scratch for NVFP4 GEMV output
            residual_buf: DeviceBuffer::new(hidden_bytes)?,
            down_proj_buf: DeviceBuffer::new(hidden_bytes)?,
            down_proj_f32_buf: DeviceBuffer::new(hidden_dim * 4)?, // FP32 down_proj output
            q_proj_dev: DeviceBuffer::new(q_proj_bytes)?,
            k_proj_dev: DeviceBuffer::new(kv_proj_bytes)?,
            v_proj_dev: DeviceBuffer::new(kv_proj_bytes)?,
            attn_dev: DeviceBuffer::new(attn_out_bytes)?,
            gated_attn_gate_dev: DeviceBuffer::new(gate_buf_bytes)?,
            swiglu_up_tmp: DeviceBuffer::new(expert_bytes)?,
            swiglu_gate_tmp: DeviceBuffer::new(expert_bytes)?,
            router_scores_dev: DeviceBuffer::new(router_scores_bytes)?,
            router_topk_ids_dev: DeviceBuffer::new(router_topk_ids_bytes)?,
            router_topk_weights_dev: DeviceBuffer::new(router_topk_weights_bytes)?,
            logits_dev: DeviceBuffer::new(vocab_size * std::mem::size_of::<f32>())?,
            lm_head_nvfp4: None,
            mla_q_compressed_dev: DeviceBuffer::new(mla_q_compressed_bytes)?,
            mla_q_full_dev: DeviceBuffer::new(mla_q_full_bytes)?,
            mla_kv_a_dev: DeviceBuffer::new(mla_kv_a_bytes)?,
            mla_attn_out_dev: DeviceBuffer::new(mla_attn_out_bytes)?,
            mla_o_out_dev: DeviceBuffer::new(mla_o_out_bytes)?,
            dense_ffn_gate_dev: DeviceBuffer::new(dense_ffn_inter_bytes)?,
            dense_ffn_up_dev: DeviceBuffer::new(dense_ffn_inter_bytes)?,
            dense_ffn_down_dev: DeviceBuffer::new(hidden_bytes)?,
            shared_expert_inter_dev: DeviceBuffer::new(shared_expert_inter_bytes)?,
            shared_expert_down_dev: DeviceBuffer::new(hidden_bytes)?,
            host_staging: vec![0u8; hidden_bytes],
            host_staging2: vec![0u8; hidden_bytes],
            gpu_mode,
            gpu_kv_k,
            gpu_kv_v,
            max_kv_len,
            // DeltaNet state and scratch buffers
            dn_recurrent_state,
            dn_conv_state,
            dn_qkv_proj_dev: DeviceBuffer::new(dn_min(dn_qkv_dim * 2))?, // FP16
            dn_qkv_f32_dev: DeviceBuffer::new(dn_min(dn_qkv_dim * 4))?, // FP32
            dn_conv_out_dev: DeviceBuffer::new(dn_min(dn_qkv_dim * 4))?, // FP32
            dn_z_proj_dev: DeviceBuffer::new(dn_min(dn_inner_dim * 2))?, // FP16
            dn_z_f32_dev: DeviceBuffer::new(dn_min(dn_inner_dim * 4))?, // FP32
            dn_alpha_proj_dev: DeviceBuffer::new(dn_min(dn_num_v_heads * 2))?, // FP16
            dn_alpha_f32_dev: DeviceBuffer::new(dn_min(dn_num_v_heads * 4))?, // FP32
            dn_beta_proj_dev: DeviceBuffer::new(dn_min(dn_num_v_heads * 2))?, // FP16
            dn_beta_f32_dev: DeviceBuffer::new(dn_min(dn_num_v_heads * 4))?, // FP32
            dn_gate_dev: DeviceBuffer::new(dn_min(dn_num_v_heads * 4))?, // FP32
            dn_q_norm_dev: DeviceBuffer::new(dn_min(dn_key_dim * 4))?, // FP32
            dn_k_norm_dev: DeviceBuffer::new(dn_min(dn_key_dim * 4))?, // FP32
            dn_q_rep_dev: DeviceBuffer::new(dn_min(dn_inner_dim * 4))?, // FP32 (after repeat)
            dn_k_rep_dev: DeviceBuffer::new(dn_min(dn_inner_dim * 4))?, // FP32 (after repeat)
            dn_step_out_dev: DeviceBuffer::new(dn_min(dn_inner_dim * 4))?, // FP32
            dn_norm_out_dev: DeviceBuffer::new(dn_min(dn_inner_dim * 4))?, // FP32
            dn_o_out_dev: DeviceBuffer::new(dn_min(hidden_dim * 4))?, // FP32
            dn_a_f32_dev: DeviceBuffer::new(dn_min(dn_num_v_heads * 4))?, // FP32
            dn_dt_bias_f32_dev: DeviceBuffer::new(dn_min(dn_num_v_heads * 4))?, // FP32
            dn_norm_weight_f32_dev: DeviceBuffer::new(dn_min(dn_v_head_dim * 4))?, // FP32
            // Pre-allocated DeltaNet weight staging buffers (BF16 storage)
            dn_w_qkv: DeviceBuffer::new(dn_min(dn_qkv_dim * hidden_dim * 2))?,  // seg 30
            dn_w_z: DeviceBuffer::new(dn_min(dn_inner_dim * hidden_dim * 2))?,   // seg 31
            dn_w_beta: DeviceBuffer::new(dn_min(dn_num_v_heads * hidden_dim * 2))?, // seg 32
            dn_w_alpha: DeviceBuffer::new(dn_min(dn_num_v_heads * hidden_dim * 2))?, // seg 33
            dn_w_conv: DeviceBuffer::new(dn_min(dn_qkv_dim * 4 * 2))?,          // seg 34 (conv_kernel=4)
            dn_w_dt_bias: DeviceBuffer::new(dn_min(dn_num_v_heads * 2))?,        // seg 35
            dn_w_a_log: DeviceBuffer::new(dn_min(dn_num_v_heads * 2))?,          // seg 36
            dn_w_norm: DeviceBuffer::new(dn_min(dn_v_head_dim * 2))?,            // seg 37
            dn_w_out: DeviceBuffer::new(dn_min(dn_inner_dim * hidden_dim * 2))?, // seg 38
            dn_conv_w_f32_dev: DeviceBuffer::new(dn_min(dn_qkv_dim * 4 * 4))?,  // conv FP32
            // Pre-allocated GQA attention weight staging buffers (FP16 storage)
            // Only allocated at full size if the model has GQA attention layers.
            gqa_w_q: DeviceBuffer::new(q_proj_elems * hidden_dim * 2)?,     // seg 4: [q_dim, hidden_dim]
            gqa_w_o: DeviceBuffer::new(hidden_dim * attn_out_elems * 2)?,   // seg 5: [hidden_dim, attn_dim]
            gqa_w_k: DeviceBuffer::new(num_kv_heads * head_dim * hidden_dim * 2)?, // seg 12
            gqa_w_v: DeviceBuffer::new(num_kv_heads * head_dim * hidden_dim * 2)?, // seg 13
            gqa_w_qnorm: DeviceBuffer::new(head_dim * 2)?,                  // seg 27
            gqa_w_knorm: DeviceBuffer::new(head_dim * 2)?,                  // seg 28
            attn_norm_w: DeviceBuffer::new(hidden_dim * 2)?,               // seg 6: attn pre-norm
            kv_cache,
            mla_kv_cache,
            tiered_kv,
            eviction_policy,
            position: 0,
            _worker_handles: worker_handles,
            mode_detector,
            current_mode: ActivationMode::Generalist,
            token_expert_ids: Vec::with_capacity(512), // 8 experts * 60 layers = 480
            decode_step: 0,
            shared_tensor_cache: std::collections::HashMap::new(),
            kv_b_proj_f32_cache: std::collections::HashMap::new(),
            kv_b_proj_f32_device: std::collections::HashMap::new(),
            mla_gpu_kv_latent,
            mla_gpu_kv_rope,
            mla_q_absorbed_dev: DeviceBuffer::new(mla_q_absorbed_bytes)?,
            mla_q_rope_f32_dev: DeviceBuffer::new(mla_q_rope_f32_bytes)?,
            mla_v_latent_dev: DeviceBuffer::new(mla_v_latent_bytes)?,
            mla_v_out_f32_dev: DeviceBuffer::new(mla_v_out_f32_bytes)?,
            mla_rope_freqs_dev,
            mla_kv_norm_weight_dev: std::collections::HashMap::new(),
            mla_v_out_f16_dev: DeviceBuffer::new(mla_v_out_f16_bytes)?,
            shared_tensor_cache_device: std::collections::HashMap::new(),
            pre_assembled_weights: std::collections::HashMap::new(),
            task_context: None,
            gear_profiles: std::collections::HashMap::new(),
            e_score_correction_bias: None,
            diag_enabled: std::env::var("VIB3_DIAG").map_or(false, |v| v == "1"),
            shared_expert_nvfp4: std::collections::HashMap::new(),
            shared_expert_swiglu_f32: DeviceBuffer::new(shared_inter.max(1) * 4)?, // FP32 SwiGLU output
            shared_expert_down_f32: DeviceBuffer::new(hidden_dim * 4)?, // FP32 down_proj output
            shared_expert_mid_fp4: DeviceBuffer::new((shared_inter.max(1) / 2).max(64))?, // FP4 data
            shared_expert_mid_scales: DeviceBuffer::new((shared_inter.max(1) / 32).max(4))?, // E8M0 scales
            moe_page_table_dev: None,
            d_position: DeviceBuffer::new(4)?, // int32 device scalar
            cuda_graph_exec: None,
            cuda_graph: None,
            capturing_graph: false,
        };

        // Pre-assemble all shared weight tensors into permanent VRAM buffers.
        // This eliminates ~10.6 GB/token of D2D copies during inference.
        let assembled = engine.preassemble_all_shared_weights();
        if assembled > 0 {
            tracing::info!("Weight preassembly complete: {} tensors ready for zero-copy inference", assembled);
        }

        // SKIPPED: sequential→split-half repack is no longer needed.
        // The MXFP4→NVFP4 converter now copies GGML qs bytes directly,
        // which are already in split-half format.  Tiled repack (below)
        // still runs to convert row-major split-half → tiled layout.
        if engine.model_config.expert_dtype == DType::NVFP4 && engine.buffer_mgr.is_fully_resident() {
            tracing::info!("Skipping seq→split-half repack (data already split-half from GGML)");
        }

        // Repack all NVFP4 expert weight pages from split-half to TILED layout.
        // Tiled layout: 16 rows × 32 bytes per K-tile contiguous (512 bytes per tile).
        // This enables coalesced memory access in the K-parallel MoE GEMV kernels.
        // Must run AFTER the split-half repack above.
        if engine.model_config.expert_dtype == DType::NVFP4 && engine.buffer_mgr.is_fully_resident() {
            let tile_start = Instant::now();
            let hidden_dim = engine.model_config.hidden_dim as usize;
            let expert_hidden_dim = engine.model_config.expert_hidden_dim as usize;

            // Max FP4 data size across expert page types (up/gate and down are equal here)
            let max_fp4_data = {
                let upgate = (hidden_dim / 2) * expert_hidden_dim;
                let down = (expert_hidden_dim / 2) * hidden_dim;
                upgate.max(down)
            };

            if max_fp4_data > 0 {
                match DeviceBuffer::new(max_fp4_data) {
                    Ok(temp_buf) => {
                        let mut tiled_count = 0usize;
                        let page_catalog = engine.model_file.page_catalog();
                        for entry in page_catalog.iter() {
                            if entry.expert == 0xFFFF || entry.segment > 2 {
                                continue;
                            }
                            let page_id = entry.page_id();
                            if let Some(handle) = engine.buffer_mgr.get_page_resident(&page_id) {
                                if handle.device_ptr.is_null() {
                                    continue;
                                }

                                let (k, m_slice) = match entry.segment {
                                    0 | 1 => (hidden_dim, entry.row_count as usize),
                                    2 => (expert_hidden_dim, entry.row_count as usize),
                                    _ => unreachable!(),
                                };

                                // Verify alignment requirements for tiled repack
                                if k % 64 != 0 || m_slice % 16 != 0 {
                                    tracing::warn!(
                                        "Skipping tiled repack for expert page (M={}, K={}): alignment",
                                        m_slice, k
                                    );
                                    continue;
                                }

                                if let Err(e) = kernels::repack_row_to_tiled(
                                    handle.device_ptr as *mut u8,
                                    temp_buf.as_mut_ptr(),
                                    m_slice,
                                    k,
                                    &engine.stream,
                                ) {
                                    tracing::warn!(
                                        "Tiled repack failed for expert page (M={}, K={}): {}",
                                        m_slice, k, e
                                    );
                                } else {
                                    tiled_count += 1;
                                }
                            }
                        }
                        engine.stream.synchronize()?;
                        tracing::info!(
                            "Tiled repack: {} NVFP4 expert weight pages in {:.1}ms (temp buf {:.1} MB)",
                            tiled_count,
                            tile_start.elapsed().as_millis(),
                            max_fp4_data as f64 / (1024.0 * 1024.0),
                        );
                    }
                    Err(e) => {
                        tracing::warn!(
                            "Failed to allocate tiled repack temp buffer ({:.1} MB): {}",
                            max_fp4_data as f64 / (1024.0 * 1024.0), e
                        );
                    }
                }
            }
        }

        // Build device-side page pointer table for GPU-only MoE dispatch.
        // Layout: [num_moe_layers * num_experts * 3] u64 device pointers.
        // Index: moe_layer * num_experts * 3 + expert_id * 3 + segment
        // This eliminates per-token host sync + H2D memcpys for expert pointer arrays.
        if engine.model_config.expert_dtype == DType::NVFP4 && engine.buffer_mgr.is_fully_resident() {
            let table_start = Instant::now();
            let num_moe_layers = engine.model_config.num_moe_layers as usize;
            let num_experts = engine.model_config.num_experts as usize;
            let dense_offset = engine.model_config.dense_layer_idx as u16;
            let table_entries = num_moe_layers * num_experts * 3;
            let mut table_host = vec![0u64; table_entries];
            let mut populated = 0usize;

            for moe_layer in 0..num_moe_layers {
                let storage_layer = moe_layer as u16 + dense_offset;
                for expert_id in 0..num_experts {
                    let pages = engine.model_file.pages_for_expert(storage_layer, expert_id as u16);
                    for entry in pages {
                        if entry.segment > 2 {
                            continue;
                        }
                        let page_id = entry.page_id();
                        if let Some(handle) = engine.buffer_mgr.get_page_resident(&page_id) {
                            if !handle.device_ptr.is_null() {
                                let idx = moe_layer * num_experts * 3
                                    + expert_id * 3
                                    + entry.segment as usize;
                                table_host[idx] = handle.device_ptr as u64;
                                populated += 1;
                            }
                        }
                    }
                }
            }

            // Upload to device
            let table_bytes = table_entries * std::mem::size_of::<u64>();
            match DeviceBuffer::new(table_bytes) {
                Ok(table_buf) => {
                    // SAFETY: table_host is a flat Vec<u64>, table_buf is device memory of the same size.
                    let host_bytes = unsafe {
                        std::slice::from_raw_parts(
                            table_host.as_ptr() as *const u8,
                            table_bytes,
                        )
                    };
                    if let Err(e) = table_buf.copy_from_host(host_bytes) {
                        tracing::warn!("Failed to upload MoE page table to device: {}", e);
                    } else {
                        engine.moe_page_table_dev = Some(table_buf);
                        tracing::info!(
                            "MoE page table built: {} entries ({} populated, {:.1} KB) in {:.1}ms",
                            table_entries,
                            populated,
                            table_bytes as f64 / 1024.0,
                            table_start.elapsed().as_millis(),
                        );
                    }
                }
                Err(e) => {
                    tracing::warn!("Failed to allocate MoE page table: {}", e);
                }
            }
        }

        // Pre-warm shared expert tensors into device cache for ALL MoE layers.
        // This ensures segments 3 (router), 7 (moe_norm), 14/15/16 (shared expert up/gate/down),
        // and 26 (shared expert sigmoid gate) are allocated and populated in
        // shared_tensor_cache_device BEFORE inference begins. Without this, step 1 would
        // trigger cudaMalloc via ensure_shared_tensor_device, preventing CUDA graph capture
        // at step 1 (since cudaMalloc is illegal during graph capture).
        #[cfg(feature = "cuda")]
        if engine.buffer_mgr.is_fully_resident() && engine.model_config.num_experts > 0 {
            let warm_start = Instant::now();
            let num_layers = engine.model_config.num_layers as u16;
            let shared_segs: &[u16] = &[3, 7, 14, 15, 16, 26];
            let mut warmed = 0usize;
            for layer_idx in 0..num_layers {
                for &seg in shared_segs {
                    if engine.ensure_shared_tensor_device(layer_idx, seg).await {
                        warmed += 1;
                    }
                }
            }
            engine.stream.synchronize()?;
            tracing::info!(
                "Pre-warmed {} shared expert tensors ({} layers × {} segments) in {:.0}ms",
                warmed, num_layers, shared_segs.len(), warm_start.elapsed().as_millis(),
            );
        }

        // Pre-warm MoE GPU buffers (intermediate per-expert buffers + device pointer arrays).
        // This ensures vib3_launch_moe_experts_fused_gpu() won't call cudaMalloc or
        // synchronous cudaMemcpy during CUDA graph capture.
        #[cfg(feature = "cuda")]
        if engine.buffer_mgr.is_fully_resident() && engine.model_config.num_experts > 0 {
            let m_mid = engine.model_config.expert_hidden_dim as i32; // 1024
            let k_mid = engine.model_config.expert_hidden_dim as i32; // 1024
            let err = unsafe { cuda_ffi::vib3_moe_prewarm_gpu_bufs(m_mid, k_mid) };
            if err != 0 {
                tracing::warn!("MoE GPU buffer pre-warm failed (err={})", err);
            } else {
                tracing::info!("Pre-warmed MoE GPU buffers (M_mid={}, K_mid={})", m_mid, k_mid);
            }
        }

        // Eagerly convert lm_head to NVFP4 at load time (instead of lazily on first token).
        // This saves ~17ms on step 0 by moving the FP16→NVFP4 conversion + tiled repack
        // out of the critical decode path.
        #[cfg(feature = "cuda")]
        if engine.buffer_mgr.is_fully_resident() {
            let lm_start = Instant::now();
            let vocab_size = engine.model_config.vocab_size as usize;
            let hidden_dim = engine.model_config.hidden_dim as usize;
            let fp16_lm_head = match std::env::var("VIB3_FP16_LM_HEAD") {
                Ok(v) => v == "1",
                Err(_) => engine.model_config.architecture == "qwen3_5_moe",
            };
            engine.ensure_shared_tensor_device(0, 11).await;
            if let Some((lm_head_ptr, lm_head_size)) = engine.get_device_tensor(0, 11) {
                let expected_fp16 = vocab_size * hidden_dim * 2;
                if lm_head_size == expected_fp16 {
                    if fp16_lm_head {
                        tracing::info!(
                            "Skipping lm_head NVFP4 load-time conversion (FP16 policy active)"
                        );
                    } else {
                        tracing::info!(
                            "Converting lm_head to NVFP4 at load time: {}x{} FP16 ({:.1} MB)",
                            vocab_size, hidden_dim, lm_head_size as f64 / (1024.0 * 1024.0)
                        );
                        match kernels::fp16_to_nvfp4_weight(lm_head_ptr, vocab_size, hidden_dim, &engine.stream) {
                            Ok(nvfp4_buf) => {
                                engine.stream.synchronize()?;
                                // Repack to tiled layout for coalesced GEMV
                                let fp4_data_size = vocab_size * hidden_dim.div_ceil(2);
                                match DeviceBuffer::new(fp4_data_size) {
                                    Ok(temp_buf) => {
                                        if let Err(e) = kernels::repack_row_to_tiled(
                                            nvfp4_buf.as_mut_ptr(),
                                            temp_buf.as_mut_ptr(),
                                            vocab_size,
                                            hidden_dim,
                                            &engine.stream,
                                        ) {
                                            tracing::warn!("lm_head tiled repack failed: {}", e);
                                        } else {
                                            engine.stream.synchronize()?;
                                        }
                                        // temp_buf dropped here, frees VRAM
                                    }
                                    Err(e) => {
                                        tracing::warn!("lm_head tiled repack temp alloc failed: {}", e);
                                    }
                                }
                                tracing::info!(
                                    "lm_head NVFP4 conversion complete: {:.1} MB in {:.0}ms",
                                    nvfp4_buf.size() as f64 / (1024.0 * 1024.0),
                                    lm_start.elapsed().as_millis(),
                                );
                                engine.lm_head_nvfp4 = Some(nvfp4_buf);
                            }
                            Err(e) => {
                                tracing::warn!("lm_head NVFP4 conversion failed at load time: {}", e);
                            }
                        }
                    }
                }
            }
        }

        Ok(engine)
    }

    /// Load e_score_correction_bias from an external binary file.
    ///
    /// File format: [num_layers * num_experts] F32, row-major.
    /// Each layer has num_experts contiguous F32 values.
    /// Layer 0 (dense) should be all zeros.
    pub fn load_e_score_correction_bias(&mut self, path: &std::path::Path) -> Result<()> {
        let num_layers = self.model_config.num_layers as usize;
        let num_experts = self.model_config.num_experts as usize;
        let expected_bytes = num_layers * num_experts * 4;

        let data = std::fs::read(path)?;
        if data.len() != expected_bytes {
            return Err(Error::InvalidFormat {
                reason: format!(
                    "e_score_correction_bias size mismatch: got {} bytes, expected {} ({} layers × {} experts × 4)",
                    data.len(), expected_bytes, num_layers, num_experts,
                ),
            });
        }

        let floats = unsafe {
            std::slice::from_raw_parts(data.as_ptr() as *const f32, num_layers * num_experts)
        };
        let mut bias_per_layer = Vec::with_capacity(num_layers);
        for layer in 0..num_layers {
            let start = layer * num_experts;
            bias_per_layer.push(floats[start..start + num_experts].to_vec());
        }

        // Log stats for a few layers
        for &l in &[1usize, 30, 60] {
            if l < num_layers {
                let b = &bias_per_layer[l];
                let min = b.iter().cloned().fold(f32::INFINITY, f32::min);
                let max = b.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                tracing::info!(
                    "e_score_correction_bias layer {}: min={:.4}, max={:.4}",
                    l, min, max,
                );
            }
        }

        tracing::info!(
            "Loaded e_score_correction_bias: {} layers × {} experts from {}",
            num_layers, num_experts, path.display(),
        );
        self.e_score_correction_bias = Some(bias_per_layer);
        Ok(())
    }

    // ── Task Context (Gearbox integration — Phase A) ──────────────────

    /// Set the task context for the current/next inference request.
    ///
    /// When a `TaskContext` is provided, the engine uses the gear signal for:
    /// - **Mode detection override** (Phase A): The gear's expected mode
    ///   (specialist/generalist) takes effect immediately, skipping the
    ///   128-token entropy warmup window.
    /// - **Proactive cache warming** (Phase B): Pin the gear's working set
    ///   in T1 before generation starts.
    /// - **Filtered HNSW search** (Phase C, when implemented): Narrow page
    ///   retrieval to the active gear's domain.
    /// - **Eviction priority** (Phase D, when implemented): Tag cached pages
    ///   with gear context for cross-gear eviction decisions.
    ///
    /// When `None`, falls back to entropy-based mode detection and unfiltered search.
    ///
    /// Called by the API handler before each inference request.
    pub async fn set_task_context(&mut self, ctx: Option<TaskContext>) {
        let old_gear = self
            .task_context
            .as_ref()
            .and_then(|tc| tc.gear.clone());

        let new_gear = ctx.as_ref().and_then(|tc| tc.gear.clone());

        let gear_changed = old_gear != new_gear;

        if gear_changed {
            if let Some(ref ctx) = ctx {
                let gear_name = ctx.gear.as_deref().unwrap_or("none");
                let mode = ctx.expected_mode().map(|m| m.name()).unwrap_or("unknown");
                tracing::info!(
                    "Task context changed: gear={}, expected_mode={}, alpha={:?}, phase={:?}",
                    gear_name,
                    mode,
                    ctx.alpha,
                    ctx.phase,
                );

                // Phase A: Override mode detection when gear signal is present
                if self.config.gear.enabled && self.config.gear.override_mode_detection {
                    if let Some(expected_mode) = ctx.expected_mode() {
                        if let Some(ref mut detector) = self.mode_detector {
                            detector.force_mode(expected_mode);
                        }
                        self.current_mode = expected_mode;
                        tracing::info!(
                            "Mode forced to {} from gear signal (gear={})",
                            expected_mode,
                            gear_name,
                        );
                    }
                }
            } else {
                tracing::info!("Task context cleared (falling back to entropy-based detection)");
            }
        }

        self.task_context = ctx;

        if gear_changed && self.config.gear.enabled {
            // Phase D: Update buffer manager's current gear for page tagging + eviction
            if self.config.gear.gear_aware_eviction {
                self.buffer_mgr.set_current_gear(new_gear.clone());
            }

            // Phase C: Update planner's gear domain filter for filtered HNSW search
            if self.config.gear.filtered_hnsw {
                let domains = new_gear
                    .as_deref()
                    .and_then(|g| self.config.gear.gear_domains.get(g))
                    .cloned()
                    .unwrap_or_default();
                self.planner.set_gear_domains(domains);
            }

            // Phase B: Proactive cache warming on gear change
            if self.config.gear.proactive_cache_warming {
                self.on_task_context_change(
                    old_gear.as_deref(),
                    new_gear.as_deref(),
                )
                .await;
            }
        }
    }

    /// Get the current task context, if any.
    pub fn task_context(&self) -> Option<&TaskContext> {
        self.task_context.as_ref()
    }

    /// Load gear profiles from a JSON file.
    ///
    /// The JSON file maps gear names to profiles containing per-layer hot expert IDs.
    /// Generated by Gearbox training analysis (tools/train_gearbox.py).
    ///
    /// Expected format:
    /// ```json
    /// {
    ///   "code": {
    ///     "name": "code",
    ///     "hot_experts": [[352, 276, ...], [101, 55, ...], ...],
    ///     "total_unique_experts": 304,
    ///     "estimated_vram_gb": 3.5
    ///   },
    ///   ...
    /// }
    /// ```
    pub fn load_gear_profiles(&mut self, path: &str) -> usize {
        let data = match std::fs::read_to_string(path) {
            Ok(d) => d,
            Err(e) => {
                tracing::warn!("Failed to read gear profiles from {}: {}", path, e);
                return 0;
            }
        };

        let json: serde_json::Value = match serde_json::from_str(&data) {
            Ok(j) => j,
            Err(e) => {
                tracing::warn!("Failed to parse gear profiles JSON: {}", e);
                return 0;
            }
        };

        let obj = match json.as_object() {
            Some(o) => o,
            None => {
                tracing::warn!("Gear profiles JSON is not an object");
                return 0;
            }
        };

        let mut count = 0;
        for (gear_name, profile_val) in obj {
            let hot_experts_val = match profile_val.get("hot_experts") {
                Some(v) => v,
                None => continue,
            };

            let hot_experts: Vec<Vec<u16>> = match hot_experts_val.as_array() {
                Some(layers) => layers
                    .iter()
                    .map(|layer_val| {
                        layer_val
                            .as_array()
                            .map(|experts| {
                                experts
                                    .iter()
                                    .filter_map(|e| e.as_u64().map(|v| v as u16))
                                    .collect()
                            })
                            .unwrap_or_default()
                    })
                    .collect(),
                None => continue,
            };

            let total_pages = hot_experts
                .iter()
                .map(|layer| layer.len())
                .sum::<usize>()
                * self.model_config.pages_per_expert();
            let vram_required = total_pages * PAGE_SIZE;

            let profile = SpecialistProfile {
                name: gear_name.clone(),
                centroid: vec![], // No centroid needed for gear-based profiles
                hot_experts,
                total_pages,
                vram_required,
            };

            tracing::info!(
                "Loaded gear profile '{}': {} layers, {} pages, {:.1} GB",
                gear_name,
                profile.hot_experts.len(),
                total_pages,
                vram_required as f64 / (1024.0 * 1024.0 * 1024.0),
            );

            self.gear_profiles.insert(gear_name.clone(), profile);
            count += 1;
        }

        tracing::info!("Loaded {} gear profiles from {}", count, path);
        count
    }

    /// Get the number of loaded gear profiles.
    pub fn gear_profile_count(&self) -> usize {
        self.gear_profiles.len()
    }

    // ── Phase B: Proactive Cache Warming ─────────────────────────────

    /// Handle a task context change by warming the cache for the new gear.
    ///
    /// This is the core Phase B method. When the gear changes:
    /// 1. If the previous gear was specialist → unpin those pages
    /// 2. Look up the new gear's specialist profile
    /// 3. Pin the new gear's hot expert pages in T1
    /// 4. Force mode to specialist/generalist based on gear
    ///
    /// This eliminates the 128-token entropy warmup window — the cache is
    /// warm from token 1 because the gear signal is available before generation.
    async fn on_task_context_change(
        &mut self,
        old_gear: Option<&str>,
        new_gear: Option<&str>,
    ) {
        // Step 1: Unpin old gear's pages if we were in specialist mode
        if old_gear.is_some() && self.current_mode == ActivationMode::Specialist {
            let unpinned = self.buffer_mgr.unpin_expert_cluster();
            if unpinned > 0 {
                tracing::info!(
                    "Gear transition: unpinned {} pages from previous gear '{}'",
                    unpinned,
                    old_gear.unwrap_or("unknown"),
                );
            }
        }

        // Step 2: Look up the new gear's profile and pin its working set
        if let Some(gear_name) = new_gear {
            if let Some(profile) = self.gear_profiles.get(gear_name).cloned() {
                let total_experts: usize = profile.hot_experts.iter().map(|l| l.len()).sum();

                if total_experts > 0 {
                    // Build (layer_idx, expert_id) pairs for buffer manager
                    let mut pin_targets: Vec<(u16, u16)> = Vec::new();
                    let dense_offset = self.model_config.dense_layer_idx as u16;

                    for (moe_layer, experts) in profile.hot_experts.iter().enumerate() {
                        let layer_idx = moe_layer as u16 + dense_offset;
                        for &expert_id in experts {
                            pin_targets.push((layer_idx, expert_id));
                        }
                    }

                    tracing::info!(
                        "Gear '{}': pinning {} experts across {} layers ({} expert-layer instances, ~{:.1} GB)",
                        gear_name,
                        profile.unique_expert_count(),
                        profile.hot_experts.len(),
                        pin_targets.len(),
                        profile.vram_required as f64 / (1024.0 * 1024.0 * 1024.0),
                    );

                    match self.buffer_mgr.pin_expert_cluster(&pin_targets).await {
                        Ok(pinned) => {
                            tracing::info!(
                                "Gear '{}': pinned {} pages in T1 (target: {} pages)",
                                gear_name,
                                pinned,
                                profile.total_pages,
                            );
                        }
                        Err(e) => {
                            tracing::error!(
                                "Gear '{}': failed to pin expert cluster: {}",
                                gear_name,
                                e,
                            );
                        }
                    }
                }
            } else {
                tracing::debug!(
                    "No gear profile loaded for '{}' — cache warming skipped",
                    gear_name,
                );
            }
        }

        // Step 3: Update mode and notify planner
        self.planner.set_mode(self.current_mode);
    }

    /// Generate text from a prompt.
    pub async fn generate(&mut self, prompt: &str) -> Result<GenerateResult> {
        self.generate_with_params(prompt, SamplingParams::default())
            .await
    }

    /// Generate text with custom sampling parameters.
    pub async fn generate_with_params(
        &mut self,
        prompt: &str,
        params: SamplingParams,
    ) -> Result<GenerateResult> {
        let start = Instant::now();

        // Tokenize
        let normalized_prompt =
            Self::normalize_prompt_for_model(&self.model_config.architecture, prompt);
        let input_tokens = self.tokenizer.encode(&normalized_prompt);
        let prompt_len = input_tokens.len();
        tracing::debug!("Prompt: {} tokens", prompt_len);

        // Prefill
        let prefill_start = Instant::now();
        self.prefill(&input_tokens).await?;
        let ttft = prefill_start.elapsed().as_secs_f64() * 1000.0;

        // Generate
        let max_tokens = params.max_tokens;
        let mut generated = Vec::new();

        for step in 0..max_tokens {
            let token_id = self.generate_token(step, &params, &generated).await?;

            if self.tokenizer.is_eos(token_id) {
                break;
            }
            if params.stop_tokens.contains(&token_id) {
                break;
            }
            generated.push(token_id);
        }

        let total_ms = start.elapsed().as_secs_f64() * 1000.0;
        let gen_time = total_ms - ttft;
        let tps = if gen_time > 0.0 {
            generated.len() as f64 / (gen_time / 1000.0)
        } else {
            0.0
        };

        let text = self.tokenizer.decode(&generated);

        Ok(GenerateResult {
            text,
            tokens_generated: generated.len(),
            prompt_tokens: prompt_len,
            tokens_per_second: tps,
            time_to_first_token_ms: ttft,
            total_time_ms: total_ms,
            stats: self.stats.snapshot(),
        })
    }

    /// Process input prompt (prefill phase).
    ///
    /// For each token in the prompt, we:
    /// 1. Look up token embedding (or fallback to pattern)
    /// 2. Run through all transformer layers (attention + MoE interleaved per layer)
    /// 3. Populate KV cache at each position for autoregressive generation
    ///
    /// This builds the full KV cache so that the decode phase only needs
    /// to process one token at a time.
    ///
    /// ## Transformer layer structure (pre-norm architecture)
    ///
    /// ```text
    /// for each layer:
    ///   normed = RMSNorm(hidden_state)
    ///   hidden_state = hidden_state + Attention(normed)     # residual uses ORIGINAL
    ///   normed = RMSNorm(hidden_state)
    ///   hidden_state = hidden_state + MoE(normed)           # residual uses ORIGINAL
    /// ```
    async fn prefill(&mut self, tokens: &[u32]) -> Result<()> {
        let hidden_dim = self.model_config.hidden_dim as usize;

        // Reset KV cache and position for new sequence
        self.kv_cache.clear();
        if let Some(ref mut mla_kv) = self.mla_kv_cache {
            mla_kv.clear();
        }
        if let Some(ref mut tiered_kv) = self.tiered_kv {
            tiered_kv.clear();
        }
        // Reset DeltaNet recurrent + conv states for a fresh sequence.
        // Without this, token-0 can inherit stale state from a previous request.
        for state in &mut self.dn_recurrent_state {
            state.zero();
        }
        for conv_state in &mut self.dn_conv_state {
            conv_state.zero();
        }
        self.position = 0;

        if tokens.is_empty() {
            return Ok(());
        }

        // Load full embedding table to device (VRAM) for GPU kernels
        self.ensure_shared_tensor_device(0, 10).await; // segment 10 = embeddings

        let num_layers = self.model_config.num_layers;
        let dense_layer_idx = self.model_config.dense_layer_idx;
        let num_moe_layers = self.model_config.num_moe_layers;

        // Process each prompt token through all layers
        for (tok_idx, &token_id) in tokens.iter().enumerate() {
            // 1. Embedding lookup for this token
            if let Some((embed_ptr, _embed_size)) = self.get_device_tensor(0, 10) {
                kernels::embedding_lookup(
                    embed_ptr,
                    token_id,
                    hidden_dim,
                    self.hidden_state.as_mut_ptr(),
                );
            } else {
                // Fallback: deterministic embedding from token ID — build on host, copy to device
                let hidden_bytes = hidden_dim * std::mem::size_of::<f16>();
                let mut host_embed = vec![0u8; hidden_bytes];
                let state = unsafe {
                    std::slice::from_raw_parts_mut(host_embed.as_mut_ptr() as *mut f16, hidden_dim)
                };
                for (d, s) in state.iter_mut().enumerate().take(hidden_dim) {
                    let freq = (d as f32 + 1.0) * 0.001;
                    let val = (token_id as f32 * freq).sin() * 0.1;
                    *s = f16::from_f32(val);
                }
                let _ = self.hidden_state.copy_from_host(&host_embed);
            }

            // Initialize FP32 hidden state accumulator from FP16 embedding
            kernels::f16_to_f32(
                self.hidden_state.as_ptr(),
                self.hidden_state_f32.as_mut_ptr(),
                hidden_dim,
                &self.stream,
            )?;

            // Debug: log embedding output for every token in prefill
            if self.diag_enabled {
                self.stream.synchronize().ok();
                let hidden_bytes = hidden_dim * std::mem::size_of::<f16>();
                let mut h_buf = vec![0u8; hidden_bytes];
                self.hidden_state.copy_to_host(&mut h_buf).ok();
                let h_f16 = unsafe { std::slice::from_raw_parts(h_buf.as_ptr() as *const f16, hidden_dim) };
                let h_f32: Vec<f32> = h_f16.iter().map(|v| v.to_f32()).collect();
                let min = h_f32.iter().cloned().fold(f32::INFINITY, f32::min);
                let max = h_f32.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let mean = h_f32.iter().sum::<f32>() / hidden_dim as f32;
                let l2 = h_f32.iter().map(|v| v * v).sum::<f32>().sqrt();
                let nans = h_f32.iter().filter(|v| v.is_nan()).count();
                tracing::info!(
                    "EMBEDDING DIAG tok={} id={}: first8=[{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6}], min={:.6}, max={:.6}, mean={:.6}, L2={:.2}, nan={}",
                    tok_idx, token_id,
                    h_f32[0], h_f32[1], h_f32[2], h_f32[3], h_f32[4], h_f32[5], h_f32[6], h_f32[7],
                    min, max, mean, l2, nans
                );
                // Dump embedding FP32 for every token
                let dump_dir = "/home/brian/code/vib3/dump";
                let _ = std::fs::create_dir_all(dump_dir);
                let f32_bytes: Vec<u8> = h_f32.iter().flat_map(|v| v.to_le_bytes()).collect();
                let dump_path = format!("{}/vib3_embed_f32_tok{}.bin", dump_dir, tok_idx);
                let _ = std::fs::write(&dump_path, &f32_bytes);
            }

            // 2. Run through all transformer layers at this position
            //    Each layer: Attention sublayer → MoE/FFN sublayer (interleaved)
            self.position = tok_idx;

            // Update device-side position scalar so CUDA KV-cache append and
            // decode-attention kernels see the correct sequence position.
            #[cfg(feature = "cuda")]
            {
                let stream_ptr = self.stream.raw_ptr();
                let err = unsafe {
                    cuda_ffi::vib3_update_device_int32(
                        self.d_position.as_mut_ptr(),
                        tok_idx as i32,
                        stream_ptr,
                    )
                };
                if err != 0 {
                    tracing::warn!("Failed to update device position during prefill (err={})", err);
                }
            }

            let is_last_token = tok_idx == tokens.len() - 1;

            for layer_idx in 0..num_layers as u16 {
                // ── Attention sublayer (with pre-norm + residual) ──
                self.run_attention_layer(layer_idx).await?;

                // Detailed diagnostic: after attention, before MoE
                if self.diag_enabled && is_last_token && layer_idx <= 5 {
                    self.stream.synchronize()?;
                    // Read FP32 accumulator (the authoritative hidden state)
                    let f32_bytes = hidden_dim * 4;
                    let mut diag_buf = vec![0u8; f32_bytes];
                    self.hidden_state_f32.copy_to_host(&mut diag_buf)?;
                    let h_f32 = unsafe { std::slice::from_raw_parts(diag_buf.as_ptr() as *const f32, hidden_dim) };
                    let h_l2 = h_f32.iter().map(|v| v * v).sum::<f32>().sqrt();
                    let h_max = h_f32.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                    let h_min = h_f32.iter().cloned().fold(f32::INFINITY, f32::min);
                    tracing::info!(
                        "LAYER{} DIAG after_attn (f32): L2={:.4}, min={:.6}, max={:.6}, first4=[{:.6},{:.6},{:.6},{:.6}]",
                        layer_idx, h_l2, h_min, h_max, h_f32[0], h_f32[1], h_f32[2], h_f32[3],
                    );
                    // Dump to file
                    let dump_dir = "/home/brian/code/vib3/dump";
                    let _ = std::fs::create_dir_all(dump_dir);
                    let dump_path = format!("{}/vib3_mixtral_after_attn_f32_L{}_lastpos.bin", dump_dir, layer_idx);
                    let _ = std::fs::write(&dump_path, &diag_buf);
                }

                // ── MoE/FFN sublayer (with pre-norm + residual) ──
                let is_moe_layer = layer_idx >= dense_layer_idx as u16
                    && (layer_idx - dense_layer_idx as u16) < num_moe_layers as u16;
                let is_dense_ffn_layer = (layer_idx as u32) < dense_layer_idx;

                if is_moe_layer {
                    self.run_moe_sublayer(layer_idx).await?;
                } else if is_dense_ffn_layer {
                    self.run_dense_ffn_sublayer(layer_idx).await?;
                }

                // Diagnostic: log hidden state stats after each layer
                // Dump for ALL tokens at layer 0, and for last token at other layers
                let should_dump_layer = self.diag_enabled && (
                    layer_idx == 0  // always dump layer 0 for all tokens
                    || (is_last_token && (layer_idx <= 15 || layer_idx % 5 == 0 || layer_idx >= num_layers as u16 - 2))
                );
                if should_dump_layer {
                    self.stream.synchronize()?;
                    let f32_bytes = hidden_dim * 4;
                    let mut diag_buf = vec![0u8; f32_bytes];
                    self.hidden_state_f32.copy_to_host(&mut diag_buf)?;
                    let h_f32 = unsafe {
                        std::slice::from_raw_parts(diag_buf.as_ptr() as *const f32, hidden_dim)
                    };
                    let h_max = h_f32.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                    let h_min = h_f32.iter().cloned().fold(f32::INFINITY, f32::min);
                    let h_mean = h_f32.iter().sum::<f32>() / hidden_dim as f32;
                    let h_nan = h_f32.iter().filter(|v| v.is_nan()).count();
                    let h_l2 = h_f32.iter().map(|v| v * v).sum::<f32>().sqrt();
                    tracing::info!(
                        "PREFILL DIAG tok={} layer={}: min={:.4}, max={:.4}, mean={:.6}, L2={:.2}, nan={}",
                        tok_idx, layer_idx, h_min, h_max, h_mean, h_l2, h_nan,
                    );

                    // Dump FP32 hidden state to file
                    {
                        let dump_dir = "/home/brian/code/vib3/dump";
                        let _ = std::fs::create_dir_all(dump_dir);
                        let dump_path = format!("{}/vib3_postlayer_f32_L{}_tok{}.bin", dump_dir, layer_idx, tok_idx);
                        let _ = std::fs::write(&dump_path, &diag_buf);
                    }
                }
            }

            // Advance tiered KV cache position after all layers for this token
            if let Some(ref mut tiered_kv) = self.tiered_kv {
                tiered_kv.advance_position();
            }
        }

        self.position = tokens.len();

        tracing::info!(
            "Prefill complete ({} tokens, position={})",
            tokens.len(),
            self.position
        );
        Ok(())
    }

    /// Run attention for a single layer at the current position.
    ///
    /// Dispatches to the appropriate attention implementation:
    /// 1. **MLA path** (Kimi K2.5, DeepSeek-V3): Uses compressed latent KV cache.
    ///    Loads q_a_proj, q_b_proj, kv_a_proj, kv_b_proj (segments 20-23).
    /// 2. **GPU-projected GQA path**: Q/K/V/O projections on GPU, attention on CPU.
    /// 3. **CPU fallback GQA path**: Everything on CPU (when GPU weights unavailable).
    ///
    /// Pre-norm architecture: RMSNorm is applied to a COPY of hidden_state
    /// (the original is preserved for the residual connection).
    async fn run_attention_layer(&mut self, layer_idx: u16) -> Result<()> {
        let profile = self.diag_enabled && self.decode_step == 1;

        // Evict only attention-specific segments from other layers.
        // Attention weights (Q/K/V/O projections, Q/K norms) are large (~156MB per layer)
        // and only needed for the current attention layer.
        // MoE segments (norm=7, router=3, shared expert=14/15/16, gate=26) are small
        // (~20MB per layer) and kept permanently cached to avoid cudaMalloc/cudaFree churn.
        // DeltaNet weights use pre-allocated buffers and don't touch this cache.
        // SKIP during graph capture: retain() drops DeviceBuffer → cudaFree, which is
        // illegal inside a CUDA graph capture region (causes error 901).
        let t_evict = Instant::now();
        if !self.capturing_graph {
            const ATTN_SEGMENTS: [u16; 6] = [4, 5, 12, 13, 27, 28];
            self.shared_tensor_cache_device.retain(|key, _| {
                let cached_layer = (*key >> 16) as u16;
                let cached_segment = (*key & 0xFFFF) as u16;
                // Keep: current layer, layer 0 (embeddings), layer 0xFFFF (final norm),
                //        and any non-attention segments from other layers (MoE weights).
                cached_layer == layer_idx
                    || cached_layer == 0
                    || cached_layer == 0xFFFF
                    || !ATTN_SEGMENTS.contains(&cached_segment)
            });
        }
        let evict_us = t_evict.elapsed().as_micros() as u64;

        let hidden_dim = self.model_config.hidden_dim as usize;
        let hidden_bytes = hidden_dim * std::mem::size_of::<f16>();
        let layer = layer_idx as usize;

        // Apply attention norm: FP32 hidden_state_f32 → RMSNorm in FP32 → FP16 hidden_state.
        //
        // CRITICAL: We must NOT do f32_to_f16 BEFORE normalization. When the FP32
        // residual stream has L2 norms in the thousands (which happens by layer 10+
        // in Kimi K2.5), casting to FP16 first destroys the relative magnitudes of
        // the hidden dimensions. Normalizing in FP32 first produces values in [-3, 3]
        // which FP16 represents with full precision.
        //
        // OPTIMIZATION: For GQA layers with NVFP4, the FP16 hidden_state is not read
        // by any production code path (NVFP4 GEMV reads from FP4 quantized data).
        // Skip the FP16 norm entirely for these layers.
        let t_norm = Instant::now();
        let eps = self.model_config.rms_norm_eps;

        // Determine if this is a GQA layer with NVFP4 active (can skip FP16 norm)
        let is_deltanet_layer = self.model_config.deltanet.as_ref()
            .map_or(false, |dn| !dn.layer_is_attention[layer]);
        let is_gqa_nvfp4 = !is_deltanet_layer && {
            // GQA layers use segment 4 for Q projection
            self.get_preassembled_nvfp4(layer_idx, 4).is_some()
        };
        // DeltaNet layers check segment 30 for QKV
        let is_dn_nvfp4 = is_deltanet_layer &&
            self.get_preassembled_nvfp4(layer_idx, 30).is_some();

        // Use pre-assembled weight (zero-copy for single-page norm tensor)
        if let Some((norm_ptr, _)) = self.get_preassembled_weight(layer_idx, 6) {
            if self.diag_enabled && layer_idx == 0 && self.position == 0 {
                self.stream.synchronize()?;
                let mut w_bytes = vec![0u8; 64];
                let _ = crate::compute::cuda_ffi::memcpy_d2h(w_bytes.as_mut_ptr(), norm_ptr as *const u8, w_bytes.len());
                let mut h_bytes = vec![0u8; hidden_dim * 4];
                self.hidden_state_f32.copy_to_host(&mut h_bytes)?;
                let h = unsafe { std::slice::from_raw_parts(h_bytes.as_ptr() as *const f32, hidden_dim) };
                let h_l2 = h.iter().map(|v| v * v).sum::<f32>().sqrt();
                tracing::info!(
                    "ATTN_NORM_WDBG preasm L{} pos={}: norm_first8={:02x?}, norm_nonzero={}/64, hidden_f32_l2={:.6}, hidden_first4=[{:.6},{:.6},{:.6},{:.6}]",
                    layer_idx,
                    self.position,
                    &w_bytes[..8],
                    w_bytes.iter().filter(|b| **b != 0).count(),
                    h_l2,
                    h[0],
                    h[1],
                    h[2],
                    h[3],
                );
            }
            if is_gqa_nvfp4 {
                // GQA NVFP4: skip FP16 norm entirely — the fused kernel in
                // try_gpu_decode_attention will produce FP4 directly from FP32.
            } else if is_dn_nvfp4 {
                // DeltaNet NVFP4: produce FP16 via the fused kernel inside
                // run_deltanet_layer. Still need FP16 for alpha/beta, so the
                // fused kernel there will produce it. But we can do it here
                // instead, producing FP16 as a side-output of the fused norm+quantize.
                // For now, keep the standard FP16 norm for DeltaNet (the fused kernel
                // in run_deltanet_layer will skip FP16 since it's already done here).
                kernels::rms_norm_f32_to_f16(
                    self.hidden_state_f32.as_ptr(),
                    self.hidden_state.as_mut_ptr(),
                    norm_ptr,
                    hidden_dim,
                    eps,
                    &self.stream,
                )?;
            } else {
                // Non-NVFP4 path: always need FP16 norm
                kernels::rms_norm_f32_to_f16(
                    self.hidden_state_f32.as_ptr(),
                    self.hidden_state.as_mut_ptr(),
                    norm_ptr,
                    hidden_dim,
                    eps,
                    &self.stream,
                )?;
            }
        } else {
            // Fallback path when pre-assembled norm is unavailable:
            // 1) try fast synchronous D2D from resident snapshot,
            // 2) fall back to async page fetch + D2D,
            // 3) finally fall back to shared tensor cache entry.
            let mut loaded_bytes = self.load_tensor_direct_resident(
                layer_idx,
                6,
                self.attn_norm_w.as_ptr() as usize,
                self.attn_norm_w.size(),
            );
            if loaded_bytes == 0 {
                loaded_bytes = self
                    .load_tensor_direct(
                        layer_idx,
                        6,
                        self.attn_norm_w.as_ptr() as usize,
                        self.attn_norm_w.size(),
                    )
                    .await;
            }

            let norm_tensor = if loaded_bytes > 0 {
                Some((self.attn_norm_w.as_ptr(), self.attn_norm_w.size()))
            } else {
                let _ = self.ensure_shared_tensor_device(layer_idx, 6).await;
                self.get_preassembled_weight(layer_idx, 6).or_else(|| {
                    let cache_key = (layer_idx as u32) << 16 | 6u32;
                    self.shared_tensor_cache_device
                        .get(&cache_key)
                        .map(|dbuf| (dbuf.as_ptr(), dbuf.size()))
                })
            };

            if let Some((norm_ptr, _)) = norm_tensor {
                if self.diag_enabled && layer_idx == 0 && self.position == 0 {
                    self.stream.synchronize()?;
                    let mut w_bytes = vec![0u8; 64];
                    let _ = crate::compute::cuda_ffi::memcpy_d2h(w_bytes.as_mut_ptr(), norm_ptr as *const u8, w_bytes.len());
                    let mut h_bytes = vec![0u8; hidden_dim * 4];
                    self.hidden_state_f32.copy_to_host(&mut h_bytes)?;
                    let h = unsafe { std::slice::from_raw_parts(h_bytes.as_ptr() as *const f32, hidden_dim) };
                    let h_l2 = h.iter().map(|v| v * v).sum::<f32>().sqrt();
                    tracing::info!(
                        "ATTN_NORM_WDBG device L{} pos={}: norm_first8={:02x?}, norm_nonzero={}/64, hidden_f32_l2={:.6}, hidden_first4=[{:.6},{:.6},{:.6},{:.6}]",
                        layer_idx,
                        self.position,
                        &w_bytes[..8],
                        w_bytes.iter().filter(|b| **b != 0).count(),
                        h_l2,
                        h[0],
                        h[1],
                        h[2],
                        h[3],
                    );
                }
                kernels::rms_norm_f32_to_f16(
                    self.hidden_state_f32.as_ptr(),
                    self.hidden_state.as_mut_ptr(),
                    norm_ptr,
                    hidden_dim,
                    eps,
                    &self.stream,
                )?;
            } else {
                // Last resort: cast then normalize without weight
                kernels::f32_to_f16(
                    self.hidden_state_f32.as_ptr(),
                    self.hidden_state.as_mut_ptr(),
                    hidden_dim,
                    &self.stream,
                )?;
                kernels::rms_norm_no_weight(
                    self.hidden_state.as_mut_ptr(),
                    hidden_dim,
                    eps,
                    &self.stream,
                )?;
            }
        }
        // Dump attn_norm output for layer 0 (all tokens)
        if self.diag_enabled && layer_idx == 0 {
            self.stream.synchronize()?;
            // Read the FP16 hidden_state (attn_norm output), convert to F32 for dumping
            let mut h_buf = vec![0u8; hidden_dim * 2];
            self.hidden_state.copy_to_host(&mut h_buf)?;
            let h_f16 = unsafe { std::slice::from_raw_parts(h_buf.as_ptr() as *const f16, hidden_dim) };
            let h_f32: Vec<f32> = h_f16.iter().map(|v| v.to_f32()).collect();
            let f32_bytes: Vec<u8> = h_f32.iter().flat_map(|v| v.to_le_bytes()).collect();
            let dump_dir = "/home/brian/code/vib3/dump";
            let _ = std::fs::create_dir_all(dump_dir);
            let dump_path = format!("{}/vib3_attn_norm_0_tok{}.bin", dump_dir, self.position);
            let _ = std::fs::write(&dump_path, &f32_bytes);
        }
        // Dump hidden_state_f32 (pre-norm) and attn_norm output for GQA layers 3/7 at early positions
        if self.diag_enabled && (layer_idx == 3 || layer_idx == 7) && self.position <= 1 {
            self.stream.synchronize()?;
            let dump_dir = "/home/brian/code/vib3/dump";
            let tok = self.position;
            // FP32 hidden state (pre-norm residual)
            let mut f32_buf = vec![0u8; hidden_dim * 4];
            self.hidden_state_f32.copy_to_host(&mut f32_buf)?;
            let f32_path = format!("{}/vib3_gqa_hidden_f32_pre_norm_L{}_tok{}.bin", dump_dir, layer_idx, tok);
            let _ = std::fs::write(&f32_path, &f32_buf);
            // FP16 attn-normed hidden state (input to Q/K/V projections)
            if !is_gqa_nvfp4 {
                let mut h_buf = vec![0u8; hidden_dim * 2];
                self.hidden_state.copy_to_host(&mut h_buf)?;
                let h_path = format!("{}/vib3_gqa_attn_normed_f16_L{}_tok{}.bin", dump_dir, layer_idx, tok);
                let _ = std::fs::write(&h_path, &h_buf);
            }
            tracing::info!("GQA DIAG L{} tok{}: dumped pre-norm hidden_f32 and attn_normed", layer_idx, tok);
        }

        // Residual buffer copy: only needed for CPU fallback attention and diagnostics.
        // Skip for DeltaNet layers (they don't use it) and when diagnostics are off.
        // Also skip for GQA NVFP4 layers (hidden_state FP16 was not produced).
        // For GQA layers, defer to just before the CPU fallback if GPU decode fails.
        if !is_deltanet_layer && !is_gqa_nvfp4 && (self.diag_enabled || !self.stream.is_real()) {
            cuda_ffi::device_memcpy_d2d_async(
                self.residual_buf.as_mut_ptr(),
                self.hidden_state.as_ptr(),
                hidden_bytes,
                &self.stream,
            )?;
        }
        if profile { self.stream.synchronize()?; }
        let norm_us = t_norm.elapsed().as_micros() as u64;

        // ── DeltaNet (Gated Delta Rule) path ────────────────────────────
        // Dispatches to DeltaNet for non-attention layers in hybrid models.
        if let Some(ref dn_config) = self.model_config.deltanet.clone() {
            if !dn_config.layer_is_attention[layer] {
                let t_dn = Instant::now();
                let result = self.run_deltanet_layer(layer_idx, &dn_config).await;
                if profile {
                    self.stream.synchronize()?;
                    let dn_us = t_dn.elapsed().as_micros() as u64;
                    if layer_idx <= 4 || layer_idx % 12 == 0 || layer_idx >= 44 {
                        tracing::info!(
                            "PROFILE DN L{}: evict={:.0}us norm={:.0}us deltanet={:.0}us total={:.0}us",
                            layer_idx, evict_us, norm_us, dn_us, evict_us + norm_us + dn_us,
                        );
                    }
                }
                return result;
            }
        }

        // ── MLA attention path ──────────────────────────────────────────
        if self.model_config.mla.is_some() && self.mla_kv_cache.is_some() {
            // Try GPU path first — hidden_state is already on device, no D2H needed.
            // Pass empty slice for hidden_state — GPU path uses self.hidden_state directly.
            let mla_result = self.run_mla_attention(layer_idx, &[]).await?;

            match mla_result {
                None => {
                    // GPU path succeeded — output is in mla_o_out_dev (FP32).
                    // FP32 residual accumulation: hidden_state_f32 += mla_o_out_dev
                    kernels::residual_add_f32_f32(
                        self.hidden_state_f32.as_mut_ptr(),
                        self.mla_o_out_dev.as_ptr(),
                        hidden_dim,
                        &self.stream,
                    )?;

                    // Diagnostic: log o_proj output magnitude during prefill and decode
                    let diag_attn = self.diag_enabled && ((self.decode_step <= 3 && (layer_idx <= 2 || layer_idx == 5 || layer_idx == 6 || layer_idx == 10 || layer_idx == 30 || layer_idx == 60))
                        || (self.decode_step == 0 && (layer_idx <= 1 || layer_idx == 5 || layer_idx == 6 || layer_idx == 10 || layer_idx == 30 || layer_idx == 60))
                        || (self.decode_step == 3 && layer_idx >= 11 && layer_idx <= 29));
                    if diag_attn {
                        self.stream.synchronize()?;
                        let f32_o_bytes = hidden_dim * 4;
                        let mut o_buf = vec![0u8; f32_o_bytes];
                        self.mla_o_out_dev.copy_to_host(&mut o_buf)?;
                        let o_f32 = unsafe {
                            std::slice::from_raw_parts(o_buf.as_ptr() as *const f32, hidden_dim)
                        };
                        let o_l2 = o_f32.iter().map(|v| v * v).sum::<f32>().sqrt();
                        let o_max = o_f32.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
                        let o_nan = o_f32.iter().filter(|v| v.is_nan()).count();
                        // Also read FP32 hidden state for comparison
                        let mut f32_buf = vec![0u8; f32_o_bytes];
                        self.hidden_state_f32.copy_to_host(&mut f32_buf)?;
                        let f32_slice = unsafe {
                            std::slice::from_raw_parts(f32_buf.as_ptr() as *const f32, hidden_dim)
                        };
                        let f32_l2 = f32_slice.iter().map(|v| v * v).sum::<f32>().sqrt();
                        tracing::info!(
                            "ATTN DIAG L{} pos={}: o_proj L2={:.4}, max_abs={:.4}, nan={}, hidden_f32_after_attn L2={:.4}",
                            layer_idx, self.position, o_l2, o_max, o_nan, f32_l2,
                        );
                    }
                }
                Some(attn_output) => {
                    // CPU fallback — upload attn_output to device, FP32 accumulate.
                    let attn_raw = unsafe {
                        std::slice::from_raw_parts(
                            attn_output.as_ptr() as *const u8,
                            hidden_bytes,
                        )
                    };
                    self.hidden_state.copy_from_host(attn_raw)?;
                    // FP32 accumulation: hidden_state_f32 += f32(attn_output)
                    kernels::residual_add_fp32(
                        self.hidden_state_f32.as_mut_ptr(),
                        self.hidden_state.as_ptr(),
                        hidden_dim,
                        &self.stream,
                    )?;
                }
            }
            return Ok(());
        }

        // ── GQA attention path ─────────────────────────────────────────
        //
        // Priority order:
        //  1. GPU decode attention — everything on GPU, zero D2H/H2D (Phase 11c)
        //  2. GPU-projected attention — projections on GPU, attention on CPU (Phase 10)
        //  3. Fully CPU path — everything on CPU (fallback)

        let num_heads = self.model_config.num_heads as usize;
        let num_kv_heads = self.model_config.num_kv_heads as usize;
        let head_dim = self.model_config.effective_head_dim() as usize;
        let is_gated_attn = self.model_config.deltanet.is_some();
        let q_dim = if is_gated_attn {
            num_heads * 2 * head_dim // Qwen3.5: Q + gate interleaved = 16384
        } else {
            hidden_dim
        };
        let kv_dim = num_kv_heads * head_dim;

        // Ensure Q/K/V/O (and gated-attn norms when needed) are present on device.
        // This handles non-fully-resident deployments where direct resident staging
        // can fail and leave stale/zero buffers.
        let t_gqa_load = Instant::now();
        let mut gpu_gqa_weights_ready = true;
        if self.stream.is_real() {
            gpu_gqa_weights_ready &= self.ensure_shared_tensor_device(layer_idx, 4).await;
            gpu_gqa_weights_ready &= self.ensure_shared_tensor_device(layer_idx, 12).await;
            gpu_gqa_weights_ready &= self.ensure_shared_tensor_device(layer_idx, 13).await;
            gpu_gqa_weights_ready &= self.ensure_shared_tensor_device(layer_idx, 5).await;
            if is_gated_attn {
                gpu_gqa_weights_ready &= self.ensure_shared_tensor_device(layer_idx, 27).await;
                gpu_gqa_weights_ready &= self.ensure_shared_tensor_device(layer_idx, 28).await;
            }
        }
        if profile { self.stream.synchronize()?; }
        let gqa_load_us = t_gqa_load.elapsed().as_micros() as u64;

        // Try 1: Fully-GPU decode attention (no CPU round-trip at all)
        let t_gqa_fwd = Instant::now();
        if self.stream.is_real() && gpu_gqa_weights_ready {
            let gpu_decode =
                self.try_gpu_decode_attention(layer_idx, hidden_dim, hidden_bytes, q_dim, kv_dim)?;
            if gpu_decode {
                if profile {
                    self.stream.synchronize()?;
                    let gqa_fwd_us = t_gqa_fwd.elapsed().as_micros() as u64;
                    tracing::info!(
                        "PROFILE GQA L{}: evict={:.0}us norm={:.0}us load={:.0}us fwd={:.0}us total={:.0}us",
                        layer_idx, evict_us, norm_us, gqa_load_us, gqa_fwd_us,
                        evict_us + norm_us + gqa_load_us + gqa_fwd_us,
                    );
                }
                return Ok(());
            }
        }

        // Try 2: GPU-projected attention (projections on GPU, attention on CPU)
        let gpu_projected = self.stream.is_real()
            && gpu_gqa_weights_ready
            && layer < self.kv_cache.layers.len()
            && self.try_gpu_attention_projection(
                layer_idx,
                hidden_dim,
                hidden_bytes,
                q_dim,
                kv_dim,
            )?;

        if gpu_projected {
            return Ok(());
        }

        // ── CPU fallback GQA path ──────────────────────────────────────
        // Used when GPU weights are unavailable or tiered KV is active.

        // Populate residual_buf now (deferred from norm step for GPU fast path)
        cuda_ffi::device_memcpy_d2d_async(
            self.residual_buf.as_mut_ptr(),
            self.hidden_state.as_ptr(),
            hidden_bytes,
            &self.stream,
        )?;

        // D2H hidden_state and residual for CPU attention
        self.stream.synchronize()?;
        self.hidden_state
            .copy_to_host(&mut self.host_staging2[..hidden_bytes])?;
        self.residual_buf
            .copy_to_host(&mut self.host_staging[..hidden_bytes])?;

        let hidden_state = unsafe {
            std::slice::from_raw_parts(self.host_staging2.as_ptr() as *const f16, hidden_dim)
        };

        let q_data = self.load_shared_tensor(layer_idx, 4).await;
        let k_data = self.load_shared_tensor(layer_idx, 12).await;
        let v_data = self.load_shared_tensor(layer_idx, 13).await;
        let o_data = self.load_shared_tensor(layer_idx, 5).await;

        let expected_q_bytes = q_dim * hidden_dim * 2;
        let expected_kv_bytes = kv_dim * hidden_dim * 2;

        let qkv_buf: Option<Vec<u8>> = match (&q_data, &k_data, &v_data) {
            (Some(q), Some(k), Some(v))
                if q.len() >= expected_q_bytes
                    && k.len() >= expected_kv_bytes
                    && v.len() >= expected_kv_bytes =>
            {
                let mut combined = Vec::with_capacity(expected_q_bytes + 2 * expected_kv_bytes);
                combined.extend_from_slice(&q[..expected_q_bytes]);
                combined.extend_from_slice(&k[..expected_kv_bytes]);
                combined.extend_from_slice(&v[..expected_kv_bytes]);
                Some(combined)
            }
            (Some(q), _, _) => Some(q.clone()),
            _ => None,
        };

        let qkv_w = qkv_buf.as_ref().map(|buf| unsafe {
            let n_elements = buf.len() / std::mem::size_of::<f16>();
            std::slice::from_raw_parts(buf.as_ptr() as *const f16, n_elements)
        });
        let o_w = o_data.as_ref().map(|buf| unsafe {
            let n_elements = buf.len() / std::mem::size_of::<f16>();
            std::slice::from_raw_parts(buf.as_ptr() as *const f16, n_elements)
        });

        let attn_output = if let Some(ref mut tiered_kv) = self.tiered_kv {
            self_attention_tiered(
                hidden_state,
                qkv_w,
                o_w,
                tiered_kv,
                layer,
                self.position,
                &self.model_config,
            )
        } else if layer < self.kv_cache.layers.len() {
            self_attention_layer(
                hidden_state,
                qkv_w,
                o_w,
                &mut self.kv_cache.layers[layer],
                self.position,
                &self.model_config,
            )
        } else {
            return Ok(());
        };

        // FP32 residual: upload attn_output to device, accumulate in FP32
        let attn_raw = unsafe {
            std::slice::from_raw_parts(attn_output.as_ptr() as *const u8, hidden_bytes)
        };
        self.hidden_state.copy_from_host(attn_raw)?;
        kernels::residual_add_fp32(
            self.hidden_state_f32.as_mut_ptr(),
            self.hidden_state.as_ptr(),
            hidden_dim,
            &self.stream,
        )?;

        Ok(())
    }

    /// Run DeltaNet (Gated Delta Rule) attention for a single layer.
    ///
    /// This implements the full DeltaNet autoregressive step:
    ///   1. QKV projection: hidden → [Q, K, V] via in_proj_qkv (segment 30)
    ///   2. Conv1d + SiLU: causal depthwise conv on QKV (segment 34)
    ///   3. Q/K/V split from convolved output
    ///   4. L2 normalize Q and K
    ///   5. Repeat Q/K from num_key_heads → num_value_heads (tiled)
    ///   6. Alpha projection → gate computation (segments 33, 35, 36)
    ///   7. Beta projection → sigmoid (segment 32)
    ///   8. DeltaNet step: decay → retrieve → delta → update → query
    ///   9. Z gate projection (segment 31)
    ///  10. Gated RMSNorm: RMSNorm(step_out, norm_weight) * SiLU(z) (segment 37)
    ///  11. Output projection: inner_dim → hidden_dim (segment 38)
    ///  12. Residual add: hidden_state_f32 += o_proj output
    ///
    /// Called from `run_attention_layer` when the layer is a DeltaNet layer.
    /// The attn_norm has already been applied: `self.hidden_state` contains
    /// the FP16 normalized input, `self.hidden_state_f32` holds the FP32 residual.
    async fn run_deltanet_layer(
        &mut self,
        layer_idx: u16,
        dn_config: &crate::core::config::DeltaNetConfig,
    ) -> Result<()> {
        // No eviction or cudaMalloc needed — use pre-allocated weight staging buffers.
        // D2D copies from T1 pages into fixed buffers (~168 MB total, reused per layer).

        let hidden_dim = self.model_config.hidden_dim as usize;
        let layer = layer_idx as usize;
        let num_key_heads = dn_config.num_key_heads as usize;
        let num_v_heads = dn_config.num_value_heads as usize;
        let key_head_dim = dn_config.key_head_dim as usize;
        let v_head_dim = dn_config.value_head_dim as usize;
        let key_dim = num_key_heads * key_head_dim; // 2048
        let inner_dim = dn_config.inner_dim as usize; // 8192
        let qkv_dim = dn_config.qkv_dim() as usize; // 12288
        let conv_kernel = dn_config.conv_kernel_size as usize; // 4
        let eps = self.model_config.rms_norm_eps;

        // Map model layer index to DeltaNet state index (skip attention layers)
        let dn_state_idx = dn_config.layer_is_attention[..layer]
            .iter()
            .filter(|&&is_attn| !is_attn)
            .count();

        let profile = self.diag_enabled && self.decode_step == 1;

        // ── 1. Load DeltaNet weights — use pre-assembled zero-copy if available ──
        // For non-preassembled layers, we use paged GEMV for large multi-page tensors
        // (segments 30=QKV, 31=Z, 38=out) to eliminate D2D staging copies.
        // Small single-page tensors still use D2D staging.
        let t_load = Instant::now();
        // DeltaNet NVFP4 path is opt-in due observed accuracy regressions on Qwen3.5.
        // Default to FP16 for correctness; enable via env for experiments.
        let nvfp4_deltanet_requested = std::env::var("VIB3_NVFP4_DELTANET").map_or(false, |v| v == "1");
        let is_qwen35_deltanet = self.model_config.architecture == "qwen3_5_moe";
        // Qwen3.5 DeltaNet correctness currently relies on staged FP16 weights.
        // Pre-assembled FP16 pointers for segments 30/31/38 can regress quality.
        let disable_preassembled_deltanet_fp16 = is_qwen35_deltanet;
        // FP32-input DeltaNet projections are experimental and can regress quality
        // on some checkpoints. Keep this opt-in via env var.
        let use_f32_deltanet_proj = std::env::var("VIB3_F32_DELTANET_PROJ")
            .map_or(false, |v| v == "1");
        if use_f32_deltanet_proj && layer_idx == 0 && self.position == 0 {
            tracing::warn!(
                "VIB3_F32_DELTANET_PROJ=1 enabled (experimental); disable if output quality regresses"
            );
        }
        let allow_unsafe_qwen35_nvfp4 =
            std::env::var("VIB3_ALLOW_UNSAFE_DELTANET_NVFP4").map_or(false, |v| v == "1");
        let use_nvfp4_deltanet = nvfp4_deltanet_requested
            && (!is_qwen35_deltanet || allow_unsafe_qwen35_nvfp4);
        if nvfp4_deltanet_requested
            && is_qwen35_deltanet
            && !allow_unsafe_qwen35_nvfp4
            && layer_idx == 0
            && self.position == 0
        {
            tracing::warn!(
                "VIB3_NVFP4_DELTANET=1 ignored for qwen3_5_moe due known severe quality regression; set VIB3_ALLOW_UNSAFE_DELTANET_NVFP4=1 to override"
            );
        }
        let use_nvfp4_deltanet_qkv = use_nvfp4_deltanet
            && std::env::var("VIB3_NVFP4_DELTANET_QKV").map_or(true, |v| v == "1");
        let use_nvfp4_deltanet_z = use_nvfp4_deltanet
            && std::env::var("VIB3_NVFP4_DELTANET_Z").map_or(true, |v| v == "1");
        let use_nvfp4_deltanet_out = use_nvfp4_deltanet
            && std::env::var("VIB3_NVFP4_DELTANET_OUT").map_or(true, |v| v == "1");
        let use_scalar_dn_nvfp4 = use_nvfp4_deltanet
            && std::env::var("VIB3_NVFP4_DELTANET_SCALAR").map_or(false, |v| v == "1");
        let diag_nvfp4_compare = self.diag_enabled
            && std::env::var("VIB3_DIAG_NVFP4_COMPARE").map_or(false, |v| v == "1")
            && self.position <= 1;
        let has_qkv_fp16_preassembled = !disable_preassembled_deltanet_fp16
            && self.get_preassembled_weight(layer_idx, 30).is_some();
        let has_qkv_nvfp4_preassembled = use_nvfp4_deltanet_qkv
            && self.get_preassembled_nvfp4(layer_idx, 30).is_some();
        let has_z_fp16_preassembled = !disable_preassembled_deltanet_fp16
            && self.get_preassembled_weight(layer_idx, 31).is_some();
        let has_z_nvfp4_preassembled = use_nvfp4_deltanet_z
            && self.get_preassembled_nvfp4(layer_idx, 31).is_some();
        let has_out_fp16_preassembled = !disable_preassembled_deltanet_fp16
            && self.get_preassembled_weight(layer_idx, 38).is_some();
        let has_out_nvfp4_preassembled = use_nvfp4_deltanet_out
            && self.get_preassembled_nvfp4(layer_idx, 38).is_some();

        let mut qkv_staged_bytes = 0usize;
        if !has_qkv_fp16_preassembled && !has_qkv_nvfp4_preassembled {
            // Try fast resident snapshot first; if pages aren't resident, fall back to
            // async page loading so staging buffers are never left stale/zero.
            qkv_staged_bytes = self.load_tensor_direct_resident(
                layer_idx,
                30,
                self.dn_w_qkv.as_mut_ptr() as usize,
                self.dn_w_qkv.size(),
            );
            if qkv_staged_bytes == 0 {
                qkv_staged_bytes = self.load_tensor_direct(
                    layer_idx,
                    30,
                    self.dn_w_qkv.as_mut_ptr() as usize,
                    self.dn_w_qkv.size(),
                )
                .await;
            }
        }

        let mut z_staged_bytes = 0usize;
        if !has_z_fp16_preassembled && !has_z_nvfp4_preassembled {
            z_staged_bytes = self.load_tensor_direct_resident(
                layer_idx,
                31,
                self.dn_w_z.as_mut_ptr() as usize,
                self.dn_w_z.size(),
            );
            if z_staged_bytes == 0 {
                z_staged_bytes = self.load_tensor_direct(
                    layer_idx,
                    31,
                    self.dn_w_z.as_mut_ptr() as usize,
                    self.dn_w_z.size(),
                )
                .await;
            }
        }

        let mut out_staged_bytes = 0usize;
        if !has_out_fp16_preassembled && !has_out_nvfp4_preassembled {
            out_staged_bytes = self.load_tensor_direct_resident(
                layer_idx,
                38,
                self.dn_w_out.as_mut_ptr() as usize,
                self.dn_w_out.size(),
            );
            if out_staged_bytes == 0 {
                out_staged_bytes = self.load_tensor_direct(
                    layer_idx,
                    38,
                    self.dn_w_out.as_mut_ptr() as usize,
                    self.dn_w_out.size(),
                )
                .await;
            }
        }

        // Small segments: stage only when preassembled pointers are unavailable.
        if self.get_preassembled_weight(layer_idx, 32).is_none() {
            if self.load_tensor_direct_resident(
                layer_idx,
                32,
                self.dn_w_beta.as_mut_ptr() as usize,
                self.dn_w_beta.size(),
            ) == 0
            {
                self.load_tensor_direct(
                    layer_idx,
                    32,
                    self.dn_w_beta.as_mut_ptr() as usize,
                    self.dn_w_beta.size(),
                )
                .await;
            }
        }

        if self.get_preassembled_weight(layer_idx, 33).is_none() {
            if self.load_tensor_direct_resident(
                layer_idx,
                33,
                self.dn_w_alpha.as_mut_ptr() as usize,
                self.dn_w_alpha.size(),
            ) == 0
            {
                self.load_tensor_direct(
                    layer_idx,
                    33,
                    self.dn_w_alpha.as_mut_ptr() as usize,
                    self.dn_w_alpha.size(),
                )
                .await;
            }
        }

        if self.get_preassembled_weight(layer_idx, 34).is_none() {
            if self.load_tensor_direct_resident(
                layer_idx,
                34,
                self.dn_w_conv.as_mut_ptr() as usize,
                self.dn_w_conv.size(),
            ) == 0
            {
                self.load_tensor_direct(
                    layer_idx,
                    34,
                    self.dn_w_conv.as_mut_ptr() as usize,
                    self.dn_w_conv.size(),
                )
                .await;
            }
        }

        if self.get_preassembled_weight(layer_idx, 35).is_none() {
            if self.load_tensor_direct_resident(
                layer_idx,
                35,
                self.dn_w_dt_bias.as_mut_ptr() as usize,
                self.dn_w_dt_bias.size(),
            ) == 0
            {
                self.load_tensor_direct(
                    layer_idx,
                    35,
                    self.dn_w_dt_bias.as_mut_ptr() as usize,
                    self.dn_w_dt_bias.size(),
                )
                .await;
            }
        }

        if self.get_preassembled_weight(layer_idx, 36).is_none() {
            if self.load_tensor_direct_resident(
                layer_idx,
                36,
                self.dn_w_a_log.as_mut_ptr() as usize,
                self.dn_w_a_log.size(),
            ) == 0
            {
                self.load_tensor_direct(
                    layer_idx,
                    36,
                    self.dn_w_a_log.as_mut_ptr() as usize,
                    self.dn_w_a_log.size(),
                )
                .await;
            }
        }

        if self.get_preassembled_weight(layer_idx, 37).is_none() {
            if self.load_tensor_direct_resident(
                layer_idx,
                37,
                self.dn_w_norm.as_mut_ptr() as usize,
                self.dn_w_norm.size(),
            ) == 0
            {
                self.load_tensor_direct(
                    layer_idx,
                    37,
                    self.dn_w_norm.as_mut_ptr() as usize,
                    self.dn_w_norm.size(),
                )
                .await;
            }
        }
        if profile { self.stream.synchronize()?; }
        let load_us = t_load.elapsed().as_micros() as u64;

        let qkv_fp16_ptr = self
            .get_preassembled_weight(layer_idx, 30)
            .map(|(p, _)| p)
            .filter(|_| !disable_preassembled_deltanet_fp16)
            .or_else(|| {
                if qkv_staged_bytes > 0 {
                    Some(self.dn_w_qkv.as_ptr())
                } else {
                    None
                }
            })
            .or_else(|| {
                if disable_preassembled_deltanet_fp16 {
                    None
                } else {
                    self.get_device_tensor(layer_idx, 30).map(|(p, _)| p as *const u8)
                }
            });
        let z_fp16_ptr = self
            .get_preassembled_weight(layer_idx, 31)
            .map(|(p, _)| p)
            .filter(|_| !disable_preassembled_deltanet_fp16)
            .or_else(|| {
                if z_staged_bytes > 0 {
                    Some(self.dn_w_z.as_ptr())
                } else {
                    None
                }
            })
            .or_else(|| {
                if disable_preassembled_deltanet_fp16 {
                    None
                } else {
                    self.get_device_tensor(layer_idx, 31).map(|(p, _)| p as *const u8)
                }
            });
        let out_fp16_ptr = self
            .get_preassembled_weight(layer_idx, 38)
            .map(|(p, _)| p)
            .filter(|_| !disable_preassembled_deltanet_fp16)
            .or_else(|| {
                if out_staged_bytes > 0 {
                    Some(self.dn_w_out.as_ptr())
                } else {
                    None
                }
            })
            .or_else(|| {
                if disable_preassembled_deltanet_fp16 {
                    None
                } else {
                    self.get_device_tensor(layer_idx, 38).map(|(p, _)| p as *const u8)
                }
            });

        // Get device pointers for small segments — prefer pre-assembled, fallback to staging
        let beta_w_ptr = self.get_preassembled_weight(layer_idx, 32)
            .map(|(p, _)| p).unwrap_or(self.dn_w_beta.as_ptr());
        let alpha_w_ptr = self.get_preassembled_weight(layer_idx, 33)
            .map(|(p, _)| p).unwrap_or(self.dn_w_alpha.as_ptr());
        let conv_w_ptr = self.get_preassembled_weight(layer_idx, 34)
            .map(|(p, _)| p).unwrap_or(self.dn_w_conv.as_ptr());
        let dt_bias_ptr = self.get_preassembled_weight(layer_idx, 35)
            .map(|(p, _)| p).unwrap_or(self.dn_w_dt_bias.as_ptr());
        let a_ptr = self.get_preassembled_weight(layer_idx, 36)
            .map(|(p, _)| p).unwrap_or(self.dn_w_a_log.as_ptr());
        let norm_w_ptr = self.get_preassembled_weight(layer_idx, 37)
            .map(|(p, _)| p).unwrap_or(self.dn_w_norm.as_ptr());

        if self.diag_enabled && layer_idx == 0 && self.position == 0 {
            let qkv_dbg = if has_qkv_nvfp4_preassembled {
                self.get_preassembled_nvfp4(layer_idx, 30).map(|(p, _, _)| ("nvfp4", p))
            } else {
                qkv_fp16_ptr.map(|p| ("fp16", p))
            };
            let z_dbg = if has_z_nvfp4_preassembled {
                self.get_preassembled_nvfp4(layer_idx, 31).map(|(p, _, _)| ("nvfp4", p))
            } else {
                z_fp16_ptr.map(|p| ("fp16", p))
            };
            let out_dbg = if has_out_nvfp4_preassembled {
                self.get_preassembled_nvfp4(layer_idx, 38).map(|(p, _, _)| ("nvfp4", p))
            } else {
                out_fp16_ptr.map(|p| ("fp16", p))
            };

            self.stream.synchronize()?;
            for (name, ptr) in [
                ("qkv", qkv_dbg),
                ("z", z_dbg),
                ("out", out_dbg),
            ] {
                let Some((path, ptr)) = ptr else {
                    tracing::warn!(
                        "DELTANET_WDBG L{} pos={}: {} missing (no fp16/nvfp4 pointer)",
                        layer_idx,
                        self.position,
                        name,
                    );
                    continue;
                };
                let mut buf = vec![0u8; 64];
                let _ = crate::compute::cuda_ffi::memcpy_d2h(buf.as_mut_ptr(), ptr as *const u8, buf.len());
                let nonzero = buf.iter().filter(|b| **b != 0).count();
                tracing::info!(
                    "DELTANET_WDBG L{} pos={}: {} path={} first8={:02x?}, nonzero={}/{}",
                    layer_idx,
                    self.position,
                    name,
                    path,
                    &buf[..8],
                    nonzero,
                    buf.len(),
                );
            }
        }

        // ── 2. QKV + Z projections (batched when both NVFP4) ──
        // Check if weight is NVFP4 (4x bandwidth reduction via MMA GEMV)
        let t_qkv = Instant::now();
        let qkv_nvfp4 = if use_nvfp4_deltanet_qkv {
            self.get_preassembled_nvfp4(layer_idx, 30)
        } else {
            None
        };
        let z_nvfp4 = if use_nvfp4_deltanet_z {
            self.get_preassembled_nvfp4(layer_idx, 31)
        } else {
            None
        };
        // DeltaNet batched NVFP4 QKV+Z is opt-in while correctness is validated.
        // Default uses single GEMV launches for QKV and Z independently.
        let use_batched_deltanet_nvfp4 =
            std::env::var("VIB3_BATCHED_DELTANET_NVFP4").map_or(false, |v| v == "1");
        let batched_qkv_z = use_batched_deltanet_nvfp4 && qkv_nvfp4.is_some() && z_nvfp4.is_some();
        let qkv_norm_ptr = self
            .get_preassembled_weight(layer_idx, 6)
            .map(|(p, _)| p)
            .or_else(|| {
                self.get_device_tensor(layer_idx, 6)
                    .map(|(p, _)| p as *const u8)
            });
        if let Some((nvfp4_ptr, _m, nvfp4_k)) = qkv_nvfp4 {
            if let Some(norm_ptr) = qkv_norm_ptr {
                // DeltaNet NVFP4 activation prep: default to non-fused RMSNorm+quantize
                // while fused path correctness is being validated.
                let use_fused_dn_nvfp4_norm =
                    std::env::var("VIB3_FUSED_DELTANET_NVFP4_NORM").map_or(false, |v| v == "1");
                let eps = self.model_config.rms_norm_eps;
                if use_fused_dn_nvfp4_norm {
                    kernels::fused_rms_norm_quantize_fp4(
                        self.hidden_state_f32.as_ptr(),
                        norm_ptr,
                        self.preq_act_fp4.as_mut_ptr(),
                        self.preq_act_scales.as_mut_ptr(),
                        std::ptr::null_mut(), // FP16 already produced by shared norm
                        nvfp4_k,
                        eps,
                        &self.stream,
                    )?;
                } else {
                    kernels::rms_norm_f32(
                        self.hidden_state_f32.as_ptr(),
                        self.attn_normed_f32.as_mut_ptr(),
                        norm_ptr,
                        nvfp4_k,
                        eps,
                        &self.stream,
                    )?;
                    kernels::quantize_activation_fp4_fast(
                        self.attn_normed_f32.as_ptr(),
                        self.preq_act_fp4.as_mut_ptr(),
                        self.preq_act_scales.as_mut_ptr(),
                        nvfp4_k,
                        &self.stream,
                    )?;
                }
                if batched_qkv_z && !use_scalar_dn_nvfp4 {
                    // Batched: QKV + Z in a single kernel launch (same FP4 activation, same K)
                    let (z_nvfp4_ptr, _z_m, _z_k) = z_nvfp4.unwrap();
                    let weight_pages = [nvfp4_ptr, z_nvfp4_ptr];
                    let m_slices = [qkv_dim as i32, inner_dim as i32];
                    let mut outputs = [
                        self.dn_qkv_f32_dev.as_mut_ptr(),
                        self.dn_z_f32_dev.as_mut_ptr(),
                    ];
                    kernels::batched_gemv_mma_nvfp4_tiled(
                        self.preq_act_fp4.as_ptr(),
                        self.preq_act_scales.as_ptr(),
                        &weight_pages,
                        &m_slices,
                        &mut outputs,
                        nvfp4_k,
                        &self.stream,
                    )?;
                } else {
                    // Single QKV GEMV (Z will be done separately later)
                    if use_scalar_dn_nvfp4 {
                        kernels::gemv_scalar_nvfp4(
                            self.preq_act_fp4.as_ptr(),
                            self.preq_act_scales.as_ptr(),
                            nvfp4_ptr,
                            self.dn_qkv_f32_dev.as_mut_ptr(),
                            nvfp4_k,
                            qkv_dim,
                            &self.stream,
                        )?;
                    } else {
                        kernels::gemv_mma_nvfp4_tiled(
                            self.preq_act_fp4.as_ptr(),
                            self.preq_act_scales.as_ptr(),
                            nvfp4_ptr,
                            self.dn_qkv_f32_dev.as_mut_ptr(),
                            nvfp4_k,
                            qkv_dim,
                            &self.stream,
                        )?;
                    }
                }
            } else {
                tracing::warn!(
                    "DeltaNet L{}: NVFP4 QKV available but norm tensor (segment 6) missing; falling back to FP16 projection",
                    layer_idx
                );
                let qkv_w_ptr = qkv_fp16_ptr.ok_or_else(|| {
                    Error::ConfigError(format!(
                        "DeltaNet L{} missing QKV weight pointer (segment 30)",
                        layer_idx
                    ))
                })?;
                kernels::linear_projection(
                    self.hidden_state.as_ptr(),
                    qkv_w_ptr,
                    self.dn_qkv_proj_dev.as_mut_ptr(),
                    hidden_dim,
                    qkv_dim,
                    &self.stream,
                )?;
                kernels::f16_to_f32(
                    self.dn_qkv_proj_dev.as_ptr(),
                    self.dn_qkv_f32_dev.as_mut_ptr(),
                    qkv_dim,
                    &self.stream,
                )?;
            }
        } else {
            let qkv_w_ptr = qkv_fp16_ptr.ok_or_else(|| {
                Error::ConfigError(format!(
                    "DeltaNet L{} missing QKV weight pointer (segment 30)",
                    layer_idx
                ))
            })?;
            if use_f32_deltanet_proj {
                kernels::linear_projection_f32_to_f32(
                    self.hidden_state_f32.as_ptr(),
                    qkv_w_ptr,
                    self.dn_qkv_f32_dev.as_mut_ptr(),
                    hidden_dim,
                    qkv_dim,
                    &self.stream,
                )?;
            } else {
                kernels::linear_projection(
                    self.hidden_state.as_ptr(),
                    qkv_w_ptr,
                    self.dn_qkv_proj_dev.as_mut_ptr(),
                    hidden_dim,
                    qkv_dim,
                    &self.stream,
                )?;
                // Convert QKV from FP16 to FP32 (DeltaNet operates entirely in FP32)
                kernels::f16_to_f32(
                    self.dn_qkv_proj_dev.as_ptr(),
                    self.dn_qkv_f32_dev.as_mut_ptr(),
                    qkv_dim,
                    &self.stream,
                )?;
            }
        }
        if diag_nvfp4_compare && qkv_nvfp4.is_some() {
            let qkv_ref_ptr = qkv_fp16_ptr.or_else(|| {
                if self.load_tensor_direct_resident(
                    layer_idx,
                    30,
                    self.dn_w_qkv.as_mut_ptr() as usize,
                    self.dn_w_qkv.size(),
                ) > 0 {
                    Some(self.dn_w_qkv.as_ptr())
                } else {
                    None
                }
            });
            if let Some(qkv_w_ptr) = qkv_ref_ptr {
                kernels::linear_projection(
                    self.hidden_state.as_ptr(),
                    qkv_w_ptr,
                    self.dn_qkv_proj_dev.as_mut_ptr(),
                    hidden_dim,
                    qkv_dim,
                    &self.stream,
                )?;
                kernels::f16_to_f32(
                    self.dn_qkv_proj_dev.as_ptr(),
                    self.nvfp4_f32_scratch.as_mut_ptr(),
                    qkv_dim,
                    &self.stream,
                )?;
                self.log_projection_compare(
                    "dn_qkv",
                    layer_idx,
                    self.nvfp4_f32_scratch.as_ptr(),
                    self.dn_qkv_f32_dev.as_ptr(),
                    qkv_dim,
                )?;
            }
        }
        if profile { self.stream.synchronize()?; }
        let qkv_us = t_qkv.elapsed().as_micros() as u64;

        // Dump layer 0 intermediates for all tokens
        if self.diag_enabled && layer_idx == 0 {
            self.stream.synchronize()?;
            let dump_dir = "/home/brian/code/vib3/dump";
            let _ = std::fs::create_dir_all(dump_dir);
            let tok = self.position;
            // QKV projection (FP32, qkv_dim=8192)
            {
                let mut buf = vec![0u8; qkv_dim * 4];
                self.dn_qkv_f32_dev.copy_to_host(&mut buf)?;
                let path = format!("{}/vib3_qkv_f32_L0_tok{}.bin", dump_dir, tok);
                let _ = std::fs::write(&path, &buf);
            }
        }

        // ── 3. Conv1d + SiLU on QKV ──
        let t_mid = Instant::now();
        // Convert conv1d weights from FP16 to FP32 into a temp buffer.
        // conv1d weight: [qkv_dim, conv_kernel] FP16 stored on device.
        // We need FP32 for the kernel. Reuse dn_conv_out_dev temporarily for the weight conversion.
        // Actually, we need conv_w in FP32 AND conv_out simultaneously, so let's use a small
        // on-stack approach: the conv weight is qkv_dim * conv_kernel * 4 = 12288 * 4 * 4 = 196608 bytes.
        // We'll convert in-place by using dn_conv_out_dev for the FP32 conv weight, then do conv1d
        // into a portion of dn_qkv_f32_dev (which we've already consumed).
        //
        // Wait — the causal_conv1d kernel takes FP32 inputs and writes FP32 output.
        // We need: FP32 conv_state, FP32 new_input (from qkv_f32), FP32 conv_weight, FP32 output.
        // The conv_weight is small enough to convert on the fly.
        // Let's use a dedicated conversion: we have dn_conv_out_dev as [qkv_dim * 4] bytes = 49152 bytes.
        // Conv weight is [qkv_dim * conv_kernel] = 12288 * 4 = 49152 FP16 values = 98304 bytes → 196608 F32 bytes.
        // That's larger than dn_conv_out_dev. We need a different approach.
        //
        // Strategy: The conv weight is static per layer, so we can convert it to F32 once and cache it.
        // For now, let's convert it to F32 into a temporary host-side buffer, upload, and cache.
        // Actually simpler: just ensure the weight is FP32 on device by using a cache.
        //
        // SIMPLEST approach for now: we have dn_norm_out_dev (inner_dim * 4 = 32768 bytes) which
        // is not used until step 10. And the conv weight is 196608 bytes. That's too large.
        //
        // Let's just allocate a temporary device buffer for the FP32 conv weight.
        // This is called once per layer per token, so the cost is minimal.
        // Convert conv1d weights FP16→FP32 into pre-allocated buffer (no cudaMalloc)
        kernels::f16_to_f32(
            conv_w_ptr,
            self.dn_conv_w_f32_dev.as_mut_ptr(),
            qkv_dim * conv_kernel,
            &self.stream,
        )?;

        // Dump conv1d weight (FP32) for layer 0
        if self.diag_enabled && layer_idx == 0 && self.position == 0 {
            self.stream.synchronize()?;
            let dump_dir = "/home/brian/code/vib3/dump";
            let mut buf = vec![0u8; qkv_dim * conv_kernel * 4];
            self.dn_conv_w_f32_dev.copy_to_host(&mut buf)?;
            let path = format!("{}/vib3_conv1d_weight_L0.bin", dump_dir);
            let _ = std::fs::write(&path, &buf);
            tracing::info!("Dumped conv1d weight FP32 to {} ({} bytes)", path, buf.len());
        }

        // Run causal conv1d: updates conv_state in-place, outputs convolved+SiLU result
        kernels::causal_conv1d(
            self.dn_conv_state[dn_state_idx].as_mut_ptr(),
            self.dn_qkv_f32_dev.as_ptr(), // new_input: FP32 QKV
            self.dn_conv_w_f32_dev.as_ptr(), // conv_weight: FP32 (pre-allocated)
            self.dn_conv_out_dev.as_mut_ptr(), // output: FP32 [qkv_dim]
            qkv_dim,
            conv_kernel,
            &self.stream,
        )?;

        // Dump conv1d output for layer 0
        if self.diag_enabled && layer_idx == 0 {
            self.stream.synchronize()?;
            let dump_dir = "/home/brian/code/vib3/dump";
            let tok = self.position;
            let mut buf = vec![0u8; qkv_dim * 4];
            self.dn_conv_out_dev.copy_to_host(&mut buf)?;
            let path = format!("{}/vib3_conv_silu_L0_tok{}.bin", dump_dir, tok);
            let _ = std::fs::write(&path, &buf);
        }

        // ── 4. Split Q/K/V from conv output ──
        // Layout: [Q: key_dim=2048, K: key_dim=2048, V: inner_dim=8192]
        // All in FP32, contiguous in dn_conv_out_dev.
        let q_ptr = self.dn_conv_out_dev.as_ptr(); // offset 0
        let k_ptr = unsafe { self.dn_conv_out_dev.as_ptr().add(key_dim * 4) }; // offset key_dim*4
        let v_ptr = unsafe { self.dn_conv_out_dev.as_ptr().add(key_dim * 2 * 4) }; // offset 2*key_dim*4

        // ── 5. L2 normalize Q and K ──
        // Q: [num_key_heads, key_head_dim] → L2 norm per head
        // K: [num_key_heads, key_head_dim] → L2 norm per head
        kernels::l2_norm(
            q_ptr,
            self.dn_q_norm_dev.as_mut_ptr(),
            num_key_heads,
            key_head_dim,
            1e-12,
            &self.stream,
        )?;
        kernels::l2_norm(
            k_ptr,
            self.dn_k_norm_dev.as_mut_ptr(),
            num_key_heads,
            key_head_dim,
            1e-12,
            &self.stream,
        )?;

        // ── 6. Repeat Q/K from num_key_heads → num_value_heads (tiled) ──
        let repeat_factor = num_v_heads / num_key_heads; // 64 / 16 = 4
        kernels::repeat_tile_f32(
            self.dn_q_norm_dev.as_ptr(),
            self.dn_q_rep_dev.as_mut_ptr(),
            num_key_heads,
            key_head_dim,
            repeat_factor,
            &self.stream,
        )?;
        kernels::repeat_tile_f32(
            self.dn_k_norm_dev.as_ptr(),
            self.dn_k_rep_dev.as_mut_ptr(),
            num_key_heads,
            key_head_dim,
            repeat_factor,
            &self.stream,
        )?;

        // ── 6b. Scale Q by 1/sqrt(key_head_dim) ──
        // llama.cpp: q = ggml_scale(ctx0, q, 1.0f / sqrtf(S_k))  (delta-net-base.cpp:321)
        kernels::scale_f32(
            self.dn_q_rep_dev.as_mut_ptr(),
            num_v_heads * key_head_dim,
            1.0 / (key_head_dim as f32).sqrt(),
            &self.stream,
        )?;

        // Dump Q/K after repeat+scale for layer 0
        if self.diag_enabled && layer_idx == 0 {
            self.stream.synchronize()?;
            let dump_dir = "/home/brian/code/vib3/dump";
            let tok = self.position;
            let qk_size = num_v_heads * key_head_dim; // 32 * 128 = 4096
            // Q (after repeat + scale)
            let mut buf = vec![0u8; qk_size * 4];
            self.dn_q_rep_dev.copy_to_host(&mut buf)?;
            let path = format!("{}/vib3_q_predelta_L0_tok{}.bin", dump_dir, tok);
            let _ = std::fs::write(&path, &buf);
            // K (after repeat)
            self.dn_k_rep_dev.copy_to_host(&mut buf)?;
            let path = format!("{}/vib3_k_predelta_L0_tok{}.bin", dump_dir, tok);
            let _ = std::fs::write(&path, &buf);
        }

        // ── 7. Alpha projection → gate computation ──
        // Alpha: hidden_state (FP16) × alpha_weight → [num_v_heads] FP16 → FP32
        if use_f32_deltanet_proj {
            kernels::linear_projection_f32_to_f32(
                self.hidden_state_f32.as_ptr(),
                alpha_w_ptr,
                self.dn_alpha_f32_dev.as_mut_ptr(),
                hidden_dim,
                num_v_heads,
                &self.stream,
            )?;
        } else {
            kernels::linear_projection(
                self.hidden_state.as_ptr(),
                alpha_w_ptr,
                self.dn_alpha_proj_dev.as_mut_ptr(),
                hidden_dim,
                num_v_heads,
                &self.stream,
            )?;
            kernels::f16_to_f32(
                self.dn_alpha_proj_dev.as_ptr(),
                self.dn_alpha_f32_dev.as_mut_ptr(),
                num_v_heads,
                &self.stream,
            )?;
        }

        // Convert dt_bias and A from FP16 to FP32 (small: num_v_heads each)
        kernels::f16_to_f32(
            dt_bias_ptr,
            self.dn_dt_bias_f32_dev.as_mut_ptr(),
            num_v_heads,
            &self.stream,
        )?;
        kernels::f16_to_f32(
            a_ptr,
            self.dn_a_f32_dev.as_mut_ptr(),
            num_v_heads,
            &self.stream,
        )?;

        // Gate = A * softplus(alpha + dt_bias)
        kernels::deltanet_gate(
            self.dn_alpha_f32_dev.as_ptr(),
            self.dn_dt_bias_f32_dev.as_ptr(),
            self.dn_a_f32_dev.as_ptr(),
            self.dn_gate_dev.as_mut_ptr(),
            num_v_heads,
            &self.stream,
        )?;

        // Dump gate for layer 0
        if self.diag_enabled && layer_idx == 0 {
            self.stream.synchronize()?;
            let dump_dir = "/home/brian/code/vib3/dump";
            let tok = self.position;
            // Gate [num_v_heads] F32
            let mut buf = vec![0u8; num_v_heads * 4];
            self.dn_gate_dev.copy_to_host(&mut buf)?;
            let path = format!("{}/vib3_gate_L0_tok{}.bin", dump_dir, tok);
            let _ = std::fs::write(&path, &buf);
            // Alpha F32 [num_v_heads]
            let mut buf2 = vec![0u8; num_v_heads * 4];
            self.dn_alpha_f32_dev.copy_to_host(&mut buf2)?;
            let path2 = format!("{}/vib3_alpha_L0_tok{}.bin", dump_dir, tok);
            let _ = std::fs::write(&path2, &buf2);
        }

        // ── 8. Beta projection → sigmoid ──
        // Beta: hidden_state (FP16) × beta_weight → [num_v_heads] FP16 → FP32 → sigmoid
        if use_f32_deltanet_proj {
            kernels::linear_projection_f32_to_f32(
                self.hidden_state_f32.as_ptr(),
                beta_w_ptr,
                self.dn_beta_f32_dev.as_mut_ptr(),
                hidden_dim,
                num_v_heads,
                &self.stream,
            )?;
        } else {
            kernels::linear_projection(
                self.hidden_state.as_ptr(),
                beta_w_ptr,
                self.dn_beta_proj_dev.as_mut_ptr(),
                hidden_dim,
                num_v_heads,
                &self.stream,
            )?;
            kernels::f16_to_f32(
                self.dn_beta_proj_dev.as_ptr(),
                self.dn_beta_f32_dev.as_mut_ptr(),
                num_v_heads,
                &self.stream,
            )?;
        }
        kernels::sigmoid(
            self.dn_beta_f32_dev.as_ptr(),
            self.dn_beta_f32_dev.as_mut_ptr(), // in-place
            num_v_heads,
            &self.stream,
        )?;

        // Dump beta (post-sigmoid) for layer 0
        if self.diag_enabled && layer_idx == 0 {
            self.stream.synchronize()?;
            let dump_dir = "/home/brian/code/vib3/dump";
            let tok = self.position;
            let mut buf = vec![0u8; num_v_heads * 4];
            self.dn_beta_f32_dev.copy_to_host(&mut buf)?;
            let path = format!("{}/vib3_beta_L0_tok{}.bin", dump_dir, tok);
            let _ = std::fs::write(&path, &buf);
        }

        // ── 9. DeltaNet step: decay → retrieve → delta → update → query ──
        // state: [num_v_heads, v_head_dim, v_head_dim] FP32 (persistent per layer)
        // q: [num_v_heads, key_head_dim=128] FP32 (after repeat)
        // k: [num_v_heads, key_head_dim=128] FP32 (after repeat)
        // v: [num_v_heads, v_head_dim=128] FP32 (from conv split, inner_dim = num_v_heads * v_head_dim)
        // gate: [num_v_heads] FP32
        // beta: [num_v_heads] FP32
        // output: [num_v_heads, v_head_dim] = [inner_dim] FP32

        // Diagnostic: dump gate, beta, Q/K/V norms for L1 debugging
        if self.diag_enabled && layer_idx <= 2 && self.position <= 3 {
            self.stream.synchronize()?;
            // Gate values
            let mut gate_buf = vec![0u8; num_v_heads * 4];
            self.dn_gate_dev.copy_to_host(&mut gate_buf)?;
            let gate_f32 = unsafe { std::slice::from_raw_parts(gate_buf.as_ptr() as *const f32, num_v_heads) };
            // Beta values (post-sigmoid)
            let mut beta_buf = vec![0u8; num_v_heads * 4];
            self.dn_beta_f32_dev.copy_to_host(&mut beta_buf)?;
            let beta_f32 = unsafe { std::slice::from_raw_parts(beta_buf.as_ptr() as *const f32, num_v_heads) };
            // Q L2 per head
            let mut q_buf_bytes = vec![0u8; num_v_heads * key_head_dim * 4];
            self.dn_q_rep_dev.copy_to_host(&mut q_buf_bytes)?;
            let q_f32 = unsafe { std::slice::from_raw_parts(q_buf_bytes.as_ptr() as *const f32, num_v_heads * key_head_dim) };
            let q_head_l2: Vec<f32> = (0..std::cmp::min(num_v_heads, 4))
                .map(|h| {
                    let start = h * key_head_dim;
                    q_f32[start..start+key_head_dim].iter().map(|v| v*v).sum::<f32>().sqrt()
                })
                .collect();
            // V L2 from conv_out_dev (v_ptr is at offset 2*key_dim*4)
            let mut conv_out_bytes = vec![0u8; qkv_dim * 4];
            self.dn_conv_out_dev.copy_to_host(&mut conv_out_bytes)?;
            let conv_f32 = unsafe { std::slice::from_raw_parts(conv_out_bytes.as_ptr() as *const f32, qkv_dim) };
            let v_start = 2 * key_dim;
            let v_f32 = &conv_f32[v_start..v_start + inner_dim];
            let v_l2 = v_f32.iter().map(|v| v*v).sum::<f32>().sqrt();
            // State L2
            let state_size = num_v_heads * v_head_dim * v_head_dim;
            let mut state_bytes = vec![0u8; state_size * 4];
            self.dn_recurrent_state[dn_state_idx].copy_to_host(&mut state_bytes)?;
            let state_f32 = unsafe { std::slice::from_raw_parts(state_bytes.as_ptr() as *const f32, state_size) };
            let state_l2 = state_f32.iter().map(|v| v*v).sum::<f32>().sqrt();
            // Per-head state L2
            let head_state_size = v_head_dim * v_head_dim;
            let state_head_l2: Vec<f32> = (0..std::cmp::min(num_v_heads, 4))
                .map(|h| {
                    let start = h * head_state_size;
                    state_f32[start..start+head_state_size].iter().map(|v| v*v).sum::<f32>().sqrt()
                })
                .collect();

            tracing::info!(
                "DELTANET_DIAG L{} pos={}: gate first4=[{:.6},{:.6},{:.6},{:.6}] gate_range=[{:.6},{:.6}] \
                 beta first4=[{:.4},{:.4},{:.4},{:.4}] v_L2={:.4} state_L2={:.4} state_head_L2={:?} \
                 q_head_L2 first4=[{:.4},{:.4},{:.4},{:.4}]",
                layer_idx, self.position,
                gate_f32[0], gate_f32[1], gate_f32[2], gate_f32[3],
                gate_f32.iter().cloned().fold(f32::INFINITY, f32::min),
                gate_f32.iter().cloned().fold(f32::NEG_INFINITY, f32::max),
                beta_f32[0], beta_f32[1], beta_f32[2], beta_f32[3],
                v_l2, state_l2, state_head_l2,
                q_head_l2[0], q_head_l2[1], q_head_l2[2], q_head_l2[3],
            );
        }

        kernels::deltanet_step(
            self.dn_recurrent_state[dn_state_idx].as_mut_ptr(),
            self.dn_q_rep_dev.as_ptr(),
            self.dn_k_rep_dev.as_ptr(),
            v_ptr, // V is in dn_conv_out_dev at offset 2*key_dim*4
            self.dn_gate_dev.as_ptr(),
            self.dn_beta_f32_dev.as_ptr(),
            self.dn_step_out_dev.as_mut_ptr(),
            num_v_heads,
            v_head_dim, // vdim = key_head_dim = v_head_dim = 128
            &self.stream,
        )?;

        // Dump deltanet_step output for layer 0
        if self.diag_enabled && layer_idx == 0 {
            self.stream.synchronize()?;
            let dump_dir = "/home/brian/code/vib3/dump";
            let tok = self.position;
            let step_size = num_v_heads * v_head_dim; // inner_dim = 4096
            let mut buf = vec![0u8; step_size * 4];
            self.dn_step_out_dev.copy_to_host(&mut buf)?;
            let path = format!("{}/vib3_attn_out_L0_tok{}.bin", dump_dir, tok);
            let _ = std::fs::write(&path, &buf);
        }

        // Diagnostic: dump step output for L1 debugging
        if self.diag_enabled && layer_idx <= 2 && self.position <= 3 {
            self.stream.synchronize()?;
            let mut step_bytes = vec![0u8; inner_dim * 4];
            self.dn_step_out_dev.copy_to_host(&mut step_bytes)?;
            let step_buf = unsafe { std::slice::from_raw_parts(step_bytes.as_ptr() as *const f32, inner_dim) };
            let step_l2 = step_buf.iter().map(|v| v*v).sum::<f32>().sqrt();
            let step_max = step_buf.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let step_min = step_buf.iter().cloned().fold(f32::INFINITY, f32::min);
            // Per-head step L2
            let step_head_l2: Vec<f32> = (0..std::cmp::min(num_v_heads, 4))
                .map(|h| {
                    let start = h * v_head_dim;
                    step_buf[start..start+v_head_dim].iter().map(|v| v*v).sum::<f32>().sqrt()
                })
                .collect();
            tracing::info!(
                "DELTANET_STEP L{} pos={}: step_out L2={:.4}, range=[{:.4},{:.4}], head_L2={:?}",
                layer_idx, self.position, step_l2, step_min, step_max, step_head_l2,
            );
        }

        if profile { self.stream.synchronize()?; }
        let mid_us = t_mid.elapsed().as_micros() as u64;

        // ── 10. Z gate projection ──
        let t_zout = Instant::now();
        if batched_qkv_z {
            // Z was already computed in the batched QKV+Z launch — skip
        } else if let Some((z_nvfp4_ptr, _m, z_nvfp4_k)) = z_nvfp4 {
            // NVFP4 path: reuse the already-quantized hidden_dim activation from QKV step
            // (same hidden_state_f32 → attn_normed_f32 → FP4, same K=hidden_dim)
            // If QKV was also NVFP4, the preq_act_fp4/scales are already populated.
            // If not, we need to quantize now.
            if qkv_nvfp4.is_none() {
                let eps = self.model_config.rms_norm_eps;
                if let Some((norm_ptr, _)) = self.get_preassembled_weight(layer_idx, 6) {
                    kernels::rms_norm_f32(
                        self.hidden_state_f32.as_ptr(),
                        self.attn_normed_f32.as_mut_ptr(),
                        norm_ptr,
                        hidden_dim,
                        eps,
                        &self.stream,
                    )?;
                }
                kernels::quantize_activation_fp4_fast(
                    self.attn_normed_f32.as_ptr(),
                    self.preq_act_fp4.as_mut_ptr(),
                    self.preq_act_scales.as_mut_ptr(),
                    z_nvfp4_k,
                    &self.stream,
                )?;
            }
            // MMA GEMV → FP32 directly into dn_z_f32_dev
            if use_scalar_dn_nvfp4 {
                kernels::gemv_scalar_nvfp4(
                    self.preq_act_fp4.as_ptr(),
                    self.preq_act_scales.as_ptr(),
                    z_nvfp4_ptr,
                    self.dn_z_f32_dev.as_mut_ptr(),
                    z_nvfp4_k,
                    inner_dim,
                    &self.stream,
                )?;
            } else {
                kernels::gemv_mma_nvfp4_tiled(
                    self.preq_act_fp4.as_ptr(),
                    self.preq_act_scales.as_ptr(),
                    z_nvfp4_ptr,
                    self.dn_z_f32_dev.as_mut_ptr(),
                    z_nvfp4_k,
                    inner_dim,
                    &self.stream,
                )?;
            }
        } else {
            let z_w_ptr = z_fp16_ptr.ok_or_else(|| {
                Error::ConfigError(format!(
                    "DeltaNet L{} missing Z weight pointer (segment 31)",
                    layer_idx
                ))
            })?;
            if use_f32_deltanet_proj {
                kernels::linear_projection_f32_to_f32(
                    self.hidden_state_f32.as_ptr(),
                    z_w_ptr,
                    self.dn_z_f32_dev.as_mut_ptr(),
                    hidden_dim,
                    inner_dim,
                    &self.stream,
                )?;
            } else {
                kernels::linear_projection(
                    self.hidden_state.as_ptr(),
                    z_w_ptr,
                    self.dn_z_proj_dev.as_mut_ptr(),
                    hidden_dim,
                    inner_dim,
                    &self.stream,
                )?;
                kernels::f16_to_f32(
                    self.dn_z_proj_dev.as_ptr(),
                    self.dn_z_f32_dev.as_mut_ptr(),
                    inner_dim,
                    &self.stream,
                )?;
            }
        }
        if diag_nvfp4_compare && z_nvfp4.is_some() {
            let z_ref_ptr = z_fp16_ptr.or_else(|| {
                if self.load_tensor_direct_resident(
                    layer_idx,
                    31,
                    self.dn_w_z.as_mut_ptr() as usize,
                    self.dn_w_z.size(),
                ) > 0 {
                    Some(self.dn_w_z.as_ptr())
                } else {
                    None
                }
            });
            if let Some(z_w_ptr) = z_ref_ptr {
                kernels::linear_projection(
                    self.hidden_state.as_ptr(),
                    z_w_ptr,
                    self.dn_z_proj_dev.as_mut_ptr(),
                    hidden_dim,
                    inner_dim,
                    &self.stream,
                )?;
                kernels::f16_to_f32(
                    self.dn_z_proj_dev.as_ptr(),
                    self.nvfp4_f32_scratch.as_mut_ptr(),
                    inner_dim,
                    &self.stream,
                )?;
                self.log_projection_compare(
                    "dn_z",
                    layer_idx,
                    self.nvfp4_f32_scratch.as_ptr(),
                    self.dn_z_f32_dev.as_ptr(),
                    inner_dim,
                )?;
            }
        }

        // ── 11. Gated RMSNorm: output = RMSNorm(step_out, norm_weight) * SiLU(z) ──
        // Convert norm weight from FP16 to FP32
        kernels::f16_to_f32(
            norm_w_ptr,
            self.dn_norm_weight_f32_dev.as_mut_ptr(),
            v_head_dim, // norm_weight is [v_head_dim=128]
            &self.stream,
        )?;

        // Dump Z gate output for layer 0
        if self.diag_enabled && layer_idx == 0 {
            self.stream.synchronize()?;
            let dump_dir = "/home/brian/code/vib3/dump";
            let tok = self.position;
            let mut buf = vec![0u8; inner_dim * 4];
            self.dn_z_f32_dev.copy_to_host(&mut buf)?;
            let path = format!("{}/vib3_z_gate_L0_tok{}.bin", dump_dir, tok);
            let _ = std::fs::write(&path, &buf);
        }

        // Diagnostic: Z gate and norm weight stats before gated_rmsnorm
        if self.diag_enabled && layer_idx <= 2 && self.position <= 3 {
            self.stream.synchronize()?;
            let mut z_bytes = vec![0u8; inner_dim * 4];
            self.dn_z_f32_dev.copy_to_host(&mut z_bytes)?;
            let z_buf = unsafe { std::slice::from_raw_parts(z_bytes.as_ptr() as *const f32, inner_dim) };
            let z_l2 = z_buf.iter().map(|v| v*v).sum::<f32>().sqrt();
            let z_max = z_buf.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let z_min = z_buf.iter().cloned().fold(f32::INFINITY, f32::min);
            // Per-head Z gate L2 and max SiLU
            let z_head_info: Vec<(f32, f32)> = (0..std::cmp::min(num_v_heads, 8))
                .map(|h| {
                    let start = h * v_head_dim;
                    let head_slice = &z_buf[start..start+v_head_dim];
                    let hl2 = head_slice.iter().map(|v| v*v).sum::<f32>().sqrt();
                    let hmax = head_slice.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                    (hl2, hmax)
                })
                .collect();
            // Compute max SiLU(z) magnitude across all elements
            let max_silu = z_buf.iter().map(|&g| {
                let silu = g / (1.0 + (-g).exp());
                silu.abs()
            }).fold(0.0f32, f32::max);
            tracing::info!(
                "DELTANET_ZGATE L{} pos={}: z_L2={:.4}, z_range=[{:.4},{:.4}], max_silu={:.4}, head_info(L2,max)={:?}",
                layer_idx, self.position, z_l2, z_min, z_max, max_silu, z_head_info,
            );
        }

        kernels::gated_rmsnorm(
            self.dn_step_out_dev.as_ptr(),
            self.dn_z_f32_dev.as_ptr(),
            self.dn_norm_weight_f32_dev.as_ptr(),
            self.dn_norm_out_dev.as_mut_ptr(),
            num_v_heads,  // 64 independent heads
            v_head_dim,   // 128 elements per head
            v_head_dim,   // norm_dim cycles across group_dim
            eps,
            &self.stream,
        )?;

        // Diagnostic: gated_rmsnorm output stats
        if self.diag_enabled && layer_idx <= 2 && self.position <= 3 {
            self.stream.synchronize()?;
            let mut norm_bytes = vec![0u8; inner_dim * 4];
            self.dn_norm_out_dev.copy_to_host(&mut norm_bytes)?;
            let norm_buf = unsafe { std::slice::from_raw_parts(norm_bytes.as_ptr() as *const f32, inner_dim) };
            let norm_l2 = norm_buf.iter().map(|v| v*v).sum::<f32>().sqrt();
            let norm_max = norm_buf.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let norm_min = norm_buf.iter().cloned().fold(f32::INFINITY, f32::min);
            let norm_head_l2: Vec<f32> = (0..std::cmp::min(num_v_heads, 8))
                .map(|h| {
                    let start = h * v_head_dim;
                    norm_buf[start..start+v_head_dim].iter().map(|v| v*v).sum::<f32>().sqrt()
                })
                .collect();
            tracing::info!(
                "DELTANET_NORMOUT L{} pos={}: L2={:.4}, range=[{:.4},{:.4}], head_L2={:?}",
                layer_idx, self.position, norm_l2, norm_min, norm_max, norm_head_l2,
            );
        }

        // Dump gated_rmsnorm output for layer 0
        if self.diag_enabled && layer_idx == 0 {
            self.stream.synchronize()?;
            let dump_dir = "/home/brian/code/vib3/dump";
            let tok = self.position;
            let mut buf = vec![0u8; inner_dim * 4];
            self.dn_norm_out_dev.copy_to_host(&mut buf)?;
            let path = format!("{}/vib3_gated_rmsnorm_L0_tok{}.bin", dump_dir, tok);
            let _ = std::fs::write(&path, &buf);
        }

        // ── 12. Output projection: inner_dim → hidden_dim ──
        let out_nvfp4 = if use_nvfp4_deltanet_out {
            self.get_preassembled_nvfp4(layer_idx, 38)
        } else {
            None
        };
        if let Some((out_nvfp4_ptr, _m, out_nvfp4_k)) = out_nvfp4 {
            // NVFP4 path: quantize FP32 norm_out (inner_dim) → FP4, then MMA GEMV → FP32
            kernels::quantize_activation_fp4_fast(
                self.dn_norm_out_dev.as_ptr(),
                self.preq_inner_act_fp4.as_mut_ptr(),
                self.preq_inner_act_scales.as_mut_ptr(),
                out_nvfp4_k, // K = inner_dim
                &self.stream,
            )?;
            if use_scalar_dn_nvfp4 {
                kernels::gemv_scalar_nvfp4(
                    self.preq_inner_act_fp4.as_ptr(),
                    self.preq_inner_act_scales.as_ptr(),
                    out_nvfp4_ptr,
                    self.dn_o_out_dev.as_mut_ptr(),
                    out_nvfp4_k,
                    hidden_dim,
                    &self.stream,
                )?;
            } else {
                kernels::gemv_mma_nvfp4_tiled(
                    self.preq_inner_act_fp4.as_ptr(),
                    self.preq_inner_act_scales.as_ptr(),
                    out_nvfp4_ptr,
                    self.dn_o_out_dev.as_mut_ptr(),
                    out_nvfp4_k,
                    hidden_dim,
                    &self.stream,
                )?;
            }
        } else {
            let o_w_ptr = out_fp16_ptr.ok_or_else(|| {
                Error::ConfigError(format!(
                    "DeltaNet L{} missing output projection pointer (segment 38)",
                    layer_idx
                ))
            })?;
            kernels::linear_projection_f32_to_f32(
                self.dn_norm_out_dev.as_ptr(),
                o_w_ptr,
                self.dn_o_out_dev.as_mut_ptr(),
                inner_dim,
                hidden_dim,
                &self.stream,
            )?;
        }
        if diag_nvfp4_compare && out_nvfp4.is_some() {
            let out_ref_ptr = out_fp16_ptr.or_else(|| {
                if self.load_tensor_direct_resident(
                    layer_idx,
                    38,
                    self.dn_w_out.as_mut_ptr() as usize,
                    self.dn_w_out.size(),
                ) > 0 {
                    Some(self.dn_w_out.as_ptr())
                } else {
                    None
                }
            });
            if let Some(o_w_ptr) = out_ref_ptr {
                kernels::linear_projection_f32_to_f32(
                    self.dn_norm_out_dev.as_ptr(),
                    o_w_ptr,
                    self.nvfp4_f32_scratch.as_mut_ptr(),
                    inner_dim,
                    hidden_dim,
                    &self.stream,
                )?;
                self.log_projection_compare(
                    "dn_out",
                    layer_idx,
                    self.nvfp4_f32_scratch.as_ptr(),
                    self.dn_o_out_dev.as_ptr(),
                    hidden_dim,
                )?;
            }
        }

        // Dump out_proj output for layer 0
        if self.diag_enabled && layer_idx == 0 {
            self.stream.synchronize()?;
            let dump_dir = "/home/brian/code/vib3/dump";
            let tok = self.position;
            let mut buf = vec![0u8; hidden_dim * 4];
            self.dn_o_out_dev.copy_to_host(&mut buf)?;
            let path = format!("{}/vib3_out_proj_L0_tok{}.bin", dump_dir, tok);
            let _ = std::fs::write(&path, &buf);
        }

        // ── 13. Residual add: hidden_state_f32 += o_proj_output (FP32) ──
        kernels::residual_add_f32_f32(
            self.hidden_state_f32.as_mut_ptr(),
            self.dn_o_out_dev.as_ptr(),
            hidden_dim,
            &self.stream,
        )?;
        // Dump post-residual hidden state for layer 0
        if self.diag_enabled && layer_idx == 0 {
            self.stream.synchronize()?;
            let dump_dir = "/home/brian/code/vib3/dump";
            let tok = self.position;
            let mut buf = vec![0u8; hidden_dim * 4];
            self.hidden_state_f32.copy_to_host(&mut buf)?;
            let path = format!("{}/vib3_attn_residual_L0_tok{}.bin", dump_dir, tok);
            let _ = std::fs::write(&path, &buf);
        }
        if profile {
            self.stream.synchronize()?;
            let zout_us = t_zout.elapsed().as_micros() as u64;
            if layer_idx <= 4 || layer_idx % 12 == 0 || layer_idx >= 44 {
                tracing::info!(
                    "PROFILE DN_INNER L{}: load={:.0}us qkv={:.0}us mid={:.0}us zout={:.0}us total={:.0}us",
                    layer_idx, load_us, qkv_us, mid_us, zout_us,
                    load_us + qkv_us + mid_us + zout_us,
                );
            }
        }

        // Diagnostic: dump DeltaNet intermediate values for debugging
        if self.diag_enabled && layer_idx <= 2 && self.decode_step <= 3 {
            self.stream.synchronize()?;
            let f32_bytes = hidden_dim * 4;
            let mut diag_buf = vec![0u8; f32_bytes];
            self.dn_o_out_dev.copy_to_host(&mut diag_buf)?;
            let o_f32 = unsafe { std::slice::from_raw_parts(diag_buf.as_ptr() as *const f32, hidden_dim) };
            let o_l2 = o_f32.iter().map(|v| v * v).sum::<f32>().sqrt();
            tracing::info!(
                "DELTANET L{} pos={}: o_proj L2={:.4}, first4=[{:.6},{:.6},{:.6},{:.6}]",
                layer_idx, self.position, o_l2,
                o_f32[0], o_f32[1], o_f32[2], o_f32[3],
            );
        }

        Ok(())
    }

    /// GPU-projected attention: QKV/O projections on GPU, attention on CPU.
    ///
    /// This method contains NO `.await` calls, so raw device pointers
    /// (*const u8) can be used freely without making the caller's future !Send.
    ///
    /// Returns true if GPU projection succeeded, false if device weights
    /// are not available (caller should fall through to CPU path).
    fn try_gpu_attention_projection(
        &mut self,
        layer_idx: u16,
        hidden_dim: usize,
        _hidden_bytes: usize,
        q_dim: usize,
        kv_dim: usize,
    ) -> Result<bool> {
        let layer = layer_idx as usize;

        // Get device pointers for QKV/O weights (must all be available)
        let q_ptr = match self.get_device_tensor(layer_idx, 4) {
            Some((ptr, _)) => ptr,
            None => return Ok(false),
        };
        let k_ptr = match self.get_device_tensor(layer_idx, 12) {
            Some((ptr, _)) => ptr,
            None => return Ok(false),
        };
        let v_ptr = match self.get_device_tensor(layer_idx, 13) {
            Some((ptr, _)) => ptr,
            None => return Ok(false),
        };
        let o_ptr = match self.get_device_tensor(layer_idx, 5) {
            Some((ptr, _)) => ptr,
            None => return Ok(false),
        };

        // Use pre-allocated device scratch for projected Q/K/V vectors
        // (eliminates 3x cudaMalloc + 3x cudaFree per layer)
        let q_proj_bytes = q_dim * 2; // FP16
        let kv_proj_bytes = kv_dim * 2;

        // GPU GEMV: Q/K/V projections (write-only, no need to zero first)
        kernels::linear_projection(
            self.hidden_state.as_ptr(),
            q_ptr,
            self.q_proj_dev.as_mut_ptr(),
            hidden_dim,
            q_dim,
            &self.stream,
        )?;
        kernels::linear_projection(
            self.hidden_state.as_ptr(),
            k_ptr,
            self.k_proj_dev.as_mut_ptr(),
            hidden_dim,
            kv_dim,
            &self.stream,
        )?;
        kernels::linear_projection(
            self.hidden_state.as_ptr(),
            v_ptr,
            self.v_proj_dev.as_mut_ptr(),
            hidden_dim,
            kv_dim,
            &self.stream,
        )?;

        // Sync and D2H the small projected vectors (Q=16KB, K=4KB, V=4KB)
        self.stream.synchronize()?;
        let mut q_proj_host = vec![0u8; q_proj_bytes];
        let mut k_proj_host = vec![0u8; kv_proj_bytes];
        let mut v_proj_host = vec![0u8; kv_proj_bytes];
        self.q_proj_dev.copy_to_host(&mut q_proj_host)?;
        self.k_proj_dev.copy_to_host(&mut k_proj_host)?;
        self.v_proj_dev.copy_to_host(&mut v_proj_host)?;

        // Convert projected FP16 vectors to f32 for CPU attention
        let q_proj_f16 =
            unsafe { std::slice::from_raw_parts(q_proj_host.as_ptr() as *const f16, q_dim) };
        let k_proj_f16 =
            unsafe { std::slice::from_raw_parts(k_proj_host.as_ptr() as *const f16, kv_dim) };
        let v_proj_f16 =
            unsafe { std::slice::from_raw_parts(v_proj_host.as_ptr() as *const f16, kv_dim) };

        let q_f32: Vec<f32> = q_proj_f16.iter().map(|v| v.to_f32()).collect();
        let k_f32: Vec<f32> = k_proj_f16.iter().map(|v| v.to_f32()).collect();
        let v_f32: Vec<f32> = v_proj_f16.iter().map(|v| v.to_f32()).collect();

        // CPU: RoPE + KV cache update + multi-head attention (tiny work for decode)
        let attn_concat = self_attention_projected(
            &q_f32,
            &k_f32,
            &v_f32,
            &mut self.kv_cache.layers[layer],
            self.position,
            &self.model_config,
        );

        // H2D the attention output for GPU O projection (using pre-allocated attn_dev)
        let attn_f16: Vec<f16> = attn_concat.iter().map(|v| f16::from_f32(*v)).collect();
        let attn_bytes = attn_f16.len() * 2;
        let attn_host_bytes =
            unsafe { std::slice::from_raw_parts(attn_f16.as_ptr() as *const u8, attn_bytes) };

        self.attn_dev.copy_from_host(attn_host_bytes)?;

        // GPU GEMV: O_proj = attn_concat × O_weights^T → hidden_state
        // O weight: [hidden_dim, num_heads*head_dim], input: [num_heads*head_dim], output: [hidden_dim]
        let num_heads = self.model_config.num_heads as usize;
        let head_dim = self.model_config.effective_head_dim() as usize;
        let attn_dim = num_heads * head_dim;
        kernels::linear_projection(
            self.attn_dev.as_ptr(),
            o_ptr,
            self.hidden_state.as_mut_ptr(),
            attn_dim,
            hidden_dim,
            &self.stream,
        )?;

        // FP32 residual accumulation: hidden_state_f32 += f32(O_proj output in hidden_state)
        kernels::residual_add_fp32(
            self.hidden_state_f32.as_mut_ptr(),
            self.hidden_state.as_ptr(),
            hidden_dim,
            &self.stream,
        )?;

        tracing::debug!("GPU attn projection complete: layer={}", layer_idx);
        Ok(true)
    }

    /// Fully GPU-resident decode attention: QKV projection, RoPE, KV cache,
    /// attention scoring, and O projection all stay on device with zero D2H/H2D.
    ///
    /// This eliminates the ~100-200ms/token CPU round-trip that dominates decode
    /// latency in `try_gpu_attention_projection()`.
    ///
    /// Like `try_gpu_attention_projection()`, this is a synchronous `fn` (no `.await`)
    /// so raw device pointers can be used freely without Send issues.
    ///
    /// Returns `Ok(true)` if the GPU decode path succeeded, `Ok(false)` if device
    /// weights are unavailable (caller falls through to the CPU-round-trip path).
    #[allow(unused_variables)]
    fn try_gpu_decode_attention(
        &mut self,
        layer_idx: u16,
        hidden_dim: usize,
        hidden_bytes: usize,
        q_dim: usize,
        kv_dim: usize,
    ) -> Result<bool> {
        // This path requires CUDA kernels for RoPE, KV cache append, and attention.
        // Without CUDA, fall through to the CPU-round-trip path.
        #[cfg(not(feature = "cuda"))]
        {
            return Ok(false);
        }

        #[cfg(feature = "cuda")]
        {
            let layer = layer_idx as usize;
            let num_heads = self.model_config.num_heads as usize;
            let num_kv_heads = self.model_config.num_kv_heads as usize;
            let head_dim = self.model_config.effective_head_dim() as usize;
            let rope_base = self.model_config.rope_theta;
            let position = self.position;
            let is_gated_attn = self.model_config.deltanet.is_some();
            let eps = self.model_config.rms_norm_eps;

            // Bounds check: GPU KV cache must be allocated for this layer
            if layer >= self.gpu_kv_k.len() || layer >= self.gpu_kv_v.len() {
                return Ok(false);
            }
            if position >= self.max_kv_len {
                tracing::warn!(
                    "GPU KV cache full: position {} >= max_kv_len {}",
                    position,
                    self.max_kv_len
                );
                return Ok(false);
            }

            // Check for NVFP4 weights first (4x bandwidth reduction via MMA GEMV)
            let q_nvfp4 = self.get_preassembled_nvfp4(layer_idx, 4);
            let k_nvfp4 = self.get_preassembled_nvfp4(layer_idx, 12);
            let v_nvfp4 = self.get_preassembled_nvfp4(layer_idx, 13);
            let o_nvfp4 = self.get_preassembled_nvfp4(layer_idx, 5);
            let use_nvfp4_qkv = q_nvfp4.is_some() && k_nvfp4.is_some() && v_nvfp4.is_some();

            // Get FP16 device pointers (fallback or for non-NVFP4 paths)
            let q_ptr = if !use_nvfp4_qkv {
                match self.get_device_tensor(layer_idx, 4) {
                    Some((ptr, _)) => ptr,
                    None => return Ok(false),
                }
            } else { std::ptr::null() };
            let k_ptr = if !use_nvfp4_qkv {
                match self.get_device_tensor(layer_idx, 12) {
                    Some((ptr, _)) => ptr,
                    None => return Ok(false),
                }
            } else { std::ptr::null() };
            let v_ptr = if !use_nvfp4_qkv {
                match self.get_device_tensor(layer_idx, 13) {
                    Some((ptr, _)) => ptr,
                    None => return Ok(false),
                }
            } else { std::ptr::null() };
            let o_ptr = if o_nvfp4.is_none() {
                match self.get_device_tensor(layer_idx, 5) {
                    Some((ptr, _)) => ptr,
                    None => return Ok(false),
                }
            } else { std::ptr::null() };

            let stream_ptr = self.stream.raw_ptr();

            // Step 1: GPU GEMV — Q/K/V projections into pre-allocated device buffers
            if use_nvfp4_qkv {
                // NVFP4 path: fused RMSNorm + FP4 quantize in a single kernel
                // Reads FP32 hidden_state, applies norm, outputs FP4 directly.
                // No FP16 output needed — GQA NVFP4 path doesn't read FP16 hidden_state.
                kernels::fused_rms_norm_quantize_fp4(
                    self.hidden_state_f32.as_ptr(),
                    self.get_preassembled_weight(layer_idx, 6).map(|(p,_)| p).unwrap_or(std::ptr::null()),
                    self.preq_act_fp4.as_mut_ptr(),
                    self.preq_act_scales.as_mut_ptr(),
                    std::ptr::null_mut(), // no FP16 output needed for GQA NVFP4
                    hidden_dim,
                    eps,
                    &self.stream,
                )?;

                // Batched Q+K+V GEMV: single kernel launch, outputs into nvfp4_f32_scratch
                // Layout: [Q output: q_dim floats | K output: kv_dim floats | V output: kv_dim floats]
                let (q_nvfp4_ptr, _q_m, q_nvfp4_k) = q_nvfp4.unwrap();
                let (k_nvfp4_ptr, _k_m, k_nvfp4_k) = k_nvfp4.unwrap();
                let (v_nvfp4_ptr, _v_m, v_nvfp4_k) = v_nvfp4.unwrap();
                let q_out_ptr = self.nvfp4_f32_scratch.as_mut_ptr();
                let k_out_ptr = unsafe { self.nvfp4_f32_scratch.as_mut_ptr().add(q_dim * 4) };
                let v_out_ptr = unsafe { self.nvfp4_f32_scratch.as_mut_ptr().add((q_dim + kv_dim) * 4) };

                let weight_pages = [q_nvfp4_ptr, k_nvfp4_ptr, v_nvfp4_ptr];
                let m_slices = [q_dim as i32, kv_dim as i32, kv_dim as i32];
                let mut outputs = [q_out_ptr, k_out_ptr, v_out_ptr];
                kernels::batched_gemv_mma_nvfp4_tiled(
                    self.preq_act_fp4.as_ptr(),
                    self.preq_act_scales.as_ptr(),
                    &weight_pages,
                    &m_slices,
                    &mut outputs,
                    hidden_dim,
                    &self.stream,
                )?;

                // Convert FP32 → FP16 for each projection
                kernels::f32_to_f16(
                    self.nvfp4_f32_scratch.as_ptr(),
                    self.q_proj_dev.as_mut_ptr(),
                    q_dim,
                    &self.stream,
                )?;
                kernels::f32_to_f16(
                    unsafe { self.nvfp4_f32_scratch.as_ptr().add(q_dim * 4) },
                    self.k_proj_dev.as_mut_ptr(),
                    kv_dim,
                    &self.stream,
                )?;
                kernels::f32_to_f16(
                    unsafe { self.nvfp4_f32_scratch.as_ptr().add((q_dim + kv_dim) * 4) },
                    self.v_proj_dev.as_mut_ptr(),
                    kv_dim,
                    &self.stream,
                )?;
            } else {
                // FP16 path
                kernels::linear_projection(
                    self.hidden_state.as_ptr(),
                    q_ptr,
                    self.q_proj_dev.as_mut_ptr(),
                    hidden_dim,
                    q_dim,
                    &self.stream,
                )?;
                kernels::linear_projection(
                    self.hidden_state.as_ptr(),
                    k_ptr,
                    self.k_proj_dev.as_mut_ptr(),
                    hidden_dim,
                    kv_dim,
                    &self.stream,
                )?;
                kernels::linear_projection(
                    self.hidden_state.as_ptr(),
                    v_ptr,
                    self.v_proj_dev.as_mut_ptr(),
                    hidden_dim,
                    kv_dim,
                    &self.stream,
                )?;
            }

            if is_gated_attn {
                // ── Qwen3.5 Gated GQA Attention Pipeline ──
                //
                // q_proj_dev contains interleaved [Q_h0, gate_h0, Q_h1, gate_h1, ...]
                // Each chunk is head_dim=256 elements. Total: num_heads * 2 * head_dim = 16384.
                //
                // Pipeline:
                //   1. Q/K/V projections (done above)
                //   2. Per-head RMSNorm on Q (in interleaved buffer, stride=2*head_dim)
                //   3. Per-head RMSNorm on K (contiguous, stride=head_dim)
                //   4. Deinterleave Q+gate → separate Q and gate buffers
                //   5. Partial RoPE on Q (64/256 dims, stride=head_dim)
                //   6. Partial RoPE on K (64/256 dims, stride=head_dim)
                //   7. Append K/V to KV cache

                // GQA diagnostic: dump ALL intermediates for first GQA layer at early positions
                let gqa_diag = self.diag_enabled && (layer == 3 || layer == 7) && self.position <= 1;

                // Dump raw Q projection (interleaved Q+gate, BEFORE RMSNorm)
                if gqa_diag {
                    self.stream.synchronize()?;
                    let dump_dir = "/home/brian/code/vib3/dump";
                    let tok = self.position;
                    // Q proj: interleaved [num_heads * 2 * head_dim] FP16
                    let q_total = num_heads * 2 * head_dim;
                    let mut buf = vec![0u8; q_total * 2];
                    self.q_proj_dev.copy_to_host(&mut buf)?;
                    let path = format!("{}/vib3_gqa_q_proj_raw_f16_L{}_tok{}.bin", dump_dir, layer, tok);
                    let _ = std::fs::write(&path, &buf);
                    // K proj: [num_kv_heads * head_dim] FP16
                    let kv_total = num_kv_heads * head_dim;
                    let mut kbuf = vec![0u8; kv_total * 2];
                    self.k_proj_dev.copy_to_host(&mut kbuf)?;
                    let kpath = format!("{}/vib3_gqa_k_proj_raw_f16_L{}_tok{}.bin", dump_dir, layer, tok);
                    let _ = std::fs::write(&kpath, &kbuf);
                    // V proj: [num_kv_heads * head_dim] FP16
                    let mut vbuf = vec![0u8; kv_total * 2];
                    self.v_proj_dev.copy_to_host(&mut vbuf)?;
                    let vpath = format!("{}/vib3_gqa_v_proj_raw_f16_L{}_tok{}.bin", dump_dir, layer, tok);
                    let _ = std::fs::write(&vpath, &vbuf);
                    tracing::info!("GQA DIAG L{} tok{}: dumped raw Q/K/V projections", layer, tok);
                }
                //   8. Decode attention (Q × K^T / sqrt(head_dim), softmax, × V)
                //   9. sigmoid(gate) * attention_output
                //  10. O projection → hidden_state

                let attn_dim = num_heads * head_dim; // 8192
                let rope_dim = (head_dim as f32 * 0.25) as usize; // 64 for Qwen3.5

                // Step 2: Per-head RMSNorm on Q (still interleaved, stride = 2*head_dim)
                // q_norm weight: segment 27, shape [head_dim]
                if let Some((q_norm_ptr, _)) = self.get_device_tensor(layer_idx, 27) {
                    kernels::per_head_rmsnorm(
                        self.q_proj_dev.as_mut_ptr(),
                        q_norm_ptr,
                        head_dim,
                        2 * head_dim, // stride: skip over gate between Q heads
                        num_heads,
                        eps,
                        &self.stream,
                    )?;
                } else {
                    tracing::warn!("L{}: q_norm weight not available for gated attn", layer_idx);
                }

                // Step 3: Per-head RMSNorm on K (contiguous, stride = head_dim)
                // k_norm weight: segment 28, shape [head_dim]
                if let Some((k_norm_ptr, _)) = self.get_device_tensor(layer_idx, 28) {
                    kernels::per_head_rmsnorm(
                        self.k_proj_dev.as_mut_ptr(),
                        k_norm_ptr,
                        head_dim,
                        head_dim, // stride: contiguous heads
                        num_kv_heads,
                        eps,
                        &self.stream,
                    )?;
                } else {
                    tracing::warn!("L{}: k_norm weight not available for gated attn", layer_idx);
                }

                // Dump Q after RMSNorm (interleaved) and K after RMSNorm
                if gqa_diag {
                    self.stream.synchronize()?;
                    let dump_dir = "/home/brian/code/vib3/dump";
                    let tok = self.position;
                    let q_total = num_heads * 2 * head_dim;
                    let mut buf = vec![0u8; q_total * 2];
                    self.q_proj_dev.copy_to_host(&mut buf)?;
                    let path = format!("{}/vib3_gqa_q_after_norm_f16_L{}_tok{}.bin", dump_dir, layer, tok);
                    let _ = std::fs::write(&path, &buf);
                    let kv_total = num_kv_heads * head_dim;
                    let mut kbuf = vec![0u8; kv_total * 2];
                    self.k_proj_dev.copy_to_host(&mut kbuf)?;
                    let kpath = format!("{}/vib3_gqa_k_after_norm_f16_L{}_tok{}.bin", dump_dir, layer, tok);
                    let _ = std::fs::write(&kpath, &kbuf);
                    tracing::info!("GQA DIAG L{} tok{}: dumped Q/K after RMSNorm", layer, tok);
                }

                // Step 4: Deinterleave Q+gate → attn_dev (Q) + gated_attn_gate_dev (gate)
                // Input: q_proj_dev [num_heads * 2 * head_dim] interleaved
                // Output: attn_dev [num_heads * head_dim] = Q, gated_attn_gate_dev [same] = gate
                kernels::deinterleave_f16(
                    self.q_proj_dev.as_ptr(),
                    self.attn_dev.as_mut_ptr(),       // Q output
                    self.gated_attn_gate_dev.as_mut_ptr(), // gate output
                    head_dim,    // chunk_size
                    num_heads,   // num_chunks
                    &self.stream,
                )?;

                // Dump Q and gate after deinterleave (before RoPE)
                if gqa_diag {
                    self.stream.synchronize()?;
                    let dump_dir = "/home/brian/code/vib3/dump";
                    let tok = self.position;
                    let attn_total = num_heads * head_dim;
                    let mut qbuf = vec![0u8; attn_total * 2];
                    self.attn_dev.copy_to_host(&mut qbuf)?;
                    let qpath = format!("{}/vib3_gqa_q_deinterleaved_f16_L{}_tok{}.bin", dump_dir, layer, tok);
                    let _ = std::fs::write(&qpath, &qbuf);
                    let mut gbuf = vec![0u8; attn_total * 2];
                    self.gated_attn_gate_dev.copy_to_host(&mut gbuf)?;
                    let gpath = format!("{}/vib3_gqa_gate_deinterleaved_f16_L{}_tok{}.bin", dump_dir, layer, tok);
                    let _ = std::fs::write(&gpath, &gbuf);
                    tracing::info!("GQA DIAG L{} tok{}: dumped Q and gate after deinterleave", layer, tok);
                }

                // Step 5: Partial RoPE on Q (now in attn_dev, contiguous, stride=head_dim)
                kernels::partial_rope(
                    self.attn_dev.as_mut_ptr(),
                    head_dim,
                    rope_dim,
                    head_dim, // stride
                    num_heads,
                    self.d_position.as_ptr(), // device-side position for CUDA graph compat
                    rope_base,
                    &self.stream,
                )?;

                // Step 6: Partial RoPE on K
                kernels::partial_rope(
                    self.k_proj_dev.as_mut_ptr(),
                    head_dim,
                    rope_dim,
                    head_dim, // stride
                    num_kv_heads,
                    self.d_position.as_ptr(), // device-side position for CUDA graph compat
                    rope_base,
                    &self.stream,
                )?;

                // Dump Q and K after partial RoPE
                if gqa_diag {
                    self.stream.synchronize()?;
                    let dump_dir = "/home/brian/code/vib3/dump";
                    let tok = self.position;
                    let attn_total = num_heads * head_dim;
                    let mut qbuf = vec![0u8; attn_total * 2];
                    self.attn_dev.copy_to_host(&mut qbuf)?;
                    let qpath = format!("{}/vib3_gqa_q_after_rope_f16_L{}_tok{}.bin", dump_dir, layer, tok);
                    let _ = std::fs::write(&qpath, &qbuf);
                    let kv_total = num_kv_heads * head_dim;
                    let mut kbuf = vec![0u8; kv_total * 2];
                    self.k_proj_dev.copy_to_host(&mut kbuf)?;
                    let kpath = format!("{}/vib3_gqa_k_after_rope_f16_L{}_tok{}.bin", dump_dir, layer, tok);
                    let _ = std::fs::write(&kpath, &kbuf);
                    // Also dump V (unmodified after projection)
                    let mut vbuf = vec![0u8; kv_total * 2];
                    self.v_proj_dev.copy_to_host(&mut vbuf)?;
                    let vpath = format!("{}/vib3_gqa_v_f16_L{}_tok{}.bin", dump_dir, layer, tok);
                    let _ = std::fs::write(&vpath, &vbuf);
                    tracing::info!("GQA DIAG L{} tok{}: dumped Q/K after RoPE, V", layer, tok);
                }

                // Step 7: Append post-RoPE K/V to the GPU-resident KV cache
                let err = unsafe {
                    cuda_ffi::vib3_launch_kv_cache_append(
                        self.gpu_kv_k[layer].as_mut_ptr(),
                        self.gpu_kv_v[layer].as_mut_ptr(),
                        self.k_proj_dev.as_ptr(),
                        self.v_proj_dev.as_ptr(),
                        self.max_kv_len as i32,
                        head_dim as i32,
                        num_kv_heads as i32,
                        self.d_position.as_ptr(), // device-side position
                        stream_ptr,
                    )
                };
                if err != 0 {
                    return Err(Error::Cuda(format!(
                        "vib3_launch_kv_cache_append failed (err={})",
                        err
                    )));
                }

                // Step 8: Decode attention — Q is in attn_dev (post-RoPE, [num_heads * head_dim])
                // We need a separate buffer for attention output since attn_dev has Q.
                // Use q_proj_dev as scratch for attention output (it's large enough: 16384 >= 8192).
                let scale = 1.0f32 / (head_dim as f32).sqrt();
                let err = unsafe {
                    cuda_ffi::vib3_launch_decode_attention(
                        self.attn_dev.as_ptr(),           // Q: [num_heads * head_dim]
                        self.gpu_kv_k[layer].as_ptr(),
                        self.gpu_kv_v[layer].as_ptr(),
                        self.q_proj_dev.as_mut_ptr(),     // output: [num_heads * head_dim] (reuse q_proj_dev as scratch)
                        head_dim as i32,
                        num_heads as i32,
                        num_kv_heads as i32,
                        self.d_position.as_ptr(), // device-side position (kernel computes seq_len=pos+1)
                        self.max_kv_len as i32,
                        scale,
                        stream_ptr,
                    )
                };
                if err != 0 {
                    return Err(Error::Cuda(format!(
                        "vib3_launch_decode_attention failed (err={})",
                        err
                    )));
                }

                // Dump attention output (before gating)
                if gqa_diag {
                    self.stream.synchronize()?;
                    let dump_dir = "/home/brian/code/vib3/dump";
                    let tok = self.position;
                    let attn_total = num_heads * head_dim;
                    let mut abuf = vec![0u8; attn_total * 2];
                    self.q_proj_dev.copy_to_host(&mut abuf)?; // attn output is in q_proj_dev
                    let apath = format!("{}/vib3_gqa_attn_output_f16_L{}_tok{}.bin", dump_dir, layer, tok);
                    let _ = std::fs::write(&apath, &abuf);
                    // Also log norms
                    let f16_slice = unsafe { std::slice::from_raw_parts(abuf.as_ptr() as *const u16, attn_total) };
                    let attn_l2: f32 = f16_slice.iter()
                        .map(|&bits| { let v = half::f16::from_bits(bits).to_f32(); v * v })
                        .sum::<f32>().sqrt();
                    tracing::info!("GQA DIAG L{} tok{}: attn_output L2={:.4}", layer, tok, attn_l2);
                }

                // Step 9: Apply gating: attn_dev = attention_output * sigmoid(gate)
                // attention_output is in q_proj_dev, gate is in gated_attn_gate_dev
                // Write result to attn_dev for O projection input
                kernels::sigmoid_mul_f16(
                    self.attn_dev.as_mut_ptr(),          // output
                    self.q_proj_dev.as_ptr(),             // attention output
                    self.gated_attn_gate_dev.as_ptr(),    // gate
                    attn_dim,                             // num_heads * head_dim
                    &self.stream,
                )?;

                // Dump gated attention output (after sigmoid(gate) * attn_output)
                if gqa_diag {
                    self.stream.synchronize()?;
                    let dump_dir = "/home/brian/code/vib3/dump";
                    let tok = self.position;
                    let attn_total = num_heads * head_dim;
                    let mut gbuf = vec![0u8; attn_total * 2];
                    self.attn_dev.copy_to_host(&mut gbuf)?;
                    let gpath = format!("{}/vib3_gqa_gated_output_f16_L{}_tok{}.bin", dump_dir, layer, tok);
                    let _ = std::fs::write(&gpath, &gbuf);
                    let f16_slice = unsafe { std::slice::from_raw_parts(gbuf.as_ptr() as *const u16, attn_total) };
                    let gated_l2: f32 = f16_slice.iter()
                        .map(|&bits| { let v = half::f16::from_bits(bits).to_f32(); v * v })
                        .sum::<f32>().sqrt();
                    tracing::info!("GQA DIAG L{} tok{}: gated_output L2={:.4}", layer, tok, gated_l2);
                }

                // Step 10: O projection — attn_dev [attn_dim] → hidden_state [hidden_dim]
                if let Some((o_nvfp4_ptr, _o_m, o_nvfp4_k)) = o_nvfp4 {
                    // NVFP4 path: attn_dev (FP16) → FP32 → FP4 → MMA GEMV → FP32
                    kernels::f16_to_f32(
                        self.attn_dev.as_ptr(),
                        self.dn_qkv_f32_dev.as_mut_ptr(), // reuse as FP32 temp
                        attn_dim,
                        &self.stream,
                    )?;
                    kernels::quantize_activation_fp4_fast(
                        self.dn_qkv_f32_dev.as_ptr(),
                        self.preq_inner_act_fp4.as_mut_ptr(),
                        self.preq_inner_act_scales.as_mut_ptr(),
                        o_nvfp4_k,
                        &self.stream,
                    )?;
                    kernels::gemv_mma_nvfp4_tiled(
                        self.preq_inner_act_fp4.as_ptr(),
                        self.preq_inner_act_scales.as_ptr(),
                        o_nvfp4_ptr,
                        self.layer_output_buf.as_mut_ptr(), // FP32 output (hidden_dim*4)
                        o_nvfp4_k,
                        hidden_dim,
                        &self.stream,
                    )?;
                } else {
                    kernels::linear_projection(
                        self.attn_dev.as_ptr(),
                        o_ptr,
                        self.hidden_state.as_mut_ptr(),
                        attn_dim,
                        hidden_dim,
                        &self.stream,
                    )?;
                }

                // Dump O projection output
                if gqa_diag {
                    self.stream.synchronize()?;
                    let dump_dir = "/home/brian/code/vib3/dump";
                    let tok = self.position;
                    if o_nvfp4.is_some() {
                        // NVFP4: FP32 output in layer_output_buf
                        let mut obuf = vec![0u8; hidden_dim * 4];
                        self.layer_output_buf.copy_to_host(&mut obuf)?;
                        let opath = format!("{}/vib3_gqa_o_proj_f32_L{}_tok{}.bin", dump_dir, layer, tok);
                        let _ = std::fs::write(&opath, &obuf);
                        let f32_slice = unsafe { std::slice::from_raw_parts(obuf.as_ptr() as *const f32, hidden_dim) };
                        let o_l2 = f32_slice.iter().map(|v| v * v).sum::<f32>().sqrt();
                        tracing::info!("GQA DIAG L{} tok{}: o_proj (NVFP4→F32) L2={:.4}", layer, tok, o_l2);
                    } else {
                        // FP16 output in hidden_state
                        let mut obuf = vec![0u8; hidden_dim * 2];
                        self.hidden_state.copy_to_host(&mut obuf)?;
                        let opath = format!("{}/vib3_gqa_o_proj_f16_L{}_tok{}.bin", dump_dir, layer, tok);
                        let _ = std::fs::write(&opath, &obuf);
                        let f16_slice = unsafe { std::slice::from_raw_parts(obuf.as_ptr() as *const u16, hidden_dim) };
                        let o_l2: f32 = f16_slice.iter()
                            .map(|&bits| { let v = half::f16::from_bits(bits).to_f32(); v * v })
                            .sum::<f32>().sqrt();
                        tracing::info!("GQA DIAG L{} tok{}: o_proj (FP16) L2={:.4}", layer, tok, o_l2);
                    }
                }
            } else {
                // ── Standard (non-gated) GQA Attention Pipeline ──
                // Step 2: RoPE — apply rotary position embeddings in-place on Q and K (FP16)
                let err = unsafe {
                    cuda_ffi::vib3_launch_rope_apply(
                        self.q_proj_dev.as_mut_ptr(),
                        self.k_proj_dev.as_mut_ptr(),
                        head_dim as i32,
                        num_heads as i32,
                        num_kv_heads as i32,
                        self.d_position.as_ptr(), // device-side position
                        rope_base,
                        stream_ptr,
                    )
                };
                if err != 0 {
                    return Err(Error::Cuda(format!(
                        "vib3_launch_rope_apply failed (err={})",
                        err
                    )));
                }

                // Step 3: Append post-RoPE K/V to the GPU-resident KV cache
                let err = unsafe {
                    cuda_ffi::vib3_launch_kv_cache_append(
                        self.gpu_kv_k[layer].as_mut_ptr(),
                        self.gpu_kv_v[layer].as_mut_ptr(),
                        self.k_proj_dev.as_ptr(),
                        self.v_proj_dev.as_ptr(),
                        self.max_kv_len as i32,
                        head_dim as i32,
                        num_kv_heads as i32,
                        self.d_position.as_ptr(), // device-side position
                        stream_ptr,
                    )
                };
                if err != 0 {
                    return Err(Error::Cuda(format!(
                        "vib3_launch_kv_cache_append failed (err={})",
                        err
                    )));
                }

                // Step 4: Decode attention — single-query GQA attention on GPU
                // Output goes to attn_dev [num_heads * head_dim] FP16
                let scale = 1.0f32 / (head_dim as f32).sqrt();
                let err = unsafe {
                    cuda_ffi::vib3_launch_decode_attention(
                        self.q_proj_dev.as_ptr(),
                        self.gpu_kv_k[layer].as_ptr(),
                        self.gpu_kv_v[layer].as_ptr(),
                        self.attn_dev.as_mut_ptr(),
                        head_dim as i32,
                        num_heads as i32,
                        num_kv_heads as i32,
                        self.d_position.as_ptr(), // device-side position (kernel computes seq_len=pos+1)
                        self.max_kv_len as i32,
                        scale,
                        stream_ptr,
                    )
                };
                if err != 0 {
                    return Err(Error::Cuda(format!(
                        "vib3_launch_decode_attention failed (err={})",
                        err
                    )));
                }

                // Step 5: O projection — GPU GEMV: attn_output × O_weights^T → hidden_state
                let attn_dim = num_heads * head_dim;
                if let Some((o_nvfp4_ptr, _o_m, o_nvfp4_k)) = o_nvfp4 {
                    // NVFP4 path
                    kernels::f16_to_f32(
                        self.attn_dev.as_ptr(),
                        self.dn_qkv_f32_dev.as_mut_ptr(),
                        attn_dim,
                        &self.stream,
                    )?;
                    kernels::quantize_activation_fp4_fast(
                        self.dn_qkv_f32_dev.as_ptr(),
                        self.preq_inner_act_fp4.as_mut_ptr(),
                        self.preq_inner_act_scales.as_mut_ptr(),
                        o_nvfp4_k,
                        &self.stream,
                    )?;
                    kernels::gemv_mma_nvfp4_tiled(
                        self.preq_inner_act_fp4.as_ptr(),
                        self.preq_inner_act_scales.as_ptr(),
                        o_nvfp4_ptr,
                        self.layer_output_buf.as_mut_ptr(),
                        o_nvfp4_k,
                        hidden_dim,
                        &self.stream,
                    )?;
                } else {
                    kernels::linear_projection(
                        self.attn_dev.as_ptr(),
                        o_ptr,
                        self.hidden_state.as_mut_ptr(),
                        attn_dim,
                        hidden_dim,
                        &self.stream,
                    )?;
                }
            }

            // FP32 residual accumulation
            if o_nvfp4.is_some() {
                // NVFP4 O proj output is already FP32 in layer_output_buf
                kernels::residual_add_f32_f32(
                    self.hidden_state_f32.as_mut_ptr(),
                    self.layer_output_buf.as_ptr(),
                    hidden_dim,
                    &self.stream,
                )?;
            } else {
                // FP16 O proj output: hidden_state_f32 += f32(hidden_state)
                kernels::residual_add_fp32(
                    self.hidden_state_f32.as_mut_ptr(),
                    self.hidden_state.as_ptr(),
                    hidden_dim,
                    &self.stream,
                )?;
            }

            tracing::debug!("GPU decode attn ok: layer={}, pos={}", layer_idx, position);
            Ok(true)
        }
    }

    /// Run MLA (Multi-head Latent Attention) for one layer.
    ///
    /// GPU-accelerated: The 4 large GEMV projections (q_a, q_b, kv_a, o_proj)
    /// run on GPU via `linear_projection()`. Small intermediate results are
    /// D2H'd for CPU-side RMSNorm, RoPE, and attention dot products.
    /// Uses "absorbed attention" (DeepSeek-V2 inference optimization):
    /// instead of reconstructing K/V from kv_b_proj at every cached position,
    /// absorb kv_b_proj into Q once per layer, then compute attention scores
    /// as dot products against the compressed latent. This eliminates O(seq_len)
    /// per-head GEMV calls.
    ///
    /// kv_b_proj is loaded to host (not VRAM) since it's only used once per layer
    /// for the Q absorption step, not per-position.
    ///
    /// Weight segments:
    /// - segment 20: q_a_proj   [q_lora_rank, hidden_dim]
    /// - segment 21: q_b_proj   [num_heads*(nope+rope), q_lora_rank]
    /// - segment 22: kv_a_proj  [kv_lora_rank+rope_dim, hidden_dim]
    /// - segment 23: kv_b_proj  [num_heads*(nope+v), kv_lora_rank]  — on host
    /// - segment 5:  o_proj     [hidden_dim, num_heads*v_head_dim]
    /// Run MLA attention for one layer.
    /// Returns `Ok(None)` when GPU path was used — output is in `self.mla_o_out_dev`.
    /// Returns `Ok(Some(vec))` when CPU fallback was used — output is the returned vector.
    async fn run_mla_attention(
        &mut self,
        layer_idx: u16,
        _hidden_state: &[f16],
    ) -> Result<Option<Vec<f16>>> {
        let mla = self.model_config.mla.clone().unwrap();
        let hidden_dim = self.model_config.hidden_dim as usize;
        let num_heads = self.model_config.num_heads as usize;
        let q_lora_rank = mla.q_lora_rank as usize;
        let kv_lora_rank = mla.kv_lora_rank as usize;
        let qk_rope_dim = mla.qk_rope_head_dim as usize;
        let qk_nope_dim = mla.qk_nope_head_dim as usize;
        let v_head_dim = mla.v_head_dim as usize;
        let q_head_dim = qk_nope_dim + qk_rope_dim;
        let k_head_dim = qk_nope_dim + qk_rope_dim;
        let kv_a_dim = kv_lora_rank + qk_rope_dim;
        let q_full_dim = num_heads * q_head_dim;
        let o_dim = num_heads * v_head_dim;

        // ── Load MLA weight tensors to VRAM (GPU path) or host (CPU fallback) ──
        // Load q_a, q_b, kv_a, o_proj to VRAM for GPU GEMV
        self.ensure_shared_tensor_device(layer_idx, 20).await; // q_a_proj
        self.ensure_shared_tensor_device(layer_idx, 21).await; // q_b_proj
        self.ensure_shared_tensor_device(layer_idx, 22).await; // kv_a_proj
        self.ensure_shared_tensor_device(layer_idx, 5).await;  // o_proj
        // kv_b_proj stays on host — only load if not already cached as F32
        let kvb_data = if self.kv_b_proj_f32_cache.contains_key(&layer_idx) {
            None // Already cached as F32, skip the 16.8MB clone
        } else {
            self.load_shared_tensor(layer_idx, 23).await
        };
        // Layernorm weights (small, 1 page each)
        let q_norm_data = self.load_shared_tensor(layer_idx, 24).await;
        let kv_norm_data = self.load_shared_tensor(layer_idx, 25).await;

        let qa_device = self.get_device_tensor(layer_idx, 20).map(|(p, s)| (p as usize, s));
        let qb_device = self.get_device_tensor(layer_idx, 21).map(|(p, s)| (p as usize, s));
        let kva_device = self.get_device_tensor(layer_idx, 22).map(|(p, s)| (p as usize, s));
        let o_device = self.get_device_tensor(layer_idx, 5).map(|(p, s)| (p as usize, s));

        // Check if GPU path is available (all 4 projection weights on device)
        let use_gpu = self.stream.is_real()
            && qa_device.is_some()
            && qb_device.is_some()
            && kva_device.is_some()
            && o_device.is_some();

        if use_gpu {
            // ── GPU-accelerated MLA path (v3: fully-GPU pipeline) ──
            //
            // Pipeline (ALL compute on GPU, 1 sync at end):
            //   1. GPU: q_a_proj + kv_a_proj (simultaneous)
            //   2. GPU: RMSNorm q_compressed (in-place FP16)
            //   3. GPU: q_b_proj → q_full
            //   4. GPU: mla_kv_cache_append (fused: FP16→F32 + RMSNorm + RoPE + cache write)
            //   5. GPU: mla_q_absorb_rope (fused: Q absorption + RoPE)
            //   6. GPU: MLA decode attention (scores + softmax + weighted latent)
            //   7. GPU: V reconstruction
            //   8. GPU: f32_to_f16 + D2D copy
            //   9. GPU: o_proj
            //  10. sync + D2H output
            //
            // Falls back to CPU attention if GPU KV cache / kv_b_proj not ready.

            let (qa_addr, _) = qa_device.unwrap();
            let qa_ptr = qa_addr as *const u8;
            let (kva_addr, _) = kva_device.unwrap();
            let kva_ptr = kva_addr as *const u8;

            // Pre-step profiling: sync before launching q_a + kv_a
            // Step 1: Launch q_a_proj and kv_a_proj on GPU simultaneously
            kernels::linear_projection(
                self.hidden_state.as_ptr(),
                qa_ptr,
                self.mla_q_compressed_dev.as_mut_ptr(),
                hidden_dim,
                q_lora_rank,
                &self.stream,
            )?;
            kernels::linear_projection(
                self.hidden_state.as_ptr(),
                kva_ptr,
                self.mla_kv_a_dev.as_mut_ptr(),
                hidden_dim,
                kv_a_dim,
                &self.stream,
            )?;

            // ── Fully-GPU MLA Pipeline (0 intermediate syncs) ──
            //
            // All compute stays on GPU until the final o_proj output.
            // Pipeline:
            //   1. q_a_proj + kv_a_proj already launched above (concurrent)
            //   2. GPU RMSNorm on q_compressed (in-place FP16, q_norm weight on device)
            //   3. GPU q_b_proj (uses normed q_compressed already on device)
            //   4. GPU mla_kv_cache_append (fused: FP16→F32 + RMSNorm + RoPE + KV cache write)
            //   5. GPU mla_q_absorb_rope (fused: Q absorption + RoPE, uses q_full on device)
            //   6. GPU MLA decode attention (scores + softmax + weighted latent)
            //   7. GPU V reconstruction (kv_b_proj_v × v_latent → v_out)
            //   8. (skipped — v_out stays F32, no FP16 truncation)
            //   9. GPU o_proj (F32 in, FP16 weight, F32 out)
            //  10. Single sync + D2H output
            //
            // Syncs: 1 (step 10) — down from 3 in the previous pipeline.

            // Convert kv_b_proj to F32 once and cache (host + device)
            if !self.kv_b_proj_f32_cache.contains_key(&layer_idx) {
                if let Some(ref buf) = kvb_data {
                    let kv_b_f16 = unsafe {
                        std::slice::from_raw_parts(buf.as_ptr() as *const f16, buf.len() / 2)
                    };
                    let kv_b_f32: Vec<f32> = kv_b_f16.iter().map(|v| v.to_f32()).collect();
                    self.kv_b_proj_f32_cache.insert(layer_idx, kv_b_f32);
                }
            }

            // Upload kv_b_proj F32 to GPU if not already cached
            if !self.kv_b_proj_f32_device.contains_key(&layer_idx) {
                if let Some(kv_b) = self.kv_b_proj_f32_cache.get(&layer_idx) {
                    let kv_b_bytes = kv_b.len() * 4;
                    let dev_buf = DeviceBuffer::new(kv_b_bytes)?;
                    let src = unsafe {
                        std::slice::from_raw_parts(kv_b.as_ptr() as *const u8, kv_b_bytes)
                    };
                    dev_buf.copy_from_host(src)?;
                    self.kv_b_proj_f32_device.insert(layer_idx, dev_buf);
                }
            }

            let layer = layer_idx as usize;

            // Check if fully-GPU attention path is available
            let gpu_attn_ok = layer < self.mla_gpu_kv_latent.len()
                && self.kv_b_proj_f32_device.contains_key(&layer_idx)
                && self.mla_rope_freqs_dev.is_some();

            // Diagnostic: log which path is used for first 2 tokens at layers 0,1,30,60
            if self.position <= 1 && (layer_idx <= 1 || layer_idx == 30 || layer_idx == 60) {
                let has_kvb = self.kv_b_proj_f32_device.contains_key(&layer_idx);
                let has_rope = self.mla_rope_freqs_dev.is_some();
                let has_kv_buf = layer < self.mla_gpu_kv_latent.len();
                let mla_seq = self.mla_kv_cache.as_ref().map(|c| c.layers[layer].seq_len).unwrap_or(0);
                tracing::info!(
                    "MLA PATH pos={} L{}: gpu_ok={} (kvb_dev={}, rope={}, kv_buf={}), kv_seq_len={}",
                    self.position, layer_idx, gpu_attn_ok, has_kvb, has_rope, has_kv_buf, mla_seq,
                );
            }

            if gpu_attn_ok {
                // ── Fully-GPU MLA Attention Path (zero syncs) ──
                // All kernels queued on stream. Caller does residual add + MoE sync.

                // Step 2: GPU RMSNorm on q_compressed (in-place FP16)
                let eps = self.model_config.rms_norm_eps;
                self.ensure_shared_tensor_device(layer_idx, 24).await;
                let q_norm_dev = self.get_device_tensor(layer_idx, 24);
                if let Some((q_norm_ptr, _)) = q_norm_dev {
                    kernels::rms_norm(
                        self.mla_q_compressed_dev.as_mut_ptr(),
                        q_norm_ptr,
                        q_lora_rank,
                        eps,
                        &self.stream,
                    )?;
                } else {
                    kernels::rms_norm_no_weight(
                        self.mla_q_compressed_dev.as_mut_ptr(),
                        q_lora_rank,
                        eps,
                        &self.stream,
                    )?;
                }

                // Step 3: GPU q_b_proj — q_normed[1536] → q_full[12288]
                let (qb_addr, _) = qb_device.unwrap();
                let qb_ptr = qb_addr as *const u8;
                kernels::linear_projection(
                    self.mla_q_compressed_dev.as_ptr(),
                    qb_ptr,
                    self.mla_q_full_dev.as_mut_ptr(),
                    q_lora_rank,
                    q_full_dim,
                    &self.stream,
                )?;

                // Step 4: GPU mla_kv_cache_append — fused KV processing + cache write
                self.ensure_shared_tensor_device(layer_idx, 25).await;
                let kv_norm_ptr = self.get_device_tensor(layer_idx, 25)
                    .map(|(p, _)| p)
                    .unwrap_or(std::ptr::null());
                let rope_freqs_ptr = self.mla_rope_freqs_dev.as_ref().unwrap().as_ptr();

                let mla_kv = self.mla_kv_cache.as_mut().unwrap();
                if layer >= mla_kv.layers.len() {
                    return Ok(Some(vec![f16::from_f32(0.0); hidden_dim]));
                }
                mla_kv.layers[layer].seq_len += 1;
                let seq_len = mla_kv.layers[layer].seq_len;
                let pos = seq_len - 1;

                kernels::mla_kv_cache_append(
                    self.mla_kv_a_dev.as_ptr(),
                    kv_norm_ptr,
                    rope_freqs_ptr,
                    self.mla_gpu_kv_latent[layer].as_mut_ptr(),
                    self.mla_gpu_kv_rope[layer].as_mut_ptr(),
                    kv_lora_rank,
                    qk_rope_dim,
                    pos,
                    eps,
                    &self.stream,
                )?;

                // Step 5: GPU mla_q_absorb_rope — fused Q absorption + RoPE
                let kv_b_dev = self.kv_b_proj_f32_device.get(&layer_idx).unwrap();
                kernels::mla_q_absorb_rope(
                    self.mla_q_full_dev.as_ptr(),
                    kv_b_dev.as_ptr(),
                    rope_freqs_ptr,
                    self.mla_q_absorbed_dev.as_mut_ptr(),
                    self.mla_q_rope_f32_dev.as_mut_ptr(),
                    q_head_dim,
                    qk_nope_dim,
                    qk_rope_dim,
                    v_head_dim,
                    kv_lora_rank,
                    num_heads,
                    self.position,
                    &self.stream,
                )?;

                // Step 6: GPU MLA decode attention
                let scale = mla.softmax_scale;
                kernels::mla_decode_attention(
                    self.mla_q_absorbed_dev.as_ptr(),
                    self.mla_q_rope_f32_dev.as_ptr(),
                    self.mla_gpu_kv_latent[layer].as_ptr(),
                    self.mla_gpu_kv_rope[layer].as_ptr(),
                    self.mla_v_latent_dev.as_mut_ptr(),
                    kv_lora_rank,
                    qk_rope_dim,
                    num_heads,
                    seq_len,
                    scale,
                    &self.stream,
                )?;

                // Step 7: GPU V reconstruction
                kernels::mla_v_reconstruct(
                    self.mla_v_latent_dev.as_ptr(),
                    kv_b_dev.as_ptr(),
                    self.mla_v_out_f32_dev.as_mut_ptr(),
                    kv_lora_rank,
                    qk_nope_dim,
                    v_head_dim,
                    num_heads,
                    &self.stream,
                )?;

                // Step 8: Skip — V_out stays in F32 (mla_v_out_f32_dev).
                // The o_proj will read F32 directly to avoid FP16 truncation.

                // ── MLA intermediate diagnostics for L1 and L6 at pos 0 ──
                // Gated behind VIB3_DIAG=1 to avoid overhead in normal inference.
                let mla_diag = self.diag_enabled && self.position == 0 && (layer_idx == 1 || layer_idx == 6);
                if mla_diag {
                    self.stream.synchronize()?;

                    // Helper: compute L2 norm of FP16 buffer on host
                    fn f16_l2(buf: &[u8]) -> f32 {
                        let f16s = unsafe {
                            std::slice::from_raw_parts(buf.as_ptr() as *const f16, buf.len() / 2)
                        };
                        f16s.iter().map(|v| { let f = v.to_f32(); f * f }).sum::<f32>().sqrt()
                    }
                    fn f32_l2(buf: &[u8]) -> f32 {
                        let f32s = unsafe {
                            std::slice::from_raw_parts(buf.as_ptr() as *const f32, buf.len() / 4)
                        };
                        f32s.iter().map(|v| v * v).sum::<f32>().sqrt()
                    }
                    fn f16_first8(buf: &[u8]) -> String {
                        let f16s = unsafe {
                            std::slice::from_raw_parts(buf.as_ptr() as *const f16, buf.len() / 2)
                        };
                        let n = std::cmp::min(8, f16s.len());
                        f16s[..n].iter().map(|v| format!("{:.4}", v.to_f32())).collect::<Vec<_>>().join(", ")
                    }
                    fn f32_first8(buf: &[u8]) -> String {
                        let f32s = unsafe {
                            std::slice::from_raw_parts(buf.as_ptr() as *const f32, buf.len() / 4)
                        };
                        let n = std::cmp::min(8, f32s.len());
                        f32s[..n].iter().map(|v| format!("{:.6}", v)).collect::<Vec<_>>().join(", ")
                    }

                    // D2H from GPU KV cache at pos=0 (latent + rope written by kv_cache_append)
                    let kv_lat_bytes = kv_lora_rank * 4;
                    let mut kv_lat_buf = vec![0u8; kv_lat_bytes];
                    // pos=0 is at offset 0 in the cache buffer
                    self.mla_gpu_kv_latent[layer].copy_to_host(&mut kv_lat_buf)?;
                    let kv_rope_bytes = qk_rope_dim * 4;
                    let mut kv_rope_buf = vec![0u8; kv_rope_bytes];
                    self.mla_gpu_kv_rope[layer].copy_to_host(&mut kv_rope_buf)?;

                    // D2H q_compressed (after norm)
                    let qc_bytes = q_lora_rank * 2;
                    let mut qc_buf = vec![0u8; qc_bytes];
                    self.mla_q_compressed_dev.copy_to_host(&mut qc_buf)?;

                    // D2H q_full
                    let qf_bytes = q_full_dim * 2;
                    let mut qf_buf = vec![0u8; qf_bytes];
                    self.mla_q_full_dev.copy_to_host(&mut qf_buf)?;

                    // D2H q_absorbed
                    let qa_bytes = num_heads * kv_lora_rank * 4;
                    let mut qa_buf = vec![0u8; qa_bytes];
                    self.mla_q_absorbed_dev.copy_to_host(&mut qa_buf)?;

                    // D2H q_rope_f32
                    let qr_bytes = num_heads * qk_rope_dim * 4;
                    let mut qr_buf = vec![0u8; qr_bytes];
                    self.mla_q_rope_f32_dev.copy_to_host(&mut qr_buf)?;

                    // D2H v_latent (output of decode attention)
                    let vl_bytes = num_heads * kv_lora_rank * 4;
                    let mut vl_buf = vec![0u8; vl_bytes];
                    self.mla_v_latent_dev.copy_to_host(&mut vl_buf)?;

                    // D2H v_out_f32 (output of V reconstruction)
                    let vo_bytes = o_dim * 4;
                    let mut vo_buf = vec![0u8; vo_bytes];
                    self.mla_v_out_f32_dev.copy_to_host(&mut vo_buf)?;

                    // Note: mla_attn_out_dev is no longer used (F32 o_proj path skips it)
                    // V_out F32 goes directly to o_proj

                    tracing::info!(
                        "MLA DIAG L{} pos=0: q_compressed L2={:.4} [{}], q_full L2={:.4} [{}]",
                        layer_idx,
                        f16_l2(&qc_buf), f16_first8(&qc_buf),
                        f16_l2(&qf_buf), f16_first8(&qf_buf),
                    );
                    tracing::info!(
                        "MLA DIAG L{} pos=0: kv_latent(normed) L2={:.4} [{}], kv_rope L2={:.4} [{}]",
                        layer_idx,
                        f32_l2(&kv_lat_buf), f32_first8(&kv_lat_buf),
                        f32_l2(&kv_rope_buf), f32_first8(&kv_rope_buf),
                    );
                    tracing::info!(
                        "MLA DIAG L{} pos=0: q_absorbed L2={:.4} [{}], q_rope L2={:.4} [{}]",
                        layer_idx,
                        f32_l2(&qa_buf), f32_first8(&qa_buf),
                        f32_l2(&qr_buf), f32_first8(&qr_buf),
                    );
                    tracing::info!(
                        "MLA DIAG L{} pos=0: v_latent L2={:.4} [{}], v_out_f32 L2={:.4} [{}]",
                        layer_idx,
                        f32_l2(&vl_buf), f32_first8(&vl_buf),
                        f32_l2(&vo_buf), f32_first8(&vo_buf),
                    );
                    tracing::info!(
                        "MLA DIAG L{} pos=0: v_out_f32 L2={:.4} (direct to o_proj, no F16 intermediate)",
                        layer_idx,
                        f32_l2(&vo_buf),
                    );

                    // Dump binary files for detailed comparison
                    let dump_dir = "/model/dump";
                    let _ = std::fs::create_dir_all(dump_dir);
                    let _ = std::fs::write(format!("{}/mla_L{}_kv_latent_normed.f32", dump_dir, layer_idx), &kv_lat_buf);
                    let _ = std::fs::write(format!("{}/mla_L{}_kv_rope.f32", dump_dir, layer_idx), &kv_rope_buf);
                    let _ = std::fs::write(format!("{}/mla_L{}_q_compressed_normed.f16", dump_dir, layer_idx), &qc_buf);
                    let _ = std::fs::write(format!("{}/mla_L{}_q_full.f16", dump_dir, layer_idx), &qf_buf);
                    let _ = std::fs::write(format!("{}/mla_L{}_q_absorbed.f32", dump_dir, layer_idx), &qa_buf);
                    let _ = std::fs::write(format!("{}/mla_L{}_q_rope.f32", dump_dir, layer_idx), &qr_buf);
                    let _ = std::fs::write(format!("{}/mla_L{}_v_latent.f32", dump_dir, layer_idx), &vl_buf);
                    let _ = std::fs::write(format!("{}/mla_L{}_v_out.f32", dump_dir, layer_idx), &vo_buf);
                    tracing::info!("MLA DIAG L{}: dumped all intermediates to {}", layer_idx, dump_dir);
                }
            } else {
                // ── CPU attention fallback ──
                // This path handles the case where GPU KV cache or kv_b_proj_f32
                // isn't on device, or rope_freqs not available.
                // Must do everything on CPU: sync, D2H, norms, RoPE, attention.

                // Sync to get q_a and kv_a results
                self.stream.synchronize()?;

                // D2H q_compressed
                let q_compressed_bytes = q_lora_rank * std::mem::size_of::<f16>();
                let mut q_compressed_host = vec![0u8; q_compressed_bytes];
                self.mla_q_compressed_dev.copy_to_host(&mut q_compressed_host)?;
                let q_compressed_f16 = unsafe {
                    std::slice::from_raw_parts(q_compressed_host.as_ptr() as *const f16, q_lora_rank)
                };
                let q_compressed_f32: Vec<f32> = q_compressed_f16.iter().map(|v| v.to_f32()).collect();

                // D2H kv_a_out
                let kv_a_bytes_sz = kv_a_dim * std::mem::size_of::<f16>();
                let mut kv_a_host = vec![0u8; kv_a_bytes_sz];
                self.mla_kv_a_dev.copy_to_host(&mut kv_a_host)?;
                let kv_a_f16 = unsafe {
                    std::slice::from_raw_parts(kv_a_host.as_ptr() as *const f16, kv_a_dim)
                };
                let kv_a_f32: Vec<f32> = kv_a_f16.iter().map(|v| v.to_f32()).collect();

                // CPU RMSNorm on q_compressed
                let q_normed = if let Some(ref buf) = q_norm_data {
                    let w = unsafe { std::slice::from_raw_parts(buf.as_ptr() as *const f16, buf.len() / 2) };
                    rms_norm_f32_with_weight(&q_compressed_f32, w)
                } else {
                    rms_norm_f32(&q_compressed_f32)
                };

                // H2D q_normed for q_b_proj
                let q_normed_f16: Vec<f16> = q_normed.iter().map(|v| f16::from_f32(*v)).collect();
                let q_normed_bytes_data = unsafe {
                    std::slice::from_raw_parts(q_normed_f16.as_ptr() as *const u8, q_compressed_bytes)
                };
                self.mla_q_compressed_dev.copy_from_host(q_normed_bytes_data)?;

                // q_b_proj on GPU
                let (qb_addr, _) = qb_device.unwrap();
                let qb_ptr = qb_addr as *const u8;
                kernels::linear_projection(
                    self.mla_q_compressed_dev.as_ptr(),
                    qb_ptr,
                    self.mla_q_full_dev.as_mut_ptr(),
                    q_lora_rank,
                    q_full_dim,
                    &self.stream,
                )?;
                self.stream.synchronize()?;

                // D2H q_full
                let q_full_bytes_sz = q_full_dim * std::mem::size_of::<f16>();
                let mut q_full_host = vec![0u8; q_full_bytes_sz];
                self.mla_q_full_dev.copy_to_host(&mut q_full_host)?;
                let q_full_f16 = unsafe {
                    std::slice::from_raw_parts(q_full_host.as_ptr() as *const f16, q_full_dim)
                };
                let q_full_f32: Vec<f32> = q_full_f16.iter().map(|v| v.to_f32()).collect();

                // CPU-side KV processing
                let yarn = crate::runtime::attention::YarnRopeConfig::from_kimi_k25(qk_rope_dim);
                let kv_latent_raw = &kv_a_f32[..kv_lora_rank];
                let mut k_rope_shared = kv_a_f32[kv_lora_rank..].to_vec();

                let kv_latent_normed = if let Some(ref buf) = kv_norm_data {
                    let w = unsafe { std::slice::from_raw_parts(buf.as_ptr() as *const f16, buf.len() / 2) };
                    rms_norm_f32_with_weight(kv_latent_raw, w)
                } else {
                    rms_norm_f32(kv_latent_raw)
                };
                crate::runtime::attention::apply_yarn_rope(&mut k_rope_shared, self.position, &yarn);

                let mla_kv = self.mla_kv_cache.as_mut().unwrap();
                if layer >= mla_kv.layers.len() {
                    return Ok(Some(vec![f16::from_f32(0.0); hidden_dim]));
                }
                mla_kv.layers[layer].append(&kv_latent_normed, &k_rope_shared);
                let seq_len = mla_kv.layers[layer].seq_len;

                // Also update GPU KV cache for future use
                if layer < self.mla_gpu_kv_latent.len() {
                    let pos = seq_len - 1;
                    let latent_offset = pos * kv_lora_rank * 4;
                    let latent_bytes = unsafe {
                        std::slice::from_raw_parts(
                            kv_latent_normed.as_ptr() as *const u8,
                            kv_lora_rank * 4,
                        )
                    };
                    cuda_ffi::memcpy_h2d_sync(
                        unsafe { self.mla_gpu_kv_latent[layer].as_mut_ptr().add(latent_offset) },
                        latent_bytes.as_ptr(),
                        kv_lora_rank * 4,
                    )?;
                    let rope_offset = pos * qk_rope_dim * 4;
                    let rope_bytes = unsafe {
                        std::slice::from_raw_parts(
                            k_rope_shared.as_ptr() as *const u8,
                            qk_rope_dim * 4,
                        )
                    };
                    cuda_ffi::memcpy_h2d_sync(
                        unsafe { self.mla_gpu_kv_rope[layer].as_mut_ptr().add(rope_offset) },
                        rope_bytes.as_ptr(),
                        qk_rope_dim * 4,
                    )?;
                }

                // CPU attention with Q absorption
                let kv_b_f32_opt = self.kv_b_proj_f32_cache.get(&layer_idx);
                let mut attn_output = vec![0.0f32; o_dim];

                let mut q_heads: Vec<Vec<f32>> = Vec::with_capacity(num_heads);
                for h in 0..num_heads {
                    let head_start = h * q_head_dim;
                    let q_nope = &q_full_f32[head_start..head_start + qk_nope_dim];
                    let mut q_rope_h = q_full_f32[head_start + qk_nope_dim..head_start + q_head_dim].to_vec();
                    crate::runtime::attention::apply_yarn_rope(&mut q_rope_h, self.position, &yarn);
                    let mut q_head = Vec::with_capacity(k_head_dim);
                    q_head.extend_from_slice(q_nope);
                    q_head.extend_from_slice(&q_rope_h);
                    q_heads.push(q_head);
                }

                if let Some(kv_b) = kv_b_f32_opt {
                    let mut q_absorbed: Vec<Vec<f32>> = Vec::with_capacity(num_heads);
                    for h in 0..num_heads {
                        let head_offset = h * (qk_nope_dim + v_head_dim);
                        let q_nope = &q_full_f32[h * q_head_dim..h * q_head_dim + qk_nope_dim];
                        let mut absorbed = vec![0.0f32; kv_lora_rank];
                        for i in 0..qk_nope_dim {
                            let row_start = (head_offset + i) * kv_lora_rank;
                            let q_val = q_nope[i];
                            let kv_row = &kv_b[row_start..row_start + kv_lora_rank];
                            for j in 0..kv_lora_rank {
                                absorbed[j] += q_val * kv_row[j];
                            }
                        }
                        q_absorbed.push(absorbed);
                    }

                    let scale = mla.softmax_scale;
                    let mla_kv_ref = self.mla_kv_cache.as_ref().unwrap();
                    let mut all_scores: Vec<Vec<f32>> = Vec::with_capacity(num_heads);

                    for h in 0..num_heads {
                        let q_rope_h = &q_heads[h][qk_nope_dim..];
                        let mut scores = Vec::with_capacity(seq_len);
                        for pos in 0..seq_len {
                            let latent_start = pos * kv_lora_rank;
                            let latent = &mla_kv_ref.layers[layer].kv_latent[latent_start..latent_start + kv_lora_rank];
                            let mut dot = 0.0f32;
                            for j in 0..kv_lora_rank {
                                dot += q_absorbed[h][j] * latent[j];
                            }
                            let rope_start = pos * qk_rope_dim;
                            let k_rope_pos = &mla_kv_ref.layers[layer].k_rope[rope_start..rope_start + qk_rope_dim];
                            for d in 0..qk_rope_dim {
                                dot += q_rope_h[d] * k_rope_pos[d];
                            }
                            scores.push(dot * scale);
                        }
                        all_scores.push(scores);
                    }

                    for h in 0..num_heads {
                        let scores = &mut all_scores[h];
                        if scores.is_empty() { continue; }
                        let max_score = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                        for s in scores.iter_mut() { *s = (*s - max_score).exp(); }
                        let sum: f32 = scores.iter().sum();
                        if sum > 0.0 { for s in scores.iter_mut() { *s /= sum; } }

                        let mut v_latent = vec![0.0f32; kv_lora_rank];
                        for (pos, &w) in scores.iter().enumerate() {
                            if w < 1e-8 { continue; }
                            let latent_start = pos * kv_lora_rank;
                            let latent = &mla_kv_ref.layers[layer].kv_latent[latent_start..latent_start + kv_lora_rank];
                            for j in 0..kv_lora_rank {
                                v_latent[j] += w * latent[j];
                            }
                        }

                        let head_v_offset = h * (qk_nope_dim + v_head_dim) + qk_nope_dim;
                        let head_out_start = h * v_head_dim;
                        for d in 0..v_head_dim {
                            let row_start = (head_v_offset + d) * kv_lora_rank;
                            let kv_row = &kv_b[row_start..row_start + kv_lora_rank];
                            let mut val = 0.0f32;
                            for j in 0..kv_lora_rank {
                                val += kv_row[j] * v_latent[j];
                            }
                            attn_output[head_out_start + d] = val;
                        }
                    }
                } else {
                    // No kv_b_proj — direct latent attention (degraded quality)
                    let scale = mla.softmax_scale;
                    let mla_kv_ref = self.mla_kv_cache.as_ref().unwrap();
                    for h in 0..num_heads {
                        let q_head = &q_heads[h];
                        let mut scores = Vec::with_capacity(seq_len);
                        for pos in 0..seq_len {
                            let latent_start = pos * kv_lora_rank;
                            let latent = &mla_kv_ref.layers[layer].kv_latent[latent_start..latent_start + kv_lora_rank];
                            let mut dot = 0.0f32;
                            for d in 0..qk_nope_dim.min(kv_lora_rank) {
                                dot += q_head[d] * latent[d];
                            }
                            let rope_start = pos * qk_rope_dim;
                            let k_rope_pos = &mla_kv_ref.layers[layer].k_rope[rope_start..rope_start + qk_rope_dim];
                            for d in 0..qk_rope_dim {
                                dot += q_head[qk_nope_dim + d] * k_rope_pos[d];
                            }
                            scores.push(dot * scale);
                        }
                        if scores.is_empty() { continue; }
                        let max_score = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                        let mut exp_scores: Vec<f32> = scores.iter().map(|s| (s - max_score).exp()).collect();
                        let sum: f32 = exp_scores.iter().sum();
                        if sum > 0.0 { for s in &mut exp_scores { *s /= sum; } }
                        let head_out_start = h * v_head_dim;
                        for (pos, &w) in exp_scores.iter().enumerate() {
                            if w < 1e-8 { continue; }
                            let latent_start = pos * kv_lora_rank;
                            let latent = &mla_kv_ref.layers[layer].kv_latent[latent_start..latent_start + kv_lora_rank];
                            let len = v_head_dim.min(kv_lora_rank);
                            for d in 0..len {
                                attn_output[head_out_start + d] += w * latent[d];
                            }
                        }
                    }
                }

                // H2D attn_output as F32 for o_proj (avoid FP16 truncation)
                let attn_bytes = unsafe {
                    std::slice::from_raw_parts(attn_output.as_ptr() as *const u8, o_dim * 4)
                };
                self.mla_v_out_f32_dev.copy_from_host(attn_bytes)?;
            }

            // Step 11: O projection on GPU (F32 in, FP16 weight, F32 out):
            // V_out[8192] × O_weight^T → output[7168]
            let (o_addr, _) = o_device.unwrap();
            let o_ptr = o_addr as *const u8;

            kernels::linear_projection_f32_to_f32(
                self.mla_v_out_f32_dev.as_ptr(),
                o_ptr,
                self.mla_o_out_dev.as_mut_ptr(),
                o_dim,
                hidden_dim,
                &self.stream,
            )?;

            // GPU path complete — output is in mla_o_out_dev (FP32, hidden_dim elements).
            // Do NOT sync or D2H here — caller will do GPU residual add.

            Ok(None) // Signal: output is on GPU in mla_o_out_dev
        } else {
            // ── CPU fallback MLA path (original) ──
            // D2H hidden_state since GPU path wasn't available
            self.stream.synchronize()?;
            let hidden_bytes_for_cpu = hidden_dim * std::mem::size_of::<f16>();
            let mut cpu_hidden_buf = vec![0u8; hidden_bytes_for_cpu];
            self.hidden_state.copy_to_host(&mut cpu_hidden_buf)?;
            let cpu_hidden_state = unsafe {
                std::slice::from_raw_parts(cpu_hidden_buf.as_ptr() as *const f16, hidden_dim)
            };

            // Load all weights to host
            let qa_data = self.load_shared_tensor(layer_idx, 20).await;
            let qb_data = self.load_shared_tensor(layer_idx, 21).await;
            let kva_data = self.load_shared_tensor(layer_idx, 22).await;
            let o_data = self.load_shared_tensor(layer_idx, 5).await;
            // CPU fallback always needs kv_b_proj as F16
            let kvb_fallback = if kvb_data.is_some() {
                kvb_data
            } else {
                self.load_shared_tensor(layer_idx, 23).await
            };

            let q_a_proj = qa_data.as_ref().map(|buf| unsafe {
                std::slice::from_raw_parts(buf.as_ptr() as *const f16, buf.len() / 2)
            });
            let q_b_proj = qb_data.as_ref().map(|buf| unsafe {
                std::slice::from_raw_parts(buf.as_ptr() as *const f16, buf.len() / 2)
            });
            let kv_a_proj = kva_data.as_ref().map(|buf| unsafe {
                std::slice::from_raw_parts(buf.as_ptr() as *const f16, buf.len() / 2)
            });
            let kv_b_proj = kvb_fallback.as_ref().map(|buf| unsafe {
                std::slice::from_raw_parts(buf.as_ptr() as *const f16, buf.len() / 2)
            });
            let o_proj = o_data.as_ref().map(|buf| unsafe {
                std::slice::from_raw_parts(buf.as_ptr() as *const f16, buf.len() / 2)
            });
            let q_norm = q_norm_data.as_ref().map(|buf| unsafe {
                std::slice::from_raw_parts(buf.as_ptr() as *const f16, buf.len() / 2)
            });
            let kv_norm = kv_norm_data.as_ref().map(|buf| unsafe {
                std::slice::from_raw_parts(buf.as_ptr() as *const f16, buf.len() / 2)
            });

            let weights = MlaWeights {
                q_a_proj,
                q_b_proj,
                kv_a_proj,
                kv_b_proj,
                o_proj,
                q_norm,
                kv_norm,
            };

            let layer = layer_idx as usize;
            let mla_kv = self.mla_kv_cache.as_mut().unwrap();
            if layer >= mla_kv.layers.len() {
                return Ok(Some(vec![f16::from_f32(0.0); hidden_dim]));
            }

            let result = mla_attention_layer(
                cpu_hidden_state,
                &weights,
                &mut mla_kv.layers[layer],
                self.position,
                &self.model_config,
                &mla,
            );

            Ok(Some(result))
        }
    }

    /// Run the MoE/FFN sublayer for one layer with pre-norm + residual.
    ///
    /// ```text
    /// residual = hidden_state
    /// normed = RMSNorm(hidden_state, ffn_norm_weights)
    /// moe_output = sum(expert_weight * Expert_i(normed)) + SharedExpert(normed)
    /// hidden_state = residual + moe_output
    /// ```
    async fn run_moe_sublayer(&mut self, layer_idx: u16) -> Result<()> {
        let hidden_dim = self.model_config.hidden_dim as usize;
        let hidden_bytes = hidden_dim * std::mem::size_of::<f16>();
        let eps = self.model_config.rms_norm_eps;

        // Apply pre-MoE RMSNorm from FP32 accumulator
        // FP32 path: hidden_state_f32 → rms_norm_f32 → moe_normed_f32 (for router + experts)
        // FP16 path: hidden_state_f32 → rms_norm_f32_to_f16 → hidden_state (for diagnostics)
        self.ensure_shared_tensor_device(layer_idx, 7).await;
        if let Some((norm_ptr, norm_size)) = self.get_device_tensor(layer_idx, 7) {
            tracing::trace!(
                "MoE RMSNorm: layer={}, norm_ptr={:?}, norm_size={}, hidden_dim={}, hidden_bytes={}",
                layer_idx, norm_ptr, norm_size, hidden_dim, hidden_bytes,
            );
            // FP32 RMSNorm: reads FP32 accumulator, writes FP32 normalized output
            kernels::rms_norm_f32(
                self.hidden_state_f32.as_ptr(),
                self.moe_normed_f32.as_mut_ptr(),
                norm_ptr,
                hidden_dim,
                eps,
                &self.stream,
            )?;
            // FP32→FP16 RMSNorm: only needed for diagnostics, skip in fast path
            if self.diag_enabled {
                kernels::rms_norm_f32_to_f16(
                    self.hidden_state_f32.as_ptr(),
                    self.hidden_state.as_mut_ptr(),
                    norm_ptr,
                    hidden_dim,
                    eps,
                    &self.stream,
                )?;
            }
        } else {
            kernels::rms_norm_no_weight(
                self.hidden_state.as_mut_ptr(),
                hidden_dim,
                eps,
                &self.stream,
            )?;
        }

        // ═══ CRITICAL DIAGNOSTIC: Check moe_normed_f32 after RMSNorm ═══
        // This isolates whether the MoE all-zeros bug is in the norm or the SwiGLU kernel.
        // Fires on decode_step 0 for first 3 layers + L31, to minimize overhead.
        // Gated behind VIB3_DIAG=1 to avoid overhead in normal inference.
        if self.diag_enabled && self.decode_step <= 1 && (layer_idx <= 2 || layer_idx == 31) {
            self.stream.synchronize()?;
            let f32_bytes = hidden_dim * std::mem::size_of::<f32>();
            let mut normed_buf = vec![0u8; f32_bytes];
            self.moe_normed_f32.copy_to_host(&mut normed_buf)?;
            let normed_f32 = unsafe { std::slice::from_raw_parts(normed_buf.as_ptr() as *const f32, hidden_dim) };
            let l2 = normed_f32.iter().map(|v| v * v).sum::<f32>().sqrt();
            let max_abs = normed_f32.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
            let zeros = normed_f32.iter().filter(|v| **v == 0.0).count();
            let nans = normed_f32.iter().filter(|v| v.is_nan()).count();
            let first4: Vec<f32> = normed_f32[..4.min(hidden_dim)].iter().copied().collect();
            tracing::warn!(
                "MOE_NORMED_F32 DIAG L{} pos={} step={}: L2={:.6}, max_abs={:.6}, zeros={}/{}, nans={}, first4={:?}",
                layer_idx, self.position, self.decode_step, l2, max_abs, zeros, hidden_dim, nans, first4,
            );

            // Also check hidden_state_f32 (the input to rms_norm_f32)
            let mut input_buf = vec![0u8; f32_bytes];
            self.hidden_state_f32.copy_to_host(&mut input_buf)?;
            let input_f32 = unsafe { std::slice::from_raw_parts(input_buf.as_ptr() as *const f32, hidden_dim) };
            let in_l2 = input_f32.iter().map(|v| v * v).sum::<f32>().sqrt();
            let in_max = input_f32.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
            let in_zeros = input_f32.iter().filter(|v| **v == 0.0).count();
            tracing::warn!(
                "HIDDEN_STATE_F32 DIAG L{} pos={} step={}: L2={:.6}, max_abs={:.6}, zeros={}/{}",
                layer_idx, self.position, self.decode_step, in_l2, in_max, in_zeros, hidden_dim,
            );
        }
        // ═══ END CRITICAL DIAGNOSTIC ═══

         // Diagnostic: log pre-norm and post-norm magnitudes for last prefill token
        if self.diag_enabled && self.position > 0 && (layer_idx == 1 || layer_idx == 5 || layer_idx == 6 || layer_idx == 10 || layer_idx == 30) {
            self.stream.synchronize()?;
            let mut h_buf = vec![0u8; hidden_bytes];
            self.hidden_state.copy_to_host(&mut h_buf)?;
            let h_f16 = unsafe { std::slice::from_raw_parts(h_buf.as_ptr() as *const f16, hidden_dim) };
            let h_l2 = h_f16.iter().map(|v| { let f = v.to_f32(); f * f }).sum::<f32>().sqrt();
            let mut r_buf = vec![0u8; hidden_bytes];
            self.residual_buf.copy_to_host(&mut r_buf)?;
            let r_f16 = unsafe { std::slice::from_raw_parts(r_buf.as_ptr() as *const f16, hidden_dim) };
            let r_l2 = r_f16.iter().map(|v| { let f = v.to_f32(); f * f }).sum::<f32>().sqrt();
            tracing::info!(
                "MOE NORM DIAG L{} pos={}: pre_norm(residual) L2={:.2}, post_norm L2={:.2}",
                layer_idx, self.position, r_l2, h_l2,
            );

            // Dump hidden states for layer 6 to /model/dump/ for Python analysis
            if layer_idx == 6 {
                let dump_dir = "/model/dump";
                let _ = std::fs::create_dir_all(dump_dir);
                // Dump post-norm
                let dump_path = format!("{}/l6_postnorm_pos{}.bin", dump_dir, self.position);
                if let Ok(()) = std::fs::write(&dump_path, &h_buf) {
                    tracing::info!("DUMPED post-norm hidden state to {} ({} bytes)", dump_path, h_buf.len());
                    let first20: Vec<f32> = h_f16[..20].iter().map(|v| v.to_f32()).collect();
                    tracing::info!("  first20: {:?}", first20);
                    let max_val = h_f16.iter().map(|v| v.to_f32()).fold(f32::NEG_INFINITY, f32::max);
                    let min_val = h_f16.iter().map(|v| v.to_f32()).fold(f32::INFINITY, f32::min);
                    tracing::info!("  min={:.6}, max={:.6}", min_val, max_val);
                }
                // Dump pre-norm (residual)
                let dump_path2 = format!("{}/l6_prenorm_pos{}.bin", dump_dir, self.position);
                let _ = std::fs::write(&dump_path2, &r_buf);
                tracing::info!("DUMPED pre-norm hidden state to {}", dump_path2);
            }
        }

        // Route and execute experts
        let activation = self.run_router_for_layer(layer_idx).await?;

        // Dump MoE intermediates for ALL tokens at layer 0
        if self.diag_enabled && layer_idx == 0 {
            self.stream.synchronize()?;
            let dump_dir = "/home/brian/code/vib3/dump";
            let tok = self.position;
            let mut buf = vec![0u8; hidden_dim * 4];
            // MoE normed input
            self.moe_normed_f32.copy_to_host(&mut buf)?;
            let _ = std::fs::write(format!("{}/vib3_moe_normed_L0_tok{}.bin", dump_dir, tok), &buf);
            // Router expert IDs and weights
            let expert_ids: Vec<u16> = activation.experts.iter().map(|(id, _)| *id).collect();
            let expert_wts: Vec<f32> = activation.experts.iter().map(|(_, w)| *w).collect();
            let id_bytes: Vec<u8> = expert_ids.iter().flat_map(|id| (*id as i32).to_le_bytes()).collect();
            let wt_bytes: Vec<u8> = expert_wts.iter().flat_map(|w| w.to_le_bytes()).collect();
            let _ = std::fs::write(format!("{}/vib3_router_ids_L0_tok{}.bin", dump_dir, tok), &id_bytes);
            let _ = std::fs::write(format!("{}/vib3_router_wts_L0_tok{}.bin", dump_dir, tok), &wt_bytes);
        }

        let plan = self.planner.plan_layer(layer_idx, &activation).await?;

        // Diagnostic: log expert plan details for first decode step, first MoE layer
        // Also log for layers 1-10 at pos=0 to investigate explosion
        // Also log L0-L2 at last prefill token (pos=4) for debugging
        if self.diag_enabled && ((self.decode_step == 1 && layer_idx == 1)
            || (self.position == 0 && layer_idx <= 10)
            || (layer_idx <= 2 && self.position == 4)
            || (layer_idx == 5 || layer_idx == 6)) {
            tracing::info!(
                "EXPERT PLAN DIAG L{} pos={}: {} experts activated, router=({:?})",
                layer_idx, self.position, plan.experts.len(),
                activation.experts.iter().map(|(id, w)| format!("e{}={:.6}", id, w)).collect::<Vec<_>>().join(", "),
            );
            for (i, ep) in plan.experts.iter().enumerate() {
                let up_pages = ep.pages.iter().filter(|p| p.id.segment == 0).count();
                let gate_pages = ep.pages.iter().filter(|p| p.id.segment == 1).count();
                let down_pages = ep.pages.iter().filter(|p| p.id.segment == 2).count();
                tracing::info!(
                    "  Expert[{}]: id={}, weight={:.6}, pages: up={}, gate={}, down={}, total={}",
                    i, ep.expert_id, ep.weight, up_pages, gate_pages, down_pages, ep.pages.len(),
                );
            }
        }

        // Zero the layer output accumulator on device
        self.layer_output_buf.zero();

        // Pre-quantize the MoE normalized hidden state for MMA: done once,
        // reused by all 8 expert SwiGLU kernels (eliminates ~2304 warp amax
        // reductions per layer from the MMA inner loop).
        if self.model_config.expert_dtype == DType::NVFP4 {
            kernels::quantize_activation_fp4_fast(
                self.moe_normed_f32.as_ptr(),
                self.preq_act_fp4.as_mut_ptr(),
                self.preq_act_scales.as_mut_ptr(),
                hidden_dim,
                &self.stream,
            )?;
        }

        // Execute routed experts
        for (_i, expert_plan) in plan.experts.iter().enumerate() {
            self.execute_expert(expert_plan, hidden_dim).await?;

            // Diagnostic: after each expert at L0 or L6 for last prefill token, track per-expert accumulation
            if self.diag_enabled && (layer_idx <= 2 && self.position == 4) {
                self.stream.synchronize()?;
                let mut out_buf = vec![0u8; hidden_dim * 4]; // FP32
                self.layer_output_buf.copy_to_host(&mut out_buf)?;
                let out_f32 = unsafe { std::slice::from_raw_parts(out_buf.as_ptr() as *const f32, hidden_dim) };
                let l2 = out_f32.iter().map(|v| v * v).sum::<f32>().sqrt();
                let max_abs = out_f32.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
                tracing::info!(
                    "L{} EXPERT[{}] ACCUM pos={}: e{}*{:.6} -> accum L2={:.6}, max_abs={:.6}, first4=[{:.6},{:.6},{:.6},{:.6}]",
                    layer_idx, _i, self.position, expert_plan.expert_id, expert_plan.weight, l2, max_abs,
                    out_f32[0], out_f32[1], out_f32[2], out_f32[3],
                );
            }
            if self.diag_enabled && layer_idx == 6 && self.position == 26 {
                self.stream.synchronize()?;
                let mut out_buf = vec![0u8; hidden_dim * 4]; // FP32
                self.layer_output_buf.copy_to_host(&mut out_buf)?;
                let out_f32 = unsafe { std::slice::from_raw_parts(out_buf.as_ptr() as *const f32, hidden_dim) };
                let l2 = out_f32.iter().map(|v| v * v).sum::<f32>().sqrt();
                let max_abs = out_f32.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
                tracing::info!(
                    "L6 EXPERT[{}] ACCUM: e{}*{:.4} -> accum L2={:.4}, max_abs={:.4}",
                    _i, expert_plan.expert_id, expert_plan.weight, l2, max_abs,
                );
            }
        }

        // ═══ DIAGNOSTIC: dump MoE output BEFORE shared expert ═══
        if self.diag_enabled && layer_idx == 0 {
            self.stream.synchronize()?;
            let dump_dir = "/home/brian/code/vib3/dump";
            let tok = self.position;
            let mut out_buf = vec![0u8; hidden_dim * 4];
            self.layer_output_buf.copy_to_host(&mut out_buf)?;
            let _ = std::fs::write(
                format!("{}/vib3_moe_routed_L0_tok{}.bin", dump_dir, tok),
                &out_buf,
            );
        }

        // Run shared expert (unconditional, every token) if model has one
        if self.model_config.num_shared_experts > 0 {
            self.execute_shared_expert(layer_idx, hidden_dim).await?;
        }

        // ═══ DIAGNOSTIC: dump MoE output AFTER shared expert (full MoE output before residual) ═══
        if self.diag_enabled && layer_idx <= 2 && self.position == 4 {
            self.stream.synchronize()?;
            let mut out_buf = vec![0u8; hidden_dim * 4];
            self.layer_output_buf.copy_to_host(&mut out_buf)?;
            let out_f32 = unsafe { std::slice::from_raw_parts(out_buf.as_ptr() as *const f32, hidden_dim) };
            let l2 = out_f32.iter().map(|v| v * v).sum::<f32>().sqrt();
            tracing::warn!(
                "MOE_FULL L{} pos={}: L2={:.6}, first4=[{:.6},{:.6},{:.6},{:.6}]",
                layer_idx, self.position, l2, out_f32[0], out_f32[1], out_f32[2], out_f32[3],
            );
            let dump_dir = "/home/brian/code/vib3/dump";
            let _ = std::fs::write(
                format!("{}/vib3_moe_full_f32_L{}_pos4.bin", dump_dir, layer_idx),
                &out_buf,
            );
            // Also dump the moe_normed_f32 (input to experts)
            let mut normed_buf = vec![0u8; hidden_dim * 4];
            self.moe_normed_f32.copy_to_host(&mut normed_buf)?;
            let _ = std::fs::write(
                format!("{}/vib3_moe_normed_f32_L{}_pos4.bin", dump_dir, layer_idx),
                &normed_buf,
            );
        }

        // Diagnostic: log layer output magnitude for decode step 1 or prefill layers 1-10 (now FP32)
        if self.diag_enabled && ((self.decode_step == 1 && (layer_idx == 1 || layer_idx == 30 || layer_idx == 60))
            || (self.position == 0 && layer_idx <= 10)
            || (layer_idx == 5 || layer_idx == 6)) {
            self.stream.synchronize()?;
            let mut out_buf = vec![0u8; hidden_dim * 4]; // FP32
            self.layer_output_buf.copy_to_host(&mut out_buf)?;
            let out_f32 = unsafe { std::slice::from_raw_parts(out_buf.as_ptr() as *const f32, hidden_dim) };
            let l2 = out_f32.iter().map(|v| v * v).sum::<f32>().sqrt();
            let max_abs = out_f32.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
            let nans = out_f32.iter().filter(|v| v.is_nan()).count();
            let zeros = out_f32.iter().filter(|v| **v == 0.0).count();
            tracing::info!(
                "MOE OUTPUT DIAG decode L{}: L2={:.4}, max_abs={:.4}, nan={}, zero={}/{}",
                layer_idx, l2, max_abs, nans, zeros, hidden_dim,
            );
        }

        // Dump full MoE output for ALL tokens at layer 0
        if self.diag_enabled && layer_idx == 0 {
            self.stream.synchronize()?;
            let dump_dir = "/home/brian/code/vib3/dump";
            let tok = self.position;
            let mut buf = vec![0u8; hidden_dim * 4];
            self.layer_output_buf.copy_to_host(&mut buf)?;
            let _ = std::fs::write(format!("{}/vib3_moe_out_L0_tok{}.bin", dump_dir, tok), &buf);
        }

        // FP32 residual accumulation: hidden_state_f32 += layer_output_f32
        // Both buffers are now FP32 — no FP16 truncation in the MoE output path
        kernels::residual_add_f32_f32(
            self.hidden_state_f32.as_mut_ptr(),
            self.layer_output_buf.as_ptr(),
            hidden_dim,
            &self.stream,
        )?;

        Ok(())
    }

    /// Run the dense FFN sublayer for a dense (non-MoE) layer with pre-norm + residual.
    ///
    /// Used for layers 0..dense_layer_idx (layer 0 in Kimi K2.5).
    /// Dense FFN is a standard SwiGLU: output = down_proj(SiLU(gate_proj(x)) * up_proj(x))
    ///
    /// Weight segments:
    /// - 17: up_proj   [intermediate_size, hidden_dim]
    /// - 18: gate_proj  [intermediate_size, hidden_dim]
    /// - 19: down_proj  [hidden_dim, intermediate_size]
    /// - 7:  pre-FFN RMSNorm weight [hidden_dim]
    async fn run_dense_ffn_sublayer(&mut self, layer_idx: u16) -> Result<()> {
        let hidden_dim = self.model_config.hidden_dim as usize;
        let intermediate_size = self.model_config.dense_intermediate_size as usize;
        let eps = self.model_config.rms_norm_eps;

        if intermediate_size == 0 {
            tracing::warn!("Dense FFN: intermediate_size=0, skipping layer {}", layer_idx);
            return Ok(());
        }

        // Apply pre-FFN RMSNorm: FP32 → normalize in FP32 → FP16 output
        self.ensure_shared_tensor_device(layer_idx, 7).await;
        if let Some((norm_ptr, _)) = self.get_device_tensor(layer_idx, 7) {
            kernels::rms_norm_f32_to_f16(
                self.hidden_state_f32.as_ptr(),
                self.hidden_state.as_mut_ptr(),
                norm_ptr,
                hidden_dim,
                eps,
                &self.stream,
            )?;
        } else {
            // Fallback: cast then normalize without weight
            kernels::f32_to_f16(
                self.hidden_state_f32.as_ptr(),
                self.hidden_state.as_mut_ptr(),
                hidden_dim,
                &self.stream,
            )?;
            kernels::rms_norm_no_weight(
                self.hidden_state.as_mut_ptr(),
                hidden_dim,
                eps,
                &self.stream,
            )?;
        }

        // Load dense FFN weights to VRAM for GPU computation
        self.ensure_shared_tensor_device(layer_idx, 17).await; // up_proj
        self.ensure_shared_tensor_device(layer_idx, 18).await; // gate_proj
        self.ensure_shared_tensor_device(layer_idx, 19).await; // down_proj

        let up_device = self.get_device_tensor(layer_idx, 17);
        let gate_device = self.get_device_tensor(layer_idx, 18);
        let down_device = self.get_device_tensor(layer_idx, 19);

        if up_device.is_none() || gate_device.is_none() || down_device.is_none() {
            tracing::warn!(
                "Dense FFN L{}: weights not available on device, skipping",
                layer_idx,
            );
            // FP32 accumulator unchanged — no FFN output to add
            return Ok(());
        }

        let (up_ptr, _) = up_device.unwrap();
        let (gate_ptr, _) = gate_device.unwrap();
        let (down_ptr, _) = down_device.unwrap();

        // Fused SwiGLU on GPU: dense_ffn_gate_dev = SiLU(hidden × gate^T) * (hidden × up^T)
        // Dense FFN weights are FP16, so use the fused kernel (no temp buffers needed).
        kernels::partial_swiglu(
            self.hidden_state.as_ptr(),
            up_ptr,
            gate_ptr,
            self.dense_ffn_gate_dev.as_mut_ptr(), // reuse as SwiGLU output [intermediate_size]
            hidden_dim,
            intermediate_size,
            DType::FP16,
            &self.stream,
            None, // FP16 uses fused kernel, no temps needed
        )?;

        // Down projection on GPU: dense_ffn_down_dev = swiglu_output × down^T
        kernels::partial_matmul(
            self.dense_ffn_gate_dev.as_ptr(), // SwiGLU output [intermediate_size]
            down_ptr,
            self.dense_ffn_down_dev.as_mut_ptr(), // [hidden_dim]
            intermediate_size,
            hidden_dim,
            DType::FP16,
            &self.stream,
        )?;

        // FP32 residual accumulation: hidden_state_f32 += f32(FFN output)
        kernels::residual_add_fp32(
            self.hidden_state_f32.as_mut_ptr(),
            self.dense_ffn_down_dev.as_ptr(),
            hidden_dim,
            &self.stream,
        )?;

        tracing::trace!("Dense FFN L{}: GPU SwiGLU, intermediate_size={}", layer_idx, intermediate_size);

        Ok(())
    }

    /// Generate one token (autoregressive decode step).
    ///
    /// ## Flow
    ///
    /// **Step 0** (first token after prefill):
    ///   - hidden_state has been through all layers in prefill
    ///   - Compute logits directly, sample first output token
    ///   - Embed sampled token → hidden_state for next step
    ///
    /// **Step 1+** (subsequent tokens):
    ///   - hidden_state = embedding of previously sampled token
    ///   - Run through all transformer layers (attention + MoE)
    ///   - Compute logits, sample token
    ///   - Embed sampled token → hidden_state for next step
    async fn generate_token(
        &mut self,
        step: usize,
        params: &SamplingParams,
        recent_tokens: &[u32],
    ) -> Result<u32> {
        let compute_start = Instant::now();
        let num_layers = self.model_config.num_layers;
        let dense_layer_idx = self.model_config.dense_layer_idx;
        let num_moe_layers = self.model_config.num_moe_layers;

        // Clear per-token expert accumulator for mode detection
        self.token_expert_ids.clear();

        // Skip layer pass for step 0 — prefill already processed all layers.
        // For step 1+, the hidden_state contains the embedding of the previously
        // sampled token, which needs to go through all layers.
        if step > 0 {
            // Update device-side position scalar (for CUDA graph compatibility).
            // All position-dependent kernels (RoPE, KV append, decode attention)
            // read position from this device pointer instead of kernel arguments.
            #[cfg(feature = "cuda")]
            {
                let stream_ptr = self.stream.raw_ptr();
                let err = unsafe {
                    cuda_ffi::vib3_update_device_int32(
                        self.d_position.as_mut_ptr(),
                        self.position as i32,
                        stream_ptr,
                    )
                };
                if err != 0 {
                    tracing::warn!("Failed to update device position (err={})", err);
                }
            }

            // ── CUDA Graph fast path: replay cached graph for layer loop ──
            // Disabled by default due decode correctness regressions observed on
            // Qwen3.5 (token collapse under graph replay). Enable explicitly via:
            //   VIB3_CUDA_GRAPH=1
            let graph_enabled = cfg!(feature = "cuda")
                && std::env::var("VIB3_CUDA_GRAPH").map_or(false, |v| v == "1");
            let use_graph = graph_enabled && self.cuda_graph_exec.is_some();

            if use_graph {
                // Replay the cached CUDA graph (all 48 layers in one launch)
                #[cfg(feature = "cuda")]
                {
                    let exec = self.cuda_graph_exec.unwrap();
                    let stream_ptr = self.stream.raw_ptr();
                    let err = unsafe {
                        cuda_ffi::vib3_cuda_graph_launch(exec, stream_ptr)
                    };
                    if err != 0 {
                        tracing::warn!("CUDA graph launch failed (err={}), falling back to normal execution", err);
                        // Fall through to normal execution below
                    } else {
                        // Graph launched successfully — skip the normal layer loop
                        tracing::info!(
                            "Token step={}: CUDA graph replay, total_layers=<graph>",
                            step,
                        );
                    }
                }
            }

            if !use_graph {
            // ── Normal layer loop (also used for graph capture) ──
            let capturing = graph_enabled
                && step == 1
                && !self.diag_enabled
                && self.cuda_graph_exec.is_none()
                && self.moe_page_table_dev.is_some();

            #[cfg(feature = "cuda")]
            if capturing {
                let stream_ptr = self.stream.raw_ptr();
                let err = unsafe {
                    cuda_ffi::vib3_cuda_graph_begin_capture(stream_ptr)
                };
                if err != 0 {
                    tracing::warn!("CUDA graph capture begin failed (err={}), running without graph", err);
                } else {
                    self.capturing_graph = true;
                }
            }

            // Run through all layers: attention + MoE interleaved per layer
            let mut attn_us: u64 = 0;
            let mut moe_us: u64 = 0;
            for layer_idx in 0..num_layers as u16 {
                // ── Attention sublayer (with pre-norm + residual) ──
                let t0 = Instant::now();
                self.run_attention_layer(layer_idx).await?;
                attn_us += t0.elapsed().as_micros() as u64;

                // ── MoE/FFN sublayer (with pre-norm + residual) ──
                let is_moe_layer = layer_idx >= dense_layer_idx as u16
                    && (layer_idx - dense_layer_idx as u16) < num_moe_layers as u16;
                let is_dense_ffn_layer = (layer_idx as u32) < dense_layer_idx;

                let t1 = Instant::now();
                if is_dense_ffn_layer {
                    self.run_dense_ffn_sublayer(layer_idx).await?;
                } else if is_moe_layer {
                    // Vector index pre-warm: predict which pages this layer will need
                    let t_prewarm = Instant::now();
                    self.planner.predict_and_prewarm(layer_idx);
                    let prewarm_us = t_prewarm.elapsed().as_micros() as u64;

                    // MoE sublayer with pre-norm + residual
                    // We inline this (instead of calling run_moe_sublayer) to also
                    // collect expert IDs for mode detection and submit lookahead.
                    let hidden_dim = self.model_config.hidden_dim as usize;
                    let hidden_bytes = hidden_dim * std::mem::size_of::<f16>();
                    let eps = self.model_config.rms_norm_eps;

                    // Apply pre-MoE RMSNorm from FP32 accumulator
                    // FP32 path: hidden_state_f32 → rms_norm_f32 → moe_normed_f32 (for router + experts)
                    // FP16 path: hidden_state_f32 → rms_norm_f32_to_f16 → hidden_state (for diagnostics)
                    let t_norm = Instant::now();
                    self.ensure_shared_tensor_device(layer_idx, 7).await;
                    if let Some((norm_ptr, _)) = self.get_device_tensor(layer_idx, 7) {
                        // FP32 RMSNorm: reads FP32 accumulator, writes FP32 normalized output
                        kernels::rms_norm_f32(
                            self.hidden_state_f32.as_ptr(),
                            self.moe_normed_f32.as_mut_ptr(),
                            norm_ptr,
                            hidden_dim,
                            eps,
                            &self.stream,
                        )?;
                        // FP16 norm only needed for diagnostics — the MoE path reads
                        // moe_normed_f32 (FP32), and the next attention layer will
                        // recompute hidden_state (FP16) from hidden_state_f32.
                        if self.diag_enabled {
                            kernels::rms_norm_f32_to_f16(
                                self.hidden_state_f32.as_ptr(),
                                self.hidden_state.as_mut_ptr(),
                                norm_ptr,
                                hidden_dim,
                                eps,
                                &self.stream,
                            )?;
                        }
                    } else {
                        // Fallback: FP32→FP16 then in-place FP16 norm (no gamma weights)
                        kernels::f32_to_f16(
                            self.hidden_state_f32.as_ptr(),
                            self.hidden_state.as_mut_ptr(),
                            hidden_dim,
                            &self.stream,
                        )?;
                        kernels::rms_norm_no_weight(
                            self.hidden_state.as_mut_ptr(),
                            hidden_dim,
                            eps,
                            &self.stream,
                        )?;
                    }
                    let norm_us = t_norm.elapsed().as_micros() as u64;

                    // ── GPU-only MoE fast path: zero host sync ──
                    // When the device-side page table is available, run router + experts
                    // entirely on the GPU compute stream with no host synchronization.
                    let (router_us, routed_us) = if self.moe_page_table_dev.is_some() {
                        let tr = Instant::now();
                        let top_k = self.model_config.num_active_experts as usize;
                        let num_experts = self.model_config.num_experts as usize;

                        // Determine scoring function
                        let scoring_func = if self.model_config.scoring_func == "sigmoid" {
                            kernels::RouterScoringFunc::Sigmoid {
                                scaling_factor: self.model_config.routed_scaling_factor,
                                normalize: self.model_config.norm_topk_prob,
                            }
                        } else {
                            kernels::RouterScoringFunc::Softmax
                        };

                        // Load router weights
                        self.ensure_shared_tensor_device(layer_idx, 3).await;
                        if let Some((router_ptr, _)) = self.get_device_tensor(layer_idx, 3) {
                            // Launch router on compute stream — no sync, no D2H
                            kernels::run_router_gpu_topk_nosync(
                                self.moe_normed_f32.as_ptr(),
                                router_ptr,
                                num_experts,
                                hidden_dim,
                                top_k,
                                scoring_func,
                                self.router_scores_dev.as_mut_ptr() as *mut f32,
                                self.router_topk_ids_dev.as_mut_ptr(),
                                self.router_topk_weights_dev.as_mut_ptr(),
                                &self.stream,
                            )?;
                        }
                        let router_t = tr.elapsed().as_micros() as u64;

                        // Zero layer output + pre-quantize activation (same as old path)
                        self.layer_output_buf.zero_async(&self.stream);
                        kernels::quantize_activation_fp4_fast(
                            self.moe_normed_f32.as_ptr(),
                            self.preq_act_fp4.as_mut_ptr(),
                            self.preq_act_scales.as_mut_ptr(),
                            hidden_dim,
                            &self.stream,
                        )?;

                        // Launch fused MoE: resolve + up + gate + swiglu + down
                        let te = Instant::now();
                        let expert_hidden_dim = self.model_config.expert_hidden_dim as usize;
                        let moe_layer = (layer_idx - dense_layer_idx as u16) as usize;
                        let page_table_ptr = self.moe_page_table_dev.as_ref().unwrap().as_ptr();
                        kernels::moe_experts_fused_gpu(
                            self.preq_act_fp4.as_ptr(),
                            self.preq_act_scales.as_ptr(),
                            page_table_ptr,
                            self.router_topk_ids_dev.as_ptr(),
                            self.router_topk_weights_dev.as_ptr(),
                            moe_layer,
                            num_experts,
                            top_k,
                            hidden_dim,
                            expert_hidden_dim,
                            hidden_dim,
                            self.layer_output_buf.as_mut_ptr() as *mut f32,
                            &self.stream,
                        )?;
                        let routed_t = te.elapsed().as_micros() as u64;
                        (router_t, routed_t)
                    } else {
                        // ── Original host-sync MoE path (fallback) ──
                        let tr = Instant::now();
                        let activation = self.run_router_for_layer(layer_idx).await?;
                        let router_t = tr.elapsed().as_micros() as u64;

                        // Collect expert IDs for mode detection
                        for &(expert_id, _weight) in &activation.experts {
                            self.token_expert_ids.push(expert_id);
                        }

                        // Submit lookahead for next MoE layer
                        let moe_offset = layer_idx - dense_layer_idx as u16;
                        if moe_offset + 1 < num_moe_layers as u16 {
                            self.planner.submit_lookahead(layer_idx + 1, &activation);
                            self.planner.submit_cross_layer_prefetch(layer_idx);
                        }

                        // Plan and execute expert computation
        let plan = self.planner.plan_layer(layer_idx, &activation).await?;

                        self.layer_output_buf.zero_async(&self.stream);

                        if self.model_config.expert_dtype == DType::NVFP4 {
                            kernels::quantize_activation_fp4_fast(
                                self.moe_normed_f32.as_ptr(),
                                self.preq_act_fp4.as_mut_ptr(),
                                self.preq_act_scales.as_mut_ptr(),
                                hidden_dim,
                                &self.stream,
                            )?;
                        }

                        let te = Instant::now();
                        let used_fused = if self.model_config.expert_dtype == DType::NVFP4
                            && self.stream.is_real()
                        {
                            let expert_hidden_dim = self.model_config.expert_hidden_dim as usize;
                            let mut fused_experts: Vec<(*const u8, *const u8, *const u8, f32)> = Vec::new();
                            let mut all_single_page = true;
                            for ep in &plan.experts {
                                let mut up_ptr: *const u8 = std::ptr::null();
                                let mut gate_ptr: *const u8 = std::ptr::null();
                                let mut down_ptr: *const u8 = std::ptr::null();
                                let mut up_count = 0;
                                let mut gate_count = 0;
                                let mut down_count = 0;
                                for rp in &ep.pages {
                                    match rp.id.segment {
                                        0 => { up_ptr = rp.device_ptr as *const u8; up_count += 1; }
                                        1 => { gate_ptr = rp.device_ptr as *const u8; gate_count += 1; }
                                        2 => { down_ptr = rp.device_ptr as *const u8; down_count += 1; }
                                        _ => {}
                                    }
                                }
                                if up_count != 1 || gate_count != 1 || down_count != 1
                                    || up_ptr.is_null() || gate_ptr.is_null() || down_ptr.is_null()
                                {
                                    all_single_page = false;
                                    break;
                                }
                                fused_experts.push((up_ptr, gate_ptr, down_ptr, ep.weight));
                            }
                            if all_single_page && !fused_experts.is_empty() {
                                kernels::moe_experts_fused(
                                    self.preq_act_fp4.as_ptr(),
                                    self.preq_act_scales.as_ptr(),
                                    &fused_experts,
                                    hidden_dim,
                                    expert_hidden_dim,
                                    hidden_dim,
                                    self.layer_output_buf.as_mut_ptr() as *mut f32,
                                    &self.stream,
                                )?;
                                true
                            } else {
                                false
                            }
                        } else {
                            false
                        };
                        if !used_fused {
                            for expert_plan in &plan.experts {
                                self.execute_expert(expert_plan, hidden_dim).await?;
                            }
                        }
                        let routed_t = te.elapsed().as_micros() as u64;
                        (router_t, routed_t)
                    };
                    let lookahead_us: u64 = 0;
                    let plan_us: u64 = 0;

                    // Run shared expert (unconditional, every token)
                    let ts = Instant::now();
                    if self.model_config.num_shared_experts > 0 {
                        self.execute_shared_expert(layer_idx, hidden_dim).await?;
                    }
                    let shared_us = ts.elapsed().as_micros() as u64;

                    // Dump MoE intermediates for layer 0
                    if self.diag_enabled && layer_idx == 0 {
                        self.stream.synchronize()?;
                        let dump_dir = "/home/brian/code/vib3/dump";
                        let tok = self.position;
                        // MoE normed input
                        let mut buf = vec![0u8; hidden_dim * 4];
                        self.moe_normed_f32.copy_to_host(&mut buf)?;
                        let _ = std::fs::write(format!("{}/vib3_moe_normed_L0_tok{}.bin", dump_dir, tok), &buf);
                        // MoE output (layer_output_buf, before residual add)
                        self.layer_output_buf.copy_to_host(&mut buf)?;
                        let _ = std::fs::write(format!("{}/vib3_moe_out_L0_tok{}.bin", dump_dir, tok), &buf);
                        // Router top-k ids and weights
                        let top_k = self.model_config.num_active_experts as usize;
                        let mut id_buf = vec![0u8; top_k * 4]; // i32
                        self.router_topk_ids_dev.copy_to_host(&mut id_buf)?;
                        let _ = std::fs::write(format!("{}/vib3_router_ids_L0_tok{}.bin", dump_dir, tok), &id_buf);
                        let mut wt_buf = vec![0u8; top_k * 4]; // f32
                        self.router_topk_weights_dev.copy_to_host(&mut wt_buf)?;
                        let _ = std::fs::write(format!("{}/vib3_router_wts_L0_tok{}.bin", dump_dir, tok), &wt_buf);
                    }

                    // FP32 residual accumulation: hidden_state_f32 += layer_output_f32
                    let t_res = Instant::now();
                    kernels::residual_add_f32_f32(
                        self.hidden_state_f32.as_mut_ptr(),
                        self.layer_output_buf.as_ptr(),
                        hidden_dim,
                        &self.stream,
                    )?;
                    let resid_us = t_res.elapsed().as_micros() as u64;

                    // Diagnostic: log MoE output magnitude for steps 1-3 (now FP32)
                    if self.diag_enabled && ((step <= 3 && (layer_idx <= 2 || layer_idx == 5 || layer_idx == 6 || layer_idx == 10 || layer_idx == 30 || layer_idx >= num_layers as u16 - 2))
                        || (step == 3 && layer_idx >= 11 && layer_idx <= 29)) {
                        self.stream.synchronize()?;
                        let mut lo_buf = vec![0u8; hidden_dim * 4]; // FP32
                        self.layer_output_buf.copy_to_host(&mut lo_buf)?;
                        let lo_f32 = unsafe {
                            std::slice::from_raw_parts(lo_buf.as_ptr() as *const f32, hidden_dim)
                        };
                        let lo_l2 = lo_f32.iter().map(|v| v * v).sum::<f32>().sqrt();
                        let lo_max = lo_f32.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
                        let lo_nan = lo_f32.iter().filter(|v| v.is_nan()).count();
                        let mut hid_buf = vec![0u8; hidden_bytes];
                        self.hidden_state.copy_to_host(&mut hid_buf)?;
                        let hid_f16 = unsafe {
                            std::slice::from_raw_parts(hid_buf.as_ptr() as *const f16, hidden_dim)
                        };
                        let hid_l2 = hid_f16.iter().map(|v| { let f = v.to_f32(); f * f }).sum::<f32>().sqrt();
                        let hid_nan = hid_f16.iter().filter(|v| v.to_f32().is_nan()).count();
                        // Also read FP32 accumulator for accurate residual state
                        let mut f32_hid_buf = vec![0u8; hidden_dim * 4];
                        self.hidden_state_f32.copy_to_host(&mut f32_hid_buf)?;
                        let f32_hid = unsafe {
                            std::slice::from_raw_parts(f32_hid_buf.as_ptr() as *const f32, hidden_dim)
                        };
                        let f32_hid_l2 = f32_hid.iter().map(|v| v * v).sum::<f32>().sqrt();
                        let f32_hid_nan = f32_hid.iter().filter(|v| v.is_nan()).count();
                        tracing::info!(
                            "MOE DIAG decode L{} pos={}: moe_out L2={:.4}, max_abs={:.4}, nan={}, hidden_f16 L2={:.4}, nan={}, hidden_f32 L2={:.4}, nan={}",
                            layer_idx, self.position, lo_l2, lo_max, lo_nan, hid_l2, hid_nan, f32_hid_l2, f32_hid_nan,
                        );
                    }

                    // Log per-layer MoE breakdown for step 1 (first decode)
                    if step == 1 {
                        tracing::info!(
                            "MoE L{}: prewarm={:.1}ms norm={:.1}ms router={:.1}ms look={:.1}ms plan={:.1}ms routed={:.1}ms shared={:.1}ms resid={:.1}ms total={:.1}ms",
                            layer_idx,
                            prewarm_us as f64 / 1000.0,
                            norm_us as f64 / 1000.0,
                            router_us as f64 / 1000.0,
                            lookahead_us as f64 / 1000.0,
                            plan_us as f64 / 1000.0,
                            routed_us as f64 / 1000.0,
                            shared_us as f64 / 1000.0,
                            resid_us as f64 / 1000.0,
                            (prewarm_us + norm_us + router_us + lookahead_us + plan_us + routed_us + shared_us + resid_us) as f64 / 1000.0,
                        );
                    }
                }
                moe_us += t1.elapsed().as_micros() as u64;
            }
            tracing::info!(
                "Token step={}: attn={:.1}ms, moe={:.1}ms, total_layers={:.1}ms",
                step,
                attn_us as f64 / 1000.0,
                moe_us as f64 / 1000.0,
                (attn_us + moe_us) as f64 / 1000.0,
            );

            // ── End CUDA graph capture and instantiate ──
            #[cfg(feature = "cuda")]
            if capturing {
                self.capturing_graph = false;
                let stream_ptr = self.stream.raw_ptr();
                let mut graph: *mut std::ffi::c_void = std::ptr::null_mut();
                let err = unsafe {
                    cuda_ffi::vib3_cuda_graph_end_capture(stream_ptr, &mut graph)
                };
                if err != 0 || graph.is_null() {
                    tracing::warn!("CUDA graph capture end failed (err={}), running without graph", err);
                } else {
                    let mut exec: *mut std::ffi::c_void = std::ptr::null_mut();
                    let err2 = unsafe {
                        cuda_ffi::vib3_cuda_graph_instantiate(&mut exec, graph)
                    };
                    if err2 != 0 || exec.is_null() {
                        tracing::warn!("CUDA graph instantiate failed (err={})", err2);
                        unsafe { cuda_ffi::vib3_cuda_graph_destroy(graph); }
                    } else {
                        tracing::info!("CUDA graph captured and instantiated successfully");
                        // Launch the graph to actually execute this step's work
                        // (capture didn't execute the kernels, it only recorded them)
                        let err3 = unsafe {
                            cuda_ffi::vib3_cuda_graph_launch(exec, stream_ptr)
                        };
                        if err3 != 0 {
                            tracing::warn!("Initial CUDA graph launch failed (err={})", err3);
                            unsafe {
                                cuda_ffi::vib3_cuda_graph_exec_destroy(exec);
                                cuda_ffi::vib3_cuda_graph_destroy(graph);
                            }
                        } else {
                            self.cuda_graph_exec = Some(exec);
                            self.cuda_graph = Some(graph);
                        }
                    }
                }
            }

            } // end if !use_graph
        } // end if step > 0

        // ── Mode detection: record activations and check for transitions ──
        self.update_activation_mode().await;

        // Only advance position and KV state when layers actually ran.
        // Step 0 skips the layer pass (prefill already did it), so no KV was
        // written and position should not advance.
        if step > 0 {
            // Advance tiered KV cache position (if enabled)
            if let Some(ref mut tiered_kv) = self.tiered_kv {
                tiered_kv.advance_position();

                // Update KV cache stats
                self.stats
                    .kv_total_positions
                    .store(tiered_kv.seq_len() as u64, Ordering::Relaxed);
            }

            // ── Unified eviction check ──
            if let (Some(ref policy), Some(ref mut tiered_kv)) =
                (&self.eviction_policy, &mut self.tiered_kv)
            {
                let t1_status = self.buffer_mgr.tier_status(Tier::T1Vram);
                let kv_bytes = tiered_kv.t1_bytes_used();
                let rec = policy.evaluate(t1_status.used_pages, t1_status.total_pages, kv_bytes);

                if rec.evict_kv && rec.kv_positions_to_demote > 0 {
                    for layer in 0..self.model_config.num_layers as usize {
                        for head in 0..self.model_config.num_kv_heads as usize {
                            tiered_kv.maybe_demote_t2(layer, head);
                        }
                    }
                    tracing::debug!("Unified eviction: {}", rec.reason,);
                }
            }

            // Increment position (only when layers ran and KV was written)
            self.position += 1;
        }

        self.decode_step += 1;

        let compute_ns = compute_start.elapsed().as_nanos() as u64;
        self.stats
            .compute_ns
            .fetch_add(compute_ns, Ordering::Relaxed);

        // Diagnostic: check hidden state after all layers (first few tokens)
        // Gated behind diag_enabled to avoid pipeline stall from stream.synchronize()
        let hidden_dim = self.model_config.hidden_dim as usize;
        let hidden_bytes = hidden_dim * std::mem::size_of::<f16>();
        if self.diag_enabled && step < 5 {
            self.stream.synchronize()?;
            let f32_bytes = hidden_dim * 4;
            let mut diag_buf_f32 = vec![0u8; f32_bytes];
            self.hidden_state_f32.copy_to_host(&mut diag_buf_f32)?;
            let h_f32 = unsafe {
                std::slice::from_raw_parts(diag_buf_f32.as_ptr() as *const f32, hidden_dim)
            };
            let h_max = h_f32.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let h_min = h_f32.iter().cloned().fold(f32::INFINITY, f32::min);
            let h_mean = h_f32.iter().sum::<f32>() / hidden_dim as f32;
            let h_l2 = h_f32.iter().map(|v| v * v).sum::<f32>().sqrt();
            let h_nan = h_f32.iter().filter(|v| v.is_nan()).count();
            tracing::info!(
                "Hidden state step={} (after all layers, before final norm): min={:.4}, max={:.4}, mean={:.6}, L2={:.4}, nan={}, first4=[{:.6},{:.6},{:.6},{:.6}]",
                step, h_min, h_max, h_mean, h_l2, h_nan,
                h_f32[0], h_f32[1], h_f32[2], h_f32[3],
            );
            if step == 0 {
                let dump_path = "/home/brian/code/vib3/dump/vib3_final_hidden_f32_step0.bin";
                let _ = std::fs::write(dump_path, &diag_buf_f32);
                tracing::info!("DUMPED final FP32 hidden state to {} ({} bytes)", dump_path, f32_bytes);
            }
        }

        let _t_norm_start = std::time::Instant::now();

        // Apply final RMSNorm: FP32 → normalize in FP32 → FP16 for logit projection
        // The final norm is stored at layer=0xFFFF, segment=8 (new convention)
        // or layer=0, segment=8 (legacy GGUF conversion). Try both.
        let eps = self.model_config.rms_norm_eps;

        // Load final norm weight: try layer=0xFFFF (new convention) then layer=0 (legacy).
        // NOTE: We must NOT hold raw pointers across .await boundaries (Send requirement).
        self.ensure_shared_tensor_device(0xFFFF, 8).await;
        if self.get_device_tensor(0xFFFF, 8).is_none() {
            // Fallback: old GGUF conversions stored final norm at layer=0
            self.ensure_shared_tensor_device(0, 8).await;
        }
        let norm_tensor = self.get_device_tensor(0xFFFF, 8)
            .or_else(|| self.get_device_tensor(0, 8));
        if let Some((norm_ptr, _norm_size)) = norm_tensor {
            if step == 0 {
                tracing::info!("Final norm: loaded from .vib3 ({} bytes)", _norm_size);
            }
            kernels::rms_norm_f32_to_f16(
                self.hidden_state_f32.as_ptr(),
                self.hidden_state.as_mut_ptr(),
                norm_ptr,
                hidden_dim,
                eps,
                &self.stream,
            )?;
        } else {
            tracing::info!("Final norm: FALLBACK to no-weight RMSNorm (norm tensor not found!)");
            kernels::f32_to_f16(
                self.hidden_state_f32.as_ptr(),
                self.hidden_state.as_mut_ptr(),
                hidden_dim,
                &self.stream,
            )?;
            kernels::rms_norm_no_weight(
                self.hidden_state.as_mut_ptr(),
                hidden_dim,
                eps,
                &self.stream,
            )?;
        }

        // Dump post-norm FP16 hidden state for step 0
        if step == 0 && self.diag_enabled {
            self.stream.synchronize()?;
            let mut norm_buf = vec![0u8; hidden_bytes];
            self.hidden_state.copy_to_host(&mut norm_buf)?;
            let norm_f16 = unsafe {
                std::slice::from_raw_parts(norm_buf.as_ptr() as *const f16, hidden_dim)
            };
            let norm_l2 = norm_f16.iter().map(|v| { let f = v.to_f32(); f * f }).sum::<f32>().sqrt();
            tracing::info!(
                "Post-norm FP16 hidden state step=0: L2={:.4}, first4=[{:.6},{:.6},{:.6},{:.6}]",
                norm_l2, norm_f16[0].to_f32(), norm_f16[1].to_f32(), norm_f16[2].to_f32(), norm_f16[3].to_f32(),
            );
            let dump_path = "/home/brian/code/vib3/dump/vib3_post_norm_f16_step0.bin";
            let _ = std::fs::write(dump_path, &norm_buf);
            tracing::info!("DUMPED post-norm FP16 hidden state to {}", dump_path);
        }

        let t_norm_done = std::time::Instant::now();

        // Sync before logits to separate layer GPU time from lm_head time
        self.stream.synchronize()?;
        let t_sync_done = std::time::Instant::now();

        // Compute logits from hidden state (both in VRAM for GPU dispatch)
        let vocab_size = self.model_config.vocab_size as usize;
        let logits_bytes = vocab_size * std::mem::size_of::<f32>();

        // Optional lm_head parity diagnostics (step 0):
        // - Logits parity against paged lm_head reference
        // - Input parity to verify fast path does not mutate hidden_state
        // Enable with: VIB3_DIAG_LMHEAD_PARITY=1
        let lm_head_parity_enabled = std::env::var("VIB3_DIAG_LMHEAD_PARITY").map_or(false, |v| v == "1");
        // Force low-VRAM paged logits path (debug/validation): skip shared/NVFP4 lm_head paths.
        // Enable with: VIB3_FORCE_PAGED_LM_HEAD=1
        let force_paged_lm_head = std::env::var("VIB3_FORCE_PAGED_LM_HEAD").map_or(false, |v| v == "1");
        // Correctness-first guard: FP16 shared lm_head fast GEMV is opt-in.
        // Enable with: VIB3_ALLOW_FP16_SHARED_FAST_LM_HEAD=1
        let allow_fp16_shared_fast_lm_head =
            std::env::var("VIB3_ALLOW_FP16_SHARED_FAST_LM_HEAD").map_or(false, |v| v == "1");
        let lm_head_input_before: Option<Vec<f16>> = if lm_head_parity_enabled && step == 0 {
            self.stream.synchronize()?;
            let mut buf = vec![0u8; hidden_bytes];
            self.hidden_state.copy_to_host(&mut buf)?;
            let slice = unsafe {
                std::slice::from_raw_parts(buf.as_ptr() as *const f16, hidden_dim)
            };
            Some(slice.to_vec())
        } else {
            None
        };

        let mut lm_head_path = if force_paged_lm_head {
            "paged_forced"
        } else {
            "paged"
        };

        let maybe_logits = if force_paged_lm_head {
            None
        } else {
            // Load lm_head weights to device and convert to NVFP4 on first use
            let lm_head_ready = self.ensure_shared_tensor_device(0, 11).await;
            let mut lm_head_tensor = self.get_device_tensor(0, 11);

            if !lm_head_ready && lm_head_tensor.is_none() && self.lm_head_nvfp4.is_none() {
                tracing::warn!(
                    "lm_head tensor (L0 S11) is unavailable on device (likely VRAM OOM); falling back to paged lm_head"
                );
            }

            // Lazily convert lm_head to NVFP4 on first token.
            // Default policy: keep FP16 lm_head for Qwen3.5 until NVFP4 parity is proven.
            // Override with VIB3_FP16_LM_HEAD=0/1 as needed.
            let fp16_lm_head = match std::env::var("VIB3_FP16_LM_HEAD") {
                Ok(v) => v == "1",
                Err(_) => self.model_config.architecture == "qwen3_5_moe",
            };
            if self.lm_head_nvfp4.is_none() && !fp16_lm_head {
                if let Some((lm_head_ptr, lm_head_size)) = lm_head_tensor {
                    let expected_fp16 = vocab_size * hidden_dim * 2;
                    if lm_head_size == expected_fp16 {
                        tracing::info!(
                            "Converting lm_head to NVFP4: {}x{} FP16 ({:.1} MB) → NVFP4",
                            vocab_size, hidden_dim, lm_head_size as f64 / (1024.0 * 1024.0)
                        );
                        match kernels::fp16_to_nvfp4_weight(lm_head_ptr, vocab_size, hidden_dim, &self.stream) {
                            Ok(nvfp4_buf) => {
                                self.stream.synchronize()?;
                                tracing::info!(
                                    "lm_head NVFP4 conversion complete: {:.1} MB",
                                    nvfp4_buf.size() as f64 / (1024.0 * 1024.0)
                                );
                                // Repack lm_head FP4 data to tiled layout for coalesced GEMV
                                let fp4_data_size = vocab_size * hidden_dim.div_ceil(2);
                                match DeviceBuffer::new(fp4_data_size) {
                                    Ok(temp_buf) => {
                                        if let Err(e) = kernels::repack_row_to_tiled(
                                            nvfp4_buf.as_mut_ptr(),
                                            temp_buf.as_mut_ptr(),
                                            vocab_size,
                                            hidden_dim,
                                            &self.stream,
                                        ) {
                                            tracing::warn!("lm_head tiled repack failed: {}", e);
                                        } else {
                                            self.stream.synchronize()?;
                                            tracing::info!("lm_head tiled repack complete");
                                        }
                                        // temp_buf dropped here, frees VRAM
                                    }
                                    Err(e) => {
                                        tracing::warn!("lm_head tiled repack temp alloc failed: {}", e);
                                    }
                                }
                                self.lm_head_nvfp4 = Some(nvfp4_buf);
                            }
                            Err(e) => {
                                tracing::warn!("lm_head NVFP4 conversion failed: {}, using FP16 fallback", e);
                            }
                        }
                    }
                }

                if self.lm_head_nvfp4.is_none() {
                    lm_head_tensor = self.get_device_tensor(0, 11);
                    if lm_head_tensor.is_none() {
                        tracing::warn!(
                            "lm_head tensor (L0 S11) missing after NVFP4 conversion attempt; using paged lm_head fallback"
                        );
                    }
                }
            }

            if self.lm_head_nvfp4.is_some() {
            // NVFP4 path: hidden_state FP16 → FP32 → quantize FP4 → MMA GEMV → FP32 logits
            // If any kernel in this fast path fails, clear cached NVFP4 lm_head and
            // gracefully fall back to FP16/paged lm_head instead of failing the request.
            let nvfp4_logits = (|| -> Result<Vec<f32>> {
                let nvfp4_buf = self.lm_head_nvfp4.as_ref().expect("checked is_some");

                // Step 1: Convert hidden_state from FP16 to FP32
                kernels::f16_to_f32(
                    self.hidden_state.as_ptr(),
                    self.attn_normed_f32.as_mut_ptr(),
                    hidden_dim,
                    &self.stream,
                )?;

                // Step 2: Quantize FP32 → FP4 split-half + E8M0 scales
                kernels::quantize_activation_fp4_fast(
                    self.attn_normed_f32.as_ptr(),
                    self.preq_act_fp4.as_mut_ptr(),
                    self.preq_act_scales.as_mut_ptr(),
                    hidden_dim,
                    &self.stream,
                )?;

                // Step 3: MMA GEMV — NVFP4 lm_head × FP4 activation → FP32 logits
                kernels::gemv_mma_nvfp4_tiled(
                    self.preq_act_fp4.as_ptr(),
                    self.preq_act_scales.as_ptr(),
                    nvfp4_buf.as_ptr(),
                    self.logits_dev.as_mut_ptr(),
                    hidden_dim,
                    vocab_size,
                    &self.stream,
                )?;
                self.stream.synchronize()?;
                let mut logits = vec![0.0f32; vocab_size];
                cuda_ffi::memcpy_d2h(
                    logits.as_mut_ptr() as *mut u8,
                    self.logits_dev.as_ptr(),
                    logits_bytes,
                )?;
                Ok(logits)
            })();

            match nvfp4_logits {
                Ok(logits) => {
                    lm_head_path = "nvfp4_fast";
                    Some(logits)
                }
                Err(e) => {
                    tracing::warn!(
                        "lm_head NVFP4 path failed: {}; falling back to FP16/paged lm_head",
                        e
                    );
                    self.lm_head_nvfp4 = None;
                    None
                }
            }
            } else if let Some((lm_head_ptr, _lm_head_size)) = lm_head_tensor {
            if !allow_fp16_shared_fast_lm_head {
                if self.diag_enabled && step == 0 {
                    tracing::info!(
                        "LMHEAD_GUARD: FP16 shared fast lm_head disabled by default; using paged lm_head (set VIB3_ALLOW_FP16_SHARED_FAST_LM_HEAD=1 to override)"
                    );
                }
                None
            } else {
                // FP16 shared fast path: scalar router GEMV
                let err = unsafe {
                    cuda_ffi::vib3_launch_router_gemv(
                        self.hidden_state.as_ptr(),
                        lm_head_ptr,
                        self.logits_dev.as_mut_ptr() as *mut f32,
                        hidden_dim as i32,
                        vocab_size as i32,
                        self.stream.raw_ptr(),
                    )
                };
                if err != 0 {
                    tracing::warn!(
                        "lm_head GEMV failed (err={}), falling back to paged lm_head",
                        err
                    );
                    None
                } else {
                    self.stream.synchronize()?;
                    let mut logits = vec![0.0f32; vocab_size];
                    cuda_ffi::memcpy_d2h(
                        logits.as_mut_ptr() as *mut u8,
                        self.logits_dev.as_ptr(),
                        logits_bytes,
                    )?;
                    lm_head_path = "fp16_shared_fast";
                    Some(logits)
                }
            }
            } else {
                None
            }
        };

        let used_fast_lm_head_path = maybe_logits.is_some();
        let logits = if let Some(logits) = maybe_logits {
            logits
        } else {
            self.compute_logits_paged_lm_head(hidden_dim, vocab_size, logits_bytes)
                .await?
        };

        // Optional lm_head parity diagnostic (step 0 only): compare whichever path
        // produced `logits` against paged lm_head reference logits.
        // Enable with: VIB3_DIAG_LMHEAD_PARITY=1 (and VIB3_DIAG=1 recommended).
        if lm_head_parity_enabled && step == 0 && used_fast_lm_head_path {
            self.stream.synchronize()?;
            let mut buf_after = vec![0u8; hidden_bytes];
            self.hidden_state.copy_to_host(&mut buf_after)?;
            let hidden_after = unsafe {
                std::slice::from_raw_parts(buf_after.as_ptr() as *const f16, hidden_dim)
            };
            if let Some(hidden_before) = &lm_head_input_before {
                let mut dot = 0.0f64;
                let mut a2 = 0.0f64;
                let mut b2 = 0.0f64;
                let mut diff2 = 0.0f64;
                let mut max_abs = 0.0f32;
                for (a, b) in hidden_before.iter().zip(hidden_after.iter()) {
                    let af = a.to_f32() as f64;
                    let bf = b.to_f32() as f64;
                    dot += af * bf;
                    a2 += af * af;
                    b2 += bf * bf;
                    let d = af - bf;
                    diff2 += d * d;
                    max_abs = max_abs.max((a.to_f32() - b.to_f32()).abs());
                }
                let cosine = if a2 > 0.0 && b2 > 0.0 {
                    (dot / (a2.sqrt() * b2.sqrt())) as f32
                } else {
                    0.0
                };
                let rel_l2 = if b2 > 0.0 {
                    (diff2.sqrt() / b2.sqrt()) as f32
                } else {
                    0.0
                };
                tracing::info!(
                    "LMHEAD_INPUT_PARITY step=0: cosine={:.6}, rel_l2={:.6}, max_abs={:.6}",
                    cosine,
                    rel_l2,
                    max_abs,
                );
            }

            if lm_head_path == "fp16_shared_fast" {
                if let Some((lm_head_ptr, _)) = self.get_device_tensor(0, 11) {
                    let rows = [0usize, 1, 16, 17, 59, 760, 247804];
                    match self
                        .diagnose_lm_head_weight_row_parity(hidden_dim, vocab_size, lm_head_ptr as usize, &rows)
                        .await
                    {
                        Ok(mismatch_rows) => {
                            if mismatch_rows > 0 {
                                tracing::warn!(
                                    "LMHEAD_WEIGHT_PARITY_SUMMARY: mismatched_rows={} (paged lm_head source may be unreliable for parity reference)",
                                    mismatch_rows
                                );
                            }
                        }
                        Err(e) => {
                            tracing::warn!("LMHEAD_WEIGHT_PARITY failed: {}", e);
                        }
                    }
                }
            }

            let ref_logits = self
                .compute_logits_paged_lm_head(hidden_dim, vocab_size, logits_bytes)
                .await?;

            let mut dot = 0.0f64;
            let mut a2 = 0.0f64;
            let mut b2 = 0.0f64;
            let mut diff2 = 0.0f64;
            let mut max_abs = 0.0f32;
            for (a, b) in logits.iter().zip(ref_logits.iter()) {
                let af = *a as f64;
                let bf = *b as f64;
                dot += af * bf;
                a2 += af * af;
                b2 += bf * bf;
                let d = af - bf;
                diff2 += d * d;
                max_abs = max_abs.max((a - b).abs());
            }

            let cosine = if a2 > 0.0 && b2 > 0.0 {
                (dot / (a2.sqrt() * b2.sqrt())) as f32
            } else {
                0.0
            };
            let rel_l2 = if b2 > 0.0 {
                (diff2.sqrt() / b2.sqrt()) as f32
            } else {
                0.0
            };

            let top1_fast = logits
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(i, _)| i)
                .unwrap_or(0);
            let top1_ref = ref_logits
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(i, _)| i)
                .unwrap_or(0);

            tracing::info!(
                "LMHEAD_PARITY step=0: path={}, cosine={:.6}, rel_l2={:.6}, max_abs={:.6}, top1_fast={}, top1_ref={}",
                lm_head_path,
                cosine,
                rel_l2,
                max_abs,
                top1_fast,
                top1_ref,
            );
        }

        // Debug: inspect hidden state before logit computation
        tracing::trace!(
            "Hidden state (step={}, after final norm): computing logits",
            step,
        );

        // Debug: log logit distribution stats (gated — sorting 248K elements is expensive)
        if self.diag_enabled && (step < 5 || step % 10 == 0) {
            let max_logit = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let min_logit = logits.iter().cloned().fold(f32::INFINITY, f32::min);
            let mean_logit = logits.iter().sum::<f32>() / logits.len() as f32;
            let nan_count = logits.iter().filter(|x| x.is_nan()).count();
            let inf_count = logits.iter().filter(|x| x.is_infinite()).count();
            let top5: Vec<(usize, f32)> = {
                let mut indexed: Vec<(usize, f32)> = logits.iter().cloned().enumerate().collect();
                indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
                indexed.into_iter().take(5).collect()
            };
            tracing::info!(
                "Logits step={}: min={:.4}, max={:.4}, mean={:.4}, nan={}, inf={}, total_logits={}, top5={:?}",
                step, min_logit, max_logit, mean_logit, nan_count, inf_count, logits.len(), top5
            );
            let think_tok = 248068usize;
            if think_tok < logits.len() {
                tracing::info!("Logits step={}: <think>(248068) logit={:.4}", step, logits[think_tok]);
            } else {
                tracing::warn!("Logits step={}: <think>(248068) OUT OF RANGE (vocab_size={})", step, logits.len());
            }
        }

        let t_logits_done = std::time::Instant::now();

        let token_id = self.sampler.sample(&logits, params, recent_tokens);
        let t_sample_done = std::time::Instant::now();
        tracing::info!(
            "Token step={}: id={}, total={:.1}ms (layers_gpu_sync={:.1}ms, lm_head+logits={:.1}ms, sample={:.1}ms)",
            step, token_id,
            compute_start.elapsed().as_secs_f64() * 1000.0,
            (t_sync_done - t_norm_done).as_secs_f64() * 1000.0,
            (t_logits_done - t_sync_done).as_secs_f64() * 1000.0,
            (t_sample_done - t_logits_done).as_secs_f64() * 1000.0,
        );
        self.stats.tokens_generated.fetch_add(1, Ordering::Relaxed);

        // Update trajectory for prediction (D2H hidden_state to host for f32 conversion)
        self.stream.synchronize()?;
        self.hidden_state
            .copy_to_host(&mut self.host_staging[..hidden_bytes])?;
        let state_vec: Vec<f32> = unsafe {
            std::slice::from_raw_parts(self.host_staging.as_ptr() as *const f16, hidden_dim)
        }
        .iter()
        .map(|x| x.to_f32())
        .collect();
        self.planner.update_trajectory(state_vec);

        // Speculative prefetch via vector index
        self.planner.submit_vector_prefetch(3);

        // ── Prepare hidden_state for the NEXT decode step ──
        // Embed the newly sampled token (device-side embedding table — already loaded during prefill)
        self.ensure_shared_tensor_device(0, 10).await;
        if let Some((embed_ptr, _)) = self.get_device_tensor(0, 10) {
            kernels::embedding_lookup(
                embed_ptr,
                token_id,
                hidden_dim,
                self.hidden_state.as_mut_ptr(),
            );
        } else {
            // Fallback: deterministic embedding from token ID — build on host, copy to device
            let mut host_embed = vec![0u8; hidden_bytes];
            let state = unsafe {
                std::slice::from_raw_parts_mut(host_embed.as_mut_ptr() as *mut f16, hidden_dim)
            };
            for (d, s) in state.iter_mut().enumerate().take(hidden_dim) {
                let freq = (d as f32 + 1.0) * 0.001;
                let val = (token_id as f32 * freq).sin() * 0.1;
                *s = f16::from_f32(val);
            }
            let _ = self.hidden_state.copy_from_host(&host_embed);
        }

        // Initialize FP32 hidden state accumulator from FP16 embedding
        kernels::f16_to_f32(
            self.hidden_state.as_ptr(),
            self.hidden_state_f32.as_mut_ptr(),
            hidden_dim,
            &self.stream,
        )?;

        Ok(token_id)
    }

    /// Execute one expert's computation (SwiGLU + down_proj + weighted accumulate).
    ///
    /// Expert computation follows the SwiGLU MoE architecture:
    /// 1. `intermediate = SiLU(input × up_proj) * (input × gate_proj)`
    /// 2. `output = intermediate × down_proj`
    ///
    /// Each projection matrix may span multiple 2MB pages. Pages contain
    /// contiguous row slices: page_idx=0 has rows [0..row_count], page_idx=1
    /// has rows [row_count..2*row_count], etc.
    ///
    /// For SwiGLU: each up/gate page pair produces a slice of the intermediate
    /// vector. These slices are written to the correct offset in expert_output_buf.
    ///
    /// For down_proj: each page covers a slice of OUTPUT dimensions (hidden_dim).
    /// The dot product uses the FULL intermediate vector as input, so each page
    /// produces partial output values that are written to the correct offset.
    async fn execute_expert(
        &mut self,
        expert_plan: &crate::runtime::query_planner::ExpertPlan,
        hidden_dim: usize,
    ) -> Result<()> {
        let force_fp16_experts = std::env::var("VIB3_FP16_EXPERTS").map_or(false, |v| v == "1");
        let use_nvfp4_experts = self.model_config.expert_dtype == DType::NVFP4 && !force_fp16_experts;

        let mut up_pages = Vec::new();
        let mut gate_pages = Vec::new();
        let mut down_pages = Vec::new();

        for rp in &expert_plan.pages {
            match rp.id.segment {
                0 => up_pages.push(rp),
                1 => gate_pages.push(rp),
                2 => down_pages.push(rp),
                _ => {}
            }
        }

        // Compute engine layer for diagnostics
        let storage_layer = expert_plan.pages.first().map(|p| p.id.layer).unwrap_or(0);
        let dense_offset = self.model_config.dense_layer_idx as u16;
        let engine_layer = storage_layer.saturating_sub(dense_offset);

        // ═══ VRAM vs DISK COMPARISON: detect page corruption in transit ═══
        // Check whether the bytes the CUDA kernel is about to read in VRAM
        // actually match what's stored on disk in the .vib3 file.
        // Runs at position 0 for layers 1,5,6,7,8 to isolate where corruption starts.
        // Gated behind VIB3_DIAG=1 to avoid overhead in normal inference.
        {

            if self.diag_enabled && self.position == 0
                && (engine_layer == 1 || engine_layer == 5 || engine_layer == 6
                    || engine_layer == 7 || engine_layer == 8)
            {
                let check_size: usize = 64;
                // Check first gate page and first up page
                for (seg_name, pages_list) in [("gate", &gate_pages), ("up", &up_pages)] {
                    if let Some(pg) = pages_list.first() {
                        if !pg.device_ptr.is_null() {
                            self.stream.synchronize()?;

                            // 1. Read from VRAM
                            let mut vram_bytes = vec![0u8; check_size];
                            let _ = cuda_ffi::memcpy_d2h(
                                vram_bytes.as_mut_ptr(),
                                pg.device_ptr as *const u8,
                                check_size,
                            );

                            // 2. Read from disk: find catalog entry for this page
                            let pid = &pg.id;
                            let catalog = self.model_file.page_catalog();
                            if let Some(cat_idx) = catalog.iter().position(|e| {
                                e.layer == pid.layer
                                    && e.expert == pid.expert
                                    && e.segment == pid.segment
                                    && e.page_idx == pid.page_idx
                            }) {
                                let entry = self.model_file.page(cat_idx);
                                let raw_sz = { entry.raw_size } as usize;
                                let mut disk_buf = vec![0u8; raw_sz.max(check_size)];
                                match self.model_file.read_page_sync(cat_idx, &mut disk_buf) {
                                    Ok(bytes_read) => {
                                        let cmp_len = check_size.min(bytes_read);
                                        let matches = vram_bytes[..cmp_len] == disk_buf[..cmp_len];
                                        if matches {
                                            tracing::info!(
                                                "VRAM_DISK_CMP L{} e{} {} pg{}: MATCH ({} bytes)",
                                                engine_layer, pid.expert, seg_name, pid.page_idx, cmp_len,
                                            );
                                        } else {
                                            let n_diff = vram_bytes[..cmp_len].iter()
                                                .zip(disk_buf[..cmp_len].iter())
                                                .filter(|(a, b)| a != b)
                                                .count();
                                            tracing::error!(
                                                "VRAM_DISK_CMP L{} e{} {} pg{}: MISMATCH! {}/{} bytes differ",
                                                engine_layer, pid.expert, seg_name, pid.page_idx, n_diff, cmp_len,
                                            );
                                            tracing::error!(
                                                "  VRAM: {:02x?}",
                                                &vram_bytes[..32.min(cmp_len)],
                                            );
                                            tracing::error!(
                                                "  DISK: {:02x?}",
                                                &disk_buf[..32.min(cmp_len)],
                                            );
                                            // Also check file_offset for sanity
                                            let fo = { entry.file_offset };
                                            let cs = { entry.compressed_size };
                                            let comp = entry.compression;
                                            tracing::error!(
                                                "  file_offset={}, compressed_size={}, raw_size={}, compression={}",
                                                fo, cs, raw_sz, comp,
                                            );
                                        }
                                    }
                                    Err(e) => {
                                        tracing::error!("Failed to read page from disk: {}", e);
                                    }
                                }
                            } else {
                                tracing::error!(
                                    "VRAM_DISK_CMP: Page not found in catalog: L{} e{} seg{} pg{}",
                                    pid.layer, pid.expert, pid.segment, pid.page_idx,
                                );
                            }
                        }
                    }
                }
            }
        }
        // ═══ END VRAM vs DISK COMPARISON ═══

        // ═══ FAST PATH: Batched expert (single C call) for NVFP4 ═══
        // Requires pre-repacked split-half weights (done at load time).
        // Each expert has exactly 1 up, 1 gate, 1 down page (all single-page).
        if use_nvfp4_experts
            && up_pages.len() == 1
            && gate_pages.len() == 1
            && down_pages.len() == 1
        {
            let up_ptr = up_pages[0].device_ptr;
            let gate_ptr = gate_pages[0].device_ptr;
            let down_ptr = down_pages[0].device_ptr;

            if !up_ptr.is_null() && !gate_ptr.is_null() && !down_ptr.is_null() {
                let expert_hidden_dim = self.model_config.expert_hidden_dim as usize;

                // ═══ SCALE OFFSET DIAGNOSTIC ═══
                if self.diag_enabled && self.position == 0 && engine_layer == 0 {
                    self.stream.synchronize()?;
                    let packed_k_in = hidden_dim / 2;    // 1024
                    let fp4_data_size = packed_k_in * expert_hidden_dim; // 1024 * 512 = 524288
                    let num_groups = hidden_dim / 32;     // 64
                    let scale_data_size = expert_hidden_dim * num_groups * 2; // 512 * 64 * 2 = 65536
                    let total_expected = fp4_data_size + scale_data_size; // 589824

                    // Read bytes at the computed scale offset
                    let scale_offset = fp4_data_size;
                    let check_bytes = 32usize;
                    let mut scale_buf = vec![0u8; check_bytes];
                    // SAFETY: reading from device pointer at known offset
                    let _ = crate::compute::cuda_ffi::memcpy_d2h(
                        scale_buf.as_mut_ptr(),
                        unsafe { (up_ptr as *const u8).add(scale_offset) },
                        check_bytes,
                    );
                    // Also read 32 bytes from 16 bytes BEFORE scale offset (should be FP4 data)
                    let mut pre_scale_buf = vec![0u8; check_bytes];
                    if scale_offset >= 16 {
                        let _ = crate::compute::cuda_ffi::memcpy_d2h(
                            pre_scale_buf.as_mut_ptr(),
                            unsafe { (up_ptr as *const u8).add(scale_offset - 16) },
                            check_bytes,
                        );
                    }
                    // Read page catalog entry raw_size
                    let up_pid = &up_pages[0].id;
                    let catalog = self.model_file.page_catalog();
                    let raw_sz = catalog.iter()
                        .find(|e| e.layer == up_pid.layer && e.expert == up_pid.expert && e.segment == up_pid.segment && e.page_idx == up_pid.page_idx)
                        .map(|e| e.raw_size)
                        .unwrap_or(0);

                    tracing::error!(
                        "SCALE_OFFSET_DIAG L{} e{}: fp4_data={}, scale_data={}, total_expected={}, page_raw_size={}, \
                         at_scale_offset={:02x?}, pre_scale={:02x?}, expert_hidden_dim={}, hidden_dim={}, row_count={}",
                        engine_layer, expert_plan.expert_id,
                        fp4_data_size, scale_data_size, total_expected, raw_sz,
                        &scale_buf[..16], &pre_scale_buf[..16],
                        expert_hidden_dim, hidden_dim, up_pages[0].row_count,
                    );
                }
                // ═══ END SCALE OFFSET DIAGNOSTIC ═══

                kernels::expert_batched(
                    self.preq_act_fp4.as_ptr(),
                    self.preq_act_scales.as_ptr(),
                    up_ptr as *const u8,
                    gate_ptr as *const u8,
                    down_ptr as *const u8,
                    self.layer_output_buf.as_mut_ptr() as *mut f32,
                    expert_plan.weight,
                    hidden_dim,          // k_in
                    expert_hidden_dim,   // m_mid (up/gate output rows)
                    hidden_dim,          // m_out (down output rows)
                    &self.stream,
                )?;

                // Update prediction for pages used by this expert
                for rp in &expert_plan.pages {
                    self.buffer_mgr
                        .update_prediction(&rp.id, expert_plan.weight);
                }

                return Ok(());
            }
        }
        // ═══ END FAST PATH ═══

        // Zero the expert output buffer on device (async — no pipeline stall)
        self.expert_output_buf.zero_async(&self.stream);

        // Process paired up/gate pages with fused SwiGLU.
        // Each page pair produces m_slice output rows starting at row_start.
        // The output goes to expert_output_buf[row_start..row_start+m_slice].
        for (_page_pair_idx, (up_page, gate_page)) in up_pages.iter().zip(gate_pages.iter()).enumerate() {
            let m_slice = up_page.row_count as usize;
            let row_start = up_page.row_start as usize;
            let byte_offset = row_start * std::mem::size_of::<f16>();

            // ═══ WEIGHT PAGE DATA DIAGNOSTIC ═══
            // Read first weight bytes AND scale bytes from device to verify they're nonzero
            // Gated behind VIB3_DIAG=1 to avoid overhead in normal inference.
            if self.diag_enabled && self.position == 0 && (engine_layer <= 1) && _page_pair_idx == 0 {
                self.stream.synchronize()?;
                let packed_k = hidden_dim / 2; // 4096/2 = 2048
                let num_groups = hidden_dim.div_ceil(32); // group_size=32
                let weight_data_size = packed_k * m_slice;
                let scale_data_size = num_groups * m_slice * 2; // FP16 = 2 bytes
                let total_page_size = weight_data_size + scale_data_size;

                // Read first 64 bytes of weight data
                let check_size = 64.min(weight_data_size);
                let mut wt_buf = vec![0u8; check_size];
                let _ = crate::compute::cuda_ffi::memcpy_d2h(
                    wt_buf.as_mut_ptr(), up_page.device_ptr as *const u8, check_size,
                );
                let wt_nonzero = wt_buf.iter().filter(|b| **b != 0).count();

                // Read first 64 bytes of scale data (offset by weight_data_size)
                let scale_check = 64.min(scale_data_size);
                let mut sc_buf = vec![0u8; scale_check];
                let scale_ptr = unsafe { (up_page.device_ptr as *const u8).add(weight_data_size) };
                let _ = crate::compute::cuda_ffi::memcpy_d2h(
                    sc_buf.as_mut_ptr(), scale_ptr, scale_check,
                );
                let sc_nonzero = sc_buf.iter().filter(|b| **b != 0).count();

                // Decode first 4 scales as FP16
                let scales_u16: Vec<u16> = sc_buf.chunks_exact(2).take(4)
                    .map(|c| u16::from_le_bytes([c[0], c[1]])).collect();
                let scales_f32: Vec<f32> = scales_u16.iter()
                    .map(|&bits| half::f16::from_bits(bits).to_f32()).collect();

                tracing::error!(
                    "WEIGHT PAGE DIAG L{} e{} pg{}: total_page={}, wt_data={}, scale_data={}, \
                     wt_first8={:02x?}, wt_nonzero={}/{}, \
                     sc_first8={:02x?}, sc_nonzero={}/{}, \
                     sc_as_fp16={:?}, sc_as_f32={:?}, \
                     m_slice={}, hidden_dim={}, dtype={:?}",
                    engine_layer, expert_plan.expert_id, _page_pair_idx,
                    total_page_size, weight_data_size, scale_data_size,
                    &wt_buf[..8.min(check_size)], wt_nonzero, check_size,
                    &sc_buf[..8.min(scale_check)], sc_nonzero, scale_check,
                    scales_u16, scales_f32,
                    m_slice, hidden_dim, self.model_config.expert_dtype,
                );
            }
            // ═══ END WEIGHT PAGE DATA DIAGNOSTIC ═══

            if !up_page.device_ptr.is_null() && !gate_page.device_ptr.is_null() {
                // Segment 0 = ffn_up_exps = up_proj = w3 (linear multiply)
                // Segment 1 = ffn_gate_exps = gate_proj = w1 (SiLU applied)
                if use_nvfp4_experts {
                    // Blackwell MMA path with pre-quantized activations.
                    // The activation was pre-quantized once before the expert loop.
                    let f32_byte_offset = up_page.row_start as usize * std::mem::size_of::<f32>();
                    let mma_result = kernels::fused_swiglu_mma_nvfp4_preq(
                        self.preq_act_fp4.as_ptr(),
                        self.preq_act_scales.as_ptr(),
                        up_page.device_ptr as *const u8,
                        gate_page.device_ptr as *const u8,
                        unsafe { self.expert_output_f32_buf.as_mut_ptr().add(f32_byte_offset) },
                        hidden_dim,
                        m_slice,
                        &self.stream,
                    );
                    if mma_result.is_err() {
                        // Fallback: software FP4 dequant path
                        kernels::partial_swiglu_f32_f32out(
                            self.moe_normed_f32.as_ptr(),
                            up_page.device_ptr as *const u8,
                            gate_page.device_ptr as *const u8,
                            unsafe { self.expert_output_f32_buf.as_mut_ptr().add(f32_byte_offset) },
                            hidden_dim,
                            m_slice,
                            &self.stream,
                        )?;
                    }
                } else {
                    // FP16 output path for non-NVFP4 weights (FP16, INT4, etc.)
                    kernels::partial_swiglu_f32(
                        self.moe_normed_f32.as_ptr(),
                        up_page.device_ptr as *const u8,
                        gate_page.device_ptr as *const u8,
                        unsafe { self.expert_output_buf.as_mut_ptr().add(byte_offset) },
                        hidden_dim,
                        m_slice,
                        self.model_config.expert_dtype,
                        &self.stream,
                        Some((
                            self.swiglu_up_tmp.as_mut_ptr(),
                            self.swiglu_gate_tmp.as_mut_ptr(),
                        )),
                    )?;
                }
            }
        }

        // ═══ CRITICAL DIAGNOSTIC: SwiGLU intermediate (fires L0-L2, L31 on first token) ═══
        // Gated behind VIB3_DIAG=1 to avoid overhead in normal inference.
        if self.diag_enabled && self.position <= 1 && (engine_layer <= 2 || engine_layer == 31) {
            self.stream.synchronize()?;
            let expert_hidden_dim_diag = self.model_config.expert_hidden_dim as usize;
            if use_nvfp4_experts {
                // FP32 path diagnostic
                let mut inter_buf = vec![0u8; expert_hidden_dim_diag * 4]; // FP32
                self.expert_output_f32_buf.copy_to_host(&mut inter_buf)?;
                let inter_f32 = unsafe { std::slice::from_raw_parts(inter_buf.as_ptr() as *const f32, expert_hidden_dim_diag) };
                let l2 = inter_f32.iter().map(|v| v * v).sum::<f32>().sqrt();
                let max_abs = inter_f32.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
                let zeros = inter_f32.iter().filter(|v| **v == 0.0).count();
                let first4: Vec<f32> = inter_f32[..4.min(expert_hidden_dim_diag)].to_vec();
                tracing::warn!(
                    "SWIGLU_INTER DIAG L{} pos={} e{}: L2={:.6}, max_abs={:.6}, zeros={}/{}, up_pages={}, gate_pages={}, first4={:?} (FP32 path)",
                    engine_layer, self.position, expert_plan.expert_id, l2, max_abs, zeros, expert_hidden_dim_diag,
                    up_pages.len(), gate_pages.len(), first4,
                );
            } else {
                // FP16 path diagnostic
                let mut inter_buf = vec![0u8; expert_hidden_dim_diag * 2]; // FP16
                self.expert_output_buf.copy_to_host(&mut inter_buf)?;
                let inter_f16 = unsafe { std::slice::from_raw_parts(inter_buf.as_ptr() as *const f16, expert_hidden_dim_diag) };
                let l2 = inter_f16.iter().map(|v| { let f = v.to_f32(); f * f }).sum::<f32>().sqrt();
                let max_abs = inter_f16.iter().map(|v| v.to_f32().abs()).fold(0.0f32, f32::max);
                let zeros = inter_f16.iter().filter(|v| v.to_f32() == 0.0).count();
                let first4: Vec<f32> = inter_f16[..4.min(expert_hidden_dim_diag)].iter().map(|v| v.to_f32()).collect();
                tracing::warn!(
                    "SWIGLU_INTER DIAG L{} pos={} e{}: L2={:.6}, max_abs={:.6}, zeros={}/{}, up_pages={}, gate_pages={}, first4={:?}",
                    engine_layer, self.position, expert_plan.expert_id, l2, max_abs, zeros, expert_hidden_dim_diag,
                    up_pages.len(), gate_pages.len(), first4,
                );
                // Also check the swiglu temp buffers (up_result, gate_result for INT4 decomposed path)
                let tmp_check_size = (expert_hidden_dim_diag * 2).min(self.swiglu_up_tmp.size());
                if tmp_check_size > 0 {
                    let mut up_tmp_buf = vec![0u8; tmp_check_size];
                    self.swiglu_up_tmp.copy_to_host(&mut up_tmp_buf)?;
                    let up_tmp_f16 = unsafe { std::slice::from_raw_parts(up_tmp_buf.as_ptr() as *const f16, tmp_check_size / 2) };
                    let up_l2 = up_tmp_f16.iter().map(|v| { let f = v.to_f32(); f * f }).sum::<f32>().sqrt();
                    let mut gate_tmp_buf = vec![0u8; tmp_check_size];
                    self.swiglu_gate_tmp.copy_to_host(&mut gate_tmp_buf)?;
                    let gate_tmp_f16 = unsafe { std::slice::from_raw_parts(gate_tmp_buf.as_ptr() as *const f16, tmp_check_size / 2) };
                    let gate_l2 = gate_tmp_f16.iter().map(|v| { let f = v.to_f32(); f * f }).sum::<f32>().sqrt();
                    tracing::warn!(
                        "  SWIGLU TEMPS: up_tmp L2={:.6}, gate_tmp L2={:.6} (check_size={})",
                        up_l2, gate_l2, tmp_check_size,
                    );
                }
            }
        }

        // Apply down_proj: project expert_hidden_dim back to hidden_dim.
        // Uses pre-allocated down_proj_buf (DeviceBuffer in VRAM).
        let expert_hidden_dim = self.model_config.expert_hidden_dim as usize;
        if use_nvfp4_experts {
            // FP32 path: zero FP32 buffer, accumulate FP32→FP32
            self.down_proj_f32_buf.zero_async(&self.stream);

            // Pre-quantize the SwiGLU output for down_proj MMA
            // (same activation across all down_proj pages of this expert)
            kernels::quantize_activation_fp4(
                self.expert_output_f32_buf.as_ptr(),
                self.preq_down_act_fp4.as_mut_ptr(),
                self.preq_down_act_scales.as_mut_ptr(),
                expert_hidden_dim,
                &self.stream,
            )?;
        } else {
            self.down_proj_buf.zero_async(&self.stream);
        }

        for down_page in &down_pages {
            let m_slice = down_page.row_count as usize;
            let row_start = down_page.row_start as usize;

            if !down_page.device_ptr.is_null() {
                if use_nvfp4_experts {
                    // Pre-quantized MMA path for down_proj
                    let f32_byte_offset = row_start * std::mem::size_of::<f32>();
                    let mma_result = kernels::gemv_mma_nvfp4_preq(
                        self.preq_down_act_fp4.as_ptr(),
                        self.preq_down_act_scales.as_ptr(),
                        down_page.device_ptr as *const u8,
                        unsafe { self.down_proj_f32_buf.as_mut_ptr().add(f32_byte_offset) },
                        expert_hidden_dim,
                        m_slice,
                        &self.stream,
                    );
                    if mma_result.is_err() {
                        // Fallback: software FP4 dequant path
                        kernels::partial_matmul_nvfp4_f32(
                            self.expert_output_f32_buf.as_ptr(),
                            down_page.device_ptr as *const u8,
                            unsafe { self.down_proj_f32_buf.as_mut_ptr().add(f32_byte_offset) },
                            expert_hidden_dim,
                            m_slice,
                            &self.stream,
                        )?;
                    }
                } else {
                    // FP16 path for non-NVFP4 weights
                    let byte_offset = row_start * std::mem::size_of::<f16>();
                    kernels::partial_matmul(
                        self.expert_output_buf.as_ptr(),
                        down_page.device_ptr as *const u8,
                        unsafe { self.down_proj_buf.as_mut_ptr().add(byte_offset) },
                        expert_hidden_dim,
                        m_slice,
                        self.model_config.expert_dtype,
                        &self.stream,
                    )?;
                }
            }
        }

        // Diagnostic: per-expert down_proj output at L6, pos=0
        if self.diag_enabled && engine_layer == 6 && self.position == 0 {
            self.stream.synchronize()?;
            if use_nvfp4_experts {
                // FP32 path diagnostic
                let mut down_buf = vec![0u8; hidden_dim * 4];
                self.down_proj_f32_buf.copy_to_host(&mut down_buf)?;
                let down_f32 = unsafe { std::slice::from_raw_parts(down_buf.as_ptr() as *const f32, hidden_dim) };
                let l2 = down_f32.iter().map(|v| v * v).sum::<f32>().sqrt();
                let max_abs = down_f32.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
                let first4: Vec<f32> = down_f32[..4].to_vec();
                tracing::info!(
                    "L6 EXPERT e{} DOWN_PROJ: L2={:.4}, max_abs={:.4}, first4={:?}, pages={} (FP32 path)",
                    expert_plan.expert_id, l2, max_abs, first4, down_pages.len(),
                );
                if expert_plan.expert_id == 243 {
                    let dump_dir = "/model/dump";
                    let _ = std::fs::create_dir_all(dump_dir);
                    let _ = std::fs::write(format!("{}/e243_downproj_L6_pos0.bin", dump_dir), &down_buf);
                    let ehd = self.model_config.expert_hidden_dim as usize;
                    let mut inter_buf = vec![0u8; ehd * 4]; // FP32
                    self.expert_output_f32_buf.copy_to_host(&mut inter_buf)?;
                    let _ = std::fs::write(format!("{}/e243_swiglu_L6_pos0.bin", dump_dir), &inter_buf);
                    tracing::info!("DUMPED e243 down_proj (FP32) and intermediate (FP32) to {}", dump_dir);
                }
            } else {
                // FP16 path diagnostic
                let mut down_buf = vec![0u8; hidden_dim * 2];
                self.down_proj_buf.copy_to_host(&mut down_buf)?;
                let down_f16 = unsafe { std::slice::from_raw_parts(down_buf.as_ptr() as *const f16, hidden_dim) };
                let l2 = down_f16.iter().map(|v| { let f = v.to_f32(); f * f }).sum::<f32>().sqrt();
                let max_abs = down_f16.iter().map(|v| v.to_f32().abs()).fold(0.0f32, f32::max);
                let first4: Vec<f32> = down_f16[..4].iter().map(|v| v.to_f32()).collect();
                tracing::info!(
                    "L6 EXPERT e{} DOWN_PROJ: L2={:.4}, max_abs={:.4}, first4={:?}, pages={}",
                    expert_plan.expert_id, l2, max_abs, first4, down_pages.len(),
                );
                if expert_plan.expert_id == 243 {
                    let dump_dir = "/model/dump";
                    let _ = std::fs::create_dir_all(dump_dir);
                    let _ = std::fs::write(format!("{}/e243_downproj_L6_pos0.bin", dump_dir), &down_buf);
                    let ehd = self.model_config.expert_hidden_dim as usize;
                    let mut inter_buf = vec![0u8; ehd * 2];
                    self.expert_output_buf.copy_to_host(&mut inter_buf)?;
                    let _ = std::fs::write(format!("{}/e243_swiglu_L6_pos0.bin", dump_dir), &inter_buf);
                    tracing::info!("DUMPED e243 down_proj and intermediate to {}", dump_dir);
                }
            }
        }

        // Accumulate expert contribution into FP32 buffer
        if use_nvfp4_experts {
            // Full FP32 path: down_proj output is FP32, no FP16 truncation anywhere
            kernels::weighted_accumulate_f32_f32(
                self.layer_output_buf.as_mut_ptr(),
                self.down_proj_f32_buf.as_ptr(),
                expert_plan.weight,
                hidden_dim,
                &self.stream,
            )?;
        } else {
            // FP16→FP32 path for non-NVFP4 weights
            kernels::weighted_accumulate_f32(
                self.layer_output_buf.as_mut_ptr(),
                self.down_proj_buf.as_ptr(),
                expert_plan.weight,
                hidden_dim,
                &self.stream,
            )?;
        }

        // Update prediction for pages used by this expert
        for rp in &expert_plan.pages {
            self.buffer_mgr
                .update_prediction(&rp.id, expert_plan.weight);
        }

        Ok(())
    }

    /// Execute the shared (unconditional) expert for a layer.
    ///
    /// The shared expert has the same SwiGLU structure as routed experts
    /// but uses `shared_intermediate_size` (2048 for Kimi K2.5) instead of
    /// dense `intermediate_size` (18432). It runs on every token, unconditionally,
    /// and its output is added to the layer output with weight 1.0.
    ///
    /// Uses pre-allocated `shared_expert_inter_dev` and `shared_expert_down_dev`
    /// buffers to avoid per-layer cudaMalloc/cudaFree overhead.
    async fn execute_shared_expert(&mut self, layer: u16, hidden_dim: usize) -> Result<()> {
        let shared_hidden = self.model_config.shared_intermediate_size as usize;
        if shared_hidden == 0 {
            return Ok(());
        }

        // Load shared expert weights to device: segments 14 (up), 15 (gate), 16 (down)
        self.ensure_shared_tensor_device(layer, 14).await;
        self.ensure_shared_tensor_device(layer, 15).await;
        self.ensure_shared_tensor_device(layer, 16).await;

        // If we have the shared expert weights, compute SwiGLU on device
        let up_device = self.get_device_tensor(layer, 14);
        let gate_device = self.get_device_tensor(layer, 15);
        let up_gate_ptrs = match (up_device, gate_device) {
            (Some((up_ptr, _)), Some((gate_ptr, _))) => Some((up_ptr as usize, gate_ptr as usize)),
            _ => None,
        };
        if let Some((up_addr, gate_addr)) = up_gate_ptrs {
            let up_ptr = up_addr as *const u8;
            let gate_ptr = gate_addr as *const u8;
            // FP32-input SwiGLU for shared expert: reads FP32 moe_normed_f32, FP16 weights
            kernels::partial_swiglu_f32(
                self.moe_normed_f32.as_ptr(),
                up_ptr,
                gate_ptr,
                self.shared_expert_inter_dev.as_mut_ptr(),
                hidden_dim,
                shared_hidden,
                DType::FP16,
                &self.stream,
                None, // shared expert weights are FP16, fused kernel
            )?;

            // Dump shared expert SwiGLU intermediate at L0
            if self.diag_enabled && layer == 0 {
                self.stream.synchronize()?;
                let dump_dir = "/home/brian/code/vib3/dump";
                let tok = self.position;
                // SwiGLU intermediate is FP16, size = shared_hidden * 2 bytes
                let inter_bytes = shared_hidden * 2;
                let mut buf = vec![0u8; inter_bytes];
                self.shared_expert_inter_dev.copy_to_host(&mut buf)?;
                let _ = std::fs::write(
                    format!("{}/vib3_shexp_inter_L0_tok{}.bin", dump_dir, tok),
                    &buf,
                );
                // Convert to f32 for stats
                let f16_slice = unsafe {
                    std::slice::from_raw_parts(buf.as_ptr() as *const f16, shared_hidden)
                };
                let l2: f32 = f16_slice.iter().map(|v| { let f = v.to_f32(); f * f }).sum::<f32>().sqrt();
                let max_abs = f16_slice.iter().map(|v| v.to_f32().abs()).fold(0.0f32, f32::max);
                let zeros = f16_slice.iter().filter(|v| v.to_f32() == 0.0).count();
                tracing::info!(
                    "SHEXP_SWIGLU L{} pos={}: inter L2={:.6}, max_abs={:.6}, zeros={}/{}",
                    layer, self.position, l2, max_abs, zeros, shared_hidden,
                );
            }

            // Apply down_proj using pre-allocated shared_expert_down_dev
            if let Some((down_ptr, _)) = self.get_device_tensor(layer, 16) {
                // Zero the down output buffer before accumulating
                self.shared_expert_down_dev.zero_async(&self.stream);

                kernels::partial_matmul(
                    self.shared_expert_inter_dev.as_ptr(),
                    down_ptr,
                    self.shared_expert_down_dev.as_mut_ptr(),
                    shared_hidden,
                    hidden_dim,
                    DType::FP16,
                    &self.stream,
                )?;

                // Apply sigmoid gate to shared expert output before accumulating.
                self.ensure_shared_tensor_device(layer, 26).await;
                if let Some((gate_w_ptr, _)) = self.get_device_tensor(layer, 26) {
                    // Compute dot product: moe_normed_f32 (FP32) × gate_weight (FP16) → 1 FP32 scalar
                    kernels::linear_projection_f32_to_f32(
                        self.moe_normed_f32.as_ptr(),
                        gate_w_ptr,
                        self.shared_expert_inter_dev.as_mut_ptr(),
                        hidden_dim,
                        1,
                        &self.stream,
                    )?;

                    // Dump shared expert down_proj output BEFORE sigmoid gating at L0
                    // NOTE: shared_expert_down_dev is FP16 (output of partial_matmul)
                    if self.diag_enabled && layer == 0 {
                        self.stream.synchronize()?;
                        let dump_dir = "/home/brian/code/vib3/dump";
                        let tok = self.position;
                        // FP16 buffer: hidden_dim * 2 bytes
                        let down_bytes = hidden_dim * 2;
                        let mut buf = vec![0u8; down_bytes];
                        self.shared_expert_down_dev.copy_to_host(&mut buf)?;
                        let _ = std::fs::write(
                            format!("{}/vib3_shexp_down_L0_tok{}.bin", dump_dir, tok),
                            &buf,
                        );
                        // Convert to f32 for stats
                        let f16_slice = unsafe {
                            std::slice::from_raw_parts(buf.as_ptr() as *const f16, hidden_dim)
                        };
                        let l2: f32 = f16_slice.iter().map(|v| { let f = v.to_f32(); f * f }).sum::<f32>().sqrt();
                        let max_abs = f16_slice.iter().map(|v| v.to_f32().abs()).fold(0.0f32, f32::max);
                        tracing::info!(
                            "SHEXP_DOWN L{} pos={}: L2={:.6}, max_abs={:.6}",
                            layer, self.position, l2, max_abs,
                        );
                    }

                    // Fused sigmoid + gated accumulate on GPU
                    kernels::sigmoid_gated_accumulate_f32(
                        self.layer_output_buf.as_mut_ptr(),
                        self.shared_expert_down_dev.as_ptr(),
                        self.shared_expert_inter_dev.as_ptr(),
                        hidden_dim,
                        &self.stream,
                    )?;

                    if self.diag_enabled && layer <= 6 && self.position <= 4 {
                        self.stream.synchronize()?;
                        let mut gate_bytes = [0u8; 4];
                        cuda_ffi::memcpy_d2h(
                            gate_bytes.as_mut_ptr(),
                            self.shared_expert_inter_dev.as_ptr(),
                            4,
                        )?;
                        let gate_raw = f32::from_le_bytes(gate_bytes);
                        let gate_sigmoid = 1.0 / (1.0 + (-gate_raw).exp());
                        tracing::info!(
                            "SHARED_EXPERT_GATE L{} pos={}: raw={:.4}, sigmoid={:.4}",
                            layer, self.position, gate_raw, gate_sigmoid,
                        );
                    }
                } else {
                    // Fallback: no gating, accumulate with weight 1.0
                    kernels::weighted_accumulate_f32(
                        self.layer_output_buf.as_mut_ptr(),
                        self.shared_expert_down_dev.as_ptr(),
                        1.0,
                        hidden_dim,
                        &self.stream,
                    )?;
                }
            }
        }

        Ok(())
    }

    /// Run the router network for a specific layer.
    async fn run_router_for_layer(&mut self, layer: u16) -> Result<ExpertActivation> {
        let hidden_dim = self.model_config.hidden_dim as usize;
        let hidden_bytes = hidden_dim * std::mem::size_of::<f16>();
        let num_experts = self.model_config.num_experts as usize;
        let top_k = self.model_config.num_active_experts as usize;

        // Load router weights to device
        self.ensure_shared_tensor_device(layer, 3).await;

        let mut activation = ExpertActivation::new();

        // Determine router scoring function from model config
        let scoring_func = if self.model_config.scoring_func == "sigmoid" {
            kernels::RouterScoringFunc::Sigmoid {
                scaling_factor: self.model_config.routed_scaling_factor,
                normalize: self.model_config.norm_topk_prob,
            }
        } else {
            kernels::RouterScoringFunc::Softmax
        };

        // Try to use loaded router weights (both hidden_state and router weights on device)
        if let Some((router_ptr, router_size)) = self.get_device_tensor(layer, 3) {
            let expected_size = num_experts * hidden_dim * std::mem::size_of::<f16>();
            tracing::trace!(
                "Router GEMV: layer={}, router_ptr={:?}, router_size={}, expected={}, num_experts={}, hidden_dim={}, scores_ptr={:?}, scores_size={}",
                layer, router_ptr, router_size, expected_size, num_experts, hidden_dim,
                self.router_scores_dev.as_ptr(), self.router_scores_dev.size(),
            );
            if router_size < expected_size {
                tracing::error!(
                    "Router weight tensor UNDERSIZED: layer={}, got={} bytes, need={} bytes",
                    layer, router_size, expected_size,
                );
            }
            // GPU-fused router: GEMV + softmax/sigmoid + top-k + renormalize.
            // All work stays on GPU; only the final top-k (8×6 bytes) is D2H'd.
            // Uses router_stream to avoid synchronizing the main compute pipeline.
            self.router_event.record(&self.stream)?;
            self.router_event.wait_on_stream(&self.router_stream)?;
            let experts = kernels::run_router_gpu_topk(
                self.moe_normed_f32.as_ptr(),
                router_ptr,
                num_experts,
                hidden_dim,
                top_k,
                scoring_func,
                self.router_scores_dev.as_mut_ptr() as *mut f32,
                self.router_topk_ids_dev.as_mut_ptr(),
                self.router_topk_weights_dev.as_mut_ptr(),
                &self.router_stream,
            )?;
            activation.experts = experts;
        } else {
            // Fallback: D2H hidden_state, deterministic expert selection on CPU
            self.stream.synchronize()?;
            self.hidden_state
                .copy_to_host(&mut self.host_staging[..hidden_bytes])?;
            let state = unsafe {
                std::slice::from_raw_parts(self.host_staging.as_ptr() as *const f16, hidden_dim)
            };

            // Hash hidden state to select experts
            let mut scores: Vec<(u16, f32)> = (0..num_experts as u16)
                .map(|e| {
                    let mut score = 0.0f32;
                    let stride = hidden_dim / 8;
                    for d in (0..hidden_dim).step_by(stride.max(1)) {
                        let idx = (d + e as usize * 7) % hidden_dim;
                        score += state[idx].to_f32().abs();
                    }
                    (e, score)
                })
                .collect();

            scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            scores.truncate(top_k);

            // Apply scoring function matching model config
            activation.experts = match scoring_func {
                kernels::RouterScoringFunc::Sigmoid { scaling_factor, normalize } => {
                    let mut weighted: Vec<(u16, f32)> = scores
                        .into_iter()
                        .map(|(e, s)| {
                            let sigmoid_w = 1.0 / (1.0 + (-s).exp());
                            (e, sigmoid_w * scaling_factor)
                        })
                        .collect();
                    if normalize {
                        let total: f32 = weighted.iter().map(|(_, w)| w).sum();
                        if total > 1e-20 {
                            for (_, w) in &mut weighted {
                                *w /= total;
                            }
                        }
                    }
                    weighted
                }
                kernels::RouterScoringFunc::Softmax => {
                    let max_s = scores
                        .iter()
                        .map(|(_, s)| *s)
                        .fold(f32::NEG_INFINITY, f32::max);
                    let exp_sum: f32 = scores.iter().map(|(_, s)| (s - max_s).exp()).sum();
                    scores
                        .into_iter()
                        .map(|(e, s)| (e, (s - max_s).exp() / exp_sum))
                        .collect()
                }
            };
        }

        Ok(activation)
    }

    // ── Activation Mode Detection ──────────────────────────────────────

    /// Record this token's expert activations and check for mode transitions.
    ///
    /// Called once per decode step, after all MoE layers have run.
    /// The detector is gated by `detect_interval` — it records every token
    /// but only recomputes entropy every N tokens to amortize cost.
    async fn update_activation_mode(&mut self) {
        let detector = match self.mode_detector.as_mut() {
            Some(d) => d,
            None => return,
        };

        // Feed this token's expert IDs to the sliding window
        detector.record(&self.token_expert_ids);

        // Only run full detection at the configured interval
        let interval = self.config.activation_mode.detect_interval as u64;
        if interval == 0 || !self.decode_step.is_multiple_of(interval) {
            return;
        }

        // Need enough data before first detection
        if detector.total_tokens() < self.config.activation_mode.window_size as u64 / 2 {
            return;
        }

        let detection = detector.detect();
        let previous_mode = self.current_mode;

        if detection.mode != previous_mode {
            tracing::info!(
                "Activation mode transition: {} -> {} (entropy={:.2}, threshold={:.2}, confidence={:.2}, unique_experts={})",
                previous_mode,
                detection.mode,
                detection.entropy,
                detection.threshold,
                detection.confidence,
                detection.unique_experts,
            );

            self.current_mode = detection.mode;

            // Extract data needed for transitions before dropping the detector borrow
            let transition_data = match detection.mode {
                ActivationMode::Specialist => {
                    let max_pin = if self.config.activation_mode.max_pinned_experts > 0 {
                        self.config.activation_mode.max_pinned_experts
                    } else {
                        60
                    };
                    Some(detector.top_experts(max_pin))
                }
                ActivationMode::Generalist => None,
            };

            // Now handle transitions (detector borrow is released)
            match detection.mode {
                ActivationMode::Specialist => {
                    if let Some(hot_experts) = transition_data {
                        self.pin_specialist_cluster(&hot_experts).await;
                    }
                }
                ActivationMode::Generalist => {
                    self.handle_generalist_transition();
                }
            }

            // Notify the planner of the mode change
            self.planner.set_mode(self.current_mode);
        }
    }

    /// Pin hot expert cluster in T1 for specialist mode.
    async fn pin_specialist_cluster(&self, hot_experts: &[(u16, u32)]) {
        if hot_experts.is_empty() {
            tracing::warn!("Specialist transition but no hot experts identified");
            return;
        }

        // Build (layer, expert_id) pairs for all MoE layers.
        // In specialist mode, the same experts tend to be hot across layers,
        // so we pin each hot expert at every MoE layer.
        let mut pin_targets: Vec<(u16, u16)> = Vec::new();
        for &(expert_id, _count) in hot_experts {
            for layer in 0..self.model_config.num_moe_layers as u16 {
                let layer_idx = layer + self.model_config.dense_layer_idx as u16;
                pin_targets.push((layer_idx, expert_id));
            }
        }

        tracing::info!(
            "Specialist pinning: {} hot experts x {} MoE layers = {} expert-layer instances",
            hot_experts.len(),
            self.model_config.num_moe_layers,
            pin_targets.len(),
        );

        match self.buffer_mgr.pin_expert_cluster(&pin_targets).await {
            Ok(pinned) => {
                tracing::info!(
                    "Pinned {} pages for specialist mode (budget: {} pages)",
                    pinned,
                    self.buffer_mgr.specialist_pin_count(),
                );
            }
            Err(e) => {
                tracing::error!("Failed to pin expert cluster: {e}");
            }
        }
    }

    /// Transition to Generalist mode: unpin specialist experts.
    fn handle_generalist_transition(&self) {
        let unpinned = self.buffer_mgr.unpin_expert_cluster();
        tracing::info!(
            "Generalist transition: unpinned {} specialist pages",
            unpinned,
        );
    }

    // ── Public streaming API ─────────────────────────────────────────

    /// Tokenize and prefill a prompt (public for streaming API).
    pub async fn prefill_tokens(&mut self, tokens: &[u32]) -> Result<()> {
        self.prefill(tokens).await
    }

    /// Generate a single token (public for streaming API).
    ///
    /// Returns the token ID. The caller is responsible for checking EOS.
    pub async fn generate_one_token(
        &mut self,
        step: usize,
        params: &SamplingParams,
        recent_tokens: &[u32],
    ) -> Result<u32> {
        self.generate_token(step, params, recent_tokens).await
    }

    // ── Accessors ────────────────────────────────────────────────────

    pub fn model_config(&self) -> &ModelConfig {
        &self.model_config
    }

    pub fn stats(&self) -> StatsSnapshot {
        self.stats.snapshot()
    }

    /// Current decode position (number of KV positions written).
    pub fn position(&self) -> usize {
        self.position
    }

    /// Number of decode steps executed since engine initialization.
    pub fn decode_step(&self) -> u64 {
        self.decode_step
    }

    pub fn buffer_mgr(&self) -> &Arc<PageBufferManager> {
        &self.buffer_mgr
    }

    pub fn tokenizer(&self) -> &TokenizerWrapper {
        &self.tokenizer
    }

    /// Get the current activation mode (Generalist or Specialist).
    pub fn current_mode(&self) -> ActivationMode {
        self.current_mode
    }

    /// Get the mode detector (if enabled) for inspection.
    pub fn mode_detector(&self) -> Option<&ActivationModeDetector> {
        self.mode_detector.as_ref()
    }

    /// Whether the engine has a vector index loaded.
    pub fn has_vector_index(&self) -> bool {
        self.vector_index.is_some()
    }

    /// Get the planner (for testing/inspection).
    pub fn planner(&self) -> &QueryPlanner {
        &self.planner
    }

    /// Whether the tiered KV cache is enabled.
    pub fn has_tiered_kv(&self) -> bool {
        self.tiered_kv.is_some()
    }

    /// Get the tiered KV cache (for testing/inspection).
    pub fn tiered_kv(&self) -> Option<&TieredKvCache> {
        self.tiered_kv.as_ref()
    }

    /// Get the unified eviction policy (for testing/inspection).
    pub fn eviction_policy(&self) -> Option<&UnifiedEvictionPolicy> {
        self.eviction_policy.as_ref()
    }

    // ── Shared Tensor Loading ────────────────────────────────────────

    /// Load a complete shared tensor by assembling all pages for (layer, segment).
    ///
    /// Shared tensors (attention projections, norms, embeddings, lm_head) can
    /// span multiple 2 MB pages. This method:
    /// 1. Finds all pages for the (layer, segment) pair
    /// 2. Loads each page via the buffer manager
    /// 3. Reassembles them into a contiguous buffer in row order
    ///
    /// Returns `None` if no pages exist for this (layer, segment).
    /// Results are cached in `shared_tensor_cache` for reuse across tokens.
    async fn load_shared_tensor(&mut self, layer: u16, segment: u16) -> Option<Vec<u8>> {
        // Check cache first
        let cache_key = (layer as u32) << 16 | segment as u32;
        if let Some(cached) = self.shared_tensor_cache.get(&cache_key) {
            return Some(cached.clone());
        }

        let pages = self.model_file.pages_for_shared_segment(layer, segment);
        if pages.is_empty() {
            return None;
        }

        // Compute total tensor size from page metadata
        let total_raw_bytes: usize = pages
            .iter()
            .map(|(_, entry)| { entry.raw_size } as usize)
            .sum();

        if total_raw_bytes == 0 {
            return None;
        }

        let mut tensor_buf = vec![0u8; total_raw_bytes];

        // Load each page and copy into the correct position.
        // Pages are sorted by page_idx. Each page contains a slice of rows
        // starting at row_start. The byte offset = page_data_offset within
        // the tensor (sequentially packed).
        let mut byte_offset = 0usize;
        for (catalog_idx, entry) in &pages {
            let page_id = entry.page_id();
            let raw_size = { entry.raw_size } as usize;

            // Correctness path for lm_head (L0/S11): assemble directly from on-disk
            // page bytes by catalog index to avoid any runtime page-cache aliasing.
            if layer == 0 && segment == 11 {
                let mut page_buf = vec![0u8; raw_size];
                match self.model_file.read_page_sync(*catalog_idx, &mut page_buf) {
                    Ok(n) => {
                        let copy_size = n.min(raw_size).min(tensor_buf.len() - byte_offset);
                        if copy_size > 0 {
                            tensor_buf[byte_offset..byte_offset + copy_size]
                                .copy_from_slice(&page_buf[..copy_size]);
                        }
                    }
                    Err(e) => {
                        tracing::warn!(
                            "Failed to load shared page from disk L{} S{} P{}: {}",
                            layer,
                            segment,
                            page_id.page_idx,
                            e
                        );
                    }
                }
                byte_offset += raw_size;
                continue;
            }

            match self.buffer_mgr.get_page(&page_id).await {
                Ok(handle) => {
                    let copy_size = raw_size.min(tensor_buf.len() - byte_offset);
                    if copy_size > 0 && !handle.device_ptr.is_null() {
                        // Use memcpy_d2h for device-to-host copy (handles both
                        // VRAM pointers when CUDA is active and host pointers
                        // in CPU fallback mode).
                        if let Err(e) = cuda_ffi::memcpy_d2h(
                            tensor_buf[byte_offset..].as_mut_ptr(),
                            handle.device_ptr as *const u8,
                            copy_size,
                        ) {
                            tracing::warn!(
                                "memcpy_d2h failed for shared tensor L{} S{}: {}, using direct copy",
                                layer, segment, e
                            );
                            // Fallback: try direct copy (works if pointer is host-accessible)
                            unsafe {
                                std::ptr::copy_nonoverlapping(
                                    handle.device_ptr as *const u8,
                                    tensor_buf[byte_offset..].as_mut_ptr(),
                                    copy_size,
                                );
                            }
                        }
                    }
                }
                Err(e) => {
                    // Try direct read from mmap as fallback
                    let mut page_buf = vec![0u8; raw_size];
                    match self.model_file.read_page_sync(*catalog_idx, &mut page_buf) {
                        Ok(n) => {
                            let copy_size = n.min(tensor_buf.len() - byte_offset);
                            tensor_buf[byte_offset..byte_offset + copy_size]
                                .copy_from_slice(&page_buf[..copy_size]);
                        }
                        Err(_) => {
                            tracing::warn!(
                                "Failed to load shared page L{} S{} P{}: {}",
                                layer,
                                segment,
                                page_id.page_idx,
                                e
                            );
                        }
                    }
                }
            }
            byte_offset += raw_size;
        }

        // NOTE: The converter already converts BF16 shared tensors to FP16
        // during conversion (via kernels::convert_bf16_to_fp16). No runtime
        // conversion is needed. Older .vib3 files may still report shared_dtype
        // as BF16 in metadata, but the actual on-disk data is FP16.

        // Cache the assembled tensor
        self.shared_tensor_cache
            .insert(cache_key, tensor_buf.clone());
        Some(tensor_buf)
    }

    /// Pre-assemble all shared weight tensors into permanent VRAM buffers.
    ///
    /// Called once after model preload when all pages are resident in T1.
    /// For single-page tensors (norms, biases), stores the T1 page pointer
    /// directly (zero-copy). For multi-page tensors (projections), allocates
    /// a contiguous DeviceBuffer and does a one-time D2D assembly.
    ///
    /// After this, `get_preassembled_weight()` provides O(1) pointer lookup
    /// with zero D2D copies per token — eliminating ~10.6 GB/token of bandwidth waste.
    pub fn preassemble_all_shared_weights(&mut self) -> usize {
        if !self.buffer_mgr.is_fully_resident() {
            tracing::warn!("Cannot preassemble: model not fully resident in T1");
            return 0;
        }

        let num_layers = self.model_config.num_layers;
        let has_deltanet = self.model_config.deltanet.is_some();

        // Query current free VRAM and set a budget with a safety reserve for:
        //  - Dynamic cache for non-preassembled segments (~1 GB: router, moe_norm, shared expert)
        //  - Global tensors (embeddings + lm_head ≈ 3 GB, allocated on first token)
        //  - Runtime temp buffers, KV cache growth (~0.5 GB)
        // NOTE: If dynamic cache OOMs, it falls back to host path — functional but slow.
        // Prioritize preassembly (eliminates per-token D2D) over dynamic cache headroom.
        let free_vram = cuda_ffi::query_free_vram();
        let vram_reserve = 4400 * 1024 * 1024; // 4.3 GB for dynamic cache + runtime
        let mut vram_budget = if free_vram > vram_reserve {
            free_vram - vram_reserve
        } else {
            0
        };
        tracing::info!(
            "Preassembly VRAM budget: {:.1} MB (free={:.1} MB, reserve={:.1} MB)",
            vram_budget as f64 / (1024.0 * 1024.0),
            free_vram as f64 / (1024.0 * 1024.0),
            vram_reserve as f64 / (1024.0 * 1024.0),
        );

        let shared_nvfp4_enabled = std::env::var("VIB3_ENABLE_SHARED_NVFP4_PREASSEMBLY")
            .map_or(false, |v| v == "1");
        let is_qwen35 = self.model_config.architecture == "qwen3_5_moe";

        // Only preassemble segments that are loaded per-token via load_tensor_direct_resident
        // (the source of ~31K D2D copies per run). Segments handled by the dynamic cache
        // (3=router, 7=moe_norm, 14/15/16=shared expert, 26=gate) are left to
        // ensure_shared_tensor_device which allocates once and caches permanently.
        //
        // Segment 6 (attn_norm) is single-page zero-copy on every layer.
        let all_layer_segments: &[u16] = &[6];
        // GQA attention projections: loaded per-token into staging buffers
        let gqa_segments: &[u16] = if is_qwen35 && !shared_nvfp4_enabled {
            &[]
        } else {
            &[4, 5, 12, 13, 27, 28]
        };
        // DeltaNet projections: loaded per-token into staging buffers
        let dn_segments: &[u16] = if is_qwen35 && !shared_nvfp4_enabled {
            &[]
        } else {
            &[30, 31, 32, 33, 34, 35, 36, 37, 38]
        };
        if is_qwen35 && !shared_nvfp4_enabled {
            tracing::warn!(
                "Qwen3.5 default preassembly mode: skipping large projection preassembly (FP16 shared path) to preserve runtime headroom and correctness"
            );
        }

        let mut assembled_count = 0usize;
        let mut total_bytes = 0usize;
        let mut single_page_count = 0usize;
        let mut multi_page_count = 0usize;
        let mut skipped_budget = 0usize;

        // Helper: try to preassemble a (layer, segment) pair.
        // Returns (assembled: bool, bytes_used: usize).
        // Single-page = zero-copy (no VRAM), multi-page = allocate + D2D.
        let try_preassemble = |pre_assembled: &mut std::collections::HashMap<u32, PreAssembledWeight>,
                                    model_file: &crate::storage::format::Vib3File,
                                    buffer_mgr: &crate::storage::buffer_manager::PageBufferManager,
                                    layer_key: u16, segment: u16,
                                    budget: &mut usize|
         -> (bool, usize) {
            let cache_key = (layer_key as u32) << 16 | segment as u32;
            if pre_assembled.contains_key(&cache_key) {
                return (false, 0);
            }

            let pages = model_file.pages_for_shared_segment(layer_key, segment);
            if pages.is_empty() {
                return (false, 0);
            }

            let total_raw_bytes: usize = pages.iter().map(|(_, e)| e.raw_size as usize).sum();
            if total_raw_bytes == 0 {
                return (false, 0);
            }

            if pages.len() == 1 {
                // Single-page tensor: store T1 page pointer directly (zero-copy)
                let (_catalog_idx, entry) = &pages[0];
                let page_id = entry.page_id();
                if let Some(handle) = buffer_mgr.get_page_resident(&page_id) {
                    if !handle.device_ptr.is_null() {
                        pre_assembled.insert(
                            cache_key,
                            PreAssembledWeight::SinglePage {
                                ptr: handle.device_ptr as *const u8,
                                size: entry.raw_size as usize,
                            },
                        );
                        return (true, 0); // No extra VRAM
                    }
                }
                (false, 0)
            } else {
                // Multi-page tensor: check VRAM budget before allocating
                if total_raw_bytes > *budget {
                    return (false, 0); // Not enough budget
                }

                match DeviceBuffer::new(total_raw_bytes) {
                    Ok(dbuf) => {
                        let mut byte_offset = 0usize;
                        let mut ok = true;
                        for (_catalog_idx, entry) in &pages {
                            let page_id = entry.page_id();
                            let raw_size = entry.raw_size as usize;
                            if let Some(handle) = buffer_mgr.get_page_resident(&page_id) {
                                if !handle.device_ptr.is_null() {
                                    let copy_size = raw_size.min(total_raw_bytes - byte_offset);
                                    if cuda_ffi::device_memcpy_d2d(
                                        unsafe { dbuf.as_mut_ptr().add(byte_offset) },
                                        handle.device_ptr as *const u8,
                                        copy_size,
                                    ).is_err() {
                                        ok = false;
                                        break;
                                    }
                                } else {
                                    ok = false;
                                    break;
                                }
                            } else {
                                ok = false;
                                break;
                            }
                            byte_offset += raw_size;
                        }
                        if ok {
                            *budget -= total_raw_bytes;
                            pre_assembled.insert(
                                cache_key,
                                PreAssembledWeight::Assembled(dbuf),
                            );
                            (true, total_raw_bytes)
                        } else {
                            (false, 0)
                        }
                    }
                    Err(_) => (false, 0),
                }
            }
        };

        for layer in 0..num_layers {
            let is_attention = if has_deltanet {
                let dn_config = self.model_config.deltanet.as_ref().unwrap();
                dn_config.layer_is_attention[layer as usize]
            } else {
                true
            };

            // Always preassemble single-page norm tensors (zero-copy, no VRAM)
            for &segment in all_layer_segments {
                let (ok, bytes) = try_preassemble(
                    &mut self.pre_assembled_weights,
                    &self.model_file,
                    &self.buffer_mgr,
                    layer as u16, segment,
                    &mut vram_budget,
                );
                if ok {
                    assembled_count += 1;
                    if bytes > 0 {
                        multi_page_count += 1;
                        total_bytes += bytes;
                    } else {
                        single_page_count += 1;
                    }
                }
            }

            // For GQA/DeltaNet projection segments, preassembly must be atomic:
            // either ALL multi-page segments for this layer fit in budget, or NONE.
            // This prevents the has_preassembled guard from skipping staging loads
            // when only some segments were preassembled.
            let projection_segments: &[u16] = if is_attention {
                gqa_segments
            } else if has_deltanet {
                dn_segments
            } else {
                &[]
            };

            if !projection_segments.is_empty() {
                // Compute total multi-page VRAM needed for this layer
                let mut layer_multi_page_bytes = 0usize;
                for &segment in projection_segments {
                    let cache_key = (layer as u32) << 16 | segment as u32;
                    if self.pre_assembled_weights.contains_key(&cache_key) {
                        continue;
                    }
                    let pages = self.model_file.pages_for_shared_segment(layer as u16, segment);
                    if pages.len() > 1 {
                        let raw_bytes: usize = pages.iter().map(|(_, e)| e.raw_size as usize).sum();
                        layer_multi_page_bytes += raw_bytes;
                    }
                }

                if layer_multi_page_bytes <= vram_budget {
                    // Budget allows: preassemble all segments for this layer
                    for &segment in projection_segments {
                        let (ok, bytes) = try_preassemble(
                            &mut self.pre_assembled_weights,
                            &self.model_file,
                            &self.buffer_mgr,
                            layer as u16, segment,
                            &mut vram_budget,
                        );
                        if ok {
                            assembled_count += 1;
                            if bytes > 0 {
                                multi_page_count += 1;
                                total_bytes += bytes;
                            } else {
                                single_page_count += 1;
                            }
                        }
                    }
                } else {
                    // Not enough budget for this layer — skip all projection segments
                    for &segment in projection_segments {
                        let pages = self.model_file.pages_for_shared_segment(layer as u16, segment);
                        if pages.len() > 1 {
                            skipped_budget += 1;
                        }
                    }
                }
            }
        }

        // Global tensors (final norm, embeddings, lm_head) are handled by the
        // dynamic cache via ensure_shared_tensor_device — low frequency, not worth
        // the VRAM budget.

        tracing::info!(
            "Pre-assembled {} shared weight tensors ({} single-page zero-copy, {} multi-page = {:.1} MB extra VRAM, {} skipped for budget, {:.1} MB budget remaining)",
            assembled_count,
            single_page_count,
            multi_page_count,
            total_bytes as f64 / (1024.0 * 1024.0),
            skipped_budget,
            vram_budget as f64 / (1024.0 * 1024.0),
        );

        // ── NVFP4 conversion pass ──
        // Convert multi-page FP16 preassembled weights to NVFP4 format for ~4x bandwidth savings.
        // Single-page weights are left as FP16 (small tensors, negligible bandwidth).
        // Conversion: FP16 → split-half FP4 + E8M0 BF16 scales (MMA-ready).
        let hidden_dim = self.model_config.hidden_dim as usize;
        let dn_inner = self.model_config.deltanet.as_ref().map_or(0, |d| d.inner_dim as usize);
        let dn_qkv = self.model_config.deltanet.as_ref().map_or(0, |d| d.qkv_dim() as usize);
        let num_heads = self.model_config.num_heads as usize;
        let head_dim = self.model_config.effective_head_dim() as usize;
        let num_kv_heads = self.model_config.num_kv_heads as usize;
        let is_gated_attn = self.model_config.deltanet.is_some();
        let q_out_dim = if is_gated_attn { num_heads * 2 * head_dim } else { hidden_dim };
        let kv_dim = num_kv_heads * head_dim;
        let attn_out_dim = num_heads * head_dim;

        // Map segment → (M, K) for weight tensor [M, K] FP16
        let segment_dims = |layer: u16, segment: u16| -> Option<(usize, usize)> {
            let is_attn = if has_deltanet {
                self.model_config.deltanet.as_ref().unwrap().layer_is_attention[layer as usize]
            } else {
                true
            };
            match segment {
                // DeltaNet segments
                30 if !is_attn => Some((dn_qkv, hidden_dim)),    // QKV: [12288, 3072]
                31 if !is_attn => Some((dn_inner, hidden_dim)),   // Z: [8192, 3072]
                38 if !is_attn => Some((hidden_dim, dn_inner)),   // Out: [3072, 8192]
                // GQA segments
                4 if is_attn => Some((q_out_dim, hidden_dim)),    // Q: [16384, 3072]
                5 if is_attn => Some((hidden_dim, attn_out_dim)), // O: [3072, 8192]
                12 if is_attn => Some((kv_dim, hidden_dim)),      // K: [1024, 3072]
                13 if is_attn => Some((kv_dim, hidden_dim)),      // V: [1024, 3072]
                // GQA segments 27, 28 are norm weights [head_dim] — NOT projection matrices
                // They are single-page and should not be NVFP4-converted.
                _ => None,
            }
        };

        if is_qwen35 && !shared_nvfp4_enabled {
            tracing::warn!(
                "Skipping shared-weight NVFP4 preassembly conversion for qwen3_5_moe; using FP16 shared projections for correctness. Set VIB3_ENABLE_SHARED_NVFP4_PREASSEMBLY=1 to override"
            );
            return assembled_count;
        }

        let mut nvfp4_count = 0usize;
        let mut nvfp4_freed = 0usize;
        let mut nvfp4_allocated = 0usize;

        // Collect keys that need conversion (can't mutate while iterating)
        let keys_to_convert: Vec<u32> = self.pre_assembled_weights.keys()
            .filter(|&&key| {
                matches!(self.pre_assembled_weights.get(&key), Some(PreAssembledWeight::Assembled(_)))
            })
            .copied()
            .collect();

        for key in keys_to_convert {
            let layer = (key >> 16) as u16;
            let segment = (key & 0xFFFF) as u16;

            if let Some((m, k)) = segment_dims(layer, segment) {
                // Verify K is divisible by 64 (MMA tile requirement)
                if k % 64 != 0 || m == 0 {
                    continue;
                }
                // Verify the FP16 buffer size matches expected dimensions
                let expected_fp16_bytes = m * k * 2;
                let current_size = self.pre_assembled_weights.get(&key).map(|w| w.size()).unwrap_or(0);
                if current_size != expected_fp16_bytes {
                    tracing::warn!(
                        "NVFP4 skip: layer={} seg={} size={} expected={} (M={} K={})",
                        layer, segment, current_size, expected_fp16_bytes, m, k
                    );
                    continue;
                }

                // Convert: take the Assembled buffer, convert to NVFP4, replace
                if let Some(PreAssembledWeight::Assembled(fp16_buf)) = self.pre_assembled_weights.remove(&key) {
                    let fp16_size = fp16_buf.size();
                    match kernels::fp16_to_nvfp4_weight(fp16_buf.as_ptr(), m, k, &self.stream) {
                        Ok(nvfp4_buf) => {
                            let nvfp4_size = nvfp4_buf.size();
                            // Sync before dropping FP16 buffer (conversion runs on stream)
                            let _ = self.stream.synchronize();
                            if let Err(e) = self.maybe_audit_nvfp4_weight_pair(
                                layer,
                                segment,
                                m,
                                k,
                                &fp16_buf,
                                &nvfp4_buf,
                            ) {
                                tracing::warn!(
                                    "NVFP4_WEIGHT_AUDIT failed L{} seg{}: {}",
                                    layer,
                                    segment,
                                    e
                                );
                            }
                            nvfp4_freed += fp16_size;
                            nvfp4_allocated += nvfp4_size;
                            // Drop FP16 buffer (frees VRAM)
                            drop(fp16_buf);
                            self.pre_assembled_weights.insert(key, PreAssembledWeight::Nvfp4 {
                                buf: nvfp4_buf,
                                m,
                                k,
                            });
                            nvfp4_count += 1;
                        }
                        Err(e) => {
                            tracing::warn!("NVFP4 conversion failed for layer={} seg={}: {}", layer, segment, e);
                            // Put the FP16 buffer back
                            self.pre_assembled_weights.insert(key, PreAssembledWeight::Assembled(fp16_buf));
                        }
                    }
                }
            }
        }

        if nvfp4_count > 0 {
            tracing::info!(
                "NVFP4 converted {} weight tensors (freed {:.1} MB FP16, allocated {:.1} MB NVFP4, net savings {:.1} MB)",
                nvfp4_count,
                nvfp4_freed as f64 / (1024.0 * 1024.0),
                nvfp4_allocated as f64 / (1024.0 * 1024.0),
                (nvfp4_freed as f64 - nvfp4_allocated as f64) / (1024.0 * 1024.0),
            );

            // With freed VRAM, iteratively preassemble+convert until no more progress
            if skipped_budget > 0 {
                let mut pass = 1u32;
                loop {
                    let new_free = cuda_ffi::query_free_vram();
                    let mut new_budget = if new_free > vram_reserve { new_free - vram_reserve } else { 0 };
                    let mut extra_assembled = 0usize;
                    let mut extra_bytes = 0usize;

                    for layer in 0..num_layers {
                        let is_attention = if has_deltanet {
                            self.model_config.deltanet.as_ref().unwrap().layer_is_attention[layer as usize]
                        } else {
                            true
                        };
                        let projection_segments: &[u16] = if is_attention {
                            gqa_segments
                        } else if has_deltanet {
                            dn_segments
                        } else {
                            &[]
                        };

                        // Check if first projection segment for this layer is already assembled
                        let first_proj = projection_segments.iter()
                            .find(|&&s| s != 6)
                            .copied()
                            .unwrap_or(0);
                        let key_check = (layer as u32) << 16 | first_proj as u32;
                        if self.pre_assembled_weights.contains_key(&key_check) {
                            continue;
                        }

                        let mut layer_bytes = 0usize;
                        for &segment in projection_segments {
                            let pages = self.model_file.pages_for_shared_segment(layer as u16, segment);
                            if pages.len() > 1 {
                                let raw_bytes: usize = pages.iter().map(|(_, e)| e.raw_size as usize).sum();
                                layer_bytes += raw_bytes;
                            }
                        }

                        if layer_bytes <= new_budget {
                            for &segment in projection_segments {
                                let (ok, bytes) = try_preassemble(
                                    &mut self.pre_assembled_weights,
                                    &self.model_file,
                                    &self.buffer_mgr,
                                    layer as u16, segment,
                                    &mut new_budget,
                                );
                                if ok && bytes > 0 {
                                    extra_assembled += 1;
                                    extra_bytes += bytes;
                                }
                            }
                        }
                    }

                    if extra_assembled == 0 {
                        break; // No more progress possible
                    }

                    // Convert newly assembled FP16 weights to NVFP4
                    let new_keys: Vec<u32> = self.pre_assembled_weights.keys()
                        .filter(|&&key| {
                            matches!(self.pre_assembled_weights.get(&key), Some(PreAssembledWeight::Assembled(_)))
                        })
                        .copied()
                        .collect();

                    for key in new_keys {
                        let layer = (key >> 16) as u16;
                        let segment = (key & 0xFFFF) as u16;
                        if let Some((m, k)) = segment_dims(layer, segment) {
                            if k % 64 != 0 || m == 0 { continue; }
                            let expected = m * k * 2;
                            let current = self.pre_assembled_weights.get(&key).map(|w| w.size()).unwrap_or(0);
                            if current != expected { continue; }

                            if let Some(PreAssembledWeight::Assembled(fp16_buf)) = self.pre_assembled_weights.remove(&key) {
                                let fp16_size = fp16_buf.size();
                                match kernels::fp16_to_nvfp4_weight(fp16_buf.as_ptr(), m, k, &self.stream) {
                                    Ok(nvfp4_buf) => {
                                        let _ = self.stream.synchronize();
                                        if let Err(e) = self.maybe_audit_nvfp4_weight_pair(
                                            layer,
                                            segment,
                                            m,
                                            k,
                                            &fp16_buf,
                                            &nvfp4_buf,
                                        ) {
                                            tracing::warn!(
                                                "NVFP4_WEIGHT_AUDIT failed L{} seg{}: {}",
                                                layer,
                                                segment,
                                                e
                                            );
                                        }
                                        nvfp4_freed += fp16_size;
                                        nvfp4_allocated += nvfp4_buf.size();
                                        drop(fp16_buf);
                                        self.pre_assembled_weights.insert(key, PreAssembledWeight::Nvfp4 {
                                            buf: nvfp4_buf, m, k,
                                        });
                                        nvfp4_count += 1;
                                    }
                                    Err(_) => {
                                        self.pre_assembled_weights.insert(key, PreAssembledWeight::Assembled(fp16_buf));
                                    }
                                }
                            }
                        }
                    }

                    tracing::info!(
                        "Pass {}: assembled {} more tensors ({:.1} MB), total NVFP4: {}, net savings {:.1} MB",
                        pass, extra_assembled,
                        extra_bytes as f64 / (1024.0 * 1024.0),
                        nvfp4_count,
                        (nvfp4_freed as f64 - nvfp4_allocated as f64) / (1024.0 * 1024.0),
                    );
                    pass += 1;
                    if pass > 20 { break; } // Safety limit
                }
            }
        }

        // ── Tiled repack pass for NVFP4 shared weights ──
        // Repack FP4 data from row-major split-half to tiled layout for coalesced GEMV access.
        // Tiled layout: 16 rows × 32 bytes per K-tile contiguous (512 bytes per tile).
        // This eliminates scattered row-major access that wastes ~75% of cache line bandwidth.
        if nvfp4_count > 0 {
            let tile_start = Instant::now();
            // Find max FP4 data size across all NVFP4 weights for temp buffer
            let max_fp4_data: usize = self.pre_assembled_weights.values()
                .filter_map(|w| w.nvfp4_dims().map(|(m, k)| m * k.div_ceil(2)))
                .max()
                .unwrap_or(0);

            if max_fp4_data > 0 {
                match DeviceBuffer::new(max_fp4_data) {
                    Ok(temp_buf) => {
                        let mut tiled_count = 0usize;
                        for w in self.pre_assembled_weights.values_mut() {
                            if let PreAssembledWeight::Nvfp4 { buf, m, k } = w {
                                if *k % 64 == 0 && *m % 16 == 0 {
                                    if let Err(e) = kernels::repack_row_to_tiled(
                                        buf.as_mut_ptr(),
                                        temp_buf.as_mut_ptr(),
                                        *m,
                                        *k,
                                        &self.stream,
                                    ) {
                                        tracing::warn!("Tiled repack failed for M={} K={}: {}", m, k, e);
                                    } else {
                                        tiled_count += 1;
                                    }
                                }
                            }
                        }
                        let _ = self.stream.synchronize();
                        tracing::info!(
                            "Tiled repack: {} NVFP4 shared weights repacked in {:.1}ms (temp buf {:.1} MB)",
                            tiled_count,
                            tile_start.elapsed().as_millis(),
                            max_fp4_data as f64 / (1024.0 * 1024.0),
                        );
                    }
                    Err(e) => {
                        tracing::warn!("Failed to allocate tiled repack temp buffer ({:.1} MB): {}", max_fp4_data as f64 / (1024.0*1024.0), e);
                    }
                }
            }
        }

        assembled_count
    }

    /// Look up a pre-assembled weight tensor by (layer, segment).
    ///
    /// Returns `(device_ptr, size_bytes)` or None if not pre-assembled.
    /// This is O(1) with zero D2D copies — the hot path replacement for
    /// `load_tensor_direct()` + `get_device_tensor()`.
    #[inline]
    fn get_preassembled_weight(&self, layer: u16, segment: u16) -> Option<(*const u8, usize)> {
        let cache_key = (layer as u32) << 16 | segment as u32;
        self.pre_assembled_weights
            .get(&cache_key)
            .and_then(|w| {
                // Don't return NVFP4 weights via this path — callers expect FP16.
                // Use get_preassembled_nvfp4() for NVFP4 weights.
                if w.is_nvfp4() { None } else { Some((w.ptr(), w.size())) }
            })
    }

    /// Look up a pre-assembled NVFP4 weight tensor by (layer, segment).
    /// Returns (device_ptr, M, K) if the weight was converted to NVFP4 format.
    /// The buffer layout is [FP4 data (M*K/2 bytes) | BF16 scales (M*(K/32)*2 bytes)].
    #[inline]
    fn get_preassembled_nvfp4(&self, layer: u16, segment: u16) -> Option<(*const u8, usize, usize)> {
        let cache_key = (layer as u32) << 16 | segment as u32;
        self.pre_assembled_weights.get(&cache_key).and_then(|w| {
            w.nvfp4_dims().map(|(m, k)| (w.ptr(), m, k))
        })
    }

    /// Ensure a shared tensor is loaded into device memory (VRAM) and cached.
    ///
    /// After calling this, use `get_device_tensor(layer, segment)` to get the
    /// device pointer. The two-step API avoids holding raw pointers across
    /// await boundaries (which would make the async future non-Send).
    ///
    /// When T1 pages are already in VRAM (GPU mode with preload), this
    /// assembles the tensor directly on device via D2D copies — no host
    /// roundtrip. Falls back to host-mediated upload otherwise.
    ///
    /// Returns true if the tensor is available on device.
    async fn ensure_shared_tensor_device(&mut self, layer: u16, segment: u16) -> bool {
        let cache_key = (layer as u32) << 16 | segment as u32;

        // Check pre-assembled weights first (zero-copy, populated at init)
        if self.pre_assembled_weights.contains_key(&cache_key) {
            return true;
        }

        // Check device cache
        if self.shared_tensor_cache_device.contains_key(&cache_key) {
            return true;
        }

        // Try to assemble directly on device from T1 pages (avoids VRAM→host→VRAM roundtrip).
        // Correctness guard: for lm_head (L0/S11), prefer host-mediated assembly by default.
        // Diagnostic evidence shows direct device assembly can produce row mismatches for
        // some pages; host assembly matches on-disk bytes and paged reference.
        let allow_device_assemble_lm_head = std::env::var("VIB3_ALLOW_DEVICE_ASSEMBLE_LM_HEAD")
            .map_or(false, |v| v == "1");
        let skip_device_assemble = layer == 0 && segment == 11 && !allow_device_assemble_lm_head;
        if !skip_device_assemble {
            if self.assemble_tensor_on_device(layer, segment).await {
                return true;
            }
        } else if self.diag_enabled {
            tracing::info!(
                "LMHEAD_GUARD: skipping device assembly for shared tensor L0 S11 (set VIB3_ALLOW_DEVICE_ASSEMBLE_LM_HEAD=1 to override)"
            );
        }

        // Fallback: load to host first, then upload to device
        let host_data = match self.load_shared_tensor(layer, segment).await {
            Some(data) => data,
            None => return false,
        };

        let required_bytes = host_data.len();
        let free_vram_bytes = cuda_ffi::query_free_vram();
        if free_vram_bytes > 0 && free_vram_bytes < required_bytes {
            tracing::warn!(
                "Insufficient free VRAM before shared tensor alloc L{} S{}: need={} MiB, free={} MiB",
                layer,
                segment,
                required_bytes as f64 / (1024.0 * 1024.0),
                free_vram_bytes as f64 / (1024.0 * 1024.0),
            );
        }

        match DeviceBuffer::new(host_data.len()) {
            Ok(dbuf) => {
                if let Err(e) = dbuf.copy_from_host(&host_data) {
                    tracing::warn!(
                        "Failed to upload shared tensor L{} S{} to device: {}",
                        layer,
                        segment,
                        e
                    );
                    return false;
                }
                self.shared_tensor_cache_device.insert(cache_key, dbuf);
                true
            }
            Err(e) => {
                tracing::warn!(
                    "Failed to allocate device buffer for shared tensor L{} S{} (need={} MiB, free={} MiB): {}",
                    layer,
                    segment,
                    required_bytes as f64 / (1024.0 * 1024.0),
                    free_vram_bytes as f64 / (1024.0 * 1024.0),
                    e
                );
                false
            }
        }
    }

    /// Assemble a shared tensor directly on device via D2D copies from T1 pages.
    ///
    /// This is the fast path when all pages are already in VRAM (after preload).
    /// Returns false if any page is missing from T1.
    async fn assemble_tensor_on_device(&mut self, layer: u16, segment: u16) -> bool {
        let cache_key = (layer as u32) << 16 | segment as u32;

        let pages = self.model_file.pages_for_shared_segment(layer, segment);
        if pages.is_empty() {
            return false;
        }

        // Compute total tensor size
        let total_raw_bytes: usize = pages.iter().map(|(_, e)| e.raw_size as usize).sum();
        if total_raw_bytes == 0 {
            return false;
        }

        // Allocate device buffer for the assembled tensor
        let dbuf = match DeviceBuffer::new(total_raw_bytes) {
            Ok(b) => b,
            Err(_) => return false,
        };

        // Copy each page's data from T1 (VRAM) to the correct offset in dbuf
        let mut byte_offset = 0usize;
        for (_catalog_idx, entry) in &pages {
            let page_id = entry.page_id();
            let raw_size = entry.raw_size as usize;

            match self.buffer_mgr.get_page(&page_id).await {
                Ok(handle) if !handle.device_ptr.is_null() => {
                    let copy_size = raw_size.min(total_raw_bytes - byte_offset);
                    // D2D copy: T1 page slot → tensor buffer (both in VRAM)
                    // SAFETY: byte_offset is within dbuf bounds, checked above
                    if cuda_ffi::device_memcpy_d2d(
                        unsafe { dbuf.as_mut_ptr().add(byte_offset) },
                        handle.device_ptr as *const u8,
                        copy_size,
                    )
                    .is_err()
                    {
                        return false;
                    }
                }
                _ => return false, // Page not in T1, fall back to host path
            }
            byte_offset += raw_size;
        }

        self.shared_tensor_cache_device.insert(cache_key, dbuf);
        true
    }

    /// Get a device tensor's pointer and size (synchronous, no await).
    ///
    /// Priority: pre-assembled weights → GQA staging buffers → dynamic cache.
    fn get_device_tensor(&self, _layer: u16, segment: u16) -> Option<(*const u8, usize)> {
        // Highest priority: pre-assembled weights (zero-copy, populated at init)
        if let Some(result) = self.get_preassembled_weight(_layer, segment) {
            return Some(result);
        }
        // Next: dynamic cache populated by ensure_shared_tensor_device()
        let cache_key = (_layer as u32) << 16 | segment as u32;
        if let Some(dbuf) = self.shared_tensor_cache_device.get(&cache_key) {
            return Some((dbuf.as_ptr(), dbuf.size()));
        }
        // GQA attention weight segments use pre-allocated buffers
        // (only for non-MLA models to avoid dimension conflicts)
        if self.model_config.mla.is_none() {
            match segment {
                4 if self.gqa_w_q.size() > 0 => return Some((self.gqa_w_q.as_ptr(), self.gqa_w_q.size())),
                5 if self.gqa_w_o.size() > 0 => return Some((self.gqa_w_o.as_ptr(), self.gqa_w_o.size())),
                6 if self.attn_norm_w.size() > 0 => return Some((self.attn_norm_w.as_ptr(), self.attn_norm_w.size())),
                12 if self.gqa_w_k.size() > 0 => return Some((self.gqa_w_k.as_ptr(), self.gqa_w_k.size())),
                13 if self.gqa_w_v.size() > 0 => return Some((self.gqa_w_v.as_ptr(), self.gqa_w_v.size())),
                27 if self.gqa_w_qnorm.size() > 0 => return Some((self.gqa_w_qnorm.as_ptr(), self.gqa_w_qnorm.size())),
                28 if self.gqa_w_knorm.size() > 0 => return Some((self.gqa_w_knorm.as_ptr(), self.gqa_w_knorm.size())),
                _ => {}
            }
        }
        None
    }

    /// Load a shared tensor directly into a pre-allocated device pointer via D2D copies.
    ///
    /// Uses async D2D copies on the compute stream to avoid blocking the CPU.
    /// The copies are ordered on the stream, so subsequent kernel launches on the
    /// same stream see the data. This bypasses the shared_tensor_cache_device entirely, eliminating
    /// cudaMalloc/cudaFree overhead. The destination must be a valid VRAM pointer
    /// with at least `dst_capacity` bytes. Returns the number of bytes copied, or 0 on failure.
    #[allow(dead_code)]
    async fn load_tensor_direct(&mut self, layer: u16, segment: u16, dst_ptr: usize, dst_capacity: usize) -> usize {
        let pages = self.model_file.pages_for_shared_segment(layer, segment);
        if pages.is_empty() {
            return 0;
        }

        let total_raw_bytes: usize = pages.iter().map(|(_, e)| e.raw_size as usize).sum();
        if total_raw_bytes == 0 || total_raw_bytes > dst_capacity {
            return 0;
        }

        let mut byte_offset = 0usize;
        for (_catalog_idx, entry) in &pages {
            let page_id = entry.page_id();
            let raw_size = entry.raw_size as usize;

            match self.buffer_mgr.get_page(&page_id).await {
                Ok(handle) if !handle.device_ptr.is_null() => {
                    let copy_size = raw_size.min(total_raw_bytes - byte_offset);
                    // Async D2D on the compute stream — CPU doesn't block.
                    // Stream ordering guarantees subsequent kernels see the data.
                    if cuda_ffi::device_memcpy_d2d_async(
                        (dst_ptr + byte_offset) as *mut u8,
                        handle.device_ptr as *const u8,
                        copy_size,
                        &self.stream,
                    )
                    .is_err()
                    {
                        return 0;
                    }
                }
                _ => return 0,
            }
            byte_offset += raw_size;
        }

        total_raw_bytes
    }

    /// Fast synchronous D2D weight staging using the frozen resident snapshot.
    ///
    /// Like `load_tensor_direct` but uses `get_page_resident()` (zero-lock HashMap
    /// lookup) instead of `get_page().await` (DashMap + Mutex + stats overhead).
    /// Only works when model is fully resident (after preload). Returns 0 on failure.
    ///
    /// This eliminates ALL async overhead from the D2D weight staging hot path.
    fn load_tensor_direct_resident(
        &self,
        layer: u16,
        segment: u16,
        dst_ptr: usize,
        dst_capacity: usize,
    ) -> usize {
        let pages = self.model_file.pages_for_shared_segment(layer, segment);
        if pages.is_empty() {
            return 0;
        }

        let total_raw_bytes: usize = pages.iter().map(|(_, e)| e.raw_size as usize).sum();
        if total_raw_bytes == 0 || total_raw_bytes > dst_capacity {
            return 0;
        }

        let mut byte_offset = 0usize;
        for (_catalog_idx, entry) in &pages {
            let page_id = entry.page_id();
            let raw_size = entry.raw_size as usize;

            if let Some(handle) = self.buffer_mgr.get_page_resident(&page_id) {
                if !handle.device_ptr.is_null() {
                    let copy_size = raw_size.min(total_raw_bytes - byte_offset);
                    if cuda_ffi::device_memcpy_d2d_async(
                        (dst_ptr + byte_offset) as *mut u8,
                        handle.device_ptr as *const u8,
                        copy_size,
                        &self.stream,
                    )
                    .is_err()
                    {
                        return 0;
                    }
                } else {
                    return 0;
                }
            } else {
                return 0;
            }
            byte_offset += raw_size;
        }

        total_raw_bytes
    }
}

impl Engine {
    /// Diagnostic helper: compare selected lm_head rows from contiguous shared tensor
    /// against the same rows fetched directly from paged source.
    ///
    /// Used to isolate whether logits divergence is caused by weight source/layout
    /// mismatch (shared tensor assembly) versus GEMV kernel math.
    async fn diagnose_lm_head_weight_row_parity(
        &self,
        hidden_dim: usize,
        vocab_size: usize,
        lm_head_ptr: usize,
        rows: &[usize],
    ) -> Result<usize> {
        let bytes_per_row = hidden_dim * std::mem::size_of::<f16>();
        let page_size = 2 * 1024 * 1024usize;
        let rows_per_page = page_size / bytes_per_row;
        if rows_per_page == 0 {
            return Err(Error::Cuda(format!(
                "LMHEAD_WEIGHT_PARITY invalid geometry: hidden_dim={} bytes_per_row={} page_size={}",
                hidden_dim, bytes_per_row, page_size
            )));
        }

        let pages = self.model_file.pages_for_shared_segment(0, 11);
        if pages.is_empty() {
            return Err(Error::Cuda(
                "LMHEAD_WEIGHT_PARITY: lm_head pages missing".to_string(),
            ));
        }

        let mut mismatch_rows = 0usize;

        for &row in rows {
            if row >= vocab_size {
                continue;
            }

            let mut shared_row = vec![0u8; bytes_per_row];
            let mut paged_row = vec![0u8; bytes_per_row];

            let row_offset = row * bytes_per_row;
            cuda_ffi::memcpy_d2h(
                shared_row.as_mut_ptr(),
                (lm_head_ptr + row_offset) as *const u8,
                bytes_per_row,
            )?;

            let page_seq = row / rows_per_page;
            let row_in_page = row % rows_per_page;
            if page_seq >= pages.len() {
                continue;
            }

            let (catalog_idx, entry) = &pages[page_seq];
            let page_id = entry.page_id();
            let handle = self.buffer_mgr.get_page(&page_id).await.map_err(|e| {
                Error::Cuda(format!(
                    "LMHEAD_WEIGHT_PARITY failed to fetch page {:?}: {}",
                    page_id, e
                ))
            })?;
            if handle.device_ptr.is_null() {
                return Err(Error::Cuda(format!(
                    "LMHEAD_WEIGHT_PARITY null page ptr for {:?}",
                    page_id
                )));
            }

            let page_row_offset = row_in_page * bytes_per_row;
            cuda_ffi::memcpy_d2h(
                paged_row.as_mut_ptr(),
                unsafe { (handle.device_ptr as *const u8).add(page_row_offset) },
                bytes_per_row,
            )?;

            let shared_f16 = unsafe {
                std::slice::from_raw_parts(shared_row.as_ptr() as *const f16, hidden_dim)
            };
            let paged_f16 = unsafe {
                std::slice::from_raw_parts(paged_row.as_ptr() as *const f16, hidden_dim)
            };

            let mut dot = 0.0f64;
            let mut a2 = 0.0f64;
            let mut b2 = 0.0f64;
            let mut diff2 = 0.0f64;
            let mut max_abs = 0.0f32;
            let mut exact = 0usize;
            for (a, b) in shared_f16.iter().zip(paged_f16.iter()) {
                if a.to_bits() == b.to_bits() {
                    exact += 1;
                }
                let af = a.to_f32() as f64;
                let bf = b.to_f32() as f64;
                dot += af * bf;
                a2 += af * af;
                b2 += bf * bf;
                let d = af - bf;
                diff2 += d * d;
                max_abs = max_abs.max((a.to_f32() - b.to_f32()).abs());
            }

            let cosine = if a2 > 0.0 && b2 > 0.0 {
                (dot / (a2.sqrt() * b2.sqrt())) as f32
            } else {
                0.0
            };
            let rel_l2 = if b2 > 0.0 {
                (diff2.sqrt() / b2.sqrt()) as f32
            } else {
                0.0
            };

            tracing::info!(
                "LMHEAD_WEIGHT_PARITY row={}: cosine={:.6}, rel_l2={:.6}, max_abs={:.6}, exact_bits={}/{}",
                row,
                cosine,
                rel_l2,
                max_abs,
                exact,
                hidden_dim,
            );

            if rel_l2 > 1e-3 || max_abs > 1e-6 {
                mismatch_rows += 1;
                tracing::warn!(
                    "LMHEAD_WEIGHT_PARITY_MISMATCH row={}: page_seq={}, row_in_page={}, page_id={:?}, shared_first4=[{:.6},{:.6},{:.6},{:.6}], paged_first4=[{:.6},{:.6},{:.6},{:.6}]",
                    row,
                    page_seq,
                    row_in_page,
                    page_id,
                    shared_f16[0].to_f32(),
                    shared_f16[1].to_f32(),
                    shared_f16[2].to_f32(),
                    shared_f16[3].to_f32(),
                    paged_f16[0].to_f32(),
                    paged_f16[1].to_f32(),
                    paged_f16[2].to_f32(),
                    paged_f16[3].to_f32(),
                );

                // Alternate lookup using true per-page row counts (raw_size / bytes_per_row)
                // instead of fixed page_size geometry. If this matches shared rows while
                // fixed-geometry lookup doesn't, paging geometry is the likely root cause.
                let mut cum_rows = 0usize;
                let mut alt_match: Option<(usize, usize, PageId)> = None;
                for (alt_seq, (_alt_idx, alt_entry)) in pages.iter().enumerate() {
                    let alt_rows = (alt_entry.raw_size as usize) / bytes_per_row;
                    if row < cum_rows + alt_rows {
                        let alt_row_in_page = row - cum_rows;
                        alt_match = Some((alt_seq, alt_row_in_page, alt_entry.page_id()));
                        break;
                    }
                    cum_rows += alt_rows;
                }

                if let Some((alt_seq, alt_row_in_page, alt_page_id)) = alt_match {
                    let alt_handle = self.buffer_mgr.get_page(&alt_page_id).await.map_err(|e| {
                        Error::Cuda(format!(
                            "LMHEAD_WEIGHT_PARITY alt lookup failed to fetch page {:?}: {}",
                            alt_page_id, e
                        ))
                    })?;
                    if !alt_handle.device_ptr.is_null() {
                        let mut alt_row = vec![0u8; bytes_per_row];
                        let alt_offset = alt_row_in_page * bytes_per_row;
                        cuda_ffi::memcpy_d2h(
                            alt_row.as_mut_ptr(),
                            unsafe { (alt_handle.device_ptr as *const u8).add(alt_offset) },
                            bytes_per_row,
                        )?;
                        let alt_f16 = unsafe {
                            std::slice::from_raw_parts(alt_row.as_ptr() as *const f16, hidden_dim)
                        };
                        let mut alt_exact = 0usize;
                        let mut alt_max_abs = 0.0f32;
                        for (a, b) in shared_f16.iter().zip(alt_f16.iter()) {
                            if a.to_bits() == b.to_bits() {
                                alt_exact += 1;
                            }
                            alt_max_abs = alt_max_abs.max((a.to_f32() - b.to_f32()).abs());
                        }
                        tracing::warn!(
                            "LMHEAD_WEIGHT_PARITY_ALT row={}: alt_page_seq={}, alt_row_in_page={}, alt_page_id={:?}, alt_exact_bits={}/{}, alt_max_abs={:.6}, alt_first4=[{:.6},{:.6},{:.6},{:.6}]",
                            row,
                            alt_seq,
                            alt_row_in_page,
                            alt_page_id,
                            alt_exact,
                            hidden_dim,
                            alt_max_abs,
                            alt_f16[0].to_f32(),
                            alt_f16[1].to_f32(),
                            alt_f16[2].to_f32(),
                            alt_f16[3].to_f32(),
                        );
                    }
                }

                // Ground truth check: compare both sources against on-disk row bytes.
                let mut disk_page = vec![0u8; entry.raw_size as usize];
                match self.model_file.read_page_sync(*catalog_idx, &mut disk_page) {
                    Ok(n) if n >= (row_in_page + 1) * bytes_per_row => {
                        let start = row_in_page * bytes_per_row;
                        let end = start + bytes_per_row;
                        let disk_row = &disk_page[start..end];
                        let disk_f16 = unsafe {
                            std::slice::from_raw_parts(disk_row.as_ptr() as *const f16, hidden_dim)
                        };
                        let mut shared_disk_exact = 0usize;
                        let mut paged_disk_exact = 0usize;
                        let mut shared_disk_max_abs = 0.0f32;
                        let mut paged_disk_max_abs = 0.0f32;
                        for i in 0..hidden_dim {
                            if shared_f16[i].to_bits() == disk_f16[i].to_bits() {
                                shared_disk_exact += 1;
                            }
                            if paged_f16[i].to_bits() == disk_f16[i].to_bits() {
                                paged_disk_exact += 1;
                            }
                            shared_disk_max_abs = shared_disk_max_abs
                                .max((shared_f16[i].to_f32() - disk_f16[i].to_f32()).abs());
                            paged_disk_max_abs = paged_disk_max_abs
                                .max((paged_f16[i].to_f32() - disk_f16[i].to_f32()).abs());
                        }
                        tracing::warn!(
                            "LMHEAD_WEIGHT_PARITY_DISK row={}: shared_exact={}/{}, paged_exact={}/{}, shared_max_abs={:.6}, paged_max_abs={:.6}",
                            row,
                            shared_disk_exact,
                            hidden_dim,
                            paged_disk_exact,
                            hidden_dim,
                            shared_disk_max_abs,
                            paged_disk_max_abs,
                        );
                    }
                    Ok(n) => {
                        tracing::warn!(
                            "LMHEAD_WEIGHT_PARITY_DISK row={}: insufficient page bytes read={} needed={}",
                            row,
                            n,
                            (row_in_page + 1) * bytes_per_row,
                        );
                    }
                    Err(e) => {
                        tracing::warn!(
                            "LMHEAD_WEIGHT_PARITY_DISK row={}: read_page_sync failed: {}",
                            row,
                            e,
                        );
                    }
                }
            }
        }

        Ok(mismatch_rows)
    }

    /// Compute logits using paged lm_head weights (segment 11) without requiring
    /// a single contiguous device tensor for the full vocab projection.
    ///
    /// This is the low-VRAM fallback path for very large models where allocating
    /// `vocab_size * hidden_dim * 2` bytes in one chunk may fail.
    async fn compute_logits_paged_lm_head(
        &mut self,
        hidden_dim: usize,
        vocab_size: usize,
        logits_bytes: usize,
    ) -> Result<Vec<f32>> {
        let pages = self.model_file.pages_for_shared_segment(0, 11);
        if pages.is_empty() {
            return Err(Error::Cuda(
                "lm_head segment (L0 S11) has no pages for paged fallback".to_string(),
            ));
        }

        let bytes_per_row = hidden_dim * 2; // FP16 weights
        let page_size = 2 * 1024 * 1024usize;
        let rows_per_page = page_size / bytes_per_row;
        if rows_per_page == 0 {
            return Err(Error::Cuda(format!(
                "invalid lm_head paging geometry: hidden_dim={} bytes_per_row={} exceeds page size {}",
                hidden_dim, bytes_per_row, page_size
            )));
        }

        // Use FP32 input + FP32 output GEMV for logits.
        kernels::f16_to_f32(
            self.hidden_state.as_ptr(),
            self.attn_normed_f32.as_mut_ptr(),
            hidden_dim,
            &self.stream,
        )?;

        for (page_seq, (_idx, entry)) in pages.iter().enumerate() {
            let page_id = entry.page_id();
            let row_start = page_seq * rows_per_page;
            if row_start >= vocab_size {
                break;
            }
            let rows_in_page = (entry.raw_size as usize) / bytes_per_row;
            let row_count = rows_in_page.min(vocab_size - row_start);
            if row_count == 0 {
                continue;
            }

            let handle = self.buffer_mgr.get_page(&page_id).await.map_err(|e| {
                Error::Cuda(format!(
                    "paged lm_head fallback failed to fetch page {:?}: {}",
                    page_id, e
                ))
            })?;

            if handle.device_ptr.is_null() {
                return Err(Error::Cuda(format!(
                    "paged lm_head fallback got null device pointer for page {:?}",
                    page_id
                )));
            }

            let output_offset = row_start * std::mem::size_of::<f32>();
            kernels::linear_projection_f32_to_f32(
                self.attn_normed_f32.as_ptr(),
                handle.device_ptr as *const u8,
                unsafe { self.logits_dev.as_mut_ptr().add(output_offset) },
                hidden_dim,
                row_count,
                &self.stream,
            )?;
        }

        self.stream.synchronize()?;
        let mut logits = vec![0.0f32; vocab_size];
        cuda_ffi::memcpy_d2h(
            logits.as_mut_ptr() as *mut u8,
            self.logits_dev.as_ptr(),
            logits_bytes,
        )?;
        Ok(logits)
    }

    /// Paged FP16 GEMV: reads weight pages directly from T1 VRAM, no D2D staging.
    /// Launches one GEMV kernel per page, writing to the correct output offset.
    /// For [M, K] weight matrices split across pages by rows (M dimension):
    ///   output[row_start..row_start+row_count] = input × W_page^T
    #[allow(dead_code)]
    fn linear_projection_paged(
        &self,
        input: *const u8,
        output: *mut u8,
        layer: u16,
        segment: u16,
        k: usize,
        total_rows: usize,
    ) -> Result<()> {
        let pages = self.model_file.pages_for_shared_segment(layer, segment);
        let bytes_per_row = k * 2; // FP16
        let page_size = 2 * 1024 * 1024usize; // 2MB
        let rows_per_page = page_size / bytes_per_row;
        for (page_seq, (_idx, entry)) in pages.iter().enumerate() {
            let page_id = entry.page_id();
            if let Some(handle) = self.buffer_mgr.get_page_resident(&page_id) {
                if !handle.device_ptr.is_null() {
                    let row_start = page_seq * rows_per_page;
                    let row_count = rows_per_page.min(total_rows - row_start);
                    if row_count == 0 {
                        continue;
                    }
                    // FP16 output: 2 bytes per element
                    let output_offset = row_start * 2;
                    kernels::linear_projection(
                        input,
                        handle.device_ptr as *const u8,
                        // SAFETY: output + offset is within the pre-allocated output buffer
                        unsafe { output.add(output_offset) },
                        k,
                        row_count,
                        &self.stream,
                    ).map_err(|e| {
                        tracing::error!(
                            "linear_projection_paged FAILED: layer={} seg={} k={} row_start={} row_count={} dev_ptr={:?} err={}",
                            layer, segment, k, row_start, row_count, handle.device_ptr, e
                        );
                        e
                    })?;
                }
            }
        }
        Ok(())
    }

    /// Paged FP32-input, FP32-output GEMV: reads weight pages directly from T1 VRAM.
    /// Like `linear_projection_paged` but uses the f32in_f32out kernel variant.
    #[allow(dead_code)]
    fn linear_projection_f32_paged(
        &self,
        input: *const u8,
        output: *mut u8,
        layer: u16,
        segment: u16,
        k: usize,
        total_rows: usize,
    ) -> Result<()> {
        let pages = self.model_file.pages_for_shared_segment(layer, segment);
        let bytes_per_row = k * 2; // FP16 weights
        let page_size = 2 * 1024 * 1024usize; // 2MB
        let rows_per_page = page_size / bytes_per_row;
        for (page_seq, (_idx, entry)) in pages.iter().enumerate() {
            let page_id = entry.page_id();
            if let Some(handle) = self.buffer_mgr.get_page_resident(&page_id) {
                if !handle.device_ptr.is_null() {
                    let row_start = page_seq * rows_per_page;
                    let row_count = rows_per_page.min(total_rows - row_start);
                    if row_count == 0 {
                        continue;
                    }
                    // FP32 output: 4 bytes per element
                    let output_offset = row_start * 4;
                    kernels::linear_projection_f32_to_f32(
                        input,
                        handle.device_ptr as *const u8,
                        // SAFETY: output + offset is within the pre-allocated output buffer
                        unsafe { output.add(output_offset) },
                        k,
                        row_count,
                        &self.stream,
                    ).map_err(|e| {
                        tracing::error!(
                            "linear_projection_f32_paged FAILED: layer={} seg={} k={} row_start={} row_count={} dev_ptr={:?} err={}",
                            layer, segment, k, row_start, row_count, handle.device_ptr, e
                        );
                        e
                    })?;
                }
            }
        }
        Ok(())
    }
}

/// Detect total system RAM (bytes).
fn detect_system_ram() -> usize {
    #[cfg(target_os = "linux")]
    {
        if let Ok(meminfo) = std::fs::read_to_string("/proc/meminfo") {
            for line in meminfo.lines() {
                if line.starts_with("MemTotal:") {
                    let parts: Vec<&str> = line.split_whitespace().collect();
                    if let Some(kb) = parts.get(1) {
                        if let Ok(val) = kb.parse::<usize>() {
                            return val * 1024;
                        }
                    }
                }
            }
        }
    }
    // Fallback
    64 * 1024 * 1024 * 1024
}
