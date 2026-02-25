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
    /// Accumulated layer output buffer (FP32, [hidden_dim]) — on device.
    /// FP32 to eliminate truncation errors when accumulating 8+ expert outputs.
    layer_output_buf: DeviceBuffer,
    /// Residual connection buffer (FP16, [hidden_dim]) — on device.
    /// Holds the pre-norm hidden_state for residual add.
    residual_buf: DeviceBuffer,
    /// Down-projection scratch buffer (FP16, [hidden_dim]) — on device.
    /// Pre-allocated to avoid per-call cudaMalloc in execute_expert.
    down_proj_buf: DeviceBuffer,
    /// FP32 normalized hidden state for MoE sublayer (FP32, [hidden_dim]).
    /// Used as input to FP32-input expert matmuls and FP32-input router.
    moe_normed_f32: DeviceBuffer,

    // ── Pre-allocated attention projection scratch buffers (VRAM) ──
    // These eliminate 4x cudaMalloc + cudaFree per layer (128 calls/token).
    /// Q projection scratch (FP16, [q_dim] = [hidden_dim]).
    q_proj_dev: DeviceBuffer,
    /// K projection scratch (FP16, [kv_dim] = [num_kv_heads * head_dim]).
    k_proj_dev: DeviceBuffer,
    /// V projection scratch (FP16, [kv_dim]).
    v_proj_dev: DeviceBuffer,
    /// Attention output scratch for O projection input (FP16, [hidden_dim]).
    attn_dev: DeviceBuffer,

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
}

// SAFETY: Engine is only accessed through a tokio::sync::Mutex in the API server,
// ensuring single-threaded access. The raw pointers within (via CudaStream, buffer
// manager page pointers) are allocated/freed on the same thread.
unsafe impl Send for Engine {}
unsafe impl Sync for Engine {}

impl Engine {
    fn normalize_prompt_for_model(model_arch: &str, prompt: &str) -> String {
        // Mixtral/Mistral instruct checkpoints typically expect [INST] wrappers.
        // If the user already provided a template, preserve it as-is.
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

        // Allocate working buffers as DeviceBuffers (VRAM on GPU, host on CPU).
        let hidden_dim = model_config.hidden_dim as usize;
        let _vocab_size = model_config.vocab_size as usize;
        let expert_hidden = model_config.expert_hidden_dim as usize;
        let hidden_bytes = hidden_dim * std::mem::size_of::<f16>();
        let expert_bytes = expert_hidden * std::mem::size_of::<f16>();

        // Pre-compute sizes for attention projection scratch buffers
        let num_heads = model_config.num_heads as usize;
        let num_kv_heads = model_config.num_kv_heads as usize;
        let head_dim = if num_heads > 0 {
            hidden_dim / num_heads
        } else {
            128
        };
        let q_proj_bytes = hidden_dim * 2; // FP16
        let kv_proj_bytes = num_kv_heads * head_dim * 2; // FP16
        let router_scores_bytes = (model_config.num_experts as usize).max(1) * 4; // f32

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
        let head_dim = hidden_dim / model_config.num_heads as usize;
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

        Ok(Self {
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
            hidden_state: DeviceBuffer::new(hidden_bytes)?,
            hidden_state_f32: DeviceBuffer::new(hidden_dim * 4)?, // FP32 accumulator
            expert_output_buf: DeviceBuffer::new(expert_bytes)?,
            layer_output_buf: DeviceBuffer::new(hidden_dim * 4)?, // FP32 for precision
            moe_normed_f32: DeviceBuffer::new(hidden_dim * 4)?, // FP32 normalized hidden state for MoE
            residual_buf: DeviceBuffer::new(hidden_bytes)?,
            down_proj_buf: DeviceBuffer::new(hidden_bytes)?,
            q_proj_dev: DeviceBuffer::new(q_proj_bytes)?,
            k_proj_dev: DeviceBuffer::new(kv_proj_bytes)?,
            v_proj_dev: DeviceBuffer::new(kv_proj_bytes)?,
            attn_dev: DeviceBuffer::new(hidden_bytes)?,
            swiglu_up_tmp: DeviceBuffer::new(expert_bytes)?,
            swiglu_gate_tmp: DeviceBuffer::new(expert_bytes)?,
            router_scores_dev: DeviceBuffer::new(router_scores_bytes)?,
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
            task_context: None,
            gear_profiles: std::collections::HashMap::new(),
            e_score_correction_bias: None,
            diag_enabled: std::env::var("VIB3_DIAG").map_or(false, |v| v == "1"),
        })
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
            let token_id = self.generate_token(step, &params).await?;

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

            // Debug: log embedding output for last token in prefill
            if self.diag_enabled && tok_idx == tokens.len() - 1 {
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
            }

            // 2. Run through all transformer layers at this position
            //    Each layer: Attention sublayer → MoE/FFN sublayer (interleaved)
            self.position = tok_idx;

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

                // Diagnostic: log hidden state stats after each layer for the last prefill token
                // NOTE: We read hidden_state_f32 (the actual accumulator), not hidden_state (stale FP16)
                if self.diag_enabled && is_last_token && (layer_idx <= 15 || layer_idx % 5 == 0 || layer_idx >= num_layers as u16 - 2) {
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

                    // Dump FP32 hidden state for selected layers to compare with Python GT
                    if matches!(layer_idx, 0 | 1 | 2 | 5 | 10 | 15 | 20 | 25 | 30 | 31) {
                        let dump_dir = "/home/brian/code/vib3/dump";
                        let _ = std::fs::create_dir_all(dump_dir);
                        let dump_path = format!("{}/vib3_mixtral_hidden_f32_L{}_lastpos.bin", dump_dir, layer_idx);
                        let _ = std::fs::write(&dump_path, &diag_buf);
                        tracing::info!("DUMPED FP32 hidden state to {} ({} bytes)", dump_path, diag_buf.len());
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
        let eps = self.model_config.rms_norm_eps;
        self.ensure_shared_tensor_device(layer_idx, 6).await;
        if let Some((norm_ptr, _)) = self.get_device_tensor(layer_idx, 6) {
            kernels::rms_norm_f32_to_f16(
                self.hidden_state_f32.as_ptr(),
                self.hidden_state.as_mut_ptr(),
                norm_ptr,
                hidden_dim,
                eps,
                &self.stream,
            )?;
        } else {
            // Fallback: cast then normalize without weight (rare path)
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
        // Keep residual_buf for CPU fallback attention paths
        cuda_ffi::device_memcpy_d2d_async(
            self.residual_buf.as_mut_ptr(),
            self.hidden_state.as_ptr(),
            hidden_bytes,
            &self.stream,
        )?;

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
        let head_dim = hidden_dim / num_heads;
        let q_dim = hidden_dim;
        let kv_dim = num_kv_heads * head_dim;

        // Ensure Q/K/V/O weights are on device before entering sync code blocks
        self.ensure_shared_tensor_device(layer_idx, 4).await;
        self.ensure_shared_tensor_device(layer_idx, 12).await;
        self.ensure_shared_tensor_device(layer_idx, 13).await;
        self.ensure_shared_tensor_device(layer_idx, 5).await;

        // Try 1: Fully-GPU decode attention (no CPU round-trip at all)
        if self.stream.is_real() {
            let gpu_decode =
                self.try_gpu_decode_attention(layer_idx, hidden_dim, hidden_bytes, q_dim, kv_dim)?;
            if gpu_decode {
                return Ok(());
            }
        }

        // Try 2: GPU-projected attention (projections on GPU, attention on CPU)
        let gpu_projected = self.stream.is_real()
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
        kernels::linear_projection(
            self.attn_dev.as_ptr(),
            o_ptr,
            self.hidden_state.as_mut_ptr(),
            hidden_dim,
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
            let head_dim = hidden_dim / num_heads;
            let rope_base = self.model_config.rope_theta;
            let position = self.position;

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

            let stream_ptr = self.stream.raw_ptr();

            // Step 1: GPU GEMV — Q/K/V projections into pre-allocated device buffers
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

            // Step 2: RoPE — apply rotary position embeddings in-place on Q and K (FP16)
            let err = unsafe {
                cuda_ffi::vib3_launch_rope_apply(
                    self.q_proj_dev.as_mut_ptr(),
                    self.k_proj_dev.as_mut_ptr(),
                    head_dim as i32,
                    num_heads as i32,
                    num_kv_heads as i32,
                    position as i32,
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
                    position as i32,
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
            let seq_len = position + 1; // include the just-appended position
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
                    seq_len as i32,
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
            kernels::linear_projection(
                self.attn_dev.as_ptr(),
                o_ptr,
                self.hidden_state.as_mut_ptr(),
                hidden_dim,
                hidden_dim,
                &self.stream,
            )?;

            // Step 6: FP32 residual accumulation: hidden_state_f32 += f32(O_proj output)
            kernels::residual_add_fp32(
                self.hidden_state_f32.as_mut_ptr(),
                self.hidden_state.as_ptr(),
                hidden_dim,
                &self.stream,
            )?;

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
            // FP32→FP16 RMSNorm: normalize in FP32, output FP16 for diagnostics
            kernels::rms_norm_f32_to_f16(
                self.hidden_state_f32.as_ptr(),
                self.hidden_state.as_mut_ptr(),
                norm_ptr,
                hidden_dim,
                eps,
                &self.stream,
            )?;
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

        let plan = self.planner.plan_layer(layer_idx, &activation).await?;

        // Diagnostic: log expert plan details for first decode step, first MoE layer
        // Also log for layers 1-10 at pos=0 to investigate explosion
        if self.diag_enabled && ((self.decode_step == 1 && layer_idx == 1)
            || (self.position == 0 && layer_idx <= 10)
            || (layer_idx == 5 || layer_idx == 6)) {
            tracing::info!(
                "EXPERT PLAN DIAG L{} pos={}: {} experts activated, router=({:?})",
                layer_idx, self.position, plan.experts.len(),
                activation.experts.iter().map(|(id, w)| format!("e{}={:.4}", id, w)).collect::<Vec<_>>().join(", "),
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

        // Execute routed experts
        for (_i, expert_plan) in plan.experts.iter().enumerate() {
            self.execute_expert(expert_plan, hidden_dim).await?;

            // Diagnostic: after each expert at layer 6, check accumulated output (now FP32)
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

        // Run shared expert (unconditional, every token) if model has one
        if self.model_config.num_shared_experts > 0 {
            self.execute_shared_expert(layer_idx, hidden_dim).await?;
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
    async fn generate_token(&mut self, step: usize, params: &SamplingParams) -> Result<u32> {
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
                        // FP32→FP16 RMSNorm: normalize in FP32, output FP16
                        kernels::rms_norm_f32_to_f16(
                            self.hidden_state_f32.as_ptr(),
                            self.hidden_state.as_mut_ptr(),
                            norm_ptr,
                            hidden_dim,
                            eps,
                            &self.stream,
                        )?;
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

                    // Get active expert activations for this layer
                    let tr = Instant::now();
                    let activation = self.run_router_for_layer(layer_idx).await?;
                    let router_us = tr.elapsed().as_micros() as u64;

                    // Collect expert IDs for mode detection
                    for &(expert_id, _weight) in &activation.experts {
                        self.token_expert_ids.push(expert_id);
                    }

                    // Submit lookahead for next MoE layer (mode-aware via planner)
                    let t_look = Instant::now();
                    let moe_offset = layer_idx - dense_layer_idx as u16;
                    if moe_offset + 1 < num_moe_layers as u16 {
                        self.planner.submit_lookahead(layer_idx + 1, &activation);
                        self.planner.submit_cross_layer_prefetch(layer_idx);
                    }
                    let lookahead_us = t_look.elapsed().as_micros() as u64;

                    // Plan and execute expert computation
                    let tp = Instant::now();
                    let plan = self.planner.plan_layer(layer_idx, &activation).await?;
                    let plan_us = tp.elapsed().as_micros() as u64;

                    // Zero the layer output on device (async — no pipeline stall)
                    self.layer_output_buf.zero_async(&self.stream);

                    let te = Instant::now();
                    for expert_plan in &plan.experts {
                        self.execute_expert(expert_plan, hidden_dim).await?;
                    }
                    let routed_us = te.elapsed().as_micros() as u64;

                    // Run shared expert (unconditional, every token)
                    let ts = Instant::now();
                    if self.model_config.num_shared_experts > 0 {
                        self.execute_shared_expert(layer_idx, hidden_dim).await?;
                    }
                    let shared_us = ts.elapsed().as_micros() as u64;

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
        let hidden_dim = self.model_config.hidden_dim as usize;
        let hidden_bytes = hidden_dim * std::mem::size_of::<f16>();
        if step < 5 {
            self.stream.synchronize()?;
            let mut diag_buf = vec![0u8; hidden_bytes];
            self.hidden_state.copy_to_host(&mut diag_buf)?;
            let h_f16 = unsafe {
                std::slice::from_raw_parts(diag_buf.as_ptr() as *const f16, hidden_dim)
            };
            let h_max = h_f16.iter().map(|v| v.to_f32()).fold(f32::NEG_INFINITY, f32::max);
            let h_min = h_f16.iter().map(|v| v.to_f32()).fold(f32::INFINITY, f32::min);
            let h_mean = h_f16.iter().map(|v| v.to_f32()).sum::<f32>() / hidden_dim as f32;
            let h_nan = h_f16.iter().filter(|v| v.to_f32().is_nan()).count();
            let h_zero = h_f16.iter().filter(|v| v.to_f32() == 0.0).count();
            tracing::info!(
                "Hidden state step={} (after all layers, before final norm): min={:.4}, max={:.4}, mean={:.6}, nan={}, zero={}/{}",
                step, h_min, h_max, h_mean, h_nan, h_zero, hidden_dim
            );
        }

        // Apply final RMSNorm: FP32 → normalize in FP32 → FP16 for logit projection
        // The final norm is stored at layer=0xFFFF, segment=8
        let eps = self.model_config.rms_norm_eps;

        self.ensure_shared_tensor_device(0xFFFF, 8).await;
        if let Some((norm_ptr, _norm_size)) = self.get_device_tensor(0xFFFF, 8) {
            // Override final norm weights from external file if available.
            // The .vib3 page for the final norm is corrupted due to vision tensor
            // pages being co-stored in segment 8. Load correct weights from a
            // standalone FP16 binary file generated from the safetensors source.
            if step == 0 {
                let norm_file = std::path::Path::new("/final_norm_weight.fp16.bin");
                if norm_file.exists() {
                    match std::fs::read(norm_file) {
                        Ok(data) if data.len() == hidden_dim * 2 => {
                            if let Err(e) = cuda_ffi::memcpy_h2d_sync(
                                norm_ptr as *mut u8,
                                data.as_ptr(),
                                data.len(),
                            ) {
                                tracing::warn!("Failed to upload final norm override: {}", e);
                            } else {
                                tracing::info!(
                                    "Final norm: loaded correct weights from {} ({} bytes)",
                                    norm_file.display(), data.len(),
                                );
                            }
                        }
                        Ok(data) => {
                            tracing::warn!(
                                "Final norm override file has wrong size: {} (expected {})",
                                data.len(), hidden_dim * 2,
                            );
                        }
                        Err(e) => {
                            tracing::warn!("Failed to read final norm override: {}", e);
                        }
                    }
                }
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

        // Compute logits from hidden state (both in VRAM for GPU dispatch)
        let vocab_size = self.model_config.vocab_size as usize;

        // Load lm_head weights to device
        self.ensure_shared_tensor_device(0, 11).await;
        let logits = if let Some((lm_head_ptr, _)) = self.get_device_tensor(0, 11) {
            // Real lm_head projection — both hidden_state and lm_head on device
            kernels::compute_logits(
                self.hidden_state.as_ptr(),
                lm_head_ptr,
                vocab_size,
                hidden_dim,
            )
        } else {
            // Fallback: D2H hidden_state, compute logits on CPU
            self.stream.synchronize()?;
            self.hidden_state
                .copy_to_host(&mut self.host_staging[..hidden_bytes])?;
            let state = unsafe {
                std::slice::from_raw_parts(self.host_staging.as_ptr() as *const f16, hidden_dim)
            };
            let mut logits = vec![0.0f32; vocab_size];
            let sample_dims = hidden_dim.min(64);
            for (v, logit) in logits.iter_mut().enumerate().take(vocab_size) {
                let mut acc = 0.0f32;
                for d in 0..sample_dims {
                    let h = d * vocab_size / sample_dims + v;
                    let weight_idx = h % hidden_dim;
                    acc += state[weight_idx].to_f32();
                }
                *logit = acc;
            }
            logits
        };

        // Debug: inspect hidden state before logit computation
        tracing::trace!(
            "Hidden state (step={}, after final norm): computing logits",
            step,
        );

        // Debug: log logit distribution stats for first few tokens
        if step < 5 || step % 10 == 0 {
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
                "Logits step={}: min={:.4}, max={:.4}, mean={:.4}, nan={}, inf={}, top5={:?}",
                step, min_logit, max_logit, mean_logit, nan_count, inf_count, top5
            );
        }

        let token_id = self.sampler.sample(&logits, params);
        tracing::info!(
            "Token step={}: id={}, total={:.1}ms",
            step, token_id,
            compute_start.elapsed().as_secs_f64() * 1000.0,
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
                let scale_data_size = num_groups * m_slice * 2; // BF16 = 2 bytes
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

                // Decode first 4 scales as BF16
                let scales_u16: Vec<u16> = sc_buf.chunks_exact(2).take(4)
                    .map(|c| u16::from_le_bytes([c[0], c[1]])).collect();
                let scales_f32: Vec<f32> = scales_u16.iter()
                    .map(|&bits| f32::from_bits((bits as u32) << 16)).collect();

                tracing::error!(
                    "WEIGHT PAGE DIAG L{} e{} pg{}: total_page={}, wt_data={}, scale_data={}, \
                     wt_first8={:02x?}, wt_nonzero={}/{}, \
                     sc_first8={:02x?}, sc_nonzero={}/{}, \
                     sc_as_bf16={:?}, sc_as_f32={:?}, \
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
                // FP32-input SwiGLU: reads FP32 moe_normed_f32, writes FP16 expert intermediate
                // Segment 0 = up_proj (multiplied directly), Segment 1 = gate_proj (SiLU applied)
                // The kernel signature is (input, up_weight, gate_weight) where SiLU is applied to gate.
                kernels::partial_swiglu_f32(
                    self.moe_normed_f32.as_ptr(),
                    up_page.device_ptr as *const u8,    // segment 0 = up_proj (multiplied directly)
                    gate_page.device_ptr as *const u8,  // segment 1 = gate_proj (SiLU applied)
                    // SAFETY: byte_offset is within expert_output_buf bounds
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

        // ═══ CRITICAL DIAGNOSTIC: SwiGLU intermediate (fires L0-L2, L31 on first token) ═══
        // Gated behind VIB3_DIAG=1 to avoid overhead in normal inference.
        if self.diag_enabled && self.position <= 1 && (engine_layer <= 2 || engine_layer == 31) {
            self.stream.synchronize()?;
            let expert_hidden_dim_diag = self.model_config.expert_hidden_dim as usize;
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

        // Apply down_proj: project expert_hidden_dim back to hidden_dim.
        // Uses pre-allocated down_proj_buf (DeviceBuffer in VRAM).
        let expert_hidden_dim = self.model_config.expert_hidden_dim as usize;
        self.down_proj_buf.zero_async(&self.stream);

        for down_page in &down_pages {
            let m_slice = down_page.row_count as usize;
            let row_start = down_page.row_start as usize;
            let byte_offset = row_start * std::mem::size_of::<f16>();

            if !down_page.device_ptr.is_null() {
                kernels::partial_matmul(
                    self.expert_output_buf.as_ptr(),
                    down_page.device_ptr as *const u8,
                    // SAFETY: byte_offset is within down_proj_buf bounds
                    unsafe { self.down_proj_buf.as_mut_ptr().add(byte_offset) },
                    expert_hidden_dim,
                    m_slice,
                    self.model_config.expert_dtype,
                    &self.stream,
                )?;
            }
        }

        // Diagnostic: per-expert down_proj output at L6, pos=0
        if self.diag_enabled && engine_layer == 6 && self.position == 0 {
            self.stream.synchronize()?;
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
            // Dump e243's down_proj and intermediate for Python comparison
            if expert_plan.expert_id == 243 {
                let dump_dir = "/model/dump";
                let _ = std::fs::create_dir_all(dump_dir);
                let _ = std::fs::write(format!("{}/e243_downproj_L6_pos0.bin", dump_dir), &down_buf);
                // Also dump the SwiGLU intermediate
                let ehd = self.model_config.expert_hidden_dim as usize;
                let mut inter_buf = vec![0u8; ehd * 2];
                self.expert_output_buf.copy_to_host(&mut inter_buf)?;
                let _ = std::fs::write(format!("{}/e243_swiglu_L6_pos0.bin", dump_dir), &inter_buf);
                tracing::info!("DUMPED e243 down_proj and intermediate to {}", dump_dir);
            }
        }

        // Accumulate expert contribution into FP32 buffer (no FP16 truncation)
        kernels::weighted_accumulate_f32(
            self.layer_output_buf.as_mut_ptr(),
            self.down_proj_buf.as_ptr(),
            expert_plan.weight,
            hidden_dim,
            &self.stream,
        )?;

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
        if let (Some((up_ptr, _)), Some((gate_ptr, _))) = (up_device, gate_device) {
            // FP32-input SwiGLU for shared expert: reads FP32 moe_normed_f32, FP16 weights
            kernels::partial_swiglu_f32(
                self.moe_normed_f32.as_ptr(),
                up_ptr,
                gate_ptr,
                self.shared_expert_inter_dev.as_mut_ptr(),
                hidden_dim,
                shared_hidden,
                self.model_config.shared_dtype,
                &self.stream,
                None, // shared expert weights are FP16, fused kernel
            )?;

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
                    self.model_config.shared_dtype,
                    &self.stream,
                )?;

                // Diagnostic: shared expert output at L6, pos=0
                if self.diag_enabled && layer == 6 && self.position == 0 {
                    self.stream.synchronize()?;
                    let mut se_buf = vec![0u8; hidden_dim * 2];
                    self.shared_expert_down_dev.copy_to_host(&mut se_buf)?;
                    let se_f16 = unsafe { std::slice::from_raw_parts(se_buf.as_ptr() as *const f16, hidden_dim) };
                    let se_l2 = se_f16.iter().map(|v| { let f = v.to_f32(); f * f }).sum::<f32>().sqrt();
                    let se_max = se_f16.iter().map(|v| v.to_f32().abs()).fold(0.0f32, f32::max);
                    tracing::info!(
                        "L6 SHARED_EXPERT DOWN_PROJ: L2={:.4}, max_abs={:.4}",
                        se_l2, se_max,
                    );
                    // Also dump for Python comparison
                    let dump_dir = "/model/dump";
                    let _ = std::fs::create_dir_all(dump_dir);
                    let _ = std::fs::write(format!("{}/shared_expert_L6_pos0.bin", dump_dir), &se_buf);

                    // Dump moe_normed_f32 for Python comparison (FP32)
                    let mut normed_buf = vec![0u8; hidden_dim * 4];
                    self.moe_normed_f32.copy_to_host(&mut normed_buf)?;
                    let _ = std::fs::write(format!("{}/moe_normed_f32_L6_pos0.bin", dump_dir), &normed_buf);
                    tracing::info!("DUMPED moe_normed_f32 and shared_expert to {}", dump_dir);
                }

                // Accumulate shared expert output into FP32 buffer with weight 1.0
                kernels::weighted_accumulate_f32(
                    self.layer_output_buf.as_mut_ptr(),
                    self.shared_expert_down_dev.as_ptr(),
                    1.0, // Shared expert always has weight 1.0
                    hidden_dim,
                    &self.stream,
                )?;
            }
        }
        // If weights aren't loaded, silently skip — the shared expert
        // only runs when its weights are available in the buffer pool.

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
            // Get per-layer e_score_correction_bias (if loaded)
            let layer_bias = self.e_score_correction_bias.as_ref()
                .and_then(|biases| biases.get(layer as usize))
                .map(|v| v.as_slice());

            // Use FP32-input router for better precision (avoids FP16 truncation
            // of the normalized hidden state that causes catastrophic expert misrouting)
            let experts = kernels::run_router_f32(
                self.moe_normed_f32.as_ptr(),
                router_ptr,
                num_experts,
                hidden_dim,
                top_k,
                scoring_func,
                layer_bias,
                self.router_scores_dev.as_mut_ptr() as *mut f32,
                &self.stream,
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
    ) -> Result<u32> {
        self.generate_token(step, params).await
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

        // Check device cache first
        if self.shared_tensor_cache_device.contains_key(&cache_key) {
            return true;
        }

        // Try to assemble directly on device from T1 pages (avoids VRAM→host→VRAM roundtrip)
        if self.assemble_tensor_on_device(layer, segment).await {
            return true;
        }

        // Fallback: load to host first, then upload to device
        let host_data = match self.load_shared_tensor(layer, segment).await {
            Some(data) => data,
            None => return false,
        };

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
                    "Failed to allocate device buffer for shared tensor L{} S{}: {}",
                    layer,
                    segment,
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
    /// Must call `ensure_shared_tensor_device` first. Returns None if the
    /// tensor is not in the device cache.
    fn get_device_tensor(&self, layer: u16, segment: u16) -> Option<(*const u8, usize)> {
        let cache_key = (layer as u32) << 16 | segment as u32;
        self.shared_tensor_cache_device
            .get(&cache_key)
            .map(|dbuf| (dbuf.as_ptr(), dbuf.size()))
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
