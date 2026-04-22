//! CUDA FFI — the thin unsafe boundary layer.
//!
//! All CUDA calls go through here. Nothing above this module touches
//! raw CUDA APIs. This keeps the unsafe surface area minimal and auditable.
//!
//! When CUDA is available (feature enabled + runtime detected), we use real
//! CUDA device memory (cudaMalloc), pinned host memory (cudaMallocHost),
//! async DMA transfers (cudaMemcpyAsync), and CUDA streams. When CUDA is
//! not available, we fall back to aligned host allocations and memcpy.

// This module inherently works with raw device pointers for CUDA FFI.
// Public functions validate pointers (null checks) before dereferencing.
#![allow(clippy::not_unsafe_ptr_arg_deref)]

use crate::core::error::{Error, Result};
use crate::core::types::PAGE_SIZE;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

/// Whether real CUDA is available at runtime.
static CUDA_AVAILABLE: AtomicBool = AtomicBool::new(false);

/// Track total allocations for debugging.
static DEVICE_BYTES_ALLOCATED: AtomicUsize = AtomicUsize::new(0);
static HOST_PINNED_BYTES_ALLOCATED: AtomicUsize = AtomicUsize::new(0);

// ─── CUDA Runtime API FFI ─────────────────────────────────────────────────

#[cfg(feature = "cuda")]
mod cuda_rt {
    //! Direct FFI bindings to the CUDA Runtime API.
    //! We link against libcudart (provided by build.rs).

    use std::os::raw::{c_int, c_void};

    // cudaError_t is an enum, represented as c_int
    pub type CudaError = c_int;
    pub const CUDA_SUCCESS: CudaError = 0;

    // cudaStream_t is an opaque pointer
    pub type CudaStreamT = *mut c_void;

    // cudaMemcpyKind
    pub const MEMCPY_HOST_TO_DEVICE: c_int = 1;
    pub const MEMCPY_DEVICE_TO_HOST: c_int = 2;

    /// Opaque buffer matching `cudaDeviceProp` (1032 bytes in CUDA 12.8).
    /// We use a raw byte buffer and extract fields at known ABI offsets
    /// because the struct layout varies across CUDA versions and the Rust
    /// #[repr(C)] alignment doesn't match the CUDA compiler's padding.
    #[repr(C, align(8))]
    pub struct CudaDeviceProp {
        _data: [u8; 1032],
    }

    impl CudaDeviceProp {
        /// Device name (offset 0, 256 bytes, null-terminated).
        pub fn name(&self) -> &[u8] {
            &self._data[..256]
        }
        /// Total global memory in bytes (offset 288, 8 bytes).
        pub fn total_global_mem(&self) -> usize {
            usize::from_ne_bytes(self._data[288..296].try_into().unwrap())
        }
        /// Compute capability major (offset 360, 4 bytes).
        pub fn major(&self) -> c_int {
            c_int::from_ne_bytes(self._data[360..364].try_into().unwrap())
        }
        /// Compute capability minor (offset 364, 4 bytes).
        pub fn minor(&self) -> c_int {
            c_int::from_ne_bytes(self._data[364..368].try_into().unwrap())
        }
    }

    extern "C" {
        pub fn cudaGetDeviceCount(count: *mut c_int) -> CudaError;
        pub fn cudaSetDevice(device: c_int) -> CudaError;
        pub fn cudaGetDeviceProperties(prop: *mut CudaDeviceProp, device: c_int) -> CudaError;
        pub fn cudaMemGetInfo(free: *mut usize, total: *mut usize) -> CudaError;

        pub fn cudaMalloc(devPtr: *mut *mut c_void, size: usize) -> CudaError;
        pub fn cudaFree(devPtr: *mut c_void) -> CudaError;

        pub fn cudaMallocHost(ptr: *mut *mut c_void, size: usize) -> CudaError;
        pub fn cudaFreeHost(ptr: *mut c_void) -> CudaError;

        pub fn cudaMemcpy(
            dst: *mut c_void,
            src: *const c_void,
            count: usize,
            kind: c_int,
        ) -> CudaError;
        pub fn cudaMemcpyAsync(
            dst: *mut c_void,
            src: *const c_void,
            count: usize,
            kind: c_int,
            stream: CudaStreamT,
        ) -> CudaError;
        pub fn cudaMemset(devPtr: *mut c_void, value: c_int, count: usize) -> CudaError;
        pub fn cudaMemsetAsync(
            devPtr: *mut c_void,
            value: c_int,
            count: usize,
            stream: CudaStreamT,
        ) -> CudaError;

        pub fn cudaStreamCreate(pStream: *mut CudaStreamT) -> CudaError;
        pub fn cudaStreamSynchronize(stream: CudaStreamT) -> CudaError;
        pub fn cudaStreamDestroy(stream: CudaStreamT) -> CudaError;

        pub fn cudaEventCreateWithFlags(event: *mut CudaStreamT, flags: c_int) -> CudaError;
        pub fn cudaEventDestroy(event: CudaStreamT) -> CudaError;
        pub fn cudaEventRecord(event: CudaStreamT, stream: CudaStreamT) -> CudaError;
        pub fn cudaStreamWaitEvent(
            stream: CudaStreamT,
            event: CudaStreamT,
            flags: c_int,
        ) -> CudaError;

        pub fn cudaDeviceSynchronize() -> CudaError;
        pub fn cudaGetLastError() -> CudaError;
        pub fn cudaGetErrorString(error: CudaError) -> *const u8;
    }

    /// Convert CUDA error to a Rust string.
    pub fn error_string(err: CudaError) -> String {
        if err == CUDA_SUCCESS {
            return "success".to_string();
        }
        // SAFETY: cudaGetErrorString returns a static string pointer
        let ptr = unsafe { cudaGetErrorString(err) };
        if ptr.is_null() {
            return format!("CUDA error {}", err);
        }
        // SAFETY: the pointer is valid for the lifetime of the CUDA context
        let cstr = unsafe { std::ffi::CStr::from_ptr(ptr as *const std::ffi::c_char) };
        cstr.to_string_lossy().into_owned()
    }
}

// ─── CUDA Kernel Launcher FFI ─────────────────────────────────────────────

#[cfg(feature = "cuda")]
extern "C" {
    pub fn vib3_launch_partial_matmul_fp16(
        input: *const u8,
        weight: *const u8,
        output: *mut u8,
        k: i32,
        m_slice: i32,
        stream: *mut std::ffi::c_void,
    ) -> i32;

    /// Fast vectorized FP16 GEMV: uses float4 loads + shared memory input cache.
    /// Falls back to standard kernel if K is not aligned.
    pub fn vib3_launch_partial_matmul_fp16_fast(
        input: *const u8,
        weight: *const u8,
        output: *mut u8,
        k: i32,
        m_slice: i32,
        stream: *mut std::ffi::c_void,
    ) -> i32;

    pub fn vib3_launch_partial_matmul_int4(
        input: *const u8,
        weight_packed: *const u8,
        scales: *const u8,
        output: *mut u8,
        k: i32,
        m_slice: i32,
        group_size: i32,
        stream: *mut std::ffi::c_void,
    ) -> i32;

    pub fn vib3_launch_fused_swiglu_fp16(
        input: *const u8,
        up_weight: *const u8,
        gate_weight: *const u8,
        output: *mut u8,
        k: i32,
        m_slice: i32,
        stream: *mut std::ffi::c_void,
    ) -> i32;

    pub fn vib3_launch_silu_mul(
        up_result: *const u8,
        gate_result: *const u8,
        output: *mut u8,
        dim: i32,
        stream: *mut std::ffi::c_void,
    ) -> i32;

    pub fn vib3_launch_weighted_accumulate(
        output: *mut u8,
        expert_output: *const u8,
        weight: f32,
        dim: i32,
        stream: *mut std::ffi::c_void,
    ) -> i32;

    pub fn vib3_launch_weighted_accumulate_f32(
        output: *mut u8,
        expert_output: *const u8,
        weight: f32,
        dim: i32,
        stream: *mut std::ffi::c_void,
    ) -> i32;

    /// Weighted accumulate FP32→FP32: output[i] += weight * expert_output[i].
    /// Both input and output are FP32 (no FP16 truncation).
    pub fn vib3_launch_weighted_accumulate_f32_f32(
        output: *mut u8,
        expert_output: *const u8,
        weight: f32,
        dim: i32,
        stream: *mut std::ffi::c_void,
    ) -> i32;

    /// Weighted accumulate with device-side scalar: output[i] += dev_weight[0] * f32(expert[i]).
    /// dev_weight is a device pointer to a single FP32 scalar (avoids D2H stall).
    pub fn vib3_launch_weighted_accumulate_f32_dev_scalar(
        output: *mut u8,
        expert_output: *const u8,
        dev_weight: *const u8,
        dim: i32,
        stream: *mut std::ffi::c_void,
    ) -> i32;

    /// Sigmoid-gated FP16→FP32 accumulate: output[i] += sigmoid(gate_dev[0]) * f16(expert[i]).
    /// Gate scalar is read from device memory; sigmoid is computed on-GPU.
    /// Eliminates stream.synchronize() + D2H for the gate value.
    pub fn vib3_launch_sigmoid_gated_accumulate_f32(
        output: *mut u8,
        expert_output: *const u8,
        gate_dev: *const u8,
        dim: i32,
        stream: *mut std::ffi::c_void,
    ) -> i32;

    /// Sigmoid-gated accumulate with FP32 expert output (NVFP4 MMA path).
    /// output[i] += sigmoid(gate_dev[0]) * expert_output[i], all FP32.
    pub fn vib3_launch_sigmoid_gated_accumulate_f32in(
        output: *mut u8,
        expert_output: *const u8,
        gate_dev: *const u8,
        dim: i32,
        stream: *mut std::ffi::c_void,
    ) -> i32;

    /// FP32 SwiGLU fuse: output[i] = silu(gate[i]) * up[i], all FP32.
    pub fn vib3_launch_swiglu_fuse_f32(
        gate: *const u8,
        up: *const u8,
        output: *mut u8,
        dim: i32,
        stream: *mut std::ffi::c_void,
    ) -> i32;

    pub fn vib3_launch_residual_add_f32_f32(
        accumulator: *mut u8,
        layer_output: *const u8,
        dim: i32,
        stream: *mut std::ffi::c_void,
    ) -> i32;

    pub fn vib3_launch_router_gemv(
        hidden_state: *const u8,
        router_weights: *const u8,
        scores: *mut f32,
        hidden_dim: i32,
        num_experts: i32,
        stream: *mut std::ffi::c_void,
    ) -> i32;

    pub fn vib3_launch_rms_norm(
        x: *mut u8,
        weight: *const u8,
        hidden_dim: i32,
        eps: f32,
        stream: *mut std::ffi::c_void,
    ) -> i32;

    pub fn vib3_launch_rms_norm_no_weight(
        x: *mut u8,
        hidden_dim: i32,
        eps: f32,
        stream: *mut std::ffi::c_void,
    ) -> i32;

    // FP32 RMSNorm: FP32 input → FP32 output, FP16 weight
    pub fn vib3_launch_rms_norm_f32(
        input: *const u8,
        output: *mut u8,
        weight: *const u8,
        hidden_dim: i32,
        eps: f32,
        stream: *mut std::ffi::c_void,
    ) -> i32;

    // FP32→FP16 RMSNorm: FP32 input → normalize in FP32 → FP16 output
    pub fn vib3_launch_rms_norm_f32_to_f16(
        input: *const u8,
        output: *mut u8,
        weight: *const u8,
        hidden_dim: i32,
        eps: f32,
        stream: *mut std::ffi::c_void,
    ) -> i32;

    // FP32-input INT4 matmul
    pub fn vib3_launch_partial_matmul_int4_f32(
        input: *const u8,
        weight: *const u8,
        scales: *const u8,
        output: *mut u8,
        k: i32,
        m_slice: i32,
        group_size: i32,
        stream: *mut std::ffi::c_void,
    ) -> i32;

    // FP32-input FP16 matmul
    pub fn vib3_launch_partial_matmul_fp16_f32in(
        input: *const u8,
        weight: *const u8,
        output: *mut u8,
        k: i32,
        m_slice: i32,
        stream: *mut std::ffi::c_void,
    ) -> i32;

    // FP32-input FP16-weight matmul with FP32 output (avoids FP16 truncation)
    pub fn vib3_launch_partial_matmul_fp16_f32in_f32out(
        input: *const u8,
        weight: *const u8,
        output: *mut u8,
        k: i32,
        m_slice: i32,
        stream: *mut std::ffi::c_void,
    ) -> i32;

    // FP32-input fused SwiGLU INT4
    pub fn vib3_launch_fused_swiglu_int4_f32(
        input: *const u8,
        up_weight: *const u8,
        up_scales: *const u8,
        gate_weight: *const u8,
        gate_scales: *const u8,
        output: *mut u8,
        k: i32,
        m_slice: i32,
        group_size: i32,
        stream: *mut std::ffi::c_void,
    ) -> i32;

    // FP32-input fused SwiGLU FP16
    pub fn vib3_launch_fused_swiglu_fp16_f32in(
        input: *const u8,
        up_weight: *const u8,
        gate_weight: *const u8,
        output: *mut u8,
        k: i32,
        m_slice: i32,
        stream: *mut std::ffi::c_void,
    ) -> i32;

    // FP32-input router GEMV
    pub fn vib3_launch_router_gemv_f32(
        hidden_state: *const u8,
        router_weights: *const u8,
        scores: *mut f32,
        hidden_dim: i32,
        num_experts: i32,
        stream: *mut std::ffi::c_void,
    ) -> i32;

    // Fused router GEMV + GPU top-k (eliminates stream.synchronize()).
    // `bias` is the aux-loss-free expert selection bias (DeepSeek-V3 /
    // Kimi K2.6 exp_probs_b.bias), stored as FP16 in the .vib3 sidecar;
    // NULL for models without aux-free balancing.
    pub fn vib3_launch_router_topk(
        hidden_state: *const u8,
        router_weights: *const u8,
        scores_buf: *mut f32,
        bias: *const u8,       // FP16 [num_experts], or NULL
        out_ids: *mut u16,     // [top_k] device output
        out_weights: *mut f32, // [top_k] device output
        hidden_dim: i32,
        num_experts: i32,
        top_k: i32,
        scoring_mode: i32,   // 0=softmax, 1=sigmoid
        scaling_factor: f32, // sigmoid only
        stream: *mut std::ffi::c_void,
    ) -> i32;

    pub fn vib3_launch_embedding_lookup(
        table: *const u8,
        output: *mut u8,
        token_id: i32,
        hidden_dim: i32,
        stream: *mut std::ffi::c_void,
    ) -> i32;

    pub fn vib3_launch_residual_add(
        output: *mut u8,
        residual: *const u8,
        dim: i32,
        stream: *mut std::ffi::c_void,
    ) -> i32;

    pub fn vib3_launch_fused_residual_add(
        output: *mut u8,
        a: *const u8,
        b: *const u8,
        dim: i32,
        stream: *mut std::ffi::c_void,
    ) -> i32;

    // ── Decode attention kernels ──

    pub fn vib3_launch_rope_apply(
        q: *mut u8,
        k: *mut u8,
        head_dim: i32,
        num_q_heads: i32,
        num_kv_heads: i32,
        d_position: *const u8,
        rope_base: f32,
        stream: *mut std::ffi::c_void,
    ) -> i32;

    pub fn vib3_launch_kv_cache_append(
        k_cache: *mut u8,
        v_cache: *mut u8,
        new_k: *const u8,
        new_v: *const u8,
        max_seq_len: i32,
        head_dim: i32,
        num_kv_heads: i32,
        d_position: *const u8,
        stream: *mut std::ffi::c_void,
    ) -> i32;

    pub fn vib3_launch_decode_attention(
        q: *const u8,
        k_cache: *const u8,
        v_cache: *const u8,
        output: *mut u8,
        head_dim: i32,
        num_heads: i32,
        num_kv_heads: i32,
        d_position: *const u8,
        max_seq_len: i32,
        scale: f32,
        stream: *mut std::ffi::c_void,
    ) -> i32;

    pub fn vib3_launch_mla_decode_attention(
        q_absorbed: *const u8,
        q_rope: *const u8,
        kv_latent: *const u8,
        k_rope_cache: *const u8,
        v_latent_out: *mut u8,
        kv_lora_rank: i32,
        qk_rope_dim: i32,
        num_heads: i32,
        seq_len: i32,
        scale: f32,
        stream: *mut std::ffi::c_void,
    ) -> i32;

    pub fn vib3_launch_mla_v_reconstruct(
        v_latent: *const u8,
        kv_b_proj: *const u8,
        v_out: *mut u8,
        kv_lora_rank: i32,
        qk_nope_dim: i32,
        v_head_dim: i32,
        num_heads: i32,
        stream: *mut std::ffi::c_void,
    ) -> i32;

    pub fn vib3_launch_mla_q_absorb_rope(
        q_full: *const u8,
        kv_b_proj: *const u8,
        rope_freqs: *const u8,
        q_absorbed_out: *mut u8,
        q_rope_out: *mut u8,
        q_head_dim: i32,
        qk_nope_dim: i32,
        qk_rope_dim: i32,
        v_head_dim: i32,
        kv_lora_rank: i32,
        num_heads: i32,
        position: i32,
        stream: *mut std::ffi::c_void,
    ) -> i32;

    pub fn vib3_launch_mla_kv_cache_append(
        kv_a_out: *const u8,
        kv_norm_weight: *const u8,
        rope_freqs: *const u8,
        kv_latent_cache: *mut u8,
        k_rope_cache: *mut u8,
        kv_lora_rank: i32,
        qk_rope_dim: i32,
        position: i32,
        eps: f32,
        stream: *mut std::ffi::c_void,
    ) -> i32;

    pub fn vib3_launch_f32_to_f16(
        input: *const u8,
        output: *mut u8,
        n: i32,
        stream: *mut std::ffi::c_void,
    ) -> i32;

    pub fn vib3_launch_fp16_to_fp32(
        input: *const u8,
        output: *mut u8,
        n: i32,
        stream: *mut std::ffi::c_void,
    ) -> i32;

    pub fn vib3_launch_residual_add_fp32(
        accumulator: *mut u8,
        layer_output: *const u8,
        dim: i32,
        stream: *mut std::ffi::c_void,
    ) -> i32;

    // NVFP4 (MXFP4 E2M1) kernels — FP32 input, FP16 output
    pub fn vib3_launch_partial_matmul_nvfp4(
        input: *const u8,
        weight_packed: *const u8,
        scales: *const u8,
        output: *mut u8,
        k: i32,
        m_slice: i32,
        block_size: i32,
        stream: *mut std::ffi::c_void,
    ) -> i32;

    pub fn vib3_launch_fused_swiglu_nvfp4(
        input: *const u8,
        up_weight: *const u8,
        up_scales: *const u8,
        gate_weight: *const u8,
        gate_scales: *const u8,
        output: *mut u8,
        k: i32,
        m_slice: i32,
        block_size: i32,
        stream: *mut std::ffi::c_void,
    ) -> i32;

    // NVFP4 fused SwiGLU with FP32 output (eliminates FP16 intermediate truncation)
    pub fn vib3_launch_fused_swiglu_nvfp4_f32out(
        input: *const u8,
        up_weight: *const u8,
        up_scales: *const u8,
        gate_weight: *const u8,
        gate_scales: *const u8,
        output: *mut u8,
        k: i32,
        m_slice: i32,
        block_size: i32,
        stream: *mut std::ffi::c_void,
    ) -> i32;

    // NVFP4 matmul with FP32 input AND FP32 output (for down_proj with FP32 intermediate)
    pub fn vib3_launch_partial_matmul_nvfp4_f32out(
        input: *const u8,
        weight_packed: *const u8,
        scales: *const u8,
        output: *mut u8,
        k: i32,
        m_slice: i32,
        block_size: i32,
        stream: *mut std::ffi::c_void,
    ) -> i32;

    // NVFP4 FP16-input variant (for down-projection where intermediate is FP16)
    pub fn vib3_launch_partial_matmul_nvfp4_fp16in(
        input: *const u8,
        weight_packed: *const u8,
        scales: *const u8,
        output: *mut u8,
        k: i32,
        m_slice: i32,
        block_size: i32,
        stream: *mut std::ffi::c_void,
    ) -> i32;

    // ─── Blackwell MMA NVFP4 kernels (SM_120+) ────────────────────────────

    /// GEMV via Blackwell Tensor Core MMA: output[M_slice] = weight[M_slice,K] × input[K].
    /// Uses m16n8k64 block-scaled FP4 MMA. Weight format: sequential nibbles + BF16 scales.
    /// Repacks to split-half + E8M0 at tile-load time. FP32 input and output.
    pub fn vib3_launch_gemv_mma_nvfp4(
        input: *const u8,
        weight_packed: *const u8,
        scales: *const u8,
        output: *mut u8,
        k: i32,
        m_slice: i32,
        stream: *mut std::ffi::c_void,
    ) -> i32;

    /// Fused SwiGLU via Blackwell MMA: output = SiLU(gate·input) × (up·input).
    /// Two GEMV MMA passes (up + gate) with shared input quantization, fused activation.
    pub fn vib3_launch_fused_swiglu_mma_nvfp4(
        input: *const u8,
        up_weight: *const u8,
        up_scales: *const u8,
        gate_weight: *const u8,
        gate_scales: *const u8,
        output: *mut u8,
        k: i32,
        m_slice: i32,
        stream: *mut std::ffi::c_void,
    ) -> i32;

    /// Pre-quantize FP32 activation vector to FP4 E2M1 split-half + E8M0 scales.
    /// Done once per MoE layer, reused across all expert kernels.
    pub fn vib3_launch_quantize_activation_fp4(
        input: *const u8,    // [K] FP32
        act_fp4: *mut u8,    // [K/2] split-half FP4 output
        act_scales: *mut u8, // [K/32] E8M0 scales output
        k: i32,
        stream: *mut std::ffi::c_void,
    ) -> i32;

    /// MMA GEMV with pre-quantized activations. Skips FP32→FP4 quantization.
    pub fn vib3_launch_gemv_mma_nvfp4_preq(
        act_fp4: *const u8,    // [K/2] pre-quantized split-half FP4
        act_scales: *const u8, // [K/32] E8M0 scales
        weight_packed: *const u8,
        scales: *const u8,
        output: *mut u8,
        k: i32,
        m_slice: i32,
        stream: *mut std::ffi::c_void,
    ) -> i32;

    /// Fused SwiGLU MMA with pre-quantized activations.
    pub fn vib3_launch_fused_swiglu_mma_nvfp4_preq(
        act_fp4: *const u8,
        act_scales: *const u8,
        up_weight: *const u8,
        up_scales: *const u8,
        gate_weight: *const u8,
        gate_scales: *const u8,
        output: *mut u8,
        k: i32,
        m_slice: i32,
        stream: *mut std::ffi::c_void,
    ) -> i32;

    // ─── Optimized launchers (cached capability, no per-launch API overhead) ───

    /// In-place repack weight data from sequential to split-half format.
    pub fn vib3_launch_repack_weights_inplace(
        weight_data_ptr: *mut u8,
        weight_data_bytes: i64,
        stream: *mut std::ffi::c_void,
    ) -> i32;

    /// Batched expert: single C call for full expert pipeline.
    /// SwiGLU(up,gate) → quantize → down_proj → weighted accumulate.
    /// Uses norepack kernels (weights must be pre-repacked to split-half).
    pub fn vib3_launch_expert_batched(
        act_fp4: *const u8,
        act_scales: *const u8,
        up_weight: *const u8,
        up_scales: *const u8,
        gate_weight: *const u8,
        gate_scales: *const u8,
        down_weight: *const u8,
        down_scales: *const u8,
        layer_output: *mut f32,
        expert_weight: f32,
        k_in: i32,
        m_mid: i32,
        k_mid: i32,
        m_out: i32,
        stream: *mut std::ffi::c_void,
    ) -> i32;

    /// Fast MMA GEMV preq (cached capability check).
    pub fn vib3_launch_gemv_mma_nvfp4_preq_fast(
        act_fp4: *const u8,
        act_scales: *const u8,
        weight_packed: *const u8,
        scales: *const u8,
        output: *mut u8,
        k: i32,
        m_slice: i32,
        stream: *mut std::ffi::c_void,
    ) -> i32;

    /// Scalar GEMV with K-dimension parallelism for maximum bandwidth.
    /// 1 row per block, all threads cooperate along K → coalesced access.
    pub fn vib3_launch_gemv_scalar_nvfp4(
        act_fp4: *const u8,
        act_scales: *const u8,
        weight_packed: *const u8,
        scales: *const u8,
        output: *mut u8,
        k: i32,
        m_slice: i32,
        stream: *mut std::ffi::c_void,
    ) -> i32;

    /// Repack weight data from row-major split-half to tiled layout.
    /// Tiled layout stores 16 consecutive rows' data for each K-tile contiguously
    /// for perfectly coalesced MMA GEMV loads.
    /// temp_buf must be at least M * (K/2) bytes.
    pub fn vib3_launch_repack_row_to_tiled(
        weight_data_ptr: *mut u8,
        temp_buf: *mut u8,
        m: i32,
        k: i32,
        stream: *mut std::ffi::c_void,
    ) -> i32;

    /// Tiled MMA GEMV: reads weights from tiled layout for coalesced access.
    /// Weights must have been repacked via vib3_launch_repack_row_to_tiled.
    /// Scales remain in row-major layout.
    pub fn vib3_launch_gemv_mma_nvfp4_tiled(
        act_fp4: *const u8,
        act_scales: *const u8,
        weight: *const u8,
        scales: *const u8,
        output: *mut u8,
        k: i32,
        m_slice: i32,
        stream: *mut std::ffi::c_void,
    ) -> i32;

    /// Batched multi-matrix MMA GEMV with shared pre-quantized activation.
    /// Processes num_matrices weight matrices in a single kernel launch.
    /// All matrices share the same FP4 activation and K dimension.
    pub fn vib3_launch_batched_gemv_mma_nvfp4_preq(
        act_fp4: *const u8,
        act_scales: *const u8,
        num_matrices: i32,
        weight_pages: *const *const u8, // [num_matrices] NVFP4 packed (data+scales)
        m_slices: *const i32,           // [num_matrices] M per matrix
        outputs: *const *mut u8,        // [num_matrices] FP32 output ptrs
        k: i32,
        stream: *mut std::ffi::c_void,
    ) -> i32;

    /// Batched tiled MMA GEMV with K-parallel decomposition + atomicAdd.
    /// Up to 5 matrices sharing the same FP4 activation and K dimension.
    /// Weights must be in TILED layout. Outputs are pre-zeroed by the launcher.
    pub fn vib3_launch_batched_gemv_mma_nvfp4_tiled(
        act_fp4: *const u8,
        act_scales: *const u8,
        num_matrices: i32,
        tiled_weights: *const *const u8, // [num_matrices] tiled FP4 data ptrs
        scale_ptrs: *const *const u8,    // [num_matrices] BF16 scale ptrs (row-major)
        output_ptrs: *const *mut u8,     // [num_matrices] FP32 output ptrs
        m_slices: *const i32,            // [num_matrices] M per matrix
        k: i32,
        stream: *mut std::ffi::c_void,
    ) -> i32;

    /// Fast activation quantization (cached capability check).
    pub fn vib3_launch_quantize_activation_fp4_fast(
        input: *const u8,
        act_fp4: *mut u8,
        act_scales: *mut u8,
        k: i32,
        stream: *mut std::ffi::c_void,
    ) -> i32;

    /// Fused RMSNorm + FP4 quantize: reads FP32 hidden state, applies RMSNorm
    /// with FP16 weight, then directly quantizes to split-half FP4 + E8M0 scales.
    /// opt_f16_out may be null to skip FP16 output production.
    pub fn vib3_launch_fused_rms_norm_quantize_fp4(
        input: *const u8,       // [K] FP32 hidden state
        norm_weight: *const u8, // [K] FP16 norm weight
        act_fp4: *mut u8,       // [K/2] split-half FP4 output
        act_scales: *mut u8,    // [K/32] E8M0 scales output
        opt_f16_out: *mut u8,   // [K] optional FP16 output (null to skip)
        k: i32,
        eps: f32,
        stream: *mut std::ffi::c_void,
    ) -> i32;

    /// Fast fused SwiGLU MMA preq (cached capability, direct kernel launches).
    pub fn vib3_launch_fused_swiglu_mma_nvfp4_preq_fast(
        act_fp4: *const u8,
        act_scales: *const u8,
        up_weight: *const u8,
        up_scales: *const u8,
        gate_weight: *const u8,
        gate_scales: *const u8,
        output: *mut u8,
        k: i32,
        m_slice: i32,
        stream: *mut std::ffi::c_void,
    ) -> i32;

    /// Fused multi-expert MoE layer: processes all selected experts (up to 8)
    /// with 4 kernel launches instead of 8 × (6 kernels + 1 memset).
    /// Uses multi-expert GEMV batching across blockIdx.y for SM utilization.
    pub fn vib3_launch_moe_experts_fused(
        act_fp4: *const u8,
        act_scales: *const u8,
        up_weight_ptrs: *const *const u8, // [num_experts] device ptrs
        up_scale_ptrs: *const *const u8,
        gate_weight_ptrs: *const *const u8,
        gate_scale_ptrs: *const *const u8,
        down_weight_ptrs: *const *const u8,
        down_scale_ptrs: *const *const u8,
        expert_weights_host: *const f32, // [num_experts] routing weights
        num_experts: i32,
        k_in: i32,
        m_mid: i32,
        k_mid: i32,
        m_out: i32,
        layer_output: *mut f32,
        stream: *mut std::ffi::c_void,
    ) -> i32;

    /// GPU-only fused multi-expert MoE layer: reads expert IDs and weights
    /// directly from device memory (written by router), looks up weight page
    /// pointers from a prebuilt device-side table. Zero host synchronization.
    pub fn vib3_launch_moe_experts_fused_gpu(
        act_fp4: *const u8,
        act_scales: *const u8,
        page_table: *const u64,     // device: [layers * experts * 3]
        expert_ids: *const u16,     // device: [num_active] from router
        expert_weights: *const f32, // device: [num_active] from router
        layer: i32,
        num_experts_total: i32,
        num_active: i32,
        k_in: i32,
        m_mid: i32,
        k_mid: i32,
        m_out: i32,
        layer_output: *mut f32,
        stream: *mut std::ffi::c_void,
    ) -> i32;

    /// Batched INT4 MoE layer (multi-page). Collapses per-expert+per-page
    /// launches (8 experts × N pages × 3 stages) into exactly 2 kernel launches:
    /// one batched SwiGLU and one batched down·weighted_atomic_accumulate.
    /// `layer_output` must be pre-zeroed FP32.
    pub fn vib3_launch_moe_int4_experts_fused(
        input_f32: *const u8,
        // SwiGLU tasks (each = one up/gate page pair)
        sw_up_w: *const *const u8,
        sw_up_s: *const *const u8,
        sw_gate_w: *const *const u8,
        sw_gate_s: *const *const u8,
        sw_expert_slots: *const i32,
        sw_row_starts: *const i32,
        sw_m_slices: *const i32,
        num_sw_tasks: i32,
        max_sw_m_slice: i32,
        // Down tasks (each = one down page)
        dn_w: *const *const u8,
        dn_s: *const *const u8,
        dn_expert_slots: *const i32,
        dn_row_starts: *const i32,
        dn_m_slices: *const i32,
        dn_expert_weights_host: *const f32,
        num_dn_tasks: i32,
        max_dn_m_slice: i32,
        // Dims
        k_in: i32,
        m_mid: i32,
        k_mid: i32,
        group_size: i32,
        layer_output: *mut f32,
        stream: *mut std::ffi::c_void,
    ) -> i32;

    /// CUTLASS 4.2.1 Blackwell MLA decode (batch=1, single-page). Expects
    /// separate FP16 buffers for q_nope/q_pe/kv_c/k_pe and writes FP32 output
    /// in latent space [H, D_latent=512]. Caller applies V-up-projection and
    /// o_proj afterwards.
    ///
    /// `num_heads` must match the tile-shape H used in cutlass_mla.cu
    /// (currently 128 — callers with fewer heads must pad Q by zeros).
    /// `seq_len` == current sequence length; `page_count_total=1`,
    /// `page_size = seq_len` for the single-page shortcut.
    /// Returns 0 on success, negative on error.
    pub fn vib3_launch_cutlass_mla_decode(
        q_nope: *const u8,
        q_pe: *const u8,
        kv_c: *const u8,
        k_pe: *const u8,
        seq_lens_device: *const i32,
        page_table_device: *const i32,
        out: *mut u8,
        lse: *mut u8, // may be null
        workspace: *mut u8,
        num_heads: i32,
        seq_len: i32,
        page_count_total: i32,
        page_size: i32,
        sm_scale: f32,
        num_kv_splits: i32,
        sm_count: i32,
        stream: *mut std::ffi::c_void,
    ) -> i32;

    /// Workspace bytes required by the CUTLASS MLA decode kernel. Caller
    /// allocates this much scratch and passes to the launcher.
    pub fn vib3_cutlass_mla_workspace_size(
        num_heads: i32,
        max_seq_len: i32,
        num_batches: i32,
        sm_count: i32,
        num_kv_splits: i32,
    ) -> usize;

    /// Simple element-wise FP32 → FP16 conversion (used to stage Q/KV for
    /// the CUTLASS MLA kernel from vib3's FP32 buffers).
    pub fn vib3_mla_f32_to_f16(
        input: *const u8,
        output: *mut u8,
        n: i32,
        stream: *mut std::ffi::c_void,
    ) -> i32;

    /// Simple element-wise FP16 → FP32 conversion.
    pub fn vib3_mla_f16_to_f32(
        input: *const u8,
        output: *mut u8,
        n: i32,
        stream: *mut std::ffi::c_void,
    ) -> i32;

    /// Convert FP16 weight matrix [M, K] to NVFP4 MMA format at runtime.
    /// out_data must be M * K/2 bytes, out_scales must be M * (K/32) * 2 bytes.
    /// Output is split-half packed FP4 data + BF16 block scales, ready for MMA GEMV.
    pub fn vib3_launch_fp16_to_nvfp4_weight(
        input: *const u8,    // [M, K] FP16
        out_data: *mut u8,   // [M * K/2] FP4 data
        out_scales: *mut u8, // [M * (K/32) * 2] BF16 scales
        m: i32,
        k: i32,
        stream: *mut std::ffi::c_void,
    ) -> i32;

    // ─── DeltaNet kernels (Qwen3.5 Gated Delta Rule) ────────────────────

    /// L2 normalization: output[i] = input[i] / ||input[i]||_2.
    /// Each of `num_vecs` vectors of length `dim` is normalized independently.
    pub fn vib3_launch_l2_norm(
        input: *const u8,
        output: *mut u8,
        num_vecs: i32,
        dim: i32,
        eps: f32,
        stream: *mut std::ffi::c_void,
    ) -> i32;

    /// Causal depthwise 1D convolution with SiLU activation.
    /// Processes one new token, updates `conv_state` in-place.
    /// conv_state: [num_channels, kernel_size-1], conv_weight: [num_channels, kernel_size].
    pub fn vib3_launch_causal_conv1d(
        conv_state: *mut u8,
        new_input: *const u8,
        conv_weight: *const u8,
        output: *mut u8,
        num_channels: i32,
        kernel_size: i32,
        stream: *mut std::ffi::c_void,
    ) -> i32;

    /// Fused DeltaNet autoregressive step for all heads.
    /// Per head: decay → retrieve → delta → update → query.
    /// state: [num_heads, vdim, vdim] (mutable), q/k/v: [num_heads, vdim],
    /// gate: [num_heads] (scalar decay), beta: [num_heads] (write strength).
    pub fn vib3_launch_deltanet_step(
        state: *mut u8,
        q: *const u8,
        k: *const u8,
        v: *const u8,
        gate: *const u8,
        beta: *const u8,
        output: *mut u8,
        num_heads: i32,
        vdim: i32,
        stream: *mut std::ffi::c_void,
    ) -> i32;

    /// Gated RMSNorm: output = RMSNorm(x, weight) * SiLU(gate).
    /// Per-head normalization: `num_groups` independent heads, each over `group_dim` elements.
    /// weight has shape [norm_dim] and cycles across each head's `group_dim`.
    pub fn vib3_launch_gated_rmsnorm(
        x: *const u8,
        gate: *const u8,
        weight: *const u8,
        output: *mut u8,
        num_groups: i32,
        group_dim: i32,
        norm_dim: i32,
        eps: f32,
        stream: *mut std::ffi::c_void,
    ) -> i32;

    /// Element-wise sigmoid: output[i] = 1 / (1 + exp(-input[i])).
    pub fn vib3_launch_sigmoid(
        input: *const u8,
        output: *mut u8,
        n: i32,
        stream: *mut std::ffi::c_void,
    ) -> i32;

    /// DeltaNet gate computation: gate = -exp(A_log) * softplus(alpha + dt_bias).
    pub fn vib3_launch_deltanet_gate(
        alpha: *const u8,
        dt_bias: *const u8,
        a_log: *const u8,
        gate_out: *mut u8,
        num_heads: i32,
        stream: *mut std::ffi::c_void,
    ) -> i32;

    /// In-place FP32 scale: data[i] *= scale.
    pub fn vib3_launch_scale_f32(
        data: *mut u8,
        n: i32,
        scale: f32,
        stream: *mut std::ffi::c_void,
    ) -> i32;

    /// Repeat-interleave FP32 vectors: expand [num_groups, dim] → [num_groups*repeat, dim].
    /// Input and output must NOT alias.
    /// Result: [G0,G0,...,G1,G1,...] (each group repeated contiguously).
    pub fn vib3_launch_repeat_interleave_f32(
        input: *const u8,
        output: *mut u8,
        num_groups: i32,
        dim: i32,
        repeat: i32,
        stream: *mut std::ffi::c_void,
    ) -> i32;

    /// Tiled repeat FP32 vectors: expand [num_groups, dim] → [num_groups*repeat, dim].
    /// Input and output must NOT alias.
    /// Result: [G0,G1,...,G_{n-1}, G0,G1,...,G_{n-1}, ...] (all groups tiled).
    /// Matches the V-head tiled order produced by llama.cpp's _LinearAttentionVReorderBase.
    pub fn vib3_launch_repeat_tile_f32(
        input: *const u8,
        output: *mut u8,
        num_groups: i32,
        dim: i32,
        repeat: i32,
        stream: *mut std::ffi::c_void,
    ) -> i32;

    // ─── Qwen3.5 gated attention kernels ────────────────────────────────

    /// Per-head RMSNorm in-place (FP16). Normalizes each head independently.
    /// data: [num_heads * stride] FP16, weight: [head_dim] FP16.
    /// stride = distance between heads in elements (2*head_dim for interleaved Q+gate,
    /// head_dim for contiguous K).
    pub fn vib3_launch_per_head_rmsnorm(
        data: *mut u8,
        weight: *const u8,
        head_dim: i32,
        stride: i32,
        num_heads: i32,
        eps: f32,
        stream: *mut std::ffi::c_void,
    ) -> i32;

    /// Partial RoPE in-place (FP16). Applies rotary embeddings to only the
    /// first `rope_dim` dimensions of each head, leaving the rest unchanged.
    /// data: [num_heads, head_dim] with given stride between heads.
    pub fn vib3_launch_partial_rope(
        data: *mut u8,
        head_dim: i32,
        rope_dim: i32,
        stride: i32,
        num_heads: i32,
        d_position: *const u8,
        rope_base: f32,
        stream: *mut std::ffi::c_void,
    ) -> i32;

    /// Sigmoid-gated multiply (FP16): output[i] = input[i] * sigmoid(gate[i]).
    /// output can alias input for in-place gating.
    pub fn vib3_launch_sigmoid_mul_f16(
        output: *mut u8,
        input: *const u8,
        gate: *const u8,
        n: i32,
        stream: *mut std::ffi::c_void,
    ) -> i32;

    /// Deinterleave FP16: splits [A0(chunk), B0(chunk), A1(chunk), B1(chunk), ...]
    /// into output_a = [A0, A1, ...] and output_b = [B0, B1, ...].
    /// Used for Qwen3.5 Q+gate extraction from doubled Q projection.
    pub fn vib3_launch_deinterleave_f16(
        input: *const u8,
        output_a: *mut u8,
        output_b: *mut u8,
        chunk_size: i32,
        num_chunks: i32,
        stream: *mut std::ffi::c_void,
    ) -> i32;

    // ─── MoE GPU buffer pre-warming ─────────────────────────────────────

    /// Pre-allocate all MoE intermediate buffers and upload buffer pointers
    /// to device. Must be called BEFORE graph capture to avoid cudaMalloc
    /// and synchronous cudaMemcpy during capture.
    pub fn vib3_moe_prewarm_gpu_bufs(m_mid: i32, k_mid: i32) -> i32;

    // ─── CUDA Graph helpers ──────────────────────────────────────────────

    /// Begin CUDA stream capture for graph recording.
    pub fn vib3_cuda_graph_begin_capture(stream: *mut std::ffi::c_void) -> i32;

    /// Check stream capture status: 0=none, 1=active, 2=invalidated, -1=error.
    pub fn vib3_cuda_stream_capture_status(stream: *mut std::ffi::c_void) -> i32;

    /// End stream capture, returns opaque graph handle.
    pub fn vib3_cuda_graph_end_capture(
        stream: *mut std::ffi::c_void,
        graph_out: *mut *mut std::ffi::c_void,
    ) -> i32;

    /// Instantiate graph into executable form.
    pub fn vib3_cuda_graph_instantiate(
        exec_out: *mut *mut std::ffi::c_void,
        graph: *mut std::ffi::c_void,
    ) -> i32;

    /// Launch a previously instantiated graph on a stream.
    pub fn vib3_cuda_graph_launch(
        exec: *mut std::ffi::c_void,
        stream: *mut std::ffi::c_void,
    ) -> i32;

    /// Destroy a graph exec handle.
    pub fn vib3_cuda_graph_exec_destroy(exec: *mut std::ffi::c_void) -> i32;

    /// Destroy a graph handle.
    pub fn vib3_cuda_graph_destroy(graph: *mut std::ffi::c_void) -> i32;

    /// Update a device-side int32 scalar via async H2D memcpy.
    pub fn vib3_update_device_int32(
        d_ptr: *mut u8,
        value: i32,
        stream: *mut std::ffi::c_void,
    ) -> i32;
}

// ─── CudaDevice ──────────────────────────────────────────────────────────

/// CUDA device handle.
pub struct CudaDevice {
    device_id: i32,
    total_mem: usize,
    free_mem: usize,
    name: String,
    is_real_cuda: bool,
}

impl CudaDevice {
    /// Initialize and query a CUDA device.
    /// Falls back to a virtual CPU device if CUDA is not available.
    pub fn new(device_id: i32) -> Result<Self> {
        #[cfg(feature = "cuda")]
        {
            match Self::try_cuda_init(device_id) {
                Ok(dev) => {
                    CUDA_AVAILABLE.store(true, Ordering::Release);
                    return Ok(dev);
                }
                Err(e) => {
                    tracing::warn!("CUDA init failed, using CPU fallback: {}", e);
                }
            }
        }

        // CPU fallback: report system memory as "VRAM"
        let ram = detect_system_ram();
        let free = (ram as f64 * 0.9) as usize;
        Ok(Self {
            device_id,
            total_mem: ram,
            free_mem: free,
            name: "CPU Fallback (no GPU)".into(),
            is_real_cuda: false,
        })
    }

    #[cfg(feature = "cuda")]
    fn try_cuda_init(device_id: i32) -> Result<Self> {
        unsafe {
            // Check device count
            let mut count: i32 = 0;
            let err = cuda_rt::cudaGetDeviceCount(&mut count);
            if err != cuda_rt::CUDA_SUCCESS || count == 0 {
                return Err(Error::NoCudaDevice);
            }
            if device_id >= count {
                return Err(Error::Cuda(format!(
                    "Device {} requested but only {} devices available",
                    device_id, count
                )));
            }

            // Set active device
            let err = cuda_rt::cudaSetDevice(device_id);
            if err != cuda_rt::CUDA_SUCCESS {
                return Err(Error::Cuda(format!(
                    "cudaSetDevice({}) failed: {}",
                    device_id,
                    cuda_rt::error_string(err)
                )));
            }

            // Get device properties
            let mut prop: cuda_rt::CudaDeviceProp = std::mem::zeroed();
            let err = cuda_rt::cudaGetDeviceProperties(&mut prop, device_id);
            if err != cuda_rt::CUDA_SUCCESS {
                return Err(Error::Cuda(format!(
                    "cudaGetDeviceProperties failed: {}",
                    cuda_rt::error_string(err)
                )));
            }

            // Extract device name (null-terminated C string)
            let name_bytes = prop.name();
            let name_len = name_bytes.iter().position(|&c| c == 0).unwrap_or(255);
            let name = String::from_utf8_lossy(&name_bytes[..name_len]).to_string();

            // Get memory info
            let mut free: usize = 0;
            let mut total: usize = 0;
            let err = cuda_rt::cudaMemGetInfo(&mut free, &mut total);
            if err != cuda_rt::CUDA_SUCCESS {
                // Fall back to property value
                total = prop.total_global_mem();
                free = total;
            }

            tracing::info!(
                "CUDA device {}: {} (compute {}.{}, {:.1} GB VRAM, {:.1} GB free)",
                device_id,
                name,
                prop.major(),
                prop.minor(),
                total as f64 / (1024.0 * 1024.0 * 1024.0),
                free as f64 / (1024.0 * 1024.0 * 1024.0),
            );

            Ok(Self {
                device_id,
                total_mem: total,
                free_mem: free,
                name,
                is_real_cuda: true,
            })
        }
    }

    pub fn device_id(&self) -> i32 {
        self.device_id
    }
    pub fn total_mem(&self) -> usize {
        self.total_mem
    }
    pub fn free_mem(&self) -> usize {
        self.free_mem
    }
    pub fn name(&self) -> &str {
        &self.name
    }
    pub fn is_real_cuda(&self) -> bool {
        self.is_real_cuda
    }
}

// ─── CudaStream ──────────────────────────────────────────────────────────

/// CUDA stream for async operations.
pub struct CudaStream {
    #[cfg(feature = "cuda")]
    raw: Option<cuda_rt::CudaStreamT>,
    is_real: bool,
}

impl CudaStream {
    pub fn new(device: &CudaDevice) -> Result<Self> {
        #[cfg(feature = "cuda")]
        {
            if device.is_real_cuda() {
                unsafe {
                    let mut stream: cuda_rt::CudaStreamT = std::ptr::null_mut();
                    let err = cuda_rt::cudaStreamCreate(&mut stream);
                    if err != cuda_rt::CUDA_SUCCESS {
                        return Err(Error::Cuda(format!(
                            "cudaStreamCreate failed: {}",
                            cuda_rt::error_string(err)
                        )));
                    }
                    return Ok(Self {
                        raw: Some(stream),
                        is_real: true,
                    });
                }
            }
        }

        Ok(Self {
            #[cfg(feature = "cuda")]
            raw: None,
            is_real: false,
        })
    }

    /// Create a CPU-only stream (always uses CPU fallback, even when CUDA is available).
    /// Useful for tests that pass host-allocated buffers to kernel functions.
    pub fn cpu_only() -> Self {
        Self {
            #[cfg(feature = "cuda")]
            raw: None,
            is_real: false,
        }
    }

    /// Synchronize — block until all operations on this stream complete.
    pub fn synchronize(&self) -> Result<()> {
        #[cfg(feature = "cuda")]
        {
            if let Some(stream) = self.raw {
                unsafe {
                    let err = cuda_rt::cudaStreamSynchronize(stream);
                    if err != cuda_rt::CUDA_SUCCESS {
                        return Err(Error::Cuda(format!(
                            "cudaStreamSynchronize failed: {}",
                            cuda_rt::error_string(err)
                        )));
                    }
                }
            }
        }
        Ok(())
    }

    pub fn is_real(&self) -> bool {
        self.is_real
    }

    /// Get the raw CUDA stream pointer (for passing to kernel launchers).
    /// Returns null for CPU fallback streams.
    #[cfg(feature = "cuda")]
    pub fn raw_ptr(&self) -> *mut std::ffi::c_void {
        self.raw.unwrap_or(std::ptr::null_mut())
    }

    /// CPU fallback: always returns null.
    #[cfg(not(feature = "cuda"))]
    pub fn raw_ptr(&self) -> *mut std::ffi::c_void {
        std::ptr::null_mut()
    }
}

// SAFETY: CUDA streams are thread-safe handles. cudaStreamSynchronize and
// kernel launches on a stream can be called from any thread. The raw pointer
// is an opaque GPU driver handle, not a host memory pointer.
unsafe impl Send for CudaStream {}
unsafe impl Sync for CudaStream {}

impl Drop for CudaStream {
    fn drop(&mut self) {
        #[cfg(feature = "cuda")]
        {
            if let Some(stream) = self.raw.take() {
                if !stream.is_null() {
                    // SAFETY: we own this stream and it hasn't been destroyed
                    unsafe {
                        cuda_rt::cudaStreamDestroy(stream);
                    }
                }
            }
        }
    }
}

// ─── CudaEvent ──────────────────────────────────────────────────────────

/// CUDA event for cross-stream synchronization.
pub struct CudaEvent {
    #[cfg(feature = "cuda")]
    raw: Option<cuda_rt::CudaStreamT>, // CudaEventT has same pointer type
}

impl CudaEvent {
    /// Create a new CUDA event (with cudaEventDisableTiming for lower overhead).
    pub fn new() -> Result<Self> {
        #[cfg(feature = "cuda")]
        {
            if is_cuda_available() {
                unsafe {
                    let mut event: cuda_rt::CudaStreamT = std::ptr::null_mut();
                    // Flag 0x02 = cudaEventDisableTiming (lower overhead)
                    let err = cuda_rt::cudaEventCreateWithFlags(&mut event, 0x02);
                    if err != cuda_rt::CUDA_SUCCESS {
                        return Err(Error::Cuda(format!(
                            "cudaEventCreateWithFlags failed: {}",
                            cuda_rt::error_string(err)
                        )));
                    }
                    return Ok(Self { raw: Some(event) });
                }
            }
        }
        Ok(Self {
            #[cfg(feature = "cuda")]
            raw: None,
        })
    }

    /// Record this event on a stream. All work previously enqueued on the stream
    /// will be completed before this event is "reached".
    pub fn record(&self, stream: &CudaStream) -> Result<()> {
        #[cfg(feature = "cuda")]
        {
            if let (Some(event), true) = (self.raw, stream.is_real()) {
                unsafe {
                    let err = cuda_rt::cudaEventRecord(event, stream.raw_ptr());
                    if err != cuda_rt::CUDA_SUCCESS {
                        return Err(Error::Cuda(format!(
                            "cudaEventRecord failed: {}",
                            cuda_rt::error_string(err)
                        )));
                    }
                }
            }
        }
        Ok(())
    }

    /// Make a stream wait for this event. The stream will not execute any
    /// subsequently enqueued work until this event is completed.
    pub fn wait_on_stream(&self, stream: &CudaStream) -> Result<()> {
        #[cfg(feature = "cuda")]
        {
            if let (Some(event), true) = (self.raw, stream.is_real()) {
                unsafe {
                    let err = cuda_rt::cudaStreamWaitEvent(stream.raw_ptr(), event, 0);
                    if err != cuda_rt::CUDA_SUCCESS {
                        return Err(Error::Cuda(format!(
                            "cudaStreamWaitEvent failed: {}",
                            cuda_rt::error_string(err)
                        )));
                    }
                }
            }
        }
        Ok(())
    }
}

unsafe impl Send for CudaEvent {}
unsafe impl Sync for CudaEvent {}

impl Drop for CudaEvent {
    fn drop(&mut self) {
        #[cfg(feature = "cuda")]
        {
            if let Some(event) = self.raw.take() {
                if !event.is_null() {
                    unsafe {
                        cuda_rt::cudaEventDestroy(event);
                    }
                }
            }
        }
    }
}

// ─── Memory Allocation ──────────────────────────────────────────────────

/// Allocate device memory (VRAM via cudaMalloc), or aligned host memory for CPU fallback.
///
/// Returns a raw pointer. The caller is responsible for freeing via `device_free`.
pub fn device_alloc(size: usize) -> Result<*mut u8> {
    #[cfg(feature = "cuda")]
    {
        if is_cuda_available() {
            unsafe {
                let mut ptr: *mut std::ffi::c_void = std::ptr::null_mut();
                let err = cuda_rt::cudaMalloc(&mut ptr, size);
                if err != cuda_rt::CUDA_SUCCESS {
                    return Err(Error::Cuda(format!(
                        "cudaMalloc({} bytes) failed: {}",
                        size,
                        cuda_rt::error_string(err)
                    )));
                }
                // Zero-initialize device memory
                let err = cuda_rt::cudaMemset(ptr, 0, size);
                if err != cuda_rt::CUDA_SUCCESS {
                    tracing::warn!(
                        "cudaMemset failed (non-fatal): {}",
                        cuda_rt::error_string(err)
                    );
                }
                DEVICE_BYTES_ALLOCATED.fetch_add(size, Ordering::Relaxed);
                tracing::trace!("device_alloc (VRAM): {} bytes at {:p}", size, ptr);
                return Ok(ptr as *mut u8);
            }
        }
    }

    // CPU fallback: aligned host allocation
    let layout = std::alloc::Layout::from_size_align(size, PAGE_SIZE)
        .map_err(|e| Error::Cuda(format!("Invalid allocation layout: {}", e)))?;

    // SAFETY: layout is valid and size > 0 is guaranteed by caller
    let ptr = unsafe { std::alloc::alloc_zeroed(layout) };
    if ptr.is_null() {
        return Err(Error::Cuda(format!(
            "Failed to allocate {} bytes for device memory (CPU fallback)",
            size
        )));
    }

    DEVICE_BYTES_ALLOCATED.fetch_add(size, Ordering::Relaxed);
    tracing::trace!("device_alloc (host): {} bytes at {:p}", size, ptr);
    Ok(ptr)
}

/// Free device memory.
pub fn device_free(ptr: *mut u8, size: usize) {
    if ptr.is_null() {
        return;
    }

    #[cfg(feature = "cuda")]
    {
        if is_cuda_available() {
            // SAFETY: ptr was allocated by cudaMalloc in device_alloc
            unsafe {
                cuda_rt::cudaFree(ptr as *mut std::ffi::c_void);
            }
            DEVICE_BYTES_ALLOCATED.fetch_sub(size, Ordering::Relaxed);
            tracing::trace!("device_free (VRAM): {} bytes at {:p}", size, ptr);
            return;
        }
    }

    let layout = std::alloc::Layout::from_size_align(size, PAGE_SIZE).unwrap();
    // SAFETY: ptr was allocated by std::alloc::alloc_zeroed with this layout
    unsafe { std::alloc::dealloc(ptr, layout) };
    DEVICE_BYTES_ALLOCATED.fetch_sub(size, Ordering::Relaxed);
    tracing::trace!("device_free (host): {} bytes at {:p}", size, ptr);
}

/// Allocate pinned host memory (for DMA transfers).
/// With CUDA: uses cudaMallocHost for page-locked memory (fast DMA).
/// Without CUDA: aligned allocation with mlock.
pub fn host_alloc_pinned(size: usize) -> Result<*mut u8> {
    #[cfg(feature = "cuda")]
    {
        if is_cuda_available() {
            unsafe {
                let mut ptr: *mut std::ffi::c_void = std::ptr::null_mut();
                let err = cuda_rt::cudaMallocHost(&mut ptr, size);
                if err != cuda_rt::CUDA_SUCCESS {
                    // Fall through to non-CUDA path
                    tracing::warn!(
                        "cudaMallocHost({} bytes) failed: {}, falling back to mlock",
                        size,
                        cuda_rt::error_string(err)
                    );
                } else {
                    HOST_PINNED_BYTES_ALLOCATED.fetch_add(size, Ordering::Relaxed);
                    tracing::trace!(
                        "host_alloc_pinned (cudaMallocHost): {} bytes at {:p}",
                        size,
                        ptr
                    );
                    return Ok(ptr as *mut u8);
                }
            }
        }
    }

    // Fallback: aligned allocation + mlock
    let layout = std::alloc::Layout::from_size_align(size, PAGE_SIZE)
        .map_err(|e| Error::Cuda(format!("Invalid allocation layout: {}", e)))?;

    let ptr = unsafe { std::alloc::alloc_zeroed(layout) };
    if ptr.is_null() {
        return Err(Error::Cuda(format!(
            "Failed to allocate {} bytes for pinned host memory",
            size
        )));
    }

    // Try to pin the memory (prevent swapping) — best effort
    #[cfg(target_os = "linux")]
    unsafe {
        libc::mlock(ptr as *const libc::c_void, size);
    }

    HOST_PINNED_BYTES_ALLOCATED.fetch_add(size, Ordering::Relaxed);
    tracing::trace!("host_alloc_pinned (mlock): {} bytes at {:p}", size, ptr);
    Ok(ptr)
}

/// Free pinned host memory.
pub fn host_free_pinned(ptr: *mut u8, size: usize) {
    if ptr.is_null() {
        return;
    }

    #[cfg(feature = "cuda")]
    {
        if is_cuda_available() {
            // SAFETY: ptr was allocated by cudaMallocHost
            unsafe {
                cuda_rt::cudaFreeHost(ptr as *mut std::ffi::c_void);
            }
            HOST_PINNED_BYTES_ALLOCATED.fetch_sub(size, Ordering::Relaxed);
            tracing::trace!(
                "host_free_pinned (cudaFreeHost): {} bytes at {:p}",
                size,
                ptr
            );
            return;
        }
    }

    #[cfg(target_os = "linux")]
    unsafe {
        libc::munlock(ptr as *const libc::c_void, size);
    }

    let layout = std::alloc::Layout::from_size_align(size, PAGE_SIZE).unwrap();
    unsafe { std::alloc::dealloc(ptr, layout) };
    HOST_PINNED_BYTES_ALLOCATED.fetch_sub(size, Ordering::Relaxed);
    tracing::trace!("host_free_pinned (dealloc): {} bytes at {:p}", size, ptr);
}

/// Zero-fill device memory.
/// With CUDA: uses cudaMemset. Without CUDA: uses std::ptr::write_bytes.
pub fn device_memset(ptr: *mut u8, value: u8, size: usize) {
    if ptr.is_null() || size == 0 {
        return;
    }
    #[cfg(feature = "cuda")]
    {
        if is_cuda_available() {
            unsafe {
                cuda_rt::cudaMemset(ptr as *mut std::ffi::c_void, value as i32, size);
            }
            return;
        }
    }
    unsafe {
        std::ptr::write_bytes(ptr, value, size);
    }
}

/// Query current free VRAM in bytes (live, not cached).
/// Returns 0 if CUDA is not available or the query fails.
pub fn query_free_vram() -> usize {
    #[cfg(feature = "cuda")]
    {
        let mut free: usize = 0;
        let mut _total: usize = 0;
        // SAFETY: cudaMemGetInfo writes to the two pointers; both are valid stack vars.
        let err = unsafe { cuda_rt::cudaMemGetInfo(&mut free, &mut _total) };
        if err == cuda_rt::CUDA_SUCCESS {
            return free;
        }
    }
    0
}

/// Device-to-device memory copy.
/// With CUDA: uses cudaMemcpy(DeviceToDevice). Without CUDA: uses memcpy.
pub fn device_memcpy_d2d(dst: *mut u8, src: *const u8, size: usize) -> Result<()> {
    if dst.is_null() || src.is_null() {
        return Err(Error::Cuda("Null pointer in device_memcpy_d2d".into()));
    }
    #[cfg(feature = "cuda")]
    {
        if is_cuda_available() {
            // cudaMemcpyDeviceToDevice = 3
            let err = unsafe {
                cuda_rt::cudaMemcpy(
                    dst as *mut std::ffi::c_void,
                    src as *const std::ffi::c_void,
                    size,
                    3, // cudaMemcpyDeviceToDevice
                )
            };
            if err != cuda_rt::CUDA_SUCCESS {
                return Err(Error::Cuda(format!(
                    "cudaMemcpy D2D failed: {}",
                    cuda_rt::error_string(err)
                )));
            }
            return Ok(());
        }
    }
    unsafe {
        std::ptr::copy_nonoverlapping(src, dst, size);
    }
    Ok(())
}

/// Device-to-device memory copy (async, on a specific stream).
/// With CUDA: uses cudaMemcpyAsync(DeviceToDevice) — does NOT synchronize
/// the default stream, unlike the synchronous cudaMemcpy.
/// Without CUDA: uses memcpy.
pub fn device_memcpy_d2d_async(
    dst: *mut u8,
    src: *const u8,
    size: usize,
    stream: &CudaStream,
) -> Result<()> {
    if dst.is_null() || src.is_null() {
        return Err(Error::Cuda(
            "Null pointer in device_memcpy_d2d_async".into(),
        ));
    }
    #[cfg(feature = "cuda")]
    {
        if stream.is_real() {
            // cudaMemcpyDeviceToDevice = 3
            let err = unsafe {
                cuda_rt::cudaMemcpyAsync(
                    dst as *mut std::ffi::c_void,
                    src as *const std::ffi::c_void,
                    size,
                    3, // cudaMemcpyDeviceToDevice
                    stream.raw_ptr(),
                )
            };
            if err != cuda_rt::CUDA_SUCCESS {
                return Err(Error::Cuda(format!(
                    "cudaMemcpyAsync D2D failed: {}",
                    cuda_rt::error_string(err)
                )));
            }
            return Ok(());
        }
    }
    let _ = stream; // suppress unused warning in non-CUDA builds
    unsafe {
        std::ptr::copy_nonoverlapping(src, dst, size);
    }
    Ok(())
}

/// Zero-fill device memory (async, on a specific stream).
/// With CUDA: uses cudaMemsetAsync — does NOT synchronize the default stream.
/// Without CUDA: uses std::ptr::write_bytes.
pub fn device_memset_async(ptr: *mut u8, value: u8, size: usize, stream: &CudaStream) {
    if ptr.is_null() || size == 0 {
        return;
    }
    #[cfg(feature = "cuda")]
    {
        if stream.is_real() {
            unsafe {
                cuda_rt::cudaMemsetAsync(
                    ptr as *mut std::ffi::c_void,
                    value as i32,
                    size,
                    stream.raw_ptr(),
                );
            }
            return;
        }
    }
    let _ = stream; // suppress unused warning in non-CUDA builds
    unsafe {
        std::ptr::write_bytes(ptr, value, size);
    }
}

/// RAII wrapper for device-allocated memory (VRAM when CUDA is available).
///
/// Provides safe-ish access patterns for GPU buffer management.
/// The underlying pointer is a device pointer when CUDA is active,
/// or a host-aligned allocation in CPU fallback mode.
pub struct DeviceBuffer {
    ptr: *mut u8,
    size: usize,
}

impl DeviceBuffer {
    /// Allocate a zero-initialized device buffer.
    pub fn new(size: usize) -> Result<Self> {
        let ptr = device_alloc(size)?;
        Ok(Self { ptr, size })
    }

    /// Get the raw device pointer (for passing to kernels).
    pub fn as_ptr(&self) -> *const u8 {
        self.ptr as *const u8
    }

    /// Get a mutable device pointer (for passing to kernels).
    pub fn as_mut_ptr(&self) -> *mut u8 {
        self.ptr
    }

    /// Buffer size in bytes.
    pub fn size(&self) -> usize {
        self.size
    }

    /// Zero-fill the entire buffer on device (synchronous — stalls pipeline).
    pub fn zero(&self) {
        device_memset(self.ptr, 0, self.size);
    }

    /// Zero-fill the entire buffer on device (async — no pipeline stall).
    pub fn zero_async(&self, stream: &CudaStream) {
        device_memset_async(self.ptr, 0, self.size, stream);
    }

    /// Copy device buffer contents to a host slice.
    pub fn copy_to_host(&self, dst: &mut [u8]) -> Result<()> {
        let copy_size = dst.len().min(self.size);
        memcpy_d2h(dst.as_mut_ptr(), self.ptr as *const u8, copy_size)
    }

    /// Copy from a host slice into this device buffer.
    pub fn copy_from_host(&self, src: &[u8]) -> Result<()> {
        let copy_size = src.len().min(self.size);
        memcpy_h2d_sync(self.ptr, src.as_ptr(), copy_size)
    }

    /// Copy from another device pointer into this buffer (D2D).
    pub fn copy_from_device(&self, src: *const u8, size: usize) -> Result<()> {
        let copy_size = size.min(self.size);
        device_memcpy_d2d(self.ptr, src, copy_size)
    }

    /// Copy this buffer's contents to another device pointer (D2D).
    pub fn copy_to_device(&self, dst: *mut u8, size: usize) -> Result<()> {
        let copy_size = size.min(self.size);
        device_memcpy_d2d(dst, self.ptr as *const u8, copy_size)
    }
}

impl Drop for DeviceBuffer {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            device_free(self.ptr, self.size);
            self.ptr = std::ptr::null_mut();
        }
    }
}

// SAFETY: DeviceBuffer wraps a device pointer allocated by cudaMalloc.
// The pointer is an opaque GPU driver handle, not a host memory pointer.
// It is safe to send/share across threads as CUDA operations are thread-safe
// when using the same CUDA context.
unsafe impl Send for DeviceBuffer {}
unsafe impl Sync for DeviceBuffer {}

/// Async copy from host (pinned) to device.
/// With CUDA: uses cudaMemcpyAsync for overlapped DMA.
/// Without CUDA: synchronous memcpy.
pub fn memcpy_h2d_async(
    dst: *mut u8,
    src: *const u8,
    size: usize,
    stream: &CudaStream,
) -> Result<()> {
    if dst.is_null() || src.is_null() {
        return Err(Error::Cuda("Null pointer in memcpy_h2d".into()));
    }

    #[cfg(feature = "cuda")]
    {
        if stream.is_real() {
            unsafe {
                let err = cuda_rt::cudaMemcpyAsync(
                    dst as *mut std::ffi::c_void,
                    src as *const std::ffi::c_void,
                    size,
                    cuda_rt::MEMCPY_HOST_TO_DEVICE,
                    stream.raw_ptr(),
                );
                if err != cuda_rt::CUDA_SUCCESS {
                    return Err(Error::Cuda(format!(
                        "cudaMemcpyAsync H2D ({} bytes) failed: {}",
                        size,
                        cuda_rt::error_string(err)
                    )));
                }
            }
            return Ok(());
        }
    }

    // CPU fallback: synchronous memcpy
    // SAFETY: src and dst are valid for `size` bytes
    unsafe {
        std::ptr::copy_nonoverlapping(src, dst, size);
    }
    Ok(())
}

/// Synchronous copy from device to host.
pub fn memcpy_d2h(dst: *mut u8, src: *const u8, size: usize) -> Result<()> {
    if dst.is_null() || src.is_null() {
        return Err(Error::Cuda("Null pointer in memcpy_d2h".into()));
    }

    #[cfg(feature = "cuda")]
    {
        if is_cuda_available() {
            unsafe {
                let err = cuda_rt::cudaMemcpy(
                    dst as *mut std::ffi::c_void,
                    src as *const std::ffi::c_void,
                    size,
                    cuda_rt::MEMCPY_DEVICE_TO_HOST,
                );
                if err != cuda_rt::CUDA_SUCCESS {
                    return Err(Error::Cuda(format!(
                        "cudaMemcpy D2H ({} bytes) failed: {}",
                        size,
                        cuda_rt::error_string(err)
                    )));
                }
            }
            return Ok(());
        }
    }

    // CPU fallback
    unsafe {
        std::ptr::copy_nonoverlapping(src, dst, size);
    }
    Ok(())
}

/// Async copy from device to host (pinned).
pub fn memcpy_d2h_async(
    dst: *mut u8,
    src: *const u8,
    size: usize,
    stream: &CudaStream,
) -> Result<()> {
    if dst.is_null() || src.is_null() {
        return Err(Error::Cuda("Null pointer in memcpy_d2h".into()));
    }

    #[cfg(feature = "cuda")]
    {
        if stream.is_real() {
            unsafe {
                let err = cuda_rt::cudaMemcpyAsync(
                    dst as *mut std::ffi::c_void,
                    src as *const std::ffi::c_void,
                    size,
                    cuda_rt::MEMCPY_DEVICE_TO_HOST,
                    stream.raw_ptr(),
                );
                if err != cuda_rt::CUDA_SUCCESS {
                    return Err(Error::Cuda(format!(
                        "cudaMemcpyAsync D2H ({} bytes) failed: {}",
                        size,
                        cuda_rt::error_string(err)
                    )));
                }
            }
            return Ok(());
        }
    }

    // CPU fallback
    unsafe {
        std::ptr::copy_nonoverlapping(src, dst, size);
    }
    Ok(())
}

/// Synchronous copy from host to device (convenience wrapper).
pub fn memcpy_h2d_sync(dst: *mut u8, src: *const u8, size: usize) -> Result<()> {
    if dst.is_null() || src.is_null() {
        return Err(Error::Cuda("Null pointer in memcpy_h2d_sync".into()));
    }

    #[cfg(feature = "cuda")]
    {
        if is_cuda_available() {
            unsafe {
                let err = cuda_rt::cudaMemcpy(
                    dst as *mut std::ffi::c_void,
                    src as *const std::ffi::c_void,
                    size,
                    cuda_rt::MEMCPY_HOST_TO_DEVICE,
                );
                if err != cuda_rt::CUDA_SUCCESS {
                    return Err(Error::Cuda(format!(
                        "cudaMemcpy H2D ({} bytes) failed: {}",
                        size,
                        cuda_rt::error_string(err)
                    )));
                }
            }
            return Ok(());
        }
    }

    unsafe {
        std::ptr::copy_nonoverlapping(src, dst, size);
    }
    Ok(())
}

/// Synchronize the GPU device (wait for all pending work to complete).
#[allow(dead_code)]
pub fn device_synchronize() -> Result<()> {
    #[cfg(feature = "cuda")]
    {
        if is_cuda_available() {
            unsafe {
                let err = cuda_rt::cudaDeviceSynchronize();
                if err != cuda_rt::CUDA_SUCCESS {
                    return Err(Error::Cuda(format!(
                        "cudaDeviceSynchronize failed: {}",
                        cuda_rt::error_string(err)
                    )));
                }
            }
        }
    }
    Ok(())
}

// ─── Diagnostics ─────────────────────────────────────────────────────────

/// Check if CUDA is available at runtime.
pub fn is_cuda_available() -> bool {
    CUDA_AVAILABLE.load(Ordering::Acquire)
}

/// Check for a deferred CUDA error. Returns Ok if no error, or a descriptive
/// error with the `context` label to identify which operation caused it.
/// Used for debugging: insert after a stream.synchronize() to pinpoint the
/// kernel responsible for an illegal address or other async error.
pub fn check_last_cuda_error(context: &str) -> Result<()> {
    #[cfg(feature = "cuda")]
    {
        if is_cuda_available() {
            let err = unsafe { cuda_rt::cudaGetLastError() };
            if err != cuda_rt::CUDA_SUCCESS {
                return Err(Error::Cuda(format!(
                    "CUDA error at {}: err={}",
                    context, err,
                )));
            }
        }
    }
    let _ = context; // suppress unused warning in non-CUDA builds
    Ok(())
}

/// Total device memory currently allocated.
pub fn device_bytes_allocated() -> usize {
    DEVICE_BYTES_ALLOCATED.load(Ordering::Relaxed)
}

/// Total pinned host memory currently allocated.
pub fn host_pinned_bytes_allocated() -> usize {
    HOST_PINNED_BYTES_ALLOCATED.load(Ordering::Relaxed)
}

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
    // Fallback: assume 64 GB
    64 * 1024 * 1024 * 1024
}
