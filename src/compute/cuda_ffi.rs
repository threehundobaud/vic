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
        position: i32,
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
        position: i32,
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
        seq_len: i32,
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
