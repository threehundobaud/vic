//! GPU kernel launchers for page-level expert computation.
//!
//! These are the Rust-side launchers that invoke CUDA kernels compiled
//! from .cu files. When CUDA is not available, they execute CPU fallback
//! implementations using f16 → f32 conversion and SIMD-friendly loops.
//!
//! The CPU fallback is not fast enough for production but is correct
//! and enables full end-to-end testing without a GPU.

// Kernel functions take raw device pointers (*const u8 / *mut u8) for VRAM
// buffers. These are passed to CUDA FFI launchers; the pointers are validated
// at the FFI boundary and never dereferenced in safe Rust code.
#![allow(clippy::not_unsafe_ptr_arg_deref)]

use crate::compute::cuda_ffi::{self, CudaStream};
use crate::core::error::{Error, Result};
use crate::core::types::DType;
use half::f16;
use rayon::prelude::*;

/// Partial matrix multiply: input × weight_page_slice.
///
/// Computes `output[1, M_slice] = input[1, K] × weight[M_slice, K]^T`
/// where the weight page is a contiguous slice of rows from the full matrix.
pub fn partial_matmul(
    input: *const u8,       // [1, K] on device — FP16
    weight_page: *const u8, // [M_slice, K] on device
    output: *mut u8,        // [1, M_slice] on device — FP16
    k: usize,
    m_slice: usize,
    weight_dtype: DType,
    stream: &CudaStream,
) -> Result<()> {
    // GPU dispatch when CUDA is available
    #[cfg(feature = "cuda")]
    {
        if stream.is_real() {
            return gpu_partial_matmul(
                input,
                weight_page,
                output,
                k,
                m_slice,
                weight_dtype,
                stream,
            );
        }
    }

    // CPU fallback
    match weight_dtype {
        DType::FP16 => {
            cpu_matmul_fp16(input, weight_page, output, k, m_slice);
        }
        DType::BF16 => {
            cpu_matmul_bf16(input, weight_page, output, k, m_slice);
        }
        DType::INT4 | DType::NF4 | DType::NVFP4 => {
            cpu_matmul_int4(input, weight_page, output, k, m_slice);
        }
        DType::INT8 => {
            cpu_matmul_int8(input, weight_page, output, k, m_slice);
        }
        _ => {
            cpu_matmul_fp16(input, weight_page, output, k, m_slice);
        }
    }
    Ok(())
}

/// GPU partial matmul dispatch.
#[cfg(feature = "cuda")]
fn gpu_partial_matmul(
    input: *const u8,
    weight_page: *const u8,
    output: *mut u8,
    k: usize,
    m_slice: usize,
    weight_dtype: DType,
    stream: &CudaStream,
) -> Result<()> {
    let err = match weight_dtype {
        DType::FP16 | DType::BF16 => unsafe {
            cuda_ffi::vib3_launch_partial_matmul_fp16_fast(
                input,
                weight_page,
                output,
                k as i32,
                m_slice as i32,
                stream.raw_ptr(),
            )
        },
        DType::INT4 | DType::NF4 => {
            // INT4 layout: [rows * packed_k nibble bytes] [rows * num_groups * 2 scale bytes]
            let packed_k = k.div_ceil(2);
            let weight_data_size = packed_k * m_slice;
            // SAFETY: scales immediately follow packed weight data in the page
            let scales_ptr = unsafe { weight_page.add(weight_data_size) };
            unsafe {
                cuda_ffi::vib3_launch_partial_matmul_int4(
                    input,
                    weight_page,
                    scales_ptr,
                    output,
                    k as i32,
                    m_slice as i32,
                    INT4_GROUP_SIZE as i32,
                    stream.raw_ptr(),
                )
            }
        }
        DType::NVFP4 => {
            // NVFP4 with FP16 input (used for down-projection where intermediate is FP16)
            let packed_k = k.div_ceil(2);
            let weight_data_size = packed_k * m_slice;
            // SAFETY: FP16 scales immediately follow packed weight data
            let scales_ptr = unsafe { weight_page.add(weight_data_size) };
            unsafe {
                cuda_ffi::vib3_launch_partial_matmul_nvfp4_fp16in(
                    input,
                    weight_page,
                    scales_ptr,
                    output,
                    k as i32,
                    m_slice as i32,
                    NVFP4_BLOCK_SIZE as i32,
                    stream.raw_ptr(),
                )
            }
        }
        _ => {
            // Unsupported dtype on GPU, fall back to FP16
            unsafe {
                cuda_ffi::vib3_launch_partial_matmul_fp16(
                    input,
                    weight_page,
                    output,
                    k as i32,
                    m_slice as i32,
                    stream.raw_ptr(),
                )
            }
        }
    };
    if err != 0 {
        return Err(Error::Cuda(format!(
            "partial_matmul kernel launch failed (err={})",
            err
        )));
    }
    Ok(())
}

/// Fused partial SwiGLU.
///
/// Computes: `output[M_slice] = SiLU(input × up_page) * (input × gate_page)`
///
/// `swiglu_temps`: Optional pre-allocated temp buffers (up_tmp, gate_tmp) for INT4
/// decomposition. When Some, avoids per-call cudaMalloc/cudaFree. Buffers must be
/// at least `m_slice * sizeof(f16)` bytes.
#[allow(clippy::too_many_arguments)]
pub fn partial_swiglu(
    input: *const u8,     // [1, hidden_dim] FP16
    up_page: *const u8,   // [M_slice, hidden_dim]
    gate_page: *const u8, // [M_slice, hidden_dim]
    output: *mut u8,      // [1, M_slice] FP16
    hidden_dim: usize,
    m_slice: usize,
    weight_dtype: DType,
    stream: &CudaStream,
    swiglu_temps: Option<(*mut u8, *mut u8)>,
) -> Result<()> {
    // GPU dispatch
    #[cfg(feature = "cuda")]
    {
        if stream.is_real() {
            return gpu_partial_swiglu(
                input,
                up_page,
                gate_page,
                output,
                hidden_dim,
                m_slice,
                weight_dtype,
                stream,
                swiglu_temps,
            );
        }
    }

    // CPU fallback
    match weight_dtype {
        DType::FP16 => {
            cpu_swiglu_fp16(input, up_page, gate_page, output, hidden_dim, m_slice);
        }
        DType::INT4 | DType::NF4 | DType::NVFP4 => {
            cpu_swiglu_int4(input, up_page, gate_page, output, hidden_dim, m_slice);
        }
        _ => {
            cpu_swiglu_fp16(input, up_page, gate_page, output, hidden_dim, m_slice);
        }
    }
    Ok(())
}

/// GPU SwiGLU dispatch. For FP16: fused kernel. For INT4: decomposed into
/// two INT4 matmuls + SiLU*gate fusion kernel.
///
/// `swiglu_temps`: Optional pre-allocated temp buffers (up_tmp, gate_tmp) for INT4.
/// When Some, avoids per-call cudaMalloc/cudaFree. When None, allocates on the fly.
#[cfg(feature = "cuda")]
#[allow(clippy::too_many_arguments)]
fn gpu_partial_swiglu(
    input: *const u8,
    up_page: *const u8,
    gate_page: *const u8,
    output: *mut u8,
    hidden_dim: usize,
    m_slice: usize,
    weight_dtype: DType,
    stream: &CudaStream,
    swiglu_temps: Option<(*mut u8, *mut u8)>,
) -> Result<()> {
    match weight_dtype {
        DType::FP16 => {
            let err = unsafe {
                cuda_ffi::vib3_launch_fused_swiglu_fp16(
                    input,
                    up_page,
                    gate_page,
                    output,
                    hidden_dim as i32,
                    m_slice as i32,
                    stream.raw_ptr(),
                )
            };
            if err != 0 {
                return Err(Error::Cuda(format!(
                    "fused_swiglu_fp16 launch failed (err={})",
                    err
                )));
            }
        }
        DType::INT4 | DType::NF4 => {
            // Decompose: up_result = matmul(input, up), gate_result = matmul(input, gate),
            // output = SiLU(gate_result) * up_result
            // Note: SiLU is applied to gate_proj output per the SwiGLU reference:
            //   SwiGLU(x) = SiLU(gate_proj(x)) * up_proj(x)
            let buf_size = m_slice * 2; // FP16
            let (up_result, gate_result, allocated) = if let Some((up_tmp, gate_tmp)) = swiglu_temps
            {
                // Use pre-allocated buffers (no cudaMalloc/cudaFree overhead)
                (up_tmp, gate_tmp, false)
            } else {
                // Fallback: allocate on the fly
                let up = cuda_ffi::device_alloc(buf_size)?;
                let gate = cuda_ffi::device_alloc(buf_size)?;
                (up, gate, true)
            };

            // up matmul
            gpu_partial_matmul(
                input,
                up_page,
                up_result,
                hidden_dim,
                m_slice,
                weight_dtype,
                stream,
            )?;
            // gate matmul
            gpu_partial_matmul(
                input,
                gate_page,
                gate_result,
                hidden_dim,
                m_slice,
                weight_dtype,
                stream,
            )?;
            // SiLU(gate) * up — note: gate_result goes as first arg (SiLU applied to it)
            let err = unsafe {
                cuda_ffi::vib3_launch_silu_mul(
                    gate_result as *const u8,
                    up_result as *const u8,
                    output,
                    m_slice as i32,
                    stream.raw_ptr(),
                )
            };

            if allocated {
                cuda_ffi::device_free(up_result, buf_size);
                cuda_ffi::device_free(gate_result, buf_size);
            }

            if err != 0 {
                return Err(Error::Cuda(format!("silu_mul launch failed (err={})", err)));
            }
        }
        DType::NVFP4 => {
            // Decompose with FP16-input NVFP4 matmul: up=matmul, gate=matmul, output=SiLU(gate)*up
            let buf_size = m_slice * 2; // FP16
            let (up_result, gate_result, allocated) = if let Some((up_tmp, gate_tmp)) = swiglu_temps
            {
                (up_tmp, gate_tmp, false)
            } else {
                let up = cuda_ffi::device_alloc(buf_size)?;
                let gate = cuda_ffi::device_alloc(buf_size)?;
                (up, gate, true)
            };

            gpu_partial_matmul(
                input,
                up_page,
                up_result,
                hidden_dim,
                m_slice,
                weight_dtype,
                stream,
            )?;
            gpu_partial_matmul(
                input,
                gate_page,
                gate_result,
                hidden_dim,
                m_slice,
                weight_dtype,
                stream,
            )?;
            let err = unsafe {
                cuda_ffi::vib3_launch_silu_mul(
                    gate_result as *const u8,
                    up_result as *const u8,
                    output,
                    m_slice as i32,
                    stream.raw_ptr(),
                )
            };

            if allocated {
                cuda_ffi::device_free(up_result, buf_size);
                cuda_ffi::device_free(gate_result, buf_size);
            }

            if err != 0 {
                return Err(Error::Cuda(format!("silu_mul launch failed (err={})", err)));
            }
        }
        _ => {
            let err = unsafe {
                cuda_ffi::vib3_launch_fused_swiglu_fp16(
                    input,
                    up_page,
                    gate_page,
                    output,
                    hidden_dim as i32,
                    m_slice as i32,
                    stream.raw_ptr(),
                )
            };
            if err != 0 {
                return Err(Error::Cuda(format!(
                    "fused_swiglu_fp16 launch failed (err={})",
                    err
                )));
            }
        }
    }
    Ok(())
}

/// FP32-input partial matmul: same as partial_matmul but input is FP32.
/// Reduces precision loss in the dot product.
pub fn partial_matmul_f32(
    input: *const u8,       // [1, K] on device — FP32
    weight_page: *const u8, // [M_slice, K] on device
    output: *mut u8,        // [1, M_slice] on device — FP16
    k: usize,
    m_slice: usize,
    weight_dtype: DType,
    stream: &CudaStream,
) -> Result<()> {
    #[cfg(feature = "cuda")]
    {
        if stream.is_real() {
            let err = match weight_dtype {
                DType::FP16 | DType::BF16 => unsafe {
                    cuda_ffi::vib3_launch_partial_matmul_fp16_f32in(
                        input,
                        weight_page,
                        output,
                        k as i32,
                        m_slice as i32,
                        stream.raw_ptr(),
                    )
                },
                DType::INT4 | DType::NF4 => {
                    let packed_k = k.div_ceil(2);
                    let weight_data_size = packed_k * m_slice;
                    let scales_ptr = unsafe { weight_page.add(weight_data_size) };
                    unsafe {
                        cuda_ffi::vib3_launch_partial_matmul_int4_f32(
                            input,
                            weight_page,
                            scales_ptr,
                            output,
                            k as i32,
                            m_slice as i32,
                            INT4_GROUP_SIZE as i32,
                            stream.raw_ptr(),
                        )
                    }
                }
                DType::NVFP4 => {
                    // NVFP4 layout: [rows * packed_k E2M1 bytes] [rows * num_blocks * 2 FP16 scale bytes]
                    let packed_k = k.div_ceil(2);
                    let weight_data_size = packed_k * m_slice;
                    // SAFETY: FP16 scales immediately follow packed weight data
                    let scales_ptr = unsafe { weight_page.add(weight_data_size) };
                    unsafe {
                        cuda_ffi::vib3_launch_partial_matmul_nvfp4(
                            input,
                            weight_page,
                            scales_ptr,
                            output,
                            k as i32,
                            m_slice as i32,
                            NVFP4_BLOCK_SIZE as i32,
                            stream.raw_ptr(),
                        )
                    }
                }
                _ => unsafe {
                    cuda_ffi::vib3_launch_partial_matmul_fp16_f32in(
                        input,
                        weight_page,
                        output,
                        k as i32,
                        m_slice as i32,
                        stream.raw_ptr(),
                    )
                },
            };
            if err != 0 {
                return Err(Error::Cuda(format!(
                    "partial_matmul_f32 launch failed (err={})",
                    err
                )));
            }
            return Ok(());
        }
    }
    Err(Error::Cuda("partial_matmul_f32 requires GPU".to_string()))
}

/// FP32-input partial SwiGLU: same as partial_swiglu but input is FP32.
#[allow(clippy::too_many_arguments)]
pub fn partial_swiglu_f32(
    input: *const u8, // [1, hidden_dim] FP32
    up_page: *const u8,
    gate_page: *const u8,
    output: *mut u8, // [1, M_slice] FP16
    hidden_dim: usize,
    m_slice: usize,
    weight_dtype: DType,
    stream: &CudaStream,
    swiglu_temps: Option<(*mut u8, *mut u8)>,
) -> Result<()> {
    #[cfg(feature = "cuda")]
    {
        if stream.is_real() {
            match weight_dtype {
                DType::FP16 => {
                    let err = unsafe {
                        cuda_ffi::vib3_launch_fused_swiglu_fp16_f32in(
                            input,
                            up_page,
                            gate_page,
                            output,
                            hidden_dim as i32,
                            m_slice as i32,
                            stream.raw_ptr(),
                        )
                    };
                    if err != 0 {
                        return Err(Error::Cuda(format!(
                            "fused_swiglu_fp16_f32in launch failed (err={})",
                            err
                        )));
                    }
                }
                DType::INT4 | DType::NF4 => {
                    // Decompose: up=matmul_f32, gate=matmul_f32, output=SiLU(gate)*up
                    let buf_size = m_slice * 2; // FP16
                    let (up_result, gate_result, allocated) =
                        if let Some((up_tmp, gate_tmp)) = swiglu_temps {
                            (up_tmp, gate_tmp, false)
                        } else {
                            let up = cuda_ffi::device_alloc(buf_size)?;
                            let gate = cuda_ffi::device_alloc(buf_size)?;
                            (up, gate, true)
                        };

                    partial_matmul_f32(
                        input,
                        up_page,
                        up_result,
                        hidden_dim,
                        m_slice,
                        weight_dtype,
                        stream,
                    )?;
                    partial_matmul_f32(
                        input,
                        gate_page,
                        gate_result,
                        hidden_dim,
                        m_slice,
                        weight_dtype,
                        stream,
                    )?;

                    let err = unsafe {
                        cuda_ffi::vib3_launch_silu_mul(
                            gate_result as *const u8,
                            up_result as *const u8,
                            output,
                            m_slice as i32,
                            stream.raw_ptr(),
                        )
                    };

                    if allocated {
                        cuda_ffi::device_free(up_result, buf_size);
                        cuda_ffi::device_free(gate_result, buf_size);
                    }

                    if err != 0 {
                        return Err(Error::Cuda(format!("silu_mul launch failed (err={})", err)));
                    }
                }
                DType::NVFP4 => {
                    // Fused SwiGLU with NVFP4 E2M1 dequant — single kernel launch
                    // Layout per page: [rows * packed_k E2M1 bytes] [rows * num_blocks * 2 FP16 scale bytes]
                    let packed_k = hidden_dim.div_ceil(2);
                    let weight_data_size = packed_k * m_slice;
                    let num_blocks = hidden_dim.div_ceil(NVFP4_BLOCK_SIZE);
                    let _scales_size = num_blocks * m_slice * 2; // FP16 scales

                    // SAFETY: scales immediately follow packed weight data in each page
                    let up_scales = unsafe { up_page.add(weight_data_size) };
                    let gate_scales = unsafe { gate_page.add(weight_data_size) };

                    let err = unsafe {
                        cuda_ffi::vib3_launch_fused_swiglu_nvfp4(
                            input,
                            up_page,
                            up_scales,
                            gate_page,
                            gate_scales,
                            output,
                            hidden_dim as i32,
                            m_slice as i32,
                            NVFP4_BLOCK_SIZE as i32,
                            stream.raw_ptr(),
                        )
                    };
                    if err != 0 {
                        return Err(Error::Cuda(format!(
                            "fused_swiglu_nvfp4 launch failed (err={})",
                            err
                        )));
                    }
                }
                _ => {
                    let err = unsafe {
                        cuda_ffi::vib3_launch_fused_swiglu_fp16_f32in(
                            input,
                            up_page,
                            gate_page,
                            output,
                            hidden_dim as i32,
                            m_slice as i32,
                            stream.raw_ptr(),
                        )
                    };
                    if err != 0 {
                        return Err(Error::Cuda(format!(
                            "fused_swiglu_fp16_f32in launch failed (err={})",
                            err
                        )));
                    }
                }
            }
            return Ok(());
        }
    }
    Err(Error::Cuda("partial_swiglu_f32 requires GPU".to_string()))
}

/// FP32-in/FP32-out SwiGLU for NVFP4 weights. Eliminates FP16 truncation
/// of the SwiGLU intermediate that gets amplified by the subsequent down_proj.
pub fn partial_swiglu_f32_f32out(
    input: *const u8,   // [1, hidden_dim] FP32
    up_page: *const u8, // NVFP4 packed weight page
    gate_page: *const u8,
    output: *mut u8, // [1, M_slice] FP32 (NOT FP16!)
    hidden_dim: usize,
    m_slice: usize,
    stream: &CudaStream,
) -> Result<()> {
    #[cfg(feature = "cuda")]
    {
        if stream.is_real() {
            let packed_k = hidden_dim.div_ceil(2);
            let weight_data_size = packed_k * m_slice;

            let up_scales = unsafe { up_page.add(weight_data_size) };
            let gate_scales = unsafe { gate_page.add(weight_data_size) };

            let err = unsafe {
                cuda_ffi::vib3_launch_fused_swiglu_nvfp4_f32out(
                    input,
                    up_page,
                    up_scales,
                    gate_page,
                    gate_scales,
                    output,
                    hidden_dim as i32,
                    m_slice as i32,
                    NVFP4_BLOCK_SIZE as i32,
                    stream.raw_ptr(),
                )
            };
            if err != 0 {
                return Err(Error::Cuda(format!(
                    "fused_swiglu_nvfp4_f32out launch failed (err={})",
                    err
                )));
            }
            return Ok(());
        }
    }
    Err(Error::Cuda(
        "partial_swiglu_f32_f32out requires GPU".to_string(),
    ))
}

/// FP32-in/FP32-out matmul for NVFP4 weights. Pairs with partial_swiglu_f32_f32out
/// to keep the entire routed expert pipeline in FP32, eliminating FP16 truncation.
pub fn partial_matmul_nvfp4_f32(
    input: *const u8,       // [1, K] FP32
    weight_page: *const u8, // NVFP4 packed weight page
    output: *mut u8,        // [1, M_slice] FP32
    k: usize,
    m_slice: usize,
    stream: &CudaStream,
) -> Result<()> {
    #[cfg(feature = "cuda")]
    {
        if stream.is_real() {
            let packed_k = k.div_ceil(2);
            let weight_data_size = packed_k * m_slice;
            let scales_ptr = unsafe { weight_page.add(weight_data_size) };

            let err = unsafe {
                cuda_ffi::vib3_launch_partial_matmul_nvfp4_f32out(
                    input,
                    weight_page,
                    scales_ptr,
                    output,
                    k as i32,
                    m_slice as i32,
                    NVFP4_BLOCK_SIZE as i32,
                    stream.raw_ptr(),
                )
            };
            if err != 0 {
                return Err(Error::Cuda(format!(
                    "partial_matmul_nvfp4_f32out launch failed (err={})",
                    err
                )));
            }
            return Ok(());
        }
    }
    Err(Error::Cuda(
        "partial_matmul_nvfp4_f32 requires GPU".to_string(),
    ))
}

/// Blackwell MMA GEMV for NVFP4 weights. Uses m16n8k64 block-scaled FP4
/// Tensor Core MMA. FP32 input, FP32 output. No model re-conversion needed:
/// sequential nibbles are repacked to split-half in-kernel.
pub fn gemv_mma_nvfp4(
    input: *const u8,       // [1, K] FP32
    weight_page: *const u8, // NVFP4 packed weight page ([data | scales])
    output: *mut u8,        // [1, M_slice] FP32
    k: usize,
    m_slice: usize,
    stream: &CudaStream,
) -> Result<()> {
    #[cfg(feature = "cuda")]
    {
        if stream.is_real() {
            let packed_k = k.div_ceil(2);
            let weight_data_size = packed_k * m_slice;
            let scales_ptr = unsafe { weight_page.add(weight_data_size) };

            let err = unsafe {
                cuda_ffi::vib3_launch_gemv_mma_nvfp4(
                    input,
                    weight_page,
                    scales_ptr,
                    output,
                    k as i32,
                    m_slice as i32,
                    stream.raw_ptr(),
                )
            };
            if err != 0 {
                return Err(Error::Cuda(format!(
                    "gemv_mma_nvfp4 launch failed (err={})",
                    err
                )));
            }
            return Ok(());
        }
    }
    Err(Error::Cuda("gemv_mma_nvfp4 requires GPU".to_string()))
}

/// Blackwell MMA fused SwiGLU for NVFP4 weights. Two GEMV MMA passes
/// (up + gate) with shared input quantization, fused SiLU activation.
/// FP32 input, FP32 output. No model re-conversion needed.
pub fn fused_swiglu_mma_nvfp4(
    input: *const u8,     // [1, K] FP32
    up_page: *const u8,   // NVFP4 packed weight page (up_proj)
    gate_page: *const u8, // NVFP4 packed weight page (gate_proj)
    output: *mut u8,      // [1, M_slice] FP32
    k: usize,
    m_slice: usize,
    stream: &CudaStream,
) -> Result<()> {
    #[cfg(feature = "cuda")]
    {
        if stream.is_real() {
            let packed_k = k.div_ceil(2);
            let weight_data_size = packed_k * m_slice;

            let up_scales = unsafe { up_page.add(weight_data_size) };
            let gate_scales = unsafe { gate_page.add(weight_data_size) };

            let err = unsafe {
                cuda_ffi::vib3_launch_fused_swiglu_mma_nvfp4(
                    input,
                    up_page,
                    up_scales,
                    gate_page,
                    gate_scales,
                    output,
                    k as i32,
                    m_slice as i32,
                    stream.raw_ptr(),
                )
            };
            if err != 0 {
                return Err(Error::Cuda(format!(
                    "fused_swiglu_mma_nvfp4 launch failed (err={})",
                    err
                )));
            }
            return Ok(());
        }
    }
    Err(Error::Cuda(
        "fused_swiglu_mma_nvfp4 requires GPU".to_string(),
    ))
}

/// Pre-quantize FP32 activation vector to FP4 E2M1 split-half + E8M0 scales.
/// Done once per MoE layer, reused by all expert kernels to eliminate redundant
/// per-kernel FP32→FP4 quantization (the dominant MMA bottleneck).
pub fn quantize_activation_fp4(
    input: *const u8,    // [K] FP32 activation
    act_fp4: *mut u8,    // [K/2] pre-quantized FP4 output
    act_scales: *mut u8, // [K/32] E8M0 scales output
    k: usize,
    stream: &CudaStream,
) -> Result<()> {
    #[cfg(feature = "cuda")]
    {
        if stream.is_real() {
            let err = unsafe {
                cuda_ffi::vib3_launch_quantize_activation_fp4(
                    input,
                    act_fp4,
                    act_scales,
                    k as i32,
                    stream.raw_ptr(),
                )
            };
            if err != 0 {
                return Err(Error::Cuda(format!(
                    "quantize_activation_fp4 launch failed (err={})",
                    err
                )));
            }
            return Ok(());
        }
    }
    Err(Error::Cuda(
        "quantize_activation_fp4 requires GPU".to_string(),
    ))
}

/// Blackwell MMA GEMV with pre-quantized activations. Skips FP32→FP4
/// quantization inside the kernel — uses activations pre-quantized by
/// quantize_activation_fp4().
pub fn gemv_mma_nvfp4_preq(
    act_fp4: *const u8,     // [K/2] pre-quantized split-half FP4
    act_scales: *const u8,  // [K/32] E8M0 scales
    weight_page: *const u8, // NVFP4 packed weight page ([data | scales])
    output: *mut u8,        // [M_slice] FP32
    k: usize,
    m_slice: usize,
    stream: &CudaStream,
) -> Result<()> {
    #[cfg(feature = "cuda")]
    {
        if stream.is_real() {
            let packed_k = k.div_ceil(2);
            let weight_data_size = packed_k * m_slice;
            let scales_ptr = unsafe { weight_page.add(weight_data_size) };

            let err = unsafe {
                cuda_ffi::vib3_launch_gemv_mma_nvfp4_preq(
                    act_fp4,
                    act_scales,
                    weight_page,
                    scales_ptr,
                    output,
                    k as i32,
                    m_slice as i32,
                    stream.raw_ptr(),
                )
            };
            if err != 0 {
                return Err(Error::Cuda(format!(
                    "gemv_mma_nvfp4_preq launch failed (err={})",
                    err
                )));
            }
            return Ok(());
        }
    }
    Err(Error::Cuda("gemv_mma_nvfp4_preq requires GPU".to_string()))
}

/// Blackwell MMA fused SwiGLU with pre-quantized activations.
pub fn fused_swiglu_mma_nvfp4_preq(
    act_fp4: *const u8,    // [K/2] pre-quantized split-half FP4
    act_scales: *const u8, // [K/32] E8M0 scales
    up_page: *const u8,    // NVFP4 packed weight page (up_proj)
    gate_page: *const u8,  // NVFP4 packed weight page (gate_proj)
    output: *mut u8,       // [M_slice] FP32
    k: usize,
    m_slice: usize,
    stream: &CudaStream,
) -> Result<()> {
    #[cfg(feature = "cuda")]
    {
        if stream.is_real() {
            let packed_k = k.div_ceil(2);
            let weight_data_size = packed_k * m_slice;

            let up_scales = unsafe { up_page.add(weight_data_size) };
            let gate_scales = unsafe { gate_page.add(weight_data_size) };

            let err = unsafe {
                cuda_ffi::vib3_launch_fused_swiglu_mma_nvfp4_preq(
                    act_fp4,
                    act_scales,
                    up_page,
                    up_scales,
                    gate_page,
                    gate_scales,
                    output,
                    k as i32,
                    m_slice as i32,
                    stream.raw_ptr(),
                )
            };
            if err != 0 {
                return Err(Error::Cuda(format!(
                    "fused_swiglu_mma_nvfp4_preq launch failed (err={})",
                    err
                )));
            }
            return Ok(());
        }
    }
    Err(Error::Cuda(
        "fused_swiglu_mma_nvfp4_preq requires GPU".to_string(),
    ))
}

// ═══════════════════════════════════════════════════════════════════════════
// Optimized launchers: cached capability, batched expert, norepack
// ═══════════════════════════════════════════════════════════════════════════

/// In-place repack weight data from sequential to split-half format on GPU.
/// Only repacks the FP4 data portion (not scales).
pub fn repack_weights_inplace(
    weight_data_ptr: *mut u8,
    weight_data_bytes: usize,
    stream: &CudaStream,
) -> Result<()> {
    #[cfg(feature = "cuda")]
    {
        if stream.is_real() {
            let err = unsafe {
                cuda_ffi::vib3_launch_repack_weights_inplace(
                    weight_data_ptr,
                    weight_data_bytes as i64,
                    stream.raw_ptr(),
                )
            };
            if err != 0 {
                return Err(Error::Cuda(format!(
                    "repack_weights_inplace failed (err={})",
                    err
                )));
            }
            return Ok(());
        }
    }
    Err(Error::Cuda(
        "repack_weights_inplace requires GPU".to_string(),
    ))
}

/// Batched expert: single call for full expert pipeline.
/// Weights must be pre-repacked to split-half format.
#[allow(clippy::too_many_arguments)]
pub fn expert_batched(
    act_fp4: *const u8,
    act_scales: *const u8,
    up_page: *const u8,   // NVFP4 weight page (repacked split-half)
    gate_page: *const u8, // NVFP4 weight page (repacked split-half)
    down_page: *const u8, // NVFP4 weight page (repacked split-half)
    layer_output: *mut f32,
    expert_weight: f32,
    k_in: usize,  // hidden_dim
    m_mid: usize, // expert_hidden_dim (up/gate output rows)
    m_out: usize, // hidden_dim (down output rows)
    stream: &CudaStream,
) -> Result<()> {
    #[cfg(feature = "cuda")]
    {
        if stream.is_real() {
            let packed_k_in = k_in.div_ceil(2);
            let packed_k_mid = m_mid.div_ceil(2); // down_proj input dim = expert_hidden_dim
            let up_data_size = packed_k_in * m_mid;
            let gate_data_size = packed_k_in * m_mid;
            let down_data_size = packed_k_mid * m_out;

            let up_scales_ptr = unsafe { up_page.add(up_data_size) };
            let gate_scales_ptr = unsafe { gate_page.add(gate_data_size) };
            let down_scales_ptr = unsafe { down_page.add(down_data_size) };

            let err = unsafe {
                cuda_ffi::vib3_launch_expert_batched(
                    act_fp4,
                    act_scales,
                    up_page,
                    up_scales_ptr,
                    gate_page,
                    gate_scales_ptr,
                    down_page,
                    down_scales_ptr,
                    layer_output,
                    expert_weight,
                    k_in as i32,
                    m_mid as i32,
                    m_mid as i32, // K_mid = expert_hidden_dim = M_mid
                    m_out as i32,
                    stream.raw_ptr(),
                )
            };
            if err != 0 {
                return Err(Error::Cuda(format!(
                    "expert_batched launch failed (err={})",
                    err
                )));
            }
            return Ok(());
        }
    }
    Err(Error::Cuda("expert_batched requires GPU".to_string()))
}

/// Fused multi-expert MoE layer: processes all selected experts with
/// 4 kernel launches (up GEMV, gate GEMV, SwiGLU+quantize, down+accumulate).
///
/// `expert_pages` is a slice of (up_page, gate_page, down_page, weight) tuples.
/// All page pointers must point to pre-repacked split-half NVFP4 data.
/// `layer_output` must be pre-zeroed.
pub fn moe_experts_fused(
    act_fp4: *const u8,
    act_scales: *const u8,
    expert_pages: &[(*const u8, *const u8, *const u8, f32)], // (up, gate, down, weight)
    k_in: usize,
    m_mid: usize,
    m_out: usize,
    layer_output: *mut f32,
    stream: &CudaStream,
) -> Result<()> {
    #[cfg(feature = "cuda")]
    {
        if stream.is_real() && !expert_pages.is_empty() {
            let num_experts = expert_pages.len().min(8);
            let packed_k_in = k_in.div_ceil(2);
            let packed_k_mid = m_mid.div_ceil(2);
            let up_data_size = packed_k_in * m_mid;
            let gate_data_size = packed_k_in * m_mid;
            let down_data_size = packed_k_mid * m_out;

            // Build pointer arrays (host side)
            let mut up_w = [std::ptr::null::<u8>(); 8];
            let mut up_s = [std::ptr::null::<u8>(); 8];
            let mut gate_w = [std::ptr::null::<u8>(); 8];
            let mut gate_s = [std::ptr::null::<u8>(); 8];
            let mut down_w = [std::ptr::null::<u8>(); 8];
            let mut down_s = [std::ptr::null::<u8>(); 8];
            let mut weights = [0.0f32; 8];

            for (i, &(up_page, gate_page, down_page, weight)) in
                expert_pages.iter().take(num_experts).enumerate()
            {
                up_w[i] = up_page;
                up_s[i] = unsafe { up_page.add(up_data_size) };
                gate_w[i] = gate_page;
                gate_s[i] = unsafe { gate_page.add(gate_data_size) };
                down_w[i] = down_page;
                down_s[i] = unsafe { down_page.add(down_data_size) };
                weights[i] = weight;
            }

            let err = unsafe {
                cuda_ffi::vib3_launch_moe_experts_fused(
                    act_fp4,
                    act_scales,
                    up_w.as_ptr() as *const *const u8,
                    up_s.as_ptr() as *const *const u8,
                    gate_w.as_ptr() as *const *const u8,
                    gate_s.as_ptr() as *const *const u8,
                    down_w.as_ptr() as *const *const u8,
                    down_s.as_ptr() as *const *const u8,
                    weights.as_ptr(),
                    num_experts as i32,
                    k_in as i32,
                    m_mid as i32,
                    m_mid as i32,
                    m_out as i32,
                    layer_output,
                    stream.raw_ptr(),
                )
            };
            if err != 0 {
                return Err(Error::Cuda(format!(
                    "moe_experts_fused launch failed (err={})",
                    err
                )));
            }
            return Ok(());
        }
    }
    Err(Error::Cuda("moe_experts_fused requires GPU".to_string()))
}

/// GPU-only fused multi-expert MoE layer: reads expert IDs and weights
/// directly from device memory (written by router), looks up weight page
/// pointers from a prebuilt device-side table. Zero host synchronization.
///
/// `page_table`: device pointer to [num_moe_layers * num_experts * 3] u64
/// `expert_ids`: device pointer to [num_active] u16 (from router topk)
/// `expert_weights`: device pointer to [num_active] f32 (from router topk)
/// `moe_layer`: MoE layer index (0-based, NOT storage layer)
pub fn moe_experts_fused_gpu(
    act_fp4: *const u8,
    act_scales: *const u8,
    page_table: *const u8,     // device: [layers * experts * 3] u64
    expert_ids: *const u8,     // device: [num_active] u16
    expert_weights: *const u8, // device: [num_active] f32
    moe_layer: usize,
    num_experts_total: usize,
    num_active: usize,
    k_in: usize,
    m_mid: usize,
    m_out: usize,
    layer_output: *mut f32,
    stream: &CudaStream,
) -> Result<()> {
    #[cfg(feature = "cuda")]
    {
        if stream.is_real() && num_active > 0 {
            let err = unsafe {
                cuda_ffi::vib3_launch_moe_experts_fused_gpu(
                    act_fp4,
                    act_scales,
                    page_table as *const u64,
                    expert_ids as *const u16,
                    expert_weights as *const f32,
                    moe_layer as i32,
                    num_experts_total as i32,
                    num_active as i32,
                    k_in as i32,
                    m_mid as i32,
                    m_mid as i32, // K_mid == M_mid for SwiGLU output
                    m_out as i32,
                    layer_output,
                    stream.raw_ptr(),
                )
            };
            if err != 0 {
                return Err(Error::Cuda(format!(
                    "moe_experts_fused_gpu launch failed (err={})",
                    err
                )));
            }
            return Ok(());
        }
    }
    let _ = (
        act_fp4,
        act_scales,
        page_table,
        expert_ids,
        expert_weights,
        moe_layer,
        num_experts_total,
        num_active,
        k_in,
        m_mid,
        m_out,
        layer_output,
        stream,
    );
    Err(Error::Cuda(
        "moe_experts_fused_gpu requires GPU".to_string(),
    ))
}

/// Convert FP16 weight matrix [M, K] to NVFP4 MMA format at runtime.
/// Returns a DeviceBuffer containing:
///   - [0 .. M*K/2):           FP4 data in split-half packing
///   - [M*K/2 .. M*K/2 + M*(K/32)*2): BF16 block scales
/// The caller must keep the returned DeviceBuffer alive.
pub fn fp16_to_nvfp4_weight(
    input: *const u8,
    m: usize,
    k: usize,
    stream: &CudaStream,
) -> Result<cuda_ffi::DeviceBuffer> {
    #[cfg(feature = "cuda")]
    {
        if stream.is_real() {
            let data_bytes = m * k / 2;
            let scale_bytes = m * (k / 32) * 2;
            let total_bytes = data_bytes + scale_bytes;
            let buf = cuda_ffi::DeviceBuffer::new(total_bytes)?;
            let data_ptr = buf.as_mut_ptr();
            let scale_ptr = unsafe { data_ptr.add(data_bytes) };
            let err = unsafe {
                cuda_ffi::vib3_launch_fp16_to_nvfp4_weight(
                    input,
                    data_ptr,
                    scale_ptr,
                    m as i32,
                    k as i32,
                    stream.raw_ptr(),
                )
            };
            if err != 0 {
                return Err(Error::Cuda(format!(
                    "fp16_to_nvfp4_weight failed (err={})",
                    err
                )));
            }
            return Ok(buf);
        }
    }
    Err(Error::Cuda("fp16_to_nvfp4_weight requires GPU".to_string()))
}

/// Fast MMA GEMV preq with cached capability check.
/// Uses shared-memory staging for coalesced global reads.
pub fn gemv_mma_nvfp4_preq_fast(
    act_fp4: *const u8,
    act_scales: *const u8,
    weight_page: *const u8,
    output: *mut u8,
    k: usize,
    m_slice: usize,
    stream: &CudaStream,
) -> Result<()> {
    #[cfg(feature = "cuda")]
    {
        if stream.is_real() {
            let packed_k = k.div_ceil(2);
            let weight_data_size = packed_k * m_slice;
            let scales_ptr = unsafe { weight_page.add(weight_data_size) };

            let err = unsafe {
                cuda_ffi::vib3_launch_gemv_mma_nvfp4_preq_fast(
                    act_fp4,
                    act_scales,
                    weight_page,
                    scales_ptr,
                    output,
                    k as i32,
                    m_slice as i32,
                    stream.raw_ptr(),
                )
            };
            if err != 0 {
                return Err(Error::Cuda(format!(
                    "gemv_mma_nvfp4_preq_fast failed (err={})",
                    err
                )));
            }
            return Ok(());
        }
    }
    Err(Error::Cuda(
        "gemv_mma_nvfp4_preq_fast requires GPU".to_string(),
    ))
}

/// Scalar GEMV with K-dimension parallelism for maximum memory bandwidth.
/// 1 row per thread block, all threads cooperate along K → coalesced access.
/// Uses LUT-based FP4 dequant + FMA instead of MMA instructions.
pub fn gemv_scalar_nvfp4(
    act_fp4: *const u8,
    act_scales: *const u8,
    weight_page: *const u8,
    output: *mut u8,
    k: usize,
    m_slice: usize,
    stream: &CudaStream,
) -> Result<()> {
    #[cfg(feature = "cuda")]
    {
        if stream.is_real() {
            let packed_k = k.div_ceil(2);
            let weight_data_size = packed_k * m_slice;
            let scales_ptr = unsafe { weight_page.add(weight_data_size) };

            let err = unsafe {
                cuda_ffi::vib3_launch_gemv_scalar_nvfp4(
                    act_fp4,
                    act_scales,
                    weight_page,
                    scales_ptr,
                    output,
                    k as i32,
                    m_slice as i32,
                    stream.raw_ptr(),
                )
            };
            if err != 0 {
                return Err(Error::Cuda(format!(
                    "gemv_scalar_nvfp4 failed (err={})",
                    err
                )));
            }
            return Ok(());
        }
    }
    Err(Error::Cuda("gemv_scalar_nvfp4 requires GPU".to_string()))
}

/// Repack weight data from row-major split-half to tiled layout on GPU.
/// Tiled layout: 16 rows × 32 bytes per K-tile contiguous.
/// M must be divisible by 16, K must be divisible by 64.
/// temp_buf must be at least M * (K/2) bytes.
pub fn repack_row_to_tiled(
    weight_data_ptr: *mut u8,
    temp_buf: *mut u8,
    m: usize,
    k: usize,
    stream: &CudaStream,
) -> Result<()> {
    #[cfg(feature = "cuda")]
    {
        if stream.is_real() {
            let err = unsafe {
                cuda_ffi::vib3_launch_repack_row_to_tiled(
                    weight_data_ptr,
                    temp_buf,
                    m as i32,
                    k as i32,
                    stream.raw_ptr(),
                )
            };
            if err != 0 {
                return Err(Error::Cuda(format!(
                    "repack_row_to_tiled failed (err={})",
                    err
                )));
            }
            return Ok(());
        }
    }
    Err(Error::Cuda("repack_row_to_tiled requires GPU".to_string()))
}

/// Tiled MMA GEMV: reads weights from tiled layout for coalesced access.
/// Weights must have been repacked via repack_row_to_tiled().
/// Scales remain in row-major layout.
pub fn gemv_mma_nvfp4_tiled(
    act_fp4: *const u8,
    act_scales: *const u8,
    weight_page: *const u8,
    output: *mut u8,
    k: usize,
    m_slice: usize,
    stream: &CudaStream,
) -> Result<()> {
    #[cfg(feature = "cuda")]
    {
        if stream.is_real() {
            let packed_k = k.div_ceil(2);
            let weight_data_size = packed_k * m_slice;
            let scales_ptr = unsafe { weight_page.add(weight_data_size) };

            let err = unsafe {
                cuda_ffi::vib3_launch_gemv_mma_nvfp4_tiled(
                    act_fp4,
                    act_scales,
                    weight_page,
                    scales_ptr,
                    output,
                    k as i32,
                    m_slice as i32,
                    stream.raw_ptr(),
                )
            };
            if err != 0 {
                return Err(Error::Cuda(format!(
                    "gemv_mma_nvfp4_tiled failed (err={})",
                    err
                )));
            }
            return Ok(());
        }
    }
    Err(Error::Cuda("gemv_mma_nvfp4_tiled requires GPU".to_string()))
}

/// Batched multi-matrix MMA GEMV with shared pre-quantized FP4 activation.
/// Processes multiple NVFP4 weight matrices in a single kernel launch.
/// All matrices must share the same K dimension and the same FP4 activation input.
///
/// # Arguments
/// * `act_fp4` - Pre-quantized FP4 activation data [K/2 bytes]
/// * `act_scales` - FP4 activation E8M0 scales [K/32 bytes]
/// * `weight_pages` - Array of NVFP4 buffer pointers (each packed [FP4 data | BF16 scales])
/// * `m_slices` - Array of M dimensions for each matrix
/// * `outputs` - Array of FP32 output pointers
/// * `k` - Shared K dimension (must be divisible by 64)
/// * `stream` - CUDA stream
pub fn batched_gemv_mma_nvfp4_preq(
    act_fp4: *const u8,
    act_scales: *const u8,
    weight_pages: &[*const u8],
    m_slices: &[i32],
    outputs: &[*mut u8],
    k: usize,
    stream: &CudaStream,
) -> Result<()> {
    let n = weight_pages.len();
    assert_eq!(n, m_slices.len());
    assert_eq!(n, outputs.len());
    assert!(n > 0 && n <= 5, "batched GEMV supports 1-5 matrices");

    #[cfg(feature = "cuda")]
    {
        if stream.is_real() {
            let err = unsafe {
                cuda_ffi::vib3_launch_batched_gemv_mma_nvfp4_preq(
                    act_fp4,
                    act_scales,
                    n as i32,
                    weight_pages.as_ptr(),
                    m_slices.as_ptr(),
                    outputs.as_ptr(),
                    k as i32,
                    stream.raw_ptr(),
                )
            };
            if err != 0 {
                return Err(Error::Cuda(format!(
                    "batched_gemv_mma_nvfp4_preq failed (err={})",
                    err
                )));
            }
            return Ok(());
        }
    }
    Err(Error::Cuda(
        "batched_gemv_mma_nvfp4_preq requires GPU".to_string(),
    ))
}

/// Batched tiled MMA GEMV with K-parallel decomposition + atomicAdd.
/// Processes multiple NVFP4 weight matrices (in tiled layout) in a single kernel launch.
/// All matrices share the same FP4 activation and K dimension.
///
/// # Arguments
/// * `act_fp4` - Pre-quantized FP4 activation data [K/2 bytes]
/// * `act_scales` - FP4 activation E8M0 scales [K/32 bytes]
/// * `weight_pages` - Array of NVFP4 buffer pointers (each packed [tiled FP4 data | BF16 scales])
/// * `m_slices` - Array of M dimensions for each matrix
/// * `outputs` - Array of FP32 output pointers
/// * `k` - Shared K dimension (must be divisible by 64)
/// * `stream` - CUDA stream
pub fn batched_gemv_mma_nvfp4_tiled(
    act_fp4: *const u8,
    act_scales: *const u8,
    weight_pages: &[*const u8],
    m_slices: &[i32],
    outputs: &[*mut u8],
    k: usize,
    stream: &CudaStream,
) -> Result<()> {
    let n = weight_pages.len();
    assert_eq!(n, m_slices.len());
    assert_eq!(n, outputs.len());
    assert!(n > 0 && n <= 5, "batched tiled GEMV supports 1-5 matrices");

    #[cfg(feature = "cuda")]
    {
        if stream.is_real() {
            let packed_k = k.div_ceil(2);

            // Split each weight_page into tiled FP4 data ptr and BF16 scales ptr
            let mut tiled_ptrs: [*const u8; 5] = [std::ptr::null(); 5];
            let mut scale_ptrs: [*const u8; 5] = [std::ptr::null(); 5];
            for i in 0..n {
                tiled_ptrs[i] = weight_pages[i];
                let data_size = packed_k * m_slices[i] as usize;
                scale_ptrs[i] = unsafe { weight_pages[i].add(data_size) };
            }

            let err = unsafe {
                cuda_ffi::vib3_launch_batched_gemv_mma_nvfp4_tiled(
                    act_fp4,
                    act_scales,
                    n as i32,
                    tiled_ptrs.as_ptr(),
                    scale_ptrs.as_ptr(),
                    outputs.as_ptr(),
                    m_slices.as_ptr(),
                    k as i32,
                    stream.raw_ptr(),
                )
            };
            if err != 0 {
                return Err(Error::Cuda(format!(
                    "batched_gemv_mma_nvfp4_tiled failed (err={})",
                    err
                )));
            }
            return Ok(());
        }
    }
    Err(Error::Cuda(
        "batched_gemv_mma_nvfp4_tiled requires GPU".to_string(),
    ))
}

/// Fast activation quantization with cached capability check.
pub fn quantize_activation_fp4_fast(
    input: *const u8,
    act_fp4: *mut u8,
    act_scales: *mut u8,
    k: usize,
    stream: &CudaStream,
) -> Result<()> {
    #[cfg(feature = "cuda")]
    {
        if stream.is_real() {
            let err = unsafe {
                cuda_ffi::vib3_launch_quantize_activation_fp4_fast(
                    input,
                    act_fp4,
                    act_scales,
                    k as i32,
                    stream.raw_ptr(),
                )
            };
            if err != 0 {
                return Err(Error::Cuda(format!(
                    "quantize_activation_fp4_fast failed (err={})",
                    err
                )));
            }
            return Ok(());
        }
    }
    Err(Error::Cuda(
        "quantize_activation_fp4_fast requires GPU".to_string(),
    ))
}

/// Fast fused SwiGLU MMA preq with cached capability check.
pub fn fused_swiglu_mma_nvfp4_preq_fast(
    act_fp4: *const u8,
    act_scales: *const u8,
    up_page: *const u8,
    gate_page: *const u8,
    output: *mut u8,
    k: usize,
    m_slice: usize,
    stream: &CudaStream,
) -> Result<()> {
    #[cfg(feature = "cuda")]
    {
        if stream.is_real() {
            let packed_k = k.div_ceil(2);
            let weight_data_size = packed_k * m_slice;
            let up_scales = unsafe { up_page.add(weight_data_size) };
            let gate_scales = unsafe { gate_page.add(weight_data_size) };

            let err = unsafe {
                cuda_ffi::vib3_launch_fused_swiglu_mma_nvfp4_preq_fast(
                    act_fp4,
                    act_scales,
                    up_page,
                    up_scales,
                    gate_page,
                    gate_scales,
                    output,
                    k as i32,
                    m_slice as i32,
                    stream.raw_ptr(),
                )
            };
            if err != 0 {
                return Err(Error::Cuda(format!(
                    "fused_swiglu_mma_nvfp4_preq_fast failed (err={})",
                    err
                )));
            }
            return Ok(());
        }
    }
    Err(Error::Cuda(
        "fused_swiglu_mma_nvfp4_preq_fast requires GPU".to_string(),
    ))
}

/// FP32 RMSNorm: reads FP32 input from accumulator, writes FP32 output.
pub fn rms_norm_f32(
    input: *const u8,  // FP32 hidden state
    output: *mut u8,   // FP32 normalized output
    weight: *const u8, // FP16 norm weight
    dim: usize,
    eps: f32,
    stream: &CudaStream,
) -> Result<()> {
    #[cfg(feature = "cuda")]
    {
        if stream.is_real() {
            let err = unsafe {
                cuda_ffi::vib3_launch_rms_norm_f32(
                    input,
                    output,
                    weight,
                    dim as i32,
                    eps,
                    stream.raw_ptr(),
                )
            };
            if err != 0 {
                return Err(Error::Cuda(format!(
                    "rms_norm_f32 launch failed (err={})",
                    err
                )));
            }
            return Ok(());
        }
    }
    // CPU fallback
    // SAFETY: Caller provides valid host pointers for `dim` elements.
    let input_f32 = unsafe { std::slice::from_raw_parts(input as *const f32, dim) };
    // SAFETY: Caller provides valid writable host pointer for `dim` elements.
    let output_f32 = unsafe { std::slice::from_raw_parts_mut(output as *mut f32, dim) };
    // SAFETY: RMSNorm weight is FP16 with `dim` elements.
    let weight_f16 = unsafe { std::slice::from_raw_parts(weight as *const f16, dim) };

    let mut sum_sq = 0.0f64;
    for &value in input_f32 {
        let v = value as f64;
        sum_sq += v * v;
    }
    let inv_rms = 1.0f32 / ((sum_sq as f32 / dim as f32) + eps).sqrt();

    for i in 0..dim {
        output_f32[i] = input_f32[i] * inv_rms * weight_f16[i].to_f32();
    }

    Ok(())
}

/// FP32→FP16 RMSNorm: reads FP32 hidden state, normalizes in FP32 precision,
/// writes FP16 output. This avoids the catastrophic precision loss from casting
/// large FP32 hidden states (L2 norm ~3000+) to FP16 before normalization.
/// Post-normalization values are typically in [-3, 3], which FP16 represents fine.
pub fn rms_norm_f32_to_f16(
    input: *const u8,  // FP32 hidden state
    output: *mut u8,   // FP16 normalized output
    weight: *const u8, // FP16 norm weight
    dim: usize,
    eps: f32,
    stream: &CudaStream,
) -> Result<()> {
    #[cfg(feature = "cuda")]
    {
        if stream.is_real() {
            let err = unsafe {
                cuda_ffi::vib3_launch_rms_norm_f32_to_f16(
                    input,
                    output,
                    weight,
                    dim as i32,
                    eps,
                    stream.raw_ptr(),
                )
            };
            if err != 0 {
                return Err(Error::Cuda(format!(
                    "rms_norm_f32_to_f16 launch failed (err={})",
                    err
                )));
            }
            return Ok(());
        }
    }
    // CPU fallback
    // SAFETY: Caller provides valid host pointers for `dim` elements.
    let input_f32 = unsafe { std::slice::from_raw_parts(input as *const f32, dim) };
    // SAFETY: Caller provides valid writable host pointer for `dim` FP16 elements.
    let output_f16 = unsafe { std::slice::from_raw_parts_mut(output as *mut f16, dim) };
    // SAFETY: RMSNorm weight is FP16 with `dim` elements.
    let weight_f16 = unsafe { std::slice::from_raw_parts(weight as *const f16, dim) };

    let mut sum_sq = 0.0f64;
    for &value in input_f32 {
        let v = value as f64;
        sum_sq += v * v;
    }
    let inv_rms = 1.0f32 / ((sum_sq as f32 / dim as f32) + eps).sqrt();

    for i in 0..dim {
        let normalized = input_f32[i] * inv_rms * weight_f16[i].to_f32();
        output_f16[i] = f16::from_f32(normalized);
    }

    Ok(())
}

/// Fused RMSNorm + FP4 quantize: reads FP32 hidden state, applies RMSNorm
/// with FP16 weight, then directly quantizes to split-half FP4 + E8M0 scales.
/// Optionally produces FP16 normalized output (pass null to skip).
/// Eliminates 2 kernel launches and a global memory round-trip.
pub fn fused_rms_norm_quantize_fp4(
    input: *const u8,       // [K] FP32 hidden state
    norm_weight: *const u8, // [K] FP16 norm weight
    act_fp4: *mut u8,       // [K/2] split-half FP4 output
    act_scales: *mut u8,    // [K/32] E8M0 scales output
    opt_f16_out: *mut u8,   // [K] optional FP16 output (null to skip)
    k: usize,
    eps: f32,
    stream: &CudaStream,
) -> Result<()> {
    #[cfg(feature = "cuda")]
    {
        if stream.is_real() {
            let err = unsafe {
                cuda_ffi::vib3_launch_fused_rms_norm_quantize_fp4(
                    input,
                    norm_weight,
                    act_fp4,
                    act_scales,
                    opt_f16_out,
                    k as i32,
                    eps,
                    stream.raw_ptr(),
                )
            };
            if err != 0 {
                return Err(Error::Cuda(format!(
                    "fused_rms_norm_quantize_fp4 failed (err={})",
                    err
                )));
            }
            return Ok(());
        }
    }
    Err(Error::Cuda(
        "fused_rms_norm_quantize_fp4 requires GPU".to_string(),
    ))
}

/// FP32-input router scoring with bias.
/// Same as run_router_with_scoring_and_bias but reads FP32 hidden state.
pub fn run_router_f32(
    hidden_state_f32: *const u8, // FP32 normalized hidden state
    router_weights: *const u8,   // FP16 router weight matrix
    num_experts: usize,
    hidden_dim: usize,
    top_k: usize,
    scoring_func: RouterScoringFunc,
    bias: Option<&[f32]>,
    scores_dev: *mut f32,
    stream: &CudaStream,
) -> Result<Vec<(u16, f32)>> {
    #[cfg(feature = "cuda")]
    {
        if stream.is_real() {
            // Run FP32-input router GEMV
            let err = unsafe {
                cuda_ffi::vib3_launch_router_gemv_f32(
                    hidden_state_f32,
                    router_weights,
                    scores_dev,
                    hidden_dim as i32,
                    num_experts as i32,
                    stream.raw_ptr(),
                )
            };
            if err != 0 {
                return Err(Error::Cuda(format!(
                    "router_gemv_f32 launch failed (err={})",
                    err
                )));
            }
            stream.synchronize()?;

            // Read scores back from device and apply scoring
            let score_bytes = num_experts * std::mem::size_of::<f32>();
            let mut score_buf = vec![0u8; score_bytes];
            cuda_ffi::memcpy_d2h(score_buf.as_mut_ptr(), scores_dev as *const u8, score_bytes)?;
            let raw_scores = unsafe {
                std::slice::from_raw_parts(score_buf.as_ptr() as *const f32, num_experts)
            };

            // Apply sigmoid scoring, bias, topk selection, normalize, scale
            // (reuse the same logic from run_router_with_scoring_and_bias)
            match scoring_func {
                RouterScoringFunc::Sigmoid {
                    scaling_factor,
                    normalize,
                } => {
                    let scores: Vec<f32> = raw_scores
                        .iter()
                        .map(|&s| 1.0 / (1.0 + (-s).exp()))
                        .collect();

                    // Selection scores (with bias for expert selection)
                    let mut selection_scores = scores.clone();
                    if let Some(b) = bias {
                        for (s, &bv) in selection_scores.iter_mut().zip(b.iter()) {
                            *s += bv;
                        }
                    }

                    // Top-k selection
                    let mut indexed: Vec<(usize, f32)> = selection_scores
                        .iter()
                        .enumerate()
                        .map(|(i, &s)| (i, s))
                        .collect();
                    indexed
                        .sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
                    indexed.truncate(top_k);

                    // Get original sigmoid scores for selected experts
                    let mut topk: Vec<(u16, f32)> = indexed
                        .iter()
                        .map(|&(idx, _)| (idx as u16, scores[idx]))
                        .collect();

                    // Normalize then scale
                    if normalize {
                        let sum: f32 = topk.iter().map(|&(_, w)| w).sum();
                        if sum > 0.0 {
                            for (_, w) in topk.iter_mut() {
                                *w /= sum;
                            }
                        }
                    }
                    for (_, w) in topk.iter_mut() {
                        *w *= scaling_factor;
                    }

                    return Ok(topk);
                }
                RouterScoringFunc::Softmax => {
                    // Softmax over ALL experts, then top-k, then renormalize
                    // to sum to 1.0 (matches llama.cpp norm_w=true behavior).
                    let max_s = raw_scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                    let exp_scores: Vec<f32> =
                        raw_scores.iter().map(|&s| (s - max_s).exp()).collect();
                    let sum: f32 = exp_scores.iter().sum();
                    let probs: Vec<f32> = exp_scores.iter().map(|s| s / sum).collect();

                    let mut indexed: Vec<(usize, f32)> =
                        probs.iter().enumerate().map(|(i, &p)| (i, p)).collect();
                    indexed
                        .sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
                    indexed.truncate(top_k);

                    // Renormalize top-k weights to sum to 1.0
                    let topk_sum: f32 = indexed.iter().map(|(_, w)| w).sum();
                    if topk_sum > 0.0 {
                        return Ok(indexed
                            .iter()
                            .map(|&(i, w)| (i as u16, w / topk_sum))
                            .collect());
                    }
                    return Ok(indexed.iter().map(|&(i, w)| (i as u16, w)).collect());
                }
            }
        }
    }
    Err(Error::Cuda("run_router_f32 requires GPU".to_string()))
}

/// GPU-fused router: GEMV + softmax/sigmoid + top-k + normalize.
///
/// Eliminates the stream.synchronize() bottleneck by doing ALL router work
/// on the GPU and only transferring back the final top-k results (tiny D2H).
///
/// Returns Vec<(expert_id, weight)> sorted by weight descending.
pub fn run_router_gpu_topk(
    hidden_state_f32: *const u8, // FP32 normalized hidden state (device)
    router_weights: *const u8,   // FP16 router weight matrix (device)
    num_experts: usize,
    hidden_dim: usize,
    top_k: usize,
    scoring_func: RouterScoringFunc,
    scores_dev: *mut f32,      // [num_experts] temp buffer (device)
    topk_ids_dev: *mut u8,     // [top_k] u16 output (device)
    topk_weights_dev: *mut u8, // [top_k] f32 output (device)
    stream: &CudaStream,
) -> Result<Vec<(u16, f32)>> {
    #[cfg(feature = "cuda")]
    {
        if stream.is_real() {
            let (scoring_mode, scaling_factor) = match scoring_func {
                RouterScoringFunc::Softmax => (0i32, 1.0f32),
                RouterScoringFunc::Sigmoid {
                    scaling_factor,
                    normalize: _,
                } => (1i32, scaling_factor),
            };

            // Launch fused GEMV + top-k on GPU (no sync needed between steps)
            let err = unsafe {
                cuda_ffi::vib3_launch_router_topk(
                    hidden_state_f32,
                    router_weights,
                    scores_dev,
                    std::ptr::null(), // no bias (NULL)
                    topk_ids_dev as *mut u16,
                    topk_weights_dev as *mut f32,
                    hidden_dim as i32,
                    num_experts as i32,
                    top_k as i32,
                    scoring_mode,
                    scaling_factor,
                    stream.raw_ptr(),
                )
            };
            if err != 0 {
                return Err(Error::Cuda(format!(
                    "router_topk launch failed (err={})",
                    err
                )));
            }

            // Single sync to wait for the tiny output (instead of syncing after GEMV)
            stream.synchronize()?;

            // Small D2H: top_k × (u16 + f32) = top_k × 6 bytes
            let id_bytes = top_k * std::mem::size_of::<u16>();
            let weight_bytes = top_k * std::mem::size_of::<f32>();

            let mut id_buf = vec![0u16; top_k];
            let mut weight_buf = vec![0f32; top_k];

            cuda_ffi::memcpy_d2h(id_buf.as_mut_ptr() as *mut u8, topk_ids_dev, id_bytes)?;
            cuda_ffi::memcpy_d2h(
                weight_buf.as_mut_ptr() as *mut u8,
                topk_weights_dev,
                weight_bytes,
            )?;

            let result: Vec<(u16, f32)> = id_buf
                .iter()
                .zip(weight_buf.iter())
                .map(|(&id, &w)| (id, w))
                .collect();

            return Ok(result);
        }
    }
    Err(Error::Cuda("run_router_gpu_topk requires GPU".to_string()))
}

/// GPU router that launches the kernel but does NOT synchronize or D2H.
/// Expert IDs and weights remain on device for consumption by
/// `moe_experts_fused_gpu`. No host-side expert list is returned.
pub fn run_router_gpu_topk_nosync(
    hidden_state_f32: *const u8, // FP32 normalized hidden state (device)
    router_weights: *const u8,   // FP16 router weight matrix (device)
    num_experts: usize,
    hidden_dim: usize,
    top_k: usize,
    scoring_func: RouterScoringFunc,
    scores_dev: *mut f32,      // [num_experts] temp buffer (device)
    topk_ids_dev: *mut u8,     // [top_k] u16 output (device)
    topk_weights_dev: *mut u8, // [top_k] f32 output (device)
    stream: &CudaStream,
) -> Result<()> {
    #[cfg(feature = "cuda")]
    {
        if stream.is_real() {
            let (scoring_mode, scaling_factor) = match scoring_func {
                RouterScoringFunc::Softmax => (0i32, 1.0f32),
                RouterScoringFunc::Sigmoid {
                    scaling_factor,
                    normalize: _,
                } => (1i32, scaling_factor),
            };

            let err = unsafe {
                cuda_ffi::vib3_launch_router_topk(
                    hidden_state_f32,
                    router_weights,
                    scores_dev,
                    std::ptr::null(),
                    topk_ids_dev as *mut u16,
                    topk_weights_dev as *mut f32,
                    hidden_dim as i32,
                    num_experts as i32,
                    top_k as i32,
                    scoring_mode,
                    scaling_factor,
                    stream.raw_ptr(),
                )
            };
            if err != 0 {
                return Err(Error::Cuda(format!(
                    "router_topk nosync launch failed (err={})",
                    err
                )));
            }
            return Ok(());
        }
    }
    let _ = (
        hidden_state_f32,
        router_weights,
        num_experts,
        hidden_dim,
        top_k,
        scoring_func,
        scores_dev,
        topk_ids_dev,
        topk_weights_dev,
        stream,
    );
    Err(Error::Cuda(
        "run_router_gpu_topk_nosync requires GPU".to_string(),
    ))
}

/// Weighted accumulation: output += weight × expert_output.
///
/// Accumulates one expert's contribution into the layer output,
/// scaled by the router weight.
pub fn weighted_accumulate(
    output: *mut u8,          // [dim] FP16, accumulated
    expert_output: *const u8, // [dim] FP16, single expert
    weight: f32,
    dim: usize,
    stream: &CudaStream,
) -> Result<()> {
    #[cfg(feature = "cuda")]
    {
        if stream.is_real() {
            let err = unsafe {
                cuda_ffi::vib3_launch_weighted_accumulate(
                    output,
                    expert_output,
                    weight,
                    dim as i32,
                    stream.raw_ptr(),
                )
            };
            if err != 0 {
                return Err(Error::Cuda(format!(
                    "weighted_accumulate launch failed (err={})",
                    err
                )));
            }
            return Ok(());
        }
    }

    let out = unsafe { std::slice::from_raw_parts_mut(output as *mut f16, dim) };
    let inp = unsafe { std::slice::from_raw_parts(expert_output as *const f16, dim) };
    for (o, i) in out.iter_mut().zip(inp.iter()) {
        let val = o.to_f32() + weight * i.to_f32();
        *o = f16::from_f32(val);
    }
    Ok(())
}

/// FP32 version: accumulate weighted expert output into FP32 buffer.
/// Eliminates FP16 truncation between expert accumulations.
/// `output` is FP32 [dim], `expert_output` is FP16 [dim].
pub fn weighted_accumulate_f32(
    output: *mut u8,          // [dim] FP32, accumulated
    expert_output: *const u8, // [dim] FP16, single expert
    weight: f32,
    dim: usize,
    stream: &CudaStream,
) -> Result<()> {
    #[cfg(feature = "cuda")]
    {
        if stream.is_real() {
            let err = unsafe {
                cuda_ffi::vib3_launch_weighted_accumulate_f32(
                    output,
                    expert_output,
                    weight,
                    dim as i32,
                    stream.raw_ptr(),
                )
            };
            if err != 0 {
                return Err(Error::Cuda(format!(
                    "weighted_accumulate_f32 launch failed (err={})",
                    err
                )));
            }
            return Ok(());
        }
    }

    // CPU fallback
    let out = unsafe { std::slice::from_raw_parts_mut(output as *mut f32, dim) };
    let inp = unsafe { std::slice::from_raw_parts(expert_output as *const f16, dim) };
    for (o, i) in out.iter_mut().zip(inp.iter()) {
        *o += weight * i.to_f32();
    }
    Ok(())
}

/// FP32→FP32 weighted accumulate: output[i] += weight * expert_output[i].
/// Both buffers are FP32 — no FP16 truncation anywhere in the pipeline.
pub fn weighted_accumulate_f32_f32(
    output: *mut u8,          // [dim] FP32, accumulated
    expert_output: *const u8, // [dim] FP32, single expert
    weight: f32,
    dim: usize,
    stream: &CudaStream,
) -> Result<()> {
    #[cfg(feature = "cuda")]
    {
        if stream.is_real() {
            let err = unsafe {
                cuda_ffi::vib3_launch_weighted_accumulate_f32_f32(
                    output,
                    expert_output,
                    weight,
                    dim as i32,
                    stream.raw_ptr(),
                )
            };
            if err != 0 {
                return Err(Error::Cuda(format!(
                    "weighted_accumulate_f32_f32 launch failed (err={})",
                    err
                )));
            }
            return Ok(());
        }
    }

    // CPU fallback
    let out = unsafe { std::slice::from_raw_parts_mut(output as *mut f32, dim) };
    let inp = unsafe { std::slice::from_raw_parts(expert_output as *const f32, dim) };
    for (o, i) in out.iter_mut().zip(inp.iter()) {
        *o += weight * i;
    }
    Ok(())
}

/// Sigmoid-gated FP16→FP32 accumulate: output[i] += sigmoid(gate_dev[0]) * f16(expert[i]).
/// Gate scalar is read from device memory; sigmoid computed on-GPU. No sync needed.
pub fn sigmoid_gated_accumulate_f32(
    output: *mut u8,          // [dim] FP32, accumulated
    expert_output: *const u8, // [dim] FP16, shared expert output
    gate_dev: *const u8,      // [1] FP32, gate scalar on device
    dim: usize,
    stream: &CudaStream,
) -> Result<()> {
    #[cfg(feature = "cuda")]
    {
        if stream.is_real() {
            let err = unsafe {
                cuda_ffi::vib3_launch_sigmoid_gated_accumulate_f32(
                    output,
                    expert_output,
                    gate_dev,
                    dim as i32,
                    stream.raw_ptr(),
                )
            };
            if err != 0 {
                return Err(Error::Cuda(format!(
                    "sigmoid_gated_accumulate_f32 launch failed (err={})",
                    err
                )));
            }
            return Ok(());
        }
    }

    // CPU fallback: read gate, apply sigmoid, accumulate
    let gate_raw = unsafe { *(gate_dev as *const f32) };
    let gate_sigmoid = 1.0 / (1.0 + (-gate_raw).exp());
    let out = unsafe { std::slice::from_raw_parts_mut(output as *mut f32, dim) };
    let inp = unsafe { std::slice::from_raw_parts(expert_output as *const half::f16, dim) };
    for (o, i) in out.iter_mut().zip(inp.iter()) {
        *o += gate_sigmoid * i.to_f32();
    }
    Ok(())
}

/// FP32 SwiGLU fuse: output[i] = silu(gate[i]) * up[i], all FP32.
pub fn swiglu_fuse_f32(
    gate: *const u8,
    up: *const u8,
    output: *mut u8,
    dim: usize,
    stream: &CudaStream,
) -> Result<()> {
    #[cfg(feature = "cuda")]
    {
        if stream.is_real() {
            let err = unsafe {
                cuda_ffi::vib3_launch_swiglu_fuse_f32(
                    gate,
                    up,
                    output,
                    dim as i32,
                    stream.raw_ptr(),
                )
            };
            if err != 0 {
                return Err(Error::Cuda(format!(
                    "swiglu_fuse_f32 launch failed (err={})",
                    err
                )));
            }
            return Ok(());
        }
    }
    Err(Error::Cuda("swiglu_fuse_f32 requires GPU".to_string()))
}

/// Sigmoid-gated accumulate with FP32 expert output (NVFP4 MMA path).
/// output[i] += sigmoid(gate_dev[0]) * expert_output[i], all FP32.
pub fn sigmoid_gated_accumulate_f32in(
    output: *mut u8,
    expert_output: *const u8,
    gate_dev: *const u8,
    dim: usize,
    stream: &CudaStream,
) -> Result<()> {
    #[cfg(feature = "cuda")]
    {
        if stream.is_real() {
            let err = unsafe {
                cuda_ffi::vib3_launch_sigmoid_gated_accumulate_f32in(
                    output,
                    expert_output,
                    gate_dev,
                    dim as i32,
                    stream.raw_ptr(),
                )
            };
            if err != 0 {
                return Err(Error::Cuda(format!(
                    "sigmoid_gated_accumulate_f32in launch failed (err={})",
                    err
                )));
            }
            return Ok(());
        }
    }
    // CPU fallback
    let gate_raw = unsafe { *(gate_dev as *const f32) };
    let gate_sigmoid = 1.0 / (1.0 + (-gate_raw).exp());
    let out = unsafe { std::slice::from_raw_parts_mut(output as *mut f32, dim) };
    let inp = unsafe { std::slice::from_raw_parts(expert_output as *const f32, dim) };
    for (o, i) in out.iter_mut().zip(inp.iter()) {
        *o += gate_sigmoid * i;
    }
    Ok(())
}

/// FP32-to-FP32 residual add: accumulator[i] += layer_output[i]
/// Used when layer_output is already FP32 (from FP32 MoE accumulation).
pub fn residual_add_f32_f32(
    accumulator: *mut u8,
    layer_output: *const u8,
    dim: usize,
    stream: &CudaStream,
) -> Result<()> {
    #[cfg(feature = "cuda")]
    {
        if stream.is_real() {
            let err = unsafe {
                cuda_ffi::vib3_launch_residual_add_f32_f32(
                    accumulator,
                    layer_output,
                    dim as i32,
                    stream.raw_ptr(),
                )
            };
            if err != 0 {
                return Err(Error::Cuda(format!(
                    "residual_add_f32_f32 launch failed (err={})",
                    err
                )));
            }
            return Ok(());
        }
    }
    Err(Error::Cuda("residual_add_f32_f32 requires GPU".to_string()))
}

/// Router scoring function.
#[derive(Clone, Copy, PartialEq, Debug, Default)]
pub enum RouterScoringFunc {
    /// Standard softmax over top-K scores (Mixtral, Switch Transformer).
    #[default]
    Softmax,
    /// Sigmoid gating with scaling factor (DeepSeek-V3, Kimi K2.5).
    /// Each selected expert's weight = sigmoid(score) * scaling_factor.
    /// When `normalize` is true, weights are divided by their sum after scaling.
    Sigmoid {
        scaling_factor: f32,
        normalize: bool,
    },
}

/// Run the router network for one layer.
///
/// Router is a small linear projection + top-k selection:
///   scores = hidden_state × router_weights^T  → [num_experts]
///   top_k(scores, k) → [(expert_id, weight)]
///   weight normalization via scoring_func (softmax or sigmoid)
pub fn run_router(
    hidden_state: *const u8,   // [hidden_dim] FP16
    router_weights: *const u8, // [num_experts, hidden_dim] FP16
    num_experts: usize,
    hidden_dim: usize,
    top_k: usize,
    _stream: &CudaStream,
    scores_dev: Option<*mut u8>,
) -> Result<Vec<(u16, f32)>> {
    run_router_with_scoring(
        hidden_state,
        router_weights,
        num_experts,
        hidden_dim,
        top_k,
        RouterScoringFunc::Softmax,
        _stream,
        scores_dev,
    )
}

/// Run the router with a specific scoring function.
///
/// `scores_dev`: Optional pre-allocated device buffer (f32, [num_experts]) for router
/// GEMV output. When Some, avoids per-call cudaMalloc/cudaFree.
///
/// For Kimi K2.5: `scoring_func = Sigmoid { scaling_factor: 2.827, normalize: true }`.
/// For Mixtral: `scoring_func = Softmax`.
#[allow(clippy::too_many_arguments)]
pub fn run_router_with_scoring(
    hidden_state: *const u8,   // [hidden_dim] FP16
    router_weights: *const u8, // [num_experts, hidden_dim] FP16
    num_experts: usize,
    hidden_dim: usize,
    top_k: usize,
    scoring_func: RouterScoringFunc,
    stream: &CudaStream,
    scores_dev: Option<*mut u8>,
) -> Result<Vec<(u16, f32)>> {
    run_router_with_scoring_and_bias(
        hidden_state,
        router_weights,
        num_experts,
        hidden_dim,
        top_k,
        scoring_func,
        stream,
        scores_dev,
        None,
    )
}

/// Run router with optional e_score_correction_bias.
///
/// For sigmoid routing (DeepSeek-V3/Kimi K2.5), the bias is added to sigmoid
/// scores BEFORE top-k selection, but the final expert weights use the original
/// sigmoid values (without bias). This matches the reference implementation:
///   scores = sigmoid(linear(hidden, weight))
///   scores_for_choice = scores + e_score_correction_bias
///   top_k_indices = top_k(scores_for_choice, k)
///   weights = scores[top_k_indices] * scaling_factor  (normalized if norm_topk_prob)
#[allow(clippy::too_many_arguments)]
pub fn run_router_with_scoring_and_bias(
    hidden_state: *const u8,   // [hidden_dim] FP16
    router_weights: *const u8, // [num_experts, hidden_dim] FP16
    num_experts: usize,
    hidden_dim: usize,
    top_k: usize,
    scoring_func: RouterScoringFunc,
    stream: &CudaStream,
    scores_dev: Option<*mut u8>,
    correction_bias: Option<&[f32]>,
) -> Result<Vec<(u16, f32)>> {
    // GPU path: compute scores on GPU, copy to host for top-k selection
    #[cfg(feature = "cuda")]
    {
        if stream.is_real() {
            return gpu_run_router(
                hidden_state,
                router_weights,
                num_experts,
                hidden_dim,
                top_k,
                scoring_func,
                stream,
                scores_dev,
                correction_bias,
            );
        }
    }
    let _ = scores_dev; // suppress unused warning in non-CUDA builds

    let state = unsafe { std::slice::from_raw_parts(hidden_state as *const f16, hidden_dim) };
    let weights = unsafe {
        std::slice::from_raw_parts(router_weights as *const f16, num_experts * hidden_dim)
    };

    let state_f32: Vec<f32> = state.iter().map(|v| v.to_f32()).collect();
    let mut scores = vec![0.0f32; num_experts];
    if num_experts >= PARALLEL_ROW_THRESHOLD {
        scores.par_iter_mut().enumerate().for_each(|(e, score)| {
            let row_offset = e * hidden_dim;
            let mut acc = 0.0f32;
            for d in 0..hidden_dim {
                acc += state_f32[d] * weights[row_offset + d].to_f32();
            }
            *score = acc;
        });
    } else {
        for (e, score) in scores.iter_mut().enumerate().take(num_experts) {
            let row_offset = e * hidden_dim;
            let mut acc = 0.0f32;
            for d in 0..hidden_dim {
                acc += state_f32[d] * weights[row_offset + d].to_f32();
            }
            *score = acc;
        }
    }

    // Top-K selection — method depends on scoring function.
    // For sigmoid routing with correction_bias, top-k uses biased sigmoid scores
    // but final weights use the original (unbiased) sigmoid values.
    let result: Vec<(u16, f32)> = match scoring_func {
        RouterScoringFunc::Softmax => {
            // Softmax: top-k on raw scores, then softmax over selected
            let mut indexed: Vec<(usize, f32)> = scores.iter().copied().enumerate().collect();
            indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            indexed.truncate(top_k);

            let max_score = indexed
                .iter()
                .map(|(_, s)| *s)
                .fold(f32::NEG_INFINITY, f32::max);
            let exp_sum: f32 = indexed.iter().map(|(_, s)| (s - max_score).exp()).sum();

            indexed
                .iter()
                .map(|(idx, score)| {
                    let softmax_weight = (score - max_score).exp() / exp_sum;
                    (*idx as u16, softmax_weight)
                })
                .collect()
        }
        RouterScoringFunc::Sigmoid {
            scaling_factor,
            normalize,
        } => {
            // Sigmoid gating (DeepSeek-V3 / Kimi K2.5):
            //   1. Compute sigmoid of ALL raw scores
            //   2. Add e_score_correction_bias (if present) for top-k selection
            //   3. Top-k on biased scores
            //   4. Expert weights = original sigmoid (without bias) * scaling_factor
            //   5. Normalize weights if norm_topk_prob=true
            let sigmoid_scores: Vec<f32> =
                scores.iter().map(|s| 1.0 / (1.0 + (-s).exp())).collect();

            // Scores for top-k selection (with bias)
            let selection_scores: Vec<f32> = if let Some(bias) = correction_bias {
                sigmoid_scores
                    .iter()
                    .enumerate()
                    .map(|(i, &s)| if i < bias.len() { s + bias[i] } else { s })
                    .collect()
            } else {
                sigmoid_scores.clone()
            };

            let mut indexed: Vec<(usize, f32)> =
                selection_scores.iter().copied().enumerate().collect();
            indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            indexed.truncate(top_k);

            // Use original sigmoid scores (without bias) for weights.
            // DeepSeek-V3 / Kimi K2.5 order: normalize FIRST, then scale.
            let mut result: Vec<(u16, f32)> = indexed
                .iter()
                .map(|(idx, _)| (*idx as u16, sigmoid_scores[*idx]))
                .collect();
            if normalize {
                let total: f32 = result.iter().map(|(_, w)| w).sum();
                if total > 1e-20 {
                    for (_, w) in &mut result {
                        *w /= total;
                    }
                }
            }
            // Apply routed_scaling_factor AFTER normalization
            for (_, w) in &mut result {
                *w *= scaling_factor;
            }
            result
        }
    };

    Ok(result)
}

/// GPU router: GEMV on GPU → copy scores to host → top-K + scoring on CPU.
#[cfg(feature = "cuda")]
#[allow(clippy::too_many_arguments)]
fn gpu_run_router(
    hidden_state: *const u8,
    router_weights: *const u8,
    num_experts: usize,
    hidden_dim: usize,
    top_k: usize,
    scoring_func: RouterScoringFunc,
    stream: &CudaStream,
    scores_dev: Option<*mut u8>,
    correction_bias: Option<&[f32]>,
) -> Result<Vec<(u16, f32)>> {
    // Use pre-allocated device buffer for scores, or allocate on the fly
    let (scores_device, allocated) = if let Some(dev_ptr) = scores_dev {
        (dev_ptr, false)
    } else {
        (cuda_ffi::device_alloc(num_experts * 4)?, true)
    };

    // Launch GEMV on GPU
    let err = unsafe {
        cuda_ffi::vib3_launch_router_gemv(
            hidden_state,
            router_weights,
            scores_device as *mut f32,
            hidden_dim as i32,
            num_experts as i32,
            stream.raw_ptr(),
        )
    };
    if err != 0 {
        if allocated {
            cuda_ffi::device_free(scores_device, num_experts * 4);
        }
        return Err(Error::Cuda(format!(
            "router_gemv launch failed (err={})",
            err
        )));
    }

    // Sync and copy scores to host
    stream.synchronize()?;
    let mut scores = vec![0.0f32; num_experts];
    cuda_ffi::memcpy_d2h(
        scores.as_mut_ptr() as *mut u8,
        scores_device as *const u8,
        num_experts * 4,
    )?;
    if allocated {
        cuda_ffi::device_free(scores_device, num_experts * 4);
    }

    // Top-K selection on CPU (tiny amount of work)
    let result: Vec<(u16, f32)> = match scoring_func {
        RouterScoringFunc::Softmax => {
            let mut indexed: Vec<(usize, f32)> = scores.iter().copied().enumerate().collect();
            indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            indexed.truncate(top_k);

            let max_score = indexed
                .iter()
                .map(|(_, s)| *s)
                .fold(f32::NEG_INFINITY, f32::max);
            let exp_sum: f32 = indexed.iter().map(|(_, s)| (s - max_score).exp()).sum();
            indexed
                .iter()
                .map(|(idx, score)| (*idx as u16, (score - max_score).exp() / exp_sum))
                .collect()
        }
        RouterScoringFunc::Sigmoid {
            scaling_factor,
            normalize,
        } => {
            // Apply sigmoid to all scores, add correction bias for selection,
            // but use original sigmoid values for final weights.
            let sigmoid_scores: Vec<f32> =
                scores.iter().map(|s| 1.0 / (1.0 + (-s).exp())).collect();

            let selection_scores: Vec<f32> = if let Some(bias) = correction_bias {
                sigmoid_scores
                    .iter()
                    .enumerate()
                    .map(|(i, &s)| if i < bias.len() { s + bias[i] } else { s })
                    .collect()
            } else {
                sigmoid_scores.clone()
            };

            let mut indexed: Vec<(usize, f32)> =
                selection_scores.iter().copied().enumerate().collect();
            indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            indexed.truncate(top_k);

            // DeepSeek-V3 / Kimi K2.5 order: normalize FIRST, then scale.
            let mut result: Vec<(u16, f32)> = indexed
                .iter()
                .map(|(idx, _)| (*idx as u16, sigmoid_scores[*idx]))
                .collect();
            if normalize {
                let total: f32 = result.iter().map(|(_, w)| w).sum();
                if total > 1e-20 {
                    for (_, w) in &mut result {
                        *w /= total;
                    }
                }
            }
            // Apply routed_scaling_factor AFTER normalization
            for (_, w) in &mut result {
                *w *= scaling_factor;
            }
            result
        }
    };

    Ok(result)
}

/// RMSNorm: normalize hidden state in-place.
///
/// Computes: output[d] = (input[d] / rms) * weight[d]
/// where rms = sqrt(mean(input^2) + eps)
pub fn rms_norm(
    input: *mut u8,    // [hidden_dim] FP16, modified in-place
    weight: *const u8, // [hidden_dim] FP16 norm weights
    hidden_dim: usize,
    eps: f32,
    stream: &CudaStream,
) -> Result<()> {
    #[cfg(feature = "cuda")]
    {
        if stream.is_real() {
            let err = unsafe {
                cuda_ffi::vib3_launch_rms_norm(
                    input,
                    weight,
                    hidden_dim as i32,
                    eps,
                    stream.raw_ptr(),
                )
            };
            if err != 0 {
                return Err(Error::Cuda(format!("rms_norm launch failed (err={})", err)));
            }
            return Ok(());
        }
    }

    let state = unsafe { std::slice::from_raw_parts_mut(input as *mut f16, hidden_dim) };
    let w = unsafe { std::slice::from_raw_parts(weight as *const f16, hidden_dim) };
    let mut sum_sq = 0.0f32;
    for s in state.iter() {
        let val = s.to_f32();
        sum_sq += val * val;
    }
    let rms = ((sum_sq / hidden_dim as f32) + eps).sqrt();
    let inv_rms = 1.0 / rms;
    for (s, wt) in state.iter_mut().zip(w.iter()) {
        let val = s.to_f32() * inv_rms * wt.to_f32();
        *s = f16::from_f32(val);
    }
    Ok(())
}

/// RMSNorm without weight (unit norm).
///
/// Computes: output[d] = input[d] / rms
/// where rms = sqrt(mean(input^2) + eps)
pub fn rms_norm_no_weight(
    input: *mut u8, // [hidden_dim] FP16, modified in-place
    hidden_dim: usize,
    eps: f32,
    stream: &CudaStream,
) -> Result<()> {
    #[cfg(feature = "cuda")]
    {
        if stream.is_real() {
            let err = unsafe {
                cuda_ffi::vib3_launch_rms_norm_no_weight(
                    input,
                    hidden_dim as i32,
                    eps,
                    stream.raw_ptr(),
                )
            };
            if err != 0 {
                return Err(Error::Cuda(format!(
                    "rms_norm_no_weight launch failed (err={})",
                    err
                )));
            }
            return Ok(());
        }
    }

    let state = unsafe { std::slice::from_raw_parts_mut(input as *mut f16, hidden_dim) };
    let mut sum_sq = 0.0f32;
    for s in state.iter() {
        let val = s.to_f32();
        sum_sq += val * val;
    }
    let rms = ((sum_sq / hidden_dim as f32) + eps).sqrt();
    let inv_rms = 1.0 / rms;
    for s in state.iter_mut() {
        let val = s.to_f32() * inv_rms;
        *s = f16::from_f32(val);
    }
    Ok(())
}

// ─── CPU Fallback Implementations ────────────────────────────────────────

/// Minimum number of output rows before we spawn parallel threads.
/// Below this threshold, the thread-spawn overhead exceeds the compute savings.
/// At ~4096 cols (FP16), each row takes ~8 μs — threads need ~50 μs to spawn.
const PARALLEL_ROW_THRESHOLD: usize = 64;

/// FP16 × FP16 matmul on CPU (parallelized across output rows).
fn cpu_matmul_fp16(input: *const u8, weight: *const u8, output: *mut u8, k: usize, m_slice: usize) {
    let inp = unsafe { std::slice::from_raw_parts(input as *const f16, k) };
    let w = unsafe { std::slice::from_raw_parts(weight as *const f16, m_slice * k) };
    let out = unsafe { std::slice::from_raw_parts_mut(output as *mut f16, m_slice) };

    // Pre-convert input to f32 once (amortized across all rows)
    let inp_f32: Vec<f32> = inp.iter().map(|v| v.to_f32()).collect();

    if m_slice >= PARALLEL_ROW_THRESHOLD {
        // Parallel: each row computed independently
        out.par_iter_mut().enumerate().for_each(|(row, out_val)| {
            let row_offset = row * k;
            let mut acc = 0.0f32;
            for col in 0..k {
                acc += inp_f32[col] * w[row_offset + col].to_f32();
            }
            *out_val = f16::from_f32(acc);
        });
    } else {
        // Sequential for small slices
        for (row, out_val) in out.iter_mut().enumerate() {
            let row_offset = row * k;
            let mut acc = 0.0f32;
            for col in 0..k {
                acc += inp_f32[col] * w[row_offset + col].to_f32();
            }
            *out_val = f16::from_f32(acc);
        }
    }
}

/// BF16 × BF16 matmul on CPU (parallelized across output rows).
fn cpu_matmul_bf16(input: *const u8, weight: *const u8, output: *mut u8, k: usize, m_slice: usize) {
    let inp = unsafe { std::slice::from_raw_parts(input as *const u16, k) };
    let w = unsafe { std::slice::from_raw_parts(weight as *const u16, m_slice * k) };
    let out = unsafe { std::slice::from_raw_parts_mut(output as *mut u16, m_slice) };

    // Pre-convert input to f32 once
    let inp_f32: Vec<f32> = inp.iter().map(|v| bf16_to_f32(*v)).collect();

    if m_slice >= PARALLEL_ROW_THRESHOLD {
        out.par_iter_mut().enumerate().for_each(|(row, out_val)| {
            let row_offset = row * k;
            let mut acc = 0.0f32;
            for col in 0..k {
                acc += inp_f32[col] * bf16_to_f32(w[row_offset + col]);
            }
            *out_val = f32_to_bf16(acc);
        });
    } else {
        for (row, out_val) in out.iter_mut().enumerate() {
            let row_offset = row * k;
            let mut acc = 0.0f32;
            for col in 0..k {
                acc += inp_f32[col] * bf16_to_f32(w[row_offset + col]);
            }
            *out_val = f32_to_bf16(acc);
        }
    }
}

/// INT4 (packed) × FP16 matmul on CPU with LUT-based accumulation.
///
/// Instead of multiply-accumulate per nibble, we precompute a 16-entry
/// lookup table per input element group: `LUT[w] = input_val * dequant(w)`.
/// The inner loop becomes `acc += LUT[packed & 0x0F]; acc += LUT[packed >> 4]`
/// — no multiplies, no branches. This is the BitNet.cpp T-MAC idea adapted
/// to INT4 with group-wise quantization scales.
///
/// For K=4096, this precomputes 16 LUT entries per group of input elements
/// (one per quantization group), then processes the weight matrix with
/// pure table lookups. The LUT fits in L1 cache (16 × 4 = 64 bytes per group).
fn cpu_matmul_int4(input: *const u8, weight: *const u8, output: *mut u8, k: usize, m_slice: usize) {
    cpu_matmul_int4_grouped(input, weight, output, k, m_slice, INT4_GROUP_SIZE)
}

/// INT4 matmul with explicit group size parameter (parallelized across output rows).
///
/// This allows models with different quantization group sizes (32, 64, 128)
/// to be computed correctly without changing the global constant.
///
/// Uses a per-row LUT-based dequantization strategy:
/// - Each row has per-group scales stored as FP16 after the packed data
/// - For each group, build a 16-entry LUT: `lut[nibble] = (nibble - 8) * scale`
/// - Inner loop: `acc += inp_f32[col] * lut[packed_nibble]` (no branch, no multiply for dequant)
fn cpu_matmul_int4_grouped(
    input: *const u8,
    weight: *const u8,
    output: *mut u8,
    k: usize,
    m_slice: usize,
    group_size: usize,
) {
    let inp = unsafe { std::slice::from_raw_parts(input as *const f16, k) };
    let packed_k = k.div_ceil(2); // 2 INT4 values per byte
    let num_groups = k.div_ceil(group_size);
    let weight_bytes = packed_k * m_slice;
    let scales_offset = weight_bytes;
    let total_weight_bytes = weight_bytes + m_slice * num_groups * 2;

    let w = unsafe { std::slice::from_raw_parts(weight, total_weight_bytes) };
    let out = unsafe { std::slice::from_raw_parts_mut(output as *mut f16, m_slice) };

    // Pre-convert input to f32 (done once, shared across all rows)
    let inp_f32: Vec<f32> = inp.iter().map(|v| v.to_f32()).collect();

    // Per-row computation closure — used by both parallel and sequential paths
    let compute_row = |row: usize| -> f32 {
        let row_packed = &w[row * packed_k..(row + 1) * packed_k];
        let row_scales_start = scales_offset + row * num_groups * 2;

        // Build dequant LUT for this row's scales
        // For k <= 4096 (group_size=32), we have ≤128 groups and use a stack array.
        // For larger k (e.g. down_proj with k=14336 → 448 groups), heap-allocate.
        let mut dequant_lut_stack = [[0.0f32; 16]; 128];
        let mut dequant_lut_heap: Vec<[f32; 16]>;
        let dequant_lut: &mut [[f32; 16]] = if num_groups <= 128 {
            &mut dequant_lut_stack[..num_groups]
        } else {
            dequant_lut_heap = vec![[0.0f32; 16]; num_groups];
            &mut dequant_lut_heap[..]
        };
        let num_groups_actual = num_groups;

        for (group, lut_entry) in dequant_lut.iter_mut().enumerate().take(num_groups_actual) {
            let scale_idx = row_scales_start + group * 2;
            let scale = if scale_idx + 1 < w.len() {
                // Scales are BF16 (not FP16) — must decode as BF16
                let s_bits = u16::from_le_bytes([w[scale_idx], w[scale_idx + 1]]);
                let s = bf16_to_f32(s_bits);
                if s.is_finite() && s > 0.0 {
                    s
                } else {
                    1.0 / 8.0
                }
            } else {
                1.0 / 8.0
            };
            for nibble in 0u8..16 {
                lut_entry[nibble as usize] = (nibble as i8 - 8) as f32 * scale;
            }
        }

        // Inner loop: pure LUT lookups, no multiplies for dequantization
        let mut acc = 0.0f32;
        for (byte_idx, &packed) in row_packed.iter().enumerate().take(packed_k) {
            let col = byte_idx * 2;
            if col >= k {
                break;
            }
            let group = (col / group_size).min(num_groups_actual - 1);

            // Low nibble
            acc += inp_f32[col] * dequant_lut[group][(packed & 0x0F) as usize];
            // High nibble
            if col + 1 < k {
                acc += inp_f32[col + 1] * dequant_lut[group][(packed >> 4) as usize];
            }
        }
        acc
    };

    if m_slice >= PARALLEL_ROW_THRESHOLD {
        out.par_iter_mut().enumerate().for_each(|(row, out_val)| {
            *out_val = f16::from_f32(compute_row(row));
        });
    } else {
        for (row, out_val) in out.iter_mut().enumerate() {
            *out_val = f16::from_f32(compute_row(row));
        }
    }
}

/// INT8 × FP16 matmul on CPU with per-row dequantization scales (parallelized).
///
/// Weight layout: [m_slice * k bytes of INT8 data] [m_slice * 4 bytes of f32 per-row scales]
/// If the buffer is too small to contain per-row scales, falls back to a default scale of 1/127.
fn cpu_matmul_int8(input: *const u8, weight: *const u8, output: *mut u8, k: usize, m_slice: usize) {
    let inp = unsafe { std::slice::from_raw_parts(input as *const f16, k) };
    let weight_data_bytes = m_slice * k;

    let w = unsafe { std::slice::from_raw_parts(weight as *const i8, weight_data_bytes) };
    let out = unsafe { std::slice::from_raw_parts_mut(output as *mut f16, m_slice) };

    // Read per-row scales from after the weight data as a slice (Send + Sync safe)
    let scales: &[f32] =
        unsafe { std::slice::from_raw_parts(weight.add(weight_data_bytes) as *const f32, m_slice) };

    // Pre-convert input to f32 once
    let inp_f32: Vec<f32> = inp.iter().map(|v| v.to_f32()).collect();

    let compute_row = |row: usize| -> f16 {
        let s = scales[row];
        let scale = if s.is_finite() && s > 0.0 && s < 1000.0 {
            s
        } else {
            1.0f32 / 127.0
        };

        let mut acc = 0.0f32;
        let row_offset = row * k;
        for col in 0..k {
            acc += inp_f32[col] * w[row_offset + col] as f32 * scale;
        }
        f16::from_f32(acc)
    };

    if m_slice >= PARALLEL_ROW_THRESHOLD {
        out.par_iter_mut().enumerate().for_each(|(row, out_val)| {
            *out_val = compute_row(row);
        });
    } else {
        for (row, out_val) in out.iter_mut().enumerate() {
            *out_val = compute_row(row);
        }
    }
}

/// FP16 fused SwiGLU on CPU (parallelized across output rows).
fn cpu_swiglu_fp16(
    input: *const u8,
    up_page: *const u8,
    gate_page: *const u8,
    output: *mut u8,
    hidden_dim: usize,
    m_slice: usize,
) {
    let inp = unsafe { std::slice::from_raw_parts(input as *const f16, hidden_dim) };
    let up = unsafe { std::slice::from_raw_parts(up_page as *const f16, m_slice * hidden_dim) };
    let gate = unsafe { std::slice::from_raw_parts(gate_page as *const f16, m_slice * hidden_dim) };
    let out = unsafe { std::slice::from_raw_parts_mut(output as *mut f16, m_slice) };

    // Pre-convert input to f32 once
    let inp_f32: Vec<f32> = inp.iter().map(|v| v.to_f32()).collect();

    let compute_row = |row: usize| -> f16 {
        let mut up_acc = 0.0f32;
        let mut gate_acc = 0.0f32;
        let row_offset = row * hidden_dim;

        for d in 0..hidden_dim {
            up_acc += inp_f32[d] * up[row_offset + d].to_f32();
            gate_acc += inp_f32[d] * gate[row_offset + d].to_f32();
        }

        // SwiGLU: SiLU(gate) * up — SiLU applied to gate_proj output
        let silu = gate_acc / (1.0 + (-gate_acc).exp());
        f16::from_f32(silu * up_acc)
    };

    if m_slice >= PARALLEL_ROW_THRESHOLD {
        out.par_iter_mut().enumerate().for_each(|(row, out_val)| {
            *out_val = compute_row(row);
        });
    } else {
        for (row, out_val) in out.iter_mut().enumerate() {
            *out_val = compute_row(row);
        }
    }
}

/// INT4 fused SwiGLU on CPU with per-group dequantization scales (parallelized).
///
/// Uses the same grouped INT4 layout as `cpu_matmul_int4_grouped`:
/// Weight data: `[m_slice * packed_k bytes]`
/// Scales:      `[m_slice * num_groups * 2 bytes (FP16)]`
///
/// Dequantization: `(nibble - 8) * scale_for_group`
///
/// This is the single hottest kernel in MoE inference — called twice per expert
/// per layer (up+gate fused). Parallelized across output rows with Rayon.
fn cpu_swiglu_int4(
    input: *const u8,
    up_page: *const u8,
    gate_page: *const u8,
    output: *mut u8,
    hidden_dim: usize,
    m_slice: usize,
) {
    let inp = unsafe { std::slice::from_raw_parts(input as *const f16, hidden_dim) };
    let out = unsafe { std::slice::from_raw_parts_mut(output as *mut f16, m_slice) };

    let group_size = INT4_GROUP_SIZE;
    let packed_k = hidden_dim.div_ceil(2);
    let num_groups = hidden_dim.div_ceil(group_size);
    let weight_bytes = packed_k * m_slice;
    let total_bytes = weight_bytes + m_slice * num_groups * 2;

    let up_data = unsafe { std::slice::from_raw_parts(up_page, total_bytes) };
    let gate_data = unsafe { std::slice::from_raw_parts(gate_page, total_bytes) };

    // Pre-convert input to f32 (shared across all rows)
    let inp_f32: Vec<f32> = inp.iter().map(|v| v.to_f32()).collect();

    // Per-row computation closure
    let compute_row = |row: usize| -> f16 {
        let up_row_packed = &up_data[row * packed_k..(row + 1) * packed_k];
        let gate_row_packed = &gate_data[row * packed_k..(row + 1) * packed_k];
        let up_scales_start = weight_bytes + row * num_groups * 2;
        let gate_scales_start = weight_bytes + row * num_groups * 2;

        // Build dequant LUTs for this row
        // For hidden_dim <= 4096 (group_size=32), we have ≤128 groups and use stack arrays.
        // For larger dimensions, heap-allocate.
        let mut up_lut_stack = [[0.0f32; 16]; 128];
        let mut gate_lut_stack = [[0.0f32; 16]; 128];
        let mut up_lut_heap: Vec<[f32; 16]>;
        let mut gate_lut_heap: Vec<[f32; 16]>;
        let (up_lut, gate_lut): (&mut [[f32; 16]], &mut [[f32; 16]]) = if num_groups <= 128 {
            (
                &mut up_lut_stack[..num_groups],
                &mut gate_lut_stack[..num_groups],
            )
        } else {
            up_lut_heap = vec![[0.0f32; 16]; num_groups];
            gate_lut_heap = vec![[0.0f32; 16]; num_groups];
            (&mut up_lut_heap[..], &mut gate_lut_heap[..])
        };
        let ng = num_groups;

        for group in 0..ng {
            let up_scale_idx = up_scales_start + group * 2;
            let up_scale = if up_scale_idx + 1 < up_data.len() {
                // Scales are BF16 (not FP16) — must decode as BF16
                let s_bits = u16::from_le_bytes([up_data[up_scale_idx], up_data[up_scale_idx + 1]]);
                let s = bf16_to_f32(s_bits);
                if s.is_finite() && s > 0.0 {
                    s
                } else {
                    1.0 / 8.0
                }
            } else {
                1.0 / 8.0
            };

            let gate_scale_idx = gate_scales_start + group * 2;
            let gate_scale = if gate_scale_idx + 1 < gate_data.len() {
                // Scales are BF16 (not FP16) — must decode as BF16
                let s_bits =
                    u16::from_le_bytes([gate_data[gate_scale_idx], gate_data[gate_scale_idx + 1]]);
                let s = bf16_to_f32(s_bits);
                if s.is_finite() && s > 0.0 {
                    s
                } else {
                    1.0 / 8.0
                }
            } else {
                1.0 / 8.0
            };

            for nibble in 0u8..16 {
                up_lut[group][nibble as usize] = (nibble as i8 - 8) as f32 * up_scale;
                gate_lut[group][nibble as usize] = (nibble as i8 - 8) as f32 * gate_scale;
            }
        }

        // Inner loop: dual LUT-based dot product
        let mut up_acc = 0.0f32;
        let mut gate_acc = 0.0f32;

        for byte_idx in 0..packed_k {
            let col = byte_idx * 2;
            if col >= hidden_dim {
                break;
            }
            let group = (col / group_size).min(ng - 1);

            let up_packed = up_row_packed[byte_idx];
            let gate_packed = gate_row_packed[byte_idx];

            up_acc += inp_f32[col] * up_lut[group][(up_packed & 0x0F) as usize];
            gate_acc += inp_f32[col] * gate_lut[group][(gate_packed & 0x0F) as usize];

            if col + 1 < hidden_dim {
                up_acc += inp_f32[col + 1] * up_lut[group][(up_packed >> 4) as usize];
                gate_acc += inp_f32[col + 1] * gate_lut[group][(gate_packed >> 4) as usize];
            }
        }

        // SwiGLU: SiLU(gate) * up — SiLU applied to gate_proj output
        let silu = gate_acc / (1.0 + (-gate_acc).exp());
        f16::from_f32(silu * up_acc)
    };

    if m_slice >= PARALLEL_ROW_THRESHOLD {
        out.par_iter_mut().enumerate().for_each(|(row, out_val)| {
            *out_val = compute_row(row);
        });
    } else {
        for (row, out_val) in out.iter_mut().enumerate() {
            *out_val = compute_row(row);
        }
    }
}

// ─── RoPE (Rotary Position Embeddings) ───────────────────────────────────

/// Apply Rotary Position Embeddings in-place to a Q or K vector.
///
/// For each pair of dimensions (2i, 2i+1):
///   q'[2i]   = q[2i] * cos(θ) - q[2i+1] * sin(θ)
///   q'[2i+1] = q[2i] * sin(θ) + q[2i+1] * cos(θ)
/// where θ = position / 10000^(2i/dim)
pub fn apply_rope(
    x: &mut [f32], // [head_dim] in f32
    position: usize,
    head_dim: usize,
    rope_base: f32, // typically 10000.0
) {
    let half_dim = head_dim / 2;
    for i in 0..half_dim {
        let freq = 1.0 / rope_base.powf(2.0 * i as f32 / head_dim as f32);
        let theta = position as f32 * freq;
        let cos_t = theta.cos();
        let sin_t = theta.sin();

        let x0 = x[2 * i];
        let x1 = x[2 * i + 1];
        x[2 * i] = x0 * cos_t - x1 * sin_t;
        x[2 * i + 1] = x0 * sin_t + x1 * cos_t;
    }
}

/// Apply RoPE to FP16 data in-place.
pub fn apply_rope_fp16(
    x: *mut u8, // [head_dim] FP16
    position: usize,
    head_dim: usize,
    rope_base: f32,
) {
    let data = unsafe { std::slice::from_raw_parts_mut(x as *mut f16, head_dim) };
    let mut buf: Vec<f32> = data.iter().map(|v| v.to_f32()).collect();
    apply_rope(&mut buf, position, head_dim, rope_base);
    for (d, v) in data.iter_mut().zip(buf.iter()) {
        *d = f16::from_f32(*v);
    }
}

// ─── Attention ───────────────────────────────────────────────────────────

/// Single-head attention: Q × K^T / sqrt(d) → softmax → × V
///
/// Computes one attention head given Q vector, K cache, V cache.
/// Returns the attention output vector [head_dim].
pub fn attention_head(
    q: &[f32],       // [head_dim]
    k_cache: &[f32], // [seq_len, head_dim]
    v_cache: &[f32], // [seq_len, head_dim]
    seq_len: usize,
    head_dim: usize,
) -> Vec<f32> {
    if seq_len == 0 {
        return vec![0.0; head_dim];
    }

    let scale = 1.0 / (head_dim as f32).sqrt();

    // Compute attention scores: Q · K^T (for each position)
    let mut scores = vec![0.0f32; seq_len];
    for pos in 0..seq_len {
        let k_row = &k_cache[pos * head_dim..(pos + 1) * head_dim];
        let mut dot = 0.0f32;
        for d in 0..head_dim {
            dot += q[d] * k_row[d];
        }
        scores[pos] = dot * scale;
    }

    // Softmax
    let max_score = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut exp_scores: Vec<f32> = scores.iter().map(|s| (s - max_score).exp()).collect();
    let sum: f32 = exp_scores.iter().sum();
    if sum > 0.0 {
        for s in &mut exp_scores {
            *s /= sum;
        }
    }

    // Weighted sum of V
    let mut output = vec![0.0f32; head_dim];
    for pos in 0..seq_len {
        let v_row = &v_cache[pos * head_dim..(pos + 1) * head_dim];
        let w = exp_scores[pos];
        for d in 0..head_dim {
            output[d] += w * v_row[d];
        }
    }

    output
}

/// Multi-head attention with Grouped Query Attention (GQA).
///
/// Q: [num_heads, head_dim]
/// K/V: [num_kv_heads, head_dim] (shared across Q head groups)
/// K/V caches: [num_kv_heads, seq_len, head_dim]
///
/// Returns: [hidden_dim] = concat of all head outputs
pub fn multi_head_attention(
    q_heads: &[Vec<f32>],  // [num_heads][head_dim]
    k_caches: &[Vec<f32>], // [num_kv_heads][seq_len * head_dim]
    v_caches: &[Vec<f32>], // [num_kv_heads][seq_len * head_dim]
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    seq_len: usize,
) -> Vec<f32> {
    let heads_per_kv = num_heads / num_kv_heads.max(1);
    let mut output = Vec::with_capacity(num_heads * head_dim);

    for (h, q_head) in q_heads.iter().enumerate().take(num_heads) {
        let kv_idx = h / heads_per_kv.max(1);
        let kv_idx = kv_idx.min(num_kv_heads.saturating_sub(1));

        let head_out = attention_head(
            q_head,
            &k_caches[kv_idx],
            &v_caches[kv_idx],
            seq_len,
            head_dim,
        );
        output.extend_from_slice(&head_out);
    }

    output
}

/// Sparse attention over a subset of positions (retrieval-based attention).
///
/// Instead of computing attention over all seq_len positions, this computes
/// attention only over a selected subset of positions. The positions are
/// provided as separate K and V vectors (already gathered from the tiered
/// KV cache by the retrieval step).
///
/// This is the "attention-as-query-plan" computation:
/// 1. KV index searches for top-k positions (already done by caller)
/// 2. K/V vectors for those positions are gathered (already done by caller)
/// 3. This function computes Q·K^T softmax → weighted V sum over the subset
///
/// Accuracy vs exhaustive attention depends on the ANN index recall.
/// With HNSW at >95% recall, the output is nearly identical.
pub fn sparse_attention_head(
    q: &[f32],              // [head_dim]
    k_vectors: &[Vec<f32>], // [num_positions][head_dim] — gathered K vectors
    v_vectors: &[Vec<f32>], // [num_positions][head_dim] — gathered V vectors
    head_dim: usize,
) -> Vec<f32> {
    let num_positions = k_vectors.len();
    if num_positions == 0 {
        return vec![0.0; head_dim];
    }

    let scale = 1.0 / (head_dim as f32).sqrt();

    // Compute attention scores: Q · K^T for each retrieved position
    let mut scores = Vec::with_capacity(num_positions);
    for k_vec in k_vectors {
        let mut dot = 0.0f32;
        for d in 0..head_dim.min(k_vec.len()) {
            dot += q[d] * k_vec[d];
        }
        scores.push(dot * scale);
    }

    // Softmax over retrieved positions
    let max_score = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut exp_scores: Vec<f32> = scores.iter().map(|s| (s - max_score).exp()).collect();
    let sum: f32 = exp_scores.iter().sum();
    if sum > 0.0 {
        for s in &mut exp_scores {
            *s /= sum;
        }
    }

    // Weighted sum of V vectors
    let mut output = vec![0.0f32; head_dim];
    for (pos_idx, v_vec) in v_vectors.iter().enumerate() {
        let w = exp_scores[pos_idx];
        for d in 0..head_dim.min(v_vec.len()) {
            output[d] += w * v_vec[d];
        }
    }

    output
}

/// Multi-head sparse attention with GQA over tiered KV cache.
///
/// This is the tiered equivalent of `multi_head_attention`. Instead of
/// operating on contiguous K/V cache arrays, it takes per-head gathered
/// K/V vectors (already retrieved from the tiered KV cache).
///
/// `k_per_head`: [num_kv_heads][num_positions][head_dim]
/// `v_per_head`: [num_kv_heads][num_positions][head_dim]
pub fn multi_head_sparse_attention(
    q_heads: &[Vec<f32>],         // [num_heads][head_dim]
    k_per_head: &[Vec<Vec<f32>>], // [num_kv_heads][num_positions][head_dim]
    v_per_head: &[Vec<Vec<f32>>], // [num_kv_heads][num_positions][head_dim]
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
) -> Vec<f32> {
    let heads_per_kv = num_heads / num_kv_heads.max(1);
    let mut output = Vec::with_capacity(num_heads * head_dim);

    for (h, q_head) in q_heads.iter().enumerate().take(num_heads) {
        let kv_idx = (h / heads_per_kv.max(1)).min(num_kv_heads.saturating_sub(1));

        let head_out = if kv_idx < k_per_head.len() && !k_per_head[kv_idx].is_empty() {
            sparse_attention_head(q_head, &k_per_head[kv_idx], &v_per_head[kv_idx], head_dim)
        } else {
            vec![0.0; head_dim]
        };

        output.extend_from_slice(&head_out);
    }

    output
}

// ─── Fused Residual Add ──────────────────────────────────────────────────

/// Fused residual add: output[d] = a[d] + b[d] for FP16 vectors.
///
/// Used for residual connections: hidden_state = pre_norm + transform(post_norm).
pub fn fused_residual_add(
    output: *mut u8, // [dim] FP16 on device
    a: *const u8,    // [dim] FP16 on device
    b: *const u8,    // [dim] FP16 on device
    dim: usize,
    stream: &CudaStream,
) -> Result<()> {
    #[cfg(feature = "cuda")]
    {
        if stream.is_real() {
            let err = unsafe {
                cuda_ffi::vib3_launch_fused_residual_add(output, a, b, dim as i32, stream.raw_ptr())
            };
            if err != 0 {
                return Err(Error::Cuda(format!(
                    "fused_residual_add launch failed (err={})",
                    err
                )));
            }
            return Ok(());
        }
    }

    // CPU fallback
    let out = unsafe { std::slice::from_raw_parts_mut(output as *mut f16, dim) };
    let va = unsafe { std::slice::from_raw_parts(a as *const f16, dim) };
    let vb = unsafe { std::slice::from_raw_parts(b as *const f16, dim) };
    for d in 0..dim {
        out[d] = f16::from_f32(va[d].to_f32() + vb[d].to_f32());
    }
    Ok(())
}

// ─── Embedding / lm_head ─────────────────────────────────────────────────

/// Embedding lookup: token_id → hidden_state vector.
///
/// Reads one row from the embedding table.
pub fn embedding_lookup(
    embedding_table: *const u8, // [vocab_size, hidden_dim] FP16
    token_id: u32,
    hidden_dim: usize,
    output: *mut u8, // [hidden_dim] FP16
) {
    // GPU dispatch: embedding table and output are both in VRAM
    #[cfg(feature = "cuda")]
    {
        if cuda_ffi::is_cuda_available() {
            let err = unsafe {
                cuda_ffi::vib3_launch_embedding_lookup(
                    embedding_table,
                    output,
                    token_id as i32,
                    hidden_dim as i32,
                    std::ptr::null_mut(), // default stream
                )
            };
            if err == 0 {
                // Sync to ensure completion before returning
                let _ = cuda_ffi::device_synchronize();
                return;
            }
            // Fall through to CPU on error
        }
    }

    let table = unsafe {
        std::slice::from_raw_parts(
            embedding_table as *const f16,
            (token_id as usize + 1) * hidden_dim,
        )
    };
    let out = unsafe { std::slice::from_raw_parts_mut(output as *mut f16, hidden_dim) };
    let row_start = token_id as usize * hidden_dim;
    out.copy_from_slice(&table[row_start..row_start + hidden_dim]);
}

/// INT8 quantized embedding lookup with dequantization.
///
/// Storage format: `[vocab_size * hidden_dim bytes of INT8]` followed by
/// `[vocab_size * 2 bytes of FP16 per-row scales]`.
///
/// Dequantization: `output[d] = (int8_val as f32) * scale`, producing FP16.
///
/// This reduces embedding table size by 50% vs FP16 with negligible quality
/// loss — embeddings are over-parameterized and tolerate aggressive quantization.
/// Memory savings: for a 256K vocab × 7168 dim model, INT8 saves ~875 MB.
pub fn embedding_lookup_int8(
    embedding_table: *const u8, // [vocab_size * hidden_dim] INT8 + [vocab_size * 2] FP16 scales
    token_id: u32,
    vocab_size: usize,
    hidden_dim: usize,
    output: *mut u8, // [hidden_dim] FP16
) {
    let int8_data_bytes = vocab_size * hidden_dim;
    let table =
        unsafe { std::slice::from_raw_parts(embedding_table as *const i8, int8_data_bytes) };
    let scales_ptr = unsafe { embedding_table.add(int8_data_bytes) };
    let scales = unsafe { std::slice::from_raw_parts(scales_ptr as *const f16, vocab_size) };
    let out = unsafe { std::slice::from_raw_parts_mut(output as *mut f16, hidden_dim) };

    let row_start = token_id as usize * hidden_dim;
    let scale = scales[token_id as usize].to_f32();

    for (out_val, &tbl_val) in out.iter_mut().zip(table[row_start..].iter()) {
        let int_val = tbl_val as f32;
        *out_val = f16::from_f32(int_val * scale);
    }
}

/// Quantize an FP16 embedding table to INT8 + per-row FP16 scales.
///
/// For each row (token), finds max absolute value, computes `scale = max_abs / 127`,
/// and quantizes each element as `round(val / scale)` clamped to [-128, 127].
///
/// Returns: packed buffer `[vocab_size * hidden_dim INT8 bytes] [vocab_size * 2 FP16 scale bytes]`
pub fn quantize_embeddings_to_int8(
    fp16_table: &[f16],
    vocab_size: usize,
    hidden_dim: usize,
) -> Vec<u8> {
    let int8_bytes = vocab_size * hidden_dim;
    let scales_bytes = vocab_size * 2;
    let mut result = vec![0u8; int8_bytes + scales_bytes];

    let (int8_region, scales_region) = result.split_at_mut(int8_bytes);

    for v in 0..vocab_size {
        let row_start = v * hidden_dim;
        let row = &fp16_table[row_start..row_start + hidden_dim];

        // Find max absolute value
        let max_abs = row.iter().map(|x| x.to_f32().abs()).fold(0.0f32, f32::max);

        let scale = if max_abs > 0.0 { max_abs / 127.0 } else { 1.0 };

        // Quantize row
        let int8_row_start = v * hidden_dim;
        for (d, &r) in row.iter().enumerate() {
            let val = r.to_f32() / scale;
            let quantized = val.round().clamp(-128.0, 127.0) as i8;
            int8_region[int8_row_start + d] = quantized as u8;
        }

        // Store scale as FP16
        let scale_f16 = f16::from_f32(scale);
        let scale_bytes = scale_f16.to_le_bytes();
        let scale_offset = v * 2;
        scales_region[scale_offset] = scale_bytes[0];
        scales_region[scale_offset + 1] = scale_bytes[1];
    }

    result
}

// ─── Transport Quantization ──────────────────────────────────────────────

/// Dequantize a page from INT8 transport format to FP16 compute format.
///
/// **Transport quantization** stores weight pages at INT8 on NVMe (half the
/// bytes of FP16), then dequantizes to FP16 during the T2→T1 DMA transfer.
/// Since NVMe bandwidth is the bottleneck, halving the page size on disk
/// effectively doubles read throughput. The CPU dequant runs at ~10 GB/s
/// and overlaps with GPU compute on the current layer.
///
/// Layout of `src`: `[num_elements INT8 bytes] [num_rows * 2 bytes FP16 per-row scales]`
/// where `num_rows = num_elements / cols_per_row` (one scale per matrix row).
///
/// Writes FP16 values to `dst`.
pub fn dequant_page_int8_to_fp16(
    src: &[u8],
    dst: &mut [u8],
    num_elements: usize,
    cols_per_row: usize,
) {
    let num_rows = if cols_per_row > 0 {
        num_elements.div_ceil(cols_per_row)
    } else {
        1
    };
    let int8_bytes = num_elements;
    let scales_start = int8_bytes;

    let out = unsafe { std::slice::from_raw_parts_mut(dst.as_mut_ptr() as *mut f16, num_elements) };

    for row in 0..num_rows {
        // Read per-row scale
        let scale_offset = scales_start + row * 2;
        let scale = if scale_offset + 1 < src.len() {
            let s = f16::from_le_bytes([src[scale_offset], src[scale_offset + 1]]);
            let sv = s.to_f32();
            if sv.is_finite() && sv > 0.0 {
                sv
            } else {
                1.0 / 127.0
            }
        } else {
            1.0 / 127.0
        };

        let row_start = row * cols_per_row;
        let row_end = (row_start + cols_per_row).min(num_elements);
        for i in row_start..row_end {
            if i < int8_bytes {
                let int_val = src[i] as i8 as f32;
                out[i] = f16::from_f32(int_val * scale);
            }
        }
    }
}

/// Quantize FP16 page data to INT8 transport format.
///
/// Input: `[num_elements] FP16` values.
/// Output: `[num_elements INT8 bytes] [num_rows * 2 bytes FP16 per-row scales]`
///
/// Used by the converter when `--transport-quant` is enabled.
pub fn quant_page_fp16_to_int8(
    fp16_data: &[u8],
    num_elements: usize,
    cols_per_row: usize,
) -> Vec<u8> {
    let num_rows = if cols_per_row > 0 {
        num_elements.div_ceil(cols_per_row)
    } else {
        1
    };
    let int8_bytes = num_elements;
    let scales_bytes = num_rows * 2;
    let mut result = vec![0u8; int8_bytes + scales_bytes];

    let src = unsafe { std::slice::from_raw_parts(fp16_data.as_ptr() as *const f16, num_elements) };
    let (int8_region, scales_region) = result.split_at_mut(int8_bytes);

    for row in 0..num_rows {
        let row_start = row * cols_per_row;
        let row_end = (row_start + cols_per_row).min(num_elements);
        let row_slice = &src[row_start..row_end];

        // Find max absolute value in row
        let max_abs = row_slice
            .iter()
            .map(|x| x.to_f32().abs())
            .fold(0.0f32, f32::max);
        let scale = if max_abs > 0.0 { max_abs / 127.0 } else { 1.0 };

        // Quantize
        for (col_idx, &val) in row_slice.iter().enumerate() {
            let q = (val.to_f32() / scale).round().clamp(-128.0, 127.0) as i8;
            int8_region[row_start + col_idx] = q as u8;
        }

        // Store scale
        let scale_f16 = f16::from_f32(scale);
        let sb = scale_f16.to_le_bytes();
        scales_region[row * 2] = sb[0];
        scales_region[row * 2 + 1] = sb[1];
    }

    result
}

/// Linear projection: output = input × weight^T (GEMV).
///
/// Used for lm_head, Q/K/V projections, O projection.
/// input: [1, in_dim] FP16
/// weight: [out_dim, in_dim] FP16
/// output: [1, out_dim] FP16
pub fn linear_projection(
    input: *const u8,
    weight: *const u8,
    output: *mut u8,
    in_dim: usize,
    out_dim: usize,
    stream: &CudaStream,
) -> Result<()> {
    // GPU dispatch: uses the FP16 matmul kernel
    #[cfg(feature = "cuda")]
    {
        if stream.is_real() {
            let err = unsafe {
                cuda_ffi::vib3_launch_partial_matmul_fp16(
                    input,
                    weight,
                    output,
                    in_dim as i32,
                    out_dim as i32,
                    stream.raw_ptr(),
                )
            };
            if err != 0 {
                return Err(Error::Cuda(format!(
                    "linear_projection launch failed (err={})",
                    err
                )));
            }
            return Ok(());
        }
    }

    cpu_matmul_fp16(input, weight, output, in_dim, out_dim);
    Ok(())
}

/// Linear projection with F32 input and F32 output: output = input × weight^T (GEMV).
///
/// Avoids FP16 truncation on both input and output sides.
/// Used for the MLA O-projection where V_out is F32 and we want to keep
/// the result in F32 before adding to the FP32 residual stream.
/// input: [1, in_dim] FP32
/// weight: [out_dim, in_dim] FP16
/// output: [1, out_dim] FP32
pub fn linear_projection_f32_to_f32(
    input: *const u8,
    weight: *const u8,
    output: *mut u8,
    in_dim: usize,
    out_dim: usize,
    stream: &CudaStream,
) -> Result<()> {
    #[cfg(feature = "cuda")]
    {
        if stream.is_real() {
            let err = unsafe {
                cuda_ffi::vib3_launch_partial_matmul_fp16_f32in_f32out(
                    input,
                    weight,
                    output,
                    in_dim as i32,
                    out_dim as i32,
                    stream.raw_ptr(),
                )
            };
            if err != 0 {
                return Err(Error::Cuda(format!(
                    "linear_projection_f32_to_f32 launch failed (err={})",
                    err
                )));
            }
            return Ok(());
        }
    }

    // CPU fallback: F32 input × FP16 weight → F32 output
    let inp = unsafe { std::slice::from_raw_parts(input as *const f32, in_dim) };
    let w = unsafe { std::slice::from_raw_parts(weight as *const f16, out_dim * in_dim) };
    let out = unsafe { std::slice::from_raw_parts_mut(output as *mut f32, out_dim) };
    for row in 0..out_dim {
        let mut acc = 0.0f32;
        let row_offset = row * in_dim;
        for d in 0..in_dim {
            acc += inp[d] * w[row_offset + d].to_f32();
        }
        out[row] = acc;
    }
    Ok(())
}

/// Compute logits from hidden state using lm_head weights (parallelized).
///
/// logits[v] = hidden_state · lm_head_weights[v] for v in 0..vocab_size
/// Returns f32 logits for sampling.
///
/// This is a significant compute target: 32K rows for Mixtral, 164K for Kimi K2.5.
/// Parallelized across vocab rows with Rayon — each row is an independent dot product.
pub fn compute_logits(
    hidden_state: *const u8, // [hidden_dim] FP16
    lm_head: *const u8,      // [vocab_size, hidden_dim] FP16
    vocab_size: usize,
    hidden_dim: usize,
) -> Vec<f32> {
    // GPU path: compute logits on GPU, copy to host for sampling
    #[cfg(feature = "cuda")]
    {
        if cuda_ffi::is_cuda_available() {
            if let Ok(logits) = gpu_compute_logits(hidden_state, lm_head, vocab_size, hidden_dim) {
                return logits;
            }
            // Fall through to CPU on error
        }
    }

    let state = unsafe { std::slice::from_raw_parts(hidden_state as *const f16, hidden_dim) };
    let weights =
        unsafe { std::slice::from_raw_parts(lm_head as *const f16, vocab_size * hidden_dim) };

    let state_f32: Vec<f32> = state.iter().map(|v| v.to_f32()).collect();
    let mut logits = vec![0.0f32; vocab_size];

    logits.par_iter_mut().enumerate().for_each(|(v, logit)| {
        let row_offset = v * hidden_dim;
        let mut acc = 0.0f32;
        for d in 0..hidden_dim {
            acc += state_f32[d] * weights[row_offset + d].to_f32();
        }
        *logit = acc;
    });

    logits
}

/// GPU logits: uses router_gemv kernel (same GEMV operation).
#[cfg(feature = "cuda")]
fn gpu_compute_logits(
    hidden_state: *const u8,
    lm_head: *const u8,
    vocab_size: usize,
    hidden_dim: usize,
) -> Result<Vec<f32>> {
    // Allocate device buffer for logits
    let logits_device = cuda_ffi::device_alloc(vocab_size * 4)?;

    // Launch GEMV (same kernel as router, just larger)
    let err = unsafe {
        cuda_ffi::vib3_launch_router_gemv(
            hidden_state,
            lm_head,
            logits_device as *mut f32,
            hidden_dim as i32,
            vocab_size as i32,
            std::ptr::null_mut(), // default stream
        )
    };
    if err != 0 {
        cuda_ffi::device_free(logits_device, vocab_size * 4);
        return Err(Error::Cuda(format!(
            "compute_logits launch failed (err={})",
            err
        )));
    }

    // Sync and copy to host
    cuda_ffi::device_synchronize()?;
    let mut logits = vec![0.0f32; vocab_size];
    cuda_ffi::memcpy_d2h(
        logits.as_mut_ptr() as *mut u8,
        logits_device as *const u8,
        vocab_size * 4,
    )?;
    cuda_ffi::device_free(logits_device, vocab_size * 4);

    Ok(logits)
}

// ─── Weight Quantization (FP16/BF16/FP32 → INT4) ────────────────────────

/// Default group size for INT4 quantization.
///
/// Kimi K2.5 uses group_size=32 (compressed-tensors format).
/// Many other models (GPTQ, AWQ) use group_size=128.
/// The converter and dequant kernels should use the model's native group size.
pub const INT4_GROUP_SIZE: usize = 32;

/// Legacy group size for models using GPTQ/AWQ with group_size=128.
pub const INT4_GROUP_SIZE_128: usize = 128;

/// Block size for NVFP4 (MXFP4 E2M1) quantization.
/// Each block of 32 elements shares a single FP16 scale factor.
pub const NVFP4_BLOCK_SIZE: usize = 32;

/// Quantize a weight matrix from f32 values to packed INT4 + per-group FP16 scales.
///
/// This is the **forward quantization** function — the inverse of the LUT-based
/// dequantization in `cpu_matmul_int4`. It converts a dense weight matrix into
/// the compact format used on disk and by the INT4 compute kernels.
///
/// ## Layout
///
/// For a weight matrix of shape `[rows, cols]`:
/// - `packed_k = cols.div_ceil(2)` bytes per row (2 nibbles per byte)
/// - `num_groups = (cols + GROUP_SIZE - 1) / GROUP_SIZE` groups per row
/// - Output: `[rows * packed_k bytes of INT4 data] [rows * num_groups * 2 bytes of FP16 scales]`
///
/// ## Quantization scheme
///
/// Per group of `GROUP_SIZE` columns within each row:
/// 1. Find `max_abs = max(|w_i|)` over the group
/// 2. Compute `scale = max_abs / 7.0` (INT4 signed range is [-8, +7] with offset)
/// 3. Quantize: `q = clamp(round(w / scale) + 8, 0, 15)` → nibble value
/// 4. Pack two nibbles per byte: `byte = lo_nibble | (hi_nibble << 4)`
///
/// The dequantization formula is: `(nibble - 8) * scale`, matching `cpu_matmul_int4`.
///
/// ## Arguments
///
/// - `weights_f32`: flattened `[rows, cols]` weight matrix in f32
/// - `rows`: number of rows (output dimension)
/// - `cols`: number of columns (input dimension)
///
/// ## Returns
///
/// Packed buffer: `[rows * packed_k INT4 bytes] [rows * num_groups * 2 FP16 scale bytes]`
pub fn quantize_weights_to_int4(weights_f32: &[f32], rows: usize, cols: usize) -> Vec<u8> {
    assert_eq!(weights_f32.len(), rows * cols);

    let packed_k = cols.div_ceil(2);
    let num_groups = cols.div_ceil(INT4_GROUP_SIZE);
    let int4_bytes = rows * packed_k;
    let scales_bytes = rows * num_groups * 2; // FP16 per group
    let mut result = vec![0u8; int4_bytes + scales_bytes];

    let (int4_region, scales_region) = result.split_at_mut(int4_bytes);

    for row in 0..rows {
        let row_data = &weights_f32[row * cols..(row + 1) * cols];

        // Compute per-group scales
        for group in 0..num_groups {
            let group_start = group * INT4_GROUP_SIZE;
            let group_end = (group_start + INT4_GROUP_SIZE).min(cols);
            let group_slice = &row_data[group_start..group_end];

            let max_abs = group_slice.iter().map(|x| x.abs()).fold(0.0f32, f32::max);

            // scale such that max_abs maps to 7 (the positive range of our offset scheme)
            let scale = if max_abs > 0.0 { max_abs / 7.0 } else { 1.0 };

            // Store scale as BF16
            let scale_bf16 = f32_to_bf16(scale);
            let sb = scale_bf16.to_le_bytes();
            let scale_offset = (row * num_groups + group) * 2;
            scales_region[scale_offset] = sb[0];
            scales_region[scale_offset + 1] = sb[1];

            // Quantize and pack nibbles
            #[allow(clippy::needless_range_loop)]
            for col in group_start..group_end {
                let val = row_data[col];
                // q = round(val / scale) + 8, clamped to [0, 15]
                let q = ((val / scale).round() as i32 + 8).clamp(0, 15) as u8;

                let byte_idx = col / 2;
                let row_byte_offset = row * packed_k + byte_idx;
                if col % 2 == 0 {
                    // Low nibble
                    int4_region[row_byte_offset] |= q & 0x0F;
                } else {
                    // High nibble
                    int4_region[row_byte_offset] |= (q & 0x0F) << 4;
                }
            }
        }
    }

    result
}

/// Quantize FP16 weight data to INT4 format.
///
/// Convenience wrapper that converts FP16 → f32 → INT4.
pub fn quantize_fp16_to_int4(fp16_data: &[u8], rows: usize, cols: usize) -> Vec<u8> {
    let num_elements = rows * cols;
    let fp16_slice =
        unsafe { std::slice::from_raw_parts(fp16_data.as_ptr() as *const f16, num_elements) };
    let f32_data: Vec<f32> = fp16_slice.iter().map(|v| v.to_f32()).collect();
    quantize_weights_to_int4(&f32_data, rows, cols)
}

/// Quantize BF16 weight data to INT4 format.
///
/// Convenience wrapper that converts BF16 → f32 → INT4.
pub fn quantize_bf16_to_int4(bf16_data: &[u8], rows: usize, cols: usize) -> Vec<u8> {
    let num_elements = rows * cols;
    let bf16_slice =
        unsafe { std::slice::from_raw_parts(bf16_data.as_ptr() as *const u16, num_elements) };
    let f32_data: Vec<f32> = bf16_slice.iter().map(|v| bf16_to_f32(*v)).collect();
    quantize_weights_to_int4(&f32_data, rows, cols)
}

/// Dequantize INT4 packed data (with per-group scales) back to f32.
///
/// This is the inverse of `quantize_weights_to_int4`.
/// The input layout is: `[rows * packed_k INT4 bytes] [rows * num_groups * scale_bytes_each]`
///
/// The scale format may be BF16 or FP16 (2 bytes each). The `scale_format` argument
/// selects the interpretation:
/// - `DType::BF16` → scales are BF16 (compressed-tensors format from HuggingFace)
/// - `DType::FP16` → scales are FP16
///
/// Dequant formula: `(nibble - 8) * scale`
pub fn dequantize_int4_to_f32(
    int4_data: &[u8],
    rows: usize,
    cols: usize,
    group_size: usize,
    scale_format: DType,
) -> Vec<f32> {
    use rayon::prelude::*;

    let packed_k = cols.div_ceil(2);
    let num_groups = cols.div_ceil(group_size);
    let total_int4_bytes = rows * packed_k;

    let mut result = vec![0.0f32; rows * cols];

    // Pre-decode all scales to f32 to avoid per-element branching
    let total_scales = rows * num_groups;
    let scales_f32: Vec<f32> = (0..total_scales)
        .map(|i| {
            let offset = total_int4_bytes + i * 2;
            if offset + 1 < int4_data.len() {
                let sb = [int4_data[offset], int4_data[offset + 1]];
                match scale_format {
                    DType::BF16 => bf16_to_f32(u16::from_le_bytes(sb)),
                    _ => f16::from_le_bytes(sb).to_f32(),
                }
            } else {
                1.0
            }
        })
        .collect();

    // Parallel dequant per row
    result
        .par_chunks_mut(cols)
        .enumerate()
        .for_each(|(row, row_out)| {
            let row_packed = &int4_data
                [row * packed_k..row * packed_k + packed_k.min(total_int4_bytes - row * packed_k)];
            let row_scales = &scales_f32[row * num_groups..(row + 1) * num_groups];

            // Process pairs of columns (one byte = two nibbles)
            for byte_idx in 0..packed_k {
                let packed_byte = row_packed[byte_idx];
                let lo = (packed_byte & 0x0F) as f32 - 8.0;
                let hi = ((packed_byte >> 4) & 0x0F) as f32 - 8.0;

                let col0 = byte_idx * 2;
                let col1 = col0 + 1;

                let scale0 = row_scales[col0 / group_size];
                row_out[col0] = lo * scale0;

                if col1 < cols {
                    let scale1 = row_scales[col1 / group_size];
                    row_out[col1] = hi * scale1;
                }
            }
        });

    result
}

/// Direct INT4→NVFP4 conversion without f32 intermediate.
///
/// Takes INT4 packed data (with BF16 scales) from compressed-tensors format and
/// converts to NVFP4 (E2M1 with FP16 scales) in a single pass. This is much faster
/// than dequant→requant for model conversion.
///
/// Input layout: `[rows * packed_k INT4 bytes] [rows * num_groups * 2 BF16 scale bytes]`
/// Output layout: `[rows * packed_k E2M1 bytes] [rows * num_blocks * 2 FP16 scale bytes]`
pub fn convert_int4_to_nvfp4(
    int4_data: &[u8],
    rows: usize,
    cols: usize,
    group_size: usize,
) -> Vec<u8> {
    use rayon::prelude::*;

    // Map a non-negative value (already divided by scale) to the nearest E2M1 nibble.
    #[inline]
    fn nearest_e2m1(val: f32) -> u8 {
        if val < 0.25 {
            0
        } else if val < 0.75 {
            1
        } else if val < 1.25 {
            2
        } else if val < 1.75 {
            3
        } else if val < 2.5 {
            4
        } else if val < 3.5 {
            5
        } else if val < 5.0 {
            6
        } else {
            7
        }
    }

    let packed_k = cols.div_ceil(2);
    let num_groups_in = cols.div_ceil(group_size);
    let num_blocks_out = cols.div_ceil(NVFP4_BLOCK_SIZE);
    let total_int4_bytes = rows * packed_k;

    let data_bytes = rows * packed_k; // Same packed size for E2M1
    let scales_bytes = rows * num_blocks_out * 2;

    // Process each row independently in parallel, collecting per-row results
    let row_results: Vec<(Vec<u8>, Vec<u8>)> = (0..rows)
        .into_par_iter()
        .map(|row| {
            // Dequantize this row to f32
            let row_packed = &int4_data[row * packed_k..(row + 1) * packed_k];
            let mut row_f32 = vec![0.0f32; cols];

            for byte_idx in 0..packed_k {
                let packed_byte = row_packed[byte_idx];
                let lo = (packed_byte & 0x0F) as f32 - 8.0;
                let hi = ((packed_byte >> 4) & 0x0F) as f32 - 8.0;

                let col0 = byte_idx * 2;
                let col1 = col0 + 1;

                let scale_offset = total_int4_bytes + (row * num_groups_in + col0 / group_size) * 2;
                let scale0 = if scale_offset + 1 < int4_data.len() {
                    let sb = [int4_data[scale_offset], int4_data[scale_offset + 1]];
                    bf16_to_f32(u16::from_le_bytes(sb))
                } else {
                    1.0
                };
                row_f32[col0] = lo * scale0;

                if col1 < cols {
                    let scale_offset1 =
                        total_int4_bytes + (row * num_groups_in + col1 / group_size) * 2;
                    let scale1 = if scale_offset1 + 1 < int4_data.len() {
                        let sb = [int4_data[scale_offset1], int4_data[scale_offset1 + 1]];
                        bf16_to_f32(u16::from_le_bytes(sb))
                    } else {
                        1.0
                    };
                    row_f32[col1] = hi * scale1;
                }
            }

            // Quantize this row to NVFP4
            let mut row_packed_out = vec![0u8; packed_k];
            let mut row_scales = vec![0u8; num_blocks_out * 2];

            for block in 0..num_blocks_out {
                let block_start = block * NVFP4_BLOCK_SIZE;
                let block_end = (block_start + NVFP4_BLOCK_SIZE).min(cols);

                let max_abs = row_f32[block_start..block_end]
                    .iter()
                    .map(|x| x.abs())
                    .fold(0.0f32, f32::max);

                let scale = if max_abs > 0.0 { max_abs / 6.0 } else { 1.0 };
                let scale_f16 = f16::from_f32(scale);
                let sb = scale_f16.to_le_bytes();
                row_scales[block * 2] = sb[0];
                row_scales[block * 2 + 1] = sb[1];

                for col in block_start..block_end {
                    let val = row_f32[col];
                    let sign = if val < 0.0 { 1u8 } else { 0u8 };
                    let mag_idx = nearest_e2m1((val.abs() / scale).min(6.0));
                    let nibble = mag_idx | (sign << 3);

                    let byte_idx = col / 2;
                    if col % 2 == 0 {
                        row_packed_out[byte_idx] |= nibble & 0x0F;
                    } else {
                        row_packed_out[byte_idx] |= (nibble & 0x0F) << 4;
                    }
                }
            }

            (row_packed_out, row_scales)
        })
        .collect();

    // Assemble output: [all packed data] [all scales]
    let mut result = vec![0u8; data_bytes + scales_bytes];
    for (row, (packed, scales)) in row_results.iter().enumerate() {
        result[row * packed_k..(row + 1) * packed_k].copy_from_slice(packed);
        let scale_offset = data_bytes + row * num_blocks_out * 2;
        result[scale_offset..scale_offset + num_blocks_out * 2].copy_from_slice(scales);
    }

    result
}

/// Quantize a weight matrix from f32 values to packed NVFP4 (MXFP4 E2M1) + per-block FP16 scales.
///
/// ## MXFP4 E2M1 Format
///
/// Each 4-bit value encodes: 1 sign bit + 2 exponent bits + 1 mantissa bit.
/// Representable magnitudes: {0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0}
///
/// ## Layout
///
/// For a weight matrix of shape `[rows, cols]`:
/// - `packed_k = cols.div_ceil(2)` bytes per row (2 nibbles per byte)
/// - `num_blocks = (cols + NVFP4_BLOCK_SIZE - 1) / NVFP4_BLOCK_SIZE` blocks per row
/// - Output: `[rows * packed_k bytes of E2M1 data] [rows * num_blocks * 2 bytes of FP16 scales]`
///
/// ## Quantization scheme
///
/// Per block of `NVFP4_BLOCK_SIZE` columns within each row:
/// 1. Find `max_abs = max(|w_i|)` over the block
/// 2. Compute `scale = max_abs / 6.0` (max E2M1 magnitude is 6.0)
/// 3. Quantize: find nearest E2M1 value to `|w / scale|`, apply sign → nibble
/// 4. Pack two nibbles per byte: `byte = lo_nibble | (hi_nibble << 4)`
///
/// The dequantization formula is: `nvfp4_lut[nibble] * scale`, matching the CUDA kernel.
pub fn quantize_weights_to_nvfp4(weights_f32: &[f32], rows: usize, cols: usize) -> Vec<u8> {
    assert_eq!(weights_f32.len(), rows * cols);

    // E2M1 magnitude values (positive side of the LUT, indices 0-7)
    const E2M1_VALUES: [f32; 8] = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0];

    // Map a non-negative value (already divided by scale) to the nearest E2M1 nibble.
    // Returns unsigned index 0-7 into E2M1_VALUES.
    #[inline]
    fn nearest_e2m1(val: f32) -> u8 {
        // Binary-search style: compare against midpoints between adjacent E2M1 values
        // Midpoints: 0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0
        if val < 0.25 {
            0 // 0.0
        } else if val < 0.75 {
            1 // 0.5
        } else if val < 1.25 {
            2 // 1.0
        } else if val < 1.75 {
            3 // 1.5
        } else if val < 2.5 {
            4 // 2.0
        } else if val < 3.5 {
            5 // 3.0
        } else if val < 5.0 {
            6 // 4.0
        } else {
            7 // 6.0
        }
    }
    // Silence unused warning for the constant array (used conceptually for documentation)
    let _ = E2M1_VALUES;

    let packed_k = cols.div_ceil(2);
    let num_blocks = cols.div_ceil(NVFP4_BLOCK_SIZE);
    let data_bytes = rows * packed_k;
    let scales_bytes = rows * num_blocks * 2; // FP16 per block
    let mut result = vec![0u8; data_bytes + scales_bytes];

    let (data_region, scales_region) = result.split_at_mut(data_bytes);

    for row in 0..rows {
        let row_data = &weights_f32[row * cols..(row + 1) * cols];

        for block in 0..num_blocks {
            let block_start = block * NVFP4_BLOCK_SIZE;
            let block_end = (block_start + NVFP4_BLOCK_SIZE).min(cols);
            let block_slice = &row_data[block_start..block_end];

            let max_abs = block_slice.iter().map(|x| x.abs()).fold(0.0f32, f32::max);

            // Scale so that max_abs maps to 6.0 (the max representable E2M1 magnitude)
            let scale = if max_abs > 0.0 { max_abs / 6.0 } else { 1.0 };

            // Store scale as FP16
            let scale_f16 = f16::from_f32(scale);
            let sb = scale_f16.to_le_bytes();
            let scale_offset = (row * num_blocks + block) * 2;
            scales_region[scale_offset] = sb[0];
            scales_region[scale_offset + 1] = sb[1];

            // Quantize and pack nibbles
            #[allow(clippy::needless_range_loop)]
            for col in block_start..block_end {
                let val = row_data[col];
                let sign = if val < 0.0 { 1u8 } else { 0u8 };
                let mag_idx = nearest_e2m1((val.abs() / scale).min(6.0));
                // E2M1 nibble: positive values at indices 0-7, negative at 8-15 (sign bit is MSB)
                let nibble = mag_idx | (sign << 3);

                let byte_idx = col / 2;
                let row_byte_offset = row * packed_k + byte_idx;
                if col % 2 == 0 {
                    data_region[row_byte_offset] |= nibble & 0x0F;
                } else {
                    data_region[row_byte_offset] |= (nibble & 0x0F) << 4;
                }
            }
        }
    }

    result
}

/// Quantize FP16 weight data to NVFP4 format.
///
/// Convenience wrapper that converts FP16 → f32 → NVFP4.
pub fn quantize_fp16_to_nvfp4(fp16_data: &[u8], rows: usize, cols: usize) -> Vec<u8> {
    let num_elements = rows * cols;
    let fp16_slice =
        unsafe { std::slice::from_raw_parts(fp16_data.as_ptr() as *const f16, num_elements) };
    let f32_data: Vec<f32> = fp16_slice.iter().map(|v| v.to_f32()).collect();
    quantize_weights_to_nvfp4(&f32_data, rows, cols)
}

/// Quantize BF16 weight data to NVFP4 format.
///
/// Convenience wrapper that converts BF16 → f32 → NVFP4.
pub fn quantize_bf16_to_nvfp4(bf16_data: &[u8], rows: usize, cols: usize) -> Vec<u8> {
    let num_elements = rows * cols;
    let bf16_slice =
        unsafe { std::slice::from_raw_parts(bf16_data.as_ptr() as *const u16, num_elements) };
    let f32_data: Vec<f32> = bf16_slice.iter().map(|v| bf16_to_f32(*v)).collect();
    quantize_weights_to_nvfp4(&f32_data, rows, cols)
}

/// Convert BF16 raw bytes to FP16 raw bytes (element-wise cast through f32).
///
/// Used when the source model has BF16 weights but we want to store or compute in FP16.
pub fn convert_bf16_to_fp16(bf16_data: &[u8], num_elements: usize) -> Vec<u8> {
    let bf16_slice =
        unsafe { std::slice::from_raw_parts(bf16_data.as_ptr() as *const u16, num_elements) };
    let mut result = vec![0u8; num_elements * 2];
    let fp16_slice =
        unsafe { std::slice::from_raw_parts_mut(result.as_mut_ptr() as *mut f16, num_elements) };
    for (fp16_val, &bf16_val) in fp16_slice.iter_mut().zip(bf16_slice.iter()) {
        *fp16_val = f16::from_f32(bf16_to_f32(bf16_val));
    }
    result
}

/// Convert FP32 raw bytes to FP16 raw bytes (element-wise cast).
pub fn convert_f32_to_fp16(f32_data: &[u8], num_elements: usize) -> Vec<u8> {
    let f32_slice =
        unsafe { std::slice::from_raw_parts(f32_data.as_ptr() as *const f32, num_elements) };
    let mut result = vec![0u8; num_elements * 2];
    let fp16_slice =
        unsafe { std::slice::from_raw_parts_mut(result.as_mut_ptr() as *mut f16, num_elements) };
    for (fp16_val, &f32_val) in fp16_slice.iter_mut().zip(f32_slice.iter()) {
        *fp16_val = f16::from_f32(f32_val);
    }
    result
}

// ─── BF16 helpers ────────────────────────────────────────────────────────

fn bf16_to_f32(bits: u16) -> f32 {
    half::bf16::from_bits(bits).to_f32()
}

fn f32_to_bf16(val: f32) -> u16 {
    half::bf16::from_f32(val).to_bits()
}

// ─── MLA Decode Attention GPU Kernels ────────────────────────────────────

/// GPU MLA decode attention: fused Q·K score + softmax + weighted V latent accumulation.
///
/// Operates on compressed MLA KV cache (latent + rope), avoiding kv_b_proj during attention.
/// All inputs/outputs are F32 device pointers.
///
/// q_absorbed: [num_heads * kv_lora_rank] F32 — Q_nope × kv_b_k^T (already absorbed)
/// q_rope:     [num_heads * qk_rope_dim] F32 — RoPE'd query rope component
/// kv_latent:  [seq_len * kv_lora_rank] F32 — compressed KV cache
/// k_rope:     [seq_len * qk_rope_dim] F32 — RoPE'd K rope cache
/// v_latent_out: [num_heads * kv_lora_rank] F32 — output weighted latent per head
pub fn mla_decode_attention(
    q_absorbed: *const u8,
    q_rope: *const u8,
    kv_latent: *const u8,
    k_rope_cache: *const u8,
    v_latent_out: *mut u8,
    kv_lora_rank: usize,
    qk_rope_dim: usize,
    num_heads: usize,
    seq_len: usize,
    scale: f32,
    stream: &CudaStream,
) -> Result<()> {
    #[cfg(feature = "cuda")]
    {
        if stream.is_real() {
            let err = unsafe {
                cuda_ffi::vib3_launch_mla_decode_attention(
                    q_absorbed,
                    q_rope,
                    kv_latent,
                    k_rope_cache,
                    v_latent_out,
                    kv_lora_rank as i32,
                    qk_rope_dim as i32,
                    num_heads as i32,
                    seq_len as i32,
                    scale,
                    stream.raw_ptr(),
                )
            };
            if err != 0 {
                return Err(Error::Cuda(format!(
                    "mla_decode_attention launch failed (err={})",
                    err
                )));
            }
            return Ok(());
        }
    }
    // CPU fallback — not implemented, GPU required for MLA decode attention
    Err(Error::Cuda("mla_decode_attention requires GPU".to_string()))
}

/// GPU MLA V reconstruction: compute v_out from weighted latent and kv_b_proj_v.
///
/// v_latent: [num_heads * kv_lora_rank] F32 — weighted latent accumulation per head
/// kv_b_proj: [(num_heads * (nope+v)), kv_lora_rank] F32 — full kv_b_proj weight
/// v_out: [num_heads * v_head_dim] F32 — reconstructed V output per head
pub fn mla_v_reconstruct(
    v_latent: *const u8,
    kv_b_proj: *const u8,
    v_out: *mut u8,
    kv_lora_rank: usize,
    qk_nope_dim: usize,
    v_head_dim: usize,
    num_heads: usize,
    stream: &CudaStream,
) -> Result<()> {
    #[cfg(feature = "cuda")]
    {
        if stream.is_real() {
            let err = unsafe {
                cuda_ffi::vib3_launch_mla_v_reconstruct(
                    v_latent,
                    kv_b_proj,
                    v_out,
                    kv_lora_rank as i32,
                    qk_nope_dim as i32,
                    v_head_dim as i32,
                    num_heads as i32,
                    stream.raw_ptr(),
                )
            };
            if err != 0 {
                return Err(Error::Cuda(format!(
                    "mla_v_reconstruct launch failed (err={})",
                    err
                )));
            }
            return Ok(());
        }
    }
    Err(Error::Cuda("mla_v_reconstruct requires GPU".to_string()))
}

/// GPU MLA Q absorption + RoPE: fused kernel that computes absorbed Q and applies RoPE.
///
/// q_full: [num_heads * q_head_dim] FP16 — q_b_proj output (on device)
/// kv_b_proj: [num_heads*(nope+v), kv_lora_rank] F32 — kv_b weight (on device)
/// rope_freqs: [qk_rope_dim/2] F32 — precomputed YaRN frequencies (on device)
/// q_absorbed_out: [num_heads * kv_lora_rank] F32
/// q_rope_out: [num_heads * qk_rope_dim] F32
pub fn mla_q_absorb_rope(
    q_full: *const u8,
    kv_b_proj: *const u8,
    rope_freqs: *const u8,
    q_absorbed_out: *mut u8,
    q_rope_out: *mut u8,
    q_head_dim: usize,
    qk_nope_dim: usize,
    qk_rope_dim: usize,
    v_head_dim: usize,
    kv_lora_rank: usize,
    num_heads: usize,
    position: usize,
    stream: &CudaStream,
) -> Result<()> {
    #[cfg(feature = "cuda")]
    {
        if stream.is_real() {
            let err = unsafe {
                cuda_ffi::vib3_launch_mla_q_absorb_rope(
                    q_full,
                    kv_b_proj,
                    rope_freqs,
                    q_absorbed_out,
                    q_rope_out,
                    q_head_dim as i32,
                    qk_nope_dim as i32,
                    qk_rope_dim as i32,
                    v_head_dim as i32,
                    kv_lora_rank as i32,
                    num_heads as i32,
                    position as i32,
                    stream.raw_ptr(),
                )
            };
            if err != 0 {
                return Err(Error::Cuda(format!(
                    "mla_q_absorb_rope launch failed (err={})",
                    err
                )));
            }
            return Ok(());
        }
    }
    Err(Error::Cuda("mla_q_absorb_rope requires GPU".to_string()))
}

/// GPU MLA KV cache append: converts kv_a output (FP16) to F32, applies RMSNorm to
/// latent portion, applies RoPE to rope portion, writes both to GPU KV cache.
pub fn mla_kv_cache_append(
    kv_a_out: *const u8,
    kv_norm_weight: *const u8, // can be null
    rope_freqs: *const u8,
    kv_latent_cache: *mut u8,
    k_rope_cache: *mut u8,
    kv_lora_rank: usize,
    qk_rope_dim: usize,
    position: usize,
    eps: f32,
    stream: &CudaStream,
) -> Result<()> {
    #[cfg(feature = "cuda")]
    {
        if stream.is_real() {
            let err = unsafe {
                cuda_ffi::vib3_launch_mla_kv_cache_append(
                    kv_a_out,
                    kv_norm_weight,
                    rope_freqs,
                    kv_latent_cache,
                    k_rope_cache,
                    kv_lora_rank as i32,
                    qk_rope_dim as i32,
                    position as i32,
                    eps,
                    stream.raw_ptr(),
                )
            };
            if err != 0 {
                return Err(Error::Cuda(format!(
                    "mla_kv_cache_append launch failed (err={})",
                    err
                )));
            }
            return Ok(());
        }
    }
    Err(Error::Cuda("mla_kv_cache_append requires GPU".to_string()))
}

/// GPU F32→FP16 conversion.
pub fn f32_to_f16(input: *const u8, output: *mut u8, n: usize, stream: &CudaStream) -> Result<()> {
    #[cfg(feature = "cuda")]
    {
        if stream.is_real() {
            let err = unsafe {
                cuda_ffi::vib3_launch_f32_to_f16(input, output, n as i32, stream.raw_ptr())
            };
            if err != 0 {
                return Err(Error::Cuda(format!(
                    "f32_to_f16 launch failed (err={})",
                    err
                )));
            }
            return Ok(());
        }
    }
    // CPU fallback
    // SAFETY: Caller guarantees `input` points to at least `n` f32 values and
    // `output` points to at least `n` f16 values. Buffers are non-overlapping.
    let input_f32 = unsafe { std::slice::from_raw_parts(input as *const f32, n) };
    // SAFETY: See comment above; output has capacity for `n` f16 elements.
    let output_f16 = unsafe { std::slice::from_raw_parts_mut(output as *mut f16, n) };
    for i in 0..n {
        output_f16[i] = f16::from_f32(input_f32[i]);
    }
    Ok(())
}

/// GPU FP16→F32 conversion.
/// Converts `n` half-precision elements to single-precision.
pub fn f16_to_f32(input: *const u8, output: *mut u8, n: usize, stream: &CudaStream) -> Result<()> {
    #[cfg(feature = "cuda")]
    {
        if stream.is_real() {
            let err = unsafe {
                cuda_ffi::vib3_launch_fp16_to_fp32(input, output, n as i32, stream.raw_ptr())
            };
            if err != 0 {
                return Err(Error::Cuda(format!(
                    "f16_to_f32 launch failed (err={})",
                    err
                )));
            }
            return Ok(());
        }
    }
    // CPU fallback
    // SAFETY: Caller guarantees `input` points to at least `n` f16 values and
    // `output` points to at least `n` f32 values. Buffers are non-overlapping.
    let input_f16 = unsafe { std::slice::from_raw_parts(input as *const f16, n) };
    // SAFETY: See comment above; output has capacity for `n` f32 elements.
    let output_f32 = unsafe { std::slice::from_raw_parts_mut(output as *mut f32, n) };
    for i in 0..n {
        output_f32[i] = input_f16[i].to_f32();
    }
    Ok(())
}

/// GPU FP32 residual accumulation: accumulator[i] += f32(layer_output_f16[i]).
/// In-place on the FP32 accumulator buffer. `layer_output` is FP16.
/// Used for FP32 hidden state accumulation across transformer layers.
pub fn residual_add_fp32(
    accumulator: *mut u8,
    layer_output: *const u8,
    dim: usize,
    stream: &CudaStream,
) -> Result<()> {
    #[cfg(feature = "cuda")]
    {
        if stream.is_real() {
            let err = unsafe {
                cuda_ffi::vib3_launch_residual_add_fp32(
                    accumulator,
                    layer_output,
                    dim as i32,
                    stream.raw_ptr(),
                )
            };
            if err != 0 {
                return Err(Error::Cuda(format!(
                    "residual_add_fp32 launch failed (err={})",
                    err
                )));
            }
            return Ok(());
        }
    }
    // CPU fallback
    // SAFETY: Caller provides `dim` FP32 accumulator elements and `dim` FP16
    // layer output elements in valid, non-overlapping host buffers.
    let accumulator_f32 = unsafe { std::slice::from_raw_parts_mut(accumulator as *mut f32, dim) };
    // SAFETY: See comment above; layer output has `dim` FP16 elements.
    let layer_output_f16 = unsafe { std::slice::from_raw_parts(layer_output as *const f16, dim) };
    for i in 0..dim {
        accumulator_f32[i] += layer_output_f16[i].to_f32();
    }
    Ok(())
}

/// Causal depthwise 1D convolution with SiLU activation.
/// Processes one new token, updates `conv_state` in-place.
/// conv_state: [num_channels, kernel_size-1], conv_weight: [num_channels, kernel_size].
pub fn causal_conv1d(
    conv_state: *mut u8,    // [num_channels, kernel_size-1] FP32, updated in-place
    new_input: *const u8,   // [num_channels] FP32
    conv_weight: *const u8, // [num_channels, kernel_size] FP32
    output: *mut u8,        // [num_channels] FP32
    num_channels: usize,
    kernel_size: usize,
    stream: &CudaStream,
) -> Result<()> {
    #[cfg(feature = "cuda")]
    {
        if stream.is_real() {
            let err = unsafe {
                cuda_ffi::vib3_launch_causal_conv1d(
                    conv_state,
                    new_input,
                    conv_weight,
                    output,
                    num_channels as i32,
                    kernel_size as i32,
                    stream.raw_ptr(),
                )
            };
            if err != 0 {
                return Err(Error::Cuda(format!(
                    "causal_conv1d launch failed (err={})",
                    err
                )));
            }
            return Ok(());
        }
    }
    // CPU fallback
    let state_len = kernel_size.saturating_sub(1);

    // SAFETY: Caller provides valid host buffers with the documented shapes.
    let conv_state_f32 = unsafe {
        std::slice::from_raw_parts_mut(conv_state as *mut f32, num_channels * state_len)
    };
    // SAFETY: Caller provides valid host buffers with `num_channels` elements.
    let new_input_f32 = unsafe { std::slice::from_raw_parts(new_input as *const f32, num_channels) };
    // SAFETY: Caller provides `num_channels * kernel_size` FP32 weights.
    let conv_weight_f32 = unsafe {
        std::slice::from_raw_parts(conv_weight as *const f32, num_channels * kernel_size)
    };
    // SAFETY: Caller provides writable host output with `num_channels` FP32 elements.
    let output_f32 = unsafe { std::slice::from_raw_parts_mut(output as *mut f32, num_channels) };

    for ch in 0..num_channels {
        let state_base = ch * state_len;
        let weight_base = ch * kernel_size;

        let mut acc = 0.0f32;
        for i in 0..state_len {
            acc += conv_state_f32[state_base + i] * conv_weight_f32[weight_base + i];
        }
        acc += new_input_f32[ch] * conv_weight_f32[weight_base + state_len];

        output_f32[ch] = acc / (1.0 + (-acc).exp());

        if state_len > 0 {
            for i in 0..(state_len - 1) {
                conv_state_f32[state_base + i] = conv_state_f32[state_base + i + 1];
            }
            conv_state_f32[state_base + state_len - 1] = new_input_f32[ch];
        }
    }

    Ok(())
}

/// L2 normalization: output[i] = input[i] / ||input[i]||_2.
/// Each of `num_vecs` vectors of length `dim` is normalized independently.
pub fn l2_norm(
    input: *const u8, // [num_vecs, dim] FP32
    output: *mut u8,  // [num_vecs, dim] FP32
    num_vecs: usize,
    dim: usize,
    eps: f32,
    stream: &CudaStream,
) -> Result<()> {
    #[cfg(feature = "cuda")]
    {
        if stream.is_real() {
            let err = unsafe {
                cuda_ffi::vib3_launch_l2_norm(
                    input,
                    output,
                    num_vecs as i32,
                    dim as i32,
                    eps,
                    stream.raw_ptr(),
                )
            };
            if err != 0 {
                return Err(Error::Cuda(format!("l2_norm launch failed (err={})", err)));
            }
            return Ok(());
        }
    }
    // CPU fallback
    // SAFETY: Caller provides `num_vecs * dim` FP32 elements for input/output.
    let input_f32 = unsafe { std::slice::from_raw_parts(input as *const f32, num_vecs * dim) };
    // SAFETY: Caller provides writable output for `num_vecs * dim` FP32 elements.
    let output_f32 = unsafe { std::slice::from_raw_parts_mut(output as *mut f32, num_vecs * dim) };

    for vec_idx in 0..num_vecs {
        let base = vec_idx * dim;
        let mut sum_sq = 0.0f64;
        for i in 0..dim {
            let v = input_f32[base + i] as f64;
            sum_sq += v * v;
        }
        let inv_norm = 1.0f32 / ((sum_sq as f32).sqrt() + eps);
        for i in 0..dim {
            output_f32[base + i] = input_f32[base + i] * inv_norm;
        }
    }

    Ok(())
}

/// Tiled repeat FP32 vectors: expand [num_groups, dim] → [num_groups*repeat, dim].
/// Result: [G0,G1,...,G_{n-1}, G0,G1,...,G_{n-1}, ...] (all groups tiled).
/// Matches the V-head tiled order produced by llama.cpp's _LinearAttentionVReorderBase.
pub fn repeat_tile_f32(
    input: *const u8, // [num_groups, dim] FP32
    output: *mut u8,  // [num_groups * repeat, dim] FP32
    num_groups: usize,
    dim: usize,
    repeat: usize,
    stream: &CudaStream,
) -> Result<()> {
    #[cfg(feature = "cuda")]
    {
        if stream.is_real() {
            let err = unsafe {
                cuda_ffi::vib3_launch_repeat_tile_f32(
                    input,
                    output,
                    num_groups as i32,
                    dim as i32,
                    repeat as i32,
                    stream.raw_ptr(),
                )
            };
            if err != 0 {
                return Err(Error::Cuda(format!(
                    "repeat_tile_f32 launch failed (err={})",
                    err
                )));
            }
            return Ok(());
        }
    }
    Err(Error::Cuda("repeat_tile_f32 requires GPU".to_string()))
}

/// DeltaNet gate computation: gate = -exp(A_log) * softplus(alpha + dt_bias).
pub fn deltanet_gate(
    alpha: *const u8,   // [num_heads] FP32
    dt_bias: *const u8, // [num_heads] FP32
    a_log: *const u8,   // [num_heads] FP32
    gate_out: *mut u8,  // [num_heads] FP32
    num_heads: usize,
    stream: &CudaStream,
) -> Result<()> {
    #[cfg(feature = "cuda")]
    {
        if stream.is_real() {
            let err = unsafe {
                cuda_ffi::vib3_launch_deltanet_gate(
                    alpha,
                    dt_bias,
                    a_log,
                    gate_out,
                    num_heads as i32,
                    stream.raw_ptr(),
                )
            };
            if err != 0 {
                return Err(Error::Cuda(format!(
                    "deltanet_gate launch failed (err={})",
                    err
                )));
            }
            return Ok(());
        }
    }
    Err(Error::Cuda("deltanet_gate requires GPU".to_string()))
}

/// Fused DeltaNet autoregressive step for all heads.
/// Per head: decay → retrieve → delta → update → query.
/// state: [num_heads, vdim, vdim] (mutable), q/k/v: [num_heads, vdim],
/// gate: [num_heads] (scalar decay), beta: [num_heads] (write strength).
pub fn deltanet_step(
    state: *mut u8,  // [num_heads, vdim, vdim] FP32, updated in-place
    q: *const u8,    // [num_heads, vdim] FP32
    k: *const u8,    // [num_heads, vdim] FP32
    v: *const u8,    // [num_heads, vdim] FP32
    gate: *const u8, // [num_heads] FP32
    beta: *const u8, // [num_heads] FP32
    output: *mut u8, // [num_heads, vdim] FP32
    num_heads: usize,
    vdim: usize,
    stream: &CudaStream,
) -> Result<()> {
    #[cfg(feature = "cuda")]
    {
        if stream.is_real() {
            let err = unsafe {
                cuda_ffi::vib3_launch_deltanet_step(
                    state,
                    q,
                    k,
                    v,
                    gate,
                    beta,
                    output,
                    num_heads as i32,
                    vdim as i32,
                    stream.raw_ptr(),
                )
            };
            if err != 0 {
                return Err(Error::Cuda(format!(
                    "deltanet_step launch failed (err={})",
                    err
                )));
            }
            return Ok(());
        }
    }
    Err(Error::Cuda("deltanet_step requires GPU".to_string()))
}

/// In-place FP32 scale: data[i] *= scale.
pub fn scale_f32(
    data: *mut u8, // [n] FP32, scaled in-place
    n: usize,
    scale: f32,
    stream: &CudaStream,
) -> Result<()> {
    #[cfg(feature = "cuda")]
    {
        if stream.is_real() {
            let err =
                unsafe { cuda_ffi::vib3_launch_scale_f32(data, n as i32, scale, stream.raw_ptr()) };
            if err != 0 {
                return Err(Error::Cuda(format!(
                    "scale_f32 launch failed (err={})",
                    err
                )));
            }
            return Ok(());
        }
    }
    Err(Error::Cuda("scale_f32 requires GPU".to_string()))
}

/// Element-wise sigmoid: output[i] = 1 / (1 + exp(-input[i])).
pub fn sigmoid(
    input: *const u8, // [n] FP32
    output: *mut u8,  // [n] FP32 (can alias input for in-place)
    n: usize,
    stream: &CudaStream,
) -> Result<()> {
    #[cfg(feature = "cuda")]
    {
        if stream.is_real() {
            let err =
                unsafe { cuda_ffi::vib3_launch_sigmoid(input, output, n as i32, stream.raw_ptr()) };
            if err != 0 {
                return Err(Error::Cuda(format!("sigmoid launch failed (err={})", err)));
            }
            return Ok(());
        }
    }
    Err(Error::Cuda("sigmoid requires GPU".to_string()))
}

/// Gated RMSNorm: output = RMSNorm(x, weight) * SiLU(gate).
/// Per-head normalization: `num_groups` independent heads, each over `group_dim` elements.
/// weight has shape [norm_dim] and cycles across each head's `group_dim`.
pub fn gated_rmsnorm(
    x: *const u8,      // [num_groups * group_dim] FP32
    gate: *const u8,   // [num_groups * group_dim] FP32
    weight: *const u8, // [norm_dim] FP32
    output: *mut u8,   // [num_groups * group_dim] FP32
    num_groups: usize,
    group_dim: usize,
    norm_dim: usize,
    eps: f32,
    stream: &CudaStream,
) -> Result<()> {
    #[cfg(feature = "cuda")]
    {
        if stream.is_real() {
            let err = unsafe {
                cuda_ffi::vib3_launch_gated_rmsnorm(
                    x,
                    gate,
                    weight,
                    output,
                    num_groups as i32,
                    group_dim as i32,
                    norm_dim as i32,
                    eps,
                    stream.raw_ptr(),
                )
            };
            if err != 0 {
                return Err(Error::Cuda(format!(
                    "gated_rmsnorm launch failed (err={})",
                    err
                )));
            }
            return Ok(());
        }
    }
    Err(Error::Cuda("gated_rmsnorm requires GPU".to_string()))
}

/// Deinterleave FP16: splits [A0(chunk), B0(chunk), A1(chunk), B1(chunk), ...]
/// into output_a = [A0, A1, ...] and output_b = [B0, B1, ...].
/// Used for Qwen3.5 Q+gate extraction from doubled Q projection.
pub fn deinterleave_f16(
    input: *const u8,  // [num_chunks * 2 * chunk_size] FP16 interleaved
    output_a: *mut u8, // [num_chunks * chunk_size] FP16
    output_b: *mut u8, // [num_chunks * chunk_size] FP16
    chunk_size: usize,
    num_chunks: usize,
    stream: &CudaStream,
) -> Result<()> {
    #[cfg(feature = "cuda")]
    {
        if stream.is_real() {
            let err = unsafe {
                cuda_ffi::vib3_launch_deinterleave_f16(
                    input,
                    output_a,
                    output_b,
                    chunk_size as i32,
                    num_chunks as i32,
                    stream.raw_ptr(),
                )
            };
            if err != 0 {
                return Err(Error::Cuda(format!(
                    "deinterleave_f16 launch failed (err={})",
                    err
                )));
            }
            return Ok(());
        }
    }
    Err(Error::Cuda("deinterleave_f16 requires GPU".to_string()))
}

/// Partial RoPE in-place (FP16). Applies rotary embeddings to only the
/// first `rope_dim` dimensions of each head, leaving the rest unchanged.
/// data: [num_heads, head_dim] with given stride between heads.
/// d_position: device pointer to int32 position scalar.
pub fn partial_rope(
    data: *mut u8, // [num_heads * stride] FP16, modified in-place
    head_dim: usize,
    rope_dim: usize,
    stride: usize,
    num_heads: usize,
    d_position: *const u8, // device pointer to int32
    rope_base: f32,
    stream: &CudaStream,
) -> Result<()> {
    #[cfg(feature = "cuda")]
    {
        if stream.is_real() {
            let err = unsafe {
                cuda_ffi::vib3_launch_partial_rope(
                    data,
                    head_dim as i32,
                    rope_dim as i32,
                    stride as i32,
                    num_heads as i32,
                    d_position,
                    rope_base,
                    stream.raw_ptr(),
                )
            };
            if err != 0 {
                return Err(Error::Cuda(format!(
                    "partial_rope launch failed (err={})",
                    err
                )));
            }
            return Ok(());
        }
    }
    Err(Error::Cuda("partial_rope requires GPU".to_string()))
}

/// Per-head RMSNorm in-place (FP16). Normalizes each head independently.
/// data: [num_heads * stride] FP16, weight: [head_dim] FP16.
/// stride = distance between heads in elements (2*head_dim for interleaved Q+gate,
/// head_dim for contiguous K).
pub fn per_head_rmsnorm(
    data: *mut u8,     // [num_heads * stride] FP16, modified in-place
    weight: *const u8, // [head_dim] FP16
    head_dim: usize,
    stride: usize,
    num_heads: usize,
    eps: f32,
    stream: &CudaStream,
) -> Result<()> {
    #[cfg(feature = "cuda")]
    {
        if stream.is_real() {
            let err = unsafe {
                cuda_ffi::vib3_launch_per_head_rmsnorm(
                    data,
                    weight,
                    head_dim as i32,
                    stride as i32,
                    num_heads as i32,
                    eps,
                    stream.raw_ptr(),
                )
            };
            if err != 0 {
                return Err(Error::Cuda(format!(
                    "per_head_rmsnorm launch failed (err={})",
                    err
                )));
            }
            return Ok(());
        }
    }
    Err(Error::Cuda("per_head_rmsnorm requires GPU".to_string()))
}

/// Sigmoid-gated multiply (FP16): output[i] = input[i] * sigmoid(gate[i]).
/// output can alias input for in-place gating.
pub fn sigmoid_mul_f16(
    output: *mut u8,  // [n] FP16
    input: *const u8, // [n] FP16
    gate: *const u8,  // [n] FP16
    n: usize,
    stream: &CudaStream,
) -> Result<()> {
    #[cfg(feature = "cuda")]
    {
        if stream.is_real() {
            let err = unsafe {
                cuda_ffi::vib3_launch_sigmoid_mul_f16(
                    output,
                    input,
                    gate,
                    n as i32,
                    stream.raw_ptr(),
                )
            };
            if err != 0 {
                return Err(Error::Cuda(format!(
                    "sigmoid_mul_f16 launch failed (err={})",
                    err
                )));
            }
            return Ok(());
        }
    }
    Err(Error::Cuda("sigmoid_mul_f16 requires GPU".to_string()))
}
