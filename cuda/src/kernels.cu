// vib3 CUDA kernels — partial matmul, fused SwiGLU, and supporting ops.
//
// Compiled with nvcc, linked into Rust via build.rs.
// These kernels operate on PAGE-LEVEL weight slices, not full matrices.
//
// Build: nvcc -c -O3 -arch=sm_90 -arch=sm_100 kernels.cu -o kernels.o

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_fp4.h>
#include <cstdint>
#include <cstdio>

// ─── Configuration ──────────────────────────────────────────────────────
//
// GEMV tiling: each output row is computed by THREADS_PER_ROW threads
// cooperating via warp shuffle + shared memory reduction.
// This is ~20× faster than 1-thread-per-row for K=4096.

#define THREADS_PER_ROW 128
#define ROWS_PER_BLOCK 2
#define BLOCK_SIZE (THREADS_PER_ROW * ROWS_PER_BLOCK)  // 256
#define WARP_SIZE 32

// Fast vectorized FP16 GEMV / SwiGLU: 4 rows per block × 64 threads per row
#define FAST_SWIGLU_ROWS 4
#define FAST_SWIGLU_TPR 64
#define FAST_SWIGLU_BLK (FAST_SWIGLU_ROWS * FAST_SWIGLU_TPR) // 256

// ─── Device Helpers ──────────────────────────────────────────────────────

// BF16 → float conversion. BF16 is the upper 16 bits of IEEE 754 float32,
// so conversion is just a 16-bit left shift. No exponent/mantissa remapping.
__device__ __forceinline__ float bf16_to_float(unsigned short bf16) {
    unsigned int bits = ((unsigned int)bf16) << 16;
    float result;
    memcpy(&result, &bits, sizeof(float));
    return result;
}

// FP16 → float conversion (used by NVFP4 scale reading).
__device__ __forceinline__ float fp16_scale_to_float(unsigned short fp16_bits) {
    return __half2float(*reinterpret_cast<const half*>(&fp16_bits));
}

// BF16 → float conversion for INT4 quantization scales.
// vib3 quantize_weights_to_int4() stores scales as BF16 (sign=1, exp=8, mantissa=7)
// which has the same exponent range as FP32.  Reading BF16 bits as FP16 would
// produce wildly wrong values (the 8-bit BF16 exponent overflows the 5-bit FP16 exponent).
__device__ __forceinline__ float bf16_scale_to_float(unsigned short bf16_bits) {
    // BF16 is the upper 16 bits of an FP32: just shift left by 16.
    unsigned int fp32_bits = (unsigned int)bf16_bits << 16;
    return __uint_as_float(fp32_bits);
}

// ─── Device Kernels ──────────────────────────────────────────────────────

// Warp-level reduction (sum)
__device__ __forceinline__ float warp_reduce_sum(float val) {
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

// Tiled GEMV FP16: each row uses THREADS_PER_ROW threads for the K-dim reduction
// Grid: (ceil(M_slice / ROWS_PER_BLOCK)) blocks of BLOCK_SIZE threads
__global__ void vib3_partial_matmul_fp16(
    const half* __restrict__ input,
    const half* __restrict__ weight,
    half* __restrict__ output,
    int K,
    int M_slice
) {
    // Which row within the block (0 or 1)
    const int local_row = threadIdx.x / THREADS_PER_ROW;
    const int lane = threadIdx.x % THREADS_PER_ROW;
    const int row = blockIdx.x * ROWS_PER_BLOCK + local_row;

    if (row >= M_slice) return;

    // Partial dot product: each thread handles K/THREADS_PER_ROW elements
    const half* row_ptr = weight + (long long)row * K;
    float acc = 0.0f;
    for (int k = lane; k < K; k += THREADS_PER_ROW) {
        acc += __half2float(input[k]) * __half2float(row_ptr[k]);
    }

    // Warp-level reduction
    acc = warp_reduce_sum(acc);

    // Inter-warp reduction via shared memory
    // THREADS_PER_ROW/WARP_SIZE = 4 warps per row
    __shared__ float smem[ROWS_PER_BLOCK * (THREADS_PER_ROW / WARP_SIZE)];
    const int warp_id = lane / WARP_SIZE;
    const int warp_lane = lane % WARP_SIZE;

    if (warp_lane == 0) {
        smem[local_row * (THREADS_PER_ROW / WARP_SIZE) + warp_id] = acc;
    }
    __syncthreads();

    // First warp of each row does final reduction
    // All 32 lanes of warp 0 must participate; lanes >= n_warps contribute 0.
    if (warp_id == 0) {
        float final_acc = (warp_lane < (THREADS_PER_ROW / WARP_SIZE))
            ? smem[local_row * (THREADS_PER_ROW / WARP_SIZE) + warp_lane]
            : 0.0f;
        final_acc = warp_reduce_sum(final_acc);
        if (warp_lane == 0) {
            output[row] = __float2half(final_acc);
        }
    }
}

// Tiled GEMV INT4: dequantize on the fly with per-group BF16 scales
// Same tiling strategy as FP16 but with INT4 unpacking
__global__ void vib3_partial_matmul_int4(
    const half* __restrict__ input,
    const uint8_t* __restrict__ weight,
    const unsigned short* __restrict__ scales,  // BF16 scale values (from quantize_weights_to_int4)
    half* __restrict__ output,
    int K,
    int M_slice,
    int group_size
) {
    const int local_row = threadIdx.x / THREADS_PER_ROW;
    const int lane = threadIdx.x % THREADS_PER_ROW;
    const int row = blockIdx.x * ROWS_PER_BLOCK + local_row;

    if (row >= M_slice) return;

    const int packed_k = (K + 1) / 2;
    const int num_groups = (K + group_size - 1) / group_size;
    const uint8_t* row_weight = weight + (long long)row * packed_k;
    const unsigned short* row_scales = scales + (long long)row * num_groups;

    float acc = 0.0f;
    // Each thread processes elements at stride THREADS_PER_ROW, stepping by 2 (one byte = 2 weights)
    for (int k = lane * 2; k < K; k += THREADS_PER_ROW * 2) {
        int byte_idx = k / 2;
        uint8_t packed = row_weight[byte_idx];
        int w0 = (int)((packed >> 0) & 0xF) - 8;
        int w1 = (int)((packed >> 4) & 0xF) - 8;

        int group = k / group_size;
        float scale = bf16_scale_to_float(row_scales[group]);

        acc += __half2float(input[k]) * (float)w0 * scale;
        if (k + 1 < K) {
            acc += __half2float(input[k + 1]) * (float)w1 * scale;
        }
    }

    // Warp reduction
    acc = warp_reduce_sum(acc);

    // Inter-warp reduction
    __shared__ float smem[ROWS_PER_BLOCK * (THREADS_PER_ROW / WARP_SIZE)];
    const int warp_id = lane / WARP_SIZE;
    const int warp_lane = lane % WARP_SIZE;

    if (warp_lane == 0) {
        smem[local_row * (THREADS_PER_ROW / WARP_SIZE) + warp_id] = acc;
    }
    __syncthreads();

    // All 32 lanes of warp 0 must participate; lanes >= n_warps contribute 0.
    if (warp_id == 0) {
        float final_acc = (warp_lane < (THREADS_PER_ROW / WARP_SIZE))
            ? smem[local_row * (THREADS_PER_ROW / WARP_SIZE) + warp_lane]
            : 0.0f;
        final_acc = warp_reduce_sum(final_acc);
        if (warp_lane == 0) {
            output[row] = __float2half(final_acc);
        }
    }
}

// FP32-input INT4 matmul: accepts FP32 input vector instead of FP16
// Reduces precision loss in the dot product by avoiding FP16 input truncation
__global__ void vib3_partial_matmul_int4_f32(
    const float* __restrict__ input,
    const uint8_t* __restrict__ weight,
    const unsigned short* __restrict__ scales,
    half* __restrict__ output,
    int K,
    int M_slice,
    int group_size
) {
    const int local_row = threadIdx.x / THREADS_PER_ROW;
    const int lane = threadIdx.x % THREADS_PER_ROW;
    const int row = blockIdx.x * ROWS_PER_BLOCK + local_row;

    if (row >= M_slice) return;

    const int packed_k = (K + 1) / 2;
    const int num_groups = (K + group_size - 1) / group_size;
    const uint8_t* row_weight = weight + (long long)row * packed_k;
    const unsigned short* row_scales = scales + (long long)row * num_groups;

    float acc = 0.0f;
    for (int k = lane * 2; k < K; k += THREADS_PER_ROW * 2) {
        int byte_idx = k / 2;
        uint8_t packed = row_weight[byte_idx];
        int w0 = (int)((packed >> 0) & 0xF) - 8;
        int w1 = (int)((packed >> 4) & 0xF) - 8;

        int group = k / group_size;
        float scale = bf16_scale_to_float(row_scales[group]);

        acc += input[k] * (float)w0 * scale;
        if (k + 1 < K) {
            acc += input[k + 1] * (float)w1 * scale;
        }
    }

    // Warp reduction
    acc = warp_reduce_sum(acc);

    // Inter-warp reduction
    __shared__ float smem[ROWS_PER_BLOCK * (THREADS_PER_ROW / WARP_SIZE)];
    const int warp_id = lane / WARP_SIZE;
    const int warp_lane = lane % WARP_SIZE;

    if (warp_lane == 0) {
        smem[local_row * (THREADS_PER_ROW / WARP_SIZE) + warp_id] = acc;
    }
    __syncthreads();

    if (warp_id == 0) {
        float final_acc = (warp_lane < (THREADS_PER_ROW / WARP_SIZE))
            ? smem[local_row * (THREADS_PER_ROW / WARP_SIZE) + warp_lane]
            : 0.0f;
        final_acc = warp_reduce_sum(final_acc);
        if (warp_lane == 0) {
            output[row] = __float2half(final_acc);
        }
    }
}

// FP32-input FP16 matmul: accepts FP32 input vector, FP16 weight matrix
__global__ void vib3_partial_matmul_fp16_f32in(
    const float* __restrict__ input,
    const half* __restrict__ weight,
    half* __restrict__ output,
    int K,
    int M_slice
) {
    const int local_row = threadIdx.x / THREADS_PER_ROW;
    const int lane = threadIdx.x % THREADS_PER_ROW;
    const int row = blockIdx.x * ROWS_PER_BLOCK + local_row;

    if (row >= M_slice) return;

    const half* row_w = weight + (long long)row * K;

    float acc = 0.0f;
    for (int k = lane; k < K; k += THREADS_PER_ROW) {
        acc += input[k] * __half2float(row_w[k]);
    }

    acc = warp_reduce_sum(acc);

    __shared__ float smem[ROWS_PER_BLOCK * (THREADS_PER_ROW / WARP_SIZE)];
    const int warp_id = lane / WARP_SIZE;
    const int warp_lane = lane % WARP_SIZE;

    if (warp_lane == 0) {
        smem[local_row * (THREADS_PER_ROW / WARP_SIZE) + warp_id] = acc;
    }
    __syncthreads();

    if (warp_id == 0) {
        float final_acc = (warp_lane < (THREADS_PER_ROW / WARP_SIZE))
            ? smem[local_row * (THREADS_PER_ROW / WARP_SIZE) + warp_lane]
            : 0.0f;
        final_acc = warp_reduce_sum(final_acc);
        if (warp_lane == 0) {
            output[row] = __float2half(final_acc);
        }
    }
}

// FP32-input fused SwiGLU INT4: reads FP32 input, INT4 weights
__global__ void vib3_fused_swiglu_int4_f32(
    const float* __restrict__ input,
    const uint8_t* __restrict__ up_weight,
    const unsigned short* __restrict__ up_scales,
    const uint8_t* __restrict__ gate_weight,
    const unsigned short* __restrict__ gate_scales,
    half* __restrict__ output,
    int K,
    int M_slice,
    int group_size
) {
    const int local_row = threadIdx.x / THREADS_PER_ROW;
    const int lane = threadIdx.x % THREADS_PER_ROW;
    const int row = blockIdx.x * ROWS_PER_BLOCK + local_row;

    if (row >= M_slice) return;

    const int packed_k = (K + 1) / 2;
    const int num_groups = (K + group_size - 1) / group_size;
    const uint8_t* up_row = up_weight + (long long)row * packed_k;
    const unsigned short* up_s = up_scales + (long long)row * num_groups;
    const uint8_t* gate_row = gate_weight + (long long)row * packed_k;
    const unsigned short* gate_s = gate_scales + (long long)row * num_groups;

    float up_acc = 0.0f;
    float gate_acc = 0.0f;
    for (int k = lane * 2; k < K; k += THREADS_PER_ROW * 2) {
        int byte_idx = k / 2;
        int group = k / group_size;

        uint8_t up_packed = up_row[byte_idx];
        float up_scale = bf16_scale_to_float(up_s[group]);
        int up_w0 = (int)((up_packed >> 0) & 0xF) - 8;
        int up_w1 = (int)((up_packed >> 4) & 0xF) - 8;

        uint8_t gate_packed = gate_row[byte_idx];
        float gate_scale = bf16_scale_to_float(gate_s[group]);
        int gate_w0 = (int)((gate_packed >> 0) & 0xF) - 8;
        int gate_w1 = (int)((gate_packed >> 4) & 0xF) - 8;

        float inp0 = input[k];
        up_acc += inp0 * (float)up_w0 * up_scale;
        gate_acc += inp0 * (float)gate_w0 * gate_scale;

        if (k + 1 < K) {
            float inp1 = input[k + 1];
            up_acc += inp1 * (float)up_w1 * up_scale;
            gate_acc += inp1 * (float)gate_w1 * gate_scale;
        }
    }

    up_acc = warp_reduce_sum(up_acc);
    gate_acc = warp_reduce_sum(gate_acc);

    __shared__ float smem[ROWS_PER_BLOCK * (THREADS_PER_ROW / WARP_SIZE) * 2];
    const int warp_id = lane / WARP_SIZE;
    const int warp_lane = lane % WARP_SIZE;
    const int n_warps = THREADS_PER_ROW / WARP_SIZE;

    if (warp_lane == 0) {
        smem[(local_row * n_warps + warp_id) * 2 + 0] = up_acc;
        smem[(local_row * n_warps + warp_id) * 2 + 1] = gate_acc;
    }
    __syncthreads();

    if (warp_id == 0) {
        float final_up = (warp_lane < n_warps)
            ? smem[(local_row * n_warps + warp_lane) * 2 + 0]
            : 0.0f;
        float final_gate = (warp_lane < n_warps)
            ? smem[(local_row * n_warps + warp_lane) * 2 + 1]
            : 0.0f;
        final_up = warp_reduce_sum(final_up);
        final_gate = warp_reduce_sum(final_gate);
        if (warp_lane == 0) {
            // SwiGLU: SiLU(gate) * up
            float silu = final_gate / (1.0f + expf(-final_gate));
            output[row] = __float2half(silu * final_up);
        }
    }
}

// FP32-input fused SwiGLU FP16: reads FP32 input, FP16 weights
__global__ void vib3_fused_swiglu_fp16_f32in(
    const float* __restrict__ input,
    const half* __restrict__ up_weight,
    const half* __restrict__ gate_weight,
    half* __restrict__ output,
    int K,
    int M_slice
) {
    const int local_row = threadIdx.x / THREADS_PER_ROW;
    const int lane = threadIdx.x % THREADS_PER_ROW;
    const int row = blockIdx.x * ROWS_PER_BLOCK + local_row;

    if (row >= M_slice) return;

    const half* up_ptr = up_weight + (long long)row * K;
    const half* gate_ptr = gate_weight + (long long)row * K;

    float up_acc = 0.0f;
    float gate_acc = 0.0f;
    for (int k = lane; k < K; k += THREADS_PER_ROW) {
        float inp = input[k];
        up_acc   += inp * __half2float(up_ptr[k]);
        gate_acc += inp * __half2float(gate_ptr[k]);
    }

    up_acc = warp_reduce_sum(up_acc);
    gate_acc = warp_reduce_sum(gate_acc);

    __shared__ float smem[ROWS_PER_BLOCK * (THREADS_PER_ROW / WARP_SIZE) * 2];
    const int warp_id = lane / WARP_SIZE;
    const int warp_lane = lane % WARP_SIZE;
    const int n_warps = THREADS_PER_ROW / WARP_SIZE;

    if (warp_lane == 0) {
        smem[(local_row * n_warps + warp_id) * 2 + 0] = up_acc;
        smem[(local_row * n_warps + warp_id) * 2 + 1] = gate_acc;
    }
    __syncthreads();

    if (warp_id == 0) {
        float final_up = (warp_lane < n_warps)
            ? smem[(local_row * n_warps + warp_lane) * 2 + 0]
            : 0.0f;
        float final_gate = (warp_lane < n_warps)
            ? smem[(local_row * n_warps + warp_lane) * 2 + 1]
            : 0.0f;
        final_up = warp_reduce_sum(final_up);
        final_gate = warp_reduce_sum(final_gate);
        if (warp_lane == 0) {
            float silu = final_gate / (1.0f + expf(-final_gate));
            output[row] = __float2half(silu * final_up);
        }
    }
}

// ─── NVFP4 (MXFP4 E2M1) Kernels ───────────────────────────────────────────
//
// MXFP4 format: E2M1 (1 sign, 2 exponent, 1 mantissa)
// Values: {0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0} × {+1, -1}
// Block scaling: one FP16 scale per block of 32 elements
//
// Packed layout: 2 E2M1 nibbles per byte (same as INT4)
// Scale layout: FP16 scales, one per group of 32 weights
//
// Advantages over INT4:
//   - Floating-point encoding better represents the weight distribution
//   - log-spaced values capture both small (near-zero) and large values
//   - Native Blackwell tensor core support (sm_100+) for future batched path

// E2M1 lookup table: map 4-bit E2M1 encoding to float.
// bit3=sign, bit2-1=exponent, bit0=mantissa
// E2M1: subnormal when exp=0, normal otherwise
//   exp=0, man=0 → 0.0         exp=0, man=1 → 0.5
//   exp=1, man=0 → 1.0         exp=1, man=1 → 1.5
//   exp=2, man=0 → 2.0         exp=2, man=1 → 3.0
//   exp=3, man=0 → 4.0         exp=3, man=1 → 6.0
__device__ __constant__ float nvfp4_lut[16] = {
     0.0f,  0.5f,  1.0f,  1.5f,  2.0f,  3.0f,  4.0f,  6.0f,  // positive
    -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f,  // negative
};

// Tiled GEMV NVFP4: dequantize E2M1 on the fly with per-block BF16 scales
// FP32 input for maximum precision in the accumulation path
// Note: scales are stored as BF16 (from E8M0→BF16 conversion), NOT FP16
__global__ void vib3_partial_matmul_nvfp4(
    const float* __restrict__ input,
    const uint8_t* __restrict__ weight,   // packed E2M1 nibbles (2 per byte)
    const unsigned short* __restrict__ scales, // BF16 scale per block of 32
    half* __restrict__ output,
    int K,
    int M_slice,
    int block_size                        // typically 32
) {
    const int local_row = threadIdx.x / THREADS_PER_ROW;
    const int lane = threadIdx.x % THREADS_PER_ROW;
    const int row = blockIdx.x * ROWS_PER_BLOCK + local_row;

    if (row >= M_slice) return;

    const int packed_k = (K + 1) / 2;
    const int num_blocks = (K + block_size - 1) / block_size;
    const uint8_t* row_weight = weight + (long long)row * packed_k;
    const unsigned short* row_scales = scales + (long long)row * num_blocks;

    float acc = 0.0f;
    for (int k = lane * 2; k < K; k += THREADS_PER_ROW * 2) {
        int byte_idx = k / 2;
        uint8_t packed = row_weight[byte_idx];

        // Unpack two E2M1 nibbles and dequantize via LUT
        float w0 = nvfp4_lut[packed & 0xF];
        float w1 = nvfp4_lut[(packed >> 4) & 0xF];

        // Block scale (BF16 from E8M0 conversion)
        int blk = k / block_size;
        float scale = bf16_to_float(row_scales[blk]);

        acc += input[k] * w0 * scale;
        if (k + 1 < K) {
            acc += input[k + 1] * w1 * scale;
        }
    }

    // Warp reduction
    acc = warp_reduce_sum(acc);

    // Inter-warp reduction
    __shared__ float smem[ROWS_PER_BLOCK * (THREADS_PER_ROW / WARP_SIZE)];
    const int warp_id = lane / WARP_SIZE;
    const int warp_lane = lane % WARP_SIZE;

    if (warp_lane == 0) {
        smem[local_row * (THREADS_PER_ROW / WARP_SIZE) + warp_id] = acc;
    }
    __syncthreads();

    if (warp_id == 0) {
        float final_acc = (warp_lane < (THREADS_PER_ROW / WARP_SIZE))
            ? smem[local_row * (THREADS_PER_ROW / WARP_SIZE) + warp_lane]
            : 0.0f;
        final_acc = warp_reduce_sum(final_acc);
        if (warp_lane == 0) {
            output[row] = __float2half(final_acc);
        }
    }
}

// Fused SwiGLU NVFP4: SiLU(gate·x) * (up·x) with MXFP4 weights
// FP32 input for maximum precision
// Note: scales are BF16 (from E8M0→BF16 conversion), NOT FP16
__global__ void vib3_fused_swiglu_nvfp4(
    const float* __restrict__ input,
    const uint8_t* __restrict__ up_weight,
    const unsigned short* __restrict__ up_scales,
    const uint8_t* __restrict__ gate_weight,
    const unsigned short* __restrict__ gate_scales,
    half* __restrict__ output,
    int K,
    int M_slice,
    int block_size
) {
    const int local_row = threadIdx.x / THREADS_PER_ROW;
    const int lane = threadIdx.x % THREADS_PER_ROW;
    const int row = blockIdx.x * ROWS_PER_BLOCK + local_row;

    if (row >= M_slice) return;

    const int packed_k = (K + 1) / 2;
    const int num_blocks = (K + block_size - 1) / block_size;
    const uint8_t* up_row = up_weight + (long long)row * packed_k;
    const unsigned short* up_s = up_scales + (long long)row * num_blocks;
    const uint8_t* gate_row = gate_weight + (long long)row * packed_k;
    const unsigned short* gate_s = gate_scales + (long long)row * num_blocks;

    float up_acc = 0.0f;
    float gate_acc = 0.0f;
    for (int k = lane * 2; k < K; k += THREADS_PER_ROW * 2) {
        int byte_idx = k / 2;
        int blk = k / block_size;

        uint8_t up_packed = up_row[byte_idx];
        float up_scale = bf16_to_float(up_s[blk]);
        float up_w0 = nvfp4_lut[up_packed & 0xF];
        float up_w1 = nvfp4_lut[(up_packed >> 4) & 0xF];

        uint8_t gate_packed = gate_row[byte_idx];
        float gate_scale = bf16_to_float(gate_s[blk]);
        float gate_w0 = nvfp4_lut[gate_packed & 0xF];
        float gate_w1 = nvfp4_lut[(gate_packed >> 4) & 0xF];

        float inp0 = input[k];
        up_acc += inp0 * up_w0 * up_scale;
        gate_acc += inp0 * gate_w0 * gate_scale;

        if (k + 1 < K) {
            float inp1 = input[k + 1];
            up_acc += inp1 * up_w1 * up_scale;
            gate_acc += inp1 * gate_w1 * gate_scale;
        }
    }

    up_acc = warp_reduce_sum(up_acc);
    gate_acc = warp_reduce_sum(gate_acc);

    __shared__ float smem[ROWS_PER_BLOCK * (THREADS_PER_ROW / WARP_SIZE) * 2];
    const int warp_id = lane / WARP_SIZE;
    const int warp_lane = lane % WARP_SIZE;
    const int n_warps = THREADS_PER_ROW / WARP_SIZE;

    if (warp_lane == 0) {
        smem[(local_row * n_warps + warp_id) * 2 + 0] = up_acc;
        smem[(local_row * n_warps + warp_id) * 2 + 1] = gate_acc;
    }
    __syncthreads();

    if (warp_id == 0) {
        float final_up = (warp_lane < n_warps)
            ? smem[(local_row * n_warps + warp_lane) * 2 + 0]
            : 0.0f;
        float final_gate = (warp_lane < n_warps)
            ? smem[(local_row * n_warps + warp_lane) * 2 + 1]
            : 0.0f;
        final_up = warp_reduce_sum(final_up);
        final_gate = warp_reduce_sum(final_gate);
        if (warp_lane == 0) {
            // SwiGLU: SiLU(gate) * up
            float silu = final_gate / (1.0f + expf(-final_gate));
            output[row] = __float2half(silu * final_up);
        }
    }
}

// ─── NVFP4 FP16-input variants (for down-projection where intermediate is FP16) ───

// Tiled GEMV NVFP4 with FP16 input: same as vib3_partial_matmul_nvfp4 but reads half*
// Note: scales are BF16 (from E8M0→BF16 conversion), NOT FP16
__global__ void vib3_partial_matmul_nvfp4_fp16in(
    const half* __restrict__ input,
    const uint8_t* __restrict__ weight,
    const unsigned short* __restrict__ scales,
    half* __restrict__ output,
    int K,
    int M_slice,
    int block_size
) {
    const int local_row = threadIdx.x / THREADS_PER_ROW;
    const int lane = threadIdx.x % THREADS_PER_ROW;
    const int row = blockIdx.x * ROWS_PER_BLOCK + local_row;

    if (row >= M_slice) return;

    const int packed_k = (K + 1) / 2;
    const int num_blocks = (K + block_size - 1) / block_size;
    const uint8_t* row_weight = weight + (long long)row * packed_k;
    const unsigned short* row_scales = scales + (long long)row * num_blocks;

    float acc = 0.0f;
    for (int k = lane * 2; k < K; k += THREADS_PER_ROW * 2) {
        int byte_idx = k / 2;
        uint8_t packed = row_weight[byte_idx];

        float w0 = nvfp4_lut[packed & 0xF];
        float w1 = nvfp4_lut[(packed >> 4) & 0xF];

        int blk = k / block_size;
        float scale = bf16_to_float(row_scales[blk]);

        acc += __half2float(input[k]) * w0 * scale;
        if (k + 1 < K) {
            acc += __half2float(input[k + 1]) * w1 * scale;
        }
    }

    acc = warp_reduce_sum(acc);

    __shared__ float smem[ROWS_PER_BLOCK * (THREADS_PER_ROW / WARP_SIZE)];
    const int warp_id = lane / WARP_SIZE;
    const int warp_lane = lane % WARP_SIZE;

    if (warp_lane == 0) {
        smem[local_row * (THREADS_PER_ROW / WARP_SIZE) + warp_id] = acc;
    }
    __syncthreads();

    if (warp_id == 0) {
        float final_acc = (warp_lane < (THREADS_PER_ROW / WARP_SIZE))
            ? smem[local_row * (THREADS_PER_ROW / WARP_SIZE) + warp_lane]
            : 0.0f;
        final_acc = warp_reduce_sum(final_acc);
        if (warp_lane == 0) {
            output[row] = __float2half(final_acc);
        }
    }
}

// ─── End NVFP4 Kernels ─────────────────────────────────────────────────────

// FP32-input router GEMV: scores[e] = f32_hidden[d] · fp16_weights[e, d]
// Tiled router GEMV: FP32 input × FP16 weights → FP32 scores.
// Uses THREADS_PER_ROW threads per expert with warp + inter-warp reduction,
// same pattern as vib3_partial_matmul_fp16. 128 threads per row, 2 rows per
// block → 128 blocks for 256 experts, spreading across many SMs.
// The old 1-thread-per-expert kernel took ~0.5ms (1 SM); this takes ~5-10μs.
__global__ void vib3_router_gemv_f32(
    const float* __restrict__ hidden_state,
    const half* __restrict__ router_weights,
    float* __restrict__ scores,
    int hidden_dim,
    int num_experts
) {
    const int local_row = threadIdx.x / THREADS_PER_ROW;
    const int lane = threadIdx.x % THREADS_PER_ROW;
    const int expert = blockIdx.x * ROWS_PER_BLOCK + local_row;

    if (expert >= num_experts) return;

    // Tiled dot product: each thread handles hidden_dim/THREADS_PER_ROW elements
    const half* row_ptr = router_weights + (long long)expert * hidden_dim;
    float acc = 0.0f;
    for (int d = lane; d < hidden_dim; d += THREADS_PER_ROW) {
        acc += hidden_state[d] * __half2float(row_ptr[d]);
    }

    // Warp-level reduction
    acc = warp_reduce_sum(acc);

    // Inter-warp reduction via shared memory
    __shared__ float smem[ROWS_PER_BLOCK * (THREADS_PER_ROW / WARP_SIZE)];
    const int warp_id = lane / WARP_SIZE;
    const int warp_lane = lane % WARP_SIZE;

    if (warp_lane == 0) {
        smem[local_row * (THREADS_PER_ROW / WARP_SIZE) + warp_id] = acc;
    }
    __syncthreads();

    if (warp_id == 0) {
        float final_acc = (warp_lane < (THREADS_PER_ROW / WARP_SIZE))
            ? smem[local_row * (THREADS_PER_ROW / WARP_SIZE) + warp_lane]
            : 0.0f;
        final_acc = warp_reduce_sum(final_acc);
        if (warp_lane == 0) {
            scores[expert] = final_acc;
        }
    }
}

// Tiled fused SwiGLU FP16: output = SiLU(input × up) * (input × gate)
// Same tiling: THREADS_PER_ROW threads per row, two accumulators
__global__ void vib3_fused_swiglu_fp16(
    const half* __restrict__ input,
    const half* __restrict__ up_weight,
    const half* __restrict__ gate_weight,
    half* __restrict__ output,
    int K,
    int M_slice
) {
    const int local_row = threadIdx.x / THREADS_PER_ROW;
    const int lane = threadIdx.x % THREADS_PER_ROW;
    const int row = blockIdx.x * ROWS_PER_BLOCK + local_row;

    if (row >= M_slice) return;

    const half* up_ptr = up_weight + (long long)row * K;
    const half* gate_ptr = gate_weight + (long long)row * K;

    float up_acc = 0.0f;
    float gate_acc = 0.0f;
    for (int k = lane; k < K; k += THREADS_PER_ROW) {
        float inp = __half2float(input[k]);
        up_acc   += inp * __half2float(up_ptr[k]);
        gate_acc += inp * __half2float(gate_ptr[k]);
    }

    // Warp reduction for both accumulators
    up_acc = warp_reduce_sum(up_acc);
    gate_acc = warp_reduce_sum(gate_acc);

    // Inter-warp reduction (need 2 floats per warp per row)
    __shared__ float smem[ROWS_PER_BLOCK * (THREADS_PER_ROW / WARP_SIZE) * 2];
    const int warp_id = lane / WARP_SIZE;
    const int warp_lane = lane % WARP_SIZE;
    const int n_warps = THREADS_PER_ROW / WARP_SIZE;

    if (warp_lane == 0) {
        smem[(local_row * n_warps + warp_id) * 2 + 0] = up_acc;
        smem[(local_row * n_warps + warp_id) * 2 + 1] = gate_acc;
    }
    __syncthreads();

    // All 32 lanes of warp 0 must participate; lanes >= n_warps contribute 0.
    if (warp_id == 0) {
        float final_up = (warp_lane < n_warps)
            ? smem[(local_row * n_warps + warp_lane) * 2 + 0]
            : 0.0f;
        float final_gate = (warp_lane < n_warps)
            ? smem[(local_row * n_warps + warp_lane) * 2 + 1]
            : 0.0f;
        final_up = warp_reduce_sum(final_up);
        final_gate = warp_reduce_sum(final_gate);
        if (warp_lane == 0) {
            // SwiGLU: SiLU(gate) * up — SiLU applied to gate_proj output
            float silu = final_gate / (1.0f + expf(-final_gate));
            output[row] = __float2half(silu * final_up);
        }
    }
}

// SiLU-multiply: output[i] = SiLU(a[i]) * b[i]  (for decomposed INT4 SwiGLU)
// Caller is responsible for passing gate_proj output as `a` and up_proj output as `b`.
__global__ void vib3_silu_mul(
    const half* __restrict__ a,  // SiLU applied to this (should be gate_proj output)
    const half* __restrict__ b,  // multiplied directly (should be up_proj output)
    half* __restrict__ output,
    int dim
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= dim) return;

    float a_val = __half2float(a[i]);
    float b_val = __half2float(b[i]);
    float silu = a_val / (1.0f + expf(-a_val));
    output[i] = __float2half(silu * b_val);
}

// Weighted accumulation: output[i] += weight * expert_output[i]
__global__ void vib3_weighted_accumulate(
    half* __restrict__ output,
    const half* __restrict__ expert_output,
    float weight,
    int dim
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= dim) return;

    float val = __half2float(output[i]) + weight * __half2float(expert_output[i]);
    output[i] = __float2half(val);
}

// FP32 version: accumulate weighted expert output into FP32 buffer
// Eliminates FP16 truncation between expert accumulations
__global__ void vib3_weighted_accumulate_f32(
    float* __restrict__ output,
    const half* __restrict__ expert_output,
    float weight,
    int dim
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= dim) return;

    output[i] += weight * __half2float(expert_output[i]);
}

// FP32-to-FP32 residual add: accumulator[i] += layer_output[i]
// Used when layer_output is already FP32 (from FP32 MoE accumulation)
__global__ void vib3_residual_add_f32_f32_kernel(
    float* __restrict__ accumulator,
    const float* __restrict__ layer_output,
    int dim
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < dim) {
        accumulator[idx] += layer_output[idx];
    }
}

// Router GEMV: scores[e] = hidden[d] · weights[e, d]
__global__ void vib3_router_gemv(
    const half* __restrict__ hidden_state,
    const half* __restrict__ router_weights,
    float* __restrict__ scores,
    int hidden_dim,
    int num_experts
) {
    int expert = blockIdx.x * blockDim.x + threadIdx.x;
    if (expert >= num_experts) return;

    float acc = 0.0f;
    for (int d = 0; d < hidden_dim; d++) {
        acc += __half2float(hidden_state[d]) *
               __half2float(router_weights[expert * hidden_dim + d]);
    }
    scores[expert] = acc;
}

// RMSNorm with weight: x[d] = (x[d] / rms) * weight[d]
__global__ void vib3_rms_norm(
    half* __restrict__ x,
    const half* __restrict__ weight,
    int hidden_dim,
    float eps
) {
    // Single-block kernel: all threads cooperate to compute sum of squares
    extern __shared__ float shared[];

    int tid = threadIdx.x;
    int stride = blockDim.x;

    // Each thread computes partial sum of squares
    float partial_ss = 0.0f;
    for (int d = tid; d < hidden_dim; d += stride) {
        float val = __half2float(x[d]);
        partial_ss += val * val;
    }
    shared[tid] = partial_ss;
    __syncthreads();

    // Reduction to compute total sum of squares
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared[tid] += shared[tid + s];
        }
        __syncthreads();
    }

    float rms = sqrtf(shared[0] / (float)hidden_dim + eps);
    float inv_rms = 1.0f / rms;

    // Normalize and scale
    for (int d = tid; d < hidden_dim; d += stride) {
        float val = __half2float(x[d]) * inv_rms * __half2float(weight[d]);
        x[d] = __float2half(val);
    }
}

// RMSNorm without weight: x[d] = x[d] / rms
__global__ void vib3_rms_norm_no_weight(
    half* __restrict__ x,
    int hidden_dim,
    float eps
) {
    extern __shared__ float shared[];
    int tid = threadIdx.x;
    int stride = blockDim.x;

    float partial_ss = 0.0f;
    for (int d = tid; d < hidden_dim; d += stride) {
        float val = __half2float(x[d]);
        partial_ss += val * val;
    }
    shared[tid] = partial_ss;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) shared[tid] += shared[tid + s];
        __syncthreads();
    }

    float rms = sqrtf(shared[0] / (float)hidden_dim + eps);
    float inv_rms = 1.0f / rms;

    for (int d = tid; d < hidden_dim; d += stride) {
        float val = __half2float(x[d]) * inv_rms;
        x[d] = __float2half(val);
    }
}

// FP32 RMSNorm: reads FP32 input, writes FP32 output, uses FP16 weight
// Eliminates FP16 truncation in the normalization path
__global__ void vib3_rms_norm_f32(
    const float* __restrict__ input,   // FP32 hidden state
    float* __restrict__ output,        // FP32 normalized output
    const half* __restrict__ weight,   // FP16 norm weight
    int hidden_dim,
    float eps
) {
    extern __shared__ float shared[];
    int tid = threadIdx.x;
    int stride = blockDim.x;

    // Compute sum of squares
    float partial_ss = 0.0f;
    for (int d = tid; d < hidden_dim; d += stride) {
        float val = input[d];
        partial_ss += val * val;
    }
    shared[tid] = partial_ss;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) shared[tid] += shared[tid + s];
        __syncthreads();
    }

    float rms = sqrtf(shared[0] / (float)hidden_dim + eps);
    float inv_rms = 1.0f / rms;

    // Normalize and scale
    for (int d = tid; d < hidden_dim; d += stride) {
        output[d] = input[d] * inv_rms * __half2float(weight[d]);
    }
}

// FP32→FP16 RMSNorm: reads FP32 input, normalizes in FP32, writes FP16 output.
// This avoids the catastrophic precision loss from casting FP32 hidden states
// (L2 norm ~3000+) to FP16 BEFORE normalization. Post-norm values are typically
// in [-3, 3] which fits FP16 perfectly.
__global__ void vib3_rms_norm_f32_to_f16(
    const float* __restrict__ input,   // FP32 hidden state (L2 can be very large)
    half* __restrict__ output,         // FP16 normalized output (safe: post-norm values small)
    const half* __restrict__ weight,   // FP16 norm weight
    int hidden_dim,
    float eps
) {
    extern __shared__ float shared[];
    int tid = threadIdx.x;
    int stride = blockDim.x;

    // Compute sum of squares in FP32
    float partial_ss = 0.0f;
    for (int d = tid; d < hidden_dim; d += stride) {
        float val = input[d];
        partial_ss += val * val;
    }
    shared[tid] = partial_ss;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) shared[tid] += shared[tid + s];
        __syncthreads();
    }

    float rms = sqrtf(shared[0] / (float)hidden_dim + eps);
    float inv_rms = 1.0f / rms;

    // Normalize in FP32, scale, then cast to FP16
    for (int d = tid; d < hidden_dim; d += stride) {
        float val = input[d] * inv_rms * __half2float(weight[d]);
        output[d] = __float2half(val);
    }
}

// Embedding lookup: copy row token_id from table to output
__global__ void vib3_embedding_lookup(
    const half* __restrict__ table,
    half* __restrict__ output,
    int token_id,
    int hidden_dim
) {
    int d = blockIdx.x * blockDim.x + threadIdx.x;
    if (d >= hidden_dim) return;
    output[d] = table[token_id * hidden_dim + d];
}

// Residual add: output[d] += residual[d]
__global__ void vib3_residual_add(
    half* __restrict__ output,
    const half* __restrict__ residual,
    int dim
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= dim) return;
    float val = __half2float(output[i]) + __half2float(residual[i]);
    output[i] = __float2half(val);
}


// --- Fused residual add: output = a + b ---
__global__ void vib3_fused_residual_add(half* output, const half* a, const half* b, int dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < dim) {
        output[idx] = __hadd(a[idx], b[idx]);
    }
}

// ─── Host Launcher Functions (callable from Rust via extern "C") ─────────

extern "C" {

int vib3_launch_partial_matmul_fp16(
    const void* input, const void* weight, void* output,
    int K, int M_slice, void* stream
) {
    int blocks = (M_slice + ROWS_PER_BLOCK - 1) / ROWS_PER_BLOCK;
    cudaStream_t s = (cudaStream_t)stream;
    vib3_partial_matmul_fp16<<<blocks, BLOCK_SIZE, 0, s>>>(
        (const half*)input, (const half*)weight, (half*)output, K, M_slice
    );
    return (int)cudaGetLastError();
}

int vib3_launch_partial_matmul_int4(
    const void* input, const void* weight_packed, const void* scales,
    void* output, int K, int M_slice, int group_size, void* stream
) {
    int blocks = (M_slice + ROWS_PER_BLOCK - 1) / ROWS_PER_BLOCK;
    cudaStream_t s = (cudaStream_t)stream;
    vib3_partial_matmul_int4<<<blocks, BLOCK_SIZE, 0, s>>>(
        (const half*)input, (const uint8_t*)weight_packed,
        (const unsigned short*)scales, (half*)output, K, M_slice, group_size
    );
    return (int)cudaGetLastError();
}

int vib3_launch_fused_swiglu_fp16(
    const void* input, const void* up_weight, const void* gate_weight,
    void* output, int K, int M_slice, void* stream
) {
    int blocks = (M_slice + ROWS_PER_BLOCK - 1) / ROWS_PER_BLOCK;
    cudaStream_t s = (cudaStream_t)stream;
    vib3_fused_swiglu_fp16<<<blocks, BLOCK_SIZE, 0, s>>>(
        (const half*)input, (const half*)up_weight, (const half*)gate_weight,
        (half*)output, K, M_slice
    );
    return (int)cudaGetLastError();
}

int vib3_launch_silu_mul(
    const void* up_result, const void* gate_result, void* output,
    int dim, void* stream
) {
    cudaStream_t s = (cudaStream_t)stream;
    int blocks = (dim + 255) / 256;
    vib3_silu_mul<<<blocks, 256, 0, s>>>(
        (const half*)up_result, (const half*)gate_result, (half*)output, dim
    );
    return (int)cudaGetLastError();
}

// FP32 SwiGLU fuse: output[i] = silu(gate[i]) * up[i], all FP32
__global__ void vib3_swiglu_fuse_f32_standalone_kernel(
    const float* __restrict__ gate,
    const float* __restrict__ up,
    float* __restrict__ output,
    int N
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        float g = gate[i];
        float sigmoid_g = 1.0f / (1.0f + expf(-g));
        output[i] = g * sigmoid_g * up[i];
    }
}

int vib3_launch_swiglu_fuse_f32(
    const void* gate, const void* up, void* output,
    int dim, void* stream
) {
    cudaStream_t s = (cudaStream_t)stream;
    int blocks = (dim + 255) / 256;
    vib3_swiglu_fuse_f32_standalone_kernel<<<blocks, 256, 0, s>>>(
        (const float*)gate, (const float*)up, (float*)output, dim
    );
    return (int)cudaGetLastError();
}

int vib3_launch_weighted_accumulate(
    void* output, const void* expert_output, float weight,
    int dim, void* stream
) {
    cudaStream_t s = (cudaStream_t)stream;
    int blocks = (dim + 255) / 256;
    vib3_weighted_accumulate<<<blocks, 256, 0, s>>>(
        (half*)output, (const half*)expert_output, weight, dim
    );
    return (int)cudaGetLastError();
}

int vib3_launch_weighted_accumulate_f32(
    void* output, const void* expert_output, float weight,
    int dim, void* stream
) {
    cudaStream_t s = (cudaStream_t)stream;
    int blocks = (dim + 255) / 256;
    vib3_weighted_accumulate_f32<<<blocks, 256, 0, s>>>(
        (float*)output, (const half*)expert_output, weight, dim
    );
    return (int)cudaGetLastError();
}

int vib3_launch_residual_add_f32_f32(
    void* accumulator, const void* layer_output, int dim, void* stream
) {
    cudaStream_t s = (cudaStream_t)stream;
    int blocks = (dim + 255) / 256;
    vib3_residual_add_f32_f32_kernel<<<blocks, 256, 0, s>>>(
        (float*)accumulator, (const float*)layer_output, dim
    );
    return (int)cudaGetLastError();
}

int vib3_launch_router_gemv(
    const void* hidden_state, const void* router_weights, float* scores,
    int hidden_dim, int num_experts, void* stream
) {
    cudaStream_t s = (cudaStream_t)stream;
    int blocks = (num_experts + 255) / 256;
    vib3_router_gemv<<<blocks, 256, 0, s>>>(
        (const half*)hidden_state, (const half*)router_weights, scores,
        hidden_dim, num_experts
    );
    return (int)cudaGetLastError();
}

int vib3_launch_rms_norm(
    void* x, const void* weight, int hidden_dim, float eps, void* stream
) {
    cudaStream_t s = (cudaStream_t)stream;
    int smem = 256 * sizeof(float);
    vib3_rms_norm<<<1, 256, smem, s>>>(
        (half*)x, (const half*)weight, hidden_dim, eps
    );
    return (int)cudaGetLastError();
}

int vib3_launch_rms_norm_no_weight(
    void* x, int hidden_dim, float eps, void* stream
) {
    cudaStream_t s = (cudaStream_t)stream;
    int smem = 256 * sizeof(float);
    vib3_rms_norm_no_weight<<<1, 256, smem, s>>>(
        (half*)x, hidden_dim, eps
    );
    return (int)cudaGetLastError();
}

int vib3_launch_rms_norm_f32(
    const void* input, void* output, const void* weight,
    int hidden_dim, float eps, void* stream
) {
    cudaStream_t s = (cudaStream_t)stream;
    int smem = 256 * sizeof(float);
    vib3_rms_norm_f32<<<1, 256, smem, s>>>(
        (const float*)input, (float*)output, (const half*)weight, hidden_dim, eps
    );
    return (int)cudaGetLastError();
}

int vib3_launch_rms_norm_f32_to_f16(
    const void* input, void* output, const void* weight,
    int hidden_dim, float eps, void* stream
) {
    cudaStream_t s = (cudaStream_t)stream;
    int smem = 256 * sizeof(float);
    vib3_rms_norm_f32_to_f16<<<1, 256, smem, s>>>(
        (const float*)input, (half*)output, (const half*)weight, hidden_dim, eps
    );
    return (int)cudaGetLastError();
}

int vib3_launch_partial_matmul_int4_f32(
    const void* input, const void* weight, const void* scales,
    void* output, int K, int M_slice, int group_size, void* stream
) {
    cudaStream_t s = (cudaStream_t)stream;
    int blocks = (M_slice + ROWS_PER_BLOCK - 1) / ROWS_PER_BLOCK;
    vib3_partial_matmul_int4_f32<<<blocks, THREADS_PER_ROW * ROWS_PER_BLOCK, 0, s>>>(
        (const float*)input, (const uint8_t*)weight,
        (const unsigned short*)scales, (half*)output,
        K, M_slice, group_size
    );
    return (int)cudaGetLastError();
}

int vib3_launch_partial_matmul_fp16_f32in(
    const void* input, const void* weight, void* output,
    int K, int M_slice, void* stream
) {
    cudaStream_t s = (cudaStream_t)stream;
    int blocks = (M_slice + ROWS_PER_BLOCK - 1) / ROWS_PER_BLOCK;
    vib3_partial_matmul_fp16_f32in<<<blocks, THREADS_PER_ROW * ROWS_PER_BLOCK, 0, s>>>(
        (const float*)input, (const half*)weight, (half*)output,
        K, M_slice
    );
    return (int)cudaGetLastError();
}

// FP32-input FP16-weight matmul with FP32 output
__global__ void vib3_partial_matmul_fp16_f32in_f32out(
    const float* __restrict__ input,
    const half* __restrict__ weight,
    float* __restrict__ output,
    int K,
    int M_slice
) {
    const int local_row = threadIdx.x / THREADS_PER_ROW;
    const int lane = threadIdx.x % THREADS_PER_ROW;
    const int row = blockIdx.x * ROWS_PER_BLOCK + local_row;
    if (row >= M_slice) return;
    const half* row_w = weight + (long long)row * K;
    float acc = 0.0f;
    for (int k = lane; k < K; k += THREADS_PER_ROW) {
        acc += input[k] * __half2float(row_w[k]);
    }
    acc = warp_reduce_sum(acc);
    __shared__ float smem_fp32out[ROWS_PER_BLOCK * (THREADS_PER_ROW / WARP_SIZE)];
    const int warp_id = lane / WARP_SIZE;
    const int warp_lane = lane % WARP_SIZE;
    if (warp_lane == 0) {
        smem_fp32out[local_row * (THREADS_PER_ROW / WARP_SIZE) + warp_id] = acc;
    }
    __syncthreads();
    if (warp_id == 0) {
        float final_acc = (warp_lane < (THREADS_PER_ROW / WARP_SIZE))
            ? smem_fp32out[local_row * (THREADS_PER_ROW / WARP_SIZE) + warp_lane]
            : 0.0f;
        final_acc = warp_reduce_sum(final_acc);
        if (warp_lane == 0) {
            output[row] = final_acc;
        }
    }
}

int vib3_launch_partial_matmul_fp16_f32in_f32out(
    const void* input, const void* weight, void* output,
    int K, int M_slice, void* stream
) {
    cudaStream_t s = (cudaStream_t)stream;
    int blocks = (M_slice + ROWS_PER_BLOCK - 1) / ROWS_PER_BLOCK;
    vib3_partial_matmul_fp16_f32in_f32out<<<blocks, THREADS_PER_ROW * ROWS_PER_BLOCK, 0, s>>>(
        (const float*)input, (const half*)weight, (float*)output,
        K, M_slice
    );
    return (int)cudaGetLastError();
}

int vib3_launch_fused_swiglu_int4_f32(
    const void* input, const void* up_weight, const void* up_scales,
    const void* gate_weight, const void* gate_scales,
    void* output, int K, int M_slice, int group_size, void* stream
) {
    cudaStream_t s = (cudaStream_t)stream;
    int blocks = (M_slice + ROWS_PER_BLOCK - 1) / ROWS_PER_BLOCK;
    vib3_fused_swiglu_int4_f32<<<blocks, THREADS_PER_ROW * ROWS_PER_BLOCK, 0, s>>>(
        (const float*)input,
        (const uint8_t*)up_weight, (const unsigned short*)up_scales,
        (const uint8_t*)gate_weight, (const unsigned short*)gate_scales,
        (half*)output, K, M_slice, group_size
    );
    return (int)cudaGetLastError();
}

// ── Fast vectorized FP16 SwiGLU for shared expert ──
// Uses float4 loads (8 halfs per load) and shared memory for input caching.
// 4 rows per block × 64 threads per row = 256 threads.
// For K=3072: each thread processes 48 elements via 6 float4 loads per weight matrix.
// FAST_SWIGLU_ROWS/TPR/BLK defined at top of file.

__global__ void vib3_fused_swiglu_fp16_f32in_fast(
    const float* __restrict__ input,
    const half* __restrict__ up_weight,
    const half* __restrict__ gate_weight,
    half* __restrict__ output,
    int K, int M_slice
) {
    const int local_row = threadIdx.x / FAST_SWIGLU_TPR;
    const int lane = threadIdx.x % FAST_SWIGLU_TPR;
    const int row = blockIdx.x * FAST_SWIGLU_ROWS + local_row;
    const bool row_valid = (row < M_slice);

    // Shared memory layout: [K halfs for input cache] [16 floats for reduction]
    extern __shared__ char shared_bytes[];
    half* smem_input = reinterpret_cast<half*>(shared_bytes);
    float* reduce_smem = reinterpret_cast<float*>(shared_bytes + K * sizeof(half));

    // Cache input in shared memory (FP32 → FP16 to halve smem usage)
    for (int k = threadIdx.x; k < K; k += FAST_SWIGLU_BLK) {
        smem_input[k] = __float2half(input[k]);
    }
    __syncthreads();

    float up_acc = 0.0f;
    float gate_acc = 0.0f;

    if (row_valid) {
        const half* up_ptr = up_weight + (long long)row * K;
        const half* gate_ptr = gate_weight + (long long)row * K;

        // Coalesced vectorized loop
        for (int k = lane * 8; k < K; k += FAST_SWIGLU_TPR * 8) {
            float4 up_v = *reinterpret_cast<const float4*>(up_ptr + k);
            float4 gate_v = *reinterpret_cast<const float4*>(gate_ptr + k);
            float4 inp_v = *reinterpret_cast<const float4*>(smem_input + k);

            half2* up_h = reinterpret_cast<half2*>(&up_v);
            half2* g_h = reinterpret_cast<half2*>(&gate_v);
            half2* i_h = reinterpret_cast<half2*>(&inp_v);

            #pragma unroll
            for (int p = 0; p < 4; p++) {
                up_acc += __half2float(up_h[p].x) * __half2float(i_h[p].x)
                        + __half2float(up_h[p].y) * __half2float(i_h[p].y);
                gate_acc += __half2float(g_h[p].x) * __half2float(i_h[p].x)
                          + __half2float(g_h[p].y) * __half2float(i_h[p].y);
            }
        }
    }

    // Warp reduction (64 threads per row = 2 warps)
    up_acc = warp_reduce_sum(up_acc);
    gate_acc = warp_reduce_sum(gate_acc);

    // Inter-warp reduction
    const int warp_id = lane / WARP_SIZE;
    const int warp_lane = lane % WARP_SIZE;

    if (warp_lane == 0) {
        reduce_smem[(local_row * 2 + warp_id) * 2 + 0] = up_acc;
        reduce_smem[(local_row * 2 + warp_id) * 2 + 1] = gate_acc;
    }
    __syncthreads();

    if (row_valid && warp_id == 0 && warp_lane == 0) {
        float final_up = reduce_smem[local_row * 4 + 0] + reduce_smem[local_row * 4 + 2];
        float final_gate = reduce_smem[local_row * 4 + 1] + reduce_smem[local_row * 4 + 3];
        float silu = final_gate / (1.0f + expf(-final_gate));
        output[row] = __float2half(silu * final_up);
    }
}

// ── Fast vectorized FP16 GEMV for shared expert down_proj ──
// float4 loads for weight, smem-cached FP16 input.
// 4 rows per block × 64 threads per row = 256 threads.
__global__ void vib3_partial_matmul_fp16_fast(
    const half* __restrict__ input,
    const half* __restrict__ weight,
    half* __restrict__ output,
    int K, int M_slice
) {
    const int local_row = threadIdx.x / FAST_SWIGLU_TPR;
    const int lane = threadIdx.x % FAST_SWIGLU_TPR;
    const int row = blockIdx.x * FAST_SWIGLU_ROWS + local_row;
    const bool row_valid = (row < M_slice);

    // Shared memory: [K halfs for input cache] [8 floats for reduction]
    extern __shared__ char shared_bytes2[];
    half* smem_in = reinterpret_cast<half*>(shared_bytes2);
    float* reduce_smem2 = reinterpret_cast<float*>(shared_bytes2 + K * sizeof(half));

    for (int k = threadIdx.x; k < K; k += FAST_SWIGLU_BLK) {
        smem_in[k] = input[k];
    }
    __syncthreads();

    float acc = 0.0f;

    if (row_valid) {
        const half* row_ptr = weight + (long long)row * K;

        // Coalesced vectorized loop
        for (int k = lane * 8; k < K; k += FAST_SWIGLU_TPR * 8) {
            float4 w_v = *reinterpret_cast<const float4*>(row_ptr + k);
            float4 i_v = *reinterpret_cast<const float4*>(smem_in + k);

            half2* w_h = reinterpret_cast<half2*>(&w_v);
            half2* i_h = reinterpret_cast<half2*>(&i_v);

            #pragma unroll
            for (int p = 0; p < 4; p++) {
                acc += __half2float(w_h[p].x) * __half2float(i_h[p].x)
                     + __half2float(w_h[p].y) * __half2float(i_h[p].y);
            }
        }
    }

    acc = warp_reduce_sum(acc);

    const int warp_id = lane / WARP_SIZE;
    const int warp_lane = lane % WARP_SIZE;

    if (warp_lane == 0) {
        reduce_smem2[local_row * 2 + warp_id] = acc;
    }
    __syncthreads();

    if (row_valid && warp_id == 0 && warp_lane == 0) {
        float final_acc = reduce_smem2[local_row * 2 + 0] + reduce_smem2[local_row * 2 + 1];
        output[row] = __float2half(final_acc);
    }
}

int vib3_launch_fused_swiglu_fp16_f32in(
    const void* input, const void* up_weight, const void* gate_weight,
    void* output, int K, int M_slice, void* stream
) {
    cudaStream_t s = (cudaStream_t)stream;
    // Use fast kernel when K is a multiple of 512 (FAST_SWIGLU_TPR * 8)
    if (K % (FAST_SWIGLU_TPR * 8) == 0) {
        int blocks = (M_slice + FAST_SWIGLU_ROWS - 1) / FAST_SWIGLU_ROWS;
        // smem: K halfs for input + 16 floats for reduction (4 rows × 2 warps × 2 vals)
        int smem_bytes = K * (int)sizeof(half) + FAST_SWIGLU_ROWS * 2 * 2 * (int)sizeof(float);
        vib3_fused_swiglu_fp16_f32in_fast<<<blocks, FAST_SWIGLU_BLK, smem_bytes, s>>>(
            (const float*)input, (const half*)up_weight, (const half*)gate_weight,
            (half*)output, K, M_slice
        );
    } else {
        int blocks = (M_slice + ROWS_PER_BLOCK - 1) / ROWS_PER_BLOCK;
        vib3_fused_swiglu_fp16_f32in<<<blocks, THREADS_PER_ROW * ROWS_PER_BLOCK, 0, s>>>(
            (const float*)input, (const half*)up_weight, (const half*)gate_weight,
            (half*)output, K, M_slice
        );
    }
    return (int)cudaGetLastError();
}

int vib3_launch_partial_matmul_fp16_fast(
    const void* input, const void* weight, void* output,
    int K, int M_slice, void* stream
) {
    cudaStream_t s = (cudaStream_t)stream;
    if (K % (FAST_SWIGLU_TPR * 8) == 0 && K >= 512) {
        int blocks = (M_slice + FAST_SWIGLU_ROWS - 1) / FAST_SWIGLU_ROWS;
        int smem_bytes = K * (int)sizeof(half) + FAST_SWIGLU_ROWS * 2 * (int)sizeof(float);
        vib3_partial_matmul_fp16_fast<<<blocks, FAST_SWIGLU_BLK, smem_bytes, s>>>(
            (const half*)input, (const half*)weight, (half*)output, K, M_slice
        );
    } else {
        int blocks = (M_slice + ROWS_PER_BLOCK - 1) / ROWS_PER_BLOCK;
        vib3_partial_matmul_fp16<<<blocks, BLOCK_SIZE, 0, s>>>(
            (const half*)input, (const half*)weight, (half*)output, K, M_slice
        );
    }
    return (int)cudaGetLastError();
}

int vib3_launch_router_gemv_f32(
    const void* hidden_state, const void* router_weights, float* scores,
    int hidden_dim, int num_experts, void* stream
) {
    cudaStream_t s = (cudaStream_t)stream;
    // Tiled: ROWS_PER_BLOCK experts per block, THREADS_PER_ROW threads per expert
    int blocks = (num_experts + ROWS_PER_BLOCK - 1) / ROWS_PER_BLOCK;
    vib3_router_gemv_f32<<<blocks, BLOCK_SIZE, 0, s>>>(
        (const float*)hidden_state, (const half*)router_weights, scores,
        hidden_dim, num_experts
    );
    return (int)cudaGetLastError();
}

int vib3_launch_embedding_lookup(
    const void* table, void* output, int token_id, int hidden_dim, void* stream
) {
    cudaStream_t s = (cudaStream_t)stream;
    int blocks = (hidden_dim + 255) / 256;
    vib3_embedding_lookup<<<blocks, 256, 0, s>>>(
        (const half*)table, (half*)output, token_id, hidden_dim
    );
    return (int)cudaGetLastError();
}

int vib3_launch_residual_add(
    void* output, const void* residual, int dim, void* stream
) {
    cudaStream_t s = (cudaStream_t)stream;
    int blocks = (dim + 255) / 256;
    vib3_residual_add<<<blocks, 256, 0, s>>>(
        (half*)output, (const half*)residual, dim
    );
    return (int)cudaGetLastError();
}

int vib3_launch_fused_residual_add(
    void* output, const void* a, const void* b, int dim, void* stream
) {
    cudaStream_t s = (cudaStream_t)stream;
    int blocks = (dim + 255) / 256;
    vib3_fused_residual_add<<<blocks, 256, 0, s>>>(
        (half*)output, (const half*)a, (const half*)b, dim
    );
    return (int)cudaGetLastError();
}

int vib3_launch_partial_matmul_nvfp4(
    const void* input, const void* weight_packed, const void* scales,
    void* output, int K, int M_slice, int block_size, void* stream
) {
    cudaStream_t s = (cudaStream_t)stream;
    int blocks = (M_slice + ROWS_PER_BLOCK - 1) / ROWS_PER_BLOCK;
    vib3_partial_matmul_nvfp4<<<blocks, ROWS_PER_BLOCK * THREADS_PER_ROW, 0, s>>>(
        (const float*)input, (const uint8_t*)weight_packed,
        (const unsigned short*)scales, (half*)output, K, M_slice, block_size
    );
    return (int)cudaGetLastError();
}

int vib3_launch_fused_swiglu_nvfp4(
    const void* input, const void* up_weight, const void* up_scales,
    const void* gate_weight, const void* gate_scales,
    void* output, int K, int M_slice, int block_size, void* stream
) {
    cudaStream_t s = (cudaStream_t)stream;
    int blocks = (M_slice + ROWS_PER_BLOCK - 1) / ROWS_PER_BLOCK;
    vib3_fused_swiglu_nvfp4<<<blocks, ROWS_PER_BLOCK * THREADS_PER_ROW, 0, s>>>(
        (const float*)input, (const uint8_t*)up_weight, (const unsigned short*)up_scales,
        (const uint8_t*)gate_weight, (const unsigned short*)gate_scales,
        (half*)output, K, M_slice, block_size
    );
    return (int)cudaGetLastError();
}

int vib3_launch_partial_matmul_nvfp4_fp16in(
    const void* input, const void* weight_packed, const void* scales,
    void* output, int K, int M_slice, int block_size, void* stream
) {
    cudaStream_t s = (cudaStream_t)stream;
    int blocks = (M_slice + ROWS_PER_BLOCK - 1) / ROWS_PER_BLOCK;
    vib3_partial_matmul_nvfp4_fp16in<<<blocks, ROWS_PER_BLOCK * THREADS_PER_ROW, 0, s>>>(
        (const half*)input, (const uint8_t*)weight_packed,
        (const unsigned short*)scales, (half*)output, K, M_slice, block_size
    );
    return (int)cudaGetLastError();
}

} // extern "C" (original launchers)

// ─── MLA Decode Attention Kernel ─────────────────────────────────────────
#define MLA_ATTN_BLOCK 256

__global__ void vib3_mla_decode_attn_kernel(
    const float* __restrict__ q_absorbed,
    const float* __restrict__ q_rope,
    const float* __restrict__ kv_latent,
    const float* __restrict__ k_rope_cache,
    float* __restrict__ v_latent_out,
    int kv_lora_rank, int qk_rope_dim, int num_heads, int seq_len, float scale
) {
    int head = blockIdx.x;
    if (head >= num_heads) return;
    int tid = threadIdx.x;
    const float* q_abs_h = q_absorbed + head * kv_lora_rank;
    const float* q_rope_h = q_rope + head * qk_rope_dim;
    extern __shared__ float smem_mla[];
    float* scores = smem_mla;
    for (int pos = 0; pos < seq_len; pos++) {
        const float* lat = kv_latent + pos * kv_lora_rank;
        const float* kr = k_rope_cache + pos * qk_rope_dim;
        float partial = 0.0f;
        for (int j = tid; j < kv_lora_rank; j += MLA_ATTN_BLOCK) partial += q_abs_h[j] * lat[j];
        for (int j = tid; j < qk_rope_dim; j += MLA_ATTN_BLOCK) partial += q_rope_h[j] * kr[j];
        for (int offset = 16; offset > 0; offset >>= 1)
            partial += __shfl_down_sync(0xFFFFFFFF, partial, offset);
        __shared__ float warp_sums[8];
        int warp_id = tid / 32, lane = tid % 32;
        if (lane == 0) warp_sums[warp_id] = partial;
        __syncthreads();
        if (tid == 0) {
            float total = 0.0f;
            for (int w = 0; w < MLA_ATTN_BLOCK / 32; w++) total += warp_sums[w];
            scores[pos] = total * scale;
        }
        __syncthreads();
    }
    // Broadcast global_max via a DEDICATED shared slot, not smem_mla[0]:
    // `scores == smem_mla`, so writing smem_mla[0] would clobber scores[0]
    // and cause softmax to compute expf(0)=1.0 for position 0 regardless of
    // its actual score — over-weighting position 0 at all positions where
    // seq_len > 1.
    __shared__ float gmax_shared;
    if (tid == 0) {
        float m = -1e30f;
        for (int p = 0; p < seq_len; p++) {
            if (scores[p] > m) m = scores[p];
        }
        gmax_shared = m;
    }
    __syncthreads();
    float global_max = gmax_shared;
    float total_exp = 0.0f;
    if (tid == 0) {
        for (int p = 0; p < seq_len; p++) { float e = expf(scores[p] - global_max); scores[p] = e; total_exp += e; }
        float inv = (total_exp > 0.0f) ? (1.0f / total_exp) : 0.0f;
        for (int p = 0; p < seq_len; p++) scores[p] *= inv;
    }
    __syncthreads();
    float* out_h = v_latent_out + head * kv_lora_rank;
    for (int j = tid; j < kv_lora_rank; j += MLA_ATTN_BLOCK) {
        float acc = 0.0f;
        for (int pos = 0; pos < seq_len; pos++) acc += scores[pos] * kv_latent[pos * kv_lora_rank + j];
        out_h[j] = acc;
    }
}

__global__ void vib3_mla_v_reconstruct(
    const float* __restrict__ v_latent, const float* __restrict__ kv_b_proj,
    float* __restrict__ v_out, int kv_lora_rank, int qk_nope_dim, int v_head_dim, int num_heads
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = num_heads * v_head_dim;
    if (idx >= total) return;
    int h = idx / v_head_dim, d = idx % v_head_dim;
    int row = h * (qk_nope_dim + v_head_dim) + qk_nope_dim + d;
    const float* kv_row = kv_b_proj + row * kv_lora_rank;
    const float* v_lat_h = v_latent + h * kv_lora_rank;
    float acc = 0.0f;
    for (int j = 0; j < kv_lora_rank; j++) acc += kv_row[j] * v_lat_h[j];
    v_out[idx] = acc;
}

extern "C" {

int vib3_launch_mla_decode_attention(
    const void* q_absorbed, const void* q_rope,
    const void* kv_latent, const void* k_rope_cache,
    void* v_latent_out, int kv_lora_rank, int qk_rope_dim,
    int num_heads, int seq_len, float scale, void* stream
) {
    cudaStream_t s = (cudaStream_t)stream;
    int smem_bytes = seq_len * sizeof(float);
    if (smem_bytes < 32) smem_bytes = 32;
    vib3_mla_decode_attn_kernel<<<num_heads, MLA_ATTN_BLOCK, smem_bytes, s>>>(
        (const float*)q_absorbed, (const float*)q_rope,
        (const float*)kv_latent, (const float*)k_rope_cache,
        (float*)v_latent_out, kv_lora_rank, qk_rope_dim, num_heads, seq_len, scale
    );
    return (int)cudaGetLastError();
}

int vib3_launch_mla_v_reconstruct(
    const void* v_latent, const void* kv_b_proj, void* v_out,
    int kv_lora_rank, int qk_nope_dim, int v_head_dim, int num_heads, void* stream
) {
    cudaStream_t s = (cudaStream_t)stream;
    int total = num_heads * v_head_dim;
    int blocks = (total + 255) / 256;
    vib3_mla_v_reconstruct<<<blocks, 256, 0, s>>>(
        (const float*)v_latent, (const float*)kv_b_proj, (float*)v_out,
        kv_lora_rank, qk_nope_dim, v_head_dim, num_heads
    );
    return (int)cudaGetLastError();
}

} // extern "C" (MLA launchers)

// ─── MLA Q Absorption + RoPE + KV Cache Append ──────────────────────────

#define Q_ABSORB_BLOCK 256

__global__ void vib3_mla_q_absorb_rope_kernel(
    const half* __restrict__ q_full, const float* __restrict__ kv_b_proj,
    const float* __restrict__ rope_freqs,
    float* __restrict__ q_absorbed_out, float* __restrict__ q_rope_out,
    int q_head_dim, int qk_nope_dim, int qk_rope_dim, int v_head_dim,
    int kv_lora_rank, int num_heads, int position
) {
    int head = blockIdx.x;
    if (head >= num_heads) return;
    int tid = threadIdx.x;
    int head_row_offset = head * (qk_nope_dim + v_head_dim);
    const half* q_nope = q_full + head * q_head_dim;
    for (int j = tid; j < kv_lora_rank; j += Q_ABSORB_BLOCK) {
        float acc = 0.0f;
        for (int i = 0; i < qk_nope_dim; i++)
            acc += __half2float(q_nope[i]) * kv_b_proj[(head_row_offset + i) * kv_lora_rank + j];
        q_absorbed_out[head * kv_lora_rank + j] = acc;
    }
    const half* q_rope_src = q_full + head * q_head_dim + qk_nope_dim;
    int half_rope = qk_rope_dim / 2;
    for (int i = tid; i < half_rope; i += Q_ABSORB_BLOCK) {
        float freq = rope_freqs[i];
        float theta = (float)position * freq;
        float cos_t = cosf(theta), sin_t = sinf(theta);
        float x0 = __half2float(q_rope_src[2*i]), x1 = __half2float(q_rope_src[2*i+1]);
        q_rope_out[head * qk_rope_dim + 2*i]   = x0 * cos_t - x1 * sin_t;
        q_rope_out[head * qk_rope_dim + 2*i+1] = x0 * sin_t + x1 * cos_t;
    }
}

__global__ void vib3_mla_kv_cache_append_kernel(
    const half* __restrict__ kv_a_out, const half* __restrict__ kv_norm_weight,
    const float* __restrict__ rope_freqs,
    float* __restrict__ kv_latent_cache, float* __restrict__ k_rope_cache,
    int kv_lora_rank, int qk_rope_dim, int position, float eps
) {
    extern __shared__ float shared_kv[];
    int tid = threadIdx.x, stride = blockDim.x;
    float partial_ss = 0.0f;
    for (int d = tid; d < kv_lora_rank; d += stride) {
        float val = __half2float(kv_a_out[d]);
        shared_kv[d] = val;
        partial_ss += val * val;
    }
    __shared__ float ss_reduce[256];
    ss_reduce[tid] = partial_ss;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) { if (tid < s) ss_reduce[tid] += ss_reduce[tid + s]; __syncthreads(); }
    float inv_rms = 1.0f / sqrtf(ss_reduce[0] / (float)kv_lora_rank + eps);
    float* latent_dst = kv_latent_cache + position * kv_lora_rank;
    for (int d = tid; d < kv_lora_rank; d += stride) {
        float val = shared_kv[d] * inv_rms;
        if (kv_norm_weight != nullptr) val *= __half2float(kv_norm_weight[d]);
        latent_dst[d] = val;
    }
    int half_rope = qk_rope_dim / 2;
    const half* k_rope_src = kv_a_out + kv_lora_rank;
    float* rope_dst = k_rope_cache + position * qk_rope_dim;
    for (int i = tid; i < half_rope; i += stride) {
        float freq = rope_freqs[i];
        float theta = (float)position * freq;
        float cos_t = cosf(theta), sin_t = sinf(theta);
        float x0 = __half2float(k_rope_src[2*i]), x1 = __half2float(k_rope_src[2*i+1]);
        rope_dst[2*i]   = x0 * cos_t - x1 * sin_t;
        rope_dst[2*i+1] = x0 * sin_t + x1 * cos_t;
    }
}

// ─── Type Conversion Kernels ────────────────────────────────────────────

__global__ void vib3_f32_to_f16_kernel(const float* __restrict__ input, half* __restrict__ output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) output[idx] = __float2half(input[idx]);
}

__global__ void vib3_fp16_to_fp32_kernel(const half* __restrict__ input, float* __restrict__ output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) output[idx] = __half2float(input[idx]);
}

__global__ void vib3_residual_add_fp32_kernel(float* __restrict__ accumulator, const half* __restrict__ layer_output, int dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < dim) accumulator[idx] += __half2float(layer_output[idx]);
}

extern "C" {

int vib3_launch_mla_q_absorb_rope(
    const void* q_full, const void* kv_b_proj, const void* rope_freqs,
    void* q_absorbed_out, void* q_rope_out,
    int q_head_dim, int qk_nope_dim, int qk_rope_dim, int v_head_dim,
    int kv_lora_rank, int num_heads, int position, void* stream
) {
    cudaStream_t s = (cudaStream_t)stream;
    vib3_mla_q_absorb_rope_kernel<<<num_heads, Q_ABSORB_BLOCK, 0, s>>>(
        (const half*)q_full, (const float*)kv_b_proj, (const float*)rope_freqs,
        (float*)q_absorbed_out, (float*)q_rope_out,
        q_head_dim, qk_nope_dim, qk_rope_dim, v_head_dim, kv_lora_rank, num_heads, position
    );
    return (int)cudaGetLastError();
}

int vib3_launch_mla_kv_cache_append(
    const void* kv_a_out, const void* kv_norm_weight, const void* rope_freqs,
    void* kv_latent_cache, void* k_rope_cache,
    int kv_lora_rank, int qk_rope_dim, int position, float eps, void* stream
) {
    cudaStream_t s = (cudaStream_t)stream;
    int smem_bytes = kv_lora_rank * sizeof(float);
    vib3_mla_kv_cache_append_kernel<<<1, 256, smem_bytes, s>>>(
        (const half*)kv_a_out, (const half*)kv_norm_weight, (const float*)rope_freqs,
        (float*)kv_latent_cache, (float*)k_rope_cache, kv_lora_rank, qk_rope_dim, position, eps
    );
    return (int)cudaGetLastError();
}

int vib3_launch_f32_to_f16(const void* input, void* output, int n, void* stream) {
    cudaStream_t s = (cudaStream_t)stream;
    vib3_f32_to_f16_kernel<<<(n+255)/256, 256, 0, s>>>((const float*)input, (half*)output, n);
    return (int)cudaGetLastError();
}

int vib3_launch_fp16_to_fp32(const void* input, void* output, int n, void* stream) {
    cudaStream_t s = (cudaStream_t)stream;
    vib3_fp16_to_fp32_kernel<<<(n+255)/256, 256, 0, s>>>((const half*)input, (float*)output, n);
    return (int)cudaGetLastError();
}

int vib3_launch_residual_add_fp32(void* accumulator, const void* layer_output, int dim, void* stream) {
    cudaStream_t s = (cudaStream_t)stream;
    vib3_residual_add_fp32_kernel<<<(dim+255)/256, 256, 0, s>>>((float*)accumulator, (const half*)layer_output, dim);
    return (int)cudaGetLastError();
}

} // extern "C" (MLA + conversion launchers)

// ─── Decode Attention Kernels ────────────────────────────────────────────

__global__ void vib3_rope_kernel(
    half* __restrict__ data, int head_dim, int total_heads, const int* __restrict__ d_position, float rope_base
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int half_dim = head_dim / 2;
    int total_pairs = total_heads * half_dim;
    if (idx >= total_pairs) return;
    int position = *d_position;
    int head = idx / half_dim, i = idx % half_dim;
    float freq = 1.0f / powf(rope_base, 2.0f * (float)i / (float)head_dim);
    float theta = (float)position * freq;
    float cos_t = cosf(theta), sin_t = sinf(theta);
    int offset = head * head_dim + 2 * i;
    float x0 = __half2float(data[offset]), x1 = __half2float(data[offset + 1]);
    data[offset]     = __float2half(x0 * cos_t - x1 * sin_t);
    data[offset + 1] = __float2half(x0 * sin_t + x1 * cos_t);
}

__global__ void vib3_kv_append_kernel(
    half* __restrict__ k_cache, half* __restrict__ v_cache,
    const half* __restrict__ new_k, const half* __restrict__ new_v,
    int max_seq_len, int head_dim, int num_kv_heads, const int* __restrict__ d_position
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = num_kv_heads * head_dim;
    if (idx >= total) return;
    int position = *d_position;
    int h = idx / head_dim, d = idx % head_dim;
    int cache_offset = h * max_seq_len * head_dim + position * head_dim + d;
    k_cache[cache_offset] = new_k[idx];
    v_cache[cache_offset] = new_v[idx];
}

#define ATTN_BLOCK_SIZE 256

__global__ void vib3_decode_attn_kernel(
    const half* __restrict__ q, const half* __restrict__ k_cache,
    const half* __restrict__ v_cache, half* __restrict__ output,
    int head_dim, int num_heads, int num_kv_heads, const int* __restrict__ d_position, int max_seq_len, float scale
) {
    int head = blockIdx.x;
    if (head >= num_heads) return;
    int seq_len = *d_position + 1;
    int tid = threadIdx.x;
    int heads_per_kv = num_heads / max(num_kv_heads, 1);
    int kv_head = head / max(heads_per_kv, 1);
    if (kv_head >= num_kv_heads) kv_head = num_kv_heads - 1;
    const half* q_head = q + head * head_dim;
    const half* k_head_cache = k_cache + kv_head * max_seq_len * head_dim;
    const half* v_head_cache = v_cache + kv_head * max_seq_len * head_dim;
    float local_max = -1e30f;
    for (int pos = tid; pos < seq_len; pos += ATTN_BLOCK_SIZE) {
        float score = 0.0f;
        const half* k_pos = k_head_cache + pos * head_dim;
        for (int d = 0; d < head_dim; d++) score += __half2float(q_head[d]) * __half2float(k_pos[d]);
        score *= scale;
        if (score > local_max) local_max = score;
    }
    __shared__ float smem_max[ATTN_BLOCK_SIZE];
    smem_max[tid] = local_max;
    __syncthreads();
    for (int s = ATTN_BLOCK_SIZE / 2; s > 0; s >>= 1) {
        if (tid < s && smem_max[tid + s] > smem_max[tid]) smem_max[tid] = smem_max[tid + s];
        __syncthreads();
    }
    float global_max = smem_max[0];
    extern __shared__ float smem_v[];
    for (int d = tid; d < head_dim; d += ATTN_BLOCK_SIZE) smem_v[d] = 0.0f;
    __syncthreads();
    float local_exp_sum = 0.0f;
    for (int pos = tid; pos < seq_len; pos += ATTN_BLOCK_SIZE) {
        float score = 0.0f;
        const half* k_pos = k_head_cache + pos * head_dim;
        for (int d = 0; d < head_dim; d++) score += __half2float(q_head[d]) * __half2float(k_pos[d]);
        score = expf(score * scale - global_max);
        local_exp_sum += score;
        const half* v_pos = v_head_cache + pos * head_dim;
        for (int d = 0; d < head_dim; d++) atomicAdd(&smem_v[d], score * __half2float(v_pos[d]));
    }
    __shared__ float smem_exp[ATTN_BLOCK_SIZE];
    smem_exp[tid] = local_exp_sum;
    __syncthreads();
    for (int s = ATTN_BLOCK_SIZE / 2; s > 0; s >>= 1) { if (tid < s) smem_exp[tid] += smem_exp[tid + s]; __syncthreads(); }
    float inv_sum = (smem_exp[0] > 0.0f) ? (1.0f / smem_exp[0]) : 0.0f;
    for (int d = tid; d < head_dim; d += ATTN_BLOCK_SIZE)
        output[head * head_dim + d] = __float2half(smem_v[d] * inv_sum);
}

extern "C" {

int vib3_launch_rope_apply(
    void* q, void* k, int head_dim, int num_q_heads, int num_kv_heads,
    const void* d_position, float rope_base, void* stream
) {
    cudaStream_t s = (cudaStream_t)stream;
    const int* dp = (const int*)d_position;
    int q_pairs = num_q_heads * (head_dim / 2);
    vib3_rope_kernel<<<(q_pairs+255)/256, 256, 0, s>>>((half*)q, head_dim, num_q_heads, dp, rope_base);
    int k_pairs = num_kv_heads * (head_dim / 2);
    vib3_rope_kernel<<<(k_pairs+255)/256, 256, 0, s>>>((half*)k, head_dim, num_kv_heads, dp, rope_base);
    return (int)cudaGetLastError();
}

int vib3_launch_kv_cache_append(
    void* k_cache, void* v_cache, const void* new_k, const void* new_v,
    int max_seq_len, int head_dim, int num_kv_heads, const void* d_position, void* stream
) {
    cudaStream_t s = (cudaStream_t)stream;
    int total = num_kv_heads * head_dim;
    vib3_kv_append_kernel<<<(total+255)/256, 256, 0, s>>>(
        (half*)k_cache, (half*)v_cache, (const half*)new_k, (const half*)new_v,
        max_seq_len, head_dim, num_kv_heads, (const int*)d_position
    );
    return (int)cudaGetLastError();
}

int vib3_launch_decode_attention(
    const void* q, const void* k_cache, const void* v_cache, void* output,
    int head_dim, int num_heads, int num_kv_heads,
    const void* d_position, int max_seq_len, float scale, void* stream
) {
    cudaStream_t s = (cudaStream_t)stream;
    int smem_bytes = head_dim * sizeof(float);
    vib3_decode_attn_kernel<<<num_heads, ATTN_BLOCK_SIZE, smem_bytes, s>>>(
        (const half*)q, (const half*)k_cache, (const half*)v_cache, (half*)output,
        head_dim, num_heads, num_kv_heads, (const int*)d_position, max_seq_len, scale
    );
    return (int)cudaGetLastError();
}

} // extern "C" (decode attention launchers)

// ═══════════════════════════════════════════════════════════════════════════
// Reconstructed Kernels: F32-output NVFP4, DeltaNet, Utilities, MMA
// ═══════════════════════════════════════════════════════════════════════════

// ─── NVFP4 F32-output Kernels ──────────────────────────────────────────────

// Fused SwiGLU NVFP4 with FP32 output (eliminates FP16 intermediate truncation)
// Note: scales are BF16 (from E8M0→BF16 conversion), NOT FP16
__global__ void vib3_fused_swiglu_nvfp4_f32out_kernel(
    const float* __restrict__ input,
    const uint8_t* __restrict__ up_weight,
    const unsigned short* __restrict__ up_scales,
    const uint8_t* __restrict__ gate_weight,
    const unsigned short* __restrict__ gate_scales,
    float* __restrict__ output,
    int K,
    int M_slice,
    int block_size
) {
    const int local_row = threadIdx.x / THREADS_PER_ROW;
    const int lane = threadIdx.x % THREADS_PER_ROW;
    const int row = blockIdx.x * ROWS_PER_BLOCK + local_row;
    if (row >= M_slice) return;

    const int packed_k = (K + 1) / 2;
    const int num_blocks = (K + block_size - 1) / block_size;
    const uint8_t* up_row = up_weight + (long long)row * packed_k;
    const unsigned short* up_s = up_scales + (long long)row * num_blocks;
    const uint8_t* gate_row = gate_weight + (long long)row * packed_k;
    const unsigned short* gate_s = gate_scales + (long long)row * num_blocks;

    float up_acc = 0.0f, gate_acc = 0.0f;
    for (int k = lane * 2; k < K; k += THREADS_PER_ROW * 2) {
        int byte_idx = k / 2;
        int blk = k / block_size;

        uint8_t up_packed = up_row[byte_idx];
        float up_scale = bf16_to_float(up_s[blk]);
        float up_w0 = nvfp4_lut[up_packed & 0xF];
        float up_w1 = nvfp4_lut[(up_packed >> 4) & 0xF];

        uint8_t gate_packed = gate_row[byte_idx];
        float gate_scale = bf16_to_float(gate_s[blk]);
        float gate_w0 = nvfp4_lut[gate_packed & 0xF];
        float gate_w1 = nvfp4_lut[(gate_packed >> 4) & 0xF];

        float inp0 = input[k];
        up_acc += inp0 * up_w0 * up_scale;
        gate_acc += inp0 * gate_w0 * gate_scale;
        if (k + 1 < K) {
            float inp1 = input[k + 1];
            up_acc += inp1 * up_w1 * up_scale;
            gate_acc += inp1 * gate_w1 * gate_scale;
        }
    }

    up_acc = warp_reduce_sum(up_acc);
    gate_acc = warp_reduce_sum(gate_acc);

    __shared__ float smem[ROWS_PER_BLOCK * (THREADS_PER_ROW / WARP_SIZE) * 2];
    const int warp_id = lane / WARP_SIZE;
    const int warp_lane = lane % WARP_SIZE;
    const int n_warps = THREADS_PER_ROW / WARP_SIZE;

    if (warp_lane == 0) {
        smem[(local_row * n_warps + warp_id) * 2 + 0] = up_acc;
        smem[(local_row * n_warps + warp_id) * 2 + 1] = gate_acc;
    }
    __syncthreads();

    if (warp_id == 0) {
        float final_up = (warp_lane < n_warps)
            ? smem[(local_row * n_warps + warp_lane) * 2 + 0] : 0.0f;
        float final_gate = (warp_lane < n_warps)
            ? smem[(local_row * n_warps + warp_lane) * 2 + 1] : 0.0f;
        final_up = warp_reduce_sum(final_up);
        final_gate = warp_reduce_sum(final_gate);
        if (warp_lane == 0) {
            float silu = final_gate / (1.0f + expf(-final_gate));
            output[row] = silu * final_up;  // FP32 output
        }
    }
}

// Partial matmul NVFP4 with FP32 output
// Note: scales are BF16 (from E8M0→BF16 conversion), NOT FP16
__global__ void vib3_partial_matmul_nvfp4_f32out_kernel(
    const float* __restrict__ input,
    const uint8_t* __restrict__ weight,
    const unsigned short* __restrict__ scales,
    float* __restrict__ output,
    int K,
    int M_slice,
    int block_size
) {
    const int local_row = threadIdx.x / THREADS_PER_ROW;
    const int lane = threadIdx.x % THREADS_PER_ROW;
    const int row = blockIdx.x * ROWS_PER_BLOCK + local_row;
    if (row >= M_slice) return;

    const int packed_k = (K + 1) / 2;
    const int num_blocks = (K + block_size - 1) / block_size;
    const uint8_t* row_weight = weight + (long long)row * packed_k;
    const unsigned short* row_scales = scales + (long long)row * num_blocks;

    float acc = 0.0f;
    for (int k = lane * 2; k < K; k += THREADS_PER_ROW * 2) {
        int byte_idx = k / 2;
        uint8_t packed = row_weight[byte_idx];
        float w0 = nvfp4_lut[packed & 0xF];
        float w1 = nvfp4_lut[(packed >> 4) & 0xF];

        int blk = k / block_size;
        float scale = bf16_to_float(row_scales[blk]);

        acc += input[k] * w0 * scale;
        if (k + 1 < K) {
            acc += input[k + 1] * w1 * scale;
        }
    }

    acc = warp_reduce_sum(acc);

    __shared__ float smem[ROWS_PER_BLOCK * (THREADS_PER_ROW / WARP_SIZE)];
    const int warp_id = lane / WARP_SIZE;
    const int warp_lane = lane % WARP_SIZE;

    if (warp_lane == 0) {
        smem[local_row * (THREADS_PER_ROW / WARP_SIZE) + warp_id] = acc;
    }
    __syncthreads();

    if (warp_id == 0) {
        float final_acc = (warp_lane < (THREADS_PER_ROW / WARP_SIZE))
            ? smem[local_row * (THREADS_PER_ROW / WARP_SIZE) + warp_lane] : 0.0f;
        final_acc = warp_reduce_sum(final_acc);
        if (warp_lane == 0) {
            output[row] = final_acc;  // FP32 output
        }
    }
}

// FP32→FP32 weighted accumulate (both buffers FP32)
__global__ void vib3_weighted_accumulate_f32_f32_kernel(
    float* __restrict__ output,
    const float* __restrict__ expert_output,
    float weight,
    int dim
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= dim) return;
    output[i] += weight * expert_output[i];
}

// Sigmoid-gated FP16→FP32 weighted accumulate: output[i] += sigmoid(gate[0]) * expert_output[i]
// The gate scalar is read from device memory (a single FP32 value), sigmoid is applied on-GPU.
// This eliminates a stream.synchronize() + D2H copy for the gate value.
__global__ void vib3_sigmoid_gated_accumulate_f32_kernel(
    float* __restrict__ output,
    const half* __restrict__ expert_output,
    const float* __restrict__ gate_dev,  // 1-element FP32 on device
    int dim
) {
    __shared__ float gate_sigmoid;
    if (threadIdx.x == 0) {
        float g = gate_dev[0];
        gate_sigmoid = 1.0f / (1.0f + expf(-g));
    }
    __syncthreads();

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= dim) return;
    output[i] += gate_sigmoid * __half2float(expert_output[i]);
}

// ─── DeltaNet Kernels ──────────────────────────────────────────────────────

// L2 normalization: output = input / ||input||_2, per vector.
// One block per vector, 256 threads cooperate.
__global__ void vib3_l2_norm_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int num_vecs,
    int dim,
    float eps
) {
    int vec = blockIdx.x;
    if (vec >= num_vecs) return;
    int tid = threadIdx.x;

    extern __shared__ float shared[];

    const float* vec_in = input + (long long)vec * dim;
    float* vec_out = output + (long long)vec * dim;

    float partial_ss = 0.0f;
    for (int d = tid; d < dim; d += blockDim.x) {
        float val = vec_in[d];
        partial_ss += val * val;
    }
    shared[tid] = partial_ss;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) shared[tid] += shared[tid + s];
        __syncthreads();
    }

    float inv_norm = rsqrtf(shared[0] + eps);

    for (int d = tid; d < dim; d += blockDim.x) {
        vec_out[d] = vec_in[d] * inv_norm;
    }
}

// Causal depthwise 1D convolution with SiLU activation.
// One thread per channel. Updates conv_state in-place (shift left, append new input).
__global__ void vib3_causal_conv1d_kernel(
    float* __restrict__ conv_state,       // [num_channels, kernel_size-1]
    const float* __restrict__ new_input,  // [num_channels]
    const float* __restrict__ conv_weight, // [num_channels, kernel_size]
    float* __restrict__ output,           // [num_channels]
    int num_channels,
    int kernel_size
) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= num_channels) return;

    int state_width = kernel_size - 1;
    float* state = conv_state + c * state_width;
    const float* w = conv_weight + c * kernel_size;

    // Convolution: state[0..state_width-1] * weight[0..state_width-1] + new_input * weight[last]
    float acc = 0.0f;
    for (int i = 0; i < state_width; i++) {
        acc += state[i] * w[i];
    }
    acc += new_input[c] * w[state_width];

    // Update state: shift left, append new input
    for (int i = 0; i < state_width - 1; i++) {
        state[i] = state[i + 1];
    }
    if (state_width > 0) {
        state[state_width - 1] = new_input[c];
    }

    // SiLU activation
    output[c] = acc / (1.0f + expf(-acc));
}

// Fused DeltaNet autoregressive step for all heads.
// One block per head. Per head: decay -> retrieve -> delta -> update -> output.
// state[h]: [vdim, vdim], q/k/v[h]: [vdim], gate[h]: scalar, beta[h]: scalar.
__global__ void vib3_deltanet_step_kernel(
    float* __restrict__ state,
    const float* __restrict__ q,
    const float* __restrict__ k,
    const float* __restrict__ v,
    const float* __restrict__ gate,
    const float* __restrict__ beta,
    float* __restrict__ output,
    int num_heads,
    int vdim
) {
    int h = blockIdx.x;
    if (h >= num_heads) return;
    int tid = threadIdx.x;

    extern __shared__ float smem[];
    float* s_k = smem;
    float* s_q = smem + vdim;
    float* s_v = smem + 2 * vdim;

    float g = gate[h];
    float b = beta[h];

    // Load k, q, v to shared memory
    for (int j = tid; j < vdim; j += blockDim.x) {
        s_k[j] = k[h * vdim + j];
        s_q[j] = q[h * vdim + j];
        s_v[j] = v[h * vdim + j];
    }
    __syncthreads();

    float* state_h = state + (long long)h * vdim * vdim;

    // Each thread handles one or more rows of the state matrix
    for (int i = tid; i < vdim; i += blockDim.x) {
        float* state_row = state_h + i * vdim;

        // Decay + Retrieve: retrieved = (decayed_state_row) dot k
        // g is the log-space gate = -exp(A_log) * softplus(alpha + dt_bias) (always negative)
        // Actual decay factor is exp(g) ∈ (0, 1]
        float decay = expf(g);
        float retrieved = 0.0f;
        for (int j = 0; j < vdim; j++) {
            state_row[j] *= decay;
            retrieved += state_row[j] * s_k[j];
        }

        // Delta: beta * (v[i] - retrieved)
        float delta = b * (s_v[i] - retrieved);

        // Update + Output: state_row += delta * k, output = state_row dot q
        float out = 0.0f;
        for (int j = 0; j < vdim; j++) {
            state_row[j] += delta * s_k[j];
            out += state_row[j] * s_q[j];
        }
        output[h * vdim + i] = out;
    }
}

// Gated RMSNorm: output = RMSNorm(x, weight) * SiLU(gate).
// One block per group (head), 256 threads cooperate.
// weight cycles: weight[d % norm_dim].
__global__ void vib3_gated_rmsnorm_kernel(
    const float* __restrict__ x,
    const float* __restrict__ gate_data,
    const float* __restrict__ weight,
    float* __restrict__ output,
    int num_groups,
    int group_dim,
    int norm_dim,
    float eps
) {
    int g = blockIdx.x;
    if (g >= num_groups) return;
    int tid = threadIdx.x;

    extern __shared__ float shared[];

    const float* x_g = x + (long long)g * group_dim;
    const float* gate_g = gate_data + (long long)g * group_dim;
    float* out_g = output + (long long)g * group_dim;

    // Compute sum of squares for RMSNorm
    float partial_ss = 0.0f;
    for (int d = tid; d < group_dim; d += blockDim.x) {
        float val = x_g[d];
        partial_ss += val * val;
    }
    shared[tid] = partial_ss;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) shared[tid] += shared[tid + s];
        __syncthreads();
    }

    float rms = sqrtf(shared[0] / (float)group_dim + eps);
    float inv_rms = 1.0f / rms;

    // Normalize, apply cycling weight, multiply by SiLU(gate)
    for (int d = tid; d < group_dim; d += blockDim.x) {
        float norm_val = x_g[d] * inv_rms * weight[d % norm_dim];
        float gate_val = gate_g[d];
        float silu_gate = gate_val / (1.0f + expf(-gate_val));
        out_g[d] = norm_val * silu_gate;
    }
}

// Element-wise sigmoid: output[i] = 1 / (1 + exp(-input[i]))
__global__ void vib3_sigmoid_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    output[i] = 1.0f / (1.0f + expf(-input[i]));
}

// DeltaNet gate: gate = -exp(A_log) * softplus(alpha + dt_bias)
// One thread per head.
__global__ void vib3_deltanet_gate_kernel(
    const float* __restrict__ alpha,
    const float* __restrict__ dt_bias,
    const float* __restrict__ a_log,
    float* __restrict__ gate_out,
    int num_heads
) {
    int h = blockIdx.x * blockDim.x + threadIdx.x;
    if (h >= num_heads) return;
    float a = alpha[h] + dt_bias[h];
    float softplus = logf(1.0f + expf(a));
    gate_out[h] = a_log[h] * softplus;
}

// ─── Utility Kernels ───────────────────────────────────────────────────────

// In-place FP32 scale: data[i] *= scale
__global__ void vib3_scale_f32_kernel(
    float* __restrict__ data,
    int n,
    float scale
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    data[i] *= scale;
}

// Tiled repeat: [num_groups, dim] -> [num_groups*repeat, dim]
// Layout: [G0,G1,...,G_{n-1}, G0,G1,...,G_{n-1}, ...]
// Matches the V-head tiled order produced by llama.cpp's _LinearAttentionVReorderBase.
__global__ void vib3_repeat_tile_f32_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int num_groups,
    int dim,
    int repeat
) {
    int total = num_groups * repeat * dim;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    // idx maps to output; source is input[idx % (num_groups * dim)]
    int gd = idx % (num_groups * dim);
    output[idx] = input[gd];
}

// Per-head RMSNorm in-place (FP16).
// data: [num_heads * stride] FP16, weight: [head_dim] FP16.
// One block per head, 256 threads cooperate.
__global__ void vib3_per_head_rmsnorm_kernel(
    half* __restrict__ data,
    const half* __restrict__ weight,
    int head_dim,
    int stride,
    int num_heads,
    float eps
) {
    int head = blockIdx.x;
    if (head >= num_heads) return;
    int tid = threadIdx.x;

    extern __shared__ float shared[];

    half* head_data = data + head * stride;

    // Compute sum of squares
    float partial_ss = 0.0f;
    for (int d = tid; d < head_dim; d += blockDim.x) {
        float val = __half2float(head_data[d]);
        partial_ss += val * val;
    }
    shared[tid] = partial_ss;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) shared[tid] += shared[tid + s];
        __syncthreads();
    }

    float rms = sqrtf(shared[0] / (float)head_dim + eps);
    float inv_rms = 1.0f / rms;

    for (int d = tid; d < head_dim; d += blockDim.x) {
        float val = __half2float(head_data[d]) * inv_rms * __half2float(weight[d]);
        head_data[d] = __float2half(val);
    }
}

// Partial RoPE in-place (FP16): apply rotary embeddings to first rope_dim dims.
// data: [num_heads, stride] FP16.
__global__ void vib3_partial_rope_kernel(
    half* __restrict__ data,
    int head_dim,
    int rope_dim,
    int stride,
    int num_heads,
    const int* __restrict__ d_position,
    float rope_base
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int half_rope = rope_dim / 2;
    int total_pairs = num_heads * half_rope;
    if (idx >= total_pairs) return;

    int position = *d_position;
    int head = idx / half_rope;
    int i = idx % half_rope;

    float freq = 1.0f / powf(rope_base, 2.0f * (float)i / (float)rope_dim);
    float theta = (float)position * freq;
    float cos_t = cosf(theta);
    float sin_t = sinf(theta);

    int offset = head * stride + 2 * i;
    float x0 = __half2float(data[offset]);
    float x1 = __half2float(data[offset + 1]);
    data[offset]     = __float2half(x0 * cos_t - x1 * sin_t);
    data[offset + 1] = __float2half(x0 * sin_t + x1 * cos_t);
}

// Sigmoid-gated multiply (FP16): output[i] = input[i] * sigmoid(gate[i])
__global__ void vib3_sigmoid_mul_f16_kernel(
    half* __restrict__ output,
    const half* __restrict__ input,
    const half* __restrict__ gate,
    int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float inp = __half2float(input[i]);
    float g = __half2float(gate[i]);
    float sig = 1.0f / (1.0f + expf(-g));
    output[i] = __float2half(inp * sig);
}

// Deinterleave FP16: [A0(chunk),B0(chunk),A1(chunk),B1(chunk),...] ->
//   output_a = [A0,A1,...], output_b = [B0,B1,...].
__global__ void vib3_deinterleave_f16_kernel(
    const half* __restrict__ input,
    half* __restrict__ output_a,
    half* __restrict__ output_b,
    int chunk_size,
    int num_chunks
) {
    int total = num_chunks * chunk_size;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;

    int c = idx / chunk_size;
    int d = idx % chunk_size;

    output_a[c * chunk_size + d] = input[c * 2 * chunk_size + d];
    output_b[c * chunk_size + d] = input[c * 2 * chunk_size + chunk_size + d];
}

// ─── Launchers for Reconstructed Kernels ──────────────────────────────────

extern "C" {

int vib3_launch_fused_swiglu_nvfp4_f32out(
    const void* input, const void* up_weight, const void* up_scales,
    const void* gate_weight, const void* gate_scales,
    void* output, int K, int M_slice, int block_size, void* stream
) {
    cudaStream_t s = (cudaStream_t)stream;
    int blocks = (M_slice + ROWS_PER_BLOCK - 1) / ROWS_PER_BLOCK;
    vib3_fused_swiglu_nvfp4_f32out_kernel<<<blocks, ROWS_PER_BLOCK * THREADS_PER_ROW, 0, s>>>(
        (const float*)input,
        (const uint8_t*)up_weight, (const unsigned short*)up_scales,
        (const uint8_t*)gate_weight, (const unsigned short*)gate_scales,
        (float*)output, K, M_slice, block_size
    );
    return (int)cudaGetLastError();
}

int vib3_launch_partial_matmul_nvfp4_f32out(
    const void* input, const void* weight_packed, const void* scales,
    void* output, int K, int M_slice, int block_size, void* stream
) {
    cudaStream_t s = (cudaStream_t)stream;
    int blocks = (M_slice + ROWS_PER_BLOCK - 1) / ROWS_PER_BLOCK;
    vib3_partial_matmul_nvfp4_f32out_kernel<<<blocks, ROWS_PER_BLOCK * THREADS_PER_ROW, 0, s>>>(
        (const float*)input,
        (const uint8_t*)weight_packed, (const unsigned short*)scales,
        (float*)output, K, M_slice, block_size
    );
    return (int)cudaGetLastError();
}

int vib3_launch_weighted_accumulate_f32_f32(
    void* output, const void* expert_output, float weight,
    int dim, void* stream
) {
    cudaStream_t s = (cudaStream_t)stream;
    int blocks = (dim + 255) / 256;
    vib3_weighted_accumulate_f32_f32_kernel<<<blocks, 256, 0, s>>>(
        (float*)output, (const float*)expert_output, weight, dim
    );
    return (int)cudaGetLastError();
}

int vib3_launch_sigmoid_gated_accumulate_f32(
    void* output, const void* expert_output, const void* gate_dev,
    int dim, void* stream
) {
    cudaStream_t s = (cudaStream_t)stream;
    int blocks = (dim + 255) / 256;
    vib3_sigmoid_gated_accumulate_f32_kernel<<<blocks, 256, 0, s>>>(
        (float*)output, (const half*)expert_output, (const float*)gate_dev, dim
    );
    return (int)cudaGetLastError();
}

// Variant: FP32 expert output (for NVFP4 MMA path shared expert)
__global__ void vib3_sigmoid_gated_accumulate_f32in_kernel(
    float* __restrict__ output,
    const float* __restrict__ expert_output,
    const float* __restrict__ gate_dev,
    int dim
) {
    __shared__ float gate_sigmoid;
    if (threadIdx.x == 0) {
        float g = gate_dev[0];
        gate_sigmoid = 1.0f / (1.0f + expf(-g));
    }
    __syncthreads();

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= dim) return;
    output[i] += gate_sigmoid * expert_output[i];
}

int vib3_launch_sigmoid_gated_accumulate_f32in(
    void* output, const void* expert_output, const void* gate_dev,
    int dim, void* stream
) {
    cudaStream_t s = (cudaStream_t)stream;
    int blocks = (dim + 255) / 256;
    vib3_sigmoid_gated_accumulate_f32in_kernel<<<blocks, 256, 0, s>>>(
        (float*)output, (const float*)expert_output, (const float*)gate_dev, dim
    );
    return (int)cudaGetLastError();
}

int vib3_launch_l2_norm(
    const void* input, void* output, int num_vecs, int dim,
    float eps, void* stream
) {
    cudaStream_t s = (cudaStream_t)stream;
    int smem_bytes = 256 * sizeof(float);
    vib3_l2_norm_kernel<<<num_vecs, 256, smem_bytes, s>>>(
        (const float*)input, (float*)output, num_vecs, dim, eps
    );
    return (int)cudaGetLastError();
}

int vib3_launch_causal_conv1d(
    void* conv_state, const void* new_input, const void* conv_weight,
    void* output, int num_channels, int kernel_size, void* stream
) {
    cudaStream_t s = (cudaStream_t)stream;
    int blocks = (num_channels + 255) / 256;
    vib3_causal_conv1d_kernel<<<blocks, 256, 0, s>>>(
        (float*)conv_state, (const float*)new_input,
        (const float*)conv_weight, (float*)output,
        num_channels, kernel_size
    );
    return (int)cudaGetLastError();
}

int vib3_launch_deltanet_step(
    void* state, const void* q, const void* k, const void* v,
    const void* gate, const void* beta, void* output,
    int num_heads, int vdim, void* stream
) {
    cudaStream_t s = (cudaStream_t)stream;
    int smem_bytes = 3 * vdim * sizeof(float);
    vib3_deltanet_step_kernel<<<num_heads, 256, smem_bytes, s>>>(
        (float*)state, (const float*)q, (const float*)k, (const float*)v,
        (const float*)gate, (const float*)beta, (float*)output,
        num_heads, vdim
    );
    return (int)cudaGetLastError();
}

int vib3_launch_gated_rmsnorm(
    const void* x, const void* gate, const void* weight,
    void* output, int num_groups, int group_dim, int norm_dim,
    float eps, void* stream
) {
    cudaStream_t s = (cudaStream_t)stream;
    int smem_bytes = 256 * sizeof(float);
    vib3_gated_rmsnorm_kernel<<<num_groups, 256, smem_bytes, s>>>(
        (const float*)x, (const float*)gate, (const float*)weight,
        (float*)output, num_groups, group_dim, norm_dim, eps
    );
    return (int)cudaGetLastError();
}

int vib3_launch_sigmoid(
    const void* input, void* output, int n, void* stream
) {
    cudaStream_t s = (cudaStream_t)stream;
    int blocks = (n + 255) / 256;
    vib3_sigmoid_kernel<<<blocks, 256, 0, s>>>(
        (const float*)input, (float*)output, n
    );
    return (int)cudaGetLastError();
}

int vib3_launch_deltanet_gate(
    const void* alpha, const void* dt_bias, const void* a_log,
    void* gate_out, int num_heads, void* stream
) {
    cudaStream_t s = (cudaStream_t)stream;
    int blocks = (num_heads + 255) / 256;
    vib3_deltanet_gate_kernel<<<blocks, 256, 0, s>>>(
        (const float*)alpha, (const float*)dt_bias, (const float*)a_log,
        (float*)gate_out, num_heads
    );
    return (int)cudaGetLastError();
}

int vib3_launch_scale_f32(
    void* data, int n, float scale, void* stream
) {
    cudaStream_t s = (cudaStream_t)stream;
    int blocks = (n + 255) / 256;
    vib3_scale_f32_kernel<<<blocks, 256, 0, s>>>(
        (float*)data, n, scale
    );
    return (int)cudaGetLastError();
}

int vib3_launch_repeat_tile_f32(
    const void* input, void* output, int num_groups, int dim,
    int repeat, void* stream
) {
    cudaStream_t s = (cudaStream_t)stream;
    int total = num_groups * repeat * dim;
    int blocks = (total + 255) / 256;
    vib3_repeat_tile_f32_kernel<<<blocks, 256, 0, s>>>(
        (const float*)input, (float*)output,
        num_groups, dim, repeat
    );
    return (int)cudaGetLastError();
}

int vib3_launch_per_head_rmsnorm(
    void* data, const void* weight, int head_dim, int stride,
    int num_heads, float eps, void* stream
) {
    cudaStream_t s = (cudaStream_t)stream;
    int smem_bytes = 256 * sizeof(float);
    vib3_per_head_rmsnorm_kernel<<<num_heads, 256, smem_bytes, s>>>(
        (half*)data, (const half*)weight,
        head_dim, stride, num_heads, eps
    );
    return (int)cudaGetLastError();
}

int vib3_launch_partial_rope(
    void* data, int head_dim, int rope_dim, int stride,
    int num_heads, const void* d_position, float rope_base, void* stream
) {
    cudaStream_t s = (cudaStream_t)stream;
    int half_rope = rope_dim / 2;
    int total_pairs = num_heads * half_rope;
    int blocks = (total_pairs + 255) / 256;
    vib3_partial_rope_kernel<<<blocks, 256, 0, s>>>(
        (half*)data, head_dim, rope_dim, stride,
        num_heads, (const int*)d_position, rope_base
    );
    return (int)cudaGetLastError();
}

int vib3_launch_sigmoid_mul_f16(
    void* output, const void* input, const void* gate,
    int n, void* stream
) {
    cudaStream_t s = (cudaStream_t)stream;
    int blocks = (n + 255) / 256;
    vib3_sigmoid_mul_f16_kernel<<<blocks, 256, 0, s>>>(
        (half*)output, (const half*)input, (const half*)gate, n
    );
    return (int)cudaGetLastError();
}

int vib3_launch_deinterleave_f16(
    const void* input, void* output_a, void* output_b,
    int chunk_size, int num_chunks, void* stream
) {
    cudaStream_t s = (cudaStream_t)stream;
    int total = num_chunks * chunk_size;
    int blocks = (total + 255) / 256;
    vib3_deinterleave_f16_kernel<<<blocks, 256, 0, s>>>(
        (const half*)input, (half*)output_a, (half*)output_b,
        chunk_size, num_chunks
    );
    return (int)cudaGetLastError();
}

// ─── MMA Blackwell NVFP4 GEMV Kernel ─────────────────────────────────────
//
// Blackwell-native FP4 Tensor Core GEMV using mma.sync.aligned.kind::mxf4
// m16n8k64 instruction for MoE expert matmuls (M=1 token decode).
//
// Design:
// - A (weights) = [M_slice, K] in vib3 sequential nibble format
// - B (activations) = [K] FP32, quantized to FP4 E2M1 on-the-fly
// - C (output) = [M_slice] FP32 accumulated
// - Each warp handles 16 output rows
// - K processed in chunks of 64 (one MMA per chunk)
// - Weight data repacked from sequential to split-half in shared memory
// - Activation quantized per 32-element group with E8M0 scales
//
// Guarded: only compiled for sm_120a (Blackwell), falls back to software path.

// Repack 16 bytes (32 FP4 nibbles) from sequential format to split-half format.
// Sequential: byte j = {elem[2j], elem[2j+1]}
// Split-half: byte j = {elem[j], elem[j+16]}
__device__ __forceinline__ void repack_seq_to_split(
    const uint8_t* __restrict__ seq,   // 16 bytes in
    uint8_t* __restrict__ split        // 16 bytes out
) {
    #pragma unroll
    for (int j = 0; j < 8; j++) {
        uint8_t s0 = seq[j];
        uint8_t s1 = seq[j + 8];
        split[2 * j]     = (s0 & 0x0F) | ((s1 & 0x0F) << 4);
        split[2 * j + 1] = (s0 >> 4)   | (s1 & 0xF0);
    }
}

// Compute E8M0 scale for a group of FP32 values (matching llama.cpp)
__device__ __forceinline__ uint8_t compute_e8m0(float amax) {
    if (!(amax > 0.0f)) return 0;
    const float e = log2f(amax);
    const int e_int = __float2int_rn(e);
    int biased = e_int - 2 + 127;  // FP4_E2M1_EMAX = 2
    biased = max(biased, 0);
    biased = min(biased, 254);
    return (uint8_t)biased;
}

// Convert E8M0 to FP32 scale value
__device__ __forceinline__ float e8m0_to_f32(uint8_t e) {
    // E8M0 is just a biased exponent: value = 2^(e-127)
    // Represent as float by setting exponent field
    unsigned int bits = ((unsigned int)e) << 23;
    float result;
    memcpy(&result, &bits, sizeof(float));
    return result;
}

// ── MMA self-test kernel ──
// Tests the MMA instruction with a known pattern: all weights = 1.0 (FP4),
// all activations = 1.0 (FP4). With K=64, each output should be 64.0 * scale_A * scale_B.
// With scale = 127 (E8M0 for 1.0), result should be 64.0.
//
// FP4 E2M1 value 1.0 = 0b0100 = 4 (sign=0, exp=10, man=0 -> (-1)^0 * 2^(2-1) * 1.0 = 2.0)
// Wait no: E2M1 encoding: 0b0100 = exp=10=2, mant=0 -> value = 2^(2-1)*(1+0) = 2.0
// 0b0010 = exp=01=1, mant=0 -> value = 2^(1-1)*(1+0) = 1.0
// So FP4 1.0 = 0b0010 = 2
//
// Split-half byte with elem[j]=0b0010 and elem[j+16]=0b0010:
//   low nibble = 0b0010, high nibble = 0b0010 -> byte = 0x22
//
// E8M0 scale for 1.0: exponent of 1.0 is 0, so biased = 0-2+127 = 125
// Actually for activation: amax=1.0, e=round(log2(1.0))-2+127 = 0-2+127 = 125
// e8m0_to_f32(125) = 2^(125-127) = 2^(-2) = 0.25
// So the true activation values after scale: 1.0 * (1/0.25) = 4.0
// Then quantized: FP4(4.0) = 0b0110 = 6 (exp=11=3, mant=0 -> 2^(3-1) = 4.0)
//
// Let me use a simpler approach: directly create known-good FP4 data in split-half format.

__global__ void vib3_mma_selftest_kernel(float* __restrict__ results) {
#if __CUDA_ARCH__ >= 1200
    // Test: A has distinct rows, B is uniform activation = 1.0 (FP4).
    // Row r: all K elements = FP4 value (r+1) where r=0..15.
    // Activation: all elements = 1.0 (FP4 = 0b0010 = 2).
    // With scale 127 (E8M0 for 1.0): each output = K * row_val * 1.0 * 1.0 * 1.0
    // Wait: E8M0 127 = 2^(127-127) = 1.0. So scales are unity.
    // Result for row r: 64 * fp4_val(r) * 1.0 * 1.0 * 1.0 = 64 * fp4_val(r)
    //
    // FP4 E2M1 values: 0=0, 1=0.5, 2=1.0, 3=1.5, 4=2.0, 5=3.0, 6=4.0, 7=6.0
    //   (signed: 8-15 are negatives of 0-7)
    //
    // Use row_val = 2 (FP4 1.0) for all rows for simplicity. Expected = 64.
    // But also test with different values per row.
    //
    // Row 0: all elems = FP4 0b0001 (0.5) -> expected = 64 * 0.5 = 32.0
    // Row 1: all elems = FP4 0b0010 (1.0) -> expected = 64 * 1.0 = 64.0
    // Row 2: all elems = FP4 0b0011 (1.5) -> expected = 64 * 1.5 = 96.0
    // ...etc. But we need the nibbles in split-half format.
    //
    // For simplicity, let me test the REPACK function:
    // Create sequential format data, repack it, run MMA, check result.

    const int lane_id = threadIdx.x;
    const int groupID = lane_id >> 2;
    const int tid_in_grp = lane_id & 3;

    __shared__ uint8_t seq_data[16];   // 32 FP4 in sequential format
    __shared__ uint8_t split_data[16]; // 32 FP4 in split-half format
    __shared__ uint8_t wt_sh[512];     // 16 rows * 32 bytes
    __shared__ uint8_t act_sh[32];     // 64 FP4 elements

    // ── Test repack ──
    // Sequential: byte j = {elem[2j] in low nibble, elem[2j+1] in high nibble}
    // Create: elem[i] = (i % 8) for i=0..31
    // seq[0] = {elem[0], elem[1]} = {0, 1} = 0x10
    // seq[1] = {elem[2], elem[3]} = {2, 3} = 0x32
    // ...
    // seq[j] = {elem[2j], elem[2j+1]} = (2j+1)<<4 | (2j)
    if (lane_id < 16) {
        int e0 = (2 * lane_id) % 8;
        int e1 = (2 * lane_id + 1) % 8;
        seq_data[lane_id] = (uint8_t)((e1 << 4) | e0);
    }
    __syncwarp();

    if (lane_id == 0) {
        repack_seq_to_split(seq_data, split_data);
    }
    __syncwarp();

    // Verify: split_data[j] should have {elem[j], elem[j+16]} 
    // elem[j] = j%8, elem[j+16] = (j+16)%8 = j%8  (since 16%8=0)
    // So split_data[j] = (j%8) | ((j%8)<<4)
    // E.g., j=0: {0,0}=0x00, j=1: {1,1}=0x11, j=5: {5,5}=0x55, j=7: {7,7}=0x77
    // j=8: elem[8]=0, elem[24]=0 -> 0x00
    // j=9: elem[9]=1, elem[25]=1 -> 0x11
    if (lane_id < 16) {
        uint8_t expected = (uint8_t)(((lane_id % 8) << 4) | (lane_id % 8));
        uint8_t actual = split_data[lane_id];
        results[lane_id] = (actual == expected) ? 1.0f : -(float)(int)actual;
        results[16 + lane_id] = (float)(int)actual;
        results[32 + lane_id] = (float)(int)expected;
    }
    __syncwarp();

    // ── Test 2: MMA with uniform data + unity scales ──
    for (int i = lane_id; i < 512; i += 32) {
        wt_sh[i] = 0x22; // FP4 1.0 = 0b0010 in both nibbles
    }
    if (lane_id < 32) {
        act_sh[lane_id] = 0x22;
    }
    __syncwarp();

    const int* wt_qs = (const int*)wt_sh;
    const int wt_stride = 8; // 32 bytes per row / 4 bytes per int32
    // Block 0 = int32[0..3], Block 1 = int32[4..7] within each row
    int a0 = wt_qs[groupID * wt_stride + tid_in_grp];           // PTX reg0: row groupID, K block 0
    int a1 = wt_qs[(groupID + 8) * wt_stride + tid_in_grp];     // PTX reg1: row groupID+8, K block 0
    int a2 = wt_qs[groupID * wt_stride + 4 + tid_in_grp];       // PTX reg2: row groupID, K block 1
    int a3 = wt_qs[(groupID + 8) * wt_stride + 4 + tid_in_grp]; // PTX reg3: row groupID+8, K block 1

    const int* act_qs = (const int*)act_sh;
    int b0 = act_qs[tid_in_grp];
    int b1 = act_qs[tid_in_grp + 4];

    uint32_t sa = 127u | (127u << 8);
    uint32_t sb = 127u | (127u << 8);

    float c0 = 0.0f, c1 = 0.0f, c2 = 0.0f, c3 = 0.0f;

    asm volatile(
        "mma.sync.aligned.kind::mxf4.block_scale.scale_vec::2X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue8m0 "
        "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, "
        "%10, {0, 0}, %11, {0, 0};"
        : "+f"(c0), "+f"(c1), "+f"(c2), "+f"(c3)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
          "r"(b0), "r"(b1),
          "r"(sa), "r"(sb)
    );

    // results[48..175] = per-lane MMA outputs
    results[48 + lane_id * 4 + 0] = c0;
    results[48 + lane_id * 4 + 1] = c1;
    results[48 + lane_id * 4 + 2] = c2;
    results[48 + lane_id * 4 + 3] = c3;

    // ── Test 3: Non-uniform data (catches K-block indexing errors) ──
    // Weight row 0: K block 0 (K[0..31]) = FP4(2.0), K block 1 (K[32..63]) = FP4(1.0)
    // Weight rows 1-15: all FP4(1.0)
    // Activation: k[i] = FP4(1.0) for all i
    // Scale = 127 (1.0)
    //
    // Expected row 0: 32 * 2.0*1.0 + 32 * 1.0*1.0 = 64 + 32 = 96
    // Expected rows 1-15: 64 * 1.0*1.0 = 64

    // FP4 1.0 = 0b0010, FP4 2.0 = 0b0100
    // split-half byte for {1.0, 1.0} = 0x22
    // split-half byte for {2.0, 2.0} = 0x44

    // Fill weight: all 0x22 first (all rows, both K blocks = FP4 1.0)
    for (int i = lane_id; i < 512; i += 32) {
        wt_sh[i] = 0x22;
    }
    __syncwarp();
    // Row 0, K block 0 (bytes 0-15): set to 0x44 (FP4 2.0)
    // Row 0, K block 1 (bytes 16-31): stays 0x22 (FP4 1.0)
    if (lane_id < 16) {
        wt_sh[lane_id] = 0x44;
    }
    __syncwarp();

    // Activation: all 0x22
    if (lane_id < 32) {
        act_sh[lane_id] = 0x22;
    }
    __syncwarp();

    // Reload A (PTX ISA physical register layout: reg0=rowG blk0, reg1=rowG+8 blk0, reg2=rowG blk1, reg3=rowG+8 blk1)
    a0 = ((const int*)wt_sh)[groupID * 8 + tid_in_grp];           // PTX reg0: row groupID, K block 0
    a1 = ((const int*)wt_sh)[(groupID + 8) * 8 + tid_in_grp];     // PTX reg1: row groupID+8, K block 0
    a2 = ((const int*)wt_sh)[groupID * 8 + 4 + tid_in_grp];       // PTX reg2: row groupID, K block 1
    a3 = ((const int*)wt_sh)[(groupID + 8) * 8 + 4 + tid_in_grp]; // PTX reg3: row groupID+8, K block 1

    // Reload B
    b0 = ((const int*)act_sh)[tid_in_grp];
    b1 = ((const int*)act_sh)[tid_in_grp + 4];

    sa = 127u | (127u << 8);
    sb = 127u | (127u << 8);

    float d0 = 0.0f, d1 = 0.0f, d2 = 0.0f, d3 = 0.0f;
    asm volatile(
        "mma.sync.aligned.kind::mxf4.block_scale.scale_vec::2X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue8m0 "
        "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, "
        "%10, {0, 0}, %11, {0, 0};"
        : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
          "r"(b0), "r"(b1),
          "r"(sa), "r"(sb)
    );

    // results[176..303] = non-uniform test
    results[176 + lane_id * 4 + 0] = d0;
    results[176 + lane_id * 4 + 1] = d1;
    results[176 + lane_id * 4 + 2] = d2;
    results[176 + lane_id * 4 + 3] = d3;
#endif
}

// ── Single-tile comparison kernel ──
// Processes ONE tile (16 rows × K) via both MMA and software dequant,
// writes both results for host-side comparison.
// Outputs: results[0..15] = MMA results for rows 0..15
//          results[16..31] = Software dequant results for rows 0..15
//          results[32..63] = Debug: first 32 dequantized weight values for row 0 (software)
//          results[64..95] = Debug: first 32 input FP32 values
//          results[96..97] = Debug: {scaleA_e8m0_row0_g0, scaleA_e8m0_row0_g1} as floats
//          results[98..99] = Debug: {actscale_g0, actscale_g1} as floats
__global__ void vib3_mma_tile_compare_kernel(
    const float* __restrict__ input_fp32,       // [K] FP32
    const uint8_t* __restrict__ weight_packed,  // [M, K/2] NVFP4 sequential
    const unsigned short* __restrict__ scales,  // [M, K/32] BF16 scales
    float* __restrict__ results,                // output buffer
    int K,
    int M_slice
) {
#if __CUDA_ARCH__ >= 1200
    const int lane_id = threadIdx.x;
    const int groupID = lane_id >> 2;
    const int tid_in_grp = lane_id & 3;
    const int packed_k = K / 2;
    const int num_groups = K / 32;

    // ── Software dequant path (each thread computes one row's dot product) ──
    // Thread 0..15 compute rows 0..15, threads 16..31 idle for this part
    float sw_result = 0.0f;
    if (lane_id < 16) {
        int row = lane_id;
        if (row < M_slice) {
            for (int k = 0; k < K; k++) {
                // Read weight nibble
                int byte_idx = k / 2;
                uint8_t byte_val = weight_packed[(long long)row * packed_k + byte_idx];
                uint8_t nibble = (k & 1) ? (byte_val >> 4) : (byte_val & 0x0F);

                // FP4 E2M1 LUT decode
                const float lut[16] = {0, 0.5, 1, 1.5, 2, 3, 4, 6,
                                        0, -0.5, -1, -1.5, -2, -3, -4, -6};
                float fp4_val = lut[nibble];

                // Apply scale
                int group = k / 32;
                unsigned short bf16_s = scales[(long long)row * num_groups + group];
                uint8_t e8m0 = (bf16_s >> 7) & 0xFF;
                unsigned int sbits = ((unsigned int)e8m0) << 23;
                float scale_f;
                memcpy(&scale_f, &sbits, sizeof(float));

                float dequant = fp4_val * scale_f;
                sw_result += dequant * input_fp32[k];
            }
        }
    }

    // ── Debug: dump first 32 dequantized weights and first 32 inputs ──
    if (lane_id < 32 && lane_id < K) {
        // Dequant weight[row=0][k=lane_id]
        int byte_idx = lane_id / 2;
        uint8_t byte_val = weight_packed[byte_idx]; // row 0
        uint8_t nibble = (lane_id & 1) ? (byte_val >> 4) : (byte_val & 0x0F);
        const float lut[16] = {0, 0.5, 1, 1.5, 2, 3, 4, 6,
                                0, -0.5, -1, -1.5, -2, -3, -4, -6};
        float fp4_val = lut[nibble];
        unsigned short bf16_s = scales[lane_id / 32]; // row 0, group
        uint8_t e8m0 = (bf16_s >> 7) & 0xFF;
        unsigned int sbits = ((unsigned int)e8m0) << 23;
        float scale_f;
        memcpy(&scale_f, &sbits, sizeof(float));
        results[32 + lane_id] = fp4_val * scale_f;

        // Input
        results[64 + lane_id] = input_fp32[lane_id];
    }

    // Store debug scales
    if (lane_id == 0) {
        unsigned short bf16_s0 = scales[0]; // row 0, group 0
        unsigned short bf16_s1 = scales[1]; // row 0, group 1
        results[96] = (float)((bf16_s0 >> 7) & 0xFF);
        results[97] = (float)((bf16_s1 >> 7) & 0xFF);
    }
    __syncwarp();

    // Store software results
    if (lane_id < 16) {
        results[16 + lane_id] = sw_result;
    }

    // ── MMA path (full K accumulation) ──
    float c0 = 0.0f, c1 = 0.0f, c2 = 0.0f, c3 = 0.0f;

    __shared__ uint8_t wt_tile[512]; // 16 rows × 32 bytes
    __shared__ uint8_t act_tile[64]; // 64 FP4 nibbles + padding

    for (int k0 = 0; k0 < K; k0 += 64) {
        // Quantize activations
        uint8_t act_scales_local[2];
        for (int g = 0; g < 2; g++) {
            int k_base = k0 + g * 32;
            float val = 0.0f;
            if (k_base + lane_id < K) {
                val = input_fp32[k_base + lane_id];
            }

            float amax = fabsf(val);
            #pragma unroll
            for (int mask = 16; mask > 0; mask >>= 1) {
                amax = fmaxf(amax, __shfl_xor_sync(0xFFFFFFFF, amax, mask, 32));
            }

            uint8_t e = compute_e8m0(amax);
            act_scales_local[g] = e;
            float inv_s = (amax == 0.0f) ? 0.0f : __frcp_rn(e8m0_to_f32(e));
            float scaled_val = val * inv_s;

            int act_group_id = lane_id / 4;
            int act_lane_in_grp = lane_id % 4;
            int base = act_group_id * 2;

            float val0 = __shfl_sync(0xFFFFFFFF, scaled_val, base, 32);
            float val1 = __shfl_sync(0xFFFFFFFF, scaled_val, base + 16, 32);
            float val2 = __shfl_sync(0xFFFFFFFF, scaled_val, base + 1, 32);
            float val3 = __shfl_sync(0xFFFFFFFF, scaled_val, base + 17, 32);

            if (act_lane_in_grp == 0) {
                __nv_fp4x4_e2m1 fp4_packed(make_float4(val0, val1, val2, val3));
                *(uint16_t*)(act_tile + g * 16 + act_group_id * 2) = *(uint16_t*)&fp4_packed;
            }
        }

        // Store activation scale debug for first K chunk
        if (k0 == 0 && lane_id == 0) {
            results[98] = (float)act_scales_local[0];
            results[99] = (float)act_scales_local[1];
        }
        __syncwarp();

        // Load and repack weights
        int my_block = lane_id;
        int my_row = my_block / 2;
        int my_blk = my_block % 2;
        int global_row = my_row; // tile_row = 0

        if (global_row < M_slice) {
            int k_offset = k0 + my_blk * 32;
            const uint8_t* src = weight_packed + (long long)global_row * packed_k + k_offset / 2;
            uint8_t* dst = wt_tile + my_row * 32 + my_blk * 16;
            repack_seq_to_split(src, dst);
        } else {
            uint8_t* dst = wt_tile + my_row * 32 + my_blk * 16;
            for (int i = 0; i < 4; i++) ((uint32_t*)dst)[i] = 0;
        }
        __syncwarp();

        // Load A registers: block 0 = int32[0..3], block 1 = int32[4..7] per row
        const int* wt_qs = (const int*)wt_tile;
        const int wt_stride = 8; // 32 bytes per row / 4
        int a0 = wt_qs[groupID * wt_stride + tid_in_grp];           // PTX reg0: row groupID, K block 0
        int a1 = wt_qs[(groupID + 8) * wt_stride + tid_in_grp];    // PTX reg1: row groupID+8, K block 0
        int a2 = wt_qs[groupID * wt_stride + 4 + tid_in_grp];      // PTX reg2: row groupID, K block 1
        int a3 = wt_qs[(groupID + 8) * wt_stride + 4 + tid_in_grp]; // PTX reg3: row groupID+8, K block 1

        // Load B registers
        const int* act_qs = (const int*)act_tile;
        int b0 = act_qs[tid_in_grp];
        int b1 = act_qs[tid_in_grp + 4];

        // Load scales
        int tidx = groupID + (tid_in_grp & 1) * 8;
        int scale_row = tidx; // tile_row = 0
        uint32_t sa = 0;
        if (scale_row < M_slice) {
            int g0 = k0 / 32;
            unsigned short bf16_s0 = scales[(long long)scale_row * num_groups + g0];
            unsigned short bf16_s1 = scales[(long long)scale_row * num_groups + g0 + 1];
            uint8_t e0 = (bf16_s0 >> 7) & 0xFF;
            uint8_t e1 = (bf16_s1 >> 7) & 0xFF;
            sa = (uint32_t)e0 | ((uint32_t)e1 << 8);
        }

        uint32_t sb = (uint32_t)act_scales_local[0] | ((uint32_t)act_scales_local[1] << 8);

        asm volatile(
            "mma.sync.aligned.kind::mxf4.block_scale.scale_vec::2X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue8m0 "
            "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, "
            "%10, {0, 0}, %11, {0, 0};"
            : "+f"(c0), "+f"(c1), "+f"(c2), "+f"(c3)
            : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
              "r"(b0), "r"(b1),
              "r"(sa), "r"(sb)
        );
    }

    // Write MMA results: only tid_in_grp==0 threads write
    if (tid_in_grp == 0) {
        results[groupID] = c0;
        results[groupID + 8] = c2;
    }
#endif
}

// ── Activation pre-quantization kernel ──
// Quantizes an FP32 activation vector [K] into FP4 E2M1 sequential format
// that can be directly consumed by the Blackwell MMA instruction.
// This is done ONCE per MoE layer and reused across all 8+1 expert kernels.
//
// Output layout:
//   act_fp4: [K/2] bytes — sequential packed FP4 nibbles (32 bytes per 64-element tile)
//   act_scales: [K/32] uint8 E8M0 — one scale per group of 32 elements
//
// Launch: 1 block per K/64 tile, 32 threads per block (1 warp).
// Each warp quantizes 64 FP32 values (2 groups of 32).
__global__ void vib3_quantize_activation_fp4_kernel(
    const float* __restrict__ input,    // [K] FP32
    uint8_t* __restrict__ act_fp4,      // [K/2] sequential FP4 output
    uint8_t* __restrict__ act_scales,   // [K/32] E8M0 scales output
    int K
) {
    const int tile_idx = blockIdx.x;  // which K/64 tile
    const int lane_id = threadIdx.x;
    const int k0 = tile_idx * 64;

    if (k0 >= K) return;

#if __CUDA_ARCH__ >= 1200
    const int group_id = lane_id / 4;
    const int lane_in_group = lane_id % 4;
    const int seq_base = group_id * 4;

    // Process 2 groups of 32
    for (int g = 0; g < 2; g++) {
        int k_base = k0 + g * 32;

        // Each thread loads one value
        float val = 0.0f;
        if (k_base + lane_id < K) {
            val = input[k_base + lane_id];
        }

        // Warp-wide amax reduction
        float amax = fabsf(val);
        #pragma unroll
        for (int mask = 16; mask > 0; mask >>= 1) {
            amax = fmaxf(amax, __shfl_xor_sync(0xFFFFFFFF, amax, mask, 32));
        }

        uint8_t e = compute_e8m0(amax);
        float inv_s = (amax == 0.0f) ? 0.0f : __frcp_rn(e8m0_to_f32(e));
        float scaled_val = val * inv_s;

        // Gather 4 consecutive values for sequential nibble packing
        // (MMA expects sequential: byte j = {elem[2j] lo, elem[2j+1] hi})
        float val0 = __shfl_sync(0xFFFFFFFF, scaled_val, seq_base, 32);
        float val1 = __shfl_sync(0xFFFFFFFF, scaled_val, seq_base + 1, 32);
        float val2 = __shfl_sync(0xFFFFFFFF, scaled_val, seq_base + 2, 32);
        float val3 = __shfl_sync(0xFFFFFFFF, scaled_val, seq_base + 3, 32);

        if (lane_in_group == 0) {
            __nv_fp4x4_e2m1 fp4_packed(make_float4(val0, val1, val2, val3));
            // Write 2 bytes (4 FP4 nibbles) into output
            *(uint16_t*)(act_fp4 + tile_idx * 32 + g * 16 + group_id * 2) = *(uint16_t*)&fp4_packed;
        }

        // Thread 0 writes the scale for this group
        if (lane_id == 0) {
            act_scales[tile_idx * 2 + g] = e;
        }
    }
#endif
}

// Launcher for activation pre-quantization.
// Returns 0 on success, -1 if arch < 12.
int vib3_launch_quantize_activation_fp4(
    const void* input,      // [K] FP32
    void* act_fp4,          // [K/2] output
    void* act_scales,       // [K/32] output
    int K,
    void* stream
) {
    int dev;
    cudaGetDevice(&dev);
    int major;
    cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev);
    if (major < 12) return -1;

    cudaStream_t s = (cudaStream_t)stream;
    int num_tiles = (K + 63) / 64;
    vib3_quantize_activation_fp4_kernel<<<num_tiles, 32, 0, s>>>(
        (const float*)input,
        (uint8_t*)act_fp4,
        (uint8_t*)act_scales,
        K
    );
    return (int)cudaGetLastError();
}

// ── MMA GEMV kernel with pre-quantized activations ──
// Same as vib3_gemv_mma_nvfp4_kernel but skips the expensive
// per-tile FP32→FP4 activation quantization. Instead reads
// pre-quantized FP4 data and E8M0 scales from buffers that
// were computed once per MoE layer.
//
// This eliminates the dominant bottleneck: ~48 K-iterations ×
// ~18 kernel launches × warp amax reduction per tile.
__global__ void vib3_gemv_mma_nvfp4_preq_kernel(
    const uint8_t* __restrict__ act_fp4,    // [K/2] pre-quantized split-half FP4
    const uint8_t* __restrict__ act_scales,  // [K/32] E8M0 scales
    const uint8_t* __restrict__ weight,      // [M_slice, K/2] packed NVFP4
    const unsigned short* __restrict__ scales, // [M_slice, K/32] BF16 scales
    float* __restrict__ output,              // [M_slice] FP32
    int K,
    int M_slice
) {
    const int warp_id = threadIdx.y;
    const int lane_id = threadIdx.x;
    const int num_warps = blockDim.y;

    const int tile_row = (blockIdx.x * num_warps + warp_id) * 16;
    if (tile_row >= M_slice) return;

    const int packed_k = K / 2;
    const int num_groups = K / 32;

    const int groupID = lane_id >> 2;
    const int tid_in_grp = lane_id & 3;

    float c0 = 0.0f, c1 = 0.0f, c2 = 0.0f, c3 = 0.0f;

    // Shared memory for repacked weight tile only (no activation tile needed!)
    // 16 rows × 32 bytes = 512 bytes per warp
    extern __shared__ char mma_smem[];
    uint8_t* wt_tile = (uint8_t*)(mma_smem + warp_id * 512);

    for (int k0 = 0; k0 < K; k0 += 64) {
        int tile_idx = k0 / 64;

        // ── 1. Load pre-quantized activation data (no quantization needed!) ──
        // Direct load from pre-quantized buffer — this is the entire optimization
        const int* act_qs = (const int*)(act_fp4 + tile_idx * 32);
        int b0 = act_qs[tid_in_grp];      // k[0:31] portion
        int b1 = act_qs[tid_in_grp + 4];  // k[32:63] portion

        // Load pre-computed E8M0 scales for activation
        uint8_t as0 = act_scales[tile_idx * 2];
        uint8_t as1 = act_scales[tile_idx * 2 + 1];
        uint32_t sb = (uint32_t)as0 | ((uint32_t)as1 << 8);

        // ── 2. Load and repack weight data for 16 rows × 64 K elements ──
        int my_block = lane_id;
        int my_row = my_block / 2;
        int my_blk = my_block % 2;
        int global_row = tile_row + my_row;

        if (global_row < M_slice) {
            int k_offset = k0 + my_blk * 32;
            const uint8_t* src = weight + (long long)global_row * packed_k + k_offset / 2;
            uint8_t* dst = wt_tile + my_row * 32 + my_blk * 16;
            repack_seq_to_split(src, dst);
        } else {
            uint8_t* dst = wt_tile + my_row * 32 + my_blk * 16;
            #pragma unroll
            for (int i = 0; i < 4; i++) {
                ((uint32_t*)dst)[i] = 0;
            }
        }
        __syncwarp();

        // ── 3. Load A (weight) registers from shared memory ──
        const int* wt_qs = (const int*)wt_tile;
        const int wt_stride = 8;
        int a0 = wt_qs[groupID * wt_stride + tid_in_grp];
        int a1 = wt_qs[(groupID + 8) * wt_stride + tid_in_grp];
        int a2 = wt_qs[groupID * wt_stride + 4 + tid_in_grp];
        int a3 = wt_qs[(groupID + 8) * wt_stride + 4 + tid_in_grp];

        // ── 4. Compute weight scale registers ──
        int tidx = groupID + (tid_in_grp & 1) * 8;
        int scale_row = tile_row + tidx;
        uint32_t sa = 0;
        if (scale_row < M_slice) {
            int g0 = k0 / 32;
            unsigned short bf16_s0 = scales[(long long)scale_row * num_groups + g0];
            unsigned short bf16_s1 = scales[(long long)scale_row * num_groups + g0 + 1];
            uint8_t e0 = (bf16_s0 >> 7) & 0xFF;
            uint8_t e1 = (bf16_s1 >> 7) & 0xFF;
            sa = (uint32_t)e0 | ((uint32_t)e1 << 8);
        }

        // ── 5. Issue MMA instruction ──
#if __CUDA_ARCH__ >= 1200
        asm volatile(
            "mma.sync.aligned.kind::mxf4.block_scale.scale_vec::2X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue8m0 "
            "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, "
            "%10, {0, 0}, %11, {0, 0};"
            : "+f"(c0), "+f"(c1), "+f"(c2), "+f"(c3)
            : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
              "r"(b0), "r"(b1),
              "r"(sa), "r"(sb)
        );
#endif
    }

    // ── 6. Write output ──
    if (tid_in_grp == 0) {
        int row0 = tile_row + groupID;
        int row1 = tile_row + groupID + 8;
        if (row0 < M_slice) output[row0] = c0;
        if (row1 < M_slice) output[row1] = c2;
    }
}

// ── Original MMA GEMV kernel (with inline activation quantization) ──
// Each block has multiple warps, each warp processes 16 output rows.
// blockDim.x = 32 (warp size), blockDim.y = num_warps
// gridDim.x = ceil(M_slice / (16 * num_warps))
__global__ void vib3_gemv_mma_nvfp4_kernel(
    const float* __restrict__ input,        // [K] FP32
    const uint8_t* __restrict__ weight,     // [M_slice, K/2] packed NVFP4
    const unsigned short* __restrict__ scales,  // [M_slice, K/32] BF16 scales
    float* __restrict__ output,             // [M_slice] FP32
    int K,
    int M_slice
) {
    const int warp_id = threadIdx.y;
    const int lane_id = threadIdx.x;
    const int num_warps = blockDim.y;

    // Which 16-row tile this warp processes
    const int tile_row = (blockIdx.x * num_warps + warp_id) * 16;
    if (tile_row >= M_slice) return;

    const int packed_k = K / 2;  // bytes per row for weights
    const int num_groups = K / 32; // number of scale groups per row

    // MMA thread decomposition
    const int groupID = lane_id >> 2;        // 0..7
    const int tid_in_grp = lane_id & 3;      // 0..3

    // Accumulator registers (C/D): 4 x FP32
    float c0 = 0.0f, c1 = 0.0f, c2 = 0.0f, c3 = 0.0f;

    // Shared memory for repacked weight tile and quantized activations
    // Weight tile: 16 rows × (K_chunk/2) bytes per chunk, but we process K in chunks of 64
    // 16 rows × 32 bytes = 512 bytes per MMA tile (split-half format)
    // Activation: 32 bytes (64 FP4 nibbles) + 4 bytes (2 E8M0 scales packed)
    extern __shared__ char mma_smem[];

    // Per-warp shared memory regions
    // Weight tile: 16 rows × 16 bytes per 32-elem block × 2 blocks = 512 bytes
    uint8_t* wt_tile = (uint8_t*)(mma_smem + warp_id * (512 + 64));
    // Activation tile: 32 bytes (64 nibbles in split-half) + padding
    uint8_t* act_tile = wt_tile + 512;

    // Loop over K in chunks of 64
    for (int k0 = 0; k0 < K; k0 += 64) {
        // ── 1. Quantize 64 activation FP32 values to FP4 E2M1 ──
        // Each warp cooperatively quantizes 64 values (2 groups of 32)
        // Using warp shuffle for amax reduction

        // Process 2 groups of 32 (group 0: k0..k0+31, group 1: k0+32..k0+63)
        uint8_t act_scales[2];
        for (int g = 0; g < 2; g++) {
            int k_base = k0 + g * 32;
            // Each thread loads one value
            float val = 0.0f;
            if (lane_id < 32 && k_base + lane_id < K) {
                val = input[k_base + lane_id];
            }

            // Warp-wide amax reduction
            float amax = fabsf(val);
            #pragma unroll
            for (int mask = 16; mask > 0; mask >>= 1) {
                amax = fmaxf(amax, __shfl_xor_sync(0xFFFFFFFF, amax, mask, 32));
            }

            uint8_t e = compute_e8m0(amax);
            act_scales[g] = e;
            float inv_s = (amax == 0.0f) ? 0.0f : __frcp_rn(e8m0_to_f32(e));

            // Quantize: scale and convert to FP4 E2M1
            float scaled_val = val * inv_s;

            // Pack into split-half format using hardware intrinsic
            // Need to gather 4 values per thread for __nv_fp4x4_e2m1
            // Split-half interleaving: byte j = {elem[j], elem[j+16]}
            // Thread layout for gathering: group_id = lane_id / 4, lane_in_group = lane_id % 4
            // base = group_id * 2
            int act_group_id = lane_id / 4;    // 0..7
            int act_lane_in_grp = lane_id % 4; // 0..3
            int base = act_group_id * 2;

            // Gather 4 values for split-half packing: {elem[base], elem[base+16], elem[base+1], elem[base+17]}
            float val0 = __shfl_sync(0xFFFFFFFF, scaled_val, base, 32);
            float val1 = __shfl_sync(0xFFFFFFFF, scaled_val, base + 16, 32);
            float val2 = __shfl_sync(0xFFFFFFFF, scaled_val, base + 1, 32);
            float val3 = __shfl_sync(0xFFFFFFFF, scaled_val, base + 17, 32);

            if (act_lane_in_grp == 0) {
                __nv_fp4x4_e2m1 fp4_packed(make_float4(val0, val1, val2, val3));
                // Store 2 bytes (4 FP4 nibbles) into activation tile
                // Offset: g * 16 bytes + group_id * 2 bytes
                *(uint16_t*)(act_tile + g * 16 + act_group_id * 2) = *(uint16_t*)&fp4_packed;
            }
        }
        __syncwarp();

        // ── 2. Load and repack weight data for 16 rows × 64 K elements ──
        // Total: 16 rows × 32 bytes (64 nibbles) = 512 bytes sequential
        // Need to repack each 32-elem block from sequential to split-half
        // 16 rows × 2 blocks × 16 bytes = 512 bytes
        // Use all 32 threads to cooperatively load/repack
        // Each thread handles 16 bytes (one 32-elem block)
        // 32 threads × 16 bytes = 512 bytes = exactly what we need

        int my_block = lane_id;  // 0..31: covers 16 rows × 2 blocks
        int my_row = my_block / 2;
        int my_blk = my_block % 2;
        int global_row = tile_row + my_row;

        if (global_row < M_slice) {
            // Source: sequential format at row global_row, k_offset = k0 + my_blk*32
            int k_offset = k0 + my_blk * 32;
            const uint8_t* src = weight + (long long)global_row * packed_k + k_offset / 2;
            // Destination: split-half format in shared memory
            uint8_t* dst = wt_tile + my_row * 32 + my_blk * 16;
            repack_seq_to_split(src, dst);
        } else {
            // Zero-pad for rows beyond M_slice
            uint8_t* dst = wt_tile + my_row * 32 + my_blk * 16;
            #pragma unroll
            for (int i = 0; i < 4; i++) {
                ((uint32_t*)dst)[i] = 0;
            }
        }
        __syncwarp();

        // ── 3. Load A (weight) registers from shared memory ──
        // PTX ISA m16n8k64 physical register layout (NOT llama.cpp's load_generic):
        //   reg0: row groupID,   8 FP4 elems from K block 0 (K[0..31])
        //   reg1: row groupID+8, 8 FP4 elems from K block 0 (K[0..31])
        //   reg2: row groupID,   8 FP4 elems from K block 1 (K[32..63])
        //   reg3: row groupID+8, 8 FP4 elems from K block 1 (K[32..63])
        //
        // Shared memory layout: wt_tile[row * 32 + byte_offset]
        //   Bytes 0-15 (int32[0..3]) = K block 0 (32 FP4 in split-half)
        //   Bytes 16-31 (int32[4..7]) = K block 1 (32 FP4 in split-half)
        //
        // Each tid_in_grp (0-3) reads one int32 (4 bytes = 8 FP4 nibbles) per block.
        // Note: llama.cpp uses ldmatrix for hardware rearrangement; we load manually.

        const int* wt_qs = (const int*)wt_tile;
        const int wt_stride = 8; // 32 bytes per row / 4 bytes per int32
        // Block 0 (K[0..31]) = int32[0..3], Block 1 (K[32..63]) = int32[4..7]
        int a0 = wt_qs[groupID * wt_stride + tid_in_grp];           // PTX reg0: row groupID, K block 0
        int a1 = wt_qs[(groupID + 8) * wt_stride + tid_in_grp];    // PTX reg1: row groupID+8, K block 0
        int a2 = wt_qs[groupID * wt_stride + 4 + tid_in_grp];      // PTX reg2: row groupID, K block 1
        int a3 = wt_qs[(groupID + 8) * wt_stride + 4 + tid_in_grp]; // PTX reg3: row groupID+8, K block 1

        // ── 4. Load B (activation) registers from shared memory ──
        // B is 64×8 col-major. For GEMV, all 8 columns are identical.
        // Register 0: rows=[tid_in_grp*8..tid_in_grp*8+7] of k[0:31], col=groupID
        // Register 1: rows=[tid_in_grp*8+32..tid_in_grp*8+39] of k[32:63], col=groupID
        //
        // Since B is the activation vector (same for all cols), the activation tile
        // has 32 bytes total (64 FP4 nibbles in split-half, 2 blocks of 16 bytes).
        // The B register needs the data at the right k-positions.
        //
        // For the activation, we stored it like llama.cpp's format:
        //   act_tile[0..15] = first 32-elem block (k[0:31]) in split-half
        //   act_tile[16..31] = second 32-elem block (k[32:63]) in split-half
        //
        // B reg 0 needs k-positions [tid_in_grp*8..tid_in_grp*8+7] from block 0
        // In split-half int32 array: position tid_in_grp*2 and tid_in_grp*2+1
        // But wait — B has shape 8×64 transposed, and each thread loads based on its groupID (column)
        // For GEMV all cols same, so we just broadcast.
        //
        // Actually for B with col-major 64×8:
        //   B reg 0 (b0..b7): rows = tid_in_grp*8..tid_in_grp*8+7, col = groupID
        //   B reg 1 (b8..b15): rows = tid_in_grp*8+32..tid_in_grp*8+39, col = groupID
        //
        // Since all columns are the same vector, we can load from the activation tile:
        // The split-half block has 16 bytes = 4 int32.
        // int32[i] covers bytes 4i..4i+3 which in split-half covers elements at specific positions.
        //
        // For B, the load pattern from llama.cpp uses load_generic for tile<8,8,int>:
        //   y_qs loaded at [j * stride + k_offset] where j indexes B columns
        //   For GEMV, all j give same data.
        //
        // The B fragment for each thread needs:
        //   reg 0: 4 bytes covering k-positions tid_in_grp*8..tid_in_grp*8+7 from block 0
        //   reg 1: 4 bytes covering k-positions tid_in_grp*8+32..tid_in_grp*8+39 from block 1
        //
        // In split-half format, the 16 bytes for a 32-elem block are indexed as:
        //   int32[t] at byte offset 4t covers: {elem[2t], elem[2t+16], elem[2t+1], elem[2t+17]}
        //
        // But the B register layout expects 8 consecutive k-positions per register.
        // With split-half, tid_in_grp*8 consecutive elements don't map to a single int32.
        //
        // Actually, I think for the B operand, we need the same layout as the activation
        // tile that was created in the same split-half format as llama.cpp's quantize_mmq_mxfp4.
        // The MMA hardware knows how to interpret the split-half packed data from each register.
        //
        // For B with tile<8,8,int>, ne=2, and the NVIDIA path:
        //   get_i(l) = threadIdx.x / 4 = groupID (but B's i is the K dimension)
        //   get_j(l) = (l * 4) + (threadIdx.x % 4) = l*4 + tid_in_grp
        //   stride = ...
        //
        // Wait — for B in the MMA, the data is loaded per-column (N dimension),
        // and within each column, the K elements are distributed.
        // For GEMV all N-columns are the same, so we can use any column.
        //
        // Let me just load B the same way as llama.cpp: from a per-column activation buffer.
        // The activation tile has format matching block_fp4_mmq.qs (split-half, 32 bytes for 64 elems).
        //
        // For tile<8,8,int> B operand loading (from llama.cpp mmq.cuh line 1053):
        //   load_generic(B, y_qs + j0 * stride_y + k01, stride_y)
        //   where y_qs starts at offset 4 (past the scales header) in block_fp4_mmq
        //
        // For B, ne=2 (registers per thread), and:
        //   b0: i=groupID, j=tid_in_grp  → load from y_qs[groupID * stride_y + tid_in_grp]
        //   b1: i=groupID, j=tid_in_grp + 4 → load from y_qs[groupID * stride_y + tid_in_grp + 4]
        //
        // Since our activation is the same for all columns, we can set stride_y = 0 for GEMV
        // and load: b0 = act_int32[tid_in_grp], b1 = act_int32[tid_in_grp + 4]

        const int* act_qs = (const int*)act_tile;
        int b0 = act_qs[tid_in_grp];      // k[0:31] portion
        int b1 = act_qs[tid_in_grp + 4];  // k[32:63] portion

        // ── 5. Compute scale registers ──
        // scale_A: 2 E8M0 bytes packed in uint32 (for K=64 = 2 groups of 32)
        // Each thread needs the scale for its rows (groupID and groupID+8)
        //
        // Our BF16 scales need to be converted to E8M0 for the MMA.
        // BF16 scale was created as (e8m0 << 7), so to recover E8M0: (bf16_bits >> 7) & 0xFF
        //
        // For scale_A with scale_vec::2X:
        // Thread pair (laneid%4 == 0 or 1) with thread-id-a=0 supplies scales for one row
        // Thread pair (laneid%4 == 2 or 3) with thread-id-a=1 supplies scales for another row
        //
        // Based on llama.cpp (line 1040-1042):
        //   tidx = threadIdx.x / 4 + (threadIdx.x % 2) * 8
        //   scaleA = *(x_sc + (row + tidx) * stride + k_block_offset)
        //
        // For our case, we need per-row scales for the current K chunk.
        // The scale for row r, group g (0 or 1 for k0..k0+31 and k0+32..k0+63) is:
        //   scales[r * num_groups + (k0/32 + g)]
        //
        // We need to pack 2 scales (for the 2 groups in K=64) into one uint32.
        // For each row this thread is responsible for:
        int tidx = groupID + (tid_in_grp & 1) * 8;
        int scale_row = tile_row + tidx;
        uint32_t sa = 0;
        if (scale_row < M_slice) {
            int g0 = k0 / 32;
            unsigned short bf16_s0 = scales[(long long)scale_row * num_groups + g0];
            unsigned short bf16_s1 = scales[(long long)scale_row * num_groups + g0 + 1];
            uint8_t e0 = (bf16_s0 >> 7) & 0xFF;
            uint8_t e1 = (bf16_s1 >> 7) & 0xFF;
            sa = (uint32_t)e0 | ((uint32_t)e1 << 8);
        }

        // scale_B: 2 E8M0 bytes for the activation scales
        // For scale_vec::2X, scale_B has shape 2×N.
        // With thread-id-b selecting which thread provides the scale.
        // For GEMV all columns same, so all threads provide the same scales.
        uint32_t sb = (uint32_t)act_scales[0] | ((uint32_t)act_scales[1] << 8);

        // ── 6. Issue the MMA instruction ──
#if __CUDA_ARCH__ >= 1200
        asm volatile(
            "mma.sync.aligned.kind::mxf4.block_scale.scale_vec::2X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue8m0 "
            "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, "
            "%10, {0, 0}, %11, {0, 0};"
            : "+f"(c0), "+f"(c1), "+f"(c2), "+f"(c3)
            : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
              "r"(b0), "r"(b1),
              "r"(sa), "r"(sb)
        );
#endif // __CUDA_ARCH__ >= 1200
    }

    // ── 7. Write output ──
    // C layout: c0 = row groupID, col tid_in_grp*2
    //           c1 = row groupID, col tid_in_grp*2+1
    //           c2 = row groupID+8, col tid_in_grp*2
    //           c3 = row groupID+8, col tid_in_grp*2+1
    //
    // For GEMV, all 8 columns should give the same result.
    // We can pick column 0 (tid_in_grp=0, using c0 and c2) and write.
    // But to be safe, let's have tid_in_grp=0 write both rows.
    if (tid_in_grp == 0) {
        int row0 = tile_row + groupID;
        int row1 = tile_row + groupID + 8;
        if (row0 < M_slice) output[row0] = c0;
        if (row1 < M_slice) output[row1] = c2;
    }
}

// ─── MMA Launcher Functions ──────────────────────────────────────────────

static bool mma_selftest_done = false;

int vib3_launch_gemv_mma_nvfp4(
    const void* input, const void* weight_packed, const void* scales,
    void* output, int K, int M_slice, void* stream
) {
    int dev;
    cudaGetDevice(&dev);
    int major;
    cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev);

    if (major >= 12) {
        cudaStream_t s = (cudaStream_t)stream;

        // Run self-test once on first call
        if (!mma_selftest_done) {
            mma_selftest_done = true;
            float* d_results;
            cudaMalloc(&d_results, 512 * sizeof(float));
            cudaMemset(d_results, 0, 512 * sizeof(float));
            vib3_mma_selftest_kernel<<<1, 32, 0, s>>>(d_results);
            cudaStreamSynchronize(s);
            cudaError_t err = cudaGetLastError();

            float h_results[512];
            cudaMemcpy(h_results, d_results, 512 * sizeof(float), cudaMemcpyDeviceToHost);

            // Verify non-uniform test (most sensitive check)
            float row0 = h_results[176 + 0*4 + 0]; // groupID=0, c0 = row 0
            float row8 = h_results[176 + 0*4 + 2]; // groupID=0, c2 = row 8
            bool selftest_ok = (err == cudaSuccess && row0 == 96.0f && row8 == 64.0f);
            if (!selftest_ok) {
                fprintf(stderr, "[MMA SELFTEST FAILED] err=%d row0=%.1f(exp 96) row8=%.1f(exp 64) — falling back to software\n",
                        (int)err, row0, row8);
                cudaFree(d_results);
                return vib3_launch_partial_matmul_nvfp4_f32out(
                    input, weight_packed, scales, output, K, M_slice, 32, stream);
            }
            fprintf(stderr, "[MMA] selftest PASS — using Blackwell MMA path\n");
            cudaFree(d_results);
        }

        // Launch MMA GEMV kernel
        const int warps_per_block = 4;
        dim3 block(32, warps_per_block);
        int tiles = (M_slice + 15) / 16;
        int grid_x = (tiles + warps_per_block - 1) / warps_per_block;
        int smem = warps_per_block * (512 + 64);

        vib3_gemv_mma_nvfp4_kernel<<<grid_x, block, smem, s>>>(
            (const float*)input,
            (const uint8_t*)weight_packed,
            (const unsigned short*)scales,
            (float*)output,
            K, M_slice
        );
        return 0;
    }

    // Fallback for pre-Blackwell: software dequant
    return vib3_launch_partial_matmul_nvfp4_f32out(
        input, weight_packed, scales, output, K, M_slice, 32, stream);
}

// Forward declaration of norepack kernel (defined later in file)
__global__ void vib3_gemv_mma_nvfp4_preq_norepack_kernel(
    const uint8_t* __restrict__ act_fp4,
    const uint8_t* __restrict__ act_scales,
    const uint8_t* __restrict__ weight,
    const unsigned short* __restrict__ scales,
    float* __restrict__ output,
    int K,
    int M_slice
);

// ── Pre-quantized MMA GEMV launcher ──
// Like vib3_launch_gemv_mma_nvfp4 but takes pre-quantized activations.
// No self-test (assumes it already passed from the regular MMA launcher).
int vib3_launch_gemv_mma_nvfp4_preq(
    const void* act_fp4, const void* act_scales,
    const void* weight_packed, const void* scales,
    void* output, int K, int M_slice, void* stream
) {
    int dev;
    cudaGetDevice(&dev);
    int major;
    cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev);

    if (major >= 12) {
        cudaStream_t s = (cudaStream_t)stream;

        const int warps_per_block = 4;
        dim3 block(32, warps_per_block);
        int tiles = (M_slice + 15) / 16;
        int grid_x = (tiles + warps_per_block - 1) / warps_per_block;
        // Use norepack kernel: expert pages are already in split-half format
        // (repacked at startup by repack_weights_inplace_kernel)
        vib3_gemv_mma_nvfp4_preq_norepack_kernel<<<grid_x, block, 0, s>>>(
            (const uint8_t*)act_fp4,
            (const uint8_t*)act_scales,
            (const uint8_t*)weight_packed,
            (const unsigned short*)scales,
            (float*)output,
            K, M_slice
        );
        return 0;
    }

    // Fallback: can't use pre-quantized without MMA
    return -1;
}

// SwiGLU fusion kernel: output[i] = SiLU(gate[i]) * up[i]
// SiLU(x) = x * sigmoid(x) = x / (1 + exp(-x))
__global__ void vib3_swiglu_fuse_f32_kernel(
    const float* __restrict__ gate,
    const float* __restrict__ up,
    float* __restrict__ output,
    int N
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        float g = gate[i];
        float sigmoid_g = 1.0f / (1.0f + expf(-g));
        output[i] = g * sigmoid_g * up[i];
    }
}

// Static temp buffers for SwiGLU MMA (allocated once)
static float* swiglu_up_buf = nullptr;
static float* swiglu_gate_buf = nullptr;
static int swiglu_buf_size = 0;

int vib3_launch_fused_swiglu_mma_nvfp4(
    const void* input, const void* up_weight, const void* up_scales,
    const void* gate_weight, const void* gate_scales,
    void* output, int K, int M_slice, void* stream
) {
    int dev;
    cudaGetDevice(&dev);
    int major;
    cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev);

    if (major >= 12) {
        cudaStream_t s = (cudaStream_t)stream;

        // Ensure temp buffers are large enough
        if (M_slice > swiglu_buf_size) {
            if (swiglu_up_buf) cudaFree(swiglu_up_buf);
            if (swiglu_gate_buf) cudaFree(swiglu_gate_buf);
            swiglu_buf_size = M_slice;
            cudaMalloc(&swiglu_up_buf, swiglu_buf_size * sizeof(float));
            cudaMalloc(&swiglu_gate_buf, swiglu_buf_size * sizeof(float));
        }

        // Run MMA GEMV for up_proj
        int ret = vib3_launch_gemv_mma_nvfp4(input, up_weight, up_scales, swiglu_up_buf, K, M_slice, stream);
        if (ret != 0) goto fallback;

        // Run MMA GEMV for gate_proj
        ret = vib3_launch_gemv_mma_nvfp4(input, gate_weight, gate_scales, swiglu_gate_buf, K, M_slice, stream);
        if (ret != 0) goto fallback;

        // Fuse: output = SiLU(gate) * up
        int threads = 256;
        int blocks = (M_slice + threads - 1) / threads;
        vib3_swiglu_fuse_f32_kernel<<<blocks, threads, 0, s>>>(swiglu_gate_buf, swiglu_up_buf, (float*)output, M_slice);

        return 0;
    }

fallback:
    return vib3_launch_fused_swiglu_nvfp4_f32out(
        input, up_weight, up_scales, gate_weight, gate_scales,
        output, K, M_slice, 32, stream);
}

// ── Pre-quantized fused SwiGLU MMA ──
// Like vib3_launch_fused_swiglu_mma_nvfp4 but uses pre-quantized activations.
int vib3_launch_fused_swiglu_mma_nvfp4_preq(
    const void* act_fp4, const void* act_scales,
    const void* up_weight, const void* up_scales,
    const void* gate_weight, const void* gate_scales,
    void* output, int K, int M_slice, void* stream
) {
    int dev;
    cudaGetDevice(&dev);
    int major;
    cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev);

    if (major >= 12) {
        cudaStream_t s = (cudaStream_t)stream;

        // Ensure temp buffers
        if (M_slice > swiglu_buf_size) {
            if (swiglu_up_buf) cudaFree(swiglu_up_buf);
            if (swiglu_gate_buf) cudaFree(swiglu_gate_buf);
            swiglu_buf_size = M_slice;
            cudaMalloc(&swiglu_up_buf, swiglu_buf_size * sizeof(float));
            cudaMalloc(&swiglu_gate_buf, swiglu_buf_size * sizeof(float));
        }

        // Run pre-quantized MMA GEMV for up_proj
        int ret = vib3_launch_gemv_mma_nvfp4_preq(act_fp4, act_scales, up_weight, up_scales, swiglu_up_buf, K, M_slice, stream);
        if (ret != 0) return ret;

        // Run pre-quantized MMA GEMV for gate_proj
        ret = vib3_launch_gemv_mma_nvfp4_preq(act_fp4, act_scales, gate_weight, gate_scales, swiglu_gate_buf, K, M_slice, stream);
        if (ret != 0) return ret;

        // Fuse: output = SiLU(gate) * up
        int threads = 256;
        int blocks = (M_slice + threads - 1) / threads;
        vib3_swiglu_fuse_f32_kernel<<<blocks, threads, 0, s>>>(swiglu_gate_buf, swiglu_up_buf, (float*)output, M_slice);

        return 0;
    }

    return -1;  // pre-quantized requires MMA
}

// ─── GPU-side Router Top-K ──────────────────────────────────────────────────
// Fused softmax/sigmoid + top-k selection + renormalization.
// Eliminates the stream.synchronize() + D2H + CPU top-k bottleneck.
//
// Launch: 1 block of num_experts threads (must be power-of-2, typically 256).
// Output: top_k expert IDs (uint16) + weights (float), ready for small D2H.

__device__ __forceinline__ float warp_reduce_max(float val) {
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        float other = __shfl_down_sync(0xffffffff, val, offset);
        val = fmaxf(val, other);
    }
    return val;
}

// Bitonic sort step: compare-and-swap for (key, index) pairs in shared memory.
// Sorts in DESCENDING order by key (highest scores first).
__device__ __forceinline__ void bitonic_cas_descending(
    float* __restrict__ s_keys,
    int* __restrict__ s_indices,
    int i, int j
) {
    if (s_keys[i] < s_keys[j]) {
        float tk = s_keys[i]; s_keys[i] = s_keys[j]; s_keys[j] = tk;
        int ti = s_indices[i]; s_indices[i] = s_indices[j]; s_indices[j] = ti;
    }
}

// Fused router top-k kernel.
//
// scoring_mode: 0 = softmax (full softmax over all experts, top-k, renormalize)
//               1 = sigmoid (per-expert sigmoid, optional bias for selection, top-k, normalize+scale)
//
// For softmax: bias is ignored, scaling_factor is ignored.
// For sigmoid: bias is added to sigmoid scores for SELECTION only (top-k picking),
//              but the output weights use un-biased sigmoid scores.
//              Then if normalize: weights /= sum(weights), then weights *= scaling_factor.
__global__ void vib3_router_topk_kernel(
    const float* __restrict__ raw_scores,  // [num_experts] from router GEMV
    const __half* __restrict__ bias,       // [num_experts] FP16 aux-free bias, or NULL
    unsigned short* __restrict__ out_ids,   // [top_k] output expert IDs
    float* __restrict__ out_weights,        // [top_k] output routing weights
    int num_experts,
    int num_experts_padded,                 // next pow2 ≥ num_experts; shared-mem + block size
    int top_k,
    int scoring_mode,                       // 0=softmax, 1=sigmoid
    float scaling_factor                    // only for sigmoid mode
) {
    // Shared memory for bitonic sort: keys (selection scores) + indices.
    // Allocated for num_experts_padded so the bitonic sort never goes out of
    // bounds when num_experts is not a power of 2 (Kimi K2.6 has 384 experts;
    // padded to 512). Padding slots are filled with -INF keys so they lose
    // every compare-and-swap and can never appear in the top-k output.
    extern __shared__ char smem_raw[];
    float* s_keys = (float*)smem_raw;
    int* s_indices = (int*)(s_keys + num_experts_padded);
    // For sigmoid: also store un-biased sigmoid scores for final weight output
    float* s_sigmoid_scores = (float*)(s_indices + num_experts_padded);

    int tid = threadIdx.x;

    if (scoring_mode == 0) {
        // ── Softmax mode ──
        // CRITICAL: Thread 0 computes softmax sequentially to match CPU
        // floating-point summation order exactly. Warp-tree reductions
        // produce slightly different sums due to FP associativity,
        // which compounds across 48 MoE layers into wrong output.

        // Step 1: All threads load raw scores into shared memory. Padded
        // slots (tid >= num_experts) get -INF so they fall out of top-k.
        s_keys[tid] = (tid < num_experts) ? raw_scores[tid] : -1e30f;
        __syncthreads();

        // Step 2: Thread 0 computes softmax sequentially (matches CPU order)
        if (tid == 0) {
            // Find max (sequential, matches Rust iter().fold(NEG_INF, max))
            float global_max = s_keys[0];
            for (int i = 1; i < num_experts; i++) {
                if (s_keys[i] > global_max) global_max = s_keys[i];
            }

            // Compute exp(x - max) and sum (sequential, matches Rust iter().sum())
            float global_sum = 0.0f;
            for (int i = 0; i < num_experts; i++) {
                float e = expf(s_keys[i] - global_max);
                s_keys[i] = e;
                global_sum += e;
            }

            // Divide by sum to get probabilities
            if (global_sum > 0.0f) {
                float inv_sum = 1.0f / global_sum;
                for (int i = 0; i < num_experts; i++) {
                    s_keys[i] *= inv_sum;
                }
            }
            // Fill padded region with -INF so it never wins a compare-and-swap
            for (int i = num_experts; i < num_experts_padded; i++) {
                s_keys[i] = -1e30f;
            }
        }
        __syncthreads();

        // All threads set up indices for bitonic sort
        s_indices[tid] = tid;
        __syncthreads();

    } else {
        // ── Sigmoid mode ──
        bool live = (tid < num_experts);
        float raw_val = live ? raw_scores[tid] : 0.0f;
        float sig = live ? (1.0f / (1.0f + expf(-raw_val))) : 0.0f;
        s_sigmoid_scores[tid] = sig;  // store un-biased score for final weights

        // Selection score = sigmoid + bias (if present). Padded threads
        // (tid >= num_experts) get -INF so they never appear in top-k.
        // Bias is stored as FP16 in the .vib3 sidecar (DeepSeek-V3 /
        // Kimi K2.6 exp_probs_b.bias is tiny so FP16 is adequate).
        float sel;
        if (live) {
            sel = sig;
            if (bias != NULL) sel += __half2float(bias[tid]);
        } else {
            sel = -1e30f;
        }

        s_keys[tid] = sel;
        s_indices[tid] = tid;
    }
    __syncthreads();

    // ── Bitonic sort (descending) over num_experts_padded elements ──
    // num_experts_padded is power of 2 by construction. Real experts occupy
    // the first num_experts slots; padding slots carry -INF so they sink to
    // the bottom of the sorted order and never appear in the top-k output.
    for (int k = 2; k <= num_experts_padded; k <<= 1) {
        for (int j = k >> 1; j > 0; j >>= 1) {
            int ixj = tid ^ j;
            if (ixj > tid) {
                // Ascending or descending based on position in bitonic sequence
                if ((tid & k) == 0) {
                    // ascending half → we want descending, so swap if keys[tid] < keys[ixj]
                    bitonic_cas_descending(s_keys, s_indices, tid, ixj);
                } else {
                    // descending half → swap if keys[tid] > keys[ixj]
                    bitonic_cas_descending(s_keys, s_indices, ixj, tid);
                }
            }
            __syncthreads();
        }
    }

    // ── Thread 0 writes top-k results ──
    if (tid == 0) {
        if (scoring_mode == 0) {
            // Softmax: extract top-k softmax probabilities, then renormalize
            // so they sum to 1.0.  llama.cpp passes norm_w=true for
            // Qwen3.5-35B-A3B (see qwen35moe.cpp:383), which divides
            // each weight by sum(weights) after top-k selection.
            float topk_sum = 0.0f;
            for (int i = 0; i < top_k; i++) {
                out_ids[i] = (unsigned short)s_indices[i];
                out_weights[i] = s_keys[i];
                topk_sum += s_keys[i];
            }
            if (topk_sum > 0.0f) {
                float inv_sum = 1.0f / topk_sum;
                for (int i = 0; i < top_k; i++) {
                    out_weights[i] *= inv_sum;
                }
            }
        } else {
            // Sigmoid: use un-biased sigmoid scores, normalize, scale
            float topk_sum = 0.0f;
            for (int i = 0; i < top_k; i++) {
                int idx = s_indices[i];
                float w = s_sigmoid_scores[idx];
                out_weights[i] = w;
                topk_sum += w;
            }
            // Normalize if needed (always normalize for now; Rust can skip scaling_factor=1)
            if (topk_sum > 0.0f) {
                float inv_sum = 1.0f / topk_sum;
                for (int i = 0; i < top_k; i++) {
                    out_weights[i] *= inv_sum;
                }
            }
            // Apply scaling factor
            for (int i = 0; i < top_k; i++) {
                out_ids[i] = (unsigned short)s_indices[i];
                out_weights[i] *= scaling_factor;
            }
        }
    }
}

// Launcher for the fused router GEMV + top-k kernel.
// Runs the existing router_gemv_f32 kernel, then the top-k kernel.
// Output is written to out_ids and out_weights on device (caller does small D2H).
int vib3_launch_router_topk(
    const void* hidden_state,      // [hidden_dim] FP32
    const void* router_weights,    // [num_experts × hidden_dim] FP16
    float* scores_buf,             // [num_experts] FP32 temp buffer (device)
    const __half* bias,            // [num_experts] FP16 aux-free bias, or NULL
    unsigned short* out_ids,       // [top_k] output (device)
    float* out_weights,            // [top_k] output (device)
    int hidden_dim,
    int num_experts,
    int top_k,
    int scoring_mode,              // 0=softmax, 1=sigmoid
    float scaling_factor,          // sigmoid only
    void* stream
) {
    cudaStream_t s = (cudaStream_t)stream;

    // Step 1: Router GEMV (tiled, multi-block)
    int blocks = (num_experts + ROWS_PER_BLOCK - 1) / ROWS_PER_BLOCK;
    vib3_router_gemv_f32<<<blocks, BLOCK_SIZE, 0, s>>>(
        (const float*)hidden_state, (const half*)router_weights, scores_buf,
        hidden_dim, num_experts
    );

    // Step 2: Top-k selection kernel. The bitonic sort requires a power-of-2
    // element count; num_experts for K2.6 is 384 (not pow2), so we round up
    // to the next power of 2 (512) and fill the padding slots with -INF so
    // they never appear in the top-k output. Up to block-size limit of 1024.
    int num_experts_padded = 1;
    while (num_experts_padded < num_experts) num_experts_padded <<= 1;
    // Shared memory: s_keys[N_pad] + s_indices[N_pad] + s_sigmoid_scores[N_pad]
    int smem_bytes =
        num_experts_padded * (sizeof(float) + sizeof(int) + sizeof(float));
    vib3_router_topk_kernel<<<1, num_experts_padded, smem_bytes, s>>>(
        scores_buf, bias, out_ids, out_weights,
        num_experts, num_experts_padded, top_k, scoring_mode, scaling_factor
    );

    return (int)cudaGetLastError();
}

// ═══════════════════════════════════════════════════════════════════════════
// OPTIMIZATION PASS: Cached capability, in-place repack, no-repack MMA,
// batched expert launcher.
// ═══════════════════════════════════════════════════════════════════════════

// ── Cached SM capability (eliminates cudaGetDevice + cudaDeviceGetAttribute per launch) ──
static int g_sm_major = -1;

static inline int get_sm_major() {
    if (__builtin_expect(g_sm_major >= 0, 1)) return g_sm_major;
    int dev;
    cudaGetDevice(&dev);
    int major;
    cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev);
    g_sm_major = major;
    return major;
}

// ── In-place repack kernel: sequential → split-half on all weight bytes ──
// Converts 32-nibble blocks from sequential byte[j]={elem[2j],elem[2j+1]}
// to split-half byte[j]={elem[j],elem[j+16]}.
// Processes 16 bytes at a time (one 32-nibble block).
// Launch with ceil(total_bytes / 16) threads.
__global__ void vib3_repack_weights_inplace_kernel(
    uint8_t* __restrict__ data,
    long long total_weight_bytes  // only the packed FP4 data portion, not scales
) {
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long block_start = idx * 16;
    if (block_start + 16 > total_weight_bytes) return;

    uint8_t* ptr = data + block_start;

    // Read 16 bytes
    uint8_t tmp[16];
    #pragma unroll
    for (int i = 0; i < 16; i++) tmp[i] = ptr[i];

    // Repack: sequential → split-half
    // Input:  byte j = {elem[2j] in low nibble, elem[2j+1] in high nibble}
    // Output: byte j = {elem[j] in low nibble, elem[j+16] in high nibble}
    #pragma unroll
    for (int j = 0; j < 8; j++) {
        uint8_t s0 = tmp[j];
        uint8_t s1 = tmp[j + 8];
        ptr[2 * j]     = (s0 & 0x0F) | ((s1 & 0x0F) << 4);
        ptr[2 * j + 1] = (s0 >> 4)   | (s1 & 0xF0);
    }
}

// Launcher: repack one weight page in-place.
// weight_data_ptr points to the FP4 packed data (NOT scales).
// weight_data_bytes is the size of just the FP4 data (packed_k * m_slice).
int vib3_launch_repack_weights_inplace(
    void* weight_data_ptr,
    long long weight_data_bytes,
    void* stream
) {
    cudaStream_t s = (cudaStream_t)stream;
    long long num_blocks_16 = (weight_data_bytes + 15) / 16;
    int threads = 256;
    int grid = (int)((num_blocks_16 + threads - 1) / threads);
    if (grid <= 0) return 0;
    vib3_repack_weights_inplace_kernel<<<grid, threads, 0, s>>>(
        (uint8_t*)weight_data_ptr, weight_data_bytes
    );
    return (int)cudaGetLastError();
}

// ── Tiled weight repack: row-major split-half → tiled layout ──
// Reorganizes FP4 weight data so that 16 consecutive rows' data for the same
// K-tile are stored contiguously. This enables perfectly coalesced loads in GEMV.
//
// Row-major layout:  weight[row * packed_k + k_byte]
// Tiled layout:      weight[(row_block * num_k_tiles + k_tile) * 512 + row_in_block * 32 + byte_in_tile]
//
// Where: row_block = row / 16, row_in_block = row % 16, k_tile = k / 64,
//        num_k_tiles = K / 64, and each tile = 16 rows × 32 bytes = 512 bytes.
//
// M must be divisible by 16, K must be divisible by 64.
// Uses a temp buffer (same size as data) for the reorder.
__global__ void vib3_repack_row_to_tiled_kernel(
    const uint8_t* __restrict__ src,   // row-major split-half [M, packed_k]
    uint8_t* __restrict__ dst,         // tiled sequential output [M, packed_k] (same total size)
    int M,
    int packed_k  // = K / 2
) {
    // Each thread copies one 32-byte chunk (one row's contribution to one K-tile)
    const int num_k_tiles = packed_k / 32;  // packed_k = K/2, each tile covers 64 K-elems = 32 bytes
    const int num_row_blocks = M / 16;
    const long long total_chunks = (long long)num_row_blocks * num_k_tiles * 16;  // = M * num_k_tiles

    long long chunk_idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (chunk_idx >= total_chunks) return;

    // Decompose chunk_idx into (row_block, k_tile, row_in_block)
    int row_in_block = (int)(chunk_idx % 16);
    long long tile_idx = chunk_idx / 16;
    int k_tile = (int)(tile_idx % num_k_tiles);
    int row_block = (int)(tile_idx / num_k_tiles);

    int row = row_block * 16 + row_in_block;

    // Source: row-major
    long long src_offset = (long long)row * packed_k + (long long)k_tile * 32;
    // Destination: tiled
    long long dst_offset = ((long long)row_block * num_k_tiles + k_tile) * 512 + (long long)row_in_block * 32;

    // Convert 32 bytes from split-half to sequential nibble packing.
    // Split-half:  byte j has {elem[j] lo, elem[j+16] hi} for j=0..15
    // Sequential:  byte j has {elem[2j] lo, elem[2j+1] hi} for j=0..15
    // The Blackwell MMA instruction expects sequential packing in registers.
    const uint8_t* s = src + src_offset;
    uint8_t* d = dst + dst_offset;

    #pragma unroll
    for (int half = 0; half < 2; half++) {
        const uint8_t* sh = s + half * 16;
        uint8_t* dh = d + half * 16;
        #pragma unroll
        for (int j = 0; j < 8; j++) {
            uint8_t b0 = sh[2 * j];
            uint8_t b1 = sh[2 * j + 1];
            dh[j]     = (b0 & 0x0F) | ((b1 & 0x0F) << 4);  // {elem[2j], elem[2j+1]}
            dh[j + 8] = (b0 >> 4)   | (b1 & 0xF0);          // {elem[2j+16], elem[2j+17]}
        }
    }
}

// Launcher: repack row-major → tiled layout using a temp buffer.
// After this call, the data at weight_data_ptr is in tiled layout.
// temp_buf must be at least weight_data_bytes in size.
extern "C" int vib3_launch_repack_row_to_tiled(
    void* weight_data_ptr,     // [M, packed_k] split-half row-major → tiled
    void* temp_buf,            // [M * packed_k] temp buffer (same size)
    int M,
    int K,
    void* stream
) {
    cudaStream_t s = (cudaStream_t)stream;
    int packed_k = K / 2;
    int num_k_tiles = packed_k / 32;
    int num_row_blocks = M / 16;
    long long total_chunks = (long long)num_row_blocks * num_k_tiles * 16;

    if (total_chunks <= 0) return 0;

    int threads = 256;
    int grid = (int)((total_chunks + threads - 1) / threads);

    // Step 1: row-major → tiled into temp_buf
    vib3_repack_row_to_tiled_kernel<<<grid, threads, 0, s>>>(
        (const uint8_t*)weight_data_ptr,
        (uint8_t*)temp_buf,
        M, packed_k
    );

    // Step 2: copy tiled data back to original location
    long long total_bytes = (long long)M * packed_k;
    cudaMemcpyAsync(weight_data_ptr, temp_buf, total_bytes, cudaMemcpyDeviceToDevice, s);

    return (int)cudaGetLastError();
}

// ── Tiled MMA GEMV kernel: reads from tiled weight layout ──
// Weights stored as: tile_base = (row_block * num_k_tiles + k_tile) * 512
// Within tile: 16 rows × 32 bytes contiguous.
// K-parallel decomposition: gridDim.y = K_BLOCKS, each block processes a K-range.
// Output uses atomicAdd for cross-block accumulation (output must be pre-zeroed).
// This increases occupancy from ~9 warps/SM to ~36+ warps/SM.
__global__ void vib3_gemv_mma_nvfp4_tiled_kernel(
    const uint8_t* __restrict__ act_fp4,    // [K/2] pre-quantized split-half FP4
    const uint8_t* __restrict__ act_scales,  // [K/32] E8M0 scales
    const uint8_t* __restrict__ weight,      // [M, K/2] TILED split-half NVFP4
    const unsigned short* __restrict__ scales, // [M, K/32] BF16 scales (still row-major)
    float* __restrict__ output,              // [M] FP32 (pre-zeroed)
    int K,
    int M_slice,
    int k_tiles_per_block                    // how many K-tiles each block processes
) {
    const int warp_id = threadIdx.y;
    const int lane_id = threadIdx.x;
    const int num_warps = blockDim.y;

    const int row_block = blockIdx.x * num_warps + warp_id;  // which 16-row block
    const int tile_row = row_block * 16;
    if (tile_row >= M_slice) return;

    const int packed_k = K / 2;
    const int num_k_tiles = packed_k / 32;
    const int num_groups = K / 32;

    // K-range for this block (gridDim.y direction)
    const int k_block_id = blockIdx.y;
    const int k_tile_start = k_block_id * k_tiles_per_block;
    int k_tile_end = k_tile_start + k_tiles_per_block;
    if (k_tile_end > num_k_tiles) k_tile_end = num_k_tiles;

    const int groupID = lane_id >> 2;     // 0-7: which row pair
    const int tid_in_grp = lane_id & 3;   // 0-3: which int32 within row

    // Scale row indices for this thread (MMA scale_vec::2X layout)
    int tidx = groupID + (tid_in_grp & 1) * 8;
    int scale_row = tile_row + tidx;

    float c0 = 0.0f, c1 = 0.0f, c2 = 0.0f, c3 = 0.0f;

    // Base pointer for this warp's tile sequence in tiled layout
    const int* wt_base = (const int*)(weight + (long long)row_block * num_k_tiles * 512);

    // Pre-compute scale base pointer
    const unsigned short* sr = (scale_row < M_slice)
        ? scales + (long long)scale_row * num_groups : nullptr;

    for (int k_tile = k_tile_start; k_tile < k_tile_end; k_tile++) {
        // ── Direct register loads from tiled layout ──
        const int* tile = wt_base + k_tile * 128;
        int a0 = tile[groupID * 8 + tid_in_grp];
        int a2 = tile[groupID * 8 + 4 + tid_in_grp];
        int a1 = tile[(groupID + 8) * 8 + tid_in_grp];
        int a3 = tile[(groupID + 8) * 8 + 4 + tid_in_grp];

        // ── Activation loads ──
        const int* act_qs = (const int*)(act_fp4 + k_tile * 32);
        int b0 = act_qs[tid_in_grp];
        int b1 = act_qs[tid_in_grp + 4];
        uint8_t as0 = act_scales[k_tile * 2];
        uint8_t as1 = act_scales[k_tile * 2 + 1];
        uint32_t sb = (uint32_t)as0 | ((uint32_t)as1 << 8);

        // ── Weight scales ──
        uint32_t sa = 0;
        if (sr) {
            int g0 = k_tile * 2;
            unsigned short bf16_s0 = sr[g0];
            unsigned short bf16_s1 = sr[g0 + 1];
            uint8_t e0 = (bf16_s0 >> 7) & 0xFF;
            uint8_t e1 = (bf16_s1 >> 7) & 0xFF;
            sa = (uint32_t)e0 | ((uint32_t)e1 << 8);
        }

#if __CUDA_ARCH__ >= 1200
        asm volatile(
            "mma.sync.aligned.kind::mxf4.block_scale.scale_vec::2X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue8m0 "
            "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, "
            "%10, {0, 0}, %11, {0, 0};"
            : "+f"(c0), "+f"(c1), "+f"(c2), "+f"(c3)
            : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
              "r"(b0), "r"(b1),
              "r"(sa), "r"(sb)
        );
#endif
    }

    // ── Write output via atomicAdd (K-parallel accumulation) ──
    if (tid_in_grp == 0) {
        int row0 = tile_row + groupID;
        int row1 = tile_row + groupID + 8;
        if (row0 < M_slice) atomicAdd(&output[row0], c0);
        if (row1 < M_slice) atomicAdd(&output[row1], c2);
    }
}

// ── Tiled GEMV launcher (K-parallel) ──
extern "C" int vib3_launch_gemv_mma_nvfp4_tiled(
    const void* act_fp4,
    const void* act_scales,
    const void* weight,       // TILED layout
    const void* scales,       // row-major BF16
    void* output,
    int K,
    int M_slice,
    void* stream
) {
    if (get_sm_major() < 12) return -1;
    cudaStream_t s = (cudaStream_t)stream;

    const int warps_per_block = 4;
    const int packed_k = K / 2;
    const int num_k_tiles = packed_k / 32;

    // Choose K-blocks to target ~32+ warps per SM for good occupancy
    // row_blocks = M_slice / 16, total_warps_base = row_blocks
    // Want total_blocks >= 84 SMs * 8 (for ~32 warps/SM at 4 warps/block)
    int row_blocks = (M_slice + 15) / 16;
    int grid_x = (row_blocks + warps_per_block - 1) / warps_per_block;

    // Target: at least 672 total blocks (84 SMs × 8 blocks/SM)
    // k_blocks = max(1, ceil(672 / grid_x))
    // But cap at num_k_tiles (each K-block needs at least 1 tile)
    int target_blocks = 84 * 8;  // 672
    int k_blocks = (target_blocks + grid_x - 1) / grid_x;
    if (k_blocks < 1) k_blocks = 1;
    if (k_blocks > num_k_tiles) k_blocks = num_k_tiles;
    int k_tiles_per_block = (num_k_tiles + k_blocks - 1) / k_blocks;

    // Zero output for atomicAdd accumulation
    cudaMemsetAsync(output, 0, M_slice * sizeof(float), s);

    dim3 block(32, warps_per_block);
    dim3 grid(grid_x, k_blocks);

    vib3_gemv_mma_nvfp4_tiled_kernel<<<grid, block, 0, s>>>(
        (const uint8_t*)act_fp4,
        (const uint8_t*)act_scales,
        (const uint8_t*)weight,
        (const unsigned short*)scales,
        (float*)output,
        K, M_slice, k_tiles_per_block
    );
    return (int)cudaGetLastError();
}

// ── Scalar GEMV kernel: K-dimension parallelism for maximum bandwidth ──
// Instead of MMA (which forces M-dim parallelism = scattered row access),
// this kernel assigns 1 row per thread block and parallelizes along K.
// All threads read contiguous bytes from the SAME row → coalesced access.
// Uses register-based FP4 dequant + FMA, warp shuffle + shared memory reduction.
//
// Key optimizations:
//   - 128 threads per block, 1 row per block → fully coalesced weight reads
//   - Vectorized loads (4 bytes per load via int32)
//   - FP4 dequant via device function (NOT constant memory LUT, which serializes)
//   - Combined scale pre-computation per group
//   - Unrolled inner loop for K=3072 (typical case)
//
#define SCALAR_NWARPS 4
#define SCALAR_THREADS (SCALAR_NWARPS * 32)

// FP4 E2M1 dequant via arithmetic (avoids constant memory serialization)
// bit3=sign, bit2-1=exp, bit0=man
// val = (sign ? -1 : 1) * 2^(exp-1) * (1 + man*0.5)
// Special: exp=0 → subnormal: val = (sign ? -1 : 1) * 0.5 * man
__device__ __forceinline__ float fp4_dequant(uint8_t nibble) {
    // LUT in registers — compiler will use immediate constants
    // Values: 0, 0.5, 1, 1.5, 2, 3, 4, 6, -0, -0.5, -1, -1.5, -2, -3, -4, -6
    constexpr float lut[16] = {
        0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
        -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f
    };
    return lut[nibble];
}

__global__ void vib3_gemv_scalar_nvfp4_kernel(
    const uint8_t* __restrict__ act_fp4,    // [K/2] pre-quantized split-half FP4
    const uint8_t* __restrict__ act_scales,  // [K/32] E8M0 scales
    const uint8_t* __restrict__ weight,      // [M_slice, K/2] ALREADY split-half NVFP4
    const unsigned short* __restrict__ scales, // [M_slice, K/32] BF16 scales
    float* __restrict__ output,              // [M_slice] FP32
    int K,
    int M_slice
) {
    const int row = blockIdx.x;
    if (row >= M_slice) return;

    const int tid = threadIdx.y * 32 + threadIdx.x;
    const int packed_k = K / 2;          // bytes of weight data per row
    const int num_groups = K / 32;       // number of 32-element groups per row

    const uint8_t* row_weight = weight + (long long)row * packed_k;
    const unsigned short* row_scales = scales + (long long)row * num_groups;

    float acc = 0.0f;

    // Process one group (16 bytes = 32 elements) at a time
    // With 128 threads, 8 threads per group, or groups processed in parallel
    // Each group: 16 bytes of weight, 16 bytes of activation, 1 BF16 scale, 1 E8M0 scale
    // 128 threads / 16 bytes per group = 8 groups in parallel per iteration
    // Total groups = K/32. Iterations = K/32 / 8 = K/256
    
    // Thread assignment: tid maps to byte within a group stride
    // tid = group_offset * 16 + byte_within_group
    // where group_offset = tid / 16, byte_within_group = tid % 16
    // Process groups: group = iteration * (SCALAR_THREADS/16) + tid/16
    
    const int bytes_per_iter = SCALAR_THREADS;  // 128 bytes = 8 groups
    const int groups_per_iter = bytes_per_iter / 16;  // 8 groups
    
    for (int byte_off = tid; byte_off < packed_k; byte_off += bytes_per_iter) {
        int group = byte_off >> 4;       // byte_off / 16
        int j = byte_off & 15;           // byte_off % 16

        // Vectorized weight load — single byte
        uint8_t w_byte = row_weight[byte_off];
        
        // Activation at same group/position
        uint8_t a_byte = act_fp4[byte_off];  // act layout is same: group*16 + j

        // Dequant both nibbles (register LUT, no serialization)
        float w0 = fp4_dequant(w_byte & 0xF);
        float w1 = fp4_dequant((w_byte >> 4) & 0xF);
        float a0 = fp4_dequant(a_byte & 0xF);
        float a1 = fp4_dequant((a_byte >> 4) & 0xF);

        // Combined scale = weight_scale * activation_scale
        float w_scale = bf16_to_float(row_scales[group]);
        float a_scale = e8m0_to_f32(act_scales[group]);
        
        acc += (w0 * a0 + w1 * a1) * w_scale * a_scale;
    }

    // Warp-level reduction
    acc = warp_reduce_sum(acc);

    // Inter-warp reduction via shared memory
    __shared__ float smem[SCALAR_NWARPS];
    const int warp_id = threadIdx.y;
    const int lane_id = threadIdx.x;

    if (lane_id == 0) {
        smem[warp_id] = acc;
    }
    __syncthreads();

    // Warp 0, lane 0 does final reduction and write
    if (warp_id == 0 && lane_id == 0) {
        float result = smem[0];
        for (int i = 1; i < SCALAR_NWARPS; i++) {
            result += smem[i];
        }
        output[row] = result;
    }
}

// ── Launcher for scalar GEMV ──
int vib3_launch_gemv_scalar_nvfp4(
    const void* act_fp4, const void* act_scales,
    const void* weight_packed, const void* scales,
    void* output, int K, int M_slice, void* stream
) {
    cudaStream_t s = (cudaStream_t)stream;
    dim3 block(32, SCALAR_NWARPS);  // 128 threads
    dim3 grid(M_slice);             // 1 block per row
    vib3_gemv_scalar_nvfp4_kernel<<<grid, block, 0, s>>>(
        (const uint8_t*)act_fp4,
        (const uint8_t*)act_scales,
        (const uint8_t*)weight_packed,
        (const unsigned short*)scales,
        (float*)output,
        K, M_slice
    );
    return 0;
}

// ── No-repack MMA GEMV kernel: weights already in split-half format ──
// Uses K-unrolling by 4 to exploit L1 cache line reuse: each cache line
// (128 bytes) covers exactly 4 consecutive K-tiles (4 × 32 bytes).
// After fetching the first tile, tiles 2-4 hit L1 cache instead of DRAM.
// This improves effective bandwidth from ~12.5% to ~50-100% of peak.
__global__ void vib3_gemv_mma_nvfp4_preq_norepack_kernel(
    const uint8_t* __restrict__ act_fp4,    // [K/2] pre-quantized split-half FP4
    const uint8_t* __restrict__ act_scales,  // [K/32] E8M0 scales
    const uint8_t* __restrict__ weight,      // [M_slice, K/2] ALREADY split-half NVFP4
    const unsigned short* __restrict__ scales, // [M_slice, K/32] BF16 scales
    float* __restrict__ output,              // [M_slice] FP32
    int K,
    int M_slice
) {
    const int warp_id = threadIdx.y;
    const int lane_id = threadIdx.x;
    const int num_warps = blockDim.y;

    const int tile_row = (blockIdx.x * num_warps + warp_id) * 16;
    if (tile_row >= M_slice) return;

    const int packed_k = K / 2;
    const int num_groups = K / 32;

    const int groupID = lane_id >> 2;
    const int tid_in_grp = lane_id & 3;

    // Pre-compute row base pointers (hoist out of loop)
    int row0 = tile_row + groupID;
    int row1 = tile_row + groupID + 8;
    const uint8_t* r0 = (row0 < M_slice) ? weight + (long long)row0 * packed_k : nullptr;
    const uint8_t* r1 = (row1 < M_slice) ? weight + (long long)row1 * packed_k : nullptr;

    // Scale row for this thread
    int tidx = groupID + (tid_in_grp & 1) * 8;
    int scale_row = tile_row + tidx;
    const unsigned short* sr = (scale_row < M_slice)
        ? scales + (long long)scale_row * num_groups : nullptr;

    float c0 = 0.0f, c1 = 0.0f, c2 = 0.0f, c3 = 0.0f;

    // Unrolled by 4: process 256 K-elements (4 MMA tiles of k=64) per iteration
    // Each cache line = 128 bytes = 4 tiles × 32 bytes, so this maximizes reuse.
    int k0 = 0;
    for (; k0 + 256 <= K; k0 += 256) {
        #pragma unroll 4
        for (int t = 0; t < 4; t++) {
            int kk = k0 + t * 64;
            int tile_idx = kk / 64;

            // Activation loads (small, likely L1-cached across all blocks)
            const int* act_qs = (const int*)(act_fp4 + tile_idx * 32);
            int b0 = act_qs[tid_in_grp];
            int b1 = act_qs[tid_in_grp + 4];
            uint8_t as0 = act_scales[tile_idx * 2];
            uint8_t as1 = act_scales[tile_idx * 2 + 1];
            uint32_t sb = (uint32_t)as0 | ((uint32_t)as1 << 8);

            // Weight loads — tiles 0-3 within the same cache line per row
            int kb0 = kk / 2;
            int kb1 = kk / 2 + 16;
            int a0 = 0, a1 = 0, a2 = 0, a3 = 0;
            if (r0) {
                a0 = ((const int*)(r0 + kb0))[tid_in_grp];
                a2 = ((const int*)(r0 + kb1))[tid_in_grp];
            }
            if (r1) {
                a1 = ((const int*)(r1 + kb0))[tid_in_grp];
                a3 = ((const int*)(r1 + kb1))[tid_in_grp];
            }

            // Weight scales
            uint32_t sa = 0;
            if (sr) {
                int g0 = kk / 32;
                unsigned short bf16_s0 = sr[g0];
                unsigned short bf16_s1 = sr[g0 + 1];
                uint8_t e0 = (bf16_s0 >> 7) & 0xFF;
                uint8_t e1 = (bf16_s1 >> 7) & 0xFF;
                sa = (uint32_t)e0 | ((uint32_t)e1 << 8);
            }

#if __CUDA_ARCH__ >= 1200
            asm volatile(
                "mma.sync.aligned.kind::mxf4.block_scale.scale_vec::2X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue8m0 "
                "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, "
                "%10, {0, 0}, %11, {0, 0};"
                : "+f"(c0), "+f"(c1), "+f"(c2), "+f"(c3)
                : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
                  "r"(b0), "r"(b1),
                  "r"(sa), "r"(sb)
            );
#endif
        }
    }

    // Handle remaining tiles (K not divisible by 256)
    for (; k0 < K; k0 += 64) {
        int tile_idx = k0 / 64;
        const int* act_qs = (const int*)(act_fp4 + tile_idx * 32);
        int b0 = act_qs[tid_in_grp];
        int b1 = act_qs[tid_in_grp + 4];
        uint8_t as0 = act_scales[tile_idx * 2];
        uint8_t as1 = act_scales[tile_idx * 2 + 1];
        uint32_t sb = (uint32_t)as0 | ((uint32_t)as1 << 8);

        int kb0 = k0 / 2;
        int kb1 = k0 / 2 + 16;
        int a0 = 0, a1 = 0, a2 = 0, a3 = 0;
        if (r0) {
            a0 = ((const int*)(r0 + kb0))[tid_in_grp];
            a2 = ((const int*)(r0 + kb1))[tid_in_grp];
        }
        if (r1) {
            a1 = ((const int*)(r1 + kb0))[tid_in_grp];
            a3 = ((const int*)(r1 + kb1))[tid_in_grp];
        }

        uint32_t sa = 0;
        if (sr) {
            int g0 = k0 / 32;
            unsigned short bf16_s0 = sr[g0];
            unsigned short bf16_s1 = sr[g0 + 1];
            uint8_t e0 = (bf16_s0 >> 7) & 0xFF;
            uint8_t e1 = (bf16_s1 >> 7) & 0xFF;
            sa = (uint32_t)e0 | ((uint32_t)e1 << 8);
        }

#if __CUDA_ARCH__ >= 1200
        asm volatile(
            "mma.sync.aligned.kind::mxf4.block_scale.scale_vec::2X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue8m0 "
            "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, "
            "%10, {0, 0}, %11, {0, 0};"
            : "+f"(c0), "+f"(c1), "+f"(c2), "+f"(c3)
            : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
              "r"(b0), "r"(b1),
              "r"(sa), "r"(sb)
        );
#endif
    }

    // ── Write output ──
    if (tid_in_grp == 0) {
        if (row0 < M_slice) output[row0] = c0;
        if (row1 < M_slice) output[row1] = c2;
    }
}

// ── Batched expert launcher ──
// Executes the ENTIRE expert computation in a single C call:
//   1. SwiGLU: MMA GEMV up_proj → temp, MMA GEMV gate_proj → temp, fuse
//   2. Quantize SwiGLU output to FP4
//   3. MMA GEMV down_proj
//   4. Weighted accumulate into layer output
//
// Uses pre-allocated static temp buffers. No cudaMalloc in hot path.
// No cudaGetDevice per call — uses cached capability.
// Uses norepack kernels — weights must be pre-repacked to split-half.

// Static temp buffers for batched expert (sized for max M_slice)
static float* g_expert_up_buf = nullptr;
static float* g_expert_gate_buf = nullptr;
static uint8_t* g_expert_down_act_fp4 = nullptr;
static uint8_t* g_expert_down_act_scales = nullptr;
static float* g_expert_down_out_buf = nullptr;
static int g_expert_buf_up_size = 0;    // max M_slice for up/gate
static int g_expert_buf_down_size = 0;  // max M_slice for down
static int g_expert_diag_call = 0;      // diagnostic counter

int vib3_launch_expert_batched(
    // Pre-quantized activation (shared across all experts in this layer)
    const void* act_fp4,           // [K_in/2] FP4 split-half
    const void* act_scales,        // [K_in/32] E8M0
    // Weight pages (already repacked to split-half)
    const void* up_weight,         // [M_mid, K_in/2] split-half NVFP4
    const void* up_scales,         // [M_mid, K_in/32] BF16
    const void* gate_weight,       // [M_mid, K_in/2] split-half NVFP4
    const void* gate_scales,       // [M_mid, K_in/32] BF16
    const void* down_weight,       // [M_out, K_mid/2] split-half NVFP4
    const void* down_scales,       // [M_out, K_mid/32] BF16
    // Output accumulation
    float* layer_output,           // [M_out] FP32, accumulated (+=)
    float expert_weight,           // routing weight for this expert
    // Dimensions
    int K_in,                      // hidden_dim (3072)
    int M_mid,                     // expert_hidden_dim (1024) — up/gate output rows
    int K_mid,                     // expert_hidden_dim (1024) — down input cols
    int M_out,                     // hidden_dim (3072) — down output rows
    void* stream
) {
    if (get_sm_major() < 12) return -1;
    cudaStream_t s = (cudaStream_t)stream;

    // Ensure temp buffers for up/gate (M_mid sized)
    if (M_mid > g_expert_buf_up_size) {
        if (g_expert_up_buf) cudaFree(g_expert_up_buf);
        if (g_expert_gate_buf) cudaFree(g_expert_gate_buf);
        g_expert_buf_up_size = M_mid;
        cudaMalloc(&g_expert_up_buf, M_mid * sizeof(float));
        cudaMalloc(&g_expert_gate_buf, M_mid * sizeof(float));
    }

    // Ensure temp buffers for down proj input quantization + output
    if (M_out > g_expert_buf_down_size) {
        if (g_expert_down_act_fp4) cudaFree(g_expert_down_act_fp4);
        if (g_expert_down_act_scales) cudaFree(g_expert_down_act_scales);
        if (g_expert_down_out_buf) cudaFree(g_expert_down_out_buf);
        g_expert_buf_down_size = M_out;
        // FP4 buffer for expert_hidden_dim input (K_mid sized, but allocate M_out for safety)
        cudaMalloc(&g_expert_down_act_fp4, (K_mid / 2 + 64) * sizeof(uint8_t));
        cudaMalloc(&g_expert_down_act_scales, (K_mid / 32 + 4) * sizeof(uint8_t));
        cudaMalloc(&g_expert_down_out_buf, M_out * sizeof(float));
    }

    const int warps_per_block = 4;
    dim3 block(32, warps_per_block);

    // ── Step 1: MMA GEMV up_proj (tiled layout) ──
    {
        int packed_k_in = K_in / 2;
        int num_k_tiles = packed_k_in / 32;
        int row_blocks = (M_mid + 15) / 16;
        int grid_x = (row_blocks + warps_per_block - 1) / warps_per_block;
        int target_blocks = 84 * 8;
        int k_blocks = (target_blocks + grid_x - 1) / grid_x;
        if (k_blocks < 1) k_blocks = 1;
        if (k_blocks > num_k_tiles) k_blocks = num_k_tiles;
        k_blocks = 1;  // DEBUG: force single k-block to eliminate atomicAdd
        int k_tiles_per_block = (num_k_tiles + k_blocks - 1) / k_blocks;

        cudaMemsetAsync(g_expert_up_buf, 0, M_mid * sizeof(float), s);
        dim3 grid(grid_x, k_blocks);
        vib3_gemv_mma_nvfp4_tiled_kernel<<<grid, block, 0, s>>>(
            (const uint8_t*)act_fp4,
            (const uint8_t*)act_scales,
            (const uint8_t*)up_weight,
            (const unsigned short*)up_scales,
            g_expert_up_buf,
            K_in, M_mid, k_tiles_per_block
        );
    }

    // ── Step 2: MMA GEMV gate_proj (tiled layout) ──
    {
        int packed_k_in = K_in / 2;
        int num_k_tiles = packed_k_in / 32;
        int row_blocks = (M_mid + 15) / 16;
        int grid_x = (row_blocks + warps_per_block - 1) / warps_per_block;
        int target_blocks = 84 * 8;
        int k_blocks = (target_blocks + grid_x - 1) / grid_x;
        if (k_blocks < 1) k_blocks = 1;
        if (k_blocks > num_k_tiles) k_blocks = num_k_tiles;
        k_blocks = 1;  // DEBUG: force single k-block to eliminate atomicAdd
        int k_tiles_per_block = (num_k_tiles + k_blocks - 1) / k_blocks;

        cudaMemsetAsync(g_expert_gate_buf, 0, M_mid * sizeof(float), s);
        dim3 grid(grid_x, k_blocks);
        vib3_gemv_mma_nvfp4_tiled_kernel<<<grid, block, 0, s>>>(
            (const uint8_t*)act_fp4,
            (const uint8_t*)act_scales,
            (const uint8_t*)gate_weight,
            (const unsigned short*)gate_scales,
            g_expert_gate_buf,
            K_in, M_mid, k_tiles_per_block
        );
    }

    // ── Diagnostic: dump up_proj and gate_proj GEMV output for first few calls ──
    if (g_expert_diag_call < 8) {
        cudaStreamSynchronize(s);
        float* h_up = (float*)malloc(M_mid * sizeof(float));
        float* h_gate = (float*)malloc(M_mid * sizeof(float));
        cudaMemcpy(h_up, g_expert_up_buf, M_mid * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_gate, g_expert_gate_buf, M_mid * sizeof(float), cudaMemcpyDeviceToHost);
        double up_l2 = 0, gate_l2 = 0;
        float up_max = 0, gate_max = 0;
        for (int i = 0; i < M_mid; i++) {
            up_l2 += (double)h_up[i] * h_up[i];
            gate_l2 += (double)h_gate[i] * h_gate[i];
            if (fabsf(h_up[i]) > up_max) up_max = fabsf(h_up[i]);
            if (fabsf(h_gate[i]) > gate_max) gate_max = fabsf(h_gate[i]);
        }
        up_l2 = sqrt(up_l2);
        gate_l2 = sqrt(gate_l2);
        fprintf(stderr, "[EXPERT_DIAG call=%d] up_proj: L2=%.6f max=%.6f first4=[%.6f,%.6f,%.6f,%.6f] | "
                        "gate_proj: L2=%.6f max=%.6f first4=[%.6f,%.6f,%.6f,%.6f] | K_in=%d M_mid=%d\n",
                g_expert_diag_call, up_l2, up_max, h_up[0], h_up[1], h_up[2], h_up[3],
                gate_l2, gate_max, h_gate[0], h_gate[1], h_gate[2], h_gate[3], K_in, M_mid);
        // Dump binary files for per-expert comparison with llama.cpp
        {
            char fname[256];
            snprintf(fname, sizeof(fname),
                     "/home/brian/code/vib3/dump/vib3_expert_up_call%d.bin",
                     g_expert_diag_call);
            FILE* f = fopen(fname, "wb");
            if (f) { fwrite(h_up, sizeof(float), M_mid, f); fclose(f); }

            snprintf(fname, sizeof(fname),
                     "/home/brian/code/vib3/dump/vib3_expert_gate_call%d.bin",
                     g_expert_diag_call);
            f = fopen(fname, "wb");
            if (f) { fwrite(h_gate, sizeof(float), M_mid, f); fclose(f); }
        }
        free(h_up);
        free(h_gate);
    }

    // ── CPU reference matmul for call 0: full dequant + multi-row check ──
    if (g_expert_diag_call == 0) {
        cudaStreamSynchronize(s);
        int packed_k_in = K_in / 2;
        int num_k_tiles = packed_k_in / 32;
        int num_groups = K_in / 32;
        int row_blocks = (M_mid + 15) / 16;

        // 1. Read ALL up_proj tiled weight data (all row_blocks)
        long long total_wt_bytes = (long long)row_blocks * num_k_tiles * 512;
        uint8_t* h_wt = (uint8_t*)malloc(total_wt_bytes);
        cudaMemcpy(h_wt, up_weight, total_wt_bytes, cudaMemcpyDeviceToHost);

        // 2. Read ALL activation FP4 data + scales
        int act_data_bytes = packed_k_in;
        int act_scale_bytes = K_in / 32;
        uint8_t* h_act = (uint8_t*)malloc(act_data_bytes);
        uint8_t* h_act_sc = (uint8_t*)malloc(act_scale_bytes);
        cudaMemcpy(h_act, act_fp4, act_data_bytes, cudaMemcpyDeviceToHost);
        cudaMemcpy(h_act_sc, act_scales, act_scale_bytes, cudaMemcpyDeviceToHost);

        // 3. Read ALL weight scales (M_mid rows × num_groups BF16 values)
        long long total_sc_bytes = (long long)M_mid * num_groups * 2;
        unsigned short* h_wt_sc = (unsigned short*)malloc(total_sc_bytes);
        cudaMemcpy(h_wt_sc, up_scales, total_sc_bytes, cudaMemcpyDeviceToHost);

        // FP4 E2M1 dequant table
        const float fp4_lut[16] = {
            0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
            -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f
        };

        auto e8m0_to_f = [](uint8_t e) -> float {
            if (e == 0) return 0.0f;
            uint32_t bits = (uint32_t)e << 23;
            float f; memcpy(&f, &bits, 4); return f;
        };
        auto bf16_to_f = [](unsigned short bf16) -> float {
            uint32_t bits = (uint32_t)bf16 << 16;
            float f; memcpy(&f, &bits, 4); return f;
        };

        // 4. Dequantize activation (sequential format)
        // act_fp4 is produced by vib3_quantize_activation_fp4_kernel:
        //   byte j = {elem[2j], elem[2j+1]} with j in [0, 31] per 64-elem tile.
        // bytes 0-15 use scale s0 (K 0-31), bytes 16-31 use scale s1 (K 32-63).
        float* act_deq = (float*)calloc(K_in, sizeof(float));
        for (int t = 0; t < num_k_tiles; t++) {
            float s0 = e8m0_to_f(h_act_sc[t * 2]);
            float s1 = e8m0_to_f(h_act_sc[t * 2 + 1]);
            for (int b = 0; b < 32; b++) {
                uint8_t byte = h_act[t * 32 + b];
                float as_scale = (b < 16) ? s0 : s1;
                act_deq[t * 64 + 2 * b] = fp4_lut[byte & 0xF] * as_scale;
                act_deq[t * 64 + 2 * b + 1] = fp4_lut[(byte >> 4) & 0xF] * as_scale;
            }
        }

        // 5. Compute CPU matmul for ALL rows and compare with GPU
        float* h_gpu_out = (float*)malloc(M_mid * sizeof(float));
        cudaMemcpy(h_gpu_out, g_expert_up_buf, M_mid * sizeof(float), cudaMemcpyDeviceToHost);

        double cpu_l2 = 0.0, gpu_l2 = 0.0, diff_l2 = 0.0, dot_cg = 0.0;
        int n_big_diff = 0;

        // Check specific rows: 0, 1, 255, 400, 500, 511
        int check_rows[] = {0, 1, 255, 400, 500, 511};
        for (int ri = 0; ri < 6; ri++) {
            int row = check_rows[ri];
            if (row >= M_mid) continue;
            int rb = row / 16;
            int row_in_block = row % 16;

            // Dequant this row from tiled layout
            double cpu_dot = 0.0;
            int n_nonzero_scales = 0;
            for (int t = 0; t < num_k_tiles; t++) {
                // Weight scale for this row, this k-tile
                float ws0 = bf16_to_f(h_wt_sc[row * num_groups + t * 2]);
                float ws1 = bf16_to_f(h_wt_sc[row * num_groups + t * 2 + 1]);

                // Check if scales are "normal" (not tiny)
                uint16_t sc_raw0 = h_wt_sc[row * num_groups + t * 2];
                if (sc_raw0 > 0x0100) n_nonzero_scales++;

                // Weight data: tiled offset = rb * num_k_tiles * 512 + t * 512 + row_in_block * 32
                // Split-half mapping per 64-elem tile:
                //   b in [0,15]: low->k=b,     high->k=b+16   (scale ws0)
                //   b in [16,31]: low->k=b+16, high->k=b+32   (scale ws1)
                long long wt_off = (long long)rb * num_k_tiles * 512 + (long long)t * 512 + row_in_block * 32;
                for (int b = 0; b < 32; b++) {
                    uint8_t byte = h_wt[wt_off + b];
                    float ws = (b < 16) ? ws0 : ws1;
                    float w0 = fp4_lut[byte & 0xF] * ws;
                    float w1 = fp4_lut[(byte >> 4) & 0xF] * ws;
                    int k0 = (b < 16) ? b : (b + 16);
                    int k1 = k0 + 16;
                    cpu_dot += (double)w0 * (double)act_deq[t * 64 + k0];
                    cpu_dot += (double)w1 * (double)act_deq[t * 64 + k1];
                }
            }

            fprintf(stderr, "[CPU_REF call=0] row=%d: CPU_dot=%.6f, GPU=%.6f, diff=%.6f, normal_scales=%d/%d, "
                            "wt_sc_bf16[0]=0x%04x\n",
                    row, cpu_dot, h_gpu_out[row], cpu_dot - h_gpu_out[row],
                    n_nonzero_scales, num_k_tiles,
                    h_wt_sc[row * num_groups]);
        }

        // Full CPU matmul for cosine similarity
        // Also track top-10 worst rows
        // Also compute swapped-scale version
        float worst_diff[10] = {0};
        int worst_row[10] = {0};
        float worst_cpu[10] = {0};
        float worst_gpu[10] = {0};
        double swap_l2 = 0.0, swap_dot_cg = 0.0;

        for (int row = 0; row < M_mid; row++) {
            int rb = row / 16;
            int rib = row % 16;
            double cpu_dot = 0.0;
            double swap_dot = 0.0;
            for (int t = 0; t < num_k_tiles; t++) {
                float ws0 = bf16_to_f(h_wt_sc[row * num_groups + t * 2]);
                float ws1 = bf16_to_f(h_wt_sc[row * num_groups + t * 2 + 1]);
                long long wt_off = (long long)rb * num_k_tiles * 512 + (long long)t * 512 + rib * 32;
                for (int b = 0; b < 32; b++) {
                    uint8_t byte = h_wt[wt_off + b];
                    float ws = (b < 16) ? ws0 : ws1;
                    float ws_swap = (b < 16) ? ws1 : ws0;
                    float w0 = fp4_lut[byte & 0xF] * ws;
                    float w1 = fp4_lut[(byte >> 4) & 0xF] * ws;
                    int k0 = (b < 16) ? b : (b + 16);
                    int k1 = k0 + 16;
                    cpu_dot += (double)w0 * (double)act_deq[t * 64 + k0];
                    cpu_dot += (double)w1 * (double)act_deq[t * 64 + k1];
                    // Swapped: ws0 ↔ ws1
                    float w0s = fp4_lut[byte & 0xF] * ws_swap;
                    float w1s = fp4_lut[(byte >> 4) & 0xF] * ws_swap;
                    swap_dot += (double)w0s * (double)act_deq[t * 64 + k0];
                    swap_dot += (double)w1s * (double)act_deq[t * 64 + k1];
                }
            }
            float cpu_val = (float)cpu_dot;
            float swap_val = (float)swap_dot;
            float gpu_val = h_gpu_out[row];
            cpu_l2 += cpu_val * cpu_val;
            gpu_l2 += gpu_val * gpu_val;
            dot_cg += cpu_val * gpu_val;
            diff_l2 += (cpu_val - gpu_val) * (cpu_val - gpu_val);
            swap_l2 += swap_val * swap_val;
            swap_dot_cg += swap_val * gpu_val;
            float ad = fabsf(cpu_val - gpu_val);
            if (ad > 0.1f) n_big_diff++;
            // Insert into worst-10 (simple insertion sort)
            for (int wi = 0; wi < 10; wi++) {
                if (ad > worst_diff[wi]) {
                    for (int wj = 9; wj > wi; wj--) {
                        worst_diff[wj] = worst_diff[wj-1];
                        worst_row[wj] = worst_row[wj-1];
                        worst_cpu[wj] = worst_cpu[wj-1];
                        worst_gpu[wj] = worst_gpu[wj-1];
                    }
                    worst_diff[wi] = ad;
                    worst_row[wi] = row;
                    worst_cpu[wi] = cpu_val;
                    worst_gpu[wi] = gpu_val;
                    break;
                }
            }
        }
        cpu_l2 = sqrt(cpu_l2);
        gpu_l2 = sqrt(gpu_l2);
        diff_l2 = sqrt(diff_l2);
        swap_l2 = sqrt(swap_l2);
        double cosine = (cpu_l2 > 0 && gpu_l2 > 0) ? dot_cg / (cpu_l2 * gpu_l2) : 0.0;
        double swap_cosine = (swap_l2 > 0 && gpu_l2 > 0) ? swap_dot_cg / (swap_l2 * gpu_l2) : 0.0;
        fprintf(stderr, "[CPU_REF call=0] FULL: CPU_L2=%.6f, GPU_L2=%.6f, diff_L2=%.6f, cosine=%.6f, SWAP_cosine=%.6f, big_diffs=%d/%d\n",
                cpu_l2, gpu_l2, diff_l2, cosine, swap_cosine, n_big_diff, M_mid);

        // Dump top-10 worst rows
        for (int wi = 0; wi < 10 && worst_diff[wi] > 0.001f; wi++) {
            int wr = worst_row[wi];
            int wrb = wr / 16;
            int wrib = wr % 16;
            // Check scale pattern for this row
            int n_normal = 0;
            int n_mismatched = 0;
            for (int t = 0; t < num_k_tiles; t++) {
                uint16_t s0 = h_wt_sc[wr * num_groups + t * 2];
                uint16_t s1 = h_wt_sc[wr * num_groups + t * 2 + 1];
                if (s0 > 0x0100) n_normal++;
                if (s0 != s1) n_mismatched++;
            }
            fprintf(stderr, "[CPU_REF call=0]   worst[%d]: row=%d (rb=%d,rib=%d) CPU=%.6f GPU=%.6f diff=%.6f normal_sc=%d/%d mismatch=%d/%d sc0=0x%04x\n",
                    wi, wr, wrb, wrib, worst_cpu[wi], worst_gpu[wi], worst_diff[wi],
                    n_normal, num_k_tiles, n_mismatched, num_k_tiles, h_wt_sc[wr * num_groups]);
        }
        // Also dump scale pairs for worst row in detail
        {
            int wr = worst_row[0];
            fprintf(stderr, "[CPU_REF call=0]   worst_row=%d scale_pairs:", wr);
            for (int t = 0; t < num_k_tiles && t < 8; t++) {
                fprintf(stderr, " [0x%04x,0x%04x]",
                        h_wt_sc[wr * num_groups + t * 2],
                        h_wt_sc[wr * num_groups + t * 2 + 1]);
            }
            fprintf(stderr, "\n");
        }

        // Per-tile CPU dot product breakdown for worst row and row 1
        // Also compute SWAPPED-scale version to test if scales are reversed in MMA
        {
            int diag_rows[] = {worst_row[0], 1};
            for (int dr = 0; dr < 2; dr++) {
                int wr = diag_rows[dr];
                if (wr >= M_mid) continue;
                int wrb = wr / 16;
                int wrib = wr % 16;
                double running = 0.0;
                double running_swap = 0.0;
                fprintf(stderr, "[CPU_REF call=0]   per_tile row=%d:", wr);
                for (int t = 0; t < num_k_tiles; t++) {
                    float ws0 = bf16_to_f(h_wt_sc[wr * num_groups + t * 2]);
                    float ws1 = bf16_to_f(h_wt_sc[wr * num_groups + t * 2 + 1]);
                    long long wt_off = (long long)wrb * num_k_tiles * 512 + (long long)t * 512 + wrib * 32;
                    double tile_dot = 0.0;
                    double tile_dot_swap = 0.0;
                    for (int b = 0; b < 32; b++) {
                        uint8_t byte = h_wt[wt_off + b];
                        float ws = (b < 16) ? ws0 : ws1;
                        float ws_swap = (b < 16) ? ws1 : ws0;
                        float w0 = fp4_lut[byte & 0xF] * ws;
                        float w1 = fp4_lut[(byte >> 4) & 0xF] * ws;
                        float w0s = fp4_lut[byte & 0xF] * ws_swap;
                        float w1s = fp4_lut[(byte >> 4) & 0xF] * ws_swap;
                        int k0 = (b < 16) ? b : (b + 16);
                        int k1 = k0 + 16;
                        tile_dot += (double)w0 * (double)act_deq[t * 64 + k0];
                        tile_dot += (double)w1 * (double)act_deq[t * 64 + k1];
                        tile_dot_swap += (double)w0s * (double)act_deq[t * 64 + k0];
                        tile_dot_swap += (double)w1s * (double)act_deq[t * 64 + k1];
                    }
                    running += tile_dot;
                    running_swap += tile_dot_swap;
                    if (t < 4 || t == num_k_tiles - 1) {
                        fprintf(stderr, " t%d=%.4f(s=0x%04x,0x%04x)",
                                t, tile_dot,
                                h_wt_sc[wr * num_groups + t * 2],
                                h_wt_sc[wr * num_groups + t * 2 + 1]);
                    }
                }
                fprintf(stderr, " total=%.6f SWAPPED=%.6f GPU=%.6f\n", running, running_swap, (double)h_gpu_out[wr]);
            }
        }

        // Dump act scales and a sample of weight scales for row 400
        fprintf(stderr, "[CPU_REF call=0]   act_scales[0..7]=%u,%u,%u,%u,%u,%u,%u,%u\n",
                h_act_sc[0], h_act_sc[1], h_act_sc[2], h_act_sc[3],
                h_act_sc[4], h_act_sc[5], h_act_sc[6], h_act_sc[7]);
        if (M_mid > 400) {
            fprintf(stderr, "[CPU_REF call=0]   wt_sc_row400[0..3]=0x%04x,0x%04x,0x%04x,0x%04x\n",
                    h_wt_sc[400 * num_groups], h_wt_sc[400 * num_groups + 1],
                    h_wt_sc[400 * num_groups + 2], h_wt_sc[400 * num_groups + 3]);
        }

        free(h_wt);
        free(h_act);
        free(h_act_sc);
        free(h_wt_sc);
        free(act_deq);
        free(h_gpu_out);
    }

    // ── Step 3: Fused SwiGLU: output = SiLU(gate) * up ──
    {
        int threads = 256;
        int blocks_swiglu = (M_mid + threads - 1) / threads;
        vib3_swiglu_fuse_f32_kernel<<<blocks_swiglu, threads, 0, s>>>(
            g_expert_gate_buf, g_expert_up_buf, g_expert_up_buf, M_mid
        );
        // Reuse g_expert_up_buf as SwiGLU output (in-place into up_buf)
    }

    // ── Diagnostic: dump SwiGLU output ──
    if (g_expert_diag_call < 8) {
        cudaStreamSynchronize(s);
        float* h_swiglu = (float*)malloc(M_mid * sizeof(float));
        cudaMemcpy(h_swiglu, g_expert_up_buf, M_mid * sizeof(float), cudaMemcpyDeviceToHost);
        double swiglu_l2 = 0;
        float swiglu_max = 0;
        for (int i = 0; i < M_mid; i++) {
            swiglu_l2 += (double)h_swiglu[i] * h_swiglu[i];
            if (fabsf(h_swiglu[i]) > swiglu_max) swiglu_max = fabsf(h_swiglu[i]);
        }
        swiglu_l2 = sqrt(swiglu_l2);
        fprintf(stderr, "[EXPERT_DIAG call=%d] swiglu: L2=%.6f max=%.6f first4=[%.6f,%.6f,%.6f,%.6f]\n",
                g_expert_diag_call, swiglu_l2, swiglu_max,
                h_swiglu[0], h_swiglu[1], h_swiglu[2], h_swiglu[3]);
        {
            char fname[256];
            snprintf(fname, sizeof(fname),
                     "/home/brian/code/vib3/dump/vib3_expert_swiglu_call%d.bin",
                     g_expert_diag_call);
            FILE* f = fopen(fname, "wb");
            if (f) { fwrite(h_swiglu, sizeof(float), M_mid, f); fclose(f); }
        }
        free(h_swiglu);
    }

    // ── Step 4: Quantize SwiGLU output for down_proj ──
    {
        int num_tiles = (K_mid + 63) / 64;
        vib3_quantize_activation_fp4_kernel<<<num_tiles, 32, 0, s>>>(
            g_expert_up_buf,
            g_expert_down_act_fp4,
            g_expert_down_act_scales,
            K_mid
        );
    }

    // ── Step 5: MMA GEMV down_proj (tiled layout) ──
    cudaMemsetAsync(g_expert_down_out_buf, 0, M_out * sizeof(float), s);
    {
        int packed_k_mid = K_mid / 2;
        int num_k_tiles = packed_k_mid / 32;
        int row_blocks = (M_out + 15) / 16;
        int grid_x = (row_blocks + warps_per_block - 1) / warps_per_block;
        int target_blocks = 84 * 8;
        int k_blocks = (target_blocks + grid_x - 1) / grid_x;
        if (k_blocks < 1) k_blocks = 1;
        if (k_blocks > num_k_tiles) k_blocks = num_k_tiles;
        k_blocks = 1;  // DEBUG: force single k-block to eliminate atomicAdd
        int k_tiles_per_block = (num_k_tiles + k_blocks - 1) / k_blocks;

        dim3 grid(grid_x, k_blocks);
        vib3_gemv_mma_nvfp4_tiled_kernel<<<grid, block, 0, s>>>(
            g_expert_down_act_fp4,
            g_expert_down_act_scales,
            (const uint8_t*)down_weight,
            (const unsigned short*)down_scales,
            g_expert_down_out_buf,
            K_mid, M_out, k_tiles_per_block
        );
    }

    // ── Diagnostic: dump down_proj output ──
    if (g_expert_diag_call < 8) {
        cudaStreamSynchronize(s);
        float* h_down = (float*)malloc(M_out * sizeof(float));
        cudaMemcpy(h_down, g_expert_down_out_buf, M_out * sizeof(float), cudaMemcpyDeviceToHost);
        double down_l2 = 0;
        float down_max = 0;
        for (int i = 0; i < M_out; i++) {
            down_l2 += (double)h_down[i] * h_down[i];
            if (fabsf(h_down[i]) > down_max) down_max = fabsf(h_down[i]);
        }
        down_l2 = sqrt(down_l2);
        fprintf(stderr, "[EXPERT_DIAG call=%d] down_proj: L2=%.6f max=%.6f first4=[%.6f,%.6f,%.6f,%.6f] | "
                        "expert_weight=%.6f K_mid=%d M_out=%d\n",
                g_expert_diag_call, down_l2, down_max,
                h_down[0], h_down[1], h_down[2], h_down[3],
                expert_weight, K_mid, M_out);
        {
            char fname[256];
            snprintf(fname, sizeof(fname),
                     "/home/brian/code/vib3/dump/vib3_expert_down_call%d.bin",
                     g_expert_diag_call);
            FILE* f = fopen(fname, "wb");
            if (f) { fwrite(h_down, sizeof(float), M_out, f); fclose(f); }
        }
        free(h_down);
        g_expert_diag_call++;
    }

    // ── Step 6: Weighted accumulate into layer output ──
    {
        int threads = 256;
        int blocks_acc = (M_out + threads - 1) / threads;
        vib3_weighted_accumulate_f32_f32_kernel<<<blocks_acc, threads, 0, s>>>(
            layer_output, g_expert_down_out_buf, expert_weight, M_out
        );
    }

    return 0;
}

// ── Cached-capability versions of existing launchers ──
// For kernels that aren't part of the batched expert path but still
// suffer from per-launch cudaGetDevice overhead.

int vib3_launch_gemv_mma_nvfp4_preq_fast(
    const void* act_fp4, const void* act_scales,
    const void* weight_packed, const void* scales,
    void* output, int K, int M_slice, void* stream
) {
    if (get_sm_major() < 12) return -1;
    cudaStream_t s = (cudaStream_t)stream;
    const int warps_per_block = 4;
    dim3 block(32, warps_per_block);
    int tiles = (M_slice + 15) / 16;
    int grid_x = (tiles + warps_per_block - 1) / warps_per_block;
    // Use norepack kernel: weights from fp16_to_nvfp4_weight are already split-half
    vib3_gemv_mma_nvfp4_preq_norepack_kernel<<<grid_x, block, 0, s>>>(
        (const uint8_t*)act_fp4,
        (const uint8_t*)act_scales,
        (const uint8_t*)weight_packed,
        (const unsigned short*)scales,
        (float*)output,
        K, M_slice
    );
    return 0;
}

int vib3_launch_quantize_activation_fp4_fast(
    const void* input, void* act_fp4, void* act_scales, int K, void* stream
) {
    if (get_sm_major() < 12) return -1;
    cudaStream_t s = (cudaStream_t)stream;
    int num_tiles = (K + 63) / 64;
    vib3_quantize_activation_fp4_kernel<<<num_tiles, 32, 0, s>>>(
        (const float*)input,
        (uint8_t*)act_fp4,
        (uint8_t*)act_scales,
        K
    );
    return 0;
}

int vib3_launch_fused_swiglu_mma_nvfp4_preq_fast(
    const void* act_fp4, const void* act_scales,
    const void* up_weight, const void* up_scales,
    const void* gate_weight, const void* gate_scales,
    void* output, int K, int M_slice, void* stream
) {
    if (get_sm_major() < 12) return -1;
    cudaStream_t s = (cudaStream_t)stream;

    // Ensure temp buffers
    if (M_slice > swiglu_buf_size) {
        if (swiglu_up_buf) cudaFree(swiglu_up_buf);
        if (swiglu_gate_buf) cudaFree(swiglu_gate_buf);
        swiglu_buf_size = M_slice;
        cudaMalloc(&swiglu_up_buf, swiglu_buf_size * sizeof(float));
        cudaMalloc(&swiglu_gate_buf, swiglu_buf_size * sizeof(float));
    }

    const int warps_per_block = 4;
    dim3 block(32, warps_per_block);
    int tiles = (M_slice + 15) / 16;
    int grid_x = (tiles + warps_per_block - 1) / warps_per_block;
    int smem = warps_per_block * 512;

    // Up proj
    vib3_gemv_mma_nvfp4_preq_kernel<<<grid_x, block, smem, s>>>(
        (const uint8_t*)act_fp4, (const uint8_t*)act_scales,
        (const uint8_t*)up_weight, (const unsigned short*)up_scales,
        swiglu_up_buf, K, M_slice
    );

    // Gate proj
    vib3_gemv_mma_nvfp4_preq_kernel<<<grid_x, block, smem, s>>>(
        (const uint8_t*)act_fp4, (const uint8_t*)act_scales,
        (const uint8_t*)gate_weight, (const unsigned short*)gate_scales,
        swiglu_gate_buf, K, M_slice
    );

    // Fuse
    int threads = 256;
    int blocks_fuse = (M_slice + threads - 1) / threads;
    vib3_swiglu_fuse_f32_kernel<<<blocks_fuse, threads, 0, s>>>(
        swiglu_gate_buf, swiglu_up_buf, (float*)output, M_slice
    );

    return 0;
}

// ════════════════════════════════════════════════════════════════════════════
// Multi-Expert Fused MoE Layer Kernels (Tiled + K-Parallel)
// ════════════════════════════════════════════════════════════════════════════
//
// These kernels process ALL selected experts (up to 8) in a single launch,
// using blockIdx.z to index into per-expert weight pointers, blockIdx.y for
// K-parallel decomposition, and blockIdx.x for row tiles.
// Weights must be in TILED layout (16 rows × 32 bytes per K-tile contiguous).
// Outputs use atomicAdd for K-parallel accumulation (must be pre-zeroed).
//
// Kernel 1: up+gate GEMV → SwiGLU → quantize FP4 (all experts, 1 launch each)
// Kernel 2: down GEMV → weighted atomicAdd to layer output (all experts, 1 launch)

#define MOE_MAX_EXPERTS 8

// Per-expert intermediate buffers (8 experts)
static float* g_moe_up_bufs[MOE_MAX_EXPERTS] = {};
static float* g_moe_gate_bufs[MOE_MAX_EXPERTS] = {};
static uint8_t* g_moe_mid_fp4[MOE_MAX_EXPERTS] = {};
static uint8_t* g_moe_mid_scales[MOE_MAX_EXPERTS] = {};
static int g_moe_buf_M_mid = 0;
static int g_moe_buf_K_mid = 0;

static void ensure_moe_bufs(int M_mid, int K_mid) {
    if (M_mid <= g_moe_buf_M_mid && K_mid <= g_moe_buf_K_mid) return;
    for (int i = 0; i < MOE_MAX_EXPERTS; i++) {
        if (g_moe_up_bufs[i]) cudaFree(g_moe_up_bufs[i]);
        if (g_moe_gate_bufs[i]) cudaFree(g_moe_gate_bufs[i]);
        if (g_moe_mid_fp4[i]) cudaFree(g_moe_mid_fp4[i]);
        if (g_moe_mid_scales[i]) cudaFree(g_moe_mid_scales[i]);
        cudaMalloc(&g_moe_up_bufs[i], M_mid * sizeof(float));
        cudaMalloc(&g_moe_gate_bufs[i], M_mid * sizeof(float));
        cudaMalloc(&g_moe_mid_fp4[i], (K_mid / 2 + 64));
        cudaMalloc(&g_moe_mid_scales[i], (K_mid / 32 + 4));
    }
    g_moe_buf_M_mid = M_mid;
    g_moe_buf_K_mid = K_mid;
}

// ── Kernel 1: Multi-expert tiled GEMV (up OR gate) with K-parallel ──
// blockIdx.z = expert index, blockIdx.y = K-block, blockIdx.x = row tile
// Weights must be in TILED layout. Output must be pre-zeroed (atomicAdd).
__global__ void vib3_moe_multi_gemv_kernel(
    const uint8_t* __restrict__ act_fp4,
    const uint8_t* __restrict__ act_scales,
    const uint8_t* const* __restrict__ weight_ptrs,  // [num_experts] tiled FP4
    const unsigned short* const* __restrict__ scale_ptrs, // [num_experts] row-major BF16
    float* const* __restrict__ output_ptrs,           // [num_experts] pre-zeroed FP32
    int K, int M_slice,
    int k_tiles_per_block
) {
    const int expert_idx = blockIdx.z;
    const uint8_t* weight = weight_ptrs[expert_idx];
    const unsigned short* scales = scale_ptrs[expert_idx];
    float* output = output_ptrs[expert_idx];

    const int warp_id = threadIdx.y;
    const int lane_id = threadIdx.x;
    const int num_warps = blockDim.y;

    const int row_block = blockIdx.x * num_warps + warp_id;
    const int tile_row = row_block * 16;
    if (tile_row >= M_slice) return;

    const int packed_k = K / 2;
    const int num_k_tiles = packed_k / 32;
    const int num_groups = K / 32;

    // K-range for this block
    const int k_block_id = blockIdx.y;
    const int k_tile_start = k_block_id * k_tiles_per_block;
    int k_tile_end = k_tile_start + k_tiles_per_block;
    if (k_tile_end > num_k_tiles) k_tile_end = num_k_tiles;

    const int groupID = lane_id >> 2;
    const int tid_in_grp = lane_id & 3;

    int tidx = groupID + (tid_in_grp & 1) * 8;
    int scale_row = tile_row + tidx;

    float c0 = 0.0f, c1 = 0.0f, c2 = 0.0f, c3 = 0.0f;

    // Tiled layout base pointer for this row_block
    const int* wt_base = (const int*)(weight + (long long)row_block * num_k_tiles * 512);

    const unsigned short* sr = (scale_row < M_slice)
        ? scales + (long long)scale_row * num_groups : nullptr;

    for (int k_tile = k_tile_start; k_tile < k_tile_end; k_tile++) {
        const int* tile = wt_base + k_tile * 128;
        int a0 = tile[groupID * 8 + tid_in_grp];
        int a2 = tile[groupID * 8 + 4 + tid_in_grp];
        int a1 = tile[(groupID + 8) * 8 + tid_in_grp];
        int a3 = tile[(groupID + 8) * 8 + 4 + tid_in_grp];

        const int* act_qs = (const int*)(act_fp4 + k_tile * 32);
        int b0 = act_qs[tid_in_grp];
        int b1 = act_qs[tid_in_grp + 4];
        uint8_t as0 = act_scales[k_tile * 2];
        uint8_t as1 = act_scales[k_tile * 2 + 1];
        uint32_t sb = (uint32_t)as0 | ((uint32_t)as1 << 8);

        uint32_t sa = 0;
        if (sr) {
            int g0 = k_tile * 2;
            unsigned short bf16_s0 = sr[g0];
            unsigned short bf16_s1 = sr[g0 + 1];
            uint8_t e0 = (bf16_s0 >> 7) & 0xFF;
            uint8_t e1 = (bf16_s1 >> 7) & 0xFF;
            sa = (uint32_t)e0 | ((uint32_t)e1 << 8);
        }

#if __CUDA_ARCH__ >= 1200
        asm volatile(
            "mma.sync.aligned.kind::mxf4.block_scale.scale_vec::2X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue8m0 "
            "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, "
            "%10, {0, 0}, %11, {0, 0};"
            : "+f"(c0), "+f"(c1), "+f"(c2), "+f"(c3)
            : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
              "r"(b0), "r"(b1),
              "r"(sa), "r"(sb)
        );
#endif
    }

    if (tid_in_grp == 0) {
        int row0 = tile_row + groupID;
        int row1 = tile_row + groupID + 8;
        if (row0 < M_slice) atomicAdd(&output[row0], c0);
        if (row1 < M_slice) atomicAdd(&output[row1], c2);
    }
}

// ── Kernel 2: Multi-expert SwiGLU + quantize FP4 ──
// Fuses SwiGLU(gate, up) and quantize-to-FP4 into one kernel.
// blockIdx.y = expert, blockIdx.x = K/64 tile
__global__ void vib3_moe_swiglu_quantize_kernel(
    float* const* __restrict__ gate_bufs,  // [num_experts], each [M_mid]
    float* const* __restrict__ up_bufs,    // [num_experts], each [M_mid]
    uint8_t* const* __restrict__ fp4_outs, // [num_experts], each [K_mid/2]
    uint8_t* const* __restrict__ scale_outs, // [num_experts], each [K_mid/32]
    int K_mid
) {
    const int expert_idx = blockIdx.y;
    const float* gate = gate_bufs[expert_idx];
    const float* up = up_bufs[expert_idx];
    uint8_t* act_fp4 = fp4_outs[expert_idx];
    uint8_t* act_scales = scale_outs[expert_idx];

    const int tile_idx = blockIdx.x;
    const int lane_id = threadIdx.x;
    const int k0 = tile_idx * 64;
    if (k0 >= K_mid) return;

#if __CUDA_ARCH__ >= 1200
    const int group_id = lane_id / 4;
    const int lane_in_group = lane_id % 4;
    const int base = group_id * 2;

    for (int g = 0; g < 2; g++) {
        int k_base = k0 + g * 32;

        // Load + apply SwiGLU inline
        float val = 0.0f;
        if (k_base + lane_id < K_mid) {
            float g_val = gate[k_base + lane_id];
            float u_val = up[k_base + lane_id];
            float sigmoid_g = 1.0f / (1.0f + expf(-g_val));
            val = g_val * sigmoid_g * u_val;
        }

        // Warp-wide amax reduction
        float amax = fabsf(val);
        #pragma unroll
        for (int mask = 16; mask > 0; mask >>= 1) {
            amax = fmaxf(amax, __shfl_xor_sync(0xFFFFFFFF, amax, mask, 32));
        }

        uint8_t e = compute_e8m0(amax);
        float inv_s = (amax == 0.0f) ? 0.0f : __frcp_rn(e8m0_to_f32(e));
        float scaled_val = val * inv_s;

        // Gather 4 values for split-half packing
        float val0 = __shfl_sync(0xFFFFFFFF, scaled_val, base, 32);
        float val1 = __shfl_sync(0xFFFFFFFF, scaled_val, base + 16, 32);
        float val2 = __shfl_sync(0xFFFFFFFF, scaled_val, base + 1, 32);
        float val3 = __shfl_sync(0xFFFFFFFF, scaled_val, base + 17, 32);

        if (lane_in_group == 0) {
            __nv_fp4x4_e2m1 fp4_packed(make_float4(val0, val1, val2, val3));
            *(uint16_t*)(act_fp4 + tile_idx * 32 + g * 16 + group_id * 2) = *(uint16_t*)&fp4_packed;
        }

        if (lane_id == 0) {
            act_scales[tile_idx * 2 + g] = e;
        }
    }
#endif
}

// ── Kernel 3: Multi-expert tiled down GEMV + weighted accumulate (K-parallel) ──
// blockIdx.z = expert, blockIdx.y = K-block, blockIdx.x = row tile
// Weights must be in TILED layout. Uses atomicAdd for both K-parallel
// accumulation and weighted addition to the shared layer_output.
__global__ void vib3_moe_down_accumulate_kernel(
    const uint8_t* const* __restrict__ act_fp4_ptrs,  // [num_experts] FP4 mid
    const uint8_t* const* __restrict__ act_scale_ptrs, // [num_experts] E8M0
    const uint8_t* const* __restrict__ weight_ptrs,    // [num_experts] tiled down weight
    const unsigned short* const* __restrict__ scale_ptrs, // [num_experts] down scales
    const float* __restrict__ expert_weights,           // [num_experts] routing weights
    float* __restrict__ layer_output,                   // [M_out] accumulated output
    int K, int M_out,
    int k_tiles_per_block
) {
    const int expert_idx = blockIdx.z;
    const uint8_t* act_fp4 = act_fp4_ptrs[expert_idx];
    const uint8_t* act_scales = act_scale_ptrs[expert_idx];
    const uint8_t* weight = weight_ptrs[expert_idx];
    const unsigned short* scales = scale_ptrs[expert_idx];
    const float ew = expert_weights[expert_idx];

    const int warp_id = threadIdx.y;
    const int lane_id = threadIdx.x;
    const int num_warps = blockDim.y;

    const int row_block = blockIdx.x * num_warps + warp_id;
    const int tile_row = row_block * 16;
    if (tile_row >= M_out) return;

    const int packed_k = K / 2;
    const int num_k_tiles = packed_k / 32;
    const int num_groups = K / 32;

    // K-range for this block
    const int k_block_id = blockIdx.y;
    const int k_tile_start = k_block_id * k_tiles_per_block;
    int k_tile_end = k_tile_start + k_tiles_per_block;
    if (k_tile_end > num_k_tiles) k_tile_end = num_k_tiles;

    const int groupID = lane_id >> 2;
    const int tid_in_grp = lane_id & 3;

    int tidx = groupID + (tid_in_grp & 1) * 8;
    int scale_row = tile_row + tidx;

    float c0 = 0.0f, c1 = 0.0f, c2 = 0.0f, c3 = 0.0f;

    // Tiled layout base pointer
    const int* wt_base = (const int*)(weight + (long long)row_block * num_k_tiles * 512);

    const unsigned short* sr = (scale_row < M_out)
        ? scales + (long long)scale_row * num_groups : nullptr;

    for (int k_tile = k_tile_start; k_tile < k_tile_end; k_tile++) {
        const int* tile = wt_base + k_tile * 128;
        int a0 = tile[groupID * 8 + tid_in_grp];
        int a2 = tile[groupID * 8 + 4 + tid_in_grp];
        int a1 = tile[(groupID + 8) * 8 + tid_in_grp];
        int a3 = tile[(groupID + 8) * 8 + 4 + tid_in_grp];

        const int* act_qs = (const int*)(act_fp4 + k_tile * 32);
        int b0 = act_qs[tid_in_grp];
        int b1 = act_qs[tid_in_grp + 4];
        uint8_t as0 = act_scales[k_tile * 2];
        uint8_t as1 = act_scales[k_tile * 2 + 1];
        uint32_t sb = (uint32_t)as0 | ((uint32_t)as1 << 8);

        uint32_t sa = 0;
        if (sr) {
            int g0 = k_tile * 2;
            unsigned short bf16_s0 = sr[g0];
            unsigned short bf16_s1 = sr[g0 + 1];
            uint8_t e0 = (bf16_s0 >> 7) & 0xFF;
            uint8_t e1 = (bf16_s1 >> 7) & 0xFF;
            sa = (uint32_t)e0 | ((uint32_t)e1 << 8);
        }

#if __CUDA_ARCH__ >= 1200
        asm volatile(
            "mma.sync.aligned.kind::mxf4.block_scale.scale_vec::2X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue8m0 "
            "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, "
            "%10, {0, 0}, %11, {0, 0};"
            : "+f"(c0), "+f"(c1), "+f"(c2), "+f"(c3)
            : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
              "r"(b0), "r"(b1),
              "r"(sa), "r"(sb)
        );
#endif
    }

    // Weighted atomicAdd to shared layer output
    if (tid_in_grp == 0) {
        int row0 = tile_row + groupID;
        int row1 = tile_row + groupID + 8;
        if (row0 < M_out) atomicAdd(&layer_output[row0], ew * c0);
        if (row1 < M_out) atomicAdd(&layer_output[row1], ew * c2);
    }
}

// ── Static device-side pointer arrays for kernel launch ──
static const uint8_t** d_moe_weight_ptrs = nullptr;
static const unsigned short** d_moe_scale_ptrs = nullptr;
static float** d_moe_output_ptrs = nullptr;
static const uint8_t** d_moe_fp4_ptrs = nullptr;
static const uint8_t** d_moe_fp4_scale_ptrs = nullptr;
static float** d_moe_gate_ptrs = nullptr;
static float** d_moe_up_ptrs = nullptr;
static uint8_t** d_moe_mid_fp4_ptrs = nullptr;
static uint8_t** d_moe_mid_scale_ptrs = nullptr;
static float* d_moe_expert_weights = nullptr;
static bool d_moe_ptrs_allocated = false;

static void ensure_moe_device_ptrs() {
    if (d_moe_ptrs_allocated) return;
    cudaMalloc(&d_moe_weight_ptrs, MOE_MAX_EXPERTS * sizeof(void*));
    cudaMalloc(&d_moe_scale_ptrs, MOE_MAX_EXPERTS * sizeof(void*));
    cudaMalloc(&d_moe_output_ptrs, MOE_MAX_EXPERTS * sizeof(void*));
    cudaMalloc(&d_moe_fp4_ptrs, MOE_MAX_EXPERTS * sizeof(void*));
    cudaMalloc(&d_moe_fp4_scale_ptrs, MOE_MAX_EXPERTS * sizeof(void*));
    cudaMalloc(&d_moe_gate_ptrs, MOE_MAX_EXPERTS * sizeof(void*));
    cudaMalloc(&d_moe_up_ptrs, MOE_MAX_EXPERTS * sizeof(void*));
    cudaMalloc(&d_moe_mid_fp4_ptrs, MOE_MAX_EXPERTS * sizeof(void*));
    cudaMalloc(&d_moe_mid_scale_ptrs, MOE_MAX_EXPERTS * sizeof(void*));
    cudaMalloc(&d_moe_expert_weights, MOE_MAX_EXPERTS * sizeof(float));
    d_moe_ptrs_allocated = true;
}

// ── Fused MoE layer launcher ──
// Processes ALL selected experts (up to 8) with just 4 kernel launches:
//   1. Multi-expert up GEMV
//   2. Multi-expert gate GEMV
//   3. Multi-expert SwiGLU + quantize FP4
//   4. Multi-expert down GEMV + weighted accumulate
//
// vs. previous: 8 experts × (6 kernels + 1 memset) = 56 GPU operations
int vib3_launch_moe_experts_fused(
    // Pre-quantized activation (shared across all experts)
    const void* act_fp4,
    const void* act_scales,
    // Per-expert weight page pointers (host arrays of device pointers)
    const void* const* up_weight_ptrs,    // [num_experts]
    const void* const* up_scale_ptrs,     // [num_experts]
    const void* const* gate_weight_ptrs,  // [num_experts]
    const void* const* gate_scale_ptrs,   // [num_experts]
    const void* const* down_weight_ptrs,  // [num_experts]
    const void* const* down_scale_ptrs,   // [num_experts]
    const float* expert_weights_host,     // [num_experts] routing weights
    int num_experts,
    // Dimensions
    int K_in,    // hidden_dim (3072)
    int M_mid,   // expert_hidden_dim (1024) — up/gate output rows
    int K_mid,   // expert_hidden_dim (1024) — down input cols
    int M_out,   // hidden_dim (3072) — down output rows
    float* layer_output,  // [M_out] pre-zeroed FP32 accumulator
    void* stream
) {
    if (get_sm_major() < 12) return -1;
    if (num_experts <= 0 || num_experts > MOE_MAX_EXPERTS) return -2;
    cudaStream_t s = (cudaStream_t)stream;

    // Ensure intermediate buffers
    ensure_moe_bufs(M_mid, K_mid);
    ensure_moe_device_ptrs();

    // Build host-side pointer arrays
    const uint8_t* h_up_w[MOE_MAX_EXPERTS];
    const unsigned short* h_up_s[MOE_MAX_EXPERTS];
    const uint8_t* h_gate_w[MOE_MAX_EXPERTS];
    const unsigned short* h_gate_s[MOE_MAX_EXPERTS];
    const uint8_t* h_down_w[MOE_MAX_EXPERTS];
    const unsigned short* h_down_s[MOE_MAX_EXPERTS];

    for (int i = 0; i < num_experts; i++) {
        h_up_w[i] = (const uint8_t*)up_weight_ptrs[i];
        h_up_s[i] = (const unsigned short*)up_scale_ptrs[i];
        h_gate_w[i] = (const uint8_t*)gate_weight_ptrs[i];
        h_gate_s[i] = (const unsigned short*)gate_scale_ptrs[i];
        h_down_w[i] = (const uint8_t*)down_weight_ptrs[i];
        h_down_s[i] = (const unsigned short*)down_scale_ptrs[i];
    }

    // Upload pointer arrays to device (async)
    cudaMemcpyAsync((void*)d_moe_weight_ptrs, h_up_w, num_experts * sizeof(void*), cudaMemcpyHostToDevice, s);
    cudaMemcpyAsync((void*)d_moe_scale_ptrs, h_up_s, num_experts * sizeof(void*), cudaMemcpyHostToDevice, s);
    cudaMemcpyAsync((void*)d_moe_output_ptrs, g_moe_up_bufs, num_experts * sizeof(void*), cudaMemcpyHostToDevice, s);

    // ── K-parallel grid computation for up/gate (M=M_mid, K=K_in) ──
    const int warps_per_block = 4;
    dim3 block(32, warps_per_block);

    int packed_k_in = K_in / 2;
    int num_k_tiles_in = packed_k_in / 32;
    int upgate_tiles = (M_mid + 15) / 16;
    int upgate_grid_x = (upgate_tiles + warps_per_block - 1) / warps_per_block;

    // Target ~672 total blocks (84 SMs × 8), factoring in num_experts in z-dim
    int target_blocks = 84 * 8;
    int upgate_base_blocks = upgate_grid_x * num_experts;
    int upgate_k_blocks = (target_blocks + upgate_base_blocks - 1) / upgate_base_blocks;
    if (upgate_k_blocks < 1) upgate_k_blocks = 1;
    if (upgate_k_blocks > num_k_tiles_in) upgate_k_blocks = num_k_tiles_in;
    int upgate_k_tiles_per_block = (num_k_tiles_in + upgate_k_blocks - 1) / upgate_k_blocks;

    // ── Step 1: Multi-expert up GEMV (tiled + K-parallel) ──
    // Zero up buffers for atomicAdd accumulation
    for (int i = 0; i < num_experts; i++) {
        cudaMemsetAsync(g_moe_up_bufs[i], 0, M_mid * sizeof(float), s);
    }
    {
        dim3 grid(upgate_grid_x, upgate_k_blocks, num_experts);
        vib3_moe_multi_gemv_kernel<<<grid, block, 0, s>>>(
            (const uint8_t*)act_fp4, (const uint8_t*)act_scales,
            d_moe_weight_ptrs, d_moe_scale_ptrs,
            d_moe_output_ptrs,
            K_in, M_mid, upgate_k_tiles_per_block
        );
    }

    // ── Step 2: Multi-expert gate GEMV (tiled + K-parallel) ──
    cudaMemcpyAsync((void*)d_moe_weight_ptrs, h_gate_w, num_experts * sizeof(void*), cudaMemcpyHostToDevice, s);
    cudaMemcpyAsync((void*)d_moe_scale_ptrs, h_gate_s, num_experts * sizeof(void*), cudaMemcpyHostToDevice, s);
    cudaMemcpyAsync((void*)d_moe_output_ptrs, g_moe_gate_bufs, num_experts * sizeof(void*), cudaMemcpyHostToDevice, s);
    // Zero gate buffers for atomicAdd accumulation
    for (int i = 0; i < num_experts; i++) {
        cudaMemsetAsync(g_moe_gate_bufs[i], 0, M_mid * sizeof(float), s);
    }
    {
        dim3 grid(upgate_grid_x, upgate_k_blocks, num_experts);
        vib3_moe_multi_gemv_kernel<<<grid, block, 0, s>>>(
            (const uint8_t*)act_fp4, (const uint8_t*)act_scales,
            d_moe_weight_ptrs, d_moe_scale_ptrs,
            d_moe_output_ptrs,
            K_in, M_mid, upgate_k_tiles_per_block
        );
    }

    // ── Step 3: Multi-expert SwiGLU + quantize FP4 ──
    cudaMemcpyAsync((void*)d_moe_gate_ptrs, g_moe_gate_bufs, num_experts * sizeof(void*), cudaMemcpyHostToDevice, s);
    cudaMemcpyAsync((void*)d_moe_up_ptrs, g_moe_up_bufs, num_experts * sizeof(void*), cudaMemcpyHostToDevice, s);
    cudaMemcpyAsync((void*)d_moe_mid_fp4_ptrs, g_moe_mid_fp4, num_experts * sizeof(void*), cudaMemcpyHostToDevice, s);
    cudaMemcpyAsync((void*)d_moe_mid_scale_ptrs, g_moe_mid_scales, num_experts * sizeof(void*), cudaMemcpyHostToDevice, s);
    {
        int num_tiles = (K_mid + 63) / 64;
        dim3 grid(num_tiles, num_experts);
        vib3_moe_swiglu_quantize_kernel<<<grid, 32, 0, s>>>(
            (float* const*)d_moe_gate_ptrs,
            (float* const*)d_moe_up_ptrs,
            (uint8_t* const*)d_moe_mid_fp4_ptrs,
            (uint8_t* const*)d_moe_mid_scale_ptrs,
            K_mid
        );
    }

    // ── K-parallel grid computation for down (M=M_out, K=K_mid) ──
    int packed_k_mid = K_mid / 2;
    int num_k_tiles_mid = packed_k_mid / 32;
    int down_tiles = (M_out + 15) / 16;
    int down_grid_x = (down_tiles + warps_per_block - 1) / warps_per_block;

    int down_base_blocks = down_grid_x * num_experts;
    int down_k_blocks = (target_blocks + down_base_blocks - 1) / down_base_blocks;
    if (down_k_blocks < 1) down_k_blocks = 1;
    if (down_k_blocks > num_k_tiles_mid) down_k_blocks = num_k_tiles_mid;
    int down_k_tiles_per_block = (num_k_tiles_mid + down_k_blocks - 1) / down_k_blocks;

    // ── Step 4: Multi-expert down GEMV + weighted accumulate (tiled + K-parallel) ──
    cudaMemcpyAsync((void*)d_moe_weight_ptrs, h_down_w, num_experts * sizeof(void*), cudaMemcpyHostToDevice, s);
    cudaMemcpyAsync((void*)d_moe_scale_ptrs, h_down_s, num_experts * sizeof(void*), cudaMemcpyHostToDevice, s);
    cudaMemcpyAsync((void*)d_moe_expert_weights, expert_weights_host, num_experts * sizeof(float), cudaMemcpyHostToDevice, s);
    // Note: layer_output is already pre-zeroed by caller (for weighted atomicAdd from all experts)
    {
        dim3 grid(down_grid_x, down_k_blocks, num_experts);
        vib3_moe_down_accumulate_kernel<<<grid, block, 0, s>>>(
            (const uint8_t* const*)d_moe_mid_fp4_ptrs,
            (const uint8_t* const*)d_moe_mid_scale_ptrs,
            d_moe_weight_ptrs, d_moe_scale_ptrs,
            d_moe_expert_weights,
            layer_output,
            K_mid, M_out, down_k_tiles_per_block
        );
    }

    return 0;
}

// ════════════════════════════════════════════════════════════════════════════
// Batched INT4 MoE (analogue of NVFP4 moe_experts_fused for Q4_0 / INT4 weights)
// ════════════════════════════════════════════════════════════════════════════
//
// Two kernel launches per MoE layer instead of 3 × num_active:
//   1. vib3_moe_multi_int4_swiglu — batched up·gate·silu_mul across all experts,
//      writes per-expert FP16 intermediate buffers.
//   2. vib3_moe_multi_int4_down_accum — batched down GEMV + weighted atomicAdd
//      into the shared FP32 layer_output buffer.
//
// Reduction for K2.6 (num_active=8, 60 MoE layers):
//   Before: 8 × 3 launches × 60 layers = 1440 launches/token
//   After:  2 launches × 60 layers = 120 launches/token
//
// Saves ~1320 launches × ~5 µs = ~6 ms/token of launch overhead.

// Per-expert FP16 intermediate buffers for batched INT4 SwiGLU → down_proj handoff.
static half* g_moe_int4_mid_fp16[MOE_MAX_EXPERTS] = {};
static int g_moe_int4_buf_M_mid = 0;

static void ensure_moe_int4_bufs(int M_mid) {
    if (M_mid <= g_moe_int4_buf_M_mid) return;
    for (int i = 0; i < MOE_MAX_EXPERTS; i++) {
        if (g_moe_int4_mid_fp16[i]) cudaFree(g_moe_int4_mid_fp16[i]);
        cudaMalloc(&g_moe_int4_mid_fp16[i], M_mid * sizeof(half));
    }
    g_moe_int4_buf_M_mid = M_mid;
}

// Per-task pointer arrays for INT4 MoE (each expert may be split across
// multiple weight pages → up to MOE_MAX_EXPERTS × MOE_MAX_PAGES tasks).
#define MOE_INT4_MAX_TASKS 128
static const uint8_t**        d_moe_int4_up_w = nullptr;
static const unsigned short** d_moe_int4_up_s = nullptr;
static const uint8_t**        d_moe_int4_gate_w = nullptr;
static const unsigned short** d_moe_int4_gate_s = nullptr;
static const uint8_t**        d_moe_int4_down_w = nullptr;
static const unsigned short** d_moe_int4_down_s = nullptr;
static half**                 d_moe_int4_mid_out_ptrs = nullptr; // per-swiglu-task (offset to row_start)
static const half**           d_moe_int4_mid_in_ptrs = nullptr;  // per-down-task (full expert buffer)
static float*                 d_moe_int4_task_expert_wts = nullptr;
static int*                   d_moe_int4_sw_m_slices = nullptr;
static int*                   d_moe_int4_dn_m_slices = nullptr;
static int*                   d_moe_int4_dn_row_starts = nullptr;
static bool                   d_moe_int4_ptrs_allocated = false;

static void ensure_moe_int4_device_ptrs() {
    if (d_moe_int4_ptrs_allocated) return;
    cudaMalloc(&d_moe_int4_up_w,   MOE_INT4_MAX_TASKS * sizeof(void*));
    cudaMalloc(&d_moe_int4_up_s,   MOE_INT4_MAX_TASKS * sizeof(void*));
    cudaMalloc(&d_moe_int4_gate_w, MOE_INT4_MAX_TASKS * sizeof(void*));
    cudaMalloc(&d_moe_int4_gate_s, MOE_INT4_MAX_TASKS * sizeof(void*));
    cudaMalloc(&d_moe_int4_down_w, MOE_INT4_MAX_TASKS * sizeof(void*));
    cudaMalloc(&d_moe_int4_down_s, MOE_INT4_MAX_TASKS * sizeof(void*));
    cudaMalloc(&d_moe_int4_mid_out_ptrs, MOE_INT4_MAX_TASKS * sizeof(void*));
    cudaMalloc(&d_moe_int4_mid_in_ptrs,  MOE_INT4_MAX_TASKS * sizeof(void*));
    cudaMalloc(&d_moe_int4_task_expert_wts, MOE_INT4_MAX_TASKS * sizeof(float));
    cudaMalloc(&d_moe_int4_sw_m_slices, MOE_INT4_MAX_TASKS * sizeof(int));
    cudaMalloc(&d_moe_int4_dn_m_slices, MOE_INT4_MAX_TASKS * sizeof(int));
    cudaMalloc(&d_moe_int4_dn_row_starts, MOE_INT4_MAX_TASKS * sizeof(int));
    d_moe_int4_ptrs_allocated = true;
}

// Kernel 1 — batched INT4 fused SwiGLU, multi-page.
// Each "task" is one (up_page, gate_page) pair (may be a slice of one expert's
// up/gate matrix). The task owns its own m_slice of the expert's intermediate
// buffer — row_start is already baked into output_ptrs (caller offsets).
//
// Grid: (ceil(max_m_slice/ROWS_PER_BLOCK), num_tasks)
// Block: THREADS_PER_ROW × ROWS_PER_BLOCK = BLOCK_SIZE threads.
__global__ void vib3_moe_multi_int4_swiglu_kernel(
    const float* __restrict__ input,                              // [K] FP32 shared
    const uint8_t* const* __restrict__ up_weight_ptrs,            // [num_tasks]
    const unsigned short* const* __restrict__ up_scale_ptrs,
    const uint8_t* const* __restrict__ gate_weight_ptrs,
    const unsigned short* const* __restrict__ gate_scale_ptrs,
    half* const* __restrict__ output_ptrs,                        // [num_tasks] offset to row_start
    const int* __restrict__ m_slices,                             // [num_tasks]
    int K,
    int group_size
) {
    const int task_idx = blockIdx.y;
    const int m_slice_local = m_slices[task_idx];
    const uint8_t* up_weight = up_weight_ptrs[task_idx];
    const unsigned short* up_scales = up_scale_ptrs[task_idx];
    const uint8_t* gate_weight = gate_weight_ptrs[task_idx];
    const unsigned short* gate_scales = gate_scale_ptrs[task_idx];
    half* output = output_ptrs[task_idx];

    const int local_row = threadIdx.x / THREADS_PER_ROW;
    const int lane = threadIdx.x % THREADS_PER_ROW;
    const int row = blockIdx.x * ROWS_PER_BLOCK + local_row;

    if (row >= m_slice_local) return;

    const int packed_k = (K + 1) / 2;
    const int num_groups = (K + group_size - 1) / group_size;
    const uint8_t* up_row = up_weight + (long long)row * packed_k;
    const unsigned short* up_s = up_scales + (long long)row * num_groups;
    const uint8_t* gate_row = gate_weight + (long long)row * packed_k;
    const unsigned short* gate_s = gate_scales + (long long)row * num_groups;

    float up_acc = 0.0f;
    float gate_acc = 0.0f;
    for (int k = lane * 2; k < K; k += THREADS_PER_ROW * 2) {
        int byte_idx = k / 2;
        int group = k / group_size;

        uint8_t up_packed = up_row[byte_idx];
        float up_scale = bf16_scale_to_float(up_s[group]);
        int up_w0 = (int)((up_packed >> 0) & 0xF) - 8;
        int up_w1 = (int)((up_packed >> 4) & 0xF) - 8;

        uint8_t gate_packed = gate_row[byte_idx];
        float gate_scale = bf16_scale_to_float(gate_s[group]);
        int gate_w0 = (int)((gate_packed >> 0) & 0xF) - 8;
        int gate_w1 = (int)((gate_packed >> 4) & 0xF) - 8;

        float inp0 = input[k];
        up_acc += inp0 * (float)up_w0 * up_scale;
        gate_acc += inp0 * (float)gate_w0 * gate_scale;

        if (k + 1 < K) {
            float inp1 = input[k + 1];
            up_acc += inp1 * (float)up_w1 * up_scale;
            gate_acc += inp1 * (float)gate_w1 * gate_scale;
        }
    }

    up_acc = warp_reduce_sum(up_acc);
    gate_acc = warp_reduce_sum(gate_acc);

    __shared__ float smem[ROWS_PER_BLOCK * (THREADS_PER_ROW / WARP_SIZE) * 2];
    const int warp_id = lane / WARP_SIZE;
    const int warp_lane = lane % WARP_SIZE;
    const int n_warps = THREADS_PER_ROW / WARP_SIZE;

    if (warp_lane == 0) {
        smem[(local_row * n_warps + warp_id) * 2 + 0] = up_acc;
        smem[(local_row * n_warps + warp_id) * 2 + 1] = gate_acc;
    }
    __syncthreads();

    if (warp_id == 0) {
        float final_up = (warp_lane < n_warps)
            ? smem[(local_row * n_warps + warp_lane) * 2 + 0]
            : 0.0f;
        float final_gate = (warp_lane < n_warps)
            ? smem[(local_row * n_warps + warp_lane) * 2 + 1]
            : 0.0f;
        final_up = warp_reduce_sum(final_up);
        final_gate = warp_reduce_sum(final_gate);
        if (warp_lane == 0) {
            float silu = final_gate / (1.0f + expf(-final_gate));
            output[row] = __float2half(silu * final_up);
        }
    }
}

// Kernel 2 — batched INT4 down-proj GEMV + weighted atomicAdd into layer_output.
// Each "task" is one down_page for one expert. Different experts may have
// different row_start / m_slice per page; different pages of the same expert
// cover disjoint rows of layer_output.
//
// Grid: (ceil(max_m_slice/ROWS_PER_BLOCK), num_tasks)
// atomicAdd to layer_output because multiple tasks (different experts) may
// contribute to the same row.
__global__ void vib3_moe_multi_int4_down_accum_kernel(
    const half* const* __restrict__ mid_ptrs,               // [num_tasks] × [K] FP16 (shared across tasks for same expert)
    const uint8_t* const* __restrict__ down_weight_ptrs,    // [num_tasks]
    const unsigned short* const* __restrict__ down_scale_ptrs, // [num_tasks]
    const float* __restrict__ task_expert_weights,          // [num_tasks] — expert routing weight replicated across its pages
    const int* __restrict__ row_starts,                     // [num_tasks]
    const int* __restrict__ m_slices,                       // [num_tasks]
    float* __restrict__ layer_output,                       // [M_out] FP32, pre-zeroed
    int K,
    int group_size
) {
    const int task_idx = blockIdx.y;
    const int m_slice_local = m_slices[task_idx];
    const int row_start = row_starts[task_idx];
    const half* input = mid_ptrs[task_idx];
    const uint8_t* weight = down_weight_ptrs[task_idx];
    const unsigned short* scales = down_scale_ptrs[task_idx];
    const float expert_w = task_expert_weights[task_idx];

    const int local_row = threadIdx.x / THREADS_PER_ROW;
    const int lane = threadIdx.x % THREADS_PER_ROW;
    const int local = blockIdx.x * ROWS_PER_BLOCK + local_row;

    if (local >= m_slice_local) return;
    const int row = local;               // row within page's weight matrix
    const int global_row = row_start + local; // absolute row in layer_output

    const int packed_k = (K + 1) / 2;
    const int num_groups = (K + group_size - 1) / group_size;
    const uint8_t* row_weight = weight + (long long)row * packed_k;
    const unsigned short* row_scales = scales + (long long)row * num_groups;

    float acc = 0.0f;
    for (int k = lane * 2; k < K; k += THREADS_PER_ROW * 2) {
        int byte_idx = k / 2;
        uint8_t packed = row_weight[byte_idx];
        int w0 = (int)((packed >> 0) & 0xF) - 8;
        int w1 = (int)((packed >> 4) & 0xF) - 8;

        int group = k / group_size;
        float scale = bf16_scale_to_float(row_scales[group]);

        acc += __half2float(input[k]) * (float)w0 * scale;
        if (k + 1 < K) {
            acc += __half2float(input[k + 1]) * (float)w1 * scale;
        }
    }

    acc = warp_reduce_sum(acc);

    __shared__ float smem[ROWS_PER_BLOCK * (THREADS_PER_ROW / WARP_SIZE)];
    const int warp_id = lane / WARP_SIZE;
    const int warp_lane = lane % WARP_SIZE;
    const int n_warps = THREADS_PER_ROW / WARP_SIZE;

    if (warp_lane == 0) {
        smem[local_row * n_warps + warp_id] = acc;
    }
    __syncthreads();

    if (warp_id == 0) {
        float final_acc = (warp_lane < n_warps)
            ? smem[local_row * n_warps + warp_lane]
            : 0.0f;
        final_acc = warp_reduce_sum(final_acc);
        if (warp_lane == 0) {
            atomicAdd(&layer_output[global_row], expert_w * final_acc);
        }
    }
}

// Launcher for batched INT4 MoE (multi-page).
//
// Inputs are "tasks". For SwiGLU, each task = one (up_page, gate_page) pair of
// a single expert. For down, each task = one down_page of a single expert.
// Each task also names an `expert_slot` (0..MOE_MAX_EXPERTS-1) which
// determines which internal FP16 intermediate buffer the SwiGLU writes into
// and the down kernel reads from. Different pages of the same expert share
// the same expert_slot. Per-task `row_start` indexes into that buffer
// (SwiGLU) or into `layer_output` (down).
//
// layer_output must be pre-zeroed (atomicAdd accumulation across tasks).
// input_f32 is the shared activation (moe_normed_f32, FP32).
int vib3_launch_moe_int4_experts_fused(
    const void* input_f32,
    // SwiGLU tasks (one per up/gate page pair)
    const void* const* sw_up_w,
    const void* const* sw_up_s,
    const void* const* sw_gate_w,
    const void* const* sw_gate_s,
    const int* sw_expert_slots,   // [num_sw_tasks] in 0..MOE_MAX_EXPERTS
    const int* sw_row_starts,     // [num_sw_tasks] — row offset within expert intermediate
    const int* sw_m_slices,       // [num_sw_tasks]
    int num_sw_tasks,
    int max_sw_m_slice,
    // Down tasks (one per down page)
    const void* const* dn_w,
    const void* const* dn_s,
    const int* dn_expert_slots,
    const int* dn_row_starts,     // [num_dn_tasks] — row offset within layer_output
    const int* dn_m_slices,
    const float* dn_expert_weights_host, // [num_dn_tasks]
    int num_dn_tasks,
    int max_dn_m_slice,
    // Dims
    int K_in,
    int M_mid,
    int K_mid,
    int group_size,
    float* layer_output,
    void* stream
) {
    if (num_sw_tasks <= 0 || num_sw_tasks > MOE_INT4_MAX_TASKS) return -2;
    if (num_dn_tasks <= 0 || num_dn_tasks > MOE_INT4_MAX_TASKS) return -3;
    cudaStream_t s = (cudaStream_t)stream;

    ensure_moe_int4_bufs(M_mid);
    ensure_moe_int4_device_ptrs();

    // Build SwiGLU per-task output pointers: g_moe_int4_mid_fp16[slot] + row_start
    half* h_sw_mid_out[MOE_INT4_MAX_TASKS];
    for (int i = 0; i < num_sw_tasks; i++) {
        int slot = sw_expert_slots[i];
        int row_start = sw_row_starts[i];
        h_sw_mid_out[i] = g_moe_int4_mid_fp16[slot] + row_start;
    }
    // Build down per-task input pointers: g_moe_int4_mid_fp16[slot] (full buffer)
    const half* h_dn_mid_in[MOE_INT4_MAX_TASKS];
    for (int i = 0; i < num_dn_tasks; i++) {
        int slot = dn_expert_slots[i];
        h_dn_mid_in[i] = g_moe_int4_mid_fp16[slot];
    }

    // Upload task pointer/int arrays to device. All async on the given stream.
    cudaMemcpyAsync((void*)d_moe_int4_up_w,   sw_up_w,   num_sw_tasks * sizeof(void*), cudaMemcpyHostToDevice, s);
    cudaMemcpyAsync((void*)d_moe_int4_up_s,   sw_up_s,   num_sw_tasks * sizeof(void*), cudaMemcpyHostToDevice, s);
    cudaMemcpyAsync((void*)d_moe_int4_gate_w, sw_gate_w, num_sw_tasks * sizeof(void*), cudaMemcpyHostToDevice, s);
    cudaMemcpyAsync((void*)d_moe_int4_gate_s, sw_gate_s, num_sw_tasks * sizeof(void*), cudaMemcpyHostToDevice, s);
    cudaMemcpyAsync((void*)d_moe_int4_mid_out_ptrs, h_sw_mid_out, num_sw_tasks * sizeof(void*), cudaMemcpyHostToDevice, s);
    cudaMemcpyAsync((void*)d_moe_int4_sw_m_slices, sw_m_slices, num_sw_tasks * sizeof(int), cudaMemcpyHostToDevice, s);

    cudaMemcpyAsync((void*)d_moe_int4_down_w, dn_w, num_dn_tasks * sizeof(void*), cudaMemcpyHostToDevice, s);
    cudaMemcpyAsync((void*)d_moe_int4_down_s, dn_s, num_dn_tasks * sizeof(void*), cudaMemcpyHostToDevice, s);
    cudaMemcpyAsync((void*)d_moe_int4_mid_in_ptrs, h_dn_mid_in, num_dn_tasks * sizeof(void*), cudaMemcpyHostToDevice, s);
    cudaMemcpyAsync((void*)d_moe_int4_task_expert_wts, dn_expert_weights_host, num_dn_tasks * sizeof(float), cudaMemcpyHostToDevice, s);
    cudaMemcpyAsync((void*)d_moe_int4_dn_row_starts, dn_row_starts, num_dn_tasks * sizeof(int), cudaMemcpyHostToDevice, s);
    cudaMemcpyAsync((void*)d_moe_int4_dn_m_slices, dn_m_slices, num_dn_tasks * sizeof(int), cudaMemcpyHostToDevice, s);

    // Launch 1: batched SwiGLU across all (expert, up/gate page) tasks.
    {
        int blocks_m = (max_sw_m_slice + ROWS_PER_BLOCK - 1) / ROWS_PER_BLOCK;
        dim3 grid(blocks_m, num_sw_tasks);
        vib3_moe_multi_int4_swiglu_kernel<<<grid, BLOCK_SIZE, 0, s>>>(
            (const float*)input_f32,
            d_moe_int4_up_w, d_moe_int4_up_s,
            d_moe_int4_gate_w, d_moe_int4_gate_s,
            d_moe_int4_mid_out_ptrs,
            d_moe_int4_sw_m_slices,
            K_in, group_size
        );
    }

    // Launch 2: batched down + weighted atomicAdd across all (expert, down page) tasks.
    {
        int blocks_m = (max_dn_m_slice + ROWS_PER_BLOCK - 1) / ROWS_PER_BLOCK;
        dim3 grid(blocks_m, num_dn_tasks);
        vib3_moe_multi_int4_down_accum_kernel<<<grid, BLOCK_SIZE, 0, s>>>(
            d_moe_int4_mid_in_ptrs,
            d_moe_int4_down_w, d_moe_int4_down_s,
            d_moe_int4_task_expert_wts,
            d_moe_int4_dn_row_starts,
            d_moe_int4_dn_m_slices,
            layer_output,
            K_mid, group_size
        );
    }

    return (int)cudaGetLastError();
}

// ════════════════════════════════════════════════════════════════════════════
// GPU-Only MoE Dispatch (zero host sync)
// ════════════════════════════════════════════════════════════════════════════
//
// Uses a device-side page pointer table to eliminate host-side router sync
// and H2D memcpys. The router writes expert IDs/weights to device memory,
// and a small resolve kernel looks up weight page pointers from a prebuilt table.
//
// Table layout: [num_layers * num_experts * 3] uint64_t page pointers
// Index: layer * num_experts_total * 3 + expert_id * 3 + segment (0=up, 1=gate, 2=down)

// Static device arrays for resolved up/gate/down pointers (separate from existing d_moe_*)
static const uint8_t** d_gpu_up_weight_ptrs = nullptr;
static const unsigned short** d_gpu_up_scale_ptrs = nullptr;
static const uint8_t** d_gpu_gate_weight_ptrs = nullptr;
static const unsigned short** d_gpu_gate_scale_ptrs = nullptr;
static const uint8_t** d_gpu_down_weight_ptrs = nullptr;
static const unsigned short** d_gpu_down_scale_ptrs = nullptr;
static float* d_gpu_expert_weights = nullptr;
static bool d_gpu_moe_ptrs_allocated = false;
static bool d_gpu_moe_bufs_uploaded = false;

static void ensure_gpu_moe_ptrs() {
    if (d_gpu_moe_ptrs_allocated) return;
    cudaMalloc(&d_gpu_up_weight_ptrs, MOE_MAX_EXPERTS * sizeof(void*));
    cudaMalloc(&d_gpu_up_scale_ptrs, MOE_MAX_EXPERTS * sizeof(void*));
    cudaMalloc(&d_gpu_gate_weight_ptrs, MOE_MAX_EXPERTS * sizeof(void*));
    cudaMalloc(&d_gpu_gate_scale_ptrs, MOE_MAX_EXPERTS * sizeof(void*));
    cudaMalloc(&d_gpu_down_weight_ptrs, MOE_MAX_EXPERTS * sizeof(void*));
    cudaMalloc(&d_gpu_down_scale_ptrs, MOE_MAX_EXPERTS * sizeof(void*));
    cudaMalloc(&d_gpu_expert_weights, MOE_MAX_EXPERTS * sizeof(float));
    d_gpu_moe_ptrs_allocated = true;
}

// Resolve expert IDs → 6 pointer arrays + weights, all on GPU.
// Single warp (32 threads), only first num_active threads active.
__global__ void vib3_moe_resolve_ptrs_kernel(
    const uint64_t* __restrict__ page_table,
    const uint16_t* __restrict__ expert_ids,
    const float* __restrict__ expert_weights_in,
    int layer, int num_experts_total, int num_active,
    int up_data_size, int gate_data_size, int down_data_size,
    const uint8_t** __restrict__ up_w_out,
    const unsigned short** __restrict__ up_s_out,
    const uint8_t** __restrict__ gate_w_out,
    const unsigned short** __restrict__ gate_s_out,
    const uint8_t** __restrict__ down_w_out,
    const unsigned short** __restrict__ down_s_out,
    float* __restrict__ ew_out
) {
    int idx = threadIdx.x;
    if (idx >= num_active) return;

    uint16_t eid = expert_ids[idx];
    int base = layer * num_experts_total * 3 + (int)eid * 3;
    uint64_t up_ptr = page_table[base + 0];
    uint64_t gate_ptr = page_table[base + 1];
    uint64_t down_ptr = page_table[base + 2];

    up_w_out[idx]   = (const uint8_t*)up_ptr;
    up_s_out[idx]   = (const unsigned short*)(up_ptr + up_data_size);
    gate_w_out[idx] = (const uint8_t*)gate_ptr;
    gate_s_out[idx] = (const unsigned short*)(gate_ptr + gate_data_size);
    down_w_out[idx] = (const uint8_t*)down_ptr;
    down_s_out[idx] = (const unsigned short*)(down_ptr + down_data_size);
    ew_out[idx]     = expert_weights_in[idx];
}

int vib3_launch_moe_experts_fused_gpu(
    const void* act_fp4, const void* act_scales,
    const uint64_t* page_table,
    const uint16_t* expert_ids,
    const float* expert_weights,
    int layer, int num_experts_total, int num_active,
    int K_in, int M_mid, int K_mid, int M_out,
    float* layer_output,
    void* stream
) {
    if (get_sm_major() < 12) return -1;
    if (num_active <= 0 || num_active > MOE_MAX_EXPERTS) return -2;
    cudaStream_t s = (cudaStream_t)stream;

    ensure_moe_bufs(M_mid, K_mid);
    ensure_moe_device_ptrs();
    ensure_gpu_moe_ptrs();

    // One-time upload of intermediate buffer pointers
    if (!d_gpu_moe_bufs_uploaded) {
        cudaMemcpy((void*)d_moe_output_ptrs, g_moe_up_bufs, MOE_MAX_EXPERTS * sizeof(void*), cudaMemcpyHostToDevice);
        cudaMemcpy((void*)d_moe_gate_ptrs, g_moe_gate_bufs, MOE_MAX_EXPERTS * sizeof(void*), cudaMemcpyHostToDevice);
        cudaMemcpy((void*)d_moe_up_ptrs, g_moe_up_bufs, MOE_MAX_EXPERTS * sizeof(void*), cudaMemcpyHostToDevice);
        cudaMemcpy((void*)d_moe_mid_fp4_ptrs, g_moe_mid_fp4, MOE_MAX_EXPERTS * sizeof(void*), cudaMemcpyHostToDevice);
        cudaMemcpy((void*)d_moe_mid_scale_ptrs, g_moe_mid_scales, MOE_MAX_EXPERTS * sizeof(void*), cudaMemcpyHostToDevice);
        d_gpu_moe_bufs_uploaded = true;
    }

    int packed_k_in = (K_in + 1) / 2;
    int packed_k_mid = (K_mid + 1) / 2;
    int up_data_size = packed_k_in * M_mid;
    int gate_data_size = packed_k_in * M_mid;  // same as up
    int down_data_size = packed_k_mid * M_out;

    // ── Step 0: Resolve expert IDs → pointer arrays (1 warp, ~1μs) ──
    vib3_moe_resolve_ptrs_kernel<<<1, 32, 0, s>>>(
        page_table, expert_ids, expert_weights,
        layer, num_experts_total, num_active,
        up_data_size, gate_data_size, down_data_size,
        d_gpu_up_weight_ptrs, d_gpu_up_scale_ptrs,
        d_gpu_gate_weight_ptrs, d_gpu_gate_scale_ptrs,
        d_gpu_down_weight_ptrs, d_gpu_down_scale_ptrs,
        d_gpu_expert_weights
    );

    // ── K-parallel grid computation for up/gate (M=M_mid, K=K_in) ──
    const int warps_per_block = 4;
    dim3 block(32, warps_per_block);

    int packed_k_in_val = K_in / 2;
    int num_k_tiles_in = packed_k_in_val / 32;
    int upgate_tiles = (M_mid + 15) / 16;
    int upgate_grid_x = (upgate_tiles + warps_per_block - 1) / warps_per_block;

    int target_blocks = 84 * 8;  // 672
    int upgate_base_blocks = upgate_grid_x * num_active;
    int upgate_k_blocks = (target_blocks + upgate_base_blocks - 1) / upgate_base_blocks;
    if (upgate_k_blocks < 1) upgate_k_blocks = 1;
    if (upgate_k_blocks > num_k_tiles_in) upgate_k_blocks = num_k_tiles_in;
    int upgate_k_tiles_per_block = (num_k_tiles_in + upgate_k_blocks - 1) / upgate_k_blocks;

    // ── Step 1: Multi-expert up GEMV (tiled + K-parallel) ──
    // Zero up buffers for atomicAdd accumulation
    for (int i = 0; i < num_active; i++) {
        cudaMemsetAsync(g_moe_up_bufs[i], 0, M_mid * sizeof(float), s);
    }
    {
        dim3 grid(upgate_grid_x, upgate_k_blocks, num_active);
        vib3_moe_multi_gemv_kernel<<<grid, block, 0, s>>>(
            (const uint8_t*)act_fp4, (const uint8_t*)act_scales,
            d_gpu_up_weight_ptrs, d_gpu_up_scale_ptrs,
            d_moe_output_ptrs,  // → g_moe_up_bufs (intermediate FP32)
            K_in, M_mid, upgate_k_tiles_per_block
        );
    }

    // ── Step 2: Multi-expert gate GEMV (tiled + K-parallel) ──
    // Zero gate buffers for atomicAdd accumulation
    for (int i = 0; i < num_active; i++) {
        cudaMemsetAsync(g_moe_gate_bufs[i], 0, M_mid * sizeof(float), s);
    }
    {
        dim3 grid(upgate_grid_x, upgate_k_blocks, num_active);
        vib3_moe_multi_gemv_kernel<<<grid, block, 0, s>>>(
            (const uint8_t*)act_fp4, (const uint8_t*)act_scales,
            d_gpu_gate_weight_ptrs, d_gpu_gate_scale_ptrs,
            (float**)d_moe_gate_ptrs,  // → g_moe_gate_bufs (intermediate FP32)
            K_in, M_mid, upgate_k_tiles_per_block
        );
    }

    // ── Step 3: Multi-expert SwiGLU + quantize FP4 ──
    {
        int num_tiles = (K_mid + 63) / 64;
        dim3 grid(num_tiles, num_active);
        vib3_moe_swiglu_quantize_kernel<<<grid, 32, 0, s>>>(
            (float* const*)d_moe_gate_ptrs,
            (float* const*)d_moe_up_ptrs,
            (uint8_t* const*)d_moe_mid_fp4_ptrs,
            (uint8_t* const*)d_moe_mid_scale_ptrs,
            K_mid
        );
    }

    // ── K-parallel grid computation for down (M=M_out, K=K_mid) ──
    int packed_k_mid_val = K_mid / 2;
    int num_k_tiles_mid = packed_k_mid_val / 32;
    int down_tiles = (M_out + 15) / 16;
    int down_grid_x = (down_tiles + warps_per_block - 1) / warps_per_block;

    int down_base_blocks = down_grid_x * num_active;
    int down_k_blocks = (target_blocks + down_base_blocks - 1) / down_base_blocks;
    if (down_k_blocks < 1) down_k_blocks = 1;
    if (down_k_blocks > num_k_tiles_mid) down_k_blocks = num_k_tiles_mid;
    int down_k_tiles_per_block = (num_k_tiles_mid + down_k_blocks - 1) / down_k_blocks;

    // ── Step 4: Multi-expert down GEMV + weighted accumulate (tiled + K-parallel) ──
    // Note: layer_output is already pre-zeroed by caller
    {
        dim3 grid(down_grid_x, down_k_blocks, num_active);
        vib3_moe_down_accumulate_kernel<<<grid, block, 0, s>>>(
            (const uint8_t* const*)d_moe_mid_fp4_ptrs,
            (const uint8_t* const*)d_moe_mid_scale_ptrs,
            (const uint8_t* const*)d_gpu_down_weight_ptrs,
            (const unsigned short* const*)d_gpu_down_scale_ptrs,
            d_gpu_expert_weights,
            layer_output,
            K_mid, M_out, down_k_tiles_per_block
        );
    }

    return 0;
}

// ════════════════════════════════════════════════════════════════════════════
// FP16 Weight → NVFP4 MMA Format Conversion (runtime quantization)
// ════════════════════════════════════════════════════════════════════════════
//
// Converts an FP16 weight matrix [M, K] into the NVFP4 MMA-compatible format:
//   - Data section: M × (K/2) bytes, FP4 E2M1 nibbles in split-half packing
//   - Scale section: M × (K/32) × 2 bytes, BF16 block scales
//
// The output layout matches what vib3_gemv_mma_nvfp4_preq_norepack_kernel and
// vib3_moe_multi_gemv_kernel expect:
//   scales[row * num_groups + group] = bf16 where bf16 = (e8m0 << 7)
//
// Grid: blockIdx.x = row, blockIdx.y = K/32 group index
// Block: 32 threads (one warp)

__global__ void vib3_fp16_to_nvfp4_weight_kernel(
    const __half* __restrict__ input,       // [M, K] FP16 weight matrix
    uint8_t* __restrict__ out_data,         // [M * K/2] split-half FP4 output
    unsigned short* __restrict__ out_scales, // [M * (K/32)] BF16 scales
    int M, int K
) {
    const int row = blockIdx.x;
    const int group = blockIdx.y;
    const int lane = threadIdx.x;
    if (row >= M) return;

    const int k_base = group * 32;
    if (k_base >= K) return;

    // Load one FP16 value per thread (32 threads = 32 values = 1 scale group)
    float val = 0.0f;
    if (k_base + lane < K) {
        val = __half2float(input[(long long)row * K + k_base + lane]);
    }

    // Warp-wide amax reduction for E8M0 scale
    float amax = fabsf(val);
    #pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
        amax = fmaxf(amax, __shfl_xor_sync(0xFFFFFFFF, amax, mask, 32));
    }

    uint8_t e = compute_e8m0(amax);
    float inv_s = (amax == 0.0f) ? 0.0f : __frcp_rn(e8m0_to_f32(e));
    float scaled_val = val * inv_s;

    // Pack into split-half FP4 format
    // Within a group of 32: split into low half [0..15] and high half [16..31]
    // Byte j contains: low_nibble = elem[j], high_nibble = elem[j+16]
    // This is done via __nv_fp4x4_e2m1 which packs 4 values into 2 bytes.
    //
    // The MMA instruction reads data in 64-element tiles (2 groups of 32).
    // Each 32-element group occupies 16 bytes in split-half format.
    // Layout within a K/64 tile:
    //   bytes [0..15]:  group 0 (first 32 elements)
    //   bytes [16..31]: group 1 (next 32 elements)
    //
    // Within 16 bytes for a group, thread mapping for __nv_fp4x4_e2m1:
    //   group_id = lane / 4 (0..7), each writes 2 bytes at offset group_id*2
    //   fp4_packed = {elem[base], elem[base+16], elem[base+1], elem[base+17]}
    //   where base = group_id * 2
    const int group_id = lane / 4;
    const int lane_in_group = lane % 4;
    const int base = group_id * 2;

    float val0 = __shfl_sync(0xFFFFFFFF, scaled_val, base, 32);
    float val1 = __shfl_sync(0xFFFFFFFF, scaled_val, base + 16, 32);
    float val2 = __shfl_sync(0xFFFFFFFF, scaled_val, base + 1, 32);
    float val3 = __shfl_sync(0xFFFFFFFF, scaled_val, base + 17, 32);

#if __CUDA_ARCH__ >= 1200
    if (lane_in_group == 0) {
        __nv_fp4x4_e2m1 fp4_packed(make_float4(val0, val1, val2, val3));
        // Location within row's data:
        // tile_idx = group / 2 (which K/64 tile)
        // sub_group = group % 2 (which half of the tile)
        int tile_idx = group / 2;
        int sub_group = group % 2;
        long long byte_offset = (long long)row * (K / 2) + tile_idx * 32 + sub_group * 16 + group_id * 2;
        *(uint16_t*)(out_data + byte_offset) = *(uint16_t*)&fp4_packed;
    }
#else
    // Software fallback: manual FP4 quantization for pre-Blackwell
    if (lane_in_group == 0) {
        auto to_fp4 = [](float v) -> uint8_t {
            // FP4 E2M1: {0, 0.5, 1, 1.5, 2, 3, 4, 6}, sign bit 3
            float av = fabsf(v);
            uint8_t nibble;
            if (av < 0.25f) nibble = 0;
            else if (av < 0.75f) nibble = 1;
            else if (av < 1.25f) nibble = 2;
            else if (av < 1.75f) nibble = 3;
            else if (av < 2.5f) nibble = 4;
            else if (av < 3.5f) nibble = 5;
            else if (av < 5.0f) nibble = 6;
            else nibble = 7;
            if (v < 0.0f) nibble |= 8;
            return nibble;
        };
        uint8_t n0 = to_fp4(val0);
        uint8_t n1 = to_fp4(val1);
        uint8_t n2 = to_fp4(val2);
        uint8_t n3 = to_fp4(val3);
        // Split-half: byte0 = {n0_lo, n1_lo}, byte1 = {n2_lo, n3_lo}
        uint8_t byte0 = (n0 & 0xF) | ((n1 & 0xF) << 4);
        uint8_t byte1 = (n2 & 0xF) | ((n3 & 0xF) << 4);
        int tile_idx = group / 2;
        int sub_group = group % 2;
        long long byte_offset = (long long)row * (K / 2) + tile_idx * 32 + sub_group * 16 + group_id * 2;
        out_data[byte_offset] = byte0;
        out_data[byte_offset + 1] = byte1;
    }
#endif

    // Thread 0 writes the BF16 scale for this group
    if (lane == 0) {
        unsigned short bf16_scale = (unsigned short)e << 7;
        out_scales[(long long)row * (K / 32) + group] = bf16_scale;
    }
}

// Launcher: convert FP16 weight matrix [M, K] to NVFP4 MMA format.
// out_data must be pre-allocated: M * K/2 bytes
// out_scales must be pre-allocated: M * (K/32) * 2 bytes
// Returns 0 on success.
int vib3_launch_fp16_to_nvfp4_weight(
    const void* input,       // [M, K] FP16
    void* out_data,          // [M * K/2] FP4 data
    void* out_scales,        // [M * (K/32)] BF16 scales
    int M, int K,
    void* stream
) {
    int num_groups = K / 32;
    dim3 grid(M, num_groups);
    dim3 block(32);
    cudaStream_t s = (cudaStream_t)stream;
    vib3_fp16_to_nvfp4_weight_kernel<<<grid, block, 0, s>>>(
        (const __half*)input,
        (uint8_t*)out_data,
        (unsigned short*)out_scales,
        M, K
    );
    return 0;
}

// ── Batched multi-matrix GEMV kernel (zero-copy arguments) ──
// Processes up to 3 NVFP4 weight matrices sharing the same FP4 activation input and K.
// All pointer/M data is passed directly as kernel arguments (no device-side indirection).
// blockIdx.y selects which matrix; blockIdx.x selects tile within that matrix.
// Grid: (max_tiles, num_matrices), block: (32, warps_per_block)
//
// This eliminates multiple sequential kernel launches when projections share the same
// quantized activation (e.g., DeltaNet QKV+Z, GQA Q+K+V).
__global__ void vib3_batched_gemv_mma_nvfp4_preq_kernel(
    const uint8_t* __restrict__ act_fp4,        // [K/2] shared pre-quantized FP4
    const uint8_t* __restrict__ act_scales,      // [K/32] shared E8M0 scales
    // Matrix 0
    const uint8_t* __restrict__ weight0, const unsigned short* __restrict__ scales0,
    float* __restrict__ output0, int M0,
    // Matrix 1
    const uint8_t* __restrict__ weight1, const unsigned short* __restrict__ scales1,
    float* __restrict__ output1, int M1,
    // Matrix 2
    const uint8_t* __restrict__ weight2, const unsigned short* __restrict__ scales2,
    float* __restrict__ output2, int M2,
    int K
) {
    const int mat_idx = blockIdx.y;

    // Select matrix based on blockIdx.y
    const uint8_t* weight;
    const unsigned short* scales;
    float* output;
    int M_slice;
    if (mat_idx == 0) { weight = weight0; scales = scales0; output = output0; M_slice = M0; }
    else if (mat_idx == 1) { weight = weight1; scales = scales1; output = output1; M_slice = M1; }
    else { weight = weight2; scales = scales2; output = output2; M_slice = M2; }

    const int warp_id = threadIdx.y;
    const int lane_id = threadIdx.x;
    const int num_warps = blockDim.y;

    const int tile_row = (blockIdx.x * num_warps + warp_id) * 16;
    if (tile_row >= M_slice) return;

    const int packed_k = K / 2;
    const int num_groups = K / 32;
    const int groupID = lane_id >> 2;
    const int tid_in_grp = lane_id & 3;

    float c0 = 0.0f, c1 = 0.0f, c2 = 0.0f, c3 = 0.0f;

    for (int k0 = 0; k0 < K; k0 += 64) {
        int tile_idx = k0 / 64;
        const int* act_qs = (const int*)(act_fp4 + tile_idx * 32);
        int b0 = act_qs[tid_in_grp];
        int b1 = act_qs[tid_in_grp + 4];
        uint8_t as0 = act_scales[tile_idx * 2];
        uint8_t as1 = act_scales[tile_idx * 2 + 1];
        uint32_t sb = (uint32_t)as0 | ((uint32_t)as1 << 8);

        int a0 = 0, a1 = 0, a2 = 0, a3 = 0;
        int row0 = tile_row + groupID;
        int row1 = tile_row + groupID + 8;
        int k_byte_offset_blk0 = k0 / 2;
        int k_byte_offset_blk1 = k0 / 2 + 16;

        if (row0 < M_slice) {
            const uint8_t* r0 = weight + (long long)row0 * packed_k;
            a0 = ((const int*)(r0 + k_byte_offset_blk0))[tid_in_grp];
            a2 = ((const int*)(r0 + k_byte_offset_blk1))[tid_in_grp];
        }
        if (row1 < M_slice) {
            const uint8_t* r1 = weight + (long long)row1 * packed_k;
            a1 = ((const int*)(r1 + k_byte_offset_blk0))[tid_in_grp];
            a3 = ((const int*)(r1 + k_byte_offset_blk1))[tid_in_grp];
        }

        int tidx = groupID + (tid_in_grp & 1) * 8;
        int scale_row = tile_row + tidx;
        uint32_t sa = 0;
        if (scale_row < M_slice) {
            int g0 = k0 / 32;
            unsigned short bf16_s0 = scales[(long long)scale_row * num_groups + g0];
            unsigned short bf16_s1 = scales[(long long)scale_row * num_groups + g0 + 1];
            uint8_t e0 = (bf16_s0 >> 7) & 0xFF;
            uint8_t e1 = (bf16_s1 >> 7) & 0xFF;
            sa = (uint32_t)e0 | ((uint32_t)e1 << 8);
        }

#if __CUDA_ARCH__ >= 1200
        asm volatile(
            "mma.sync.aligned.kind::mxf4.block_scale.scale_vec::2X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue8m0 "
            "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, "
            "%10, {0, 0}, %11, {0, 0};"
            : "+f"(c0), "+f"(c1), "+f"(c2), "+f"(c3)
            : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
              "r"(b0), "r"(b1),
              "r"(sa), "r"(sb)
        );
#endif
    }

    if (tid_in_grp == 0) {
        int row0 = tile_row + groupID;
        int row1 = tile_row + groupID + 8;
        if (row0 < M_slice) output[row0] = c0;
        if (row1 < M_slice) output[row1] = c2;
    }
}

int vib3_launch_batched_gemv_mma_nvfp4_preq(
    const void* act_fp4, const void* act_scales,
    int num_matrices,
    const void* const* weight_pages,  // [num_matrices] NVFP4 buffer ptrs (data+scales packed)
    const int* m_slices,              // [num_matrices] M per matrix
    void* const* outputs,             // [num_matrices] FP32 output ptrs
    int K,
    void* stream
) {
    if (get_sm_major() < 12) return -1;
    if (num_matrices <= 0 || num_matrices > 3) return -2;
    cudaStream_t s = (cudaStream_t)stream;

    int packed_k = K / 2;

    // Extract weight/scale pointers for each matrix (null for unused slots)
    const uint8_t* w[3] = {0, 0, 0};
    const unsigned short* sc[3] = {0, 0, 0};
    float* out[3] = {0, 0, 0};
    int ms[3] = {0, 0, 0};
    int max_m = 0;

    for (int i = 0; i < num_matrices; i++) {
        w[i] = (const uint8_t*)weight_pages[i];
        sc[i] = (const unsigned short*)(w[i] + (long long)m_slices[i] * packed_k);
        out[i] = (float*)outputs[i];
        ms[i] = m_slices[i];
        if (ms[i] > max_m) max_m = ms[i];
    }

    const int warps_per_block = 4;
    dim3 block(32, warps_per_block);
    int max_tiles = (max_m + 15) / 16;
    int grid_x = (max_tiles + warps_per_block - 1) / warps_per_block;
    dim3 grid(grid_x, num_matrices);

    vib3_batched_gemv_mma_nvfp4_preq_kernel<<<grid, block, 0, s>>>(
        (const uint8_t*)act_fp4,
        (const uint8_t*)act_scales,
        w[0], sc[0], out[0], ms[0],
        w[1], sc[1], out[1], ms[1],
        w[2], sc[2], out[2], ms[2],
        K
    );

    return 0;
}

// ── Batched tiled GEMV kernel: K-parallel + atomicAdd, up to 5 matrices ──
// Like vib3_gemv_mma_nvfp4_tiled_kernel but processes multiple matrices in a
// single launch. blockIdx.z selects the matrix, blockIdx.y for K-parallel,
// blockIdx.x for row tiles. All matrices share the same FP4 activation and K.
// Weights must be in TILED layout. Output must be pre-zeroed.
__global__ void vib3_batched_gemv_mma_nvfp4_tiled_kernel(
    const uint8_t* __restrict__ act_fp4,        // [K/2] shared pre-quantized FP4
    const uint8_t* __restrict__ act_scales,      // [K/32] shared E8M0 scales
    // Matrix pointers passed as arrays (indexed by blockIdx.z)
    const uint8_t* __restrict__ weight0, const unsigned short* __restrict__ scales0,
    float* __restrict__ output0, int M0,
    const uint8_t* __restrict__ weight1, const unsigned short* __restrict__ scales1,
    float* __restrict__ output1, int M1,
    const uint8_t* __restrict__ weight2, const unsigned short* __restrict__ scales2,
    float* __restrict__ output2, int M2,
    const uint8_t* __restrict__ weight3, const unsigned short* __restrict__ scales3,
    float* __restrict__ output3, int M3,
    const uint8_t* __restrict__ weight4, const unsigned short* __restrict__ scales4,
    float* __restrict__ output4, int M4,
    int K,
    int k_tiles_per_block
) {
    const int mat_idx = blockIdx.z;

    // Select matrix based on blockIdx.z
    const uint8_t* weight;
    const unsigned short* wscales;
    float* output;
    int M_slice;
    if (mat_idx == 0)      { weight = weight0; wscales = scales0; output = output0; M_slice = M0; }
    else if (mat_idx == 1) { weight = weight1; wscales = scales1; output = output1; M_slice = M1; }
    else if (mat_idx == 2) { weight = weight2; wscales = scales2; output = output2; M_slice = M2; }
    else if (mat_idx == 3) { weight = weight3; wscales = scales3; output = output3; M_slice = M3; }
    else                   { weight = weight4; wscales = scales4; output = output4; M_slice = M4; }

    if (M_slice <= 0) return;

    const int warp_id = threadIdx.y;
    const int lane_id = threadIdx.x;
    const int num_warps = blockDim.y;

    const int row_block = blockIdx.x * num_warps + warp_id;  // which 16-row block
    const int tile_row = row_block * 16;
    if (tile_row >= M_slice) return;

    const int packed_k = K / 2;
    const int num_k_tiles = packed_k / 32;
    const int num_groups = K / 32;

    // K-range for this block (gridDim.y direction)
    const int k_block_id = blockIdx.y;
    const int k_tile_start = k_block_id * k_tiles_per_block;
    int k_tile_end = k_tile_start + k_tiles_per_block;
    if (k_tile_end > num_k_tiles) k_tile_end = num_k_tiles;

    const int groupID = lane_id >> 2;
    const int tid_in_grp = lane_id & 3;

    // Scale row indices for this thread (MMA scale_vec::2X layout)
    int tidx = groupID + (tid_in_grp & 1) * 8;
    int scale_row = tile_row + tidx;

    float c0 = 0.0f, c1 = 0.0f, c2 = 0.0f, c3 = 0.0f;

    // Base pointer for this warp's tile sequence in tiled layout
    // In tiled layout: row_block's data for k_tile is at (row_block * num_k_tiles + k_tile) * 512
    const int* wt_base = (const int*)(weight + (long long)row_block * num_k_tiles * 512);

    // Pre-compute scale base pointer
    const unsigned short* sr = (scale_row < M_slice)
        ? wscales + (long long)scale_row * num_groups : nullptr;

    for (int k_tile = k_tile_start; k_tile < k_tile_end; k_tile++) {
        // ── Direct register loads from tiled layout ──
        const int* tile = wt_base + k_tile * 128;  // 512 bytes / 4 = 128 ints
        int a0 = tile[groupID * 8 + tid_in_grp];
        int a2 = tile[groupID * 8 + 4 + tid_in_grp];
        int a1 = tile[(groupID + 8) * 8 + tid_in_grp];
        int a3 = tile[(groupID + 8) * 8 + 4 + tid_in_grp];

        // ── Activation loads ──
        const int* act_qs = (const int*)(act_fp4 + k_tile * 32);
        int b0 = act_qs[tid_in_grp];
        int b1 = act_qs[tid_in_grp + 4];
        uint8_t as0 = act_scales[k_tile * 2];
        uint8_t as1 = act_scales[k_tile * 2 + 1];
        uint32_t sb = (uint32_t)as0 | ((uint32_t)as1 << 8);

        // ── Weight scales ──
        uint32_t sa = 0;
        if (sr) {
            int g0 = k_tile * 2;
            unsigned short bf16_s0 = sr[g0];
            unsigned short bf16_s1 = sr[g0 + 1];
            uint8_t e0 = (bf16_s0 >> 7) & 0xFF;
            uint8_t e1 = (bf16_s1 >> 7) & 0xFF;
            sa = (uint32_t)e0 | ((uint32_t)e1 << 8);
        }

#if __CUDA_ARCH__ >= 1200
        asm volatile(
            "mma.sync.aligned.kind::mxf4.block_scale.scale_vec::2X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue8m0 "
            "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, "
            "%10, {0, 0}, %11, {0, 0};"
            : "+f"(c0), "+f"(c1), "+f"(c2), "+f"(c3)
            : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
              "r"(b0), "r"(b1),
              "r"(sa), "r"(sb)
        );
#endif
    }

    // ── Write output via atomicAdd (K-parallel accumulation) ──
    if (tid_in_grp == 0) {
        int row0 = tile_row + groupID;
        int row1 = tile_row + groupID + 8;
        if (row0 < M_slice) atomicAdd(&output[row0], c0);
        if (row1 < M_slice) atomicAdd(&output[row1], c2);
    }
}

// ── Batched tiled GEMV launcher (K-parallel) ──
// Up to 5 matrices, all sharing the same FP4 activation and K.
// Weights must be in TILED layout. Outputs are pre-zeroed by this launcher.
extern "C" int vib3_launch_batched_gemv_mma_nvfp4_tiled(
    const void* act_fp4, const void* act_scales,
    int num_matrices,
    // Each matrix: tiled_weight_ptr, scales_ptr (row-major BF16), output_ptr, M
    // Passed as arrays of pointers/ints
    const void* const* tiled_weights,
    const void* const* scale_ptrs,
    void* const* output_ptrs,
    const int* m_slices,
    int K,
    void* stream
) {
    if (get_sm_major() < 12) return -1;
    if (num_matrices <= 0 || num_matrices > 5) return -2;
    cudaStream_t s = (cudaStream_t)stream;

    const int warps_per_block = 4;
    const int packed_k = K / 2;
    const int num_k_tiles = packed_k / 32;

    // Find max M to determine grid_x
    int max_m = 0;
    for (int i = 0; i < num_matrices; i++) {
        if (m_slices[i] > max_m) max_m = m_slices[i];
    }

    int row_blocks = (max_m + 15) / 16;
    int grid_x = (row_blocks + warps_per_block - 1) / warps_per_block;

    // Target ~672 total row×k blocks for good occupancy
    int target_blocks = 84 * 8;  // 672
    // Adjust target by num_matrices (more matrices = fewer k_blocks needed per matrix)
    int effective_grid_x = grid_x * num_matrices;
    int k_blocks = (target_blocks + effective_grid_x - 1) / effective_grid_x;
    if (k_blocks < 1) k_blocks = 1;
    if (k_blocks > num_k_tiles) k_blocks = num_k_tiles;
    int k_tiles_per_block = (num_k_tiles + k_blocks - 1) / k_blocks;

    // Zero ALL output buffers for atomicAdd accumulation
    for (int i = 0; i < num_matrices; i++) {
        cudaMemsetAsync(output_ptrs[i], 0, m_slices[i] * sizeof(float), s);
    }

    // Extract pointers (null for unused slots)
    const uint8_t* w[5] = {0, 0, 0, 0, 0};
    const unsigned short* sc[5] = {0, 0, 0, 0, 0};
    float* out[5] = {0, 0, 0, 0, 0};
    int ms[5] = {0, 0, 0, 0, 0};

    for (int i = 0; i < num_matrices; i++) {
        w[i] = (const uint8_t*)tiled_weights[i];
        sc[i] = (const unsigned short*)scale_ptrs[i];
        out[i] = (float*)output_ptrs[i];
        ms[i] = m_slices[i];
    }

    dim3 block(32, warps_per_block);
    dim3 grid(grid_x, k_blocks, num_matrices);

    vib3_batched_gemv_mma_nvfp4_tiled_kernel<<<grid, block, 0, s>>>(
        (const uint8_t*)act_fp4,
        (const uint8_t*)act_scales,
        w[0], sc[0], out[0], ms[0],
        w[1], sc[1], out[1], ms[1],
        w[2], sc[2], out[2], ms[2],
        w[3], sc[3], out[3], ms[3],
        w[4], sc[4], out[4], ms[4],
        K, k_tiles_per_block
    );

    return (int)cudaGetLastError();
}

// ═══════════════════════════════════════════════════════════════════════════
// MoE GPU buffer pre-warming (must be called BEFORE graph capture)
// ═══════════════════════════════════════════════════════════════════════════
//
// Pre-allocates all MoE intermediate buffers, device pointer arrays, and
// uploads buffer pointers to device. This ensures that
// vib3_launch_moe_experts_fused_gpu() will not call cudaMalloc or synchronous
// cudaMemcpy during graph capture (both are illegal during capture).

int vib3_moe_prewarm_gpu_bufs(int M_mid, int K_mid) {
    // 1. Allocate per-expert intermediate buffers (8 × {up, gate, mid_fp4, mid_scales})
    ensure_moe_bufs(M_mid, K_mid);

    // 2. Allocate device-side pointer arrays used by fused kernels
    ensure_moe_device_ptrs();

    // 3. Allocate GPU-only MoE pointer arrays (resolve kernel outputs)
    ensure_gpu_moe_ptrs();

    // 4. Upload intermediate buffer pointers to device (synchronous — safe before capture)
    if (!d_gpu_moe_bufs_uploaded) {
        cudaMemcpy((void*)d_moe_output_ptrs, g_moe_up_bufs, MOE_MAX_EXPERTS * sizeof(void*), cudaMemcpyHostToDevice);
        cudaMemcpy((void*)d_moe_gate_ptrs, g_moe_gate_bufs, MOE_MAX_EXPERTS * sizeof(void*), cudaMemcpyHostToDevice);
        cudaMemcpy((void*)d_moe_up_ptrs, g_moe_up_bufs, MOE_MAX_EXPERTS * sizeof(void*), cudaMemcpyHostToDevice);
        cudaMemcpy((void*)d_moe_mid_fp4_ptrs, g_moe_mid_fp4, MOE_MAX_EXPERTS * sizeof(void*), cudaMemcpyHostToDevice);
        cudaMemcpy((void*)d_moe_mid_scale_ptrs, g_moe_mid_scales, MOE_MAX_EXPERTS * sizeof(void*), cudaMemcpyHostToDevice);
        d_gpu_moe_bufs_uploaded = true;
    }

    return (int)cudaGetLastError();
}

// ═══════════════════════════════════════════════════════════════════════════
// CUDA Graph helpers
// ═══════════════════════════════════════════════════════════════════════════

int vib3_cuda_graph_begin_capture(void* stream) {
    cudaStream_t s = (cudaStream_t)stream;
    return (int)cudaStreamBeginCapture(s, cudaStreamCaptureModeRelaxed);
}

// Returns the capture status of the stream: 0=none, 1=active, 2=invalidated, -1=error
int vib3_cuda_stream_capture_status(void* stream) {
    cudaStream_t s = (cudaStream_t)stream;
    cudaStreamCaptureStatus status;
    cudaError_t err = cudaStreamGetCaptureInfo(s, &status, NULL);
    if (err != cudaSuccess) return -1;
    return (int)status;
}

int vib3_cuda_graph_end_capture(void* stream, void** graph_out) {
    cudaStream_t s = (cudaStream_t)stream;
    cudaGraph_t graph;
    cudaError_t err = cudaStreamEndCapture(s, &graph);
    if (err != cudaSuccess) return (int)err;
    *graph_out = (void*)graph;
    return 0;
}

int vib3_cuda_graph_instantiate(void** exec_out, void* graph) {
    cudaGraphExec_t exec;
    cudaError_t err = cudaGraphInstantiate(&exec, (cudaGraph_t)graph, 0);
    if (err != cudaSuccess) return (int)err;
    *exec_out = (void*)exec;
    return 0;
}

int vib3_cuda_graph_launch(void* exec, void* stream) {
    return (int)cudaGraphLaunch((cudaGraphExec_t)exec, (cudaStream_t)stream);
}

int vib3_cuda_graph_exec_destroy(void* exec) {
    return (int)cudaGraphExecDestroy((cudaGraphExec_t)exec);
}

int vib3_cuda_graph_destroy(void* graph) {
    return (int)cudaGraphDestroy((cudaGraph_t)graph);
}

// ══════════════════════════════════════════════════════════════════════
// Fused RMSNorm + FP4 Quantize kernel
// Reads FP32 hidden state, applies RMSNorm with FP16 weight, then directly
// quantizes to split-half FP4 + E8M0 scales in a single kernel.
// Optionally produces FP16 normalized output (for DeltaNet alpha/beta paths).
// Eliminates: 2 kernel launches, 1 global memory round-trip of FP32 normed data,
//             and 1 redundant RMS computation.
// ══════════════════════════════════════════════════════════════════════
__global__ void vib3_fused_rms_norm_quantize_fp4_kernel(
    const float* __restrict__ input,       // [K] FP32 hidden state
    const half* __restrict__ norm_weight,   // [K] FP16 norm weight
    uint8_t* __restrict__ act_fp4,         // [K/2] split-half FP4 output
    uint8_t* __restrict__ act_scales,      // [K/32] E8M0 scales output
    half* __restrict__ opt_f16_out,        // [K] optional FP16 output (may be NULL)
    int K,
    float eps
) {
#if __CUDA_ARCH__ >= 1200
    // Strategy: Use shared memory to hold the full normalized vector.
    // Phase 1: All threads cooperate to compute sum-of-squares reduction.
    // Phase 2: Each thread normalizes its elements and writes to shared memory.
    // Phase 3: Each warp quantizes a group of 32 elements from shared memory to FP4.
    //
    // Thread count: 256 threads (8 warps).
    // For K=3072: each thread handles 3072/256 = 12 elements.

    extern __shared__ float smem[];  // [K] for normalized values

    const int tid = threadIdx.x;
    const int nthreads = blockDim.x;

    // Phase 1: Compute sum of squares with warp-level reduction
    float partial_ss = 0.0f;
    for (int d = tid; d < K; d += nthreads) {
        float val = input[d];
        partial_ss += val * val;
    }

    // Warp reduction
    #pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
        partial_ss += __shfl_xor_sync(0xFFFFFFFF, partial_ss, mask, 32);
    }

    // Inter-warp reduction via shared memory
    const int warp_id = tid / 32;
    const int lane_id = tid % 32;
    __shared__ float warp_sums[8];  // max 8 warps for 256 threads
    if (lane_id == 0) {
        warp_sums[warp_id] = partial_ss;
    }
    __syncthreads();

    float total_ss;
    if (tid == 0) {
        total_ss = 0.0f;
        for (int w = 0; w < (nthreads + 31) / 32; w++) {
            total_ss += warp_sums[w];
        }
        warp_sums[0] = total_ss;
    }
    __syncthreads();
    total_ss = warp_sums[0];

    float inv_rms = rsqrtf(total_ss / (float)K + eps);

    // Phase 2: Normalize and write to shared memory (and optionally to FP16 output)
    for (int d = tid; d < K; d += nthreads) {
        float normalized = input[d] * inv_rms * __half2float(norm_weight[d]);
        smem[d] = normalized;
        if (opt_f16_out) {
            opt_f16_out[d] = __float2half(normalized);
        }
    }
    __syncthreads();

    // Phase 3: Quantize from shared memory to FP4
    // Each group of 32 elements = 1 E8M0 scale + 16 bytes of FP4 data (split-half).
    // We have K/64 tiles, each tile = 2 groups of 32.
    // With 256 threads (8 warps), process 8 tiles simultaneously.
    // For K=3072, num_tiles = 3072/64 = 48, so we loop 48/8 = 6 iterations.
    const int num_tiles = K / 64;
    const int warps_total = nthreads / 32;

    for (int tile_idx = warp_id; tile_idx < num_tiles; tile_idx += warps_total) {
        const int k0 = tile_idx * 64;
        const int group_id = lane_id / 4;
        const int lane_in_group = lane_id % 4;
        const int base = group_id * 2;

        // Process 2 groups of 32
        for (int g = 0; g < 2; g++) {
            int k_base = k0 + g * 32;

            // Each thread reads one value from shared memory
            float val = smem[k_base + lane_id];

            // Warp-wide amax reduction
            float amax = fabsf(val);
            #pragma unroll
            for (int mask = 16; mask > 0; mask >>= 1) {
                amax = fmaxf(amax, __shfl_xor_sync(0xFFFFFFFF, amax, mask, 32));
            }

            uint8_t e = compute_e8m0(amax);
            float inv_s = (amax == 0.0f) ? 0.0f : __frcp_rn(e8m0_to_f32(e));
            float scaled_val = val * inv_s;

            // Gather 4 values for split-half packing
            float val0 = __shfl_sync(0xFFFFFFFF, scaled_val, base, 32);
            float val1 = __shfl_sync(0xFFFFFFFF, scaled_val, base + 16, 32);
            float val2 = __shfl_sync(0xFFFFFFFF, scaled_val, base + 1, 32);
            float val3 = __shfl_sync(0xFFFFFFFF, scaled_val, base + 17, 32);

            if (lane_in_group == 0) {
                __nv_fp4x4_e2m1 fp4_packed(make_float4(val0, val1, val2, val3));
                *(uint16_t*)(act_fp4 + tile_idx * 32 + g * 16 + group_id * 2) = *(uint16_t*)&fp4_packed;
            }

            // Thread 0 in warp writes the scale for this group
            if (lane_id == 0) {
                act_scales[tile_idx * 2 + g] = e;
            }
        }
    }
#endif
}

int vib3_launch_fused_rms_norm_quantize_fp4(
    const void* input,         // [K] FP32 hidden state
    const void* norm_weight,   // [K] FP16 norm weight
    void* act_fp4,             // [K/2] split-half FP4 output
    void* act_scales,          // [K/32] E8M0 scales output
    void* opt_f16_out,         // [K] optional FP16 output (NULL to skip)
    int K,
    float eps,
    void* stream
) {
    if (get_sm_major() < 12) return -1;
    cudaStream_t s = (cudaStream_t)stream;
    // Single block, 256 threads. Shared memory: K * sizeof(float) for normalized values.
    int smem_bytes = K * sizeof(float);
    vib3_fused_rms_norm_quantize_fp4_kernel<<<1, 256, smem_bytes, s>>>(
        (const float*)input,
        (const half*)norm_weight,
        (uint8_t*)act_fp4,
        (uint8_t*)act_scales,
        (half*)opt_f16_out,
        K,
        eps
    );
    return 0;
}

int vib3_update_device_int32(void* d_ptr, int value, void* stream) {
    cudaStream_t s = (cudaStream_t)stream;
    cudaError_t err = cudaMemcpyAsync(d_ptr, &value, sizeof(int),
                                      cudaMemcpyHostToDevice, s);
    if (err != cudaSuccess) return (int)err;
    return (int)cudaStreamSynchronize(s);
}

} // extern "C" (reconstructed kernels)
