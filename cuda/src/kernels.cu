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

// ─── Device Helpers ──────────────────────────────────────────────────────

// BF16 → float conversion. BF16 is the upper 16 bits of IEEE 754 float32,
// so conversion is just a 16-bit left shift. No exponent/mantissa remapping.
__device__ __forceinline__ float bf16_to_float(unsigned short bf16) {
    unsigned int bits = ((unsigned int)bf16) << 16;
    float result;
    memcpy(&result, &bits, sizeof(float));
    return result;
}

// FP16 → float conversion for INT4 quantization scales.
// compressed-tensors stores scales as IEEE 754 half-precision (FP16),
// NOT BF16. Using bf16_to_float on FP16 data produces ~1e-23 values
// instead of ~0.003, causing matmul output to flush to zero.
__device__ __forceinline__ float fp16_scale_to_float(unsigned short fp16_bits) {
    return __half2float(*reinterpret_cast<const half*>(&fp16_bits));
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

// Tiled GEMV INT4: dequantize on the fly with per-group scales
// Same tiling strategy as FP16 but with INT4 unpacking
__global__ void vib3_partial_matmul_int4(
    const half* __restrict__ input,
    const uint8_t* __restrict__ weight,
    const unsigned short* __restrict__ scales,  // FP16 scale values (from compressed-tensors)
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
        float scale = fp16_scale_to_float(row_scales[group]);

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
        float scale = fp16_scale_to_float(row_scales[group]);

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
        float up_scale = fp16_scale_to_float(up_s[group]);
        int up_w0 = (int)((up_packed >> 0) & 0xF) - 8;
        int up_w1 = (int)((up_packed >> 4) & 0xF) - 8;

        uint8_t gate_packed = gate_row[byte_idx];
        float gate_scale = fp16_scale_to_float(gate_s[group]);
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

// Tiled GEMV NVFP4: dequantize E2M1 on the fly with per-block FP16 scales
// FP32 input for maximum precision in the accumulation path
__global__ void vib3_partial_matmul_nvfp4(
    const float* __restrict__ input,
    const uint8_t* __restrict__ weight,   // packed E2M1 nibbles (2 per byte)
    const half* __restrict__ scales,      // FP16 scale per block of 32
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
    const half* row_scales = scales + (long long)row * num_blocks;

    float acc = 0.0f;
    for (int k = lane * 2; k < K; k += THREADS_PER_ROW * 2) {
        int byte_idx = k / 2;
        uint8_t packed = row_weight[byte_idx];

        // Unpack two E2M1 nibbles and dequantize via LUT
        float w0 = nvfp4_lut[packed & 0xF];
        float w1 = nvfp4_lut[(packed >> 4) & 0xF];

        // Block scale (FP16)
        int blk = k / block_size;
        float scale = __half2float(row_scales[blk]);

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
__global__ void vib3_fused_swiglu_nvfp4(
    const float* __restrict__ input,
    const uint8_t* __restrict__ up_weight,
    const half* __restrict__ up_scales,
    const uint8_t* __restrict__ gate_weight,
    const half* __restrict__ gate_scales,
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
    const half* up_s = up_scales + (long long)row * num_blocks;
    const uint8_t* gate_row = gate_weight + (long long)row * packed_k;
    const half* gate_s = gate_scales + (long long)row * num_blocks;

    float up_acc = 0.0f;
    float gate_acc = 0.0f;
    for (int k = lane * 2; k < K; k += THREADS_PER_ROW * 2) {
        int byte_idx = k / 2;
        int blk = k / block_size;

        uint8_t up_packed = up_row[byte_idx];
        float up_scale = __half2float(up_s[blk]);
        float up_w0 = nvfp4_lut[up_packed & 0xF];
        float up_w1 = nvfp4_lut[(up_packed >> 4) & 0xF];

        uint8_t gate_packed = gate_row[byte_idx];
        float gate_scale = __half2float(gate_s[blk]);
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
__global__ void vib3_partial_matmul_nvfp4_fp16in(
    const half* __restrict__ input,
    const uint8_t* __restrict__ weight,
    const half* __restrict__ scales,
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
    const half* row_scales = scales + (long long)row * num_blocks;

    float acc = 0.0f;
    for (int k = lane * 2; k < K; k += THREADS_PER_ROW * 2) {
        int byte_idx = k / 2;
        uint8_t packed = row_weight[byte_idx];

        float w0 = nvfp4_lut[packed & 0xF];
        float w1 = nvfp4_lut[(packed >> 4) & 0xF];

        int blk = k / block_size;
        float scale = __half2float(row_scales[blk]);

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
__global__ void vib3_router_gemv_f32(
    const float* __restrict__ hidden_state,
    const half* __restrict__ router_weights,
    float* __restrict__ scores,
    int hidden_dim,
    int num_experts
) {
    int expert = blockIdx.x * blockDim.x + threadIdx.x;
    if (expert >= num_experts) return;

    float acc = 0.0f;
    for (int d = 0; d < hidden_dim; d++) {
        acc += hidden_state[d] *
               __half2float(router_weights[expert * hidden_dim + d]);
    }
    scores[expert] = acc;
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

// FP32 RMSNorm: reads FP32 input from hidden_state_f32, outputs FP32 to separate buffer
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

// FP32→FP16 RMSNorm: normalize in FP32 precision, output FP16
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

// FP32-input INT4 matmul
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

// FP32-input FP16 matmul
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

// FP32-input FP16-weight matmul with FP32 output (avoids FP16 truncation of result)
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
            output[row] = final_acc;
        }
    }
}

// FP32-input FP16-weight matmul, FP32 output — launcher
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

// FP32-input fused SwiGLU INT4
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

// FP32-input fused SwiGLU FP16
int vib3_launch_fused_swiglu_fp16_f32in(
    const void* input, const void* up_weight, const void* gate_weight,
    void* output, int K, int M_slice, void* stream
) {
    cudaStream_t s = (cudaStream_t)stream;
    int blocks = (M_slice + ROWS_PER_BLOCK - 1) / ROWS_PER_BLOCK;
    vib3_fused_swiglu_fp16_f32in<<<blocks, THREADS_PER_ROW * ROWS_PER_BLOCK, 0, s>>>(
        (const float*)input, (const half*)up_weight, (const half*)gate_weight,
        (half*)output, K, M_slice
    );
    return (int)cudaGetLastError();
}

// FP32-input router GEMV
int vib3_launch_router_gemv_f32(
    const void* hidden_state, const void* router_weights, float* scores,
    int hidden_dim, int num_experts, void* stream
) {
    cudaStream_t s = (cudaStream_t)stream;
    int blocks = (num_experts + 255) / 256;
    vib3_router_gemv_f32<<<blocks, 256, 0, s>>>(
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
        (const float*)input,
        (const uint8_t*)weight_packed,
        (const half*)scales,
        (half*)output,
        K, M_slice, block_size
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
        (const float*)input,
        (const uint8_t*)up_weight,
        (const half*)up_scales,
        (const uint8_t*)gate_weight,
        (const half*)gate_scales,
        (half*)output,
        K, M_slice, block_size
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
        (const half*)input,
        (const uint8_t*)weight_packed,
        (const half*)scales,
        (half*)output,
        K, M_slice, block_size
    );
    return (int)cudaGetLastError();
}

} // extern "C" (original launchers)

// ─── MLA Decode Attention Kernel ─────────────────────────────────────────
//
// Fused kernel for Multi-head Latent Attention (MLA) decode.
// Operates on COMPRESSED KV cache (latent + rope) — no kv_b_proj needed during attention.
//
// Input:
//   q_absorbed: [num_heads * kv_lora_rank] F32 — already absorbed: Q_nope × kv_b_k^T
//   q_rope:     [num_heads * qk_rope_dim] F32 — RoPE-applied query rope component
//   kv_latent:  [seq_len * kv_lora_rank] F32 — compressed KV cache (normed)
//   k_rope:     [seq_len * qk_rope_dim] F32 — RoPE'd key rope cache
//
// Output:
//   v_latent_out: [num_heads * kv_lora_rank] F32 — weighted latent accumulation per head
//
// Algorithm per head h:
//   For each position p:
//     score_nope = dot(q_absorbed[h], kv_latent[p])    — [kv_lora_rank] dot product
//     score_rope = dot(q_rope[h], k_rope[p])           — [qk_rope_dim] dot product
//     score = (score_nope + score_rope) * scale
//   softmax over positions
//   v_latent_out[h] = Σ_p softmax[p] * kv_latent[p]   — weighted sum
//
// Optimized MLA decode attention kernel.
//
// Key optimizations vs. previous version:
// 1. Cooperative dot products: 256 threads cooperate on each position's 576-element dot product
//    using warp-level reduction, instead of each thread doing a full serial dot product.
// 2. Scores stored in shared memory (no recomputation for Phase 2).
// 3. V accumulation uses per-thread partial sums (no atomicAdd to shared memory).
// 4. Single-pass online softmax (compute max, exp, and V accumulation incrementally).
//
// Launch: <<<num_heads, 256, smem_bytes>>> where smem_bytes = max(seq_len, kv_lora_rank) * sizeof(float)
// For short seq_len (decode), we have 256 threads cooperating on 576-element dot products.

#define MLA_ATTN_BLOCK 256

__global__ void vib3_mla_decode_attn_kernel(
    const float* __restrict__ q_absorbed,   // [num_heads * kv_lora_rank]
    const float* __restrict__ q_rope,       // [num_heads * qk_rope_dim]
    const float* __restrict__ kv_latent,    // [seq_len * kv_lora_rank]
    const float* __restrict__ k_rope_cache, // [seq_len * qk_rope_dim]
    float* __restrict__ v_latent_out,       // [num_heads * kv_lora_rank]
    int kv_lora_rank,
    int qk_rope_dim,
    int num_heads,
    int seq_len,
    float scale
) {
    int head = blockIdx.x;
    if (head >= num_heads) return;

    int tid = threadIdx.x;
    int total_k_dim = kv_lora_rank + qk_rope_dim; // 576 for Kimi K2.5

    // Per-head Q pointers
    const float* q_abs_h = q_absorbed + head * kv_lora_rank;
    const float* q_rope_h = q_rope + head * qk_rope_dim;

    // Shared memory layout: [seq_len] floats for scores
    extern __shared__ float smem[];
    float* scores = smem; // [seq_len]

    // ── Phase 1: Cooperative dot products → scores[pos] ──
    // Each position's dot product (576 elements) is split across 256 threads.
    // Each thread handles ceil(576/256) ≈ 2-3 elements per position.
    for (int pos = 0; pos < seq_len; pos++) {
        const float* lat = kv_latent + pos * kv_lora_rank;
        const float* kr = k_rope_cache + pos * qk_rope_dim;

        // Each thread computes partial dot product over its assigned dimensions
        float partial = 0.0f;
        // Absorbed (nope) part: kv_lora_rank elements
        for (int j = tid; j < kv_lora_rank; j += MLA_ATTN_BLOCK) {
            partial += q_abs_h[j] * lat[j];
        }
        // Rope part: qk_rope_dim elements
        for (int j = tid; j < qk_rope_dim; j += MLA_ATTN_BLOCK) {
            partial += q_rope_h[j] * kr[j];
        }

        // Warp-level reduction (no shared memory needed for this)
        for (int offset = 16; offset > 0; offset >>= 1) {
            partial += __shfl_down_sync(0xFFFFFFFF, partial, offset);
        }

        // Inter-warp reduction: warp leaders write to shared memory
        __shared__ float warp_sums[8]; // 256 threads / 32 = 8 warps
        int warp_id = tid / 32;
        int lane = tid % 32;
        if (lane == 0) warp_sums[warp_id] = partial;
        __syncthreads();

        if (tid == 0) {
            float total = 0.0f;
            int nwarps = MLA_ATTN_BLOCK / 32;
            for (int w = 0; w < nwarps; w++) total += warp_sums[w];
            scores[pos] = total * scale;
        }
        __syncthreads();
    }

    // ── Phase 2: Softmax on scores ──
    // Find max (tid 0 since seq_len is small)
    float global_max = -1e30f;
    if (tid == 0) {
        for (int p = 0; p < seq_len; p++) {
            if (scores[p] > global_max) global_max = scores[p];
        }
    }
    // Broadcast max via shared memory
    if (tid == 0) smem[0] = global_max;
    __syncthreads();
    global_max = smem[0];

    // Compute exp and sum (tid 0, seq_len is small)
    float total_exp = 0.0f;
    if (tid == 0) {
        for (int p = 0; p < seq_len; p++) {
            float e = expf(scores[p] - global_max);
            scores[p] = e;
            total_exp += e;
        }
        float inv = (total_exp > 0.0f) ? (1.0f / total_exp) : 0.0f;
        for (int p = 0; p < seq_len; p++) {
            scores[p] *= inv;
        }
    }
    __syncthreads();

    // ── Phase 3: Weighted V latent accumulation ──
    // Each thread computes a subset of the kv_lora_rank output dimensions.
    // No atomics needed — threads work on disjoint output elements.
    float* out_h = v_latent_out + head * kv_lora_rank;
    for (int j = tid; j < kv_lora_rank; j += MLA_ATTN_BLOCK) {
        float acc = 0.0f;
        for (int pos = 0; pos < seq_len; pos++) {
            acc += scores[pos] * kv_latent[pos * kv_lora_rank + j];
        }
        out_h[j] = acc;
    }
}

// ─── V Reconstruction GEMV Kernel ────────────────────────────────────────
//
// Given v_latent_weighted[num_heads, kv_lora_rank] and kv_b_proj_v[num_heads, v_head_dim, kv_lora_rank],
// compute v_out[num_heads, v_head_dim]:
//   v_out[h][d] = dot(kv_b_v[h*v_head_dim+d, :], v_latent[h, :])
//
// This is a batched GEMV: num_heads * v_head_dim output values, each a dot product of length kv_lora_rank.
// kv_b_proj layout: [(h*(nope+v)+nope+d) * kv_lora_rank + j] for V row d of head h.
//
// We launch one thread per output element (num_heads * v_head_dim = 64 * 128 = 8192).

__global__ void vib3_mla_v_reconstruct(
    const float* __restrict__ v_latent,      // [num_heads * kv_lora_rank]
    const float* __restrict__ kv_b_proj,     // [(num_heads * (nope+v)), kv_lora_rank] row-major
    float* __restrict__ v_out,               // [num_heads * v_head_dim]
    int kv_lora_rank,
    int qk_nope_dim,
    int v_head_dim,
    int num_heads
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = num_heads * v_head_dim;
    if (idx >= total) return;

    int h = idx / v_head_dim;
    int d = idx % v_head_dim;

    // kv_b_proj row for this output: head_v_offset = h * (nope + v) + nope + d
    int row = h * (qk_nope_dim + v_head_dim) + qk_nope_dim + d;
    const float* kv_row = kv_b_proj + row * kv_lora_rank;
    const float* v_lat_h = v_latent + h * kv_lora_rank;

    float acc = 0.0f;
    for (int j = 0; j < kv_lora_rank; j++) {
        acc += kv_row[j] * v_lat_h[j];
    }
    v_out[idx] = acc;
}

// ─── MLA Launchers ───────────────────────────────────────────────────────

extern "C" {

int vib3_launch_mla_decode_attention(
    const void* q_absorbed, const void* q_rope,
    const void* kv_latent, const void* k_rope_cache,
    void* v_latent_out,
    int kv_lora_rank, int qk_rope_dim,
    int num_heads, int seq_len, float scale,
    void* stream
) {
    cudaStream_t s = (cudaStream_t)stream;
    // Shared memory: max(seq_len, 8) floats for scores + 8 floats for warp_sums
    // (warp_sums is __shared__ inside kernel, not dynamic, so only scores need dynamic smem)
    int smem_bytes = seq_len * sizeof(float);
    if (smem_bytes < 32) smem_bytes = 32; // minimum
    vib3_mla_decode_attn_kernel<<<num_heads, MLA_ATTN_BLOCK, smem_bytes, s>>>(
        (const float*)q_absorbed, (const float*)q_rope,
        (const float*)kv_latent, (const float*)k_rope_cache,
        (float*)v_latent_out,
        kv_lora_rank, qk_rope_dim, num_heads, seq_len, scale
    );
    return (int)cudaGetLastError();
}

int vib3_launch_mla_v_reconstruct(
    const void* v_latent, const void* kv_b_proj,
    void* v_out,
    int kv_lora_rank, int qk_nope_dim, int v_head_dim, int num_heads,
    void* stream
) {
    cudaStream_t s = (cudaStream_t)stream;
    int total = num_heads * v_head_dim;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    vib3_mla_v_reconstruct<<<blocks, threads, 0, s>>>(
        (const float*)v_latent, (const float*)kv_b_proj,
        (float*)v_out,
        kv_lora_rank, qk_nope_dim, v_head_dim, num_heads
    );
    return (int)cudaGetLastError();
}

} // extern "C" (MLA launchers)

// ─── MLA Q Absorption + RoPE Kernel ─────────────────────────────────────
//
// Fused kernel: For each head h:
//   1. Q_absorbed[h][j] = Σ_i q_full_fp16[h*q_head_dim + i] * kv_b_proj_f32[(h*(nope+v) + i)*kv_lora_rank + j]
//      for i in [0, qk_nope_dim), j in [0, kv_lora_rank)
//   2. Q_rope[h][d] = RoPE(q_full_fp16[h*q_head_dim + qk_nope_dim + d], position, freqs[d])
//      for d in [0, qk_rope_dim)
//
// Threading: one block per head (64 blocks), 256 threads per block.
// Each thread computes multiple output elements of Q_absorbed.

#define Q_ABSORB_BLOCK 256

__global__ void vib3_mla_q_absorb_rope_kernel(
    const half* __restrict__ q_full,       // [num_heads * q_head_dim] FP16
    const float* __restrict__ kv_b_proj,   // [(num_heads*(nope+v)) * kv_lora_rank] F32
    const float* __restrict__ rope_freqs,  // [qk_rope_dim/2] precomputed frequencies
    float* __restrict__ q_absorbed_out,    // [num_heads * kv_lora_rank] F32
    float* __restrict__ q_rope_out,        // [num_heads * qk_rope_dim] F32
    int q_head_dim,     // nope + rope (e.g. 192)
    int qk_nope_dim,    // 128
    int qk_rope_dim,    // 64
    int v_head_dim,     // 128
    int kv_lora_rank,   // 512
    int num_heads,      // 64
    int position
) {
    int head = blockIdx.x;
    if (head >= num_heads) return;
    int tid = threadIdx.x;

    // ── Part 1: Q absorption ──
    // Each thread computes a subset of the kv_lora_rank output dims
    int head_row_offset = head * (qk_nope_dim + v_head_dim); // row offset in kv_b_proj
    const half* q_nope = q_full + head * q_head_dim;

    for (int j = tid; j < kv_lora_rank; j += Q_ABSORB_BLOCK) {
        float acc = 0.0f;
        for (int i = 0; i < qk_nope_dim; i++) {
            float q_val = __half2float(q_nope[i]);
            float kv_val = kv_b_proj[(head_row_offset + i) * kv_lora_rank + j];
            acc += q_val * kv_val;
        }
        q_absorbed_out[head * kv_lora_rank + j] = acc;
    }

    // ── Part 2: RoPE on q_rope ──
    // Each thread handles a pair of rope dims
    const half* q_rope_src = q_full + head * q_head_dim + qk_nope_dim;
    int half_rope = qk_rope_dim / 2;
    for (int i = tid; i < half_rope; i += Q_ABSORB_BLOCK) {
        float freq = rope_freqs[i];
        float theta = (float)position * freq;
        float cos_t = cosf(theta);
        float sin_t = sinf(theta);

        float x0 = __half2float(q_rope_src[2 * i]);
        float x1 = __half2float(q_rope_src[2 * i + 1]);
        q_rope_out[head * qk_rope_dim + 2 * i]     = x0 * cos_t - x1 * sin_t;
        q_rope_out[head * qk_rope_dim + 2 * i + 1]  = x0 * sin_t + x1 * cos_t;
    }
}

// ─── MLA KV Cache Append (FP16→F32 + RMSNorm + RoPE) ────────────────────
//
// Takes kv_a output (FP16), splits into latent + rope components,
// applies RMSNorm to latent (with weight), applies RoPE to rope,
// writes both to the GPU KV cache as F32 at the given position.
//
// Single block kernel (kv_lora_rank=512, rope_dim=64 — small enough).

__global__ void vib3_mla_kv_cache_append_kernel(
    const half* __restrict__ kv_a_out,       // [kv_lora_rank + qk_rope_dim] FP16
    const half* __restrict__ kv_norm_weight,  // [kv_lora_rank] FP16 (can be null)
    const float* __restrict__ rope_freqs,    // [qk_rope_dim/2]
    float* __restrict__ kv_latent_cache,     // [max_seq * kv_lora_rank] F32 — write position
    float* __restrict__ k_rope_cache,        // [max_seq * qk_rope_dim] F32 — write position
    int kv_lora_rank,    // 512
    int qk_rope_dim,     // 64
    int position,
    float eps
) {
    extern __shared__ float shared[];
    int tid = threadIdx.x;
    int stride = blockDim.x;

    // ── RMSNorm on latent portion ──
    // Convert FP16→F32 and compute sum of squares
    float partial_ss = 0.0f;
    for (int d = tid; d < kv_lora_rank; d += stride) {
        float val = __half2float(kv_a_out[d]);
        shared[d] = val; // store F32 in shared memory (reuse as temp buffer)
        partial_ss += val * val;
    }

    // Reduce sum of squares
    __shared__ float ss_reduce[256];
    ss_reduce[tid] = partial_ss;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) ss_reduce[tid] += ss_reduce[tid + s];
        __syncthreads();
    }
    float rms = sqrtf(ss_reduce[0] / (float)kv_lora_rank + eps);
    float inv_rms = 1.0f / rms;

    // Write normed latent to cache at position offset
    float* latent_dst = kv_latent_cache + position * kv_lora_rank;
    for (int d = tid; d < kv_lora_rank; d += stride) {
        float val = shared[d] * inv_rms;
        if (kv_norm_weight != nullptr) {
            val *= __half2float(kv_norm_weight[d]);
        }
        latent_dst[d] = val;
    }

    // ── RoPE on rope portion ──
    int half_rope = qk_rope_dim / 2;
    const half* k_rope_src = kv_a_out + kv_lora_rank;
    float* rope_dst = k_rope_cache + position * qk_rope_dim;
    for (int i = tid; i < half_rope; i += stride) {
        float freq = rope_freqs[i];
        float theta = (float)position * freq;
        float cos_t = cosf(theta);
        float sin_t = sinf(theta);

        float x0 = __half2float(k_rope_src[2 * i]);
        float x1 = __half2float(k_rope_src[2 * i + 1]);
        rope_dst[2 * i]     = x0 * cos_t - x1 * sin_t;
        rope_dst[2 * i + 1] = x0 * sin_t + x1 * cos_t;
    }
}

// ─── F32 to FP16 conversion ─────────────────────────────────────────────

__global__ void vib3_f32_to_f16_kernel(
    const float* __restrict__ input,
    half* __restrict__ output,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = __float2half(input[idx]);
    }
}

// ─── FP16 → FP32 conversion ────────────────────────────────────────────

__global__ void vib3_fp16_to_fp32_kernel(
    const half* __restrict__ input,
    float* __restrict__ output,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = __half2float(input[idx]);
    }
}

// ─── FP32 residual accumulation (FP32 += FP16) ─────────────────────────
// In-place: accumulator[i] += __half2float(layer_output[i])
// Used for FP32 hidden state accumulation across transformer layers.

__global__ void vib3_residual_add_fp32_kernel(
    float* __restrict__ accumulator,
    const half* __restrict__ layer_output,
    int dim
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < dim) {
        accumulator[idx] += __half2float(layer_output[idx]);
    }
}

// ─── New MLA Launchers ──────────────────────────────────────────────────

extern "C" {

int vib3_launch_mla_q_absorb_rope(
    const void* q_full, const void* kv_b_proj, const void* rope_freqs,
    void* q_absorbed_out, void* q_rope_out,
    int q_head_dim, int qk_nope_dim, int qk_rope_dim, int v_head_dim,
    int kv_lora_rank, int num_heads, int position,
    void* stream
) {
    cudaStream_t s = (cudaStream_t)stream;
    vib3_mla_q_absorb_rope_kernel<<<num_heads, Q_ABSORB_BLOCK, 0, s>>>(
        (const half*)q_full, (const float*)kv_b_proj, (const float*)rope_freqs,
        (float*)q_absorbed_out, (float*)q_rope_out,
        q_head_dim, qk_nope_dim, qk_rope_dim, v_head_dim,
        kv_lora_rank, num_heads, position
    );
    return (int)cudaGetLastError();
}

int vib3_launch_mla_kv_cache_append(
    const void* kv_a_out, const void* kv_norm_weight, const void* rope_freqs,
    void* kv_latent_cache, void* k_rope_cache,
    int kv_lora_rank, int qk_rope_dim, int position, float eps,
    void* stream
) {
    cudaStream_t s = (cudaStream_t)stream;
    // Shared memory: kv_lora_rank floats for latent temp + 256 floats for reduction
    int smem_bytes = kv_lora_rank * sizeof(float);
    vib3_mla_kv_cache_append_kernel<<<1, 256, smem_bytes, s>>>(
        (const half*)kv_a_out, (const half*)kv_norm_weight, (const float*)rope_freqs,
        (float*)kv_latent_cache, (float*)k_rope_cache,
        kv_lora_rank, qk_rope_dim, position, eps
    );
    return (int)cudaGetLastError();
}

int vib3_launch_f32_to_f16(
    const void* input, void* output, int n, void* stream
) {
    cudaStream_t s = (cudaStream_t)stream;
    int blocks = (n + 255) / 256;
    vib3_f32_to_f16_kernel<<<blocks, 256, 0, s>>>(
        (const float*)input, (half*)output, n
    );
    return (int)cudaGetLastError();
}

int vib3_launch_fp16_to_fp32(
    const void* input, void* output, int n, void* stream
) {
    cudaStream_t s = (cudaStream_t)stream;
    int blocks = (n + 255) / 256;
    vib3_fp16_to_fp32_kernel<<<blocks, 256, 0, s>>>(
        (const half*)input, (float*)output, n
    );
    return (int)cudaGetLastError();
}

int vib3_launch_residual_add_fp32(
    void* accumulator, const void* layer_output, int dim, void* stream
) {
    cudaStream_t s = (cudaStream_t)stream;
    int blocks = (dim + 255) / 256;
    vib3_residual_add_fp32_kernel<<<blocks, 256, 0, s>>>(
        (float*)accumulator, (const half*)layer_output, dim
    );
    return (int)cudaGetLastError();
}

} // extern "C" (new MLA launchers)

// ─── Decode Attention Kernels ─────────────────────────────────────────────
//
// These kernels keep the entire attention sublayer on GPU for decode (seq_len=1 query),
// eliminating the GPU→CPU→GPU round-trip that was the dominant bottleneck.

// ─── RoPE Kernel ─────────────────────────────────────────────────────────

// Each thread rotates one pair (x[2i], x[2i+1]) with the RoPE angle.
__global__ void vib3_rope_kernel(
    half* __restrict__ data,   // [total_heads * head_dim] FP16
    int head_dim,
    int total_heads,
    int position,
    float rope_base
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int half_dim = head_dim / 2;
    int total_pairs = total_heads * half_dim;
    if (idx >= total_pairs) return;

    int head = idx / half_dim;
    int i = idx % half_dim;

    // Compute frequency: freq = 1 / base^(2i / head_dim)
    float freq = 1.0f / powf(rope_base, 2.0f * (float)i / (float)head_dim);
    float theta = (float)position * freq;
    float cos_t = cosf(theta);
    float sin_t = sinf(theta);

    int offset = head * head_dim + 2 * i;
    float x0 = __half2float(data[offset]);
    float x1 = __half2float(data[offset + 1]);
    data[offset]     = __float2half(x0 * cos_t - x1 * sin_t);
    data[offset + 1] = __float2half(x0 * sin_t + x1 * cos_t);
}

// ─── KV Cache Append Kernel ──────────────────────────────────────────────

// Copy new_k/new_v into the cache at the given position.
// Grid: ceil(num_kv_heads * head_dim / 256) blocks.
__global__ void vib3_kv_append_kernel(
    half* __restrict__ k_cache,   // [num_kv_heads, max_seq_len, head_dim]
    half* __restrict__ v_cache,
    const half* __restrict__ new_k,  // [num_kv_heads * head_dim]
    const half* __restrict__ new_v,
    int max_seq_len,
    int head_dim,
    int num_kv_heads,
    int position
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = num_kv_heads * head_dim;
    if (idx >= total) return;

    int h = idx / head_dim;
    int d = idx % head_dim;

    // Cache layout: cache[h * max_seq_len * head_dim + position * head_dim + d]
    int cache_offset = h * max_seq_len * head_dim + position * head_dim + d;
    k_cache[cache_offset] = new_k[idx];
    v_cache[cache_offset] = new_v[idx];
}

// ─── Decode Attention Kernel ─────────────────────────────────────────────
//
// Single-query attention for decode: Q is one vector per head.
// One block per Q head. Threads cooperate across positions.
//
// Phase 1: Compute Q · K^T scores, track max for numerically stable softmax.
// Phase 2: Compute exp(score - max) and accumulate sum + weighted V.
//
// GQA: Q head h maps to KV head (h / heads_per_kv_group).
//
// Thread budget: 256 threads per block, each thread handles
// ceil(seq_len / 256) positions.

#define ATTN_BLOCK_SIZE 256

__global__ void vib3_decode_attn_kernel(
    const half* __restrict__ q,        // [num_heads * head_dim]
    const half* __restrict__ k_cache,  // [num_kv_heads * max_seq_len * head_dim]
    const half* __restrict__ v_cache,
    half* __restrict__ output,         // [num_heads * head_dim]
    int head_dim,
    int num_heads,
    int num_kv_heads,
    int seq_len,
    int max_seq_len,
    float scale
) {
    int head = blockIdx.x;  // one block per Q head
    if (head >= num_heads) return;

    int tid = threadIdx.x;
    int heads_per_kv = num_heads / max(num_kv_heads, 1);
    int kv_head = head / max(heads_per_kv, 1);
    if (kv_head >= num_kv_heads) kv_head = num_kv_heads - 1;

    // Pointers for this head
    const half* q_head = q + head * head_dim;
    const half* k_head_cache = k_cache + kv_head * max_seq_len * head_dim;
    const half* v_head_cache = v_cache + kv_head * max_seq_len * head_dim;

    // ── Phase 1: Compute scores and find max ──
    // Each thread handles multiple positions
    float local_max = -1e30f;
    for (int pos = tid; pos < seq_len; pos += ATTN_BLOCK_SIZE) {
        float score = 0.0f;
        const half* k_pos = k_head_cache + pos * head_dim;
        for (int d = 0; d < head_dim; d++) {
            score += __half2float(q_head[d]) * __half2float(k_pos[d]);
        }
        score *= scale;
        if (score > local_max) local_max = score;
    }

    // Block-level max reduction via shared memory
    __shared__ float smem_max[ATTN_BLOCK_SIZE];
    smem_max[tid] = local_max;
    __syncthreads();
    for (int s = ATTN_BLOCK_SIZE / 2; s > 0; s >>= 1) {
        if (tid < s && smem_max[tid + s] > smem_max[tid]) {
            smem_max[tid] = smem_max[tid + s];
        }
        __syncthreads();
    }
    float global_max = smem_max[0];

    // ── Phase 2: Compute exp(score - max), accumulate sum and weighted V ──
    // Each thread accumulates partial V output and partial exp sum.
    extern __shared__ float smem_v[];  // [head_dim] for the final V accumulation

    // Initialize shared V to zero
    for (int d = tid; d < head_dim; d += ATTN_BLOCK_SIZE) {
        smem_v[d] = 0.0f;
    }
    __syncthreads();

    float local_exp_sum = 0.0f;
    // Use a per-thread local buffer for V accumulation to avoid atomics
    // We'll reduce at the end.
    // For memory efficiency, accumulate in registers if head_dim is small enough.
    // head_dim=128 for Mixtral: 128 floats = 512 bytes per thread — too much for registers.
    // Instead, use a two-pass approach with atomicAdd to shared memory.
    for (int pos = tid; pos < seq_len; pos += ATTN_BLOCK_SIZE) {
        float score = 0.0f;
        const half* k_pos = k_head_cache + pos * head_dim;
        for (int d = 0; d < head_dim; d++) {
            score += __half2float(q_head[d]) * __half2float(k_pos[d]);
        }
        score = expf(score * scale - global_max);
        local_exp_sum += score;

        // Weighted V accumulation
        const half* v_pos = v_head_cache + pos * head_dim;
        for (int d = 0; d < head_dim; d++) {
            atomicAdd(&smem_v[d], score * __half2float(v_pos[d]));
        }
    }

    // Reduce exp_sum across threads
    __shared__ float smem_exp[ATTN_BLOCK_SIZE];
    smem_exp[tid] = local_exp_sum;
    __syncthreads();
    for (int s = ATTN_BLOCK_SIZE / 2; s > 0; s >>= 1) {
        if (tid < s) smem_exp[tid] += smem_exp[tid + s];
        __syncthreads();
    }
    float total_exp = smem_exp[0];

    // Normalize V output and write to global memory
    float inv_sum = (total_exp > 0.0f) ? (1.0f / total_exp) : 0.0f;
    for (int d = tid; d < head_dim; d += ATTN_BLOCK_SIZE) {
        output[head * head_dim + d] = __float2half(smem_v[d] * inv_sum);
    }
}

// ─── Launchers ───────────────────────────────────────────────────────────

extern "C" {

int vib3_launch_rope_apply(
    void* q, void* k,
    int head_dim, int num_q_heads, int num_kv_heads,
    int position, float rope_base, void* stream
) {
    cudaStream_t s = (cudaStream_t)stream;

    // Apply RoPE to Q heads
    int q_pairs = num_q_heads * (head_dim / 2);
    int q_blocks = (q_pairs + 255) / 256;
    vib3_rope_kernel<<<q_blocks, 256, 0, s>>>(
        (half*)q, head_dim, num_q_heads, position, rope_base
    );

    // Apply RoPE to K heads
    int k_pairs = num_kv_heads * (head_dim / 2);
    int k_blocks = (k_pairs + 255) / 256;
    vib3_rope_kernel<<<k_blocks, 256, 0, s>>>(
        (half*)k, head_dim, num_kv_heads, position, rope_base
    );

    return (int)cudaGetLastError();
}

int vib3_launch_kv_cache_append(
    void* k_cache, void* v_cache,
    const void* new_k, const void* new_v,
    int max_seq_len, int head_dim, int num_kv_heads,
    int position, void* stream
) {
    cudaStream_t s = (cudaStream_t)stream;
    int total = num_kv_heads * head_dim;
    int blocks = (total + 255) / 256;
    vib3_kv_append_kernel<<<blocks, 256, 0, s>>>(
        (half*)k_cache, (half*)v_cache,
        (const half*)new_k, (const half*)new_v,
        max_seq_len, head_dim, num_kv_heads, position
    );
    return (int)cudaGetLastError();
}

int vib3_launch_decode_attention(
    const void* q, const void* k_cache, const void* v_cache,
    void* output,
    int head_dim, int num_heads, int num_kv_heads,
    int seq_len, int max_seq_len, float scale,
    void* stream
) {
    cudaStream_t s = (cudaStream_t)stream;
    // One block per Q head, shared memory for V accumulation
    int smem_bytes = head_dim * sizeof(float);
    vib3_decode_attn_kernel<<<num_heads, ATTN_BLOCK_SIZE, smem_bytes, s>>>(
        (const half*)q, (const half*)k_cache, (const half*)v_cache,
        (half*)output,
        head_dim, num_heads, num_kv_heads,
        seq_len, max_seq_len, scale
    );
    return (int)cudaGetLastError();
}

} // extern "C"
