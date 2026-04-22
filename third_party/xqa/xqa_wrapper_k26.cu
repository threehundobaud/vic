// SPDX-License-Identifier: Apache-2.0
// vib3 FFI wrapper for the K26 MLA variant of xqa's mla_sm120 kernel.
//
// Bridges vib3's contiguous-per-seq MLA layout (separate q_nope / q_pe and
// kv_c / k_pe buffers) to xqa's expected concatenated 576-wide input +
// paged KV cache with a page-index list.
//
// Exposed symbol:
//   int vib3_launch_xqa_mla_decode_k26(
//       const void* q_nope,       // [H=64, 512] half
//       const void* q_pe,         // [H=64,  64] half
//       const void* kv_c,         // [seq_len, 512] half  contiguous
//       const void* k_pe,         // [seq_len,  64] half  contiguous
//       const int*  seq_lens_d,   // [1] int32
//       void*       out,          // [H=64, 512] half
//       void*       workspace,    // scratch + concat Q/KV + page list
//       int num_heads, int seq_len, float sm_scale,
//       int sm_count, void* stream);
//
// The workspace layout is computed in vib3_xqa_mla_workspace_size_k26 so
// the Rust caller can allocate a single buffer and we slice it internally.

#define VIB3_MLA_K26 1
#include "defines_k26.h"
#include "defines.h"
#include "mha.h"

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <cstring>
#include <stdexcept>

// Host-side declaration of launchMLA as actually defined in mla_sm120_k26.cu.
// Note: upstream's mha.h declares launchMLA without kv_stride args, but the
// *definition* in mla_sm120.cu takes them — we match the definition to keep
// C++ name mangling consistent at link time.
extern void launchMLA(
    cudaDeviceProp const& prop, uint32_t inputSeqLen, float qScale,
    float const* qScalePtr, OutputHead* output, InputHead const* q,
    GMemCacheHead* kCacheVLLM, GMemCacheHead* vCacheVLLM,
    KVCachePageIndex const* kvCachePageList, uint32_t maxSeqLen,
    uint32_t const* seqLen, uint32_t batchSize, float kvCacheScale,
    float const* kvScalePtr, uint32_t* semaphores, void* scratch,
    bool enable_pdl, uint64_t kv_stride_page, uint64_t kv_stride_token,
    uint64_t kv_stride_head, cudaStream_t stream);

namespace vib3_k26_wrapper {

// Kernel: concatenate [q_nope(512) | q_pe(64)] along the head dim into Q[H, 576].
__global__ void pack_q_kernel(half const* __restrict__ q_nope,
                              half const* __restrict__ q_pe,
                              half* __restrict__ q_out,
                              int num_heads) {
  int const h = blockIdx.x;
  int const tid = threadIdx.x;
  if (h >= num_heads) return;
  half const* nope_row = q_nope + h * 512;
  half const* pe_row = q_pe + h * 64;
  half* out_row = q_out + h * 576;
  // 512 / 128 = 4 vectors of 128 halves per warp-stride; simple element loop is fine.
  for (int i = tid; i < 512; i += blockDim.x) {
    out_row[i] = nope_row[i];
  }
  for (int i = tid; i < 64; i += blockDim.x) {
    out_row[512 + i] = pe_row[i];
  }
}

// Kernel: concatenate [kv_c(512) | k_pe(64)] into a page-laid-out KV cache.
// Target layout:
//   kv_cache[page_id][head = 0][token_in_page][dim]  (VLLM-style)
// With nbKHeads = 1 and TOKENS_PER_PAGE pages, we write token (page_id * P + t)
// to kv_cache[page_id][0][t][0..575].
__global__ void pack_kv_kernel(half const* __restrict__ kv_c,
                               half const* __restrict__ k_pe,
                               half* __restrict__ kv_out,
                               int seq_len,
                               int tokens_per_page) {
  int const t = blockIdx.x;
  int const tid = threadIdx.x;
  if (t >= seq_len) return;
  int const page_id = t / tokens_per_page;
  int const token_in_page = t % tokens_per_page;

  half const* kvc_row = kv_c + t * 512;
  half const* kpe_row = k_pe + t * 64;
  // Page stride = tokens_per_page * 576 halves; within a page, head 0 is the
  // entire page (nbKHeads = 1).
  half* out_row = kv_out + (page_id * tokens_per_page + token_in_page) * 576;

  for (int i = tid; i < 512; i += blockDim.x) {
    out_row[i] = kvc_row[i];
  }
  for (int i = tid; i < 64; i += blockDim.x) {
    out_row[512 + i] = kpe_row[i];
  }
}

// Kernel: fill a contiguous page-index list [0, 1, ..., n_pages - 1].
__global__ void fill_page_list_kernel(int32_t* __restrict__ page_list,
                                      int n_pages) {
  int const i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n_pages) {
    page_list[i] = i;
  }
}

struct WorkspaceLayout {
  size_t q_packed_off;       // H * 576 halves
  size_t kv_packed_off;      // max_pages * TOKENS_PER_PAGE * 576 halves
  size_t page_list_off;      // max_pages int32
  size_t semaphores_off;     // 1 uint32 (per input token); we decode 1 at a time
  size_t kernel_scratch_off; // xqa's scratch (CgaXBuffer + PartialResult)
  size_t total;
};

static WorkspaceLayout compute_layout(int num_heads, int seq_len) {
  WorkspaceLayout ws{};
  auto const align = [](size_t x, size_t a) { return (x + a - 1) & ~(a - 1); };
  size_t cur = 0;
  ws.q_packed_off = cur;
  cur = align(cur + size_t(num_heads) * 576 * sizeof(half), 256);
  int const max_pages = (seq_len + TOKENS_PER_PAGE - 1) / TOKENS_PER_PAGE;
  int const padded_tokens = max_pages * TOKENS_PER_PAGE;
  ws.kv_packed_off = cur;
  cur = align(cur + size_t(padded_tokens) * 576 * sizeof(half), 256);
  ws.page_list_off = cur;
  cur = align(cur + size_t(max_pages) * sizeof(int32_t), 256);
  ws.semaphores_off = cur;
  cur = align(cur + sizeof(uint32_t), 256);
  ws.kernel_scratch_off = cur;
  // xqa's scratch holds CgaXBuffer array and PartialResult array. Size is
  // conservative: nbCgas * sizeof(CgaXBuffer * nbProducerCtasPerCga) +
  // nbCgas * sizeof(PartialResult). For 1 seq, nbSubSeq = 1, so nbCgas = 1.
  // Upper-bound each to 64 KB → 128 KB total.
  cur = align(cur + 128 * 1024, 256);
  ws.total = cur;
  return ws;
}

}  // namespace vib3_k26_wrapper

extern "C" size_t vib3_xqa_mla_workspace_size_k26(int num_heads, int seq_len) {
  return vib3_k26_wrapper::compute_layout(num_heads, seq_len).total;
}

extern "C" int vib3_launch_xqa_mla_decode_k26(
    void const* q_nope, void const* q_pe,
    void const* kv_c, void const* k_pe,
    int const* seq_lens_d,
    void* out,
    void* workspace,
    int num_heads, int seq_len,
    float sm_scale, int /*sm_count*/,
    void* stream) {
  if (num_heads != 64) {
    return -1;  // only K2.6's 64-head shape is supported in this port.
  }
  if (seq_len <= 0) {
    return -2;
  }
  cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);

  auto const layout = vib3_k26_wrapper::compute_layout(num_heads, seq_len);
  auto* ws = static_cast<uint8_t*>(workspace);
  half* q_packed = reinterpret_cast<half*>(ws + layout.q_packed_off);
  half* kv_packed = reinterpret_cast<half*>(ws + layout.kv_packed_off);
  int32_t* page_list = reinterpret_cast<int32_t*>(ws + layout.page_list_off);
  uint32_t* semaphores = reinterpret_cast<uint32_t*>(ws + layout.semaphores_off);
  void* kernel_scratch = ws + layout.kernel_scratch_off;

  int const max_pages = (seq_len + TOKENS_PER_PAGE - 1) / TOKENS_PER_PAGE;
  int const padded_seq_len = max_pages * TOKENS_PER_PAGE;

  // Zero the kernel scratch + semaphores + tail of padded KV (so causal-mask
  // reads beyond seq_len see clean data and don't contribute spuriously).
  cudaError_t err = cudaMemsetAsync(semaphores, 0, sizeof(uint32_t), s);
  if (err != cudaSuccess) return -10;
  err = cudaMemsetAsync(kernel_scratch, 0, 128 * 1024, s);
  if (err != cudaSuccess) return -11;

  // Stage 1: pack Q (64 heads, 576 dim) and KV cache (padded to page grid).
  vib3_k26_wrapper::pack_q_kernel<<<num_heads, 128, 0, s>>>(
      reinterpret_cast<half const*>(q_nope),
      reinterpret_cast<half const*>(q_pe),
      q_packed, num_heads);
  vib3_k26_wrapper::pack_kv_kernel<<<seq_len, 128, 0, s>>>(
      reinterpret_cast<half const*>(kv_c),
      reinterpret_cast<half const*>(k_pe),
      kv_packed, seq_len, TOKENS_PER_PAGE);
  {
    int const blocks = (max_pages + 31) / 32;
    vib3_k26_wrapper::fill_page_list_kernel<<<blocks, 32, 0, s>>>(
        page_list, max_pages);
  }
  err = cudaGetLastError();
  if (err != cudaSuccess) return -20;

  // Zero out the tail of padded KV beyond seq_len so causal-masked tokens
  // contribute 0 after softmax mask (safety: mask should already zero these,
  // but cleaner to fill with zeros since our "real" seq_len is tracked via
  // seq_lens_d which the kernel reads to mask).
  if (padded_seq_len > seq_len) {
    size_t const pad_tokens = size_t(padded_seq_len - seq_len);
    half* pad_start = kv_packed + size_t(seq_len) * 576;
    cudaMemsetAsync(pad_start, 0, pad_tokens * 576 * sizeof(half), s);
  }

  // Stage 2: launch the MLA kernel.
  static cudaDeviceProp s_prop;
  static bool s_prop_inited = false;
  if (!s_prop_inited) {
    cudaGetDeviceProperties(&s_prop, 0);
    s_prop_inited = true;
  }

  uint32_t batch_size = 1;
  uint32_t input_seq_len = 1;            // decode: 1 Q token per step
  uint32_t max_seq_len = uint32_t(padded_seq_len);

  // qScale applied to QK; xqa multiplies by log2e internally. Caller supplies
  // 1/sqrt(head_dim) — we pass it through as qScale.
  float q_scale = sm_scale;
  float kv_cache_scale = 1.f;  // FP16 → no explicit scaling

  // KV cache strides (in elements, not bytes). Our packed KV layout is:
  //   kv_packed[page_id * tokens_per_page * 576 + token_in_page * 576 + dim].
  // xqa's TMA descriptor wants stride_page / stride_token / stride_head in
  // elements. With nbKHeads=1, head stride is 576 (one full row per head).
  uint64_t const stride_head  = 576u;
  uint64_t const stride_token = 576u;
  uint64_t const stride_page  = uint64_t(TOKENS_PER_PAGE) * 576u;

  try {
    launchMLA(
        s_prop, input_seq_len, q_scale, /*qScalePtr=*/nullptr,
        reinterpret_cast<OutputHead*>(out),
        reinterpret_cast<InputHead const*>(q_packed),
        reinterpret_cast<GMemCacheHead*>(kv_packed),   // K cache
        reinterpret_cast<GMemCacheHead*>(kv_packed),   // V cache (aliased — V reads first 512 cols)
        reinterpret_cast<KVCachePageIndex const*>(page_list),
        max_seq_len,
        reinterpret_cast<uint32_t const*>(seq_lens_d),
        batch_size, kv_cache_scale, /*kvScalePtr=*/nullptr,
        semaphores, kernel_scratch,
        /*enable_pdl=*/false, stride_page, stride_token, stride_head, s);
  } catch (std::exception const& e) {
    fprintf(stderr, "[vib3_xqa_k26] launchMLA threw: %s\n", e.what());
    return -30;
  } catch (...) {
    fprintf(stderr, "[vib3_xqa_k26] launchMLA threw unknown exception\n");
    return -31;
  }

  return int(cudaGetLastError());
}
