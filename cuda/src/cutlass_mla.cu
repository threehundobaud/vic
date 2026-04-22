// vib3 CUTLASS MLA decode wrapper (Blackwell sm_100 / sm_120a).
//
// Replaces the 9-kernel MLA decode pipeline with one persistent fused kernel
// from CUTLASS's examples/77_blackwell_fmha. Adapted from the SGLang/vLLM
// wrapper (sm100_cutlass_mla_kernel.cu) but with torch::Tensor stripped out
// — arguments are raw device pointers + strides so it can be called from
// vib3's Rust runtime via a plain extern "C" entry point.
//
// Expected input layout:
//   q_nope:  [num_heads, D_latent = 512]  FP16,  absorbed Q (after mla_q_absorb_rope)
//   q_pe:    [num_heads, D_rope  =  64]  FP16,  RoPE'd Q
//   kv_c:    [seq_len,   D_latent = 512]  FP16,  compressed latent KV
//   k_pe:    [seq_len,   D_rope  =  64]  FP16,  RoPE'd K
//   seq_lens:[1]                         int32
// Output:
//   out:     [num_heads, D_latent = 512]  FP32, attention in latent space.
//            Caller applies V-up-projection (kv_b_proj V-partition) + o_proj.

#if defined(__CUDACC__) && CUDA_VERSION >= 12040

#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include "cutlass/cutlass.h"
#include "cutlass/kernel_hardware_info.h"

#include <cute/tensor.hpp>

#include "device/sm100_mla.hpp"
#include "kernel/sm100_mla_tile_scheduler.hpp"

using namespace cute;
using namespace cutlass::fmha::kernel;

// Forward declare extern "C" entry point.
extern "C" int vib3_launch_cutlass_mla_decode(
    const void* q_nope,            // [H, 512] FP16
    const void* q_pe,              // [H, 64] FP16
    const void* kv_c,              // [seq_len, 512] FP16 — contiguous single-sequence cache
    const void* k_pe,              // [seq_len, 64] FP16
    const int* seq_lens_device,    // [1] int32
    const int* page_table_device,  // [1, pages_per_seq] int32 (page_size = seq_len, 1 page)
    void* out,                     // [H, 512] FP32
    void* lse,                     // [H] FP32 or nullptr (log-sum-exp; we don't need it)
    void* workspace,               // scratch, size from vib3_cutlass_mla_workspace_size
    int num_heads,
    int seq_len,
    int page_count_total,
    int page_size,
    float sm_scale,
    int num_kv_splits,
    int sm_count,
    void* stream
);

template <bool v>
struct IsPersistent {
    static const bool value = v;
};

template <typename T, typename TOut, bool IsPaged128, typename PersistenceOption = IsPersistent<true>>
struct MlaSm100 {
    using Element = T;
    using ElementAcc = float;
    using ElementOut = TOut;

    using TileShape = Shape<_128, _128, Shape<_512, _64>>;
    using TileShapeH = cute::tuple_element_t<0, TileShape>;
    using TileShapeD = cute::tuple_element_t<2, TileShape>;

    using ProblemShape = cute::tuple<TileShapeH, int, TileShapeD, int>;

    using StrideQ = cute::tuple<int64_t, _1, int64_t>;  // H D B
    using StrideK = cute::tuple<int64_t, _1, int64_t>;  // K D B
    using StrideO = StrideK;                            // H D B
    using StrideLSE = cute::tuple<_1, int>;             // H B

    using TileScheduler = std::conditional_t<
        PersistenceOption::value,
        Sm100MlaPersistentTileScheduler,
        Sm100MlaIndividualTileScheduler>;

    using FmhaKernel = cutlass::fmha::kernel::Sm100FmhaMlaKernelTmaWarpspecialized<
        TileShape,
        Element,
        ElementAcc,
        ElementOut,
        ElementAcc,
        TileScheduler,
        /*kIsCpAsync=*/!IsPaged128>;
    using Fmha = cutlass::fmha::device::MLA<FmhaKernel>;
};

// Build kernel arguments from raw pointers + strides.
template <typename T>
typename T::Fmha::Arguments build_args(
    const void* q_nope, const void* q_pe,
    const void* kv_c, const void* k_pe,
    const int* seq_lens, const int* page_table,
    void* out, void* lse,
    int num_heads, int max_seq_len,
    int page_count_total, int page_size,
    float sm_scale, int num_kv_splits, int sm_count
) {
    cutlass::KernelHardwareInfo hw_info;
    hw_info.device_id = 0;
    hw_info.sm_count = sm_count > 0
        ? sm_count
        : cutlass::KernelHardwareInfo::query_device_multiprocessor_count(hw_info.device_id);

    constexpr int batches = 1;  // vib3 decodes one token at a time
    using TileShapeH = typename T::TileShapeH;
    using TileShapeD = typename T::TileShapeD;
    auto problem_shape = cute::make_tuple(TileShapeH{}, max_seq_len, TileShapeD{}, batches);

    auto [H, K, D, B] = problem_shape;
    auto [D_latent, D_rope] = D;

    using StrideQ = typename T::StrideQ;
    using StrideK = typename T::StrideK;
    using StrideO = typename T::StrideO;
    using StrideLSE = typename T::StrideLSE;

    // Q: [H=128, D, B=1], rows are heads, head_dim stride=1, batch stride=H*D
    StrideQ stride_Q_nope = cute::make_tuple(
        static_cast<int64_t>(D_latent), _1{}, static_cast<int64_t>(num_heads * D_latent));
    StrideQ stride_Q_pe = cute::make_tuple(
        static_cast<int64_t>(D_rope), _1{}, static_cast<int64_t>(num_heads * D_rope));

    // KV cache: per-page layout [page_size=128, D_latent + D_rope] concatenated.
    // stride_C is (row_stride, elem_stride, page_stride).
    StrideK stride_C_latent = cute::make_tuple(
        static_cast<int64_t>(D_latent + D_rope), _1{},
        static_cast<int64_t>(page_size * (D_latent + D_rope)));
    StrideK stride_K_rope = cute::make_tuple(
        static_cast<int64_t>(D_latent + D_rope), _1{},
        static_cast<int64_t>(page_size * (D_latent + D_rope)));

    StrideLSE stride_PT = cute::make_stride(_1{}, static_cast<int>(max_seq_len / page_size));
    StrideLSE stride_LSE = cute::make_tuple(_1{}, static_cast<int>(0 + H));
    StrideO stride_O = cute::make_tuple(
        static_cast<int64_t>(D_latent), _1{}, static_cast<int64_t>(num_heads * D_latent));

    using Element = typename T::Element;
    using ElementOut = typename T::ElementOut;
    using ElementAcc = typename T::ElementAcc;

    typename T::Fmha::Arguments args{
        problem_shape,
        {  // mainloop
            sm_scale,
            reinterpret_cast<Element*>(const_cast<void*>(q_nope)), stride_Q_nope,
            reinterpret_cast<Element*>(const_cast<void*>(q_pe)),   stride_Q_pe,
            reinterpret_cast<Element*>(const_cast<void*>(kv_c)),   stride_C_latent,
            reinterpret_cast<Element*>(const_cast<void*>(k_pe)),   stride_K_rope,
            const_cast<int*>(seq_lens),
            const_cast<int*>(page_table),
            stride_PT,
            page_count_total,
            page_size,
        },
        {  // epilogue
            reinterpret_cast<ElementOut*>(out), stride_O,
            reinterpret_cast<ElementAcc*>(lse),  // may be nullptr
            stride_LSE,
        },
        hw_info,
        num_kv_splits,
        nullptr,  // is_var_split_kv
    };

    T::Fmha::set_split_kv(args);
    return args;
}

template <typename Element, typename ElementOut, bool IsPaged128, typename PersistenceOption>
static int run_mla(
    const void* q_nope, const void* q_pe,
    const void* kv_c, const void* k_pe,
    const int* seq_lens, const int* page_table,
    void* out, void* lse, void* workspace,
    int num_heads, int max_seq_len,
    int page_count_total, int page_size,
    float sm_scale, int num_kv_splits, int sm_count,
    cudaStream_t stream
) {
    using MlaType = MlaSm100<Element, ElementOut, IsPaged128, PersistenceOption>;
    typename MlaType::Fmha fmha;
    auto args = build_args<MlaType>(
        q_nope, q_pe, kv_c, k_pe, seq_lens, page_table,
        out, lse,
        num_heads, max_seq_len, page_count_total, page_size,
        sm_scale, num_kv_splits, sm_count);

    if (fmha.can_implement(args) != cutlass::Status::kSuccess) return -10;
    if (fmha.initialize(args, workspace, stream) != cutlass::Status::kSuccess) return -11;
    if (fmha.run(args, workspace, stream) != cutlass::Status::kSuccess) return -12;
    return 0;
}

extern "C" int vib3_launch_cutlass_mla_decode(
    const void* q_nope, const void* q_pe,
    const void* kv_c, const void* k_pe,
    const int* seq_lens_device, const int* page_table_device,
    void* out, void* lse,
    void* workspace,
    int num_heads, int seq_len,
    int page_count_total, int page_size,
    float sm_scale, int num_kv_splits, int sm_count,
    void* stream
) {
    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);
    const bool is_paged_128 = (page_size == 128);
    const bool manual_split = (num_kv_splits > 1);

    // FP16 I/O for K2.6 (MLA projections are FP16).
    if (is_paged_128) {
        if (!manual_split) {
            return run_mla<cutlass::half_t, cutlass::half_t, true, IsPersistent<true>>(
                q_nope, q_pe, kv_c, k_pe, seq_lens_device, page_table_device,
                out, lse, workspace,
                num_heads, seq_len, page_count_total, page_size,
                sm_scale, num_kv_splits, sm_count, s);
        } else {
            return run_mla<cutlass::half_t, cutlass::half_t, true, IsPersistent<false>>(
                q_nope, q_pe, kv_c, k_pe, seq_lens_device, page_table_device,
                out, lse, workspace,
                num_heads, seq_len, page_count_total, page_size,
                sm_scale, num_kv_splits, sm_count, s);
        }
    } else {
        if (!manual_split) {
            return run_mla<cutlass::half_t, cutlass::half_t, false, IsPersistent<true>>(
                q_nope, q_pe, kv_c, k_pe, seq_lens_device, page_table_device,
                out, lse, workspace,
                num_heads, seq_len, page_count_total, page_size,
                sm_scale, num_kv_splits, sm_count, s);
        } else {
            return run_mla<cutlass::half_t, cutlass::half_t, false, IsPersistent<false>>(
                q_nope, q_pe, kv_c, k_pe, seq_lens_device, page_table_device,
                out, lse, workspace,
                num_heads, seq_len, page_count_total, page_size,
                sm_scale, num_kv_splits, sm_count, s);
        }
    }
}

extern "C" size_t vib3_cutlass_mla_workspace_size(
    int num_heads, int max_seq_len, int num_batches, int sm_count, int num_kv_splits
) {
    using MlaType = MlaSm100<cutlass::half_t, cutlass::half_t, /*IsPaged128=*/true>;
    typename MlaType::Fmha::Arguments args{};
    using TileShapeH = typename MlaType::TileShapeH;
    using TileShapeD = typename MlaType::TileShapeD;
    args.problem_shape = cute::make_tuple(
        TileShapeH{}, static_cast<int>(max_seq_len), TileShapeD{}, static_cast<int>(num_batches));
    args.hw_info.sm_count = sm_count > 0
        ? sm_count
        : cutlass::KernelHardwareInfo::query_device_multiprocessor_count(0);
    args.split_kv = static_cast<int>(num_kv_splits);
    MlaType::Fmha::set_split_kv(args);
    return MlaType::Fmha::get_workspace_size(args);
}

#else  // CUDA_VERSION < 12040 — stub out

extern "C" int vib3_launch_cutlass_mla_decode(
    const void*, const void*, const void*, const void*,
    const int*, const int*,
    void*, void*, void*,
    int, int, int, int,
    float, int, int,
    void*
) {
    return -100; // unsupported
}

extern "C" size_t vib3_cutlass_mla_workspace_size(int, int, int, int, int) {
    return 0;
}

#endif
