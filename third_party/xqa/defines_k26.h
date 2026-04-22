// SPDX-License-Identifier: Apache-2.0
// vib3 MLA K26 variant — defines overlay.
//
// Include this header *before* defines.h when compiling the k26 kernel
// (or pre-define VIB3_MLA_K26 on the nvcc command line).

#pragma once

#ifndef VIB3_MLA_K26
#define VIB3_MLA_K26 1
#endif

// K2.6 has 64 attention heads per KV head. DeepSeek-style MLA still uses
// headElems = 576 (512 compressed + 64 RoPE) for K and 512 for V.
#define HEAD_GRP_SIZE 64
#define HEAD_ELEMS 576

// FP16 inputs and cache. (xqa's default IS_MLA gate would force FP8 e4m3.)
#define INPUT_FP16 1
#define CACHE_ELEM_ENUM 0   // 0 == INPUT_ELEM (half)

// Paged KV with a single page-per-seq equal to seq_len is the vib3 layout;
// xqa accepts that via makeTensorMapForPagedKVCache when we supply
// tokensPerPage = min(seq_len, 128).
#ifndef TOKENS_PER_PAGE
#define TOKENS_PER_PAGE 64
#endif

// Non-MLA xqa gates require these to stay unset; we're in the IS_MLA path so
// nothing relies on INPUT_FP16 for dispatch — it's only the dtype alias.
#define SLIDING_WINDOW 0
#define SPEC_DEC 0
#define BEAM_WIDTH 1
#define USE_INPUT_KV 0
#define LOW_PREC_OUTPUT 0
