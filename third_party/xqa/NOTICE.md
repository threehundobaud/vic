xqa sources vendored from flashinfer (NVIDIA TensorRT-LLM) — Apache-2.0.

Upstream: https://github.com/flashinfer-ai/flashinfer (csrc/xqa/)

Unmodified files carry their original SPDX headers. Files suffixed `_k26`
are vib3 derivations covered under this NOTICE:

  - mla_sm120_k26.cu     — MLA decode kernel, derived from mla_sm120.cu.
                            Changes: HEAD_GRP_SIZE 128->64, FP8 e4m3 inputs
                            and KV cache replaced with FP16, MMA atom from
                            m16n8k32.e4m3 to m16n8k16.f16, X softmax buffer
                            packed as FP16 instead of quantized FP8, V-buffer
                            split halved (partElemsV 128->64) to fit sm_120a
                            smem, XV double-buffering disabled.

  - xqa_wrapper_k26.cu    — extern "C" entry point for vib3's runtime.

  - defines_k26.h         — configuration overrides for IS_MLA_K26 variant.

Original Apache-2.0 license terms apply. See upstream for full text.
