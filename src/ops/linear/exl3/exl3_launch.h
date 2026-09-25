#pragma once

#include <cuda_runtime.h>

#include <cstdint>

namespace ninfer::ops::detail {

// Row-major operands in the vendored kernel's terms: A is [m, k] FP16, C is [m, n] FP32. NInfer's
// column-major [K, T] activations and [N, T] outputs are the same bytes with m = T.
struct Exl3GemmOperands {
    std::int32_t bits = 0;
    const void* a     = nullptr; // FP16 [m, k]; overwritten in place by the input rotation
    const void* codes = nullptr; // exl3_tile_v1 code plane, [N/16][K/16] tiles
    const void* suh   = nullptr; // FP16 [k]
    const void* svh   = nullptr; // FP16 [n]
    float* c          = nullptr; // FP32 [m, n]
    std::int32_t m    = 0;
    std::int32_t k    = 0;
    std::int32_t n    = 0;
};

void exl3_gemm_launch(const Exl3GemmOperands& operands, cudaStream_t stream);

// x BF16 [count] -> a FP16 [count].
void exl3_bf16_to_fp16_launch(const void* x, void* a, std::int64_t count, cudaStream_t stream);

// out[t * out_stride + i] = bf16(c[t * n + i] (+ residual[t * residual_stride + i])), strides in
// elements, for i < n and t < tokens.
void exl3_store_bf16_launch(const float* c, void* out, std::int64_t out_stride,
                            const void* residual, std::int64_t residual_stride, std::int32_t n,
                            std::int32_t tokens, cudaStream_t stream);

// Decodes output rows [first_row, first_row + rows) of the trellis into BF16 [rows, K] in the
// original basis, both rotations and sign vectors applied. first_row and rows are multiples of 128.
struct Exl3ReconstructOperands {
    std::int32_t bits     = 0;
    const void* codes     = nullptr; // exl3_tile_v1 code plane of the whole parent
    const void* suh       = nullptr; // FP16 [K]
    const void* svh       = nullptr; // FP16 [N], whole parent
    void* weight          = nullptr; // BF16 [rows, K]
    std::int32_t k        = 0;
    std::int32_t first_row = 0;
    std::int32_t rows     = 0;
};

void exl3_reconstruct_launch(const Exl3ReconstructOperands& operands, cudaStream_t stream);

// out[t * out_stride + r] = bf16(sum_k weight[r, k] x[t, k] (+ residual[t * residual_stride + r]))
// for r < rows, t < tokens: a BF16 Tensor Core GEMM over a reconstructed slice, FP32 accumulation,
// one rounding. out and residual point at the slice's first row; strides are in elements.
struct Exl3DenseOperands {
    const void* x                = nullptr; // BF16 [tokens, K] contiguous
    const void* weight           = nullptr; // BF16 [rows, K]
    void* out                    = nullptr;
    const void* residual         = nullptr;
    std::int32_t k               = 0;
    std::int32_t rows            = 0; // multiple of 64
    std::int32_t tokens          = 0;
    std::int32_t out_stride      = 0;
    std::int32_t residual_stride = 0;
};

// Input row counts the dense GEMM is compiled for.
[[nodiscard]] bool exl3_dense_supported(std::int32_t input_rows) noexcept;

void exl3_dense_launch(const Exl3DenseOperands& operands, cudaStream_t stream);

} // namespace ninfer::ops::detail
