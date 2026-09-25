#pragma once

#include "core/arena.h"
#include "core/tensor.h"
#include "core/weight.h"
#include "ninfer/ops/linear.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace ninfer::ops::detail {

[[nodiscard]] bool is_exl3(QType qtype) noexcept;

// Scratch for one EXL3 linear over `max_tokens` columns: the FP16 activation copy the kernel
// rotates in place, and the FP32 product.
[[nodiscard]] std::size_t exl3_linear_workspace_capacity_bytes(std::int32_t output_rows,
                                                               std::int32_t input_rows,
                                                               std::int32_t max_tokens);

// out[N, T] (row stride out.nb[1], so a row slice of a wider tensor is accepted) = W x, plus
// `residual` when given (same shape and strides rule as out).
void exl3_linear(const Tensor& x, const Weight& weight, Tensor& out, const Tensor* residual,
                 WorkspaceArena& workspace, cudaStream_t stream);

void exl3_dispatch(const Tensor& x, const Weight& weight, Tensor& out, LinearPolicy policy,
                   WorkspaceArena* workspace, cudaStream_t stream);

} // namespace ninfer::ops::detail
