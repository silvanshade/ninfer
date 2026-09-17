#pragma once

#include "ninfer/ops/position.h"
#include <cstdint>

namespace ninfer::ops {

__global__ void fill_i32_positions_kernel(std::int32_t* positions, std::int32_t count,
                                          std::int32_t start) {
    const std::int32_t i = static_cast<std::int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
    if (i < count) { positions[i] = start + i; }
}

__global__ void offset_i32_positions_kernel(const std::int32_t* source, const std::int32_t* delta,
                                            std::int32_t* destination, std::int32_t count) {
    const std::int32_t i = static_cast<std::int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
    if (i < count) { destination[i] = source[i] + delta[0]; }
}

__global__ void scale_rope_positions_kernel(std::int32_t* positions, std::int32_t count,
                                           std::uint32_t original_context, float factor) {
    const auto i = static_cast<std::int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
    if (i < count) positions[i] = scale_rope_position(positions[i], original_context, factor);
}

} // namespace ninfer::ops
