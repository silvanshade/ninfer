#pragma once

#include "core/tensor.h"

#include <cuda_runtime.h>

#include <cstdint>

namespace ninfer::ops {

/**
 * Op: fill_i32_positions
 *
 * Math / indexing:
 *   positions[i] = start + i, 0 <= i < T.
 *
 * Logical shapes:
 *   positions is a contiguous I32 vector [T].
 *
 * Numeric:
 *   T is positive, start is nonnegative, and start+T must not exceed INT32_MAX. Thus every
 *   emitted value is a nonnegative I32 position.
 *
 * Effects:
 *   Writes the full positions vector.
 *
 * Workspace:
 *   None. The Op has no other state side effect.
 */
void fill_i32_positions(Tensor& positions, std::int32_t start, cudaStream_t stream);

/**
 * Op: offset_i32_positions
 *
 * Math / indexing:
 *   destination[i] = source[i] + delta[0], 0 <= i < T.
 *
 * Logical shapes:
 *   source and destination are contiguous I32 vectors [T]; delta is an I32 scalar [1].
 *
 * Numeric:
 *   Callers provide values whose sums are representable by I32.
 *
 * Effects:
 *   Writes the full destination. Source and destination may alias; delta must not alias a written
 *   destination element.
 *
 * Workspace:
 *   None. The Op has no other state side effect.
 */
void offset_i32_positions(const Tensor& source, const Tensor& delta, Tensor& destination,
                          cudaStream_t stream);

/** Piecewise integer position transform. Native positions are unchanged; beyond the threshold,
 * round the scaled excess to nearest, with positive half ties rounded up. Requires finite
 * factor >= 1 and a nonnegative original_context. Logical KV and RNG positions are not scaled. */
__host__ __device__ inline std::int32_t
scale_rope_position(std::int32_t position, std::uint32_t original_context, float factor) {
    if (factor == 1.0F || position <= static_cast<std::int32_t>(original_context)) return position;
    return static_cast<std::int32_t>(original_context) +
           static_cast<std::int32_t>(
               static_cast<double>(position - static_cast<std::int32_t>(original_context)) /
                   factor +
               0.5);
}

/** Applies scale_rope_position in place to a nonempty contiguous I32 vector.
 * No workspace or other state effects. Invalid factor/threshold/shape throws invalid_argument. */
void scale_rope_positions(Tensor& positions, std::uint32_t original_context, float factor,
                          cudaStream_t stream);

/** Out-of-place form: destination[i] = scale_rope_position(source[i], ...), for a drafter that
 * attends at scaled positions while its cache, masks and slots keep the logical ones. Source and
 * destination are nonempty contiguous I32 vectors of equal length and may alias. A unit factor
 * still copies when they do not alias. Invalid factor/threshold/shape throws invalid_argument. */
void scale_rope_positions_into(const Tensor& source, Tensor& destination,
                               std::uint32_t original_context, float factor, cudaStream_t stream);

} // namespace ninfer::ops
