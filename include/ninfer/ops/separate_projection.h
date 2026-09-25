#pragma once

// ninfer::ops - one input projected through separately stored parents.

#include "core/arena.h"
#include "core/tensor.h"
#include "core/weight.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <span>

namespace ninfer::ops {

/**
 * Returns the transient capacity SeparateProjection requires for every T in the inclusive
 * `[min_tokens,max_tokens]` interval over `parts`. Unregistered parents or intervals throw.
 */
[[nodiscard]] std::size_t separate_projection_workspace_capacity_bytes(std::span<const Weight> parts,
                                                                       std::int32_t min_tokens,
                                                                       std::int32_t max_tokens);

/**
 * Op: separate_projection
 *
 * Math / indexing:
 *   The logical output rows are the concatenation of `outputs` in order. `parts[i]` is a complete
 *   `[N_i,K]` parent producing the next N_i logical rows: out = parts[i] * x[:,t]. A part never
 *   straddles two outputs, and the parts cover every logical row exactly once.
 *
 * Logical shapes:
 *   x is contiguous BF16 `[K,T]` with T positive. Each output is BF16 `[R_j,T]` with unit element
 *   stride and any even row stride, so a row slice of a wider tensor is accepted. Registered parents
 *   are EXL3 trellises (Exl3Tile, K1..K8 MUL1) with N_i and K multiples of 128; an EXL3 parent
 *   carries its own input rotation, so projections stored apart run one GEMM per parent.
 *
 * Numeric:
 *   Each part follows Linear's contract for its parent: exact decode of the trellis and both
 *   Hadamard rotations, FP16 activation and FP32 accumulation privately, one BF16 store.
 *
 * Effects:
 *   Writes every logical output row; x, parents, outputs and workspace must not overlap.
 *
 * Workspace:
 *   Caller-owned call-scoped storage sized by separate_projection_workspace_capacity_bytes().
 */
void separate_projection(const Tensor& x, std::span<const Weight> parts, std::span<Tensor> outputs,
                         WorkspaceArena& workspace, cudaStream_t stream);

} // namespace ninfer::ops
