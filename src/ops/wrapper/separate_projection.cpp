#include "ninfer/ops/separate_projection.h"

#include "ops/linear/exl3/exl3_dispatch.h"

#include <algorithm>
#include <stdexcept>

namespace ninfer::ops {
namespace {

void require(bool condition, const char* message) {
    if (!condition) { throw std::invalid_argument(message); }
}

} // namespace

std::size_t separate_projection_workspace_capacity_bytes(std::span<const Weight> parts,
                                                         std::int32_t min_tokens,
                                                         std::int32_t max_tokens) {
    require(!parts.empty(), "separate_projection workspace: no parents");
    require(min_tokens > 0 && max_tokens >= min_tokens,
            "separate_projection workspace: invalid token interval");
    // Parts run one after another, each inside its own workspace scope.
    std::size_t bytes = 0;
    for (const auto& part : parts) {
        require(detail::is_exl3(part.qtype),
                "separate_projection workspace: parents must be EXL3 trellises");
        bytes = std::max(bytes,
                         detail::exl3_linear_workspace_capacity_bytes(part.n, part.k, max_tokens));
    }
    return bytes;
}

void separate_projection(const Tensor& x, std::span<const Weight> parts, std::span<Tensor> outputs,
                         WorkspaceArena& workspace, cudaStream_t stream) {
    require(!parts.empty() && !outputs.empty(), "separate_projection: no parents or outputs");
    std::size_t output = 0;
    std::int32_t row   = 0;
    for (const auto& part : parts) {
        require(detail::is_exl3(part.qtype), "separate_projection: parents must be EXL3 trellises");
        require(output < outputs.size() && row + part.n <= outputs[output].ne[0],
                "separate_projection: a parent straddles outputs or exceeds them");
        Tensor rows = outputs[output].slice(0, row, part.n);
        detail::exl3_linear(x, part, rows, nullptr, workspace, stream);
        row += part.n;
        if (row == outputs[output].ne[0]) {
            ++output;
            row = 0;
        }
    }
    require(output == outputs.size() && row == 0,
            "separate_projection: parents do not cover the outputs");
}

} // namespace ninfer::ops
