#include "models/qwen3_5/execution/ffn.h"

#include "core/layout.h"
#include "ninfer/ops/linear.h"
#include "ninfer/ops/linear_add.h"
#include "ninfer/ops/linear_swiglu.h"
#include "ninfer/ops/residual_add.h"
#include "ninfer/ops/separate_projection.h"
#include "ninfer/ops/silu_mul.h"

#include <array>
#include <stdexcept>

namespace ninfer::models::qwen3_5::execution {

std::size_t ffn_workspace_bytes(const FfnParameters& parameters, std::int32_t first,
                                std::int32_t last, bool mtp) {
    if (first <= 0 || last < first) { throw std::invalid_argument("FFN: invalid column interval"); }
    if (const auto* moe = std::get_if<ops::SparseMoeWeights>(&parameters)) {
        return ops::sparse_moe_workspace_capacity_bytes(moe->routed_gate_up.qtype,
                                                        moe->routed_down.qtype, first, last);
    }
    const auto& p    = std::get<DenseParameters>(parameters);
    const auto& down = p.down.weight;
    WorkspaceLayoutBuilder layout;
    if (const auto* separate = std::get_if<ops::SeparateProjectionWeights>(&p.gate_up)) {
        // Both trellises write one materialized [gate; up] plane, then SwiGLU, then down.
        const auto rows = separate->parts.front().n * 2;
        (void)layout.alloc(DType::BF16, {rows, last});
        {
            auto scope = layout.scope();
            (void)layout.alloc_bytes(
                ops::separate_projection_workspace_capacity_bytes(separate->parts, first, last));
        }
        (void)layout.alloc(DType::BF16, {rows / 2, last});
        (void)layout.alloc_bytes(ops::linear_add_workspace_capacity_bytes(
            down.qtype, down.n, down.k, p.down.policy, first, last));
        return layout.peak_bytes(1);
    }
    const auto& gate_up = std::get<LinearParameters>(p.gate_up);
    const auto& gu      = gate_up.weight;
    if (mtp) {
        (void)layout.alloc(DType::BF16, {gu.n, last});
        {
            auto scope = layout.scope();
            (void)layout.alloc_bytes(ops::linear_workspace_capacity_bytes(
                gu.qtype, gu.n, gu.k, gate_up.policy, first, last));
        }
        (void)layout.alloc(DType::BF16, {gu.n / 2, last});
        (void)layout.alloc(DType::BF16, {down.n, last});
        (void)layout.alloc_bytes(ops::linear_workspace_capacity_bytes(down.qtype, down.n, down.k,
                                                                      p.down.policy, first, last));
    } else {
        (void)layout.alloc(DType::BF16, {gu.n / 2, last});
        {
            auto scope = layout.scope();
            (void)layout.alloc_bytes(ops::linear_swiglu_workspace_capacity_bytes(
                gu.qtype, gu.n, gu.k, gate_up.policy, first, last));
        }
        {
            auto scope = layout.scope();
            (void)layout.alloc_bytes(ops::linear_add_workspace_capacity_bytes(
                down.qtype, down.n, down.k, p.down.policy, first, last));
        }
    }
    return layout.peak_bytes(1);
}

void ffn(const Tensor& hidden, const FfnParameters& parameters, Tensor& residual,
         const ops::SparseMoeHints& hints, WorkspaceArena& workspace, cudaStream_t stream,
         bool mtp) {
    auto scope         = workspace.scope();
    const auto columns = hidden.ne[1];
    if (const auto* moe = std::get_if<ops::SparseMoeWeights>(&parameters)) {
        const auto storage =
            workspace.alloc_bytes(ffn_workspace_bytes(parameters, columns, columns));
        WorkspaceArena scratch(storage);
        ops::sparse_moe(hidden, *moe, ops::SparseMoeEpilogue::AddResidual, residual, hints, scratch,
                        stream);
        return;
    }
    const auto& p    = std::get<DenseParameters>(parameters);
    const auto& down = p.down.weight;
    if (const auto* separate = std::get_if<ops::SeparateProjectionWeights>(&p.gate_up)) {
        const auto rows = separate->parts.front().n * 2;
        Tensor gate_up  = workspace.alloc(DType::BF16, {rows, columns});
        {
            auto call = workspace.scope();
            std::array outputs{gate_up};
            ops::separate_projection(hidden, separate->parts, outputs, workspace, stream);
        }
        Tensor activation = workspace.alloc(DType::BF16, {rows / 2, columns});
        ops::silu_mul(gate_up.slice(0, 0, rows / 2), gate_up.slice(0, rows / 2, rows / 2),
                      activation, stream);
        ops::linear_add(activation, down, residual, p.down.policy, workspace, stream);
        return;
    }
    const auto& gate_up_parameters = std::get<LinearParameters>(p.gate_up);
    const auto& gu                 = gate_up_parameters.weight;
    if (mtp) {
        Tensor gate_up = workspace.alloc(DType::BF16, {gu.n, columns});
        {
            auto call = workspace.scope();
            ops::linear(hidden, gu, gate_up, gate_up_parameters.policy, workspace, stream);
        }
        Tensor activation = workspace.alloc(DType::BF16, {gu.n / 2, columns});
        ops::silu_mul(gate_up.slice(0, 0, gu.n / 2), gate_up.slice(0, gu.n / 2, gu.n / 2),
                      activation, stream);
        Tensor delta = workspace.alloc(DType::BF16, {down.n, columns});
        ops::linear(activation, down, delta, p.down.policy, workspace, stream);
        ops::residual_add(delta, residual, stream);
        return;
    }
    Tensor activation = workspace.alloc(DType::BF16, {gu.n / 2, columns});
    {
        auto call = workspace.scope();
        ops::linear_swiglu(hidden, gu, activation, gate_up_parameters.policy, workspace, stream);
    }
    ops::linear_add(activation, down, residual, p.down.policy, workspace, stream);
}

} // namespace ninfer::models::qwen3_5::execution
