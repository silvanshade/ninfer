#include "ops/linear/exl3/exl3_dispatch.h"

#include "core/dtype.h"
#include "ops/linear/exl3/exl3_launch.h"

#include <algorithm>
#include <stdexcept>

namespace ninfer::ops::detail {
namespace {

std::size_t align256(std::size_t bytes) { return (bytes + 255) & ~std::size_t{255}; }

void require(bool condition, const char* message) {
    if (!condition) { throw std::invalid_argument(message); }
}

// The GEMM reduces into an FP32 [T,N] plane before the BF16 store. Columns run in chunks so that
// plane stays near this bound: the 248320-row output head at a 2048-column prefill chunk would
// otherwise stage 2 GB, which the runtime reservation pays for at every context size. Each chunk
// rereads the weight, so the bound stays large enough that the extra reads are small next to the
// arithmetic of the columns they serve.
constexpr std::size_t kStagingBytes = std::size_t{128} << 20;

std::int32_t chunk_columns(std::int32_t output_rows, std::int32_t tokens) {
    const auto fit = kStagingBytes / (static_cast<std::size_t>(output_rows) * 4);
    // Whole 16-column tiles, and at least one of them, even for a row count beyond the bound.
    const auto columns = std::max<std::size_t>(16, fit / 16 * 16);
    return static_cast<std::int32_t>(std::min<std::size_t>(columns, tokens));
}

// From this many tokens on, the weight is decoded once per call into BF16 and multiplied by a dense
// Tensor Core GEMM; below it, the trellis kernel decodes in the matmul loop, which re-decodes every
// tile once per 16-token block. On the 27B shapes (RTX 5090) the dense route wins every projection
// from 256 tokens, reaching about 2.5x the trellis kernel at 2048; between 128 and 256 the winner
// depends on the shape, and below 128 the one-off decode dominates.
constexpr std::int32_t kDenseMinTokens = 256;

// Bound on one reconstructed slice. Every 27B projection fits whole, so each is one decode and one
// GEMM with the full row count to spread over the device; only the output head is sliced. Smaller
// slices starve the K=17408 projection of row blocks.
constexpr std::size_t kSliceBytes = std::size_t{192} << 20;

std::int32_t slice_rows(std::int32_t output_rows, std::int32_t input_rows) {
    const auto fit = kSliceBytes / (static_cast<std::size_t>(input_rows) * 2) / 128 * 128;
    return static_cast<std::int32_t>(
        std::min<std::size_t>(std::max<std::size_t>(fit, 128), output_rows));
}

bool dense_path(std::int32_t input_rows, std::int32_t tokens) {
    return tokens >= kDenseMinTokens && exl3_dense_supported(input_rows);
}

void validate(const Tensor& x, const Weight& w, const Tensor& out, const Tensor* residual) {
    require(is_exl3(w.qtype) && w.layout == QuantLayout::Exl3Tile,
            "exl3 linear: weight is not an EXL3 trellis");
    require(w.qdata != nullptr && w.qhigh != nullptr && w.scales != nullptr,
            "exl3 linear: missing code, suh or svh plane");
    require(w.n > 0 && w.k > 0 && w.n % 128 == 0 && w.k % 128 == 0,
            "exl3 linear: N and K must be multiples of 128");
    require(x.dtype == DType::BF16 && out.dtype == DType::BF16, "exl3 linear: x/out must be BF16");
    require(x.ne[0] == w.k && x.ne[2] == 1 && x.ne[3] == 1 && x.is_contiguous(),
            "exl3 linear: x must be contiguous [K,T]");
    require(out.ne[0] == w.n && out.ne[1] == x.ne[1] && out.ne[2] == 1 && out.ne[3] == 1 &&
                out.nb[0] == 2 && out.nb[1] % 2 == 0,
            "exl3 linear: out must be [N,T] with unit element stride");
    if (residual != nullptr) {
        require(residual->dtype == DType::BF16 && residual->ne[0] == w.n &&
                    residual->ne[1] == x.ne[1] && residual->nb[0] == 2 && residual->nb[1] % 2 == 0,
                "exl3 linear: residual must match out");
    }
}

} // namespace

bool is_exl3(QType qtype) noexcept { return exl3_bitrate(qtype) != 0; }

std::size_t exl3_linear_workspace_capacity_bytes(std::int32_t output_rows, std::int32_t input_rows,
                                                 std::int32_t max_tokens) {
    require(output_rows > 0 && input_rows > 0 && max_tokens > 0,
            "exl3 linear workspace: invalid profile");
    std::size_t bytes = 0;
    const bool dense  = dense_path(input_rows, max_tokens);
    const std::int32_t trellis_tokens = dense ? std::min(max_tokens, kDenseMinTokens - 1) : max_tokens;
    if (trellis_tokens > 0) {
        const auto tokens = static_cast<std::size_t>(chunk_columns(output_rows, trellis_tokens));
        bytes = align256(tokens * static_cast<std::size_t>(input_rows) * 2) +
                align256(tokens * static_cast<std::size_t>(output_rows) * 4);
    }
    if (dense) {
        bytes = std::max(bytes, align256(static_cast<std::size_t>(slice_rows(output_rows, input_rows)) *
                                         static_cast<std::size_t>(input_rows) * 2));
    }
    return bytes;
}

void exl3_linear(const Tensor& x, const Weight& w, Tensor& out, const Tensor* residual,
                 WorkspaceArena& workspace, cudaStream_t stream) {
    validate(x, w, out, residual);
    const std::int32_t tokens = x.ne[1];
    require(tokens > 0, "exl3 linear: T must be positive");
    const auto bits = static_cast<std::int32_t>(exl3_bitrate(w.qtype));
    if (dense_path(w.k, tokens)) {
        const std::int32_t slice = slice_rows(w.n, w.k);
        auto scope               = workspace.scope();
        const auto weight =
            workspace.alloc_bytes(static_cast<std::size_t>(slice) * w.k * 2, 256);
        Exl3ReconstructOperands decode;
        decode.bits   = bits;
        decode.codes  = w.qdata;
        decode.suh    = w.qhigh;
        decode.svh    = w.scales;
        decode.weight = weight.data;
        decode.k      = w.k;
        Exl3DenseOperands gemm;
        gemm.x               = x.data;
        gemm.weight          = weight.data;
        gemm.k               = w.k;
        gemm.tokens          = tokens;
        gemm.out_stride      = static_cast<std::int32_t>(out.nb[1] / 2);
        gemm.residual_stride = residual ? static_cast<std::int32_t>(residual->nb[1] / 2) : 0;
        for (std::int32_t first = 0; first < w.n; first += slice) {
            const std::int32_t rows = std::min(slice, w.n - first);
            decode.first_row        = first;
            decode.rows             = rows;
            exl3_reconstruct_launch(decode, stream);
            gemm.rows     = rows;
            gemm.out      = static_cast<char*>(out.data) + std::size_t(first) * 2;
            gemm.residual = residual ? static_cast<const char*>(residual->data) + std::size_t(first) * 2
                                     : nullptr;
            exl3_dense_launch(gemm, stream);
        }
        return;
    }
    const std::int32_t chunk = chunk_columns(w.n, tokens);
    auto scope               = workspace.scope();
    const auto a = workspace.alloc_bytes(static_cast<std::size_t>(chunk) * w.k * 2, 256);
    const auto c = workspace.alloc_bytes(static_cast<std::size_t>(chunk) * w.n * 4, 256);
    Exl3GemmOperands operands;
    operands.bits  = bits;
    operands.a     = a.data;
    operands.codes = w.qdata;
    operands.suh   = w.qhigh;
    operands.svh   = w.scales;
    operands.c     = static_cast<float*>(c.data);
    operands.k     = w.k;
    operands.n     = w.n;
    for (std::int32_t first = 0; first < tokens; first += chunk) {
        const std::int32_t columns = std::min(chunk, tokens - first);
        exl3_bf16_to_fp16_launch(static_cast<const char*>(x.data) + std::size_t(first) * x.nb[1],
                                 a.data, static_cast<std::int64_t>(columns) * w.k, stream);
        operands.m = columns;
        exl3_gemm_launch(operands, stream);
        exl3_store_bf16_launch(
            static_cast<const float*>(c.data),
            static_cast<char*>(out.data) + std::size_t(first) * out.nb[1], out.nb[1] / 2,
            residual ? static_cast<const char*>(residual->data) + std::size_t(first) * residual->nb[1]
                     : nullptr,
            residual ? residual->nb[1] / 2 : 0, w.n, columns, stream);
    }
}

void exl3_dispatch(const Tensor& x, const Weight& weight, Tensor& out, LinearPolicy,
                   WorkspaceArena* workspace, cudaStream_t stream) {
    require(workspace != nullptr, "exl3 linear requires caller workspace");
    exl3_linear(x, weight, out, nullptr, *workspace, stream);
}

} // namespace ninfer::ops::detail
