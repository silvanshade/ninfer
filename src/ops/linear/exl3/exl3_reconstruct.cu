#include "core/device.h"

#include <cuda_fp16.h>

#include "util.cuh"
#include "ptx.cuh"
#include "quant/exl3_dq.cuh"
#include "quant/hadamard_inner.cuh"
// The vendored util.cuh defines a function-like `cuda_check` macro that collides with
// NInfer's `CUDA_CHECK` expansion.
#undef cuda_check

#include "ops/common/math.h"
#include "ops/linear/bf16/bf16_config.h"
#include "ops/linear/bf16/bf16_gemm_mma.cuh"
#include "ops/linear/exl3/exl3_launch.h"

#include <cuda_bf16.h>

#include <array>
#include <stdexcept>

namespace ninfer::ops::detail {
namespace {

// Decodes one 128x128 tile of the trellis into the original basis and stores it transposed, as
// BF16 rows of the [N, K] weight: W = diag(suh) . H128 . W_hat . H128 . diag(svh) per tile, with
// 1/sqrt(128) per side. The dequant and both butterflies follow exllamav3 `reconstruct_had_tile`
// (reconstruct.cu at the vendored revision); what differs is the tile read, which follows the
// NInfer [N/16][K/16] grid, and the store, which transposes through shared memory.
template <int Bits>
__global__ __launch_bounds__(256) void exl3_reconstruct_kernel(
    __nv_bfloat16* __restrict__ out, const std::uint16_t* __restrict__ codes,
    const half* __restrict__ suh, const half* __restrict__ svh, int blocks_k, int k_len,
    int first_tile_n) {
    constexpr int packed_size = 16 * Bits; // uint16 words per 16x16 tile
    constexpr int j_int4      = packed_size / 8;
    constexpr float r_scale   = 0.08838834764831845f;

    const int t       = static_cast<int>(threadIdx.x);
    const int lane_id = t % 32;
    const int warp_id = t / 32;
    const int kb      = static_cast<int>(blockIdx.x);
    const int nb      = first_tile_n + static_cast<int>(blockIdx.y);

    __shared__ std::uint32_t s_packed[8][8][packed_size / 2];
    __shared__ half2 stile[128 * 64];

    auto tix = [&](int row, int q, int p) { return row * 64 + (q ^ ((row >> 2) & 31)) * 2 + p; };

    // For each of the 8 output-row subtiles the 8 input-row subtiles are one contiguous run.
    for (int u = t; u < 8 * 8 * j_int4; u += 256) {
        const int wn   = u / (8 * j_int4);
        const int rest = u % (8 * j_int4);
        const int j    = rest / j_int4;
        const int r    = rest % j_int4;
        const std::uint16_t* tile =
            codes +
            static_cast<std::size_t>((nb * 8 + wn) * blocks_k + kb * 8 + j) * packed_size;
        reinterpret_cast<int4*>(s_packed[j][wn])[r] = reinterpret_cast<const int4*>(tile)[r];
    }
    __syncthreads();

    // Dequant: warp w decodes output subtile w for all 8 input subtiles, into k-row-major stile.
#pragma unroll 1
    for (int j = 0; j < 8; ++j) {
        const int wn = warp_id;
        FragB frag[2];
        dq_dispatch<Bits, 2>(s_packed[j][wn], lane_id * 8, frag[0], frag[1]);

        const half2 n0 = __shfl_down_sync(0xFFFFFFFF, frag[0][0], 4, 32);
        const half2 n1 = __shfl_down_sync(0xFFFFFFFF, frag[0][1], 4, 32);
        const half2 n2 = __shfl_down_sync(0xFFFFFFFF, frag[1][0], 4, 32);
        const half2 n3 = __shfl_down_sync(0xFFFFFFFF, frag[1][1], 4, 32);

        if (!(lane_id & 4)) {
            const half2 m0 = __halves2half2(__low2half(frag[0][0]), __low2half(n0));
            const half2 m1 = __halves2half2(__high2half(frag[0][0]), __high2half(n0));
            const half2 m2 = __halves2half2(__low2half(frag[0][1]), __low2half(n1));
            const half2 m3 = __halves2half2(__high2half(frag[0][1]), __high2half(n1));
            const half2 m4 = __halves2half2(__low2half(frag[1][0]), __low2half(n2));
            const half2 m5 = __halves2half2(__high2half(frag[1][0]), __high2half(n2));
            const half2 m6 = __halves2half2(__low2half(frag[1][1]), __low2half(n3));
            const half2 m7 = __halves2half2(__high2half(frag[1][1]), __high2half(n3));
            const int r0   = j * 16 + (lane_id % 4) * 2;
            const int r1   = r0 + 1;
            const int r2   = r0 + 8;
            const int r3   = r0 + 9;
            const int c0   = lane_id / 8;
            const int q0 = (wn * 8 + c0) >> 1, p0 = c0 & 1;
            const int q1 = (wn * 8 + c0 + 4) >> 1, p1 = c0 & 1;
            stile[tix(r0, q0, p0)] = m0;
            stile[tix(r1, q0, p0)] = m1;
            stile[tix(r2, q0, p0)] = m2;
            stile[tix(r3, q0, p0)] = m3;
            stile[tix(r0, q1, p1)] = m4;
            stile[tix(r1, q1, p1)] = m5;
            stile[tix(r2, q1, p1)] = m6;
            stile[tix(r3, q1, p1)] = m7;
        }
    }
    __syncthreads();

    // Column transform (along k), unchanged from upstream.
    const half2 rs2 = __float2half2_rn(r_scale);
#pragma unroll
    for (int qq = 0; qq < 4; ++qq) {
        const int q  = warp_id * 4 + qq;
        const int qs = q ^ lane_id;
        half2 a[4], b[4];
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const half4 v = *reinterpret_cast<const half4*>(stile + (lane_id * 4 + i) * 64 + qs * 2);
            a[i]          = v.x;
            b[i]          = v.y;
        }
#pragma unroll
        for (int x = 0; x < 2; ++x) {
            half2* v       = x == 0 ? a : b;
            const half2 s0 = __hadd2(v[0], v[1]), d0 = __hsub2(v[0], v[1]);
            const half2 s1 = __hadd2(v[2], v[3]), d1 = __hsub2(v[2], v[3]);
            v[0]           = __hmul2(__hadd2(s0, s1), rs2);
            v[1]           = __hmul2(__hadd2(d0, d1), rs2);
            v[2]           = __hmul2(__hsub2(s0, s1), rs2);
            v[3]           = __hmul2(__hsub2(d0, d1), rs2);
#pragma unroll
            for (int i = 0; i < 4; ++i) v[i] = shuffle_had_h2x32(v[i], lane_id);
        }
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            half4 v;
            v.x = a[i];
            v.y = b[i];
            *reinterpret_cast<half4*>(stile + (lane_id * 4 + i) * 64 + qs * 2) = v;
        }
    }
    __syncthreads();

    // Row transform (along n): lane l ends holding final columns 4l..4l+3 of k-row R. Values stay
    // in registers until every warp has finished reading stile, which the transpose then reuses.
    const half4 sv4 = reinterpret_cast<const half4*>(svh)[nb * 32 + lane_id];
    const float sv[4] = {__low2float(sv4.x), __high2float(sv4.x), __low2float(sv4.y),
                         __high2float(sv4.y)};
    float value[16][4];
#pragma unroll
    for (int rr = 0; rr < 16; ++rr) {
        const int row    = warp_id * 16 + rr;
        const int base   = row * 64 + (lane_id ^ ((row >> 2) & 31)) * 2;
        const half2 v01  = stile[base];
        const half2 v23  = stile[base + 1];
        const float v0 = __low2float(v01), v1 = __high2float(v01);
        const float v2 = __low2float(v23), v3 = __high2float(v23);
        const float s0 = v0 + v1, d0 = v0 - v1;
        const float s1 = v2 + v3, d1 = v2 - v3;
        half2 h01      = __hmul2(__floats2half2_rn(s0 + s1, d0 + d1), rs2);
        half2 h23      = __hmul2(__floats2half2_rn(s0 - s1, d0 - d1), rs2);
        h01            = shuffle_had_h2x32(h01, lane_id);
        h23            = shuffle_had_h2x32(h23, lane_id);
        const float su = __half2float(suh[kb * 128 + row]);
        value[rr][0]   = __low2float(h01) * su * sv[0];
        value[rr][1]   = __high2float(h01) * su * sv[1];
        value[rr][2]   = __low2float(h23) * su * sv[2];
        value[rr][3]   = __high2float(h23) * su * sv[3];
    }
    __syncthreads();

    // Transpose through shared memory: element (n, k) of the tile sits at n * 128 + (k ^ swizzle(n)),
    // swizzle(n) = ((n >> 2) & 31) << 1, so the column writes below spread over all banks and each
    // 4-element k group stays one aligned 8-byte word (with its two pairs swapped when bit 1 is set).
    auto* ttile = reinterpret_cast<__nv_bfloat16*>(stile);
#pragma unroll
    for (int rr = 0; rr < 16; ++rr) {
        const int k = warp_id * 16 + rr;
#pragma unroll
        for (int c = 0; c < 4; ++c) {
            const int n                          = lane_id * 4 + c;
            ttile[n * 128 + (k ^ (lane_id << 1))] = __float2bfloat16_rn(value[rr][c]);
        }
    }
    __syncthreads();

    const int out_row0 = (nb - first_tile_n) * 128;
#pragma unroll 4
    for (int i = 0; i < 16; ++i) {
        const int n      = warp_id * 16 + i;
        const int swz    = ((n >> 2) & 31) << 1;
        const int group  = (lane_id * 4) ^ (swz & ~3);
        uint2 word       = *reinterpret_cast<const uint2*>(ttile + n * 128 + group);
        if (swz & 2) { word = make_uint2(word.y, word.x); }
        *reinterpret_cast<uint2*>(out + static_cast<std::size_t>(out_row0 + n) * k_len + kb * 128 +
                                  lane_id * 4) = word;
    }
}

using ReconstructKernel = void (*)(__nv_bfloat16*, const std::uint16_t*, const half*, const half*,
                                   int, int, int);

constexpr std::array<ReconstructKernel, 9> kReconstructKernels = {
    nullptr,
    exl3_reconstruct_kernel<1>,
    exl3_reconstruct_kernel<2>,
    exl3_reconstruct_kernel<3>,
    exl3_reconstruct_kernel<4>,
    exl3_reconstruct_kernel<5>,
    exl3_reconstruct_kernel<6>,
    exl3_reconstruct_kernel<7>,
    exl3_reconstruct_kernel<8>,
};

struct Exl3MmaOutputTile {
    __nv_bfloat16* data;
    std::int32_t leading_dim;
    const __nv_bfloat16* residual;
    std::int32_t residual_leading_dim;

    __device__ __forceinline__ void store(std::int32_t row, std::int32_t token, float value) const {
        if (residual != nullptr) {
            value += __bfloat162float(
                residual[static_cast<std::int64_t>(token) * residual_leading_dim + row]);
        }
        data[static_cast<std::int64_t>(token) * leading_dim + row] = __float2bfloat16_rn(value);
    }
};

struct Exl3MmaOutput {
    __nv_bfloat16* data;
    std::int32_t leading_dim;
    const __nv_bfloat16* residual;
    std::int32_t residual_leading_dim;

    __device__ __forceinline__ Exl3MmaOutputTile tile(std::int32_t) const {
        return {data, leading_dim, residual, residual_leading_dim};
    }
};

using Exl3Mma = Bf16MmaSchedule<64, 128, 64, 32, 32, 2, 2, Cache::cg, Cache::cg,
                                Bf16MmaFragmentPipeline::PingPong, Bf16MmaRaster::TokenFast>;
static_assert(Exl3Mma::kSharedBytes <= 48 * 1024);
// Token-fast raster derives a block's row tile from the grid alone, so a geometry of one block row
// serves any row count that is a multiple of it: the grid, not the geometry, spans the slice.
static_assert(Exl3Mma::kRaster == Bf16MmaRaster::TokenFast);

template <std::int32_t InputRows, bool FullTokens>
void launch_mma(const Exl3DenseOperands& o, cudaStream_t stream) {
    using Geometry   = Bf16Geometry<Exl3Mma::kBlockRows, InputRows>;
    const int blocks = o.rows / Exl3Mma::kBlockRows * div_up(o.tokens, Exl3Mma::kBlockCols);
    const Exl3MmaOutput output{static_cast<__nv_bfloat16*>(o.out), o.out_stride,
                               static_cast<const __nv_bfloat16*>(o.residual), o.residual_stride};
    bf16_gemm_mma_kernel<Geometry, Exl3Mma, FullTokens>
        <<<blocks, Exl3Mma::kThreads, Exl3Mma::kSharedBytes, stream>>>(
            static_cast<const __nv_bfloat16*>(o.x), static_cast<const __nv_bfloat16*>(o.weight),
            output, o.tokens);
    CUDA_CHECK(cudaGetLastError());
}

template <std::int32_t InputRows>
void launch_mma(const Exl3DenseOperands& o, cudaStream_t stream) {
    if (o.tokens % Exl3Mma::kBlockCols == 0) {
        launch_mma<InputRows, true>(o, stream);
    } else {
        launch_mma<InputRows, false>(o, stream);
    }
}

} // namespace

bool exl3_dense_supported(std::int32_t input_rows) noexcept {
    return input_rows == 5120 || input_rows == 6144 || input_rows == 17408;
}

void exl3_reconstruct_launch(const Exl3ReconstructOperands& o, cudaStream_t stream) {
    if (o.bits < 1 || o.bits > 8 || o.k <= 0 || o.k % 128 != 0 || o.first_row < 0 ||
        o.first_row % 128 != 0 || o.rows <= 0 || o.rows % 128 != 0 || o.codes == nullptr ||
        o.suh == nullptr || o.svh == nullptr || o.weight == nullptr) {
        throw std::invalid_argument("exl3 reconstruct: invalid operands");
    }
    const dim3 grid(o.k / 128, o.rows / 128);
    kReconstructKernels[o.bits]<<<grid, 256, 0, stream>>>(
        static_cast<__nv_bfloat16*>(o.weight), static_cast<const std::uint16_t*>(o.codes),
        static_cast<const half*>(o.suh), static_cast<const half*>(o.svh), o.k / 16, o.k,
        o.first_row / 128);
    CUDA_CHECK(cudaGetLastError());
}

void exl3_dense_launch(const Exl3DenseOperands& o, cudaStream_t stream) {
    if (o.rows <= 0 || o.rows % Exl3Mma::kBlockRows != 0 || o.tokens <= 0 || o.x == nullptr ||
        o.weight == nullptr || o.out == nullptr) {
        throw std::invalid_argument("exl3 dense: invalid operands");
    }
    switch (o.k) {
    case 5120: launch_mma<5120>(o, stream); return;
    case 6144: launch_mma<6144>(o, stream); return;
    case 17408: launch_mma<17408>(o, stream); return;
    default: throw std::invalid_argument("exl3 dense: unsupported input rows");
    }
}

} // namespace ninfer::ops::detail
