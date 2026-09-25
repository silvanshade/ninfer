#include "core/device.h"

#define NINFER_EXL3_NK_TILES 1
#include "ops/linear/exl3/exl3_units.cuh"
#include "quant/exl3_gemm_kernel.cuh"
// The vendored util.cuh defines a function-like `cuda_check` macro that collides with
// NInfer's `CUDA_CHECK` expansion.
#undef cuda_check

#include "ops/linear/exl3/exl3_launch.h"

#include <cuda_bf16.h>

#include <array>
#include <mutex>
#include <stdexcept>
#include <string>

namespace ninfer::ops::detail {
namespace {

constexpr int kTilesizeK[] = {EXL3_GEMM_TILESIZE_K};
constexpr int kTilesizeN[] = {EXL3_GEMM_TILESIZE_N};
constexpr int kBlockDim[]  = {EXL3_GEMM_BLOCKDIM};

fp_exl3_gemm_kernel* const kKernels[9] = {
    nullptr,
    ninfer_exl3_gemm_mul1_b1,
    ninfer_exl3_gemm_mul1_b2,
    ninfer_exl3_gemm_mul1_b3,
    ninfer_exl3_gemm_mul1_b4,
    ninfer_exl3_gemm_mul1_b5,
    ninfer_exl3_gemm_mul1_b6,
    ninfer_exl3_gemm_mul1_b7,
    ninfer_exl3_gemm_mul1_b8,
};

// Shape choice copied from exllamav3 `select_gemm_shape` (exl3_kernel_map.cu, single GEMM) for the
// Hopper/Blackwell class, the only one NInfer builds for (SM120).
int select_shape(int size_k, int size_n, int bits) {
    const bool mod_256 = size_n % 256 == 0;
    const bool mod_512 = size_n % 512 == 0;
    if ((bits == 4 || bits == 2) && size_k <= 2048) return 1;
    if (bits >= 7) {
        if (mod_256 && size_n <= 8192) return size_k > 32768 ? 3 : 2;
        if (mod_512 && size_n > 32768) return 4;
        return 2;
    }
    if (mod_256 && size_n <= 4096) return size_k > 8192 && bits >= 3 ? 3 : 2;
    if (mod_512 && size_n > 16384) return 4;
    if (mod_256) return 3;
    return 2;
}

struct DeviceState {
    int sms     = 0;
    int* locks  = nullptr;
    std::array<bool, 9 * (EXL3_GEMM_NUM_SHAPES + 1)> attributes{};
};

// The cross-block reduction locks must be zero between launches; the kernel leaves them zeroed.
// They are allocated once per device, outside any capture.
DeviceState& device_state(cudaStream_t stream) {
    static std::mutex mutex;
    static std::array<DeviceState, 16> states;
    int device = 0;
    CUDA_CHECK(cudaGetDevice(&device));
    if (device < 0 || device >= static_cast<int>(states.size())) {
        throw std::runtime_error("exl3 linear: device index out of range");
    }
    std::lock_guard lock(mutex);
    auto& state = states[device];
    if (state.locks == nullptr) {
        cudaStreamCaptureStatus capture = cudaStreamCaptureStatusNone;
        CUDA_CHECK(cudaStreamIsCapturing(stream, &capture));
        if (capture != cudaStreamCaptureStatusNone) {
            throw std::runtime_error("exl3 linear: first launch on a device must precede capture");
        }
        CUDA_CHECK(cudaDeviceGetAttribute(&state.sms, cudaDevAttrMultiProcessorCount, device));
        const std::size_t bytes = (MAX_TILES_C + MAX_BARRIERS * 2 + MOE_SCHED_INTS) * sizeof(int);
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&state.locks), bytes));
        CUDA_CHECK(cudaMemset(state.locks, 0, bytes));
    }
    return state;
}

__global__ void bf16_to_fp16_kernel(const __nv_bfloat16* __restrict__ x, half* __restrict__ a,
                                    std::int64_t count) {
    const std::int64_t stride = static_cast<std::int64_t>(gridDim.x) * blockDim.x;
    for (std::int64_t i = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         i < count; i += stride) {
        a[i] = __float2half_rn(__bfloat162float(x[i]));
    }
}

__global__ void store_bf16_kernel(const float* __restrict__ c, __nv_bfloat16* __restrict__ out,
                                  std::int64_t out_stride, const __nv_bfloat16* residual,
                                  std::int64_t residual_stride, int n) {
    const int t = static_cast<int>(blockIdx.y);
    for (int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x); i < n;
         i += static_cast<int>(gridDim.x * blockDim.x)) {
        float value = c[static_cast<std::int64_t>(t) * n + i];
        if (residual != nullptr) {
            value += __bfloat162float(residual[static_cast<std::int64_t>(t) * residual_stride + i]);
        }
        out[static_cast<std::int64_t>(t) * out_stride + i] = __float2bfloat16_rn(value);
    }
}

} // namespace

void exl3_gemm_launch(const Exl3GemmOperands& o, cudaStream_t stream) {
    if (o.bits < 1 || o.bits > 8 || o.m <= 0 || o.k % 128 != 0 || o.n % 128 != 0 || o.k <= 0 ||
        o.n <= 0 || o.a == nullptr || o.codes == nullptr || o.suh == nullptr ||
        o.svh == nullptr || o.c == nullptr) {
        throw std::invalid_argument("exl3 linear: invalid GEMM operands");
    }
    auto& state         = device_state(stream);
    const int shape     = select_shape(o.k, o.n, o.bits);
    const int tilesize_k = kTilesizeK[shape];
    const int tilesize_n = kTilesizeN[shape];
    if (o.k % tilesize_k != 0 || o.n % tilesize_n != 0) {
        throw std::invalid_argument("exl3 linear: shape " + std::to_string(shape) +
                                    " does not tile K=" + std::to_string(o.k) +
                                    " N=" + std::to_string(o.n));
    }
    if (static_cast<std::int64_t>(o.n / 128) * ((o.m + 15) / 16) > MAX_TILES_C) {
        throw std::invalid_argument("exl3 linear: output exceeds the lock buffer");
    }
    fp_exl3_gemm_kernel kernel = kKernels[o.bits][shape];
    auto& attribute            = state.attributes[o.bits * (EXL3_GEMM_NUM_SHAPES + 1) + shape];
    if (!attribute) {
        CUDA_CHECK(cudaFuncSetAttribute(reinterpret_cast<const void*>(kernel),
                                        cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_MAX));
        attribute = true;
    }
    const int slices = o.k / tilesize_k * (o.n / tilesize_n);
    const int blocks = slices < state.sms ? (slices > 0 ? slices : 1) : state.sms;

    const half* a        = static_cast<const half*>(o.a);
    const auto* b        = static_cast<const std::uint16_t*>(o.codes);
    void* c              = o.c;
    int size_m           = o.m;
    int size_k           = o.k;
    int size_n           = o.n;
    int* locks           = state.locks;
    const half* suh      = static_cast<const half*>(o.suh);
    half* a_had          = const_cast<half*>(a);
    const half* svh      = static_cast<const half*>(o.svh);
    void* arguments[]    = {&a, &b, &c, &size_m, &size_k, &size_n, &locks, &suh, &a_had, &svh};
    CUDA_CHECK(cudaLaunchCooperativeKernel(reinterpret_cast<const void*>(kernel), blocks,
                                           kBlockDim[shape], arguments, SMEM_MAX, stream));
}

void exl3_bf16_to_fp16_launch(const void* x, void* a, std::int64_t count, cudaStream_t stream) {
    const int blocks = static_cast<int>(count / 256 < 1024 ? (count + 255) / 256 : 1024);
    bf16_to_fp16_kernel<<<blocks, 256, 0, stream>>>(static_cast<const __nv_bfloat16*>(x),
                                                   static_cast<half*>(a), count);
    CUDA_CHECK(cudaGetLastError());
}

void exl3_store_bf16_launch(const float* c, void* out, std::int64_t out_stride,
                            const void* residual, std::int64_t residual_stride, std::int32_t n,
                            std::int32_t tokens, cudaStream_t stream) {
    const dim3 grid((n + 255) / 256, tokens);
    store_bf16_kernel<<<grid, 256, 0, stream>>>(c, static_cast<__nv_bfloat16*>(out), out_stride,
                                                static_cast<const __nv_bfloat16*>(residual),
                                                residual_stride, n);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops::detail
