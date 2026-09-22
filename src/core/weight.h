#pragma once

#include "core/dtype.h"

#include <cstdint>

namespace ninfer {

enum class QType : std::uint16_t {
    Q4_G64_FP16         = 0,
    Q5_G64_FP16         = 1,
    Q6_G64_FP16         = 2,
    Q8_G32_FP16         = 3,
    BF16                = 4,
    FP32                = 5,
    INT32               = 6,
    NVFP4               = 7,
    FP8_E4M3FN_ROW_BF16 = 8,
    // EXL3 trellis codes. Both the bitrate and the codebook are part of the format rather than
    // parameters beside it: the rate fixes the tile width (16*K uint16 per 16x16 tile) and the
    // codebook fixes what a 16-bit trellis word decodes to, and a trellis has no group scales to
    // attach a size to. mul1 is exllamav3's default and the only codebook admitted here; a name
    // per (rate, codebook) keeps the container's format string self-describing, as every other
    // row here is.
    EXL3_K3_MUL1 = 9,
    EXL3_K4_MUL1 = 10,
};

enum class QuantLayout : std::uint16_t {
    RowSplit            = 0,
    Contiguous          = 1,
    BlockScaleK16M128x4 = 2,
    RowScale            = 3,
    // Trellis tiles of 16x16 weights, followed by the input- and output-side Hadamard scale
    // vectors (suh over K, svh over N).
    Exl3Tile = 4,
};

// Bits per weight for the EXL3 trellis formats; zero for every other format.
[[nodiscard]] constexpr std::uint32_t exl3_bitrate(QType format) noexcept {
    switch (format) {
    case QType::EXL3_K3_MUL1:
        return 3;
    case QType::EXL3_K4_MUL1:
        return 4;
    default:
        return 0;
    }
}

struct Weight {
    const void* payload            = nullptr;
    std::uint64_t payload_bytes    = 0;
    std::uint64_t high_plane_bytes = 0;
    QType qtype                    = QType::Q4_G64_FP16;
    std::uint32_t group_size       = 0;
    std::int32_t shape[4]          = {1, 1, 1, 1};
    std::int32_t padded_shape[4]   = {1, 1, 1, 1};
    std::uint32_t ndim             = 0;

    const void* qdata          = nullptr;
    const void* qhigh          = nullptr;
    const void* scales         = nullptr;
    std::int32_t n             = 0;
    std::int32_t k             = 0;
    std::int32_t group         = 0;
    QuantLayout layout         = QuantLayout::RowSplit;
    DType scale_dtype          = DType::FP32;
    std::int32_t scale_ne[4]   = {1, 1, 1, 1};
    std::int64_t scale_nb[4]   = {0, 0, 0, 0};
    float weight_scale_divisor = 0.0F;
    float input_scale_divisor  = 0.0F;
};

} // namespace ninfer
