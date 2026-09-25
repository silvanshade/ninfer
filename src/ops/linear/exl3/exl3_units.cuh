#pragma once

// One translation unit per trellis bitrate instantiates the vendored exllamav3 GEMM for the
// `mul1` codebook with FP32 output, one kernel per upstream tile shape (index 1..4, 0 unused).

#include <cooperative_groups.h>
#include <cuda_fp16.h>
namespace cg = cooperative_groups;

#include "util.h"
#include "util.cuh"
#include "ptx.cuh"
#include "quant/exl3_kernel_map.cuh"

#define NINFER_EXL3_UNIT_EXTERN(K) extern fp_exl3_gemm_kernel ninfer_exl3_gemm_mul1_b##K[];

NINFER_EXL3_UNIT_EXTERN(1)
NINFER_EXL3_UNIT_EXTERN(2)
NINFER_EXL3_UNIT_EXTERN(3)
NINFER_EXL3_UNIT_EXTERN(4)
NINFER_EXL3_UNIT_EXTERN(5)
NINFER_EXL3_UNIT_EXTERN(6)
NINFER_EXL3_UNIT_EXTERN(7)
NINFER_EXL3_UNIT_EXTERN(8)

#define NINFER_EXL3_UNIT(K)                                                                     \
    fp_exl3_gemm_kernel ninfer_exl3_gemm_mul1_b##K[] = {EXL3_GEMM_KERNEL_INSTANCES(K, true, 2)};
