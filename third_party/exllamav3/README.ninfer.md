# exllamav3 EXL3 GEMM kernels

Source: `https://github.com/turboderp-org/exllamav3`, commit
`6b84a21b6f1e5da3f291b9e1019061f0de788279`, directory `exllamav3/exllamav3_ext/`. MIT license,
see `LICENSE`.

Vendored: the header-only kernel path of the EXL3 GEMM (`exl3_gemm_kernel`), which dequantizes
trellis tiles in the matmul loop and applies both Hadamard axis transforms. Files:

- `util.h`, `util.cuh`, `ptx.cuh`, `compat.cuh`
- `quant/exl3_gemm_kernel.cuh`, `quant/exl3_gemm_inner.cuh`, `quant/exl3_dq.cuh`,
  `quant/codebook.cuh`, `quant/hadamard_inner.cuh`, `quant/exl3_kernel_map.cuh`,
  `quant/exl3_devctx.cuh`

Not vendored: the PyTorch-facing host launchers, the autotuner, the GEMV and MoE paths, and the
kernel tables. NInfer instantiates the `mul1` codebook with FP32 output only
(`src/ops/linear/exl3/`) and owns its own launcher.

Adapted, not vendored: `src/ops/linear/exl3/exl3_reconstruct.cu` follows the dequant and the two
Hadamard butterflies of `reconstruct_had_tile` in `quant/reconstruct.cu` at the same commit. It
reads tiles from the NInfer grid, stores BF16 transposed to `[N, K]`, and feeds NInfer's own BF16
Tensor Core GEMM, which stands in for exllamav3's `hgemm_recon` on large token counts.

## Local patch

`quant/exl3_gemm_inner.cuh` changes only where the kernel reads trellis tiles from global memory,
under `NINFER_EXL3_NK_TILES`. `exl3_tile_v1` (`docs/maintainer/storage-layouts.md` §6) stores the
16×16 tile grid row-major over output rows, `[N/16][K/16]`, where exllamav3 stores `[K/16][N/16]`.
The tile words themselves are identical. With the macro defined, the per-chunk global offset follows
the NInfer grid; the shared-memory stage keeps the upstream order, so nothing downstream of the
copy changes. Column-slice mode (`size_n_stride`) is not used with that layout.
