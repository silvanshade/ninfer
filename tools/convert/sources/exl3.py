"""Interpret an EXL3 quantized linear: trellis tiles, both axis vectors, and the codebook marker.

The words are preserved exactly. The only thing this transforms is the tile grid: exllamav3 stores
a matrix as `[K/16, N/16]` tiles of a `[K,N]` matrix, NInfer stores `[N,K]`, so the grid transposes
while every tile's `16 * bits` words travel unchanged.
"""

from __future__ import annotations

from math import prod

import torch

from tools.artifact.codecs.exl3 import dequantize_exl3_tile, encode_exl3_tile
from tools.artifact.formats import Exl3Format, get_format
from .logical import LogicalSource, TrellisRows
from .safetensors import SafetensorsSource

TILE = 16
HADAMARD = 128
MUL1_MULTIPLIER = 0x83DCD12D


def _marker(store: SafetensorsSource, prefix: str) -> str:
    """Name the format from the checkpoint's own codebook marker, or refuse it."""

    if store.has(f"{prefix}.mcg"):
        raise ValueError(f"{prefix}: the mcg codebook has no NInfer format")
    if not store.has(f"{prefix}.mul1"):
        raise ValueError(
            f"{prefix}: no codebook marker, so the plain 3inst codebook, "
            "which has no NInfer format"
        )
    word = store.read_flat(f"{prefix}.mul1").to(torch.int64).reshape(-1)
    if word.numel() != 1 or int(word[0]) & 0xFFFFFFFF != MUL1_MULTIPLIER:
        raise ValueError(f"{prefix}: mul1 marker is not the reference multiplier")
    return "mul1"


def exl3_linear_source(store: SafetensorsSource, prefix: str) -> LogicalSource:
    """Read one EXL3 linear as an `[N,K]` trellis source."""

    codebook = _marker(store, prefix)
    trellis = store.describe(f"{prefix}.trellis")
    if len(trellis.shape) != 3 or trellis.dtype != "I16":
        raise ValueError(f"{prefix}.trellis: expected a three-dimensional I16 tensor")
    tiles_k, tiles_n, words = trellis.shape
    k, n = tiles_k * TILE, tiles_n * TILE
    if words % TILE:
        raise ValueError(
            f"{prefix}.trellis: {words} words per tile is a half-integer rate, "
            "which has no NInfer format"
        )
    bits = words // TILE
    format = f"exl3_k{bits}_{codebook}"
    spec = get_format(format)
    if not isinstance(spec, Exl3Format):
        raise ValueError(f"{prefix}: {bits}-bit trellis has no NInfer format")
    if n % HADAMARD or k % HADAMARD:
        raise ValueError(f"{prefix}: [{n},{k}] is not rotatable in 128-element blocks")
    for name, length in ((f"{prefix}.suh", k), (f"{prefix}.svh", n)):
        info = store.describe(name)
        if info.dtype != "F16" or prod(info.shape) != length:
            raise ValueError(f"{name}: expected {length} F16 words")

    def vectors() -> tuple[torch.Tensor, torch.Tensor]:
        return store.read_flat(f"{prefix}.suh"), store.read_flat(f"{prefix}.svh")

    def read_trellis(begin: int, end: int) -> TrellisRows:
        if begin % TILE or end % TILE or not 0 <= begin < end <= n:
            raise ValueError(f"{prefix}: trellis rows [{begin},{end}) are not tile bands")
        suh, svh = vectors()
        # The source grid is K-major, so a band of output rows is a column slice of it: read the
        # whole grid's rows for those tile columns, then transpose the grid into row-major order.
        columns = torch.arange(begin // TILE, end // TILE)
        grid = store.read_flat(f"{prefix}.trellis").reshape(tiles_k, tiles_n, words)
        tiles = grid[:, columns, :].permute(1, 0, 2).contiguous()
        return TrellisRows(format, tiles, suh, svh[begin:end])

    def read_values(begin: int, end: int) -> torch.Tensor:
        if begin % (HADAMARD * k) or end % (HADAMARD * k):
            raise ValueError(
                f"{prefix}: a trellis decodes in 128-row blocks, so element range "
                f"[{begin},{end}) must cover whole blocks"
            )
        first, last = begin // k, end // k
        rows = read_trellis(first, last)
        payload = encode_exl3_tile(
            rows.tiles, rows.suh, rows.svh, (last - first, k), format
        )
        return dequantize_exl3_tile(payload, (last - first, k), format).reshape(-1)

    return LogicalSource(
        (n, k),
        f"{store.path}:{prefix} ({format})",
        read_values,
        read_trellis=read_trellis,
        origin=f"{store.path}:{prefix}",
    )


def trellis_format(source: LogicalSource) -> str | None:
    """Name the trellis format a source carries, or nothing when it carries none.

    A recipe does not choose the rate or the codebook for an imported checkpoint; it asks what the
    words already are.
    """

    if source.read_trellis is None:
        return None
    return source.read_trellis(0, TILE).format
