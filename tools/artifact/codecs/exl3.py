"""Exact trellis code words and Hadamard axis vectors in exl3_tile_v1 layout.

The decode here is the format's reference implementation, not a fast one: it builds the whole
65536-entry mul1 codebook and materializes the rotation, so an independent consumer can be checked
against it word for word.
"""

from __future__ import annotations

from typing import Sequence

import torch

from ..formats import Exl3Format, get_format
from ..layouts import exl3_tile_geometry
from ._tensor_bytes import (
    Payload,
    _payload_length,
    _payload_tensor,
)

TILE = 16
HADAMARD = 128
MUL1_MULTIPLIER = 0x83DCD12D
MUL1_SCALE_WORD = 0x1EEE
MUL1_BIAS_WORD = 0xC931


def _format(format: str | Exl3Format) -> Exl3Format:
    spec = get_format(format) if isinstance(format, str) else format
    if not isinstance(spec, Exl3Format):
        raise ValueError("exl3_tile_v1 requires a trellis format")
    return spec


def _half_from_word(word: int) -> float:
    return (
        torch.tensor([word], dtype=torch.int32)
        .to(torch.int16)
        .view(torch.float16)
        .float()
        .item()
    )


def mul1_codebook() -> torch.Tensor:
    """Every decoded mul1 value, indexed by its 16-bit trellis word."""

    word = torch.arange(1 << 16, dtype=torch.int64)
    product = (word * MUL1_MULTIPLIER) & 0xFFFFFFFF
    byte_sum = sum((product >> shift) & 0xFF for shift in (0, 8, 16, 24))
    step = (1024 + byte_sum).float()
    return (step * _half_from_word(MUL1_SCALE_WORD) + _half_from_word(MUL1_BIAS_WORD)).to(
        torch.float16
    )


def _exact_tile_words(
    tiles: torch.Tensor, n: int, k: int, bits: int, label: str
) -> torch.Tensor:
    shape = (n // TILE, k // TILE, TILE * bits)
    if tiles.dtype != torch.int16 or tuple(tiles.shape) != shape:
        raise TypeError(f"{label} must be int16 words with shape {shape}")
    return tiles.detach().contiguous().cpu()


def _exact_half_vector(vector: torch.Tensor, length: int, label: str) -> torch.Tensor:
    if vector.dtype != torch.float16 or tuple(vector.shape) != (length,):
        raise TypeError(f"{label} must be FP16 with shape ({length},)")
    return vector.detach().contiguous().cpu()


def trellis_band_bytes(
    tiles: torch.Tensor,
    suh: torch.Tensor,
    svh: torch.Tensor,
    rows: int,
    k: int,
    format: str | Exl3Format,
) -> tuple[bytes, bytes, bytes]:
    """Validate and serialize one band's three planes: codes, `suh` over K, `svh` over the band.

    A band is any whole number of tile rows. Only the parent matrix has to be rotatable, so this
    checks the tiling rather than the 128-element blocking.
    """

    spec = _format(format)
    if rows <= 0 or rows % TILE or k <= 0 or k % TILE:
        raise ValueError("a trellis band is a positive whole number of 16x16 tiles")
    words = _exact_tile_words(tiles, rows, k, spec.bits, "trellis tiles")
    input_vector = _exact_half_vector(suh, k, "trellis suh")
    output_vector = _exact_half_vector(svh, rows, "trellis svh")
    return (
        words.numpy().tobytes(),
        input_vector.numpy().tobytes(),
        output_vector.numpy().tobytes(),
    )


def encode_exl3_tile(
    tiles: torch.Tensor,
    suh: torch.Tensor,
    svh: torch.Tensor,
    shape: Sequence[int],
    format: str | Exl3Format,
) -> bytes:
    """Encode exact trellis tiles and both binary16 axis vectors."""

    spec = _format(format)
    geometry = exl3_tile_geometry(spec, shape)
    codes, input_vector, output_vector = trellis_band_bytes(
        tiles, suh, svh, geometry.n, geometry.k, spec
    )
    payload = bytearray(geometry.payload_bytes)
    payload[: geometry.code_plane_bytes] = codes
    payload[geometry.suh_offset : geometry.suh_offset + geometry.suh_bytes] = (
        input_vector
    )
    payload[geometry.svh_offset : geometry.svh_offset + geometry.svh_bytes] = (
        output_vector
    )
    return bytes(payload)


def decode_exl3_tile_words(
    payload: Payload,
    shape: Sequence[int],
    format: str | Exl3Format,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Decode exact trellis tiles, the suh vector over K and the svh vector over N."""

    spec = _format(format)
    geometry = exl3_tile_geometry(spec, shape)
    if _payload_length(payload) != geometry.payload_bytes:
        raise ValueError(
            f"trellis payload has {_payload_length(payload)} bytes, "
            f"expected {geometry.payload_bytes}"
        )
    raw = _payload_tensor(payload, torch.device("cpu"))
    tiles = (
        raw[: geometry.code_plane_bytes]
        .clone()
        .view(torch.int16)
        .reshape(geometry.n // TILE, geometry.k // TILE, TILE * spec.bits)
    )
    suh = (
        raw[geometry.suh_offset : geometry.suh_offset + geometry.suh_bytes]
        .clone()
        .view(torch.float16)
    )
    svh = (
        raw[geometry.svh_offset : geometry.svh_offset + geometry.svh_bytes]
        .clone()
        .view(torch.float16)
    )
    return tiles, suh, svh


def tensor_core_permutation() -> torch.Tensor:
    """Row-major tile position of every trellis element, in EXL3's tensor-core order.

    Element ``p`` of a tile decodes to row-major position ``permutation[p]`` of a ``16 x 16`` tile
    whose rows run along K, the orientation the EXL3 quantizer works in.
    """

    lane = torch.arange(32).unsqueeze(1)
    row = (lane % 4) * 2 + torch.tensor([0, 1, 8, 9, 0, 1, 8, 9]).unsqueeze(0)
    column = lane // 4 + torch.tensor([0, 0, 0, 0, 8, 8, 8, 8]).unsqueeze(0)
    return (row * TILE + column).reshape(-1)


def trellis_states(tiles: torch.Tensor, bits: int) -> torch.Tensor:
    """Read every tile's 256 trellis words from its tail-biting ring.

    A tile's ``16 * bits`` stored words are a bitstream whose word ``w`` is stored at ``w ^ 1``,
    each word most significant bit first. Element ``t`` is the 16-bit window ending at stream bit
    ``(t + 1) * bits - 1``, so its last ``bits`` bits are element ``t``'s own step and the rest is
    the state it inherits.
    """

    weights_per_tile = TILE * TILE
    stream_bits = weights_per_tile * bits
    words = tiles.reshape(-1, TILE * bits).to(torch.int64) & 0xFFFF
    words = words[:, torch.arange(TILE * bits) ^ 1]
    ring = (words.unsqueeze(-1) >> torch.arange(15, -1, -1)) & 1
    ring = ring.reshape(words.shape[0], stream_bits)
    window = torch.arange(weights_per_tile).unsqueeze(1) * bits + bits - 16
    source = (window + torch.arange(16).unsqueeze(0)) % stream_bits
    return (ring[:, source] << torch.arange(15, -1, -1)).sum(dim=-1)


def hadamard_matrix(order: int = HADAMARD) -> torch.Tensor:
    """The Sylvester Hadamard matrix of the given order, scaled by 1 / sqrt(order)."""

    if order <= 0 or order & (order - 1):
        raise ValueError("the Sylvester construction needs a power-of-two order")
    matrix = torch.ones((1, 1), dtype=torch.float32)
    while matrix.shape[0] < order:
        matrix = torch.cat(
            (
                torch.cat((matrix, matrix), dim=1),
                torch.cat((matrix, -matrix), dim=1),
            ),
            dim=0,
        )
    return matrix / (order**0.5)


def decode_exl3_codes(
    tiles: torch.Tensor,
    shape: Sequence[int],
    format: str | Exl3Format,
) -> torch.Tensor:
    """Decode the trellis codes to their binary16 values, before either rotation.

    The result is a ``[N,K]`` matrix: a tile's elements are unpermuted from tensor-core order into
    a row-major ``16 x 16`` tile whose rows run along K, then transposed into NInfer's
    output-major orientation.
    """

    spec = _format(format)
    geometry = exl3_tile_geometry(spec, shape)
    values = mul1_codebook()[trellis_states(tiles, spec.bits)]
    rows = torch.empty_like(values)
    rows[:, tensor_core_permutation()] = values
    return (
        rows.reshape(geometry.n // TILE, geometry.k // TILE, TILE, TILE)
        .permute(0, 3, 1, 2)
        .reshape(geometry.n, geometry.k)
    )


def dequantize_exl3_tile(
    payload: Payload,
    shape: Sequence[int],
    format: str | Exl3Format,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Reconstruct a trellis matrix from its exact stored words."""

    spec = _format(format)
    geometry = exl3_tile_geometry(spec, shape)
    tiles, suh, svh = decode_exl3_tile_words(payload, shape, spec)
    decoded = decode_exl3_codes(tiles, shape, spec).float()
    hadamard = hadamard_matrix()
    rotated = torch.einsum(
        "ij,nbj->nbi",
        hadamard,
        decoded.reshape(geometry.n, geometry.k // HADAMARD, HADAMARD),
    ).reshape(geometry.n, geometry.k)
    scaled = rotated * suh.float().unsqueeze(0)
    rotated = torch.einsum(
        "ij,bjk->bik",
        hadamard,
        scaled.reshape(geometry.n // HADAMARD, HADAMARD, geometry.k),
    ).reshape(geometry.n, geometry.k)
    return (rotated * svh.float().unsqueeze(1)).to(dtype)
