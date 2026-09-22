from __future__ import annotations

import pytest

from tools.artifact.layouts import (
    encoded_size,
    exl3_tile_geometry,
)
from tools.convert.recipe import default_layout


def test_exl3_tile_layout_places_both_hadamard_vectors_after_the_codes():
    # The native reader computes the same numbers for this shape
    # (tests/artifact/test_reader.cpp, "EXL3 K=4 plane layout").
    geometry = exl3_tile_geometry("exl3_k4_mul1", (256, 128))
    assert (
        geometry.code_row_bytes,
        geometry.code_plane_bytes,
        geometry.suh_offset,
        geometry.suh_bytes,
        geometry.svh_offset,
        geometry.svh_bytes,
        geometry.payload_bytes,
    ) == (64, 16384, 16384, 256, 16640, 512, 17152)
    assert encoded_size("exl3_tile_v1", "exl3_k4_mul1", (256, 128)) == 17152

    narrow = exl3_tile_geometry("exl3_k3_mul1", (256, 128))
    assert (narrow.code_row_bytes, narrow.code_plane_bytes) == (48, 12288)
    assert narrow.suh_bytes == geometry.suh_bytes
    assert narrow.svh_bytes == geometry.svh_bytes


def test_exl3_tile_layout_rejects_untileable_shapes_and_other_formats():
    with pytest.raises(ValueError, match="divisible by 128"):
        exl3_tile_geometry("exl3_k4_mul1", (256, 64))
    with pytest.raises(ValueError, match="divisible by 128"):
        exl3_tile_geometry("exl3_k4_mul1", (64, 128))
    with pytest.raises(ValueError, match="trellis format"):
        exl3_tile_geometry("nvfp4", (256, 128))
    with pytest.raises(ValueError, match="does not accept format"):
        encoded_size("exl3_tile_v1", "nvfp4", (256, 128))
    with pytest.raises(ValueError, match="does not accept format"):
        encoded_size("block_scale_k16_m128x4_v1", "exl3_k4_mul1", (128, 64))


def test_trellis_formats_select_the_trellis_layout():
    assert default_layout("exl3_k3_mul1") == "exl3_tile_v1"
    assert default_layout("exl3_k4_mul1") == "exl3_tile_v1"
