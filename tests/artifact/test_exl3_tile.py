from __future__ import annotations

import pytest
import torch

from tools.artifact.codecs.exl3 import (
    decode_exl3_codes,
    decode_exl3_tile_words,
    dequantize_exl3_tile,
    encode_exl3_tile,
    hadamard_matrix,
    mul1_codebook,
    trellis_states,
)
from tools.artifact.layouts import (
    encoded_size,
    exl3_tile_geometry,
)
from tools.convert.recipe import default_layout


def _random_trellis(shape, bits, seed=20260921):
    generator = torch.Generator().manual_seed(seed)
    n, k = shape
    words = torch.randint(
        -32768,
        32768,
        (n // 16, k // 16, 16 * bits),
        generator=generator,
        dtype=torch.int16,
    )
    suh = torch.randn(k, generator=generator).to(torch.float16)
    svh = torch.randn(n, generator=generator).to(torch.float16)
    return words, suh, svh


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


def test_trellis_ring_carries_state_between_consecutive_weights():
    shape, bits = (128, 128), 4
    words, _, _ = _random_trellis(shape, bits)
    states = trellis_states(words, bits)
    assert states.shape == (64, 256)
    assert int(states.min()) >= 0 and int(states.max()) <= 0xFFFF
    # Element t's window ends where element t+1's begins, so t+1 inherits all but t's own step,
    # around the whole tail-biting ring. A wrong bit order or a missed word swap breaks this.
    following = states.roll(-1, dims=1)
    assert torch.equal(following >> bits, states & ((1 << (16 - bits)) - 1))


def test_mul1_codebook_decodes_the_reference_words():
    codebook = mul1_codebook()
    assert codebook.dtype == torch.float16 and codebook.shape == (65536,)
    probes = {0x0000: -3.453125, 0x0001: 0.64111328125, 0x1234: -0.5771484375}
    for word, value in probes.items():
        assert codebook[word].item() == value


def test_trellis_payload_round_trips_and_reconstructs_through_both_rotations():
    shape, bits = (128, 256), 3
    words, suh, svh = _random_trellis(shape, bits)
    payload = encode_exl3_tile(words, suh, svh, shape, "exl3_k3_mul1")
    assert len(payload) == encoded_size("exl3_tile_v1", "exl3_k3_mul1", shape)

    stored, stored_suh, stored_svh = decode_exl3_tile_words(
        payload, shape, "exl3_k3_mul1"
    )
    assert torch.equal(stored, words)
    assert torch.equal(stored_suh.view(torch.int16), suh.view(torch.int16))
    assert torch.equal(stored_svh.view(torch.int16), svh.view(torch.int16))

    geometry = exl3_tile_geometry("exl3_k3_mul1", shape)
    assert payload[geometry.code_plane_bytes : geometry.suh_offset] == bytes(
        geometry.suh_offset - geometry.code_plane_bytes
    )
    assert payload[geometry.suh_offset + geometry.suh_bytes : geometry.svh_offset] == (
        bytes(geometry.svh_offset - geometry.suh_offset - geometry.suh_bytes)
    )

    # The reconstruction rotates K in 128-blocks, scales by suh, rotates N in 128-blocks, then
    # scales by svh. Comparing against block-diagonal matrices catches a swapped axis or a
    # rotation applied to the wrong side.
    decoded = decode_exl3_codes(stored, shape, "exl3_k3_mul1").float()
    hadamard = hadamard_matrix()
    rotate_k = torch.block_diag(*([hadamard] * (shape[1] // 128)))
    rotate_n = torch.block_diag(*([hadamard] * (shape[0] // 128)))
    expected = rotate_n @ ((decoded @ rotate_k) * suh.float()) * svh.float().unsqueeze(1)
    assert torch.allclose(
        dequantize_exl3_tile(payload, shape, "exl3_k3_mul1"), expected, atol=1e-4
    )


def test_hadamard_matrix_is_orthogonal_and_symmetric():
    hadamard = hadamard_matrix()
    assert hadamard.shape == (128, 128)
    assert torch.equal(hadamard, hadamard.T)
    assert torch.allclose(hadamard @ hadamard, torch.eye(128), atol=1e-6)
    # Sylvester entries are the parity of the bitwise-and of their indices.
    index = torch.arange(128)
    parity = torch.tensor(
        [[bin(int(i) & int(j)).count("1") & 1 for j in index] for i in index]
    )
    assert torch.equal(hadamard.sign(), torch.where(parity == 1, -1.0, 1.0))
