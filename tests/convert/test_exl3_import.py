from __future__ import annotations

import json

import pytest
from safetensors.torch import save_file
import torch

from tools.artifact.codecs.exl3 import decode_exl3_tile_words, encode_exl3_tile
from tools.artifact.reader import Artifact
from tools.artifact.schema import TensorSpec
from tools.artifact.tensor_output import TensorOutput
from tools.artifact.writer import ArtifactWriter
from tools.convert.methods import PrepareRequest, MethodInput, import_exl3
from tools.convert.sources.exl3 import MUL1_MULTIPLIER, exl3_linear_source
from tools.convert.sources.safetensors import SafetensorsSource

BITS = 4


def _checkpoint(path, shapes, *, marker="mul1", seed=7):
    """Write a checkpoint in the source's own convention: [K,N] with a [K/16, N/16] tile grid."""

    generator = torch.Generator().manual_seed(seed)
    tensors = {}
    for prefix, (n, k) in shapes.items():
        tensors[f"{prefix}.trellis"] = torch.randint(
            -32768,
            32768,
            (k // 16, n // 16, 16 * BITS),
            generator=generator,
            dtype=torch.int16,
        )
        tensors[f"{prefix}.suh"] = torch.randn(k, generator=generator).to(torch.float16)
        tensors[f"{prefix}.svh"] = torch.randn(n, generator=generator).to(torch.float16)
        if marker is not None:
            tensors[f"{prefix}.{marker}"] = torch.tensor(
                [MUL1_MULTIPLIER], dtype=torch.int64
            ).to(torch.int32)
    shard = "model.safetensors"
    save_file(tensors, path / shard)
    (path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {name: shard for name in tensors}}), encoding="utf-8"
    )
    (path / "config.json").write_text("{}")
    return tensors


def _prepare(target, sources, rows_per_chunk=512):
    return PrepareRequest(
        target=target,
        inputs=tuple(
            MethodInput(name, source, ()) for name, source in sources.items()
        ),
        policies={},
        parameters={},
        rows_per_chunk=rows_per_chunk,
    )


def _write(path, spec, prepared):
    with ArtifactWriter(
        path, [spec], components={"text": {"config": {}}}, bindings={}
    ) as writer:
        prepared.produce(TensorOutput(writer, spec.id))
    return Artifact(path).read_object(spec.id)


def test_import_transposes_the_tile_grid_and_keeps_every_word(tmp_path):
    shape = (256, 128)
    tensors = _checkpoint(tmp_path, {"proj": shape})
    with SafetensorsSource(tmp_path) as store:
        source = exl3_linear_source(store, "proj")
        assert source.shape == shape
        spec = TensorSpec("w", shape, f"exl3_k{BITS}_mul1", "exl3_tile_v1")
        payload = _write(
            tmp_path / "out.ninfer",
            spec,
            import_exl3(_prepare(spec, {"proj": source}, rows_per_chunk=64)),
        )

    tiles, suh, svh = decode_exl3_tile_words(payload, shape, f"exl3_k{BITS}_mul1")
    assert torch.equal(tiles, tensors["proj.trellis"].permute(1, 0, 2))
    assert torch.equal(suh.view(torch.int16), tensors["proj.suh"].view(torch.int16))
    assert torch.equal(svh.view(torch.int16), tensors["proj.svh"].view(torch.int16))


def test_adjacent_projections_cannot_share_a_trellis_parent(tmp_path):
    # suh spans the whole K axis, so two projections can occupy one parent only if their input
    # vectors are identical word for word. Quantization derives suh per tensor, so in practice
    # they are not, and a trellis import cannot fuse q/k/v the way an NVFP4 import does.
    parts = {"q": (256, 128), "k": (128, 128)}
    _checkpoint(tmp_path, parts, seed=11)
    with SafetensorsSource(tmp_path) as store:
        sources = {name: exl3_linear_source(store, name) for name in parts}
        assert not torch.equal(
            sources["q"].read_trellis(0, 16).suh.view(torch.int16),
            sources["k"].read_trellis(0, 16).suh.view(torch.int16),
        )
        spec = TensorSpec("w", (384, 128), f"exl3_k{BITS}_mul1", "exl3_tile_v1")
        with pytest.raises(ValueError, match="suh changed"):
            _write(
                tmp_path / "packed.ninfer",
                spec,
                import_exl3(_prepare(spec, sources)),
            )


def test_import_refuses_a_checkpoint_it_cannot_name(tmp_path):
    _checkpoint(tmp_path, {"proj": (256, 128)}, marker=None)
    with SafetensorsSource(tmp_path) as store:
        with pytest.raises(ValueError, match="3inst"):
            exl3_linear_source(store, "proj")

    other = tmp_path / "mcg"
    other.mkdir()
    _checkpoint(other, {"proj": (256, 128)}, marker="mcg")
    with SafetensorsSource(other) as store:
        with pytest.raises(ValueError, match="mcg codebook"):
            exl3_linear_source(store, "proj")

    narrow = tmp_path / "narrow"
    narrow.mkdir()
    _checkpoint(narrow, {"proj": (256, 64)})
    with SafetensorsSource(narrow) as store:
        with pytest.raises(ValueError, match="rotatable"):
            exl3_linear_source(store, "proj")


def test_imported_words_reconstruct_the_same_matrix_as_the_source(tmp_path):
    shape = (128, 128)
    tensors = _checkpoint(tmp_path, {"proj": shape}, seed=13)
    with SafetensorsSource(tmp_path) as store:
        source = exl3_linear_source(store, "proj")
        values = source.values().reshape(shape)

    # The source decodes through the same words the import writes, in NInfer's orientation.
    payload = encode_exl3_tile(
        tensors["proj.trellis"].permute(1, 0, 2),
        tensors["proj.suh"],
        tensors["proj.svh"],
        shape,
        f"exl3_k{BITS}_mul1",
    )
    from tools.artifact.codecs.exl3 import dequantize_exl3_tile

    assert torch.equal(values, dequantize_exl3_tile(payload, shape, f"exl3_k{BITS}_mul1"))
