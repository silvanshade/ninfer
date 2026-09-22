from __future__ import annotations

from dataclasses import replace
import json
from functools import partial

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
from tools.convert.sources.compressed_tensors import matrix_source
from tools.convert.sources.logical import select_rows
from tools.convert.sources.safetensors import SafetensorsSource
from tools.convert.model import Model, Parameter
from tools.convert.recipe import Recipe
from tools.convert.official_recipes import _exl3_dense

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


def test_a_trellis_linear_resolves_from_its_planes_and_slices_at_tile_bands(tmp_path):
    # An EXL3 linear has no weight tensor, so resolution keys on the planes while the recipe keeps
    # naming the parameter as it does everywhere else.
    shape = (256, 128)
    tensors = _checkpoint(tmp_path, {"model.layers.0.self_attn.q_proj": shape}, seed=17)
    with SafetensorsSource(tmp_path) as store:
        source = matrix_source(store, "model.layers.0.self_attn.q_proj.weight", shape)
        band = select_rows(source, ((128, 256),))
        assert band.shape == (128, 128)
        rows = band.read_trellis(0, 128)
        grid = tensors["model.layers.0.self_attn.q_proj.trellis"].permute(1, 0, 2)
        assert torch.equal(rows.tiles, grid[8:])
        assert torch.equal(
            rows.svh.view(torch.int16),
            tensors["model.layers.0.self_attn.q_proj.svh"][128:].view(torch.int16),
        )
        # A band keeps its parent's whole input vector, because suh spans K.
        assert torch.equal(
            rows.suh.view(torch.int16),
            tensors["model.layers.0.self_attn.q_proj.suh"].view(torch.int16),
        )
        with pytest.raises(ValueError, match="tile bands"):
            select_rows(source, ((8, 24),)).read_trellis(0, 16)


def test_packing_keeps_separate_projections_separate(tmp_path):
    # Each projection brought its own suh, so one parent cannot hold them and the recipe writes
    # the linears the checkpoint ships rather than fusing anything.
    names = ("query", "key")
    _checkpoint(tmp_path, {name: (128, 128) for name in names})
    store = SafetensorsSource(tmp_path)
    with store:
        model = Model({"text": {"config": {}}})
        for name in names:
            model.add(
                Parameter(
                    name, (128, 128), exl3_linear_source(store, name), inputs=("input",)
                )
            )
        model.packing_groups = [tuple(names)]
        recipe = Recipe(model)
        recipe.assign(names, format=f"exl3_k{BITS}_mul1", method=import_exl3)
        prepared = recipe.prepare(device="cpu")

    assert len(prepared.weights) == 2
    assert all(job.spec.shape == (128, 128) for job in prepared.weights)
    assert prepared.bindings["query"] != prepared.bindings["key"]


def test_two_parents_with_one_suh_between_them_still_stay_apart(tmp_path):
    # A quantizer derives suh from the activation a linear reads, and q/k/v read the same one, so
    # a checkpoint can ship three parents whose suh words agree exactly. Equal words are not one
    # parent: fusing them would be a conversion-time decision the checkpoint never made, which is
    # the thing the owner ruled against, so the packer keys on the source and not on the vector.
    names = ("query", "key")
    _checkpoint(tmp_path, {name: (128, 128) for name in names})
    shared = torch.randn(128, generator=torch.Generator().manual_seed(3)).to(torch.float16)
    store = SafetensorsSource(tmp_path)
    with store:
        sources = {}
        for name in names:
            source = exl3_linear_source(store, name)
            rows = source.read_trellis
            sources[name] = replace(
                source,
                read_trellis=lambda begin, end, rows=rows: replace(
                    rows(begin, end), suh=shared
                ),
            )
        model = Model({"text": {"config": {}}})
        for name in names:
            model.add(Parameter(name, (128, 128), sources[name], inputs=("input",)))
        model.packing_groups = [tuple(names)]
        recipe = Recipe(model)
        recipe.assign(names, format=f"exl3_k{BITS}_mul1", method=import_exl3)
        prepared = recipe.prepare(device="cpu")

    assert len(prepared.weights) == 2
    assert prepared.bindings["query"] != prepared.bindings["key"]


def test_packing_joins_projections_sliced_from_one_source(tmp_path):
    shape = (256, 128)
    _checkpoint(tmp_path, {"in_proj_qkv": shape})
    store = SafetensorsSource(tmp_path)
    with store:
        source = exl3_linear_source(store, "in_proj_qkv")
        model = Model({"text": {"config": {}}})
        for name, rows in (("query", (0, 128)), ("value", (128, 256))):
            model.add(
                Parameter(
                    name, (128, 128), select_rows(source, (rows,)), inputs=("input",)
                )
            )
        model.packing_groups = [("query", "value")]
        recipe = Recipe(model)
        recipe.assign(
            ("query", "value"), format=f"exl3_k{BITS}_mul1", method=import_exl3
        )
        prepared = recipe.prepare(device="cpu")

    # Both slices carry their parent's suh, so the GDN projections still share one parent.
    assert len(prepared.weights) == 1
    assert prepared.weights[0].spec.shape == shape


def test_the_recipe_takes_what_the_checkpoint_quantized_and_nothing_else(tmp_path):
    # A checkpoint that quantized the attention projections but left the GDN gate alone: the
    # recipe imports the first as trellises at the checkpoint's own rate and falls back to the
    # model's groupwise representation for the second, without requantizing either.
    trellis = {
        "model.layers.0.self_attn.q_proj": (256, 128),
        "model.layers.0.self_attn.k_proj": (128, 128),
    }
    tensors = _checkpoint(tmp_path, trellis)
    tensors["model.layers.0.linear_attn.in_proj_a.weight"] = torch.zeros(
        128, 128, dtype=torch.float16
    )
    save_file(tensors, tmp_path / "model.safetensors")
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {n: "model.safetensors" for n in tensors}}),
        encoding="utf-8",
    )

    store = SafetensorsSource(tmp_path)
    with store:
        model = Model({"text": {"config": {}}})
        for name, prefix, shape in (
            ("text/layers/0/attention/query", "model.layers.0.self_attn.q_proj", (256, 128)),
            ("text/layers/0/attention/key", "model.layers.0.self_attn.k_proj", (128, 128)),
            ("text/layers/0/gdn/gate", "model.layers.0.linear_attn.in_proj_a", (128, 128)),
        ):
            factory = partial(matrix_source, name=f"{prefix}.weight", shape=shape)
            model.add(
                Parameter(
                    name,
                    shape,
                    factory(store),
                    source_factory=lambda s, f, factory=factory: factory(s),
                    inputs=("input",),
                )
            )
        recipe = Recipe(model)
        _exl3_dense(model, recipe, {"quantized": store}, "q8_g32_fp16")
        prepared = recipe.prepare(device="cpu")
        formats = {job.spec.id: job.spec.format for job in prepared.weights}

    assert set(formats.values()) == {f"exl3_k{BITS}_mul1", "q5_g64_fp16"}
    assert len(formats) == 3
