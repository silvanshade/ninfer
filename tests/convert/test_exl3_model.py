"""Convert a whole (small) EXL3 model, end to end, through the real builder and writer."""

from __future__ import annotations

import json

from safetensors.torch import save_file
import torch

from tools.artifact.reader import Artifact
from tools.convert.official_recipes import qwen3_8_27b_exl3
from tools.convert.pipeline import convert
from tools.convert.qwen3_5 import build_model
from tools.convert.recipe import Recipe
from tools.convert.sources.exl3 import MUL1_MULTIPLIER
from tools.convert.sources.safetensors import SafetensorsSource

BITS = 4
H = 128
HEADS, KV_HEADS, HEAD_DIM = 2, 1, 128
INTERMEDIATE = 128
KEY_GROUP = VALUE_GROUP = 128
VOCAB = 128


def _config():
    return {
        "architectures": ["Qwen3_5ForConditionalGeneration"],
        "text_config": {
            "hidden_size": H,
            "vocab_size": VOCAB,
            "num_hidden_layers": 2,
            "max_position_embeddings": 256,
            "layer_types": ["linear_attention", "full_attention"],
            "num_attention_heads": HEADS,
            "num_key_value_heads": KV_HEADS,
            "head_dim": HEAD_DIM,
            "intermediate_size": INTERMEDIATE,
            "rope_parameters": {
                "partial_rotary_factor": 0.5,
                "mrope_section": [21, 11, 0],
            },
            "linear_num_key_heads": 1,
            "linear_key_head_dim": KEY_GROUP,
            "linear_num_value_heads": 1,
            "linear_value_head_dim": VALUE_GROUP,
            "linear_conv_kernel_dim": 3,
        },
    }


def _trellis(tensors, prefix, n, k, generator):
    tensors[f"{prefix}.trellis"] = torch.randint(
        -32768,
        32768,
        (k // 16, n // 16, 16 * BITS),
        generator=generator,
        dtype=torch.int16,
    )
    tensors[f"{prefix}.suh"] = torch.randn(k, generator=generator).to(torch.float16)
    tensors[f"{prefix}.svh"] = torch.randn(n, generator=generator).to(torch.float16)
    tensors[f"{prefix}.mul1"] = torch.tensor([MUL1_MULTIPLIER], dtype=torch.int64).to(
        torch.int32
    )


def _checkpoint(path):
    """An EXL3 quantization of the small model: trellises for the linears, plain rest."""

    generator = torch.Generator().manual_seed(99)
    channels = 2 * KEY_GROUP + VALUE_GROUP
    query = HEADS * HEAD_DIM
    tensors: dict[str, torch.Tensor] = {}

    def plain(name, *shape, dtype=torch.float16):
        tensors[name] = torch.randn(*shape, generator=generator).to(dtype)

    pre = "model.language_model."
    plain(pre + "embed_tokens.weight", VOCAB, H)
    plain(pre + "norm.weight", H)
    _trellis(tensors, "lm_head", VOCAB, H, generator)
    for index, kind in enumerate(_config()["text_config"]["layer_types"]):
        layer = f"{pre}layers.{index}."
        plain(layer + "input_layernorm.weight", H)
        plain(layer + "post_attention_layernorm.weight", H)
        if kind == "full_attention":
            _trellis(tensors, layer + "self_attn.q_proj", 2 * query, H, generator)
            _trellis(
                tensors, layer + "self_attn.k_proj", KV_HEADS * HEAD_DIM, H, generator
            )
            _trellis(
                tensors, layer + "self_attn.v_proj", KV_HEADS * HEAD_DIM, H, generator
            )
            _trellis(tensors, layer + "self_attn.o_proj", H, query, generator)
            for role in ("q_norm", "k_norm"):
                plain(layer + "self_attn." + role + ".weight", HEAD_DIM)
        else:
            _trellis(tensors, layer + "linear_attn.in_proj_qkv", channels, H, generator)
            _trellis(
                tensors, layer + "linear_attn.in_proj_z", VALUE_GROUP, H, generator
            )
            _trellis(tensors, layer + "linear_attn.out_proj", H, VALUE_GROUP, generator)
            # The quantizer leaves these alone, so the recipe must fall back for them.
            plain(layer + "linear_attn.in_proj_a.weight", 1, H)
            plain(layer + "linear_attn.in_proj_b.weight", 1, H)
            plain(layer + "linear_attn.conv1d.weight", channels, 1, 3)
            plain(layer + "linear_attn.norm.weight", VALUE_GROUP)
            plain(layer + "linear_attn.A_log", 1, dtype=torch.float32)
            plain(layer + "linear_attn.dt_bias", 1, dtype=torch.float32)
        _trellis(tensors, layer + "mlp.gate_proj", INTERMEDIATE, H, generator)
        _trellis(tensors, layer + "mlp.up_proj", INTERMEDIATE, H, generator)
        _trellis(tensors, layer + "mlp.down_proj", H, INTERMEDIATE, generator)

    path.mkdir(parents=True, exist_ok=True)
    config = _config()
    config["tie_word_embeddings"] = False
    (path / "config.json").write_text(json.dumps(config))
    save_file(tensors, path / "model.safetensors")
    (path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {name: "model.safetensors" for name in tensors}})
    )
    for role in ("tokenizer.json", "tokenizer_config.json", "generation_config.json"):
        (path / role).write_text(
            json.dumps({"model": {"vocab": {str(i): i for i in range(8)}}})
            if role == "tokenizer.json"
            else "{}"
        )
    (path / "chat_template.jinja").write_text("{{ messages }}")
    return tensors


def test_an_exl3_quantization_converts_end_to_end(tmp_path):
    source = tmp_path / "source"
    _checkpoint(source)
    path = tmp_path / "small.ninfer"
    with SafetensorsSource(source) as store:
        model = build_model(store)
        recipe = Recipe(model)
        qwen3_8_27b_exl3(model, recipe, {"quantized": store})
        report = convert(model, recipe, path, device="cpu", rows_per_chunk=64)

    with Artifact(path) as artifact:
        directory = artifact.directory
    by_id = {obj.id: obj for obj in directory.objects}

    def parents(name):
        binding = directory.bindings[name]
        return [part["object"] for part in binding.get("parts", [binding])]

    def fmt(name):
        return {by_id[object_id].format for object_id in parents(name)}

    # Every projection the quantizer took arrives as a trellis, at the rate its planes declare.
    assert fmt("text/layers/1/attention/query") == {f"exl3_k{BITS}_mul1"}
    assert fmt("text/layers/0/mlp/down") == {f"exl3_k{BITS}_mul1"}
    assert fmt("text/output_head") == {f"exl3_k{BITS}_mul1"}
    # The ones it left alone fall back rather than being requantized into a trellis.
    assert all(
        not value.startswith("exl3_") for value in fmt("text/layers/0/gdn/a_projection")
    )
    assert report["formats"][f"exl3_k{BITS}_mul1"] >= 8

    # Owner ruling: attention consumes the three shipped linears as they are, unfused.
    attention = [
        parents(f"text/layers/1/attention/{role}") for role in ("query", "key", "value")
    ]
    assert all(len(part) == 1 for part in attention)
    assert len({part[0] for part in attention}) == 3

    # The GDN pack ships fused, so its roles are row spans of that one trellis, never requantized.
    gdn = [f"text/layers/0/gdn/{role}" for role in ("query", "key", "value")]
    assert all(fmt(name) == {f"exl3_k{BITS}_mul1"} for name in gdn)
    sources = {
        entry["object"]: entry["sources"]
        for entry in report["methods"]
        if entry["method"] == "import_exl3"
    }
    assert all(
        "in_proj_qkv" in source
        for name in gdn
        for object_id in parents(name)
        for source in sources[object_id]
    )
