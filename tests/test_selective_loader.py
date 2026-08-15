import pytest
import json
import struct
import torch

from spikingbrain_cpu.block import GLABlock
from spikingbrain_cpu.selective_loader import (
    IndexPlanner,
    LoaderLimitError,
    RealTensorLoader,
    assign_layer_tensor,
)


CONFIG = {
    "hidden_size": 128,
    "intermediate_size": 256,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "num_hidden_layers": 2,
    "vocab_size": 100,
}


def write_f32_safetensor(path, name, values):
    raw = struct.pack(f"<{len(values)}f", *values)
    header = json.dumps(
        {name: {"dtype": "F32", "shape": [len(values)], "data_offsets": [0, len(raw)]}},
        separators=(",", ":"),
    ).encode()
    header += b" " * ((8 - len(header) % 8) % 8)
    path.write_bytes(struct.pack("<Q", len(header)) + header + raw)


def test_index_planner_infers_shapes_and_split_without_shards():
    weight_map = {
        "model.embeddings.weight": "a.safetensors",
        "model.layers.0.attn.q_proj.weight": "a.safetensors",
        "model.layers.0.attn.q_proj.weight_quantizer.scales": "a.safetensors",
        "model.layers.1.mlp.down_proj.weight": "b.safetensors",
        "model.layers.1.mlp.down_proj.weight_quantizer.scales": "b.safetensors",
        "model.norm.weight": "b.safetensors",
        "lm_head.weight": "b.safetensors",
    }
    planner = IndexPlanner(CONFIG, {"weight_map": weight_map})
    tensors = {tensor.name: tensor for tensor in planner.all_tensors()}
    assert tensors["model.embeddings.weight"].shape == (100, 128)
    assert tensors["model.layers.0.attn.q_proj.weight"].shape == (128, 128)
    assert tensors["model.layers.0.attn.q_proj.weight_quantizer.scales"].shape == (128, 1, 1)
    assert tensors["model.layers.1.mlp.down_proj.weight"].shape == (128, 256)
    atlas, argo = planner.plan_split(1)
    assert {tensor.name for tensor in atlas.tensors} == {
        "model.embeddings.weight",
        "model.layers.0.attn.q_proj.weight",
        "model.layers.0.attn.q_proj.weight_quantizer.scales",
    }
    assert len(argo.tensors) == 4


def test_index_planner_rejects_total_size_mismatch():
    planner = IndexPlanner(
        CONFIG,
        {
            "metadata": {"total_size": 1},
            "weight_map": {"model.norm.weight": "one.safetensors"},
        },
    )
    with pytest.raises(ValueError, match="index declares"):
        planner.all_tensors()


def test_index_planner_rejects_unknown_tensor():
    planner = IndexPlanner(CONFIG, {"weight_map": {"unknown.weight": "x"}})
    with pytest.raises(ValueError, match="unsupported tensor"):
        planner.all_tensors()


def test_real_loader_reads_only_selected_synthetic_tensor(tmp_path):
    tensor = torch.arange(128, dtype=torch.float32)
    write_f32_safetensor(
        tmp_path / "one.safetensors", "model.norm.weight", list(range(128))
    )
    index = {
        "metadata": {"total_size": tensor.numel() * tensor.element_size()},
        "weight_map": {"model.norm.weight": "one.safetensors"},
    }
    planner = IndexPlanner(CONFIG, index)
    loader = RealTensorLoader(planner, tmp_path, tensor.numel() * 4, 2048)
    loaded = list(loader.iter_materialized(("model.norm.weight",)))
    assert len(loaded) == 1
    assert torch.equal(loaded[0][1], tensor)
    assert loader.opened_shards == ["one.safetensors"]
    assert loader.materialized_bytes == tensor.numel() * 4


def test_real_loader_enforces_byte_limit_before_opening_shard(tmp_path):
    tensor = torch.zeros(128)
    write_f32_safetensor(
        tmp_path / "one.safetensors", "model.norm.weight", [0.0] * 128
    )
    planner = IndexPlanner(
        CONFIG,
        {
            "metadata": {"total_size": 512},
            "weight_map": {"model.norm.weight": "one.safetensors"},
        },
    )
    loader = RealTensorLoader(planner, tmp_path, max_bytes=511, max_rss_mib=2048)
    with pytest.raises(LoaderLimitError, match="byte limit exceeded"):
        list(loader.iter_materialized(("model.norm.weight",)))
    assert loader.opened_shards == []


def test_assign_layer_tensor_replaces_meta_parameter_without_copy():
    block = GLABlock(
        hidden_size=128,
        intermediate_size=256,
        num_heads=4,
        num_key_value_heads=2,
        group_size=128,
        device="meta",
    )
    tensor = torch.randn(128, 128)
    assign_layer_tensor(block, 14, "model.layers.14.attn.q_proj.weight", tensor)
    assert block.q_proj.weight.device.type == "cpu"
    assert block.q_proj.weight.data_ptr() == tensor.data_ptr()
