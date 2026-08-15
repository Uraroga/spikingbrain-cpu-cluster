import pytest
import torch

from spikingbrain_cpu.block import GLACache, KVCache
from spikingbrain_cpu.model_partition import ArgoStage, AtlasStage


def test_atlas_stage_is_meta_and_shares_exactly_one_quant_buffer():
    stage = AtlasStage()
    assert stage.quant_buffer_identity_count() == 1
    assert all(parameter.device.type == "meta" for parameter in stage.parameters())
    assert list(stage.layers) == [str(index) for index in range(14)]


def test_atlas_stage_builds_independent_global_layer_caches():
    caches = AtlasStage().make_decode_caches(position=3)
    assert set(caches) == set(range(14))
    assert all(cache.next_position == 3 for cache in caches.values())
    assert len({id(cache) for cache in caches.values()}) == 14


def test_atlas_stage_rejects_use_before_loading():
    stage = AtlasStage()
    with pytest.raises(RuntimeError, match="embedding is not loaded"):
        stage.embed(torch.tensor([[1]]))
    with pytest.raises(RuntimeError, match="must be loaded"):
        stage.forward_layers(torch.zeros(1, 1, 3584))


def test_argo_stage_is_meta_and_shares_exactly_one_quant_buffer():
    stage = ArgoStage()
    assert stage.quant_buffer_identity_count() == 1
    assert stage.quant_buffer.tensor is None
    assert all(parameter.device.type == "meta" for parameter in stage.parameters())
    assert list(stage.layers) == [str(index) for index in range(14, 28)]


def test_argo_stage_builds_independent_global_layer_caches():
    stage = ArgoStage()
    caches = stage.make_decode_caches(position=3)
    assert set(caches) == set(range(14, 28))
    for index, cache in caches.items():
        assert isinstance(cache, GLACache if index % 2 == 0 else KVCache)
        assert cache.next_position == 3
    assert len({cache.state.data_ptr() for index, cache in caches.items() if index % 2 == 0}) == 7
    assert len({cache.key.data_ptr() for index, cache in caches.items() if index % 2 == 1}) == 7


def test_argo_stage_rejects_forward_before_loading():
    stage = ArgoStage()
    with pytest.raises(RuntimeError, match="must be loaded"):
        stage.forward_layers(torch.zeros(1, 1, 3584))
