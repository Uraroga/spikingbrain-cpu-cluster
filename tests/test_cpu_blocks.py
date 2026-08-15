import pytest
import torch

from spikingbrain_cpu.block import (
    GLABlock,
    GLACache,
    KVCache,
    QuantBuffer,
    SlidingWindowAttentionBlock,
)


def make_gla() -> GLABlock:
    torch.manual_seed(31)
    return GLABlock(
        hidden_size=16,
        intermediate_size=32,
        num_heads=4,
        num_key_value_heads=2,
        group_size=4,
    ).eval()


def make_attention() -> SlidingWindowAttentionBlock:
    torch.manual_seed(32)
    return SlidingWindowAttentionBlock(
        hidden_size=16,
        intermediate_size=32,
        num_heads=4,
        num_key_value_heads=2,
        sliding_window=16,
        group_size=4,
    ).eval()


@pytest.mark.parametrize("factory,cache_type", [(make_gla, GLACache), (make_attention, KVCache)])
def test_complete_block_shape_dtype_finite_and_cache(factory, cache_type):
    block = factory()
    x = torch.randn(2, 5, 16)
    with torch.inference_mode():
        output, cache = block(x)
    assert output.shape == x.shape
    assert output.dtype == x.dtype
    assert isinstance(cache, cache_type)
    assert cache.next_position == 5
    assert cache.nbytes > 0
    assert torch.isfinite(output).all()


@pytest.mark.parametrize("factory", [make_gla, make_attention])
def test_complete_block_is_deterministic_without_cache(factory):
    block = factory()
    x = torch.randn(1, 4, 16)
    with torch.inference_mode():
        first, _ = block(x)
        second, _ = block(x)
    assert torch.equal(first, second)


@pytest.mark.parametrize("factory", [make_gla, make_attention])
def test_progressive_decode_matches_single_prefill(factory):
    block = factory()
    x = torch.randn(1, 6, 16)
    with torch.inference_mode():
        expected, _ = block(x)
        cache = None
        pieces = []
        for token in range(x.shape[1]):
            output, cache = block(x[:, token : token + 1], cache)
            pieces.append(output)
    actual = torch.cat(pieces, dim=1)
    assert torch.allclose(actual, expected, rtol=1e-5, atol=1e-5)
    assert cache is not None and cache.next_position == x.shape[1]


@pytest.mark.parametrize("factory", [make_gla, make_attention])
def test_zero_projections_leave_residual_unchanged(factory):
    block = factory()
    with torch.no_grad():
        for module in block.modules():
            if hasattr(module, "weight") and getattr(module, "weight") is not None:
                module.weight.zero_()
            if hasattr(module, "bias") and getattr(module, "bias") is not None:
                module.bias.zero_()
    x = torch.randn(1, 3, 16)
    with torch.inference_mode():
        output, _ = block(x)
    assert torch.equal(output, x)


def test_attention_cache_respects_sliding_window_capacity():
    torch.manual_seed(33)
    block = SlidingWindowAttentionBlock(
        hidden_size=16,
        intermediate_size=32,
        num_heads=4,
        num_key_value_heads=2,
        sliding_window=2,
        group_size=4,
    ).eval()
    cache = None
    with torch.inference_mode():
        for _ in range(7):
            _, cache = block(torch.randn(1, 1, 16), cache)
    assert cache is not None
    assert cache.key.shape[2] == 3  # two previous positions plus current
    assert cache.value.shape == cache.key.shape
    assert cache.next_position == 7


def test_one_shared_quant_buffer_grows_to_largest_projection():
    buffer = QuantBuffer()
    block = GLABlock(
        hidden_size=16,
        intermediate_size=32,
        num_heads=4,
        num_key_value_heads=2,
        group_size=4,
        buffer=buffer,
    ).eval()
    with torch.inference_mode():
        block(torch.randn(1, 1, 16))
    assert buffer.tensor is not None
    assert buffer.tensor.numel() == 16 * 32
