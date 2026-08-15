import math

import pytest
import torch
import torch.nn.functional as F

from spikingbrain_cpu.ops import (
    RMSNorm,
    dynamic_spikes,
    fake_quantize_weight,
    gla_recurrent,
    quant_linear,
    quantize_symmetric,
    sliding_window_attention,
    swiglu,
    swish,
)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_rmsnorm_matches_formula_and_preserves_shape_dtype(dtype):
    torch.manual_seed(1)
    x = torch.randn(2, 3, 8, dtype=dtype)
    norm = RMSNorm(8, eps=1e-6).to(dtype=dtype)
    norm.weight.data.copy_(torch.linspace(0.5, 1.5, 8, dtype=dtype))

    with torch.inference_mode():
        actual = norm(x)
        work = x.float() if dtype == torch.bfloat16 else x
        expected = work * torch.rsqrt(work.square().mean(-1, keepdim=True) + 1e-6)
        expected = expected.to(dtype) * norm.weight

    assert actual.shape == x.shape
    assert actual.dtype == dtype
    assert torch.allclose(actual, expected, rtol=2e-2 if dtype == torch.bfloat16 else 1e-6)
    assert torch.isfinite(actual).all()


def test_rmsnorm_prenorm_adds_and_returns_residual():
    norm = RMSNorm(3)
    x = torch.tensor([[[1.0, 2.0, 3.0]]])
    residual = torch.tensor([[[0.5, -0.5, 1.0]]])
    with torch.inference_mode():
        output, combined = norm(x, residual, prenorm=True)
        reference = norm(x + residual)
    assert torch.equal(combined, x + residual)
    assert torch.equal(output, reference)


def test_swish_and_swiglu_match_pytorch_formula():
    torch.manual_seed(2)
    gate = torch.randn(2, 3, 7)
    value = torch.randn(2, 3, 7)
    weight = torch.randn(5, 7)
    bias = torch.randn(5)

    with torch.inference_mode():
        assert torch.equal(swish(gate), F.silu(gate))
        actual = swiglu(gate, value, weight, bias)
        expected = F.linear(F.silu(gate) * value, weight, bias)

    assert actual.shape == (2, 3, 5)
    assert actual.dtype == gate.dtype
    assert torch.allclose(actual, expected)
    assert torch.isfinite(actual).all()


def test_swiglu_rejects_different_shapes():
    with pytest.raises(ValueError):
        swiglu(torch.zeros(2, 3), torch.zeros(2, 4))


def test_attention_is_causal():
    # Changing future keys/values cannot affect earlier output positions.
    q = torch.ones(1, 1, 4, 2)
    k = torch.ones(1, 1, 4, 2)
    v1 = torch.tensor([[[[1.0], [2.0], [3.0], [4.0]]]])
    v2 = v1.clone()
    v2[:, :, 3] = 999.0

    with torch.inference_mode():
        out1 = sliding_window_attention(q, k, v1, window_size=4)
        out2 = sliding_window_attention(q, k, v2, window_size=4)

    assert torch.equal(out1[:, :, :3], out2[:, :, :3])
    assert not torch.equal(out1[:, :, 3:], out2[:, :, 3:])


def test_attention_applies_sliding_window_exactly():
    # Equal scores make the result the average of the visible values.  A window
    # of one includes the current token and exactly one predecessor, matching
    # FlashAttention's (left, right) convention.
    q = torch.ones(1, 1, 4, 1)
    k = torch.ones(1, 1, 4, 1)
    v = torch.tensor([[[[1.0], [2.0], [4.0], [8.0]]]])

    with torch.inference_mode():
        actual = sliding_window_attention(q, k, v, window_size=1, scale=1.0)
    expected = torch.tensor([[[[1.0], [1.5], [3.0], [6.0]]]])

    assert actual.shape == v.shape
    assert actual.dtype == q.dtype
    assert torch.allclose(actual, expected)
    assert torch.isfinite(actual).all()


def test_attention_supports_grouped_query_heads_and_masked_rows():
    torch.manual_seed(3)
    q = torch.randn(2, 4, 3, 2)
    k = torch.randn(2, 2, 3, 2)
    v = torch.randn(2, 2, 3, 5)
    mask = torch.tensor([[1, 1, 1], [0, 0, 0]], dtype=torch.bool)

    with torch.inference_mode():
        output = sliding_window_attention(q, k, v, window_size=3, attention_mask=mask)

    assert output.shape == (2, 4, 3, 5)
    assert output.dtype == q.dtype
    assert torch.equal(output[1], torch.zeros_like(output[1]))
    assert torch.isfinite(output).all()


def test_attention_is_deterministic():
    torch.manual_seed(4)
    q = torch.randn(1, 2, 5, 4)
    k = torch.randn(1, 1, 5, 4)
    v = torch.randn(1, 1, 5, 3)
    with torch.inference_mode():
        first = sliding_window_attention(q, k, v, 3)
        second = sliding_window_attention(q, k, v, 3)
    assert torch.equal(first, second)


def test_gla_scalar_recurrence_has_readable_expected_values():
    # state_t = 0.5 * state_(t-1) + 1 gives [1, 1.5, 1.75].
    q = torch.ones(1, 1, 3, 1)
    k = torch.ones_like(q)
    v = torch.ones_like(q)
    log_gate = torch.full_like(q, math.log(0.5))

    with torch.inference_mode():
        output, state = gla_recurrent(q, k, v, log_gate, scale=1.0)

    assert output.shape == (1, 1, 3, 1)
    assert output.dtype == q.dtype
    assert torch.allclose(output.flatten(), torch.tensor([1.0, 1.5, 1.75]))
    assert torch.allclose(state.flatten(), torch.tensor([1.75]))
    assert torch.isfinite(output).all() and torch.isfinite(state).all()


def test_gla_cache_split_matches_single_pass():
    torch.manual_seed(5)
    q = torch.randn(1, 2, 6, 3)
    k = torch.relu(torch.randn_like(q))
    v = torch.randn(1, 2, 6, 4)
    log_gate = F.logsigmoid(torch.randn_like(q)) / 16

    with torch.inference_mode():
        complete, complete_state = gla_recurrent(q, k, v, log_gate)
        first, cached = gla_recurrent(q[:, :, :4], k[:, :, :4], v[:, :, :4], log_gate[:, :, :4])
        second, split_state = gla_recurrent(
            q[:, :, 4:], k[:, :, 4:], v[:, :, 4:], log_gate[:, :, 4:], initial_state=cached
        )

    assert torch.allclose(torch.cat((first, second), dim=2), complete, atol=1e-6)
    assert torch.allclose(split_state, complete_state, atol=1e-6)
    assert torch.isfinite(complete).all()


def test_gla_is_deterministic_and_preserves_bfloat16_output():
    torch.manual_seed(6)
    q = torch.randn(1, 2, 4, 3, dtype=torch.bfloat16)
    k = torch.randn_like(q)
    v = torch.randn(1, 2, 4, 2, dtype=torch.bfloat16)
    log_gate = F.logsigmoid(torch.randn_like(q).float()).to(torch.bfloat16) / 16
    with torch.inference_mode():
        first, state1 = gla_recurrent(q, k, v, log_gate)
        second, state2 = gla_recurrent(q, k, v, log_gate)
    assert first.dtype == torch.bfloat16 and state1.dtype == torch.bfloat16
    assert torch.equal(first, second) and torch.equal(state1, state2)
    assert torch.isfinite(first).all()


def test_fake_quantization_matches_checkpoint_formula_without_mutation():
    weight = torch.tensor([[0.2, 0.6, 1.4, -1.6], [2.1, -2.4, 3.7, -3.2]])
    original = weight.clone()
    scales = torch.tensor([[[0.5], [1.0]], [[1.0], [2.0]]])

    with torch.inference_mode():
        actual = fake_quantize_weight(weight, scales, group_size=2)
        expected = ((weight.reshape(2, 2, 2) / scales).round() * scales).reshape_as(weight)

    assert torch.equal(actual, expected)
    assert torch.equal(weight, original)
    assert actual.data_ptr() != weight.data_ptr()


def test_dynamic_spikes_matches_checkpoint_formula():
    x = torch.tensor([[[0.2, -0.7, 1.4, -2.1]]])
    with torch.inference_mode():
        spikes, threshold = dynamic_spikes(x, k=3.0)
    expected_threshold = (x.abs().mean(-1, keepdim=True).float() / 3.0).clamp(1e-5, 1e4)
    assert torch.equal(threshold, expected_threshold)
    assert torch.equal(spikes, (x / expected_threshold).round())


def test_symmetric_fake_quant_matches_checkpoint_formula():
    x = torch.tensor([[[0.1, -0.7, 1.4, -2.1]]])
    with torch.inference_mode():
        actual = quantize_symmetric(x, bits=8)
    scale = (x.abs().amax(-1, keepdim=True).float() / 127).clamp(1e-5, 1e4)
    expected = (x / scale).round().clamp(-128, 127) * scale
    assert torch.equal(actual, expected)
    assert actual.dtype == x.dtype


def test_quant_linear_reusable_buffer_matches_temporary_path():
    torch.manual_seed(7)
    x = torch.randn(2, 4)
    weight = torch.randn(3, 4)
    bias = torch.randn(3)
    scales = torch.rand(3, 2, 1) + 0.1
    buffer = torch.empty_like(weight)
    original = weight.clone()

    with torch.inference_mode():
        allocating = quant_linear(x, weight, scales, bias, group_size=2)
        reused = quant_linear(x, weight, scales, bias, group_size=2, out=buffer)
        spikes, threshold = dynamic_spikes(x)
        quantized_weight = fake_quantize_weight(weight, scales, group_size=2)
        reference = F.linear((spikes * threshold).to(x.dtype), quantized_weight, bias)

    assert torch.equal(allocating, reused)
    assert torch.equal(allocating, reference)
    assert torch.equal(weight, original)
    assert buffer.data_ptr() != weight.data_ptr()
    assert allocating.shape == (2, 3)
