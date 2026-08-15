"""Small, readable CPU reference operations used by SpikingBrain.

These implementations intentionally use only public PyTorch operations.  They
do not import the checkpoint's Python modules, Triton, FlashAttention or fla.
They target inference and numerical validation, not peak performance.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
from torch import Tensor, nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    """RMSNorm compatible with the checkpoint's non-residual use.

    Reduction is performed in float32 for float16/bfloat16 inputs, matching the
    reference implementation shipped beside the checkpoint.  The result keeps
    the input dtype.
    """

    def __init__(
        self, hidden_size: int, eps: float = 1e-6, device: torch.device | str | None = None
    ) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size, device=device))
        self.eps = eps

    def forward(
        self, x: Tensor, residual: Optional[Tensor] = None, prenorm: bool = False
    ) -> Tensor | Tuple[Tensor, Tensor]:
        if residual is not None:
            if residual.shape != x.shape:
                raise ValueError("residual must have the same shape as x")
            x = x + residual
        residual_output = x
        input_dtype = x.dtype
        work = x.float() if input_dtype in (torch.float16, torch.bfloat16) else x
        variance = work.square().mean(dim=-1, keepdim=True)
        normalized = work * torch.rsqrt(variance + self.eps)
        output = normalized.to(input_dtype) * self.weight.to(input_dtype)
        return (output, residual_output) if prenorm else output


def swish(x: Tensor) -> Tensor:
    """Pure PyTorch SiLU/swish replacement for ``fla.modules.activations``."""

    return F.silu(x)


def swiglu(
    gate: Tensor,
    value: Tensor,
    weight: Optional[Tensor] = None,
    bias: Optional[Tensor] = None,
) -> Tensor:
    """Compute ``silu(gate) * value`` and optionally a final linear layer.

    Passing ``weight`` reproduces the checkpoint's
    ``swiglu_linear(gate, value, down_proj.weight, down_proj.bias)`` call.
    """

    if gate.shape != value.shape:
        raise ValueError(f"gate and value must have equal shapes: {gate.shape} != {value.shape}")
    activated = F.silu(gate) * value
    return F.linear(activated, weight, bias) if weight is not None else activated


def _repeat_kv(x: Tensor, num_query_heads: int) -> Tensor:
    if x.ndim != 4:
        raise ValueError("q, k and v must have shape [batch, heads, sequence, dim]")
    kv_heads = x.shape[1]
    if num_query_heads % kv_heads:
        raise ValueError("number of query heads must be divisible by KV heads")
    return torch.repeat_interleave(x, num_query_heads // kv_heads, dim=1)


def sliding_window_attention(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    window_size: int,
    attention_mask: Optional[Tensor] = None,
    scale: Optional[float] = None,
) -> Tensor:
    """Reference causal grouped-query sliding-window attention.

    Args:
        q: ``[batch, query_heads, query_length, key_dim]``.
        k: ``[batch, kv_heads, key_length, key_dim]``.
        v: ``[batch, kv_heads, key_length, value_dim]``.
        window_size: Number of previous positions visible to each query.  The
            current position is also visible, matching FlashAttention's
            ``window_size=(left, 0)`` convention used by the checkpoint.
        attention_mask: Optional key-validity mask ``[batch, key_length]`` where
            nonzero/True entries are visible.  Query positions are aligned to
            the right of the key sequence, which also supports cached decoding.
    """

    if window_size < 0:
        raise ValueError("window_size must be non-negative")
    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
        raise ValueError("q, k and v must be rank-4 tensors")
    if q.device.type != "cpu" or k.device.type != "cpu" or v.device.type != "cpu":
        raise ValueError("this reference implementation is CPU-only")
    if q.shape[0] != k.shape[0] or k.shape[:3] != v.shape[:3]:
        raise ValueError("batch, KV-head and key-length dimensions must agree")
    if q.shape[-1] != k.shape[-1]:
        raise ValueError("query and key dimensions must agree")
    if q.shape[2] > k.shape[2]:
        raise ValueError("query length cannot exceed key length")

    k = _repeat_kv(k, q.shape[1])
    v = _repeat_kv(v, q.shape[1])
    q_len, k_len = q.shape[2], k.shape[2]
    query_positions = torch.arange(k_len - q_len, k_len, device=q.device)
    key_positions = torch.arange(k_len, device=q.device)
    distance = query_positions[:, None] - key_positions[None, :]
    allowed = (distance >= 0) & (distance <= window_size)
    allowed = allowed[None, None, :, :]

    if attention_mask is not None:
        if attention_mask.shape != (q.shape[0], k_len):
            raise ValueError("attention_mask must have shape [batch, key_length]")
        allowed = allowed & attention_mask.to(torch.bool)[:, None, None, :]

    score_scale = scale if scale is not None else q.shape[-1] ** -0.5
    work_q = q.float() if q.dtype in (torch.float16, torch.bfloat16) else q
    work_k = k.float() if k.dtype in (torch.float16, torch.bfloat16) else k
    scores = torch.matmul(work_q, work_k.transpose(-1, -2)) * score_scale
    scores = scores.masked_fill(~allowed, -torch.inf)

    # Causality guarantees one key per query unless an external mask removes all
    # of them.  Return zero for such fully masked rows instead of producing NaN.
    valid_row = allowed.any(dim=-1, keepdim=True)
    scores = torch.where(valid_row, scores, torch.zeros_like(scores))
    probabilities = torch.softmax(scores, dim=-1)
    probabilities = torch.where(valid_row, probabilities, torch.zeros_like(probabilities))
    work_v = v.float() if probabilities.dtype != v.dtype else v
    return torch.matmul(probabilities, work_v).to(q.dtype)


def gla_recurrent(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    log_gate: Tensor,
    initial_state: Optional[Tensor] = None,
    attention_mask: Optional[Tensor] = None,
    scale: Optional[float] = None,
) -> Tuple[Tensor, Tensor]:
    """Sequential reference Gated Linear Attention.

    All four inputs use ``[batch, heads, sequence, dim]``; value dim may differ
    from key dim. ``log_gate`` is a per-key-feature log decay and is normally
    ``logsigmoid(gk) / gate_logit_normalizer`` in SpikingBrain.

    Recurrence for each token ``t``::

        state_t = exp(log_gate_t) * state_(t-1) + k_t outer v_t
        output_t = (q_t * scale) @ state_t

    The explicit loop is deliberate: it is a correctness baseline for a future
    optimized CPU kernel and naturally supports cached one-token decoding.
    """

    if any(x.ndim != 4 for x in (q, k, v, log_gate)):
        raise ValueError("q, k, v and log_gate must be rank-4 tensors")
    if any(x.device.type != "cpu" for x in (q, k, v, log_gate)):
        raise ValueError("this reference implementation is CPU-only")
    if q.shape[:3] != k.shape[:3] or q.shape[:3] != v.shape[:3]:
        raise ValueError("q, k and v batch/head/sequence dimensions must agree")
    if q.shape != log_gate.shape or q.shape[-1] != k.shape[-1]:
        raise ValueError("q, k and log_gate must have equal key dimensions")

    batch, heads, length, key_dim = q.shape
    value_dim = v.shape[-1]
    accumulation_dtype = (
        torch.float32 if q.dtype in (torch.float16, torch.bfloat16) else q.dtype
    )
    if initial_state is None:
        state = torch.zeros(
            batch, heads, key_dim, value_dim, dtype=accumulation_dtype, device=q.device
        )
    else:
        expected = (batch, heads, key_dim, value_dim)
        if initial_state.shape != expected:
            raise ValueError(f"initial_state must have shape {expected}")
        state = initial_state.to(accumulation_dtype).clone()

    q_work = q.to(accumulation_dtype)
    k_work = k.to(accumulation_dtype)
    v_work = v.to(accumulation_dtype)
    gate_work = log_gate.to(accumulation_dtype)
    score_scale = scale if scale is not None else key_dim**-0.5
    outputs = []

    if attention_mask is not None and attention_mask.shape != (batch, length):
        raise ValueError("attention_mask must have shape [batch, sequence]")

    for token in range(length):
        decay = torch.exp(gate_work[:, :, token]).unsqueeze(-1)
        value_t = v_work[:, :, token]
        if attention_mask is not None:
            value_t = value_t * attention_mask[:, token, None, None].to(value_t.dtype)
        state = state * decay + torch.einsum(
            "bhk,bhv->bhkv", k_work[:, :, token], value_t
        )
        output_t = torch.einsum("bhk,bhkv->bhv", q_work[:, :, token] * score_scale, state)
        outputs.append(output_t)

    output = torch.stack(outputs, dim=2) if outputs else q.new_empty(batch, heads, 0, value_dim)
    return output.to(q.dtype), state.to(q.dtype)


def fake_quantize_weight(weight: Tensor, scales: Tensor, group_size: int = 128) -> Tensor:
    """Reproduce QuantLinear's FP32 weight fake-quantization.

    The returned tensor is a new, full-size dense tensor.  This is faithful to
    the checkpoint but important for RAM planning: no storage compression occurs.
    """

    if weight.ndim != 2 or group_size <= 0 or weight.shape[1] % group_size:
        raise ValueError("weight input dimension must be divisible by group_size")
    grouped = weight.reshape(weight.shape[0], -1, group_size)
    expected = (weight.shape[0], weight.shape[1] // group_size, 1)
    if scales.shape != expected:
        raise ValueError(f"scales must have shape {expected}")
    return ((grouped / scales).round() * scales).reshape_as(weight)


def dynamic_spikes(x: Tensor, k: float = 3.0) -> Tuple[Tensor, Tensor]:
    """CPU form of the checkpoint's activation spike fake-quantization.

    The optional bitwise neuron path in ``neuron.py`` encodes and immediately
    decodes the rounded integers without clipping, so its inference result is
    equal to the rounded values returned here; constructing the spike sequence
    would only consume extra memory.
    """

    if k <= 0:
        raise ValueError("k must be positive")
    threshold = (x.abs().mean(dim=-1, keepdim=True).float() / k).clamp(1e-5, 1e4)
    spikes = (x / threshold).round()
    return spikes, threshold


def quantize_symmetric(x: Tensor, bits: int = 8) -> Tensor:
    """Per-vector symmetric fake quantization used for attention K/V."""

    if bits < 2:
        raise ValueError("bits must be at least 2")
    q_max = (1 << (bits - 1)) - 1
    q_min = -(1 << (bits - 1))
    scale = (x.abs().amax(dim=-1, keepdim=True).float() / q_max).clamp(1e-5, 1e4)
    integers = (x / scale).round().clamp(q_min, q_max)
    return (integers * scale).to(x.dtype)


def quant_linear(
    x: Tensor,
    weight: Tensor,
    scales: Tensor,
    bias: Optional[Tensor] = None,
    group_size: int = 128,
    out: Optional[Tensor] = None,
    dynamic_sfr: float = 3.0,
) -> Tensor:
    """Fake-quantized linear with an optional reusable weight-sized buffer.

    Supplying ``out`` avoids the chain of multiple weight-sized temporaries, but
    still needs one dense buffer because ``F.linear`` cannot consume the rounded
    values lazily.  Neither path modifies the original weight or scales.
    """

    expected = (weight.shape[0], weight.shape[1] // group_size, 1)
    if scales.shape != expected:
        raise ValueError(f"scales must have shape {expected}")
    spikes, threshold = dynamic_spikes(x, dynamic_sfr)
    quantized_input = (spikes * threshold).to(x.dtype)
    if out is None:
        quantized = fake_quantize_weight(weight, scales, group_size)
    else:
        if out.shape != weight.shape or out.dtype != weight.dtype or out.device != weight.device:
            raise ValueError("out must match weight shape, dtype and device")
        grouped_out = out.view(weight.shape[0], -1, group_size)
        torch.div(weight.view_as(grouped_out), scales, out=grouped_out)
        grouped_out.round_().mul_(scales)
        quantized = out
    return F.linear(quantized_input, quantized, bias)
