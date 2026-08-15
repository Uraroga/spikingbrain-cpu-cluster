"""Synthetic CPU blocks mirroring the SpikingBrain checkpoint structure."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Optional

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .ops import (
    RMSNorm,
    dynamic_spikes,
    gla_recurrent,
    quant_linear,
    quantize_symmetric,
    sliding_window_attention,
    swiglu,
)


class QuantBuffer:
    """One reusable fake-quantized-weight buffer shared by all projections."""

    def __init__(self) -> None:
        self.tensor: Optional[Tensor] = None

    def reserve(self, elements: int, dtype: torch.dtype = torch.float32) -> None:
        if self.tensor is None or self.tensor.numel() < elements or self.tensor.dtype != dtype:
            self.tensor = torch.empty(elements, dtype=dtype, device="cpu")

    def view_for(self, weight: Tensor) -> Tensor:
        self.reserve(weight.numel(), weight.dtype)
        assert self.tensor is not None
        return self.tensor[: weight.numel()].view_as(weight)

    @property
    def nbytes(self) -> int:
        return 0 if self.tensor is None else self.tensor.numel() * self.tensor.element_size()


class SyntheticQuantLinear(nn.Module):
    """Checkpoint-compatible FP32 QuantLinear for synthetic block benchmarks."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        buffer: QuantBuffer,
        bias: bool = True,
        group_size: int = 128,
        device: torch.device | str | None = None,
    ) -> None:
        super().__init__()
        if in_features % group_size:
            raise ValueError("in_features must be divisible by group_size")
        self.in_features = in_features
        self.out_features = out_features
        self.group_size = group_size
        self.buffer = buffer
        self.weight = nn.Parameter(
            torch.empty(out_features, in_features, device=device), requires_grad=False
        )
        self.bias = (
            nn.Parameter(torch.empty(out_features, device=device), requires_grad=False)
            if bias
            else None
        )
        self.register_buffer(
            "scales",
            torch.empty(out_features, in_features // group_size, 1, device=device),
        )
        if self.weight.device.type != "meta":
            self.reset_parameters()

    def reset_parameters(self) -> None:
        bound = self.in_features**-0.5
        with torch.no_grad():
            self.weight.uniform_(-bound, bound)
            self.scales.fill_(max(bound / 16, 1e-5))
            if self.bias is not None:
                self.bias.zero_()

    def forward(self, x: Tensor) -> Tensor:
        return quant_linear(
            x,
            self.weight,
            self.scales,
            self.bias,
            group_size=self.group_size,
            out=self.buffer.view_for(self.weight),
        )


class SyntheticMLP(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        buffer: QuantBuffer,
        group_size: int,
        device: torch.device | str | None = None,
    ) -> None:
        super().__init__()
        self.gate_proj = SyntheticQuantLinear(
            hidden_size,
            intermediate_size,
            buffer,
            bias=False,
            group_size=group_size,
            device=device,
        )
        self.up_proj = SyntheticQuantLinear(
            hidden_size,
            intermediate_size,
            buffer,
            bias=False,
            group_size=group_size,
            device=device,
        )
        # The checkpoint calls swiglu_linear with down_proj.weight directly, so
        # down_proj does not run QuantLinear.forward in the real model code.
        self.down_proj = SyntheticQuantLinear(
            intermediate_size,
            hidden_size,
            buffer,
            bias=False,
            group_size=group_size,
            device=device,
        )

    def forward(self, x: Tensor) -> Tensor:
        gate = self.gate_proj(x)
        value = self.up_proj(x)
        return swiglu(gate, value, self.down_proj.weight, self.down_proj.bias)


@dataclass(frozen=True)
class GLACache:
    state: Tensor
    next_position: int

    @property
    def nbytes(self) -> int:
        return self.state.numel() * self.state.element_size()


@dataclass(frozen=True)
class KVCache:
    key: Tensor
    value: Tensor
    next_position: int

    @property
    def nbytes(self) -> int:
        return (self.key.numel() * self.key.element_size()) + (
            self.value.numel() * self.value.element_size()
        )


class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, base: float = 1_000_000.0) -> None:
        super().__init__()
        inv_freq = 1.0 / (
            base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, positions: Tensor, dtype: torch.dtype) -> tuple[Tensor, Tensor]:
        frequencies = torch.outer(positions.float(), self.inv_freq)
        embedding = torch.cat((frequencies, frequencies), dim=-1)
        return embedding.cos().to(dtype), embedding.sin().to(dtype)


def _rotate_half(x: Tensor) -> Tensor:
    first, second = x.chunk(2, dim=-1)
    return torch.cat((-second, first), dim=-1)


def _apply_rope(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    cos = cos[None, None, :, :]
    sin = sin[None, None, :, :]
    return x * cos + _rotate_half(x) * sin


class _BaseBlock(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        buffer: Optional[QuantBuffer],
        group_size: int,
        norm_eps: float,
        device: torch.device | str | None,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.quant_buffer = buffer if buffer is not None else QuantBuffer()
        self.attn_norm = RMSNorm(hidden_size, norm_eps, device=device)
        self.mlp_norm = RMSNorm(hidden_size, norm_eps, device=device)
        self.mlp = SyntheticMLP(
            hidden_size, intermediate_size, self.quant_buffer, group_size, device=device
        )

    def _finish_block(self, attention_output: Tensor, residual: Tensor) -> Tensor:
        normalized, residual = self.mlp_norm(attention_output, residual, prenorm=True)
        return residual + self.mlp(normalized)

    @property
    def parameter_nbytes(self) -> int:
        return sum(p.numel() * p.element_size() for p in self.parameters()) + sum(
            b.numel() * b.element_size() for b in self.buffers()
        )


class GLABlock(_BaseBlock):
    def __init__(
        self,
        hidden_size: int = 3584,
        intermediate_size: int = 18944,
        num_heads: int = 28,
        num_key_value_heads: int = 4,
        gate_logit_normalizer: int = 16,
        group_size: int = 128,
        norm_eps: float = 1e-6,
        buffer: Optional[QuantBuffer] = None,
        device: torch.device | str | None = None,
    ) -> None:
        super().__init__(hidden_size, intermediate_size, buffer, group_size, norm_eps, device)
        if hidden_size % num_heads or num_heads % num_key_value_heads:
            raise ValueError("invalid head configuration")
        self.num_heads = num_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = hidden_size // num_heads
        self.kv_dim = num_key_value_heads * self.head_dim
        self.gate_logit_normalizer = gate_logit_normalizer
        self.q_proj = SyntheticQuantLinear(
            hidden_size, hidden_size, self.quant_buffer, True, group_size, device
        )
        self.k_proj = SyntheticQuantLinear(
            hidden_size, self.kv_dim, self.quant_buffer, True, group_size, device
        )
        self.v_proj = SyntheticQuantLinear(
            hidden_size, self.kv_dim, self.quant_buffer, True, group_size, device
        )
        self.gk_proj = SyntheticQuantLinear(
            hidden_size, self.kv_dim, self.quant_buffer, True, group_size, device
        )
        self.g_norm = RMSNorm(self.head_dim, norm_eps, device=device)
        self.o_proj = SyntheticQuantLinear(
            hidden_size, hidden_size, self.quant_buffer, False, group_size, device
        )

    def _attention(
        self, normalized: Tensor, cache: Optional[GLACache]
    ) -> tuple[Tensor, GLACache]:
        batch, length, _ = normalized.shape
        q = F.relu(self.q_proj(normalized)).view(
            batch, length, self.num_heads, self.head_dim
        ).transpose(1, 2)
        k = F.relu(self.k_proj(normalized)).view(
            batch, length, self.num_key_value_heads, self.head_dim
        ).transpose(1, 2)
        v = self.v_proj(normalized).view(
            batch, length, self.num_key_value_heads, self.head_dim
        ).transpose(1, 2)
        log_gate = (F.logsigmoid(self.gk_proj(normalized)) / self.gate_logit_normalizer).view(
            batch, length, self.num_key_value_heads, self.head_dim
        ).transpose(1, 2)
        repeat = self.num_heads // self.num_key_value_heads
        k = torch.repeat_interleave(k, repeat, dim=1)
        v = torch.repeat_interleave(v, repeat, dim=1)
        log_gate = torch.repeat_interleave(log_gate, repeat, dim=1)
        core, state = gla_recurrent(
            q, k, v, log_gate, initial_state=None if cache is None else cache.state
        )
        core = self.g_norm(core).transpose(1, 2).reshape(batch, length, self.hidden_size)
        output = self.o_proj(core)
        next_position = (0 if cache is None else cache.next_position) + length
        return output, GLACache(state, next_position)

    def forward(
        self, hidden_states: Tensor, cache: Optional[GLACache] = None
    ) -> tuple[Tensor, GLACache]:
        residual = hidden_states
        normalized = self.attn_norm(hidden_states)
        attention_output, new_cache = self._attention(normalized, cache)
        return self._finish_block(attention_output, residual), new_cache


class SlidingWindowAttentionBlock(_BaseBlock):
    def __init__(
        self,
        hidden_size: int = 3584,
        intermediate_size: int = 18944,
        num_heads: int = 28,
        num_key_value_heads: int = 4,
        sliding_window: int = 4096,
        rope_theta: float = 1_000_000.0,
        group_size: int = 128,
        norm_eps: float = 1e-6,
        buffer: Optional[QuantBuffer] = None,
        device: torch.device | str | None = None,
    ) -> None:
        super().__init__(hidden_size, intermediate_size, buffer, group_size, norm_eps, device)
        if hidden_size % num_heads or num_heads % num_key_value_heads:
            raise ValueError("invalid head configuration")
        self.num_heads = num_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = hidden_size // num_heads
        self.kv_dim = num_key_value_heads * self.head_dim
        self.sliding_window = sliding_window
        self.q_proj = SyntheticQuantLinear(
            hidden_size, hidden_size, self.quant_buffer, True, group_size, device
        )
        self.k_proj = SyntheticQuantLinear(
            hidden_size, self.kv_dim, self.quant_buffer, True, group_size, device
        )
        self.v_proj = SyntheticQuantLinear(
            hidden_size, self.kv_dim, self.quant_buffer, True, group_size, device
        )
        self.o_proj = SyntheticQuantLinear(
            hidden_size, hidden_size, self.quant_buffer, False, group_size, device
        )
        self.rotary = RotaryEmbedding(self.head_dim, rope_theta)

    def _attention(
        self, normalized: Tensor, cache: Optional[KVCache]
    ) -> tuple[Tensor, KVCache]:
        batch, length, _ = normalized.shape
        start = 0 if cache is None else cache.next_position
        positions = torch.arange(start, start + length, device=normalized.device)
        q = self.q_proj(normalized).view(
            batch, length, self.num_heads, self.head_dim
        ).transpose(1, 2)
        k = self.k_proj(normalized).view(
            batch, length, self.num_key_value_heads, self.head_dim
        ).transpose(1, 2)
        v = self.v_proj(normalized).view(
            batch, length, self.num_key_value_heads, self.head_dim
        ).transpose(1, 2)
        cos, sin = self.rotary(positions, q.dtype)
        q = _apply_rope(q, cos, sin)
        k = _apply_rope(k, cos, sin)
        q_spikes, q_threshold = dynamic_spikes(q, 3.0)
        q = (q_spikes * q_threshold).to(q.dtype)
        if cache is not None:
            k = torch.cat((cache.key, k), dim=2)
            v = torch.cat((cache.value, v), dim=2)
        max_cached = self.sliding_window + 1
        if k.shape[2] > max_cached:
            k = k[:, :, -max_cached:]
            v = v[:, :, -max_cached:]
        core = sliding_window_attention(
            q, quantize_symmetric(k), quantize_symmetric(v), self.sliding_window
        )
        core = core.transpose(1, 2).reshape(batch, length, self.hidden_size)
        output = self.o_proj(core)
        return output, KVCache(k, v, start + length)

    def forward(
        self, hidden_states: Tensor, cache: Optional[KVCache] = None
    ) -> tuple[Tensor, KVCache]:
        residual = hidden_states
        normalized = self.attn_norm(hidden_states)
        attention_output, new_cache = self._attention(normalized, cache)
        return self._finish_block(attention_output, residual), new_cache


def timed_forward(block: nn.Module, hidden: Tensor, cache=None) -> tuple[Tensor, object, float]:
    """Convenience used by the benchmark without adding instrumentation overhead."""
    start = time.perf_counter()
    output, new_cache = block(hidden, cache)
    return output, new_cache, (time.perf_counter() - start) * 1_000
