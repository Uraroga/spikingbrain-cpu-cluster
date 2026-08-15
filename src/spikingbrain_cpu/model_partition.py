"""CPU model partitions built incrementally from selective real tensors."""

from __future__ import annotations

import time
from typing import Optional

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .block import (
    GLABlock,
    GLACache,
    KVCache,
    QuantBuffer,
    SlidingWindowAttentionBlock,
)
from .ops import RMSNorm
from .selective_loader import RealTensorLoader, materialize_layer


class AtlasStage(nn.Module):
    first_layer = 0
    last_layer = 13

    def __init__(self) -> None:
        super().__init__()
        self.quant_buffer = QuantBuffer()
        self.embeddings = nn.Embedding(152064, 3584, device="meta")
        self.layers = nn.ModuleDict(
            {
                str(index): (
                    GLABlock(buffer=self.quant_buffer, device="meta")
                    if index % 2 == 0
                    else SlidingWindowAttentionBlock(buffer=self.quant_buffer, device="meta")
                )
                for index in range(self.first_layer, self.last_layer + 1)
            }
        )
        self.embedding_loaded = False
        self.loaded_layers: set[int] = set()
        self.loaded_tensor_count = 0
        self.loaded_logical_bytes = 0

    def load_embedding(self, loader: RealTensorLoader) -> None:
        if self.embedding_loaded:
            raise RuntimeError("embedding already loaded")
        before = loader.materialized_bytes
        _, tensor = next(loader.iter_materialized(("model.embeddings.weight",)))
        if tuple(tensor.shape) != tuple(self.embeddings.weight.shape):
            raise ValueError("embedding shape mismatch")
        self.embeddings.weight = nn.Parameter(tensor, requires_grad=False)
        self.embedding_loaded = True
        self.loaded_tensor_count += 1
        self.loaded_logical_bytes += loader.materialized_bytes - before

    def load_layer(self, layer_idx: int, loader: RealTensorLoader) -> tuple[str, ...]:
        if not self.first_layer <= layer_idx <= self.last_layer:
            raise ValueError(f"layer {layer_idx} is outside AtlasStage")
        if layer_idx in self.loaded_layers:
            raise RuntimeError(f"layer {layer_idx} is already loaded")
        before = loader.materialized_bytes
        names = materialize_layer(self.layers[str(layer_idx)], layer_idx, loader)
        self.loaded_layers.add(layer_idx)
        self.loaded_tensor_count += len(names)
        self.loaded_logical_bytes += loader.materialized_bytes - before
        return names

    def allocate_quant_buffer(self) -> None:
        if self.loaded_layers != set(range(self.first_layer, self.last_layer + 1)):
            raise RuntimeError("all layers must be loaded before sizing the shared buffer")
        largest = max(
            parameter.numel()
            for layer in self.layers.values()
            for parameter in layer.parameters()
        )
        self.quant_buffer.reserve(largest)
        assert self.quant_buffer.tensor is not None
        self.quant_buffer.tensor.zero_()

    def quant_buffer_identity_count(self) -> int:
        return len({id(layer.quant_buffer) for layer in self.layers.values()})

    def make_decode_caches(self, batch_size: int = 1, position: int = 4096):
        caches = {}
        for layer_idx in range(self.first_layer, self.last_layer + 1):
            if layer_idx % 2 == 0:
                caches[layer_idx] = GLACache(
                    torch.zeros(batch_size, 28, 128, 128), next_position=position
                )
            else:
                caches[layer_idx] = KVCache(
                    torch.zeros(batch_size, 4, position, 128),
                    torch.zeros(batch_size, 4, position, 128),
                    next_position=position,
                )
        return caches

    @staticmethod
    def cache_nbytes(caches: dict[int, GLACache | KVCache]) -> int:
        return sum(cache.nbytes for cache in caches.values())

    def embed(self, token_ids: Tensor) -> Tensor:
        if not self.embedding_loaded:
            raise RuntimeError("embedding is not loaded")
        return self.embeddings(token_ids)

    def forward_layers(self, hidden_states: Tensor, caches=None, profile_layers=False):
        expected = set(range(self.first_layer, self.last_layer + 1))
        if self.loaded_layers != expected:
            raise RuntimeError("all AtlasStage layers must be loaded before forward")
        caches = {} if caches is None else caches
        new_caches, timings, shapes = {}, {}, {}
        for layer_idx in range(self.first_layer, self.last_layer + 1):
            start = time.perf_counter()
            hidden_states, new_cache = self.layers[str(layer_idx)](
                hidden_states, caches.get(layer_idx)
            )
            if profile_layers:
                timings[layer_idx] = (time.perf_counter() - start) * 1000
                shapes[layer_idx] = {
                    "shape": tuple(hidden_states.shape),
                    "dtype": str(hidden_states.dtype),
                    "finite": bool(torch.isfinite(hidden_states).all().item()),
                }
            new_caches[layer_idx] = new_cache
        return hidden_states, new_caches, timings, shapes


class ArgoStage(nn.Module):
    first_layer = 14
    last_layer = 27

    def __init__(self) -> None:
        super().__init__()
        self.quant_buffer = QuantBuffer()
        blocks = {}
        for layer_idx in range(self.first_layer, self.last_layer + 1):
            if layer_idx % 2 == 0:
                block = GLABlock(buffer=self.quant_buffer, device="meta")
            else:
                block = SlidingWindowAttentionBlock(buffer=self.quant_buffer, device="meta")
            blocks[str(layer_idx)] = block
        self.layers = nn.ModuleDict(blocks)
        self.final_norm = RMSNorm(3584, eps=1e-6, device="meta")
        self.lm_head = nn.Linear(3584, 152064, bias=False, device="meta")
        self.loaded_layers: set[int] = set()
        self.norm_loaded = False
        self.lm_head_loaded = False
        self.loaded_tensor_count = 0
        self.loaded_logical_bytes = 0

    def load_layer(self, layer_idx: int, loader: RealTensorLoader) -> tuple[str, ...]:
        if not self.first_layer <= layer_idx <= self.last_layer:
            raise ValueError(f"layer {layer_idx} is outside ArgoStage")
        if layer_idx in self.loaded_layers:
            raise RuntimeError(f"layer {layer_idx} is already loaded")
        before = loader.materialized_bytes
        names = materialize_layer(self.layers[str(layer_idx)], layer_idx, loader)
        self.loaded_layers.add(layer_idx)
        self.loaded_tensor_count += len(names)
        self.loaded_logical_bytes += loader.materialized_bytes - before
        return names

    def load_final_norm(self, loader: RealTensorLoader) -> None:
        if self.norm_loaded:
            raise RuntimeError("final norm already loaded")
        before = loader.materialized_bytes
        plan, tensor = next(loader.iter_materialized(("model.norm.weight",)))
        if tuple(tensor.shape) != tuple(self.final_norm.weight.shape):
            raise ValueError("final norm shape mismatch")
        self.final_norm.weight = nn.Parameter(tensor, requires_grad=False)
        self.norm_loaded = True
        self.loaded_tensor_count += 1
        self.loaded_logical_bytes += loader.materialized_bytes - before

    def load_lm_head(self, loader: RealTensorLoader) -> None:
        if self.lm_head_loaded:
            raise RuntimeError("lm_head already loaded")
        before = loader.materialized_bytes
        plan, tensor = next(loader.iter_materialized(("lm_head.weight",)))
        if tuple(tensor.shape) != tuple(self.lm_head.weight.shape):
            raise ValueError("lm_head shape mismatch")
        self.lm_head.weight = nn.Parameter(tensor, requires_grad=False)
        self.lm_head_loaded = True
        self.loaded_tensor_count += 1
        self.loaded_logical_bytes += loader.materialized_bytes - before

    def allocate_quant_buffer(self) -> None:
        if self.loaded_layers != set(range(self.first_layer, self.last_layer + 1)):
            raise RuntimeError("all layers must be loaded before sizing the shared buffer")
        largest = max(
            parameter.numel()
            for layer in self.layers.values()
            for parameter in layer.parameters()
        )
        self.quant_buffer.reserve(largest)
        assert self.quant_buffer.tensor is not None
        self.quant_buffer.tensor.zero_()

    def quant_buffer_identity_count(self) -> int:
        return len({id(layer.quant_buffer) for layer in self.layers.values()})

    def make_decode_caches(self, batch_size: int = 1, position: int = 4096):
        caches = {}
        for layer_idx in range(self.first_layer, self.last_layer + 1):
            if layer_idx % 2 == 0:
                caches[layer_idx] = GLACache(
                    torch.zeros(batch_size, 28, 128, 128), next_position=position
                )
            else:
                caches[layer_idx] = KVCache(
                    torch.zeros(batch_size, 4, position, 128),
                    torch.zeros(batch_size, 4, position, 128),
                    next_position=position,
                )
        return caches

    @staticmethod
    def cache_nbytes(caches: dict[int, GLACache | KVCache]) -> int:
        return sum(cache.nbytes for cache in caches.values())

    def forward_layers(
        self,
        hidden_states: Tensor,
        caches: Optional[dict[int, GLACache | KVCache]] = None,
        profile_layers: bool = False,
    ):
        expected = set(range(self.first_layer, self.last_layer + 1))
        if self.loaded_layers != expected:
            raise RuntimeError("all ArgoStage layers must be loaded before forward")
        caches = {} if caches is None else caches
        new_caches = {}
        timings = {}
        shapes = {}
        for layer_idx in range(self.first_layer, self.last_layer + 1):
            start = time.perf_counter()
            hidden_states, new_cache = self.layers[str(layer_idx)](
                hidden_states, caches.get(layer_idx)
            )
            if profile_layers:
                timings[layer_idx] = (time.perf_counter() - start) * 1000
                shapes[layer_idx] = {
                    "shape": tuple(hidden_states.shape),
                    "dtype": str(hidden_states.dtype),
                    "finite": bool(torch.isfinite(hidden_states).all().item()),
                }
            new_caches[layer_idx] = new_cache
        return hidden_states, new_caches, timings, shapes

    def apply_final_norm(self, hidden_states: Tensor) -> Tensor:
        if not self.norm_loaded:
            raise RuntimeError("final norm is not loaded")
        return self.final_norm(hidden_states)

    def project_logits(self, hidden_states: Tensor) -> Tensor:
        if not self.lm_head_loaded:
            raise RuntimeError("lm_head is not loaded")
        return F.linear(hidden_states, self.lm_head.weight)
