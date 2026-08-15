"""Index-only planning for a future selective Safetensors loader.

This module deliberately does not import safetensors and never opens shard
files. Shapes are inferred from config.json plus the checkpoint tensor names.
"""

from __future__ import annotations

import argparse
import gc
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import torch
from torch import Tensor, nn
from safetensors import safe_open


FP32_BYTES = 4


@dataclass(frozen=True)
class TensorPlan:
    name: str
    shard: str
    shape: tuple[int, ...]
    dtype: str = "float32"

    @property
    def nbytes(self) -> int:
        elements = 1
        for dimension in self.shape:
            elements *= dimension
        return elements * FP32_BYTES


@dataclass(frozen=True)
class PartitionPlan:
    node: str
    tensors: tuple[TensorPlan, ...]

    @property
    def nbytes(self) -> int:
        return sum(tensor.nbytes for tensor in self.tensors)


class IndexPlanner:
    def __init__(self, config: Mapping[str, object], index: Mapping[str, object]) -> None:
        self.config = config
        raw_map = index.get("weight_map")
        if not isinstance(raw_map, dict):
            raise ValueError("index is missing weight_map")
        self.weight_map = {str(name): str(shard) for name, shard in raw_map.items()}
        self.expected_total = int(index.get("metadata", {}).get("total_size", 0))

    @classmethod
    def from_files(cls, config_path: Path, index_path: Path) -> "IndexPlanner":
        with config_path.open(encoding="utf-8") as stream:
            config = json.load(stream)
        with index_path.open(encoding="utf-8") as stream:
            index = json.load(stream)
        return cls(config, index)

    @property
    def hidden(self) -> int:
        return int(self.config["hidden_size"])

    @property
    def intermediate(self) -> int:
        return int(self.config["intermediate_size"])

    @property
    def head_dim(self) -> int:
        return self.hidden // int(self.config["num_attention_heads"])

    @property
    def kv_dim(self) -> int:
        return int(self.config["num_key_value_heads"]) * self.head_dim

    @staticmethod
    def _linear_shape(suffix: str, in_features: int, out_features: int) -> tuple[int, ...]:
        if suffix == "weight":
            return (out_features, in_features)
        if suffix == "bias":
            return (out_features,)
        if suffix == "weight_quantizer.scales":
            return (out_features, in_features // 128, 1)
        raise ValueError(f"unknown linear tensor suffix: {suffix}")

    def infer_shape(self, name: str) -> tuple[int, ...]:
        if name in {"model.embeddings.weight", "lm_head.weight"}:
            return (int(self.config["vocab_size"]), self.hidden)
        if name == "model.norm.weight":
            return (self.hidden,)

        match = re.fullmatch(r"model\.layers\.(\d+)\.(.+)", name)
        if not match:
            raise ValueError(f"unsupported tensor name: {name}")
        remainder = match.group(2)
        if remainder in {"attn_norm.weight", "mlp_norm.weight"}:
            return (self.hidden,)
        if remainder == "attn.g_norm.weight":
            return (self.head_dim,)

        linear = re.fullmatch(
            r"(attn\.(?:q_proj|k_proj|v_proj|gk_proj|o_proj)|"
            r"mlp\.(?:gate_proj|up_proj|down_proj))\.(.+)",
            remainder,
        )
        if not linear:
            raise ValueError(f"unsupported layer tensor name: {name}")
        module, suffix = linear.groups()
        if module == "attn.q_proj" or module == "attn.o_proj":
            return self._linear_shape(suffix, self.hidden, self.hidden)
        if module in {"attn.k_proj", "attn.v_proj", "attn.gk_proj"}:
            return self._linear_shape(suffix, self.hidden, self.kv_dim)
        if module in {"mlp.gate_proj", "mlp.up_proj"}:
            return self._linear_shape(suffix, self.hidden, self.intermediate)
        if module == "mlp.down_proj":
            return self._linear_shape(suffix, self.intermediate, self.hidden)
        raise AssertionError(module)

    def all_tensors(self) -> tuple[TensorPlan, ...]:
        tensors = tuple(
            TensorPlan(name, shard, self.infer_shape(name))
            for name, shard in sorted(self.weight_map.items())
        )
        inferred_total = sum(tensor.nbytes for tensor in tensors)
        if self.expected_total and inferred_total != self.expected_total:
            raise ValueError(
                f"inferred {inferred_total} bytes, index declares {self.expected_total}"
            )
        return tensors

    def plan_split(self, split_layer: int) -> tuple[PartitionPlan, PartitionPlan]:
        layer_count = int(self.config["num_hidden_layers"])
        if not 0 < split_layer < layer_count:
            raise ValueError(f"split_layer must be between 1 and {layer_count - 1}")
        atlas, argo = [], []
        for tensor in self.all_tensors():
            match = re.match(r"model\.layers\.(\d+)\.", tensor.name)
            if tensor.name == "model.embeddings.weight":
                atlas.append(tensor)
            elif match and int(match.group(1)) < split_layer:
                atlas.append(tensor)
            else:
                argo.append(tensor)
        return PartitionPlan("atlas5", tuple(atlas)), PartitionPlan("argo3", tuple(argo))


def process_memory_mib() -> tuple[float, float]:
    values: dict[str, float] = {}
    for line in Path("/proc/self/status").read_text().splitlines():
        key, _, rest = line.partition(":")
        if key in {"VmRSS", "VmHWM"}:
            values[key] = int(rest.split()[0]) / 1024
    return values.get("VmRSS", 0.0), values.get("VmHWM", 0.0)


class LoaderLimitError(RuntimeError):
    pass


class RealTensorLoader:
    """Materialize only explicitly selected tensors through Safetensors mmap."""

    def __init__(
        self,
        planner: IndexPlanner,
        model_dir: Path,
        max_bytes: int,
        max_rss_mib: float,
    ) -> None:
        if max_bytes <= 0 or max_rss_mib <= 0:
            raise ValueError("max_bytes and max_rss_mib must be positive")
        self.planner = planner
        self.model_dir = model_dir.resolve()
        self.max_bytes = max_bytes
        self.max_rss_mib = max_rss_mib
        self.materialized_bytes = 0
        self.opened_shards: list[str] = []
        self._plans = {tensor.name: tensor for tensor in planner.all_tensors()}
        self._check_rss("before loading")

    def _check_rss(self, context: str) -> None:
        rss, peak = process_memory_mib()
        if rss > self.max_rss_mib or peak > self.max_rss_mib:
            raise LoaderLimitError(
                f"RSS limit exceeded {context}: rss={rss:.2f} MiB, "
                f"peak={peak:.2f} MiB, limit={self.max_rss_mib:.2f} MiB"
            )

    def _shard_path(self, shard: str) -> Path:
        path = (self.model_dir / shard).resolve()
        if path.parent != self.model_dir:
            raise ValueError(f"unsafe shard path in index: {shard}")
        if not path.is_file():
            raise FileNotFoundError(path)
        return path

    def names_for_prefix(self, prefix: str) -> tuple[str, ...]:
        return tuple(name for name in self.planner.weight_map if name.startswith(prefix))

    def iter_materialized(self, names: Iterable[str]):
        requested = tuple(dict.fromkeys(names))
        missing = [name for name in requested if name not in self._plans]
        if missing:
            raise KeyError(f"tensor names absent from index: {missing}")
        requested_bytes = sum(self._plans[name].nbytes for name in requested)
        if self.materialized_bytes + requested_bytes > self.max_bytes:
            raise LoaderLimitError(
                f"byte limit exceeded before loading: requested={requested_bytes}, "
                f"already={self.materialized_bytes}, limit={self.max_bytes}"
            )

        by_shard: dict[str, list[str]] = {}
        for name in requested:
            by_shard.setdefault(self._plans[name].shard, []).append(name)
        for shard, shard_names in by_shard.items():
            path = self._shard_path(shard)
            self.opened_shards.append(shard)
            with safe_open(path, framework="pt", device="cpu") as handle:
                available = set(handle.keys())
                for name in shard_names:
                    if name not in available:
                        raise KeyError(f"{name} is mapped to {shard} but absent from its header")
                    plan = self._plans[name]
                    tensor = handle.get_tensor(name)
                    if tensor.dtype != torch.float32:
                        raise TypeError(f"{name}: expected float32, found {tensor.dtype}")
                    if tuple(tensor.shape) != plan.shape:
                        raise ValueError(
                            f"{name}: expected shape {plan.shape}, found {tuple(tensor.shape)}"
                        )
                    actual_bytes = tensor.numel() * tensor.element_size()
                    if actual_bytes != plan.nbytes:
                        raise ValueError(
                            f"{name}: expected {plan.nbytes} bytes, found {actual_bytes}"
                        )
                    self.materialized_bytes += actual_bytes
                    self._check_rss(f"after mapping {name}")
                    yield plan, tensor


def assign_layer_tensor(block: nn.Module, layer_idx: int, name: str, tensor: Tensor) -> None:
    prefix = f"model.layers.{layer_idx}."
    if not name.startswith(prefix):
        raise ValueError(f"{name} does not belong to layer {layer_idx}")
    path = name[len(prefix) :]
    if path.startswith("attn."):
        path = path[len("attn.") :]
    path = path.replace("weight_quantizer.scales", "scales")
    parts = path.split(".")
    target: object = block
    for part in parts[:-1]:
        target = getattr(target, part)
    attribute = parts[-1]
    expected = getattr(target, attribute)
    if tuple(expected.shape) != tuple(tensor.shape):
        raise ValueError(
            f"target shape mismatch for {name}: {tuple(expected.shape)} != {tuple(tensor.shape)}"
        )
    if isinstance(expected, nn.Parameter):
        setattr(target, attribute, nn.Parameter(tensor, requires_grad=False))
    else:
        setattr(target, attribute, tensor)


def materialize_layer(block: nn.Module, layer_idx: int, loader: RealTensorLoader) -> tuple[str, ...]:
    names = loader.names_for_prefix(f"model.layers.{layer_idx}.")
    for plan, tensor in loader.iter_materialized(names):
        assign_layer_tensor(block, layer_idx, plan.name, tensor)
    remaining_meta = [
        name
        for name, tensor in list(block.named_parameters()) + list(block.named_buffers())
        if tensor.device.type == "meta"
    ]
    if remaining_meta:
        raise RuntimeError(f"unmaterialized meta tensors: {remaining_meta}")
    return names


def _print_partition(partition: PartitionPlan, list_tensors: bool) -> None:
    print(
        f"{partition.node}: tensors={len(partition.tensors)} "
        f"bytes={partition.nbytes} GiB={partition.nbytes / 2**30:.6f}"
    )
    if list_tensors:
        for tensor in partition.tensors:
            print(
                f"  {tensor.name}\t{tensor.shard}\t{list(tensor.shape)}\t"
                f"{tensor.dtype}\t{tensor.nbytes}"
            )


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--split-layer", type=int)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--materialize-name", action="append")
    parser.add_argument("--summary-only", action="store_true")
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument("--max-bytes", type=int)
    parser.add_argument("--max-rss-mib", type=float)
    args = parser.parse_args(argv)
    planner = IndexPlanner.from_files(args.config, args.index)
    if args.materialize_name:
        if args.model_dir is None or args.max_bytes is None or args.max_rss_mib is None:
            parser.error("real materialization requires --model-dir, --max-bytes and --max-rss-mib")
        loader = RealTensorLoader(
            planner, args.model_dir, args.max_bytes, args.max_rss_mib
        )
        for plan, tensor in loader.iter_materialized(args.materialize_name):
            before, _ = process_memory_mib()
            checksum = tensor.view(-1)[::1024].double().sum().item()
            after, peak = process_memory_mib()
            print(
                f"{plan.name}\t{plan.shard}\t{list(plan.shape)}\t{tensor.dtype}\t"
                f"{plan.nbytes}\trss_before={before:.2f}\trss_after={after:.2f}\t"
                f"peak={peak:.2f}\tsampled_sum={checksum:.9g}"
            )
        return
    if args.split_layer is None:
        parser.error("--dry-run requires --split-layer")
    atlas, argo = planner.plan_split(args.split_layer)
    print(f"dry-run split: atlas5=[embedding, layers 0..{args.split_layer - 1}]")
    print(
        f"dry-run split: argo3=[layers {args.split_layer}.."
        f"{int(planner.config['num_hidden_layers']) - 1}, norm, lm_head]"
    )
    _print_partition(atlas, not args.summary_only)
    _print_partition(argo, not args.summary_only)
    print(
        f"total: tensors={len(atlas.tensors) + len(argo.tensors)} "
        f"bytes={atlas.nbytes + argo.nbytes} "
        f"GiB={(atlas.nbytes + argo.nbytes) / 2**30:.6f}"
    )


if __name__ == "__main__":
    main()
