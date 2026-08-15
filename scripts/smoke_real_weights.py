#!/usr/bin/env python3
"""Controlled real-weight smoke tests; intended only for read-only argo3 mounts."""

from __future__ import annotations

import argparse
import gc
import json
import platform
import statistics
import sys
import time
from pathlib import Path
from typing import Callable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import torch
import torch.nn.functional as F

from spikingbrain_cpu.block import GLABlock, GLACache
from spikingbrain_cpu.ops import gla_recurrent
from spikingbrain_cpu.selective_loader import (
    IndexPlanner,
    LoaderLimitError,
    RealTensorLoader,
    materialize_layer,
    process_memory_mib,
)


def memory() -> dict[str, float]:
    rss, peak = process_memory_mib()
    return {"rss_mib": round(rss, 2), "peak_mib": round(peak, 2)}


def guard_rss(limit: float, context: str) -> None:
    rss, peak = process_memory_mib()
    if rss > limit or peak > limit:
        raise LoaderLimitError(
            f"RSS limit exceeded {context}: rss={rss:.2f}, peak={peak:.2f}, limit={limit:.2f} MiB"
        )


def touch_tensor(tensor: torch.Tensor) -> float:
    # One FP32 value per 4 KiB page forces mmap pages resident without a copy.
    return tensor.view(-1)[::1024].sum().item()


def touch_module(module: torch.nn.Module) -> float:
    checksum = 0.0
    with torch.inference_mode():
        for tensor in list(module.parameters()) + list(module.buffers()):
            checksum += touch_tensor(tensor)
    return checksum


def measure(operation: Callable[[], object], warmup: int = 1, iterations: int = 3):
    with torch.inference_mode():
        for _ in range(warmup):
            operation()
        samples = []
        last = None
        for _ in range(iterations):
            start = time.perf_counter()
            last = operation()
            samples.append((time.perf_counter() - start) * 1000)
    return {
        "median_ms": round(statistics.median(samples), 3),
        "min_ms": round(min(samples), 3),
        "max_ms": round(max(samples), 3),
    }, last


def planner_from_model(model_dir: Path) -> IndexPlanner:
    return IndexPlanner.from_files(
        model_dir / "config.json", model_dir / "model.safetensors.index.json"
    )


def run_small(args, planner: IndexPlanner) -> dict:
    names = (
        "model.layers.14.attn_norm.weight",
        "model.layers.14.attn.k_proj.weight",
        "model.layers.14.mlp.gate_proj.weight",
    )
    records = []
    for name in names:
        gc.collect()
        before = memory()
        plan = {p.name: p for p in planner.all_tensors()}[name]
        loader = RealTensorLoader(planner, args.model_dir, plan.nbytes, args.max_rss_mib)
        iterator = loader.iter_materialized((name,))
        loaded_plan, tensor = next(iterator)
        mapped = memory()
        checksum = touch_tensor(tensor)
        touched = memory()
        iterator.close()
        del tensor, iterator, loader
        gc.collect()
        released = memory()
        records.append(
            {
                "name": name,
                "shard": loaded_plan.shard,
                "shape": loaded_plan.shape,
                "dtype": "float32",
                "elements": loaded_plan.nbytes // 4,
                "bytes": loaded_plan.nbytes,
                "rss_before_mib": before["rss_mib"],
                "rss_after_map_mib": mapped["rss_mib"],
                "rss_after_touch_mib": touched["rss_mib"],
                "rss_after_release_mib": released["rss_mib"],
                "process_peak_mib": touched["peak_mib"],
                "sampled_checksum": checksum,
            }
        )
        guard_rss(args.max_rss_mib, f"after {name}")
    return {"mode": "small", "tensors": records}


def run_layer(args, planner: IndexPlanner) -> dict:
    layer_idx = 14
    before = memory()
    block = GLABlock(device="meta").eval()
    after_meta = memory()
    loader = RealTensorLoader(planner, args.model_dir, args.max_bytes, args.max_rss_mib)
    load_start = time.perf_counter()
    names = materialize_layer(block, layer_idx, loader)
    load_ms = (time.perf_counter() - load_start) * 1000
    after_map = memory()
    checksum = touch_module(block)
    after_touch = memory()

    largest = max(parameter.numel() for parameter in block.parameters())
    block.quant_buffer.reserve(largest)
    assert block.quant_buffer.tensor is not None
    block.quant_buffer.tensor.zero_()
    after_buffer = memory()

    torch.manual_seed(404)
    hidden = torch.randn(1, 1, 3584)
    cache = GLACache(torch.zeros(1, 28, 128, 128), next_position=4096)
    normalized = block.attn_norm(hidden)
    q = torch.randn(1, 28, 1, 128)
    k = torch.relu(torch.randn_like(q))
    v = torch.randn_like(q)
    gate = F.logsigmoid(torch.randn_like(q)) / 16

    quant_timing, _ = measure(lambda: block.mlp.gate_proj(normalized))
    core_timing, _ = measure(lambda: gla_recurrent(q, k, v, gate, cache.state))
    mlp_timing, _ = measure(lambda: block.mlp(normalized))
    block_timing, last = measure(lambda: block(hidden, cache))
    output, output_cache = last
    guard_rss(args.max_rss_mib, "after real layer forwards")
    after_forward = memory()
    result = {
        "mode": "layer",
        "layer": layer_idx,
        "layer_type": "gla",
        "tensor_count": len(names),
        "opened_shards": list(dict.fromkeys(loader.opened_shards)),
        "materialized_bytes": loader.materialized_bytes,
        "load_ms": round(load_ms, 3),
        "sampled_checksum": checksum,
        "memory": {
            "before": before,
            "after_meta_construction": after_meta,
            "after_mmap_assignment": after_map,
            "after_page_touch": after_touch,
            "after_quant_buffer": after_buffer,
            "after_forwards": after_forward,
            "quant_buffer_mib": round(block.quant_buffer.nbytes / 2**20, 2),
            "cache_mib": round(cache.nbytes / 2**20, 2),
        },
        "validation": {
            "output_shape": list(output.shape),
            "output_dtype": str(output.dtype),
            "output_finite": bool(torch.isfinite(output).all().item()),
            "cache_shape": list(output_cache.state.shape),
            "cache_position": output_cache.next_position,
        },
        "timings": {
            "quantlinear_gate_projection_decode": quant_timing,
            "gla_core_decode": core_timing,
            "mlp_decode": mlp_timing,
            "block_decode_cached": block_timing,
        },
    }
    del block, output, output_cache, hidden, cache, loader
    gc.collect()
    result["memory_after_release"] = memory()
    return result


def run_lm_head(args, planner: IndexPlanner) -> dict:
    name = "lm_head.weight"
    before = memory()
    loader = RealTensorLoader(planner, args.model_dir, args.max_bytes, args.max_rss_mib)
    load_start = time.perf_counter()
    iterator = loader.iter_materialized((name,))
    plan, weight = next(iterator)
    iterator.close()
    load_ms = (time.perf_counter() - load_start) * 1000
    after_map = memory()
    checksum = touch_tensor(weight)
    after_touch = memory()
    hidden = torch.randn(1, 3584)
    projection_timing, logits = measure(lambda: F.linear(hidden, weight))
    argmax_timing, argmax = measure(lambda: torch.argmax(logits, dim=-1))
    topk_timing, topk = measure(lambda: torch.topk(logits, k=5, dim=-1))
    guard_rss(args.max_rss_mib, "after lm_head projection")
    after_forward = memory()
    result = {
        "mode": "lm_head",
        "name": name,
        "shard": plan.shard,
        "shape": plan.shape,
        "dtype": "float32",
        "elements": weight.numel(),
        "bytes": plan.nbytes,
        "opened_shards": list(dict.fromkeys(loader.opened_shards)),
        "load_ms": round(load_ms, 3),
        "sampled_checksum": checksum,
        "memory": {
            "before": before,
            "after_mmap_assignment": after_map,
            "after_page_touch": after_touch,
            "after_projection": after_forward,
        },
        "projection": {
            "timing": projection_timing,
            "logits_shape": list(logits.shape),
            "logits_dtype": str(logits.dtype),
            "logits_bytes": logits.numel() * logits.element_size(),
            "finite": bool(torch.isfinite(logits).all().item()),
        },
        "argmax": {"timing": argmax_timing, "index": int(argmax.item())},
        "topk": {"timing": topk_timing, "indices": topk.indices.tolist()},
    }
    del weight, logits, hidden, loader, iterator, argmax, topk
    gc.collect()
    result["memory_after_release"] = memory()
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("small", "layer", "lm-head"), required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--max-bytes", type=int, required=True)
    parser.add_argument("--max-rss-mib", type=float, required=True)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    planner = planner_from_model(args.model_dir)
    if args.mode == "small":
        result = run_small(args, planner)
    elif args.mode == "layer":
        result = run_layer(args, planner)
    else:
        result = run_lm_head(args, planner)
    result["environment"] = {
        "hostname": platform.node(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "threads": torch.get_num_threads(),
        "cuda_available": torch.cuda.is_available(),
        "max_bytes": args.max_bytes,
        "max_rss_mib": args.max_rss_mib,
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
