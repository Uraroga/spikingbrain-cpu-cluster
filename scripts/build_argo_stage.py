#!/usr/bin/env python3
"""Incrementally build and validate the real argo3 stage under hard limits."""

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

from spikingbrain_cpu.model_partition import ArgoStage
from spikingbrain_cpu.selective_loader import (
    IndexPlanner,
    LoaderLimitError,
    RealTensorLoader,
    process_memory_mib,
)


CHECKPOINTS = {1, 2, 4, 8, 14}


def proc_status() -> dict[str, float]:
    values = {}
    for line in Path("/proc/self/status").read_text().splitlines():
        key, _, rest = line.partition(":")
        if key in {"VmRSS", "VmHWM", "VmSwap"}:
            values[key] = round(int(rest.split()[0]) / 1024, 2)
    return {
        "rss_mib": values.get("VmRSS", 0.0),
        "hwm_mib": values.get("VmHWM", 0.0),
        "swap_mib": values.get("VmSwap", 0.0),
    }


def host_memory() -> dict[str, float]:
    wanted = {"MemAvailable", "MemFree", "Cached", "SwapTotal", "SwapFree"}
    values = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        key = line.split(":", 1)[0]
        if key in wanted:
            values[key] = round(int(line.split()[1]) / 1024, 2)
    return {
        "available_mib": values.get("MemAvailable", 0.0),
        "free_mib": values.get("MemFree", 0.0),
        "cached_mib": values.get("Cached", 0.0),
        "swap_total_mib": values.get("SwapTotal", 0.0),
        "swap_free_mib": values.get("SwapFree", 0.0),
    }


def guard_rss(limit: float, context: str) -> None:
    rss, hwm = process_memory_mib()
    if rss > limit or hwm > limit:
        raise LoaderLimitError(
            f"RSS limit exceeded {context}: rss={rss:.2f}, hwm={hwm:.2f}, limit={limit:.2f} MiB"
        )


def touch_module(module: torch.nn.Module) -> float:
    checksum = 0.0
    with torch.inference_mode():
        for tensor in list(module.parameters()) + list(module.buffers()):
            checksum += tensor.view(-1)[::1024].sum().item()
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--max-bytes", type=int, required=True)
    parser.add_argument("--max-rss-mib", type=float, required=True)
    parser.add_argument("--min-host-available-mib", type=float, default=6144)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    torch.manual_seed(505)

    planner = IndexPlanner.from_files(
        args.model_dir / "config.json", args.model_dir / "model.safetensors.index.json"
    )
    atlas_plan, argo_plan = planner.plan_split(14)
    exact_required_bytes = argo_plan.nbytes
    if exact_required_bytes > args.max_bytes:
        raise LoaderLimitError(
            f"planned argo stage is {exact_required_bytes} bytes, limit is {args.max_bytes}"
        )

    process_before = proc_status()
    host_before = host_memory()
    stage = ArgoStage().eval()
    after_meta = proc_status()
    loader = RealTensorLoader(planner, args.model_dir, args.max_bytes, args.max_rss_mib)
    all_plans = planner.all_tensors()
    layer_expected = {
        index: sum(
            tensor.nbytes
            for tensor in all_plans
            if tensor.name.startswith(f"model.layers.{index}.")
        )
        for index in range(14, 28)
    }

    checkpoints = []
    cumulative_load_ms = 0.0
    checksum = 0.0
    for count, layer_idx in enumerate(range(14, 28), start=1):
        start = time.perf_counter()
        names = stage.load_layer(layer_idx, loader)
        checksum += touch_module(stage.layers[str(layer_idx)])
        cumulative_load_ms += (time.perf_counter() - start) * 1000
        guard_rss(args.max_rss_mib, f"after layer {layer_idx}")
        if host_memory()["available_mib"] < args.min_host_available_mib:
            raise LoaderLimitError(f"host available RAM floor crossed after layer {layer_idx}")
        if count in CHECKPOINTS:
            current = proc_status()
            expected_mib = stage.loaded_logical_bytes / 2**20
            observed_delta = current["rss_mib"] - after_meta["rss_mib"]
            if observed_delta > expected_mib + 1024:
                raise LoaderLimitError(
                    f"unexpected RSS overhead at {count} layers: delta={observed_delta:.2f} MiB, "
                    f"logical={expected_mib:.2f} MiB"
                )
            record = {
                "layers": count,
                "last_global_layer": layer_idx,
                "tensor_count": stage.loaded_tensor_count,
                "logical_bytes": stage.loaded_logical_bytes,
                "expected_bytes_from_plan": sum(layer_expected[i] for i in range(14, layer_idx + 1)),
                "rss_mib": current["rss_mib"],
                "hwm_mib": current["hwm_mib"],
                "rss_delta_from_meta_mib": round(observed_delta, 2),
                "unique_shards": list(dict.fromkeys(loader.opened_shards)),
                "shard_open_events": len(loader.opened_shards),
                "cumulative_load_and_touch_ms": round(cumulative_load_ms, 3),
                "host_available_mib": host_memory()["available_mib"],
            }
            checkpoints.append(record)
            print(
                f"checkpoint {count}/14: rss={current['rss_mib']:.2f} MiB "
                f"logical={expected_mib:.2f} MiB",
                file=sys.stderr,
                flush=True,
            )

    layer_unique_shards = list(dict.fromkeys(loader.opened_shards))

    before_buffer = proc_status()
    stage.allocate_quant_buffer()
    after_buffer = proc_status()
    if stage.quant_buffer_identity_count() != 1:
        raise RuntimeError("ArgoStage does not share exactly one QuantBuffer")
    guard_rss(args.max_rss_mib, "after QuantBuffer")

    before_caches = proc_status()
    caches = stage.make_decode_caches(position=4096)
    after_caches = proc_status()
    cache_bytes = stage.cache_nbytes(caches)
    if len({id(cache) for cache in caches.values()}) != 14:
        raise RuntimeError("cache objects are unexpectedly shared")
    if len({cache.next_position for cache in caches.values()}) != 1:
        raise RuntimeError("cache positions disagree")

    hidden = torch.randn(1, 1, 3584)
    validation_start = time.perf_counter()
    with torch.inference_mode():
        layer_output, validation_caches, layer_times, layer_diagnostics = stage.forward_layers(
            hidden, caches, profile_layers=True
        )
    validation_ms = (time.perf_counter() - validation_start) * 1000
    if not all(item["finite"] for item in layer_diagnostics.values()):
        raise FloatingPointError("non-finite output found in layer sequence")
    if any(cache.next_position != 4097 for cache in validation_caches.values()):
        raise RuntimeError("one or more caches did not advance to position 4097")
    guard_rss(args.max_rss_mib, "after validation forward")
    after_validation = proc_status()

    benchmark, benchmark_last = measure(lambda: stage.forward_layers(hidden, caches))
    guard_rss(args.max_rss_mib, "after stage benchmark")
    after_benchmark = proc_status()
    gla_times = [value for index, value in layer_times.items() if index % 2 == 0]
    attention_times = [value for index, value in layer_times.items() if index % 2 == 1]

    norm_before = proc_status()
    norm_load_start = time.perf_counter()
    stage.load_final_norm(loader)
    norm_checksum = touch_module(stage.final_norm)
    norm_load_ms = (time.perf_counter() - norm_load_start) * 1000
    norm_after_load = proc_status()
    norm_timing, normalized = measure(lambda: stage.apply_final_norm(layer_output))
    if not torch.isfinite(normalized).all():
        raise FloatingPointError("final norm produced non-finite values")

    rss_before_head = proc_status()
    host_before_head = host_memory()
    head_plan = next(tensor for tensor in all_plans if tensor.name == "lm_head.weight")
    if (
        args.max_rss_mib - rss_before_head["rss_mib"] < head_plan.nbytes / 2**20 + 1024
        or host_before_head["available_mib"] < head_plan.nbytes / 2**20 + args.min_host_available_mib
    ):
        raise LoaderLimitError("insufficient conservative margin for lm_head")
    head_load_start = time.perf_counter()
    stage.load_lm_head(loader)
    head_checksum = touch_module(stage.lm_head)
    head_load_ms = (time.perf_counter() - head_load_start) * 1000
    rss_after_head = proc_status()
    guard_rss(args.max_rss_mib, "after lm_head")

    head_timing, logits = measure(lambda: stage.project_logits(normalized))
    argmax_timing, token = measure(lambda: torch.argmax(logits, dim=-1))
    topk_timing, topk = measure(lambda: torch.topk(logits, 5, dim=-1))
    if not torch.isfinite(logits).all():
        raise FloatingPointError("lm_head produced non-finite logits")

    full_start = time.perf_counter()
    with torch.inference_mode():
        full_hidden, _, _, _ = stage.forward_layers(hidden, caches)
        full_hidden = stage.apply_final_norm(full_hidden)
        full_logits = stage.project_logits(full_hidden)
        full_token = torch.argmax(full_logits, dim=-1)
    full_ms = (time.perf_counter() - full_start) * 1000
    if not torch.isfinite(full_logits).all():
        raise FloatingPointError("full stage produced non-finite logits")
    guard_rss(args.max_rss_mib, "after full stage")
    final_process = proc_status()
    final_host = host_memory()

    result = {
        "environment": {
            "hostname": platform.node(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "threads": torch.get_num_threads(),
            "cuda_available": torch.cuda.is_available(),
            "max_bytes": args.max_bytes,
            "max_rss_mib": args.max_rss_mib,
            "min_host_available_mib": args.min_host_available_mib,
        },
        "plan": {
            "first_layer": 14,
            "last_layer": 27,
            "exact_required_bytes_layers_norm_head": exact_required_bytes,
            "planned_tensor_count": len(argo_plan.tensors),
        },
        "loading": {
            "process_before": process_before,
            "host_before": host_before,
            "after_meta_stage": after_meta,
            "checkpoints": checkpoints,
            "sampled_checksum": checksum,
            "unique_layer_shards": layer_unique_shards,
        },
        "quant_buffer": {
            "identity_count": stage.quant_buffer_identity_count(),
            "bytes": stage.quant_buffer.nbytes,
            "before": before_buffer,
            "after": after_buffer,
        },
        "caches": {
            "count": len(caches),
            "bytes": cache_bytes,
            "before": before_caches,
            "after": after_caches,
            "types": {index: type(cache).__name__ for index, cache in caches.items()},
            "all_positions_before": sorted({cache.next_position for cache in caches.values()}),
            "all_positions_after": sorted(
                {cache.next_position for cache in validation_caches.values()}
            ),
        },
        "validation_forward": {
            "total_ms": round(validation_ms, 3),
            "output_shape": list(layer_output.shape),
            "output_dtype": str(layer_output.dtype),
            "finite": bool(torch.isfinite(layer_output).all().item()),
            "per_layer_ms": layer_times,
            "per_layer": layer_diagnostics,
            "mean_gla_ms": round(statistics.mean(gla_times), 3),
            "mean_attention_ms": round(statistics.mean(attention_times), 3),
            "memory_after": after_validation,
        },
        "stage_benchmark": {
            **benchmark,
            "warmup": 1,
            "iterations": 3,
            "memory_after": after_benchmark,
        },
        "final_norm": {
            "load_ms": round(norm_load_ms, 3),
            "sampled_checksum": norm_checksum,
            "memory_before": norm_before,
            "memory_after_load": norm_after_load,
            "timing": norm_timing,
            "shape": list(normalized.shape),
            "dtype": str(normalized.dtype),
            "finite": bool(torch.isfinite(normalized).all().item()),
        },
        "lm_head": {
            "load_ms": round(head_load_ms, 3),
            "sampled_checksum": head_checksum,
            "memory_before": rss_before_head,
            "host_before": host_before_head,
            "memory_after_load": rss_after_head,
            "timing": head_timing,
            "logits_shape": list(logits.shape),
            "logits_finite": bool(torch.isfinite(logits).all().item()),
            "argmax_timing": argmax_timing,
            "argmax": int(token.item()),
            "topk_timing": topk_timing,
            "topk": topk.indices.tolist(),
        },
        "complete_stage": {
            "time_ms": round(full_ms, 3),
            "logits_shape": list(full_logits.shape),
            "logits_finite": bool(torch.isfinite(full_logits).all().item()),
            "token_id": int(full_token.item()),
            "process_final": final_process,
            "host_final": final_host,
        },
        "loader": {
            "loaded_tensor_count": stage.loaded_tensor_count,
            "loaded_logical_bytes": stage.loaded_logical_bytes,
            "materialized_bytes": loader.materialized_bytes,
            "all_unique_shards": list(dict.fromkeys(loader.opened_shards)),
            "shard_open_events": len(loader.opened_shards),
        },
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
