#!/usr/bin/env python3
"""Benchmark one full-size synthetic SpikingBrain block per process."""

from __future__ import annotations

import argparse
import json
import os
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

from spikingbrain_cpu.block import GLABlock, GLACache, KVCache, SlidingWindowAttentionBlock
from spikingbrain_cpu.ops import gla_recurrent, sliding_window_attention


HIDDEN_SIZE = 3584
INTERMEDIATE_SIZE = 18944
NUM_HEADS = 28
NUM_KV_HEADS = 4
HEAD_DIM = 128
WINDOW = 4096


def proc_memory() -> dict[str, float]:
    values = {}
    for line in Path("/proc/self/status").read_text().splitlines():
        key, _, rest = line.partition(":")
        if key in {"VmRSS", "VmHWM"}:
            values[key.lower() + "_mib"] = round(int(rest.split()[0]) / 1024, 2)
    return values


def available_ram_mib() -> float:
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return round(int(line.split()[1]) / 1024, 1)
    raise RuntimeError("MemAvailable not found")


def cpu_model() -> str:
    for line in Path("/proc/cpuinfo").read_text().splitlines():
        if line.startswith("model name"):
            return line.split(":", 1)[1].strip()
    return platform.processor() or "unknown"


def measure(operation: Callable[[], object], warmup: int, iterations: int) -> dict[str, float]:
    with torch.inference_mode():
        for _ in range(warmup):
            operation()
        samples = []
        for _ in range(iterations):
            start = time.perf_counter()
            operation()
            samples.append((time.perf_counter() - start) * 1000)
    return {
        "median_ms": round(statistics.median(samples), 3),
        "min_ms": round(min(samples), 3),
        "max_ms": round(max(samples), 3),
    }


def largest_projection_elements(block) -> int:
    return max(
        module.weight.numel()
        for module in block.modules()
        if hasattr(module, "weight") and isinstance(getattr(module, "weight"), torch.Tensor)
    )


def build_gla_cache() -> GLACache:
    return GLACache(torch.randn(1, NUM_HEADS, HEAD_DIM, HEAD_DIM), next_position=4096)


def build_attention_cache() -> KVCache:
    return KVCache(
        torch.randn(1, NUM_KV_HEADS, WINDOW, HEAD_DIM),
        torch.randn(1, NUM_KV_HEADS, WINDOW, HEAD_DIM),
        next_position=WINDOW,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--block", choices=("gla", "attention"), required=True)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--prefill-length", type=int, default=4)
    args = parser.parse_args()

    torch.set_num_threads(args.threads)
    torch.manual_seed(2026)
    before = proc_memory()
    available_before = available_ram_mib()

    if args.block == "gla":
        block = GLABlock().eval()
    else:
        block = SlidingWindowAttentionBlock().eval()
    after_construction = proc_memory()

    largest_elements = largest_projection_elements(block)
    block.quant_buffer.reserve(largest_elements)
    assert block.quant_buffer.tensor is not None
    block.quant_buffer.tensor.zero_()  # commit pages before measuring forwards
    after_buffer = proc_memory()
    cache = build_gla_cache() if args.block == "gla" else build_attention_cache()
    after_cache = proc_memory()

    decode_input = torch.randn(1, 1, HIDDEN_SIZE)
    prefill_input = torch.randn(1, args.prefill_length, HIDDEN_SIZE)
    normalized_decode = block.attn_norm(decode_input)

    if args.block == "gla":
        q = torch.randn(1, NUM_HEADS, 1, HEAD_DIM)
        k = F.relu(torch.randn_like(q))
        v = torch.randn_like(q)
        log_gate = F.logsigmoid(torch.randn_like(q)) / 16
        core_operation = lambda: gla_recurrent(q, k, v, log_gate, cache.state)
    else:
        q = torch.randn(1, NUM_HEADS, 1, HEAD_DIM)
        core_operation = lambda: sliding_window_attention(
            q, cache.key, cache.value, WINDOW
        )

    operations = {
        "quantlinear_gate_projection_decode": lambda: block.mlp.gate_proj(normalized_decode),
        "attention_or_gla_core_decode": core_operation,
        "mlp_decode": lambda: block.mlp(normalized_decode),
        "block_decode_cached": lambda: block(decode_input, cache),
        "block_prefill_short": lambda: block(prefill_input, None),
    }
    timings = {
        name: measure(operation, args.warmup, args.iterations)
        for name, operation in operations.items()
    }
    after_forward = proc_memory()

    result = {
        "environment": {
            "hostname": platform.node(),
            "cpu": cpu_model(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "threads": torch.get_num_threads(),
            "cuda_available": torch.cuda.is_available(),
            "available_ram_mib_before": available_before,
        },
        "configuration": {
            "block": args.block,
            "hidden_size": HIDDEN_SIZE,
            "intermediate_size": INTERMEDIATE_SIZE,
            "num_heads": NUM_HEADS,
            "num_kv_heads": NUM_KV_HEADS,
            "head_dim": HEAD_DIM,
            "sliding_window": WINDOW,
            "prefill_length": args.prefill_length,
            "warmup": args.warmup,
            "iterations": args.iterations,
            "seed": 2026,
        },
        "memory": {
            "rss_before_construction_mib": before.get("vmrss_mib"),
            "rss_after_construction_mib": after_construction.get("vmrss_mib"),
            "rss_after_buffer_mib": after_buffer.get("vmrss_mib"),
            "rss_after_cache_mib": after_cache.get("vmrss_mib"),
            "rss_after_forwards_mib": after_forward.get("vmrss_mib"),
            "process_peak_mib": after_forward.get("vmhwm_mib"),
            "block_parameters_and_scales_mib": round(block.parameter_nbytes / 2**20, 2),
            "quant_buffer_mib": round(block.quant_buffer.nbytes / 2**20, 2),
            "cache_mib": round(cache.nbytes / 2**20, 2),
        },
        "timings": timings,
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
