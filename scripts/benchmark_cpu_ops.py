#!/usr/bin/env python3
"""Small synthetic CPU benchmarks; never reads model files or checkpoints."""

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
from torch.profiler import ProfilerActivity, profile

from spikingbrain_cpu.ops import (
    RMSNorm,
    gla_recurrent,
    quant_linear,
    sliding_window_attention,
    swiglu,
)


def measure(operation: Callable[[], object], warmup: int, iterations: int) -> dict[str, float]:
    with torch.inference_mode():
        for _ in range(warmup):
            operation()
        timings = []
        for _ in range(iterations):
            start = time.perf_counter()
            operation()
            timings.append((time.perf_counter() - start) * 1_000)
    return {
        "median_ms": round(statistics.median(timings), 3),
        "min_ms": round(min(timings), 3),
        "max_ms": round(max(timings), 3),
    }


def cpu_model() -> str:
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "unknown"


def available_ram_mib() -> float | None:
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemAvailable:"):
                return round(int(line.split()[1]) / 1024, 1)
    except OSError:
        pass
    return None


def allocator_bytes(operation: Callable[[], object]) -> int:
    """Total positive CPU allocator traffic reported for one operation."""
    with torch.inference_mode(), profile(
        activities=[ProfilerActivity.CPU], profile_memory=True
    ) as trace:
        operation()
    return sum(max(0, event.self_cpu_memory_usage) for event in trace.key_averages())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--threads", type=int, default=min(4, os.cpu_count() or 1))
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    torch.manual_seed(2026)

    # Model-compatible dimensions, but deliberately short sequences and no full
    # projection matrices. These benchmark the fallback kernels, not a full layer.
    rms_x = torch.randn(1, 32, 3584)
    rms = RMSNorm(3584).eval()
    gate = torch.randn(1, 8, 18944)
    value = torch.randn_like(gate)
    q_attn = torch.randn(1, 28, 64, 128)
    k_attn = torch.randn(1, 4, 64, 128)
    v_attn = torch.randn_like(k_attn)
    q_gla = torch.randn(1, 28, 32, 128)
    k_gla = torch.relu(torch.randn_like(q_gla))
    v_gla = torch.randn_like(q_gla)
    gate_gla = F.logsigmoid(torch.randn_like(q_gla)) / 16
    q_decode_attn = torch.randn(1, 28, 1, 128)
    k_decode_attn = torch.randn(1, 4, 4096, 128)
    v_decode_attn = torch.randn_like(k_decode_attn)
    q_decode_gla = torch.randn(1, 28, 1, 128)
    k_decode_gla = torch.relu(torch.randn_like(q_decode_gla))
    v_decode_gla = torch.randn_like(q_decode_gla)
    gate_decode_gla = F.logsigmoid(torch.randn_like(q_decode_gla)) / 16
    gla_state = torch.randn(1, 28, 128, 128)
    quant_weight = torch.randn(1024, 1024)
    quant_scales = torch.rand(1024, 8, 1) + 0.1
    quant_input = torch.randn(1, 1024)
    quant_buffer = torch.empty_like(quant_weight)

    allocating_quant = lambda: quant_linear(
        quant_input, quant_weight, quant_scales, group_size=128
    )
    buffered_quant = lambda: quant_linear(
        quant_input, quant_weight, quant_scales, group_size=128, out=quant_buffer
    )

    benchmarks = {
        "rmsnorm_b1_l32_h3584": lambda: rms(rms_x),
        "swiglu_core_b1_l8_i18944": lambda: swiglu(gate, value),
        "attention_b1_l64_h28_kv4_d128_w64": lambda: sliding_window_attention(
            q_attn, k_attn, v_attn, 64
        ),
        "attention_decode_q1_k4096_h28_kv4_d128_w4096": lambda: sliding_window_attention(
            q_decode_attn, k_decode_attn, v_decode_attn, 4096
        ),
        "gla_recurrent_b1_l32_h28_d128": lambda: gla_recurrent(
            q_gla, k_gla, v_gla, gate_gla
        ),
        "gla_decode_l1_h28_d128_cached": lambda: gla_recurrent(
            q_decode_gla,
            k_decode_gla,
            v_decode_gla,
            gate_decode_gla,
            initial_state=gla_state,
        ),
        "quantlinear_allocating_1024x1024": allocating_quant,
        "quantlinear_buffered_1024x1024": buffered_quant,
    }
    results = {
        "environment": {
            "hostname": platform.node(),
            "cpu": cpu_model(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "threads": torch.get_num_threads(),
            "available_ram_mib": available_ram_mib(),
            "cuda_available": torch.cuda.is_available(),
        },
        "benchmarks": {
            name: measure(operation, args.warmup, args.iterations)
            for name, operation in benchmarks.items()
        },
        "quantlinear_ram": {
            "synthetic_matrix_mib": round(quant_weight.numel() * quant_weight.element_size() / 2**20, 2),
            "synthetic_allocator_traffic_allocating_mib": round(allocator_bytes(allocating_quant) / 2**20, 2),
            "synthetic_allocator_traffic_buffered_mib": round(allocator_bytes(buffered_quant) / 2**20, 2),
            "largest_checkpoint_matrix_assumed": "18944 x 3584 FP32",
            "matrix_mib": round(18944 * 3584 * 4 / 2**20, 2),
            "minimum_extra_mib_with_reusable_buffer": round(18944 * 3584 * 4 / 2**20, 2),
            "estimated_peak_extra_mib_allocating_expression": round(2 * 18944 * 3584 * 4 / 2**20, 2),
            "note": "Allocator traffic is cumulative, not peak RSS. The original expression allocates div, round and mul results; two matrix-sized tensors can overlap. The buffered path allocates the output buffer once outside the profiled forward.",
        },
    }
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
