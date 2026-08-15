#!/usr/bin/env python3
"""One real, local AtlasStage forward for independent-process diagnosis."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, "/app/scripts")

import torch

from distributed_stage import load_stage, memory


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--max-bytes", type=int, default=15_384_997_376)
    parser.add_argument("--max-rss-mib", type=float, default=23_552)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--token", type=int, default=42)
    args = parser.parse_args()
    args.rank = 0
    torch.set_num_threads(args.threads)

    started = time.perf_counter()
    stage, loading = load_stage(0, args)
    load_ms = (time.perf_counter() - started) * 1000
    token = torch.tensor([[args.token]], dtype=torch.long)
    started = time.perf_counter()
    with torch.inference_mode():
        hidden = stage.embed(token)
        hidden, caches, _, _ = stage.forward_layers(hidden, {})
    forward_ms = (time.perf_counter() - started) * 1000
    print(json.dumps({
        "torch": torch.__version__,
        "capability": torch.backends.cpu.get_cpu_capability(),
        "loading": loading,
        "load_ms": load_ms,
        "forward_ms": forward_ms,
        "shape": list(hidden.shape),
        "finite": bool(torch.isfinite(hidden).all()),
        "checksum": float(hidden.reshape(-1)[::128].sum()),
        "cache_positions": sorted({cache.next_position for cache in caches.values()}),
        "memory": memory(),
    }, indent=2))


if __name__ == "__main__":
    main()
