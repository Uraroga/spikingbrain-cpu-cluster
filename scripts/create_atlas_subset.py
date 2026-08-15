#!/usr/bin/env python3
"""Plan or stream-copy the atlas-only tensors without materializing tensors."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import struct
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from spikingbrain_cpu.selective_loader import IndexPlanner

EXPECTED_COUNT = 295
EXPECTED_BYTES = 15_384_983_040
BUFFER_SIZE = 8 * 1024 * 1024


def read_header(path: Path):
    with path.open("rb") as stream:
        raw = stream.read(8)
        if len(raw) != 8:
            raise ValueError(f"invalid Safetensors file: {path}")
        length = struct.unpack("<Q", raw)[0]
        header = json.loads(stream.read(length))
    return 8 + length, header


def copy_range(source, target, offset: int, size: int, digest) -> None:
    source.seek(offset)
    remaining = size
    while remaining:
        chunk = source.read(min(BUFFER_SIZE, remaining))
        if not chunk:
            raise EOFError("unexpected EOF while copying tensor data")
        target.write(chunk)
        digest.update(chunk)
        remaining -= len(chunk)


def write_subset_shard(source_path: Path, output_path: Path, names: list[str]) -> dict:
    data_base, source_header = read_header(source_path)
    selected, cursor = {}, 0
    for name in names:
        entry = source_header.get(name)
        if not isinstance(entry, dict):
            raise KeyError(f"{name} absent from {source_path.name}")
        start, end = entry["data_offsets"]
        size = end - start
        selected[name] = {
            "dtype": entry["dtype"],
            "shape": entry["shape"],
            "data_offsets": [cursor, cursor + size],
        }
        cursor += size
    header = {"__metadata__": {"format": "pt"}, **selected}
    raw_header = json.dumps(header, separators=(",", ":")).encode()
    raw_header += b" " * ((-len(raw_header)) % 8)
    prefix = struct.pack("<Q", len(raw_header)) + raw_header
    partial = output_path.with_suffix(output_path.suffix + ".partial")
    digest = hashlib.sha256()
    with source_path.open("rb") as source, partial.open("xb") as target:
        target.write(prefix)
        digest.update(prefix)
        for name in names:
            start, end = source_header[name]["data_offsets"]
            copy_range(source, target, data_base + start, end - start, digest)
        target.flush()
        os.fsync(target.fileno())
    partial.replace(output_path)
    return {"filename": output_path.name, "size": output_path.stat().st_size, "sha256": digest.hexdigest()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--extract", action="store_true")
    args = parser.parse_args()
    planner = IndexPlanner.from_files(
        args.model_dir / "config.json", args.model_dir / "model.safetensors.index.json"
    )
    atlas, _ = planner.plan_split(14)
    if len(atlas.tensors) != EXPECTED_COUNT or atlas.nbytes != EXPECTED_BYTES:
        raise RuntimeError(
            f"atlas plan mismatch: tensors={len(atlas.tensors)}, bytes={atlas.nbytes}; "
            f"expected {EXPECTED_COUNT}, {EXPECTED_BYTES}"
        )
    manifest = {
        "partition": "atlas5",
        "split_layer": 14,
        "tensor_count": len(atlas.tensors),
        "total_tensor_bytes": atlas.nbytes,
        "tensors": [
            {"name": t.name, "source_shard": t.shard, "shape": list(t.shape), "dtype": t.dtype, "bytes": t.nbytes}
            for t in atlas.tensors
        ],
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"validated atlas plan: {len(atlas.tensors)} tensors, {atlas.nbytes} bytes")
    if not args.extract:
        return
    if args.output_dir is None:
        raise ValueError("--output-dir is required with --extract")
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=False)
    grouped: dict[str, list[str]] = {}
    for tensor in atlas.tensors:
        grouped.setdefault(tensor.shard, []).append(tensor.name)
    file_records, weight_map = [], {}
    count = len(grouped)
    for number, (source_shard, names) in enumerate(grouped.items(), start=1):
        target_name = f"model-{number:05d}-of-{count:05d}.safetensors"
        print(f"streaming {source_shard} -> {target_name} ({len(names)} tensors)", flush=True)
        record = write_subset_shard(args.model_dir / source_shard, output / target_name, names)
        file_records.append(record)
        weight_map.update({name: target_name for name in names})
    index = {"metadata": {"total_size": atlas.nbytes}, "weight_map": weight_map}
    (output / "model.safetensors.index.json").write_text(json.dumps(index, indent=2) + "\n")
    shutil.copyfile(args.model_dir / "config.json", output / "config.json")
    shutil.copyfile(args.manifest, output / "atlas_tensor_manifest.json")
    for name in ("model.safetensors.index.json", "config.json", "atlas_tensor_manifest.json"):
        path = output / name
        file_records.append({"filename": name, "size": path.stat().st_size, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    checksum_manifest = {
        "files": file_records,
        "total_file_bytes": sum(item["size"] for item in file_records),
        "total_tensor_bytes": atlas.nbytes,
    }
    (output / "checksums.json").write_text(json.dumps(checksum_manifest, indent=2) + "\n")
    print(json.dumps(checksum_manifest, indent=2))


if __name__ == "__main__":
    main()
