#!/usr/bin/env python3
import argparse, hashlib, json
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("directory", type=Path)
args = parser.parse_args()
manifest = json.loads((args.directory / "checksums.json").read_text())
for record in manifest["files"]:
    path = args.directory / record["filename"]
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    actual = digest.hexdigest()
    if path.stat().st_size != record["size"] or actual != record["sha256"]:
        raise SystemExit(f"FAILED: {record['filename']}")
    print(f"OK {record['filename']} {actual}")
print(f"verified {len(manifest['files'])} files")
