#!/usr/bin/env python3
from __future__ import annotations

import argparse
import io
import json
import zipfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import repeat
from pathlib import Path

import numpy as np


REQUIRED_KEYS = ("future_length", "memory_condition_length", "has_reference_tail")


def _npy_bytes(value: int, dtype) -> bytes:
    buffer = io.BytesIO()
    np.lib.format.write_array(buffer, np.asarray(value, dtype=dtype), allow_pickle=False)
    return buffer.getvalue()


def _upgrade_file(path: str, future_length: int, memory_condition_length: int, dry_run: bool = False) -> tuple[str, str]:
    npz_path = Path(path)
    has_reference_tail = int(memory_condition_length > 0)
    payloads = {
        "future_length.npy": _npy_bytes(future_length, np.int64),
        "memory_condition_length.npy": _npy_bytes(memory_condition_length, np.int64),
        "has_reference_tail.npy": _npy_bytes(has_reference_tail, np.int64),
    }
    with zipfile.ZipFile(npz_path, "r") as zf:
        names = set(zf.namelist())
    missing = [name for name in payloads if name not in names]
    if not missing:
        return str(npz_path), "ok"
    if dry_run:
        return str(npz_path), "missing"
    with zipfile.ZipFile(npz_path, "a", compression=zipfile.ZIP_STORED) as zf:
        for name in missing:
            zf.writestr(name, payloads[name])
    return str(npz_path), "upgraded"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("cache_dir", type=Path)
    parser.add_argument("--split", default="train")
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    manifest_path = args.cache_dir / "manifest.json"
    if not manifest_path.exists():
        raise SystemExit(f"missing manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    future_length = int(manifest["future_length"])
    memory_condition_length = int(manifest["memory_condition_length"])

    index_path = args.cache_dir / args.split / "index.jsonl"
    if not index_path.exists():
        raise SystemExit(f"missing index: {index_path}")
    paths = []
    with index_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            entry = json.loads(line)
            sample_path = Path(entry["path"])
            if not sample_path.is_absolute():
                sample_path = index_path.parent / sample_path
            paths.append(str(sample_path))

    counts = {"ok": 0, "missing": 0, "upgraded": 0, "error": 0}
    with ProcessPoolExecutor(max_workers=max(1, args.workers)) as pool:
        results = pool.map(
            _upgrade_file,
            paths,
            repeat(future_length),
            repeat(memory_condition_length),
            repeat(args.dry_run),
            chunksize=64,
        )
        for i, result in enumerate(results, 1):
            try:
                _path, status = result
            except Exception as exc:
                counts["error"] += 1
                if counts["error"] <= 10:
                    print(f"ERROR {exc}", flush=True)
                continue
            counts[status] = counts.get(status, 0) + 1
            if i % 5000 == 0 or i == len(paths):
                print(f"progress {i}/{len(paths)} {counts}", flush=True)
    if counts["error"]:
        raise SystemExit(f"metadata upgrade finished with errors: {counts}")
    print(f"done {counts}", flush=True)


if __name__ == "__main__":
    main()
