from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _sample_rows(rows: list[dict[str, Any]], limit: int, seed: int) -> list[dict[str, Any]]:
    if limit <= 0 or len(rows) <= limit:
        return rows
    return random.Random(seed).sample(rows, limit)


def audit_split(cache_root: Path, split: str, limit: int, seed: int) -> dict[str, Any]:
    index_path = cache_root / split / "index.jsonl"
    rows = _read_jsonl(index_path)
    sampled = _sample_rows(rows, limit, seed)

    result: dict[str, Any] = {
        "split": split,
        "count": len(rows),
        "sampled": len(sampled),
        "keys": None,
        "frame_shape": None,
        "all_zero_frame_stacks": 0,
        "low_frame_diff_stacks": 0,
        "matched_history_candidate_hits": 0,
        "matched_history_total": 0,
        "match_valid_missing": 0,
        "match_valid_inconsistent": 0,
        "match_valid_true": 0,
        "test_type_counts": {},
        "mean_frame_absdiff": None,
        "bad_paths": [],
    }
    diffs: list[float] = []
    test_types: Counter[int] = Counter()

    for row in sampled:
        rel_path = row["path"]
        npz_path = cache_root / split / rel_path
        try:
            data = np.load(npz_path)
            if result["keys"] is None:
                result["keys"] = sorted(data.files)
            frame_key = "frames" if "frames" in data else "video"
            frames = data[frame_key]
            if result["frame_shape"] is None:
                result["frame_shape"] = list(frames.shape)
            if frames.max() == 0:
                result["all_zero_frame_stacks"] += 1
            if frames.shape[0] > 1:
                diff = float(np.abs(frames[1:].astype(np.int16) - frames[:-1].astype(np.int16)).mean())
                diffs.append(diff)
                if diff < 0.5:
                    result["low_frame_diff_stacks"] += 1
            if "matched_history_t" in data:
                result["matched_history_total"] += 1
                matched_t = int(np.asarray(data["matched_history_t"]).reshape(-1)[0])
                candidate_indices = (
                    set(map(int, data["candidate_history_indices"].tolist()))
                    if "candidate_history_indices" in data
                    else set()
                )
                hit = matched_t in candidate_indices
                if hit:
                    result["matched_history_candidate_hits"] += 1
                if "match_valid" not in data:
                    result["match_valid_missing"] += 1
                else:
                    match_valid = bool(int(np.asarray(data["match_valid"]).reshape(-1)[0]))
                    result["match_valid_true"] += int(match_valid)
                    if match_valid != hit:
                        result["match_valid_inconsistent"] += 1
            test_type_key = "test_type" if "test_type" in data else "test_type_id"
            if test_type_key in data:
                test_type = int(np.asarray(data[test_type_key]).reshape(-1)[0])
                test_types[test_type] += 1
        except Exception as exc:  # noqa: BLE001 - this is an audit report.
            result["bad_paths"].append({"path": rel_path, "error": repr(exc)})

    result["test_type_counts"] = dict(sorted(test_types.items()))
    if diffs:
        result["mean_frame_absdiff"] = float(sum(diffs) / len(diffs))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-root", required=True, type=Path)
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    parser.add_argument("--limit", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--fail-on-bad", action="store_true")
    args = parser.parse_args()

    reports = [audit_split(args.cache_root, split, args.limit, args.seed) for split in args.splits]
    for report in reports:
        print(json.dumps(report, sort_keys=True))

    if args.fail_on_bad:
        failed = []
        for report in reports:
            if report["count"] <= 0:
                failed.append(f"{report['split']}: empty split")
            if report["bad_paths"]:
                failed.append(f"{report['split']}: unreadable samples={len(report['bad_paths'])}")
            if report["all_zero_frame_stacks"]:
                failed.append(f"{report['split']}: all-zero frame stacks={report['all_zero_frame_stacks']}")
            if report["match_valid_missing"]:
                failed.append(f"{report['split']}: samples missing match_valid={report['match_valid_missing']}")
            if report["match_valid_inconsistent"]:
                failed.append(
                    f"{report['split']}: match_valid inconsistent={report['match_valid_inconsistent']}"
                )
        if failed:
            raise SystemExit("cache quality audit failed: " + "; ".join(failed))


if __name__ == "__main__":
    main()
