from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect PTM sample-level NPZ cache.")
    parser.add_argument("--cache-root", required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--limit", type=int, default=16)
    parser.add_argument("--require-consistent-match-valid", action="store_true")
    args = parser.parse_args()

    split_dir = Path(args.cache_root) / args.split
    files = sorted(split_dir.glob("sample_*.npz"))
    inspected = []
    hits = 0
    for path in files[: args.limit]:
        with np.load(path, allow_pickle=False) as data:
            memory_indices = data["memory_indices"].astype(int).tolist() if "memory_indices" in data else []
            candidate_indices = (
                data["candidate_history_indices"].astype(int).tolist()
                if "candidate_history_indices" in data
                else []
            )
            matched = int(data["matched_history_t"]) if "matched_history_t" in data else None
            label = int(data["matched_history_index"]) if "matched_history_index" in data else None
            match_valid = int(data["match_valid"]) if "match_valid" in data else None
            hit = matched in candidate_indices if matched is not None else False
            hits += int(hit)
            inspected.append(
                {
                    "file": path.name,
                    "matched_history_t": matched,
                    "matched_history_index": label,
                    "match_valid": match_valid,
                    "memory_indices": memory_indices,
                    "candidate_history_indices": candidate_indices,
                    "matched_in_candidates": hit,
                }
            )

    summary = {
        "cache_root": str(Path(args.cache_root)),
        "split": args.split,
        "samples": len(files),
        "inspected": len(inspected),
        "matched_candidate_hits": hits,
        "first": inspected[:3],
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    if args.require_consistent_match_valid:
        bad = [
            row
            for row in inspected
            if row["match_valid"] is None or bool(row["match_valid"]) != bool(row["matched_in_candidates"])
        ]
        if bad:
            raise SystemExit(f"match_valid inconsistent for {len(bad)}/{len(inspected)} inspected samples")


if __name__ == "__main__":
    main()
