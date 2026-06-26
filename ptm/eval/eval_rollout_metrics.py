from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect rollout metrics produced by WorldMem validation.")
    parser.add_argument("--metrics_json", required=True)
    parser.add_argument("--out", default="outputs/ptm/rollout_metrics.json")
    args = parser.parse_args()
    with Path(args.metrics_json).open("r", encoding="utf-8") as f:
        metrics = json.load(f)
    selected = {
        key: float(metrics[key])
        for key in ("lpips", "psnr", "mse", "dino_similarity", "clip_similarity")
        if key in metrics
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with Path(args.out).open("w", encoding="utf-8") as f:
        json.dump(selected, f, indent=2, sort_keys=True)
    print(selected)


if __name__ == "__main__":
    main()
