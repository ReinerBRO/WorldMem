from __future__ import annotations

import argparse

from .common import binary_accuracy, read_jsonl, write_metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate PTM landmark persistence predictions.")
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--out", default="outputs/ptm/landmark_persistence_metrics.json")
    args = parser.parse_args()
    records = [r for r in read_jsonl(args.predictions) if r.get("test_type") == "landmark_revisit"]
    metrics = {
        "landmark_hit_rate": binary_accuracy(records, "landmark_visible_prob", "landmark_visible"),
        "num_landmark_records": float(len(records)),
    }
    write_metrics(metrics, args.out)
    print(metrics)


if __name__ == "__main__":
    main()
