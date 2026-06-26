from __future__ import annotations

import argparse

from .common import binary_accuracy, read_jsonl, write_metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate PTM loop revisit predictions.")
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--out", default="outputs/ptm/loop_revisit_metrics.json")
    args = parser.parse_args()
    records = [r for r in read_jsonl(args.predictions) if r.get("test_type") == "loop_return"]
    metrics = {
        "loop_return_accuracy": binary_accuracy(records, "loop_return_prob", "returns_to_seen_place"),
        "num_loop_records": float(len(records)),
    }
    write_metrics(metrics, args.out)
    print(metrics)


if __name__ == "__main__":
    main()
