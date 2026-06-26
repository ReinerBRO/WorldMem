from __future__ import annotations

import argparse

from .common import binary_accuracy, read_jsonl, write_metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate PTM object persistence predictions.")
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--out", default="outputs/ptm/object_persistence_metrics.json")
    args = parser.parse_args()
    records = [r for r in read_jsonl(args.predictions) if r.get("test_type") == "object_persistence"]
    metrics = {
        "object_persistence_accuracy": binary_accuracy(records, "object_exists_prob", "object_exists_at_return"),
        "num_object_records": float(len(records)),
    }
    write_metrics(metrics, args.out)
    print(metrics)


if __name__ == "__main__":
    main()
