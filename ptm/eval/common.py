from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Iterable


def read_jsonl(path: str | Path) -> list[dict]:
    records = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def binary_accuracy(records: Iterable[dict], pred_key: str, label_key: str) -> float:
    total = 0
    correct = 0
    for record in records:
        if pred_key not in record or label_key not in record:
            continue
        pred = float(record[pred_key]) >= 0.5
        label = bool(record[label_key])
        total += 1
        correct += int(pred == label)
    return correct / total if total else 0.0


def write_metrics(metrics: dict[str, float], out: str | Path) -> None:
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.suffix == ".csv":
        with out.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=sorted(metrics))
            writer.writeheader()
            writer.writerow(metrics)
    else:
        with out.open("w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2, sort_keys=True)
