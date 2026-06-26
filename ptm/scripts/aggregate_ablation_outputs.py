#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _binary_bce(records: list[dict[str, Any]], prob_key: str, label_key: str, test_type_id: int) -> float | None:
    subset = [record for record in records if int(record.get("test_type_id", -1)) == int(test_type_id)]
    if not subset:
        return None
    probs = torch.tensor([float(record[prob_key]) for record in subset], dtype=torch.float32).clamp(1e-6, 1 - 1e-6)
    labels = torch.tensor([float(record[label_key]) for record in subset], dtype=torch.float32)
    return float(F.binary_cross_entropy(probs, labels).item())


def _binary_acc(records: list[dict[str, Any]], prob_key: str, label_key: str, test_type_id: int) -> float | None:
    subset = [record for record in records if int(record.get("test_type_id", -1)) == int(test_type_id)]
    if not subset:
        return None
    preds = torch.tensor([float(record[prob_key]) >= 0.5 for record in subset], dtype=torch.bool)
    labels = torch.tensor([float(record[label_key]) >= 0.5 for record in subset], dtype=torch.bool)
    return float((preds == labels).float().mean().item())


def _mean(records: list[dict[str, Any]], key: str) -> float | None:
    values = [float(record[key]) for record in records if key in record and record[key] is not None]
    if not values:
        return None
    return float(torch.tensor(values, dtype=torch.float32).mean().item())


def summarize_direct(mode_dir: Path) -> dict[str, Any] | None:
    files = sorted(mode_dir.rglob("ptm_eval/*future_test_predictions_step*.jsonl"))
    records: list[dict[str, Any]] = []
    for path in files:
        records.extend(_read_jsonl(path))
    if not records:
        return None
    valid_match = [record for record in records if bool(record.get("match_valid", False))]
    summary: dict[str, Any] = {
        "num_samples": len(records),
        "num_rank_files": len(files),
        "future_test_loss": _mean(records, "ptm_future_test_loss"),
        "future_embedding_mse": _mean(records, "future_embedding_mse"),
        "future_embedding_loss": _mean(records, "ptm_future_embedding_loss"),
        "loop_return_loss": _mean(records, "ptm_loop_return_loss"),
        "loop_return_bce": _binary_bce(records, "loop_return_prob", "returns_to_seen_place_label", 1),
        "loop_return_accuracy": _binary_acc(records, "loop_return_prob", "returns_to_seen_place_label", 1),
        "landmark_visible_loss": _mean(records, "ptm_landmark_visible_loss"),
        "landmark_visible_bce": _binary_bce(records, "landmark_visible_prob", "landmark_visible_label", 2),
        "landmark_visible_accuracy": _binary_acc(records, "landmark_visible_prob", "landmark_visible_label", 2),
        "object_exists_loss": _mean(records, "ptm_object_exists_loss"),
        "object_exists_bce": _binary_bce(records, "object_exists_prob", "object_exists_at_return_label", 3),
        "object_exists_accuracy": _binary_acc(records, "object_exists_prob", "object_exists_at_return_label", 3),
        "matched_history_loss": _mean(records, "ptm_matched_history_loss"),
        "matched_history_num_valid": len(valid_match),
        "matched_history_accuracy": None,
    }
    if valid_match:
        correct = [
            int(record["matched_history_index_pred"]) == int(record["matched_history_index_label"])
            for record in valid_match
        ]
        summary["matched_history_accuracy"] = float(torch.tensor(correct, dtype=torch.float32).mean().item())
    return summary


def _weighted_metric(payloads: list[dict[str, Any]], key: str, weight_key: str = "num_samples") -> float | None:
    weighted = []
    weights = []
    for payload in payloads:
        if payload is None or key not in payload or payload[key] is None:
            continue
        weight = int(payload.get(weight_key, 0))
        if weight <= 0:
            continue
        weighted.append(float(payload[key]) * weight)
        weights.append(weight)
    if not weights:
        return None
    return float(sum(weighted) / sum(weights))


def _aggregate_generation_scope(rank_payloads: list[dict[str, Any]], scope: str) -> dict[str, float | None] | None:
    metrics = []
    for payload in rank_payloads:
        value = payload.get(scope)
        if value is not None:
            metrics.append({"num_samples": payload.get("num_samples", 0), **value})
    if not metrics:
        return None
    return {
        "psnr": _weighted_metric(metrics, "psnr"),
        "mse": _weighted_metric(metrics, "mse"),
        "lpips": _weighted_metric(metrics, "lpips"),
    }


def summarize_generation(mode_dir: Path) -> dict[str, Any] | None:
    files = sorted(mode_dir.rglob("generation_eval/*generation_metrics_step*.json"))
    all_rank_files = [path for path in files if "_all_ranks_" in path.name]
    if all_rank_files:
        files = all_rank_files
    else:
        files = [path for path in files if "_all_ranks_" not in path.name]
    payloads = [json.loads(path.read_text(encoding="utf-8")) for path in files]
    if not payloads:
        return None
    summary: dict[str, Any] = {
        "num_samples": sum(int(payload.get("num_samples", 0)) for payload in payloads),
        "num_rank_files": len(files),
        "overall": _aggregate_generation_scope(payloads, "overall"),
        "target_window": _aggregate_generation_scope(payloads, "target_window"),
        "late_horizon": _aggregate_generation_scope(payloads, "late_horizon"),
        "subsets": {},
    }
    for subset_name in ("loop_return", "landmark_visible", "object_exists"):
        subset_payloads = []
        for payload in payloads:
            subset = payload.get("subsets", {}).get(subset_name, {})
            metrics = subset.get("metrics")
            if metrics is not None:
                subset_payloads.append({"num_samples": int(subset.get("num_samples", 0)), **metrics})
        summary["subsets"][subset_name] = {
            "num_samples": sum(int(item["num_samples"]) for item in subset_payloads),
            "metrics": {
                "psnr": _weighted_metric(subset_payloads, "psnr"),
                "mse": _weighted_metric(subset_payloads, "mse"),
                "lpips": _weighted_metric(subset_payloads, "lpips"),
            } if subset_payloads else None,
        }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("eval_root", type=Path)
    parser.add_argument("--kind", choices=("direct", "generation"), required=True)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--modes", default=None, help="comma-separated mode names; default auto-discover subdirs")
    args = parser.parse_args()

    if args.modes:
        modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    else:
        modes = sorted(d.name for d in args.eval_root.iterdir() if d.is_dir())
    summary = {}
    for mode in modes:
        mode_dir = args.eval_root / mode
        if args.kind == "direct":
            value = summarize_direct(mode_dir)
        else:
            value = summarize_generation(mode_dir)
        if value is not None:
            summary[mode] = value

    output = args.output or (args.eval_root / f"{args.kind}_summary.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(output)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
