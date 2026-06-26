from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def flatten_metrics(path: Path) -> dict[str, float | str]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    row: dict[str, float | str] = {"run": path.parent.name, "source": str(path)}
    for key, value in data.items():
        if isinstance(value, (int, float)):
            row[key] = float(value)
    return row


def markdown_table(rows: list[dict[str, float | str]], fields: list[str]) -> str:
    lines = ["| " + " | ".join(fields) + " |", "| " + " | ".join(["---"] * len(fields)) + " |"]
    for row in rows:
        values = []
        for field in fields:
            value = row.get(field, "")
            values.append(f"{value:.4f}" if isinstance(value, float) else str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize PTM eval JSON metrics into CSV and Markdown.")
    parser.add_argument("--runs", nargs="+", required=True)
    parser.add_argument("--out_dir", default="outputs/ptm/summary")
    args = parser.parse_args()

    metric_files: list[Path] = []
    for run in args.runs:
        path = Path(run)
        if path.is_file() and path.suffix == ".json":
            metric_files.append(path)
        else:
            metric_files.extend(sorted(path.glob("**/*metrics.json")))
            metric_files.extend(sorted(path.glob("**/summary.json")))
    rows = [flatten_metrics(path) for path in metric_files]
    fields = sorted({field for row in rows for field in row})
    if "run" in fields:
        fields.remove("run")
        fields.insert(0, "run")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "ptm_results.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    with (out_dir / "ptm_results.md").open("w", encoding="utf-8") as f:
        f.write(markdown_table(rows, fields))
    print(f"wrote {len(rows)} rows to {out_dir}")


if __name__ == "__main__":
    main()
