from __future__ import annotations

import argparse
import json
import math
import random
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

try:
    import cv2
except ImportError as exc:  # pragma: no cover
    raise SystemExit("opencv-python/cv2 is required for mp4 audit") from exc


REQUIRED_FILES = (
    "frames.mp4",
    "frames_index.jsonl",
    "actions.jsonl",
    "poses.jsonl",
    "events.jsonl",
    "tests.jsonl",
    "metadata.json",
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def complete_episode(path: Path) -> bool:
    if not all((path / name).is_file() for name in REQUIRED_FILES):
        return False
    return (path / "frames.mp4").stat().st_size > 0 and (path / "metadata.json").stat().st_size > 0


def episode_id(path: Path) -> int:
    return int(path.name.split("_")[-1])


def select_sample(episodes: list[Path], n: int, seed: int) -> list[Path]:
    if len(episodes) <= n:
        return episodes
    # Mix uniform coverage and random coverage so the sample includes early/mid/late episodes.
    stride_pick = [episodes[round(i * (len(episodes) - 1) / (n - 1))] for i in range(n)]
    rng = random.Random(seed)
    random_pick = rng.sample(episodes, n)
    selected: dict[int, Path] = {}
    for path in stride_pick[: n // 2] + random_pick:
        selected[episode_id(path)] = path
        if len(selected) >= n:
            break
    if len(selected) < n:
        for path in stride_pick:
            selected[episode_id(path)] = path
            if len(selected) >= n:
                break
    return [selected[key] for key in sorted(selected)]


def read_video(path: Path) -> tuple[list[np.ndarray], dict[str, Any]]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"failed to open video: {path}")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    reported_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    frames: list[np.ndarray] = []
    while True:
        ok, frame_bgr = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    cap.release()
    return frames, {
        "reported_frames": reported_frames,
        "decoded_frames": len(frames),
        "width": width,
        "height": height,
        "fps": fps,
    }


def frame_metrics(frames: list[np.ndarray]) -> dict[str, Any]:
    if not frames:
        return {"error": "no decoded frames"}
    arr = np.stack(frames).astype(np.float32)
    means = arr.mean(axis=(1, 2, 3))
    stds = arr.std(axis=(1, 2, 3))
    if len(frames) > 1:
        diffs = np.mean(np.abs(arr[1:] - arr[:-1]), axis=(1, 2, 3))
    else:
        diffs = np.asarray([], dtype=np.float32)
    return {
        "mean_min": float(means.min()),
        "mean_max": float(means.max()),
        "mean_avg": float(means.mean()),
        "std_min": float(stds.min()),
        "std_avg": float(stds.mean()),
        "diff_mean": float(diffs.mean()) if len(diffs) else 0.0,
        "diff_p05": float(np.percentile(diffs, 5)) if len(diffs) else 0.0,
        "diff_p95": float(np.percentile(diffs, 95)) if len(diffs) else 0.0,
        "near_black_frames": int((means < 5.0).sum()),
        "near_white_frames": int((means > 250.0).sum()),
        "low_texture_frames": int((stds < 3.0).sum()),
        "near_frozen_transitions": int((diffs < 0.5).sum()) if len(diffs) else 0,
    }


def action_metrics(actions: list[dict[str, Any]]) -> dict[str, Any]:
    nonzero_counter: Counter[str] = Counter()
    nonzero_rows = 0
    compact_lengths: Counter[int] = Counter()
    for row in actions:
        oasis = row.get("oasis_action", {})
        row_nonzero = []
        for key, value in oasis.items():
            try:
                v = float(value)
            except Exception:
                continue
            if abs(v) > 1e-6:
                nonzero_counter[key] += 1
                row_nonzero.append(key)
        if row_nonzero:
            nonzero_rows += 1
        compact = row.get("minedojo_action", {}).get("compact")
        if isinstance(compact, list):
            compact_lengths[len(compact)] += 1
    return {
        "rows": len(actions),
        "nonzero_rows": nonzero_rows,
        "nonzero_keys": dict(sorted(nonzero_counter.items())),
        "compact_lengths": dict(sorted(compact_lengths.items())),
    }


def put_label(img: np.ndarray, text: str) -> np.ndarray:
    out = img.copy()
    cv2.rectangle(out, (0, 0), (out.shape[1], 18), (0, 0, 0), thickness=-1)
    cv2.putText(out, text, (4, 13), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def make_contact_sheets(sample_reports: list[dict[str, Any]], output_dir: Path, rows_per_sheet: int = 10) -> list[str]:
    sheet_paths: list[str] = []
    thumb_w, thumb_h = 160, 90
    frame_slots = [0, 32, 64, 96, 127]
    for sheet_idx in range(math.ceil(len(sample_reports) / rows_per_sheet)):
        chunk = sample_reports[sheet_idx * rows_per_sheet : (sheet_idx + 1) * rows_per_sheet]
        rows = []
        for report in chunk:
            video_path = Path(report["episode_dir"]) / "frames.mp4"
            frames, _ = read_video(video_path)
            thumbs = []
            for idx in frame_slots:
                frame = frames[min(idx, len(frames) - 1)] if frames else np.zeros((thumb_h, thumb_w, 3), dtype=np.uint8)
                bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                thumb = cv2.resize(bgr, (thumb_w, thumb_h), interpolation=cv2.INTER_AREA)
                label = f"ep{report['episode_id']} t{idx}"
                thumbs.append(put_label(thumb, label))
            rows.append(np.concatenate(thumbs, axis=1))
        sheet = np.concatenate(rows, axis=0)
        path = output_dir / f"sample50_contact_sheet_{sheet_idx:02d}.jpg"
        cv2.imwrite(str(path), sheet, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
        sheet_paths.append(str(path))
    return sheet_paths


def audit_episode(path: Path, expected_frames: int, expected_height: int, expected_width: int) -> dict[str, Any]:
    errors: list[str] = []
    missing_files = [name for name in REQUIRED_FILES if not (path / name).is_file()]
    if missing_files:
        errors.append(f"missing_files:{missing_files}")

    metadata = json.loads((path / "metadata.json").read_text(encoding="utf-8"))
    frames_index = read_jsonl(path / "frames_index.jsonl")
    actions = read_jsonl(path / "actions.jsonl")
    poses = read_jsonl(path / "poses.jsonl")
    events = read_jsonl(path / "events.jsonl")
    tests = read_jsonl(path / "tests.jsonl")
    frames, video = read_video(path / "frames.mp4")
    visual = frame_metrics(frames)
    action = action_metrics(actions)

    if video["decoded_frames"] != expected_frames:
        errors.append(f"decoded_frames:{video['decoded_frames']}!=expected:{expected_frames}")
    if video["height"] != expected_height or video["width"] != expected_width:
        errors.append(f"resolution:{video['height']}x{video['width']}!=expected:{expected_height}x{expected_width}")
    for name, rows in (("frames_index", frames_index), ("actions", actions), ("poses", poses)):
        if len(rows) != expected_frames:
            errors.append(f"{name}_rows:{len(rows)}!=expected:{expected_frames}")
    if not tests:
        errors.append("tests_empty")
    if visual.get("near_black_frames", 0) > 0:
        errors.append(f"near_black_frames:{visual['near_black_frames']}")
    if visual.get("low_texture_frames", 0) > expected_frames // 4:
        errors.append(f"many_low_texture_frames:{visual['low_texture_frames']}")
    if visual.get("near_frozen_transitions", 0) > int(expected_frames * 0.9):
        errors.append(f"near_frozen_transitions:{visual['near_frozen_transitions']}")
    if action["nonzero_rows"] == 0:
        errors.append("all_actions_zero")

    return {
        "episode_id": episode_id(path),
        "episode_dir": str(path),
        "metadata": {
            "backend": metadata.get("backend"),
            "family": metadata.get("family"),
            "seed": metadata.get("seed"),
            "seed_base": metadata.get("seed_base"),
            "episode_attempt": metadata.get("episode_attempt"),
            "frames_per_episode": metadata.get("frames_per_episode"),
            "height": metadata.get("height"),
            "width": metadata.get("width"),
            "frame_storage": metadata.get("frame_storage"),
            "reuse_env": metadata.get("reuse_env"),
        },
        "line_counts": {
            "frames_index": len(frames_index),
            "actions": len(actions),
            "poses": len(poses),
            "events": len(events),
            "tests": len(tests),
        },
        "video": video,
        "visual": visual,
        "action": action,
        "errors": errors,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", default="ptm_minedojo_data/stage1_360x640")
    parser.add_argument("--split", default="train")
    parser.add_argument("--sample_size", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20260622)
    parser.add_argument("--expected_frames", type=int, default=128)
    parser.add_argument("--expected_height", type=int, default=360)
    parser.add_argument("--expected_width", type=int, default=640)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    data_root = Path(args.data_root)
    split_root = data_root / args.split
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    episode_dirs = sorted((p for p in split_root.glob("episode_*") if p.is_dir()), key=episode_id)
    complete = [p for p in episode_dirs if complete_episode(p)]
    missing_ids = []
    if complete:
        complete_ids = {episode_id(p) for p in complete}
        for idx in range(max(complete_ids) + 1):
            if idx not in complete_ids:
                missing_ids.append(idx)
    sample = select_sample(complete, args.sample_size, args.seed)

    sample_reports = []
    for path in sample:
        sample_reports.append(audit_episode(path, args.expected_frames, args.expected_height, args.expected_width))

    aggregate_actions: Counter[str] = Counter()
    families: Counter[str] = Counter()
    failed = []
    for report in sample_reports:
        families[str(report["metadata"].get("family"))] += 1
        aggregate_actions.update(report["action"]["nonzero_keys"])
        if report["errors"]:
            failed.append({"episode_id": report["episode_id"], "errors": report["errors"]})

    contact_sheets = make_contact_sheets(sample_reports, output_dir)
    summary = {
        "data_root": str(data_root),
        "split": args.split,
        "episode_dirs": len(episode_dirs),
        "complete_episodes": len(complete),
        "missing_ids_up_to_max_complete": missing_ids,
        "sample_size": len(sample_reports),
        "sample_ids": [r["episode_id"] for r in sample_reports],
        "sample_failed": failed,
        "sample_passed": len(failed) == 0,
        "families": dict(sorted(families.items())),
        "aggregate_action_nonzero_counts": dict(sorted(aggregate_actions.items())),
        "contact_sheets": contact_sheets,
    }
    report = {"summary": summary, "episodes": sample_reports}
    report_path = output_dir / "sample50_audit_report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"REPORT {report_path}")


if __name__ == "__main__":
    main()
