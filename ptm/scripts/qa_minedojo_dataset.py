from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def episode_dirs(root: Path, split: str) -> list[Path]:
    split_root = root / split
    return sorted(path for path in split_root.glob("episode_*") if path.is_dir())


def line_count(path: Path) -> int:
    if not path.exists():
        return -1
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for _ in handle)


def inspect_episode(path: Path, height: int, width: int, frames: int) -> tuple[dict, list[np.ndarray]]:
    result = {"episode": path.name, "ok": True, "errors": []}
    frame_samples: list[np.ndarray] = []
    metadata_path = path / "metadata.json"
    video_path = path / "frames.mp4"
    if not metadata_path.exists():
        result["ok"] = False
        result["errors"].append("missing metadata.json")
    else:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        result["metadata"] = metadata
        if metadata.get("height") != height or metadata.get("width") != width:
            result["ok"] = False
            result["errors"].append(f"metadata_size={metadata.get('height')}x{metadata.get('width')}")
        if metadata.get("backend") != "minedojo":
            result["ok"] = False
            result["errors"].append(f"backend={metadata.get('backend')}")

    for name in ("actions.jsonl", "poses.jsonl", "tests.jsonl"):
        count = line_count(path / name)
        result[f"{name}_lines"] = count
        if count <= 0:
            result["ok"] = False
            result["errors"].append(f"bad {name} lines={count}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        result["ok"] = False
        result["errors"].append("cannot open frames.mp4")
        return result, frame_samples

    reported_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    reported_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    reported_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    result["video"] = {
        "frames": reported_frames,
        "height": reported_height,
        "width": reported_width,
    }
    if reported_frames != frames or reported_height != height or reported_width != width:
        result["ok"] = False
        result["errors"].append(f"video_shape={reported_frames}x{reported_height}x{reported_width}")

    sample_indices = sorted(set([0, frames // 4, frames // 2, (frames * 3) // 4, frames - 1]))
    means = []
    prev = None
    diffs = []
    all_frames = 0
    static_pairs = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        idx = all_frames
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        means.append(float(rgb.mean()))
        if prev is not None:
            diff = float(np.mean(np.abs(rgb.astype(np.float32) - prev.astype(np.float32))))
            diffs.append(diff)
            if diff < 0.5:
                static_pairs += 1
        if idx in sample_indices:
            thumb = cv2.resize(rgb, (160, 90), interpolation=cv2.INTER_AREA)
            frame_samples.append(thumb)
        prev = rgb
        all_frames += 1
    cap.release()

    result["decoded_frames"] = all_frames
    result["mean_pixel"] = float(np.mean(means)) if means else 0.0
    result["mean_abs_frame_diff"] = float(np.mean(diffs)) if diffs else 0.0
    result["static_pair_ratio"] = static_pairs / max(1, len(diffs))
    if all_frames != frames:
        result["ok"] = False
        result["errors"].append(f"decoded_frames={all_frames}")
    if result["mean_pixel"] < 3.0 or result["mean_pixel"] > 252.0:
        result["ok"] = False
        result["errors"].append(f"mean_pixel={result['mean_pixel']:.2f}")
    if result["static_pair_ratio"] > 0.98:
        result["ok"] = False
        result["errors"].append(f"static_pair_ratio={result['static_pair_ratio']:.3f}")
    return result, frame_samples


def write_contact_sheet(samples: list[tuple[str, np.ndarray]], out_path: Path) -> None:
    if not samples:
        return
    cell_h, cell_w = samples[0][1].shape[:2]
    cols = 5
    rows = (len(samples) + cols - 1) // cols
    sheet = np.zeros((rows * cell_h, cols * cell_w, 3), dtype=np.uint8)
    for idx, (_, image) in enumerate(samples):
        row, col = divmod(idx, cols)
        sheet[row * cell_h : (row + 1) * cell_h, col * cell_w : (col + 1) * cell_w] = image
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR))


def main() -> None:
    parser = argparse.ArgumentParser(description="QA generated MineDojo PTM episodes.")
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--height", type=int, default=360)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--frames", type=int, default=128)
    parser.add_argument("--sample", type=int, default=16)
    parser.add_argument("--report", required=True)
    parser.add_argument("--contact_sheet", required=True)
    args = parser.parse_args()

    root = Path(args.data_root)
    episodes = episode_dirs(root, args.split)
    selected = episodes[: args.sample]
    results = []
    sheet_samples: list[tuple[str, np.ndarray]] = []
    for episode in selected:
        result, frames = inspect_episode(episode, args.height, args.width, args.frames)
        results.append(result)
        for frame in frames:
            sheet_samples.append((episode.name, frame))

    report = {
        "data_root": str(root),
        "split": args.split,
        "episodes_total": len(episodes),
        "episodes_checked": len(results),
        "ok": all(item["ok"] for item in results) and bool(results),
        "results": results,
    }
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    write_contact_sheet(sheet_samples, Path(args.contact_sheet))
    print(json.dumps({k: report[k] for k in ("data_root", "episodes_total", "episodes_checked", "ok")}, sort_keys=True))
    if not report["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
