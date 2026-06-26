from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .schemas import read_jsonl, required_episode_files

try:
    import cv2
except ImportError:  # pragma: no cover - depends on local env.
    cv2 = None

try:
    import imageio.v3 as iio
except ImportError:  # pragma: no cover - depends on local env.
    iio = None


def count_mp4_frames(path: Path) -> tuple[int | None, str | None]:
    if cv2 is not None:
        cap = cv2.VideoCapture(str(path))
        video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        return video_frames, None
    if iio is not None:
        try:
            return sum(1 for _ in iio.imiter(path)), None
        except Exception as exc:  # pragma: no cover - backend specific.
            return None, f"imageio could not read frames.mp4: {exc}"
    return None, "frames.mp4 exists but neither opencv-python nor imageio is available"


def verify_episode(episode_dir: Path, strict_events: bool = False) -> list[str]:
    errors: list[str] = []
    for name in required_episode_files():
        if not (episode_dir / name).exists():
            errors.append(f"missing {name}")
    if errors:
        return errors

    frames = read_jsonl(episode_dir / "frames_index.jsonl")
    actions = read_jsonl(episode_dir / "actions.jsonl")
    poses = read_jsonl(episode_dir / "poses.jsonl")
    events = read_jsonl(episode_dir / "events.jsonl")
    tests = read_jsonl(episode_dir / "tests.jsonl")
    metadata = json.loads((episode_dir / "metadata.json").read_text(encoding="utf-8"))
    if metadata.get("ptm_tests_filter_static_future") is not True:
        errors.append("metadata ptm_tests_filter_static_future is not true")

    if (episode_dir / "frames.mp4").exists():
        video_frames, error = count_mp4_frames(episode_dir / "frames.mp4")
        if error is not None or video_frames is None:
            errors.append(error or "frames.mp4 is unreadable")
            return errors
    elif (episode_dir / "frames.npz").exists():
        video_frames = int(np.load(episode_dir / "frames.npz")["frames"].shape[0])
    else:
        errors.append("missing compressed frames: expected frames.mp4 or frames.npz")
        return errors

    if video_frames <= 0:
        errors.append("frames.mp4 is empty or unreadable")
    expected = min(video_frames, len(frames), len(actions), len(poses))
    if len(frames) != video_frames:
        errors.append(f"frames_index count {len(frames)} != video frame count {video_frames}")
    if len(actions) != expected:
        errors.append(f"actions count {len(actions)} misaligned with expected {expected}")
    if len(poses) != expected:
        errors.append(f"poses count {len(poses)} misaligned with expected {expected}")
    if not tests:
        errors.append("tests.jsonl is empty")
    for i, pose in enumerate(poses):
        for key in ("x", "y", "z", "yaw", "pitch"):
            if key not in pose or pose[key] is None:
                errors.append(f"pose {i} missing {key}")
                break
    for i, test in enumerate(tests):
        target_t = int(test.get("target_t", -1))
        if target_t < 0 or target_t >= expected:
            errors.append(f"test {i} target_t {target_t} outside [0,{expected})")
        if "labels" not in test:
            errors.append(f"test {i} missing labels")
        labels = test.get("labels", {})
        if labels.get("future_static_filtered_candidate") is True:
            errors.append(f"test {i} is marked future_static_filtered_candidate")
        for key in ("future_path_length", "future_yaw_path", "future_action_rate"):
            if key not in labels:
                errors.append(f"test {i} missing {key}")
    if strict_events:
        for i, event in enumerate(events):
            event_type = event.get("event_type", "")
            if event_type.startswith(("place", "set", "delete")) and not (
                event.get("success_verified_by_voxel") or event.get("success_verified_by_inventory_delta")
            ):
                errors.append(f"event {i} block edit is not verified")
    return errors


def find_episodes(data_root: Path) -> list[Path]:
    required = required_episode_files()
    episodes = sorted(
        path
        for path in data_root.glob("**/episode_*")
        if path.is_dir()
        and not path.name.endswith(".lock")
        and all((path / name).exists() for name in required)
    )
    if not episodes and all((data_root / name).exists() for name in required_episode_files()):
        episodes = [data_root]
    return episodes


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify PTM episode dataset integrity.")
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--strict_events", action="store_true")
    args = parser.parse_args()

    data_root = Path(args.data_root)
    episodes = find_episodes(data_root)
    corrupted = []
    for episode in episodes:
        errors = verify_episode(episode, strict_events=args.strict_events)
        if errors:
            corrupted.append({"episode": str(episode), "errors": errors})

    summary = {
        "data_root": str(data_root),
        "episodes": len(episodes),
        "corrupted_episodes": len(corrupted),
        "corrupted": corrupted[:20],
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    if corrupted:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
