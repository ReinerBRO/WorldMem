from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from ptm.data.minedojo_generator import _expected_labels
from ptm.data.schemas import EventRecord, PoseRecord, TestRecord, oasis_action_to_vector, yaw_distance, write_jsonl
from ptm.data.window_filters import has_pose_discontinuity


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _read_metadata(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _write_metadata(path: Path, metadata: dict[str, Any]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, sort_keys=True, indent=2)
        f.write("\n")
    tmp.replace(path)


def _episode_dirs(data_root: Path, split: str) -> list[Path]:
    split_dir = data_root / split
    if not split_dir.exists():
        return []
    return sorted(
        path
        for path in split_dir.glob("episode_*")
        if path.is_dir() and not path.name.endswith(".lock")
    )


def _load_poses(path: Path) -> list[PoseRecord]:
    return [PoseRecord(**record) for record in _read_jsonl(path)]


def _load_events(path: Path) -> list[EventRecord]:
    return [EventRecord(**record) for record in _read_jsonl(path)]


def _future_motion_stats(
    poses: list[PoseRecord],
    actions: list[dict[str, Any]],
    query_t: int,
    target_t: int,
) -> dict[str, float]:
    start = max(0, min(int(query_t), len(poses) - 1))
    end = max(start, min(int(target_t), len(poses) - 1))
    path_length = 0.0
    yaw_path = 0.0
    for idx in range(start + 1, end + 1):
        prev = poses[idx - 1]
        cur = poses[idx]
        path_length += float(np.sqrt((cur.x - prev.x) ** 2 + (cur.z - prev.z) ** 2))
        yaw_path += float(yaw_distance(cur.yaw, prev.yaw))

    action_end = min(end, len(actions) - 1)
    action_start = min(start, action_end)
    nonzero_actions = 0
    action_count = max(1, action_end - action_start + 1)
    for record in actions[action_start : action_end + 1]:
        vector = oasis_action_to_vector(record.get("oasis_action", {}))
        nonzero_actions += int(bool(np.abs(vector).sum() > 0))

    return {
        "future_path_length": float(path_length),
        "future_yaw_path": float(yaw_path),
        "future_action_rate": float(nonzero_actions / action_count),
    }


def _yaw_distance_array(values: np.ndarray, target: float) -> np.ndarray:
    diff = np.abs(values.astype(np.float64) - float(target)) % 360.0
    return np.minimum(diff, 360.0 - diff)


def _build_dense_tests_fast(
    poses: list[PoseRecord],
    family: str,
    events: list[EventRecord],
    *,
    frames_per_episode: int,
    expected_labels: dict[str, object],
    history_length: int,
    future_length: int,
    stride: int,
    actions: list[dict[str, Any]] | None = None,
    filter_static_future: bool = False,
    min_future_path: float = 0.25,
    min_future_yaw_path: float = 5.0,
    min_future_action_rate: float = 0.05,
) -> list[TestRecord]:
    if frames_per_episode <= 1:
        raise ValueError("frames_per_episode must be > 1")
    if len(poses) < frames_per_episode:
        raise ValueError("poses must contain at least frames_per_episode records")

    xs = np.asarray([pose.x for pose in poses[:frames_per_episode]], dtype=np.float64)
    ys = np.asarray([pose.y for pose in poses[:frames_per_episode]], dtype=np.float64)
    zs = np.asarray([pose.z for pose in poses[:frames_per_episode]], dtype=np.float64)
    yaws = np.asarray([pose.yaw for pose in poses[:frames_per_episode]], dtype=np.float64)

    max_query = frames_per_episode - future_length - 1
    first_query = min(history_length, max_query)
    if first_query < 1:
        first_query = max(0, frames_per_episode // 2)
    query_times = list(range(first_query, max_query + 1, max(1, stride)))
    if not query_times:
        query_times = [max(0, min(frames_per_episode - 2, first_query))]

    expected = expected_labels or {}
    tests: list[TestRecord] = []
    for query_t in query_times:
        history_start = max(0, query_t - history_length + 1)
        history_end = query_t
        future_start = query_t
        future_end = min(frames_per_episode - 1, query_t + future_length)
        target_t = future_end
        if has_pose_discontinuity(poses, history_start, future_end):
            continue
        motion = _future_motion_stats(poses, actions or [], query_t, target_t)
        is_static_future = (
            motion["future_path_length"] < float(min_future_path)
            and motion["future_yaw_path"] < float(min_future_yaw_path)
            and motion["future_action_rate"] < float(min_future_action_rate)
        )
        if filter_static_future and is_static_future:
            continue

        hist_slice = slice(history_start, history_end + 1)
        pose_dist = np.sqrt((xs[hist_slice] - xs[target_t]) ** 2 + (zs[hist_slice] - zs[target_t]) ** 2)
        yaw_dist = _yaw_distance_array(yaws[hist_slice], float(yaws[target_t]))
        scores = pose_dist + yaw_dist / 180.0
        best_rel = int(np.argmin(scores))
        matched_t = history_start + best_rel

        labels: dict[str, object] = {
            "target_frame_idx": target_t,
            "matched_history_t": matched_t,
            "matched_history_index": matched_t - history_start,
            "pose_distance": float(pose_dist[best_rel]),
            "yaw_distance": float(yaw_dist[best_rel]),
            "returns_to_seen_place": bool(expected.get("returns_to_seen_place", False)),
            "landmark_visible": bool(expected.get("landmark_visible", False)),
            "object_exists_at_return": bool(expected.get("object_exists_at_return", False)),
        }
        labels.update(motion)
        labels["future_static_filtered_candidate"] = bool(is_static_future)

        if family.startswith("loop_return"):
            test_type = "loop_return"
        elif family.startswith("landmark_revisit"):
            test_type = "landmark_revisit"
            event = events[0] if events else None
            event_verified = bool(
                event
                and (event.success_verified_by_voxel or event.success_verified_by_inventory_delta)
            )
            labels["landmark_event_verified"] = event_verified
            labels["landmark_id"] = "landmark_0"
            labels["landmark_type"] = event.block_type if event else "unknown"
            labels["landmark_position"] = event.target_block_position if event else None
        elif family.startswith("object_persistence"):
            test_type = "object_persistence"
            event = events[0] if events else None
            event_verified = bool(
                event
                and (event.success_verified_by_voxel or event.success_verified_by_inventory_delta)
            )
            labels["object_event_verified"] = event_verified
            labels["event_type"] = event.event_type if event else "place_block"
            labels["block_type"] = event.block_type if event else "unknown"
            labels["target_block_position"] = event.target_block_position if event else None
        else:
            test_type = "normal_rollout"
            labels["local_pose_delta"] = {
                "dx": float(xs[target_t] - xs[query_t]),
                "dy": float(ys[target_t] - ys[query_t]),
                "dz": float(zs[target_t] - zs[query_t]),
                "dyaw": float(yaw_distance(float(yaws[target_t]), float(yaws[query_t]))),
            }

        tests.append(
            TestRecord(
                query_t=query_t,
                history_start_t=history_start,
                history_end_t=history_end,
                future_start_t=future_start,
                future_end_t=future_end,
                test_type=test_type,
                target_t=target_t,
                labels=labels,
            )
        )
    return tests


def rebuild_episode_tests(
    episode_dir: Path,
    *,
    history_length: int,
    future_length: int,
    stride: int,
    backup_suffix: str,
    dry_run: bool,
    filter_static_future: bool,
    min_future_path: float,
    min_future_yaw_path: float,
    min_future_action_rate: float,
) -> dict[str, Any]:
    metadata_path = episode_dir / "metadata.json"
    tests_path = episode_dir / "tests.jsonl"
    poses_path = episode_dir / "poses.jsonl"
    events_path = episode_dir / "events.jsonl"
    actions_path = episode_dir / "actions.jsonl"

    if not metadata_path.exists():
        raise FileNotFoundError(f"missing metadata.json: {episode_dir}")
    if not poses_path.exists():
        raise FileNotFoundError(f"missing poses.jsonl: {episode_dir}")

    metadata = _read_metadata(metadata_path)
    family = metadata.get("family") or metadata.get("ptm_family")
    if not family:
        raise ValueError(f"missing family/ptm_family in {metadata_path}")

    poses = _load_poses(poses_path)
    events = _load_events(events_path)
    actions = _read_jsonl(actions_path)
    frames_per_episode = int(metadata.get("frames_per_episode") or len(poses))
    expected = metadata.get("expected") or _expected_labels(str(family))

    tests = _build_dense_tests_fast(
        poses=poses,
        family=str(family),
        events=events,
        frames_per_episode=frames_per_episode,
        expected_labels=expected,
        history_length=history_length,
        future_length=future_length,
        stride=stride,
        actions=actions,
        filter_static_future=filter_static_future,
        min_future_path=min_future_path,
        min_future_yaw_path=min_future_yaw_path,
        min_future_action_rate=min_future_action_rate,
    )
    old_count = len(_read_jsonl(tests_path))
    new_count = len(tests)
    test_types = Counter(test.test_type for test in tests)

    if not dry_run:
        if tests_path.exists() and backup_suffix:
            backup_path = tests_path.with_name(f"tests{backup_suffix}.jsonl")
            if not backup_path.exists():
                shutil.copy2(tests_path, backup_path)
        write_jsonl(tests_path, tests)
        metadata.update(
            {
                "ptm_tests_history_length": int(history_length),
                "ptm_tests_future_length": int(future_length),
                "ptm_tests_stride": int(stride),
                "ptm_tests_filter_static_future": bool(filter_static_future),
                "ptm_tests_min_future_path": float(min_future_path),
                "ptm_tests_min_future_yaw_path": float(min_future_yaw_path),
                "ptm_tests_min_future_action_rate": float(min_future_action_rate),
                "ptm_tests_count": int(new_count),
                "ptm_tests_rebuilt_at": datetime.now(timezone.utc).isoformat(),
            }
        )
        _write_metadata(metadata_path, metadata)

    return {
        "episode": str(episode_dir),
        "family": str(family),
        "old_count": old_count,
        "new_count": new_count,
        "test_types": dict(test_types),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Rebuild PTM future-test records densely for existing MineDojo long trajectories."
    )
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    parser.add_argument("--history-length", type=int, default=600)
    parser.add_argument("--future-length", type=int, default=100)
    parser.add_argument("--stride", type=int, default=8)
    parser.add_argument("--backup-suffix", default=".sparse16_backup")
    parser.add_argument("--filter-static-future", action="store_true")
    parser.add_argument("--min-future-path", type=float, default=0.25)
    parser.add_argument("--min-future-yaw-path", type=float, default=5.0)
    parser.add_argument("--min-future-action-rate", type=float, default=0.05)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.history_length < 1:
        raise ValueError("--history-length must be >= 1")
    if args.future_length < 1:
        raise ValueError("--future-length must be >= 1")
    if args.stride < 1:
        raise ValueError("--stride must be >= 1")

    data_root = Path(args.data_root)
    totals: dict[str, Any] = {}

    for split in args.splits:
        episodes = _episode_dirs(data_root, split)
        split_types: Counter[str] = Counter()
        family_counts: Counter[str] = Counter()
        old_total = 0
        new_total = 0
        per_family_new: defaultdict[str, int] = defaultdict(int)

        for episode_dir in episodes:
            result = rebuild_episode_tests(
                episode_dir,
                history_length=args.history_length,
                future_length=args.future_length,
                stride=args.stride,
                backup_suffix=args.backup_suffix,
                dry_run=args.dry_run,
                filter_static_future=args.filter_static_future,
                min_future_path=args.min_future_path,
                min_future_yaw_path=args.min_future_yaw_path,
                min_future_action_rate=args.min_future_action_rate,
            )
            old_total += int(result["old_count"])
            new_total += int(result["new_count"])
            family = str(result["family"])
            family_counts[family] += 1
            per_family_new[family] += int(result["new_count"])
            split_types.update(result["test_types"])

        totals[split] = {
            "episodes": len(episodes),
            "old_tests": old_total,
            "new_tests": new_total,
            "test_types": dict(split_types),
            "families": dict(family_counts),
            "tests_per_family": dict(per_family_new),
        }

    print(json.dumps(totals, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
