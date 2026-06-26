from __future__ import annotations

import argparse
import hashlib
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .ptm_dataset import _episode_dirs
from .schemas import TEST_TYPE_TO_ID, oasis_action_to_vector, read_jsonl
from .window_filters import has_pose_discontinuity

try:
    import cv2
except ImportError:  # pragma: no cover - depends on runtime env.
    cv2 = None

try:
    import imageio.v3 as iio
except ImportError:  # pragma: no cover - depends on runtime env.
    iio = None


@dataclass(frozen=True)
class CacheTask:
    split: str
    episode_dir: str
    sample_offset: int
    out_dir: str
    height: int
    width: int
    context_length: int
    future_length: int
    test_future_action_length: int
    memory_condition_length: int
    max_history_candidates: int
    skip_existing: bool
    compressed: bool
    full_decode: bool
    memory_strategy: str
    window_centers: tuple[str, ...]
    selected_test_indices: tuple[int, ...] | None = None


def _pose_records_to_array(records: list[dict[str, Any]]) -> np.ndarray:
    rows = []
    for record in records:
        rows.append(
            [
                float(record.get("x", 0.0)),
                float(record.get("y", 0.0)),
                float(record.get("z", 0.0)),
                float(record.get("pitch", 0.0)),
                float(record.get("yaw", 0.0)),
            ]
        )
    return np.asarray(rows, dtype=np.float32)


def _pad_or_trim_array(array: np.ndarray, length: int) -> np.ndarray:
    if array.shape[0] == length:
        return array
    if array.shape[0] > length:
        return array[-length:]
    pad_shape = (length - array.shape[0],) + tuple(array.shape[1:])
    return np.concatenate([np.zeros(pad_shape, dtype=array.dtype), array], axis=0)


def _pad_or_trim_indices(indices: list[int], length: int) -> list[int]:
    if len(indices) == length:
        return indices
    if len(indices) > length:
        return indices[-length:]
    return [-1] * (length - len(indices)) + indices


def _slice_and_pad(array: np.ndarray, start: int, end: int, length: int) -> np.ndarray:
    if end < start:
        return _pad_or_trim_array(array[:0], length)
    return _pad_or_trim_array(array[start : end + 1], length)


def _timestamp_slice(start: int, end: int, length: int) -> np.ndarray:
    if end < start:
        values = np.zeros((0,), dtype=np.int64)
    else:
        values = np.arange(start, end + 1, dtype=np.int64)
    return _pad_or_trim_array(values, length)


def _normalize_rgb_frame(frame: np.ndarray, height: int, width: int) -> np.ndarray:
    if frame.ndim == 2:
        frame = np.repeat(frame[..., None], 3, axis=2)
    if frame.shape[0] in (1, 3, 4) and frame.shape[-1] not in (3, 4):
        frame = np.transpose(frame, (1, 2, 0))
    if frame.shape[-1] > 3:
        frame = frame[..., :3]
    if frame.shape[-1] != 3:
        raise ValueError(f"expected RGB frame, got shape {frame.shape}")
    if frame.shape[0] != height or frame.shape[1] != width:
        if cv2 is None:
            raise RuntimeError("opencv-python is required to resize cached frames")
        frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_LINEAR)
    if frame.dtype != np.uint8:
        frame = np.clip(frame, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(frame)


def _read_frame(cap: Any, frame_idx: int, height: int, width: int) -> np.ndarray:
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
    ok, frame_bgr = cap.read()
    if not ok or frame_bgr is None:
        raise RuntimeError(f"failed to read frame {frame_idx} from video")
    if frame_bgr.shape[0] != height or frame_bgr.shape[1] != width:
        frame_bgr = cv2.resize(frame_bgr, (width, height), interpolation=cv2.INTER_LINEAR)
    return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)


def _decode_mp4(path: Path, height: int, width: int) -> np.ndarray:
    frames = []
    if cv2 is not None:
        cap = cv2.VideoCapture(str(path))
        try:
            while cap.isOpened():
                ok, frame_bgr = cap.read()
                if not ok or frame_bgr is None:
                    break
                if frame_bgr.shape[0] != height or frame_bgr.shape[1] != width:
                    frame_bgr = cv2.resize(frame_bgr, (width, height), interpolation=cv2.INTER_LINEAR)
                frames.append(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        finally:
            cap.release()
    elif iio is not None:
        for frame in iio.imiter(path):
            frames.append(_normalize_rgb_frame(frame, height, width))
    else:
        raise RuntimeError("opencv-python or imageio is required to decode mp4 episodes")
    if not frames:
        raise RuntimeError(f"opencv returned no frames for {path}")
    return np.stack(frames, axis=0).astype(np.uint8, copy=False)


def _read_frame_map(frame_source: Any, frame_indices: list[int], height: int, width: int) -> dict[int, np.ndarray]:
    frames: dict[int, np.ndarray] = {}
    for frame_idx in sorted({int(i) for i in frame_indices if int(i) >= 0}):
        if isinstance(frame_source, np.ndarray):
            if frame_idx >= frame_source.shape[0]:
                raise IndexError(f"frame index {frame_idx} >= decoded frame count {frame_source.shape[0]}")
            frames[frame_idx] = _normalize_rgb_frame(frame_source[frame_idx], height, width)
        else:
            frames[frame_idx] = _read_frame(frame_source, frame_idx, height, width)
    return frames


def _stack_frames(indices: list[int], frame_map: dict[int, np.ndarray], height: int, width: int) -> np.ndarray:
    zero = np.zeros((height, width, 3), dtype=np.uint8)
    frames = []
    for index in indices:
        frame_idx = int(index)
        if frame_idx < 0:
            frames.append(zero)
            continue
        if frame_idx not in frame_map:
            raise KeyError(f"missing decoded frame {frame_idx}")
        frames.append(frame_map[frame_idx])
    return np.stack(frames, axis=0)


def _clamp_frame(frame_idx: int, num_frames: int, max_t: int | None = None) -> int:
    upper = num_frames - 1 if max_t is None else min(num_frames - 1, max_t)
    return max(0, min(int(frame_idx), upper))


def _unique_preserving_order(values: list[int]) -> list[int]:
    seen = set()
    out = []
    for value in values:
        value = int(value)
        if value in seen:
            continue
        seen.add(value)
        out.append(value)
    return out


def _future_motion_score(test: dict[str, Any]) -> float:
    labels = test.get("labels", {})
    path = float(labels.get("future_path_length", 0.0))
    yaw = float(labels.get("future_yaw_path", 0.0))
    action_rate = float(labels.get("future_action_rate", 0.0))
    return action_rate * 1000.0 + path + yaw / 180.0


def _window_center_t(test: dict[str, Any], spec: str, num_frames: int) -> tuple[str, int] | None:
    query_t = int(test["query_t"])
    target_t = int(test["target_t"])
    future_end_t = min(num_frames - 1, int(test.get("future_end_t", target_t)))
    spec = str(spec).strip().lower().replace("-", "_")
    if spec == "target":
        center_t = target_t
        window_kind = "target"
    elif spec.startswith("late"):
        suffix = spec.removeprefix("late").lstrip("_")
        if not suffix:
            raise ValueError(f"late window spec must include an offset, got {spec!r}")
        center_t = query_t + int(suffix)
        window_kind = f"late_{int(suffix)}"
    else:
        raise ValueError(f"unknown window center spec {spec!r}; expected target or late<offset>")
    if center_t < 0 or center_t > future_end_t or center_t >= num_frames:
        return None
    return window_kind, center_t


def _late_offset_from_spec(spec: str) -> int | None:
    spec = str(spec).strip().lower().replace("-", "_")
    if not spec.startswith("late"):
        return None
    suffix = spec.removeprefix("late").lstrip("_")
    if not suffix:
        raise ValueError(f"late window spec must include an offset, got {spec!r}")
    return int(suffix)


def _event_candidate_times(events: list[dict[str, Any]], query_t: int, num_frames: int) -> list[int]:
    candidates = []
    for event in events:
        try:
            event_t = int(event.get("t", -1))
        except (TypeError, ValueError):
            continue
        if event_t < 0 or event_t > query_t:
            continue
        candidates.append(_clamp_frame(event_t, num_frames, query_t))
        if event_t + 1 <= query_t:
            candidates.append(_clamp_frame(event_t + 1, num_frames, query_t))
    return candidates


def _select_memory_indices(
    test: dict[str, Any],
    events: list[dict[str, Any]],
    num_frames: int,
    start: int,
    query_t: int,
    memory_length: int,
    strategy: str,
) -> list[int]:
    if memory_length <= 0:
        return []
    if strategy != "causal_slots":
        raise ValueError(f"unknown memory strategy {strategy!r}; expected causal_slots")

    history_start = _clamp_frame(int(test.get("history_start_t", 0)), num_frames, query_t)
    history_end = _clamp_frame(int(test.get("history_end_t", query_t)), num_frames, query_t)
    span_end = max(history_start, min(history_end, query_t))

    priority = []
    priority.extend(_event_candidate_times(events, query_t, num_frames))
    priority.extend(
        [
            history_start,
            history_start + (span_end - history_start) // 4,
            (history_start + span_end) // 2,
            history_start + (span_end - history_start) * 3 // 4,
            max(history_start, query_t - 1),
            query_t,
        ]
    )
    priority = [_clamp_frame(value, num_frames, query_t) for value in priority]

    if span_end <= history_start:
        anchors = [history_start]
    else:
        anchors = [int(round(value)) for value in np.linspace(history_start, span_end, num=max(memory_length * 2, 2))]
    candidates = _unique_preserving_order(priority + [_clamp_frame(value, num_frames, query_t) for value in anchors])

    selected = candidates[:memory_length]
    if len(selected) < memory_length:
        fill = selected[-1] if selected else history_start
        selected.extend([fill] * (memory_length - len(selected)))
    return sorted(selected[:memory_length])


def _history_match_label(
    matched_t: int,
    memory_indices: list[int],
    main_indices: list[int],
    history_context_end_index: int,
    max_history_candidates: int,
) -> tuple[int, int, list[int]]:
    if history_context_end_index >= 0:
        context_indices = [
            int(index)
            for index in main_indices[: history_context_end_index + 1]
            if int(index) >= 0
        ]
    else:
        context_indices = []
    candidates = _unique_preserving_order(
        [int(index) for index in memory_indices if int(index) >= 0] + context_indices
    )
    candidates = candidates[:max_history_candidates]
    for position, index in enumerate(candidates):
        if int(index) == int(matched_t):
            return position, 1, candidates
    return 0, 0, candidates


def _sample_payload(
    frame_source: Any,
    test: dict[str, Any],
    events: list[dict[str, Any]],
    actions: np.ndarray,
    poses: np.ndarray,
    num_frames: int,
    task: CacheTask,
    window_kind: str,
    center_t: int,
) -> dict[str, np.ndarray]:
    query_t = int(test["query_t"])
    target_t = int(test["target_t"])
    if target_t < 0 or target_t >= num_frames:
        raise ValueError(f"target_t={target_t} is outside episode length {num_frames}")
    context_len = task.context_length
    future_len = task.future_length
    rollout_length = context_len + future_len

    start = int(center_t) - context_len
    end = start + rollout_length - 1
    if start < 0 or end >= num_frames:
        raise ValueError(
            f"{window_kind} window [{start}, {end}] around center_t={center_t} exceeds episode length {num_frames}"
        )

    main_indices = list(range(start, end + 1))
    if len(main_indices) != rollout_length:
        raise ValueError(f"invalid rollout length {len(main_indices)}; expected {rollout_length}")
    generation_center_index = int(center_t) - start
    if generation_center_index != context_len:
        raise ValueError(
            f"generation center index {generation_center_index} must equal context_length {context_len}"
        )
    query_index_in_video = query_t - start if start <= query_t <= end else -1
    history_context_end_index = query_index_in_video
    ptm_recent_end_index = context_len
    target_index_in_video = target_t - start if start <= target_t <= end else -1
    if window_kind == "target" and target_index_in_video != generation_center_index:
        raise ValueError(
            f"target window must contain target_t at generation center; got target_index={target_index_in_video}, "
            f"center_index={generation_center_index}"
        )

    main_actions = actions[start : end + 1]
    main_poses = poses[start : end + 1]
    main_timestamp = np.arange(start, end + 1, dtype=np.int64)

    memory_indices: list[int] = []
    if task.memory_condition_length > 0:
        memory_indices = _select_memory_indices(
            test=test,
            events=events,
            num_frames=num_frames,
            start=start,
            query_t=query_t,
            memory_length=task.memory_condition_length,
            strategy=task.memory_strategy,
        )
        main_indices = main_indices + memory_indices
        main_actions = np.concatenate(
            [main_actions, np.stack([actions[idx] if idx >= 0 else np.zeros_like(actions[0]) for idx in memory_indices])],
            axis=0,
        )
        main_poses = np.concatenate(
            [main_poses, np.stack([poses[idx] if idx >= 0 else np.zeros_like(poses[0]) for idx in memory_indices])],
            axis=0,
        )
        main_timestamp = np.concatenate(
            [main_timestamp, np.asarray([idx if idx >= 0 else 0 for idx in memory_indices], dtype=np.int64)], axis=0
        )

    f0 = max(0, int(test["future_start_t"]))
    f1 = min(num_frames - 1, int(test["future_end_t"]))

    frame_map = _read_frame_map(frame_source, main_indices + [target_t], task.height, task.width)
    video = _stack_frames(main_indices, frame_map, task.height, task.width)
    target_frame = frame_map[target_t]

    raw_labels = dict(test.get("labels", {}))
    matched_t = int(raw_labels.get("matched_history_t", start))
    matched_t = _clamp_frame(matched_t, num_frames, query_t)
    matched_history_index, match_valid, candidate_history_indices = _history_match_label(
        matched_t=matched_t,
        memory_indices=memory_indices,
        main_indices=main_indices[:rollout_length],
        history_context_end_index=history_context_end_index,
        max_history_candidates=task.max_history_candidates,
    )

    return {
        "video": video.astype(np.uint8, copy=False),
        "actions": main_actions.astype(np.float32, copy=False),
        "poses": main_poses.astype(np.float32, copy=False),
        "timestamp": main_timestamp.astype(np.int64, copy=False),
        "future_actions": _slice_and_pad(actions, f0, f1, task.test_future_action_length).astype(
            np.float32, copy=False
        ),
        "target_frames": target_frame.astype(np.uint8, copy=False),
        "test_type_id": np.asarray(int(test.get("test_type_id", TEST_TYPE_TO_ID[test["test_type"]])), dtype=np.int64),
        "matched_history_index": np.asarray(matched_history_index, dtype=np.int64),
        "match_valid": np.asarray(match_valid, dtype=np.int64),
        "returns_to_seen_place": np.asarray(float(bool(raw_labels.get("returns_to_seen_place", False))), dtype=np.float32),
        "landmark_visible": np.asarray(float(bool(raw_labels.get("landmark_visible", False))), dtype=np.float32),
        "object_exists_at_return": np.asarray(float(bool(raw_labels.get("object_exists_at_return", False))), dtype=np.float32),
        "query_index_in_video": np.asarray(query_index_in_video, dtype=np.int64),
        "target_index_in_video": np.asarray(target_index_in_video, dtype=np.int64),
        "query_t": np.asarray(query_t, dtype=np.int64),
        "target_t": np.asarray(target_t, dtype=np.int64),
        "window_center_t": np.asarray(int(center_t), dtype=np.int64),
        "window_kind": np.asarray(str(window_kind)),
        "generation_center_index_in_video": np.asarray(generation_center_index, dtype=np.int64),
        "ptm_recent_end_index": np.asarray(ptm_recent_end_index, dtype=np.int64),
        "history_context_end_index": np.asarray(history_context_end_index, dtype=np.int64),
        "context_length": np.asarray(context_len, dtype=np.int64),
        "future_length": np.asarray(future_len, dtype=np.int64),
        "memory_condition_length": np.asarray(task.memory_condition_length, dtype=np.int64),
        "has_reference_tail": np.asarray(int(task.memory_condition_length > 0), dtype=np.int64),
        "memory_indices": np.asarray(memory_indices, dtype=np.int64),
        "candidate_history_indices": np.asarray(candidate_history_indices, dtype=np.int64),
        "candidate_history_count": np.asarray(len(candidate_history_indices), dtype=np.int64),
        "matched_history_t": np.asarray(matched_t, dtype=np.int64),
    }


def _save_npz(path: Path, payload: dict[str, np.ndarray], compressed: bool) -> None:
    tmp_path = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with tmp_path.open("wb") as handle:
        if compressed:
            np.savez_compressed(handle, **payload)
        else:
            np.savez(handle, **payload)
    os.replace(tmp_path, path)


def _process_episode(task: CacheTask) -> dict[str, Any]:
    episode_dir = Path(task.episode_dir)
    video_path = episode_dir / "frames.mp4"
    npz_path = episode_dir / "frames.npz"
    if not video_path.exists() and not npz_path.exists():
        raise FileNotFoundError(f"{episode_dir} has neither frames.mp4 nor frames.npz")

    metadata_path = episode_dir / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else {}
    episode_family = str(metadata.get("ptm_family", metadata.get("family", "")))
    tests = read_jsonl(episode_dir / "tests.jsonl")
    events = read_jsonl(episode_dir / "events.jsonl")
    action_records = read_jsonl(episode_dir / "actions.jsonl")
    pose_records = read_jsonl(episode_dir / "poses.jsonl")
    tests = [
        test
        for test in tests
        if not has_pose_discontinuity(
            pose_records,
            int(test.get("history_start_t", 0)),
            int(test.get("future_end_t", test.get("target_t", 0))),
        )
    ]
    actions = np.stack([oasis_action_to_vector(record["oasis_action"]) for record in action_records]).astype(np.float32)
    poses = _pose_records_to_array(pose_records)

    frame_source: Any
    release_source = False
    if video_path.exists() and task.full_decode:
        frame_source = _decode_mp4(video_path, task.height, task.width)
        video_frames = int(frame_source.shape[0])
    elif video_path.exists():
        if cv2 is None:
            raise RuntimeError("opencv-python is required to build the NPZ cache from mp4")
        frame_source = cv2.VideoCapture(str(video_path))
        release_source = True
        video_frames = int(frame_source.get(cv2.CAP_PROP_FRAME_COUNT))
    else:
        frame_source = np.load(npz_path, allow_pickle=False)["frames"]
        video_frames = int(frame_source.shape[0])
    try:
        num_frames = min(video_frames if video_frames > 0 else len(actions), len(actions), len(poses))
        if num_frames <= 0:
            raise ValueError(f"{episode_dir} has no usable frames")

        selected = set(task.selected_test_indices) if task.selected_test_indices is not None else None
        indexed_tests = [
            (local_idx, test)
            for local_idx, test in enumerate(tests)
            if selected is None or local_idx in selected
        ]

        entries = []
        written = 0
        skipped = 0
        episode_key = hashlib.sha1(str(episode_dir).encode("utf-8")).hexdigest()[:12]
        for output_idx, (local_idx, test) in enumerate(indexed_tests):
            for spec in task.window_centers:
                center = _window_center_t(test, spec, num_frames)
                if center is None:
                    continue
                window_kind, center_t = center
                start = center_t - task.context_length
                end = start + task.context_length + task.future_length - 1
                if start < 0 or end >= num_frames:
                    continue
                sample_name = f"{episode_key}_test{local_idx:05d}_{window_kind}.npz"
                sample_path = Path(task.out_dir) / task.split / sample_name
                if task.skip_existing and sample_path.exists():
                    skipped += 1
                else:
                    payload = _sample_payload(
                        frame_source,
                        test,
                        events,
                        actions,
                        poses,
                        num_frames,
                        task,
                        window_kind,
                        center_t,
                    )
                    _save_npz(sample_path, payload, task.compressed)
                    written += 1
                entries.append(
                    {
                        "path": sample_name,
                        "episode_dir": str(episode_dir),
                        "episode_family": episode_family,
                        "test_idx": local_idx,
                        "test_type": test["test_type"],
                        "window_kind": window_kind,
                        "window_center_t": center_t,
                    }
                )
        return {"split": task.split, "episode_dir": str(episode_dir), "entries": entries, "written": written, "skipped": skipped}
    finally:
        if release_source:
            frame_source.release()


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    with tmp_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
    os.replace(tmp_path, path)


def build_npz_cache(
    data_root: str | Path,
    out_dir: str | Path,
    splits: list[str],
    workers: int = 8,
    height: int = 360,
    width: int = 640,
    context_length: int = 4,
    future_length: int = 4,
    test_future_action_length: int | None = None,
    memory_condition_length: int = 8,
    max_history_candidates: int = 256,
    skip_existing: bool = False,
    compressed: bool = False,
    full_decode: bool = True,
    memory_strategy: str = "causal_slots",
    window_centers: list[str] | tuple[str, ...] = ("target", "late50", "late75", "late100"),
    max_samples_per_split: int | None = None,
) -> dict[str, Any]:
    data_root = Path(data_root)
    out_dir = Path(out_dir)
    if test_future_action_length is None:
        late_offsets = [
            offset
            for offset in (_late_offset_from_spec(item) for item in window_centers)
            if offset is not None
        ]
        test_future_action_length = max([int(future_length)] + late_offsets)
    manifest: dict[str, Any] = {
        "data_root": str(data_root),
        "out_dir": str(out_dir),
        "height": int(height),
        "width": int(width),
        "context_length": int(context_length),
        "future_length": int(future_length),
        "test_future_action_length": int(test_future_action_length),
        "memory_condition_length": int(memory_condition_length),
        "max_history_candidates": int(max_history_candidates),
        "compressed": bool(compressed),
        "full_decode": bool(full_decode),
        "memory_strategy": str(memory_strategy),
        "window_centers": [str(item) for item in window_centers],
        "max_samples_per_split": max_samples_per_split,
        "splits": {},
    }

    for split in splits:
        episode_dirs = _episode_dirs(data_root, split)
        if not episode_dirs:
            raise FileNotFoundError(f"no PTM episodes found in {data_root / split}")
        selected_by_episode: dict[str, set[int]] | None = None
        if max_samples_per_split is not None and int(max_samples_per_split) > 0:
            candidates: list[tuple[float, int, str, int]] = []
            for episode_order, episode_dir in enumerate(episode_dirs):
                tests = read_jsonl(episode_dir / "tests.jsonl")
                pose_records = read_jsonl(episode_dir / "poses.jsonl")
                tests = [
                    test
                    for test in tests
                    if not has_pose_discontinuity(
                        pose_records,
                        int(test.get("history_start_t", 0)),
                        int(test.get("future_end_t", test.get("target_t", 0))),
                    )
                ]
                for local_idx, test in enumerate(tests):
                    score = _future_motion_score(test) if split in {"val", "test"} else -float(len(candidates))
                    candidates.append((score, episode_order, str(episode_dir), local_idx))
            if split in {"val", "test"}:
                candidates.sort(key=lambda item: item[0], reverse=True)
            selected_by_episode = {}
            for _score, _episode_order, episode_dir, local_idx in candidates[: int(max_samples_per_split)]:
                selected_by_episode.setdefault(episode_dir, set()).add(local_idx)

        tasks: list[CacheTask] = []
        sample_offset = 0
        for episode_dir in episode_dirs:
            selected_indices = None
            if selected_by_episode is not None:
                selected_set = selected_by_episode.get(str(episode_dir), set())
                if not selected_set:
                    continue
                selected_indices = tuple(sorted(selected_set))
                test_count = len(selected_indices)
            else:
                test_count = len(read_jsonl(episode_dir / "tests.jsonl"))
            if test_count == 0:
                continue
            tasks.append(
                CacheTask(
                    split=split,
                    episode_dir=str(episode_dir),
                    sample_offset=sample_offset,
                    out_dir=str(out_dir),
                    height=int(height),
                    width=int(width),
                    context_length=int(context_length),
                    future_length=int(future_length),
                    test_future_action_length=int(test_future_action_length),
                    memory_condition_length=int(memory_condition_length),
                    max_history_candidates=int(max_history_candidates),
                    skip_existing=bool(skip_existing),
                    compressed=bool(compressed),
                    full_decode=bool(full_decode),
                    memory_strategy=str(memory_strategy),
                    window_centers=tuple(str(item) for item in window_centers),
                    selected_test_indices=selected_indices,
                )
            )
            sample_offset += test_count

        entries: list[dict[str, Any]] = []
        written = 0
        skipped = 0
        errors: list[dict[str, str]] = []
        if tasks:
            with ProcessPoolExecutor(max_workers=max(1, int(workers))) as executor:
                futures = {executor.submit(_process_episode, task): task for task in tasks}
                for completed, future in enumerate(as_completed(futures), start=1):
                    task = futures[future]
                    try:
                        result = future.result()
                    except Exception as exc:  # pragma: no cover - exercised in cluster runtime.
                        errors.append({"episode_dir": task.episode_dir, "error": repr(exc)})
                    else:
                        entries.extend(result["entries"])
                        written += int(result["written"])
                        skipped += int(result["skipped"])
                    if completed == 1 or completed % 25 == 0 or completed == len(tasks):
                        print(
                            json.dumps(
                                {
                                    "split": split,
                                    "episodes_done": completed,
                                    "episodes_total": len(tasks),
                                    "samples_indexed": len(entries),
                                    "written": written,
                                    "skipped": skipped,
                                    "errors": len(errors),
                                },
                                sort_keys=True,
                            ),
                            flush=True,
                        )

        entries.sort(key=lambda item: item["path"])
        if errors:
            manifest["splits"][split] = {
                "episodes": len(tasks),
                "samples": len(entries),
                "written": written,
                "skipped": skipped,
                "errors": errors,
            }
            continue

        _write_jsonl(out_dir / split / "index.jsonl", entries)
        manifest["splits"][split] = {
            "episodes": len(tasks),
            "samples": len(entries),
            "written": written,
            "skipped": skipped,
            "index": str(out_dir / split / "index.jsonl"),
        }

    manifest_path = out_dir / "manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_manifest = manifest_path.with_name(f"{manifest_path.name}.tmp.{os.getpid()}")
    tmp_manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp_manifest, manifest_path)

    failed = {split: data for split, data in manifest["splits"].items() if data.get("errors")}
    if failed:
        raise RuntimeError(f"failed to build some cache entries: {json.dumps(failed, sort_keys=True)[:2000]}")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Build sample-level NPZ cache for PTM WorldMem training.")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--height", type=int, default=360)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--context-length", type=int, default=4)
    parser.add_argument("--future-length", type=int, default=4)
    parser.add_argument(
        "--test-future-action-length",
        type=int,
        default=None,
        help=(
            "Number of future actions stored for PTM future-test decoding; "
            "defaults to the largest requested late window offset, or --future-length."
        ),
    )
    parser.add_argument("--memory-condition-length", type=int, default=8)
    parser.add_argument("--max-history-candidates", type=int, default=256)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--compressed", action="store_true")
    parser.add_argument("--no-full-decode", action="store_true")
    parser.add_argument(
        "--memory-strategy",
        default="causal_slots",
        choices=["causal_slots"],
        help=(
            "causal_slots is the main PTM strategy and only uses information "
            "available up to query_t."
        ),
    )
    parser.add_argument(
        "--window-centers",
        nargs="+",
        default=["target", "late50", "late75", "late100"],
        help="Training windows to build for each future test: target or late<offset>.",
    )
    parser.add_argument(
        "--max-samples-per-split",
        type=int,
        default=None,
        help="Optional fixed subset size per split. Useful for 600+100 generation validation caches.",
    )
    args = parser.parse_args()

    manifest = build_npz_cache(
        data_root=args.data_root,
        out_dir=args.out_dir,
        splits=args.splits,
        workers=args.workers,
        height=args.height,
        width=args.width,
        context_length=args.context_length,
        future_length=args.future_length,
        test_future_action_length=args.test_future_action_length,
        memory_condition_length=args.memory_condition_length,
        max_history_candidates=args.max_history_candidates,
        skip_existing=args.skip_existing,
        compressed=args.compressed,
        full_decode=not args.no_full_decode,
        memory_strategy=args.memory_strategy,
        window_centers=args.window_centers,
        max_samples_per_split=args.max_samples_per_split,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
