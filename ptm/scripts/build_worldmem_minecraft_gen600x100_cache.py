#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np


ACTION_DIM = 25


def _convert_worldmem_actions(actions: np.ndarray) -> np.ndarray:
    """Match datasets/video/minecraft_video_dataset.py::convert_action_space."""
    actions = np.asarray(actions)
    out = np.zeros((len(actions), ACTION_DIM), dtype=np.float32)
    if actions.ndim != 2 or actions.shape[1] < 8:
        raise ValueError(f"expected WorldMem compact actions with shape (T,>=8), got {actions.shape}")
    out[actions[:, 0] == 1, 11] = 1
    out[actions[:, 0] == 2, 12] = 1
    out[actions[:, 4] == 11, 16] = -1
    out[actions[:, 4] == 13, 16] = 1
    out[actions[:, 3] == 11, 15] = -1
    out[actions[:, 3] == 13, 15] = 1
    out[actions[:, 5] == 6, 24] = 1
    out[actions[:, 5] == 1, 24] = 1
    out[actions[:, 1] == 1, 13] = 1
    out[actions[:, 1] == 2, 14] = 1
    out[actions[:, 7] == 1, 2] = 1
    return out


def _select_stratified(paths: list[Path], data_root: Path, max_samples: int | None) -> list[Path]:
    if max_samples is None or max_samples <= 0 or len(paths) <= max_samples:
        return paths
    groups: dict[str, list[Path]] = defaultdict(list)
    for path in paths:
        rel = path.relative_to(data_root)
        key = rel.parts[0] if rel.parts else "unknown"
        groups[key].append(path)
    for values in groups.values():
        values.sort()
    selected: list[Path] = []
    keys = sorted(groups)
    cursor = {key: 0 for key in keys}
    while len(selected) < max_samples:
        progressed = False
        for key in keys:
            idx = cursor[key]
            if idx >= len(groups[key]):
                continue
            selected.append(groups[key][idx])
            cursor[key] += 1
            progressed = True
            if len(selected) >= max_samples:
                break
        if not progressed:
            break
    return selected


def _memory_indices(start: int, context_length: int, memory_length: int) -> list[int]:
    if memory_length <= 0:
        return []
    end = start + context_length - 1
    if memory_length == 1:
        return [end]
    return [int(round(x)) for x in np.linspace(start, end, num=memory_length)]


def _read_rollout_frames(
    video_path: Path,
    start: int,
    context_length: int,
    future_length: int,
    memory_indices: list[int],
    height: int,
    width: int,
) -> tuple[np.ndarray, np.ndarray]:
    rollout_length = context_length + future_length
    main_start = start
    main_end = start + rollout_length - 1
    max_index = max([main_end, *memory_indices])
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"could not open video: {video_path}")
    main_frames: list[np.ndarray] = []
    memory_frames: dict[int, np.ndarray] = {}
    frame_idx = 0
    try:
        while frame_idx <= max_index:
            ok, frame = cap.read()
            if not ok:
                raise RuntimeError(f"{video_path} ended at frame {frame_idx}; need frame {max_index}")
            if frame_idx >= main_start and frame_idx <= main_end or frame_idx in memory_indices:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                if frame.shape[0] != height or frame.shape[1] != width:
                    frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
                frame = np.ascontiguousarray(frame, dtype=np.uint8)
                if frame_idx >= main_start and frame_idx <= main_end:
                    main_frames.append(frame)
                if frame_idx in memory_indices:
                    memory_frames[frame_idx] = frame
            frame_idx += 1
    finally:
        cap.release()
    if len(main_frames) != rollout_length:
        raise RuntimeError(f"{video_path} yielded {len(main_frames)} rollout frames; expected {rollout_length}")
    video = np.concatenate(
        [np.stack(main_frames, axis=0), np.stack([memory_frames[idx] for idx in memory_indices], axis=0)],
        axis=0,
    )
    return video, main_frames[context_length]


def _candidate_history_indices(
    memory_indices: list[int],
    start: int,
    history_context_end_index: int,
    max_history_candidates: int,
) -> list[int]:
    context_indices = list(range(start, start + history_context_end_index + 1))
    seen = set()
    out: list[int] = []
    for value in [*memory_indices, *context_indices]:
        if value in seen:
            continue
        seen.add(value)
        out.append(int(value))
        if len(out) >= max_history_candidates:
            break
    return out


def _save_npz(path: Path, payload: dict[str, np.ndarray], compressed: bool) -> None:
    tmp_path = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with tmp_path.open("wb") as handle:
        if compressed:
            np.savez_compressed(handle, **payload)
        else:
            np.savez(handle, **payload)
    os.replace(tmp_path, path)


def _build_sample(
    video_path: Path,
    data_root: Path,
    out_split_dir: Path,
    sample_idx: int,
    args: argparse.Namespace,
) -> dict[str, Any]:
    action_path = video_path.with_suffix(".npz")
    if not action_path.exists():
        raise FileNotFoundError(f"missing action/pose npz next to {video_path}")
    sidecar = np.load(action_path, allow_pickle=False)
    actions = _convert_worldmem_actions(sidecar["actions"])
    poses = np.asarray(sidecar["poses"], dtype=np.float32)
    if poses.shape[1] != 5:
        raise ValueError(f"expected poses shape (T,5), got {poses.shape} for {action_path}")

    start = int(args.start_frame)
    context_length = int(args.context_length)
    future_length = int(args.future_length)
    memory_length = int(args.memory_condition_length)
    rollout_length = context_length + future_length
    target_t = start + context_length
    end = start + rollout_length - 1
    if end >= len(actions) or end >= len(poses):
        raise ValueError(f"{video_path} has too few actions/poses for window [{start}, {end}]")

    memory_indices = _memory_indices(start, context_length, memory_length)
    video, target_frame = _read_rollout_frames(
        video_path,
        start,
        context_length,
        future_length,
        memory_indices,
        int(args.height),
        int(args.width),
    )
    main_indices = list(range(start, end + 1))
    all_indices = [*main_indices, *memory_indices]
    history_context_end_index = context_length - 1
    candidate_indices = _candidate_history_indices(
        memory_indices,
        start,
        history_context_end_index,
        int(args.max_history_candidates),
    )
    if len(candidate_indices) < int(args.max_history_candidates):
        candidate_indices.extend([candidate_indices[-1] if candidate_indices else start] * (int(args.max_history_candidates) - len(candidate_indices)))

    rel = video_path.relative_to(data_root)
    biome = rel.parts[0] if len(rel.parts) >= 1 else "unknown"
    seed = rel.parts[1] if len(rel.parts) >= 2 else "unknown"
    sample_name = f"worldmem_{sample_idx:05d}_{biome}_{seed}_{video_path.stem}_target.npz"
    payload = {
        "video": video.astype(np.uint8, copy=False),
        "actions": actions[all_indices].astype(np.float32, copy=False),
        "poses": poses[all_indices].astype(np.float32, copy=False),
        "timestamp": np.asarray(all_indices, dtype=np.int64),
        "future_actions": actions[target_t : target_t + future_length].astype(np.float32, copy=False),
        "target_frames": target_frame.astype(np.uint8, copy=False),
        "test_type_id": np.asarray(0, dtype=np.int64),
        "matched_history_index": np.asarray(0, dtype=np.int64),
        "match_valid": np.asarray(0, dtype=np.int64),
        "returns_to_seen_place": np.asarray(0.0, dtype=np.float32),
        "landmark_visible": np.asarray(0.0, dtype=np.float32),
        "object_exists_at_return": np.asarray(0.0, dtype=np.float32),
        "query_index_in_video": np.asarray(history_context_end_index, dtype=np.int64),
        "target_index_in_video": np.asarray(context_length, dtype=np.int64),
        "query_t": np.asarray(start + history_context_end_index, dtype=np.int64),
        "target_t": np.asarray(target_t, dtype=np.int64),
        "window_center_t": np.asarray(target_t, dtype=np.int64),
        "window_kind": np.asarray("target"),
        "generation_center_index_in_video": np.asarray(context_length, dtype=np.int64),
        "ptm_recent_end_index": np.asarray(context_length, dtype=np.int64),
        "history_context_end_index": np.asarray(history_context_end_index, dtype=np.int64),
        "context_length": np.asarray(context_length, dtype=np.int64),
        "future_length": np.asarray(future_length, dtype=np.int64),
        "memory_condition_length": np.asarray(memory_length, dtype=np.int64),
        "has_reference_tail": np.asarray(int(memory_length > 0), dtype=np.int64),
        "memory_indices": np.asarray(memory_indices, dtype=np.int64),
        "candidate_history_indices": np.asarray(candidate_indices, dtype=np.int64),
        "candidate_history_count": np.asarray(len(candidate_indices), dtype=np.int64),
        "matched_history_t": np.asarray(start, dtype=np.int64),
    }
    _save_npz(out_split_dir / sample_name, payload, bool(args.compressed))
    return {
        "episode_dir": str(video_path.with_suffix("")),
        "episode_family": biome,
        "path": sample_name,
        "test_idx": int(sample_idx),
        "test_type": "normal_rollout",
        "window_center_t": int(target_t),
        "window_kind": "target",
        "source_mp4": str(video_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a PTM gen600x100 NPZ cache from WorldMem Minecraft test mp4/npz pairs.")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--max-samples", type=int, default=16)
    parser.add_argument("--start-frame", type=int, default=100)
    parser.add_argument("--height", type=int, default=360)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--context-length", type=int, default=600)
    parser.add_argument("--future-length", type=int, default=100)
    parser.add_argument("--memory-condition-length", type=int, default=8)
    parser.add_argument("--max-history-candidates", type=int, default=16)
    parser.add_argument("--compressed", action="store_true")
    args = parser.parse_args()

    data_root = args.data_root.resolve()
    out_dir = args.out_dir.resolve()
    mp4_paths = sorted(data_root.glob("*/*/*.mp4"))
    if not mp4_paths:
        raise SystemExit(f"no mp4 files under {data_root}")
    selected = _select_stratified(mp4_paths, data_root, args.max_samples)
    split_dir = out_dir / args.split
    split_dir.mkdir(parents=True, exist_ok=True)

    entries: list[dict[str, Any]] = []
    for sample_idx, video_path in enumerate(selected):
        entry = _build_sample(video_path, data_root, split_dir, sample_idx, args)
        entries.append(entry)
        print(json.dumps({"sample": sample_idx, "path": entry["path"], "source_mp4": str(video_path)}, sort_keys=True), flush=True)

    index_path = split_dir / "index.jsonl"
    with index_path.open("w", encoding="utf-8") as handle:
        for entry in entries:
            handle.write(json.dumps(entry, sort_keys=True) + "\n")

    manifest = {
        "data_root": str(data_root),
        "out_dir": str(out_dir),
        "height": int(args.height),
        "width": int(args.width),
        "context_length": int(args.context_length),
        "future_length": int(args.future_length),
        "test_future_action_length": int(args.future_length),
        "memory_condition_length": int(args.memory_condition_length),
        "max_history_candidates": int(args.max_history_candidates),
        "compressed": bool(args.compressed),
        "full_decode": True,
        "memory_strategy": "causal_slots",
        "window_centers": ["target"],
        "source": "worldmem_minecraft_test_mp4_npz",
        "start_frame": int(args.start_frame),
        "max_samples_per_split": int(args.max_samples),
        "splits": {
            args.split: {
                "episodes": len(entries),
                "samples": len(entries),
                "written": len(entries),
                "skipped": 0,
                "index": str(index_path),
            }
        },
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"out_dir": str(out_dir), "samples": len(entries), "index": str(index_path)}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
