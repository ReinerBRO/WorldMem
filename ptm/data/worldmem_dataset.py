from __future__ import annotations

import json
from collections import OrderedDict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .ptm_dataset import _episode_dirs, _pad_or_trim_time, _read_frames
from .schemas import TEST_TYPE_TO_ID, oasis_action_to_vector, read_jsonl
from .window_filters import has_pose_discontinuity


def _npz_video_to_tensor(frames: np.ndarray) -> torch.Tensor:
    if frames.ndim == 3:
        if frames.shape[0] in (1, 3, 4) and frames.shape[-1] not in (3, 4):
            frames = np.transpose(frames, (1, 2, 0))
        if frames.shape[-1] > 3:
            frames = frames[..., :3]
        return torch.from_numpy(np.ascontiguousarray(frames)).permute(2, 0, 1).float() / 255.0
    if frames.ndim != 4:
        raise ValueError(f"expected cached video with 3 or 4 dims, got shape {frames.shape}")
    if frames.shape[1] in (1, 3, 4) and frames.shape[-1] not in (3, 4):
        frames = np.transpose(frames, (0, 2, 3, 1))
    if frames.shape[-1] > 3:
        frames = frames[..., :3]
    if frames.shape[-1] != 3:
        raise ValueError(f"expected cached RGB video, got shape {frames.shape}")
    return torch.from_numpy(np.ascontiguousarray(frames)).permute(0, 3, 1, 2).float() / 255.0


def _pose_records_to_tensor(records: list[dict[str, Any]]) -> torch.Tensor:
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
    return torch.tensor(rows, dtype=torch.float32)


def _split_names(value: Any) -> set[str]:
    if value is None:
        return {"train"}
    if isinstance(value, str):
        return {part.strip() for part in value.split(",") if part.strip()}
    return {str(part).strip() for part in value if str(part).strip()}


def _future_motion_score(test: dict[str, Any]) -> float:
    labels = test.get("labels", {})
    path = float(labels.get("future_path_length", 0.0))
    yaw = float(labels.get("future_yaw_path", 0.0))
    action_rate = float(labels.get("future_action_rate", 0.0))
    return action_rate * 1000.0 + path + yaw / 180.0


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


def _history_match_label(
    matched_t: int,
    memory_indices: list[int],
    context_indices: list[int],
    max_history_candidates: int,
) -> tuple[int, int, list[int]]:
    candidates = _unique_preserving_order(
        [int(index) for index in memory_indices if int(index) >= 0]
        + [int(index) for index in context_indices if int(index) >= 0]
    )
    candidates = candidates[:max_history_candidates]
    for position, index in enumerate(candidates):
        if int(index) == int(matched_t):
            return position, 1, candidates
    return 0, 0, candidates


class PTMWorldMemDataset(torch.utils.data.Dataset):
    """PTM episode dataset that feeds WorldMem training plus future-test labels."""

    def __init__(self, cfg: Any, split: str = "training"):
        super().__init__()
        split_map = {"training": "train", "validation": "val", "test": "test"}
        self.split = split_map.get(split, split)
        self.is_eval_split = self.split in {"val", "test"}
        self.save_dir = Path(cfg.save_dir)
        self.resolution = cfg.resolution
        self.memory_condition_length = int(getattr(cfg, "memory_condition_length", 0))
        self.future_length = int(getattr(cfg, "future_length", 64))
        self.ptm_context_length = int(getattr(cfg, "ptm_context_length", 32))
        self.ptm_future_length = int(getattr(cfg, "ptm_future_length", self.future_length))
        if self.is_eval_split:
            valid_context = getattr(cfg, "ptm_context_length_valid", None)
            valid_future = getattr(cfg, "ptm_future_length_valid", None)
            valid_frames = getattr(cfg, "n_frames_valid", None)
            if valid_context is not None:
                self.ptm_context_length = int(valid_context)
            if valid_future is not None:
                self.ptm_future_length = int(valid_future)
            elif valid_frames is not None and valid_context is not None:
                self.ptm_future_length = int(valid_frames) - self.ptm_context_length
            self.future_length = self.ptm_future_length
        self.n_frames = self.ptm_context_length + self.ptm_future_length
        self.max_history_candidates = int(getattr(cfg, "max_history_candidates", 256))
        self.frame_skip = int(getattr(cfg, "frame_skip", 1))
        self.video_cache_size = int(getattr(cfg, "video_cache_size", -1))

        self.use_npz_cache = False
        self.cache_index: list[dict[str, Any]] = []
        self.npz_cache_manifest: dict[str, Any] = {}
        self.npz_split_dir: Path | None = None

        # Eval splits use npz_cache_dir_val / npz_cache_dir_test if configured,
        # otherwise fall back to npz_cache_dir.
        eval_cache_key = f"npz_cache_dir_{self.split}"
        npz_cache_dir = getattr(cfg, eval_cache_key, None) or getattr(cfg, "npz_cache_dir", None)
        npz_cache_splits = _split_names(getattr(cfg, "npz_cache_splits", "train"))
        if npz_cache_dir and ("all" in npz_cache_splits or self.split in npz_cache_splits):
            cache_root = Path(npz_cache_dir)
            manifest_path = cache_root / "manifest.json"
            if manifest_path.exists():
                self.npz_cache_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            index_path = cache_root / self.split / "index.jsonl"
            if not index_path.exists():
                raise FileNotFoundError(f"missing PTM NPZ cache index: {index_path}")
            self.cache_index = read_jsonl(index_path)
            if not self.cache_index:
                raise ValueError(f"empty PTM NPZ cache index: {index_path}")
            cache_indices_file = getattr(cfg, f"npz_cache_indices_file_{self.split}", None)
            if cache_indices_file is None:
                cache_indices_file = getattr(cfg, "npz_cache_indices_file", None)
            if cache_indices_file:
                selected_indices = []
                indices_path = Path(str(cache_indices_file))
                with indices_path.open("r", encoding="utf-8") as handle:
                    for line in handle:
                        line = line.strip()
                        if not line:
                            continue
                        selected_indices.append(int(line))
                if not selected_indices:
                    raise ValueError(f"empty PTM NPZ cache indices file: {indices_path}")
                cache_size = len(self.cache_index)
                invalid = [index for index in selected_indices if index < 0 or index >= cache_size]
                if invalid:
                    raise ValueError(
                        f"PTM NPZ cache indices out of range for {index_path}: {invalid[:8]}"
                    )
                self.cache_index = [self.cache_index[index] for index in selected_indices]
            self.npz_split_dir = index_path.parent
            self.use_npz_cache = True
            return

        self.episode_dirs = _episode_dirs(self.save_dir, self.split)
        if not self.episode_dirs:
            raise FileNotFoundError(f"no PTM episodes found in {self.save_dir}/{self.split}")

        self.index: list[tuple[Path, int]] = []
        self._test_cache: dict[Path, list[dict[str, Any]]] = {}
        self._episode_family: dict[Path, str] = {}
        for episode_dir in self.episode_dirs:
            metadata_path = episode_dir / "metadata.json"
            if metadata_path.exists():
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                self._episode_family[episode_dir] = str(metadata.get("ptm_family", metadata.get("family", "")))
            else:
                self._episode_family[episode_dir] = ""
            tests = read_jsonl(episode_dir / "tests.jsonl")
            if self.is_eval_split:
                poses = read_jsonl(episode_dir / "poses.jsonl")
                tests = [
                    test
                    for test in tests
                    if not has_pose_discontinuity(
                        poses,
                        int(test.get("history_start_t", 0)),
                        int(test.get("future_end_t", test.get("target_t", 0))),
                    )
                ]
            self._test_cache[episode_dir] = tests
            self.index.extend((episode_dir, i) for i in range(len(tests)))
        if self.split in {"val", "test"}:
            self.index.sort(
                key=lambda item: _future_motion_score(self._test_cache[item[0]][item[1]]),
                reverse=True,
            )
        indices_file = getattr(cfg, f"indices_file_{self.split}", None)
        if indices_file is None:
            indices_file = getattr(cfg, "indices_file", None)
        if indices_file:
            selected_indices = []
            indices_path = Path(str(indices_file))
            with indices_path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    selected_indices.append(int(line))
            if not selected_indices:
                raise ValueError(f"empty PTM dataset indices file: {indices_path}")
            invalid = [index for index in selected_indices if index < 0 or index >= len(self.index)]
            if invalid:
                raise ValueError(
                    f"PTM dataset indices out of range for split {self.split}: {invalid[:8]}"
                )
            self.index = [self.index[index] for index in selected_indices]

        self._video_cache: OrderedDict[Path, torch.Tensor] = OrderedDict()
        self._actions_cache: dict[Path, torch.Tensor] = {}
        self._poses_cache: dict[Path, torch.Tensor] = {}

    def __len__(self) -> int:
        if self.use_npz_cache:
            return len(self.cache_index)
        return len(self.index)

    def _video(self, episode_dir: Path) -> torch.Tensor:
        if self.video_cache_size == 0:
            return _read_frames(episode_dir, self.resolution)
        if episode_dir in self._video_cache:
            self._video_cache.move_to_end(episode_dir)
            return self._video_cache[episode_dir]
        video = _read_frames(episode_dir, self.resolution)
        self._video_cache[episode_dir] = video
        if self.video_cache_size > 0:
            while len(self._video_cache) > self.video_cache_size:
                self._video_cache.popitem(last=False)
        return video

    def _actions(self, episode_dir: Path) -> torch.Tensor:
        if episode_dir not in self._actions_cache:
            records = read_jsonl(episode_dir / "actions.jsonl")
            vectors = [oasis_action_to_vector(record["oasis_action"]) for record in records]
            self._actions_cache[episode_dir] = torch.from_numpy(np.stack(vectors)).float()
        return self._actions_cache[episode_dir]

    def _poses(self, episode_dir: Path) -> torch.Tensor:
        if episode_dir not in self._poses_cache:
            self._poses_cache[episode_dir] = _pose_records_to_tensor(read_jsonl(episode_dir / "poses.jsonl"))
        return self._poses_cache[episode_dir]

    def _getitem_npz(self, idx: int) -> dict[str, Any]:
        assert self.npz_split_dir is not None
        entry = self.cache_index[idx]
        sample_path = Path(entry["path"])
        if not sample_path.is_absolute():
            sample_path = self.npz_split_dir / sample_path
        with np.load(sample_path, allow_pickle=False) as data:
            cached_video = _npz_video_to_tensor(data["video"])[:: self.frame_skip]
            cached_actions = torch.from_numpy(np.asarray(data["actions"])).float()[:: self.frame_skip]
            cached_poses = torch.from_numpy(np.asarray(data["poses"])).float()[:: self.frame_skip]
            cached_timestamp = torch.from_numpy(np.asarray(data["timestamp"])).long()[:: self.frame_skip]
            context_length = int(data["context_length"].item())
            future_length = int(data["future_length"].item())
            memory_condition_length = int(data["memory_condition_length"].item())
            has_reference_tail = bool(int(data["has_reference_tail"].item()))
            if has_reference_tail and self.memory_condition_length == 0:
                main_length = context_length + future_length
                cached_video = cached_video[:main_length]
                cached_actions = cached_actions[:main_length]
                cached_poses = cached_poses[:main_length]
                cached_timestamp = cached_timestamp[:main_length]
                memory_condition_length = 0
                has_reference_tail = False
            labels = {
                "test_type_id": torch.tensor(int(data["test_type_id"].item()), dtype=torch.long),
                "matched_history_index": torch.tensor(int(data["matched_history_index"].item()), dtype=torch.long),
                "match_valid": torch.tensor(bool(data["match_valid"].item()), dtype=torch.bool),
                "returns_to_seen_place": torch.tensor(float(data["returns_to_seen_place"].item()), dtype=torch.float32),
                "landmark_visible": torch.tensor(float(data["landmark_visible"].item()), dtype=torch.float32),
                "object_exists_at_return": torch.tensor(
                    float(data["object_exists_at_return"].item()), dtype=torch.float32
                ),
            }
            return {
                "video": cached_video,
                "actions": cached_actions,
                "poses": cached_poses,
                "timestamp": cached_timestamp,
                "future_actions": torch.from_numpy(np.asarray(data["future_actions"])).float(),
                "target_frames": _npz_video_to_tensor(data["target_frames"]),
                "memory_labels": labels,
                "test_type": entry.get("test_type", "unknown"),
                "episode_dir": entry.get("episode_dir", ""),
                "episode_family": entry.get("episode_family", ""),
                "query_index_in_video": torch.tensor(int(data["query_index_in_video"].item()), dtype=torch.long),
                "target_index_in_video": torch.tensor(int(data["target_index_in_video"].item()), dtype=torch.long),
                "query_t": torch.tensor(int(data["query_t"].item()), dtype=torch.long),
                "target_t": torch.tensor(int(data["target_t"].item()), dtype=torch.long),
                "window_center_t": torch.tensor(int(data["window_center_t"].item()), dtype=torch.long),
                "window_kind": str(data["window_kind"].item()),
                "generation_center_index_in_video": torch.tensor(
                    int(data["generation_center_index_in_video"].item()), dtype=torch.long
                ),
                "ptm_recent_end_index": torch.tensor(int(data["ptm_recent_end_index"].item()), dtype=torch.long),
                "history_context_end_index": torch.tensor(int(data["history_context_end_index"].item()), dtype=torch.long),
                "context_length": torch.tensor(context_length, dtype=torch.long),
                "future_length": torch.tensor(future_length, dtype=torch.long),
                "memory_condition_length": torch.tensor(memory_condition_length, dtype=torch.long),
                "has_reference_tail": torch.tensor(has_reference_tail, dtype=torch.bool),
            }

    def __getitem__(self, idx: int) -> dict[str, Any]:
        if not self.use_npz_cache:
            raise RuntimeError(
                "PTMWorldMemDataset requires NPZ cache for all splits. "
                f"Set npz_cache_dir and ensure split '{self.split}' is in npz_cache_splits."
            )
        return self._getitem_npz(idx)


def collate_worldmem_ptm(batch: list[dict[str, Any]]) -> dict[str, Any]:
    labels: dict[str, list[Any]] = {}
    for item in batch:
        for key, value in item["memory_labels"].items():
            if torch.is_tensor(value):
                labels.setdefault(key, []).append(value)
            elif isinstance(value, (bool, int, float)):
                labels.setdefault(key, []).append(value)
    tensor_labels = {}
    for key, values in labels.items():
        dtype = torch.long if key in {"test_type_id", "matched_history_index"} else torch.float32
        if values and torch.is_tensor(values[0]):
            tensor_labels[key] = torch.stack([v.to(dtype=dtype) for v in values], dim=0)
        else:
            tensor_labels[key] = torch.tensor(values, dtype=dtype)
    return {
        "video": torch.stack([item["video"] for item in batch], dim=0),
        "actions": torch.stack([item["actions"] for item in batch], dim=0),
        "poses": torch.stack([item["poses"] for item in batch], dim=0),
        "timestamp": torch.stack([item["timestamp"] for item in batch], dim=0),
        "future_actions": torch.stack([item["future_actions"] for item in batch], dim=0),
        "target_frames": torch.stack([item["target_frames"] for item in batch], dim=0),
        "memory_labels": tensor_labels,
        "test_type": [item["test_type"] for item in batch],
        "episode_dir": [item["episode_dir"] for item in batch],
        "episode_family": [item.get("episode_family", "") for item in batch],
        "query_index_in_video": torch.stack([item["query_index_in_video"] for item in batch], dim=0),
        "target_index_in_video": torch.stack([item["target_index_in_video"] for item in batch], dim=0),
        "query_t": torch.stack([item["query_t"] for item in batch], dim=0),
        "target_t": torch.stack([item["target_t"] for item in batch], dim=0),
        "window_center_t": torch.stack([item["window_center_t"] for item in batch], dim=0),
        "window_kind": [item["window_kind"] for item in batch],
        "generation_center_index_in_video": torch.stack(
            [item["generation_center_index_in_video"] for item in batch], dim=0
        ),
        "ptm_recent_end_index": torch.stack([item["ptm_recent_end_index"] for item in batch], dim=0),
        "history_context_end_index": torch.stack([item["history_context_end_index"] for item in batch], dim=0),
        "context_length": torch.stack([item["context_length"] for item in batch], dim=0),
        "future_length": torch.stack([item["future_length"] for item in batch], dim=0),
        "memory_condition_length": torch.stack([item["memory_condition_length"] for item in batch], dim=0),
        "has_reference_tail": torch.stack([item["has_reference_tail"] for item in batch], dim=0),
    }
