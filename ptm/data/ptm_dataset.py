from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from .schemas import TEST_TYPE_TO_ID, oasis_action_to_vector, read_jsonl, required_episode_files

try:
    import cv2
except ImportError:  # pragma: no cover - depends on local env.
    cv2 = None

try:
    import imageio.v3 as iio
except ImportError:  # pragma: no cover - depends on local env.
    iio = None


@dataclass(frozen=True)
class IndexedTest:
    episode_dir: Path
    test_idx: int


def _episode_dirs(data_root: Path, split: str | None) -> list[Path]:
    if split is not None and (data_root / split).exists():
        root = data_root / split
    else:
        root = data_root
    required = required_episode_files()
    dirs = sorted(
        path
        for path in root.glob("episode_*")
        if path.is_dir()
        and not path.name.endswith(".lock")
        and all((path / name).exists() for name in required)
    )
    if not dirs and all((root / name).exists() for name in required_episode_files()):
        dirs = [root]
    return dirs


def _numpy_frame_to_tensor(frame: np.ndarray) -> torch.Tensor:
    if frame.ndim == 2:
        frame = np.repeat(frame[..., None], 3, axis=2)
    if frame.shape[-1] > 3:
        frame = frame[..., :3]
    frame = np.ascontiguousarray(frame)
    return torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0


def _read_mp4(path: Path) -> torch.Tensor:
    errors = []
    if cv2 is not None:
        cap = cv2.VideoCapture(str(path))
        frames = []
        while cap.isOpened():
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(_numpy_frame_to_tensor(frame))
        cap.release()
        if frames:
            return torch.stack(frames, dim=0)
        errors.append("opencv returned no frames")

    if iio is not None:
        try:
            frames = [_numpy_frame_to_tensor(frame) for frame in iio.imiter(path)]
        except Exception as exc:  # pragma: no cover - backend specific.
            errors.append(f"imageio failed: {exc}")
        else:
            if frames:
                return torch.stack(frames, dim=0)
            errors.append("imageio returned no frames")

    details = "; ".join(errors) if errors else "no mp4 backend available"
    raise RuntimeError(f"could not read frames from {path}: {details}")


def _resolution_size(resolution: int | tuple[int, int] | list[int] | None) -> tuple[int, int] | None:
    if resolution is None:
        return None
    if isinstance(resolution, int):
        return (resolution, resolution)
    values = tuple(int(value) for value in resolution)
    if len(values) != 2:
        raise ValueError(f"resolution must be an int or [height,width], got {resolution!r}")
    return values


def _read_frames(episode_dir: Path, resolution: int | tuple[int, int] | list[int] | None) -> torch.Tensor:
    if (episode_dir / "frames.mp4").exists():
        video = _read_mp4(episode_dir / "frames.mp4")
    elif (episode_dir / "frames.npz").exists():
        frames = np.load(episode_dir / "frames.npz")["frames"]
        video = torch.from_numpy(frames).permute(0, 3, 1, 2).float() / 255.0
    else:
        raise FileNotFoundError(f"{episode_dir} has neither frames.mp4 nor frames.npz")
    size = _resolution_size(resolution)
    if size is not None and tuple(video.shape[-2:]) != size:
        video = F.interpolate(video, size=size, mode="bilinear", align_corners=False)
    return video


def _pad_or_trim_time(tensor: torch.Tensor, length: int) -> torch.Tensor:
    if tensor.shape[0] == length:
        return tensor
    if tensor.shape[0] > length:
        return tensor[-length:]
    pad_shape = (length - tensor.shape[0],) + tuple(tensor.shape[1:])
    return torch.cat([torch.zeros(pad_shape, dtype=tensor.dtype), tensor], dim=0)


class PTMDataset(Dataset):
    """Loads PTM episode directories and samples future-test training items."""

    def __init__(
        self,
        data_root: str | Path,
        split: str | None = "train",
        history_length: int = 64,
        future_length: int = 64,
        resolution: int | None = 128,
        max_history_candidates: int = 256,
    ):
        super().__init__()
        self.data_root = Path(data_root)
        self.split = split
        self.history_length = int(history_length)
        self.future_length = int(future_length)
        self.resolution = resolution
        self.max_history_candidates = int(max_history_candidates)
        self.episode_dirs = _episode_dirs(self.data_root, split)
        if not self.episode_dirs:
            raise FileNotFoundError(f"no episode directories found under {self.data_root}")

        self.tests: list[IndexedTest] = []
        self._test_cache: dict[Path, list[dict[str, Any]]] = {}
        for episode_dir in self.episode_dirs:
            tests = read_jsonl(episode_dir / "tests.jsonl")
            self._test_cache[episode_dir] = tests
            self.tests.extend(IndexedTest(episode_dir, i) for i in range(len(tests)))
        if not self.tests:
            raise ValueError(f"no tests found under {self.data_root}")

        self._video_cache: dict[Path, torch.Tensor] = {}
        self._actions_cache: dict[Path, torch.Tensor] = {}

    def __len__(self) -> int:
        return len(self.tests)

    def _video(self, episode_dir: Path) -> torch.Tensor:
        if episode_dir not in self._video_cache:
            self._video_cache[episode_dir] = _read_frames(episode_dir, self.resolution)
        return self._video_cache[episode_dir]

    def _actions(self, episode_dir: Path) -> torch.Tensor:
        if episode_dir not in self._actions_cache:
            records = read_jsonl(episode_dir / "actions.jsonl")
            vectors = [oasis_action_to_vector(record["oasis_action"]) for record in records]
            self._actions_cache[episode_dir] = torch.from_numpy(np.stack(vectors)).float()
        return self._actions_cache[episode_dir]

    def __getitem__(self, idx: int) -> dict[str, Any]:
        indexed = self.tests[idx]
        test = self._test_cache[indexed.episode_dir][indexed.test_idx]
        video = self._video(indexed.episode_dir)
        actions = self._actions(indexed.episode_dir)
        num_frames = min(video.shape[0], actions.shape[0])

        h0 = max(0, int(test["history_start_t"]))
        h1 = min(num_frames - 1, int(test["history_end_t"]))
        f0 = max(0, int(test["future_start_t"]))
        f1 = min(num_frames - 1, int(test["future_end_t"]))
        target_t = min(num_frames - 1, int(test["target_t"]))

        history_frames = _pad_or_trim_time(video[h0 : h1 + 1], self.history_length)
        past_actions = _pad_or_trim_time(actions[h0 : h1 + 1], self.history_length)
        future_actions = _pad_or_trim_time(actions[f0 : f1 + 1], self.future_length)
        target_frame = video[target_t]

        labels = dict(test.get("labels", {}))
        labels["test_type_id"] = int(test.get("test_type_id", TEST_TYPE_TO_ID[test["test_type"]]))
        matched_t = max(0, min(int(labels.get("matched_history_t", h0)), num_frames - 1))
        matched_history_index = matched_t - h0
        labels["match_valid"] = float(h0 <= matched_t <= h1 and matched_history_index < self.max_history_candidates)
        labels["matched_history_index"] = matched_history_index if labels["match_valid"] else 0
        labels["returns_to_seen_place"] = float(bool(labels.get("returns_to_seen_place", False)))
        labels["landmark_visible"] = float(bool(labels.get("landmark_visible", False)))
        labels["object_exists_at_return"] = float(bool(labels.get("object_exists_at_return", False)))

        return {
            "history_frames": history_frames,
            "past_actions": past_actions,
            "future_actions": future_actions,
            "memory_labels": labels,
            "target_frames": target_frame,
            "episode_dir": str(indexed.episode_dir),
            "test_type": test["test_type"],
        }


def collate_ptm_batch(batch: list[dict[str, Any]]) -> dict[str, Any]:
    labels: dict[str, list[Any]] = {}
    for item in batch:
        for key, value in item["memory_labels"].items():
            if isinstance(value, (bool, int, float)):
                labels.setdefault(key, []).append(value)

    tensor_labels: dict[str, torch.Tensor] = {}
    for key, values in labels.items():
        dtype = torch.long if key in {"test_type_id", "matched_history_index"} else torch.float32
        tensor_labels[key] = torch.tensor(values, dtype=dtype)

    return {
        "history_frames": torch.stack([item["history_frames"] for item in batch], dim=0),
        "past_actions": torch.stack([item["past_actions"] for item in batch], dim=0),
        "future_actions": torch.stack([item["future_actions"] for item in batch], dim=0),
        "memory_labels": tensor_labels,
        "target_frames": torch.stack([item["target_frames"] for item in batch], dim=0),
        "episode_dir": [item["episode_dir"] for item in batch],
        "test_type": [item["test_type"] for item in batch],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="PTM dataset loader debug entry point.")
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--history_length", type=int, default=64)
    parser.add_argument("--future_length", type=int, default=64)
    parser.add_argument("--resolution", type=int, default=128)
    parser.add_argument("--debug_batch", action="store_true")
    parser.add_argument("--batch_size", type=int, default=2)
    args = parser.parse_args()

    dataset = PTMDataset(
        args.data_root,
        split=args.split,
        history_length=args.history_length,
        future_length=args.future_length,
        resolution=args.resolution,
    )
    print(f"episodes={len(dataset.episode_dirs)} tests={len(dataset)}")
    if args.debug_batch:
        loader = DataLoader(dataset, batch_size=args.batch_size, collate_fn=collate_ptm_batch)
        batch = next(iter(loader))
        summary = {
            key: tuple(value.shape)
            for key, value in batch.items()
            if torch.is_tensor(value)
        }
        summary["labels"] = {key: tuple(value.shape) for key, value in batch["memory_labels"].items()}
        print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
