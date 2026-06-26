from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np


ACTION_KEYS = [
    "inventory",
    "ESC",
    "hotbar.1",
    "hotbar.2",
    "hotbar.3",
    "hotbar.4",
    "hotbar.5",
    "hotbar.6",
    "hotbar.7",
    "hotbar.8",
    "hotbar.9",
    "forward",
    "back",
    "left",
    "right",
    "cameraY",
    "cameraX",
    "jump",
    "sneak",
    "sprint",
    "swapHands",
    "attack",
    "use",
    "pickItem",
    "drop",
]

TEST_TYPE_TO_ID = {
    "normal_rollout": 0,
    "loop_return": 1,
    "landmark_revisit": 2,
    "object_persistence": 3,
}
ID_TO_TEST_TYPE = {value: key for key, value in TEST_TYPE_TO_ID.items()}


@dataclass
class PoseRecord:
    t: int
    x: float
    y: float
    z: float
    yaw: float
    pitch: float
    biome: str = "unknown"


@dataclass
class EventRecord:
    t: int
    event_type: str
    block_type: str | None = None
    agent_pose: dict[str, float] | None = None
    target_block_position: dict[str, int] | None = None
    success_verified_by_voxel: bool = False
    success_verified_by_inventory_delta: bool = False
    metadata: dict[str, Any] | None = None


@dataclass
class TestRecord:
    query_t: int
    history_start_t: int
    history_end_t: int
    future_start_t: int
    future_end_t: int
    test_type: str
    target_t: int
    labels: dict[str, Any]

    def to_json(self) -> dict[str, Any]:
        data = asdict(self)
        data["test_type_id"] = TEST_TYPE_TO_ID[self.test_type]
        return data


def write_jsonl(path: str | Path, records: list[dict[str, Any] | Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            if hasattr(record, "to_json"):
                payload = record.to_json()
            elif hasattr(record, "__dataclass_fields__"):
                payload = asdict(record)
            else:
                payload = record
            f.write(json.dumps(payload, sort_keys=True) + "\n")


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def pose_distance(a: dict[str, Any] | PoseRecord, b: dict[str, Any] | PoseRecord) -> float:
    if isinstance(a, PoseRecord):
        a = asdict(a)
    if isinstance(b, PoseRecord):
        b = asdict(b)
    return math.sqrt((float(a["x"]) - float(b["x"])) ** 2 + (float(a["z"]) - float(b["z"])) ** 2)


def yaw_distance(a: float, b: float) -> float:
    diff = abs(float(a) - float(b)) % 360.0
    return min(diff, 360.0 - diff)


def compact_minedojo_action_to_dict(action: np.ndarray | list[int] | tuple[int, ...]) -> dict[str, Any]:
    arr = np.asarray(action).astype(float).tolist()
    return {
        "compact": arr,
        "forward": int(arr[0] == 1) if len(arr) > 0 else 0,
        "back": int(arr[0] == 2) if len(arr) > 0 else 0,
        "left": int(arr[1] == 1) if len(arr) > 1 else 0,
        "right": int(arr[1] == 2) if len(arr) > 1 else 0,
        "jump": int(len(arr) > 5 and arr[5] == 1),
        "attack": int(len(arr) > 5 and arr[5] == 3),
        "use": int(len(arr) > 5 and arr[5] == 1),
        "place": int(len(arr) > 5 and arr[5] == 6),
        "camera_dx": float(arr[4] - 12) if len(arr) > 4 else 0.0,
        "camera_dy": float(arr[3] - 12) if len(arr) > 3 else 0.0,
    }


def compact_minedojo_action_to_oasis(action: np.ndarray | list[int] | tuple[int, ...]) -> dict[str, float]:
    arr = np.asarray(action)
    oasis = {key: 0.0 for key in ACTION_KEYS}
    if arr.shape[0] > 0:
        oasis["forward"] = float(arr[0] == 1)
        oasis["back"] = float(arr[0] == 2)
    if arr.shape[0] > 1:
        oasis["left"] = float(arr[1] == 1)
        oasis["right"] = float(arr[1] == 2)
    if arr.shape[0] > 3:
        oasis["cameraY"] = float(arr[3] - 12)
    if arr.shape[0] > 4:
        oasis["cameraX"] = float(arr[4] - 12)
    if arr.shape[0] > 5:
        oasis["jump"] = float(arr[5] == 1)
        oasis["attack"] = float(arr[5] == 3)
        oasis["use"] = float(arr[5] in (1, 6))
    if arr.shape[0] > 7 and arr[7] == 1:
        oasis["hotbar.1"] = 1.0
    return oasis


def oasis_action_to_vector(action: dict[str, float]) -> np.ndarray:
    return np.asarray([float(action.get(key, 0.0)) for key in ACTION_KEYS], dtype=np.float32)


def normalize_rgb_array(rgb: np.ndarray) -> np.ndarray:
    rgb = np.asarray(rgb)
    if rgb.ndim != 3:
        raise ValueError(f"expected one RGB frame, got shape {rgb.shape}")
    if rgb.shape[0] in (1, 3, 4) and rgb.shape[-1] not in (3, 4):
        rgb = np.transpose(rgb, (1, 2, 0))
    if rgb.shape[-1] == 4:
        rgb = rgb[..., :3]
    if rgb.shape[-1] != 3:
        raise ValueError(f"expected RGB frame with 3 channels, got shape {rgb.shape}")
    if rgb.dtype != np.uint8:
        rgb = np.clip(rgb, 0, 255).astype(np.uint8)
    return rgb


def required_episode_files() -> tuple[str, ...]:
    return (
        "frames_index.jsonl",
        "actions.jsonl",
        "poses.jsonl",
        "events.jsonl",
        "tests.jsonl",
        "metadata.json",
    )
