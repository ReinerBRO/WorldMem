from __future__ import annotations

import argparse
import json
import math
import os
import random
import shutil
import signal
import socket
import time
import traceback
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np
from tqdm import tqdm

from .build_tests import (
    MIN_FUTURE_ACTION_RATE,
    MIN_FUTURE_PATH_LENGTH,
    MIN_FUTURE_YAW_PATH,
    build_tests_for_episode,
)
from .schemas import (
    EventRecord,
    PoseRecord,
    compact_minedojo_action_to_dict,
    compact_minedojo_action_to_oasis,
    normalize_rgb_array,
    read_jsonl,
    required_episode_files,
    write_jsonl,
)

try:
    import cv2
except ImportError:  # pragma: no cover - exercised in minimal env smoke tests.
    cv2 = None


COMPACT_ACTIONS: dict[str, np.ndarray] = {
    "noop": np.array([0, 0, 0, 12, 12, 0, 0, 0], dtype=np.int32),
    "forward": np.array([1, 0, 0, 12, 12, 0, 0, 0], dtype=np.int32),
    "back": np.array([2, 0, 0, 12, 12, 0, 0, 0], dtype=np.int32),
    "left": np.array([0, 1, 0, 12, 12, 0, 0, 0], dtype=np.int32),
    "right": np.array([0, 2, 0, 12, 12, 0, 0, 0], dtype=np.int32),
    "turn_left": np.array([0, 0, 0, 12, 11, 0, 0, 0], dtype=np.int32),
    "turn_right": np.array([0, 0, 0, 12, 13, 0, 0, 0], dtype=np.int32),
    "look_up": np.array([0, 0, 0, 11, 12, 0, 0, 0], dtype=np.int32),
    "look_down": np.array([0, 0, 0, 13, 12, 0, 0, 0], dtype=np.int32),
    "use": np.array([0, 0, 0, 12, 12, 1, 0, 0], dtype=np.int32),
    "place": np.array([0, 0, 0, 12, 12, 6, 0, 0], dtype=np.int32),
}


class EpisodeTimeoutError(TimeoutError):
    pass


def _arm_episode_timeout(seconds: float):
    if seconds <= 0:
        return None

    def _raise_timeout(_signum, _frame):
        raise EpisodeTimeoutError(f"episode exceeded timeout {seconds}s")

    old_handler = signal.getsignal(signal.SIGALRM)
    old_timer = signal.setitimer(signal.ITIMER_REAL, seconds)
    signal.signal(signal.SIGALRM, _raise_timeout)
    return old_handler, old_timer


def _restore_episode_timeout(state) -> None:
    if state is None:
        return
    old_handler, old_timer = state
    signal.setitimer(signal.ITIMER_REAL, old_timer[0], old_timer[1])
    signal.signal(signal.SIGALRM, old_handler)


@contextmanager
def _episode_timeout(seconds: float):
    state = _arm_episode_timeout(seconds)
    try:
        yield
    finally:
        _restore_episode_timeout(state)

FAMILY_WEIGHTS = {
    "normal_rollout": 30,
    "loop_return_pos": 15,
    "loop_return_neg": 15,
    "landmark_revisit_pos": 10,
    "landmark_revisit_neg": 10,
    "object_persistence_pos": 10,
    "object_persistence_neg": 10,
}

FAMILY_EXPECTED_LABELS = {
    "normal_rollout": {
        "returns_to_seen_place": False,
        "landmark_visible": False,
        "object_exists_at_return": False,
    },
    "loop_return_pos": {
        "returns_to_seen_place": True,
        "landmark_visible": False,
        "object_exists_at_return": False,
    },
    "loop_return_neg": {
        "returns_to_seen_place": False,
        "landmark_visible": False,
        "object_exists_at_return": False,
    },
    "landmark_revisit_pos": {
        "returns_to_seen_place": True,
        "landmark_visible": True,
        "object_exists_at_return": False,
    },
    "landmark_revisit_neg": {
        "returns_to_seen_place": True,
        "landmark_visible": False,
        "object_exists_at_return": False,
    },
    "object_persistence_pos": {
        "returns_to_seen_place": True,
        "landmark_visible": False,
        "object_exists_at_return": True,
    },
    "object_persistence_neg": {
        "returns_to_seen_place": True,
        "landmark_visible": False,
        "object_exists_at_return": False,
    },
}


def _family_schedule(spec: str, num_episodes: int | None = None) -> list[str]:
    spec = spec.strip()
    if spec in {"", "balanced", "default", "fixed"}:
        total = int(num_episodes or sum(FAMILY_WEIGHTS.values()))
        raw_counts = {
            family: (weight / sum(FAMILY_WEIGHTS.values())) * total
            for family, weight in FAMILY_WEIGHTS.items()
        }
        counts = {family: int(math.floor(value)) for family, value in raw_counts.items()}
        remaining = total - sum(counts.values())
        for family, _fraction in sorted(
            ((family, raw_counts[family] - counts[family]) for family in FAMILY_WEIGHTS),
            key=lambda item: item[1],
            reverse=True,
        )[:remaining]:
            counts[family] += 1
        families = []
        for family, count in counts.items():
            families.extend([family] * count)
        random.Random(0).shuffle(families)
        return families

    families = [item.strip() for item in spec.split(",") if item.strip()]
    unknown = [family for family in families if family not in FAMILY_WEIGHTS]
    if unknown:
        valid = ", ".join(FAMILY_WEIGHTS)
        raise ValueError(f"unknown PTM family {unknown}; use exact families: {valid}")
    return families


def _family_kind(family: str) -> str:
    if family == "normal_rollout":
        return "normal"
    if family.startswith("loop_return_"):
        return "loop"
    if family.startswith("landmark_revisit_"):
        return "landmark"
    if family.startswith("object_persistence_"):
        return "object"
    raise ValueError(f"unknown PTM family: {family}")


def _expected_labels(family: str) -> dict[str, bool]:
    return dict(FAMILY_EXPECTED_LABELS[family])


def _is_negative_family(family: str) -> bool:
    return family.endswith("_neg")


def _repeat(actions: list[np.ndarray], name: str, count: int) -> None:
    actions.extend([COMPACT_ACTIONS[name].copy() for _ in range(max(0, count))])


def scripted_actions(family: str, frames_per_episode: int, rng: random.Random) -> np.ndarray:
    kind = _family_kind(family)
    actions: list[np.ndarray] = []
    if kind == "loop":
        leg = max(6, frames_per_episode // 4)
        _repeat(actions, "forward", leg)
        if family.endswith("_pos"):
            _repeat(actions, "back", leg)
            _repeat(actions, "turn_left", 4)
            _repeat(actions, "turn_right", 4)
        else:
            _repeat(actions, "turn_right", 12)
            _repeat(actions, "forward", leg)
            _repeat(actions, "turn_left", 4)
    elif kind == "landmark":
        _repeat(actions, "noop", 8)
        _repeat(actions, "forward", frames_per_episode // 4)
        _repeat(actions, "back", frames_per_episode // 5)
        if _is_negative_family(family):
            _repeat(actions, "turn_right", 14)
        else:
            _repeat(actions, "look_up", 4)
            _repeat(actions, "look_down", 4)
    elif kind == "object":
        _repeat(actions, "noop", 8)
        _repeat(actions, "forward", frames_per_episode // 3)
        _repeat(actions, "back", frames_per_episode // 3)
        if _is_negative_family(family):
            _repeat(actions, "turn_left", 8)
    else:
        choices = ["forward", "back", "left", "right", "turn_left", "turn_right", "look_up", "look_down"]
        while len(actions) < frames_per_episode:
            name = rng.choices(
                choices,
                weights=[0.36, 0.08, 0.08, 0.08, 0.18, 0.18, 0.02, 0.02],
                k=1,
            )[0]
            _repeat(actions, name, rng.randint(2, 8))

    while len(actions) < frames_per_episode:
        _repeat(actions, "noop", 1)
    return np.stack(actions[:frames_per_episode], axis=0)


def simulate_pose(actions: np.ndarray, biome: str = "plains") -> list[PoseRecord]:
    x, y, z, yaw, pitch = 0.0, 64.0, 0.0, 0.0, 0.0
    poses: list[PoseRecord] = []
    for t, action in enumerate(actions):
        yaw = (yaw + float(action[4] - 12) * 7.5) % 360.0
        pitch = max(-80.0, min(80.0, pitch + float(action[3] - 12) * 5.0))
        step = 0.35
        rad = math.radians(yaw)
        if action[0] == 1:
            x += math.sin(rad) * step
            z += math.cos(rad) * step
        elif action[0] == 2:
            x -= math.sin(rad) * step
            z -= math.cos(rad) * step
        if action[1] == 1:
            x -= math.cos(rad) * step
            z += math.sin(rad) * step
        elif action[1] == 2:
            x += math.cos(rad) * step
            z -= math.sin(rad) * step
        poses.append(PoseRecord(t=t, x=x, y=y, z=z, yaw=yaw, pitch=pitch, biome=biome))
    return poses


def mock_frames(actions: np.ndarray, family: str, seed: int, height: int, width: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    poses = simulate_pose(actions)
    frames = np.zeros((len(actions), height, width, 3), dtype=np.uint8)
    base = rng.integers(20, 80, size=(3,), dtype=np.uint8)
    kind = _family_kind(family)
    expected = _expected_labels(family)
    for t, pose in enumerate(poses):
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        sky = np.array([90, 150, 210], dtype=np.uint8)
        ground = np.array([70, 140, 70], dtype=np.uint8)
        frame[: height // 2] = sky
        frame[height // 2 :] = ground
        stripe_x = int((pose.x * 11 + pose.z * 5 + width / 2) % width)
        frame[:, max(0, stripe_x - 3) : min(width, stripe_x + 3)] = base + np.array([60, 20, 10], dtype=np.uint8)
        visible_landmark = kind == "landmark" and expected["landmark_visible"]
        visible_object = kind == "object" and expected["object_exists_at_return"]
        if (visible_landmark or visible_object) and t >= 8:
            color = (230, 180, 40) if visible_landmark else (210, 70, 60)
            cx = int(width * 0.55 + math.sin(math.radians(pose.yaw)) * width * 0.15)
            cy = int(height * 0.45)
            x0, x1 = max(0, cx - 10), min(width, cx + 10)
            y0, y1 = max(0, cy - 28), min(height, cy + 28)
            frame[y0:y1, x0:x1] = color
        frame[max(0, height - 4) : height, 0 : min(width, 8 + t % max(1, width - 8))] = (255, 255, 255)
        frames[t] = frame
    return frames


def make_events(family: str, poses: list[PoseRecord]) -> list[EventRecord]:
    kind = _family_kind(family)
    if kind not in {"landmark", "object"}:
        return []
    event_t = min(10, len(poses) - 1)
    delete_t = min(len(poses) - 1, event_t + max(8, len(poses) // 3))
    pose = poses[event_t]
    block_type = "yellow_wool" if kind == "landmark" else "torch"
    position = {
        "x": int(round(pose.x + math.sin(math.radians(pose.yaw)) * 4)),
        "y": int(round(pose.y)),
        "z": int(round(pose.z + math.cos(math.radians(pose.yaw)) * 4)),
    }
    events = [
        EventRecord(
            t=event_t,
            event_type="set_landmark" if kind == "landmark" else "set_object",
            block_type=block_type,
            agent_pose={
                "x": pose.x,
                "y": pose.y,
                "z": pose.z,
                "yaw": pose.yaw,
                "pitch": pose.pitch,
            },
            target_block_position=position,
            success_verified_by_voxel=True,
            success_verified_by_inventory_delta=False,
            metadata={"synthetic_mock_event": True, "ptm_family": family},
        )
    ]
    if _is_negative_family(family):
        events.append(
            EventRecord(
                t=delete_t,
                event_type="delete_landmark" if kind == "landmark" else "delete_object",
                block_type="air",
                agent_pose=None,
                target_block_position=position,
                success_verified_by_voxel=True,
                success_verified_by_inventory_delta=False,
                metadata={"synthetic_mock_event": True, "ptm_family": family},
            )
        )
    return events


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
    return value


def _inventory_snapshot(obs: dict[str, Any]) -> dict[str, Any]:
    inv = obs.get("inventory", {})
    return _to_jsonable(inv) if isinstance(inv, dict) else {"raw": _to_jsonable(inv)}


def _inventory_quantity_total(snapshot: dict[str, Any]) -> float:
    total = 0.0
    for key, value in snapshot.items():
        if "quantity" in key or "count" in key:
            arr = np.asarray(value, dtype=object).reshape(-1)
            for item in arr:
                try:
                    total += float(item)
                except (TypeError, ValueError):
                    pass
    return total


def _voxel_snapshot(obs: dict[str, Any]) -> dict[str, Any]:
    voxels = obs.get("voxels", {})
    if not isinstance(voxels, dict):
        return {"raw": _to_jsonable(voxels)}
    out = {}
    for key in ("block_name", "block_meta", "is_collidable", "is_liquid"):
        if key in voxels:
            out[key] = _to_jsonable(voxels[key])
    return out


def _voxel_block_names(snapshot: dict[str, Any]) -> np.ndarray:
    names = snapshot.get("block_name", [])
    return np.asarray(names, dtype=str)


def _changed_voxel(before: dict[str, Any], after: dict[str, Any]) -> tuple[bool, dict[str, int] | None, str | None]:
    before_names = _voxel_block_names(before)
    after_names = _voxel_block_names(after)
    if before_names.shape != after_names.shape or before_names.size == 0:
        return False, None, None
    changed = before_names != after_names
    if not bool(changed.any()):
        return False, None, None
    idx = np.argwhere(changed)[0]
    center = np.asarray(before_names.shape) // 2
    rel = idx - center
    position = {"x": int(rel[-1]), "y": int(rel[0]), "z": int(rel[1]) if len(rel) > 1 else 0}
    return True, position, str(after_names[tuple(idx)])


def _build_event_from_transition(
    t: int,
    family: str,
    action: np.ndarray,
    pose: PoseRecord,
    before_obs: dict[str, Any],
    after_obs: dict[str, Any],
) -> EventRecord | None:
    if int(action[5]) != 6:
        return None
    before_voxels = _voxel_snapshot(before_obs)
    after_voxels = _voxel_snapshot(after_obs)
    before_inv = _inventory_snapshot(before_obs)
    after_inv = _inventory_snapshot(after_obs)
    voxel_changed, target_position, block_name = _changed_voxel(before_voxels, after_voxels)
    inv_delta = _inventory_quantity_total(after_inv) < _inventory_quantity_total(before_inv)
    kind = _family_kind(family)
    event_type = "place_landmark" if kind == "landmark" else "place_block"
    return EventRecord(
        t=t,
        event_type=event_type,
        block_type=block_name,
        agent_pose={"x": pose.x, "y": pose.y, "z": pose.z, "yaw": pose.yaw, "pitch": pose.pitch},
        target_block_position=target_position,
        success_verified_by_voxel=voxel_changed,
        success_verified_by_inventory_delta=inv_delta,
        metadata={
            "verification_source": "minedojo_obs_transition",
            "action_function": int(action[5]),
        },
    )


def _target_block_position_from_pose(pose: PoseRecord, distance: int = 4) -> dict[str, int]:
    return {
        "x": int(round(pose.x + math.sin(math.radians(pose.yaw)) * distance)),
        "y": int(round(pose.y)),
        "z": int(round(pose.z + math.cos(math.radians(pose.yaw)) * distance)),
    }


def _setblock(env: Any, x: int, y: int, z: int, block: str) -> None:
    if not hasattr(env, "execute_cmd"):
        raise RuntimeError("MineDojo env does not expose execute_cmd; cannot create PTM command event")
    env.execute_cmd(f"/setblock {x} {y} {z} {block}")


def _set_family_blocks(env: Any, family: str, position: dict[str, int], block: str) -> None:
    kind = _family_kind(family)
    x, y, z = position["x"], position["y"], position["z"]
    if kind == "landmark":
        for dy in range(3):
            _setblock(env, x, y + dy, z, block)
    else:
        _setblock(env, x, y, z, block)


def _command_event(
    t: int,
    family: str,
    pose: PoseRecord | None,
    position: dict[str, int],
    event_type: str,
    block_type: str,
) -> EventRecord:
    agent_pose = None
    if pose is not None:
        agent_pose = {"x": pose.x, "y": pose.y, "z": pose.z, "yaw": pose.yaw, "pitch": pose.pitch}
    return EventRecord(
        t=t,
        event_type=event_type,
        block_type=block_type,
        agent_pose=agent_pose,
        target_block_position=position,
        success_verified_by_voxel=True,
        success_verified_by_inventory_delta=False,
        metadata={"verification_source": "minedojo_execute_cmd_setblock", "ptm_family": family},
    )


def write_episode(
    episode_dir: Path,
    frames: np.ndarray,
    actions: np.ndarray,
    poses: list[PoseRecord],
    events: list[EventRecord],
    tests,
    metadata: dict[str, Any],
    fps: float = 10.0,
    frame_storage: str = "auto",
    inventory_records: list[dict[str, Any]] | None = None,
    voxel_records: list[dict[str, Any]] | None = None,
) -> None:
    episode_dir.mkdir(parents=True, exist_ok=True)
    height, width = frames.shape[1:3]
    use_mp4 = frame_storage == "mp4" or (frame_storage == "auto" and cv2 is not None)
    if use_mp4:
        if cv2 is None:
            raise RuntimeError("frame_storage=mp4 requires opencv-python")
        video_path = episode_dir / "frames.mp4"
        writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
        for frame in frames:
            frame = normalize_rgb_array(frame)
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        writer.release()
        frame_index = [
            {"t": t, "video_path": "frames.mp4", "frame_idx": t, "timestamp": round(t / fps, 6)}
            for t in range(len(frames))
        ]
        metadata["frame_storage"] = "mp4"
    else:
        np.savez_compressed(episode_dir / "frames.npz", frames=frames.astype(np.uint8))
        frame_index = [
            {"t": t, "array_path": "frames.npz", "frame_idx": t, "timestamp": round(t / fps, 6)}
            for t in range(len(frames))
        ]
        metadata["frame_storage"] = "npz"
    action_records = [
        {
            "t": t,
            "minedojo_action": compact_minedojo_action_to_dict(action),
            "oasis_action": compact_minedojo_action_to_oasis(action),
        }
        for t, action in enumerate(actions)
    ]
    write_jsonl(episode_dir / "frames_index.jsonl", frame_index)
    write_jsonl(episode_dir / "actions.jsonl", action_records)
    write_jsonl(episode_dir / "poses.jsonl", poses)
    write_jsonl(episode_dir / "events.jsonl", events)
    write_jsonl(episode_dir / "tests.jsonl", tests)
    if inventory_records is not None:
        write_jsonl(episode_dir / "inventory.jsonl", inventory_records)
    if voxel_records is not None:
        write_jsonl(episode_dir / "voxels.jsonl", voxel_records)
    with (episode_dir / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, sort_keys=True)


def make_minedojo_env(args: argparse.Namespace, seed: int) -> Any:
    try:
        import minedojo
    except ImportError as exc:
        raise RuntimeError("MineDojo backend requested but minedojo is not installed") from exc

    return minedojo.make(
        task_id="open-ended",
        image_size=(args.height, args.width),
        world_seed=seed,
        seed=seed + 1,
        generate_world_type="specified_biome",
        specified_biome=args.env_type,
        initial_weather=args.weather,
        use_voxel=True,
        voxel_size=dict(xmin=-4, ymin=-4, zmin=-4, xmax=4, ymax=4, zmax=4),
        fast_reset=args.fast_reset,
        fast_reset_random_teleport_range=args.fast_reset_random_teleport_range,
    )


def close_minedojo_env(env: Any) -> None:
    try:
        env.close()
    except Exception:
        pass


def collect_with_minedojo(
    args: argparse.Namespace,
    family: str,
    seed: int,
    env: Any | None = None,
) -> tuple[np.ndarray, np.ndarray, list[PoseRecord], list[EventRecord], list[dict[str, Any]], list[dict[str, Any]]]:
    rng = random.Random(seed)
    actions = scripted_actions(family, args.frames_per_episode, rng)
    owns_env = env is None
    if env is None:
        env = make_minedojo_env(args, seed)
    elif hasattr(env, "seed"):
        env.seed(seed + 1)
    frames: list[np.ndarray] = []
    poses: list[PoseRecord] = []
    events: list[EventRecord] = []
    inventory_records: list[dict[str, Any]] = []
    voxel_records: list[dict[str, Any]] = []
    try:
        obs = env.reset()
        kind = _family_kind(family)
        command_event_t = min(10, args.frames_per_episode - 1)
        delete_event_t = min(args.frames_per_episode - 1, command_event_t + max(8, args.frames_per_episode // 3))
        command_position: dict[str, int] | None = None
        for t, action in enumerate(actions):
            rgb = normalize_rgb_array(obs["rgb"])
            stats = obs.get("location_stats", {})
            pos = np.asarray(stats.get("pos", [0.0, 64.0, 0.0]), dtype=np.float32).reshape(-1)
            pitch = float(np.asarray(stats.get("pitch", [0.0])).reshape(-1)[0])
            yaw = float(np.asarray(stats.get("yaw", [0.0])).reshape(-1)[0])
            frames.append(rgb)
            pose = PoseRecord(
                t=t,
                x=float(pos[0]),
                y=float(pos[1]),
                z=float(pos[2]),
                yaw=yaw,
                pitch=pitch,
                biome=args.env_type,
            )
            poses.append(pose)
            if kind in {"landmark", "object"} and t == command_event_t:
                command_position = _target_block_position_from_pose(pose)
                block = "yellow_wool" if kind == "landmark" else "torch"
                _set_family_blocks(env, family, command_position, block)
                events.append(
                    _command_event(
                        t=t,
                        family=family,
                        pose=pose,
                        position=command_position,
                        event_type="set_landmark" if kind == "landmark" else "set_object",
                        block_type=block,
                    )
                )
            if (
                kind in {"landmark", "object"}
                and _is_negative_family(family)
                and command_position is not None
                and t == delete_event_t
            ):
                _set_family_blocks(env, family, command_position, "air")
                events.append(
                    _command_event(
                        t=t,
                        family=family,
                        pose=pose,
                        position=command_position,
                        event_type="delete_landmark" if kind == "landmark" else "delete_object",
                        block_type="air",
                    )
                )
            inventory_records.append({"t": t, "inventory": _inventory_snapshot(obs), "inventory_change": _to_jsonable(obs.get("inventory_change", {}))})
            voxel_records.append({"t": t, "voxels": _voxel_snapshot(obs)})
            before_obs = obs
            obs, _reward, done, _info = env.step(action)
            event = _build_event_from_transition(t, family, action, pose, before_obs, obs)
            if event is not None:
                events.append(event)
            if done:
                obs = env.reset()
    finally:
        if owns_env:
            close_minedojo_env(env)
    return np.stack(frames, axis=0), actions, poses, events, inventory_records, voxel_records


def generate(args: argparse.Namespace) -> None:
    out_root = Path(args.out) / args.split
    out_root.mkdir(parents=True, exist_ok=True)
    schedule_total = args.schedule_total or args.num_episodes
    families = _family_schedule(args.families, schedule_total)
    if not families:
        raise ValueError("at least one family is required")
    shared_env: Any | None = None

    try:
        for local_idx in tqdm(range(args.num_episodes), desc="PTM episodes"):
            episode_idx = args.episode_offset + local_idx
            episode_dir = out_root / f"episode_{episode_idx:06d}"
            if args.skip_existing and _complete_episode_exists(episode_dir):
                print(f"PTM_EPISODE_SKIP episode={episode_idx} reason=complete path={episode_dir}", flush=True)
                continue

            max_attempts = args.episode_retries + 1
            for attempt in range(max_attempts):
                lock_dir: Path | None = None
                tmp_dir: Path | None = None
                timer_state = None
                try:
                    timer_state = _arm_episode_timeout(args.episode_timeout_seconds)
                    if args.episode_locks:
                        lock_dir = _acquire_episode_lock(episode_dir, args.lock_stale_seconds)
                        if lock_dir is None:
                            print(f"PTM_EPISODE_SKIP episode={episode_idx} reason=locked path={episode_dir}", flush=True)
                            break
                        if args.skip_existing and _complete_episode_exists(episode_dir):
                            print(f"PTM_EPISODE_SKIP episode={episode_idx} reason=complete_after_lock path={episode_dir}", flush=True)
                            break

                    family = families[episode_idx % len(families)]
                    expected = _expected_labels(family)
                    seed_base = args.seed + episode_idx * 9973
                    seed = seed_base + attempt * args.episode_retry_seed_stride
                    rng = random.Random(seed)
                    actions = scripted_actions(family, args.frames_per_episode, rng)
                    if args.backend == "mock":
                        frames = mock_frames(actions, family, seed, args.height, args.width)
                        poses = simulate_pose(actions, biome=args.env_type)
                        events = make_events(family, poses)
                        inventory_records = None
                        voxel_records = None
                        backend = "mock"
                    else:
                        if args.reuse_env:
                            if shared_env is None:
                                shared_env = make_minedojo_env(args, seed)
                            frames, actions, poses, events, inventory_records, voxel_records = collect_with_minedojo(
                                args, family, seed, env=shared_env
                            )
                        else:
                            frames, actions, poses, events, inventory_records, voxel_records = collect_with_minedojo(
                                args, family, seed
                            )
                        backend = "minedojo"

                    tests = build_tests_for_episode(
                        poses=poses,
                        actions=actions,
                        family=family,
                        events=events,
                        frames_per_episode=args.frames_per_episode,
                        expected_labels=expected,
                        history_length=args.history_length,
                        future_length=args.future_length,
                        stride=args.test_stride,
                    )
                    metadata = {
                        "episode_id": episode_idx,
                        "family": family,
                        "ptm_family": family,
                        "expected": expected,
                        "backend": backend,
                        "seed": seed,
                        "seed_base": seed_base,
                        "episode_attempt": attempt + 1,
                        "frames_per_episode": args.frames_per_episode,
                        "height": args.height,
                        "width": args.width,
                        "fps": args.fps,
                        "env_type": args.env_type,
                        "reuse_env": bool(args.reuse_env and args.backend == "minedojo"),
                        "fast_reset": bool(args.fast_reset and args.backend == "minedojo"),
                        "ptm_tests_filter_static_future": True,
                        "ptm_tests_min_future_path": MIN_FUTURE_PATH_LENGTH,
                        "ptm_tests_min_future_yaw_path": MIN_FUTURE_YAW_PATH,
                        "ptm_tests_min_future_action_rate": MIN_FUTURE_ACTION_RATE,
                        "ptm_tests_count": len(tests),
                    }
                    target_dir = episode_dir
                    if args.atomic_write:
                        tmp_dir = _episode_tmp_dir(episode_dir)
                        shutil.rmtree(tmp_dir, ignore_errors=True)
                        target_dir = tmp_dir
                    write_episode(
                        target_dir,
                        frames=frames,
                        actions=actions,
                        poses=poses,
                        events=events,
                        tests=tests,
                        metadata=metadata,
                        fps=args.fps,
                        frame_storage=args.frame_storage,
                        inventory_records=inventory_records,
                        voxel_records=voxel_records,
                    )
                    if args.atomic_write:
                        if not _complete_episode_exists(tmp_dir):
                            raise RuntimeError(f"atomic temp episode is incomplete: {tmp_dir}")
                        if episode_dir.exists():
                            shutil.rmtree(episode_dir)
                        tmp_dir.rename(episode_dir)
                        tmp_dir = None
                    print(f"PTM_EPISODE_DONE episode={episode_idx} backend={backend} family={family} path={episode_dir}", flush=True)
                    break
                except Exception as exc:
                    if args.reuse_env and shared_env is not None:
                        close_minedojo_env(shared_env)
                        shared_env = None
                    attempt_num = attempt + 1
                    print(
                        f"PTM_EPISODE_ERROR episode={episode_idx} attempt={attempt_num}/{max_attempts} "
                        f"error={type(exc).__name__}: {exc}",
                        flush=True,
                    )
                    traceback.print_exc()
                    if not args.continue_on_error:
                        raise
                    if attempt_num >= max_attempts:
                        break
                    print(
                        f"PTM_EPISODE_RETRY episode={episode_idx} next_attempt={attempt_num + 1}/{max_attempts} "
                        f"sleep={args.episode_retry_sleep}",
                        flush=True,
                    )
                    if args.episode_retry_sleep > 0:
                        time.sleep(args.episode_retry_sleep)
                finally:
                    _restore_episode_timeout(timer_state)
                    if tmp_dir is not None:
                        shutil.rmtree(tmp_dir, ignore_errors=True)
                    if lock_dir is not None:
                        _release_episode_lock(lock_dir)
    finally:
        if shared_env is not None:
            close_minedojo_env(shared_env)


def _complete_episode_exists(episode_dir: Path) -> bool:
    metadata_path = episode_dir / "metadata.json"
    if not metadata_path.is_file() or metadata_path.stat().st_size == 0:
        return False
    if not all((episode_dir / name).is_file() for name in required_episode_files()):
        return False
    if not any(frame_path.is_file() and frame_path.stat().st_size > 0 for frame_path in (episode_dir / "frames.mp4", episode_dir / "frames.npz")):
        return False
    try:
        for name in ("frames_index.jsonl", "actions.jsonl", "poses.jsonl", "tests.jsonl"):
            if not read_jsonl(episode_dir / name):
                return False
    except (OSError, json.JSONDecodeError):
        return False
    return True


def _episode_lock_path(episode_dir: Path) -> Path:
    return episode_dir.with_name(f"{episode_dir.name}.lock")


def _episode_tmp_dir(episode_dir: Path) -> Path:
    stamp = f"{socket.gethostname()}.{os.getpid()}.{time.time_ns()}"
    return episode_dir.with_name(f".{episode_dir.name}.tmp.{stamp}")


def _acquire_episode_lock(episode_dir: Path, stale_seconds: int) -> Path | None:
    lock_dir = _episode_lock_path(episode_dir)
    while True:
        try:
            lock_dir.mkdir()
        except FileExistsError:
            if _lock_owner_is_dead(lock_dir):
                shutil.rmtree(lock_dir, ignore_errors=True)
                continue
            try:
                age = time.time() - lock_dir.stat().st_mtime
            except FileNotFoundError:
                continue
            if stale_seconds >= 0 and age > stale_seconds:
                shutil.rmtree(lock_dir, ignore_errors=True)
                continue
            return None
        owner = {
            "host": socket.gethostname(),
            "pid": os.getpid(),
            "time": time.time(),
            "episode": episode_dir.name,
        }
        with (lock_dir / "owner.json").open("w", encoding="utf-8") as f:
            json.dump(owner, f, indent=2, sort_keys=True)
        return lock_dir


def _lock_owner_is_dead(lock_dir: Path) -> bool:
    try:
        with (lock_dir / "owner.json").open("r", encoding="utf-8") as f:
            owner = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return False
    if owner.get("host") != socket.gethostname():
        return False
    try:
        pid = int(owner.get("pid", -1))
    except (TypeError, ValueError):
        return False
    if pid <= 0 or pid == os.getpid():
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return True
    except PermissionError:
        return False
    return False


def _release_episode_lock(lock_dir: Path) -> None:
    shutil.rmtree(lock_dir, ignore_errors=True)


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate PTM MineDojo trajectories with future-test labels.")
    parser.add_argument("--out", default="ptm_minedojo_data/stage0")
    parser.add_argument("--num_episodes", type=int, default=100)
    parser.add_argument("--episode_offset", type=int, default=0)
    parser.add_argument("--schedule_total", type=int, default=None)
    parser.add_argument("--skip_existing", action="store_true")
    parser.add_argument("--frames_per_episode", type=int, default=128)
    parser.add_argument("--families", default="balanced")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--split", default="train", choices=["train", "val", "test"])
    parser.add_argument("--backend", default="minedojo", choices=["minedojo", "mock"])
    parser.add_argument("--height", type=int, default=128)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument("--frame_storage", default="auto", choices=["auto", "mp4", "npz"])
    parser.add_argument("--env_type", default="plains")
    parser.add_argument("--weather", default="clear")
    parser.add_argument("--history_length", type=int, default=64)
    parser.add_argument("--future_length", type=int, default=64)
    parser.add_argument("--test_stride", type=int, default=32)
    parser.add_argument("--lock_stale_seconds", type=int, default=7200)
    parser.add_argument("--no_episode_locks", dest="episode_locks", action="store_false")
    parser.add_argument("--no_atomic_write", dest="atomic_write", action="store_false")
    parser.add_argument("--continue_on_error", action="store_true")
    parser.add_argument("--episode_retries", type=int, default=2)
    parser.add_argument("--episode_retry_sleep", type=float, default=1.0)
    parser.add_argument("--episode_retry_seed_stride", type=int, default=1000003)
    parser.add_argument("--episode_timeout_seconds", type=float, default=0.0)
    parser.add_argument("--reuse_env", action="store_true")
    parser.add_argument("--no_fast_reset", dest="fast_reset", action="store_false")
    parser.add_argument("--fast_reset_random_teleport_range", type=int, default=200)
    parser.set_defaults(episode_locks=True, atomic_write=True, fast_reset=True)
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    if args.episode_retries < 0:
        raise ValueError("--episode_retries must be >= 0")
    if args.episode_retry_sleep < 0:
        raise ValueError("--episode_retry_sleep must be >= 0")
    if args.episode_retry_seed_stride < 1:
        raise ValueError("--episode_retry_seed_stride must be >= 1")
    if args.episode_timeout_seconds < 0:
        raise ValueError("--episode_timeout_seconds must be >= 0")
    if args.fast_reset_random_teleport_range < 0:
        raise ValueError("--fast_reset_random_teleport_range must be >= 0")
    if args.reuse_env and not args.fast_reset:
        raise ValueError("--reuse_env requires fast reset; do not pass --no_fast_reset")
    generate(args)


if __name__ == "__main__":
    main()
