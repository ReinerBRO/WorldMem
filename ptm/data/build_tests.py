from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict
from typing import Any

from .schemas import (
    EventRecord,
    PoseRecord,
    TestRecord,
    compact_minedojo_action_to_oasis,
    oasis_action_to_vector,
    pose_distance,
    yaw_distance,
)
from .window_filters import has_pose_discontinuity


MIN_FUTURE_PATH_LENGTH = 0.25
MIN_FUTURE_YAW_PATH = 5.0
MIN_FUTURE_ACTION_RATE = 0.05


def _nearest_history_pose(
    poses: Sequence[PoseRecord],
    query_pose: PoseRecord,
    start_t: int,
    end_t: int,
) -> tuple[int, float, float]:
    best_t = start_t
    best_pose_dist = float("inf")
    best_yaw_dist = float("inf")
    for t in range(start_t, max(start_t, end_t) + 1):
        dist = pose_distance(poses[t], query_pose)
        yaw = yaw_distance(poses[t].yaw, query_pose.yaw)
        score = dist + yaw / 180.0
        best_score = best_pose_dist + best_yaw_dist / 180.0
        if score < best_score:
            best_t = t
            best_pose_dist = dist
            best_yaw_dist = yaw
    return best_t, best_pose_dist, best_yaw_dist


def _action_is_nonzero(action: Any) -> bool:
    if isinstance(action, dict):
        if "oasis_action" in action:
            vector = oasis_action_to_vector(action["oasis_action"])
        elif "compact" in action:
            vector = oasis_action_to_vector(compact_minedojo_action_to_oasis(action["compact"]))
        else:
            vector = oasis_action_to_vector(action)
    else:
        vector = oasis_action_to_vector(compact_minedojo_action_to_oasis(action))
    return bool(abs(vector).sum() > 0)


def _future_motion_stats(
    poses: Sequence[PoseRecord],
    actions: Sequence[Any],
    query_t: int,
    target_t: int,
) -> dict[str, float]:
    if len(actions) <= target_t:
        raise ValueError(
            f"actions must cover target_t={target_t}; got {len(actions)} actions"
        )

    path_length = 0.0
    yaw_path = 0.0
    for idx in range(query_t + 1, target_t + 1):
        prev = poses[idx - 1]
        cur = poses[idx]
        path_length += pose_distance(prev, cur)
        yaw_path += yaw_distance(cur.yaw, prev.yaw)

    future_actions = actions[query_t : target_t + 1]
    nonzero_actions = sum(1 for action in future_actions if _action_is_nonzero(action))
    action_rate = nonzero_actions / max(1, len(future_actions))

    return {
        "future_path_length": float(path_length),
        "future_yaw_path": float(yaw_path),
        "future_action_rate": float(action_rate),
    }


def _is_static_future(stats: dict[str, float]) -> bool:
    return (
        stats["future_path_length"] < MIN_FUTURE_PATH_LENGTH
        and stats["future_yaw_path"] < MIN_FUTURE_YAW_PATH
        and stats["future_action_rate"] < MIN_FUTURE_ACTION_RATE
    )


def build_tests_for_episode(
    poses: Sequence[PoseRecord],
    actions: Sequence[Any],
    family: str,
    events: Sequence[EventRecord],
    frames_per_episode: int,
    expected_labels: dict[str, object] | None = None,
    history_length: int = 64,
    future_length: int = 64,
    stride: int = 32,
) -> list[TestRecord]:
    """Package future-test records from scripted trajectory metadata."""

    if frames_per_episode <= 1:
        raise ValueError("frames_per_episode must be > 1")
    if len(poses) < frames_per_episode:
        raise ValueError("poses must contain at least frames_per_episode records")
    if len(actions) < frames_per_episode:
        raise ValueError("actions must contain at least frames_per_episode records")

    tests: list[TestRecord] = []
    max_query = frames_per_episode - future_length - 1
    first_query = min(history_length, max_query)
    if first_query < 1:
        first_query = max(0, frames_per_episode // 2)
    query_times = list(range(first_query, max_query + 1, max(1, stride)))
    if not query_times:
        query_times = [max(0, min(frames_per_episode - 2, first_query))]

    for query_t in query_times:
        history_start = max(0, query_t - history_length + 1)
        history_end = query_t
        future_start = query_t
        future_end = min(frames_per_episode - 1, query_t + future_length)
        target_t = future_end
        if has_pose_discontinuity(poses, history_start, future_end):
            continue
        motion = _future_motion_stats(poses, actions, query_t, target_t)
        is_static_future = _is_static_future(motion)
        if is_static_future:
            continue

        target_pose = poses[target_t]
        matched_t, pose_dist, yaw_dist = _nearest_history_pose(
            poses,
            target_pose,
            history_start,
            history_end,
        )

        expected = expected_labels or {}
        labels: dict[str, object] = {
            "target_frame_idx": target_t,
            "matched_history_t": matched_t,
            "matched_history_index": matched_t - history_start,
            "pose_distance": pose_dist,
            "yaw_distance": yaw_dist,
            "returns_to_seen_place": bool(expected.get("returns_to_seen_place", False)),
            "landmark_visible": bool(expected.get("landmark_visible", False)),
            "object_exists_at_return": bool(expected.get("object_exists_at_return", False)),
        }
        labels.update(motion)
        labels["future_static_filtered_candidate"] = False

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
                "dx": target_pose.x - poses[query_t].x,
                "dy": target_pose.y - poses[query_t].y,
                "dz": target_pose.z - poses[query_t].z,
                "dyaw": yaw_distance(target_pose.yaw, poses[query_t].yaw),
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

    if not tests:
        raise RuntimeError(
            "episode produced no valid future tests after motion filtering "
            f"(min_path={MIN_FUTURE_PATH_LENGTH}, min_yaw={MIN_FUTURE_YAW_PATH}, "
            f"min_action_rate={MIN_FUTURE_ACTION_RATE})"
        )

    return tests


def records_to_json(records: Sequence[PoseRecord | EventRecord | TestRecord]) -> list[dict]:
    out = []
    for record in records:
        if isinstance(record, TestRecord):
            out.append(record.to_json())
        else:
            out.append(asdict(record))
    return out
