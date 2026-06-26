from __future__ import annotations

from typing import Any, Sequence


def _field(record: Any, name: str) -> float:
    if isinstance(record, dict):
        return float(record[name])
    return float(getattr(record, name))


def has_pose_discontinuity(
    poses: Sequence[Any],
    start_t: int,
    end_t: int,
    *,
    max_step_distance: float = 4.0,
    max_vertical_step: float = 2.5,
) -> bool:
    """Detect respawn/teleport-like jumps inside a supervision window."""
    if not poses:
        return True

    start = max(0, int(start_t))
    end = min(len(poses) - 1, int(end_t))
    if start >= end:
        return False

    for t in range(start + 1, end + 1):
        prev = poses[t - 1]
        cur = poses[t]
        dx = _field(cur, "x") - _field(prev, "x")
        dy = _field(cur, "y") - _field(prev, "y")
        dz = _field(cur, "z") - _field(prev, "z")
        step_distance = (dx * dx + dy * dy + dz * dz) ** 0.5
        if step_distance > max_step_distance or abs(dy) > max_vertical_step:
            return True

    return False
