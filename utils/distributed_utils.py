from __future__ import annotations

import os


def _rank_from_env() -> int:
    for name in ("RANK", "LOCAL_RANK", "SLURM_PROCID"):
        value = os.environ.get(name)
        if value is None or value == "":
            continue
        try:
            return int(value)
        except ValueError:
            return 0
    return 0


is_rank_zero = _rank_from_env() == 0
