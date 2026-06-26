#!/usr/bin/env python3
"""Inspect PTM MineDojo generation status without trusting stale pidfiles."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import re
import subprocess
import time


def complete_episode(path: Path) -> bool:
    return (path / "frames.mp4").is_file() and (path / "frames.mp4").stat().st_size > 0 and (
        path / "metadata.json"
    ).is_file() and (path / "metadata.json").stat().st_size > 0


def episode_id(path: Path) -> int:
    match = re.search(r"episode_(\d+)$", path.name)
    return int(match.group(1)) if match else -1


def count_data(train_root: Path, target: int) -> None:
    now = time.time()
    complete = []
    for episode_dir in train_root.glob("episode_*"):
        if complete_episode(episode_dir):
            complete.append(episode_dir)
    complete.sort(key=episode_id)
    recent10 = sum(1 for d in complete if now - (d / "frames.mp4").stat().st_mtime < 600)
    recent30 = sum(1 for d in complete if now - (d / "frames.mp4").stat().st_mtime < 1800)
    ids = {episode_id(d) for d in complete}
    missing = [i for i in range(target) if i not in ids]
    newest = sorted(complete, key=lambda d: (d / "frames.mp4").stat().st_mtime)[-8:]
    print(f"DATA complete={len(complete)} remaining={max(target - len(complete), 0)} recent10={recent10} recent30={recent30}")
    print("DATA newest=" + " ".join(f"{episode_id(d)}" for d in newest))
    print("DATA missing_head=" + " ".join(map(str, missing[:24])))


def read_tail(path: Path, max_bytes: int = 240_000) -> str:
    try:
        size = path.stat().st_size
        with path.open("rb") as f:
            if size > max_bytes:
                f.seek(size - max_bytes)
            return f.read().decode("utf-8", "replace")
    except FileNotFoundError:
        return ""


def summarize_logs(logdir: Path, host: str, include_old_w14: bool = False, old_logdir: Path | None = None) -> None:
    logs = sorted(logdir.glob(f"worker_{host}_*.log"))
    print(f"LOGS host={host} count={len(logs)} dir={logdir}")
    totals = {
        "batch_start": 0,
        "ep_done": 0,
        "ep_fail": 0,
        "batch_done": 0,
        "timeout": 0,
        "traceback": 0,
        "xorgfail": 0,
        "no_progress": 0,
    }
    for log in logs:
        text = read_tail(log)
        vals = {
            "batch_start": text.count("MICRO_BATCH_START"),
            "ep_done": text.count("MICRO_EP_DONE"),
            "ep_fail": text.count("MICRO_EP_FAIL"),
            "batch_done": text.count("MICRO_BATCH_DONE"),
            "timeout": text.count("rc=124"),
            "traceback": text.count("Traceback"),
            "xorgfail": text.count("Xorg exited before socket ready"),
            "no_progress": text.count("MICRO_NO_PROGRESS"),
        }
        for key, value in vals.items():
            totals[key] += value
        interesting = vals["ep_done"] or vals["ep_fail"] or vals["timeout"] or vals["traceback"] or vals["xorgfail"]
        if interesting:
            print(
                "LOG",
                log.name,
                " ".join(f"{key}={value}" for key, value in vals.items()),
            )
            print("TAIL", " | ".join(text.splitlines()[-5:])[-1000:])
    print("LOG_TOTALS", " ".join(f"{key}={value}" for key, value in totals.items()))
    if include_old_w14 and old_logdir is not None:
        for log in sorted(old_logdir.glob("worker_legacy_w14_*.log")):
            text = read_tail(log)
            print(
                "OLD_W14",
                log.name,
                f"done={text.count('MICRO_EP_DONE')}",
                f"fail={text.count('MICRO_EP_FAIL')}",
                f"timeout={text.count('rc=124')}",
                f"traceback={text.count('Traceback')}",
            )
            print("OLD_W14_TAIL", " | ".join(text.splitlines()[-8:])[-1200:])


def summarize_processes(run_tag: str, data_token: str) -> None:
    ps = subprocess.check_output(
        ["ps", "-eo", "pid,ppid,pgid,stat,pcpu,pmem,etime,cmd"],
        text=True,
        errors="replace",
    )
    lines = [
        line
        for line in ps.splitlines()
        if run_tag in line or ("ptm.data.minedojo_generator" in line and data_token in line)
    ]
    batch_generators = [line for line in lines if "ptm.data.minedojo_generator" in line and "--num_episodes 1 " not in line]
    single_generators = [line for line in lines if "ptm.data.minedojo_generator" in line and "--num_episodes 1 " in line]
    xorg = [line for line in lines if "/Xorg" in line or " Xorg " in line]
    java = [line for line in ps.splitlines() if "GradleStart" in line or "net.minecraft.launchwrapper.Launch" in line]
    java = [line for line in java if run_tag in line or data_token in line]
    print(
        f"PROCS matched={len(lines)} batch_generators={len(batch_generators)} "
        f"single_generators={len(single_generators)} xorg={len(xorg)} java={len(java)}"
    )
    for label, seq in [
        ("PROC_BATCH", batch_generators[:40]),
        ("PROC_SINGLE", single_generators[:20]),
        ("PROC_XORG", xorg[:20]),
        ("PROC_JAVA", java[:20]),
    ]:
        for line in seq:
            print(label, line[:260])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", required=True)
    parser.add_argument("--data-root", default="/gfs/space/private/zjc/ptm/ptm_minedojo_data/stage1_360x640")
    parser.add_argument("--run-tag", default="ptm_stage1_360x640_batch_20260622_145231")
    parser.add_argument("--old-tag", default="ptm_stage1_360x640_dual_20260622_142454")
    parser.add_argument("--target", type=int, default=3000)
    args = parser.parse_args()

    data_root = Path(args.data_root)
    log_root = Path("/gfs/space/private/zjc/logs")
    print(f"HOST requested={args.host} actual={os.uname().nodename} time={time.strftime('%Y-%m-%dT%H:%M:%S%z')}")
    count_data(data_root / "train", args.target)
    summarize_processes(args.run_tag, "stage1_360x640")
    summarize_logs(
        log_root / args.run_tag,
        args.host,
        include_old_w14=args.host == "legacy",
        old_logdir=log_root / args.old_tag,
    )


if __name__ == "__main__":
    main()
