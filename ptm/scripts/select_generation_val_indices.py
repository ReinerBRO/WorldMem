#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from types import SimpleNamespace

from ptm.data.worldmem_dataset import PTMWorldMemDataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Select fixed PTM raw validation indices for 600+100 generation ablation."
    )
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--num-ranks", type=int, default=8)
    parser.add_argument("--batch-size-per-rank", type=int, default=2)
    parser.add_argument("--context-length", type=int, default=600)
    parser.add_argument("--future-length", type=int, default=100)
    parser.add_argument("--memory-condition-length", type=int, default=8)
    parser.add_argument("--height", type=int, default=360)
    parser.add_argument("--width", type=int, default=640)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.batch_size_per_rank != 2:
        raise SystemExit("selector currently expects batch-size-per-rank=2 for hard-shuffle pairs")

    cfg = SimpleNamespace(
        save_dir=args.data_root,
        resolution=[args.height, args.width],
        memory_condition_length=args.memory_condition_length,
        future_length=4,
        ptm_context_length=4,
        ptm_future_length=4,
        ptm_context_length_valid=args.context_length,
        ptm_future_length_valid=args.future_length,
        n_frames_valid=args.context_length + args.future_length,
        max_history_candidates=16,
        frame_skip=1,
        video_cache_size=0,
    )
    dataset = PTMWorldMemDataset(cfg, split="validation")

    def meta(position: int) -> tuple[str, str, int]:
        episode_dir, test_idx = dataset.index[position]
        return str(episode_dir), dataset._episode_family.get(episode_dir, ""), int(test_idx)

    selected: list[int] = []
    used: set[int] = set()
    used_episodes: set[str] = set()
    for position in range(len(dataset.index)):
        episode_dir, _, _ = meta(position)
        if episode_dir in used_episodes:
            continue
        selected.append(position)
        used.add(position)
        used_episodes.add(episode_dir)
        if len(selected) == args.num_ranks:
            break
    if len(selected) != args.num_ranks:
        raise SystemExit(f"could not select first half: {len(selected)}")

    second_half: list[int] = []
    for rank, first_position in enumerate(selected):
        first_episode, first_family, _ = meta(first_position)
        best = None
        fallback = None
        for position in range(len(dataset.index)):
            if position in used:
                continue
            episode_dir, family, _ = meta(position)
            if episode_dir == first_episode:
                continue
            if fallback is None:
                fallback = position
            if first_family and family and family != first_family:
                best = position
                break
        chosen = best if best is not None else fallback
        if chosen is None:
            raise SystemExit(f"could not pair rank {rank} for index {first_position}")
        second_half.append(chosen)
        used.add(chosen)

    selected.extend(second_half)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(str(index) for index in selected) + "\n", encoding="utf-8")

    print(f"wrote={output}")
    print("selected_indices=" + ",".join(str(index) for index in selected))
    for rank in range(args.num_ranks):
        a = selected[rank]
        b = selected[rank + args.num_ranks]
        episode_a, family_a, test_a = meta(a)
        episode_b, family_b, test_b = meta(b)
        print(
            "rank{rank}: {a} test{test_a} {ep_a} family={family_a} | "
            "{b} test{test_b} {ep_b} family={family_b} "
            "diff_episode={diff_episode} diff_family={diff_family}".format(
                rank=rank,
                a=a,
                test_a=test_a,
                ep_a=Path(episode_a).name,
                family_a=family_a,
                b=b,
                test_b=test_b,
                ep_b=Path(episode_b).name,
                family_b=family_b,
                diff_episode=episode_a != episode_b,
                diff_family=family_a != family_b,
            )
        )


if __name__ == "__main__":
    main()
