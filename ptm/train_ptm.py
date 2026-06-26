from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from .data.ptm_dataset import PTMDataset, collate_ptm_batch
from .losses.future_test_losses import FutureTestLoss, FutureTestLossConfig
from .memory.predictive_test_memory import FutureTestDecoder, PredictiveTestMemory


def image_grid_embedding(frames: torch.Tensor, grid: int = 8) -> torch.Tensor:
    """Frozen frame embedding used for PTM head-only smoke training."""

    if frames.ndim == 4:
        pooled = F.adaptive_avg_pool2d(frames, (grid, grid))
        return pooled.flatten(1)
    if frames.ndim != 5:
        raise ValueError(f"expected [B,T,C,H,W] or [B,C,H,W], got {tuple(frames.shape)}")
    batch, time = frames.shape[:2]
    pooled = F.adaptive_avg_pool2d(frames.reshape(batch * time, *frames.shape[2:]), (grid, grid))
    return pooled.flatten(1).view(batch, time, -1)


def train(args: argparse.Namespace) -> dict[str, float]:
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    dataset = PTMDataset(
        args.data_root,
        split=args.split,
        history_length=args.history_length,
        future_length=args.future_length,
        resolution=args.resolution,
        max_history_candidates=args.max_history_candidates,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_ptm_batch,
        drop_last=False,
    )
    frame_dim = 3 * args.embedding_grid * args.embedding_grid
    memory = PredictiveTestMemory(
        frame_dim=frame_dim,
        action_dim=len(dataset._actions(dataset.episode_dirs[0])[0]),
        memory_dim=args.memory_dim,
        num_memory_tokens=args.num_memory_tokens,
        num_layers=args.num_layers,
        dropout=args.dropout,
        max_history=args.history_length,
        token_dropout=args.token_dropout,
    ).to(device)
    decoder = FutureTestDecoder(
        memory_dim=args.memory_dim,
        action_dim=len(dataset._actions(dataset.episode_dirs[0])[0]),
        future_embedding_dim=frame_dim,
        max_history_candidates=args.max_history_candidates,
        dropout=args.dropout,
    ).to(device)
    loss_fn = FutureTestLoss(
        FutureTestLossConfig(
            w_embed=args.w_embed,
            w_loop=args.w_loop,
            w_match=args.w_match,
            w_landmark=args.w_landmark,
            w_object=args.w_object,
        )
    )
    params = list(memory.parameters()) + list(decoder.parameters())
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "train_metrics.jsonl"
    last_metrics: dict[str, float] = {}
    with log_path.open("w", encoding="utf-8") as log_f:
        for epoch in range(args.epochs):
            pbar = tqdm(loader, desc=f"ptm epoch {epoch}")
            for step, batch in enumerate(pbar):
                history_frames = batch["history_frames"].to(device)
                past_actions = batch["past_actions"].to(device)
                future_actions = batch["future_actions"].to(device)
                labels = {key: value.to(device) for key, value in batch["memory_labels"].items()}
                target_frames = batch["target_frames"].to(device)

                history_embeddings = image_grid_embedding(history_frames, grid=args.embedding_grid)
                target_embeddings = image_grid_embedding(target_frames, grid=args.embedding_grid)
                memory_tokens = memory(history_embeddings, past_actions)
                predictions = decoder(memory_tokens, future_actions, labels["test_type_id"])
                loss, components = loss_fn(predictions, labels, target_embeddings)

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(params, args.grad_clip)
                optimizer.step()

                last_metrics = {key: float(value.detach().cpu()) for key, value in components.items()}
                last_metrics.update({"epoch": epoch, "step": step})
                log_f.write(json.dumps(last_metrics, sort_keys=True) + "\n")
                pbar.set_postfix(loss=last_metrics["total"])
                if args.max_steps and step + 1 >= args.max_steps:
                    break

    checkpoint = {
        "memory": memory.state_dict(),
        "decoder": decoder.state_dict(),
        "args": vars(args),
        "last_metrics": last_metrics,
    }
    torch.save(checkpoint, out_dir / "ptm_head_only.pt")
    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(last_metrics, f, indent=2, sort_keys=True)
    return last_metrics


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train PTM memory and future-test heads.")
    parser.add_argument("--data_root", default="ptm_minedojo_data/stage0")
    parser.add_argument("--split", default="train")
    parser.add_argument("--output_dir", default="outputs/ptm_smoke")
    parser.add_argument("--history_length", type=int, default=64)
    parser.add_argument("--future_length", type=int, default=64)
    parser.add_argument("--resolution", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max_steps", type=int, default=0)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--embedding_grid", type=int, default=8)
    parser.add_argument("--memory_dim", type=int, default=256)
    parser.add_argument("--num_memory_tokens", type=int, default=16)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--token_dropout", type=float, default=0.1)
    parser.add_argument("--max_history_candidates", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--w_embed", type=float, default=1.0)
    parser.add_argument("--w_loop", type=float, default=0.5)
    parser.add_argument("--w_match", type=float, default=0.5)
    parser.add_argument("--w_landmark", type=float, default=0.5)
    parser.add_argument("--w_object", type=float, default=0.5)
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    metrics = train(args)
    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
