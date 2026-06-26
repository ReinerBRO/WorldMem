from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


class TokenDropout(nn.Module):
    """Drops whole memory tokens during training while preserving tensor shape."""

    def __init__(self, p: float = 0.0):
        super().__init__()
        if not 0.0 <= p < 1.0:
            raise ValueError(f"token dropout must be in [0, 1), got {p}")
        self.p = float(p)

    def forward(self, memory_tokens: torch.Tensor) -> torch.Tensor:
        if not self.training or self.p == 0.0:
            return memory_tokens
        if memory_tokens.ndim != 3:
            raise ValueError(
                "memory_tokens must have shape [batch, num_tokens, dim], "
                f"got {tuple(memory_tokens.shape)}"
            )
        keep = torch.rand(
            memory_tokens.shape[:2],
            device=memory_tokens.device,
            dtype=memory_tokens.dtype,
        ) >= self.p
        keep = keep.unsqueeze(-1)
        return memory_tokens * keep / (1.0 - self.p)


@dataclass(frozen=True)
class BottleneckStats:
    l2: torch.Tensor
    variance: torch.Tensor


class BottleneckLoss(nn.Module):
    """Regularizes bounded PTM tokens without introducing a discrete codebook."""

    def __init__(self, l2_weight: float = 0.0, variance_weight: float = 0.0):
        super().__init__()
        self.l2_weight = float(l2_weight)
        self.variance_weight = float(variance_weight)

    def stats(self, memory_tokens: torch.Tensor) -> BottleneckStats:
        if memory_tokens.ndim != 3:
            raise ValueError(
                "memory_tokens must have shape [batch, num_tokens, dim], "
                f"got {tuple(memory_tokens.shape)}"
            )
        l2 = memory_tokens.pow(2).mean()
        token_var = memory_tokens.var(dim=1, unbiased=False).mean()
        return BottleneckStats(l2=l2, variance=token_var)

    def forward(self, memory_tokens: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        stats = self.stats(memory_tokens)
        loss = memory_tokens.new_zeros(())
        components: dict[str, torch.Tensor] = {}
        if self.l2_weight:
            components["bottleneck_l2"] = stats.l2 * self.l2_weight
            loss = loss + components["bottleneck_l2"]
        if self.variance_weight:
            # Penalize collapsed token sets: low variance should cost more.
            components["bottleneck_variance"] = (1.0 / (stats.variance + 1e-6)) * self.variance_weight
            loss = loss + components["bottleneck_variance"]
        return loss, components
