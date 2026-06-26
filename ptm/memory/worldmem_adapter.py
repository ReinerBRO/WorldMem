from __future__ import annotations

import torch
from torch import nn


class PTMWorldMemAdapter(nn.Module):
    """Maps PTM tokens into WorldMem's existing reference-frame interface.

    WorldMem already accepts appended memory/reference latents. This adapter is
    the low-risk integration path requested in the spec: PTM tokens are converted
    into pseudo reference latents plus action/pose conditioning tensors, so the
    DiT memory-attention path can consume them without rewriting the backbone.
    """

    def __init__(
        self,
        memory_dim: int,
        latent_channels: int = 16,
        latent_height: int = 16,
        latent_width: int = 16,
        action_dim: int = 25,
        pose_dim: int = 5,
    ):
        super().__init__()
        self.memory_dim = int(memory_dim)
        self.latent_channels = int(latent_channels)
        self.latent_height = int(latent_height)
        self.latent_width = int(latent_width)
        self.action_dim = int(action_dim)
        self.pose_dim = int(pose_dim)

        latent_dim = latent_channels * latent_height * latent_width
        self.latent_proj = nn.Sequential(
            nn.LayerNorm(memory_dim),
            nn.Linear(memory_dim, latent_dim),
        )
        self.action_proj = nn.Linear(memory_dim, action_dim)
        self.pose_proj = nn.Linear(memory_dim, pose_dim)

    def forward(self, memory_tokens: torch.Tensor) -> dict[str, torch.Tensor]:
        if memory_tokens.ndim != 3:
            raise ValueError(f"memory_tokens must be [B,M,D], got {tuple(memory_tokens.shape)}")
        batch, num_tokens, _ = memory_tokens.shape
        latents = self.latent_proj(memory_tokens)
        latents = latents.view(
            batch,
            num_tokens,
            self.latent_channels,
            self.latent_height,
            self.latent_width,
        )
        return {
            "latent_reference_frames": latents.permute(1, 0, 2, 3, 4).contiguous(),
            "reference_actions": self.action_proj(memory_tokens).permute(1, 0, 2).contiguous(),
            "reference_poses": self.pose_proj(memory_tokens).permute(1, 0, 2).contiguous(),
        }
