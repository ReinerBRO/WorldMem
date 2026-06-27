from __future__ import annotations

import math
from collections.abc import Sequence

import torch
from torch import nn
from torch.nn import functional as F

from .bottleneck import TokenDropout


TEST_TYPES = {
    "normal_rollout": 0,
    "loop_return": 1,
    "landmark_revisit": 2,
    "object_persistence": 3,
}


def _pool_frame_latents(history_frame_latents: torch.Tensor) -> torch.Tensor:
    """Convert frame latents/images to [B, T, C] features when needed."""

    if history_frame_latents.ndim == 3:
        return history_frame_latents
    if history_frame_latents.ndim < 4:
        raise ValueError(
            "history_frame_latents must be [B,T,D] or [B,T,C,...], "
            f"got {tuple(history_frame_latents.shape)}"
        )
    reduce_dims = tuple(range(3, history_frame_latents.ndim))
    return history_frame_latents.mean(dim=reduce_dims)


class PredictiveTestMemory(nn.Module):
    """Bounded memory encoder trained through executable future-test losses.

    The encoder only returns a fixed number of memory tokens. Downstream test
    decoders and generation adapters must read these tokens instead of raw
    history, which is the core leakage-prevention constraint for PTM.
    """

    def __init__(
        self,
        frame_dim: int,
        action_dim: int,
        memory_dim: int = 1024,
        num_memory_tokens: int = 16,
        num_layers: int = 4,
        dropout: float = 0.1,
        pose_dim: int = 0,
        max_history: int = 512,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        token_dropout: float = 0.0,
    ):
        super().__init__()
        if num_memory_tokens <= 0:
            raise ValueError("num_memory_tokens must be positive")
        if memory_dim % num_heads != 0:
            raise ValueError("memory_dim must be divisible by num_heads")

        self.frame_dim = int(frame_dim)
        self.action_dim = int(action_dim)
        self.pose_dim = int(pose_dim)
        self.memory_dim = int(memory_dim)
        self.num_memory_tokens = int(num_memory_tokens)
        self.max_history = int(max_history)

        input_dim = self.frame_dim + self.action_dim + self.pose_dim
        self.input_proj = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, memory_dim),
        )
        self.position_embedding = nn.Parameter(torch.zeros(max_history, memory_dim))
        self.memory_queries = nn.Parameter(torch.randn(num_memory_tokens, memory_dim) * 0.02)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=memory_dim,
            nhead=num_heads,
            dim_feedforward=int(memory_dim * mlp_ratio),
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(memory_dim)
        self.token_dropout = TokenDropout(token_dropout)

    def forward(
        self,
        history_frame_latents: torch.Tensor,
        past_actions: torch.Tensor,
        pose_tokens: torch.Tensor | None = None,
        key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        frame_features = _pool_frame_latents(history_frame_latents)
        if frame_features.shape[-1] != self.frame_dim:
            raise ValueError(f"expected frame_dim={self.frame_dim}, got {frame_features.shape[-1]}")
        if past_actions.ndim != 3:
            raise ValueError(f"past_actions must be [B,T,A], got {tuple(past_actions.shape)}")
        if past_actions.shape[:2] != frame_features.shape[:2]:
            raise ValueError("history_frame_latents and past_actions must share [B,T]")
        if past_actions.shape[-1] != self.action_dim:
            raise ValueError(f"expected action_dim={self.action_dim}, got {past_actions.shape[-1]}")

        inputs = [frame_features, past_actions.to(frame_features.dtype)]
        if self.pose_dim:
            if pose_tokens is None:
                pose_tokens = frame_features.new_zeros(*frame_features.shape[:2], self.pose_dim)
            if pose_tokens.shape[:2] != frame_features.shape[:2] or pose_tokens.shape[-1] != self.pose_dim:
                raise ValueError(
                    f"pose_tokens must be [B,T,{self.pose_dim}], got {tuple(pose_tokens.shape)}"
                )
            inputs.append(pose_tokens.to(frame_features.dtype))

        seq = torch.cat(inputs, dim=-1)
        batch, history, _ = seq.shape
        if history > self.max_history:
            seq = seq[:, -self.max_history :]
            if key_padding_mask is not None:
                key_padding_mask = key_padding_mask[:, -self.max_history :]
            history = self.max_history

        history_tokens = self.input_proj(seq)
        history_tokens = history_tokens + self.position_embedding[:history].unsqueeze(0)
        memory_queries = self.memory_queries.unsqueeze(0).expand(batch, -1, -1)
        tokens = torch.cat([memory_queries, history_tokens], dim=1)

        if key_padding_mask is not None:
            if key_padding_mask.shape != (batch, history):
                raise ValueError(
                    f"key_padding_mask must be [B,T]={batch, history}, got {tuple(key_padding_mask.shape)}"
                )
            query_mask = torch.zeros(batch, self.num_memory_tokens, device=key_padding_mask.device, dtype=torch.bool)
            key_padding_mask = torch.cat([query_mask, key_padding_mask.bool()], dim=1)

        encoded = self.encoder(tokens, src_key_padding_mask=key_padding_mask)
        memory_tokens = self.norm(encoded[:, : self.num_memory_tokens])
        return self.token_dropout(memory_tokens)


class FutureTestDecoder(nn.Module):
    """Predicts future executable-test outcomes from PTM tokens and future actions."""

    def __init__(
        self,
        memory_dim: int,
        action_dim: int,
        future_embedding_dim: int,
        max_history_candidates: int = 256,
        num_test_types: int = len(TEST_TYPES),
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.memory_dim = int(memory_dim)
        self.action_dim = int(action_dim)
        self.future_embedding_dim = int(future_embedding_dim)
        self.max_history_candidates = int(max_history_candidates)

        self.future_action_encoder = nn.GRU(
            input_size=action_dim,
            hidden_size=memory_dim,
            batch_first=True,
            num_layers=1,
        )
        self.test_type_embedding = nn.Embedding(num_test_types, memory_dim)
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=memory_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.summary = nn.Sequential(
            nn.LayerNorm(memory_dim),
            nn.Linear(memory_dim, memory_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.future_embedding_head = nn.Linear(memory_dim, future_embedding_dim)
        self.loop_return_head = nn.Linear(memory_dim, 1)
        self.match_history_head = nn.Linear(memory_dim, max_history_candidates)
        self.landmark_visible_head = nn.Linear(memory_dim, 1)
        self.object_exists_head = nn.Linear(memory_dim, 1)

    @staticmethod
    def encode_test_types(test_type: torch.Tensor | Sequence[str], device: torch.device) -> torch.Tensor:
        if torch.is_tensor(test_type):
            return test_type.to(device=device, dtype=torch.long)
        ids = []
        for name in test_type:
            if name not in TEST_TYPES:
                raise KeyError(f"unknown test_type {name!r}; expected one of {sorted(TEST_TYPES)}")
            ids.append(TEST_TYPES[name])
        return torch.tensor(ids, device=device, dtype=torch.long)

    def forward(
        self,
        memory_tokens: torch.Tensor,
        future_actions: torch.Tensor,
        test_type: torch.Tensor | Sequence[str],
        candidate_history_embeddings: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if memory_tokens.ndim != 3:
            raise ValueError(f"memory_tokens must be [B,M,D], got {tuple(memory_tokens.shape)}")
        if future_actions.ndim != 3:
            raise ValueError(f"future_actions must be [B,K,A], got {tuple(future_actions.shape)}")
        if future_actions.shape[0] != memory_tokens.shape[0]:
            raise ValueError("memory_tokens and future_actions must share batch size")

        _, hidden = self.future_action_encoder(future_actions.to(memory_tokens.dtype))
        action_query = hidden[-1]
        type_ids = self.encode_test_types(test_type, memory_tokens.device)
        query = action_query + self.test_type_embedding(type_ids)
        attended, attn_weights = self.cross_attention(
            query=query.unsqueeze(1),
            key=memory_tokens,
            value=memory_tokens,
            need_weights=True,
        )
        state = self.summary(attended.squeeze(1))

        if candidate_history_embeddings is not None:
            if candidate_history_embeddings.ndim != 3:
                raise ValueError("candidate_history_embeddings must be [B,N,D]")
            candidate = candidate_history_embeddings.to(state.dtype)
            match_logits = torch.einsum("bd,bnd->bn", state, candidate) / (state.shape[-1] ** 0.5)
        else:
            match_logits = self.match_history_head(state)

        return {
            "future_embedding": self.future_embedding_head(state),
            "loop_return_logit": self.loop_return_head(state).squeeze(-1),
            "match_history_logits": match_logits,
            "landmark_visible_logit": self.landmark_visible_head(state).squeeze(-1),
            "object_exists_logit": self.object_exists_head(state).squeeze(-1),
            "memory_attention": attn_weights,
        }


class FutureSupervisedVisualMemorySelector(nn.Module):
    """Selects high-bandwidth visual memory values from historical latent candidates.

    The selector deliberately separates "which candidate is future-relevant" from
    "what the candidate looks like":

    - candidate_embeddings() projects each candidate latent to the PTM memory space
      for the future-test matched-history classifier.
    - selected_visual_tokens() keeps the selected candidate's visual latent content,
      pools it into one or more visual tokens, and projects those tokens for DiT
      cross-attention.

    The default proxy_topk path preserves the V5 behavior: hard top-k indices
    are chosen from future-supervised scores, while selected token magnitudes
    are weighted by differentiable softmax probabilities. The slot_router path
    keeps the same visual memory budget but replaces hard membership with
    sparse differentiable routing slots whose logits combine a generation
    query with the PTM matched-history prior.
    """

    def __init__(
        self,
        frame_dim: int,
        memory_dim: int = 1024,
        top_k: int = 8,
        pool: str = "grid2x2",
        dropout: float = 0.0,
        routing_mode: str = "proxy_topk",
        route_top_m: int = 8,
        route_tau: float = 0.2,
        route_prior_alpha: float = 1.0,
        route_dim: int | None = None,
    ):
        super().__init__()
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        pool = str(pool).strip().lower()
        if pool not in {"global", "grid2x2"}:
            raise ValueError("pool must be 'global' or 'grid2x2'")
        routing_mode = str(routing_mode).strip().lower()
        if routing_mode not in {"proxy_topk", "slot_router"}:
            raise ValueError("routing_mode must be 'proxy_topk' or 'slot_router'")
        if route_top_m <= 0:
            raise ValueError("route_top_m must be positive")
        if route_tau <= 0:
            raise ValueError("route_tau must be positive")
        self.frame_dim = int(frame_dim)
        self.memory_dim = int(memory_dim)
        self.top_k = int(top_k)
        self.pool = pool
        self.routing_mode = routing_mode
        self.route_top_m = int(route_top_m)
        self.route_tau = float(route_tau)
        self.route_prior_alpha = float(route_prior_alpha)
        self.route_dim = int(route_dim or memory_dim)
        if self.route_dim <= 0:
            raise ValueError("route_dim must be positive")

        self.candidate_key_proj = nn.Sequential(
            nn.LayerNorm(frame_dim),
            nn.Linear(frame_dim, memory_dim),
        )
        self.visual_value_proj = nn.Sequential(
            nn.LayerNorm(frame_dim),
            nn.Linear(frame_dim, memory_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(memory_dim, memory_dim),
        )
        if self.routing_mode == "slot_router":
            self.route_query_proj = nn.Sequential(
                nn.LayerNorm(memory_dim),
                nn.Linear(memory_dim, self.route_dim),
            )
            self.route_key_proj = nn.Sequential(
                nn.LayerNorm(memory_dim),
                nn.Linear(memory_dim, self.route_dim),
            )
            self.route_slot_embedding = nn.Parameter(torch.randn(self.top_k, self.route_dim) * 0.02)

    def _global_candidate_features(self, candidate_latents: torch.Tensor) -> torch.Tensor:
        if candidate_latents.ndim != 5:
            raise ValueError(
                "candidate_latents must be [B,N,C,H,W], "
                f"got {tuple(candidate_latents.shape)}"
            )
        return candidate_latents.mean(dim=(-2, -1))

    def candidate_embeddings(self, candidate_latents: torch.Tensor) -> torch.Tensor:
        features = self._global_candidate_features(candidate_latents)
        return self.candidate_key_proj(features)

    def _pool_visual_values(self, candidate_latents: torch.Tensor) -> torch.Tensor:
        batch, candidates, channels, height, width = candidate_latents.shape
        flat = candidate_latents.reshape(batch * candidates, channels, height, width)
        if self.pool == "global":
            pooled = flat.mean(dim=(-2, -1)).reshape(batch, candidates, 1, channels)
        else:
            pooled = F.adaptive_avg_pool2d(flat, output_size=(2, 2))
            pooled = pooled.permute(0, 2, 3, 1).reshape(batch, candidates, 4, channels)
        return pooled

    def selected_visual_tokens(
        self,
        candidate_latents: torch.Tensor,
        scores: torch.Tensor,
        candidate_mask: torch.Tensor | None = None,
        top_k: int | None = None,
    ) -> torch.Tensor:
        if candidate_latents.ndim != 5:
            raise ValueError(
                "candidate_latents must be [B,N,C,H,W], "
                f"got {tuple(candidate_latents.shape)}"
            )
        if scores.ndim != 2:
            raise ValueError(f"scores must be [B,N], got {tuple(scores.shape)}")
        batch, candidates = scores.shape
        if candidate_latents.shape[:2] != (batch, candidates):
            raise ValueError(
                "candidate_latents and scores must share [B,N], "
                f"got {tuple(candidate_latents.shape[:2])} vs {tuple(scores.shape)}"
            )
        if candidates <= 0:
            return candidate_latents.new_zeros(batch, 0, self.memory_dim)

        k = min(int(top_k or self.top_k), int(candidates))
        masked_scores = scores
        if candidate_mask is not None:
            if candidate_mask.shape != scores.shape:
                raise ValueError(
                    f"candidate_mask must match scores {tuple(scores.shape)}, got {tuple(candidate_mask.shape)}"
                )
            # Keep at least one finite entry per sample to avoid NaNs on
            # degenerate padded batches; invalid-only rows select index 0 but
            # then get zero weights from the mask below.
            has_valid = candidate_mask.any(dim=1)
            safe_mask = candidate_mask.clone()
            if not bool(has_valid.all()):
                safe_mask[~has_valid, 0] = True
            masked_scores = scores.masked_fill(~safe_mask, torch.finfo(scores.dtype).min)

        top_indices = torch.topk(masked_scores, k=k, dim=1).indices
        gather_shape = (batch, k, *candidate_latents.shape[2:])
        gather_indices = top_indices[:, :, None, None, None].expand(gather_shape)
        selected_latents = candidate_latents.gather(dim=1, index=gather_indices)

        pooled = self._pool_visual_values(selected_latents)
        tokens = self.visual_value_proj(pooled.reshape(batch, k * pooled.shape[2], pooled.shape[-1]))

        probs = torch.softmax(masked_scores, dim=1)
        selected_weights = probs.gather(dim=1, index=top_indices)
        if candidate_mask is not None:
            selected_valid = candidate_mask.gather(dim=1, index=top_indices).to(selected_weights.dtype)
            selected_weights = selected_weights * selected_valid
        selected_weights = selected_weights / selected_weights.sum(dim=1, keepdim=True).clamp_min(1e-6)
        selected_weights = selected_weights * float(k)
        if pooled.shape[2] > 1:
            selected_weights = selected_weights[:, :, None].expand(-1, -1, pooled.shape[2]).reshape(batch, -1)
        tokens = tokens * selected_weights[:, :, None].to(tokens.dtype)
        return tokens

    def routed_visual_tokens(
        self,
        candidate_latents: torch.Tensor,
        candidate_embeddings: torch.Tensor,
        prior_scores: torch.Tensor,
        memory_tokens: torch.Tensor,
        candidate_mask: torch.Tensor | None = None,
        top_k: int | None = None,
    ) -> torch.Tensor:
        """Route visual candidates through differentiable budgeted memory slots.

        Each slot sees all K candidates, keeps only its top-m sparse support,
        and returns the weighted visual value mixture. With grid2x2 pooling this
        preserves V5's frame budget: top_k routed slots become top_k * 4 visual
        DiT tokens, just like top_k selected frames in proxy_topk mode.
        """

        if self.routing_mode != "slot_router":
            raise ValueError("routed_visual_tokens requires routing_mode='slot_router'")
        if candidate_latents.ndim != 5:
            raise ValueError(
                "candidate_latents must be [B,N,C,H,W], "
                f"got {tuple(candidate_latents.shape)}"
            )
        if candidate_embeddings.ndim != 3:
            raise ValueError(
                "candidate_embeddings must be [B,N,D], "
                f"got {tuple(candidate_embeddings.shape)}"
            )
        if prior_scores.ndim != 2:
            raise ValueError(f"prior_scores must be [B,N], got {tuple(prior_scores.shape)}")
        if memory_tokens.ndim != 3:
            raise ValueError(f"memory_tokens must be [B,M,D], got {tuple(memory_tokens.shape)}")

        batch, candidates = prior_scores.shape
        if candidates <= 0:
            return candidate_latents.new_zeros(batch, 0, self.memory_dim)
        if candidate_latents.shape[:2] != (batch, candidates):
            raise ValueError(
                "candidate_latents and prior_scores must share [B,N], "
                f"got {tuple(candidate_latents.shape[:2])} vs {tuple(prior_scores.shape)}"
            )
        if candidate_embeddings.shape[:2] != (batch, candidates):
            raise ValueError(
                "candidate_embeddings and prior_scores must share [B,N], "
                f"got {tuple(candidate_embeddings.shape[:2])} vs {tuple(prior_scores.shape)}"
            )
        if candidate_embeddings.shape[-1] != self.memory_dim:
            raise ValueError(
                f"candidate_embeddings must have dim={self.memory_dim}, "
                f"got {candidate_embeddings.shape[-1]}"
            )
        if memory_tokens.shape[0] != batch or memory_tokens.shape[-1] != self.memory_dim:
            raise ValueError(
                "memory_tokens must share batch and memory_dim with candidates, "
                f"got {tuple(memory_tokens.shape)}"
            )

        slots = min(int(top_k or self.top_k), int(candidates), int(self.route_slot_embedding.shape[0]))
        if slots <= 0:
            return candidate_latents.new_zeros(batch, 0, self.memory_dim)
        top_m = min(self.route_top_m, candidates)

        safe_mask = None
        valid_rows = None
        if candidate_mask is not None:
            if candidate_mask.shape != prior_scores.shape:
                raise ValueError(
                    f"candidate_mask must match prior_scores {tuple(prior_scores.shape)}, "
                    f"got {tuple(candidate_mask.shape)}"
                )
            safe_mask = candidate_mask.to(device=prior_scores.device, dtype=torch.bool).clone()
            valid_rows = safe_mask.any(dim=1)
            if not bool(valid_rows.all()):
                safe_mask[~valid_rows, 0] = True

        pooled = self._pool_visual_values(candidate_latents)
        patches = int(pooled.shape[2])
        value_tokens = self.visual_value_proj(pooled.reshape(batch, candidates * patches, pooled.shape[-1]))
        value_tokens = value_tokens.reshape(batch, candidates, patches, self.memory_dim)

        summary = memory_tokens.mean(dim=1)
        slot_q = self.route_query_proj(summary)[:, None, :]
        slot_q = slot_q + self.route_slot_embedding[:slots].to(slot_q.dtype)[None, :, :]
        cand_k = self.route_key_proj(candidate_embeddings.to(memory_tokens.dtype))
        route_logits = torch.einsum("bsd,bnd->bsn", slot_q, cand_k) / math.sqrt(float(self.route_dim))
        route_logits = route_logits + self.route_prior_alpha * prior_scores.to(route_logits.dtype)[:, None, :]
        if safe_mask is not None:
            route_logits = route_logits.masked_fill(~safe_mask[:, None, :], torch.finfo(route_logits.dtype).min)

        top_values, top_indices = torch.topk(route_logits, k=top_m, dim=-1)
        weights = torch.softmax(top_values / self.route_tau, dim=-1)
        batch_indices = torch.arange(batch, device=value_tokens.device)[:, None, None]
        gathered_values = value_tokens[batch_indices, top_indices]
        routed = (weights[:, :, :, None, None].to(gathered_values.dtype) * gathered_values).sum(dim=2)
        if valid_rows is not None:
            routed = routed * valid_rows.to(device=routed.device, dtype=routed.dtype)[:, None, None, None]
        return routed.reshape(batch, slots * patches, self.memory_dim)
