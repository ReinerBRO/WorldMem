from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class FutureTestLossConfig:
    w_embed: float = 1.0
    w_loop: float = 0.5
    w_match: float = 0.5
    w_landmark: float = 0.5
    w_object: float = 0.5


class FutureTestLoss(nn.Module):
    """Composite PTM loss from the implementation spec."""

    def __init__(self, cfg: FutureTestLossConfig | None = None):
        super().__init__()
        self.cfg = cfg or FutureTestLossConfig()

    @staticmethod
    def _as_float(labels: dict[str, torch.Tensor], key: str, ref: torch.Tensor) -> torch.Tensor | None:
        value = labels.get(key)
        if value is None:
            return None
        return value.to(device=ref.device, dtype=ref.dtype)

    @staticmethod
    def _as_long(labels: dict[str, torch.Tensor], key: str, ref: torch.Tensor) -> torch.Tensor | None:
        value = labels.get(key)
        if value is None:
            return None
        return value.to(device=ref.device, dtype=torch.long)

    @staticmethod
    def _select_mask(labels: dict[str, torch.Tensor], ref: torch.Tensor, test_type_ids: set[int]) -> torch.Tensor | None:
        ids = labels.get("test_type_id")
        if ids is None:
            return None
        ids = ids.to(device=ref.device, dtype=torch.long)
        mask = torch.zeros_like(ids, dtype=torch.bool)
        for test_type_id in test_type_ids:
            mask |= ids == test_type_id
        return mask

    @staticmethod
    def _masked_bce(logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor | None:
        if mask is not None:
            if not bool(mask.any()):
                return None
            logits = logits[mask]
            targets = targets[mask]
        return F.binary_cross_entropy_with_logits(logits, targets)

    @staticmethod
    def _masked_ce(logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor | None:
        if mask is not None:
            if not bool(mask.any()):
                return None
            logits = logits[mask]
            targets = targets[mask]
        return F.cross_entropy(logits, targets)

    def forward(
        self,
        predictions: dict[str, torch.Tensor],
        labels: dict[str, torch.Tensor],
        target_embeddings: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        components: dict[str, torch.Tensor] = {}
        device_ref = next(iter(predictions.values()))
        total = device_ref.new_zeros(())

        if target_embeddings is not None and self.cfg.w_embed:
            pred = predictions["future_embedding"]
            target = target_embeddings.to(device=pred.device, dtype=pred.dtype)
            if pred.shape != target.shape:
                raise ValueError(f"future embedding shape mismatch: pred={pred.shape}, target={target.shape}")
            components["future_embedding"] = F.mse_loss(pred, target) * self.cfg.w_embed
            total = total + components["future_embedding"]

        loop = self._as_float(labels, "returns_to_seen_place", predictions["loop_return_logit"])
        if loop is not None and self.cfg.w_loop:
            loop_loss = self._masked_bce(
                predictions["loop_return_logit"],
                loop,
                self._select_mask(labels, predictions["loop_return_logit"], {1}),
            )
            if loop_loss is not None:
                components["loop_return"] = loop_loss * self.cfg.w_loop
                total = total + components["loop_return"]

        matched = self._as_long(labels, "matched_history_index", predictions["match_history_logits"])
        if matched is not None and self.cfg.w_match:
            logits = predictions["match_history_logits"]
            match_valid = labels.get("match_valid")
            if match_valid is None:
                raise KeyError("match_valid is required for matched_history loss")
            match_valid = match_valid.to(device=logits.device, dtype=torch.bool)
            type_mask = self._select_mask(labels, logits, {1, 2, 3})
            mask = match_valid if type_mask is None else match_valid & type_mask
            if bool(mask.any()):
                matched_selected = matched[mask]
                if bool((matched_selected < 0).any()) or bool((matched_selected >= logits.shape[-1]).any()):
                    raise ValueError(
                        "matched_history_index contains an out-of-range valid target "
                        f"for {logits.shape[-1]} match logits"
                    )
                match_loss = F.cross_entropy(logits[mask], matched_selected)
                components["matched_history"] = match_loss * self.cfg.w_match
                total = total + components["matched_history"]

        landmark = self._as_float(labels, "landmark_visible", predictions["landmark_visible_logit"])
        if landmark is not None and self.cfg.w_landmark:
            landmark_loss = self._masked_bce(
                predictions["landmark_visible_logit"],
                landmark,
                self._select_mask(labels, predictions["landmark_visible_logit"], {2}),
            )
            if landmark_loss is not None:
                components["landmark_visible"] = landmark_loss * self.cfg.w_landmark
                total = total + components["landmark_visible"]

        obj = self._as_float(labels, "object_exists_at_return", predictions["object_exists_logit"])
        if obj is not None and self.cfg.w_object:
            object_loss = self._masked_bce(
                predictions["object_exists_logit"],
                obj,
                self._select_mask(labels, predictions["object_exists_logit"], {3}),
            )
            if object_loss is not None:
                components["object_exists"] = object_loss * self.cfg.w_object
                total = total + components["object_exists"]

        components["total"] = total
        return total, components
