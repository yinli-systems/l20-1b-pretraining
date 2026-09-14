"""Small, independently attachable router for evidence-directed visual acquisition.

The module does not modify the frozen vision tower, bridge, or language model.
It predicts one coarse evidence patch (or STOP) and the expected reduction in
correct-answer NLL from acquiring one high-resolution crop.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn


@dataclass(frozen=True)
class EvidenceTargets:
    pointer: Tensor
    observed_gain: Tensor
    zoom_is_beneficial: Tensor


class EvidenceAcquisitionRouter(nn.Module):
    """Question-conditioned visual-token pointer, STOP action, and gain regressor."""

    def __init__(self, vision_dim: int, language_dim: int, rank: int = 64) -> None:
        super().__init__()
        if rank < 1:
            raise ValueError("rank must be positive")
        self.rank = rank
        self.text_norm = nn.RMSNorm(language_dim)
        self.vision_norm = nn.LayerNorm(vision_dim)
        self.text_pool_key = nn.Linear(language_dim, rank, bias=False)
        self.text_pool_value = nn.Linear(language_dim, rank, bias=False)
        self.text_pool_query = nn.Parameter(torch.empty(rank))
        self.vision_key = nn.Linear(vision_dim, rank, bias=False)
        self.vision_value = nn.Linear(vision_dim, rank, bias=False)
        self.stop_key = nn.Parameter(torch.empty(rank))
        self.logit_scale = nn.Parameter(torch.tensor(math.log(math.sqrt(rank))))
        self.gain_head = nn.Sequential(
            nn.RMSNorm(3 * rank),
            nn.Linear(3 * rank, rank, bias=False),
            nn.SiLU(),
            nn.Linear(rank, 1),
        )
        nn.init.normal_(self.text_pool_query, std=rank**-0.5)
        nn.init.normal_(self.stop_key, std=rank**-0.5)
        # A newly attached router starts neutral about the value of another view.
        nn.init.zeros_(self.gain_head[-1].weight)
        nn.init.zeros_(self.gain_head[-1].bias)

    def prompt_query(
        self,
        text_embeddings: Tensor,
        text_attention_mask: Tensor,
        labels: Tensor | None,
    ) -> Tensor:
        if text_embeddings.ndim != 3:
            raise ValueError("text_embeddings must be [batch, tokens, width]")
        if text_attention_mask.shape != text_embeddings.shape[:2]:
            raise ValueError("text_attention_mask shape mismatch")
        prompt_mask = text_attention_mask.bool()
        if labels is not None:
            if labels.shape != text_embeddings.shape[:2]:
                raise ValueError("labels shape mismatch")
            prompt_mask = prompt_mask & labels.eq(-100)
        if not prompt_mask.any(dim=1).all():
            raise ValueError("router requires at least one prompt token per sample")
        text = self.text_norm(text_embeddings)
        scores = torch.einsum(
            "btd,d->bt", self.text_pool_key(text), self.text_pool_query
        )
        scores = scores.masked_fill(~prompt_mask, torch.finfo(scores.dtype).min)
        weights = torch.softmax(scores.float(), dim=-1).to(text.dtype)
        return torch.einsum("bt,btd->bd", weights, self.text_pool_value(text))

    def forward(
        self,
        local_vision_features: Tensor,
        text_embeddings: Tensor,
        text_attention_mask: Tensor,
        labels: Tensor | None = None,
        patch_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        if local_vision_features.ndim != 3:
            raise ValueError("local_vision_features must be [batch, patches, width]")
        if local_vision_features.shape[0] != text_embeddings.shape[0]:
            raise ValueError("vision/text batch mismatch")
        if patch_mask is not None:
            if patch_mask.shape != local_vision_features.shape[:2] or patch_mask.dtype != torch.bool:
                raise ValueError("patch_mask must be boolean [batch, patches]")
            if not patch_mask.any(dim=1).all():
                raise ValueError("every sample must contain at least one valid patch")

        query = self.prompt_query(text_embeddings, text_attention_mask, labels)
        vision = self.vision_norm(local_vision_features)
        query_key = F.normalize(query.float(), dim=-1)
        visual_key = F.normalize(self.vision_key(vision).float(), dim=-1)
        stop_key = F.normalize(self.stop_key.float(), dim=-1)
        scale = self.logit_scale.float().clamp(max=math.log(100.0)).exp()
        patch_logits = torch.einsum("bd,bnd->bn", query_key, visual_key) * scale
        if patch_mask is not None:
            patch_logits = patch_logits.masked_fill(~patch_mask, float("-inf"))
        stop_logits = torch.einsum("bd,d->b", query_key, stop_key).unsqueeze(1) * scale
        pointer_logits = torch.cat((patch_logits, stop_logits), dim=1)

        pointer_probabilities = torch.softmax(pointer_logits, dim=-1)
        evidence = torch.einsum(
            "bn,bnd->bd",
            pointer_probabilities[:, :-1].to(vision.dtype),
            self.vision_value(vision),
        )
        gain_features = torch.cat((query, evidence, query * evidence), dim=-1)
        predicted_gain = self.gain_head(gain_features).squeeze(-1).float()
        return pointer_logits, predicted_gain


def build_evidence_targets(
    global_correct_answer_nll: Tensor,
    best_crop_correct_answer_nll: Tensor,
    best_patch_index: Tensor,
    patch_count: int,
    minimum_gain: float,
) -> EvidenceTargets:
    """Build STOP/pointer targets from frozen counterfactual crop outcomes."""
    if patch_count < 1:
        raise ValueError("patch_count must be positive")
    if global_correct_answer_nll.shape != best_crop_correct_answer_nll.shape:
        raise ValueError("global/crop NLL shape mismatch")
    if best_patch_index.shape != global_correct_answer_nll.shape:
        raise ValueError("best_patch_index shape mismatch")
    if best_patch_index.dtype != torch.long:
        raise ValueError("best_patch_index must be torch.long")
    if not ((best_patch_index >= 0) & (best_patch_index < patch_count)).all():
        raise ValueError("best_patch_index outside patch grid")
    observed_gain = (
        global_correct_answer_nll.float() - best_crop_correct_answer_nll.float()
    ).detach()
    zoom_is_beneficial = observed_gain > float(minimum_gain)
    stop_index = torch.full_like(best_patch_index, patch_count)
    pointer = torch.where(zoom_is_beneficial, best_patch_index, stop_index)
    return EvidenceTargets(pointer, observed_gain, zoom_is_beneficial)


def evidence_router_loss(
    pointer_logits: Tensor,
    predicted_gain: Tensor,
    targets: EvidenceTargets,
    pointer_weight: float = 1.0,
    gain_weight: float = 1.0,
) -> tuple[Tensor, dict[str, Tensor]]:
    if pointer_logits.ndim != 2:
        raise ValueError("pointer_logits must be [batch, patches_plus_stop]")
    if predicted_gain.shape != targets.observed_gain.shape:
        raise ValueError("predicted/observed gain shape mismatch")
    if targets.pointer.shape != predicted_gain.shape:
        raise ValueError("pointer target shape mismatch")
    pointer_loss = F.cross_entropy(pointer_logits.float(), targets.pointer)
    gain_loss = F.smooth_l1_loss(predicted_gain.float(), targets.observed_gain.float())
    total = float(pointer_weight) * pointer_loss + float(gain_weight) * gain_loss
    return total, {"pointer_ce": pointer_loss.detach(), "gain_huber": gain_loss.detach()}


def choose_evidence_action(
    pointer_logits: Tensor,
    predicted_gain: Tensor,
    minimum_predicted_gain: float,
    minimum_action_probability: float,
) -> Tensor:
    """Return a patch index or the final STOP index under calibrated thresholds."""
    if pointer_logits.ndim != 2 or predicted_gain.ndim != 1:
        raise ValueError("invalid router output shape")
    if pointer_logits.shape[0] != predicted_gain.shape[0]:
        raise ValueError("router output batch mismatch")
    probabilities = torch.softmax(pointer_logits.float(), dim=-1)
    probability, action = probabilities.max(dim=-1)
    stop = torch.full_like(action, pointer_logits.shape[1] - 1)
    acquire = (
        action.ne(stop)
        & predicted_gain.ge(float(minimum_predicted_gain))
        & probability.ge(float(minimum_action_probability))
    )
    return torch.where(acquire, action, stop)


def patch_index_to_crop_box(
    patch_index: int,
    grid_size: int,
    expansion_cells: float = 1.0,
) -> tuple[float, float, float, float]:
    """Map a coarse patch to an expanded normalized [x1,y1,x2,y2] crop box."""
    if grid_size < 1:
        raise ValueError("grid_size must be positive")
    if not 0 <= patch_index < grid_size * grid_size:
        raise ValueError("patch_index outside grid")
    if expansion_cells < 0:
        raise ValueError("expansion_cells must be non-negative")
    row, column = divmod(patch_index, grid_size)
    margin = expansion_cells / grid_size
    x1 = max(0.0, column / grid_size - margin)
    y1 = max(0.0, row / grid_size - margin)
    x2 = min(1.0, (column + 1) / grid_size + margin)
    y2 = min(1.0, (row + 1) / grid_size + margin)
    return x1, y1, x2, y2
