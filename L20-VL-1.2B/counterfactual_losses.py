"""Losses and metrics for counterfactual evidence-preserving compression."""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import random
from typing import Iterable, Sequence

import torch
import torch.nn.functional as F
from torch import Tensor


def masked_sequence_logprob(
    logits: Tensor, token_ids: Tensor, token_mask: Tensor
) -> Tensor:
    """Sum answer-token log probabilities for already aligned logits/targets."""
    if logits.ndim != 3 or token_ids.shape != logits.shape[:2]:
        raise ValueError("expected logits [batch,tokens,vocab] and aligned token ids")
    if token_mask.shape != token_ids.shape or token_mask.dtype != torch.bool:
        raise ValueError("token_mask must be boolean [batch,tokens]")
    token_logprob = F.log_softmax(logits.float(), dim=-1).gather(
        -1, token_ids.unsqueeze(-1)
    ).squeeze(-1)
    return (token_logprob * token_mask).sum(dim=-1)


def candidate_classification_loss(
    logits: Tensor,
    targets: Tensor,
    target_indices: Tensor,
    temperature: float = 1.0,
) -> tuple[Tensor, Tensor]:
    """Cross-entropy over paired sequence scores in evaluator candidate order."""
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if logits.shape[0] % 2 or target_indices.shape != (logits.shape[0] // 2,):
        raise ValueError("expected exactly two candidates per target index")
    shifted = targets[:, 1:]
    target_mask = shifted != -100
    safe_targets = shifted.masked_fill(~target_mask, 0)
    sequence_logprob = masked_sequence_logprob(
        logits[:, :-1], safe_targets, target_mask
    )
    candidate_scores = sequence_logprob.view(-1, 2)
    loss = F.cross_entropy(candidate_scores / temperature, target_indices)
    accuracy = (candidate_scores.argmax(dim=-1) == target_indices).float().mean()
    return loss, accuracy


def candidate_distillation_loss(
    student_scores: Tensor, teacher_scores: Tensor, temperature: float = 1.0
) -> Tensor:
    """KL-distill the teacher distribution over the same answer candidates."""
    if student_scores.shape != teacher_scores.shape or student_scores.shape[-1] != 2:
        raise ValueError("student and teacher must have matching two-candidate scores")
    if temperature <= 0:
        raise ValueError("distillation temperature must be positive")
    student_flat = student_scores.reshape(-1, 2)
    teacher_flat = teacher_scores.reshape(-1, 2)
    return F.kl_div(
        F.log_softmax(student_flat / temperature, dim=-1),
        F.softmax(teacher_flat.detach() / temperature, dim=-1),
        reduction="batchmean",
    ) * temperature**2


def answer_preference(answer_logprob: Tensor, alternative_logprob: Tensor) -> Tensor:
    if answer_logprob.shape != alternative_logprob.shape:
        raise ValueError("candidate log-probability shapes differ")
    return answer_logprob - alternative_logprob


def paired_preference_delta(base_preference: Tensor, edited_preference: Tensor) -> Tensor:
    if base_preference.shape != edited_preference.shape:
        raise ValueError("paired preference shapes differ")
    return base_preference - edited_preference


def evidence_delta_loss(
    student_base_preference: Tensor,
    student_edited_preference: Tensor,
    teacher_base_preference: Tensor,
    teacher_edited_preference: Tensor,
    teacher_pair_correct: Tensor,
    beta: float = 1.0,
) -> Tensor:
    """Huber-match cross-image preference changes on reliable teacher pairs only."""
    if teacher_pair_correct.dtype != torch.bool:
        raise ValueError("teacher_pair_correct must be boolean")
    student_delta = paired_preference_delta(
        student_base_preference, student_edited_preference
    )
    teacher_delta = paired_preference_delta(
        teacher_base_preference, teacher_edited_preference
    ).detach()
    if teacher_pair_correct.shape != student_delta.shape:
        raise ValueError("teacher reliability mask shape differs from preference batch")
    if not teacher_pair_correct.any():
        return student_delta.sum() * 0.0
    return F.smooth_l1_loss(
        student_delta[teacher_pair_correct], teacher_delta[teacher_pair_correct], beta=beta
    )


def invariance_delta_loss(
    student_base_preference: Tensor,
    student_invariant_preference: Tensor,
    teacher_base_preference: Tensor,
    teacher_invariant_preference: Tensor,
    teacher_pair_correct: Tensor,
    beta: float = 1.0,
) -> Tensor:
    """Match the full-token model's residual response to irrelevant edits."""
    return evidence_delta_loss(
        student_base_preference,
        student_invariant_preference,
        teacher_base_preference,
        teacher_invariant_preference,
        teacher_pair_correct,
        beta,
    )


def pair_joint_correct(base_correct: Tensor, edited_correct: Tensor) -> Tensor:
    if base_correct.dtype != torch.bool or edited_correct.dtype != torch.bool:
        raise ValueError("correctness tensors must be boolean")
    if base_correct.shape != edited_correct.shape:
        raise ValueError("paired correctness shapes differ")
    return base_correct & edited_correct


def compression_induced_failure(
    full_base_correct: Tensor,
    full_edited_correct: Tensor,
    compressed_base_correct: Tensor,
    compressed_edited_correct: Tensor,
) -> tuple[Tensor, Tensor]:
    """Return failure indicators and the full-token-correct eligibility mask."""
    eligible = pair_joint_correct(full_base_correct, full_edited_correct)
    compressed_joint = pair_joint_correct(
        compressed_base_correct, compressed_edited_correct
    )
    return eligible & ~compressed_joint, eligible


@dataclass(frozen=True)
class BootstrapInterval:
    estimate: float
    lower: float
    upper: float
    clusters: int
    samples: int
    resamples: int


def paired_cluster_bootstrap(
    differences: Sequence[float],
    cluster_ids: Sequence[str],
    *,
    resamples: int = 10_000,
    confidence: float = 0.95,
    seed: int = 20260913,
) -> BootstrapInterval:
    """Bootstrap a paired mean difference by scene/image family."""
    if len(differences) != len(cluster_ids) or not differences:
        raise ValueError("differences and cluster_ids must be non-empty and equally sized")
    if resamples < 1 or not 0.0 < confidence < 1.0:
        raise ValueError("invalid bootstrap configuration")
    grouped: dict[str, list[float]] = defaultdict(list)
    for difference, cluster in zip(differences, cluster_ids):
        grouped[str(cluster)].append(float(difference))
    cluster_means = [sum(values) / len(values) for _, values in sorted(grouped.items())]
    rng = random.Random(seed)
    draws = []
    count = len(cluster_means)
    for _ in range(resamples):
        draws.append(sum(cluster_means[rng.randrange(count)] for _ in range(count)) / count)
    draws.sort()
    alpha = (1.0 - confidence) / 2.0
    lower_index = max(0, int(alpha * resamples))
    upper_index = min(resamples - 1, int((1.0 - alpha) * resamples) - 1)
    return BootstrapInterval(
        estimate=sum(cluster_means) / count,
        lower=draws[lower_index],
        upper=draws[upper_index],
        clusters=count,
        samples=len(differences),
        resamples=resamples,
    )
