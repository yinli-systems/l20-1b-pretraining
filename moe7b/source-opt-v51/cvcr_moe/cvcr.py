"""Control-variate counterfactual routing primitives.

Only the local-credit estimator is claimed to be conditionally unbiased. The
injected router update is an auxiliary surrogate, not an unbiased gradient of
discrete Top-k language-model routing.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


def control_variate_estimate(
    baseline: torch.Tensor,
    exact: torch.Tensor,
    included: torch.Tensor,
    inclusion_probability: torch.Tensor,
) -> torch.Tensor:
    """Horvitz-Thompson residual correction around a control variate."""
    probability = inclusion_probability.clamp_min(torch.finfo(baseline.dtype).tiny)
    return baseline + included.to(baseline.dtype) * (exact - baseline) / probability


class CreditPredictor(nn.Module):
    """Factorized input-conditioned predictor of expert output.

    mu_e(h) = m_e + Q^T (u_e * P h). This preserves the intended
    input-conditioned control variate while evaluating all expert credits in
    O(Tdr + TEr), rather than materializing a T x E x d tensor.
    """

    def __init__(self, hidden_size: int, num_experts: int, rank: int) -> None:
        super().__init__()
        self.default = nn.Parameter(torch.zeros(num_experts, hidden_size))
        self.input_projection = nn.Parameter(torch.empty(rank, hidden_size))
        self.output_projection = nn.Parameter(torch.empty(rank, hidden_size))
        self.expert_scale = nn.Parameter(torch.ones(num_experts, rank))
        nn.init.normal_(self.input_projection, std=0.02)
        nn.init.normal_(self.output_projection, std=0.02)

    def projected_hidden(self, hidden: torch.Tensor) -> torch.Tensor:
        return F.linear(hidden, self.input_projection)

    def selected_outputs(
        self, projected_hidden: torch.Tensor, expert_ids: torch.Tensor
    ) -> torch.Tensor:
        latent = projected_hidden * self.expert_scale.index_select(0, expert_ids)
        return self.default.index_select(0, expert_ids) + F.linear(
            latent, self.output_projection.t()
        )


class ProbeSampler(nn.Module):
    def __init__(
        self,
        num_experts: int,
        fraction: float,
        probability_floor: float,
        mode: str,
        ema_decay: float,
    ) -> None:
        super().__init__()
        self.num_experts = num_experts
        self.fraction = fraction
        self.probability_floor = probability_floor
        self.mode = mode
        self.ema_decay = ema_decay
        self.register_buffer("residual_variance", torch.ones(num_experts))
        self.register_buffer("measured_cost", torch.ones(num_experts))

    def _base_probabilities(self) -> torch.Tensor:
        if self.mode == "uniform":
            weights = torch.ones_like(self.residual_variance)
        else:
            weights = torch.sqrt(
                self.residual_variance.clamp_min(1e-12)
                / self.measured_cost.clamp_min(1e-12)
            )
        weights = weights + self.probability_floor
        return weights / weights.sum()

    @torch.no_grad()
    def sample(
        self, selected_ids: torch.Tensor, fraction: float | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return probed rows, inactive expert ids, and exact inclusion q."""
        fraction = self.fraction if fraction is None else fraction
        tokens = selected_ids.shape[0]
        device = selected_ids.device
        inactive_experts = self.num_experts - selected_ids.shape[1]
        if self.mode == "uniform":
            # Amortize the probe GEMM with one shared candidate expert and a
            # fixed-size sample with replacement from the rows where it is
            # inactive.  Sampling with replacement preserves a static GPU
            # shape even during early router collapse.  If M experts have at
            # least one inactive row, expert e has N_e eligible rows, and n
            # probes are drawn, an eligible pair has exact marginal
            # q=(1/M)*(1-(1-1/N_e)^n).
            # Cross-token samples are correlated, affecting variance but not
            # Horvitz-Thompson unbiasedness.  Fixed n also prevents a new
            # torch.compile graph for every random probe count.
            probe_count = round(tokens * fraction)
            if fraction <= inactive_experts / self.num_experts:
                if probe_count == 0:
                    return (
                        torch.empty(0, device=device, dtype=torch.long),
                        torch.empty(0, device=device, dtype=selected_ids.dtype),
                        torch.empty(
                            0, device=device, dtype=self.residual_variance.dtype
                        ),
                    )
                selected_counts = torch.zeros(
                    self.num_experts, device=device, dtype=torch.float32
                )
                selected_counts.scatter_add_(
                    0,
                    selected_ids.reshape(-1),
                    torch.ones_like(selected_ids, dtype=torch.float32).reshape(-1),
                )
                inactive_counts = tokens - selected_counts
                eligible = inactive_counts > 0
                eligible_count = eligible.sum()
                shared_expert = torch.multinomial(
                    eligible.to(torch.float32), 1, replacement=True
                ).squeeze(0).to(selected_ids.dtype)
                inactive = (selected_ids != shared_expert).all(dim=-1)
                inactive_count = inactive.sum().to(torch.float32)
                rows = torch.multinomial(
                    inactive.to(torch.float32), probe_count, replacement=True
                )
                sampled = shared_expert.expand(probe_count)
                probability = (
                    1.0
                    - torch.pow(
                        1.0 - inactive_count.reciprocal(), probe_count
                    )
                ) / eligible_count
                return rows, sampled, probability.expand(probe_count)

        # ABP, and the high-budget uniform fallback, use an independent
        # per-token categorical draw.
        token_mask = torch.rand(tokens, device=device) < fraction
        base = self._base_probabilities().to(device).expand(tokens, -1).clone()
        base.scatter_(1, selected_ids, 0.0)
        base /= base.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        sampled = torch.multinomial(base, num_samples=1).squeeze(-1)
        chosen_probability = base.gather(1, sampled[:, None]).squeeze(-1)
        inclusion = fraction * chosen_probability
        sampled = torch.where(token_mask, sampled, torch.full_like(sampled, -1))
        inclusion = torch.where(token_mask, inclusion, torch.ones_like(inclusion))
        rows = token_mask.nonzero(as_tuple=False).squeeze(-1)
        return (
            rows,
            sampled.index_select(0, rows),
            inclusion.index_select(0, rows),
        )

    @torch.no_grad()
    def observe_residuals(
        self, expert_ids: torch.Tensor, squared_error: torch.Tensor
    ) -> None:
        if not expert_ids.numel():
            return
        sums = torch.zeros_like(self.residual_variance)
        counts = torch.zeros_like(self.residual_variance)
        sums.scatter_add_(0, expert_ids, squared_error.to(sums.dtype))
        counts.scatter_add_(0, expert_ids, torch.ones_like(squared_error, dtype=counts.dtype))
        means = sums / counts.clamp_min(1.0)
        updated = self.ema_decay * self.residual_variance + (
            1.0 - self.ema_decay
        ) * means
        self.residual_variance.copy_(
            torch.where(counts > 0, updated, self.residual_variance)
        )


def _router_credit_gradient(
    grad_output: torch.Tensor,
    logits: torch.Tensor,
    selected_ids: torch.Tensor,
    selected_outputs: torch.Tensor,
    reference: torch.Tensor,
    probe_rows: torch.Tensor,
    probe_ids: torch.Tensor,
    probe_outputs: torch.Tensor,
    inclusion_probability: torch.Tensor,
    projected_hidden: torch.Tensor,
    predictor_default: torch.Tensor,
    predictor_output_projection: torch.Tensor,
    predictor_expert_scale: torch.Tensor,
    coefficient: torch.Tensor,
) -> torch.Tensor:
    grad = grad_output.reshape(-1, grad_output.shape[-1])
    credit_dtype = torch.bfloat16 if grad.is_cuda else torch.float32
    grad_compute = grad.to(credit_dtype)
    probabilities = logits.float().softmax(dim=-1)
    reference_compute = reference.to(credit_dtype)
    selected_compute = selected_outputs.to(credit_dtype)
    projected = projected_hidden.to(credit_dtype)
    default = predictor_default.to(credit_dtype)
    output_projection = predictor_output_projection.to(credit_dtype)
    scale = predictor_expert_scale.to(credit_dtype)

    # Tensor-core matmuls dominate this training-only estimator.  Keep them in
    # BF16 on CUDA, then promote credits and router reductions to FP32.
    grad_default = (grad_compute @ default.t()).float()
    grad_latent = grad_compute @ output_projection.t()
    predicted_dot = grad_default + torch.einsum(
        "tr,er,tr->te", grad_latent, scale, projected
    ).float()
    reference_dot = torch.sum(
        grad_compute * reference_compute,
        dim=-1,
        keepdim=True,
        dtype=torch.float32,
    )
    estimated_credit = -predicted_dot + reference_dot

    selected_credit = -torch.sum(
        grad_compute[:, None, :]
        * (selected_compute - reference_compute[:, None, :]),
        dim=-1,
        dtype=torch.float32,
    )
    estimated_credit.scatter_(
        1, selected_ids, selected_credit.to(estimated_credit.dtype)
    )

    if probe_rows.numel():
        exact = -torch.sum(
            grad_compute.index_select(0, probe_rows)
            * (
                probe_outputs.to(credit_dtype)
                - reference_compute.index_select(0, probe_rows)
            ),
            dim=-1,
            dtype=torch.float32,
        )
        baseline = estimated_credit[probe_rows, probe_ids]
        q = inclusion_probability.float()
        estimated_credit[probe_rows, probe_ids] = baseline + (exact - baseline) / q

    centered = estimated_credit - (
        probabilities * estimated_credit
    ).sum(dim=-1, keepdim=True)
    # Gradient of -lambda * sum_e stopgrad(A_e) log p_e.
    router_gradient = -coefficient.float() * (
        centered - probabilities * centered.sum(dim=-1, keepdim=True)
    )
    return router_gradient.to(logits.dtype)


# The unfused estimator has very low FLOPs but launches hundreds of tiny
# kernels per step.  Compile this fixed-shape backward subgraph independently
# so the ordinary model can remain eager when whole-model compilation is not a
# win for grouped MoE dispatch.
_compiled_router_credit_gradient = torch.compile(
    _router_credit_gradient, fullgraph=True, dynamic=False
)


class _InjectCounterfactualCredit(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        output: torch.Tensor,
        router_logits: torch.Tensor,
        selected_ids: torch.Tensor,
        selected_outputs: torch.Tensor,
        reference_output: torch.Tensor,
        probe_rows: torch.Tensor,
        probe_ids: torch.Tensor,
        probe_outputs: torch.Tensor,
        inclusion_probability: torch.Tensor,
        projected_hidden: torch.Tensor,
        predictor_default: torch.Tensor,
        predictor_output_projection: torch.Tensor,
        predictor_expert_scale: torch.Tensor,
        coefficient: torch.Tensor,
    ) -> torch.Tensor:
        ctx.save_for_backward(
            router_logits,
            selected_ids,
            selected_outputs,
            reference_output,
            probe_rows,
            probe_ids,
            probe_outputs,
            inclusion_probability,
            projected_hidden,
            predictor_default,
            predictor_output_projection,
            predictor_expert_scale,
            coefficient,
        )
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        (
            logits,
            selected_ids,
            selected_outputs,
            reference,
            probe_rows,
            probe_ids,
            probe_outputs,
            inclusion_probability,
            projected_hidden,
            predictor_default,
            predictor_output_projection,
            predictor_expert_scale,
            coefficient,
        ) = ctx.saved_tensors
        with torch.no_grad():
            implementation = (
                _compiled_router_credit_gradient
                if grad_output.is_cuda
                else _router_credit_gradient
            )
            router_gradient = implementation(
                grad_output,
                logits,
                selected_ids,
                selected_outputs,
                reference,
                probe_rows,
                probe_ids,
                probe_outputs,
                inclusion_probability,
                projected_hidden,
                predictor_default,
                predictor_output_projection,
                predictor_expert_scale,
                coefficient,
            )

        return (grad_output, router_gradient) + (None,) * 12


def inject_counterfactual_credit(
    output: torch.Tensor,
    router_logits: torch.Tensor,
    selected_ids: torch.Tensor,
    selected_outputs: torch.Tensor,
    reference_output: torch.Tensor,
    probe_rows: torch.Tensor,
    probe_ids: torch.Tensor,
    probe_outputs: torch.Tensor,
    inclusion_probability: torch.Tensor,
    projected_hidden: torch.Tensor,
    predictor: CreditPredictor,
    coefficient: float,
) -> torch.Tensor:
    coefficient_tensor = output.new_tensor(coefficient, dtype=torch.float32)
    return _InjectCounterfactualCredit.apply(
        output,
        router_logits,
        selected_ids,
        selected_outputs,
        reference_output,
        probe_rows,
        probe_ids,
        probe_outputs,
        inclusion_probability,
        projected_hidden,
        predictor.default,
        predictor.output_projection,
        predictor.expert_scale,
        coefficient_tensor,
    )
