"""Dropless Top-2 MoE decoder with training-only CVCR instrumentation."""

from __future__ import annotations

import math

import torch
import torch.distributed as dist
import torch.distributed.nn.functional as dist_nn
from torch import nn
from torch.nn import functional as F

from .config import ModelConfig
from .cvcr import CreditPredictor, ProbeSampler, inject_counterfactual_credit


def invert_permutation(order: torch.Tensor) -> torch.Tensor:
    """Invert a one-dimensional permutation without paying for a second sort."""
    inverse = torch.empty_like(order)
    inverse.scatter_(
        0,
        order,
        torch.arange(order.numel(), device=order.device, dtype=order.dtype),
    )
    return inverse


def _scattermoe_functions():
    """Import the pinned optional backend only when it is selected."""
    try:
        from scattermoe.parallel_experts import flatten_sort_count, parallel_linear
    except ImportError as error:
        raise RuntimeError(
            "expert_backend='scattermoe' requires scattermoe commit "
            "47b5e1502e5a10e82c8e5945d761b877849871e7"
        ) from error
    return flatten_sort_count, parallel_linear


@torch.compiler.disable
def _liger_linear_cross_entropy(
    loss_module: nn.Module,
    weight: torch.Tensor,
    hidden: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:
    """Keep Liger's custom fused operator outside Dynamo's fake-tensor trace.

    Liger 0.8.2 executes correctly on torch 2.11, but its internal
    ``addmm(out_dtype=..., out=...)`` is not accepted by that release's fake
    tensor implementation.  This graph boundary preserves the fused CUDA
    operator while allowing the rest of the model to remain compiled.
    """
    return loss_module(weight, hidden, targets)


class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        value = hidden.float()
        value = value * torch.rsqrt(value.square().mean(-1, keepdim=True) + self.eps)
        return value.to(hidden.dtype) * self.weight.to(hidden.dtype)


def rotate_half(value: torch.Tensor) -> torch.Tensor:
    first, second = value.chunk(2, dim=-1)
    return torch.cat((-second, first), dim=-1)


class Attention(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.query_heads = config.num_attention_heads
        self.kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        hidden = config.hidden_size
        self.query = nn.Linear(hidden, self.query_heads * self.head_dim, bias=False)
        self.key = nn.Linear(hidden, self.kv_heads * self.head_dim, bias=False)
        self.value = nn.Linear(hidden, self.kv_heads * self.head_dim, bias=False)
        self.output = nn.Linear(hidden, hidden, bias=False)

    def forward(
        self, hidden: torch.Tensor, cosine: torch.Tensor, sine: torch.Tensor
    ) -> torch.Tensor:
        batch, sequence, _ = hidden.shape
        query = self.query(hidden).view(
            batch, sequence, self.query_heads, self.head_dim
        ).transpose(1, 2)
        key = self.key(hidden).view(
            batch, sequence, self.kv_heads, self.head_dim
        ).transpose(1, 2)
        value = self.value(hidden).view(
            batch, sequence, self.kv_heads, self.head_dim
        ).transpose(1, 2)
        cosine = cosine.to(query.dtype)
        sine = sine.to(query.dtype)
        query = query * cosine + rotate_half(query) * sine
        key = key * cosine + rotate_half(key) * sine
        attended = F.scaled_dot_product_attention(
            query,
            key,
            value,
            is_causal=True,
            enable_gqa=self.query_heads != self.kv_heads,
        )
        return self.output(attended.transpose(1, 2).contiguous().view(batch, sequence, -1))


class ScatterExpertLinear(nn.Module):
    """One expert projection and one FSDP lifecycle boundary.

    Keeping gate/up and down in separate modules lets composable FSDP reshard
    one projection before materializing the other projection or its full
    ScatterMoE weight-gradient workspace.
    """

    def __init__(self, experts: int, output_features: int, input_features: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(
            torch.empty(experts, output_features, input_features)
        )

    def forward(
        self,
        hidden: torch.Tensor,
        top_k: int,
        sorted_ids: torch.Tensor,
        sorted_scattered: torch.Tensor,
        offsets: torch.Tensor,
        *,
        grouped_in: bool = False,
        grouped_out: bool = False,
        detach_parameters: bool = False,
    ) -> torch.Tensor:
        _, parallel_linear = _scattermoe_functions()
        weight = self.weight.detach() if detach_parameters else self.weight
        return parallel_linear(
            hidden,
            weight.to(torch.bfloat16).permute(0, 2, 1),
            top_k,
            sorted_ids,
            sorted_scattered,
            offsets,
            grouped_in=grouped_in,
            grouped_out=grouped_out,
        )


class PartitionedScatterExpertLinear(nn.Module):
    """Output-partitioned expert projection with independent FSDP units.

    ScatterMoE's weight-gradient kernel materializes one dense BF16 workspace
    with the same shape as the expert weight.  Partitioning only the output
    dimension preserves the exact linear map while allowing each workspace to
    be reduced and released before the next partition is processed.
    """

    def __init__(
        self,
        experts: int,
        output_features: int,
        input_features: int,
        partitions: int,
    ) -> None:
        super().__init__()
        if output_features % partitions:
            raise ValueError("output features must divide projection partitions")
        width = output_features // partitions
        self.projections = nn.ModuleList(
            [
                ScatterExpertLinear(experts, width, input_features)
                for _ in range(partitions)
            ]
        )

    @property
    def weight(self) -> torch.Tensor:
        return torch.cat([projection.weight for projection in self.projections], dim=1)

    def forward(
        self,
        hidden: torch.Tensor,
        top_k: int,
        sorted_ids: torch.Tensor,
        sorted_scattered: torch.Tensor,
        offsets: torch.Tensor,
        *,
        grouped_in: bool = False,
        grouped_out: bool = False,
        detach_parameters: bool = False,
    ) -> torch.Tensor:
        return torch.cat(
            [
                projection(
                    hidden,
                    top_k,
                    sorted_ids,
                    sorted_scattered,
                    offsets,
                    grouped_in=grouped_in,
                    grouped_out=grouped_out,
                    detach_parameters=detach_parameters,
                )
                for projection in self.projections
            ],
            dim=-1,
        )


class ExpertBank(nn.Module):
    """SwiGLU experts with jagged grouped GEMM and an exact loop fallback."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        experts = config.num_experts
        hidden = config.hidden_size
        intermediate = config.expert_intermediate_size
        self.backend = config.expert_backend
        self.global_num_experts = experts
        self.expert_parallel_group = None
        self.expert_parallel_rank = 0
        self.expert_parallel_size = 1
        self.local_expert_start = 0
        partitions = config.expert_projection_partitions
        projection_type = (
            ScatterExpertLinear
            if partitions == 1
            else lambda e, o, i: PartitionedScatterExpertLinear(
                e, o, i, partitions
            )
        )
        self.gate_up_projection = projection_type(experts, 2 * intermediate, hidden)
        self.down_projection = projection_type(experts, hidden, intermediate)

    def fsdp_projection_units(self) -> list[ScatterExpertLinear]:
        units: list[ScatterExpertLinear] = []
        for projection in (self.gate_up_projection, self.down_projection):
            if isinstance(projection, PartitionedScatterExpertLinear):
                units.extend(projection.projections)
            else:
                units.append(projection)
        return units

    @property
    def gate_up(self) -> torch.Tensor:
        return self.gate_up_projection.weight

    @property
    def down(self) -> torch.Tensor:
        return self.down_projection.weight

    @staticmethod
    def _slice_projection_experts(
        projection: ScatterExpertLinear | PartitionedScatterExpertLinear,
        start: int,
        stop: int,
    ) -> None:
        units = (
            projection.projections
            if isinstance(projection, PartitionedScatterExpertLinear)
            else (projection,)
        )
        for unit in units:
            unit.weight = nn.Parameter(unit.weight[start:stop].detach().clone())

    def enable_expert_parallel(self, rank: int, world_size: int, group=None) -> None:
        """Keep one disjoint expert shard and route assignments with all-to-all.

        Every rank constructs the same globally seeded expert bank before this
        method is called. Slicing afterwards therefore preserves the exact
        global initialization while avoiding any expert-gradient all-reduce.
        """
        if self.expert_parallel_size != 1:
            raise RuntimeError("expert parallelism was already enabled")
        if world_size < 2 or self.global_num_experts % world_size:
            raise ValueError("expert count must divide the expert-parallel world size")
        if not 0 <= rank < world_size:
            raise ValueError("expert-parallel rank is outside the process group")
        local_experts = self.global_num_experts // world_size
        start = rank * local_experts
        stop = start + local_experts
        self._slice_projection_experts(self.gate_up_projection, start, stop)
        self._slice_projection_experts(self.down_projection, start, stop)
        self.expert_parallel_group = group
        self.expert_parallel_rank = rank
        self.expert_parallel_size = world_size
        self.local_expert_start = start

    @property
    def local_num_experts(self) -> int:
        return self.global_num_experts // self.expert_parallel_size

    def _expert_parallel(
        self,
        hidden: torch.Tensor,
        expert_ids: torch.Tensor,
        detach_parameters: bool,
    ) -> torch.Tensor:
        """Dispatch Top-K assignments to their owner ranks and return in place.

        The two autograd-aware all-to-all calls carry activation gradients back
        to the source tokens. Integer routing metadata uses the same frozen
        split vectors and does not participate in autograd.
        """
        if expert_ids.ndim == 1:
            expert_ids = expert_ids[:, None]
        assignments = hidden.repeat_interleave(expert_ids.shape[1], dim=0)
        flat_ids = expert_ids.reshape(-1)
        owners = torch.div(
            flat_ids, self.local_num_experts, rounding_mode="floor"
        )
        send_order = owners.argsort(stable=True)
        send_hidden = assignments.index_select(0, send_order).contiguous()
        send_ids = (
            flat_ids.index_select(0, send_order) - self.local_expert_start
        ).contiguous()
        # Convert global ids to the receiver-local range before transmission.
        sorted_owners = owners.index_select(0, send_order)
        send_ids = send_ids - (
            sorted_owners - self.expert_parallel_rank
        ) * self.local_num_experts
        send_counts = torch.bincount(
            owners, minlength=self.expert_parallel_size
        ).to(dtype=torch.int64)
        recv_counts = torch.empty_like(send_counts)
        dist.all_to_all_single(
            recv_counts, send_counts, group=self.expert_parallel_group
        )
        send_splits = [int(value) for value in send_counts.cpu().tolist()]
        recv_splits = [int(value) for value in recv_counts.cpu().tolist()]
        received_rows = sum(recv_splits)

        recv_hidden_buffer = hidden.new_empty((received_rows, hidden.shape[-1]))
        recv_hidden = dist_nn.all_to_all_single(
            recv_hidden_buffer,
            send_hidden,
            output_split_sizes=recv_splits,
            input_split_sizes=send_splits,
            group=self.expert_parallel_group,
        )
        recv_ids = flat_ids.new_empty((received_rows,))
        dist.all_to_all_single(
            recv_ids,
            send_ids,
            output_split_sizes=recv_splits,
            input_split_sizes=send_splits,
            group=self.expert_parallel_group,
        )
        if recv_ids.numel() and (
            int(recv_ids.min()) < 0 or int(recv_ids.max()) >= self.local_num_experts
        ):
            raise RuntimeError("received an assignment outside the local expert shard")
        local_outputs = self.forward_assignments(
            recv_hidden,
            recv_ids,
            detach_parameters=detach_parameters,
            _skip_expert_parallel=True,
        )
        returned_buffer = hidden.new_empty(send_hidden.shape)
        returned = dist_nn.all_to_all_single(
            returned_buffer,
            local_outputs,
            output_split_sizes=send_splits,
            input_split_sizes=recv_splits,
            group=self.expert_parallel_group,
        )
        restored = torch.empty_like(returned)
        restored.index_copy_(0, send_order, returned)
        return restored

    def _grouped(
        self,
        hidden: torch.Tensor,
        expert_ids: torch.Tensor,
        detach_parameters: bool,
    ) -> torch.Tensor:
        order = expert_ids.argsort(stable=True)
        inverse = invert_permutation(order)
        sorted_hidden = hidden.index_select(0, order).to(torch.bfloat16)
        sorted_experts = expert_ids.index_select(0, order)
        unique, counts = sorted_experts.unique_consecutive(return_counts=True)
        offsets = counts.cumsum(0).to(torch.int32)
        gate_up = self.gate_up.detach() if detach_parameters else self.gate_up
        down = self.down.detach() if detach_parameters else self.down
        gate_up = gate_up.index_select(0, unique)
        down = down.index_select(0, unique)
        gate_up = gate_up.to(torch.bfloat16)
        down = down.to(torch.bfloat16)
        # torch 2.11's native kernel consumes [group, K, N], despite an older
        # functional docstring describing the common weight as [group, N, K].
        projected = F.grouped_mm(
            sorted_hidden, gate_up.transpose(1, 2).contiguous(), offs=offsets
        )
        gate, up = projected.chunk(2, dim=-1)
        activated = F.silu(gate) * up
        result = F.grouped_mm(
            activated, down.transpose(1, 2).contiguous(), offs=offsets
        )
        return result.index_select(0, inverse)

    def _scattermoe(
        self,
        hidden: torch.Tensor,
        expert_ids: torch.Tensor,
        detach_parameters: bool,
    ) -> torch.Tensor:
        """Device-resident scattered expert GEMMs with full autograd support.

        ``hidden`` contains one row per token while ``expert_ids`` is ``[T, K]``.
        The returned rows are in token-major, slot-minor assignment order so the
        caller retains the exact existing FP32 routing-weight mixture.
        """
        flatten_sort_count, _ = _scattermoe_functions()
        if expert_ids.ndim == 1:
            expert_ids = expert_ids[:, None]
        top_k = expert_ids.shape[1]
        sorted_ids, sorted_scattered, offsets = flatten_sort_count(
            expert_ids, num_experts=self.gate_up.shape[0]
        )
        # Match the existing grouped_mm arithmetic contract: expert operands
        # and outputs are BF16 while parameter storage and accumulated grads
        # remain FP32.
        hidden = hidden.to(torch.bfloat16)
        projected = self.gate_up_projection(
            hidden,
            top_k,
            sorted_ids,
            sorted_scattered,
            offsets,
            grouped_out=True,
            detach_parameters=detach_parameters,
        )
        gate, up = projected.chunk(2, dim=-1)
        activated = F.silu(gate) * up
        return self.down_projection(
            activated,
            1,
            sorted_ids,
            sorted_scattered,
            offsets,
            grouped_in=True,
            detach_parameters=detach_parameters,
        )

    def _loop(
        self,
        hidden: torch.Tensor,
        expert_ids: torch.Tensor,
        detach_parameters: bool,
    ) -> torch.Tensor:
        output = torch.empty_like(hidden)
        gate_up = self.gate_up.detach() if detach_parameters else self.gate_up
        down = self.down.detach() if detach_parameters else self.down
        for expert in expert_ids.unique():
            index = int(expert.item())
            mask = expert_ids == expert
            values = hidden[mask]
            projected = F.linear(values, gate_up[index])
            gate, up = projected.chunk(2, dim=-1)
            output[mask] = F.linear(F.silu(gate) * up, down[index])
        return output

    def forward_assignments(
        self,
        hidden: torch.Tensor,
        expert_ids: torch.Tensor,
        *,
        detach_parameters: bool = False,
        _skip_expert_parallel: bool = False,
    ) -> torch.Tensor:
        if not hidden.numel():
            return torch.empty_like(hidden)
        if self.expert_parallel_size > 1 and not _skip_expert_parallel:
            return self._expert_parallel(hidden, expert_ids, detach_parameters)
        if self.backend == "scattermoe" and hidden.is_cuda:
            return self._scattermoe(hidden, expert_ids, detach_parameters)
        if self.backend == "grouped_mm" and hidden.is_cuda:
            return self._grouped(hidden, expert_ids, detach_parameters)
        return self._loop(hidden, expert_ids, detach_parameters)

    def forward_topk(
        self,
        hidden: torch.Tensor,
        expert_ids: torch.Tensor,
        *,
        detach_parameters: bool = False,
    ) -> torch.Tensor:
        """Execute Top-K assignments, avoiding repeated hidden rows when possible."""
        if self.expert_parallel_size > 1:
            return self._expert_parallel(hidden, expert_ids, detach_parameters)
        if self.backend == "scattermoe" and hidden.is_cuda:
            return self._scattermoe(hidden, expert_ids, detach_parameters)
        assignment_hidden = hidden.repeat_interleave(expert_ids.shape[-1], dim=0)
        return self.forward_assignments(
            assignment_hidden,
            expert_ids.reshape(-1),
            detach_parameters=detach_parameters,
        )


class MoELayer(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.router = nn.Linear(config.hidden_size, config.num_experts, bias=False)
        self.experts = ExpertBank(config)
        if config.cvcr_enabled:
            self.credit_predictor = CreditPredictor(
                config.hidden_size, config.num_experts, config.cvcr_rank
            )
            self.probe_sampler = ProbeSampler(
                config.num_experts,
                config.probe_fraction,
                config.probe_probability_floor,
                config.probe_sampler,
                config.abp_ema_decay,
            )
            self.cvcr_active = True
        else:
            self.credit_predictor = None
            self.probe_sampler = None
            self.cvcr_active = False

    def _router_auxiliary(
        self, logits: torch.Tensor, selected_ids: torch.Tensor
    ) -> torch.Tensor:
        probabilities = logits.float().softmax(dim=-1)
        dispatch = torch.zeros_like(probabilities)
        dispatch.scatter_(1, selected_ids, 1.0 / selected_ids.shape[-1])
        balance = self.config.num_experts * (
            probabilities.mean(dim=0) * dispatch.mean(dim=0)
        ).sum()
        z_loss = logits.float().logsumexp(dim=-1).square().mean()
        return (
            self.config.router_aux_loss_coefficient * balance
            + self.config.router_z_loss_coefficient * z_loss
        )

    def forward(self, hidden: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        original_shape = hidden.shape
        flat = hidden.reshape(-1, hidden.shape[-1])
        # ``flat.float()`` alone is not sufficient inside CUDA autocast: linear
        # is autocast-eligible and can still run in BF16.  Routing decisions and
        # the load-balancing reductions are part of the numerical recipe, so
        # execute the projection explicitly in FP32.
        with torch.autocast(device_type=flat.device.type, enabled=False):
            logits = F.linear(flat.float(), self.router.weight.float())
        top_logits, selected_ids = logits.topk(self.config.experts_per_token, dim=-1)
        selected_weights = top_logits.softmax(dim=-1).to(flat.dtype)

        assignment_outputs = self.experts.forward_topk(flat, selected_ids)
        selected_outputs = assignment_outputs.view(
            flat.shape[0], self.config.experts_per_token, flat.shape[-1]
        )
        if self.config.cvcr_enabled:
            mixed = (selected_outputs * selected_weights[..., None]).sum(dim=1)
        else:
            # The plain Top-2 path does not reuse the per-slot outputs.  Avoid
            # retaining a second [tokens, top_k, hidden] allocation at the
            # tight ScatterMoE backward peak.
            selected_outputs.mul_(selected_weights[..., None])
            mixed = selected_outputs.sum(dim=1)
        router_auxiliary = self._router_auxiliary(logits, selected_ids)
        predictor_loss = mixed.new_zeros(())

        if self.training and self.config.cvcr_enabled and self.cvcr_active:
            assert self.credit_predictor is not None
            assert self.probe_sampler is not None
            interval = self.config.cvcr_update_interval
            active_rows, probe_ids, inclusion = self.probe_sampler.sample(
                selected_ids, self.config.probe_fraction * interval
            )
            projected_hidden = self.credit_predictor.projected_hidden(flat.detach())
            if active_rows.numel():
                probe_outputs = self.experts.forward_assignments(
                    flat.detach().index_select(0, active_rows),
                    probe_ids,
                    detach_parameters=True,
                ).detach()
                predicted = self.credit_predictor.selected_outputs(
                    projected_hidden.index_select(0, active_rows), probe_ids
                )
                squared_error = (predicted.float() - probe_outputs.float()).square().mean(dim=-1)
                predictor_loss = (
                    interval
                    * self.config.predictor_loss_coefficient
                    * squared_error.mean()
                )
                # Uniform sampling does not consume the residual-variance EMA;
                # avoid an otherwise needless reduction on the hot path.
                if self.config.probe_sampler == "abp":
                    self.probe_sampler.observe_residuals(
                        probe_ids, squared_error.detach()
                    )
            else:
                probe_outputs = flat.new_empty((0, flat.shape[-1]))
                predictor_loss = predictor_loss + 0.0 * sum(
                    parameter.reshape(-1)[0]
                    for parameter in self.credit_predictor.parameters()
                )
            mixed = inject_counterfactual_credit(
                mixed,
                logits,
                selected_ids,
                selected_outputs,
                mixed.detach(),
                active_rows,
                probe_ids,
                probe_outputs,
                inclusion,
                projected_hidden,
                self.credit_predictor,
                interval * self.config.cvcr_coefficient,
            )
        return mixed.view(original_shape), router_auxiliary, predictor_loss


class DecoderBlock(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.attention = Attention(config)
        self.moe = MoELayer(config)
        self.input_norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_attention_norm = RMSNorm(config.hidden_size, config.rms_norm_eps)

    def forward(
        self, hidden: torch.Tensor, cosine: torch.Tensor, sine: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden = hidden + self.attention(self.input_norm(hidden), cosine, sine)
        moe_output, router_auxiliary, predictor_loss = self.moe(
            self.post_attention_norm(hidden)
        )
        return hidden + moe_output, router_auxiliary, predictor_loss


class MoELanguageModel(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            [DecoderBlock(config) for _ in range(config.num_hidden_layers)]
        )
        self.final_norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        if config.tie_word_embeddings:
            self.lm_head.weight = self.embedding.weight
        if config.loss_backend == "liger":
            try:
                from liger_kernel.transformers.fused_linear_cross_entropy import (
                    LigerFusedLinearCrossEntropyLoss,
                )
            except ImportError as error:
                raise RuntimeError(
                    "loss_backend='liger' requires the pinned liger-kernel runtime"
                ) from error
            self.fused_linear_cross_entropy = LigerFusedLinearCrossEntropyLoss(
                reduction="mean", accum_dtype=torch.float32
            )
        else:
            self.fused_linear_cross_entropy = None
        # CVCR modules must not perturb the deployed model's initialization.
        # A method pair with the same process seed therefore starts from exactly
        # the same embeddings, attention, experts, and router parameters.
        initialization_seed = torch.initial_seed()
        deployed_generator = torch.Generator(device="cpu")
        deployed_generator.manual_seed(initialization_seed)
        for name, parameter in self.named_parameters():
            if ".credit_predictor." in name:
                continue
            if parameter.ndim > 1:
                nn.init.normal_(
                    parameter,
                    mean=0.0,
                    std=config.initializer_range,
                    generator=deployed_generator,
                )
        residual_std = config.initializer_range / math.sqrt(2 * config.num_hidden_layers)
        for layer in self.layers:
            nn.init.normal_(
                layer.attention.output.weight,
                std=residual_std,
                generator=deployed_generator,
            )
            down_projection = layer.moe.experts.down_projection
            if isinstance(down_projection, PartitionedScatterExpertLinear):
                for projection in down_projection.projections:
                    nn.init.normal_(
                        projection.weight,
                        std=residual_std,
                        generator=deployed_generator,
                    )
            else:
                nn.init.normal_(
                    down_projection.weight,
                    std=residual_std,
                    generator=deployed_generator,
                )
        inverse = 1.0 / (
            config.rope_theta
            ** (torch.arange(0, config.head_dim, 2).float() / config.head_dim)
        )
        frequencies = torch.outer(
            torch.arange(config.max_position_embeddings).float(), inverse
        )
        embedding = torch.cat([frequencies, frequencies], dim=-1)[None, None, :, :]
        self.register_buffer("cosine", embedding.cos(), persistent=False)
        self.register_buffer("sine", embedding.sin(), persistent=False)

    def forward(
        self,
        input_ids: torch.Tensor,
        targets: torch.Tensor | None = None,
        all_targets_ignored: bool = False,
    ):
        sequence = input_ids.shape[1]
        hidden = self.embedding(input_ids)
        router_auxiliary = hidden.new_zeros(())
        predictor_loss = hidden.new_zeros(())
        for layer in self.layers:
            hidden, layer_router_auxiliary, layer_predictor_loss = layer(
                hidden,
                self.cosine[:, :, :sequence],
                self.sine[:, :, :sequence],
            )
            router_auxiliary = router_auxiliary + layer_router_auxiliary
            predictor_loss = predictor_loss + layer_predictor_loss
        hidden = self.final_norm(hidden)
        if targets is None:
            return self.lm_head(hidden)
        # Formal stage manifests can end with rank-local padding whose targets
        # are all ignored.  Every rank must still execute the same sharded
        # LM-head path, so make one harmless target valid and zero the result
        # afterwards instead of skipping the collective-bearing operation.
        loss_targets = targets
        if all_targets_ignored:
            loss_targets = targets.clone()
            loss_targets.reshape(-1)[0] = 0
        if self.fused_linear_cross_entropy is not None:
            cross_entropy = _liger_linear_cross_entropy(
                self.fused_linear_cross_entropy,
                self.lm_head.weight,
                hidden.reshape(-1, hidden.shape[-1]),
                loss_targets.reshape(-1),
            )
        else:
            logits = self.lm_head(hidden)
            cross_entropy = F.cross_entropy(
                logits.float().reshape(-1, self.config.vocab_size),
                loss_targets.reshape(-1),
            )
        if all_targets_ignored:
            cross_entropy = cross_entropy * 0.0
        total = cross_entropy + router_auxiliary + predictor_loss
        return total, cross_entropy.detach(), router_auxiliary.detach(), predictor_loss.detach()

    def set_cvcr_active(self, active: bool) -> None:
        """Select the temporally batched CVCR graph for this microbatch."""
        for layer in self.layers:
            layer.moe.cvcr_active = active

    def enable_expert_parallel(self, rank: int, world_size: int, group=None) -> None:
        for layer in self.layers:
            layer.moe.experts.enable_expert_parallel(rank, world_size, group)

    def active_flops_per_token(self, sequence_length: int) -> int:
        """Legacy full-square training-matmul convention retained for replay."""
        config = self.config
        hidden = config.hidden_size
        attention = (
            2 * hidden * hidden
            + 2 * hidden * config.num_key_value_heads * config.head_dim
        )
        experts = (
            3
            * hidden
            * config.expert_intermediate_size
            * config.experts_per_token
        )
        router = hidden * config.num_experts
        body = config.num_hidden_layers * (attention + experts + router)
        train_matmuls = 6 * (body + hidden * config.vocab_size)
        attention_matmuls = (
            12 * config.num_hidden_layers * hidden * sequence_length
        )
        return train_matmuls + attention_matmuls

    def causal_matmul_flops_per_token(self, sequence_length: int) -> int:
        """Useful training matmuls with exact causal attention-pair accounting."""
        config = self.config
        hidden = config.hidden_size
        attention = (
            2 * hidden * hidden
            + 2 * hidden * config.num_key_value_heads * config.head_dim
        )
        experts = (
            3
            * hidden
            * config.expert_intermediate_size
            * config.experts_per_token
        )
        router = hidden * config.num_experts
        body = config.num_hidden_layers * (attention + experts + router)
        train_matmuls = 6 * (body + hidden * config.vocab_size)
        causal_attention_matmuls = (
            6 * config.num_hidden_layers * hidden * (sequence_length + 1)
        )
        return train_matmuls + causal_attention_matmuls

    def expected_probe_flops_per_token(self) -> float:
        if not self.config.cvcr_enabled:
            return 0.0
        # One forward-only SwiGLU expert on the sampled token.
        return (
            self.config.probe_fraction
            * 6
            * self.config.hidden_size
            * self.config.expert_intermediate_size
            * self.config.num_hidden_layers
        )

    def inference_state_dict(self) -> dict[str, torch.Tensor]:
        return {
            key: value
            for key, value in self.state_dict().items()
            if ".credit_predictor." not in key and ".probe_sampler." not in key
        }
