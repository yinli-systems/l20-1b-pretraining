"""Minimal decoder-only visual bridge for bounded Stage-0/Stage-1 tests.

This module intentionally does not own the language or vision parent.  It only
compresses patch features, projects them to the language width, and constructs
masked-loss prefix embeddings.  Text-only inputs are returned byte-for-byte as
the caller supplied them.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn


@dataclass(frozen=True)
class CompressionSpec:
    input_tokens: int
    target_ratio: float
    requested_output_tokens: int | None = None

    def __post_init__(self) -> None:
        if self.input_tokens < 1:
            raise ValueError("input_tokens must be positive")
        if self.target_ratio <= 0:
            raise ValueError("target_ratio must be positive")
        if self.requested_output_tokens is not None and not (
            1 <= self.requested_output_tokens <= self.input_tokens
        ):
            raise ValueError("requested_output_tokens must be in [1, input_tokens]")

    @property
    def output_tokens(self) -> int:
        if self.requested_output_tokens is not None:
            return self.requested_output_tokens
        if self.target_ratio == 1:
            return self.input_tokens
        return max(1, round(self.input_tokens / self.target_ratio))

    @property
    def achieved_ratio(self) -> float:
        return self.input_tokens / self.output_tokens


class LearnedQueryCompressor(nn.Module):
    """Compress patch features with learned queries and one cross-attention block."""

    def __init__(
        self,
        vision_dim: int,
        output_tokens: int,
        num_heads: int = 12,
        mlp_ratio: int = 2,
    ) -> None:
        super().__init__()
        if vision_dim % num_heads:
            raise ValueError("vision_dim must be divisible by num_heads")
        if output_tokens < 1:
            raise ValueError("output_tokens must be positive")
        self.output_tokens = output_tokens
        self.queries = nn.Parameter(torch.empty(1, output_tokens, vision_dim))
        self.key_norm = nn.LayerNorm(vision_dim)
        self.query_norm = nn.LayerNorm(vision_dim)
        self.attention = nn.MultiheadAttention(
            vision_dim, num_heads, dropout=0.0, batch_first=True
        )
        hidden = vision_dim * mlp_ratio
        self.ff_norm = nn.LayerNorm(vision_dim)
        self.ff = nn.Sequential(
            nn.Linear(vision_dim, hidden, bias=False),
            nn.GELU(approximate="tanh"),
            nn.Linear(hidden, vision_dim, bias=False),
        )
        nn.init.normal_(self.queries, std=0.02)

    def forward(self, features: Tensor, patch_mask: Tensor | None = None) -> Tensor:
        if features.ndim != 3:
            raise ValueError("features must have shape [batch, patches, width]")
        if patch_mask is not None:
            if patch_mask.shape != features.shape[:2] or patch_mask.dtype != torch.bool:
                raise ValueError("patch_mask must be boolean [batch, patches]")
            if not patch_mask.any(dim=1).all():
                raise ValueError("every sample must contain at least one valid patch")
        query = self.queries.expand(features.shape[0], -1, -1)
        residual = query
        query = self.query_norm(query)
        key_value = self.key_norm(features)
        attended, _ = self.attention(
            query,
            key_value,
            key_value,
            key_padding_mask=None if patch_mask is None else ~patch_mask,
            need_weights=False,
        )
        query = residual + attended
        return query + self.ff(self.ff_norm(query))


class AdaptiveSpatialPoolCompressor(nn.Module):
    """Parameter-free 2D adaptive pooling baseline in the spirit of DeCo."""

    def __init__(self, input_tokens: int, output_tokens: int) -> None:
        super().__init__()
        self.input_grid = math.isqrt(input_tokens)
        self.output_grid = math.isqrt(output_tokens)
        if self.input_grid**2 != input_tokens:
            raise ValueError("spatial pooling requires a square input patch grid")
        if self.output_grid**2 != output_tokens:
            raise ValueError("spatial pooling requires a square output patch grid")

    def forward(self, features: Tensor, patch_mask: Tensor | None = None) -> Tensor:
        if features.ndim != 3:
            raise ValueError("features must have shape [batch, patches, width]")
        if features.shape[1] != self.input_grid**2:
            raise ValueError("feature count does not match the configured input grid")
        if patch_mask is not None and not patch_mask.all():
            raise ValueError("spatial pooling baseline does not support partial patch masks")
        batch, _, width = features.shape
        grid = features.transpose(1, 2).reshape(
            batch, width, self.input_grid, self.input_grid
        )
        pooled = F.adaptive_avg_pool2d(grid, (self.output_grid, self.output_grid))
        return pooled.flatten(2).transpose(1, 2).contiguous()


class SpatialQueryCompressor(nn.Module):
    """Spatially anchored learned resampler with a small attention residual."""

    def __init__(
        self,
        input_tokens: int,
        output_tokens: int,
        vision_dim: int,
        num_heads: int = 12,
        mlp_ratio: int = 2,
    ) -> None:
        super().__init__()
        self.pool = AdaptiveSpatialPoolCompressor(input_tokens, output_tokens)
        self.query_norm = nn.LayerNorm(vision_dim)
        self.key_norm = nn.LayerNorm(vision_dim)
        self.attention = nn.MultiheadAttention(
            vision_dim, num_heads, dropout=0.0, batch_first=True
        )
        hidden = vision_dim * mlp_ratio
        self.ff_norm = nn.LayerNorm(vision_dim)
        self.ff = nn.Sequential(
            nn.Linear(vision_dim, hidden, bias=False),
            nn.GELU(approximate="tanh"),
            nn.Linear(hidden, vision_dim, bias=False),
        )
        self.attention_scale = nn.Parameter(torch.tensor(0.1))
        self.ff_scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, features: Tensor, patch_mask: Tensor | None = None) -> Tensor:
        pooled = self.pool(features, patch_mask)
        key_value = self.key_norm(features)
        attended, _ = self.attention(
            self.query_norm(pooled), key_value, key_value, need_weights=False
        )
        state = pooled + self.attention_scale * attended
        return state + self.ff_scale * self.ff(self.ff_norm(state))


class BindingResidual(nn.Module):
    """Low-rank 14x14-to-7x7 spatial residual without adding visual tokens."""

    def __init__(
        self,
        input_tokens: int,
        output_tokens: int,
        vision_dim: int,
        language_dim: int,
        rank: int,
    ) -> None:
        super().__init__()
        if rank < 1:
            raise ValueError("binding residual rank must be positive")
        self.input_grid = math.isqrt(input_tokens)
        self.output_grid = math.isqrt(output_tokens)
        if self.input_grid**2 != input_tokens or self.output_grid**2 != output_tokens:
            raise ValueError("binding residual requires square input/output token grids")
        if self.input_grid != 2 * self.output_grid:
            raise ValueError("binding residual currently requires a 2x spatial reduction")
        self.rank = rank
        self.read = nn.Linear(vision_dim, rank, bias=False)
        self.write = nn.Linear(4 * rank, language_dim, bias=False)
        nn.init.normal_(self.read.weight, std=vision_dim**-0.5)
        nn.init.zeros_(self.write.weight)

    def forward(self, features: Tensor) -> Tensor:
        if features.ndim != 3:
            raise ValueError("binding residual features must be [batch, patches, width]")
        if features.shape[1] != self.input_grid**2:
            raise ValueError("binding residual feature count does not match its input grid")
        low_rank = self.read(features).reshape(
            features.shape[0], self.input_grid, self.input_grid, self.rank
        )
        # Fixed top-left, top-right, bottom-left, bottom-right order in each 2x2 cell.
        packed = low_rank.reshape(
            features.shape[0], self.output_grid, 2, self.output_grid, 2, self.rank
        ).permute(0, 1, 3, 2, 4, 5).reshape(
            features.shape[0], self.output_grid**2, 4 * self.rank
        )
        return self.write(packed)


class QueryConditionedReadout(nn.Module):
    """Read one question-conditioned value from fixed-count visual tokens."""

    def __init__(self, language_dim: int, rank: int) -> None:
        super().__init__()
        if rank < 1:
            raise ValueError("query readout rank must be positive")
        self.rank = rank
        self.text_norm = nn.RMSNorm(language_dim)
        self.visual_norm = nn.RMSNorm(language_dim)
        self.text_key = nn.Linear(language_dim, rank, bias=False)
        self.text_value = nn.Linear(language_dim, rank, bias=False)
        self.text_pool_query = nn.Parameter(torch.empty(rank))
        self.visual_key = nn.Linear(language_dim, rank, bias=False)
        self.visual_value = nn.Linear(language_dim, rank, bias=False)
        self.logit_scale = nn.Parameter(torch.tensor(math.log(math.sqrt(rank))))
        self.output = nn.Linear(rank, language_dim, bias=False)
        nn.init.normal_(self.text_pool_query, std=rank**-0.5)
        nn.init.zeros_(self.output.weight)

    def prompt_query(
        self,
        text_embeddings: Tensor,
        text_attention_mask: Tensor,
        labels: Tensor | None,
    ) -> Tensor:
        prompt_mask = text_attention_mask.bool()
        if labels is not None:
            prompt_mask = prompt_mask & labels.eq(-100)
        if not prompt_mask.any(dim=1).all():
            raise ValueError("query readout requires at least one prompt token")
        text = self.text_norm(text_embeddings)
        keys = self.text_key(text)
        scores = torch.einsum("btd,d->bt", keys, self.text_pool_query)
        scores = scores.masked_fill(~prompt_mask, torch.finfo(scores.dtype).min)
        weights = torch.softmax(scores.float(), dim=-1).to(text.dtype)
        return torch.einsum("bt,btd->bd", weights, self.text_value(text))

    def forward(
        self,
        address_visual: Tensor,
        value_visual: Tensor,
        text_embeddings: Tensor,
        text_attention_mask: Tensor,
        labels: Tensor | None,
    ) -> tuple[Tensor, Tensor]:
        if address_visual.shape != value_visual.shape:
            raise ValueError("query readout address/value visual shapes must match")
        query = self.prompt_query(text_embeddings, text_attention_mask, labels)
        address = self.visual_norm(address_visual)
        value = self.visual_norm(value_visual)
        query_key = F.normalize(query.float(), dim=-1)
        visual_key = F.normalize(self.visual_key(address).float(), dim=-1)
        scale = self.logit_scale.float().clamp(max=math.log(100.0)).exp()
        logits = torch.einsum("bd,bnd->bn", query_key, visual_key) * scale
        attention = torch.softmax(logits.float(), dim=-1).to(value.dtype)
        readout = torch.einsum(
            "bn,bnd->bd", attention, self.visual_value(value)
        )
        return self.output(readout), attention


class VisionProjector(nn.Module):
    def __init__(self, vision_dim: int = 768, language_dim: int = 2048) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.LayerNorm(vision_dim),
            nn.Linear(vision_dim, language_dim, bias=False),
            nn.GELU(approximate="tanh"),
            nn.Linear(language_dim, language_dim, bias=False),
            nn.RMSNorm(language_dim),
        )

    def forward(self, features: Tensor) -> Tensor:
        return self.network(features)


class MultimodalBridge(nn.Module):
    """Create projected visual tokens and inject them as a causal prefix."""

    def __init__(
        self,
        input_tokens: int = 196,
        target_ratio: float = 4,
        output_tokens: int | None = None,
        vision_dim: int = 768,
        language_dim: int = 2048,
        num_heads: int = 12,
        compressor_kind: str = "learned_query",
        binding_residual_rank: int | None = None,
        query_readout_rank: int | None = None,
    ) -> None:
        super().__init__()
        self.spec = CompressionSpec(input_tokens, target_ratio, output_tokens)
        if compressor_kind not in {"learned_query", "spatial_pool", "spatial_query"}:
            raise ValueError(f"unsupported compressor_kind: {compressor_kind}")
        self.compressor_kind = compressor_kind
        if target_ratio == 1 and output_tokens is None:
            self.compressor = None
        elif compressor_kind == "learned_query":
            self.compressor = LearnedQueryCompressor(
                vision_dim, self.spec.output_tokens, num_heads=num_heads
            )
        elif compressor_kind == "spatial_pool":
            self.compressor = AdaptiveSpatialPoolCompressor(
                input_tokens, self.spec.output_tokens
            )
        else:
            self.compressor = SpatialQueryCompressor(
                input_tokens, self.spec.output_tokens, vision_dim, num_heads=num_heads
            )
        self.projector = VisionProjector(vision_dim, language_dim)
        self.binding_residual = (
            None
            if binding_residual_rank is None
            else BindingResidual(
                input_tokens,
                self.spec.output_tokens,
                vision_dim,
                language_dim,
                binding_residual_rank,
            )
        )
        self.query_readout = (
            None
            if query_readout_rank is None
            else QueryConditionedReadout(language_dim, query_readout_rank)
        )
        self.image_start = nn.Parameter(torch.empty(1, 1, language_dim))
        self.image_end = nn.Parameter(torch.empty(1, 1, language_dim))
        nn.init.normal_(self.image_start, std=0.02)
        nn.init.normal_(self.image_end, std=0.02)

    def compress(self, features: Tensor, patch_mask: Tensor | None = None) -> Tensor:
        if features.shape[1] != self.spec.input_tokens:
            raise ValueError(
                f"expected {self.spec.input_tokens} patches, got {features.shape[1]}"
            )
        if self.compressor is not None:
            return self.compressor(features, patch_mask)
        if patch_mask is not None:
            if patch_mask.shape != features.shape[:2] or patch_mask.dtype != torch.bool:
                raise ValueError("patch_mask must be boolean [batch, patches]")
            features = features * patch_mask.unsqueeze(-1)
        return features

    def core_visual_tokens(self, features: Tensor, patch_mask: Tensor | None = None) -> Tensor:
        return self.projector(self.compress(features, patch_mask))

    def residual_visual_tokens(self, features: Tensor) -> Tensor:
        if self.binding_residual is None:
            raise RuntimeError("bridge does not contain a binding residual")
        return self.binding_residual(features)

    def visual_tokens(
        self,
        features: Tensor,
        patch_mask: Tensor | None = None,
        binding_residual_features: Tensor | None = None,
    ) -> Tensor:
        visual = self.core_visual_tokens(features, patch_mask)
        if self.binding_residual is not None:
            residual_source = features if binding_residual_features is None else binding_residual_features
            visual = visual + self.residual_visual_tokens(residual_source)
        elif binding_residual_features is not None:
            raise RuntimeError("binding residual features supplied to a bridge without that path")
        return visual

    def inject(
        self,
        text_embeddings: Tensor,
        text_attention_mask: Tensor,
        labels: Tensor | None,
        vision_features: Tensor | None,
        patch_mask: Tensor | None = None,
        binding_residual_features: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor | None]:
        inputs, attention_mask, targets, _ = self.inject_with_readout(
            text_embeddings,
            text_attention_mask,
            labels,
            vision_features,
            patch_mask=patch_mask,
            binding_residual_features=binding_residual_features,
        )
        return inputs, attention_mask, targets

    def inject_with_readout(
        self,
        text_embeddings: Tensor,
        text_attention_mask: Tensor,
        labels: Tensor | None,
        vision_features: Tensor | None,
        patch_mask: Tensor | None = None,
        binding_residual_features: Tensor | None = None,
        query_readout_value_features: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor | None, Tensor | None]:
        if vision_features is None:
            if query_readout_value_features is not None:
                raise RuntimeError("query readout value features require vision features")
            return text_embeddings, text_attention_mask, labels, None
        if text_embeddings.ndim != 3:
            raise ValueError("text_embeddings must be [batch, tokens, width]")
        if text_attention_mask.shape != text_embeddings.shape[:2]:
            raise ValueError("text_attention_mask shape mismatch")
        if labels is not None and labels.shape != text_embeddings.shape[:2]:
            raise ValueError("labels shape mismatch")
        batch = text_embeddings.shape[0]
        visual = self.visual_tokens(
            vision_features, patch_mask, binding_residual_features=binding_residual_features
        )
        if visual.shape[0] != batch or visual.shape[2] != text_embeddings.shape[2]:
            raise ValueError("visual/text batch or width mismatch")
        start = self.image_start.expand(batch, -1, -1)
        end = self.image_end.expand(batch, -1, -1)
        readout_attention = None
        if self.query_readout is not None:
            value_visual = (
                visual
                if query_readout_value_features is None
                else self.visual_tokens(query_readout_value_features, patch_mask)
            )
            readout, readout_attention = self.query_readout(
                visual,
                value_visual,
                text_embeddings,
                text_attention_mask,
                labels,
            )
            end = end + readout.unsqueeze(1)
        elif query_readout_value_features is not None:
            raise RuntimeError("query readout value features supplied without a query readout")
        inputs = torch.cat((start, visual, end, text_embeddings), dim=1)
        prefix = torch.ones(
            batch,
            visual.shape[1] + 2,
            dtype=text_attention_mask.dtype,
            device=text_attention_mask.device,
        )
        attention_mask = torch.cat((prefix, text_attention_mask), dim=1)
        if labels is None:
            return inputs, attention_mask, None, readout_attention
        ignored = torch.full(
            (batch, visual.shape[1] + 2),
            -100,
            dtype=labels.dtype,
            device=labels.device,
        )
        return inputs, attention_mask, torch.cat((ignored, labels), dim=1), readout_attention


def trainable_parameter_count(module: nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters() if parameter.requires_grad)


def freeze(module: nn.Module) -> None:
    module.requires_grad_(False)
    module.eval()


def vision_features(vision: nn.Module, pixel_values: Tensor, layer: int = -1) -> Tensor:
    """Return a declared vision layer while preserving legacy last-layer behavior."""
    if layer == -1:
        return vision(pixel_values=pixel_values).last_hidden_state
    output = vision(pixel_values=pixel_values, output_hidden_states=True)
    if output.hidden_states is None:
        raise RuntimeError("vision encoder did not return hidden states")
    try:
        features = output.hidden_states[layer]
    except IndexError as error:
        raise ValueError(f"vision feature layer {layer} is out of range") from error
    if features.ndim != 3:
        raise RuntimeError("selected vision features must be [batch, patches, width]")
    return features


def bridge_from_architecture(architecture: dict) -> MultimodalBridge:
    """Construct a bridge from an auditable protocol architecture block."""
    compressor = architecture.get("compressor", "none")
    if compressor == "none":
        target_ratio = 1
        output_tokens = None
        compressor_kind = "learned_query"
    elif compressor in {"learned_query", "spatial_pool", "spatial_query"}:
        output_tokens = architecture.get("output_visual_tokens")
        if output_tokens is not None:
            output_tokens = int(output_tokens)
            input_tokens = int(architecture.get("visual_tokens", 196))
            target_ratio = input_tokens / output_tokens
        else:
            target_ratio = float(architecture["target_ratio"])
        compressor_kind = compressor
    else:
        raise ValueError(f"unsupported protocol compressor: {compressor}")
    return MultimodalBridge(
        input_tokens=int(architecture.get("visual_tokens", 196)),
        target_ratio=target_ratio,
        output_tokens=output_tokens,
        compressor_kind=compressor_kind,
        binding_residual_rank=(
            None
            if architecture.get("binding_residual_rank") is None
            else int(architecture["binding_residual_rank"])
        ),
        query_readout_rank=(
            None
            if architecture.get("query_readout_rank") is None
            else int(architecture["query_readout_rank"])
        ),
    )


def load_bridge_parent(
    bridge: MultimodalBridge,
    state: dict[str, Tensor],
    allow_new_compressor: bool = False,
) -> list[str]:
    """Strictly load a bridge, optionally allowing only a new compressor subtree."""
    incompatible = bridge.load_state_dict(state, strict=False)
    missing = sorted(incompatible.missing_keys)
    unexpected = sorted(incompatible.unexpected_keys)
    allowed_prefixes = ("compressor.", "binding_residual.", "query_readout.")
    allowed_missing = allow_new_compressor and bool(missing) and all(
        name.startswith(allowed_prefixes) for name in missing
    )
    if unexpected or (missing and not allowed_missing):
        raise RuntimeError(
            f"bridge parent mismatch: missing={missing}, unexpected={unexpected}"
        )
    return missing
