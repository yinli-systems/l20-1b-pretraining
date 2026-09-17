"""Frozen model and method configuration."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int = 50_280
    hidden_size: int = 1_024
    expert_intermediate_size: int = 1_024
    num_hidden_layers: int = 10
    num_attention_heads: int = 16
    num_key_value_heads: int = 4
    num_experts: int = 16
    experts_per_token: int = 2
    max_position_embeddings: int = 2_048
    rms_norm_eps: float = 1e-5
    rope_theta: float = 10_000.0
    initializer_range: float = 0.02
    tie_word_embeddings: bool = True
    eos_token_id: int = 50_279
    pad_token_id: int = 1
    router_aux_loss_coefficient: float = 1e-3
    router_z_loss_coefficient: float = 1e-4
    router_dtype: str = "float32"
    expert_backend: str = "grouped_mm"
    expert_projection_partitions: int = 1
    loss_backend: str = "eager"
    method: str = "top2"
    cvcr_rank: int = 8
    cvcr_coefficient: float = 0.05
    predictor_loss_coefficient: float = 1e-3
    probe_fraction: float = 0.0625
    cvcr_update_interval: int = 1
    probe_sampler: str = "uniform"
    probe_probability_floor: float = 1e-4
    abp_ema_decay: float = 0.99

    def __post_init__(self) -> None:
        if self.hidden_size % self.num_attention_heads:
            raise ValueError("hidden_size must divide num_attention_heads")
        if self.num_attention_heads % self.num_key_value_heads:
            raise ValueError("query heads must divide key/value heads")
        if self.head_dim % 2:
            raise ValueError("RoPE requires an even head dimension")
        if not 0 < self.experts_per_token < self.num_experts:
            raise ValueError("experts_per_token must be in [1, num_experts)")
        if self.method not in {"top2", "cvcr", "cvcr_abp"}:
            raise ValueError(f"unsupported method: {self.method}")
        if self.expert_backend not in {"grouped_mm", "scattermoe", "loop"}:
            raise ValueError(f"unsupported expert backend: {self.expert_backend}")
        if self.expert_projection_partitions < 1:
            raise ValueError("expert_projection_partitions must be positive")
        if (2 * self.expert_intermediate_size) % self.expert_projection_partitions:
            raise ValueError("gate/up width must divide expert_projection_partitions")
        if self.hidden_size % self.expert_projection_partitions:
            raise ValueError("hidden size must divide expert_projection_partitions")
        if self.loss_backend not in {"eager", "liger"}:
            raise ValueError(f"unsupported loss backend: {self.loss_backend}")
        if self.probe_sampler not in {"uniform", "abp"}:
            raise ValueError(f"unsupported probe sampler: {self.probe_sampler}")
        if self.method == "cvcr_abp" and self.probe_sampler != "abp":
            raise ValueError("cvcr_abp requires the abp probe sampler")
        if not 0.0 <= self.probe_fraction <= 1.0:
            raise ValueError("probe_fraction must be in [0, 1]")
        if self.cvcr_update_interval < 1:
            raise ValueError("cvcr_update_interval must be positive")
        if self.probe_fraction * self.cvcr_update_interval > 1.0:
            raise ValueError("temporally batched probe fraction exceeds 1")
        if not 0 <= self.eos_token_id < self.vocab_size:
            raise ValueError("eos_token_id must be inside the configured vocabulary")
        if not 0 <= self.pad_token_id < self.vocab_size:
            raise ValueError("pad_token_id must be inside the configured vocabulary")
        if not 0.0 < self.probe_probability_floor < 1.0:
            raise ValueError("probe_probability_floor must be in (0, 1)")
        if not 0.0 < self.abp_ema_decay < 1.0:
            raise ValueError("abp_ema_decay must be in (0, 1)")

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_attention_heads

    @property
    def cvcr_enabled(self) -> bool:
        return self.method in {"cvcr", "cvcr_abp"}

    def deployed_parameter_count(self) -> int:
        d = self.hidden_size
        f = self.expert_intermediate_size
        layers = self.num_hidden_layers
        embeddings = self.vocab_size * d * (1 if self.tie_word_embeddings else 2)
        attention = 2 * d * d + 2 * d * self.num_key_value_heads * self.head_dim
        experts = 3 * d * f * self.num_experts
        router = d * self.num_experts
        norms = 2 * d
        return embeddings + layers * (attention + experts + router + norms) + d

    def active_parameter_count(self) -> int:
        d = self.hidden_size
        f = self.expert_intermediate_size
        layers = self.num_hidden_layers
        embeddings = self.vocab_size * d * (1 if self.tie_word_embeddings else 2)
        attention = 2 * d * d + 2 * d * self.num_key_value_heads * self.head_dim
        experts = 3 * d * f * self.experts_per_token
        router = d * self.num_experts
        norms = 2 * d
        return embeddings + layers * (attention + experts + router + norms) + d

    def predictor_parameter_count(self) -> int:
        if not self.cvcr_enabled:
            return 0
        per_layer = (
            self.num_experts * self.hidden_size
            + 2 * self.cvcr_rank * self.hidden_size
            + self.num_experts * self.cvcr_rank
        )
        return self.num_hidden_layers * per_layer

    def training_parameter_count(self) -> int:
        return self.deployed_parameter_count() + self.predictor_parameter_count()

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_json(cls, path: Path) -> "ModelConfig":
        with path.open() as handle:
            return cls(**json.load(handle))
