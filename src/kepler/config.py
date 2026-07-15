from __future__ import annotations

from dataclasses import asdict, dataclass, field as dataclass_field
from typing import Any


@dataclass(frozen=True)
class FieldConfig:
    p: int = 257
    eta: float = 0.1
    d_max: int = 64

    def validate(self) -> None:
        if self.p != 257:
            raise ValueError("Kepler v3.2.2 is preregistered for p=257")
        if not 0.0 < self.eta < (self.p - 1) / self.p:
            raise ValueError("eta is outside the admissible range")


@dataclass(frozen=True)
class ModelFamilyConfig:
    target_core_parameters: tuple[int, ...] = (
        2_000_000,
        10_000_000,
        20_000_000,
        50_000_000,
    )
    n_layers: int = 6
    n_heads: int = 8
    context_length: int = 128
    visible_prefix_length: int = 64
    mlp_ratio: int = 4
    dropout: float = 0.0
    position_encoding: str = "learned_absolute"
    tie_embeddings: bool = True
    bias: bool = True
    # Disabled for Phase 2/3. Phase 4 uses dataclasses.replace(..., num_tasks=2).
    num_tasks: int = 0

    def validate(self, field_cfg: FieldConfig) -> None:
        if self.visible_prefix_length < field_cfg.d_max:
            raise ValueError("visible prefix C must be at least d_max")
        if self.context_length <= self.visible_prefix_length:
            raise ValueError("context must leave at least one scored token")
        if self.n_heads <= 0 or self.n_layers <= 0:
            raise ValueError("depth and head count must be positive")
        if self.position_encoding != "learned_absolute":
            raise ValueError("v3.2.2 freezes learned absolute position encoding")
        if not self.tie_embeddings:
            raise ValueError("v3.2.2 requires tied input/output embeddings")
        if self.num_tasks < 0:
            raise ValueError("num_tasks cannot be negative")


@dataclass(frozen=True)
class OptimizerConfig:
    learning_rate: float = 3.0e-4
    min_learning_rate: float = 3.0e-5
    betas: tuple[float, float] = (0.9, 0.95)
    weight_decay: float = 0.1
    grad_clip_norm: float = 1.0
    warmup_tokens: int = 5_000_000
    decay_tokens: int = 160_000_000


@dataclass(frozen=True)
class TrainingConfig:
    checkpoints: tuple[int, ...] = (
        5_000_000,
        10_000_000,
        20_000_000,
        40_000_000,
        80_000_000,
        160_000_000,
    )
    batch_size: int = 32
    seed_primary: int = 0
    seed_reproduction: int = 1
    rule_instances: int = 3
    reproduction_rule_instance: int = 0
    rule_seed: int = 322
    train_stream_seed: int = 10_000
    evaluation_stream_seed: int = 20_000
    probe_stream_seed: int = 30_000
    amp_dtype: str = "bfloat16"


@dataclass(frozen=True)
class GateConfig:
    phase0_delta: float = 0.04
    phase0_samples: int = 1_000_000
    phase0_context_buckets: int = 4
    phase1_loss_relative_tolerance: float = 5.0e-4
    phase1_gradient_relative_tolerance: float = 5.0e-3


@dataclass(frozen=True)
class ExperimentConfig:
    field: FieldConfig = dataclass_field(default_factory=FieldConfig)
    model: ModelFamilyConfig = dataclass_field(default_factory=ModelFamilyConfig)
    optimizer: OptimizerConfig = dataclass_field(default_factory=OptimizerConfig)
    training: TrainingConfig = dataclass_field(default_factory=TrainingConfig)
    gates: GateConfig = dataclass_field(default_factory=GateConfig)

    def __post_init__(self) -> None:
        self.field.validate()
        self.model.validate(self.field)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


DEFAULT_CONFIG = ExperimentConfig()
