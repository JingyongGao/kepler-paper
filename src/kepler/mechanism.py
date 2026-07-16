from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .config import DEFAULT_CONFIG, ExperimentConfig
from .data import split_seed
from .durability import atomic_write_json
from .environment import file_sha256, git_commit
from .learnability import LearnabilityRun
from .model import ModelSize, size_spec
from .topology import FiniteFieldRule


AUDIT_FLAGS = {
    "exploratory": True,
    "diagnostic_only": True,
    "excluded_from_confirmatory_fit": True,
}


@dataclass(frozen=True)
class MechanismRun:
    run_id: str
    description: str
    purpose: str
    model_kind: str
    target_core_parameters: int
    actual_core_parameters: int
    total_parameters: int
    width: int | None
    rule: dict[str, object]
    loss_type: str
    exposure_unit: str
    maximum_exposure: int
    checkpoints: tuple[int, ...]
    evaluation_seed: int
    data_seed: int
    model_initialization_seed: int
    probe_checkpoints: tuple[int, ...]
    probe_layers: tuple[int, ...]
    exploratory: bool = True
    diagnostic_only: bool = True
    excluded_from_confirmatory_fit: bool = True


def _rule(k: int, d: int, lags: tuple[int, ...], coefficients: tuple[int, ...]):
    return FiniteFieldRule(
        p=257,
        eta=0.1,
        k=k,
        d=d,
        lags=lags,
        coefficients=coefficients,
        instance=0,
    )


def build_mechanism_isolation_runs(
    cfg: ExperimentConfig = DEFAULT_CONFIG,
) -> tuple[MechanismRun, ...]:
    short = (500_000, 1_000_000, 2_000_000, 5_000_000, 10_000_000)
    long = (5_000_000, 10_000_000, 20_000_000, 40_000_000)
    size_2m = size_spec(2_000_000, cfg.model, cfg.field.p)
    size_10m = size_spec(10_000_000, cfg.model, cfg.field.p)
    size_parent_transformer = ModelSize(
        2_000_000,
        size_2m.width,
        size_2m.actual_core_parameters,
        size_2m.total_parameters + size_2m.width,
    )
    size_parent_mlp = ModelSize(2_000_000, 168, 2_045_940, 2_132_292)
    rules = {
        "M1": _rule(1, 4, (4,), (45,)),
        "M2": _rule(2, 2, (1, 2), (45, 36)),
        "M3": _rule(2, 2, (1, 2), (45, 36)),
        "M4": _rule(2, 2, (1, 2), (45, 36)),
    }

    def run(
        identifier: str,
        description: str,
        purpose: str,
        kind: str,
        size,
        loss: str,
        unit: str,
        checkpoints: tuple[int, ...],
        probes: tuple[int, ...],
        layers: tuple[int, ...],
        offset: int,
    ) -> MechanismRun:
        rule = rules[identifier]
        return MechanismRun(
            run_id=f"mechanism_{identifier}",
            description=description,
            purpose=purpose,
            model_kind=kind,
            target_core_parameters=size.target_core_parameters,
            actual_core_parameters=size.actual_core_parameters,
            total_parameters=size.total_parameters,
            width=size.width,
            rule=rule.to_dict(),
            loss_type=loss,
            exposure_unit=unit,
            maximum_exposure=checkpoints[-1],
            checkpoints=checkpoints,
            evaluation_seed=split_seed(82_000 + offset, rule.k, rule.d, 0, 0),
            data_seed=split_seed(81_000 + offset, rule.k, rule.d, 0, 0),
            model_initialization_seed=0,
            probe_checkpoints=probes,
            probe_layers=layers,
        )

    runs = (
        run(
            "M1",
            "long-span single-parent retrieval control",
            "isolate lag-4 retrieval without a multi-parent interaction",
            "recurrent_transformer",
            size_2m,
            "sampled_nll",
            "fresh_supervised_tokens",
            short,
            (500_000, 2_000_000, 10_000_000),
            (0, 3, 6),
            1,
        ),
        run(
            "M2",
            "short-span two-parent interaction control",
            "remove long span while retaining the pure two-parent interaction",
            "recurrent_transformer",
            size_10m,
            "sampled_nll",
            "fresh_supervised_tokens",
            long,
            (5_000_000, 20_000_000, 40_000_000),
            (0, 3, 6),
            2,
        ),
        run(
            "M3",
            "parent-exposed Transformer",
            "test modular pair composition without recurrence, distractors, or support discovery",
            "parent_exposed_transformer",
            size_parent_transformer,
            "oracle_soft",
            "independent_uniform_parent_pairs",
            short,
            (500_000, 2_000_000, 10_000_000),
            (0, 3, 6),
            3,
        ),
        run(
            "M4",
            "concatenated-parent MLP",
            "remove attention while retaining the same modular pair target",
            "concatenated_parent_mlp",
            size_parent_mlp,
            "oracle_soft",
            "independent_uniform_parent_pairs",
            short,
            (),
            (),
            4,
        ),
    )
    if any(
        not (item.exploratory and item.diagnostic_only and item.excluded_from_confirmatory_fit)
        for item in runs
    ):
        raise AssertionError("all mechanism runs must be diagnostic-only and excluded")
    return runs


def mechanism_manifest_payload(cfg: ExperimentConfig = DEFAULT_CONFIG) -> dict[str, Any]:
    paths = {
        "production": "artifacts/manifest.json",
        "pilot": "artifacts/pilot_manifest.json",
        "learnability_audit": "artifacts/learnability_audit_manifest.json",
    }
    return {
        "project": "Project Kepler",
        "kind": "mechanism_isolation_audit",
        "created_from_git_commit": git_commit(Path.cwd()),
        "prior_manifest_sha256": {name: file_sha256(path) for name, path in paths.items()},
        "prior_manifests_mutated": False,
        "flags": dict(AUDIT_FLAGS),
        "coefficients_reused_from_existing_k2_d4_rule0": [45, 36],
        "gradient_signal_audit": {
            "trained": False,
            "p": 257,
            "model": "2M",
            "loss": "exact_soft_label_population_cross_entropy",
            "model_initializations": 8,
            "independent_batches_per_initialization_and_task": 32,
            "tasks": [
                {"k": 1, "d": 1, "lags": [1], "coefficients": [45]},
                {"k": 1, "d": 4, "lags": [4], "coefficients": [45]},
                {"k": 2, "d": 2, "lags": [1, 2], "coefficients": [45, 36]},
                {"k": 2, "d": 4, "lags": [1, 4], "coefficients": [45, 36]},
            ],
        },
        "runs": [asdict(item) for item in build_mechanism_isolation_runs(cfg)],
        "forbidden_automatic_extensions": [
            "p=17",
            "Fourier initialization",
            "curriculum learning",
            "global optimizer changes",
            "remaining micro-pilot",
            "production trajectories",
        ],
    }


def write_mechanism_manifest(
    output: str | Path = "artifacts/mechanism_isolation_manifest.json",
    cfg: ExperimentConfig = DEFAULT_CONFIG,
) -> tuple[Path, str]:
    path = atomic_write_json(output, mechanism_manifest_payload(cfg))
    return path, file_sha256(path)


def find_mechanism_run(run_id: str, cfg: ExperimentConfig = DEFAULT_CONFIG) -> MechanismRun:
    for run in build_mechanism_isolation_runs(cfg):
        if run.run_id == run_id:
            return run
    raise KeyError(f"unknown mechanism run: {run_id}")


def as_learnability_run(run: MechanismRun, cfg: ExperimentConfig = DEFAULT_CONFIG):
    if run.model_kind != "recurrent_transformer":
        raise ValueError("only recurrent mechanism controls use the Episode runner")
    rule = FiniteFieldRule(**{**run.rule, "lags": tuple(run.rule["lags"]), "coefficients": tuple(run.rule["coefficients"])})
    return LearnabilityRun(
        run_id=run.run_id,
        description=run.description,
        k=rule.k,
        d=rule.d,
        target_core_parameters=run.target_core_parameters,
        width=int(run.width),
        actual_core_parameters=run.actual_core_parameters,
        total_parameters=run.total_parameters,
        rule_instance=rule.instance,
        rule=rule.to_dict(),
        model_initialization_seed=run.model_initialization_seed,
        data_seed=run.data_seed,
        evaluation_seed=run.evaluation_seed,
        probe_seeds={
            "train": split_seed(83_000, rule.k, rule.d, rule.instance, 0),
            "validation": split_seed(84_000, rule.k, rule.d, rule.instance, 0),
            "test": split_seed(85_000, rule.k, rule.d, rule.instance, 0),
        },
        loss_type=run.loss_type,
        maximum_supervised_tokens=run.maximum_exposure,
        checkpoints=run.checkpoints,
        evaluation_episodes=256,
        probe_checkpoints=run.probe_checkpoints,
        probe_layers=run.probe_layers,
        probe_examples={"train": 4096, "validation": 1024, "test": 1024},
        fixed_mlp_hidden=256,
    )
