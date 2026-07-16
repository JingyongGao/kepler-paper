from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any

from .config import DEFAULT_CONFIG, ExperimentConfig
from .data import split_seed
from .durability import atomic_write_json
from .environment import file_sha256, git_commit
from .model import size_spec
from .parent_pair_split import COEFFICIENTS, P


@dataclass(frozen=True)
class CompositionalRun:
    run_id: str
    arm: str
    description: str
    model_kind: str
    target_core_parameters: int
    actual_core_parameters: int
    total_parameters: int
    width: int
    p: int
    coefficients: tuple[int, int]
    loss_type: str
    sampling_support: str
    model_initialization_seed: int
    data_seed: int
    evaluation_seed: int
    probe_seed: int
    maximum_pair_exposures: int
    checkpoints: tuple[int, ...]
    batch_size: int
    exploratory: bool = True
    diagnostic_only: bool = True
    excluded_from_confirmatory_scaling_fit: bool = True


def build_compositional_runs(
    cfg: ExperimentConfig = DEFAULT_CONFIG,
) -> tuple[CompositionalRun, ...]:
    checkpoints = (
        500_000,
        1_000_000,
        2_000_000,
        5_000_000,
        10_000_000,
        20_000_000,
        40_000_000,
    )
    size = size_spec(2_000_000, cfg.model, P)
    total_with_query = size.total_parameters + size.width
    output = []
    for seed in (0, 1, 2):
        paired_data_seed = split_seed(111_000, 2, 2, 0, seed)
        for arm, support, description in (
            ("G1", "full", "full-support parent-pair reference"),
            ("G2", "train", "frozen-80%-support compositional generalization test"),
        ):
            output.append(
                CompositionalRun(
                    run_id=f"composition_{arm}_s{seed}",
                    arm=arm,
                    description=description,
                    model_kind="parent_exposed_transformer",
                    target_core_parameters=2_000_000,
                    actual_core_parameters=size.actual_core_parameters,
                    total_parameters=total_with_query,
                    width=size.width,
                    p=P,
                    coefficients=COEFFICIENTS,
                    loss_type="oracle_soft",
                    sampling_support=support,
                    model_initialization_seed=seed,
                    data_seed=paired_data_seed,
                    evaluation_seed=split_seed(112_000, 2, 2, 0, seed),
                    probe_seed=split_seed(113_000, 2, 2, 0, seed),
                    maximum_pair_exposures=checkpoints[-1],
                    checkpoints=checkpoints,
                    batch_size=4096,
                )
            )
    runs = tuple(output)
    if len(runs) != 6 or any(
        not (
            run.exploratory
            and run.diagnostic_only
            and run.excluded_from_confirmatory_scaling_fit
        )
        for run in runs
    ):
        raise AssertionError("the compositional manifest must contain six excluded runs")
    return runs


def compositional_manifest_payload(
    split_path: str | Path = "artifacts/parent_pair_split.json",
    cfg: ExperimentConfig = DEFAULT_CONFIG,
) -> dict[str, Any]:
    split = json.loads(Path(split_path).read_text(encoding="utf-8"))
    prior = {
        name: file_sha256(path)
        for name, path in {
            "production": "artifacts/manifest.json",
            "pilot": "artifacts/pilot_manifest.json",
            "learnability": "artifacts/learnability_audit_manifest.json",
            "mechanism": "artifacts/mechanism_isolation_manifest.json",
        }.items()
    }
    return {
        "project": "Project Kepler v3.3",
        "kind": "compositional_generalization_gate",
        "created_from_git_commit": git_commit(Path.cwd()),
        "production_status": "suspended_before_execution",
        "prior_manifests_modified": False,
        "prior_manifest_sha256": prior,
        "parent_pair_split": {
            "path": str(split_path),
            "file_sha256": file_sha256(split_path),
            "assignment_sha256": split["assignment_sha256"],
            "preregistered_sha256_seed": split["preregistered_sha256_seed"],
            "train_pairs": split["train_pairs"],
            "heldout_pairs": split["heldout_pairs"],
        },
        "paired_protocol": {
            "model_initialization_seeds": [0, 1, 2],
            "same_initialization_within_seed": True,
            "same_data_rng_seed_within_seed": True,
            "same_optimizer_schedule_and_evaluation_code": True,
            "batch_size": 4096,
            "objective": "exact_soft_label_cross_entropy",
            "checkpoints": [500_000, 1_000_000, 2_000_000, 5_000_000, 10_000_000, 20_000_000, 40_000_000],
        },
        "flags": {
            "exploratory": True,
            "diagnostic_only": True,
            "excluded_from_confirmatory_scaling_fit": True,
        },
        "runs": [asdict(run) for run in build_compositional_runs(cfg)],
        "forbidden_in_this_gate": [
            "production trajectories",
            "remaining micro-pilot",
            "M2 extension",
            "field-size sweep",
            "Fourier initialization",
            "curriculum learning",
            "recurrent parent markers",
            "bilinear or gated MLPs",
            "global optimizer changes",
        ],
    }


def write_compositional_manifest(
    output: str | Path = "artifacts/compositional_generalization_manifest.json",
    split_path: str | Path = "artifacts/parent_pair_split.json",
    cfg: ExperimentConfig = DEFAULT_CONFIG,
) -> tuple[Path, str]:
    path = atomic_write_json(output, compositional_manifest_payload(split_path, cfg))
    return path, file_sha256(path)


def find_compositional_run(
    run_id: str, cfg: ExperimentConfig = DEFAULT_CONFIG
) -> CompositionalRun:
    for run in build_compositional_runs(cfg):
        if run.run_id == run_id:
            return run
    raise KeyError(f"unknown compositional run: {run_id}")
