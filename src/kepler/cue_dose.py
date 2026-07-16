from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .compositional import build_compositional_runs
from .config import DEFAULT_CONFIG, ExperimentConfig
from .cue_distribution import sample_cue_pairs
from .data import split_seed
from .durability import atomic_write_json
from .environment import canonical_hash, file_sha256, git_commit
from .model import size_spec
from .pair_diagnostics import _pair_centers, _soft_loss, make_pair_model
from .topology import sample_noise


TRAIN_CHECKPOINTS = (
    500_000,
    1_000_000,
    2_000_000,
    3_000_000,
    4_000_000,
    5_000_000,
    5_500_000,
    6_000_000,
    7_500_000,
    10_000_000,
)
PROBE_CHECKPOINTS = (2_000_000, 3_000_000, 4_000_000, 5_000_000, 6_000_000, 7_500_000, 10_000_000)
GRADIENT_AUDIT_CHECKPOINTS = (0, 2_000_000, 4_000_000, 5_000_000, 6_000_000)


@dataclass(frozen=True)
class CueDoseRun:
    run_id: str
    condition_id: str
    condition_order: int
    m: int
    target_singleton_cue_index: float
    lambda_value: float
    B_m: tuple[int, ...]
    model_kind: str
    target_core_parameters: int
    actual_core_parameters: int
    total_parameters: int
    width: int
    p: int
    coefficients: tuple[int, int]
    cue_coefficient_c: int
    loss_type: str
    model_initialization_seed: int
    data_seed: int
    evaluation_seed: int
    probe_seed: int
    gradient_audit_seed: int
    pre_switch_support: str
    post_switch_support: str
    switch_exposure: int
    maximum_pair_exposures: int
    checkpoints: tuple[int, ...]
    probe_checkpoints: tuple[int, ...]
    gradient_audit_checkpoints: tuple[int, ...]
    batch_size: int
    execution_mode: str
    reused_source_run_id: str | None
    exploratory: bool = True
    diagnostic_only: bool = True
    excluded_from_confirmatory_scaling_fit: bool = True


def _state_hash(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in model.state_dict().items():
        digest.update(name.encode("utf-8"))
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def assert_c0_reuse_equivalence(
    cfg: ExperimentConfig = DEFAULT_CONFIG,
) -> dict[str, Any]:
    source_runs = {
        run.run_id: run for run in build_compositional_runs(cfg) if run.arm == "G1"
    }
    results = {}
    for seed in (0, 1, 2):
        source = source_runs[f"composition_G1_s{seed}"]
        expected_data_seed = split_seed(111_000, 2, 2, 0, seed)
        torch.manual_seed(seed)
        source_model = make_pair_model(source, cfg)
        source_hash = _state_hash(source_model)
        torch.manual_seed(seed)
        candidate_model = make_pair_model(source, cfg)
        candidate_hash = _state_hash(candidate_model)

        baseline_rng = np.random.default_rng(expected_data_seed)
        candidate_rng = np.random.default_rng(expected_data_seed)
        sampler_equal = True
        label_equal = True
        for count in (4096, 4096, 733):
            baseline_linear = baseline_rng.integers(
                0, cfg.field.p * cfg.field.p, size=count, dtype=np.int64
            )
            candidate_pairs, candidate_linear = sample_cue_pairs(
                candidate_rng, count, 0.0, 64
            )
            baseline_pairs = np.stack(
                (baseline_linear // cfg.field.p, baseline_linear % cfg.field.p), axis=1
            )
            sampler_equal &= bool(np.array_equal(baseline_linear, candidate_linear))
            sampler_equal &= bool(np.array_equal(baseline_pairs, candidate_pairs))
            centers = _pair_centers(
                baseline_pairs, tuple(source.coefficients), cfg.field.p
            )
            labels_a = (
                centers
                + sample_noise(baseline_rng, count, p=cfg.field.p, eta=cfg.field.eta)
            ) % cfg.field.p
            labels_b = (
                centers
                + sample_noise(candidate_rng, count, p=cfg.field.p, eta=cfg.field.eta)
            ) % cfg.field.p
            label_equal &= bool(np.array_equal(labels_a, labels_b))
        rng_equal = canonical_hash(baseline_rng.bit_generator.state) == canonical_hash(
            candidate_rng.bit_generator.state
        )

        source_run_record = json.loads(
            Path(f"artifacts/compositional_runs/{source.run_id}/run.json").read_text(
                encoding="utf-8"
            )
        )
        checkpoint = torch.load(
            f"artifacts/compositional_runs/{source.run_id}/checkpoint_2000000.pt",
            map_location="cpu",
            weights_only=False,
        )
        fixed_logits = torch.arange(4 * cfg.field.p, dtype=torch.float32).reshape(
            4, cfg.field.p
        ) / 1000.0
        fixed_centers = torch.tensor([0, 1, 17, 256])
        loss_a = _soft_loss(
            fixed_logits, fixed_centers, cfg.field.p, cfg.field.eta
        )
        loss_b = _soft_loss(
            fixed_logits.clone(), fixed_centers.clone(), cfg.field.p, cfg.field.eta
        )
        checks = {
            "model_config": (
                source.model_kind == "parent_exposed_transformer"
                and source.target_core_parameters == 2_000_000
                and source.p == cfg.field.p
                and tuple(source.coefficients) == (45, 36)
            ),
            "initial_weights": source_hash == candidate_hash,
            "optimizer_config": canonical_hash(
                source_run_record["optimizer"]
            ) == canonical_hash(cfg.to_dict()["optimizer"]),
            "scheduler": (
                checkpoint["scheduler_state"]["kind"] == "absolute_exposure_cosine"
                and checkpoint["scheduler_state"]["supervised_examples_seen"] == 2_000_000
            ),
            "pair_sampler": sampler_equal and rng_equal,
            "sampled_label_rng": label_equal,
            "objective": bool(torch.equal(loss_a, loss_b)) and source.loss_type == "oracle_soft",
            "checkpoint_exposure_definition": (
                source.batch_size == 4096
                and {
                    point for point in source.checkpoints if point <= 10_000_000
                }.issubset(set(TRAIN_CHECKPOINTS))
            ),
            "registered_seeds": (
                source.model_initialization_seed == seed
                and source.data_seed == expected_data_seed
            ),
        }
        results[str(seed)] = {
            "checks": checks,
            "passed": all(checks.values()),
            "source_run_id": source.run_id,
            "source_initial_weight_sha256": source_hash,
            "candidate_initial_weight_sha256": candidate_hash,
            "source_checkpoint_2M_sha256": file_sha256(
                f"artifacts/compositional_runs/{source.run_id}/checkpoint_2000000.pt"
            ),
        }
    return {
        "all_passed": all(result["passed"] for result in results.values()),
        "seeds": results,
    }


def build_cue_dose_runs(
    audit_path: str | Path = "artifacts/cue_distribution_audit.json",
    cfg: ExperimentConfig = DEFAULT_CONFIG,
) -> tuple[CueDoseRun, ...]:
    audit = json.loads(Path(audit_path).read_text(encoding="utf-8"))
    reuse = assert_c0_reuse_equivalence(cfg)
    size = size_spec(2_000_000, cfg.model, cfg.field.p)
    runs = []
    for order, condition in enumerate(audit["conditions"]):
        for seed in range(5):
            reuse_source = condition["condition_id"] == "C0" and seed < 3 and reuse["all_passed"]
            runs.append(
                CueDoseRun(
                    run_id=f"cue_{condition['condition_id']}_s{seed}",
                    condition_id=condition["condition_id"],
                    condition_order=order,
                    m=int(condition["m"]),
                    target_singleton_cue_index=float(
                        condition["target_singleton_cue_index"]
                    ),
                    lambda_value=float(condition["lambda"]),
                    B_m=tuple(int(value) for value in condition["B_m"]),
                    model_kind="parent_exposed_transformer",
                    target_core_parameters=2_000_000,
                    actual_core_parameters=size.actual_core_parameters,
                    total_parameters=size.total_parameters + size.width,
                    width=size.width,
                    p=cfg.field.p,
                    coefficients=(45, 36),
                    cue_coefficient_c=int(audit["cue_coefficient_c"]),
                    loss_type="oracle_soft",
                    model_initialization_seed=seed,
                    data_seed=split_seed(111_000, 2, 2, 0, seed),
                    evaluation_seed=split_seed(132_000, 2, 2, 0, seed),
                    probe_seed=split_seed(133_000, 2, 2, 0, seed),
                    gradient_audit_seed=split_seed(134_000, 2, 2, 0, seed),
                    pre_switch_support=(
                        "uniform_full" if condition["lambda"] == 0.0 else "cue_full_support"
                    ),
                    post_switch_support="uniform_full",
                    switch_exposure=5_000_000,
                    maximum_pair_exposures=10_000_000,
                    checkpoints=TRAIN_CHECKPOINTS,
                    probe_checkpoints=PROBE_CHECKPOINTS,
                    gradient_audit_checkpoints=GRADIENT_AUDIT_CHECKPOINTS,
                    batch_size=4096,
                    execution_mode=(
                        "frozen_G1_reuse_with_deterministic_replay_from_2M"
                        if reuse_source
                        else "new_training"
                    ),
                    reused_source_run_id=(
                        f"composition_G1_s{seed}" if reuse_source else None
                    ),
                )
            )
    if len(runs) != 30 or any(
        not (run.exploratory and run.diagnostic_only and run.excluded_from_confirmatory_scaling_fit)
        for run in runs
    ):
        raise AssertionError("cue-dose gate must contain 30 excluded cells")
    return tuple(runs)


def cue_dose_manifest_payload(
    audit_path: str | Path = "artifacts/cue_distribution_audit.json",
    cfg: ExperimentConfig = DEFAULT_CONFIG,
) -> dict[str, Any]:
    reuse = assert_c0_reuse_equivalence(cfg)
    if not reuse["all_passed"]:
        control_policy = "run_all_five_C0_controls_new"
    else:
        control_policy = "reuse_seeds_0_to_2_with_replay_for_missing_checkpoints_and_run_3_to_4_new"
    prior = {
        name: file_sha256(path)
        for name, path in {
            "production": "artifacts/manifest.json",
            "pilot": "artifacts/pilot_manifest.json",
            "learnability": "artifacts/learnability_audit_manifest.json",
            "mechanism": "artifacts/mechanism_isolation_manifest.json",
            "composition": "artifacts/compositional_generalization_manifest.json",
            "transfer": "artifacts/full_support_transfer_manifest.json",
        }.items()
    }
    return {
        "project": "Project Kepler v3.5",
        "kind": "full_support_cue_dose_causal_gate",
        "created_from_git_commit": git_commit(Path.cwd()),
        "prior_manifests_modified": False,
        "prior_manifest_sha256": prior,
        "cue_distribution_audit": {
            "path": str(audit_path),
            "sha256": file_sha256(audit_path),
        },
        "flags": {
            "exploratory": True,
            "diagnostic_only": True,
            "excluded_from_confirmatory_scaling_fit": True,
        },
        "paired_protocol": {
            "model_initialization_seeds": [0, 1, 2, 3, 4],
            "conditions": 6,
            "seed_condition_cells": 30,
            "same_initialization_within_seed": True,
            "same_base_data_seed_within_seed": True,
            "same_optimizer_and_absolute_exposure_scheduler": True,
            "objective": "exact_soft_label_cross_entropy",
            "sampler_switch_at_absolute_exposure": 5_000_000,
            "zero_optimizer_step_post_switch_output_equality_required": True,
            "maximum_total_exposure": 10_000_000,
        },
        "control_reuse": {
            "policy": control_policy,
            "assertions": reuse,
        },
        "transition_definitions": {
            "probe_checkpoint_sequence": list(PROBE_CHECKPOINTS),
            "onset": (
                "first probe checkpoint with full-support R<0.9 and layer-6 linear "
                "accuracy>0.5, with both conditions retained at the next probe checkpoint"
            ),
            "establishment": (
                "first checkpoint with full-support R<0.1, center accuracy>0.9, "
                "and layer-6 linear accuracy>0.9"
            ),
            "ordered_trend_conditions": ["C0", "C1", "C2", "C3"],
            "ordered_trend_test": (
                "exact blocked Page-style permutation test over within-seed onset-ordinal "
                "ranks; no onset by 10M receives the terminal ordinal after 10M"
            ),
        },
        "runs": [asdict(run) for run in build_cue_dose_runs(audit_path, cfg)],
        "forbidden": [
            "production trajectories",
            "recurrent experiments or bridges",
            "new hard support masks",
            "global optimizer changes",
            "p sweeps",
            "Fourier initialization",
            "extensions beyond 10M",
        ],
    }


def write_cue_dose_manifest(
    output: str | Path = "artifacts/cue_dose_manifest.json",
    audit_path: str | Path = "artifacts/cue_distribution_audit.json",
    cfg: ExperimentConfig = DEFAULT_CONFIG,
) -> tuple[Path, str]:
    path = atomic_write_json(output, cue_dose_manifest_payload(audit_path, cfg))
    return path, file_sha256(path)


def find_cue_dose_run(
    run_id: str,
    audit_path: str | Path = "artifacts/cue_distribution_audit.json",
) -> CueDoseRun:
    for run in build_cue_dose_runs(audit_path):
        if run.run_id == run_id:
            return run
    raise KeyError(f"unknown cue-dose run: {run_id}")
