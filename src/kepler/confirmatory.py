from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

from .config import DEFAULT_CONFIG, ExperimentConfig
from .confirmatory_distribution import (
    BLOCK_RNG_SEED_SHA256,
    MODEL_INITIALIZATION_SEED_SHA256,
)
from .cue_dose import PROBE_CHECKPOINTS, TRAIN_CHECKPOINTS
from .durability import atomic_write_json
from .environment import file_sha256, git_commit
from .model import size_spec


@dataclass(frozen=True)
class ConfirmatoryRun:
    run_id: str
    block_id: str
    rule_id: str
    rule_index: int
    coefficients: tuple[int, int]
    cue_coefficient_c: int
    permutation_id: str
    permutation_index: int
    permutation_seed_sha256: str
    B_m: tuple[int, ...]
    condition_id: str
    condition_order: int
    target_singleton_cue_index: float
    lambda_value: float
    model_kind: str
    target_core_parameters: int
    actual_core_parameters: int
    total_parameters: int
    width: int
    p: int
    eta: float
    m: int
    loss_type: str
    model_seed_index: int
    model_initialization_seed: int
    data_seed: int
    evaluation_seed: int
    probe_seed: int
    pre_switch_support: str
    post_switch_support: str
    switch_exposure: int
    maximum_pair_exposures: int
    checkpoints: tuple[int, ...]
    probe_checkpoints: tuple[int, ...]
    batch_size: int
    confirmatory_replication: bool = True
    excluded_from_original_scaling_fit: bool = True


def _derived_seed(master_sha256: str, label: str) -> int:
    digest = hashlib.sha256(
        bytes.fromhex(master_sha256) + label.encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:8], "big") % (2**31 - 1)


def model_initialization_seeds() -> tuple[int, int]:
    values = tuple(
        _derived_seed(MODEL_INITIALIZATION_SEED_SHA256, f"model_seed_{index}")
        for index in range(2)
    )
    if len(set(values)) != 2 or any(value in range(5) for value in values):
        raise AssertionError("confirmatory model seeds must be two new distinct seeds")
    return values


def _block_stream_seeds(block_id: str) -> dict[str, int]:
    return {
        kind: _derived_seed(BLOCK_RNG_SEED_SHA256, f"{block_id}:{kind}")
        for kind in ("data", "evaluation", "probe")
    }


def build_confirmatory_runs(
    audit_path: str | Path = "artifacts/confirmatory_distribution_audit.json",
    cfg: ExperimentConfig = DEFAULT_CONFIG,
) -> tuple[ConfirmatoryRun, ...]:
    audit = json.loads(Path(audit_path).read_text(encoding="utf-8"))
    size = size_spec(2_000_000, cfg.model, cfg.field.p)
    conditions = {
        (
            item["rule_id"],
            item["permutation_id"],
            item["condition_id"],
        ): item
        for item in audit["conditions"]
    }
    model_seeds = model_initialization_seeds()
    runs: list[ConfirmatoryRun] = []
    for rule_index, rule in enumerate(audit["rules"]):
        for permutation_index, permutation in enumerate(audit["permutations"]):
            for model_seed_index, model_seed in enumerate(model_seeds):
                block_id = (
                    f"{rule['rule_id']}_{permutation['permutation_id']}"
                    f"_init_{model_seed_index}"
                )
                streams = _block_stream_seeds(block_id)
                for condition_order, condition_id in enumerate(("R0", "R1", "R2")):
                    condition = conditions[
                        (rule["rule_id"], permutation["permutation_id"], condition_id)
                    ]
                    runs.append(
                        ConfirmatoryRun(
                            run_id=f"confirm_{block_id}_{condition_id}",
                            block_id=block_id,
                            rule_id=rule["rule_id"],
                            rule_index=rule_index,
                            coefficients=tuple(rule["coefficients"]),
                            cue_coefficient_c=int(rule["cue_coefficient_c"]),
                            permutation_id=permutation["permutation_id"],
                            permutation_index=permutation_index,
                            permutation_seed_sha256=permutation["seed_sha256"],
                            B_m=tuple(condition["B_m"]),
                            condition_id=condition_id,
                            condition_order=condition_order,
                            target_singleton_cue_index=float(
                                condition["target_singleton_cue_index"]
                            ),
                            lambda_value=float(condition["lambda"]),
                            model_kind="parent_exposed_transformer",
                            target_core_parameters=2_000_000,
                            actual_core_parameters=size.actual_core_parameters,
                            total_parameters=size.total_parameters + size.width,
                            width=size.width,
                            p=cfg.field.p,
                            eta=cfg.field.eta,
                            m=int(condition["m"]),
                            loss_type="oracle_soft",
                            model_seed_index=model_seed_index,
                            model_initialization_seed=model_seed,
                            data_seed=streams["data"],
                            evaluation_seed=streams["evaluation"],
                            probe_seed=streams["probe"],
                            pre_switch_support=(
                                "uniform_full"
                                if condition_id == "R0"
                                else "confirmatory_cue_full_support"
                            ),
                            post_switch_support="uniform_full",
                            switch_exposure=5_000_000,
                            maximum_pair_exposures=10_000_000,
                            checkpoints=TRAIN_CHECKPOINTS,
                            probe_checkpoints=PROBE_CHECKPOINTS,
                            batch_size=4096,
                        )
                    )
    if len(runs) != 24:
        raise AssertionError("confirmatory replication must contain 24 trajectories")
    return tuple(runs)


def confirmatory_manifest_payload(
    audit_path: str | Path = "artifacts/confirmatory_distribution_audit.json",
    cfg: ExperimentConfig = DEFAULT_CONFIG,
) -> dict[str, Any]:
    prior = {
        path: file_sha256(path)
        for path in (
            "artifacts/manifest.json",
            "artifacts/pilot_manifest.json",
            "artifacts/learnability_audit_manifest.json",
            "artifacts/mechanism_isolation_manifest.json",
            "artifacts/compositional_generalization_manifest.json",
            "artifacts/full_support_transfer_manifest.json",
            "artifacts/cue_dose_manifest.json",
        )
    }
    return {
        "project": "Project Kepler v3.6",
        "kind": "confirmatory_cross_rule_cue_dose_replication",
        "created_from_git_commit": git_commit(Path.cwd()),
        "prior_manifests_modified": False,
        "prior_manifest_sha256": prior,
        "frozen_v3_5_result_sha256": file_sha256(
            "artifacts/cue_dose_results.json"
        ),
        "distribution_audit": {
            "path": str(audit_path),
            "sha256": file_sha256(audit_path),
            "sha256_sidecar": str(Path(audit_path).with_suffix(".json.sha256")),
        },
        "generation_seeds_sha256": {
            "rules": json.loads(Path(audit_path).read_text(encoding="utf-8"))[
                "generation_seeds_sha256"
            ]["rules"],
            "cue_permutations": json.loads(
                Path(audit_path).read_text(encoding="utf-8")
            )["generation_seeds_sha256"]["cue_permutations"],
            "model_initializations": MODEL_INITIALIZATION_SEED_SHA256,
            "block_rng_streams": BLOCK_RNG_SEED_SHA256,
        },
        "flags": {
            "confirmatory_replication": True,
            "excluded_from_original_scaling_fit": True,
        },
        "factorial_design": {
            "rules": 2,
            "cue_permutations": 2,
            "model_initialization_seeds": 2,
            "cue_doses": 3,
            "paired_blocks": 8,
            "trajectories": 24,
            "condition_order": ["R0", "R1", "R2"],
            "same_initial_weights_within_block": True,
            "same_noncondition_rng_fields_within_block": True,
        },
        "frozen_protocol": {
            "model": "v3.5 parent-exposed approximately 2M Transformer",
            "objective": "exact_soft_label_cross_entropy",
            "optimizer": cfg.to_dict()["optimizer"],
            "scheduler": "absolute_exposure_cosine",
            "switch_exposure": 5_000_000,
            "maximum_exposure": 10_000_000,
            "checkpoints": [0, *TRAIN_CHECKPOINTS],
            "probe_checkpoints": list(PROBE_CHECKPOINTS),
            "gradient_audits": "not_run_unless_prespecified_numerical_anomaly",
        },
        "transition_definitions": {
            "onset": (
                "first probe checkpoint with full-support R<0.9 and layer-6 "
                "linear accuracy>0.5, with both retained at the next registered "
                "probe checkpoint"
            ),
            "establishment": (
                "first probe checkpoint with full-support R<0.1, center "
                "accuracy>0.9, and layer-6 linear accuracy>0.9"
            ),
            "terminal_onset_ordinal": 8,
            "ten_M_cannot_initiate_onset": True,
        },
        "primary_test": {
            "hypothesis": "R0 onset later than R1 and R1 onset later than R2",
            "test": "one-sided exact blocked Page-style permutation test",
            "within_block_assignments": 6,
            "blocks": 8,
            "exact_assignment_count": 6**8,
            "confirmatory_alpha": 0.05,
            "v3_5_results_excluded_from_primary_p_value": True,
        },
        "runs": [asdict(run) for run in build_confirmatory_runs(audit_path, cfg)],
        "forbidden": [
            "recurrent experiments or bridges",
            "production matrix",
            "model architecture changes",
            "optimizer changes",
            "scheduler changes",
            "field-size or eta changes",
            "objective changes",
            "v3.5 rules, cue permutations, or checkpoints",
            "model-size increases",
            "extensions beyond 10M",
        ],
    }


def write_confirmatory_manifest(
    output: str | Path = "artifacts/cue_dose_confirmatory_manifest.json",
    audit_path: str | Path = "artifacts/confirmatory_distribution_audit.json",
    cfg: ExperimentConfig = DEFAULT_CONFIG,
) -> tuple[Path, str]:
    path = atomic_write_json(
        output, confirmatory_manifest_payload(audit_path, cfg)
    )
    return path, file_sha256(path)


def find_confirmatory_run(
    run_id: str,
    audit_path: str | Path = "artifacts/confirmatory_distribution_audit.json",
) -> ConfirmatoryRun:
    for run in build_confirmatory_runs(audit_path):
        if run.run_id == run_id:
            return run
    raise KeyError(f"unknown confirmatory run: {run_id}")
