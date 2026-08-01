from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

from .config import DEFAULT_CONFIG, ExperimentConfig
from .confirmatory import model_initialization_seeds
from .cue_dose import PROBE_CHECKPOINTS, TRAIN_CHECKPOINTS
from .durability import atomic_write_bytes, atomic_write_json
from .environment import file_sha256, git_commit
from .model import size_spec


SCALE_TARGETS = (1_000_000, 5_000_000)
CONDITION_ORDER = ("R0", "R2")
FIXED_PERMUTATION_ID = "permutation_0"
ROBUSTNESS_BLOCK_RNG_SEED_SHA256 = hashlib.sha256(
    b"Project Kepler ICLR 2027 cross-scale robustness block streams v1"
).hexdigest()


@dataclass(frozen=True)
class ScaleRobustnessRun:
    run_id: str
    block_id: str
    scale_id: str
    rule_id: str
    rule_index: int
    coefficients: tuple[int, int]
    cue_coefficient_c: int
    permutation_id: str
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
    engineering_learnability_pilot: bool
    post_confirmatory_robustness_extension: bool = True
    excluded_from_original_v3_6_p_value: bool = True
    not_a_scaling_law_fit: bool = True


def _derived_seed(master_sha256: str, label: str) -> int:
    digest = hashlib.sha256(
        bytes.fromhex(master_sha256) + label.encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:8], "big") % (2**31 - 1)


def _scale_id(target: int) -> str:
    return f"{target // 1_000_000}m"


def _block_stream_seeds(target: int, rule_id: str, model_seed_index: int) -> dict[str, int]:
    block = f"{_scale_id(target)}:{rule_id}:init_{model_seed_index}"
    return {
        kind: _derived_seed(ROBUSTNESS_BLOCK_RNG_SEED_SHA256, f"{block}:{kind}")
        for kind in ("data", "evaluation", "probe")
    }


def build_scale_robustness_runs(
    audit_path: str | Path = "artifacts/confirmatory_distribution_audit.json",
    cfg: ExperimentConfig = DEFAULT_CONFIG,
) -> tuple[ScaleRobustnessRun, ...]:
    audit = json.loads(Path(audit_path).read_text(encoding="utf-8"))
    rules = {item["rule_id"]: item for item in audit["rules"]}
    conditions = {
        (item["rule_id"], item["permutation_id"], item["condition_id"]): item
        for item in audit["conditions"]
    }
    model_seeds = model_initialization_seeds()
    runs: list[ScaleRobustnessRun] = []
    for target in SCALE_TARGETS:
        size = size_spec(target, cfg.model, cfg.field.p)
        for rule_index, rule_id in enumerate(("rule_0", "rule_1")):
            rule = rules[rule_id]
            for model_seed_index, model_seed in enumerate(model_seeds):
                streams = _block_stream_seeds(target, rule_id, model_seed_index)
                block_id = f"scale_{_scale_id(target)}_{rule_id}_init_{model_seed_index}"
                for condition_order, condition_id in enumerate(CONDITION_ORDER):
                    condition = conditions[(rule_id, FIXED_PERMUTATION_ID, condition_id)]
                    runs.append(
                        ScaleRobustnessRun(
                            run_id=f"{block_id}_{condition_id}",
                            block_id=block_id,
                            scale_id=_scale_id(target),
                            rule_id=rule_id,
                            rule_index=rule_index,
                            coefficients=tuple(rule["coefficients"]),
                            cue_coefficient_c=int(rule["cue_coefficient_c"]),
                            permutation_id=FIXED_PERMUTATION_ID,
                            permutation_seed_sha256=condition["permutation_seed_sha256"],
                            B_m=tuple(condition["B_m"]),
                            condition_id=condition_id,
                            condition_order=condition_order,
                            target_singleton_cue_index=float(
                                condition["target_singleton_cue_index"]
                            ),
                            lambda_value=float(condition["lambda"]),
                            model_kind="parent_exposed_transformer",
                            target_core_parameters=target,
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
                            engineering_learnability_pilot=(
                                rule_id == "rule_0" and model_seed_index == 0
                            ),
                        )
                    )
    if len(runs) != 16:
        raise AssertionError("scale robustness extension must contain 16 trajectories")
    return tuple(runs)


def scale_robustness_manifest_payload(
    audit_path: str | Path = "artifacts/confirmatory_distribution_audit.json",
    cfg: ExperimentConfig = DEFAULT_CONFIG,
) -> dict[str, Any]:
    v36_manifest = json.loads(
        Path("artifacts/cue_dose_confirmatory_manifest.json").read_text(encoding="utf-8")
    )
    prior_paths = (
        "artifacts/manifest.json",
        "artifacts/pilot_manifest.json",
        "artifacts/learnability_audit_manifest.json",
        "artifacts/mechanism_isolation_manifest.json",
        "artifacts/compositional_generalization_manifest.json",
        "artifacts/full_support_transfer_manifest.json",
        "artifacts/cue_dose_manifest.json",
        "artifacts/cue_dose_confirmatory_manifest.json",
        "artifacts/cue_dose_confirmatory_results.json",
    )
    runs = build_scale_robustness_runs(audit_path, cfg)
    pilot_run_ids = [run.run_id for run in runs if run.engineering_learnability_pilot]
    transition_definitions = dict(v36_manifest["transition_definitions"])
    return {
        "project": "Project Kepler ICLR 2027 robustness extension",
        "kind": "post_confirmatory_cross_scale_robustness",
        "created_from_git_commit": git_commit(Path.cwd()),
        "prior_manifests_modified": False,
        "prior_manifest_and_result_sha256": {
            path: file_sha256(path) for path in prior_paths
        },
        "frozen_v3_6_sources": {
            "distribution_audit": {
                "path": str(audit_path),
                "sha256": file_sha256(audit_path),
            },
            "manifest": {
                "path": "artifacts/cue_dose_confirmatory_manifest.json",
                "sha256": file_sha256("artifacts/cue_dose_confirmatory_manifest.json"),
            },
            "result": {
                "path": "artifacts/cue_dose_confirmatory_results.json",
                "sha256": file_sha256("artifacts/cue_dose_confirmatory_results.json"),
            },
            "coefficient_rule_seed_sha256": v36_manifest["generation_seeds_sha256"][
                "rules"
            ],
            "model_initialization_seed_sha256": v36_manifest[
                "generation_seeds_sha256"
            ]["model_initializations"],
            "cue_permutation": FIXED_PERMUTATION_ID,
            "cue_permutation_seed_sha256": v36_manifest[
                "generation_seeds_sha256"
            ]["cue_permutations"][0],
        },
        "flags": {
            "post_confirmatory_robustness_extension": True,
            "excluded_from_original_v3_6_p_value": True,
            "not_a_scaling_law_fit": True,
        },
        "factorial_design": {
            "model_scales": 2,
            "coefficient_rules": 2,
            "model_initialization_seeds": 2,
            "conditions": 2,
            "paired_blocks": 8,
            "trajectories": 16,
            "scales_target_core_parameters": list(SCALE_TARGETS),
            "condition_order": list(CONDITION_ORDER),
            "same_initial_weights_within_rule_seed_scale_block": True,
            "same_noncondition_rng_fields_within_rule_seed_scale_block": True,
            "width_only_scaling": True,
        },
        "frozen_protocol": {
            "model": "parent-exposed six-layer Transformer",
            "model_scale_convention": (
                "select the closest width divisible by eight; change width only; "
                "count Transformer blocks and final normalization as N_core"
            ),
            "p": cfg.field.p,
            "eta": cfg.field.eta,
            "objective": "exact_soft_label_cross_entropy",
            "sampled_nll_is_not_an_optimization_objective": True,
            "optimizer": cfg.to_dict()["optimizer"],
            "optimizer_parameterization_identical_at_both_scales": True,
            "scheduler": "absolute_exposure_cosine",
            "cue_bearing_sampling_through_exposure": 5_000_000,
            "uniform_full_support_from_exposure": 5_000_000,
            "maximum_exposure": 10_000_000,
            "zero_step_switch_verification": True,
            "checkpoints": [0, *TRAIN_CHECKPOINTS],
            "probe_checkpoints": list(PROBE_CHECKPOINTS),
            "registered_probes": "unchanged v3.6 layers 0, 3, and 6 linear, fixed Fourier, and fixed-capacity MLP probes",
        },
        "transition_definitions": transition_definitions,
        "primary_robustness_question": {
            "question": (
                "Does R2 advance verified onset timing or onset probability relative "
                "to paired R0 at both new scales?"
            ),
            "analysis_unit": "paired rule-by-initialization block within scale",
            "report_every_paired_block": True,
            "combined_confirmatory_p_value_with_v3_6": False,
            "new_confirmatory_p_value": False,
        },
        "pilot_gate": {
            "pilot_runs_are_registered_cells": True,
            "pilot_run_ids": pilot_run_ids,
            "remaining_12_runs_withheld_until_go": True,
            "forced_resume_checkpoint": 2_000_000,
            "engineering_criteria": [
                "all four pilot cells complete to 10M without OOM, device, or checkpoint errors",
                "all logged losses, gradients, throughputs, and exact evaluation metrics are finite",
                "every pilot checkpoint has a SHA-256 sidecar and the 2M checkpoint is successfully resumed without optimizer or scheduler reset",
                "median throughput is positive and memory telemetry is recorded for both scales",
                "the zero-step 5M sampler switch passes all frozen invariance checks",
            ],
            "identifiability_criteria_per_scale": [
                "at least one of R0 or R2 has verified onset by 10M",
                "R0 and R2 are not both left-censored at the first 2M probe",
                "at least one registered evaluation across the pair has 0.1 < exact full-support R < 0.9",
            ],
            "favorable_cue_effect_required": False,
            "direction_or_size_of_R2_minus_R0_ignored_for_gate": True,
            "failure_action": (
                "stop before the remaining twelve trajectories and report; do not "
                "change architecture, optimizer, objective, checkpoints, or gate"
            ),
        },
        "forbidden": [
            "modification of frozen TMLR, v1.3, or v3.6 artifacts",
            "combined confirmatory p-value with v3.6",
            "scaling-law fit",
            "recurrent bridge",
            "p sweep",
            "sampled-NLL training objective",
            "new architecture",
            "silent architecture or optimizer changes",
            "early-removal continuation without separate authorization and manifest",
        ],
        "runs": [asdict(run) for run in runs],
    }


def write_scale_robustness_manifest(
    output: str | Path = "artifacts/iclr2027_scale_robustness_manifest.json",
    audit_path: str | Path = "artifacts/confirmatory_distribution_audit.json",
) -> tuple[Path, str]:
    path = atomic_write_json(output, scale_robustness_manifest_payload(audit_path))
    checksum = file_sha256(path)
    atomic_write_bytes(
        path.with_suffix(path.suffix + ".sha256"),
        f"{checksum}  {path.name}\n".encode("utf-8"),
    )
    return path, checksum


def find_scale_robustness_run(
    run_id: str,
    audit_path: str | Path = "artifacts/confirmatory_distribution_audit.json",
) -> ScaleRobustnessRun:
    matches = [run for run in build_scale_robustness_runs(audit_path) if run.run_id == run_id]
    if len(matches) != 1:
        raise KeyError(run_id)
    return matches[0]
