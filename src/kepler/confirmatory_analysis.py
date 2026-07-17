from __future__ import annotations

import json
from pathlib import Path
from statistics import mean, median
from typing import Any, Iterable

from .cue_dose_analysis import classify_transition, exact_blocked_page_test
from .durability import atomic_write_json
from .environment import file_sha256


CONDITION_ORDER = ("R0", "R1", "R2")


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _summary_records(run_directory: Path) -> list[dict[str, Any]]:
    path = run_directory / "evaluations.jsonl"
    if not path.exists():
        raise FileNotFoundError(path)
    records = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(records) != 12:
        raise ValueError(
            f"{run_directory.name}: expected 12 evaluations, found {len(records)}"
        )
    return records


def _record_at(
    records: Iterable[dict[str, Any]],
    exposure: int,
    phase: str = "standard",
) -> dict[str, Any]:
    matches = [
        record
        for record in records
        if int(record["exposed_pairs"]) == exposure and record["phase"] == phase
    ]
    if len(matches) != 1:
        raise ValueError(
            f"expected one record for {exposure}:{phase}, found {len(matches)}"
        )
    return matches[0]


def _numeric_summary(values: Iterable[float]) -> dict[str, Any]:
    materialized = [float(value) for value in values]
    return {
        "values_in_block_order": materialized,
        "mean": mean(materialized),
        "median": median(materialized),
        "minimum": min(materialized),
        "maximum": max(materialized),
    }


def _condition_summary(cells: list[dict[str, Any]]) -> dict[str, Any]:
    transitions = [cell["transition"] for cell in cells]
    pre_positive = [
        transition["threshold_state_at_5M"] for transition in transitions
    ]
    return {
        "blocks": [cell["block_id"] for cell in cells],
        "fraction_onset_by_5M": mean(
            float(transition["onset_by_5M"]) for transition in transitions
        ),
        "fraction_onset_verified_by_10M": mean(
            float(not transition["onset_terminal_coded"])
            for transition in transitions
        ),
        "onset_exposure_by_block": [
            transition["onset_exposure"] for transition in transitions
        ],
        "onset_terminal_ordinal_by_block": [
            transition["onset_ordinal"] for transition in transitions
        ],
        "fraction_established_by_5M": mean(
            float(transition["establishment_by_5M"])
            for transition in transitions
        ),
        "fraction_established_by_10M": mean(
            float(transition["establishment_exposure"] is not None)
            for transition in transitions
        ),
        "establishment_exposure_by_block": [
            transition["establishment_exposure"] for transition in transitions
        ],
        "R_5M_pre": _numeric_summary(
            cell["R_5M_pre"] for cell in cells
        ),
        "R_10M": _numeric_summary(cell["R_10M"] for cell in cells),
        "R_change_5M_to_10M": _numeric_summary(
            cell["R_10M"] - cell["R_5M_pre"] for cell in cells
        ),
        "post_switch_threshold_retention": {
            "pre_switch_threshold_positive_blocks": sum(pre_positive),
            "fraction_of_pre_positive_retained_at_6M": (
                mean(
                    float(transition["threshold_state_at_6M"])
                    for transition, selected in zip(
                        transitions, pre_positive, strict=True
                    )
                    if selected
                )
                if any(pre_positive)
                else None
            ),
            "fraction_of_pre_positive_retained_at_10M": (
                mean(
                    float(transition["threshold_state_at_10M"])
                    for transition, selected in zip(
                        transitions, pre_positive, strict=True
                    )
                    if selected
                )
                if any(pre_positive)
                else None
            ),
        },
    }


def _paired_delta(
    cells_by_block: dict[str, dict[str, dict[str, Any]]],
    left: str,
    right: str,
    selected_blocks: list[str],
) -> dict[str, Any]:
    return {
        "comparison": f"{right}_minus_{left}",
        "blocks": selected_blocks,
        "R_5M_pre_delta": _numeric_summary(
            cells_by_block[block][right]["R_5M_pre"]
            - cells_by_block[block][left]["R_5M_pre"]
            for block in selected_blocks
        ),
        "R_10M_delta": _numeric_summary(
            cells_by_block[block][right]["R_10M"]
            - cells_by_block[block][left]["R_10M"]
            for block in selected_blocks
        ),
        "onset_terminal_ordinal_delta": _numeric_summary(
            cells_by_block[block][right]["transition"]["onset_ordinal"]
            - cells_by_block[block][left]["transition"]["onset_ordinal"]
            for block in selected_blocks
        ),
    }


def _checkpoint_metrics(
    records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    output = []
    for summary in records:
        path = Path(summary["evaluation_file"])
        evaluation = _read_json(path)
        metrics = evaluation["full_support"]
        output.append(
            {
                "exposed_pairs": int(evaluation["exposed_pairs"]),
                "phase": evaluation["phase"],
                "mean_exact_conditional_kl_nats": metrics[
                    "mean_exact_conditional_kl_nats"
                ],
                "R": metrics["R"],
                "R_bootstrap_95_ci": metrics["R_bootstrap_95_ci"],
                "bootstrap_replicates": metrics["bootstrap_replicates"],
                "bootstrap_unit": metrics["bootstrap_unit"],
                "center_state_top1_accuracy": metrics[
                    "center_state_top1_accuracy"
                ],
                "mean_probability_on_center_state": metrics[
                    "mean_probability_on_center_state"
                ],
                "mean_predictive_entropy_nats": metrics[
                    "mean_predictive_entropy_nats"
                ],
                "probe_metrics": metrics.get("probe_metrics"),
                "checkpoint_training_diagnostics": evaluation[
                    "checkpoint_training_diagnostics"
                ],
                "evaluation_file": str(path),
                "evaluation_file_sha256": file_sha256(path),
                "checkpoint": evaluation["checkpoint"],
                "checkpoint_sha256": evaluation["checkpoint_sha256"],
                "model_output_sha256": evaluation["model_output_sha256"],
            }
        )
    return output


def _stratum_summary(
    cells_by_block: dict[str, dict[str, dict[str, Any]]],
    block_metadata: dict[str, dict[str, Any]],
    blocks: list[str],
) -> dict[str, Any]:
    ordinal_blocks = {
        index: {
            condition: cells_by_block[block][condition]["transition"][
                "onset_ordinal"
            ]
            for condition in CONDITION_ORDER
        }
        for index, block in enumerate(blocks)
    }
    return {
        "blocks": blocks,
        "block_metadata": [block_metadata[block] for block in blocks],
        "condition_summaries": {
            condition: _condition_summary(
                [cells_by_block[block][condition] for block in blocks]
            )
            for condition in CONDITION_ORDER
        },
        "paired_differences": {
            "R1_vs_R0": _paired_delta(
                cells_by_block, "R0", "R1", blocks
            ),
            "R2_vs_R1": _paired_delta(
                cells_by_block, "R1", "R2", blocks
            ),
            "R2_vs_R0": _paired_delta(
                cells_by_block, "R0", "R2", blocks
            ),
        },
        "ordered_onset_test": {
            "onset_ordinals_by_block": {
                block: {
                    condition: cells_by_block[block][condition][
                        "transition"
                    ]["onset_ordinal"]
                    for condition in CONDITION_ORDER
                }
                for block in blocks
            },
            **exact_blocked_page_test(ordinal_blocks, CONDITION_ORDER),
        },
    }


def analyze_confirmatory_replication(
    run_root: str | Path = "artifacts/confirmatory_runs",
    manifest_path: str | Path = "artifacts/cue_dose_confirmatory_manifest.json",
    output_path: str | Path = "artifacts/cue_dose_confirmatory_results.json",
) -> dict[str, Any]:
    manifest_file = Path(manifest_path)
    manifest = _read_json(manifest_file)
    root = Path(run_root)
    all_cells: list[dict[str, Any]] = []
    cells_by_block: dict[str, dict[str, dict[str, Any]]] = {}
    block_metadata: dict[str, dict[str, Any]] = {}
    switch_checks = []
    for registered in manifest["runs"]:
        directory = root / registered["run_id"]
        records = _summary_records(directory)
        transition = classify_transition(records)
        r5 = _record_at(records, 5_000_000, "pre_switch")
        r10 = _record_at(records, 10_000_000)
        switch_path = directory / "switch_invariance.json"
        switch = _read_json(switch_path)
        if not switch["passed"]:
            raise AssertionError(
                f"switch invariance failed for {registered['run_id']}"
            )
        gradient_files = list(directory.glob("*.gradient.json"))
        if gradient_files:
            raise AssertionError(
                f"unregistered gradient audit found for {registered['run_id']}"
            )
        cell = {
            "run_id": registered["run_id"],
            "block_id": registered["block_id"],
            "rule_id": registered["rule_id"],
            "coefficients": registered["coefficients"],
            "permutation_id": registered["permutation_id"],
            "model_seed_index": registered["model_seed_index"],
            "condition_id": registered["condition_id"],
            "target_singleton_cue_index": registered[
                "target_singleton_cue_index"
            ],
            "lambda": registered["lambda_value"],
            "transition": transition,
            "R_5M_pre": float(r5["R_full"]),
            "R_5M_post": float(
                _record_at(records, 5_000_000, "post_switch")["R_full"]
            ),
            "R_10M": float(r10["R_full"]),
            "center_accuracy_5M_pre": float(
                r5["center_accuracy_full"]
            ),
            "center_accuracy_10M": float(r10["center_accuracy_full"]),
            "layer_6_linear_accuracy_5M_pre": float(
                r5["layer_6_linear_accuracy"]
            ),
            "layer_6_linear_accuracy_10M": float(
                r10["layer_6_linear_accuracy"]
            ),
            "checkpoint_metrics": _checkpoint_metrics(records),
            "evaluations_jsonl_sha256": file_sha256(
                directory / "evaluations.jsonl"
            ),
        }
        all_cells.append(cell)
        cells_by_block.setdefault(cell["block_id"], {})[
            cell["condition_id"]
        ] = cell
        block_metadata[cell["block_id"]] = {
            "block_id": cell["block_id"],
            "rule_id": cell["rule_id"],
            "coefficients": cell["coefficients"],
            "permutation_id": cell["permutation_id"],
            "model_seed_index": cell["model_seed_index"],
        }
        switch_checks.append(
            {
                "run_id": cell["run_id"],
                "passed": switch["passed"],
                "model_state_unchanged": switch[
                    "model_state_unchanged"
                ],
                "optimizer_state_unchanged": switch[
                    "optimizer_state_unchanged"
                ],
                "full_support_output_sha256_equal": switch[
                    "full_support_output_sha256_equal"
                ],
                "all_rng_state_checks_passed": all(
                    switch[key]
                    for key in (
                        "numpy_rng_state_unchanged",
                        "torch_cpu_rng_state_restored",
                        "torch_mps_rng_state_restored",
                        "python_rng_state_restored",
                    )
                ),
                "gradient_audit_run": switch["gradient_audit_run"],
                "sha256": file_sha256(switch_path),
            }
        )
    if len(all_cells) != 24 or len(cells_by_block) != 8:
        raise ValueError("confirmatory analysis requires all 24 cells and 8 blocks")
    if any(set(cells) != set(CONDITION_ORDER) for cells in cells_by_block.values()):
        raise ValueError("each confirmatory block must contain R0, R1, and R2")
    blocks = [
        registered["block_id"]
        for registered in manifest["runs"]
        if registered["condition_id"] == "R0"
    ]
    primary = _stratum_summary(
        cells_by_block, block_metadata, blocks
    )
    if primary["ordered_onset_test"]["exact_permutation_count"] != 6**8:
        raise AssertionError("primary exact test did not enumerate (3!)^8 assignments")
    p_value = primary["ordered_onset_test"]["exact_one_sided_p"]
    result = {
        "project": "Project Kepler v3.6",
        "kind": "confirmatory_cross_rule_cue_dose_replication_results",
        "manifest": {
            "path": str(manifest_file),
            "sha256": file_sha256(manifest_file),
        },
        "distribution_audit": manifest["distribution_audit"],
        "analysis_definitions": {
            "thresholds_changed_after_observation": False,
            "onset": manifest["transition_definitions"]["onset"],
            "establishment": manifest["transition_definitions"][
                "establishment"
            ],
            "terminal_onset_ordinal": 8,
            "ten_M_cannot_initiate_onset": True,
            "v3_5_excluded_from_primary_p_value": True,
        },
        "integrity": {
            "registered_cells": 24,
            "completed_cells": len(all_cells),
            "registered_blocks": 8,
            "complete_blocks": len(cells_by_block),
            "all_switch_invariance_checks_passed": all(
                item["passed"] for item in switch_checks
            ),
            "gradient_audits_run": False,
            "switch_checks": switch_checks,
        },
        "primary_confirmatory_test": {
            **primary,
            "confirmatory_alpha": 0.05,
            "statistically_confirmatory": p_value < 0.05,
            "v3_5_combined_in_primary": False,
        },
        "by_coefficient_rule": {
            rule_id: _stratum_summary(
                cells_by_block,
                block_metadata,
                [
                    block
                    for block in blocks
                    if block_metadata[block]["rule_id"] == rule_id
                ],
            )
            for rule_id in ("rule_0", "rule_1")
        },
        "by_cue_permutation": {
            permutation_id: _stratum_summary(
                cells_by_block,
                block_metadata,
                [
                    block
                    for block in blocks
                    if block_metadata[block]["permutation_id"]
                    == permutation_id
                ],
            )
            for permutation_id in ("permutation_0", "permutation_1")
        },
        "cells": all_cells,
        "interpretation_limits": [
            "singleton mutual information alone is not identified as the causal variable",
            "the result is not a universal Transformer capability or impossibility claim",
            "v3.5 is not combined into the primary p-value",
            "all trajectories are excluded from the original scaling fit",
        ],
        "stop_condition": {
            "maximum_pair_exposure": 10_000_000,
            "production_matrix_launched": False,
            "recurrent_bridge_launched": False,
            "model_size_increased": False,
        },
    }
    atomic_write_json(output_path, result)
    return result
