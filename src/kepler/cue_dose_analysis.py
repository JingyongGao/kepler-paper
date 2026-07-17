from __future__ import annotations

from collections import Counter
import itertools
import json
from pathlib import Path
from statistics import mean, median
from typing import Any, Iterable

from .cue_dose import PROBE_CHECKPOINTS
from .durability import atomic_write_json
from .environment import file_sha256


ORDERED_DOSE_CONDITIONS = ("C0", "C1", "C2", "C3")
GEOMETRY_CONDITIONS = ("geometry_concentrated", "C2", "geometry_diffuse")


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _summary_records(run_directory: Path) -> list[dict[str, Any]]:
    path = run_directory / "evaluations.jsonl"
    if not path.exists():
        raise FileNotFoundError(path)
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    if len(records) != 12:
        raise ValueError(f"{run_directory.name}: expected 12 evaluations, found {len(records)}")
    return records


def _record_at(
    records: Iterable[dict[str, Any]], exposure: int, phase: str = "standard"
) -> dict[str, Any]:
    matches = [
        record
        for record in records
        if int(record["exposed_pairs"]) == exposure and record["phase"] == phase
    ]
    if len(matches) != 1:
        raise ValueError(f"expected one record for {exposure}:{phase}, found {len(matches)}")
    return matches[0]


def _passes_onset(record: dict[str, Any]) -> bool:
    accuracy = record.get("layer_6_linear_accuracy")
    return (
        float(record["R_full"]) < 0.9
        and accuracy is not None
        and float(accuracy) > 0.5
    )


def _passes_establishment(record: dict[str, Any]) -> bool:
    accuracy = record.get("layer_6_linear_accuracy")
    return (
        float(record["R_full"]) < 0.1
        and float(record["center_accuracy_full"]) > 0.9
        and accuracy is not None
        and float(accuracy) > 0.9
    )


def classify_transition(records: list[dict[str, Any]]) -> dict[str, Any]:
    points = [
        _record_at(
            records,
            exposure,
            "pre_switch" if exposure == 5_000_000 else "standard",
        )
        for exposure in PROBE_CHECKPOINTS
    ]
    onset_index = None
    for index in range(len(points) - 1):
        if _passes_onset(points[index]) and _passes_onset(points[index + 1]):
            onset_index = index
            break
    establishment_index = next(
        (index for index, point in enumerate(points) if _passes_establishment(point)),
        None,
    )
    terminal_ordinal = len(points) + 1
    return {
        "probe_checkpoint_sequence": list(PROBE_CHECKPOINTS),
        "onset_exposure": (
            int(points[onset_index]["exposed_pairs"]) if onset_index is not None else None
        ),
        "onset_ordinal": onset_index + 1 if onset_index is not None else terminal_ordinal,
        "onset_terminal_coded": onset_index is None,
        "onset_by_5M": onset_index is not None
        and int(points[onset_index]["exposed_pairs"]) <= 5_000_000,
        "establishment_exposure": (
            int(points[establishment_index]["exposed_pairs"])
            if establishment_index is not None
            else None
        ),
        "establishment_by_5M": establishment_index is not None
        and int(points[establishment_index]["exposed_pairs"]) <= 5_000_000,
        "threshold_state_at_5M": _passes_onset(points[3]),
        "threshold_state_at_6M": _passes_onset(points[4]),
        "threshold_state_at_10M": _passes_onset(points[6]),
        "ten_M_cannot_initiate_onset": True,
    }


def _rank2_earlier_is_higher(values: list[int]) -> list[int]:
    """Return twice the within-block midrank, with earlier onset ranked higher."""
    ordered = sorted(values, reverse=True)
    output = []
    for value in values:
        positions = [index + 1 for index, item in enumerate(ordered) if item == value]
        output.append(positions[0] + positions[-1])
    return output


def exact_blocked_page_test(
    onset_ordinals_by_seed: dict[int, dict[str, int]],
    conditions: tuple[str, ...] = ORDERED_DOSE_CONDITIONS,
) -> dict[str, Any]:
    """Exact within-seed label-permutation Page-style ordered trend test.

    Ranks are doubled integers so tied midranks are represented exactly. A larger
    statistic means earlier transition ranks align more strongly with increasing
    registered dose order.
    """
    block_ranks2: list[list[int]] = []
    for seed in sorted(onset_ordinals_by_seed):
        values = [int(onset_ordinals_by_seed[seed][condition]) for condition in conditions]
        block_ranks2.append(_rank2_earlier_is_higher(values))
    if not block_ranks2:
        raise ValueError("at least one paired seed block is required")
    weights = tuple(range(1, len(conditions) + 1))
    observed2 = sum(
        sum(weight * rank for weight, rank in zip(weights, ranks, strict=True))
        for ranks in block_ranks2
    )
    distribution = Counter({0: 1})
    for ranks in block_ranks2:
        block = Counter(
            sum(weight * rank for weight, rank in zip(weights, permutation, strict=True))
            for permutation in itertools.permutations(ranks)
        )
        combined: Counter[int] = Counter()
        for total, count in distribution.items():
            for contribution, block_count in block.items():
                combined[total + contribution] += count * block_count
        distribution = combined
    permutations = sum(distribution.values())
    upper = sum(count for statistic, count in distribution.items() if statistic >= observed2)
    return {
        "test": "exact_blocked_Page_style_within_seed_label_permutation",
        "alternative": "earlier_onset_with_increasing_registered_C_single",
        "conditions_in_registered_order": list(conditions),
        "seed_blocks": len(block_ranks2),
        "tie_handling": "within-seed midranks represented as doubled integers",
        "observed_statistic": observed2 / 2.0,
        "observed_statistic_doubled_integer": observed2,
        "exact_permutation_count": permutations,
        "upper_tail_count": upper,
        "exact_one_sided_p": upper / permutations,
        "block_ranks": [[rank / 2.0 for rank in ranks] for ranks in block_ranks2],
    }


def _numeric_summary(values: Iterable[float]) -> dict[str, Any]:
    materialized = [float(value) for value in values]
    return {
        "values_by_seed_order": materialized,
        "mean": mean(materialized),
        "median": median(materialized),
        "minimum": min(materialized),
        "maximum": max(materialized),
    }


def _condition_summary(cells: list[dict[str, Any]]) -> dict[str, Any]:
    transitions = [cell["transition"] for cell in cells]
    r5 = [cell["R_5M_pre"] for cell in cells]
    r10 = [cell["R_10M"] for cell in cells]
    pre_pass = [transition["threshold_state_at_5M"] for transition in transitions]
    return {
        "seeds": [cell["seed"] for cell in cells],
        "fraction_onset_by_5M": mean(
            float(transition["onset_by_5M"]) for transition in transitions
        ),
        "fraction_onset_verified_by_10M": mean(
            float(not transition["onset_terminal_coded"]) for transition in transitions
        ),
        "onset_exposure_by_seed": [
            transition["onset_exposure"] for transition in transitions
        ],
        "onset_terminal_ordinal_by_seed": [
            transition["onset_ordinal"] for transition in transitions
        ],
        "fraction_established_by_5M": mean(
            float(transition["establishment_by_5M"]) for transition in transitions
        ),
        "fraction_established_by_10M": mean(
            float(transition["establishment_exposure"] is not None)
            for transition in transitions
        ),
        "establishment_exposure_by_seed": [
            transition["establishment_exposure"] for transition in transitions
        ],
        "R_5M_pre": _numeric_summary(r5),
        "R_10M": _numeric_summary(r10),
        "R_change_5M_to_10M": _numeric_summary(
            cell["R_10M"] - cell["R_5M_pre"] for cell in cells
        ),
        "post_switch_threshold_retention": {
            "pre_switch_threshold_positive_seeds": sum(pre_pass),
            "fraction_of_pre_positive_retained_at_6M": (
                mean(
                    float(transition["threshold_state_at_6M"])
                    for transition, selected in zip(transitions, pre_pass, strict=True)
                    if selected
                )
                if any(pre_pass)
                else None
            ),
            "fraction_of_pre_positive_retained_at_10M": (
                mean(
                    float(transition["threshold_state_at_10M"])
                    for transition, selected in zip(transitions, pre_pass, strict=True)
                    if selected
                )
                if any(pre_pass)
                else None
            ),
        },
    }


def _paired_delta(
    cells: dict[str, dict[int, dict[str, Any]]], left: str, right: str
) -> dict[str, Any]:
    seeds = sorted(set(cells[left]) & set(cells[right]))
    return {
        "comparison": f"{right}_minus_{left}",
        "seeds": seeds,
        "R_5M_pre_delta": _numeric_summary(
            cells[right][seed]["R_5M_pre"] - cells[left][seed]["R_5M_pre"]
            for seed in seeds
        ),
        "R_10M_delta": _numeric_summary(
            cells[right][seed]["R_10M"] - cells[left][seed]["R_10M"]
            for seed in seeds
        ),
        "onset_terminal_ordinal_delta": _numeric_summary(
            cells[right][seed]["transition"]["onset_ordinal"]
            - cells[left][seed]["transition"]["onset_ordinal"]
            for seed in seeds
        ),
    }


def _gradient_summaries(run_directory: Path) -> list[dict[str, Any]]:
    output = []
    for path in sorted(run_directory.glob("*.cue.gradient.json")):
        audit = _read_json(path)
        global_metrics = audit["groups"]["global.total"]
        output.append(
            {
                "exposed_pairs": int(audit["exposed_pairs"]),
                "phase": audit["phase"],
                "norm_of_mean_gradient": global_metrics["norm_of_mean_gradient"],
                "gradient_variance": global_metrics[
                    "gradient_variance_mean_squared_deviation"
                ],
                "gradient_snr": global_metrics["gradient_snr"],
                "mean_pairwise_gradient_cosine": global_metrics[
                    "mean_pairwise_gradient_cosine"
                ],
                "groups": audit["groups"],
                "metric_definitions": audit["metric_definitions"],
                "source": str(path),
                "sha256": file_sha256(path),
            }
        )
    if len(output) != 5:
        raise ValueError(f"{run_directory.name}: expected 5 gradient audits, found {len(output)}")
    return output


def _checkpoint_metrics(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
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


def analyze_cue_dose(
    run_root: str | Path = "artifacts/cue_dose_runs",
    manifest_path: str | Path = "artifacts/cue_dose_manifest.json",
    output_path: str | Path = "artifacts/cue_dose_results.json",
) -> dict[str, Any]:
    manifest_file = Path(manifest_path)
    manifest = _read_json(manifest_file)
    cells_by_condition: dict[str, dict[int, dict[str, Any]]] = {}
    all_cells = []
    switch_checks = []
    for registered in manifest["runs"]:
        run_directory = Path(run_root) / registered["run_id"]
        records = _summary_records(run_directory)
        transition = classify_transition(records)
        r5 = _record_at(records, 5_000_000, "pre_switch")
        r10 = _record_at(records, 10_000_000)
        switch = _read_json(run_directory / "switch_invariance.json")
        if not switch["passed"]:
            raise AssertionError(f"switch invariance failed for {registered['run_id']}")
        cell = {
            "run_id": registered["run_id"],
            "condition_id": registered["condition_id"],
            "seed": int(registered["model_initialization_seed"]),
            "m": int(registered["m"]),
            "target_singleton_cue_index": registered["target_singleton_cue_index"],
            "lambda": registered["lambda_value"],
            "transition": transition,
            "R_5M_pre": float(r5["R_full"]),
            "R_5M_post": float(
                _record_at(records, 5_000_000, "post_switch")["R_full"]
            ),
            "R_10M": float(r10["R_full"]),
            "center_accuracy_5M_pre": float(r5["center_accuracy_full"]),
            "center_accuracy_10M": float(r10["center_accuracy_full"]),
            "layer_6_linear_accuracy_5M_pre": float(r5["layer_6_linear_accuracy"]),
            "layer_6_linear_accuracy_10M": float(r10["layer_6_linear_accuracy"]),
            "checkpoint_metrics": _checkpoint_metrics(records),
            "gradient_global_metrics": _gradient_summaries(run_directory),
            "evaluations_jsonl_sha256": file_sha256(run_directory / "evaluations.jsonl"),
        }
        all_cells.append(cell)
        cells_by_condition.setdefault(cell["condition_id"], {})[cell["seed"]] = cell
        switch_checks.append(
            {
                "run_id": cell["run_id"],
                "passed": switch["passed"],
                "model_state_unchanged": switch["model_state_unchanged"],
                "optimizer_state_unchanged": switch["optimizer_state_unchanged"],
                "full_support_output_sha256_equal": switch[
                    "full_support_output_sha256_equal"
                ],
                "sha256": file_sha256(run_directory / "switch_invariance.json"),
            }
        )
    if len(all_cells) != 30 or any(len(cells) != 5 for cells in cells_by_condition.values()):
        raise ValueError("analysis requires all 30 registered cells and five seeds per condition")

    ordinal_blocks = {
        seed: {
            condition: cells_by_condition[condition][seed]["transition"]["onset_ordinal"]
            for condition in ORDERED_DOSE_CONDITIONS
        }
        for seed in range(5)
    }
    condition_summaries = {
        condition: _condition_summary([cells_by_condition[condition][seed] for seed in range(5)])
        for condition in cells_by_condition
    }
    result = {
        "project": "Project Kepler v3.5",
        "kind": "full_support_cue_dose_causal_gate_results",
        "manifest": {"path": str(manifest_file), "sha256": file_sha256(manifest_file)},
        "analysis_definitions": {
            "thresholds_changed_after_observation": False,
            "onset": manifest["transition_definitions"]["onset"],
            "establishment": manifest["transition_definitions"]["establishment"],
            "terminal_onset_rule": (
                "10M cannot initiate onset because no next registered probe checkpoint "
                "exists to verify retention; no onset is coded as ordinal 8"
            ),
            "R_switch_reference": "5M pre-switch exact full-support evaluation",
        },
        "integrity": {
            "registered_cells": 30,
            "completed_cells": len(all_cells),
            "all_switch_invariance_checks_passed": all(item["passed"] for item in switch_checks),
            "switch_checks": switch_checks,
        },
        "condition_summaries": condition_summaries,
        "paired_dose_comparisons_against_C0": {
            condition: _paired_delta(cells_by_condition, "C0", condition)
            for condition in ("C1", "C2", "C3")
        },
        "ordered_dose_trend": {
            "onset_ordinals_by_seed": ordinal_blocks,
            **exact_blocked_page_test(ordinal_blocks),
        },
        "matched_singleton_index_geometry": {
            "conditions": list(GEOMETRY_CONDITIONS),
            "comparisons_against_m64_C2": {
                condition: _paired_delta(cells_by_condition, "C2", condition)
                for condition in ("geometry_concentrated", "geometry_diffuse")
            },
        },
        "cells": all_cells,
        "interpretation_limits": [
            "gradient statistics are secondary correlates and are not causal outcomes",
            "C_single is not an additive percentage of unique information",
            "singleton mutual information alone is not called causal without joint dose and geometry support",
            "all cells are exploratory, diagnostic only, and excluded from confirmatory scaling fits",
        ],
        "stop_condition": {
            "maximum_pair_exposure": 10_000_000,
            "production_matrix_launched": False,
            "recurrent_bridge_launched": False,
            "field_sweep_launched": False,
            "Fourier_initialization_run": False,
        },
    }
    atomic_write_json(output_path, result)
    return result
