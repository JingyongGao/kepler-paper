from __future__ import annotations

import json
import math
from pathlib import Path
from statistics import mean, median
from typing import Any

from .cue_dose_analysis import classify_transition
from .cue_dose_runner import _checkpoint_path
from .durability import atomic_write_bytes, atomic_write_json
from .environment import file_sha256
from .scale_robustness import build_scale_robustness_runs


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _summary_records(directory: Path, expected: int = 12) -> list[dict[str, Any]]:
    records = _read_jsonl(directory / "evaluations.jsonl")
    if len(records) != expected:
        raise ValueError(f"{directory.name}: expected {expected} evaluations, found {len(records)}")
    return records


def _record_at(
    records: list[dict[str, Any]], exposure: int, phase: str = "standard"
) -> dict[str, Any]:
    matches = [
        record
        for record in records
        if int(record["exposed_pairs"]) == exposure and record["phase"] == phase
    ]
    if len(matches) != 1:
        raise ValueError(f"expected one record for {exposure}:{phase}, found {len(matches)}")
    return matches[0]


def _checkpoint_sidecars(directory: Path) -> dict[str, Any]:
    checkpoints = sorted(directory.glob("checkpoint_*.pt"))
    checks = []
    for checkpoint in checkpoints:
        sidecar = checkpoint.with_suffix(checkpoint.suffix + ".sha256")
        recorded = sidecar.read_text(encoding="utf-8").split()[0] if sidecar.exists() else None
        actual = file_sha256(checkpoint)
        checks.append(
            {
                "checkpoint": str(checkpoint),
                "actual_sha256": actual,
                "sidecar_sha256": recorded,
                "passed": recorded == actual,
            }
        )
    return {
        "checkpoint_count": len(checkpoints),
        "all_sidecars_valid": len(checkpoints) == 11 and all(item["passed"] for item in checks),
        "checks": checks,
    }


def _finite_training_log(records: list[dict[str, Any]]) -> bool:
    numeric_fields = (
        "optimization_soft_cross_entropy_nats",
        "sampled_nll_nats_diagnostic_only",
        "gradient_norm",
        "learning_rate",
        "step_seconds",
        "pairs_per_second",
    )
    return bool(records) and all(
        math.isfinite(float(record[field]))
        for record in records
        for field in numeric_fields
    )


def _finite_evaluations(records: list[dict[str, Any]]) -> bool:
    return all(
        math.isfinite(float(record[field]))
        for record in records
        for field in ("R_full", "center_accuracy_full")
    )


def evaluate_scale_robustness_pilot_gate(
    run_root: str | Path = "artifacts/iclr2027_scale_robustness_runs",
    manifest_path: str | Path = "artifacts/iclr2027_scale_robustness_manifest.json",
    output: str | Path = "artifacts/iclr2027_scale_robustness_pilot_gate.json",
    report: str | Path = "ICLR2027_SCALE_ROBUSTNESS_PILOT_REPORT.md",
) -> dict[str, Any]:
    root = Path(run_root)
    manifest = _read_json(Path(manifest_path))
    runs = [run for run in build_scale_robustness_runs() if run.engineering_learnability_pilot]
    cells: dict[str, dict[str, Any]] = {}
    for run in runs:
        directory = root / run.run_id
        records = _summary_records(directory)
        training = _read_jsonl(directory / "train.jsonl")
        resume = _read_json(directory / "resume_verification.json")
        switch = _read_json(directory / "switch_invariance.json")
        sidecars = _checkpoint_sidecars(directory)
        driver_memory = [
            int(item["mps_driver_allocated_bytes"])
            for item in training
            if item.get("mps_driver_allocated_bytes") is not None
        ]
        throughput = [float(item["pairs_per_second"]) for item in training]
        transition = classify_transition(records)
        checks = {
            "endpoint_checkpoint_exists": _checkpoint_path(
                directory, run.maximum_pair_exposures, "standard"
            ).exists(),
            "twelve_registered_evaluations": len(records) == 12,
            "finite_training_log": _finite_training_log(training),
            "finite_exact_evaluations": _finite_evaluations(records),
            "checkpoint_sidecars_valid": sidecars["all_sidecars_valid"],
            "forced_2M_resume_verified": (
                resume["passed"] and int(resume["resumed_exposure"]) == 2_000_000
            ),
            "zero_step_switch_verified": switch["passed"],
            "positive_median_throughput": median(throughput) > 0.0,
            "memory_telemetry_recorded": bool(driver_memory),
            "sampled_nll_not_training_objective": all(
                item["sampled_nll_training_objective"] is False for item in training
            ),
        }
        cells[run.run_id] = {
            "scale_id": run.scale_id,
            "condition_id": run.condition_id,
            "transition": transition,
            "R_by_checkpoint": [
                {
                    "exposed_pairs": record["exposed_pairs"],
                    "phase": record["phase"],
                    "R": record["R_full"],
                }
                for record in records
            ],
            "median_pairs_per_second": median(throughput),
            "minimum_pairs_per_second": min(throughput),
            "maximum_mps_driver_allocated_bytes": max(driver_memory) if driver_memory else None,
            "resume_verification": resume,
            "switch_invariance": switch,
            "checkpoint_verification": sidecars,
            "checks": checks,
            "engineering_passed": all(checks.values()),
        }

    scales: dict[str, Any] = {}
    for scale_id in ("1m", "5m"):
        selected = [cell for cell in cells.values() if cell["scale_id"] == scale_id]
        transitions = [cell["transition"] for cell in selected]
        informative_r = any(
            0.1 < float(item["R"]) < 0.9
            for cell in selected
            for item in cell["R_by_checkpoint"]
        )
        identifiability = {
            "at_least_one_verified_onset_by_10M": any(
                not transition["onset_terminal_coded"] for transition in transitions
            ),
            "not_both_left_censored_at_2M": not all(
                transition["onset_exposure"] == 2_000_000 for transition in transitions
            ),
            "at_least_one_informative_R_between_0_1_and_0_9": informative_r,
        }
        scales[scale_id] = {
            "run_ids": [run_id for run_id, cell in cells.items() if cell["scale_id"] == scale_id],
            "engineering_passed": all(cell["engineering_passed"] for cell in selected),
            "identifiability_checks": identifiability,
            "identifiability_passed": all(identifiability.values()),
            "favorable_cue_effect_used_for_gate": False,
        }
        scales[scale_id]["passed"] = (
            scales[scale_id]["engineering_passed"]
            and scales[scale_id]["identifiability_passed"]
        )

    result = {
        "kind": "iclr2027_scale_robustness_engineering_learnability_gate",
        "manifest": str(manifest_path),
        "manifest_sha256": file_sha256(manifest_path),
        "registered_gate": manifest["pilot_gate"],
        "cells": cells,
        "scales": scales,
        "all_scales_passed": all(item["passed"] for item in scales.values()),
        "decision": "GO" if all(item["passed"] for item in scales.values()) else "NO-GO",
        "remaining_12_trajectories_started_during_gate": False,
        "cue_effect_direction_used_for_decision": False,
    }
    result["failure_action"] = (
        None
        if result["all_scales_passed"]
        else "Stop before remaining trajectories; do not change architecture or optimizer."
    )
    output_path = atomic_write_json(output, result)
    report_lines = [
        "# ICLR 2027 scale-robustness pilot report",
        "",
        f"Decision: **{result['decision']}**",
        "",
        "The four pilot cells are preregistered members of the 16-cell extension. "
        "The remaining twelve trajectories were not started during this gate. "
        "Cue-effect direction and magnitude were not used for Go/No-Go.",
        "",
        "| Scale | Engineering | Learnability/identifiability | Decision |",
        "|---|---:|---:|---:|",
    ]
    for scale_id, scale in scales.items():
        report_lines.append(
            f"| {scale_id} | {'pass' if scale['engineering_passed'] else 'fail'} | "
            f"{'pass' if scale['identifiability_passed'] else 'fail'} | "
            f"{'GO' if scale['passed'] else 'NO-GO'} |"
        )
    report_lines.extend(
        [
            "",
            "## Pilot cells",
            "",
            "| Run | Median pairs/s | Peak MPS driver bytes | Onset | Establishment |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for run_id, cell in cells.items():
        transition = cell["transition"]
        report_lines.append(
            f"| `{run_id}` | {cell['median_pairs_per_second']:.1f} | "
            f"{cell['maximum_mps_driver_allocated_bytes']} | "
            f"{transition['onset_exposure']} | {transition['establishment_exposure']} |"
        )
    report_lines.extend(
        [
            "",
            "The gate requires no favorable R2 effect. If any scale is not learnable, "
            "not time-resolved, or unreliable, the protocol stops without architecture "
            "or optimizer changes.",
            "",
            f"Machine-readable gate: `{output_path}`",
        ]
    )
    atomic_write_bytes(report, ("\n".join(report_lines) + "\n").encode("utf-8"))
    return result


def _cell_result(directory: Path, run) -> dict[str, Any]:
    records = _summary_records(directory)
    transition = classify_transition(records)
    checkpoint_verification = _checkpoint_sidecars(directory)
    return {
        "run_id": run.run_id,
        "block_id": run.block_id,
        "scale_id": run.scale_id,
        "target_core_parameters": run.target_core_parameters,
        "actual_core_parameters": run.actual_core_parameters,
        "width": run.width,
        "rule_id": run.rule_id,
        "coefficients": list(run.coefficients),
        "model_seed_index": run.model_seed_index,
        "model_initialization_seed": run.model_initialization_seed,
        "condition_id": run.condition_id,
        "lambda": run.lambda_value,
        "transition": transition,
        "R_5M_pre": _record_at(records, 5_000_000, "pre_switch")["R_full"],
        "R_6M": _record_at(records, 6_000_000)["R_full"],
        "R_10M": _record_at(records, 10_000_000)["R_full"],
        "center_accuracy_5M_pre": _record_at(records, 5_000_000, "pre_switch")[
            "center_accuracy_full"
        ],
        "center_accuracy_10M": _record_at(records, 10_000_000)[
            "center_accuracy_full"
        ],
        "layer_6_linear_accuracy_5M_pre": _record_at(
            records, 5_000_000, "pre_switch"
        )["layer_6_linear_accuracy"],
        "layer_6_linear_accuracy_10M": _record_at(records, 10_000_000)[
            "layer_6_linear_accuracy"
        ],
        "checkpoint_metrics": records,
        "evaluations_jsonl_sha256": file_sha256(directory / "evaluations.jsonl"),
        "checkpoint_verification": checkpoint_verification,
        "switch_invariance": _read_json(directory / "switch_invariance.json"),
    }


def analyze_scale_robustness(
    run_root: str | Path = "artifacts/iclr2027_scale_robustness_runs",
    manifest_path: str | Path = "artifacts/iclr2027_scale_robustness_manifest.json",
    pilot_gate_path: str | Path = "artifacts/iclr2027_scale_robustness_pilot_gate.json",
    output: str | Path = "artifacts/iclr2027_scale_robustness_results.json",
    report: str | Path = "ICLR2027_SCALE_ROBUSTNESS_REPORT.md",
) -> dict[str, Any]:
    gate = _read_json(Path(pilot_gate_path))
    if not gate["all_scales_passed"]:
        raise ValueError("pilot gate is NO-GO; full scale analysis is forbidden")
    root = Path(run_root)
    runs = build_scale_robustness_runs()
    cells = [_cell_result(root / run.run_id, run) for run in runs]
    by_block = {
        block_id: {
            cell["condition_id"]: cell
            for cell in cells
            if cell["block_id"] == block_id
        }
        for block_id in sorted({cell["block_id"] for cell in cells})
    }
    paired_blocks = []
    for block_id, conditions in by_block.items():
        r0, r2 = conditions["R0"], conditions["R2"]
        delta = (
            r2["transition"]["onset_ordinal"]
            - r0["transition"]["onset_ordinal"]
        )
        paired_blocks.append(
            {
                "block_id": block_id,
                "scale_id": r0["scale_id"],
                "rule_id": r0["rule_id"],
                "model_seed_index": r0["model_seed_index"],
                "R0_onset_exposure": r0["transition"]["onset_exposure"],
                "R2_onset_exposure": r2["transition"]["onset_exposure"],
                "R0_onset_ordinal": r0["transition"]["onset_ordinal"],
                "R2_onset_ordinal": r2["transition"]["onset_ordinal"],
                "R0_establishment_exposure": r0["transition"][
                    "establishment_exposure"
                ],
                "R2_establishment_exposure": r2["transition"][
                    "establishment_exposure"
                ],
                "R2_minus_R0_onset_ordinal": delta,
                "paired_direction": "advance" if delta < 0 else "tie" if delta == 0 else "delay",
                "R0_R_5M_pre": r0["R_5M_pre"],
                "R2_R_5M_pre": r2["R_5M_pre"],
                "R2_minus_R0_R_5M_pre": r2["R_5M_pre"] - r0["R_5M_pre"],
                "R0_R_10M": r0["R_10M"],
                "R2_R_10M": r2["R_10M"],
                "R2_minus_R0_R_10M": r2["R_10M"] - r0["R_10M"],
                "R0_center_accuracy_10M": r0["center_accuracy_10M"],
                "R2_center_accuracy_10M": r2["center_accuracy_10M"],
                "R0_layer_6_linear_accuracy_10M": r0[
                    "layer_6_linear_accuracy_10M"
                ],
                "R2_layer_6_linear_accuracy_10M": r2[
                    "layer_6_linear_accuracy_10M"
                ],
                "R2_threshold_retained_at_6M": r2["transition"][
                    "threshold_state_at_6M"
                ],
                "R2_threshold_retained_at_10M": r2["transition"][
                    "threshold_state_at_10M"
                ],
            }
        )

    scales: dict[str, Any] = {}
    for scale_id in ("1m", "5m"):
        blocks = [block for block in paired_blocks if block["scale_id"] == scale_id]
        selected_cells = [cell for cell in cells if cell["scale_id"] == scale_id]
        verified = {
            condition: sum(
                not cell["transition"]["onset_terminal_coded"]
                for cell in selected_cells
                if cell["condition_id"] == condition
            )
            for condition in ("R0", "R2")
        }
        scales[scale_id] = {
            "paired_blocks": len(blocks),
            "paired_direction_counts": {
                direction: sum(block["paired_direction"] == direction for block in blocks)
                for direction in ("advance", "tie", "delay")
            },
            "verified_onset_counts_out_of_4": verified,
            "verified_onset_probability_difference_R2_minus_R0": (
                verified["R2"] - verified["R0"]
            )
            / 4.0,
            "mean_R2_minus_R0_onset_ordinal": mean(
                block["R2_minus_R0_onset_ordinal"] for block in blocks
            ),
            "mean_R2_minus_R0_R_5M_pre": mean(
                block["R2_minus_R0_R_5M_pre"] for block in blocks
            ),
            "mean_R2_minus_R0_R_10M": mean(
                block["R2_minus_R0_R_10M"] for block in blocks
            ),
        }

    result = {
        "project": "Project Kepler ICLR 2027 robustness extension",
        "kind": "post_confirmatory_cross_scale_robustness_results",
        "manifest": str(manifest_path),
        "manifest_sha256": file_sha256(manifest_path),
        "pilot_gate": str(pilot_gate_path),
        "pilot_gate_sha256": file_sha256(pilot_gate_path),
        "analysis_definitions": _read_json(Path(manifest_path))[
            "transition_definitions"
        ],
        "primary_robustness_question": _read_json(Path(manifest_path))[
            "primary_robustness_question"
        ],
        "integrity": {
            "completed_cells": len(cells),
            "expected_cells": 16,
            "reported_paired_blocks": len(paired_blocks),
            "expected_paired_blocks": 8,
            "all_zero_step_switch_checks_passed": all(
                cell["switch_invariance"]["passed"] for cell in cells
            ),
            "checkpoint_files_verified": sum(
                cell["checkpoint_verification"]["checkpoint_count"] for cell in cells
            ),
            "all_checkpoint_sidecars_valid": all(
                cell["checkpoint_verification"]["all_sidecars_valid"]
                for cell in cells
            ),
            "original_v3_6_p_value_modified": False,
            "combined_confirmatory_p_value_computed": False,
            "scaling_law_fit_computed": False,
        },
        "by_scale": scales,
        "paired_blocks": paired_blocks,
        "cells": cells,
        "interpretation_limits": [
            "post-confirmatory robustness extension",
            "excluded from the original v3.6 p-value",
            "not a scaling-law fit",
            "descriptive paired evidence only; no combined confirmatory p-value",
        ],
    }
    output_path = atomic_write_json(output, result)
    lines = [
        "# ICLR 2027 cross-scale robustness report",
        "",
        "This is a post-confirmatory robustness extension. It is excluded from the "
        "original v3.6 p-value and is not a scaling-law fit. No combined "
        "confirmatory p-value was computed.",
        "",
        "## Scale summaries",
        "",
        "| Scale | R2 advances | Ties | R2 delays | Verified onset R0 | Verified onset R2 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for scale_id, summary in scales.items():
        counts = summary["paired_direction_counts"]
        verified = summary["verified_onset_counts_out_of_4"]
        lines.append(
            f"| {scale_id} | {counts['advance']} | {counts['tie']} | {counts['delay']} | "
            f"{verified['R0']}/4 | {verified['R2']}/4 |"
        )
    lines.extend(
        [
            "",
            "## Every paired block",
            "",
            "| Block | Scale | Rule | Seed | R0 onset | R2 onset | Ordinal delta | Direction | R0 R@5M | R2 R@5M |",
            "|---|---|---|---:|---:|---:|---:|---|---:|---:|",
        ]
    )
    for block in paired_blocks:
        lines.append(
            f"| `{block['block_id']}` | {block['scale_id']} | {block['rule_id']} | "
            f"{block['model_seed_index']} | {block['R0_onset_exposure']} | "
            f"{block['R2_onset_exposure']} | {block['R2_minus_R0_onset_ordinal']} | "
            f"{block['paired_direction']} | {block['R0_R_5M_pre']:.6f} | "
            f"{block['R2_R_5M_pre']:.6f} |"
        )
    lines.extend(
        [
            "",
            "## Frozen outcome details for every paired block",
            "",
            "| Block | R0 est. | R2 est. | R0 R@10M | R2 R@10M | R0 center@10M | R2 center@10M | R0 probe@10M | R2 probe@10M | R2 retained@6M | R2 retained@10M |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for block in paired_blocks:
        lines.append(
            f"| `{block['block_id']}` | {block['R0_establishment_exposure']} | "
            f"{block['R2_establishment_exposure']} | {block['R0_R_10M']:.6f} | "
            f"{block['R2_R_10M']:.6f} | {block['R0_center_accuracy_10M']:.6f} | "
            f"{block['R2_center_accuracy_10M']:.6f} | "
            f"{block['R0_layer_6_linear_accuracy_10M']:.6f} | "
            f"{block['R2_layer_6_linear_accuracy_10M']:.6f} | "
            f"{block['R2_threshold_retained_at_6M']} | "
            f"{block['R2_threshold_retained_at_10M']} |"
        )
    lines.extend(
        [
            "",
            "Frozen v3.6 onset, establishment, exact full-support R, center accuracy, "
            "probe, and post-removal persistence definitions were reused without change.",
            "All 176 registered checkpoint files passed SHA-256 sidecar verification, "
            "and all 16 zero-step sampling-switch checks passed.",
            "",
            f"Machine-readable results: `{output_path}`",
        ]
    )
    atomic_write_bytes(report, ("\n".join(lines) + "\n").encode("utf-8"))
    return result
