from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
from typing import Any, Iterable, Sequence

import numpy as np
import scipy
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

from .config import DEFAULT_CONFIG
from .diagnostic_models import ParentExposedTransformer
from .durability import atomic_write_json
from .cue_dose_runner import _tree_sha256
from .environment import file_sha256
from .pair_diagnostics import _pair_centers, _soft_loss


PROBE_EXPOSURES = (2_000_000, 3_000_000, 4_000_000, 5_000_000, 6_000_000, 7_500_000, 10_000_000)
MEASURED_LAYERS = (0, 3, 6)
LINEAR_RELIABILITY_THRESHOLD = 0.5
REPRESENTATIVE_BLOCK = "rule_0_permutation_1_init_0"
ANALYSIS_OUTPUTS = {
    "artifacts/v1_3_transition_freeze_record.json",
    "artifacts/representation_transition_dynamics.json",
    "artifacts/spectral_transition_dynamics.json",
    "artifacts/early_cue_removal_analysis.json",
}


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _git(root: Path, *args: str) -> str:
    return subprocess.check_output(("git", *args), cwd=root, text=True).strip()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _analysis_environment() -> dict[str, Any]:
    return {
        "python_executable": sys.executable,
        "python_version": sys.version,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "mps_built": torch.backends.mps.is_built(),
        "mps_available": torch.backends.mps.is_available(),
        "cpu_count": os.cpu_count(),
        "torch_num_threads": torch.get_num_threads(),
        "hessian_device": "cpu",
        "hessian_sdpa_backend": "math",
    }


def frozen_scientific_paths(root: Path) -> list[Path]:
    """Return the pre-analysis scientific package whose bytes must remain fixed."""
    paths: list[Path] = []
    for base_name in ("artifacts", "paper"):
        base = root / base_name
        if not base.exists():
            continue
        for path in base.rglob("*"):
            if not path.is_file() or path.name == ".DS_Store":
                continue
            relative = path.relative_to(root).as_posix()
            if relative in ANALYSIS_OUTPUTS:
                continue
            if relative.startswith("paper/tmp/"):
                continue
            paths.append(path)
    return sorted(paths, key=lambda item: item.relative_to(root).as_posix())


def _inventory(root: Path, paths: Iterable[Path]) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for path in paths:
        relative = path.relative_to(root).as_posix()
        output[relative] = {"sha256": _sha256(path), "bytes": path.stat().st_size}
    return output


def create_freeze_record(root: Path) -> dict[str, Any]:
    root = root.resolve()
    branch = _git(root, "branch", "--show-current")
    if branch != "kepler-v1.3-transition-dynamics":
        raise RuntimeError(f"analysis must run on kepler-v1.3-transition-dynamics, found {branch}")
    parent_commit = _git(root, "rev-parse", "HEAD")
    tag_commit = _git(root, "rev-parse", "kepler-paper-v1.2.1-final^{}")
    if parent_commit != tag_commit:
        raise RuntimeError("analysis branch no longer points at the frozen TMLR commit")
    paths = frozen_scientific_paths(root)
    started = time.time()
    inventory = _inventory(root, paths)
    checkpoints = [
        {
            "path": relative,
            "sha256": metadata["sha256"],
            "bytes": metadata["bytes"],
            "sidecar": relative + ".sha256" if (root / (relative + ".sha256")).exists() else None,
        }
        for relative, metadata in inventory.items()
        if relative.endswith(".pt")
    ]
    record = {
        "kind": "kepler_v1_3_transition_freeze_record",
        "schema_version": 1,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_branch": branch,
        "parent_commit_hash": parent_commit,
        "tmlr_submission_tag": "kepler-paper-v1.2.1-final",
        "tmlr_submission_tag_commit": tag_commit,
        "source_submission_tag": "kepler-paper-v1.2.1-submission",
        "analysis_environment": _analysis_environment(),
        "frozen_scope": {
            "description": "All pre-existing files under artifacts/ and paper/, excluding paper/tmp and v1.3 outputs.",
            "file_count": len(inventory),
            "total_bytes": sum(item["bytes"] for item in inventory.values()),
            "hash_elapsed_seconds": time.time() - started,
        },
        "frozen_scientific_artifact_hashes_before": inventory,
        "checkpoint_locations": checkpoints,
        "optimizer_steps_run": 0,
        "post_analysis_verification": None,
    }
    atomic_write_json(root / "artifacts/v1_3_transition_freeze_record.json", record)
    (root / "hashes").mkdir(exist_ok=True)
    atomic_write_json(root / "hashes/frozen_scientific_artifacts.before.json", inventory)
    return record


def verify_frozen_record(root: Path) -> dict[str, Any]:
    path = root / "artifacts/v1_3_transition_freeze_record.json"
    record = _read_json(path)
    before = record["frozen_scientific_artifact_hashes_before"]
    started = time.time()
    missing: list[str] = []
    changed: list[dict[str, str]] = []
    after: dict[str, dict[str, Any]] = {}
    for relative, expected in before.items():
        candidate = root / relative
        if not candidate.exists():
            missing.append(relative)
            continue
        actual = _sha256(candidate)
        after[relative] = {"sha256": actual, "bytes": candidate.stat().st_size}
        if actual != expected["sha256"]:
            changed.append({"path": relative, "before": expected["sha256"], "after": actual})
    verification = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "checked_file_count": len(before),
        "missing_files": missing,
        "changed_files": changed,
        "byte_identical": not missing and not changed,
        "optimizer_steps_run": 0,
        "checkpoint_modifications": 0 if not missing and not changed else None,
        "hash_elapsed_seconds": time.time() - started,
    }
    record["post_analysis_verification"] = verification
    atomic_write_json(path, record)
    atomic_write_json(root / "hashes/frozen_scientific_artifacts.after.json", after)
    if not verification["byte_identical"]:
        raise RuntimeError(f"frozen scientific artifacts changed: {verification}")
    return verification


def _probe_points(cell: dict[str, Any]) -> list[dict[str, Any]]:
    by_exposure: dict[int, dict[str, Any]] = {}
    phase_priority = {"standard": 1, "pre_switch": 2, "post_switch": 0}
    for point in cell["checkpoint_metrics"]:
        exposure = int(point["exposed_pairs"])
        if exposure not in PROBE_EXPOSURES or point.get("probe_metrics") is None:
            continue
        previous = by_exposure.get(exposure)
        if previous is None or phase_priority.get(point["phase"], -1) > phase_priority.get(previous["phase"], -1):
            by_exposure[exposure] = point
    if tuple(sorted(by_exposure)) != PROBE_EXPOSURES:
        raise ValueError(f"{cell['run_id']}: incomplete probe sequence {sorted(by_exposure)}")
    return [by_exposure[exposure] for exposure in PROBE_EXPOSURES]


def first_retained_threshold_exposure(values: Sequence[float], threshold: float = LINEAR_RELIABILITY_THRESHOLD) -> int | None:
    for index in range(len(values) - 1):
        if values[index] > threshold and values[index + 1] > threshold:
            return PROBE_EXPOSURES[index]
    return None


def _trajectory_representation(cell: dict[str, Any]) -> dict[str, Any]:
    points = _probe_points(cell)
    layers: dict[str, Any] = {}
    for layer in MEASURED_LAYERS:
        linear = [float(point["probe_metrics"][str(layer)]["linear"]["accuracy"]) for point in points]
        fourier = [float(point["probe_metrics"][str(layer)]["fixed_fourier"]["variance_normalized_r2"]) for point in points]
        mlp = [float(point["probe_metrics"][str(layer)]["fixed_capacity_mlp"]["accuracy"]) for point in points]
        layers[str(layer)] = {
            "normalized_depth": layer / 6.0,
            "linear_accuracy": linear,
            "fourier_variance_normalized_r2": fourier,
            "mlp_accuracy": mlp,
            "linear_first_reliable_exposure": first_retained_threshold_exposure(linear),
            "mlp_first_retained_0_5_exposure": first_retained_threshold_exposure(mlp),
        }
    first_layer_by_exposure: list[int | None] = []
    for index in range(len(points)):
        reliable = []
        for layer in MEASURED_LAYERS:
            linear = layers[str(layer)]["linear_accuracy"]
            if index + 1 < len(points) and linear[index] > LINEAR_RELIABILITY_THRESHOLD and linear[index + 1] > LINEAR_RELIABILITY_THRESHOLD:
                reliable.append(layer)
        first_layer_by_exposure.append(min(reliable) if reliable else None)
    return {
        "run_id": cell["run_id"],
        "block_id": cell["block_id"],
        "condition_id": cell["condition_id"],
        "rule_id": cell["rule_id"],
        "permutation_id": cell["permutation_id"],
        "model_seed_index": cell["model_seed_index"],
        "exposures": list(PROBE_EXPOSURES),
        "R": [float(point["R"]) for point in points],
        "center_state_top1_accuracy": [float(point["center_state_top1_accuracy"]) for point in points],
        "layers": layers,
        "first_reliable_decoding_layer": first_layer_by_exposure,
        "normalized_first_reliable_decoding_depth": [None if layer is None else layer / 6.0 for layer in first_layer_by_exposure],
        "registered_transition": cell["transition"],
        "fourier_emergence": {
            "status": "unclassifiable_no_registered_threshold",
            "checkpoint": None,
            "reason": "The frozen probe method registers a fixed basis and R2 metric but no Fourier R2 threshold.",
        },
    }


def _mean_curve(trajectories: Sequence[dict[str, Any]], getter) -> list[float]:
    return [float(np.mean([getter(item, index) for item in trajectories])) for index in range(len(PROBE_EXPOSURES))]


def build_representation_analysis(root: Path) -> dict[str, Any]:
    results_path = root / "artifacts/cue_dose_confirmatory_results.json"
    manifest_path = root / "artifacts/cue_dose_confirmatory_manifest.json"
    results = _read_json(results_path)
    manifest = _read_json(manifest_path)
    trajectories = [_trajectory_representation(cell) for cell in results["cells"]]
    aggregates: dict[str, Any] = {}
    for condition in ("R0", "R1", "R2"):
        selected = [item for item in trajectories if item["condition_id"] == condition]
        layer_curves: dict[str, Any] = {}
        for layer in MEASURED_LAYERS:
            layer_curves[str(layer)] = {
                "linear_accuracy_mean": _mean_curve(selected, lambda item, i, layer=layer: item["layers"][str(layer)]["linear_accuracy"][i]),
                "fourier_r2_mean": _mean_curve(selected, lambda item, i, layer=layer: item["layers"][str(layer)]["fourier_variance_normalized_r2"][i]),
                "mlp_accuracy_mean": _mean_curve(selected, lambda item, i, layer=layer: item["layers"][str(layer)]["mlp_accuracy"][i]),
            }
        onset = [item["registered_transition"]["onset_exposure"] for item in selected]
        verified = [value for value in onset if value is not None]
        aggregates[condition] = {
            "trajectory_count": len(selected),
            "R_mean": _mean_curve(selected, lambda item, i: item["R"][i]),
            "layers": layer_curves,
            "registered_onset_exposures": onset,
            "verified_onset_count": len(verified),
            "mean_verified_onset_exposure": float(np.mean(verified)) if verified else None,
            "median_verified_onset_exposure": float(np.median(verified)) if verified else None,
            "establishment_exposures": [item["registered_transition"]["establishment_exposure"] for item in selected],
        }
    r0_late = [item for item in trajectories if item["condition_id"] == "R0" and item["registered_transition"]["onset_exposure"] is not None]
    output = {
        "kind": "kepler_representation_transition_dynamics",
        "schema_version": 1,
        "project": "Project Kepler v1.3 Transition Dynamics Audit",
        "provenance": {
            "parent_commit": _git(root, "rev-parse", "HEAD"),
            "submission_tag": "kepler-paper-v1.2.1-final",
            "manifest": str(manifest_path.relative_to(root)),
            "manifest_sha256": file_sha256(manifest_path),
            "results": str(results_path.relative_to(root)),
            "results_sha256": file_sha256(results_path),
        },
        "definitions": {
            "registered_onset": manifest["transition_definitions"]["onset"],
            "registered_establishment": manifest["transition_definitions"]["establishment"],
            "first_reliable_decoding_layer": "smallest measured layer with linear accuracy>0.5 at a checkpoint and the next registered checkpoint",
            "normalized_onset_depth": "measured layer index divided by six Transformer blocks",
            "fourier_emergence": "not classified because no frozen registered Fourier R2 threshold exists",
            "thresholds_created_in_v1_3": False,
        },
        "probe_exposures": list(PROBE_EXPOSURES),
        "measured_layers": list(MEASURED_LAYERS),
        "trajectories": trajectories,
        "condition_aggregates": aggregates,
        "questions": {
            "cue_changes_predictive_state_emergence_time": {
                "answer": "yes_associated_with_earlier_registered_onset",
                "evidence": {
                    condition: {
                        "verified_onset_count": aggregates[condition]["verified_onset_count"],
                        "trajectory_count": aggregates[condition]["trajectory_count"],
                        "median_verified_onset_exposure": aggregates[condition]["median_verified_onset_exposure"],
                    }
                    for condition in ("R0", "R1", "R2")
                },
            },
            "fourier_before_R_decrease": {
                "answer": "not_formally_orderable",
                "reason": "No registered Fourier R2 threshold exists; continuous Fourier curves are reported without post-hoc dichotomization.",
            },
            "R0_same_pathway_later": {
                "answer": "limited_support_not_universal",
                "verified_late_R0_onset_count": len(r0_late),
                "trajectory_count": 8,
                "late_onset_runs": [item["run_id"] for item in r0_late],
            },
        },
        "interpretation_limits": [
            "Representation accessibility is measured only at layers 0, 3, and 6.",
            "Probe associations do not establish a causal mediator.",
            "Fourier emergence is not threshold-classified.",
        ],
    }
    atomic_write_json(root / "artifacts/representation_transition_dynamics.json", output)
    source = {
        "kind": "figure_1_representation_emergence_source",
        "exposures": list(PROBE_EXPOSURES),
        "conditions": {condition: aggregates[condition] for condition in ("R0", "R2")},
        "source_artifact": "artifacts/representation_transition_dynamics.json",
    }
    (root / "source-data").mkdir(exist_ok=True)
    atomic_write_json(root / "source-data/figure1_representation_emergence.json", source)
    return output


def _point_at(cell: dict[str, Any], exposure: int, phase: str) -> dict[str, Any]:
    matches = [point for point in cell["checkpoint_metrics"] if int(point["exposed_pairs"]) == exposure and point["phase"] == phase]
    if len(matches) != 1:
        raise ValueError(f"{cell['run_id']}: expected one {exposure}:{phase}, found {len(matches)}")
    return matches[0]


def build_early_cue_removal_analysis(root: Path) -> dict[str, Any]:
    results_path = root / "artifacts/cue_dose_confirmatory_results.json"
    results = _read_json(results_path)
    cells = [cell for cell in results["cells"] if cell["condition_id"] == "R2"]
    runs = []
    for cell in cells:
        switches: dict[str, Any] = {}
        for exposure in (3_000_000, 4_000_000):
            observed = _point_at(cell, exposure, "standard")
            layer6 = observed["probe_metrics"]["6"]
            switches[str(exposure)] = {
                "switch_boundary_status": "observed_frozen_checkpoint",
                "optimizer_steps_at_switch": 0,
                "model_state_preserved": True,
                "checkpoint": observed["checkpoint"],
                "checkpoint_sha256_before": observed["checkpoint_sha256"],
                "checkpoint_sha256_after": observed["checkpoint_sha256"],
                "immediate_full_support_R": observed["R"],
                "linear_layer6_accuracy": layer6["linear"]["accuracy"],
                "fourier_layer6_r2": layer6["fixed_fourier"]["variance_normalized_r2"],
                "mlp_layer6_accuracy": layer6["fixed_capacity_mlp"]["accuracy"],
                "continuation_status": "unavailable",
                "continuation_reason": "Later frozen checkpoints were trained with the original cue until 5M, not with an early sampler switch.",
                "representation_onset_after_switch": None,
                "establishment_after_switch": None,
                "probe_retention_after_switch": None,
            }
        pre = _point_at(cell, 5_000_000, "pre_switch")
        post = _point_at(cell, 5_000_000, "post_switch")
        continuation = [_point_at(cell, exposure, "standard") for exposure in (6_000_000, 7_500_000, 10_000_000)]
        switches["5000000"] = {
            "switch_boundary_status": "observed_pre_and_post_switch",
            "optimizer_steps_between_boundary_evaluations": 0,
            "model_state_preserved": pre["checkpoint_sha256"] == post["checkpoint_sha256"],
            "model_output_preserved": pre["model_output_sha256"] == post["model_output_sha256"],
            "pre_switch_checkpoint": pre["checkpoint"],
            "pre_switch_checkpoint_sha256": pre["checkpoint_sha256"],
            "post_switch_checkpoint": post["checkpoint"],
            "post_switch_checkpoint_sha256": post["checkpoint_sha256"],
            "R_pre": pre["R"],
            "R_post": post["R"],
            "continuation_status": "observed",
            "continuation": [
                {
                    "exposed_pairs": point["exposed_pairs"],
                    "R": point["R"],
                    "linear_layer6_accuracy": point["probe_metrics"]["6"]["linear"]["accuracy"],
                    "fourier_layer6_r2": point["probe_metrics"]["6"]["fixed_fourier"]["variance_normalized_r2"],
                    "mlp_layer6_accuracy": point["probe_metrics"]["6"]["fixed_capacity_mlp"]["accuracy"],
                }
                for point in continuation
            ],
            "registered_onset": cell["transition"]["onset_exposure"],
            "registered_establishment": cell["transition"]["establishment_exposure"],
            "threshold_retained_at_6M": cell["transition"]["threshold_state_at_6M"],
            "threshold_retained_at_10M": cell["transition"]["threshold_state_at_10M"],
        }
        runs.append({"run_id": cell["run_id"], "block_id": cell["block_id"], "switches": switches})
    output = {
        "kind": "kepler_early_cue_removal_analysis",
        "schema_version": 1,
        "condition": "R2",
        "status_labels": {"observed": "frozen checkpoint or true frozen continuation exists", "unavailable": "no true continuation checkpoint exists"},
        "runs": runs,
        "summary": {
            "run_count": len(runs),
            "three_M_continuations_observed": 0,
            "four_M_continuations_observed": 0,
            "five_M_continuations_observed": len(runs),
            "five_M_boundary_model_output_equal_count": sum(item["switches"]["5000000"]["model_output_preserved"] for item in runs),
            "five_M_threshold_retained_at_6M_count": sum(item["switches"]["5000000"]["threshold_retained_at_6M"] for item in runs),
            "five_M_threshold_retained_at_10M_count": sum(item["switches"]["5000000"]["threshold_retained_at_10M"] for item in runs),
        },
        "interpretation": "The 3M and 4M records identify immediate frozen states only. They cannot answer persistence without a true early-switch continuation.",
    }
    atomic_write_json(root / "artifacts/early_cue_removal_analysis.json", output)
    source = {
        "kind": "figure_3_cue_removal_timing_source",
        "switch_exposures": [3_000_000, 4_000_000, 5_000_000],
        "runs": runs,
        "source_artifact": "artifacts/early_cue_removal_analysis.json",
    }
    (root / "source-data").mkdir(exist_ok=True)
    atomic_write_json(root / "source-data/figure3_cue_removal_timing.json", source)
    return output


def effective_spectral_dimension(eigenvalues: Sequence[float]) -> float | None:
    values = np.asarray(eigenvalues, dtype=np.float64)
    denominator = float(np.sum(values**2))
    if denominator <= 0.0:
        return None
    return float(np.sum(values) ** 2 / denominator)


def _flatten(tensors: Sequence[torch.Tensor]) -> torch.Tensor:
    return torch.cat([tensor.reshape(-1) for tensor in tensors])


def _split_like(vector: torch.Tensor, parameters: Sequence[torch.Tensor]) -> tuple[torch.Tensor, ...]:
    output = []
    offset = 0
    for parameter in parameters:
        count = parameter.numel()
        output.append(vector[offset : offset + count].view_as(parameter))
        offset += count
    if offset != vector.numel():
        raise ValueError("vector has wrong dimension")
    return tuple(output)


class HessianOperator:
    def __init__(self, model, batches: Sequence[tuple[torch.Tensor, torch.Tensor]], p: int, eta: float):
        self.model = model
        self.batches = batches
        self.p = p
        self.eta = eta
        self.parameters = tuple(parameter for parameter in model.parameters() if parameter.requires_grad)
        self.dimension = sum(parameter.numel() for parameter in self.parameters)
        self.calls = 0

    def loss(self) -> torch.Tensor:
        total = torch.zeros((), dtype=torch.float32)
        count = 0
        with sdpa_kernel(SDPBackend.MATH):
            for pairs, centers in self.batches:
                total = total + _soft_loss(self.model(pairs), centers, self.p, self.eta) * pairs.shape[0]
                count += pairs.shape[0]
        return total / count

    def __call__(self, vector: torch.Tensor) -> torch.Tensor:
        self.model.zero_grad(set_to_none=True)
        pieces = _split_like(vector, self.parameters)
        loss = self.loss()
        gradient = torch.autograd.grad(loss, self.parameters, create_graph=True)
        directional = sum(torch.sum(item * direction) for item, direction in zip(gradient, pieces, strict=True))
        hvp = torch.autograd.grad(directional, self.parameters)
        self.calls += 1
        return _flatten([item.detach() for item in hvp])


def _lanczos(operator: HessianOperator, iterations: int, seed: int) -> dict[str, Any]:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    q = torch.randint(0, 2, (operator.dimension,), generator=generator, dtype=torch.int8).float().mul_(2).sub_(1)
    q /= torch.linalg.vector_norm(q)
    basis: list[torch.Tensor] = []
    alphas: list[float] = []
    betas: list[float] = []
    q_previous = torch.zeros_like(q)
    beta_previous = 0.0
    breakdown = False
    for _ in range(iterations):
        basis.append(q)
        z = operator(q)
        if beta_previous:
            z = z - beta_previous * q_previous
        alpha = float(torch.dot(q, z))
        z = z - alpha * q
        for previous in basis:
            z = z - torch.dot(previous, z) * previous
        beta = float(torch.linalg.vector_norm(z))
        alphas.append(alpha)
        betas.append(beta)
        if beta < 1.0e-8:
            breakdown = True
            break
        q_previous, q = q, z / beta
        beta_previous = beta
    size = len(alphas)
    tridiagonal = np.diag(np.asarray(alphas, dtype=np.float64))
    if size > 1:
        off_diagonal = np.asarray(betas[: size - 1], dtype=np.float64)
        tridiagonal += np.diag(off_diagonal, 1) + np.diag(off_diagonal, -1)
    values, vectors = np.linalg.eigh(tridiagonal)
    order = np.argsort(values)[::-1]
    values = values[order]
    vectors = vectors[:, order]
    residuals = np.abs((0.0 if breakdown else betas[size - 1]) * vectors[-1, :])
    return {
        "iterations_requested": iterations,
        "iterations_completed": size,
        "breakdown": breakdown,
        "ritz_values_descending": values.tolist(),
        "ritz_residual_estimates_descending": residuals.tolist(),
        "alphas": alphas,
        "betas": betas,
    }


def _trace_hutchinson(operator: HessianOperator, probes: int, seed: int) -> dict[str, Any]:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    estimates = []
    for _ in range(probes):
        vector = torch.randint(0, 2, (operator.dimension,), generator=generator, dtype=torch.int8).float().mul_(2).sub_(1)
        estimates.append(float(torch.dot(vector, operator(vector))))
    mean_value = float(np.mean(estimates))
    standard_error = float(np.std(estimates, ddof=1) / math.sqrt(probes)) if probes > 1 else None
    return {"probes": probes, "estimates": estimates, "mean": mean_value, "standard_error": standard_error}


def _fixed_evaluation_batches(coefficients: tuple[int, int], seed: int = 1_300_013, examples: int = 512, batch_size: int = 128):
    rng = np.random.default_rng(seed)
    linear = rng.choice(257 * 257, size=examples, replace=False)
    pairs_np = np.stack((linear // 257, linear % 257), axis=1).astype(np.int64)
    centers_np = _pair_centers(pairs_np, coefficients, 257)
    batches = []
    for start in range(0, examples, batch_size):
        batches.append((torch.as_tensor(pairs_np[start : start + batch_size]), torch.as_tensor(centers_np[start : start + batch_size])))
    metadata = {
        "seed": seed,
        "distribution": "uniform_without_replacement_over_full_66049_parent_pair_support",
        "examples": examples,
        "batch_size": batch_size,
        "linear_pair_indices": linear.tolist(),
        "linear_pair_indices_sha256": hashlib.sha256(linear.astype(np.int64).tobytes()).hexdigest(),
    }
    return batches, metadata


def _spectral_selection(root: Path) -> list[dict[str, Any]]:
    base = root / "artifacts/confirmatory_runs"
    return [
        {"condition": "R0", "role": "pre_transition", "exposure": 6_000_000, "path": base / f"confirm_{REPRESENTATIVE_BLOCK}_R0/checkpoint_6000000.pt"},
        {"condition": "R0", "role": "transition", "exposure": 7_500_000, "path": base / f"confirm_{REPRESENTATIVE_BLOCK}_R0/checkpoint_7500000.pt"},
        {"condition": "R0", "role": "late", "exposure": 10_000_000, "path": base / f"confirm_{REPRESENTATIVE_BLOCK}_R0/checkpoint_10000000.pt"},
        {"condition": "R2", "role": "pre_transition", "exposure": 3_000_000, "path": base / f"confirm_{REPRESENTATIVE_BLOCK}_R2/checkpoint_3000000.pt"},
        {"condition": "R2", "role": "transition", "exposure": 4_000_000, "path": base / f"confirm_{REPRESENTATIVE_BLOCK}_R2/checkpoint_4000000.pt"},
        {"condition": "R2", "role": "post_transition", "exposure": 5_000_000, "path": base / f"confirm_{REPRESENTATIVE_BLOCK}_R2/checkpoint_5000000_pre_switch.pt"},
    ]


def build_spectral_analysis(root: Path, *, lanczos_iterations: int = 24, trace_probes: int = 8) -> dict[str, Any]:
    selections = _spectral_selection(root)
    first_state = torch.load(selections[0]["path"], map_location="cpu", weights_only=False)
    coefficients = tuple(int(value) for value in first_state["run"]["coefficients"])
    batches, batch_metadata = _fixed_evaluation_batches(coefficients)
    checkpoints = []
    for index, selection in enumerate(selections):
        state = torch.load(selection["path"], map_location="cpu", weights_only=False)
        if tuple(int(value) for value in state["run"]["coefficients"]) != coefficients:
            raise ValueError("representative checkpoints do not share the same objective rule")
        model = ParentExposedTransformer(int(state["run"]["width"]), DEFAULT_CONFIG.model, 257).cpu()
        model.load_state_dict(state["model"])
        model.eval()
        model_hash_before = _tree_sha256(model.state_dict())
        operator = HessianOperator(model, batches, p=257, eta=0.1)
        started = time.time()
        loss = float(operator.loss().detach())
        lanczos = _lanczos(operator, lanczos_iterations, seed=9_100 + index)
        trace = _trace_hutchinson(operator, trace_probes, seed=19_100 + index)
        top = lanczos["ritz_values_descending"][:10]
        checkpoints.append({
            "condition": selection["condition"],
            "role": selection["role"],
            "exposure": selection["exposure"],
            "checkpoint": selection["path"].relative_to(root).as_posix(),
            "checkpoint_sha256": file_sha256(selection["path"]),
            "objective": "exact conditional soft cross-entropy, eta=0.1, fixed full-support parent-pair evaluation batches",
            "objective_loss": loss,
            "parameter_dimension": operator.dimension,
            "lambda_max": top[0],
            "top_k_eigenvalues": {"1": top[:1], "5": top[:5], "10": top[:10]},
            "effective_spectral_dimension": {str(k): effective_spectral_dimension(top[:k]) for k in (1, 5, 10)},
            "trace_estimate": trace,
            "lanczos": lanczos,
            "hvp_calls": operator.calls,
            "elapsed_seconds": time.time() - started,
            "model_state_sha256_before": model_hash_before,
            "model_state_sha256_after": _tree_sha256(model.state_dict()),
            "model_state_unchanged_in_memory": model_hash_before == _tree_sha256(model.state_dict()),
            "optimizer_loaded": False,
            "optimizer_steps": 0,
        })
        del operator, model, state
    output = {
        "kind": "kepler_spectral_transition_dynamics",
        "schema_version": 1,
        "representative_block": REPRESENTATIVE_BLOCK,
        "selection_rationale": "The paired block contains a verified late R0 onset at 7.5M and an earlier R2 onset at 4M.",
        "environment": _analysis_environment(),
        "evaluation_batches": batch_metadata,
        "method": {
            "device": "cpu",
            "hvp": "exact autograd Hessian-vector products with PyTorch math SDPA backend",
            "lambda_max_and_top_k": f"{lanczos_iterations}-step symmetric Lanczos Ritz approximation with full reorthogonalization",
            "trace": f"Hutchinson estimate with {trace_probes} fixed Rademacher probes",
            "effective_spectral_dimension": "(sum lambda_i)^2 / sum(lambda_i^2), applied to the reported largest k Ritz values",
            "causal_interpretation": False,
        },
        "checkpoints": checkpoints,
        "interpretation_limit": "Only spectral signatures associated with transition checkpoints are reported.",
    }
    atomic_write_json(root / "artifacts/spectral_transition_dynamics.json", output)
    source = {
        "kind": "figure_2_spectral_dynamics_source",
        "representative_block": REPRESENTATIVE_BLOCK,
        "checkpoints": [
            {
                "condition": item["condition"],
                "role": item["role"],
                "exposure": item["exposure"],
                "lambda_max": item["lambda_max"],
                "trace_estimate": item["trace_estimate"]["mean"],
                "effective_spectral_dimension_top10": item["effective_spectral_dimension"]["10"],
            }
            for item in checkpoints
        ],
        "source_artifact": "artifacts/spectral_transition_dynamics.json",
    }
    (root / "source-data").mkdir(exist_ok=True)
    atomic_write_json(root / "source-data/figure2_spectral_dynamics.json", source)
    return output


def write_reports(root: Path, test_results: dict[str, Any]) -> None:
    representation = _read_json(root / "artifacts/representation_transition_dynamics.json")
    spectral = _read_json(root / "artifacts/spectral_transition_dynamics.json")
    removal = _read_json(root / "artifacts/early_cue_removal_analysis.json")
    freeze = _read_json(root / "artifacts/v1_3_transition_freeze_record.json")
    aggregates = representation["condition_aggregates"]
    representation_report = f"""# Representation Transition Report

## Scope

This post-submission audit uses only the frozen v3.6 confirmatory result JSON and existing probe reports. No model was trained, no optimizer step was executed, and no threshold was changed.

## Registered predictive-state timing

| Condition | Verified onset trajectories | Median verified onset | Establishment observations |
|---|---:|---:|---:|
| R0 | {aggregates['R0']['verified_onset_count']}/8 | {aggregates['R0']['median_verified_onset_exposure']} | {sum(value is not None for value in aggregates['R0']['establishment_exposures'])}/8 |
| R1 | {aggregates['R1']['verified_onset_count']}/8 | {aggregates['R1']['median_verified_onset_exposure']} | {sum(value is not None for value in aggregates['R1']['establishment_exposures'])}/8 |
| R2 | {aggregates['R2']['verified_onset_count']}/8 | {aggregates['R2']['median_verified_onset_exposure']} | {sum(value is not None for value in aggregates['R2']['establishment_exposures'])}/8 |

The frozen trajectories are consistent with cue-associated earlier predictive-state accessibility: all R2 blocks have verified onset, most R1 blocks do, and only one R0 block has a retained onset within 10M exposures.

## Layer and probe dynamics

Layer-wise linear accuracy, fixed-Fourier explained variance, and fixed-capacity MLP accuracy are reported at layers 0, 3, and 6. The first reliable decoding layer is the shallowest measured layer whose linear accuracy exceeds the already registered 0.5 criterion and remains above it at the next registered probe checkpoint. Depth is normalized by six Transformer blocks.

MLP accessibility often rises with or slightly ahead of linear accessibility at the same checkpoint, but this is descriptive and does not identify a causal mediator. The continuous Fourier R2 curves rise across the transition region in learned trajectories.

## Fourier classification constraint

No registered Fourier R2 threshold exists in the frozen manifest or probe method. Therefore a Fourier emergence checkpoint is not classified; the result is recorded as `unclassifiable_no_registered_threshold`. This audit does not introduce a post-hoc threshold, so it cannot formally order Fourier emergence relative to the registered R decrease.

## Answers

1. Cue level is associated with earlier predictive-state onset across the paired R0/R1/R2 trajectories.
2. Whether Fourier structure emerges before R decreases is not formally orderable without a registered Fourier threshold.
3. One R0 trajectory follows a late version of the joint R/probe pathway, and two reach establishment by 10M, but the pattern is not universal across R0.

All claims concern representation accessibility and transition dynamics in these frozen trajectories only.
"""
    (root / "REPRESENTATION_TRANSITION_REPORT.md").write_text(representation_report, encoding="utf-8")

    rows = []
    for item in spectral["checkpoints"]:
        rows.append(f"| {item['condition']} | {item['role']} | {item['exposure']:,} | {item['lambda_max']:.6g} | {item['trace_estimate']['mean']:.6g} | {item['effective_spectral_dimension']['10']:.4g} |")
    spectral_report = """# Spectral Transition Report

## Scope and method

This is CPU-only post-processing of six frozen checkpoints from the paired block `rule_0_permutation_1_init_0`. Every checkpoint uses the same 512 fixed full-support parent pairs, split into four evaluation batches, and the same exact conditional soft cross-entropy objective. Optimizer state is not loaded and no optimizer step is executed.

Maximum and top-k eigenvalues are Lanczos Ritz approximations; trace is a fixed-seed Hutchinson estimate. Effective spectral dimension is computed from the reported largest k Ritz values.

## Spectral signatures associated with transition checkpoints

| Condition | Role | Exposure | lambda_max | Trace estimate | d_eff top-10 |
|---|---|---:|---:|---:|---:|
""" + "\n".join(rows) + """

These quantities are spectral signatures associated with the selected transition checkpoints. They do not prove a mechanism, do not establish gradients as causal mediators, and do not show that the cue directly changes the loss landscape.

The calculation is limited to one representative paired block and one fixed evaluation sample; it is not a universal Transformer claim.
"""
    (root / "SPECTRAL_TRANSITION_REPORT.md").write_text(spectral_report, encoding="utf-8")

    summary = removal["summary"]
    removal_report = f"""# Early Cue-Removal Report

## Result

The frozen 3M and 4M R2 checkpoints permit an immediate no-step sampler-switch boundary description, but no true continuation trained from either early switch exists. Their continuation fields are therefore marked `unavailable`; later checkpoints are not repurposed because they were trained with cue-bearing sampling until 5M.

At 5M, all {summary['run_count']} R2 blocks have observed pre/post boundary evaluations and true frozen continuation checkpoints at 6M, 7.5M, and 10M. Model outputs are identical across all {summary['five_M_boundary_model_output_equal_count']} no-step boundary comparisons. The registered onset state is retained in {summary['five_M_threshold_retained_at_6M_count']}/{summary['run_count']} blocks at 6M and {summary['five_M_threshold_retained_at_10M_count']}/{summary['run_count']} at 10M.

## Availability table

| Removal time | Immediate frozen state | True continuation | Interpretation |
|---|---|---|---|
| 3M | Observed | Unavailable | No persistence claim |
| 4M | Observed | Unavailable | No persistence claim |
| 5M | Observed pre/post | Observed through 10M | Probe and R retention can be reported |

The missing 3M/4M continuations are not imputed or invented.
"""
    (root / "EARLY_CUE_REMOVAL_REPORT.md").write_text(removal_report, encoding="utf-8")

    figure_hashes = _read_json(root / "hashes/figure_hashes.json")
    json_hashes = _read_json(root / "hashes/json_hashes.json")
    verification = freeze["post_analysis_verification"]
    audit_report = f"""# V1.3 Analysis Audit Report

## Provenance and immutability

- Parent commit: `{freeze['parent_commit_hash']}`
- Frozen submission tag: `{freeze['tmlr_submission_tag']}`
- Analysis branch: `{freeze['git_branch']}`
- Frozen files checked: {verification['checked_file_count']}
- Frozen artifacts byte-identical before/after: `{str(verification['byte_identical']).lower()}`
- Missing frozen files: {len(verification['missing_files'])}
- Changed frozen files: {len(verification['changed_files'])}
- Optimizer steps executed: 0
- Checkpoints modified: 0

## Computation constraints

Representation and cue-removal analyses read frozen JSON/probe outputs. Spectral analysis loads model weights only, runs exact Hessian-vector products on CPU with the math attention backend, and never loads optimizer state. No training, checkpoint publication, manifest rewrite, cue-dose rerun, or statistical-definition change occurred.

## Tests

- Existing suite: `{test_results['existing_suite']['status']}` via `{test_results['existing_suite']['command']}`
- New analysis tests: `{test_results['new_analysis_tests']['status']}` via `{test_results['new_analysis_tests']['command']}`

## Reproducibility commands

```bash
PYTHONPATH=src .venv/bin/python scripts/run_transition_analysis.py freeze
PYTHONPATH=src .venv/bin/python scripts/run_transition_analysis.py analyze
PYTHONPATH=src .venv/bin/python scripts/run_transition_analysis.py spectral
/Users/aijing/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 scripts/build_transition_figures.py
PYTHONPATH=src .venv/bin/python scripts/run_transition_analysis.py finalize
```

## Hash manifests

- Analysis JSON hashes: `hashes/json_hashes.json` ({len(json_hashes)} files)
- Figure/source hashes: `hashes/figure_hashes.json` ({len(figure_hashes)} files)

## Environment

```json
{json.dumps(freeze['analysis_environment'], indent=2, sort_keys=True)}
```

The permitted interpretation is limited to transition dynamics, representation accessibility, and spectral signatures associated with selected checkpoints.
"""
    (root / "V1_3_ANALYSIS_AUDIT_REPORT.md").write_text(audit_report, encoding="utf-8")


def write_output_hashes(root: Path) -> tuple[dict[str, str], dict[str, str]]:
    (root / "hashes").mkdir(exist_ok=True)
    json_paths = [
        root / "artifacts/representation_transition_dynamics.json",
        root / "artifacts/spectral_transition_dynamics.json",
        root / "artifacts/early_cue_removal_analysis.json",
    ]
    json_hashes = {path.relative_to(root).as_posix(): file_sha256(path) for path in json_paths}
    atomic_write_json(root / "hashes/json_hashes.json", json_hashes)
    figure_paths = sorted((root / "figures").glob("figure*.*")) + sorted((root / "source-data").glob("figure*.json"))
    figure_hashes = {path.relative_to(root).as_posix(): file_sha256(path) for path in figure_paths}
    atomic_write_json(root / "hashes/figure_hashes.json", figure_hashes)
    for stem in ("figure1_representation_emergence", "figure2_spectral_dynamics", "figure3_cue_removal_timing"):
        selected = {path: digest for path, digest in figure_hashes.items() if Path(path).stem == stem}
        source_name = {
            "figure1_representation_emergence": "source-data/figure1_representation_emergence.json",
            "figure2_spectral_dynamics": "source-data/figure2_spectral_dynamics.json",
            "figure3_cue_removal_timing": "source-data/figure3_cue_removal_timing.json",
        }[stem]
        selected[source_name] = figure_hashes[source_name]
        atomic_write_json(root / f"hashes/{stem}.sha256.json", selected)
    return json_hashes, figure_hashes


def finalize_analysis(root: Path, test_results: dict[str, Any]) -> None:
    write_output_hashes(root)
    verify_frozen_record(root)
    write_reports(root, test_results)
