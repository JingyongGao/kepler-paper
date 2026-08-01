from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import random
import time
from typing import Any

import numpy as np
import torch
from torch.nn import functional as F

from .config import DEFAULT_CONFIG, ExperimentConfig
from .confirmatory_distribution import sample_confirmatory_pairs
from .cue_dose import PROBE_CHECKPOINTS
from .cue_dose_runner import (
    _checkpoint_path,
    _evaluation_path,
    _hash_tree,
    _snapshot,
    _tree_sha256,
    evaluate_cue_model,
)
from .diagnostic_models import ParentExposedTransformer
from .durability import (
    atomic_write_bytes,
    atomic_write_json,
    durable_torch_save,
    transactional_append_jsonl,
)
from .environment import canonical_hash, file_sha256, git_commit
from .model import size_spec
from .pair_diagnostics import (
    _gradient_norms,
    _model_metadata,
    _pair_centers,
    _soft_loss,
    _update_ratios,
)
from .scale_robustness import (
    ScaleRobustnessRun,
    build_scale_robustness_runs,
    find_scale_robustness_run,
)
from .topology import sample_noise
from .training import learning_rate_at, resolve_device


def _array_sha256(value: np.ndarray) -> str:
    digest = hashlib.sha256()
    digest.update(np.asarray(value).tobytes(order="C"))
    return digest.hexdigest()


def make_scale_robustness_model(
    run: ScaleRobustnessRun,
    cfg: ExperimentConfig = DEFAULT_CONFIG,
) -> ParentExposedTransformer:
    size = size_spec(run.target_core_parameters, cfg.model, cfg.field.p)
    if (
        size.width != run.width
        or size.actual_core_parameters != run.actual_core_parameters
        or size.total_parameters + size.width != run.total_parameters
    ):
        raise ValueError("runtime model size differs from frozen manifest")
    model = ParentExposedTransformer(size.width, cfg.model, cfg.field.p)
    if model.core_parameter_count() != run.actual_core_parameters:
        raise AssertionError("instantiated core parameter count differs from manifest")
    if model.total_parameter_count() != run.total_parameters:
        raise AssertionError("instantiated total parameter count differs from manifest")
    return model


def _memory_telemetry(device: torch.device) -> dict[str, int | None]:
    if device.type != "mps":
        return {
            "mps_current_allocated_bytes": None,
            "mps_driver_allocated_bytes": None,
        }
    return {
        "mps_current_allocated_bytes": int(torch.mps.current_allocated_memory()),
        "mps_driver_allocated_bytes": int(torch.mps.driver_allocated_memory()),
    }


def _assert_manifest(run: ScaleRobustnessRun, manifest_path: str | Path) -> str:
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    if manifest.get("kind") != "post_confirmatory_cross_scale_robustness":
        raise ValueError("wrong scale-robustness manifest kind")
    required_flags = {
        "post_confirmatory_robustness_extension": True,
        "excluded_from_original_v3_6_p_value": True,
        "not_a_scaling_law_fit": True,
    }
    if manifest.get("flags") != required_flags:
        raise ValueError("scale-robustness exclusion flags are not frozen")
    registered = {item["run_id"]: item for item in manifest["runs"]}[run.run_id]
    if canonical_hash(registered) != canonical_hash(asdict(run)):
        raise ValueError("runtime scale-robustness run differs from frozen manifest")
    return file_sha256(manifest_path)


def _save_checkpoint(
    path: Path,
    model: ParentExposedTransformer,
    optimizer: torch.optim.Optimizer,
    run: ScaleRobustnessRun,
    exposed: int,
    rng: np.random.Generator,
    pair_counts: np.ndarray,
    learning_rate: float,
    sampled_nll: float,
    soft_loss: float,
    diagnostics: dict[str, Any],
    cfg: ExperimentConfig,
) -> str:
    payload = {
        "version": "kepler-iclr2027-scale-robustness",
        "run": asdict(run),
        "config": cfg.to_dict(),
        "config_hash": canonical_hash(cfg.to_dict()),
        "git_commit": git_commit(Path.cwd()),
        "model_metadata": _model_metadata(model, run),
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "exposed_examples": exposed,
        "scheduler_state": {
            "kind": "absolute_exposure_cosine",
            "supervised_examples_seen": exposed,
            "last_learning_rate": learning_rate,
        },
        "numpy_stream_state": rng.bit_generator.state,
        "pair_frequency_counts": pair_counts,
        "torch_cpu_rng_state": torch.get_rng_state(),
        "torch_mps_rng_state": (
            torch.mps.get_rng_state().cpu()
            if torch.backends.mps.is_available() and hasattr(torch.mps, "get_rng_state")
            else None
        ),
        "python_rng_state": random.getstate(),
        "training_sampled_nll_nats_diagnostic_only": sampled_nll,
        "optimization_soft_cross_entropy_nats": soft_loss,
        "checkpoint_diagnostics": diagnostics,
        "flags": {
            "post_confirmatory_robustness_extension": True,
            "excluded_from_original_v3_6_p_value": True,
            "not_a_scaling_law_fit": True,
            "sampled_nll_training_objective": False,
        },
    }
    return durable_torch_save(
        payload,
        path,
        required_keys={
            "run",
            "model",
            "optimizer",
            "exposed_examples",
            "numpy_stream_state",
            "pair_frequency_counts",
        },
    )


def _restore_training_rng(
    cpu_state: torch.Tensor,
    mps_state: torch.Tensor | None,
    python_state,
    device: torch.device,
) -> None:
    torch.set_rng_state(cpu_state)
    if device.type == "mps" and mps_state is not None:
        torch.mps.set_rng_state(mps_state)
    random.setstate(python_state)


def _evaluate_switch_boundary(
    model: ParentExposedTransformer,
    optimizer: torch.optim.Optimizer,
    run: ScaleRobustnessRun,
    directory: Path,
    pre_checkpoint: Path,
    device: torch.device,
    checkpoint_diagnostics: dict[str, Any],
    rng: np.random.Generator,
    pair_counts: np.ndarray,
) -> None:
    cpu_state = torch.get_rng_state().clone()
    mps_state = (
        torch.mps.get_rng_state().cpu().clone()
        if device.type == "mps" and hasattr(torch.mps, "get_rng_state")
        else None
    )
    python_state = random.getstate()
    numpy_state_hash = canonical_hash(rng.bit_generator.state)
    pair_counts_hash = _array_sha256(pair_counts)
    model_hash_before = _tree_sha256(model.state_dict())
    optimizer_hash_before = _tree_sha256(optimizer.state_dict())
    pre_eval = evaluate_cue_model(
        model,
        run,
        5_000_000,
        "pre_switch",
        _evaluation_path(directory, 5_000_000, "pre_switch"),
        device=device,
        run_probes=True,
        checkpoint=pre_checkpoint,
        checkpoint_diagnostics=checkpoint_diagnostics,
    )

    post_checkpoint = _checkpoint_path(directory, 5_000_000, "post_switch")
    if post_checkpoint.exists():
        post_checkpoint.unlink()
    os.link(pre_checkpoint, post_checkpoint)
    atomic_write_bytes(
        post_checkpoint.with_suffix(post_checkpoint.suffix + ".sha256"),
        f"{file_sha256(post_checkpoint)}  {post_checkpoint.name}\n".encode("utf-8"),
    )
    post_eval = evaluate_cue_model(
        model,
        run,
        5_000_000,
        "post_switch",
        _evaluation_path(directory, 5_000_000, "post_switch"),
        device=device,
        run_probes=False,
        checkpoint=post_checkpoint,
        checkpoint_diagnostics=checkpoint_diagnostics,
    )
    post_eval["full_support"]["probe_metrics"] = pre_eval["full_support"][
        "probe_metrics"
    ]
    post_eval["probes_run"] = False
    post_eval["probe_metrics_reused_from_identical_pre_switch_state"] = True
    model_hash_after = _tree_sha256(model.state_dict())
    optimizer_hash_after = _tree_sha256(optimizer.state_dict())

    _restore_training_rng(cpu_state, mps_state, python_state, device)
    equality = {
        "optimizer_steps_between_evaluations": 0,
        "absolute_exposure_before": 5_000_000,
        "absolute_exposure_after": 5_000_000,
        "model_state_sha256_before": model_hash_before,
        "model_state_sha256_after": model_hash_after,
        "optimizer_state_sha256_before": optimizer_hash_before,
        "optimizer_state_sha256_after": optimizer_hash_after,
        "model_state_unchanged": model_hash_before == model_hash_after,
        "optimizer_state_unchanged": optimizer_hash_before == optimizer_hash_after,
        "full_support_output_sha256_equal": (
            pre_eval["model_output_sha256"] == post_eval["model_output_sha256"]
        ),
        "numpy_rng_state_unchanged": (
            canonical_hash(rng.bit_generator.state) == numpy_state_hash
        ),
        "torch_cpu_rng_state_restored": bool(
            torch.equal(torch.get_rng_state().cpu(), cpu_state.cpu())
        ),
        "torch_mps_rng_state_restored": (
            True
            if mps_state is None
            else bool(torch.equal(torch.mps.get_rng_state().cpu(), mps_state.cpu()))
        ),
        "python_rng_state_restored": random.getstate() == python_state,
        "pair_frequency_counts_unchanged": _array_sha256(pair_counts) == pair_counts_hash,
        "scheduler_state_unchanged": True,
        "sampler_before": run.pre_switch_support,
        "sampler_after": "uniform_full",
    }
    equality["passed"] = all(
        equality[key]
        for key in (
            "model_state_unchanged",
            "optimizer_state_unchanged",
            "full_support_output_sha256_equal",
            "numpy_rng_state_unchanged",
            "torch_cpu_rng_state_restored",
            "torch_mps_rng_state_restored",
            "python_rng_state_restored",
            "pair_frequency_counts_unchanged",
            "scheduler_state_unchanged",
        )
    )
    if not equality["passed"]:
        raise AssertionError("zero-step scale-robustness sampler switch changed state")
    post_eval["zero_step_switch_invariance"] = equality
    atomic_write_json(
        _evaluation_path(directory, 5_000_000, "post_switch"), post_eval
    )
    atomic_write_json(directory / "switch_invariance.json", equality)
    model.train()


def _write_resume_verification(
    directory: Path,
    checkpoint: Path,
    state: dict[str, Any],
    model: ParentExposedTransformer,
    optimizer: torch.optim.Optimizer,
    run: ScaleRobustnessRun,
    cfg: ExperimentConfig,
) -> None:
    checks = {
        "run_id_equal": state["run"]["run_id"] == run.run_id,
        "config_hash_equal": state["config_hash"] == canonical_hash(cfg.to_dict()),
        "model_state_equal_after_load": (
            _tree_sha256(model.state_dict()) == _tree_sha256(state["model"])
        ),
        "optimizer_state_equal_after_load": (
            _tree_sha256(optimizer.state_dict()) == _tree_sha256(state["optimizer"])
        ),
        "optimizer_not_reset": True,
        "scheduler_not_reset": True,
        "numpy_stream_state_present": "numpy_stream_state" in state,
        "pair_frequency_counts_present": "pair_frequency_counts" in state,
        "torch_rng_state_present": "torch_cpu_rng_state" in state,
    }
    result = {
        "run_id": run.run_id,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": file_sha256(checkpoint),
        "resumed_exposure": int(state["exposed_examples"]),
        "checks": checks,
        "passed": all(checks.values()),
    }
    if not result["passed"]:
        raise AssertionError(f"checkpoint resume verification failed: {checks}")
    atomic_write_json(directory / "resume_verification.json", result)


def train_scale_robustness_run(
    run: ScaleRobustnessRun,
    run_root: str | Path = "artifacts/iclr2027_scale_robustness_runs",
    *,
    manifest_path: str | Path = "artifacts/iclr2027_scale_robustness_manifest.json",
    audit_path: str | Path = "artifacts/confirmatory_distribution_audit.json",
    device_name: str = "auto",
    resume: str | Path | None = None,
    stop_after_exposure: int | None = None,
    cfg: ExperimentConfig = DEFAULT_CONFIG,
) -> Path:
    manifest_hash = _assert_manifest(run, manifest_path)
    endpoint = run.maximum_pair_exposures if stop_after_exposure is None else stop_after_exposure
    if endpoint not in run.checkpoints:
        raise ValueError("stop_after_exposure must be a registered checkpoint")
    device = resolve_device(device_name)
    torch.manual_seed(run.model_initialization_seed)
    np.random.seed(run.model_initialization_seed)
    random.seed(run.model_initialization_seed)
    model = make_scale_robustness_model(run, cfg).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.optimizer.learning_rate,
        betas=cfg.optimizer.betas,
        weight_decay=cfg.optimizer.weight_decay,
    )
    rng = np.random.default_rng(run.data_seed)
    pair_counts = np.zeros((run.p, run.p), dtype=np.int64)
    exposed = 0
    directory = Path(run_root) / run.run_id
    directory.mkdir(parents=True, exist_ok=True)
    if resume is not None:
        resume_path = Path(resume)
        state = torch.load(resume_path, map_location=device, weights_only=False)
        if state["run"]["run_id"] != run.run_id:
            raise ValueError("resume checkpoint belongs to another run")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        exposed = int(state["exposed_examples"])
        rng.bit_generator.state = state["numpy_stream_state"]
        pair_counts = np.asarray(state["pair_frequency_counts"], dtype=np.int64).copy()
        torch.set_rng_state(state["torch_cpu_rng_state"].cpu())
        if device.type == "mps" and state.get("torch_mps_rng_state") is not None:
            torch.mps.set_rng_state(state["torch_mps_rng_state"].cpu())
        if state.get("python_rng_state") is not None:
            random.setstate(state["python_rng_state"])
        _write_resume_verification(
            directory, resume_path, state, model, optimizer, run, cfg
        )
    if exposed >= endpoint:
        raise ValueError("run is already at or beyond the requested endpoint")

    atomic_write_json(
        directory / "run.json",
        {
            "run": asdict(run),
            "device": str(device),
            "manifest_sha256": manifest_hash,
            "confirmatory_distribution_audit_sha256": file_sha256(audit_path),
            "optimizer": cfg.to_dict()["optimizer"],
            "optimization_objective": "exact_soft_label_cross_entropy",
            "sampled_nll_training_objective": False,
            "memory_telemetry_at_start": _memory_telemetry(device),
        },
    )
    save_points = [point for point in run.checkpoints if exposed < point <= endpoint]
    metrics_path = directory / "train.jsonl"
    model.train()
    with metrics_path.open("a", encoding="utf-8") as metrics:
        while exposed < endpoint:
            boundary = min(point for point in save_points if point > exposed)
            count = min(run.batch_size, boundary - exposed)
            active_lambda = run.lambda_value if exposed < run.switch_exposure else 0.0
            pairs_np, linear = sample_confirmatory_pairs(
                rng,
                count,
                active_lambda,
                tuple(run.coefficients),
                run.cue_coefficient_c,
                run.permutation_seed_sha256,
                run.m,
            )
            centers_np = _pair_centers(pairs_np, tuple(run.coefficients), run.p)
            labels_np = (
                centers_np + sample_noise(rng, count, p=run.p, eta=cfg.field.eta)
            ) % run.p
            np.add.at(pair_counts.reshape(-1), linear, 1)
            pairs = torch.as_tensor(pairs_np, dtype=torch.long, device=device)
            centers = torch.as_tensor(centers_np, dtype=torch.long, device=device)
            labels = torch.as_tensor(labels_np, dtype=torch.long, device=device)
            lr = learning_rate_at(exposed, cfg)
            for group in optimizer.param_groups:
                group["lr"] = lr
            optimizer.zero_grad(set_to_none=True)
            started = time.perf_counter()
            logits = model(pairs).float()
            loss = _soft_loss(logits, centers, run.p, cfg.field.eta)
            sampled_nll = F.cross_entropy(logits, labels)
            loss.backward()
            checkpoint_step = exposed + count in save_points
            gradients = _gradient_norms(model) if checkpoint_step else {}
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), cfg.optimizer.grad_clip_norm
            )
            before = _snapshot(model) if checkpoint_step else None
            optimizer.step()
            ratios = _update_ratios(model, before) if before is not None else {}
            if device.type == "mps":
                torch.mps.synchronize()
            elapsed = time.perf_counter() - started
            exposed += count
            telemetry = _memory_telemetry(device)
            record = {
                "exposed_pairs": exposed,
                "active_lambda": active_lambda,
                "sampling_distribution": (
                    run.pre_switch_support if active_lambda > 0.0 else "uniform_full"
                ),
                "optimization_soft_cross_entropy_nats": float(loss.detach().cpu()),
                "sampled_nll_nats_diagnostic_only": float(sampled_nll.detach().cpu()),
                "sampled_nll_training_objective": False,
                "gradient_norm": float(grad_norm.detach().cpu()),
                "learning_rate": lr,
                "step_seconds": elapsed,
                "pairs_per_second": count / max(elapsed, 1.0e-12),
                **telemetry,
            }
            metrics.write(json.dumps(record, sort_keys=True) + "\n")
            metrics.flush()
            os.fsync(metrics.fileno())
            if exposed in save_points:
                phase = "pre_switch" if exposed == 5_000_000 else "standard"
                path = _checkpoint_path(directory, exposed, phase)
                diagnostics = {
                    "per_block_gradient_norm": gradients,
                    "per_block_parameter_update_over_parameter_norm": ratios,
                    "manifest_sha256": manifest_hash,
                    "active_lambda_for_completed_step": active_lambda,
                    "optimizer_reset": False,
                    "scheduler_reset": False,
                    "sampled_nll_training_objective": False,
                }
                _save_checkpoint(
                    path,
                    model,
                    optimizer,
                    run,
                    exposed,
                    rng,
                    pair_counts,
                    lr,
                    float(sampled_nll.detach().cpu()),
                    float(loss.detach().cpu()),
                    diagnostics,
                    cfg,
                )
                if exposed == 5_000_000:
                    _evaluate_switch_boundary(
                        model,
                        optimizer,
                        run,
                        directory,
                        path,
                        device,
                        diagnostics,
                        rng,
                        pair_counts,
                    )
    return directory


def _load_model_for_evaluation(
    run: ScaleRobustnessRun,
    directory: Path,
    exposure: int,
    phase: str,
    device: torch.device,
    cfg: ExperimentConfig,
):
    torch.manual_seed(run.model_initialization_seed)
    model = make_scale_robustness_model(run, cfg).to(device)
    if exposure == 0:
        return model, None, {}
    checkpoint = _checkpoint_path(directory, exposure, phase)
    state = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(state["model"])
    return model, checkpoint, state.get("checkpoint_diagnostics", {})


def evaluate_scale_robustness_checkpoints(
    run: ScaleRobustnessRun,
    directory: Path,
    *,
    through_exposure: int,
    device_name: str = "auto",
    cfg: ExperimentConfig = DEFAULT_CONFIG,
) -> list[dict[str, Any]]:
    device = resolve_device(device_name)
    points = [(0, "initialization")]
    for exposure in run.checkpoints:
        if exposure > through_exposure:
            continue
        if exposure == 5_000_000:
            points.extend(((exposure, "pre_switch"), (exposure, "post_switch")))
        else:
            points.append((exposure, "standard"))
    summaries_path = directory / "evaluations.jsonl"
    existing = set()
    if summaries_path.exists():
        existing = {
            json.loads(line)["checkpoint_id"]
            for line in summaries_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        }
    new = []
    for exposure, phase in points:
        checkpoint_id = f"{exposure}:{phase}"
        if checkpoint_id in existing:
            continue
        output = _evaluation_path(directory, exposure, phase)
        if output.exists():
            evaluation = json.loads(output.read_text(encoding="utf-8"))
        else:
            model, checkpoint, diagnostics = _load_model_for_evaluation(
                run, directory, exposure, phase, device, cfg
            )
            evaluation = evaluate_cue_model(
                model,
                run,
                exposure,
                phase,
                output,
                device=device,
                run_probes=exposure in PROBE_CHECKPOINTS,
                checkpoint=checkpoint,
                checkpoint_diagnostics=diagnostics,
            )
            del model
            if device.type == "mps":
                torch.mps.empty_cache()
        layer6 = evaluation["full_support"].get("probe_metrics", {}).get("6")
        summary = {
            "checkpoint_id": checkpoint_id,
            "run_id": run.run_id,
            "block_id": run.block_id,
            "scale_id": run.scale_id,
            "target_core_parameters": run.target_core_parameters,
            "rule_id": run.rule_id,
            "condition_id": run.condition_id,
            "model_seed_index": run.model_seed_index,
            "model_initialization_seed": run.model_initialization_seed,
            "exposed_pairs": exposure,
            "phase": phase,
            "R_full": evaluation["full_support"]["R"],
            "center_accuracy_full": evaluation["full_support"][
                "center_state_top1_accuracy"
            ],
            "layer_6_linear_accuracy": (
                layer6["linear"]["accuracy"] if layer6 is not None else None
            ),
            "evaluation_file": str(output.resolve()),
        }
        transactional_append_jsonl(summaries_path, summary)
        new.append(summary)
    return new


def run_scale_robustness(
    run_id: str,
    *,
    run_root: str | Path = "artifacts/iclr2027_scale_robustness_runs",
    manifest_path: str | Path = "artifacts/iclr2027_scale_robustness_manifest.json",
    audit_path: str | Path = "artifacts/confirmatory_distribution_audit.json",
    device_name: str = "auto",
    resume: str | Path | None = None,
    stop_after_exposure: int | None = None,
    cfg: ExperimentConfig = DEFAULT_CONFIG,
) -> dict[str, Any]:
    run = find_scale_robustness_run(run_id, audit_path)
    endpoint = run.maximum_pair_exposures if stop_after_exposure is None else stop_after_exposure
    directory = train_scale_robustness_run(
        run,
        run_root,
        manifest_path=manifest_path,
        audit_path=audit_path,
        device_name=device_name,
        resume=resume,
        stop_after_exposure=endpoint,
        cfg=cfg,
    )
    evaluations = evaluate_scale_robustness_checkpoints(
        run,
        directory,
        through_exposure=endpoint,
        device_name=device_name,
        cfg=cfg,
    )
    return {
        "run_id": run.run_id,
        "completed_exposure": endpoint,
        "run_directory": str(directory.resolve()),
        "new_evaluations": evaluations,
        "sampled_nll_training_objective": False,
    }


def scale_robustness_status(
    run_root: str | Path = "artifacts/iclr2027_scale_robustness_runs",
    audit_path: str | Path = "artifacts/confirmatory_distribution_audit.json",
) -> list[dict[str, Any]]:
    root = Path(run_root)
    output = []
    for run in build_scale_robustness_runs(audit_path):
        directory = root / run.run_id
        endpoint = _checkpoint_path(directory, run.maximum_pair_exposures, "standard")
        summaries = directory / "evaluations.jsonl"
        evaluations = (
            len(summaries.read_text(encoding="utf-8").splitlines())
            if summaries.exists()
            else 0
        )
        output.append(
            {
                "run_id": run.run_id,
                "block_id": run.block_id,
                "scale_id": run.scale_id,
                "condition_id": run.condition_id,
                "engineering_learnability_pilot": run.engineering_learnability_pilot,
                "status": (
                    "complete" if endpoint.exists() and evaluations == 12 else "pending"
                ),
                "evaluations": evaluations,
                "maximum_exposure": run.maximum_pair_exposures,
            }
        )
    return output
