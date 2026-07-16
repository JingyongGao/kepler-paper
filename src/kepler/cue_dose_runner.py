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

from .checkpoint_gradients import _accumulate
from .compositional_evaluation import (
    _axis_metrics,
    _axis_tail_metrics,
    _basic_partition_metrics,
)
from .config import DEFAULT_CONFIG, ExperimentConfig
from .cue_distribution import sample_cue_pairs
from .cue_dose import (
    GRADIENT_AUDIT_CHECKPOINTS,
    PROBE_CHECKPOINTS,
    CueDoseRun,
    find_cue_dose_run,
)
from .durability import (
    atomic_write_json,
    durable_torch_save,
    transactional_append_jsonl,
)
from .environment import canonical_hash, file_sha256, git_commit
from .information import predictable_information, torch_exact_conditional_kl
from .pair_diagnostics import (
    _gradient_norms,
    _model_metadata,
    _pair_centers,
    _snapshot,
    _soft_loss,
    _update_ratios,
    make_pair_model,
)
from .probes import (
    FourierStateProbe,
    LinearStateProbe,
    MLPStateProbe,
    ProbeDataset,
    _classification_metrics,
    _fourier_metrics,
    train_classification_probe,
    train_fourier_probe,
)
from .topology import sample_noise
from .training import learning_rate_at, resolve_device


def _hash_tree(value: Any, digest: hashlib._Hash) -> None:
    if isinstance(value, torch.Tensor):
        digest.update(b"tensor")
        digest.update(str(value.dtype).encode())
        digest.update(str(tuple(value.shape)).encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    elif isinstance(value, np.ndarray):
        digest.update(b"ndarray")
        digest.update(str(value.dtype).encode())
        digest.update(str(value.shape).encode())
        digest.update(value.tobytes(order="C"))
    elif isinstance(value, dict):
        digest.update(b"dict")
        for key in sorted(value, key=lambda item: str(item)):
            digest.update(str(key).encode())
            _hash_tree(value[key], digest)
    elif isinstance(value, (list, tuple)):
        digest.update(b"sequence")
        for item in value:
            _hash_tree(item, digest)
    else:
        digest.update(repr(value).encode())


def _tree_sha256(value: Any) -> str:
    digest = hashlib.sha256()
    _hash_tree(value, digest)
    return digest.hexdigest()


def _checkpoint_path(directory: Path, exposure: int, phase: str = "standard") -> Path:
    if exposure == 5_000_000:
        return directory / f"checkpoint_{exposure}_{phase}.pt"
    return directory / f"checkpoint_{exposure}.pt"


def _source_checkpoint(run: CueDoseRun, exposure: int) -> Path:
    if run.reused_source_run_id is None:
        raise ValueError("run has no frozen source")
    return Path("artifacts/compositional_runs") / run.reused_source_run_id / f"checkpoint_{exposure}.pt"


def _evaluation_path(directory: Path, exposure: int, phase: str) -> Path:
    if exposure == 0:
        return directory / "initialization.cue.eval.json"
    if exposure == 5_000_000:
        return directory / f"checkpoint_{exposure}_{phase}.cue.eval.json"
    return directory / f"checkpoint_{exposure}.cue.eval.json"


def _probe_path(directory: Path, exposure: int, phase: str) -> Path:
    if exposure == 5_000_000:
        return directory / f"checkpoint_{exposure}_{phase}.cue.probes.json"
    return directory / f"checkpoint_{exposure}.cue.probes.json"


def _gradient_path(directory: Path, exposure: int, phase: str) -> Path:
    if exposure == 0:
        return directory / "initialization.cue.gradient.json"
    if exposure == 5_000_000:
        return directory / f"checkpoint_{exposure}_{phase}.cue.gradient.json"
    return directory / f"checkpoint_{exposure}.cue.gradient.json"


@torch.no_grad()
def _forward_full_support(
    model,
    device: torch.device,
    *,
    collect_residuals: bool,
    batch_size: int = 8192,
):
    grid = np.stack(
        np.meshgrid(np.arange(257), np.arange(257), indexing="ij"), axis=-1
    ).reshape(-1, 2)
    centers_np = (45 * grid[:, 0] + 36 * grid[:, 1]) % 257
    kl_parts, correct_parts, probability_parts, entropy_parts = [], [], [], []
    residual_parts: dict[int, list[torch.Tensor]] = (
        {0: [], 3: [], 6: []} if collect_residuals else {}
    )
    output_digest = hashlib.sha256()
    for start in range(0, grid.shape[0], batch_size):
        stop = min(grid.shape[0], start + batch_size)
        parents = torch.as_tensor(grid[start:stop], dtype=torch.long, device=device)
        centers = torch.as_tensor(
            centers_np[start:stop], dtype=torch.long, device=device
        )
        if collect_residuals:
            logits, residuals = model(parents, return_residuals=True)
        else:
            logits = model(parents)
            residuals = None
        logits = logits.float()
        output_digest.update(logits.detach().cpu().contiguous().numpy().tobytes())
        log_probs = torch.log_softmax(logits, dim=-1)
        probabilities = log_probs.exp()
        kl_parts.append(
            torch_exact_conditional_kl(log_probs, centers, 257, 0.1).cpu().numpy()
        )
        correct_parts.append((logits.argmax(dim=-1) == centers).cpu().numpy())
        probability_parts.append(
            probabilities.gather(-1, centers[:, None]).squeeze(-1).cpu().numpy()
        )
        entropy_parts.append(
            (-torch.sum(probabilities * log_probs, dim=-1)).cpu().numpy()
        )
        if residuals is not None:
            for layer in residual_parts:
                residual_parts[layer].append(residuals[layer].float().cpu())
    return (
        grid,
        centers_np.astype(np.int64),
        np.concatenate(kl_parts),
        np.concatenate(correct_parts),
        np.concatenate(probability_parts),
        np.concatenate(entropy_parts),
        {layer: torch.cat(parts) for layer, parts in residual_parts.items()},
        output_digest.hexdigest(),
    )


def _train_full_support_probes(
    residuals: dict[int, torch.Tensor],
    centers: torch.Tensor,
    run: CueDoseRun,
) -> dict[str, Any]:
    rng = np.random.default_rng(run.probe_seed)
    order = rng.permutation(257 * 257)
    train_indices = torch.from_numpy(order[:4096])
    validation_indices = torch.from_numpy(order[4096:5120])
    output: dict[str, Any] = {
        "probe_train_distribution": "fixed_independent_full_support_pairs",
        "train_examples": 4096,
        "validation_examples": 1024,
        "evaluation_examples": 257 * 257,
        "probe_seed": run.probe_seed,
        "layers": {},
    }
    for layer, values in residuals.items():
        train = ProbeDataset(values[train_indices], centers[train_indices])
        validation = ProbeDataset(
            values[validation_indices], centers[validation_indices]
        )
        full = ProbeDataset(values, centers)
        width = values.shape[-1]
        seed = run.probe_seed + layer
        torch.manual_seed(seed)
        linear = LinearStateProbe(width)
        train_classification_probe(linear, train, validation, validation, seed=seed)
        torch.manual_seed(seed + 1)
        fourier = FourierStateProbe(width)
        train_fourier_probe(fourier, train, validation, validation, seed=seed + 1)
        torch.manual_seed(seed + 2)
        mlp = MLPStateProbe(width, hidden=256)
        train_classification_probe(mlp, train, validation, validation, seed=seed + 2)
        output["layers"][str(layer)] = {
            "linear": _classification_metrics(linear, full),
            "fixed_fourier": _fourier_metrics(fourier, full),
            "fixed_capacity_mlp": {
                "hidden_width": 256,
                **_classification_metrics(mlp, full),
            },
        }
    return output


def evaluate_cue_model(
    model,
    run: CueDoseRun,
    exposure: int,
    phase: str,
    output: Path,
    *,
    device: torch.device,
    run_probes: bool,
    checkpoint: Path | None,
    checkpoint_diagnostics: dict[str, Any] | None,
) -> dict[str, Any]:
    model.eval()
    (
        grid,
        centers,
        kl,
        correct,
        center_probability,
        entropy,
        residuals,
        output_hash,
    ) = _forward_full_support(model, device, collect_residuals=run_probes)
    r_values = kl / predictable_information(257, 0.1)
    full = np.ones(257 * 257, dtype=np.bool_)
    metrics = _basic_partition_metrics(
        "full",
        full,
        kl,
        r_values,
        correct,
        center_probability,
        entropy,
        grid[:, 0],
        run.evaluation_seed + exposure,
    )
    metrics["by_x1"] = _axis_metrics(
        grid[:, 0], full, r_values, correct, center_probability
    )
    metrics["by_x2"] = _axis_metrics(
        grid[:, 1], full, r_values, correct, center_probability
    )
    metrics["by_target_z"] = _axis_metrics(
        centers, full, r_values, correct, center_probability
    )
    metrics["tail_summary"] = {
        "by_x1": _axis_tail_metrics(metrics["by_x1"]),
        "by_x2": _axis_tail_metrics(metrics["by_x2"]),
        "by_target_z": _axis_tail_metrics(metrics["by_target_z"]),
    }
    probes = None
    if run_probes:
        probes = _train_full_support_probes(
            residuals, torch.as_tensor(centers, dtype=torch.long), run
        )
        metrics["probe_metrics"] = probes["layers"]
        atomic_write_json(_probe_path(output.parent, exposure, phase), probes)
    result = {
        "kind": "cue_dose_full_support_exact_evaluation",
        "run_id": run.run_id,
        "condition_id": run.condition_id,
        "model_initialization_seed": run.model_initialization_seed,
        "exposed_pairs": exposure,
        "phase": phase,
        "training_distribution_before_5M": run.pre_switch_support,
        "evaluation_distribution": "uniform_full_support_66049_pairs",
        "checkpoint": str(checkpoint) if checkpoint is not None else None,
        "checkpoint_sha256": file_sha256(checkpoint) if checkpoint is not None else None,
        "model_output_sha256": output_hash,
        "full_support": metrics,
        "probes_run": run_probes,
        "checkpoint_training_diagnostics": checkpoint_diagnostics or {},
        "finite": bool(np.isfinite(kl).all()),
    }
    atomic_write_json(output, result)
    return result


def _gradient_batches(run: CueDoseRun, device: torch.device):
    rng = np.random.default_rng(run.gradient_audit_seed)
    for _ in range(32):
        linear = rng.integers(0, 257 * 257, size=512, dtype=np.int64)
        pairs_np = np.stack((linear // 257, linear % 257), axis=1)
        centers_np = _pair_centers(pairs_np, run.coefficients, run.p)
        yield (
            torch.as_tensor(pairs_np, dtype=torch.long, device=device),
            torch.as_tensor(centers_np, dtype=torch.long, device=device),
            torch.ones(512, dtype=torch.bool, device=device),
        )


def audit_cue_gradient(
    model,
    run: CueDoseRun,
    exposure: int,
    phase: str,
    output: Path,
    *,
    device: torch.device,
    evaluation: dict[str, Any],
) -> dict[str, Any]:
    model.train()
    stats = _accumulate(
        model,
        _gradient_batches(run, device),
        run.model_kind,
        DEFAULT_CONFIG,
    )
    result = {
        "kind": "cue_dose_no_training_full_support_gradient_audit",
        "trained_during_audit": False,
        "optimizer_steps": 0,
        "run_id": run.run_id,
        "condition_id": run.condition_id,
        "model_initialization_seed": run.model_initialization_seed,
        "exposed_pairs": exposure,
        "phase": phase,
        "audit_batches": 32,
        "pairs_per_batch": 512,
        "audit_distribution": "fixed_independent_uniform_full_support",
        "metric_definitions": {
            "gradient_variance": "mean(||g_i - mean(g)||^2)",
            "gradient_snr": "||mean(g)||^2 / mean(||g_i - mean(g)||^2)",
            "mean_pairwise_gradient_cosine": (
                "mean over all unordered pairs of per-batch unit gradient vectors"
            ),
        },
        "behavior": {
            key: evaluation["full_support"][key]
            for key in (
                "R",
                "R_bootstrap_95_ci",
                "center_state_top1_accuracy",
                "mean_probability_on_center_state",
                "mean_predictive_entropy_nats",
            )
        },
        **stats,
    }
    atomic_write_json(output, result)
    return result


def _save_cue_checkpoint(
    path: Path,
    model,
    optimizer,
    run: CueDoseRun,
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
        "version": "kepler-v3.5-cue-dose",
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
        "training_sampled_nll_nats": sampled_nll,
        "optimization_soft_cross_entropy_nats": soft_loss,
        "checkpoint_diagnostics": diagnostics,
        "flags": {
            "exploratory": True,
            "diagnostic_only": True,
            "excluded_from_confirmatory_scaling_fit": True,
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


def _frozen_replay_equality(
    run: CueDoseRun,
    exposure: int,
    model,
    optimizer,
    rng: np.random.Generator,
    pair_counts: np.ndarray,
) -> dict[str, Any]:
    source_path = _source_checkpoint(run, exposure)
    source = torch.load(source_path, map_location="cpu", weights_only=False)
    checks = {
        "model": _tree_sha256(model.state_dict()) == _tree_sha256(source["model"]),
        "optimizer": _tree_sha256(optimizer.state_dict())
        == _tree_sha256(source["optimizer"]),
        "numpy_rng": canonical_hash(rng.bit_generator.state)
        == canonical_hash(source["numpy_stream_state"]),
        "pair_counts": bool(
            np.array_equal(pair_counts, np.asarray(source["pair_frequency_counts"]))
        ),
        "torch_cpu_rng": bool(
            torch.equal(torch.get_rng_state().cpu(), source["torch_cpu_rng_state"].cpu())
        ),
        "absolute_exposure": int(source["exposed_examples"]) == exposure,
    }
    result = {
        "source_checkpoint": str(source_path),
        "source_checkpoint_sha256": file_sha256(source_path),
        "checks": checks,
        "passed": all(checks.values()),
    }
    if not result["passed"]:
        raise AssertionError(f"frozen C0 replay diverged at {exposure}: {checks}")
    return result


def _assert_manifest(run: CueDoseRun, manifest_path: str | Path) -> str:
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    if manifest.get("kind") != "full_support_cue_dose_causal_gate":
        raise ValueError("wrong cue-dose manifest kind")
    if not all(manifest["flags"].values()):
        raise ValueError("cue-dose runs must be diagnostic and excluded")
    registered = {item["run_id"]: item for item in manifest["runs"]}[run.run_id]
    if canonical_hash(registered) != canonical_hash(asdict(run)):
        raise ValueError("runtime cue-dose run differs from frozen manifest")
    return file_sha256(manifest_path)


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
    model,
    optimizer,
    run: CueDoseRun,
    directory: Path,
    pre_checkpoint: Path,
    device: torch.device,
    checkpoint_diagnostics: dict[str, Any],
) -> None:
    cpu_state = torch.get_rng_state().clone()
    mps_state = (
        torch.mps.get_rng_state().cpu().clone()
        if device.type == "mps" and hasattr(torch.mps, "get_rng_state")
        else None
    )
    python_state = random.getstate()
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
    gradient = audit_cue_gradient(
        model,
        run,
        5_000_000,
        "pre_switch",
        _gradient_path(directory, 5_000_000, "pre_switch"),
        device=device,
        evaluation=pre_eval,
    )
    pre_eval["gradient_audit_file"] = str(
        _gradient_path(directory, 5_000_000, "pre_switch")
    )
    pre_eval["gradient_global_total"] = gradient["groups"]["global.total"]
    atomic_write_json(_evaluation_path(directory, 5_000_000, "pre_switch"), pre_eval)

    post_checkpoint = _checkpoint_path(directory, 5_000_000, "post_switch")
    if post_checkpoint.exists():
        post_checkpoint.unlink()
    os.link(pre_checkpoint, post_checkpoint)
    post_checksum = post_checkpoint.with_suffix(post_checkpoint.suffix + ".sha256")
    post_checksum.write_text(
        f"{file_sha256(post_checkpoint)}  {post_checkpoint.name}\n", encoding="utf-8"
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
    equality = {
        "optimizer_steps_between_evaluations": 0,
        "model_state_sha256_before": model_hash_before,
        "model_state_sha256_after": model_hash_after,
        "optimizer_state_sha256_before": optimizer_hash_before,
        "optimizer_state_sha256_after": optimizer_hash_after,
        "model_state_unchanged": model_hash_before == model_hash_after,
        "optimizer_state_unchanged": optimizer_hash_before == optimizer_hash_after,
        "full_support_output_sha256_equal": (
            pre_eval["model_output_sha256"] == post_eval["model_output_sha256"]
        ),
        "sampler_before": run.pre_switch_support,
        "sampler_after": "uniform_full",
    }
    equality["passed"] = all(
        equality[key]
        for key in (
            "model_state_unchanged",
            "optimizer_state_unchanged",
            "full_support_output_sha256_equal",
        )
    )
    if not equality["passed"]:
        raise AssertionError("zero-step switch changed model outputs or state")
    post_eval["zero_step_switch_invariance"] = equality
    atomic_write_json(_evaluation_path(directory, 5_000_000, "post_switch"), post_eval)
    atomic_write_json(directory / "switch_invariance.json", equality)
    _restore_training_rng(cpu_state, mps_state, python_state, device)
    model.train()


def train_cue_dose_run(
    run: CueDoseRun,
    run_root: str | Path = "artifacts/cue_dose_runs",
    *,
    manifest_path: str | Path = "artifacts/cue_dose_manifest.json",
    device_name: str = "auto",
    resume: str | Path | None = None,
    cfg: ExperimentConfig = DEFAULT_CONFIG,
) -> Path:
    manifest_hash = _assert_manifest(run, manifest_path)
    device = resolve_device(device_name)
    torch.manual_seed(run.model_initialization_seed)
    np.random.seed(run.model_initialization_seed)
    random.seed(run.model_initialization_seed)
    model = make_pair_model(run, cfg).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.optimizer.learning_rate,
        betas=cfg.optimizer.betas,
        weight_decay=cfg.optimizer.weight_decay,
    )
    rng = np.random.default_rng(run.data_seed)
    pair_counts = np.zeros((run.p, run.p), dtype=np.int64)
    exposed = 0
    replay_source = None
    if run.execution_mode == "frozen_G1_reuse_with_deterministic_replay_from_2M":
        replay_source = _source_checkpoint(run, 2_000_000)
        state = torch.load(replay_source, map_location=device, weights_only=False)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        exposed = int(state["exposed_examples"])
        rng.bit_generator.state = state["numpy_stream_state"]
        pair_counts = np.asarray(state["pair_frequency_counts"], dtype=np.int64).copy()
        torch.set_rng_state(state["torch_cpu_rng_state"].cpu())
        if device.type == "mps" and state.get("torch_mps_rng_state") is not None:
            torch.mps.set_rng_state(state["torch_mps_rng_state"].cpu())
    if resume is not None:
        state = torch.load(resume, map_location=device, weights_only=False)
        if state["run"]["run_id"] != run.run_id:
            raise ValueError("resume checkpoint belongs to another cue-dose run")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        exposed = int(state["exposed_examples"])
        rng.bit_generator.state = state["numpy_stream_state"]
        pair_counts = np.asarray(state["pair_frequency_counts"], dtype=np.int64)
        torch.set_rng_state(state["torch_cpu_rng_state"].cpu())
        if device.type == "mps" and state.get("torch_mps_rng_state") is not None:
            torch.mps.set_rng_state(state["torch_mps_rng_state"].cpu())
        if state.get("python_rng_state") is not None:
            random.setstate(state["python_rng_state"])
    if exposed >= run.maximum_pair_exposures:
        raise ValueError("cue-dose run is already at its registered endpoint")

    directory = Path(run_root) / run.run_id
    directory.mkdir(parents=True, exist_ok=True)
    atomic_write_json(
        directory / "run.json",
        {
            "run": asdict(run),
            "device": str(device),
            "manifest_sha256": manifest_hash,
            "cue_distribution_audit_sha256": file_sha256(
                "artifacts/cue_distribution_audit.json"
            ),
            "optimizer": cfg.to_dict()["optimizer"],
            "replay_source": str(replay_source) if replay_source else None,
        },
    )
    save_points = [point for point in run.checkpoints if point > exposed]
    metrics_path = directory / "train.jsonl"
    model.train()
    with metrics_path.open("a", encoding="utf-8") as metrics:
        while exposed < run.maximum_pair_exposures:
            boundary = min(point for point in save_points if point > exposed)
            count = min(run.batch_size, boundary - exposed)
            active_lambda = run.lambda_value if exposed < run.switch_exposure else 0.0
            pairs_np, linear = sample_cue_pairs(
                rng, count, active_lambda, run.m
            )
            centers_np = _pair_centers(pairs_np, run.coefficients, run.p)
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
            record = {
                "exposed_pairs": exposed,
                "active_lambda": active_lambda,
                "sampling_distribution": (
                    run.pre_switch_support if active_lambda > 0.0 else "uniform_full"
                ),
                "optimization_soft_cross_entropy_nats": float(loss.detach().cpu()),
                "sampled_training_nll_nats": float(sampled_nll.detach().cpu()),
                "gradient_norm": float(grad_norm.detach().cpu()),
                "learning_rate": lr,
                "step_seconds": elapsed,
                "pairs_per_second": count / max(elapsed, 1.0e-12),
            }
            metrics.write(json.dumps(record, sort_keys=True) + "\n")
            metrics.flush()
            os.fsync(metrics.fileno())
            if exposed in save_points:
                phase = "pre_switch" if exposed == 5_000_000 else "standard"
                path = _checkpoint_path(directory, exposed, phase)
                replay_equality = None
                if run.reused_source_run_id and exposed in (5_000_000, 10_000_000):
                    replay_equality = _frozen_replay_equality(
                        run, exposed, model, optimizer, rng, pair_counts
                    )
                diagnostics = {
                    "per_block_gradient_norm": gradients,
                    "per_block_parameter_update_over_parameter_norm": ratios,
                    "manifest_sha256": manifest_hash,
                    "active_lambda_for_completed_step": active_lambda,
                    "optimizer_reset": False,
                    "scheduler_reset": False,
                    "frozen_G1_replay_equality": replay_equality,
                }
                _save_cue_checkpoint(
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
                    )
    return directory


def _load_model_for_evaluation(
    run: CueDoseRun,
    directory: Path,
    exposure: int,
    phase: str,
    device: torch.device,
    cfg: ExperimentConfig,
):
    torch.manual_seed(run.model_initialization_seed)
    model = make_pair_model(run, cfg).to(device)
    if exposure == 0:
        return model, None, {}
    if run.reused_source_run_id and exposure <= 2_000_000:
        checkpoint = _source_checkpoint(run, exposure)
    else:
        checkpoint = _checkpoint_path(directory, exposure, phase)
    state = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(state["model"])
    return model, checkpoint, state.get("checkpoint_diagnostics", {})


def evaluate_all_cue_checkpoints(
    run: CueDoseRun,
    directory: Path,
    *,
    device_name: str = "auto",
    cfg: ExperimentConfig = DEFAULT_CONFIG,
) -> list[dict[str, Any]]:
    device = resolve_device(device_name)
    points = [(0, "initialization")]
    for exposure in run.checkpoints:
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
            if exposure in GRADIENT_AUDIT_CHECKPOINTS and phase != "post_switch":
                gradient_path = _gradient_path(directory, exposure, phase)
                gradient = audit_cue_gradient(
                    model,
                    run,
                    exposure,
                    phase,
                    gradient_path,
                    device=device,
                    evaluation=evaluation,
                )
                evaluation["gradient_audit_file"] = str(gradient_path)
                evaluation["gradient_global_total"] = gradient["groups"][
                    "global.total"
                ]
                atomic_write_json(output, evaluation)
            del model
            if device.type == "mps":
                torch.mps.empty_cache()
        layer6 = (
            evaluation["full_support"].get("probe_metrics", {}).get("6")
        )
        summary = {
            "checkpoint_id": checkpoint_id,
            "run_id": run.run_id,
            "condition_id": run.condition_id,
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


def run_cue_dose(
    run_id: str,
    *,
    run_root: str | Path = "artifacts/cue_dose_runs",
    manifest_path: str | Path = "artifacts/cue_dose_manifest.json",
    audit_path: str | Path = "artifacts/cue_distribution_audit.json",
    device_name: str = "auto",
    resume: str | Path | None = None,
    cfg: ExperimentConfig = DEFAULT_CONFIG,
) -> dict[str, Any]:
    run = find_cue_dose_run(run_id, audit_path)
    directory = train_cue_dose_run(
        run,
        run_root,
        manifest_path=manifest_path,
        device_name=device_name,
        resume=resume,
        cfg=cfg,
    )
    evaluations = evaluate_all_cue_checkpoints(
        run, directory, device_name=device_name, cfg=cfg
    )
    return {
        "run_id": run.run_id,
        "completed_exposure": run.maximum_pair_exposures,
        "run_directory": str(directory.resolve()),
        "new_evaluations": evaluations,
    }


def cue_dose_status(
    run_root: str | Path = "artifacts/cue_dose_runs",
    audit_path: str | Path = "artifacts/cue_distribution_audit.json",
) -> list[dict[str, Any]]:
    from .cue_dose import build_cue_dose_runs

    root = Path(run_root)
    output = []
    for run in build_cue_dose_runs(audit_path):
        directory = root / run.run_id
        final_checkpoint = _checkpoint_path(directory, 10_000_000)
        expected_evaluations = 1 + len(run.checkpoints) + 1
        evaluations = 0
        if (directory / "evaluations.jsonl").exists():
            evaluations = len(
                [
                    line
                    for line in (directory / "evaluations.jsonl")
                    .read_text(encoding="utf-8")
                    .splitlines()
                    if line.strip()
                ]
            )
        output.append(
            {
                "run_id": run.run_id,
                "condition_id": run.condition_id,
                "model_initialization_seed": run.model_initialization_seed,
                "execution_mode": run.execution_mode,
                "completed_exposure": (
                    10_000_000 if final_checkpoint.exists() else 0
                ),
                "evaluations": evaluations,
                "expected_evaluations": expected_evaluations,
                "status": (
                    "complete"
                    if final_checkpoint.exists() and evaluations == expected_evaluations
                    else "pending"
                ),
            }
        )
    return output
