from __future__ import annotations

from dataclasses import asdict, dataclass
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

from .compositional_evaluation import evaluate_compositional_checkpoint
from .config import DEFAULT_CONFIG, ExperimentConfig
from .durability import (
    atomic_write_json,
    durable_torch_save,
    transactional_append_jsonl,
)
from .environment import canonical_hash, file_sha256, git_commit
from .pair_diagnostics import (
    _gradient_norms,
    _model_metadata,
    _pair_centers,
    _snapshot,
    _soft_loss,
    _update_ratios,
    make_pair_model,
)
from .parent_pair_split import COEFFICIENTS, P, load_parent_pair_split
from .topology import sample_noise
from .training import learning_rate_at, resolve_device


TRANSFER_START = 5_000_000
TRANSFER_CHECKPOINTS = (5_500_000, 6_000_000, 7_500_000, 10_000_000, 20_000_000)


@dataclass(frozen=True)
class FullSupportTransferRun:
    run_id: str
    arm: str
    description: str
    model_kind: str
    target_core_parameters: int
    p: int
    coefficients: tuple[int, int]
    loss_type: str
    sampling_support: str
    pre_transfer_sampling_support: str
    model_initialization_seed: int
    data_seed: int
    evaluation_seed: int
    probe_seed: int
    source_run_id: str
    source_checkpoint: str
    source_checkpoint_sha256: str
    transfer_start_exposure: int
    maximum_pair_exposures: int
    checkpoints: tuple[int, ...]
    batch_size: int
    exploratory: bool = True
    diagnostic_only: bool = True
    excluded_from_confirmatory_scaling_fit: bool = True


def build_full_support_transfer_runs(
    source_root: str | Path = "artifacts/compositional_runs",
) -> tuple[FullSupportTransferRun, ...]:
    root = Path(source_root)
    runs = []
    compositional_manifest = json.loads(
        Path("artifacts/compositional_generalization_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    source_by_id = {run["run_id"]: run for run in compositional_manifest["runs"]}
    for seed in (0, 1, 2):
        source_id = f"composition_G2_s{seed}"
        source = root / source_id / f"checkpoint_{TRANSFER_START}.pt"
        registered = source_by_id[source_id]
        runs.append(
            FullSupportTransferRun(
                run_id=f"full_support_transfer_s{seed}",
                arm="T",
                description=(
                    "continue frozen G2 5M state after replacing only the pair "
                    "sampler with uniform full support"
                ),
                model_kind="parent_exposed_transformer",
                target_core_parameters=2_000_000,
                p=P,
                coefficients=COEFFICIENTS,
                loss_type="oracle_soft",
                sampling_support="full",
                pre_transfer_sampling_support="frozen_G2_training_support",
                model_initialization_seed=seed,
                data_seed=int(registered["data_seed"]),
                evaluation_seed=int(registered["evaluation_seed"]),
                probe_seed=int(registered["probe_seed"]),
                source_run_id=source_id,
                source_checkpoint=str(source),
                source_checkpoint_sha256=file_sha256(source),
                transfer_start_exposure=TRANSFER_START,
                maximum_pair_exposures=TRANSFER_CHECKPOINTS[-1],
                checkpoints=TRANSFER_CHECKPOINTS,
                batch_size=int(registered["batch_size"]),
            )
        )
    return tuple(runs)


def full_support_transfer_manifest_payload(
    source_root: str | Path = "artifacts/compositional_runs",
) -> dict[str, Any]:
    prior_paths = {
        "production": "artifacts/manifest.json",
        "pilot": "artifacts/pilot_manifest.json",
        "learnability": "artifacts/learnability_audit_manifest.json",
        "mechanism": "artifacts/mechanism_isolation_manifest.json",
        "composition": "artifacts/compositional_generalization_manifest.json",
    }
    return {
        "project": "Project Kepler v3.4",
        "kind": "full_support_transfer_gate",
        "created_from_git_commit": git_commit(Path.cwd()),
        "prior_manifests_modified": False,
        "prior_manifest_sha256": {
            name: file_sha256(path) for name, path in prior_paths.items()
        },
        "frozen_split_sha256": file_sha256("artifacts/parent_pair_split.json"),
        "flags": {
            "exploratory": True,
            "diagnostic_only": True,
            "excluded_from_confirmatory_scaling_fit": True,
        },
        "protocol": {
            "source_total_exposure": TRANSFER_START,
            "source_support": "frozen_G2_training_support",
            "post_transfer_support": "uniform_full_66049_pair_support",
            "only_sampler_replaced": True,
            "model_state_preserved": True,
            "optimizer_state_preserved_without_reset": True,
            "scheduler_state_and_absolute_exposure_preserved": True,
            "numpy_torch_cpu_torch_mps_rng_states_preserved": True,
            "python_rng_deterministically_reconstructed_because_unused_after_initialization": True,
            "objective": "exact_soft_label_cross_entropy",
            "checkpoints": list(TRANSFER_CHECKPOINTS),
            "control_comparison": (
                "use existing G1/G2 5M and 10M checkpoints to bracket 5.5M/6M/7.5M; "
                "use exact existing 10M and 20M controls at matching endpoints; no controls retrained"
            ),
        },
        "runs": [asdict(run) for run in build_full_support_transfer_runs(source_root)],
        "forbidden": [
            "production trajectories",
            "matched recurrent or objective experiments",
            "new support masks",
            "extension beyond 20M total exposure",
            "optimizer reset or global optimizer change",
        ],
    }


def write_full_support_transfer_manifest(
    output: str | Path = "artifacts/full_support_transfer_manifest.json",
    source_root: str | Path = "artifacts/compositional_runs",
) -> tuple[Path, str]:
    path = atomic_write_json(output, full_support_transfer_manifest_payload(source_root))
    return path, file_sha256(path)


def find_full_support_transfer_run(
    run_id: str, source_root: str | Path = "artifacts/compositional_runs"
) -> FullSupportTransferRun:
    for run in build_full_support_transfer_runs(source_root):
        if run.run_id == run_id:
            return run
    raise KeyError(f"unknown full-support transfer run: {run_id}")


def _array_sha256(values: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(values).tobytes(order="C")).hexdigest()


def _save_transfer_checkpoint(
    path: Path,
    model,
    optimizer,
    run: FullSupportTransferRun,
    exposed: int,
    rng: np.random.Generator,
    cumulative_pair_counts: np.ndarray,
    post_transfer_pair_counts: np.ndarray,
    source_pair_counts_sha256: str,
    learning_rate: float,
    training_nll: float,
    optimization_loss: float,
    diagnostics: dict[str, Any],
    cfg: ExperimentConfig,
) -> str:
    payload = {
        "version": "kepler-v3.4-full-support-transfer",
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
            "preserved_from_source_without_reset": True,
        },
        "transfer_state": {
            "source_run_id": run.source_run_id,
            "source_checkpoint": run.source_checkpoint,
            "source_checkpoint_sha256": run.source_checkpoint_sha256,
            "source_pair_counts_sha256": source_pair_counts_sha256,
            "transfer_start_exposure": TRANSFER_START,
            "pre_transfer_support": run.pre_transfer_sampling_support,
            "post_transfer_support": run.sampling_support,
            "post_transfer_exposures": exposed - TRANSFER_START,
            "optimizer_reset": False,
            "scheduler_reset": False,
        },
        "numpy_stream_state": rng.bit_generator.state,
        "pair_frequency_counts": cumulative_pair_counts,
        "post_transfer_pair_frequency_counts": post_transfer_pair_counts,
        "torch_cpu_rng_state": torch.get_rng_state(),
        "torch_mps_rng_state": (
            torch.mps.get_rng_state().cpu()
            if torch.backends.mps.is_available() and hasattr(torch.mps, "get_rng_state")
            else None
        ),
        "python_rng_state": random.getstate(),
        "training_sampled_nll_nats": training_nll,
        "optimization_soft_cross_entropy_nats": optimization_loss,
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
            "scheduler_state",
            "numpy_stream_state",
            "pair_frequency_counts",
            "post_transfer_pair_frequency_counts",
        },
    )


def _assert_manifest(
    run: FullSupportTransferRun, manifest_path: str | Path
) -> str:
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    if manifest.get("kind") != "full_support_transfer_gate":
        raise ValueError("wrong transfer manifest kind")
    if not all(manifest["flags"].values()):
        raise ValueError("transfer trajectories must be diagnostic and excluded")
    registered = {item["run_id"]: item for item in manifest["runs"]}[run.run_id]
    if canonical_hash(registered) != canonical_hash(asdict(run)):
        raise ValueError("runtime transfer run differs from frozen manifest")
    if file_sha256(run.source_checkpoint) != run.source_checkpoint_sha256:
        raise ValueError("frozen 5M G2 source checkpoint changed")
    return file_sha256(manifest_path)


def train_full_support_transfer(
    run: FullSupportTransferRun,
    run_root: str | Path = "artifacts/full_support_transfer_runs",
    *,
    manifest_path: str | Path = "artifacts/full_support_transfer_manifest.json",
    split_path: str | Path = "artifacts/parent_pair_split.json",
    device_name: str = "auto",
    resume: str | Path | None = None,
    stop_after_exposures: int | None = None,
    cfg: ExperimentConfig = DEFAULT_CONFIG,
) -> Path:
    manifest_hash = _assert_manifest(run, manifest_path)
    endpoint = stop_after_exposures or run.maximum_pair_exposures
    if endpoint not in run.checkpoints:
        raise ValueError("endpoint must be a registered transfer checkpoint")
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
    source_state = torch.load(run.source_checkpoint, map_location=device, weights_only=False)
    if source_state["run"]["run_id"] != run.source_run_id:
        raise ValueError("transfer source run mismatch")
    if int(source_state["exposed_examples"]) != TRANSFER_START:
        raise ValueError("transfer source must be the 5M checkpoint")
    if int(source_state["scheduler_state"]["supervised_examples_seen"]) != TRANSFER_START:
        raise ValueError("source scheduler exposure mismatch")
    model.load_state_dict(source_state["model"])
    optimizer.load_state_dict(source_state["optimizer"])
    rng = np.random.default_rng()
    rng.bit_generator.state = source_state["numpy_stream_state"]
    cumulative_counts = np.asarray(source_state["pair_frequency_counts"], dtype=np.int64).copy()
    source_counts_hash = _array_sha256(cumulative_counts)
    post_transfer_counts = np.zeros_like(cumulative_counts)
    exposed = TRANSFER_START
    torch.set_rng_state(source_state["torch_cpu_rng_state"].cpu())
    if device.type == "mps" and source_state.get("torch_mps_rng_state") is not None:
        torch.mps.set_rng_state(source_state["torch_mps_rng_state"].cpu())

    if resume is not None:
        state = torch.load(resume, map_location=device, weights_only=False)
        if state["run"]["run_id"] != run.run_id:
            raise ValueError("resume checkpoint belongs to another transfer run")
        if state["transfer_state"]["source_checkpoint_sha256"] != run.source_checkpoint_sha256:
            raise ValueError("resume source checkpoint hash mismatch")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        exposed = int(state["exposed_examples"])
        rng.bit_generator.state = state["numpy_stream_state"]
        cumulative_counts = np.asarray(state["pair_frequency_counts"], dtype=np.int64)
        post_transfer_counts = np.asarray(
            state["post_transfer_pair_frequency_counts"], dtype=np.int64
        )
        torch.set_rng_state(state["torch_cpu_rng_state"].cpu())
        if device.type == "mps" and state.get("torch_mps_rng_state") is not None:
            torch.mps.set_rng_state(state["torch_mps_rng_state"].cpu())
        if state.get("python_rng_state") is not None:
            random.setstate(state["python_rng_state"])
    if endpoint <= exposed:
        raise ValueError("endpoint must exceed current transfer exposure")

    save_points = [point for point in run.checkpoints if exposed < point <= endpoint]
    directory = Path(run_root) / run.run_id
    directory.mkdir(parents=True, exist_ok=True)
    atomic_write_json(
        directory / "run.json",
        {
            "run": asdict(run),
            "device": str(device),
            "manifest_sha256": manifest_hash,
            "split_sha256": file_sha256(split_path),
            "source_boundary_audit": {
                "source_checkpoint_sha256_verified": True,
                "model_state_loaded_without_reset": True,
                "optimizer_state_loaded_without_reset": True,
                "scheduler_absolute_exposure_preserved": True,
                "numpy_and_torch_rng_states_loaded": True,
                "only_sampler_replaced": True,
            },
            "optimizer": cfg.to_dict()["optimizer"],
        },
    )
    metrics_path = directory / "train.jsonl"
    model.train()
    with metrics_path.open("a", encoding="utf-8") as metrics:
        while exposed < endpoint:
            boundary = min(point for point in save_points if point > exposed)
            count = min(run.batch_size, boundary - exposed)
            linear = rng.integers(0, run.p * run.p, size=count, dtype=np.int64)
            pairs_np = np.stack((linear // run.p, linear % run.p), axis=1)
            centers_np = _pair_centers(pairs_np, run.coefficients, run.p)
            labels_np = (
                centers_np + sample_noise(rng, count, p=run.p, eta=cfg.field.eta)
            ) % run.p
            np.add.at(cumulative_counts, (pairs_np[:, 0], pairs_np[:, 1]), 1)
            np.add.at(post_transfer_counts, (pairs_np[:, 0], pairs_np[:, 1]), 1)
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
            if int(post_transfer_counts.sum()) != exposed - TRANSFER_START:
                raise AssertionError("post-transfer exposure accounting mismatch")
            record = {
                "total_exposed_pairs": exposed,
                "post_transfer_exposed_pairs": exposed - TRANSFER_START,
                "optimization_soft_cross_entropy_nats": float(loss.detach().cpu()),
                "sampled_training_nll_nats": float(sampled_nll.detach().cpu()),
                "gradient_norm": float(grad_norm.detach().cpu()),
                "learning_rate": lr,
                "step_seconds": elapsed,
                "pairs_per_second": count / max(elapsed, 1.0e-12),
                "sampling_support": "full",
            }
            metrics.write(json.dumps(record, sort_keys=True) + "\n")
            metrics.flush()
            os.fsync(metrics.fileno())
            if exposed in save_points:
                _save_transfer_checkpoint(
                    directory / f"checkpoint_{exposed}.pt",
                    model,
                    optimizer,
                    run,
                    exposed,
                    rng,
                    cumulative_counts,
                    post_transfer_counts,
                    source_counts_hash,
                    lr,
                    float(sampled_nll.detach().cpu()),
                    float(loss.detach().cpu()),
                    {
                        "per_block_gradient_norm": gradients,
                        "per_block_parameter_update_over_parameter_norm": ratios,
                        "manifest_sha256": manifest_hash,
                        "source_checkpoint_sha256": run.source_checkpoint_sha256,
                        "optimizer_reset": False,
                        "scheduler_reset": False,
                    },
                    cfg,
                )
    return directory


def _control_snapshot(run: FullSupportTransferRun, exposure: int) -> dict[str, Any]:
    if exposure in (10_000_000, 20_000_000):
        reference_exposures = [exposure]
        comparison_kind = "exact_exposure_match"
    else:
        reference_exposures = [5_000_000, 10_000_000]
        comparison_kind = "frozen_control_bracket_without_interpolation"
    controls = {}
    source_root = Path(run.source_checkpoint).parents[1]
    for arm in ("G1", "G2"):
        run_id = f"composition_{arm}_s{run.model_initialization_seed}"
        controls[arm] = {}
        for point in reference_exposures:
            path = source_root / run_id / f"checkpoint_{point}.composition.eval.json"
            payload = json.loads(path.read_text(encoding="utf-8"))
            controls[arm][str(point)] = {
                "evaluation_file": str(path),
                "evaluation_sha256": file_sha256(path),
                "R_full": payload["partitions"]["full"]["R"],
                "R_original_G2_train": payload["partitions"]["frozen_train"]["R"],
                "R_original_G2_heldout": payload["partitions"]["heldout"]["R"],
                "center_accuracy_full": payload["partitions"]["full"][
                    "center_state_top1_accuracy"
                ],
            }
    return {
        "comparison_kind": comparison_kind,
        "no_control_retraining": True,
        "controls": controls,
    }


def evaluate_transfer_checkpoint(
    checkpoint: str | Path,
    run: FullSupportTransferRun,
    *,
    split_path: str | Path = "artifacts/parent_pair_split.json",
    device_name: str = "auto",
    cfg: ExperimentConfig = DEFAULT_CONFIG,
) -> dict[str, Any]:
    checkpoint = Path(checkpoint)
    result = evaluate_compositional_checkpoint(
        checkpoint,
        run,
        split_path=split_path,
        device_name=device_name,
        output_label="transfer",
        cfg=cfg,
    )
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    _, _, heldout = load_parent_pair_split(split_path)
    post_counts = np.asarray(state["post_transfer_pair_frequency_counts"], dtype=np.int64).reshape(-1)
    result["registered_partitions"] = {
        "full_pair_space": "partitions.full",
        "original_G2_training_support": "partitions.frozen_train",
        "original_G2_heldout_support": "partitions.heldout",
    }
    result["transfer_diagnostics"] = {
        **state["transfer_state"],
        "post_transfer_original_train_exposures": int(post_counts[~heldout].sum()),
        "post_transfer_original_heldout_exposures": int(post_counts[heldout].sum()),
        "checkpoint_gradient_and_update_norms": state["checkpoint_diagnostics"],
    }
    result["frozen_control_comparison"] = _control_snapshot(
        run, int(state["exposed_examples"])
    )
    atomic_write_json(checkpoint.with_suffix(".transfer.eval.json"), result)
    return result


def run_full_support_transfer(
    run_id: str,
    *,
    run_root: str | Path = "artifacts/full_support_transfer_runs",
    source_root: str | Path = "artifacts/compositional_runs",
    manifest_path: str | Path = "artifacts/full_support_transfer_manifest.json",
    split_path: str | Path = "artifacts/parent_pair_split.json",
    device_name: str = "auto",
    resume: str | Path | None = None,
    stop_after_exposures: int | None = None,
    cfg: ExperimentConfig = DEFAULT_CONFIG,
) -> dict[str, Any]:
    run = find_full_support_transfer_run(run_id, source_root)
    directory = train_full_support_transfer(
        run,
        run_root,
        manifest_path=manifest_path,
        split_path=split_path,
        device_name=device_name,
        resume=resume,
        stop_after_exposures=stop_after_exposures,
        cfg=cfg,
    )
    endpoint = stop_after_exposures or run.maximum_pair_exposures
    summaries = directory / "evaluations.jsonl"
    existing = set()
    if summaries.exists():
        existing = {
            int(json.loads(line)["exposed_pairs"])
            for line in summaries.read_text(encoding="utf-8").splitlines()
            if line.strip()
        }
    new_evaluations = []
    for exposure in run.checkpoints:
        if exposure > endpoint or exposure in existing:
            continue
        result = evaluate_transfer_checkpoint(
            directory / f"checkpoint_{exposure}.pt",
            run,
            split_path=split_path,
            device_name=device_name,
            cfg=cfg,
        )
        summary = {
            "run_id": run.run_id,
            "model_initialization_seed": run.model_initialization_seed,
            "exposed_pairs": exposure,
            "post_transfer_exposed_pairs": exposure - TRANSFER_START,
            "R_full": result["partitions"]["full"]["R"],
            "R_original_G2_train": result["partitions"]["frozen_train"]["R"],
            "R_original_G2_heldout": result["partitions"]["heldout"]["R"],
            "center_accuracy_full": result["partitions"]["full"][
                "center_state_top1_accuracy"
            ],
            "full_metrics_file": str(
                (directory / f"checkpoint_{exposure}.transfer.eval.json").resolve()
            ),
        }
        transactional_append_jsonl(summaries, summary)
        new_evaluations.append(summary)
    return {
        "run_id": run.run_id,
        "completed_exposure": endpoint,
        "run_directory": str(directory.resolve()),
        "new_evaluations": new_evaluations,
    }


def full_support_transfer_status(
    run_root: str | Path = "artifacts/full_support_transfer_runs",
    source_root: str | Path = "artifacts/compositional_runs",
) -> list[dict[str, Any]]:
    root = Path(run_root)
    output = []
    for run in build_full_support_transfer_runs(source_root):
        directory = root / run.run_id
        checkpoints = [
            point for point in run.checkpoints if (directory / f"checkpoint_{point}.pt").exists()
        ]
        evaluations = [
            point
            for point in run.checkpoints
            if (directory / f"checkpoint_{point}.transfer.eval.json").exists()
        ]
        output.append(
            {
                "run_id": run.run_id,
                "model_initialization_seed": run.model_initialization_seed,
                "completed_exposure": max(checkpoints, default=TRANSFER_START),
                "target_exposure": run.maximum_pair_exposures,
                "checkpoints": checkpoints,
                "evaluations": evaluations,
                "status": (
                    "complete"
                    if checkpoints == list(run.checkpoints)
                    and evaluations == list(run.checkpoints)
                    else "pending"
                ),
            }
        )
    return output
