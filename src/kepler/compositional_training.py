from __future__ import annotations

from dataclasses import asdict
import json
import os
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch.nn import functional as F

from .compositional import CompositionalRun
from .config import DEFAULT_CONFIG, ExperimentConfig
from .durability import atomic_write_json
from .environment import file_sha256
from .pair_diagnostics import (
    _gradient_norms,
    _pair_centers,
    _save_pair_checkpoint,
    _snapshot,
    _soft_loss,
    _update_ratios,
    make_pair_model,
)
from .parent_pair_split import load_parent_pair_split
from .topology import sample_noise
from .training import learning_rate_at, resolve_device


def _assert_manifest_and_split(
    manifest_path: str | Path, split_path: str | Path
) -> tuple[str, str]:
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    if manifest.get("kind") != "compositional_generalization_gate":
        raise ValueError("wrong compositional manifest kind")
    if not all(manifest["flags"].values()):
        raise ValueError("compositional runs are not fully diagnostic/excluded")
    split_hash = file_sha256(split_path)
    if split_hash != manifest["parent_pair_split"]["file_sha256"]:
        raise ValueError("frozen parent-pair split hash changed")
    return file_sha256(manifest_path), split_hash


def train_compositional_run(
    run: CompositionalRun,
    run_root: str | Path = "artifacts/compositional_runs",
    *,
    manifest_path: str | Path = "artifacts/compositional_generalization_manifest.json",
    split_path: str | Path = "artifacts/parent_pair_split.json",
    device_name: str = "auto",
    resume: str | Path | None = None,
    stop_after_exposures: int | None = None,
    cfg: ExperimentConfig = DEFAULT_CONFIG,
) -> Path:
    manifest_hash, split_hash = _assert_manifest_and_split(manifest_path, split_path)
    endpoint = stop_after_exposures or run.maximum_pair_exposures
    if endpoint not in run.checkpoints:
        raise ValueError("endpoint must be a registered checkpoint")
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
    train_indices, heldout_indices, heldout_mask = load_parent_pair_split(split_path)
    exposed = 0
    if resume is not None:
        state = torch.load(resume, map_location=device, weights_only=False)
        if state["run"]["run_id"] != run.run_id:
            raise ValueError("resume checkpoint belongs to another run")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        exposed = int(state["exposed_examples"])
        rng.bit_generator.state = state["numpy_stream_state"]
        pair_counts = np.asarray(state["pair_frequency_counts"], dtype=np.int64)
        torch.set_rng_state(state["torch_cpu_rng_state"].cpu())
        if device.type == "mps" and state.get("torch_mps_rng_state") is not None:
            torch.mps.set_rng_state(state["torch_mps_rng_state"].cpu())
    if endpoint <= exposed:
        raise ValueError("endpoint must exceed resumed exposure")
    save_points = [point for point in run.checkpoints if exposed < point <= endpoint]
    directory = Path(run_root) / run.run_id
    directory.mkdir(parents=True, exist_ok=True)
    atomic_write_json(
        directory / "run.json",
        {
            "run": asdict(run),
            "device": str(device),
            "manifest_sha256": manifest_hash,
            "split_sha256": split_hash,
            "split_assignment_sha256": json.loads(
                Path(split_path).read_text(encoding="utf-8")
            )["assignment_sha256"],
            "optimizer": cfg.to_dict()["optimizer"],
            "pair_leakage_definition": "exposure assigned to a pair outside sampling_support",
        },
    )
    metrics_path = directory / "train.jsonl"
    model.train()
    with metrics_path.open("a", encoding="utf-8") as metrics:
        while exposed < endpoint:
            boundary = min(point for point in save_points if point > exposed)
            count = min(run.batch_size, boundary - exposed)
            if run.sampling_support == "full":
                linear = rng.integers(0, run.p * run.p, size=count, dtype=np.int64)
            else:
                linear = train_indices[rng.integers(0, train_indices.size, size=count)]
            pairs_np = np.stack((linear // run.p, linear % run.p), axis=1)
            if run.sampling_support == "train" and bool(heldout_mask[linear].any()):
                raise AssertionError("G2 emitted a held-out pair")
            centers_np = _pair_centers(pairs_np, run.coefficients, run.p)
            labels_np = (
                centers_np + sample_noise(rng, count, p=run.p, eta=cfg.field.eta)
            ) % run.p
            np.add.at(pair_counts, (pairs_np[:, 0], pairs_np[:, 1]), 1)
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
            leakage_exposures = (
                int(pair_counts.reshape(-1)[heldout_indices].sum())
                if run.sampling_support == "train"
                else 0
            )
            if leakage_exposures:
                raise AssertionError("held-out exposure count became nonzero")
            record = {
                "exposed_pairs": exposed,
                "optimization_soft_cross_entropy_nats": float(loss.detach().cpu()),
                "sampled_training_nll_nats": float(sampled_nll.detach().cpu()),
                "gradient_norm": float(grad_norm.detach().cpu()),
                "learning_rate": lr,
                "step_seconds": elapsed,
                "pairs_per_second": count / max(elapsed, 1e-12),
                "pair_leakage_count": leakage_exposures,
            }
            metrics.write(json.dumps(record, sort_keys=True) + "\n")
            metrics.flush()
            os.fsync(metrics.fileno())
            if exposed in save_points:
                _save_pair_checkpoint(
                    directory / f"checkpoint_{exposed}.pt",
                    model,
                    optimizer,
                    run,
                    exposed,
                    rng,
                    pair_counts,
                    lr,
                    float(sampled_nll.detach().cpu()),
                    float(loss.detach().cpu()),
                    {
                        "per_block_gradient_norm": gradients,
                        "per_block_parameter_update_over_parameter_norm": ratios,
                        "pair_leakage_count": leakage_exposures,
                        "manifest_sha256": manifest_hash,
                        "split_sha256": split_hash,
                    },
                    cfg,
                )
    return directory
