from __future__ import annotations

from dataclasses import asdict
import json
import math
from pathlib import Path
import random
import time
from typing import Any

import numpy as np
import torch
from torch import Tensor
from torch.nn import functional as F

from .config import DEFAULT_CONFIG, ExperimentConfig
from .diagnostic_models import ConcatenatedParentMLP, ParentExposedTransformer
from .durability import atomic_write_json, durable_torch_save, transactional_append_jsonl
from .environment import canonical_hash, file_sha256, git_commit
from .information import predictable_information, torch_exact_conditional_kl
from .mechanism import MechanismRun
from .model import size_spec
from .topology import sample_noise
from .training import learning_rate_at, resolve_device


def make_pair_model(run: MechanismRun, cfg: ExperimentConfig = DEFAULT_CONFIG):
    if run.model_kind == "parent_exposed_transformer":
        width = size_spec(2_000_000, cfg.model, cfg.field.p).width
        return ParentExposedTransformer(width, cfg.model, cfg.field.p)
    if run.model_kind == "concatenated_parent_mlp":
        return ConcatenatedParentMLP(vocab_size=cfg.field.p)
    raise ValueError(f"not a parent-pair model: {run.model_kind}")


def _pair_centers(pairs: np.ndarray, coefficients: tuple[int, int], p: int) -> np.ndarray:
    return np.asarray(
        (pairs[:, 0] * coefficients[0] + pairs[:, 1] * coefficients[1]) % p,
        dtype=np.int64,
    )


def _sample_pair_batch(
    rng: np.random.Generator,
    count: int,
    coefficients: tuple[int, int],
    p: int,
    eta: float,
):
    pairs = rng.integers(0, p, size=(count, 2), dtype=np.int64)
    centers = _pair_centers(pairs, coefficients, p)
    labels = (centers + sample_noise(rng, count, p=p, eta=eta)) % p
    return pairs, centers, labels


def _soft_loss(logits: Tensor, centers: Tensor, p: int, eta: float) -> Tensor:
    log_probs = F.log_softmax(logits.float(), dim=-1)
    q = torch.full_like(log_probs, eta / (p - 1))
    q.scatter_(-1, centers.long().unsqueeze(-1), 1.0 - eta)
    return -torch.sum(q * log_probs, dim=-1).mean()


def _diagnostic_group(name: str) -> str:
    if name.startswith("blocks."):
        return "block_" + name.split(".", 2)[1]
    if name.startswith("layers."):
        return "mlp_layer_" + name.split(".", 2)[1]
    if "embedding" in name or name == "query_embedding":
        return "embedding_interface"
    if name.startswith("final_norm"):
        return "final_norm"
    if name.startswith(("output_head", "lm_head")):
        return "output_head"
    return "other"


def _gradient_norms(model) -> dict[str, float]:
    sums: dict[str, Tensor] = {}
    for name, parameter in model.named_parameters():
        if parameter.grad is None:
            continue
        group = _diagnostic_group(name)
        value = torch.sum(parameter.grad.detach().float() ** 2)
        sums[group] = sums.get(group, torch.zeros((), device=value.device)) + value
    return {name: float(torch.sqrt(value).cpu()) for name, value in sums.items()}


def _snapshot(model) -> dict[str, Tensor]:
    return {name: parameter.detach().clone() for name, parameter in model.named_parameters()}


def _update_ratios(model, before: dict[str, Tensor]) -> dict[str, float]:
    changes: dict[str, Tensor] = {}
    magnitudes: dict[str, Tensor] = {}
    for name, parameter in model.named_parameters():
        group = _diagnostic_group(name)
        current = parameter.detach().float()
        changes[group] = changes.get(group, torch.zeros((), device=current.device)) + torch.sum(
            (current - before[name].float()) ** 2
        )
        magnitudes[group] = magnitudes.get(
            group, torch.zeros((), device=current.device)
        ) + torch.sum(current**2)
    return {
        name: float((torch.sqrt(changes[name]) / torch.sqrt(magnitudes[name]).clamp_min(1e-30)).cpu())
        for name in changes
    }


def _model_metadata(model, run: MechanismRun) -> dict[str, Any]:
    if isinstance(model, ConcatenatedParentMLP):
        return model.metadata()
    return {
        "kind": run.model_kind,
        "width": model.width,
        "n_layers": model.cfg.n_layers,
        "n_heads": model.cfg.n_heads,
        "core_parameters": model.core_parameter_count(),
        "total_parameters": model.total_parameter_count(),
        "tie_embeddings": True,
        "fixed_query_position": 2,
    }


def _save_pair_checkpoint(
    path: Path,
    model,
    optimizer,
    run: MechanismRun,
    exposed: int,
    rng: np.random.Generator,
    pair_counts: np.ndarray,
    learning_rate: float,
    training_nll: float,
    optimization_loss: float,
    diagnostics: dict[str, Any],
    cfg: ExperimentConfig,
) -> str:
    payload = {
        "version": "kepler-v3.3-mechanism-pair-diagnostic",
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
        "training_sampled_nll_nats": training_nll,
        "optimization_soft_cross_entropy_nats": optimization_loss,
        "checkpoint_diagnostics": diagnostics,
        "flags": {
            "exploratory": True,
            "diagnostic_only": True,
            "excluded_from_confirmatory_fit": True,
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


def train_pair_diagnostic(
    run: MechanismRun,
    run_root: str | Path,
    *,
    device_name: str = "auto",
    batch_size: int = 4096,
    resume: str | Path | None = None,
    stop_after_examples: int | None = None,
    cfg: ExperimentConfig = DEFAULT_CONFIG,
) -> Path:
    if run.model_kind not in {"parent_exposed_transformer", "concatenated_parent_mlp"}:
        raise ValueError("pair diagnostic runner only accepts M3/M4")
    endpoint = stop_after_examples or run.maximum_exposure
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
    pair_counts = np.zeros((cfg.field.p, cfg.field.p), dtype=np.int64)
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
    run_dir = Path(run_root) / run.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(
        run_dir / "run.json",
        {
            "run": asdict(run),
            "device": str(device),
            "batch_size": batch_size,
            "exposure_definition": "independently sampled parent-pair examples",
            "optimizer": cfg.to_dict()["optimizer"],
            "heldout_definition": "empirically zero-frequency pairs at each checkpoint",
        },
    )
    coefficients = tuple(int(v) for v in run.rule["coefficients"])
    metrics_path = run_dir / "train.jsonl"
    model.train()
    with metrics_path.open("a", encoding="utf-8") as metrics:
        while exposed < endpoint:
            boundary = min(point for point in save_points if point > exposed)
            count = min(batch_size, boundary - exposed)
            pairs_np, centers_np, labels_np = _sample_pair_batch(
                rng, count, coefficients, cfg.field.p, cfg.field.eta
            )
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
            loss = _soft_loss(logits, centers, cfg.field.p, cfg.field.eta)
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
                "exposed_examples": exposed,
                "optimization_soft_cross_entropy_nats": float(loss.detach().cpu()),
                "sampled_training_nll_nats": float(sampled_nll.detach().cpu()),
                "gradient_norm": float(grad_norm.detach().cpu()),
                "learning_rate": lr,
                "step_seconds": elapsed,
                "examples_per_second": count / max(elapsed, 1e-12),
            }
            metrics.write(json.dumps(record, sort_keys=True) + "\n")
            metrics.flush()
            if exposed in save_points:
                _save_pair_checkpoint(
                    run_dir / f"checkpoint_{exposed}.pt",
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
                    },
                    cfg,
                )
    return run_dir


def _frequency_diagnostics(
    counts: np.ndarray,
    correct: np.ndarray,
    r_values: np.ndarray,
) -> tuple[dict[str, Any], dict[str, Any], str]:
    flat = counts.reshape(-1)
    positive = flat[flat > 0]
    q25, q75 = np.quantile(positive, [0.25, 0.75]) if positive.size else (0, 0)
    groups = {
        "unseen": flat == 0,
        "low_frequency": (flat > 0) & (flat <= q25),
        "middle_frequency": (flat > q25) & (flat <= q75),
        "high_frequency": flat > q75,
    }
    by_frequency = {}
    for name, mask in groups.items():
        by_frequency[name] = {
            "pairs": int(mask.sum()),
            "min_training_frequency": int(flat[mask].min()) if mask.any() else None,
            "max_training_frequency": int(flat[mask].max()) if mask.any() else None,
            "center_accuracy": float(correct[mask].mean()) if mask.any() else None,
            "mean_R": float(r_values[mask].mean()) if mask.any() else None,
        }
    unseen = groups["unseen"]
    heldout = {
        "definition": "all full-support pairs with zero empirical training frequency",
        "pairs": int(unseen.sum()),
        "available": bool(unseen.any()),
        "center_accuracy": float(correct[unseen].mean()) if unseen.any() else None,
        "mean_R": float(r_values[unseen].mean()) if unseen.any() else None,
        "secondary_only": True,
    }
    if unseen.sum() >= 20:
        if float(correct[unseen].mean()) >= 0.8:
            conclusion = "structured_rule_evidence"
        elif float(correct[unseen].mean()) <= 0.05:
            conclusion = "lookup_or_no_rule_evidence"
        else:
            conclusion = "inconclusive_on_unseen_pairs"
    else:
        conclusion = "inconclusive_insufficient_unseen_pairs"
    return by_frequency, heldout, conclusion


@torch.no_grad()
def evaluate_pair_checkpoint(
    checkpoint: str | Path,
    run: MechanismRun,
    *,
    device_name: str = "auto",
    batch_size: int = 8192,
    bootstrap_replicates: int = 2000,
    cfg: ExperimentConfig = DEFAULT_CONFIG,
) -> dict[str, Any]:
    device = resolve_device(device_name)
    state = torch.load(checkpoint, map_location=device, weights_only=False)
    model = make_pair_model(run, cfg).to(device)
    model.load_state_dict(state["model"])
    model.eval()
    p = cfg.field.p
    grid = np.stack(np.meshgrid(np.arange(p), np.arange(p), indexing="ij"), axis=-1).reshape(-1, 2)
    coefficients = tuple(int(value) for value in run.rule["coefficients"])
    centers_np = _pair_centers(grid, coefficients, p)
    all_kl: list[np.ndarray] = []
    all_correct: list[np.ndarray] = []
    probability_sum = 0.0
    entropy_sum = 0.0
    for start in range(0, grid.shape[0], batch_size):
        stop = min(grid.shape[0], start + batch_size)
        pairs = torch.as_tensor(grid[start:stop], dtype=torch.long, device=device)
        centers = torch.as_tensor(centers_np[start:stop], dtype=torch.long, device=device)
        logits = model(pairs).float()
        log_probs = F.log_softmax(logits, dim=-1)
        probabilities = log_probs.exp()
        kl = torch_exact_conditional_kl(log_probs, centers, p, cfg.field.eta)
        all_kl.append(kl.cpu().numpy())
        all_correct.append((logits.argmax(dim=-1) == centers).cpu().numpy())
        probability_sum += float(
            probabilities.gather(-1, centers[:, None]).sum().cpu()
        )
        entropy_sum += float((-torch.sum(probabilities * log_probs, dim=-1)).sum().cpu())
    kl_values = np.concatenate(all_kl)
    correct = np.concatenate(all_correct)
    information = predictable_information(p, cfg.field.eta)
    r_values = kl_values / information
    row_r = r_values.reshape(p, p).mean(axis=1)
    rng = np.random.default_rng(run.evaluation_seed + int(state["exposed_examples"]))
    bootstrap = np.empty(bootstrap_replicates, dtype=np.float64)
    for index in range(bootstrap_replicates):
        bootstrap[index] = float(row_r[rng.integers(0, p, size=p)].mean())
    lower, upper = np.quantile(bootstrap, [0.025, 0.975])
    fresh_pairs, fresh_centers, fresh_labels = _sample_pair_batch(
        rng, 65_536, coefficients, p, cfg.field.eta
    )
    fresh_nll_sum = 0.0
    for start in range(0, fresh_pairs.shape[0], batch_size):
        stop = min(fresh_pairs.shape[0], start + batch_size)
        logits = model(torch.as_tensor(fresh_pairs[start:stop], dtype=torch.long, device=device))
        labels = torch.as_tensor(fresh_labels[start:stop], dtype=torch.long, device=device)
        fresh_nll_sum += float(F.cross_entropy(logits.float(), labels, reduction="sum").cpu())
    by_frequency, heldout, conclusion = _frequency_diagnostics(
        np.asarray(state["pair_frequency_counts"]), correct, r_values
    )
    diagnostics = state.get("checkpoint_diagnostics", {})
    result = {
        "run_id": run.run_id,
        "checkpoint": str(Path(checkpoint).resolve()),
        "exposed_examples": int(state["exposed_examples"]),
        "mean_exact_conditional_kl_nats": float(kl_values.mean()),
        "R": float(r_values.mean()),
        "R_bootstrap_95_ci": [float(lower), float(upper)],
        "bootstrap_unit": "first-parent rows of the exact 257x257 support",
        "bootstrap_replicates": bootstrap_replicates,
        "center_state_top1_accuracy": float(correct.mean()),
        "full_support_parent_pair_accuracy": float(correct.mean()),
        "mean_probability_on_center_state": probability_sum / grid.shape[0],
        "mean_predictive_entropy_nats": entropy_sum / grid.shape[0],
        "sampled_training_nll_nats": state["training_sampled_nll_nats"],
        "fresh_sample_nll_nats": fresh_nll_sum / fresh_pairs.shape[0],
        "full_support_pairs": int(grid.shape[0]),
        "performance_by_parent_pair_frequency": by_frequency,
        "heldout_pair_secondary_diagnostic": heldout,
        "structured_vs_lookup_conclusion": conclusion,
        "per_block_gradient_norm": diagnostics.get("per_block_gradient_norm", {}),
        "per_block_parameter_update_over_parameter_norm": diagnostics.get(
            "per_block_parameter_update_over_parameter_norm", {}
        ),
        "finite": bool(np.isfinite(kl_values).all()),
    }
    atomic_write_json(Path(checkpoint).with_suffix(".eval.json"), result)
    return result


@torch.no_grad()
def _extract_pair_probe_data(model, seed: int, examples: int, layers: tuple[int, ...], device: str):
    from .probes import ProbeDataset

    rng = np.random.default_rng(seed)
    pairs = rng.integers(0, 257, size=(examples, 2), dtype=np.int64)
    centers = _pair_centers(pairs, (45, 36), 257)
    tensors = torch.as_tensor(pairs, dtype=torch.long, device=device)
    _, residuals = model(tensors, return_residuals=True)
    labels = torch.as_tensor(centers, dtype=torch.long)
    return {layer: ProbeDataset(residuals[layer].float().cpu(), labels) for layer in layers}


def run_pair_checkpoint_probes(
    checkpoint: str | Path,
    run: MechanismRun,
    output: str | Path,
    *,
    device_name: str = "auto",
    cfg: ExperimentConfig = DEFAULT_CONFIG,
) -> dict[str, Any]:
    from .probes import (
        FourierStateProbe,
        LinearStateProbe,
        MLPStateProbe,
        train_classification_probe,
        train_fourier_probe,
    )

    device = resolve_device(device_name)
    state = torch.load(checkpoint, map_location=device, weights_only=False)
    model = make_pair_model(run, cfg).to(device)
    model.load_state_dict(state["model"])
    model.eval()
    split_sizes = {"train": 4096, "validation": 1024, "test": 1024}
    split_seeds = {
        "train": run.evaluation_seed + 1001,
        "validation": run.evaluation_seed + 2001,
        "test": run.evaluation_seed + 3001,
    }
    data = {
        split: _extract_pair_probe_data(
            model, split_seeds[split], examples, run.probe_layers, str(device)
        )
        for split, examples in split_sizes.items()
    }
    layer_results = {}
    for layer in run.probe_layers:
        train, validation, test = (data[name][layer] for name in ("train", "validation", "test"))
        width = train.residuals.shape[-1]
        seed = run.evaluation_seed + layer
        torch.manual_seed(seed)
        linear = train_classification_probe(
            LinearStateProbe(width), train, validation, test, seed=seed
        )
        torch.manual_seed(seed + 1)
        fourier = train_fourier_probe(
            FourierStateProbe(width), train, validation, test, seed=seed + 1
        )
        torch.manual_seed(seed + 2)
        mlp = train_classification_probe(
            MLPStateProbe(width, hidden=256), train, validation, test, seed=seed + 2
        )
        layer_results[str(layer)] = {
            "linear": linear,
            "fixed_fourier": fourier,
            "fixed_capacity_mlp": {"hidden_width": 256, **mlp},
        }
    result = {
        "run_id": run.run_id,
        "exposed_examples": int(state["exposed_examples"]),
        "probe_layers": list(run.probe_layers),
        "split_examples": split_sizes,
        "split_seeds": split_seeds,
        "layers": layer_results,
    }
    atomic_write_json(output, result)
    return result


def evaluate_all_pair_checkpoints(
    run_dir: Path,
    run: MechanismRun,
    endpoint: int,
    *,
    device_name: str,
    cfg: ExperimentConfig = DEFAULT_CONFIG,
) -> list[dict[str, Any]]:
    evaluations_file = run_dir / "evaluations.jsonl"
    existing = set()
    if evaluations_file.exists():
        existing = {
            int(json.loads(line)["exposed_examples"])
            for line in evaluations_file.read_text(encoding="utf-8").splitlines()
            if line.strip()
        }
    output = []
    for exposure in run.checkpoints:
        if exposure > endpoint or exposure in existing:
            continue
        result = evaluate_pair_checkpoint(
            run_dir / f"checkpoint_{exposure}.pt", run, device_name=device_name, cfg=cfg
        )
        transactional_append_jsonl(evaluations_file, result)
        output.append(result)
    for exposure in run.probe_checkpoints:
        if exposure > endpoint:
            continue
        path = run_dir / f"checkpoint_{exposure}.probes.json"
        if not path.exists():
            run_pair_checkpoint_probes(
                run_dir / f"checkpoint_{exposure}.pt",
                run,
                path,
                device_name=device_name,
                cfg=cfg,
            )
    return output
