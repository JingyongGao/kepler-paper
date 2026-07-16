from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .compositional import CompositionalRun
from .config import DEFAULT_CONFIG, ExperimentConfig
from .durability import atomic_write_json, transactional_append_jsonl
from .information import predictable_information, torch_exact_conditional_kl
from .pair_diagnostics import make_pair_model
from .parent_pair_split import load_parent_pair_split
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
from .training import resolve_device


def _bootstrap_rows(
    r_values: np.ndarray,
    selected: np.ndarray,
    x1: np.ndarray,
    seed: int,
    replicates: int = 2_000,
) -> list[float]:
    row_means = np.asarray(
        [r_values[selected & (x1 == value)].mean() for value in range(257)],
        dtype=np.float64,
    )
    if not np.isfinite(row_means).all():
        raise AssertionError("every first-parent row must occur in each partition")
    rng = np.random.default_rng(seed)
    draws = np.empty(replicates, dtype=np.float64)
    for index in range(replicates):
        draws[index] = float(row_means[rng.integers(0, 257, size=257)].mean())
    lower, upper = np.quantile(draws, [0.025, 0.975])
    return [float(lower), float(upper)]


def _axis_metrics(
    group_values: np.ndarray,
    selected: np.ndarray,
    r_values: np.ndarray,
    correct: np.ndarray,
    center_probability: np.ndarray,
) -> dict[str, list[float] | list[int]]:
    counts, mean_r, accuracy, mean_probability = [], [], [], []
    for value in range(257):
        mask = selected & (group_values == value)
        counts.append(int(mask.sum()))
        mean_r.append(float(r_values[mask].mean()))
        accuracy.append(float(correct[mask].mean()))
        mean_probability.append(float(center_probability[mask].mean()))
    return {
        "counts": counts,
        "R": mean_r,
        "center_accuracy": accuracy,
        "mean_center_probability": mean_probability,
    }


def _axis_tail_metrics(axis: dict[str, list[float] | list[int]]) -> dict[str, Any]:
    values = np.asarray(axis["R"], dtype=np.float64)
    order = np.argsort(values)[::-1]
    return {
        "R_quantiles": {
            "q50": float(np.quantile(values, 0.50)),
            "q90": float(np.quantile(values, 0.90)),
            "q95": float(np.quantile(values, 0.95)),
            "q99": float(np.quantile(values, 0.99)),
            "maximum": float(values.max()),
        },
        "worst_10_values": [
            {"value": int(index), "R": float(values[index])} for index in order[:10]
        ],
    }


def _basic_partition_metrics(
    name: str,
    selected: np.ndarray,
    kl: np.ndarray,
    r_values: np.ndarray,
    correct: np.ndarray,
    center_probability: np.ndarray,
    entropy: np.ndarray,
    x1: np.ndarray,
    bootstrap_seed: int,
) -> dict[str, Any]:
    return {
        "partition": name,
        "pairs": int(selected.sum()),
        "mean_exact_conditional_kl_nats": float(kl[selected].mean()),
        "R": float(r_values[selected].mean()),
        "R_bootstrap_95_ci": _bootstrap_rows(
            r_values, selected, x1, bootstrap_seed
        ),
        "bootstrap_replicates": 2_000,
        "bootstrap_unit": "first-parent rows",
        "center_state_top1_accuracy": float(correct[selected].mean()),
        "mean_probability_on_center_state": float(center_probability[selected].mean()),
        "mean_predictive_entropy_nats": float(entropy[selected].mean()),
    }


def _frequency_metrics(
    frequency: np.ndarray,
    selected: np.ndarray,
    r_values: np.ndarray,
    correct: np.ndarray,
) -> dict[str, Any]:
    positive = frequency[frequency > 0]
    q25, q75 = np.quantile(positive, [0.25, 0.75]) if positive.size else (0, 0)
    bins = {
        "zero": frequency == 0,
        "low": (frequency > 0) & (frequency <= q25),
        "middle": (frequency > q25) & (frequency <= q75),
        "high": frequency > q75,
    }
    output = {}
    for name, condition in bins.items():
        mask = selected & condition
        output[name] = {
            "pairs": int(mask.sum()),
            "min_frequency": int(frequency[mask].min()) if mask.any() else None,
            "max_frequency": int(frequency[mask].max()) if mask.any() else None,
            "R": float(r_values[mask].mean()) if mask.any() else None,
            "center_accuracy": float(correct[mask].mean()) if mask.any() else None,
        }
    return output


def _partition_probe_metrics(
    probe,
    kind: str,
    residuals: torch.Tensor,
    centers: torch.Tensor,
    mask: np.ndarray,
) -> dict[str, float]:
    torch_mask = torch.from_numpy(mask)
    dataset = ProbeDataset(residuals[torch_mask], centers[torch_mask])
    if kind == "fourier":
        return _fourier_metrics(probe, dataset)
    return _classification_metrics(probe, dataset)


def _train_partition_probes(
    residuals: dict[int, torch.Tensor],
    centers: torch.Tensor,
    masks: dict[str, np.ndarray],
    run: CompositionalRun,
) -> dict[str, Any]:
    rng = np.random.default_rng(run.probe_seed)
    train_pool = np.flatnonzero(masks["frozen_train"])
    order = rng.permutation(train_pool)
    train_indices = order[:4096]
    validation_indices = order[4096:5120]
    train_mask = np.zeros(257 * 257, dtype=np.bool_)
    validation_mask = np.zeros_like(train_mask)
    train_mask[train_indices] = True
    validation_mask[validation_indices] = True
    output: dict[str, Any] = {
        "probe_train_partition": "frozen_train_only",
        "probe_train_examples": 4096,
        "probe_validation_examples": 1024,
        "probe_seed": run.probe_seed,
        "layers": {},
    }
    for layer, values in residuals.items():
        train_mask_t = torch.from_numpy(train_mask)
        validation_mask_t = torch.from_numpy(validation_mask)
        train = ProbeDataset(values[train_mask_t], centers[train_mask_t])
        validation = ProbeDataset(values[validation_mask_t], centers[validation_mask_t])
        width = values.shape[-1]
        seed = run.probe_seed + layer
        torch.manual_seed(seed)
        linear_probe = LinearStateProbe(width)
        train_classification_probe(
            linear_probe, train, validation, validation, seed=seed
        )
        torch.manual_seed(seed + 1)
        fourier_probe = FourierStateProbe(width)
        train_fourier_probe(
            fourier_probe, train, validation, validation, seed=seed + 1
        )
        torch.manual_seed(seed + 2)
        mlp_probe = MLPStateProbe(width, hidden=256)
        train_classification_probe(
            mlp_probe, train, validation, validation, seed=seed + 2
        )
        partitions = {}
        for name in ("train_support", "frozen_train", "heldout", "full"):
            partitions[name] = {
                "linear": _partition_probe_metrics(
                    linear_probe, "classification", values, centers, masks[name]
                ),
                "fixed_fourier": _partition_probe_metrics(
                    fourier_probe, "fourier", values, centers, masks[name]
                ),
                "fixed_capacity_mlp": {
                    "hidden_width": 256,
                    **_partition_probe_metrics(
                        mlp_probe, "classification", values, centers, masks[name]
                    ),
                },
            }
        output["layers"][str(layer)] = {"partitions": partitions}
    return output


@torch.no_grad()
def _forward_full_support(model, device: torch.device, batch_size: int = 8192):
    grid = np.stack(
        np.meshgrid(np.arange(257), np.arange(257), indexing="ij"), axis=-1
    ).reshape(-1, 2)
    centers_np = (45 * grid[:, 0] + 36 * grid[:, 1]) % 257
    kl_parts, correct_parts, probability_parts, entropy_parts = [], [], [], []
    residual_parts: dict[int, list[torch.Tensor]] = {0: [], 3: [], 6: []}
    for start in range(0, grid.shape[0], batch_size):
        stop = min(grid.shape[0], start + batch_size)
        parents = torch.as_tensor(grid[start:stop], dtype=torch.long, device=device)
        centers = torch.as_tensor(centers_np[start:stop], dtype=torch.long, device=device)
        logits, residuals = model(parents, return_residuals=True)
        logits = logits.float()
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
    )


def evaluate_compositional_checkpoint(
    checkpoint: str | Path,
    run: CompositionalRun,
    *,
    split_path: str | Path = "artifacts/parent_pair_split.json",
    device_name: str = "auto",
    output_label: str = "composition",
    cfg: ExperimentConfig = DEFAULT_CONFIG,
) -> dict[str, Any]:
    device = resolve_device(device_name)
    state = torch.load(checkpoint, map_location=device, weights_only=False)
    model = make_pair_model(run, cfg).to(device)
    model.load_state_dict(state["model"])
    model.eval()
    grid, centers, kl, correct, center_probability, entropy, residuals = _forward_full_support(
        model, device
    )
    information = predictable_information(257, 0.1)
    r_values = kl / information
    train_indices, heldout_indices, heldout_mask = load_parent_pair_split(split_path)
    frozen_train_mask = ~heldout_mask
    full_mask = np.ones(257 * 257, dtype=np.bool_)
    train_support_mask = full_mask if run.sampling_support == "full" else frozen_train_mask
    masks = {
        "train_support": train_support_mask,
        "frozen_train": frozen_train_mask,
        "heldout": heldout_mask,
        "full": full_mask,
    }
    probes = _train_partition_probes(
        residuals, torch.as_tensor(centers, dtype=torch.long), masks, run
    )
    frequency = np.asarray(state["pair_frequency_counts"], dtype=np.int64).reshape(-1)
    allowed = full_mask if run.sampling_support == "full" else frozen_train_mask
    leakage_exposures = int(frequency[~allowed].sum())
    leakage_pairs = int(np.count_nonzero(frequency[~allowed]))
    if leakage_exposures != 0:
        raise AssertionError("pair leakage count must be zero")
    partitions = {}
    for offset, (name, mask) in enumerate(masks.items()):
        metrics = _basic_partition_metrics(
            name,
            mask,
            kl,
            r_values,
            correct,
            center_probability,
            entropy,
            grid[:, 0],
            run.evaluation_seed + int(state["exposed_examples"]) + offset,
        )
        metrics["by_x1"] = _axis_metrics(
            grid[:, 0], mask, r_values, correct, center_probability
        )
        metrics["by_x2"] = _axis_metrics(
            grid[:, 1], mask, r_values, correct, center_probability
        )
        metrics["by_target_z"] = _axis_metrics(
            centers, mask, r_values, correct, center_probability
        )
        metrics["tail_summary"] = {
            "by_x1": _axis_tail_metrics(metrics["by_x1"]),
            "by_x2": _axis_tail_metrics(metrics["by_x2"]),
            "by_target_z": _axis_tail_metrics(metrics["by_target_z"]),
        }
        metrics["by_pair_frequency"] = _frequency_metrics(
            frequency, mask, r_values, correct
        )
        metrics["probe_metrics"] = {
            layer: values["partitions"][name]
            for layer, values in probes["layers"].items()
        }
        partitions[name] = metrics
    train_r = partitions["train_support"]["R"]
    generalization_ratio = (
        (1.0 - partitions["heldout"]["R"]) / (1.0 - train_r)
        if train_r < 0.95
        else None
    )
    result = {
        "run_id": run.run_id,
        "arm": run.arm,
        "model_initialization_seed": run.model_initialization_seed,
        "checkpoint": str(Path(checkpoint).resolve()),
        "exposed_pairs": int(state["exposed_examples"]),
        "sampling_support": run.sampling_support,
        "partitions": partitions,
        "G": generalization_ratio,
        "G_definition": "(1 - R_heldout) / (1 - R_train_support)",
        "G_reported": generalization_ratio is not None,
        "pair_leakage_count": leakage_exposures,
        "pair_leakage_distinct_pairs": leakage_pairs,
        "late_transition_indicator": partitions["full"]["R"] < 0.95,
        "probe_protocol": {
            key: probes[key]
            for key in (
                "probe_train_partition",
                "probe_train_examples",
                "probe_validation_examples",
                "probe_seed",
            )
        },
        "training_checkpoint_diagnostics": state.get("checkpoint_diagnostics", {}),
        "finite": bool(np.isfinite(kl).all()),
    }
    output = Path(checkpoint).with_suffix(f".{output_label}.eval.json")
    atomic_write_json(output, result)
    atomic_write_json(Path(checkpoint).with_suffix(f".{output_label}.probes.json"), probes)
    return result


def evaluate_all_compositional_checkpoints(
    directory: Path,
    run: CompositionalRun,
    endpoint: int,
    *,
    split_path: str | Path,
    device_name: str,
    cfg: ExperimentConfig = DEFAULT_CONFIG,
) -> list[dict[str, Any]]:
    summaries_path = directory / "evaluations.jsonl"
    existing = set()
    if summaries_path.exists():
        existing = {
            int(json.loads(line)["exposed_pairs"])
            for line in summaries_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        }
    new_results = []
    for exposure in run.checkpoints:
        if exposure > endpoint or exposure in existing:
            continue
        result = evaluate_compositional_checkpoint(
            directory / f"checkpoint_{exposure}.pt",
            run,
            split_path=split_path,
            device_name=device_name,
            cfg=cfg,
        )
        summary = {
            "run_id": run.run_id,
            "arm": run.arm,
            "model_initialization_seed": run.model_initialization_seed,
            "exposed_pairs": exposure,
            "R_train_support": result["partitions"]["train_support"]["R"],
            "R_heldout": result["partitions"]["heldout"]["R"],
            "R_full": result["partitions"]["full"]["R"],
            "G": result["G"],
            "pair_leakage_count": result["pair_leakage_count"],
            "late_transition_indicator": result["late_transition_indicator"],
            "full_metrics_file": str(
                (directory / f"checkpoint_{exposure}.composition.eval.json").resolve()
            ),
        }
        transactional_append_jsonl(summaries_path, summary)
        new_results.append(summary)
    return new_results
