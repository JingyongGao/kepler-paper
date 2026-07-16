from __future__ import annotations

import json
from pathlib import Path
import random
from typing import Any

import numpy as np
import torch

from .checkpoint_gradients import _accumulate
from .compositional import CompositionalRun, build_compositional_runs
from .config import DEFAULT_CONFIG, ExperimentConfig
from .data import split_seed
from .durability import atomic_write_json
from .environment import file_sha256
from .pair_diagnostics import _pair_centers, make_pair_model
from .parent_pair_split import load_parent_pair_split
from .training import resolve_device


AUDIT_EXPOSURES = (0, 500_000, 2_000_000, 5_000_000, 10_000_000, 40_000_000)
GRADIENT_METRICS = (
    "norm_of_mean_gradient",
    "gradient_variance_mean_squared_deviation",
    "gradient_snr",
    "mean_pairwise_gradient_cosine",
    "mean_batch_gradient_norm",
    "rms_batch_gradient_norm",
)


def _fixed_linear_batches(
    seed: int,
    support: str,
    train_indices: np.ndarray,
    *,
    batches: int = 32,
    batch_size: int = 512,
    p: int = 257,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    if support == "full":
        return rng.integers(0, p * p, size=(batches, batch_size), dtype=np.int64)
    if support == "train":
        positions = rng.integers(
            0, train_indices.size, size=(batches, batch_size), dtype=np.int64
        )
        return train_indices[positions]
    raise ValueError("audit support must be full or train")


def _torch_batches(
    linear_batches: np.ndarray,
    device: torch.device,
    coefficients: tuple[int, int],
    p: int,
):
    for linear in linear_batches:
        pairs_np = np.stack((linear // p, linear % p), axis=1)
        centers_np = _pair_centers(pairs_np, coefficients, p)
        yield (
            torch.as_tensor(pairs_np, dtype=torch.long, device=device),
            torch.as_tensor(centers_np, dtype=torch.long, device=device),
            torch.ones(linear.size, dtype=torch.bool, device=device),
        )


def _paired_differences(g1: dict[str, Any], g2: dict[str, Any]) -> dict[str, Any]:
    common_groups = sorted(set(g1["groups"]) & set(g2["groups"]))
    return {
        "direction": "G2_minus_G1",
        "mean_exact_soft_cross_entropy_nats": (
            g2["mean_exact_soft_cross_entropy_nats"]
            - g1["mean_exact_soft_cross_entropy_nats"]
        ),
        "groups": {
            group: {
                metric: float(g2["groups"][group][metric])
                - float(g1["groups"][group][metric])
                for metric in GRADIENT_METRICS
            }
            for group in common_groups
        },
    }


def _behavior_summary(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    partitions = {}
    for name in ("train_support", "heldout", "full"):
        values = payload["partitions"][name]
        partitions[name] = {
            key: values[key]
            for key in (
                "R",
                "R_bootstrap_95_ci",
                "mean_exact_conditional_kl_nats",
                "center_state_top1_accuracy",
                "mean_probability_on_center_state",
                "mean_predictive_entropy_nats",
            )
        }
        partitions[name]["layer_6_probes"] = values["probe_metrics"]["6"]
    return {
        "evaluation_file": str(path),
        "evaluation_sha256": file_sha256(path),
        "partitions": partitions,
    }


def _load_model(
    run: CompositionalRun,
    exposure: int,
    root: Path,
    device: torch.device,
    cfg: ExperimentConfig,
) -> tuple[torch.nn.Module, dict[str, Any]]:
    torch.manual_seed(run.model_initialization_seed)
    np.random.seed(run.model_initialization_seed)
    random.seed(run.model_initialization_seed)
    model = make_pair_model(run, cfg).to(device)
    if exposure == 0:
        return model, {
            "source": "deterministic_reconstruction_before_any_optimizer_step",
            "checkpoint": None,
            "checkpoint_sha256": None,
            "behavior": None,
        }
    checkpoint = root / run.run_id / f"checkpoint_{exposure}.pt"
    state = torch.load(checkpoint, map_location=device, weights_only=False)
    if state["run"]["run_id"] != run.run_id:
        raise ValueError("checkpoint/run mismatch")
    model.load_state_dict(state["model"])
    return model, {
        "source": "frozen_checkpoint",
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": file_sha256(checkpoint),
        "behavior": _behavior_summary(
            checkpoint.with_suffix(".composition.eval.json")
        ),
    }


def _max_common_initialization_difference(differences: dict[str, Any]) -> float:
    values = [abs(float(differences["mean_exact_soft_cross_entropy_nats"]))]
    for metrics in differences["groups"].values():
        values.extend(abs(float(value)) for value in metrics.values())
    return max(values)


def run_support_gradient_dynamics_audit(
    output: str | Path = "artifacts/support_gradient_dynamics_audit.json",
    *,
    run_root: str | Path = "artifacts/compositional_runs",
    split_path: str | Path = "artifacts/parent_pair_split.json",
    device_name: str = "auto",
    batches_per_set: int = 32,
    batch_size: int = 512,
    cfg: ExperimentConfig = DEFAULT_CONFIG,
) -> dict[str, Any]:
    if batches_per_set != 32 or batch_size != 512:
        raise ValueError("registered support-gradient audit requires 32x512 batches")
    device = resolve_device(device_name)
    root = Path(run_root)
    train_indices, _, heldout_mask = load_parent_pair_split(split_path)
    run_lookup = {run.run_id: run for run in build_compositional_runs(cfg)}
    result: dict[str, Any] = {
        "kind": "paired_support_gradient_dynamics_audit",
        "trained": False,
        "optimizer_steps": 0,
        "p": cfg.field.p,
        "eta": cfg.field.eta,
        "batches_per_audit_set": batches_per_set,
        "pairs_per_batch": batch_size,
        "audit_exposures": list(AUDIT_EXPOSURES),
        "split_sha256": file_sha256(split_path),
        "metric_definitions": {
            "gradient_variance": "mean(||g_i - mean(g)||^2)",
            "gradient_snr": "||mean(g)||^2 / mean(||g_i - mean(g)||^2)",
            "mean_pairwise_gradient_cosine": (
                "mean over all unordered pairs of per-batch unit gradient vectors"
            ),
            "paired_difference": "G2 metric minus paired G1 metric",
        },
        "audit_streams": {
            "support_specific": (
                "independent frozen full-support batches for G1 and frozen "
                "training-support batches for G2"
            ),
            "common_full_support": (
                "one additional frozen full-support batch set reused for both "
                "models in a seed and across all checkpoints"
            ),
        },
        "initialization_reconstruction": (
            "torch, NumPy, and Python seeds are reset to the registered model seed "
            "before constructing the unchanged parent-exposed Transformer"
        ),
        "seeds": {},
    }
    for seed in (0, 1, 2):
        support_seed = split_seed(121_000, 2, 2, 0, seed)
        common_seed = split_seed(122_000, 2, 2, 0, seed)
        support_batches = {
            "G1": _fixed_linear_batches(support_seed, "full", train_indices),
            "G2": _fixed_linear_batches(support_seed, "train", train_indices),
        }
        common_batches = _fixed_linear_batches(common_seed, "full", train_indices)
        if bool(heldout_mask[support_batches["G2"]].any()):
            raise AssertionError("G2 support-specific audit set leaked")
        seed_result: dict[str, Any] = {
            "support_specific_seed": support_seed,
            "common_full_support_seed": common_seed,
            "checkpoints": {},
        }
        for exposure in AUDIT_EXPOSURES:
            point: dict[str, Any] = {"exposure": exposure, "arms": {}}
            for arm in ("G1", "G2"):
                run = run_lookup[f"composition_{arm}_s{seed}"]
                model, source = _load_model(run, exposure, root, device, cfg)
                model.train()
                specific_stats = _accumulate(
                    model,
                    _torch_batches(
                        support_batches[arm], device, run.coefficients, run.p
                    ),
                    run.model_kind,
                    cfg,
                )
                common_stats = _accumulate(
                    model,
                    _torch_batches(common_batches, device, run.coefficients, run.p),
                    run.model_kind,
                    cfg,
                )
                point["arms"][arm] = {
                    **source,
                    "support_specific_distribution": (
                        "full" if arm == "G1" else "frozen_G2_training_support"
                    ),
                    "support_specific": specific_stats,
                    "common_full_support": common_stats,
                }
                del model
                if device.type == "mps":
                    torch.mps.empty_cache()
            point["paired_differences"] = {
                "support_specific": _paired_differences(
                    point["arms"]["G1"]["support_specific"],
                    point["arms"]["G2"]["support_specific"],
                ),
                "common_full_support": _paired_differences(
                    point["arms"]["G1"]["common_full_support"],
                    point["arms"]["G2"]["common_full_support"],
                ),
            }
            if exposure == 0:
                maximum = _max_common_initialization_difference(
                    point["paired_differences"]["common_full_support"]
                )
                point["identical_initialization_common_full_check"] = {
                    "maximum_absolute_metric_difference": maximum,
                    "tolerance": 1.0e-10,
                    "passed": maximum < 1.0e-10,
                }
                if maximum >= 1.0e-10:
                    raise AssertionError("paired initial models are not identical")
            seed_result["checkpoints"][str(exposure)] = point
            result["seeds"][str(seed)] = seed_result
            atomic_write_json(output, result)
        result["seeds"][str(seed)] = seed_result
    atomic_write_json(output, result)
    return result
