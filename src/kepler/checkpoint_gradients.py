from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor

from .config import DEFAULT_CONFIG, ExperimentConfig
from .data import OnlineEpisodeStream, split_seed
from .durability import atomic_write_json
from .gradient_audit import (
    _OnlineGradientMoments,
    _semantic_gradient_groups,
    _soft_population_loss,
)
from .mechanism import build_mechanism_isolation_runs
from .model import make_model
from .pair_diagnostics import _pair_centers, _soft_loss, make_pair_model
from .topology import FiniteFieldRule
from .training import resolve_device


def _mlp_gradient_groups(model) -> dict[str, Tensor]:
    def flatten(values):
        return torch.cat([value.detach().float().reshape(-1) for value in values])

    embedding = [
        parameter.grad
        for name, parameter in model.named_parameters()
        if name.startswith(("parent1_embedding", "parent2_embedding"))
    ]
    output = [parameter.grad for parameter in model.output_head.parameters()]
    mlp = [parameter.grad for layer in model.layers for parameter in layer.parameters()]
    normalization = [parameter.grad for parameter in model.final_norm.parameters()]
    result = {
        "global.embedding_input": flatten(embedding),
        "global.output_head_direct": flatten(output),
        "global.mlp": flatten(mlp),
        "global.normalization": flatten(normalization),
    }
    for index, layer in enumerate(model.layers):
        result[f"mlp_layer_{index}.total"] = flatten(
            [parameter.grad for parameter in layer.parameters()]
        )
    return result


def _probe_summary(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {
        layer: {
            "linear_test_accuracy": values["linear"]["test"]["accuracy"],
            "fixed_fourier_test_r2": values["fixed_fourier"]["test"][
                "variance_normalized_r2"
            ],
            "fixed_capacity_mlp_test_accuracy": values["fixed_capacity_mlp"]["test"][
                "accuracy"
            ],
        }
        for layer, values in payload["layers"].items()
    }


def _evaluation_summary(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {
        key: payload[key]
        for key in (
            "R",
            "R_bootstrap_95_ci",
            "mean_exact_conditional_kl_nats",
            "center_state_top1_accuracy",
            "mean_probability_on_center_state",
            "mean_predictive_entropy_nats",
        )
    }


def _accumulate(
    model,
    batches,
    model_kind: str,
    cfg: ExperimentConfig,
) -> dict[str, Any]:
    accumulators: dict[str, _OnlineGradientMoments] = {}
    losses = []
    for parents_or_inputs, centers, mask in batches:
        model.zero_grad(set_to_none=True)
        if model_kind == "concatenated_parent_mlp":
            logits = model(parents_or_inputs).float()
            loss = _soft_loss(logits, centers, cfg.field.p, cfg.field.eta)
            loss.backward()
            groups = _mlp_gradient_groups(model)
        else:
            logits, residuals = model(parents_or_inputs, return_residuals=True)
            loss = _soft_population_loss(
                logits, centers, mask, cfg.field.p, cfg.field.eta
            )
            loss.backward()
            groups = _semantic_gradient_groups(
                model, logits, residuals, centers, mask, cfg.field.p, cfg.field.eta
            )
        losses.append(float(loss.detach().cpu()))
        if not accumulators:
            accumulators = {name: _OnlineGradientMoments() for name in groups}
        for name, vector in groups.items():
            accumulators[name].update(vector)
    summaries: dict[str, Any] = {
        name: value.summary() for name, value in accumulators.items()
    }
    if model_kind == "concatenated_parent_mlp":
        for name in ("global.attention_q", "global.attention_k", "global.attention_v"):
            summaries[name] = {
                "applicable": False,
                "reason": "M4 has no attention module",
            }
    return {
        "mean_exact_soft_cross_entropy_nats": float(np.mean(losses)),
        "groups": summaries,
    }


def run_checkpoint_gradient_dynamics_audit(
    output: str | Path = "artifacts/checkpoint_gradient_dynamics_audit.json",
    *,
    run_root: str | Path = "artifacts/mechanism_runs",
    device_name: str = "auto",
    batches_per_checkpoint: int = 32,
    recurrent_episode_batch_size: int = 8,
    pair_batch_size: int = 512,
    cfg: ExperimentConfig = DEFAULT_CONFIG,
) -> dict[str, Any]:
    if batches_per_checkpoint != 32:
        raise ValueError("registered checkpoint audit requires 32 fixed batches")
    device = resolve_device(device_name)
    root = Path(run_root)
    output_payload: dict[str, Any] = {
        "kind": "checkpoint_exact_soft_gradient_dynamics",
        "trained": False,
        "optimizer_steps": 0,
        "metric_definitions": {
            "gradient_variance": "mean(||g_i - mean(g)||^2)",
            "gradient_snr": "||mean(g)||^2 / mean(||g_i - mean(g)||^2)",
            "mean_pairwise_gradient_cosine": (
                "mean over all unordered pairs of per-batch unit gradient vectors"
            ),
        },
        "batches_per_checkpoint": batches_per_checkpoint,
        "fixed_batches_reused_across_checkpoints_within_each_task": True,
        "audit_batches_independent_of_training_evaluation_and_probes": True,
        "runs": {},
    }
    for run_index, run in enumerate(build_mechanism_isolation_runs(cfg)):
        run_dir = root / run.run_id
        audit_seed = split_seed(
            96_000 + run_index,
            int(run.rule["k"]),
            int(run.rule["d"]),
            int(run.rule["instance"]),
            17,
        )
        run_result: dict[str, Any] = {
            "model_kind": run.model_kind,
            "audit_seed": audit_seed,
            "checkpoints": {},
        }
        for exposure in run.checkpoints:
            checkpoint = run_dir / f"checkpoint_{exposure}.pt"
            state = torch.load(checkpoint, map_location=device, weights_only=False)
            if run.model_kind == "recurrent_transformer":
                model = make_model(run.target_core_parameters, cfg.model, cfg.field.p).to(device)
                model.load_state_dict(state["model"])
                rule_dict = dict(run.rule)
                rule_dict["lags"] = tuple(rule_dict["lags"])
                rule_dict["coefficients"] = tuple(rule_dict["coefficients"])
                rule = FiniteFieldRule(**rule_dict)
                stream = OnlineEpisodeStream(
                    rule,
                    audit_seed,
                    cfg.model.visible_prefix_length,
                    cfg.model.context_length - cfg.model.visible_prefix_length,
                    cfg.field.d_max,
                )

                def batches():
                    for _ in range(batches_per_checkpoint):
                        batch = stream.batch(recurrent_episode_batch_size)
                        inputs, _, centers, mask = batch.to_torch(str(device))
                        yield inputs, centers, mask

                batch_iterator = batches()
                eval_path = checkpoint.with_suffix(".eval.json")
            else:
                model = make_pair_model(run, cfg).to(device)
                model.load_state_dict(state["model"])
                rng = np.random.default_rng(audit_seed)

                def batches():
                    for _ in range(batches_per_checkpoint):
                        pairs = rng.integers(
                            0, cfg.field.p, size=(pair_batch_size, 2), dtype=np.int64
                        )
                        centers = _pair_centers(pairs, (45, 36), cfg.field.p)
                        yield (
                            torch.as_tensor(pairs, dtype=torch.long, device=device),
                            torch.as_tensor(centers, dtype=torch.long, device=device),
                            torch.ones(pair_batch_size, dtype=torch.bool, device=device),
                        )

                batch_iterator = batches()
                eval_path = checkpoint.with_suffix(".eval.json")
            model.train()
            stats = _accumulate(model, batch_iterator, run.model_kind, cfg)
            stats.update(
                {
                    "exposure": exposure,
                    "evaluation": _evaluation_summary(eval_path),
                    "probes": _probe_summary(checkpoint.with_suffix(".probes.json")),
                }
            )
            run_result["checkpoints"][str(exposure)] = stats
            del model
            if device.type == "mps":
                torch.mps.empty_cache()
            atomic_write_json(output, output_payload | {"runs": {**output_payload["runs"], run.run_id: run_result}})
        output_payload["runs"][run.run_id] = run_result
        atomic_write_json(output, output_payload)
    return output_payload
