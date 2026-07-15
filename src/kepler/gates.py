from __future__ import annotations

from dataclasses import asdict
import hashlib
import math
from typing import Any

import numpy as np
import torch

from .config import DEFAULT_CONFIG, ExperimentConfig
from .data import OnlineEpisodeStream
from .model import KeplerTransformer, masked_language_model_loss
from .topology import FiniteFieldRule, TOPOLOGIES, noise_probabilities


def total_variation(p: np.ndarray, q: np.ndarray) -> float:
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    if p.shape != q.shape:
        raise ValueError("TV operands must have the same shape")
    return float(0.5 * np.abs(p - q).sum())


def _history_bucket(history: np.ndarray, buckets: int) -> int:
    # Deterministic arithmetic hash avoids Python's process-randomized hash().
    weights = (np.arange(history.size, dtype=np.int64) * 2 + 1) % 104729
    return int(np.dot(history.astype(np.int64), weights) % buckets)


def phase0_gate(
    samples: int | None = None,
    delta: float | None = None,
    cfg: ExperimentConfig = DEFAULT_CONFIG,
) -> dict[str, Any]:
    n = samples or cfg.gates.phase0_samples
    tolerance = delta or cfg.gates.phase0_delta
    if n < 10_000:
        raise ValueError("Phase 0 needs at least 10,000 observations")
    topology_results: list[dict[str, Any]] = []
    expected_noise = noise_probabilities(cfg.field.p, cfg.field.eta)
    uniform = np.full(cfg.field.p, 1.0 / cfg.field.p)
    buckets = cfg.gates.phase0_context_buckets

    for topology_index, (k, d) in enumerate(TOPOLOGIES):
        rule = FiniteFieldRule.sample(k, d, 0, cfg.training.rule_seed, cfg.field)
        rng = np.random.default_rng(91_000 + topology_index)
        state = rng.integers(0, rule.p, size=rule.d, dtype=np.int64)
        residual_counts = np.zeros(rule.p, dtype=np.int64)
        token_counts = np.zeros(rule.p, dtype=np.int64)
        joint = np.zeros((rule.p, buckets), dtype=np.int64)

        nonzero = rng.random(n + cfg.model.visible_prefix_length) < rule.eta
        noise = np.zeros(nonzero.size, dtype=np.int64)
        noise[nonzero] = rng.integers(1, rule.p, size=int(nonzero.sum()))
        for i, residual in enumerate(noise):
            z = int(rule.center(state))
            token = (z + int(residual)) % rule.p
            if i >= cfg.model.visible_prefix_length:
                bucket = _history_bucket(state, buckets)
                residual_counts[residual] += 1
                token_counts[token] += 1
                joint[residual, bucket] += 1
            state[:-1] = state[1:]
            state[-1] = token

        residual_dist = residual_counts / residual_counts.sum()
        token_dist = token_counts / token_counts.sum()
        joint_dist = joint / joint.sum()
        product = joint_dist.sum(axis=1, keepdims=True) * joint_dist.sum(
            axis=0, keepdims=True
        )

        # Alignment is validated independently on the exact episode interface.
        stream = OnlineEpisodeStream(rule, 123_000 + topology_index)
        episode = stream.episode()
        scored_positions = np.flatnonzero(episode.loss_mask)
        mismatches = 0
        for input_position in scored_positions:
            history = episode.tokens[: input_position + 1]
            expected_center = int(rule.center(history))
            if expected_center != int(episode.target_centers[input_position]):
                mismatches += 1
        alignment_error_rate = mismatches / max(1, scored_positions.size)
        metrics = {
            "k": k,
            "d": d,
            "residual_marginal_tv": total_variation(residual_dist, expected_noise),
            "token_stationarity_tv": total_variation(token_dist, uniform),
            "residual_history_independence_tv": total_variation(joint_dist, product),
            "evaluation_alignment_error_rate": alignment_error_rate,
        }
        metrics["passed"] = all(
            metrics[key] < tolerance
            for key in (
                "residual_marginal_tv",
                "token_stationarity_tv",
                "residual_history_independence_tv",
                "evaluation_alignment_error_rate",
            )
        )
        topology_results.append(metrics)

    return {
        "phase": 0,
        "samples_per_topology": n,
        "delta": tolerance,
        "criterion": "every reported TV/error rate is strictly below delta",
        "topologies": topology_results,
        "passed": all(r["passed"] for r in topology_results),
    }


def _flat_gradients(model: torch.nn.Module) -> torch.Tensor:
    pieces = []
    for parameter in model.parameters():
        if parameter.grad is None:
            pieces.append(torch.zeros_like(parameter).reshape(-1).cpu())
        else:
            pieces.append(parameter.grad.detach().reshape(-1).float().cpu())
    return torch.cat(pieces)


def phase1_gate(cfg: ExperimentConfig = DEFAULT_CONFIG) -> dict[str, Any]:
    """FP32 CPU/MPS forward and first-gradient parity; deliberately no HVP."""
    if not torch.backends.mps.is_available():
        return {
            "phase": 1,
            "status": "unavailable",
            "passed": False,
            "reason": "torch.backends.mps.is_available() is false",
            "hvp_required": False,
        }
    torch.manual_seed(4_242)
    gate_cfg = cfg.model.__class__(
        target_core_parameters=cfg.model.target_core_parameters,
        n_layers=1,
        n_heads=2,
        context_length=16,
        visible_prefix_length=8,
        mlp_ratio=2,
        dropout=0.0,
        position_encoding=cfg.model.position_encoding,
        tie_embeddings=True,
        bias=True,
    )
    cpu_model = KeplerTransformer(32, gate_cfg, cfg.field.p).float()
    mps_model = KeplerTransformer(32, gate_cfg, cfg.field.p).float().to("mps")
    mps_model.load_state_dict(cpu_model.state_dict())
    inputs = torch.randint(0, cfg.field.p, (2, 15), generator=torch.Generator().manual_seed(7))
    targets = torch.randint(0, cfg.field.p, (2, 15), generator=torch.Generator().manual_seed(8))
    mask = torch.ones_like(targets, dtype=torch.bool)

    cpu_loss = masked_language_model_loss(cpu_model(inputs), targets, mask)
    cpu_loss.backward()
    cpu_grad = _flat_gradients(cpu_model)
    mps_loss = masked_language_model_loss(
        mps_model(inputs.to("mps")), targets.to("mps"), mask.to("mps")
    )
    mps_loss.backward()
    # Synchronize before copying results back to host.
    torch.mps.synchronize()
    mps_grad = _flat_gradients(mps_model)
    cpu_value = float(cpu_loss.detach())
    mps_value = float(mps_loss.detach().cpu())
    loss_relative_error = abs(mps_value - cpu_value) / max(abs(cpu_value), 1.0e-12)
    gradient_relative_error = float(
        torch.linalg.vector_norm(mps_grad - cpu_grad)
        / torch.linalg.vector_norm(cpu_grad).clamp_min(1.0e-12)
    )
    passed = (
        loss_relative_error < cfg.gates.phase1_loss_relative_tolerance
        and gradient_relative_error < cfg.gates.phase1_gradient_relative_tolerance
    )
    return {
        "phase": 1,
        "status": "complete",
        "dtype": "float32",
        "loss_cpu": cpu_value,
        "loss_mps": mps_value,
        "loss_relative_error": loss_relative_error,
        "gradient_relative_l2_error": gradient_relative_error,
        "loss_tolerance": cfg.gates.phase1_loss_relative_tolerance,
        "gradient_tolerance": cfg.gates.phase1_gradient_relative_tolerance,
        "hvp_required": False,
        "passed": passed,
    }
