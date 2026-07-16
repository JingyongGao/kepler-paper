from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from torch import Tensor
from torch.nn import functional as F

from .config import DEFAULT_CONFIG, ExperimentConfig
from .data import OnlineEpisodeStream, split_seed
from .durability import atomic_write_json
from .environment import environment_snapshot
from .model import make_model
from .topology import FiniteFieldRule
from .training import resolve_device


@dataclass
class _OnlineGradientMoments:
    """Vector Welford moments plus an online pairwise-cosine sufficient statistic."""

    count: int = 0
    mean: Tensor | None = None
    centered_sum_squares: Tensor | None = None
    unit_vector_sum: Tensor | None = None
    norm_sum: float = 0.0
    squared_norm_sum: float = 0.0

    def update(self, vector: Tensor) -> None:
        value = vector.detach().float()
        norm = torch.linalg.vector_norm(value)
        unit = value / norm.clamp_min(1.0e-30)
        self.count += 1
        self.norm_sum += float(norm.cpu())
        self.squared_norm_sum += float((norm * norm).cpu())
        if self.mean is None:
            self.mean = value.clone()
            self.centered_sum_squares = torch.zeros((), device=value.device)
            self.unit_vector_sum = unit.clone()
            return
        delta = value - self.mean
        self.mean.add_(delta / self.count)
        delta_after = value - self.mean
        self.centered_sum_squares.add_(torch.dot(delta.flatten(), delta_after.flatten()))
        self.unit_vector_sum.add_(unit)

    def summary(self) -> dict[str, float | int]:
        if self.mean is None or self.centered_sum_squares is None or self.unit_vector_sum is None:
            raise ValueError("no gradients accumulated")
        mean_norm_sq = torch.sum(self.mean * self.mean)
        variance = self.centered_sum_squares / self.count
        if self.count > 1:
            unit_sum_sq = torch.sum(self.unit_vector_sum * self.unit_vector_sum)
            cosine = (unit_sum_sq - self.count) / (self.count * (self.count - 1))
        else:
            cosine = torch.full((), float("nan"), device=self.mean.device)
        return {
            "batches": self.count,
            "norm_of_mean_gradient": float(torch.sqrt(mean_norm_sq).cpu()),
            "gradient_variance_mean_squared_deviation": float(variance.cpu()),
            "gradient_snr": float((mean_norm_sq / variance.clamp_min(1.0e-30)).cpu()),
            "mean_pairwise_gradient_cosine": float(cosine.cpu()),
            "mean_batch_gradient_norm": self.norm_sum / self.count,
            "rms_batch_gradient_norm": math.sqrt(self.squared_norm_sum / self.count),
        }


def gradient_audit_rules(cfg: ExperimentConfig = DEFAULT_CONFIG) -> dict[str, FiniteFieldRule]:
    common = dict(p=cfg.field.p, eta=cfg.field.eta, instance=0)
    return {
        "k1_d1_lag1": FiniteFieldRule(
            **common, k=1, d=1, lags=(1,), coefficients=(45,)
        ),
        "k1_d4_lag4": FiniteFieldRule(
            **common, k=1, d=4, lags=(4,), coefficients=(45,)
        ),
        "k2_d2_lags1_2": FiniteFieldRule(
            **common, k=2, d=2, lags=(1, 2), coefficients=(45, 36)
        ),
        "k2_d4_lags1_4": FiniteFieldRule(
            **common, k=2, d=4, lags=(1, 4), coefficients=(45, 36)
        ),
    }


def _flatten(tensors: Iterable[Tensor]) -> Tensor:
    values = [tensor.detach().float().reshape(-1) for tensor in tensors]
    if not values:
        raise ValueError("gradient group is empty")
    return torch.cat(values)


def _semantic_gradient_groups(
    model,
    logits: Tensor,
    residuals: list[Tensor],
    centers: Tensor,
    mask: Tensor,
    p: int,
    eta: float,
) -> dict[str, Tensor]:
    """Split tied embedding/output and fused QKV gradients into named groups."""
    with torch.no_grad():
        probabilities = F.softmax(logits.detach().float(), dim=-1)
        q = torch.full_like(probabilities, eta / (p - 1))
        q.scatter_(-1, centers.long().unsqueeze(-1), 1.0 - eta)
        output_derivative = (probabilities - q) * mask.unsqueeze(-1)
        output_derivative /= int(mask.sum())
        final_hidden = model.final_norm(residuals[-1].detach()).float()
        if output_derivative.ndim == 3:
            direct_output_weight = torch.einsum(
                "btv,btw->vw", output_derivative, final_hidden
            )
        elif output_derivative.ndim == 2:
            direct_output_weight = torch.einsum(
                "bv,bw->vw", output_derivative, final_hidden
            )
        else:
            raise ValueError("unsupported output derivative rank")
        tied_total = model.token_embedding.weight.grad.detach().float()
        input_token_weight = tied_total - direct_output_weight

        embedding_values = [input_token_weight, model.position_embedding.weight.grad]
        if hasattr(model, "query_embedding"):
            embedding_values.append(model.query_embedding.grad)
        result: dict[str, Tensor] = {
            "global.embedding_input": _flatten(embedding_values),
            "global.output_head_direct": direct_output_weight.reshape(-1),
            "global.tied_embedding_total": tied_total.reshape(-1),
        }
        global_q: list[Tensor] = []
        global_k: list[Tensor] = []
        global_v: list[Tensor] = []
        global_attn_out: list[Tensor] = []
        global_mlp: list[Tensor] = []
        global_norm: list[Tensor] = []
        for block_index, block in enumerate(model.blocks):
            width = model.width
            qkv_weight = block.attn.qkv.weight.grad
            qkv_bias = block.attn.qkv.bias.grad if block.attn.qkv.bias is not None else None
            q_values = [qkv_weight[:width]]
            k_values = [qkv_weight[width : 2 * width]]
            v_values = [qkv_weight[2 * width :]]
            if qkv_bias is not None:
                q_values.append(qkv_bias[:width])
                k_values.append(qkv_bias[width : 2 * width])
                v_values.append(qkv_bias[2 * width :])
            out_values = [parameter.grad for parameter in block.attn.out.parameters()]
            mlp_values = [parameter.grad for parameter in block.mlp.parameters()]
            norm_values = [
                parameter.grad
                for norm in (block.norm1, block.norm2)
                for parameter in norm.parameters()
            ]
            block_values = [
                parameter.grad for parameter in block.parameters() if parameter.grad is not None
            ]
            prefix = f"block_{block_index}"
            result[f"{prefix}.attention_q"] = _flatten(q_values)
            result[f"{prefix}.attention_k"] = _flatten(k_values)
            result[f"{prefix}.attention_v"] = _flatten(v_values)
            result[f"{prefix}.attention_out"] = _flatten(out_values)
            result[f"{prefix}.mlp"] = _flatten(mlp_values)
            result[f"{prefix}.normalization"] = _flatten(norm_values)
            result[f"{prefix}.total"] = _flatten(block_values)
            global_q.extend(q_values)
            global_k.extend(k_values)
            global_v.extend(v_values)
            global_attn_out.extend(out_values)
            global_mlp.extend(mlp_values)
            global_norm.extend(norm_values)
        global_norm.extend(parameter.grad for parameter in model.final_norm.parameters())
        result.update(
            {
                "global.attention_q": _flatten(global_q),
                "global.attention_k": _flatten(global_k),
                "global.attention_v": _flatten(global_v),
                "global.attention_out": _flatten(global_attn_out),
                "global.mlp": _flatten(global_mlp),
                "global.normalization": _flatten(global_norm),
            }
        )
        return result


def _soft_population_loss(logits: Tensor, centers: Tensor, mask: Tensor, p: int, eta: float):
    log_probs = F.log_softmax(logits.float(), dim=-1)
    q = torch.full_like(log_probs, eta / (p - 1))
    q.scatter_(-1, centers.long().unsqueeze(-1), 1.0 - eta)
    return (-torch.sum(q * log_probs, dim=-1))[mask].mean()


def _aggregate_initializations(
    initializations: list[dict[str, dict[str, float | int]]]
) -> dict[str, dict[str, float]]:
    output: dict[str, dict[str, float]] = {}
    for group in initializations[0]:
        output[group] = {}
        for metric in (
            "norm_of_mean_gradient",
            "gradient_variance_mean_squared_deviation",
            "gradient_snr",
            "mean_pairwise_gradient_cosine",
            "mean_batch_gradient_norm",
            "rms_batch_gradient_norm",
        ):
            values = np.asarray(
                [float(initialization[group][metric]) for initialization in initializations]
            )
            output[group][f"{metric}_mean_over_initializations"] = float(values.mean())
            output[group][f"{metric}_std_over_initializations"] = float(values.std(ddof=1))
    return output


def run_initialization_gradient_audit(
    output: str | Path = "artifacts/initialization_gradient_signal_audit.json",
    *,
    device_name: str = "auto",
    initializations: int = 8,
    batches_per_initialization: int = 32,
    batch_size: int = 8,
    cfg: ExperimentConfig = DEFAULT_CONFIG,
) -> dict[str, Any]:
    if initializations != 8 or batches_per_initialization != 32:
        raise ValueError("registered audit requires 8 initializations and 32 batches")
    device = resolve_device(device_name)
    rules = gradient_audit_rules(cfg)
    result: dict[str, Any] = {
        "kind": "no_training_initialization_gradient_signal_audit",
        "p": cfg.field.p,
        "eta": cfg.field.eta,
        "target_core_parameters": 2_000_000,
        "actual_model_initializations": initializations,
        "independent_batches_per_initialization_and_task": batches_per_initialization,
        "episodes_per_batch": batch_size,
        "scored_contexts_per_batch": batch_size
        * (cfg.model.context_length - cfg.model.visible_prefix_length),
        "loss": "exact_soft_label_population_cross_entropy",
        "trained": False,
        "online_accumulation": True,
        "full_batch_gradient_vectors_retained": False,
        "tied_head_decomposition": (
            "direct output-weight gradient is derived from dL/dlogits and final hidden; "
            "input-token gradient is tied total minus direct output contribution"
        ),
        "environment": environment_snapshot(
            cfg.to_dict(),
            production_manifest="artifacts/manifest.json",
            pilot_manifest="artifacts/pilot_manifest.json",
            seeds={"model_initializations": list(range(initializations))},
            workdir=Path.cwd(),
        ),
        "tasks": {},
    }
    scored = cfg.model.context_length - cfg.model.visible_prefix_length
    for task_offset, (task_name, rule) in enumerate(rules.items()):
        per_initialization: list[dict[str, dict[str, float | int]]] = []
        losses: list[float] = []
        for initialization in range(initializations):
            torch.manual_seed(initialization)
            np.random.seed(initialization)
            model = make_model(2_000_000, cfg.model, cfg.field.p).to(device)
            model.train()
            stream_seed = split_seed(
                71_000 + task_offset,
                rule.k,
                rule.d,
                rule.instance,
                initialization,
            )
            stream = OnlineEpisodeStream(
                rule,
                stream_seed,
                cfg.model.visible_prefix_length,
                scored,
                cfg.field.d_max,
            )
            accumulators: dict[str, _OnlineGradientMoments] = {}
            init_losses = []
            for _ in range(batches_per_initialization):
                batch = stream.batch(batch_size)
                inputs, _, centers, mask = batch.to_torch(str(device))
                model.zero_grad(set_to_none=True)
                logits, residuals = model(inputs, return_residuals=True)
                loss = _soft_population_loss(
                    logits, centers, mask, cfg.field.p, cfg.field.eta
                )
                loss.backward()
                init_losses.append(float(loss.detach().cpu()))
                groups = _semantic_gradient_groups(
                    model, logits, residuals, centers, mask, cfg.field.p, cfg.field.eta
                )
                if not accumulators:
                    accumulators = {name: _OnlineGradientMoments() for name in groups}
                for name, vector in groups.items():
                    accumulators[name].update(vector)
                del groups, logits, residuals, loss
            summaries = {name: value.summary() for name, value in accumulators.items()}
            per_initialization.append(summaries)
            losses.append(float(np.mean(init_losses)))
            del model, accumulators
            if device.type == "mps":
                torch.mps.empty_cache()
        result["tasks"][task_name] = {
            "rule": rule.to_dict(),
            "mean_population_cross_entropy_nats": float(np.mean(losses)),
            "population_cross_entropy_by_initialization": losses,
            "per_initialization": [
                {"initialization_seed": index, "groups": groups}
                for index, groups in enumerate(per_initialization)
            ],
            "aggregate": _aggregate_initializations(per_initialization),
        }
        atomic_write_json(output, result)
    atomic_write_json(output, result)
    return result
