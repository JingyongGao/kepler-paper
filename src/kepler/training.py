from __future__ import annotations

from contextlib import nullcontext
from dataclasses import asdict, replace
import json
import math
from pathlib import Path
import random
import time
from typing import Any

import numpy as np
import torch
from torch.nn import functional as F

from .config import DEFAULT_CONFIG, ExperimentConfig
from .data import OnlineEpisodeStream, split_seed
from .durability import atomic_write_json, durable_torch_save
from .environment import canonical_hash, file_sha256, git_commit
from .information import predictable_information, torch_exact_conditional_kl
from .manifest import Track, find_track, rule_from_track, track_from_dict
from .model import make_model, masked_language_model_loss, model_metadata


def resolve_device(requested: str = "auto") -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def learning_rate_at(exposed_tokens: int, cfg: ExperimentConfig = DEFAULT_CONFIG) -> float:
    opt = cfg.optimizer
    if exposed_tokens < opt.warmup_tokens:
        return opt.learning_rate * max(exposed_tokens, 1) / opt.warmup_tokens
    progress = min(
        1.0,
        (exposed_tokens - opt.warmup_tokens)
        / max(1, opt.decay_tokens - opt.warmup_tokens),
    )
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return opt.min_learning_rate + cosine * (
        opt.learning_rate - opt.min_learning_rate
    )


def _autocast(device: torch.device, enabled: bool, dtype_name: str):
    if not enabled or device.type == "cpu":
        return nullcontext()
    dtype = getattr(torch, dtype_name)
    return torch.autocast(device_type=device.type, dtype=dtype)


def _mps_rng_state() -> torch.Tensor | None:
    if torch.backends.mps.is_available() and hasattr(torch.mps, "get_rng_state"):
        return torch.mps.get_rng_state().cpu()
    return None


def _mps_memory() -> dict[str, int]:
    if not torch.backends.mps.is_available():
        return {}
    result = {
        "mps_current_allocated_bytes": int(torch.mps.current_allocated_memory()),
        "mps_driver_allocated_bytes": int(torch.mps.driver_allocated_memory()),
    }
    if hasattr(torch.mps, "recommended_max_memory"):
        result["mps_recommended_max_bytes"] = int(torch.mps.recommended_max_memory())
    return result


def _diagnostic_group(name: str) -> str:
    if name.startswith("blocks."):
        return "block_" + name.split(".", 2)[1]
    if name.startswith(("token_embedding", "position_embedding", "task_emb")):
        return "embedding_interface"
    if name.startswith("final_norm"):
        return "final_norm"
    if name.startswith("lm_head"):
        return "lm_head"
    return "other"


def _per_block_gradient_norms(model) -> dict[str, float]:
    sums: dict[str, torch.Tensor] = {}
    for name, parameter in model.named_parameters():
        if parameter.grad is None:
            continue
        group = _diagnostic_group(name)
        value = torch.sum(parameter.grad.detach().float() ** 2)
        sums[group] = sums.get(group, torch.zeros((), device=value.device)) + value
    return {group: float(torch.sqrt(value).cpu()) for group, value in sums.items()}


def _parameter_snapshot(model) -> dict[str, torch.Tensor]:
    return {
        name: parameter.detach().clone()
        for name, parameter in model.named_parameters()
    }


def _per_block_update_ratios(
    model, before: dict[str, torch.Tensor]
) -> dict[str, float]:
    update_squares: dict[str, torch.Tensor] = {}
    parameter_squares: dict[str, torch.Tensor] = {}
    for name, parameter in model.named_parameters():
        group = _diagnostic_group(name)
        current = parameter.detach().float()
        old = before[name].float()
        update = torch.sum((current - old) ** 2)
        magnitude = torch.sum(current**2)
        update_squares[group] = update_squares.get(
            group, torch.zeros((), device=update.device)
        ) + update
        parameter_squares[group] = parameter_squares.get(
            group, torch.zeros((), device=magnitude.device)
        ) + magnitude
    return {
        group: float(
            (
                torch.sqrt(update_squares[group])
                / torch.sqrt(parameter_squares[group]).clamp_min(1e-30)
            ).cpu()
        )
        for group in update_squares
    }


def _oracle_soft_loss(logits, centers, mask, p: int, eta: float):
    log_probs = F.log_softmax(logits.float(), dim=-1)
    q = torch.full_like(log_probs, eta / (p - 1))
    q.scatter_(-1, centers.long().unsqueeze(-1), 1.0 - eta)
    return (-torch.sum(q * log_probs, dim=-1))[mask].mean()


def save_checkpoint(
    path: Path,
    model,
    optimizer,
    exposed_tokens: int,
    stream: OnlineEpisodeStream,
    track: Track,
    cfg: ExperimentConfig,
    last_learning_rate: float,
    stream_seed: int,
    provenance: dict[str, Any] | None = None,
    loss_type: str = "sampled_nll",
    optimization_loss_nats: float | None = None,
    sampled_token_nll_nats: float | None = None,
    checkpoint_diagnostics: dict[str, Any] | None = None,
) -> str:
    metadata = dict(provenance or {})
    metadata.setdefault("config_hash", canonical_hash(cfg.to_dict()))
    metadata.setdefault("git_commit", git_commit(Path.cwd()))
    payload = {
        "version": "3.2.2-phase4-durable",
        "track": asdict(track),
        "config": cfg.to_dict(),
        "config_hash": metadata["config_hash"],
        "git_commit": metadata["git_commit"],
        "provenance": metadata,
        "model_metadata": model_metadata(model),
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler_state": {
            "kind": "absolute_supervised_token_cosine",
            "supervised_tokens_seen": exposed_tokens,
            "last_learning_rate": last_learning_rate,
            "next_learning_rate": learning_rate_at(exposed_tokens, cfg),
        },
        "exposed_tokens": exposed_tokens,
        "supervised_tokens_seen": exposed_tokens,
        "numpy_stream_state": stream.rng.bit_generator.state,
        "rng_states": {
            "python": random.getstate(),
            "numpy_global": np.random.get_state(),
            "online_episode_generator": stream.rng.bit_generator.state,
            "torch_cpu": torch.get_rng_state(),
            "torch_mps": _mps_rng_state(),
        },
        "rule_and_seed_ids": {
            "rule_instance": track.rule_instance,
            "rule_coefficients": track.rule["coefficients"],
            "rule_seed": cfg.training.rule_seed,
            "model_initialization_seed": track.training_seed,
            "data_seed": stream_seed,
        },
        "loss_type": loss_type,
        "optimization_loss_nats": optimization_loss_nats,
        "sampled_token_nll_nats": sampled_token_nll_nats,
        "checkpoint_diagnostics": checkpoint_diagnostics or {},
    }
    if torch.cuda.is_available():
        payload["rng_states"]["torch_cuda"] = torch.cuda.get_rng_state_all()
    return durable_torch_save(
        payload,
        path,
        required_keys={
            "model",
            "optimizer",
            "scheduler_state",
            "supervised_tokens_seen",
            "rng_states",
            "numpy_stream_state",
            "track",
            "config_hash",
            "git_commit",
        },
    )


def train_track(
    track_id: str,
    output_root: str | Path = "runs",
    max_tokens: int | None = None,
    batch_size: int | None = None,
    device_name: str = "auto",
    amp: bool = False,
    resume: str | Path | None = None,
    cfg: ExperimentConfig = DEFAULT_CONFIG,
    track_override: Track | None = None,
    checkpoint_tokens: tuple[int, ...] | None = None,
    stream_seed_override: int | None = None,
    provenance: dict[str, Any] | None = None,
    loss_type: str = "sampled_nll",
) -> Path:
    """Train a single preregistered trajectory on a never-replayed RNG stream."""
    track = track_override or find_track(track_id, cfg)
    if track.track_id != track_id:
        raise ValueError("track_override does not match track_id")
    if loss_type not in {"sampled_nll", "oracle_soft"}:
        raise ValueError("loss_type must be sampled_nll or oracle_soft")
    rule = rule_from_track(track)
    device = resolve_device(device_name)
    random.seed(track.training_seed)
    np.random.seed(track.training_seed)
    torch.manual_seed(track.training_seed)

    model = make_model(track.target_core_parameters, cfg.model, cfg.field.p).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.optimizer.learning_rate,
        betas=cfg.optimizer.betas,
        weight_decay=cfg.optimizer.weight_decay,
    )
    stream_seed = stream_seed_override or split_seed(
        cfg.training.train_stream_seed,
        track.k,
        track.d,
        track.rule_instance,
        track.training_seed,
    )
    scored_per_episode = cfg.model.context_length - cfg.model.visible_prefix_length
    stream = OnlineEpisodeStream(
        rule,
        stream_seed,
        visible_prefix_length=cfg.model.visible_prefix_length,
        scored_tokens_per_episode=scored_per_episode,
        d_max=cfg.field.d_max,
    )
    exposed = 0
    if resume is not None:
        state = torch.load(resume, map_location=device, weights_only=False)
        if state["track"]["track_id"] != track.track_id:
            raise ValueError("resume checkpoint belongs to a different track")
        if state.get("loss_type", "sampled_nll") != loss_type:
            raise ValueError("resume checkpoint uses a different training objective")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        exposed = int(state.get("supervised_tokens_seen", state["exposed_tokens"]))
        rng_states = state.get("rng_states", {})
        stream.rng.bit_generator.state = rng_states.get(
            "online_episode_generator", state["numpy_stream_state"]
        )
        if "python" in rng_states:
            random.setstate(rng_states["python"])
        if "numpy_global" in rng_states:
            np.random.set_state(rng_states["numpy_global"])
        if "torch_cpu" in rng_states:
            torch.set_rng_state(rng_states["torch_cpu"].cpu())
        if device.type == "mps" and rng_states.get("torch_mps") is not None:
            torch.mps.set_rng_state(rng_states["torch_mps"].cpu())
        if device.type == "cuda" and "torch_cuda" in rng_states:
            torch.cuda.set_rng_state_all(rng_states["torch_cuda"])
        scheduler_seen = int(state.get("scheduler_state", {}).get("supervised_tokens_seen", exposed))
        if scheduler_seen != exposed:
            raise ValueError("scheduler/token accounting mismatch in resume checkpoint")

    endpoint = max_tokens or cfg.training.checkpoints[-1]
    if endpoint <= exposed:
        raise ValueError("max_tokens must be greater than the resumed exposure")
    registered_points = checkpoint_tokens or cfg.training.checkpoints
    registered = [v for v in registered_points if exposed < v <= endpoint]
    save_points = sorted(set(registered + [endpoint]))
    batch = batch_size or cfg.training.batch_size
    run_dir = Path(output_root) / track.track_id
    run_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(
        run_dir / "run.json",
        {
            "track": asdict(track),
            "config": cfg.to_dict(),
            "device": str(device),
            "amp": amp,
            "stream_seed": stream_seed,
            "loss_type": loss_type,
            "provenance": provenance or {},
            "exposure_definition": "fresh targets selected by the NLL mask",
        },
    )

    model.train()
    metrics_path = run_dir / "train.jsonl"
    with metrics_path.open("a", encoding="utf-8") as metrics:
        while exposed < endpoint:
            next_boundary = min(v for v in save_points if v > exposed)
            capacity = batch * scored_per_episode
            score_limit = min(capacity, next_boundary - exposed)
            episode_batch = stream.batch(batch, score_limit=score_limit)
            inputs, targets, centers, mask = episode_batch.to_torch(str(device))
            lr = learning_rate_at(exposed, cfg)
            for group in optimizer.param_groups:
                group["lr"] = lr
            optimizer.zero_grad(set_to_none=True)
            if device.type == "mps":
                torch.mps.synchronize()
            step_started = time.perf_counter()
            with _autocast(device, amp, cfg.training.amp_dtype):
                logits = model(inputs)
                sampled_nll = masked_language_model_loss(logits.float(), targets, mask)
                loss = (
                    sampled_nll
                    if loss_type == "sampled_nll"
                    else _oracle_soft_loss(
                        logits, centers, mask, cfg.field.p, cfg.field.eta
                    )
                )
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError(f"non-finite loss at D={exposed}")
            loss.backward()
            is_checkpoint_step = exposed + score_limit in save_points
            per_block_gradients = (
                _per_block_gradient_norms(model) if is_checkpoint_step else {}
            )
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), cfg.optimizer.grad_clip_norm
            )
            if not bool(torch.isfinite(grad_norm)):
                raise FloatingPointError(f"non-finite gradient norm at D={exposed}")
            parameter_before = _parameter_snapshot(model) if is_checkpoint_step else None
            optimizer.step()
            update_ratios = (
                _per_block_update_ratios(model, parameter_before)
                if parameter_before is not None
                else {}
            )
            if device.type == "mps":
                torch.mps.synchronize()
            elapsed = time.perf_counter() - step_started
            exposed += score_limit
            metric = {
                "exposed_tokens": exposed,
                "loss_nats": float(loss.detach().cpu()),
                "sampled_token_nll_nats": float(sampled_nll.detach().cpu()),
                "loss_type": loss_type,
                "learning_rate": lr,
                "gradient_norm": float(grad_norm.detach().cpu()),
                "step_seconds": elapsed,
                "supervised_tokens_per_second": score_limit / max(elapsed, 1e-12),
                **_mps_memory(),
            }
            metrics.write(
                json.dumps(metric, sort_keys=True) + "\n"
            )
            metrics.flush()
            # Training metrics are append-only; make the record durable before
            # publishing any checkpoint at this exposure.
            import os

            os.fsync(metrics.fileno())
            if exposed in save_points:
                checksum = save_checkpoint(
                    run_dir / f"checkpoint_{exposed}.pt",
                    model,
                    optimizer,
                    exposed,
                    stream,
                    track,
                    cfg,
                    lr,
                    stream_seed,
                    provenance,
                    loss_type,
                    float(loss.detach().cpu()),
                    float(sampled_nll.detach().cpu()),
                    {
                        "per_block_gradient_norm": per_block_gradients,
                        "per_block_parameter_update_over_parameter_norm": update_ratios,
                    },
                )
                metric["checkpoint_sha256"] = checksum
    return run_dir


@torch.no_grad()
def evaluate_checkpoint(
    checkpoint: str | Path,
    contexts: int = 16_384,
    episodes: int | None = None,
    batch_size: int = 32,
    device_name: str = "auto",
    cfg: ExperimentConfig = DEFAULT_CONFIG,
    evaluation_seed_override: int | None = None,
    bootstrap_replicates: int = 2_000,
    task_id: int | None = None,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    if contexts <= 0:
        raise ValueError("contexts must be positive")
    device = resolve_device(device_name)
    state = torch.load(checkpoint, map_location=device, weights_only=False)
    track = track_from_dict(state["track"])
    rule = rule_from_track(track)
    num_tasks = int(state.get("model_metadata", {}).get("num_tasks", 0))
    model_cfg = replace(cfg.model, num_tasks=num_tasks)
    model = make_model(track.target_core_parameters, model_cfg, cfg.field.p).to(device)
    model.load_state_dict(state["model"])
    model.eval()
    seed = evaluation_seed_override or split_seed(
        cfg.training.evaluation_stream_seed,
        track.k,
        track.d,
        track.rule_instance,
        track.training_seed,
    )
    scored_per_episode = cfg.model.context_length - cfg.model.visible_prefix_length
    stream = OnlineEpisodeStream(
        rule,
        seed,
        cfg.model.visible_prefix_length,
        scored_per_episode,
        cfg.field.d_max,
    )
    kl_sum = 0.0
    nll_sum = 0.0
    center_correct = 0
    center_probability_sum = 0.0
    entropy_sum = 0.0
    seen = 0
    episode_target = episodes or math.ceil(contexts / scored_per_episode)
    episode_kl: list[float] = []
    episode_nll: list[float] = []
    while len(episode_kl) < episode_target:
        count = min(batch_size, episode_target - len(episode_kl))
        b = stream.batch(count)
        inputs, targets, centers, mask = b.to_torch(str(device))
        logits = model(inputs, task_id=task_id).float()
        log_probs = F.log_softmax(logits, dim=-1)
        probabilities = log_probs.exp()
        kl = torch_exact_conditional_kl(log_probs, centers, cfg.field.p, cfg.field.eta)
        nll = F.cross_entropy(
            logits.reshape(-1, cfg.field.p), targets.reshape(-1), reduction="none"
        ).reshape_as(targets)
        per_episode_kl = kl[mask].reshape(count, scored_per_episode).mean(dim=1)
        per_episode_nll = nll[mask].reshape(count, scored_per_episode).mean(dim=1)
        episode_kl.extend(float(value) for value in per_episode_kl.cpu())
        episode_nll.extend(float(value) for value in per_episode_nll.cpu())
        kl_sum += float(kl[mask].sum().cpu())
        nll_sum += float(nll[mask].sum().cpu())
        center_correct += int((logits.argmax(dim=-1)[mask] == centers[mask]).sum().cpu())
        center_probability_sum += float(
            probabilities.gather(-1, centers.long().unsqueeze(-1)).squeeze(-1)[mask].sum().cpu()
        )
        predictive_entropy = -torch.sum(probabilities * log_probs, dim=-1)
        entropy_sum += float(predictive_entropy[mask].sum().cpu())
        selected = int(mask.sum())
        seen += selected
    information = predictable_information(cfg.field.p, cfg.field.eta)
    episode_r = np.asarray(episode_kl, dtype=np.float64) / information
    bootstrap_rng = np.random.default_rng(seed + 9_973)
    bootstrap_means = np.empty(bootstrap_replicates, dtype=np.float64)
    for index in range(bootstrap_replicates):
        draw = bootstrap_rng.integers(0, episode_r.size, size=episode_r.size)
        bootstrap_means[index] = float(episode_r[draw].mean())
    lower, upper = np.quantile(bootstrap_means, [0.025, 0.975])
    result = {
        "track_id": track.track_id,
        "checkpoint": str(Path(checkpoint).resolve()),
        "exposed_tokens": int(state["exposed_tokens"]),
        "evaluation_contexts": seen,
        "mean_sample_nll_nats": nll_sum / seen,
        "sampled_token_nll_nats": state.get("sampled_token_nll_nats"),
        "fresh_episode_nll_nats": nll_sum / seen,
        "mean_exact_conditional_kl_nats": kl_sum / seen,
        "center_state_top1_accuracy": center_correct / seen,
        "mean_probability_on_center_state": center_probability_sum / seen,
        "mean_predictive_entropy_nats": entropy_sum / seen,
        "evaluation_episodes": episode_target,
        "per_episode_mean_exact_kl_nats": episode_kl,
        "per_episode_mean_sample_nll_nats": episode_nll,
        "R": (kl_sum / seen) / information,
        "R_bootstrap_95_ci": [float(lower), float(upper)],
        "bootstrap_replicates": bootstrap_replicates,
        "evaluation_stream_seed": seed,
        "finite": bool(np.isfinite(episode_r).all()),
        "loss_type": state.get("loss_type", "sampled_nll"),
        "per_block_gradient_norm": state.get("checkpoint_diagnostics", {}).get(
            "per_block_gradient_norm", {}
        ),
        "per_block_parameter_update_over_parameter_norm": state.get(
            "checkpoint_diagnostics", {}
        ).get("per_block_parameter_update_over_parameter_norm", {}),
    }
    output = Path(output_path) if output_path is not None else Path(checkpoint).with_suffix(".eval.json")
    atomic_write_json(output, result)
    return result
