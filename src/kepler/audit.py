from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import time
from typing import Any

import numpy as np
import torch

from .config import DEFAULT_CONFIG, ExperimentConfig
from .data import OnlineEpisodeStream, split_seed
from .durability import atomic_write_json, validate_checkpoint_checksum
from .environment import environment_snapshot
from .information import predictable_information, torch_exact_conditional_kl
from .manifest import find_track, rule_from_track
from .model import KeplerTransformer, masked_language_model_loss
from .training import evaluate_checkpoint, learning_rate_at, train_track


def deterministic_algorithm_audit(
    output: str | Path | None = None,
    iterations: int = 8,
    cfg: ExperimentConfig = DEFAULT_CONFIG,
) -> dict[str, Any]:
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    small_cfg = replace(
        cfg.model,
        n_layers=1,
        n_heads=2,
        context_length=32,
        visible_prefix_length=16,
        mlp_ratio=2,
        num_tasks=0,
    )
    torch.manual_seed(7_771)
    model = KeplerTransformer(32, small_cfg, cfg.field.p).to(device)
    inputs = torch.randint(0, cfg.field.p, (4, 31), device=device)
    targets = torch.randint(0, cfg.field.p, (4, 31), device=device)
    mask = torch.ones_like(targets, dtype=torch.bool)

    def measure(deterministic: bool) -> dict[str, Any]:
        previous = torch.are_deterministic_algorithms_enabled()
        try:
            torch.use_deterministic_algorithms(deterministic, warn_only=False)
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
            rates = []
            for index in range(iterations + 2):
                optimizer.zero_grad(set_to_none=True)
                if device.type == "mps":
                    torch.mps.synchronize()
                started = time.perf_counter()
                loss = masked_language_model_loss(model(inputs), targets, mask)
                loss.backward()
                optimizer.step()
                if device.type == "mps":
                    torch.mps.synchronize()
                elapsed = time.perf_counter() - started
                if index >= 2:
                    rates.append(int(mask.sum()) / elapsed)
            return {
                "supported": True,
                "tokens_per_second_median": float(np.median(rates)),
                "loss_finite": bool(torch.isfinite(loss)),
            }
        except Exception as error:
            return {
                "supported": False,
                "error": f"{type(error).__name__}: {error}",
            }
        finally:
            torch.use_deterministic_algorithms(previous, warn_only=False)

    normal = measure(False)
    deterministic = measure(True)
    slowdown = None
    materially_slower = None
    if normal.get("supported") and deterministic.get("supported"):
        slowdown = normal["tokens_per_second_median"] / max(
            deterministic["tokens_per_second_median"], 1e-12
        )
        materially_slower = slowdown > 1.25
    result = {
        "device": str(device),
        "production_setting_changed": False,
        "normal": normal,
        "deterministic": deterministic,
        "slowdown_factor": slowdown,
        "materially_slower_threshold": 1.25,
        "materially_slower": materially_slower,
        "mps_memory": {
            "current_allocated_bytes": int(torch.mps.current_allocated_memory())
            if device.type == "mps"
            else None,
            "driver_allocated_bytes": int(torch.mps.driver_allocated_memory())
            if device.type == "mps"
            else None,
        },
        "environment": environment_snapshot(cfg.to_dict(), workdir=Path.cwd()),
    }
    if output is not None:
        atomic_write_json(output, result)
    return result


def forced_resume_audit(
    output_root: str | Path = "artifacts/phase4_resume_audit",
    device: str = "auto",
    cfg: ExperimentConfig = DEFAULT_CONFIG,
) -> dict[str, Any]:
    track = find_track("k2_d4_n2m_r0_s0_primary", cfg)
    unique_root = Path(output_root) / str(time.time_ns())
    first = train_track(
        track.track_id,
        output_root=unique_root,
        max_tokens=64,
        batch_size=2,
        device_name=device,
        checkpoint_tokens=(64,),
        cfg=cfg,
    )
    checkpoint_64 = first / "checkpoint_64.pt"
    resumed = train_track(
        track.track_id,
        output_root=unique_root,
        max_tokens=128,
        batch_size=2,
        device_name=device,
        resume=checkpoint_64,
        checkpoint_tokens=(64, 128),
        cfg=cfg,
    )
    checkpoint_128 = resumed / "checkpoint_128.pt"
    state_64 = torch.load(checkpoint_64, map_location="cpu", weights_only=False)
    state_128 = torch.load(checkpoint_128, map_location="cpu", weights_only=False)

    rule = rule_from_track(track)
    stream_seed = split_seed(
        cfg.training.train_stream_seed,
        track.k,
        track.d,
        track.rule_instance,
        track.training_seed,
    )
    stream = OnlineEpisodeStream(rule, stream_seed)
    stream.rng.bit_generator.state = state_64["rng_states"]["online_episode_generator"]
    stream.batch(2, score_limit=64)
    expected_state = json.dumps(stream.rng.bit_generator.state, sort_keys=True)
    resumed_state = json.dumps(
        state_128["rng_states"]["online_episode_generator"], sort_keys=True
    )
    evaluation = evaluate_checkpoint(
        checkpoint_128,
        episodes=4,
        batch_size=2,
        device_name=device,
        cfg=cfg,
        bootstrap_replicates=200,
    )
    result = {
        "forced_interruption_at_tokens": 64,
        "resumed_to_tokens": int(state_128["supervised_tokens_seen"]),
        "token_accounting_exact": int(state_64["supervised_tokens_seen"]) == 64
        and int(state_128["supervised_tokens_seen"]) == 128,
        "scheduler_resumed": bool(
            np.isclose(
                state_128["scheduler_state"]["last_learning_rate"],
                learning_rate_at(64, cfg),
            )
        ),
        "data_generator_resumed": expected_state == resumed_state,
        "checkpoint_64_checksum_valid": validate_checkpoint_checksum(checkpoint_64),
        "checkpoint_128_checksum_valid": validate_checkpoint_checksum(checkpoint_128),
        "immediate_reload_evaluation_finite": evaluation["finite"],
        "immediate_reload_R": evaluation["R"],
        "run_directory": str(resumed.resolve()),
    }
    result["passed"] = all(
        result[key]
        for key in (
            "token_accounting_exact",
            "scheduler_resumed",
            "data_generator_resumed",
            "checkpoint_64_checksum_valid",
            "checkpoint_128_checksum_valid",
            "immediate_reload_evaluation_finite",
        )
    )
    atomic_write_json(resumed / "forced_resume_audit.json", result)
    return result


def fixed_batch_overfit_audit(
    output: str | Path = "artifacts/fixed_batch_overfit_audit.json",
    steps: int = 500,
    device: str = "auto",
    cfg: ExperimentConfig = DEFAULT_CONFIG,
) -> dict[str, Any]:
    """Isolated engineering diagnostic; never contributes to scientific fits."""
    from .model import make_model
    from .training import resolve_device

    track = find_track("k2_d4_n2m_r0_s0_primary", cfg)
    rule = rule_from_track(track)
    target_device = resolve_device(device)
    torch.manual_seed(track.training_seed)
    model = make_model(track.target_core_parameters, cfg.model, cfg.field.p).to(
        target_device
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.optimizer.learning_rate,
        betas=cfg.optimizer.betas,
        weight_decay=cfg.optimizer.weight_decay,
    )
    stream = OnlineEpisodeStream(rule, seed=919_191)
    batch = stream.batch(4)
    inputs, targets, _, mask = batch.to_torch(str(target_device))
    losses = []
    for step in range(steps):
        optimizer.zero_grad(set_to_none=True)
        loss = masked_language_model_loss(model(inputs), targets, mask)
        loss.backward()
        optimizer.step()
        if step in (0, 9, 49, 99, steps - 1):
            losses.append({"step": step + 1, "loss_nats": float(loss.detach().cpu())})
    fresh_stream = OnlineEpisodeStream(rule, seed=919_192)
    fresh = fresh_stream.batch(32)
    fresh_inputs, fresh_targets, fresh_centers, fresh_mask = fresh.to_torch(
        str(target_device)
    )
    model.eval()
    with torch.no_grad():
        fresh_logits = model(fresh_inputs).float()
        fresh_log_probs = torch.log_softmax(fresh_logits, dim=-1)
        fresh_nll = masked_language_model_loss(
            fresh_logits, fresh_targets, fresh_mask
        )
        fresh_kl = torch_exact_conditional_kl(
            fresh_log_probs, fresh_centers, cfg.field.p, cfg.field.eta
        )[fresh_mask].mean()
        fresh_r = float(
            fresh_kl.cpu() / predictable_information(cfg.field.p, cfg.field.eta)
        )
    result = {
        "scope": "isolated repeated-batch engineering diagnostic",
        "excluded_from_confirmatory_fit": True,
        "steps": steps,
        "loss_trace": losses,
        "initial_loss_nats": losses[0]["loss_nats"],
        "final_loss_nats": losses[-1]["loss_nats"],
        "loss_decreased": losses[-1]["loss_nats"] < losses[0]["loss_nats"],
        "finite": all(np.isfinite(row["loss_nats"]) for row in losses),
        "independent_fresh_episodes": 32,
        "fresh_episode_nll_nats": float(fresh_nll.cpu()),
        "fresh_exact_conditional_kl_nats": float(fresh_kl.cpu()),
        "fresh_R": fresh_r,
        "memorization_does_not_imply_rule_generalization": fresh_r > 0.5,
    }
    atomic_write_json(output, result)
    return result
