from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import random
from typing import Any

import numpy as np
import torch

from .config import DEFAULT_CONFIG, ExperimentConfig
from .durability import durable_torch_save, validate_checkpoint_checksum
from .environment import canonical_hash, environment_snapshot, git_commit
from .model import (
    assert_conditioner_attachment_equivalence,
    load_base_state_into_conditioned,
    make_model,
    model_metadata,
)
from .training import resolve_device


def attach_task_conditioner_checkpoint(
    base_checkpoint: str | Path,
    output: str | Path,
    device_name: str = "auto",
    tolerance: float = 1.0e-6,
    cfg: ExperimentConfig = DEFAULT_CONFIG,
) -> dict[str, Any]:
    """Attach two zero rows to an existing base checkpoint and validate logits."""
    device = resolve_device(device_name)
    state = torch.load(base_checkpoint, map_location=device, weights_only=False)
    if int(state.get("model_metadata", {}).get("num_tasks", 0)) != 0:
        raise ValueError("base checkpoint already has task conditioning")
    target = int(state["track"]["target_core_parameters"])
    base_model = make_model(target, cfg.model, cfg.field.p).to(device)
    base_model.load_state_dict(state["model"])
    phase4_model_cfg = replace(cfg.model, num_tasks=2)
    phase4_cfg = replace(cfg, model=phase4_model_cfg)
    conditioned = make_model(target, phase4_model_cfg, cfg.field.p).to(device)
    load_base_state_into_conditioned(conditioned, state["model"])
    tokens = torch.randint(
        0,
        cfg.field.p,
        (4, min(32, cfg.model.context_length)),
        generator=torch.Generator().manual_seed(44_004),
    )
    equivalence = assert_conditioner_attachment_equivalence(
        base_model, conditioned, tokens, atol=tolerance, rtol=tolerance
    )

    # Every arm begins with a fresh, identical optimizer after interface
    # attachment; base pretraining optimizer slots are intentionally not mixed
    # with the newly introduced parameter topology.
    optimizer = torch.optim.AdamW(
        conditioned.parameters(),
        lr=cfg.optimizer.learning_rate,
        betas=cfg.optimizer.betas,
        weight_decay=cfg.optimizer.weight_decay,
    )
    provenance = environment_snapshot(
        phase4_cfg.to_dict(), workdir=Path.cwd(), seeds={"attachment": 44_004}
    )
    payload = {
        "version": "3.2.2-phase4-task-attachment",
        "checkpoint_kind": "phase4_task_attachment",
        "base_checkpoint": str(Path(base_checkpoint).resolve()),
        "base_supervised_tokens_seen": int(
            state.get("supervised_tokens_seen", state.get("exposed_tokens", 0))
        ),
        "track": state["track"],
        "config": phase4_cfg.to_dict(),
        "config_hash": canonical_hash(phase4_cfg.to_dict()),
        "git_commit": git_commit(Path.cwd()),
        "provenance": provenance,
        "model_metadata": model_metadata(conditioned),
        "model": conditioned.state_dict(),
        "optimizer": optimizer.state_dict(),
        "optimizer_attachment_policy": "fresh identical AdamW for all five Phase-4 arms",
        "scheduler_state": {
            "kind": "phase4_local_supervised_tokens",
            "supervised_tokens_seen": 0,
            "last_learning_rate": 0.0,
            "next_learning_rate": cfg.optimizer.learning_rate,
        },
        "supervised_tokens_seen": 0,
        "exposed_tokens": 0,
        "rng_states": {
            "python": random.getstate(),
            "numpy_global": np.random.get_state(),
            "torch_cpu": torch.get_rng_state(),
            "torch_mps": torch.mps.get_rng_state().cpu()
            if torch.backends.mps.is_available() and hasattr(torch.mps, "get_rng_state")
            else None,
        },
        "attachment_equivalence": equivalence,
        "attachment_tolerance": {"atol": tolerance, "rtol": tolerance},
        "rule_and_seed_ids": state.get("rule_and_seed_ids", {}),
    }
    checksum = durable_torch_save(
        payload,
        output,
        required_keys={
            "model",
            "optimizer",
            "scheduler_state",
            "supervised_tokens_seen",
            "rng_states",
            "track",
            "config_hash",
            "git_commit",
        },
    )
    reloaded = torch.load(output, map_location=device, weights_only=False)
    verification = make_model(target, phase4_model_cfg, cfg.field.p).to(device)
    verification.load_state_dict(reloaded["model"])
    verification.eval()
    with torch.no_grad():
        a = verification(tokens.to(device), task_id=0).float()
        b = verification(tokens.to(device), task_id=1).float()
    return {
        "output": str(Path(output).resolve()),
        "sha256": checksum,
        "checksum_valid": validate_checkpoint_checksum(output),
        "num_tasks": 2,
        "task_rows_zero": bool(torch.count_nonzero(verification.task_emb.weight) == 0),
        "post_reload_task_a_b_max_abs": float((a - b).abs().max().cpu()),
        "attachment_equivalence": equivalence,
        "optimizer_attachment_policy": payload["optimizer_attachment_policy"],
    }
