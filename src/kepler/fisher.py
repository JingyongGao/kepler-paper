from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import torch
from torch import Tensor
from torch.nn import functional as F


@dataclass(frozen=True)
class AdaptationArm:
    arm_id: str
    rule: str
    task: str
    source: str
    matching: str


def registered_adaptation_arms() -> tuple[AdaptationArm, ...]:
    return (
        AdaptationArm("iid_consolidation", "old_a", "task_A", "generative", "budget"),
        AdaptationArm("task_interface_null", "old_a", "task_B", "generative", "budget"),
        AdaptationArm("compatible_extension", "new_a_prime", "task_B", "generative", "result"),
        AdaptationArm("conflict_rewrite", "new_a_prime", "task_A", "generative", "result"),
        AdaptationArm("fixed_table_memory", "fixed_table", "task_A", "memory", "budget"),
    )


class FixedRandomTable:
    """A deterministic, non-generalizing random table z_table(h)."""

    def __init__(self, seed: int, p: int = 257) -> None:
        self.seed = seed
        self.p = p
        self._key = seed.to_bytes(16, "little", signed=False)

    def center(self, history: np.ndarray) -> int:
        values = np.asarray(history, dtype="<u2")
        digest = hashlib.blake2b(values.tobytes(), key=self._key, digest_size=16).digest()
        return int.from_bytes(digest, "little") % self.p


def jeffreys_divergence_from_logits(logits_a: Tensor, logits_b: Tensor) -> Tensor:
    if logits_a.shape != logits_b.shape:
        raise ValueError("paired functional distributions must have identical shape")
    log_a = F.log_softmax(logits_a.float(), dim=-1)
    log_b = F.log_softmax(logits_b.float(), dim=-1)
    prob_a = log_a.exp()
    prob_b = log_b.exp()
    kl_ab = torch.sum(prob_a * (log_a - log_b), dim=-1)
    kl_ba = torch.sum(prob_b * (log_b - log_a), dim=-1)
    return kl_ab + kl_ba


def fisher_rao_path_length(logit_path: Sequence[Tensor]) -> dict[str, object]:
    """Compute the registered local path estimator on one fixed P_ref sample."""
    if len(logit_path) < 2:
        raise ValueError("a path needs at least two model states")
    step_lengths: list[float] = []
    for before, after in zip(logit_path, logit_path[1:]):
        mean_jeffreys = jeffreys_divergence_from_logits(before, after).mean()
        step_lengths.append(float(torch.sqrt(mean_jeffreys.clamp_min(0)).cpu()))
    return {
        "step_lengths": step_lengths,
        "L_FR": float(sum(step_lengths)),
        "steps": len(step_lengths),
    }


def state_dict_digest(state: Mapping[str, Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state):
        tensor = state[name].detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(np.asarray(tensor.shape, dtype="<i8").tobytes())
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def validate_paired_baselines(paths: Mapping[str, Sequence[str | Path]]) -> str:
    """Assert that all five paths begin from byte-identical model parameters."""
    expected_arms = {arm.arm_id for arm in registered_adaptation_arms()}
    if set(paths) != expected_arms:
        missing = expected_arms - set(paths)
        extra = set(paths) - expected_arms
        raise ValueError(f"adaptation paths mismatch; missing={missing}, extra={extra}")
    digests = []
    for arm in registered_adaptation_arms():
        checkpoints = paths[arm.arm_id]
        if len(checkpoints) < 2:
            raise ValueError(f"{arm.arm_id} requires baseline plus at least one update")
        state = torch.load(checkpoints[0], map_location="cpu", weights_only=False)
        model_state = state["model"] if "model" in state else state
        digests.append(state_dict_digest(model_state))
    if len(set(digests)) != 1:
        raise ValueError("the five local paths do not share an identical baseline")
    return digests[0]


def adaptation_protocol_payload() -> dict[str, object]:
    return {
        "arms": [asdict(arm) for arm in registered_adaptation_arms()],
        "reference_distribution": "one immutable P_ref sample shared by every arm and step",
        "path_metric": "sum sqrt(E_Pref[KL(p_s||p_s+1)+KL(p_s+1||p_s)])",
        "pairing_requirement": "byte-identical baseline parameter state",
        "result_matching": "stop at preregistered target R, never at a post-hoc chosen step",
    }
