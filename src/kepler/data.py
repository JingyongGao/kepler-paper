from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .topology import Episode, FiniteFieldRule, generate_episode


@dataclass(frozen=True)
class EpisodeBatch:
    inputs: NDArray[np.int64]
    targets: NDArray[np.int64]
    centers: NDArray[np.int64]
    loss_mask: NDArray[np.bool_]

    @property
    def scored_tokens(self) -> int:
        return int(np.count_nonzero(self.loss_mask))

    def to_torch(self, device: str):
        import torch

        return (
            torch.as_tensor(self.inputs, dtype=torch.long, device=device),
            torch.as_tensor(self.targets, dtype=torch.long, device=device),
            torch.as_tensor(self.centers, dtype=torch.long, device=device),
            torch.as_tensor(self.loss_mask, dtype=torch.bool, device=device),
        )


class OnlineEpisodeStream:
    """Stateful RNG stream: every call exposes newly sampled episodes."""

    def __init__(
        self,
        rule: FiniteFieldRule,
        seed: int,
        visible_prefix_length: int = 64,
        scored_tokens_per_episode: int = 64,
        d_max: int = 64,
    ) -> None:
        self.rule = rule
        self.rng = np.random.default_rng(seed)
        self.visible_prefix_length = visible_prefix_length
        self.scored_tokens_per_episode = scored_tokens_per_episode
        self.d_max = d_max

    def episode(self) -> Episode:
        return generate_episode(
            self.rule,
            self.rng,
            visible_prefix_length=self.visible_prefix_length,
            scored_tokens=self.scored_tokens_per_episode,
            d_max=self.d_max,
        )

    def batch(self, batch_size: int, score_limit: int | None = None) -> EpisodeBatch:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        episodes = [self.episode() for _ in range(batch_size)]
        inputs = np.stack([e.inputs for e in episodes])
        targets = np.stack([e.targets for e in episodes])
        centers = np.stack([e.target_centers for e in episodes])
        masks = np.stack([e.loss_mask for e in episodes])
        if score_limit is not None:
            if not 0 < score_limit <= int(masks.sum()):
                raise ValueError("score_limit is outside this batch")
            flat = masks.reshape(-1)
            true_positions = np.flatnonzero(flat)
            flat[true_positions[score_limit:]] = False
        return EpisodeBatch(inputs, targets, centers, masks)


def split_seed(base: int, k: int, d: int, instance: int, training_seed: int) -> int:
    """Stable split-specific seed with no Python hash randomization."""
    ss = np.random.SeedSequence([base, k, d, instance, training_seed])
    return int(ss.generate_state(1, dtype=np.uint32)[0])
