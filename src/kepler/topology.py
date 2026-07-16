from __future__ import annotations

from dataclasses import dataclass
from math import gcd
from typing import Iterable

import numpy as np
from numpy.typing import NDArray

from .config import FieldConfig


TOPOLOGIES: tuple[tuple[int, int], ...] = (
    (2, 4),
    (4, 4),
    (2, 64),
    (4, 64),
    (3, 16),
)


def anchored_lags(k: int, d: int) -> tuple[int, ...]:
    """Endpoint-anchored relative-isometric integer lag layout."""
    if k == 1:
        if d < 1:
            raise ValueError("d must be positive")
        # Single-parent controls use d as the parent's explicit lag.  They are
        # diagnostic-only and do not alter the k>=2 production layout.
        return (d,)
    if k < 1:
        raise ValueError("k must be positive")
    if d < k:
        raise ValueError("d must be at least k so lags remain distinct")
    lags = tuple(1 + ((j * (d - 1)) // (k - 1)) for j in range(k))
    if lags[0] != 1 or lags[-1] != d:
        raise AssertionError("endpoint anchoring failed")
    if len(set(lags)) != k:
        raise AssertionError("lag construction produced duplicates")
    if gcd(*lags) != 1:
        raise AssertionError("lag set must have gcd 1")
    return lags


@dataclass(frozen=True)
class FiniteFieldRule:
    p: int
    eta: float
    k: int
    d: int
    lags: tuple[int, ...]
    coefficients: tuple[int, ...]
    instance: int = 0

    def __post_init__(self) -> None:
        if self.lags != anchored_lags(self.k, self.d):
            raise ValueError("lags do not match the preregistered anchored layout")
        if len(self.coefficients) != self.k:
            raise ValueError("one coefficient is required per lag")
        if any(not 1 <= a < self.p for a in self.coefficients):
            raise ValueError("coefficients must lie in F_p^*")
        if not 0.0 < self.eta < (self.p - 1) / self.p:
            raise ValueError("invalid eta")

    @classmethod
    def sample(
        cls,
        k: int,
        d: int,
        instance: int,
        seed: int,
        field: FieldConfig | None = None,
    ) -> "FiniteFieldRule":
        cfg = field or FieldConfig()
        # The complete identity is mixed into SeedSequence, making manifest
        # expansion invariant to enumeration order.
        ss = np.random.SeedSequence([seed, cfg.p, k, d, instance])
        rng = np.random.default_rng(ss)
        coeffs = tuple(int(v) for v in rng.integers(1, cfg.p, size=k))
        return cls(
            p=cfg.p,
            eta=cfg.eta,
            k=k,
            d=d,
            lags=anchored_lags(k, d),
            coefficients=coeffs,
            instance=instance,
        )

    def center(self, history: NDArray[np.integer]) -> NDArray[np.int64]:
        """Compute z_t from histories whose final axis is ordered oldest→newest."""
        h = np.asarray(history, dtype=np.int64)
        if h.shape[-1] < self.d:
            raise ValueError(f"history requires at least d={self.d} tokens")
        indices = np.asarray([-lag for lag in self.lags], dtype=np.int64)
        parents = np.take(h, indices, axis=-1)
        coeff = np.asarray(self.coefficients, dtype=np.int64)
        return np.asarray((parents @ coeff) % self.p, dtype=np.int64)

    def to_dict(self) -> dict[str, object]:
        return {
            "p": self.p,
            "eta": self.eta,
            "k": self.k,
            "d": self.d,
            "lags": list(self.lags),
            "coefficients": list(self.coefficients),
            "instance": self.instance,
        }


def noise_probabilities(p: int = 257, eta: float = 0.1) -> NDArray[np.float64]:
    probs = np.full(p, eta / (p - 1), dtype=np.float64)
    probs[0] = 1.0 - eta
    return probs


def sample_noise(
    rng: np.random.Generator,
    size: int | tuple[int, ...],
    p: int = 257,
    eta: float = 0.1,
) -> NDArray[np.int64]:
    """Sample exactly from nu_eta without constructing a 257-way CDF."""
    nonzero = rng.random(size) < eta
    values = np.zeros(size, dtype=np.int64)
    count = int(np.count_nonzero(nonzero))
    if count:
        values[nonzero] = rng.integers(1, p, size=count)
    return values


def distinct_paired_rule(
    old_rule: FiniteFieldRule,
    seed: int,
    new_instance: int | None = None,
) -> FiniteFieldRule:
    """Construct a' with identical topology/noise and distinct coefficients."""
    instance = new_instance if new_instance is not None else old_rule.instance + 10_000
    candidate = FiniteFieldRule.sample(
        old_rule.k,
        old_rule.d,
        instance,
        seed,
        FieldConfig(p=old_rule.p, eta=old_rule.eta, d_max=max(64, old_rule.d)),
    )
    if candidate.coefficients == old_rule.coefficients:
        # Vanishingly unlikely, but distinction is a hard protocol invariant.
        candidate = FiniteFieldRule.sample(
            old_rule.k,
            old_rule.d,
            instance + 1,
            seed,
            FieldConfig(p=old_rule.p, eta=old_rule.eta, d_max=max(64, old_rule.d)),
        )
    if candidate.coefficients == old_rule.coefficients:
        raise AssertionError("failed to construct a distinct paired coefficient vector")
    return candidate


@dataclass(frozen=True)
class Episode:
    tokens: NDArray[np.int64]
    centers: NDArray[np.int64]
    loss_mask: NDArray[np.bool_]
    visible_prefix_length: int

    @property
    def inputs(self) -> NDArray[np.int64]:
        return self.tokens[:-1]

    @property
    def targets(self) -> NDArray[np.int64]:
        return self.tokens[1:]

    @property
    def target_centers(self) -> NDArray[np.int64]:
        return self.centers[1:]


def generate_episode(
    rule: FiniteFieldRule,
    rng: np.random.Generator,
    visible_prefix_length: int = 64,
    scored_tokens: int = 64,
    d_max: int = 64,
) -> Episode:
    """Generate one strictly stationary episode with an unscored burn-in.

    A uniform d-state is stationary because the endpoint coefficient at lag d
    is nonzero; the state update is a bijection conditional on each noise draw.
    We nevertheless execute the registered background burn-in and expose only
    its final C tokens.
    """
    if visible_prefix_length < d_max:
        raise ValueError("C must be at least d_max")
    if scored_tokens <= 0:
        raise ValueError("scored_tokens must be positive")

    state = rng.integers(0, rule.p, size=rule.d, dtype=np.int64)
    burn_count = max(visible_prefix_length, d_max)
    generated: list[int] = []
    generated_centers: list[int] = []
    total_new = burn_count + scored_tokens
    noise = sample_noise(rng, total_new, p=rule.p, eta=rule.eta)

    for i in range(total_new):
        z = int(rule.center(state))
        x = (z + int(noise[i])) % rule.p
        generated.append(x)
        generated_centers.append(z)
        state[:-1] = state[1:]
        state[-1] = x

    start = burn_count - visible_prefix_length
    tokens = np.asarray(generated[start:], dtype=np.int64)
    centers = np.asarray(generated_centers[start:], dtype=np.int64)
    # targets are tokens[1:]. The first scored target is token[C], predicted
    # from input position C-1, so exactly scored_tokens entries are true.
    mask = np.arange(tokens.size - 1) >= visible_prefix_length - 1
    return Episode(
        tokens=tokens,
        centers=centers,
        loss_mask=mask,
        visible_prefix_length=visible_prefix_length,
    )


def histories_from_stream(stream: Iterable[int], d: int) -> NDArray[np.int64]:
    values = np.fromiter(stream, dtype=np.int64)
    if values.size <= d:
        return np.empty((0, d), dtype=np.int64)
    return np.lib.stride_tricks.sliding_window_view(values, d)[:-1].copy()
