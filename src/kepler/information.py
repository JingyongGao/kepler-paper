from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

from .topology import noise_probabilities
from .topology import FiniteFieldRule, sample_noise


def noise_entropy(p: int = 257, eta: float = 0.1) -> float:
    probs = noise_probabilities(p, eta)
    return float(-np.sum(probs * np.log(probs)))


def predictable_information(p: int = 257, eta: float = 0.1) -> float:
    """log(p)-H(nu_eta), in nats."""
    return math.log(p) - noise_entropy(p, eta)


def analytic_parent_subset_mutual_information(
    rule: FiniteFieldRule, subset: Sequence[int]
) -> float:
    """Return I(Y; X_S) for independent uniform finite-field parents.

    Every coefficient is nonzero.  If even one parent is omitted, its scaled
    value is uniform on F_p, so its sum with all other terms and the independent
    noise is uniform and independent of X_S.  Conditioning on every parent
    instead leaves only nu_eta uncertainty.
    """
    indices = tuple(int(index) for index in subset)
    if len(set(indices)) != len(indices) or any(
        index < 0 or index >= rule.k for index in indices
    ):
        raise ValueError("subset must contain unique parent indices in 0..k-1")
    return predictable_information(rule.p, rule.eta) if len(indices) == rule.k else 0.0


def finite_sample_parent_subset_mi(
    rule: FiniteFieldRule,
    subset: Sequence[int],
    *,
    conditioning_states: int = 32,
    samples_per_state: int = 8_192,
    seed: int = 0,
) -> dict[str, float | int]:
    """Stratified Monte Carlo check of proper-subset independence.

    The selected parent tuple is sampled as an equiprobable finite set of
    conditioning states; omitted parents remain iid uniform.  The returned
    plug-in MI includes its leading finite-sample bias, while ``corrected_mi``
    removes the standard independence bias (r-1)(c-1)/(2n), in nats.
    No full parent/output contingency tensor is retained.
    """
    indices = tuple(int(index) for index in subset)
    if not indices or len(indices) >= rule.k:
        raise ValueError("finite-sample check requires a nonempty proper subset")
    if len(set(indices)) != len(indices) or any(
        index < 0 or index >= rule.k for index in indices
    ):
        raise ValueError("subset must contain unique parent indices in 0..k-1")
    if conditioning_states < 2 or samples_per_state < 1:
        raise ValueError("invalid Monte Carlo sample allocation")

    rng = np.random.default_rng(seed)
    states = rng.integers(0, rule.p, size=(conditioning_states, len(indices)), dtype=np.int64)
    # Reject duplicate conditioning tuples so the row variable is well-defined.
    while np.unique(states, axis=0).shape[0] != conditioning_states:
        states = rng.integers(
            0, rule.p, size=(conditioning_states, len(indices)), dtype=np.int64
        )
    omitted = tuple(index for index in range(rule.k) if index not in indices)
    coefficients = np.asarray(rule.coefficients, dtype=np.int64)
    counts = np.zeros((conditioning_states, rule.p), dtype=np.int64)
    for row, state in enumerate(states):
        parents = rng.integers(
            0, rule.p, size=(samples_per_state, rule.k), dtype=np.int64
        )
        parents[:, indices] = state
        centers = (parents @ coefficients) % rule.p
        outputs = (
            centers
            + sample_noise(rng, samples_per_state, p=rule.p, eta=rule.eta)
        ) % rule.p
        counts[row] = np.bincount(outputs, minlength=rule.p)

    total = int(counts.sum())
    joint = counts.astype(np.float64) / total
    p_x = joint.sum(axis=1, keepdims=True)
    p_y = joint.sum(axis=0, keepdims=True)
    positive = joint > 0
    plugin = float(
        np.sum(joint[positive] * np.log(joint[positive] / (p_x @ p_y)[positive]))
    )
    occupied_x = int(np.count_nonzero(p_x))
    occupied_y = int(np.count_nonzero(p_y))
    independence_bias = (occupied_x - 1) * (occupied_y - 1) / (2.0 * total)
    corrected = plugin - independence_bias
    return {
        "subset_size": len(indices),
        "omitted_parent_count": len(omitted),
        "conditioning_states": conditioning_states,
        "samples_per_state": samples_per_state,
        "total_samples": total,
        "plugin_mi_nats": plugin,
        "independence_bias_nats": independence_bias,
        "bias_corrected_mi_nats": corrected,
    }


def conditional_distribution(
    centers: NDArray[np.integer] | int,
    p: int = 257,
    eta: float = 0.1,
) -> NDArray[np.float64]:
    z = np.asarray(centers, dtype=np.int64)
    classes = np.arange(p, dtype=np.int64)
    residuals = (classes - z[..., None]) % p
    probs = np.full(residuals.shape, eta / (p - 1), dtype=np.float64)
    probs[residuals == 0] = 1.0 - eta
    return probs


def exact_conditional_kl_from_log_probs(
    model_log_probs: NDArray[np.floating],
    centers: NDArray[np.integer],
    p: int = 257,
    eta: float = 0.1,
) -> NDArray[np.float64]:
    log_p = np.asarray(model_log_probs, dtype=np.float64)
    z = np.asarray(centers, dtype=np.int64)
    if log_p.shape[-1] != p or log_p.shape[:-1] != z.shape:
        raise ValueError("log-probability and center shapes are not aligned")
    q = conditional_distribution(z, p=p, eta=eta)
    log_q = np.log(q)
    return np.sum(q * (log_q - log_p), axis=-1)


def normalized_excess_error(
    model_log_probs: NDArray[np.floating],
    centers: NDArray[np.integer],
    p: int = 257,
    eta: float = 0.1,
) -> float:
    kl = exact_conditional_kl_from_log_probs(model_log_probs, centers, p, eta)
    return float(np.mean(kl) / predictable_information(p, eta))


def torch_exact_conditional_kl(log_probs, centers, p: int = 257, eta: float = 0.1):
    """Differentiable exact KL; torch is imported lazily for lightweight tools."""
    import torch

    if log_probs.shape[-1] != p or log_probs.shape[:-1] != centers.shape:
        raise ValueError("log-probability and center shapes are not aligned")
    low = eta / (p - 1)
    q = torch.full_like(log_probs, low)
    q.scatter_(-1, centers.long().unsqueeze(-1), 1.0 - eta)
    return torch.sum(q * (torch.log(q) - log_probs), dim=-1)


def torch_normalized_excess_error(log_probs, centers, p: int = 257, eta: float = 0.1):
    kl = torch_exact_conditional_kl(log_probs, centers, p, eta)
    return kl.mean() / predictable_information(p, eta)
