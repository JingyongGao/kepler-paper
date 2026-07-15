from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray

from .topology import noise_probabilities


def noise_entropy(p: int = 257, eta: float = 0.1) -> float:
    probs = noise_probabilities(p, eta)
    return float(-np.sum(probs * np.log(probs)))


def predictable_information(p: int = 257, eta: float = 0.1) -> float:
    """log(p)-H(nu_eta), in nats."""
    return math.log(p) - noise_entropy(p, eta)


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
