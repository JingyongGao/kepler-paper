from __future__ import annotations

from functools import lru_cache
import hashlib
import math
from pathlib import Path
from typing import Any

import numpy as np

from .durability import atomic_write_json
from .environment import file_sha256
from .information import noise_entropy
from .parent_pair_split import COEFFICIENTS, P


ETA = 0.1
CUE_COEFFICIENT = 1
PERMUTATION_SEED_SHA256 = (
    "e6cccb69e7dba372d1a5db459b15625cc06c614ea8ba59db637d2d843ee282d9"
)
ROOT_ITERATIONS = 80
ENUMERATION_TOLERANCE = 2.0e-12
TARGET_CONDITIONS = (
    ("C0", 64, 0.0),
    ("C1", 64, 0.016),
    ("C2", 64, 0.065092),
    ("C3", 64, 0.13),
    ("geometry_concentrated", 1, 0.065092),
    ("geometry_diffuse", 128, 0.065092),
)


def cue_permutation() -> tuple[int, ...]:
    seed = bytes.fromhex(PERMUTATION_SEED_SHA256)
    scored = [
        (hashlib.sha256(seed + value.to_bytes(2, "big")).digest(), value)
        for value in range(P)
    ]
    scored.sort(key=lambda item: (item[0], item[1]))
    return tuple(value for _, value in scored)


def cue_subset(m: int) -> tuple[int, ...]:
    if not 1 <= m <= P:
        raise ValueError("cue subset size must be between 1 and p")
    return cue_permutation()[:m]


@lru_cache(maxsize=1)
def _pair_grid() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    linear = np.arange(P * P, dtype=np.int64)
    x1, x2 = linear // P, linear % P
    z = (COEFFICIENTS[0] * x1 + COEFFICIENTS[1] * x2) % P
    return x1, x2, z


def cue_pair_probabilities(lambda_value: float, m: int) -> np.ndarray:
    lam = float(lambda_value)
    if not 0.0 <= lam < 1.0:
        raise ValueError("lambda must satisfy 0 <= lambda < 1")
    if CUE_COEFFICIENT in (0, COEFFICIENTS[0]):
        raise AssertionError("registered cue coefficient is invalid")
    subset = np.zeros(P, dtype=np.bool_)
    subset[np.asarray(cue_subset(m), dtype=np.int64)] = True
    x1, _, z = _pair_grid()
    selected = subset[(z - CUE_COEFFICIENT * x1) % P]
    probabilities = np.full(P * P, (1.0 - lam) / (P * P), dtype=np.float64)
    probabilities[selected] += lam / (m * P)
    if int(selected.sum()) != m * P:
        raise AssertionError("cue component must contain exactly m*p pairs")
    if not bool((probabilities > 0.0).all()):
        raise AssertionError("full-support cue family lost strict positivity")
    if not math.isclose(float(probabilities.sum()), 1.0, abs_tol=2.0e-15):
        raise AssertionError("cue probabilities do not sum to one")
    return probabilities


def sample_cue_pairs(
    rng: np.random.Generator,
    count: int,
    lambda_value: float,
    m: int,
) -> tuple[np.ndarray, np.ndarray]:
    lam = float(lambda_value)
    if not 0.0 <= lam < 1.0:
        raise ValueError("lambda must satisfy 0 <= lambda < 1")
    if lam == 0.0:
        linear = rng.integers(0, P * P, size=count, dtype=np.int64)
        return np.stack((linear // P, linear % P), axis=1), linear
    cue_component = rng.random(count) < lam
    linear = rng.integers(0, P * P, size=count, dtype=np.int64)
    number = int(cue_component.sum())
    if number:
        x1 = rng.integers(0, P, size=number, dtype=np.int64)
        subset = np.asarray(cue_subset(m), dtype=np.int64)
        b = subset[rng.integers(0, m, size=number, dtype=np.int64)]
        inverse_36 = pow(COEFFICIENTS[1], -1, P)
        x2 = (
            (b - (COEFFICIENTS[0] - CUE_COEFFICIENT) * x1) * inverse_36
        ) % P
        linear[cue_component] = P * x1 + x2
    return np.stack((linear // P, linear % P), axis=1), linear


def _entropy(probabilities: np.ndarray, axis: int | None = None) -> np.ndarray:
    values = np.asarray(probabilities, dtype=np.float64)
    terms = np.zeros_like(values)
    positive = values > 0.0
    terms[positive] = -values[positive] * np.log(values[positive])
    return terms.sum(axis=axis)


def _noise_channel(z_distributions: np.ndarray) -> np.ndarray:
    off = ETA / (P - 1)
    return off + ((1.0 - ETA) - off) * np.asarray(z_distributions, dtype=np.float64)


def enumerate_cue_distribution(lambda_value: float, m: int) -> dict[str, Any]:
    probabilities = cue_pair_probabilities(lambda_value, m)
    x1, x2, z = _pair_grid()
    p_x1 = np.bincount(x1, weights=probabilities, minlength=P)
    p_x2 = np.bincount(x2, weights=probabilities, minlength=P)
    p_z = np.bincount(z, weights=probabilities, minlength=P)
    joint_x1_z = np.bincount(
        P * x1 + z, weights=probabilities, minlength=P * P
    ).reshape(P, P)
    joint_x2_z = np.bincount(
        P * x2 + z, weights=probabilities, minlength=P * P
    ).reshape(P, P)
    p_z_x1 = joint_x1_z / p_x1[:, None]
    p_z_x2 = joint_x2_z / p_x2[:, None]
    p_y = _noise_channel(p_z)
    p_y_x1 = _noise_channel(p_z_x1)
    p_y_x2 = _noise_channel(p_z_x2)
    h_y = float(_entropy(p_y))
    h_y_x1 = float(np.dot(p_x1, _entropy(p_y_x1, axis=1)))
    h_y_x2 = float(np.dot(p_x2, _entropy(p_y_x2, axis=1)))
    information_pair = h_y - noise_entropy(P, ETA)
    information_x1 = h_y - h_y_x1
    information_x2 = h_y - h_y_x2
    cue_index = (information_x1 + information_x2) / information_pair
    uniform = np.full(P, 1.0 / P, dtype=np.float64)
    return {
        "lambda": float(lambda_value),
        "m": int(m),
        "strictly_positive_full_support": bool((probabilities > 0.0).all()),
        "minimum_pair_probability": float(probabilities.min()),
        "maximum_pair_probability": float(probabilities.max()),
        "pair_probability_entropy_nats": float(_entropy(probabilities)),
        "expected_minimum_pair_exposure_at_5M": float(
            5_000_000 * probabilities.min()
        ),
        "entropies_nats": {
            "H_X1": float(_entropy(p_x1)),
            "H_X2": float(_entropy(p_x2)),
            "H_Z": float(_entropy(p_z)),
            "H_Y": h_y,
            "H_Y_given_X1": h_y_x1,
            "H_Y_given_X2": h_y_x2,
            "H_Y_given_X1_X2": noise_entropy(P, ETA),
        },
        "mutual_information_nats": {
            "I_Y_X1": information_x1,
            "I_Y_X2": information_x2,
            "I_Y_X1_X2": information_pair,
        },
        "singleton_cue_index": cue_index,
        "uniformity_max_absolute_error": {
            "X1": float(np.max(np.abs(p_x1 - uniform))),
            "X2": float(np.max(np.abs(p_x2 - uniform))),
            "Z": float(np.max(np.abs(p_z - uniform))),
            "Y": float(np.max(np.abs(p_y - uniform))),
        },
        "singleton_information_symmetry_absolute_error": abs(
            information_x1 - information_x2
        ),
        "joint_information_absolute_error": abs(
            information_pair - (math.log(P) - noise_entropy(P, ETA))
        ),
    }


def solve_lambda_for_cue_index(target: float, m: int) -> dict[str, Any]:
    target_value = float(target)
    if target_value == 0.0:
        audit = enumerate_cue_distribution(0.0, m)
        return {
            "target_singleton_cue_index": target_value,
            "lambda": 0.0,
            "iterations": 0,
            "absolute_residual": abs(audit["singleton_cue_index"] - target_value),
            "audit": audit,
        }
    lower = 0.0
    upper = float(np.nextafter(1.0, 0.0))
    upper_value = enumerate_cue_distribution(upper, m)["singleton_cue_index"]
    if not 0.0 < target_value < upper_value:
        raise ValueError("target cue index is outside the registered feasible interval")
    for _ in range(ROOT_ITERATIONS):
        midpoint = (lower + upper) / 2.0
        value = enumerate_cue_distribution(midpoint, m)["singleton_cue_index"]
        if value < target_value:
            lower = midpoint
        else:
            upper = midpoint
    candidates = (lower, upper, (lower + upper) / 2.0)
    audits = [enumerate_cue_distribution(value, m) for value in candidates]
    best = min(
        audits, key=lambda item: abs(item["singleton_cue_index"] - target_value)
    )
    return {
        "target_singleton_cue_index": target_value,
        "lambda": best["lambda"],
        "iterations": ROOT_ITERATIONS,
        "final_bracket": [lower, upper],
        "absolute_residual": abs(best["singleton_cue_index"] - target_value),
        "audit": best,
    }


def build_cue_distribution_audit() -> dict[str, Any]:
    conditions = []
    for name, m, target in TARGET_CONDITIONS:
        solved = solve_lambda_for_cue_index(target, m)
        audit = solved["audit"]
        errors = list(audit["uniformity_max_absolute_error"].values()) + [
            audit["singleton_information_symmetry_absolute_error"],
            audit["joint_information_absolute_error"],
            abs(
                audit["entropies_nats"]["H_Y_given_X1_X2"]
                - noise_entropy(P, ETA)
            ),
        ]
        if max(errors) >= ENUMERATION_TOLERANCE:
            raise AssertionError(f"analytic/enumeration identity failed for {name}")
        if not audit["strictly_positive_full_support"]:
            raise AssertionError("registered cue condition is not full support")
        conditions.append(
            {
                "condition_id": name,
                "m": m,
                "B_m": list(cue_subset(m)),
                **{key: value for key, value in solved.items() if key != "audit"},
                **audit,
            }
        )
    c0 = conditions[0]
    g1_error = max(
        abs(c0["minimum_pair_probability"] - 1.0 / (P * P)),
        abs(c0["maximum_pair_probability"] - 1.0 / (P * P)),
    )
    if g1_error >= 1.0e-18:
        raise AssertionError("lambda=0 does not reproduce uniform G1 probabilities")
    return {
        "kind": "full_support_cue_distribution_audit",
        "trained": False,
        "p": P,
        "eta": ETA,
        "coefficients": list(COEFFICIENTS),
        "cue_coefficient_c": CUE_COEFFICIENT,
        "cue_coefficient_constraints": {
            "c_nonzero": CUE_COEFFICIENT != 0,
            "c_not_equal_target_x1_coefficient": CUE_COEFFICIENT != COEFFICIENTS[0],
        },
        "permutation_seed_sha256": PERMUTATION_SEED_SHA256,
        "permutation_algorithm": (
            "sort field values by SHA256(seed_bytes || uint16_big_endian(value)), "
            "breaking digest ties by value"
        ),
        "field_permutation": list(cue_permutation()),
        "B_m_construction": "prefix of the single frozen field permutation",
        "sampling_algorithm": (
            "lambda=0 uses the original uniform linear-index draw; otherwise draw a "
            "Bernoulli mixture component, sample full pairs for the base component, "
            "and sample uniform x1 and uniform B_m then solve x2 for the cue component"
        ),
        "root_solver": {
            "algorithm": "deterministic bisection over [0,nextafter(1,0)]",
            "iterations": ROOT_ITERATIONS,
            "rounded_lambda_hardcoded": False,
        },
        "enumeration_tolerance": ENUMERATION_TOLERANCE,
        "analytic_identities": {
            "X1_uniform": (
                "both mixture components draw X1 uniformly"
            ),
            "X2_uniform": (
                "c != 45 makes the cue-component affine map from X1 to X2 bijective "
                "for each b"
            ),
            "Z_uniform": (
                "c != 0 makes Z=c*X1+b bijective in X1 for each b"
            ),
            "Y_uniform": "the shift-invariant noise channel preserves uniform Z",
            "singleton_symmetry": (
                "conditional target distributions have the same m boosted and p-m "
                "baseline masses for either singleton parent"
            ),
        },
        "lambda_zero_G1_check": {
            "passed": True,
            "maximum_pair_probability_error": g1_error,
        },
        "conditions": conditions,
    }


def write_cue_distribution_audit(
    output: str | Path = "artifacts/cue_distribution_audit.json",
) -> tuple[Path, str]:
    path = atomic_write_json(output, build_cue_distribution_audit())
    return path, file_sha256(path)
