from __future__ import annotations

from functools import lru_cache
import hashlib
import math
from pathlib import Path
from typing import Any, Iterator

import numpy as np

from .cue_distribution import ETA, ENUMERATION_TOLERANCE, ROOT_ITERATIONS
from .durability import atomic_write_bytes, atomic_write_json
from .environment import file_sha256
from .information import noise_entropy
from .parent_pair_split import P


RULE_SEED_SHA256 = (
    "b27ac3dd3035423188193159d1b1e2ad954adef65f3843022a63b8ff7fefff2e"
)
PERMUTATION_SEEDS_SHA256 = (
    "1f31b4ee7abe61520cbfd1d12ee54908e4cdcc5d6153fdfbcbae64d65b57f642",
    "25805e63757592cc4a5f60fe9e4b40fb8b100113b6ee6b99b0d41f8e6fcdc262",
)
MODEL_INITIALIZATION_SEED_SHA256 = (
    "5cff01a7cab164ce82f20939a9b8a99bfc35692925fff2e824dfcb2943b98856"
)
BLOCK_RNG_SEED_SHA256 = (
    "183338fcb7b65f902ba342dc62fa235813a33f430cb595083e68a1505feb23b6"
)
V3_5_PERMUTATION_SEED_SHA256 = (
    "e6cccb69e7dba372d1a5db459b15625cc06c614ea8ba59db637d2d843ee282d9"
)
V3_5_COEFFICIENTS = (45, 36)
M = 64
TARGET_CONDITIONS = (
    ("R0", 0.0),
    ("R1", 0.016),
    ("R2", 0.065092),
)


def _sha_field_stream(seed_sha256: str) -> Iterator[int]:
    seed = bytes.fromhex(seed_sha256)
    counter = 0
    while True:
        digest = hashlib.sha256(seed + counter.to_bytes(8, "big")).digest()
        for offset in range(0, len(digest), 2):
            yield int.from_bytes(digest[offset : offset + 2], "big") % P
        counter += 1


def generate_replication_rules() -> tuple[dict[str, Any], ...]:
    stream = _sha_field_stream(RULE_SEED_SHA256)
    rules: list[dict[str, Any]] = []
    while len(rules) < 2:
        a1 = next(value for value in stream if value != 0)
        a2 = next(value for value in stream if value != 0 and value != a1)
        if (a1, a2) == V3_5_COEFFICIENTS:
            continue
        c = next(
            value
            for value in stream
            if value != 0 and value != a1 and value != a2
        )
        if any(tuple(rule["coefficients"]) == (a1, a2) for rule in rules):
            continue
        rules.append(
            {
                "rule_id": f"rule_{len(rules)}",
                "coefficients": [a1, a2],
                "cue_coefficient_c": c,
            }
        )
    return tuple(rules)


def field_permutation(seed_sha256: str) -> tuple[int, ...]:
    seed = bytes.fromhex(seed_sha256)
    scored = [
        (hashlib.sha256(seed + value.to_bytes(2, "big")).digest(), value)
        for value in range(P)
    ]
    scored.sort(key=lambda item: (item[0], item[1]))
    return tuple(value for _, value in scored)


def cue_subset(permutation_seed_sha256: str, m: int = M) -> tuple[int, ...]:
    if not 1 <= m <= P:
        raise ValueError("cue subset size must be between 1 and p")
    return field_permutation(permutation_seed_sha256)[:m]


@lru_cache(maxsize=None)
def _pair_grid(coefficients: tuple[int, int]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    linear = np.arange(P * P, dtype=np.int64)
    x1, x2 = linear // P, linear % P
    z = (coefficients[0] * x1 + coefficients[1] * x2) % P
    return x1, x2, z


def confirmatory_pair_probabilities(
    lambda_value: float,
    coefficients: tuple[int, int],
    cue_coefficient_c: int,
    permutation_seed_sha256: str,
    m: int = M,
) -> np.ndarray:
    lam = float(lambda_value)
    if not 0.0 <= lam < 1.0:
        raise ValueError("lambda must satisfy 0 <= lambda < 1")
    a1, a2 = coefficients
    if (
        a1 == 0
        or a2 == 0
        or a1 == a2
        or cue_coefficient_c in (0, a1, a2)
    ):
        raise ValueError("invalid confirmatory rule or cue coefficient")
    subset = np.zeros(P, dtype=np.bool_)
    subset[np.asarray(cue_subset(permutation_seed_sha256, m), dtype=np.int64)] = True
    x1, _, z = _pair_grid(coefficients)
    selected = subset[(z - cue_coefficient_c * x1) % P]
    probabilities = np.full(P * P, (1.0 - lam) / (P * P), dtype=np.float64)
    probabilities[selected] += lam / (m * P)
    if int(selected.sum()) != m * P:
        raise AssertionError("cue component must contain exactly m*p pairs")
    if not bool((probabilities > 0.0).all()):
        raise AssertionError("confirmatory cue distribution lost strict positivity")
    if not math.isclose(float(probabilities.sum()), 1.0, abs_tol=2.0e-15):
        raise AssertionError("confirmatory cue probabilities do not sum to one")
    return probabilities


def sample_confirmatory_pairs(
    rng: np.random.Generator,
    count: int,
    lambda_value: float,
    coefficients: tuple[int, int],
    cue_coefficient_c: int,
    permutation_seed_sha256: str,
    m: int = M,
) -> tuple[np.ndarray, np.ndarray]:
    lam = float(lambda_value)
    if not 0.0 <= lam < 1.0:
        raise ValueError("lambda must satisfy 0 <= lambda < 1")
    if lam == 0.0:
        linear = rng.integers(0, P * P, size=count, dtype=np.int64)
        return np.stack((linear // P, linear % P), axis=1), linear
    a1, a2 = coefficients
    cue_component = rng.random(count) < lam
    linear = rng.integers(0, P * P, size=count, dtype=np.int64)
    number = int(cue_component.sum())
    if number:
        x1 = rng.integers(0, P, size=number, dtype=np.int64)
        subset = np.asarray(
            cue_subset(permutation_seed_sha256, m), dtype=np.int64
        )
        b = subset[rng.integers(0, m, size=number, dtype=np.int64)]
        x2 = ((b - (a1 - cue_coefficient_c) * x1) * pow(a2, -1, P)) % P
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
    return off + ((1.0 - ETA) - off) * np.asarray(
        z_distributions, dtype=np.float64
    )


def enumerate_confirmatory_distribution(
    lambda_value: float,
    coefficients: tuple[int, int],
    cue_coefficient_c: int,
    permutation_seed_sha256: str,
    m: int = M,
) -> dict[str, Any]:
    probabilities = confirmatory_pair_probabilities(
        lambda_value,
        coefficients,
        cue_coefficient_c,
        permutation_seed_sha256,
        m,
    )
    x1, x2, z = _pair_grid(coefficients)
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
    singleton_cue_index = (information_x1 + information_x2) / information_pair
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
        "singleton_cue_index": singleton_cue_index,
        "uniformity_max_absolute_error": {
            "X1": float(np.max(np.abs(p_x1 - uniform))),
            "X2": float(np.max(np.abs(p_x2 - uniform))),
            "Z": float(np.max(np.abs(p_z - uniform))),
            "Y": float(np.max(np.abs(p_y - uniform))),
        },
        "joint_information_absolute_error": abs(
            information_pair - (math.log(P) - noise_entropy(P, ETA))
        ),
    }


def solve_confirmatory_lambda(
    target: float,
    coefficients: tuple[int, int],
    cue_coefficient_c: int,
    permutation_seed_sha256: str,
    m: int = M,
) -> dict[str, Any]:
    target_value = float(target)
    if target_value == 0.0:
        audit = enumerate_confirmatory_distribution(
            0.0, coefficients, cue_coefficient_c, permutation_seed_sha256, m
        )
        return {
            "target_singleton_cue_index": target_value,
            "lambda": 0.0,
            "iterations": 0,
            "absolute_residual": abs(
                audit["singleton_cue_index"] - target_value
            ),
            "audit": audit,
        }
    lower = 0.0
    upper = float(np.nextafter(1.0, 0.0))
    upper_value = enumerate_confirmatory_distribution(
        upper, coefficients, cue_coefficient_c, permutation_seed_sha256, m
    )["singleton_cue_index"]
    if not 0.0 < target_value < upper_value:
        raise ValueError("target cue index is outside the feasible interval")
    for _ in range(ROOT_ITERATIONS):
        midpoint = (lower + upper) / 2.0
        value = enumerate_confirmatory_distribution(
            midpoint,
            coefficients,
            cue_coefficient_c,
            permutation_seed_sha256,
            m,
        )["singleton_cue_index"]
        if value < target_value:
            lower = midpoint
        else:
            upper = midpoint
    candidates = (lower, upper, (lower + upper) / 2.0)
    audits = [
        enumerate_confirmatory_distribution(
            value, coefficients, cue_coefficient_c, permutation_seed_sha256, m
        )
        for value in candidates
    ]
    best = min(
        audits,
        key=lambda item: abs(item["singleton_cue_index"] - target_value),
    )
    return {
        "target_singleton_cue_index": target_value,
        "lambda": best["lambda"],
        "iterations": ROOT_ITERATIONS,
        "final_bracket": [lower, upper],
        "absolute_residual": abs(
            best["singleton_cue_index"] - target_value
        ),
        "audit": best,
    }


def build_confirmatory_distribution_audit() -> dict[str, Any]:
    rules = generate_replication_rules()
    old_permutation = field_permutation(V3_5_PERMUTATION_SEED_SHA256)
    permutations = [
        {
            "permutation_id": f"permutation_{index}",
            "seed_sha256": seed,
            "field_permutation": list(field_permutation(seed)),
            "B_64": list(cue_subset(seed)),
        }
        for index, seed in enumerate(PERMUTATION_SEEDS_SHA256)
    ]
    if any(tuple(item["field_permutation"]) == old_permutation for item in permutations):
        raise AssertionError("confirmatory cue permutation reuses v3.5")
    conditions = []
    for rule in rules:
        coefficients = tuple(rule["coefficients"])
        c = int(rule["cue_coefficient_c"])
        for permutation in permutations:
            for condition_id, target in TARGET_CONDITIONS:
                solved = solve_confirmatory_lambda(
                    target,
                    coefficients,
                    c,
                    permutation["seed_sha256"],
                )
                audit = solved["audit"]
                errors = list(audit["uniformity_max_absolute_error"].values()) + [
                    audit["joint_information_absolute_error"],
                    abs(
                        audit["entropies_nats"]["H_Y_given_X1_X2"]
                        - noise_entropy(P, ETA)
                    ),
                    abs(audit["singleton_cue_index"] - target),
                ]
                if max(errors) >= ENUMERATION_TOLERANCE:
                    raise AssertionError(
                        f"confirmatory distribution identity failed for "
                        f"{rule['rule_id']}/{permutation['permutation_id']}/{condition_id}"
                    )
                if not audit["strictly_positive_full_support"]:
                    raise AssertionError("confirmatory distribution is not full support")
                conditions.append(
                    {
                        "distribution_id": (
                            f"{rule['rule_id']}_{permutation['permutation_id']}_{condition_id}"
                        ),
                        "rule_id": rule["rule_id"],
                        "coefficients": list(coefficients),
                        "cue_coefficient_c": c,
                        "permutation_id": permutation["permutation_id"],
                        "permutation_seed_sha256": permutation["seed_sha256"],
                        "condition_id": condition_id,
                        "B_m": permutation["B_64"],
                        **{
                            key: value
                            for key, value in solved.items()
                            if key != "audit"
                        },
                        **audit,
                    }
                )
    uniform_conditions = [
        condition for condition in conditions if condition["condition_id"] == "R0"
    ]
    uniform_error = max(
        max(
            abs(condition["minimum_pair_probability"] - 1.0 / (P * P)),
            abs(condition["maximum_pair_probability"] - 1.0 / (P * P)),
        )
        for condition in uniform_conditions
    )
    if uniform_error >= 1.0e-18:
        raise AssertionError("R0 does not reproduce exact uniform full support")
    return {
        "kind": "confirmatory_cross_rule_cue_distribution_audit",
        "trained": False,
        "p": P,
        "eta": ETA,
        "m": M,
        "generation_seeds_sha256": {
            "rules": RULE_SEED_SHA256,
            "cue_permutations": list(PERMUTATION_SEEDS_SHA256),
            "model_initializations": MODEL_INITIALIZATION_SEED_SHA256,
            "block_rng_streams": BLOCK_RNG_SEED_SHA256,
        },
        "rule_generation_algorithm": (
            "consume uint16 words from SHA256(seed_bytes || uint64_big_endian(counter)), "
            "reduce modulo 257, and accept values in preregistered constraint order"
        ),
        "rules": list(rules),
        "permutation_algorithm": (
            "sort field values by SHA256(seed_bytes || uint16_big_endian(value)), "
            "breaking digest ties by value"
        ),
        "permutations": permutations,
        "independence_checks": {
            "v3_5_coefficients_not_used": all(
                tuple(rule["coefficients"]) != V3_5_COEFFICIENTS for rule in rules
            ),
            "v3_5_permutation_not_used": all(
                permutation["seed_sha256"] != V3_5_PERMUTATION_SEED_SHA256
                and tuple(permutation["field_permutation"]) != old_permutation
                for permutation in permutations
            ),
            "two_rules_distinct": len(
                {tuple(rule["coefficients"]) for rule in rules}
            )
            == 2,
            "two_permutations_distinct": (
                permutations[0]["field_permutation"]
                != permutations[1]["field_permutation"]
            ),
        },
        "root_solver": {
            "algorithm": "independent deterministic bisection over [0,nextafter(1,0)]",
            "iterations": ROOT_ITERATIONS,
            "rounded_lambda_hardcoded": False,
        },
        "enumeration_tolerance": ENUMERATION_TOLERANCE,
        "uniform_R0_check": {
            "passed": True,
            "maximum_pair_probability_error": uniform_error,
        },
        "conditions": conditions,
    }


def write_confirmatory_distribution_audit(
    output: str | Path = "artifacts/confirmatory_distribution_audit.json",
) -> tuple[Path, str]:
    path = atomic_write_json(output, build_confirmatory_distribution_audit())
    checksum = file_sha256(path)
    atomic_write_bytes(
        path.with_suffix(path.suffix + ".sha256"),
        f"{checksum}  {path.name}\n".encode("utf-8"),
    )
    return path, checksum
