from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np

from .durability import atomic_write_json
from .environment import file_sha256
from .information import noise_entropy
from .parent_pair_split import COEFFICIENTS, P, load_parent_pair_split


ETA = 0.1
NUMERICAL_TOLERANCE = 1.0e-12


def _entropy(probabilities: np.ndarray, axis: int | None = None) -> np.ndarray:
    values = np.asarray(probabilities, dtype=np.float64)
    terms = np.zeros_like(values)
    positive = values > 0.0
    terms[positive] = -values[positive] * np.log(values[positive])
    return terms.sum(axis=axis)


def _kl_rows(conditionals: np.ndarray, marginal: np.ndarray) -> np.ndarray:
    values = np.asarray(conditionals, dtype=np.float64)
    reference = np.asarray(marginal, dtype=np.float64)
    terms = np.zeros_like(values)
    positive = values > 0.0
    terms[positive] = values[positive] * np.log(
        values[positive] / np.broadcast_to(reference, values.shape)[positive]
    )
    return terms.sum(axis=1)


def _size_distribution(values: np.ndarray) -> dict[str, Any]:
    counts = np.asarray(values, dtype=np.int64)
    unique, frequency = np.unique(counts, return_counts=True)
    return {
        "values": counts.astype(int).tolist(),
        "histogram": {
            str(int(value)): int(number) for value, number in zip(unique, frequency)
        },
        "minimum": int(counts.min()),
        "maximum": int(counts.max()),
        "mean": float(counts.mean()),
    }


def _conditional_from_counts(joint_counts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    counts = np.asarray(joint_counts, dtype=np.float64)
    parent_counts = counts.sum(axis=1)
    if bool((parent_counts == 0).any()):
        raise AssertionError("every parent value must occur in the support")
    return counts / parent_counts[:, None], parent_counts / parent_counts.sum()


def _y_from_z(z_distribution: np.ndarray) -> np.ndarray:
    off_center = ETA / (P - 1)
    return off_center + ((1.0 - ETA) - off_center) * np.asarray(
        z_distribution, dtype=np.float64
    )


def _support_result(
    name: str,
    selected: np.ndarray,
    x1: np.ndarray,
    x2: np.ndarray,
    z: np.ndarray,
) -> dict[str, Any]:
    selected = np.asarray(selected, dtype=np.bool_)
    support_size = int(selected.sum())
    x1_counts = np.bincount(x1[selected], minlength=P)
    x2_counts = np.bincount(x2[selected], minlength=P)
    z_counts = np.bincount(z[selected], minlength=P)
    x1_z_counts = np.zeros((P, P), dtype=np.int64)
    x2_z_counts = np.zeros((P, P), dtype=np.int64)
    np.add.at(x1_z_counts, (x1[selected], z[selected]), 1)
    np.add.at(x2_z_counts, (x2[selected], z[selected]), 1)

    p_z = z_counts.astype(np.float64) / support_size
    p_z_given_x1, p_x1 = _conditional_from_counts(x1_z_counts)
    p_z_given_x2, p_x2 = _conditional_from_counts(x2_z_counts)
    p_y = _y_from_z(p_z)
    p_y_given_x1 = _y_from_z(p_z_given_x1)
    p_y_given_x2 = _y_from_z(p_z_given_x2)

    h_z = float(_entropy(p_z))
    h_z_given_x1 = float(np.dot(p_x1, _entropy(p_z_given_x1, axis=1)))
    h_z_given_x2 = float(np.dot(p_x2, _entropy(p_z_given_x2, axis=1)))
    z_kl_x1 = _kl_rows(p_z_given_x1, p_z)
    z_kl_x2 = _kl_rows(p_z_given_x2, p_z)
    i_z_x1 = float(np.dot(p_x1, z_kl_x1))
    i_z_x2 = float(np.dot(p_x2, z_kl_x2))

    h_y = float(_entropy(p_y))
    h_y_given_x1 = float(np.dot(p_x1, _entropy(p_y_given_x1, axis=1)))
    h_y_given_x2 = float(np.dot(p_x2, _entropy(p_y_given_x2, axis=1)))
    y_kl_x1 = _kl_rows(p_y_given_x1, p_y)
    y_kl_x2 = _kl_rows(p_y_given_x2, p_y)
    i_y_x1 = float(np.dot(p_x1, y_kl_x1))
    i_y_x2 = float(np.dot(p_x2, y_kl_x2))
    h_noise = noise_entropy(P, ETA)
    i_y_pair = h_y - h_noise

    return {
        "support": name,
        "pairs": support_size,
        "entropies_nats": {
            "H_Z": h_z,
            "H_Z_given_X1": h_z_given_x1,
            "H_Z_given_X2": h_z_given_x2,
            "H_Y": h_y,
            "H_Y_given_X1": h_y_given_x1,
            "H_Y_given_X2": h_y_given_x2,
            "H_Y_given_X1_X2": h_noise,
        },
        "mutual_information_nats": {
            "I_Z_X1": i_z_x1,
            "I_Z_X2": i_z_x2,
            "I_Y_X1": i_y_x1,
            "I_Y_X2": i_y_x2,
            "I_Y_X1_X2": i_y_pair,
        },
        "entropy_identity_absolute_errors": {
            "I_Z_X1": abs(i_z_x1 - (h_z - h_z_given_x1)),
            "I_Z_X2": abs(i_z_x2 - (h_z - h_z_given_x2)),
            "I_Y_X1": abs(i_y_x1 - (h_y - h_y_given_x1)),
            "I_Y_X2": abs(i_y_x2 - (h_y - h_y_given_x2)),
        },
        "singleton_cue_ratio": (i_y_x1 + i_y_x2) / i_y_pair,
        "support_size_distributions": {
            "row_X1": _size_distribution(x1_counts),
            "column_X2": _size_distribution(x2_counts),
            "target_state_Z": _size_distribution(z_counts),
        },
        "marginals": {
            "P_Z": p_z.tolist(),
            "P_Y": p_y.tolist(),
            "P_X1": p_x1.tolist(),
            "P_X2": p_x2.tolist(),
        },
        "per_parent_kl_nats": {
            "KL_P_Z_given_X1_from_P_Z": {
                "values": z_kl_x1.tolist(),
                "weighted_mean": i_z_x1,
                "minimum": float(z_kl_x1.min()),
                "maximum": float(z_kl_x1.max()),
            },
            "KL_P_Z_given_X2_from_P_Z": {
                "values": z_kl_x2.tolist(),
                "weighted_mean": i_z_x2,
                "minimum": float(z_kl_x2.min()),
                "maximum": float(z_kl_x2.max()),
            },
            "KL_P_Y_given_X1_from_P_Y": {
                "values": y_kl_x1.tolist(),
                "weighted_mean": i_y_x1,
                "minimum": float(y_kl_x1.min()),
                "maximum": float(y_kl_x1.max()),
            },
            "KL_P_Y_given_X2_from_P_Y": {
                "values": y_kl_x2.tolist(),
                "weighted_mean": i_y_x2,
                "minimum": float(y_kl_x2.min()),
                "maximum": float(y_kl_x2.max()),
            },
        },
    }


def build_support_information_audit(
    split_path: str | Path = "artifacts/parent_pair_split.json",
) -> dict[str, Any]:
    linear = np.arange(P * P, dtype=np.int64)
    x1, x2 = linear // P, linear % P
    z = (COEFFICIENTS[0] * x1 + COEFFICIENTS[1] * x2) % P
    _, _, heldout = load_parent_pair_split(split_path)
    masks = {
        "G1_full_support": np.ones(P * P, dtype=np.bool_),
        "G2_training_support": ~heldout,
        "G2_heldout_support": heldout,
    }
    supports = {
        name: _support_result(name, mask, x1, x2, z)
        for name, mask in masks.items()
    }
    full = supports["G1_full_support"]
    full_mi = full["mutual_information_nats"]
    log_p = math.log(P)
    analytic_errors = {
        "H_Z_minus_log_p": abs(full["entropies_nats"]["H_Z"] - log_p),
        "H_Z_given_X1_minus_log_p": abs(
            full["entropies_nats"]["H_Z_given_X1"] - log_p
        ),
        "H_Z_given_X2_minus_log_p": abs(
            full["entropies_nats"]["H_Z_given_X2"] - log_p
        ),
        "H_Y_minus_log_p": abs(full["entropies_nats"]["H_Y"] - log_p),
        "H_Y_given_X1_minus_log_p": abs(
            full["entropies_nats"]["H_Y_given_X1"] - log_p
        ),
        "H_Y_given_X2_minus_log_p": abs(
            full["entropies_nats"]["H_Y_given_X2"] - log_p
        ),
        "I_Z_X1": abs(full_mi["I_Z_X1"]),
        "I_Z_X2": abs(full_mi["I_Z_X2"]),
        "I_Y_X1": abs(full_mi["I_Y_X1"]),
        "I_Y_X2": abs(full_mi["I_Y_X2"]),
        "singleton_cue_ratio": abs(full["singleton_cue_ratio"]),
    }
    if max(analytic_errors.values()) >= NUMERICAL_TOLERANCE:
        raise AssertionError("G1 singleton-information analytic check failed")
    return {
        "kind": "exact_support_induced_information_audit",
        "trained": False,
        "optimizer_steps": 0,
        "p": P,
        "eta": ETA,
        "coefficients": list(COEFFICIENTS),
        "logarithm_base": "natural",
        "split_file": str(Path(split_path)),
        "split_sha256": file_sha256(split_path),
        "numerical_tolerance": NUMERICAL_TOLERANCE,
        "analytic_full_support_statement": (
            "For uniform independent X1,X2 and nonzero field coefficients, "
            "Z is uniform conditional on either singleton parent; the shifted "
            "noise channel therefore also leaves Y uniform conditional on either parent."
        ),
        "G1_analytic_and_numeric_zero_check": {
            "passed": True,
            "absolute_errors": analytic_errors,
        },
        "supports": supports,
    }


def write_support_information_audit(
    output: str | Path = "artifacts/support_information_audit.json",
    split_path: str | Path = "artifacts/parent_pair_split.json",
) -> tuple[Path, str]:
    path = atomic_write_json(output, build_support_information_audit(split_path))
    return path, file_sha256(path)
