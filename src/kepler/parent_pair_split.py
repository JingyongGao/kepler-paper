from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Literal

import numpy as np

from .durability import atomic_write_json
from .environment import file_sha256


P = 257
COEFFICIENTS = (45, 36)
SPLIT_SEED_SHA256 = "ba2f84572c3b0c5d3f27b0bf5d8c8e92505cc0a5b18c9274b0ef013c3d90c02f"
HELDOUT_COUNT = 13_210


def _pair_score(x1: int, x2: int) -> bytes:
    payload = bytes.fromhex(SPLIT_SEED_SHA256)
    payload += int(x1).to_bytes(2, "big") + int(x2).to_bytes(2, "big")
    return hashlib.sha256(payload).digest()


def build_parent_pair_split() -> dict[str, Any]:
    scored = [
        (_pair_score(linear // P, linear % P), linear)
        for linear in range(P * P)
    ]
    scored.sort(key=lambda item: (item[0], item[1]))
    heldout = np.asarray(sorted(linear for _, linear in scored[:HELDOUT_COUNT]), dtype=np.int64)
    mask = np.zeros(P * P, dtype=np.uint8)
    mask[heldout] = 1
    train = np.flatnonzero(mask == 0).astype(np.int64)
    pairs = np.stack((np.arange(P * P) // P, np.arange(P * P) % P), axis=1)
    z = (COEFFICIENTS[0] * pairs[:, 0] + COEFFICIENTS[1] * pairs[:, 1]) % P

    def marginal(indices: np.ndarray, values: np.ndarray) -> list[int]:
        return np.bincount(values[indices], minlength=P).astype(int).tolist()

    marginals = {}
    for name, indices in (("train", train), ("heldout", heldout)):
        x1_counts = marginal(indices, pairs[:, 0])
        x2_counts = marginal(indices, pairs[:, 1])
        z_counts = marginal(indices, z)
        marginals[name] = {
            "x1_counts": x1_counts,
            "x2_counts": x2_counts,
            "z_counts": z_counts,
            "x1_min_max": [min(x1_counts), max(x1_counts)],
            "x2_min_max": [min(x2_counts), max(x2_counts)],
            "z_min_max": [min(z_counts), max(z_counts)],
        }
    if any(value == 0 for name in marginals.values() for key, counts in name.items() if key.endswith("_counts") for value in counts):
        raise AssertionError("each x1, x2, and z value must occur in both partitions")

    # The restricted sampler indexes only the explicit train vector.  This
    # million-draw audit supplements the construction-time set assertion.
    rng = np.random.default_rng(33_257)
    sampled = train[rng.integers(0, train.size, size=1_000_000)]
    leakage = int(mask[sampled].sum())
    if leakage != 0 or np.intersect1d(train, heldout).size:
        raise AssertionError("held-out pair leaked into the training support")
    assignment_hash = hashlib.sha256(mask.tobytes(order="C")).hexdigest()
    return {
        "kind": "frozen_ordered_parent_pair_split",
        "p": P,
        "coefficients": list(COEFFICIENTS),
        "preregistered_sha256_seed": SPLIT_SEED_SHA256,
        "algorithm": (
            "score each big-endian uint16 (x1,x2) by SHA256(seed_bytes || x1 || x2); "
            "sort lexicographically by digest then linear index; first 13210 are held out"
        ),
        "linear_index_convention": "index = 257*x1 + x2",
        "assignment_encoding": "66049 uint8 bytes in linear-index order; 0=train, 1=heldout",
        "assignment_sha256": assignment_hash,
        "total_pairs": P * P,
        "train_pairs": int(train.size),
        "heldout_pairs": int(heldout.size),
        "train_fraction": train.size / (P * P),
        "heldout_fraction": heldout.size / (P * P),
        "heldout_linear_indices": heldout.astype(int).tolist(),
        "marginals": marginals,
        "coverage_assertions": {
            "every_x1_in_both": True,
            "every_x2_in_both": True,
            "every_z_in_both": True,
        },
        "restricted_sampler_audit": {
            "draws": 1_000_000,
            "seed": 33_257,
            "pair_leakage_count": leakage,
            "construction_guarantee": "sampler indexes only the complement of heldout_linear_indices",
        },
    }


def write_parent_pair_split(
    output: str | Path = "artifacts/parent_pair_split.json",
) -> tuple[Path, str]:
    path = atomic_write_json(output, build_parent_pair_split())
    return path, file_sha256(path)


def load_parent_pair_split(
    path: str | Path = "artifacts/parent_pair_split.json",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    import json

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload["preregistered_sha256_seed"] != SPLIT_SEED_SHA256:
        raise ValueError("parent-pair split seed mismatch")
    heldout = np.asarray(payload["heldout_linear_indices"], dtype=np.int64)
    mask = np.zeros(P * P, dtype=np.bool_)
    mask[heldout] = True
    if hashlib.sha256(mask.astype(np.uint8).tobytes()).hexdigest() != payload["assignment_sha256"]:
        raise ValueError("parent-pair assignment hash mismatch")
    train = np.flatnonzero(~mask).astype(np.int64)
    return train, heldout, mask


def sample_parent_pairs(
    rng: np.random.Generator,
    count: int,
    support: Literal["full", "train"],
    split_path: str | Path = "artifacts/parent_pair_split.json",
) -> tuple[np.ndarray, np.ndarray]:
    if support == "full":
        linear = rng.integers(0, P * P, size=count, dtype=np.int64)
    elif support == "train":
        train, _, heldout_mask = load_parent_pair_split(split_path)
        linear = train[rng.integers(0, train.size, size=count)]
        if bool(heldout_mask[linear].any()):
            raise AssertionError("restricted sampler emitted a held-out pair")
    else:
        raise ValueError("support must be full or train")
    return np.stack((linear // P, linear % P), axis=1), linear
