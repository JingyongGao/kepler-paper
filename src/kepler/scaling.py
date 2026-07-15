from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import least_squares


MODEL_NAMES = ("M0", "M1", "M2")


@dataclass(frozen=True)
class ScalingData:
    topology: NDArray[np.int64]
    topology_labels: tuple[str, ...]
    n_core: NDArray[np.float64]
    tokens: NDArray[np.float64]
    response: NDArray[np.float64]
    track_ids: NDArray[np.str_]


def prepare_scaling_data(records: Sequence[Mapping[str, Any]]) -> ScalingData:
    required = {"k", "d", "N_core", "D", "R", "track_id"}
    for row in records:
        missing = required - row.keys()
        if missing:
            raise ValueError(f"scaling row is missing {sorted(missing)}")
    labels = tuple(sorted({f"k{int(r['k'])}_d{int(r['d'])}" for r in records}))
    label_to_index = {label: i for i, label in enumerate(labels)}
    topology = np.asarray(
        [label_to_index[f"k{int(r['k'])}_d{int(r['d'])}"] for r in records],
        dtype=np.int64,
    )
    n_core = np.asarray([float(r["N_core"]) for r in records], dtype=np.float64)
    tokens = np.asarray([float(r["D"]) for r in records], dtype=np.float64)
    response = np.asarray([float(r["R"]) for r in records], dtype=np.float64)
    tracks = np.asarray([str(r["track_id"]) for r in records], dtype=np.str_)
    if np.any(n_core <= 0) or np.any(tokens <= 0) or np.any(response < 0):
        raise ValueError("N_core and D must be positive and R must be nonnegative")
    return ScalingData(topology, labels, n_core, tokens, response, tracks)


def _positive(raw: NDArray[np.float64]) -> NDArray[np.float64]:
    return np.exp(np.clip(raw, -30.0, 30.0))


def _initial(data: ScalingData, model: str) -> NDArray[np.float64]:
    t = len(data.topology_labels)
    floor = max(float(np.min(data.response)) * 0.5, 1.0e-5)
    residual = max(float(np.mean(data.response)) - floor, 1.0e-4)
    a0 = 0.3
    b0 = 0.3
    if model == "M0":
        values: list[float] = []
        for i in range(t):
            mask = data.topology == i
            e = max(float(np.min(data.response[mask])) * 0.5, 1e-5)
            amp_n = residual * math.exp(a0 * float(np.mean(np.log(data.n_core[mask])))) / 2
            amp_d = residual * math.exp(b0 * float(np.mean(np.log(data.tokens[mask])))) / 2
            values.extend((math.log(e), math.log(amp_n), math.log(amp_d)))
        values.extend((math.log(a0), math.log(b0)))
        return np.asarray(values)
    if model == "M1":
        values = []
        for i in range(t):
            mask = data.topology == i
            e = max(float(np.min(data.response[mask])) * 0.5, 1e-5)
            amp_n = residual * math.exp(a0 * float(np.mean(np.log(data.n_core[mask])))) / 2
            amp_d = residual * math.exp(b0 * float(np.mean(np.log(data.tokens[mask])))) / 2
            values.extend((math.log(e), math.log(amp_n), math.log(amp_d), math.log(a0), math.log(b0)))
        return np.asarray(values)
    if model == "M2":
        values = []
        for i in range(t):
            mask = data.topology == i
            values.extend(
                (
                    math.log(max(float(np.mean(data.response[mask])), 1e-5)),
                    math.log(max(float(np.mean(data.response[mask])), 1e-5)),
                    math.log(float(np.median(data.n_core[mask]))),
                    math.log(float(np.median(data.tokens[mask]))),
                )
            )
        values.extend((math.log(1.0), math.log(1.0)))
        return np.asarray(values)
    raise ValueError(f"unknown model {model}")


def _predict(raw: NDArray[np.float64], data: ScalingData, model: str) -> NDArray[np.float64]:
    params = _positive(raw)
    t = len(data.topology_labels)
    result = np.empty_like(data.response)
    if model == "M0":
        a, b = params[-2:]
        for i in range(t):
            e, amp_n, amp_d = params[3 * i : 3 * i + 3]
            mask = data.topology == i
            result[mask] = e + amp_n * data.n_core[mask] ** (-a) + amp_d * data.tokens[mask] ** (-b)
        return result
    if model == "M1":
        for i in range(t):
            e, amp_n, amp_d, a, b = params[5 * i : 5 * i + 5]
            mask = data.topology == i
            result[mask] = e + amp_n * data.n_core[mask] ** (-a) + amp_d * data.tokens[mask] ** (-b)
        return result
    if model == "M2":
        a, b = params[-2:]
        for i in range(t):
            amp_n, amp_d, nc, dc = params[4 * i : 4 * i + 4]
            mask = data.topology == i
            # Evaluate the sigmoid denominators in log space so optimizer
            # excursions cannot overflow a raw power operation.
            n_power = np.exp(
                np.clip(a * (np.log(data.n_core[mask]) - math.log(nc)), -60.0, 60.0)
            )
            d_power = np.exp(
                np.clip(b * (np.log(data.tokens[mask]) - math.log(dc)), -60.0, 60.0)
            )
            result[mask] = amp_n / (1 + n_power) + amp_d / (1 + d_power)
        return result
    raise ValueError(f"unknown model {model}")


def _parameter_payload(raw: NDArray[np.float64], data: ScalingData, model: str) -> dict[str, Any]:
    p = _positive(raw)
    result: dict[str, Any] = {"topology": {}}
    if model == "M0":
        for i, label in enumerate(data.topology_labels):
            e, a_n, a_d = p[3 * i : 3 * i + 3]
            result["topology"][label] = {"E": e, "A": a_n, "B": a_d}
        result["shared"] = {"a": p[-2], "b": p[-1]}
    elif model == "M1":
        for i, label in enumerate(data.topology_labels):
            e, a_n, a_d, a, b = p[5 * i : 5 * i + 5]
            result["topology"][label] = {"E": e, "A": a_n, "B": a_d, "a": a, "b": b}
    else:
        for i, label in enumerate(data.topology_labels):
            a_n, a_d, nc, dc = p[4 * i : 4 * i + 4]
            result["topology"][label] = {"A": a_n, "B": a_d, "N_c": nc, "D_c": dc}
        result["shared"] = {"a": p[-2], "b": p[-1]}
    return result


def fit_model(data: ScalingData, model: str) -> dict[str, Any]:
    initial = _initial(data, model)
    fit = least_squares(
        lambda raw: _predict(raw, data, model) - data.response,
        initial,
        max_nfev=20_000,
        loss="linear",
    )
    residual = _predict(fit.x, data, model) - data.response
    return {
        "model": model,
        "success": bool(fit.success),
        "message": fit.message,
        "mse": float(np.mean(residual**2)),
        "parameters": _parameter_payload(fit.x, data, model),
        "raw": fit.x,
    }


def _subset(data: ScalingData, mask: NDArray[np.bool_]) -> ScalingData:
    # Keep the global topology label/index system. Each training fold is required
    # to contain all topologies, which is true for leave-one-trajectory-out here.
    return ScalingData(
        data.topology[mask],
        data.topology_labels,
        data.n_core[mask],
        data.tokens[mask],
        data.response[mask],
        data.track_ids[mask],
    )


def compare_models_leave_one_trajectory_out(data: ScalingData) -> dict[str, Any]:
    unique_tracks = np.unique(data.track_ids)
    if unique_tracks.size < 2:
        raise ValueError("held-out comparison needs at least two trajectories")
    squared_errors = {name: [] for name in MODEL_NAMES}
    fold_summary = []
    for track in unique_tracks:
        test_mask = data.track_ids == track
        train_mask = ~test_mask
        if set(np.unique(data.topology[train_mask])) != set(range(len(data.topology_labels))):
            raise ValueError("every training fold must retain every topology")
        train = _subset(data, train_mask)
        test = _subset(data, test_mask)
        fold = {"track_id": str(track), "mse": {}}
        for name in MODEL_NAMES:
            fit = fit_model(train, name)
            prediction = _predict(fit["raw"], test, name)
            errors = (prediction - test.response) ** 2
            squared_errors[name].extend(errors.tolist())
            fold["mse"][name] = float(np.mean(errors))
        fold_summary.append(fold)
    mse = {name: float(np.mean(values)) for name, values in squared_errors.items()}
    return {
        "holdout_unit": "entire trajectory (all D checkpoints)",
        "mse": mse,
        "error_ratio_to_M0": {name: mse[name] / mse["M0"] for name in MODEL_NAMES},
        "folds": fold_summary,
        "full_data_fits": {
            name: {k: v for k, v in fit_model(data, name).items() if k != "raw"}
            for name in MODEL_NAMES
        },
    }
