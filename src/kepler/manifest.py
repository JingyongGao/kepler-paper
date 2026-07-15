from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any

from .config import DEFAULT_CONFIG, ExperimentConfig
from .model import size_spec
from .topology import FiniteFieldRule, TOPOLOGIES


@dataclass(frozen=True)
class Track:
    track_id: str
    k: int
    d: int
    target_core_parameters: int
    width: int
    actual_core_parameters: int
    total_parameters: int
    rule_instance: int
    rule: dict[str, object]
    training_seed: int
    reproduction: bool


def _track(
    k: int,
    d: int,
    target: int,
    rule_instance: int,
    training_seed: int,
    reproduction: bool,
    cfg: ExperimentConfig,
) -> Track:
    size = size_spec(target, cfg.model, cfg.field.p)
    rule = FiniteFieldRule.sample(
        k, d, rule_instance, cfg.training.rule_seed, cfg.field
    )
    suffix = "repro" if reproduction else "primary"
    track_id = f"k{k}_d{d}_n{target // 1_000_000}m_r{rule_instance}_s{training_seed}_{suffix}"
    return Track(
        track_id=track_id,
        k=k,
        d=d,
        target_core_parameters=target,
        width=size.width,
        actual_core_parameters=size.actual_core_parameters,
        total_parameters=size.total_parameters,
        rule_instance=rule_instance,
        rule=rule.to_dict(),
        training_seed=training_seed,
        reproduction=reproduction,
    )


def build_manifest(cfg: ExperimentConfig = DEFAULT_CONFIG) -> list[Track]:
    tracks: list[Track] = []
    for k, d in TOPOLOGIES:
        for target in cfg.model.target_core_parameters:
            for instance in range(cfg.training.rule_instances):
                tracks.append(
                    _track(
                        k,
                        d,
                        target,
                        instance,
                        cfg.training.seed_primary,
                        False,
                        cfg,
                    )
                )
            tracks.append(
                _track(
                    k,
                    d,
                    target,
                    cfg.training.reproduction_rule_instance,
                    cfg.training.seed_reproduction,
                    True,
                    cfg,
                )
            )
    if len(tracks) != 80 or len({t.track_id for t in tracks}) != 80:
        raise AssertionError("the preregistered design must expand to 80 unique tracks")
    return tracks


def manifest_payload(cfg: ExperimentConfig = DEFAULT_CONFIG) -> dict[str, Any]:
    tracks = build_manifest(cfg)
    return {
        "project": "Project Kepler",
        "version": "3.2.2",
        "log_base": "e",
        "exposure_definition": "number of fresh next-token targets included in NLL",
        "config": cfg.to_dict(),
        "tracks": [asdict(t) for t in tracks],
    }


def write_manifest(path: str | Path, cfg: ExperimentConfig = DEFAULT_CONFIG) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(manifest_payload(cfg), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return output


def find_track(track_id: str, cfg: ExperimentConfig = DEFAULT_CONFIG) -> Track:
    for track in build_manifest(cfg):
        if track.track_id == track_id:
            return track
    raise KeyError(f"unknown track: {track_id}")


def rule_from_track(track: Track) -> FiniteFieldRule:
    raw = track.rule
    return FiniteFieldRule(
        p=int(raw["p"]),
        eta=float(raw["eta"]),
        k=int(raw["k"]),
        d=int(raw["d"]),
        lags=tuple(int(v) for v in raw["lags"]),  # type: ignore[arg-type]
        coefficients=tuple(int(v) for v in raw["coefficients"]),  # type: ignore[arg-type]
        instance=int(raw["instance"]),
    )


def track_from_dict(raw: dict[str, Any]) -> Track:
    return Track(
        track_id=str(raw["track_id"]),
        k=int(raw["k"]),
        d=int(raw["d"]),
        target_core_parameters=int(raw["target_core_parameters"]),
        width=int(raw["width"]),
        actual_core_parameters=int(raw["actual_core_parameters"]),
        total_parameters=int(raw["total_parameters"]),
        rule_instance=int(raw["rule_instance"]),
        rule=dict(raw["rule"]),
        training_seed=int(raw["training_seed"]),
        reproduction=bool(raw["reproduction"]),
    )
