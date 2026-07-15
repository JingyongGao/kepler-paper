from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any

from .config import DEFAULT_CONFIG, ExperimentConfig
from .data import split_seed
from .durability import atomic_write_json, transactional_append_jsonl
from .environment import environment_snapshot, file_sha256
from .manifest import Track
from .model import size_spec
from .topology import FiniteFieldRule, TOPOLOGIES, distinct_paired_rule


PILOT_LONG_CHECKPOINTS = (5_000_000, 10_000_000, 20_000_000, 40_000_000)
# The protocol permits stopping anywhere in [1M, 5M]. We preregister the lower
# endpoint before observing soak outcomes to cap engineering cost.
PILOT_SOAK_CHECKPOINTS = (1_000_000,)
PILOT_DIFFICULT_CORNER = (4, 64)


@dataclass(frozen=True)
class PilotTrack:
    track_id: str
    tier: str
    purpose: str
    k: int
    d: int
    target_core_parameters: int
    width: int
    actual_core_parameters: int
    total_parameters: int
    rule_instance: int
    rule: dict[str, object]
    paired_new_rule: dict[str, object]
    model_initialization_seed: int
    data_seed: int
    evaluation_seed: int
    maximum_supervised_tokens: int
    checkpoints: tuple[int, ...]
    evaluation_episodes: int
    num_tasks_during_baseline: int
    analysis_role: str
    include_primary_confirmatory_fit: bool

    def as_training_track(self) -> Track:
        return Track(
            track_id=self.track_id,
            k=self.k,
            d=self.d,
            target_core_parameters=self.target_core_parameters,
            width=self.width,
            actual_core_parameters=self.actual_core_parameters,
            total_parameters=self.total_parameters,
            rule_instance=self.rule_instance,
            rule=self.rule,
            training_seed=self.model_initialization_seed,
            reproduction=False,
        )


def _pilot_track(
    tier: str,
    purpose: str,
    k: int,
    d: int,
    target: int,
    instance: int,
    checkpoints: tuple[int, ...],
    cfg: ExperimentConfig,
) -> PilotTrack:
    size = size_spec(target, cfg.model, cfg.field.p)
    old_rule = FiniteFieldRule.sample(k, d, instance, cfg.training.rule_seed, cfg.field)
    new_rule = distinct_paired_rule(old_rule, cfg.training.rule_seed)
    init_seed = 0
    data_seed = split_seed(41_000, k, d, instance, init_seed)
    evaluation_seed = split_seed(42_000, k, d, instance, init_seed)
    track_id = f"pilot_{tier}_k{k}_d{d}_n{target // 1_000_000}m_r{instance}_s{init_seed}"
    return PilotTrack(
        track_id=track_id,
        tier=tier,
        purpose=purpose,
        k=k,
        d=d,
        target_core_parameters=target,
        width=size.width,
        actual_core_parameters=size.actual_core_parameters,
        total_parameters=size.total_parameters,
        rule_instance=instance,
        rule=old_rule.to_dict(),
        paired_new_rule=new_rule.to_dict(),
        model_initialization_seed=init_seed,
        data_seed=data_seed,
        evaluation_seed=evaluation_seed,
        maximum_supervised_tokens=checkpoints[-1],
        checkpoints=checkpoints,
        evaluation_episodes=256,
        num_tasks_during_baseline=0,
        analysis_role="exploratory_engineering",
        include_primary_confirmatory_fit=False,
    )


def build_pilot_manifest(cfg: ExperimentConfig = DEFAULT_CONFIG) -> list[PilotTrack]:
    tracks: list[PilotTrack] = []
    for k, d in TOPOLOGIES:
        for instance in range(2):
            tracks.append(
                _pilot_track(
                    "2m_long",
                    "learning-range and topology identifiability",
                    k,
                    d,
                    2_000_000,
                    instance,
                    PILOT_LONG_CHECKPOINTS,
                    cfg,
                )
            )
    for k, d in ((2, 4), (4, 4), (2, 64), (4, 64)):
        tracks.append(
            _pilot_track(
                "10m_long",
                "width transfer and corner coverage",
                k,
                d,
                10_000_000,
                0,
                PILOT_LONG_CHECKPOINTS,
                cfg,
            )
        )
    k, d = PILOT_DIFFICULT_CORNER
    tracks.append(
        _pilot_track(
            "50m_soak",
            "memory/checkpoint/resume/throughput only",
            k,
            d,
            50_000_000,
            0,
            PILOT_SOAK_CHECKPOINTS,
            cfg,
        )
    )
    if len(tracks) != 15 or len({track.track_id for track in tracks}) != 15:
        raise AssertionError("micro-pilot must contain 10 + 4 + 1 unique trajectories")
    return tracks


def pilot_manifest_payload(
    cfg: ExperimentConfig = DEFAULT_CONFIG,
    production_manifest: str | Path = "artifacts/manifest.json",
) -> dict[str, Any]:
    production = Path(production_manifest)
    return {
        "project": "Project Kepler",
        "version": "3.2.2-phase4-micro-pilot",
        "production_manifest_hash": file_sha256(production) if production.exists() else None,
        "production_manifest_mutated": False,
        "analysis_role": "exploratory_engineering",
        "include_primary_confirmatory_fit": False,
        "task_conditioning": {
            "baseline_num_tasks": 0,
            "phase4_num_tasks": 2,
            "mapping": {
                "iid_control": {"task": "A", "rule": "old_a"},
                "task_interface_null": {"task": "B", "rule": "old_a"},
                "compatible_extension": {"task": "B", "rule": "new_a_prime"},
                "conflicting_rewrite": {"task": "A", "rule": "new_a_prime"},
                "fixed_table_memorization": {"task": "A", "rule": "fixed_table"},
            },
        },
        "evaluation": {
            "fixed_independent_episodes": True,
            "episode_contributions_retained": True,
            "bootstrap_replicates": 2_000,
            "confidence_interval": 0.95,
        },
        "tracks": [asdict(track) for track in build_pilot_manifest(cfg)],
    }


def write_pilot_manifest(
    path: str | Path = "artifacts/pilot_manifest.json",
    cfg: ExperimentConfig = DEFAULT_CONFIG,
    production_manifest: str | Path = "artifacts/manifest.json",
) -> tuple[Path, str]:
    output = atomic_write_json(path, pilot_manifest_payload(cfg, production_manifest))
    return output, file_sha256(output)


def find_pilot_track(track_id: str, cfg: ExperimentConfig = DEFAULT_CONFIG) -> PilotTrack:
    for track in build_pilot_manifest(cfg):
        if track.track_id == track_id:
            return track
    raise KeyError(f"unknown pilot track: {track_id}")


def pilot_status(
    run_root: str | Path = "runs/pilot",
    cfg: ExperimentConfig = DEFAULT_CONFIG,
) -> list[dict[str, Any]]:
    root = Path(run_root)
    result = []
    for track in build_pilot_manifest(cfg):
        run = root / track.track_id
        completed = [
            token
            for token in track.checkpoints
            if (run / f"checkpoint_{token}.pt").exists()
        ]
        evaluations: list[dict[str, Any]] = []
        evaluation_file = run / "evaluations.jsonl"
        if evaluation_file.exists():
            evaluations = [
                json.loads(line)
                for line in evaluation_file.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
        result.append(
            {
                "track_id": track.track_id,
                "tier": track.tier,
                "topology": [track.k, track.d],
                "completed_tokens": max(completed, default=0),
                "target_tokens": track.maximum_supervised_tokens,
                "checkpoints_complete": completed,
                "evaluations_complete": sorted(
                    {int(row["exposed_tokens"]) for row in evaluations}
                ),
                "status": "complete"
                if completed and completed[-1] == track.maximum_supervised_tokens
                else ("in_progress" if completed else "pending"),
            }
        )
    return result


def _throughput_summary(metrics_path: Path) -> dict[str, Any]:
    import numpy as np

    rows = [
        json.loads(line)
        for line in metrics_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    rates = np.asarray(
        [float(row["supervised_tokens_per_second"]) for row in rows], dtype=np.float64
    )
    warm = rates[min(5, max(0, rates.size - 1)) :]
    if warm.size == 0:
        warm = rates
    memory_keys = (
        "mps_current_allocated_bytes",
        "mps_driver_allocated_bytes",
        "mps_recommended_max_bytes",
    )
    return {
        "steps": len(rows),
        "warmup_steps_excluded": int(rates.size - warm.size),
        "tokens_per_second_median": float(np.median(warm)),
        "tokens_per_second_mean": float(np.mean(warm)),
        "tokens_per_second_cv": float(np.std(warm) / max(np.mean(warm), 1e-12)),
        **{
            f"peak_{key}": max(
                (int(row[key]) for row in rows if key in row), default=None
            )
            for key in memory_keys
        },
        "all_loss_and_gradient_metrics_finite": all(
            np.isfinite(float(row["loss_nats"]))
            and np.isfinite(float(row["gradient_norm"]))
            for row in rows
        ),
    }


def run_pilot_track(
    track_id: str,
    run_root: str | Path = "runs/pilot",
    manifest_path: str | Path = "artifacts/pilot_manifest.json",
    production_manifest: str | Path = "artifacts/manifest.json",
    device: str = "auto",
    batch_size: int | None = None,
    resume: str | Path | None = None,
    stop_after_tokens: int | None = None,
    cfg: ExperimentConfig = DEFAULT_CONFIG,
) -> dict[str, Any]:
    from .training import evaluate_checkpoint, train_track

    manifest_output, pilot_hash = write_pilot_manifest(
        manifest_path, cfg, production_manifest
    )
    track = find_pilot_track(track_id, cfg)
    endpoint = stop_after_tokens or track.maximum_supervised_tokens
    if endpoint not in track.checkpoints:
        raise ValueError("stop_after_tokens must be one of the preregistered checkpoints")
    environment = environment_snapshot(
        cfg.to_dict(),
        production_manifest=production_manifest,
        pilot_manifest=manifest_output,
        seeds={
            "model": track.model_initialization_seed,
            "rule": cfg.training.rule_seed,
            "data": track.data_seed,
            "evaluation": track.evaluation_seed,
        },
        workdir=Path.cwd(),
    )
    provenance = {
        **environment,
        "pilot_manifest_hash": pilot_hash,
        "production_manifest_hash": file_sha256(production_manifest),
    }
    run_dir = train_track(
        track.track_id,
        output_root=run_root,
        max_tokens=endpoint,
        batch_size=batch_size,
        device_name=device,
        resume=resume,
        cfg=cfg,
        track_override=track.as_training_track(),
        checkpoint_tokens=track.checkpoints,
        stream_seed_override=track.data_seed,
        provenance=provenance,
    )

    evaluations_path = run_dir / "evaluations.jsonl"
    existing = set()
    if evaluations_path.exists():
        existing = {
            int(json.loads(line)["exposed_tokens"])
            for line in evaluations_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        }
    evaluation_summaries = []
    for exposure in track.checkpoints:
        if exposure > endpoint:
            continue
        checkpoint = run_dir / f"checkpoint_{exposure}.pt"
        if not checkpoint.exists() or exposure in existing:
            continue
        result = evaluate_checkpoint(
            checkpoint,
            episodes=track.evaluation_episodes,
            batch_size=batch_size or cfg.training.batch_size,
            device_name=device,
            cfg=cfg,
            evaluation_seed_override=track.evaluation_seed,
        )
        summary = {
            "track_id": track.track_id,
            "exposed_tokens": exposure,
            "R": result["R"],
            "R_bootstrap_95_ci": result["R_bootstrap_95_ci"],
            "evaluation_episodes": result["evaluation_episodes"],
            "finite": result["finite"],
            "episode_contributions_file": str(checkpoint.with_suffix(".eval.json")),
        }
        transactional_append_jsonl(evaluations_path, summary)
        evaluation_summaries.append(summary)
    throughput = _throughput_summary(run_dir / "train.jsonl")
    atomic_write_json(run_dir / "engineering_summary.json", throughput)
    return {
        "track": asdict(track),
        "run_directory": str(run_dir.resolve()),
        "environment": environment,
        "pilot_manifest_hash": pilot_hash,
        "throughput_and_memory": throughput,
        "new_evaluations": evaluation_summaries,
    }
