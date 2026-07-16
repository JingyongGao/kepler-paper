from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
from typing import Any

from .config import DEFAULT_CONFIG, ExperimentConfig
from .durability import atomic_write_json, transactional_append_jsonl
from .environment import environment_snapshot, file_sha256
from .learnability import run_checkpoint_probes
from .mechanism import (
    as_learnability_run,
    build_mechanism_isolation_runs,
    find_mechanism_run,
)
from .pair_diagnostics import evaluate_all_pair_checkpoints, train_pair_diagnostic


def _assert_manifest(path: str | Path) -> str:
    manifest = Path(path)
    if not manifest.exists():
        raise FileNotFoundError(
            "mechanism manifest must be created explicitly before running diagnostics"
        )
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    if payload.get("kind") != "mechanism_isolation_audit":
        raise ValueError("wrong manifest kind")
    if not all(payload["flags"].values()):
        raise ValueError("mechanism manifest is not fully excluded/diagnostic")
    return file_sha256(manifest)


def _run_recurrent(
    run_id: str,
    run_root: str | Path,
    manifest_hash: str,
    device_name: str,
    batch_size: int | None,
    resume: str | Path | None,
    endpoint: int,
    cfg: ExperimentConfig,
) -> dict[str, Any]:
    from .training import evaluate_checkpoint, train_track

    mechanism_run = find_mechanism_run(run_id, cfg)
    run = as_learnability_run(mechanism_run, cfg)
    provenance = {
        **environment_snapshot(
            cfg.to_dict(),
            production_manifest="artifacts/manifest.json",
            pilot_manifest="artifacts/pilot_manifest.json",
            seeds={
                "model": run.model_initialization_seed,
                "data": run.data_seed,
                "evaluation": run.evaluation_seed,
                "probe": run.probe_seeds,
            },
            workdir=Path.cwd(),
        ),
        "mechanism_isolation_manifest_hash": manifest_hash,
        "analysis_flags": {
            "exploratory": True,
            "diagnostic_only": True,
            "excluded_from_confirmatory_fit": True,
        },
    }
    directory = train_track(
        run.run_id,
        output_root=run_root,
        max_tokens=endpoint,
        batch_size=batch_size,
        device_name=device_name,
        resume=resume,
        cfg=cfg,
        track_override=run.as_training_track(),
        checkpoint_tokens=run.checkpoints,
        stream_seed_override=run.data_seed,
        provenance=provenance,
        loss_type=run.loss_type,
    )
    evaluations_file = directory / "evaluations.jsonl"
    existing = set()
    if evaluations_file.exists():
        existing = {
            int(json.loads(line)["exposed_tokens"])
            for line in evaluations_file.read_text(encoding="utf-8").splitlines()
            if line.strip()
        }
    evaluations = []
    for exposure in run.checkpoints:
        if exposure > endpoint or exposure in existing:
            continue
        checkpoint = directory / f"checkpoint_{exposure}.pt"
        result = evaluate_checkpoint(
            checkpoint,
            episodes=run.evaluation_episodes,
            batch_size=batch_size or cfg.training.batch_size,
            device_name=device_name,
            evaluation_seed_override=run.evaluation_seed,
            cfg=cfg,
        )
        summary = {
            key: result[key]
            for key in (
                "track_id",
                "exposed_tokens",
                "R",
                "R_bootstrap_95_ci",
                "mean_exact_conditional_kl_nats",
                "center_state_top1_accuracy",
                "mean_probability_on_center_state",
                "mean_predictive_entropy_nats",
                "sampled_token_nll_nats",
                "fresh_episode_nll_nats",
                "per_block_gradient_norm",
                "per_block_parameter_update_over_parameter_norm",
                "finite",
            )
        }
        transactional_append_jsonl(evaluations_file, summary)
        evaluations.append(summary)
    for exposure in run.probe_checkpoints:
        if exposure > endpoint:
            continue
        output = directory / f"checkpoint_{exposure}.probes.json"
        if not output.exists():
            run_checkpoint_probes(
                directory / f"checkpoint_{exposure}.pt",
                run,
                output,
                device_name,
                cfg,
            )
    return {
        "run": asdict(mechanism_run),
        "run_directory": str(directory.resolve()),
        "new_evaluations": evaluations,
    }


def run_mechanism_isolation(
    run_id: str,
    run_root: str | Path = "artifacts/mechanism_runs",
    manifest_path: str | Path = "artifacts/mechanism_isolation_manifest.json",
    *,
    device_name: str = "auto",
    batch_size: int | None = None,
    resume: str | Path | None = None,
    stop_after_exposure: int | None = None,
    cfg: ExperimentConfig = DEFAULT_CONFIG,
) -> dict[str, Any]:
    manifest_hash = _assert_manifest(manifest_path)
    run = find_mechanism_run(run_id, cfg)
    endpoint = stop_after_exposure or run.maximum_exposure
    if endpoint not in run.checkpoints:
        raise ValueError("stop_after_exposure must be a registered checkpoint")
    if run.model_kind == "recurrent_transformer":
        return _run_recurrent(
            run_id,
            run_root,
            manifest_hash,
            device_name,
            batch_size,
            resume,
            endpoint,
            cfg,
        )
    directory = train_pair_diagnostic(
        run,
        run_root,
        device_name=device_name,
        batch_size=batch_size or 4096,
        resume=resume,
        stop_after_examples=endpoint,
        cfg=cfg,
    )
    evaluations = evaluate_all_pair_checkpoints(
        directory, run, endpoint, device_name=device_name, cfg=cfg
    )
    return {
        "run": asdict(run),
        "run_directory": str(directory.resolve()),
        "new_evaluations": evaluations,
    }


def mechanism_status(
    run_root: str | Path = "artifacts/mechanism_runs",
    cfg: ExperimentConfig = DEFAULT_CONFIG,
) -> list[dict[str, Any]]:
    root = Path(run_root)
    output = []
    for run in build_mechanism_isolation_runs(cfg):
        directory = root / run.run_id
        checkpoints = [
            exposure
            for exposure in run.checkpoints
            if (directory / f"checkpoint_{exposure}.pt").exists()
        ]
        evaluations = []
        if (directory / "evaluations.jsonl").exists():
            key = "exposed_tokens" if run.model_kind == "recurrent_transformer" else "exposed_examples"
            evaluations = [
                int(json.loads(line)[key])
                for line in (directory / "evaluations.jsonl").read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
        probes = [
            exposure
            for exposure in run.probe_checkpoints
            if (directory / f"checkpoint_{exposure}.probes.json").exists()
        ]
        output.append(
            {
                "run_id": run.run_id,
                "model_kind": run.model_kind,
                "completed_exposure": max(checkpoints, default=0),
                "target_exposure": run.maximum_exposure,
                "checkpoints": checkpoints,
                "evaluations": evaluations,
                "probes": probes,
                "status": (
                    "complete"
                    if checkpoints and checkpoints[-1] == run.maximum_exposure
                    else ("in_progress" if checkpoints else "pending")
                ),
            }
        )
    return output
