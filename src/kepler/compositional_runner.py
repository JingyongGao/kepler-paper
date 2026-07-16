from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .compositional import build_compositional_runs, find_compositional_run
from .compositional_evaluation import evaluate_all_compositional_checkpoints
from .compositional_training import train_compositional_run
from .config import DEFAULT_CONFIG, ExperimentConfig


def run_compositional_generalization(
    run_id: str,
    run_root: str | Path = "artifacts/compositional_runs",
    *,
    manifest_path: str | Path = "artifacts/compositional_generalization_manifest.json",
    split_path: str | Path = "artifacts/parent_pair_split.json",
    device_name: str = "auto",
    resume: str | Path | None = None,
    stop_after_exposures: int | None = None,
    cfg: ExperimentConfig = DEFAULT_CONFIG,
) -> dict[str, Any]:
    run = find_compositional_run(run_id, cfg)
    endpoint = stop_after_exposures or run.maximum_pair_exposures
    directory = train_compositional_run(
        run,
        run_root,
        manifest_path=manifest_path,
        split_path=split_path,
        device_name=device_name,
        resume=resume,
        stop_after_exposures=endpoint,
        cfg=cfg,
    )
    evaluations = evaluate_all_compositional_checkpoints(
        directory,
        run,
        endpoint,
        split_path=split_path,
        device_name=device_name,
        cfg=cfg,
    )
    return {
        "run_id": run_id,
        "run_directory": str(directory.resolve()),
        "completed_exposure": endpoint,
        "new_evaluations": evaluations,
    }


def compositional_status(
    run_root: str | Path = "artifacts/compositional_runs",
    cfg: ExperimentConfig = DEFAULT_CONFIG,
) -> list[dict[str, Any]]:
    root = Path(run_root)
    output = []
    for run in build_compositional_runs(cfg):
        directory = root / run.run_id
        checkpoints = [
            exposure
            for exposure in run.checkpoints
            if (directory / f"checkpoint_{exposure}.pt").exists()
        ]
        evaluations = []
        if (directory / "evaluations.jsonl").exists():
            evaluations = [
                int(json.loads(line)["exposed_pairs"])
                for line in (directory / "evaluations.jsonl").read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
        output.append(
            {
                "run_id": run.run_id,
                "arm": run.arm,
                "model_initialization_seed": run.model_initialization_seed,
                "completed_exposure": max(checkpoints, default=0),
                "target_exposure": run.maximum_pair_exposures,
                "checkpoints": checkpoints,
                "evaluations": evaluations,
                "status": (
                    "complete"
                    if checkpoints and checkpoints[-1] == run.maximum_pair_exposures
                    and evaluations and evaluations[-1] == run.maximum_pair_exposures
                    else ("in_progress" if checkpoints else "pending")
                ),
            }
        )
    return output
