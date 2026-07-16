from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from torch.nn import functional as F

from .config import DEFAULT_CONFIG, ExperimentConfig
from .data import OnlineEpisodeStream, split_seed
from .durability import atomic_write_json, transactional_append_jsonl
from .environment import environment_snapshot, file_sha256
from .information import (
    conditional_distribution,
    normalized_excess_error,
    predictable_information,
    torch_exact_conditional_kl,
)
from .manifest import Track, rule_from_track, track_from_dict
from .model import make_model, size_spec
from .topology import FiniteFieldRule


@dataclass(frozen=True)
class LearnabilityRun:
    run_id: str
    description: str
    k: int
    d: int
    target_core_parameters: int
    width: int
    actual_core_parameters: int
    total_parameters: int
    rule_instance: int
    rule: dict[str, object]
    model_initialization_seed: int
    data_seed: int
    evaluation_seed: int
    probe_seeds: dict[str, int]
    loss_type: str
    maximum_supervised_tokens: int
    checkpoints: tuple[int, ...]
    evaluation_episodes: int
    probe_checkpoints: tuple[int, ...]
    probe_layers: tuple[int, ...]
    probe_examples: dict[str, int]
    fixed_mlp_hidden: int
    exploratory: bool = True
    diagnostic_only: bool = True
    excluded_from_confirmatory_fit: bool = True

    def as_training_track(self) -> Track:
        return Track(
            track_id=self.run_id,
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


def _audit_run(
    run_id: str,
    description: str,
    k: int,
    d: int,
    target: int,
    rule: FiniteFieldRule,
    loss_type: str,
    checkpoints: tuple[int, ...],
    probe_checkpoints: tuple[int, ...],
    seed_offset: int,
    cfg: ExperimentConfig,
) -> LearnabilityRun:
    size = size_spec(target, cfg.model, cfg.field.p)
    init_seed = 0
    return LearnabilityRun(
        run_id=run_id,
        description=description,
        k=k,
        d=d,
        target_core_parameters=target,
        width=size.width,
        actual_core_parameters=size.actual_core_parameters,
        total_parameters=size.total_parameters,
        rule_instance=rule.instance,
        rule=rule.to_dict(),
        model_initialization_seed=init_seed,
        data_seed=split_seed(51_000 + seed_offset, k, d, rule.instance, init_seed),
        evaluation_seed=split_seed(52_000 + seed_offset, k, d, rule.instance, init_seed),
        probe_seeds={
            "train": split_seed(53_000 + seed_offset, k, d, rule.instance, init_seed),
            "validation": split_seed(54_000 + seed_offset, k, d, rule.instance, init_seed),
            "test": split_seed(55_000 + seed_offset, k, d, rule.instance, init_seed),
        },
        loss_type=loss_type,
        maximum_supervised_tokens=checkpoints[-1],
        checkpoints=checkpoints,
        evaluation_episodes=256,
        probe_checkpoints=probe_checkpoints,
        probe_layers=(0, 3, 6),
        probe_examples={"train": 4096, "validation": 1024, "test": 1024},
        fixed_mlp_hidden=256,
    )


def build_learnability_manifest(
    cfg: ExperimentConfig = DEFAULT_CONFIG,
) -> tuple[LearnabilityRun, ...]:
    short = (500_000, 1_000_000, 2_000_000, 5_000_000, 10_000_000)
    long = (5_000_000, 10_000_000, 20_000_000, 40_000_000)
    positive_rule = FiniteFieldRule.sample(1, 1, 0, cfg.training.rule_seed, cfg.field)
    production_rule = FiniteFieldRule.sample(2, 4, 0, cfg.training.rule_seed, cfg.field)
    runs = (
        _audit_run(
            "learnability_A_single_parent",
            "single-parent positive control",
            1,
            1,
            2_000_000,
            positive_rule,
            "sampled_nll",
            short,
            (500_000, 2_000_000, 10_000_000),
            1,
            cfg,
        ),
        _audit_run(
            "learnability_B_capacity_threshold",
            "10M capacity-threshold calibration on preregistered (2,4) rule-0",
            2,
            4,
            10_000_000,
            production_rule,
            "sampled_nll",
            long,
            (5_000_000, 20_000_000, 40_000_000),
            2,
            cfg,
        ),
        _audit_run(
            "learnability_C_oracle_soft",
            "oracle-soft-label gradient-variance diagnostic",
            2,
            4,
            2_000_000,
            production_rule,
            "oracle_soft",
            short,
            (500_000, 2_000_000, 10_000_000),
            3,
            cfg,
        ),
    )
    if any(not (run.exploratory and run.diagnostic_only and run.excluded_from_confirmatory_fit) for run in runs):
        raise AssertionError("every learnability run must be diagnostic-only and excluded")
    return runs


def learnability_manifest_payload(
    cfg: ExperimentConfig = DEFAULT_CONFIG,
    production_manifest: str | Path = "artifacts/manifest.json",
    pilot_manifest: str | Path = "artifacts/pilot_manifest.json",
) -> dict[str, Any]:
    return {
        "project": "Project Kepler v3.2.2",
        "kind": "learnability_audit",
        "production_manifest_hash": file_sha256(production_manifest),
        "pilot_manifest_hash": file_sha256(pilot_manifest),
        "production_manifest_mutated": False,
        "pilot_manifest_mutated": False,
        "flags": {
            "exploratory": True,
            "diagnostic_only": True,
            "excluded_from_confirmatory_fit": True,
        },
        "runs": [asdict(run) for run in build_learnability_manifest(cfg)],
    }


def write_learnability_manifest(
    output: str | Path = "artifacts/learnability_audit_manifest.json",
    cfg: ExperimentConfig = DEFAULT_CONFIG,
) -> tuple[Path, str]:
    path = atomic_write_json(output, learnability_manifest_payload(cfg))
    return path, file_sha256(path)


def find_learnability_run(run_id: str, cfg: ExperimentConfig = DEFAULT_CONFIG) -> LearnabilityRun:
    for run in build_learnability_manifest(cfg):
        if run.run_id == run_id:
            return run
    raise KeyError(f"unknown learnability run: {run_id}")


def oracle_soft_label_loss(logits, centers, mask, p: int = 257, eta: float = 0.1):
    if logits.shape[:-1] != centers.shape or centers.shape != mask.shape:
        raise ValueError("soft-label inputs are not aligned")
    log_probs = F.log_softmax(logits.float(), dim=-1)
    q = torch.full_like(log_probs, eta / (p - 1))
    q.scatter_(-1, centers.long().unsqueeze(-1), 1.0 - eta)
    cross_entropy = -torch.sum(q * log_probs, dim=-1)
    return cross_entropy[mask].mean()


def no_training_metric_audit(
    checkpoint: str | Path | None = None,
    output: str | Path = "artifacts/no_training_metric_audit.json",
    cfg: ExperimentConfig = DEFAULT_CONFIG,
) -> dict[str, Any]:
    rule = FiniteFieldRule.sample(2, 4, 0, cfg.training.rule_seed, cfg.field)
    stream = OnlineEpisodeStream(rule, seed=60_001)
    batch = stream.batch(32)
    centers = batch.centers[batch.loss_mask]
    uniform_log_probs = np.full((centers.size, cfg.field.p), -math.log(cfg.field.p))
    oracle_q = conditional_distribution(centers, cfg.field.p, cfg.field.eta)
    uniform_r = normalized_excess_error(uniform_log_probs, centers, cfg.field.p, cfg.field.eta)
    oracle_r = normalized_excess_error(np.log(oracle_q), centers, cfg.field.p, cfg.field.eta)

    # Alignment is recomputed directly from h_{t-1}, independent of stored labels.
    alignment_mismatches = 0
    for episode_index in range(batch.inputs.shape[0]):
        tokens = np.concatenate((batch.inputs[episode_index, :1], batch.targets[episode_index]))
        for input_position in np.flatnonzero(batch.loss_mask[episode_index]):
            history = tokens[: input_position + 1]
            if int(rule.center(history)) != int(batch.centers[episode_index, input_position]):
                alignment_mismatches += 1

    scored_positions = np.flatnonzero(batch.loss_mask[0])[:-1]
    true_centers = batch.centers[:, scored_positions].reshape(-1)
    previous_centers = batch.centers[:, scored_positions - 1].reshape(-1)
    next_centers = batch.centers[:, scored_positions + 1].reshape(-1)
    previous_r = normalized_excess_error(
        np.log(conditional_distribution(previous_centers, cfg.field.p, cfg.field.eta)),
        true_centers,
        cfg.field.p,
        cfg.field.eta,
    )
    next_r = normalized_excess_error(
        np.log(conditional_distribution(next_centers, cfg.field.p, cfg.field.eta)),
        true_centers,
        cfg.field.p,
        cfg.field.eta,
    )

    # Pure metric batching invariance, using the exact same contexts.
    parts = []
    for start in range(0, centers.size, 137):
        stop = min(centers.size, start + 137)
        parts.append(
            normalized_excess_error(
                uniform_log_probs[start:stop], centers[start:stop], cfg.field.p, cfg.field.eta
            )
            * (stop - start)
        )
    batched_uniform_r = sum(parts) / centers.size
    result: dict[str, Any] = {
        "uniform_R": uniform_r,
        "oracle_R": oracle_r,
        "uniform_within_tolerance": abs(uniform_r - 1.0) < 1e-12,
        "oracle_within_tolerance": abs(oracle_r) < 1e-12,
        "alignment_mismatch_count": alignment_mismatches,
        "alignment_positions": int(batch.loss_mask.sum()),
        "z_t_from_h_t_minus_1": alignment_mismatches == 0,
        "shifted_previous_R": previous_r,
        "shifted_next_R": next_r,
        "shifted_states_rejected": previous_r > 0.1 and next_r > 0.1,
        "metric_batching_absolute_difference": abs(uniform_r - batched_uniform_r),
        "metric_batching_invariant": abs(uniform_r - batched_uniform_r) < 1e-12,
    }
    if checkpoint is not None:
        from .training import evaluate_checkpoint

        first = evaluate_checkpoint(
            checkpoint,
            episodes=32,
            batch_size=1,
            device_name="cpu",
            evaluation_seed_override=60_002,
            bootstrap_replicates=200,
            output_path=Path(output).with_suffix(".checkpoint_batch1.json"),
        )
        second = evaluate_checkpoint(
            checkpoint,
            episodes=32,
            batch_size=16,
            device_name="cpu",
            evaluation_seed_override=60_002,
            bootstrap_replicates=200,
            output_path=Path(output).with_suffix(".checkpoint_batch16.json"),
        )
        reloaded = evaluate_checkpoint(
            checkpoint,
            episodes=32,
            batch_size=16,
            device_name="cpu",
            evaluation_seed_override=60_002,
            bootstrap_replicates=200,
            output_path=Path(output).with_suffix(".checkpoint_reload.json"),
        )
        result["checkpoint_batching_R_absolute_difference"] = abs(first["R"] - second["R"])
        result["checkpoint_reload_R_absolute_difference"] = abs(second["R"] - reloaded["R"])
        result["checkpoint_batching_and_reload_invariant"] = (
            abs(first["R"] - second["R"]) < 1e-6
            and abs(second["R"] - reloaded["R"]) < 1e-12
        )
    result["passed"] = all(
        result[key]
        for key in (
            "uniform_within_tolerance",
            "oracle_within_tolerance",
            "z_t_from_h_t_minus_1",
            "shifted_states_rejected",
            "metric_batching_invariant",
        )
    ) and result.get("checkpoint_batching_and_reload_invariant", True)
    atomic_write_json(output, result)
    return result


def run_checkpoint_probes(
    checkpoint: str | Path,
    run: LearnabilityRun,
    output: str | Path,
    device_name: str = "auto",
    cfg: ExperimentConfig = DEFAULT_CONFIG,
) -> dict[str, Any]:
    from .probes import (
        FourierStateProbe,
        LinearStateProbe,
        MLPStateProbe,
        extract_probe_layers,
        train_classification_probe,
        train_fourier_probe,
    )
    from .training import resolve_device

    device = resolve_device(device_name)
    state = torch.load(checkpoint, map_location=device, weights_only=False)
    track = track_from_dict(state["track"])
    rule = rule_from_track(track)
    model = make_model(track.target_core_parameters, cfg.model, cfg.field.p).to(device)
    model.load_state_dict(state["model"])
    split_data: dict[str, dict[int, Any]] = {}
    for split in ("train", "validation", "test"):
        split_data[split] = extract_probe_layers(
            model,
            rule,
            run.probe_seeds[split],
            run.probe_examples[split],
            run.probe_layers,
            str(device),
            cfg,
            batch_size=32,
        )
    layer_results: dict[str, Any] = {}
    for layer in run.probe_layers:
        train = split_data["train"][layer]
        validation = split_data["validation"][layer]
        test = split_data["test"][layer]
        width = train.residuals.shape[-1]
        seed = run.probe_seeds["train"] + layer
        torch.manual_seed(seed)
        linear = train_classification_probe(
            LinearStateProbe(width, cfg.field.p),
            train,
            validation,
            test,
            seed=seed,
        )
        torch.manual_seed(seed + 1)
        fourier = train_fourier_probe(
            FourierStateProbe(width, cfg.field.p),
            train,
            validation,
            test,
            seed=seed + 1,
        )
        torch.manual_seed(seed + 2)
        mlp = train_classification_probe(
            MLPStateProbe(width, cfg.field.p, hidden=run.fixed_mlp_hidden),
            train,
            validation,
            test,
            seed=seed + 2,
        )
        layer_results[str(layer)] = {
            "linear": linear,
            "fixed_fourier": fourier,
            "fixed_capacity_mlp": {
                "hidden_width": run.fixed_mlp_hidden,
                **mlp,
            },
        }
    result = {
        "run_id": run.run_id,
        "checkpoint": str(Path(checkpoint).resolve()),
        "exposed_tokens": int(state["supervised_tokens_seen"]),
        "probe_layers": list(run.probe_layers),
        "split_examples": run.probe_examples,
        "split_seeds": run.probe_seeds,
        "layers": layer_results,
    }
    atomic_write_json(output, result)
    return result


def run_learnability_audit(
    run_id: str,
    run_root: str | Path = "artifacts/learnability_runs",
    manifest_path: str | Path = "artifacts/learnability_audit_manifest.json",
    device_name: str = "auto",
    batch_size: int | None = None,
    resume: str | Path | None = None,
    stop_after_tokens: int | None = None,
    cfg: ExperimentConfig = DEFAULT_CONFIG,
) -> dict[str, Any]:
    from .pilot import _throughput_summary
    from .training import evaluate_checkpoint, train_track

    manifest_output, manifest_hash = write_learnability_manifest(manifest_path, cfg)
    run = find_learnability_run(run_id, cfg)
    endpoint = stop_after_tokens or run.maximum_supervised_tokens
    if endpoint not in run.checkpoints:
        raise ValueError("stop_after_tokens must be a preregistered checkpoint")
    environment = environment_snapshot(
        cfg.to_dict(),
        production_manifest="artifacts/manifest.json",
        pilot_manifest="artifacts/pilot_manifest.json",
        seeds={
            "model": run.model_initialization_seed,
            "rule": cfg.training.rule_seed,
            "data": run.data_seed,
            "evaluation": run.evaluation_seed,
            "probe": run.probe_seeds,
        },
        workdir=Path.cwd(),
    )
    provenance = {
        **environment,
        "learnability_manifest_hash": manifest_hash,
        "analysis_flags": {
            "exploratory": True,
            "diagnostic_only": True,
            "excluded_from_confirmatory_fit": True,
        },
        "loss_type": run.loss_type,
    }
    run_dir = train_track(
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

    evaluation_file = run_dir / "evaluations.jsonl"
    evaluated = set()
    if evaluation_file.exists():
        evaluated = {
            int(json.loads(line)["exposed_tokens"])
            for line in evaluation_file.read_text(encoding="utf-8").splitlines()
            if line.strip()
        }
    new_evaluations = []
    for exposure in run.checkpoints:
        if exposure > endpoint or exposure in evaluated:
            continue
        checkpoint = run_dir / f"checkpoint_{exposure}.pt"
        result = evaluate_checkpoint(
            checkpoint,
            episodes=run.evaluation_episodes,
            batch_size=batch_size or cfg.training.batch_size,
            device_name=device_name,
            cfg=cfg,
            evaluation_seed_override=run.evaluation_seed,
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
        summary["full_episode_metrics_file"] = str(checkpoint.with_suffix(".eval.json"))
        transactional_append_jsonl(evaluation_file, summary)
        new_evaluations.append(summary)

    new_probes = []
    for exposure in run.probe_checkpoints:
        if exposure > endpoint:
            continue
        output = run_dir / f"checkpoint_{exposure}.probes.json"
        if output.exists():
            continue
        new_probes.append(
            run_checkpoint_probes(
                run_dir / f"checkpoint_{exposure}.pt",
                run,
                output,
                device_name,
                cfg,
            )
        )
    engineering = _throughput_summary(run_dir / "train.jsonl")
    atomic_write_json(run_dir / "engineering_summary.json", engineering)
    return {
        "run": asdict(run),
        "run_directory": str(run_dir.resolve()),
        "learnability_manifest_hash": manifest_hash,
        "new_evaluations": new_evaluations,
        "new_probe_results": [
            {"exposed_tokens": item["exposed_tokens"], "probe_layers": item["probe_layers"]}
            for item in new_probes
        ],
        "throughput_and_memory": engineering,
    }


def learnability_status(
    run_root: str | Path = "artifacts/learnability_runs",
    cfg: ExperimentConfig = DEFAULT_CONFIG,
) -> list[dict[str, Any]]:
    root = Path(run_root)
    rows = []
    for run in build_learnability_manifest(cfg):
        directory = root / run.run_id
        checkpoints = [
            exposure
            for exposure in run.checkpoints
            if (directory / f"checkpoint_{exposure}.pt").exists()
        ]
        evaluations = []
        if (directory / "evaluations.jsonl").exists():
            evaluations = [
                int(json.loads(line)["exposed_tokens"])
                for line in (directory / "evaluations.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
                if line.strip()
            ]
        probes = [
            exposure
            for exposure in run.probe_checkpoints
            if (directory / f"checkpoint_{exposure}.probes.json").exists()
        ]
        rows.append(
            {
                "run_id": run.run_id,
                "loss_type": run.loss_type,
                "completed_tokens": max(checkpoints, default=0),
                "target_tokens": run.maximum_supervised_tokens,
                "checkpoints": checkpoints,
                "evaluations": sorted(evaluations),
                "probes": probes,
                "status": "complete"
                if checkpoints and checkpoints[-1] == run.maximum_supervised_tokens
                else ("in_progress" if checkpoints else "pending"),
            }
        )
    return rows
