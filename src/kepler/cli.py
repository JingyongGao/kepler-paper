from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Any


def _json(value: Any) -> None:
    print(json.dumps(value, indent=2, sort_keys=True))


def _manifest(args: argparse.Namespace) -> int:
    from .manifest import manifest_payload, write_manifest

    if args.output:
        output = write_manifest(args.output)
        _json({"tracks": 80, "output": str(output.resolve())})
    else:
        _json(manifest_payload())
    return 0


def _sizes(_: argparse.Namespace) -> int:
    from .config import DEFAULT_CONFIG
    from .model import size_spec

    _json(
        [
            size_spec(target, DEFAULT_CONFIG.model, DEFAULT_CONFIG.field.p).__dict__
            for target in DEFAULT_CONFIG.model.target_core_parameters
        ]
    )
    return 0


def _phase0(args: argparse.Namespace) -> int:
    from .gates import phase0_gate

    result = phase0_gate(samples=args.samples, delta=args.delta)
    _json(result)
    return 0 if result["passed"] else 2


def _phase1(_: argparse.Namespace) -> int:
    from .gates import phase1_gate

    result = phase1_gate()
    _json(result)
    return 0 if result["passed"] else 2


def _train(args: argparse.Namespace) -> int:
    from .training import train_track

    output = train_track(
        args.track_id,
        output_root=args.output_root,
        max_tokens=args.max_tokens,
        batch_size=args.batch_size,
        device_name=args.device,
        amp=args.amp,
        resume=args.resume,
    )
    _json({"run_directory": str(output.resolve())})
    return 0


def _evaluate(args: argparse.Namespace) -> int:
    from .training import evaluate_checkpoint

    _json(
        evaluate_checkpoint(
            args.checkpoint,
            contexts=args.contexts,
            episodes=args.episodes,
            batch_size=args.batch_size,
            device_name=args.device,
        )
    )
    return 0


def _pilot_manifest(args: argparse.Namespace) -> int:
    from .environment import file_sha256
    from .pilot import pilot_manifest_payload, write_pilot_manifest

    if args.output:
        output, checksum = write_pilot_manifest(args.output, production_manifest=args.production_manifest)
        _json({"tracks": 15, "output": str(output.resolve()), "sha256": checksum})
    else:
        _json(pilot_manifest_payload(production_manifest=args.production_manifest))
    return 0


def _pilot_status(args: argparse.Namespace) -> int:
    from .pilot import pilot_status

    _json(pilot_status(args.run_root))
    return 0


def _pilot_run(args: argparse.Namespace) -> int:
    from .pilot import run_pilot_track

    _json(
        run_pilot_track(
            args.track_id,
            run_root=args.run_root,
            manifest_path=args.manifest,
            production_manifest=args.production_manifest,
            device=args.device,
            batch_size=args.batch_size,
            resume=args.resume,
            stop_after_tokens=args.stop_after_tokens,
        )
    )
    return 0


def _environment_audit(args: argparse.Namespace) -> int:
    from .audit import deterministic_algorithm_audit

    result = deterministic_algorithm_audit(args.output, args.iterations)
    _json(result)
    return 0 if result["deterministic"]["supported"] else 2


def _resume_audit(args: argparse.Namespace) -> int:
    from .audit import forced_resume_audit

    result = forced_resume_audit(args.output_root, args.device)
    _json(result)
    return 0 if result["passed"] else 2


def _attach_task(args: argparse.Namespace) -> int:
    from .phase4 import attach_task_conditioner_checkpoint

    _json(
        attach_task_conditioner_checkpoint(
            args.base_checkpoint,
            args.output,
            device_name=args.device,
            tolerance=args.tolerance,
        )
    )
    return 0


def _overfit_audit(args: argparse.Namespace) -> int:
    from .audit import fixed_batch_overfit_audit

    result = fixed_batch_overfit_audit(args.output, args.steps, args.device)
    _json(result)
    return 0 if result["loss_decreased"] and result["finite"] else 2


def _learnability_manifest(args: argparse.Namespace) -> int:
    from .learnability import write_learnability_manifest

    output, checksum = write_learnability_manifest(args.output)
    _json({"runs": 3, "output": str(output.resolve()), "sha256": checksum})
    return 0


def _metric_audit(args: argparse.Namespace) -> int:
    from .learnability import no_training_metric_audit

    result = no_training_metric_audit(args.checkpoint, args.output)
    _json(result)
    return 0 if result["passed"] else 2


def _learnability_run(args: argparse.Namespace) -> int:
    from .learnability import run_learnability_audit

    _json(
        run_learnability_audit(
            args.run_id,
            run_root=args.run_root,
            manifest_path=args.manifest,
            device_name=args.device,
            batch_size=args.batch_size,
            resume=args.resume,
            stop_after_tokens=args.stop_after_tokens,
        )
    )
    return 0


def _learnability_status(args: argparse.Namespace) -> int:
    from .learnability import learnability_status

    _json(learnability_status(args.run_root))
    return 0


def _mechanism_manifest(args: argparse.Namespace) -> int:
    from .mechanism import write_mechanism_manifest

    output, checksum = write_mechanism_manifest(args.output)
    _json({"runs": 4, "output": str(output.resolve()), "sha256": checksum})
    return 0


def _gradient_signal_audit(args: argparse.Namespace) -> int:
    from .gradient_audit import run_initialization_gradient_audit

    result = run_initialization_gradient_audit(
        args.output,
        device_name=args.device,
        initializations=8,
        batches_per_initialization=32,
        batch_size=args.batch_size,
    )
    _json({"output": str(Path(args.output).resolve()), "tasks": list(result["tasks"])})
    return 0


def _mechanism_run(args: argparse.Namespace) -> int:
    from .mechanism_runner import run_mechanism_isolation

    _json(
        run_mechanism_isolation(
            args.run_id,
            run_root=args.run_root,
            manifest_path=args.manifest,
            device_name=args.device,
            batch_size=args.batch_size,
            resume=args.resume,
            stop_after_exposure=args.stop_after_exposure,
        )
    )
    return 0


def _mechanism_status(args: argparse.Namespace) -> int:
    from .mechanism_runner import mechanism_status

    _json(mechanism_status(args.run_root))
    return 0


def _checkpoint_gradient_audit(args: argparse.Namespace) -> int:
    from .checkpoint_gradients import run_checkpoint_gradient_dynamics_audit

    result = run_checkpoint_gradient_dynamics_audit(
        args.output, device_name=args.device
    )
    _json({"output": str(Path(args.output).resolve()), "runs": list(result["runs"])})
    return 0


def _parent_pair_split(args: argparse.Namespace) -> int:
    from .parent_pair_split import write_parent_pair_split

    output, checksum = write_parent_pair_split(args.output)
    _json({"output": str(output.resolve()), "sha256": checksum})
    return 0


def _composition_manifest(args: argparse.Namespace) -> int:
    from .compositional import write_compositional_manifest

    output, checksum = write_compositional_manifest(args.output, args.split)
    _json({"runs": 6, "output": str(output.resolve()), "sha256": checksum})
    return 0


def _composition_run(args: argparse.Namespace) -> int:
    from .compositional_runner import run_compositional_generalization

    _json(
        run_compositional_generalization(
            args.run_id,
            run_root=args.run_root,
            manifest_path=args.manifest,
            split_path=args.split,
            device_name=args.device,
            resume=args.resume,
            stop_after_exposures=args.stop_after_exposures,
        )
    )
    return 0


def _composition_status(args: argparse.Namespace) -> int:
    from .compositional_runner import compositional_status

    _json(compositional_status(args.run_root))
    return 0


def _support_information_audit(args: argparse.Namespace) -> int:
    from .support_information import write_support_information_audit

    output, checksum = write_support_information_audit(args.output, args.split)
    _json({"output": str(output.resolve()), "sha256": checksum})
    return 0


def _support_gradient_audit(args: argparse.Namespace) -> int:
    from .support_gradient_audit import run_support_gradient_dynamics_audit

    result = run_support_gradient_dynamics_audit(
        args.output,
        run_root=args.run_root,
        split_path=args.split,
        device_name=args.device,
    )
    _json({"output": str(Path(args.output).resolve()), "seeds": list(result["seeds"])})
    return 0


def _fit_scaling(args: argparse.Namespace) -> int:
    from .scaling import compare_models_leave_one_trajectory_out, prepare_scaling_data

    with Path(args.csv).open(newline="", encoding="utf-8") as handle:
        records = list(csv.DictReader(handle))
    data = prepare_scaling_data(records)
    result = compare_models_leave_one_trajectory_out(data)
    if args.output:
        Path(args.output).write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    _json(result)
    return 0


def _adaptation_protocol(_: argparse.Namespace) -> int:
    from .fisher import adaptation_protocol_payload

    _json(adaptation_protocol_payload())
    return 0


def _probe(args: argparse.Namespace) -> int:
    import torch

    from .config import DEFAULT_CONFIG
    from .data import split_seed
    from .manifest import find_track, rule_from_track
    from .model import make_model
    from .probes import (
        FourierStateProbe,
        LinearStateProbe,
        MLPStateProbe,
        extract_probe_dataset,
        train_classification_probe,
        train_fourier_probe,
    )
    from .training import resolve_device

    device = resolve_device(args.device)
    state = torch.load(args.checkpoint, map_location=device, weights_only=False)
    track = find_track(state["track"]["track_id"])
    rule = rule_from_track(track)
    model = make_model(track.target_core_parameters, DEFAULT_CONFIG.model).to(device)
    model.load_state_dict(state["model"])
    split_sizes = (args.train_examples, args.validation_examples, args.test_examples)
    datasets = []
    for offset, examples in enumerate(split_sizes):
        seed = split_seed(
            DEFAULT_CONFIG.training.probe_stream_seed + offset,
            track.k,
            track.d,
            track.rule_instance,
            track.training_seed,
        )
        datasets.append(
            extract_probe_dataset(
                model,
                rule,
                seed,
                examples,
                args.layer,
                str(device),
                DEFAULT_CONFIG,
                args.extraction_batch_size,
            )
        )
    train, validation, test = datasets
    width = train.residuals.shape[-1]
    linear = train_classification_probe(
        LinearStateProbe(width), train, validation, test, seed=args.seed
    )
    fourier = train_fourier_probe(
        FourierStateProbe(width), train, validation, test, seed=args.seed
    )
    mlp = train_classification_probe(
        MLPStateProbe(width), train, validation, test, seed=args.seed
    )
    result = {
        "track_id": track.track_id,
        "checkpoint_exposure": state["exposed_tokens"],
        "layer": args.layer,
        "split_sizes": {
            "train": args.train_examples,
            "validation": args.validation_examples,
            "test": args.test_examples,
        },
        "linear": linear,
        "fourier": fourier,
        "mlp": mlp,
    }
    output = Path(args.output) if args.output else Path(args.checkpoint).with_suffix(
        f".probe_layer{args.layer}.json"
    )
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _json(result)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="kepler", description="Project Kepler v3.2.2")
    sub = parser.add_subparsers(dest="command", required=True)

    manifest = sub.add_parser("manifest", help="expand the preregistered 80 tracks")
    manifest.add_argument("--output")
    manifest.set_defaults(func=_manifest)

    sizes = sub.add_parser("model-sizes", help="show selected widths and actual counts")
    sizes.set_defaults(func=_sizes)

    phase0 = sub.add_parser("phase0", help="run statistical generator/alignment gates")
    phase0.add_argument("--samples", type=int)
    phase0.add_argument("--delta", type=float)
    phase0.set_defaults(func=_phase0)

    phase1 = sub.add_parser("phase1", help="run FP32 CPU/MPS parity gate")
    phase1.set_defaults(func=_phase1)

    train = sub.add_parser("train", help="train one manifest trajectory")
    train.add_argument("--track-id", required=True)
    train.add_argument("--output-root", default="runs")
    train.add_argument("--max-tokens", type=int)
    train.add_argument("--batch-size", type=int)
    train.add_argument("--device", default="auto")
    train.add_argument("--amp", action="store_true")
    train.add_argument("--resume")
    train.set_defaults(func=_train)

    evaluate = sub.add_parser("evaluate", help="evaluate exact conditional KL and R")
    evaluate.add_argument("checkpoint")
    evaluate.add_argument("--contexts", type=int, default=16_384)
    evaluate.add_argument("--episodes", type=int)
    evaluate.add_argument("--batch-size", type=int, default=32)
    evaluate.add_argument("--device", default="auto")
    evaluate.set_defaults(func=_evaluate)

    pilot_manifest = sub.add_parser(
        "pilot-manifest", help="write the separate 15-trajectory Phase-4 pilot manifest"
    )
    pilot_manifest.add_argument("--output", default="artifacts/pilot_manifest.json")
    pilot_manifest.add_argument("--production-manifest", default="artifacts/manifest.json")
    pilot_manifest.set_defaults(func=_pilot_manifest)

    pilot_status_parser = sub.add_parser("pilot-status", help="report every pilot trajectory")
    pilot_status_parser.add_argument("--run-root", default="runs/pilot")
    pilot_status_parser.set_defaults(func=_pilot_status)

    pilot_run = sub.add_parser("pilot-run", help="run one preregistered micro-pilot trajectory")
    pilot_run.add_argument("--track-id", required=True)
    pilot_run.add_argument("--run-root", default="runs/pilot")
    pilot_run.add_argument("--manifest", default="artifacts/pilot_manifest.json")
    pilot_run.add_argument("--production-manifest", default="artifacts/manifest.json")
    pilot_run.add_argument("--device", default="auto")
    pilot_run.add_argument("--batch-size", type=int)
    pilot_run.add_argument("--resume")
    pilot_run.add_argument(
        "--stop-after-tokens",
        type=int,
        help="durably stop at a preregistered checkpoint for later resume",
    )
    pilot_run.set_defaults(func=_pilot_run)

    environment_audit = sub.add_parser(
        "environment-audit", help="audit strict deterministic algorithms on local MPS"
    )
    environment_audit.add_argument("--output", default="artifacts/environment_audit.json")
    environment_audit.add_argument("--iterations", type=int, default=8)
    environment_audit.set_defaults(func=_environment_audit)

    resume_audit = sub.add_parser(
        "resume-audit", help="force an interruption and verify durable continuation"
    )
    resume_audit.add_argument("--output-root", default="artifacts/phase4_resume_audit")
    resume_audit.add_argument("--device", default="auto")
    resume_audit.set_defaults(func=_resume_audit)

    attach_task = sub.add_parser(
        "attach-task-conditioner", help="migrate a base checkpoint to two zero task rows"
    )
    attach_task.add_argument("base_checkpoint")
    attach_task.add_argument("output")
    attach_task.add_argument("--device", default="auto")
    attach_task.add_argument("--tolerance", type=float, default=1e-6)
    attach_task.set_defaults(func=_attach_task)

    overfit_audit = sub.add_parser(
        "overfit-audit", help="isolate optimizer-graph health on one repeated batch"
    )
    overfit_audit.add_argument("--output", default="artifacts/fixed_batch_overfit_audit.json")
    overfit_audit.add_argument("--steps", type=int, default=500)
    overfit_audit.add_argument("--device", default="auto")
    overfit_audit.set_defaults(func=_overfit_audit)

    learnability_manifest = sub.add_parser(
        "learnability-manifest", help="write the three-run diagnostic audit manifest"
    )
    learnability_manifest.add_argument(
        "--output", default="artifacts/learnability_audit_manifest.json"
    )
    learnability_manifest.set_defaults(func=_learnability_manifest)

    metric_audit = sub.add_parser(
        "metric-audit", help="verify uniform/oracle/indexing/batching controls"
    )
    metric_audit.add_argument("--checkpoint")
    metric_audit.add_argument("--output", default="artifacts/no_training_metric_audit.json")
    metric_audit.set_defaults(func=_metric_audit)

    learnability_run = sub.add_parser(
        "learnability-run", help="run one diagnostic-only learnability trajectory"
    )
    learnability_run.add_argument("--run-id", required=True)
    learnability_run.add_argument("--run-root", default="artifacts/learnability_runs")
    learnability_run.add_argument(
        "--manifest", default="artifacts/learnability_audit_manifest.json"
    )
    learnability_run.add_argument("--device", default="auto")
    learnability_run.add_argument("--batch-size", type=int)
    learnability_run.add_argument("--resume")
    learnability_run.add_argument("--stop-after-tokens", type=int)
    learnability_run.set_defaults(func=_learnability_run)

    learnability_status_parser = sub.add_parser(
        "learnability-status", help="report Runs A, B, and C"
    )
    learnability_status_parser.add_argument(
        "--run-root", default="artifacts/learnability_runs"
    )
    learnability_status_parser.set_defaults(func=_learnability_status)

    mechanism_manifest = sub.add_parser(
        "mechanism-manifest", help="write the isolated M1-M4 diagnostic manifest"
    )
    mechanism_manifest.add_argument(
        "--output", default="artifacts/mechanism_isolation_manifest.json"
    )
    mechanism_manifest.set_defaults(func=_mechanism_manifest)

    gradient_signal = sub.add_parser(
        "gradient-signal-audit", help="run the registered no-training 8x32 gradient audit"
    )
    gradient_signal.add_argument(
        "--output", default="artifacts/initialization_gradient_signal_audit.json"
    )
    gradient_signal.add_argument("--device", default="auto")
    gradient_signal.add_argument("--batch-size", type=int, default=8)
    gradient_signal.set_defaults(func=_gradient_signal_audit)

    mechanism_run = sub.add_parser(
        "mechanism-run", help="run one isolated diagnostic M1-M4"
    )
    mechanism_run.add_argument("--run-id", required=True)
    mechanism_run.add_argument("--run-root", default="artifacts/mechanism_runs")
    mechanism_run.add_argument(
        "--manifest", default="artifacts/mechanism_isolation_manifest.json"
    )
    mechanism_run.add_argument("--device", default="auto")
    mechanism_run.add_argument("--batch-size", type=int)
    mechanism_run.add_argument("--resume")
    mechanism_run.add_argument("--stop-after-exposure", type=int)
    mechanism_run.set_defaults(func=_mechanism_run)

    mechanism_status_parser = sub.add_parser(
        "mechanism-status", help="report M1-M4 completion"
    )
    mechanism_status_parser.add_argument(
        "--run-root", default="artifacts/mechanism_runs"
    )
    mechanism_status_parser.set_defaults(func=_mechanism_status)

    checkpoint_gradient = sub.add_parser(
        "checkpoint-gradient-audit",
        help="audit exact-soft gradient dynamics at every M1-M4 checkpoint",
    )
    checkpoint_gradient.add_argument(
        "--output", default="artifacts/checkpoint_gradient_dynamics_audit.json"
    )
    checkpoint_gradient.add_argument("--device", default="auto")
    checkpoint_gradient.set_defaults(func=_checkpoint_gradient_audit)

    parent_split = sub.add_parser(
        "parent-pair-split", help="write the frozen deterministic 80/20 pair split"
    )
    parent_split.add_argument("--output", default="artifacts/parent_pair_split.json")
    parent_split.set_defaults(func=_parent_pair_split)

    composition_manifest = sub.add_parser(
        "composition-manifest", help="write the six-run compositional gate manifest"
    )
    composition_manifest.add_argument(
        "--output", default="artifacts/compositional_generalization_manifest.json"
    )
    composition_manifest.add_argument("--split", default="artifacts/parent_pair_split.json")
    composition_manifest.set_defaults(func=_composition_manifest)

    composition_run = sub.add_parser(
        "composition-run", help="run one paired G1/G2 compositional trajectory"
    )
    composition_run.add_argument("--run-id", required=True)
    composition_run.add_argument("--run-root", default="artifacts/compositional_runs")
    composition_run.add_argument(
        "--manifest", default="artifacts/compositional_generalization_manifest.json"
    )
    composition_run.add_argument("--split", default="artifacts/parent_pair_split.json")
    composition_run.add_argument("--device", default="auto")
    composition_run.add_argument("--resume")
    composition_run.add_argument("--stop-after-exposures", type=int)
    composition_run.set_defaults(func=_composition_run)

    composition_status = sub.add_parser(
        "composition-status", help="report all six compositional trajectories"
    )
    composition_status.add_argument("--run-root", default="artifacts/compositional_runs")
    composition_status.set_defaults(func=_composition_status)

    support_information = sub.add_parser(
        "support-information-audit",
        help="enumerate exact singleton information on the frozen pair supports",
    )
    support_information.add_argument(
        "--output", default="artifacts/support_information_audit.json"
    )
    support_information.add_argument(
        "--split", default="artifacts/parent_pair_split.json"
    )
    support_information.set_defaults(func=_support_information_audit)

    support_gradient = sub.add_parser(
        "support-gradient-audit",
        help="compare paired G1/G2 gradients on support-specific and common batches",
    )
    support_gradient.add_argument(
        "--output", default="artifacts/support_gradient_dynamics_audit.json"
    )
    support_gradient.add_argument(
        "--run-root", default="artifacts/compositional_runs"
    )
    support_gradient.add_argument(
        "--split", default="artifacts/parent_pair_split.json"
    )
    support_gradient.add_argument("--device", default="auto")
    support_gradient.set_defaults(func=_support_gradient_audit)

    probe = sub.add_parser("probe", help="run the disjoint predictive-state probe array")
    probe.add_argument("checkpoint")
    probe.add_argument("--layer", type=int, required=True)
    probe.add_argument("--train-examples", type=int, default=32_768)
    probe.add_argument("--validation-examples", type=int, default=8_192)
    probe.add_argument("--test-examples", type=int, default=8_192)
    probe.add_argument("--extraction-batch-size", type=int, default=32)
    probe.add_argument("--seed", type=int, default=0)
    probe.add_argument("--device", default="auto")
    probe.add_argument("--output")
    probe.set_defaults(func=_probe)

    scaling = sub.add_parser("fit-scaling", help="compare M0/M1/M2 by trajectory holdout")
    scaling.add_argument("csv")
    scaling.add_argument("--output")
    scaling.set_defaults(func=_fit_scaling)

    protocol = sub.add_parser("adaptation-protocol", help="emit the five-arm local-path protocol")
    protocol.set_defaults(func=_adaptation_protocol)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
