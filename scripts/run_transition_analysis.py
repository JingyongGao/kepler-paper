from __future__ import annotations

import argparse
import json
from pathlib import Path

from kepler.transition_dynamics import (
    build_early_cue_removal_analysis,
    build_representation_analysis,
    build_spectral_analysis,
    create_freeze_record,
    finalize_analysis,
)


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description="Project Kepler v1.3 transition-dynamics audit")
    parser.add_argument("command", choices=("freeze", "analyze", "spectral", "finalize"))
    parser.add_argument("--lanczos-iterations", type=int, default=24)
    parser.add_argument("--trace-probes", type=int, default=8)
    args = parser.parse_args()

    if args.command == "freeze":
        result = create_freeze_record(ROOT)
        print(json.dumps({"status": "frozen", "file_count": result["frozen_scope"]["file_count"]}, sort_keys=True))
    elif args.command == "analyze":
        representation = build_representation_analysis(ROOT)
        removal = build_early_cue_removal_analysis(ROOT)
        print(json.dumps({"representation_trajectories": len(representation["trajectories"]), "cue_removal_runs": len(removal["runs"])}, sort_keys=True))
    elif args.command == "spectral":
        spectral = build_spectral_analysis(ROOT, lanczos_iterations=args.lanczos_iterations, trace_probes=args.trace_probes)
        print(json.dumps({"spectral_checkpoints": len(spectral["checkpoints"])}, sort_keys=True))
    else:
        test_path = ROOT / "hashes/test_results.json"
        if not test_path.exists():
            raise FileNotFoundError("hashes/test_results.json must be recorded after both test runs")
        finalize_analysis(ROOT, json.loads(test_path.read_text(encoding="utf-8")))
        print(json.dumps({"status": "finalized"}, sort_keys=True))


if __name__ == "__main__":
    main()
