from __future__ import annotations

import json
from pathlib import Path

from common import (
    ARTIFACTS,
    FREEZE_RECORD,
    ROOT,
    V35_RESULTS,
    V36_RESULTS,
    atomic_write_json,
    git,
    load_json,
    require_frozen_source_hashes,
    sha256,
)


FROZEN_TAG = "kepler-v3.6-confirm1"
FROZEN_COMMIT = "f9fc22d85cd52863087afbb3f924466a63b30ba2"

STATISTICAL_DEFINITION_FILES = (
    "src/kepler/cue_distribution.py",
    "src/kepler/cue_dose_analysis.py",
    "src/kepler/cue_dose_runner.py",
    "src/kepler/confirmatory_distribution.py",
    "src/kepler/confirmatory_analysis.py",
    "src/kepler/confirmatory_runner.py",
)


def tags() -> list[dict[str, str]]:
    output = []
    for name in git("tag", "--list", "--sort=refname").splitlines():
        output.append(
            {
                "name": name,
                "object": git("rev-parse", name),
                "commit": git("rev-list", "-n", "1", name),
            }
        )
    return output


def manifest_hashes() -> dict[str, str]:
    return {
        str(path.relative_to(ROOT)): sha256(path)
        for path in sorted(ARTIFACTS.glob("*manifest*.json"))
    }


def tracked_scientific_hashes() -> dict[str, str]:
    selected: set[str] = set(STATISTICAL_DEFINITION_FILES)
    selected.update(
        str(path.relative_to(ROOT))
        for pattern in ("*manifest*.json", "*results*.json", "*distribution_audit.json")
        for path in ARTIFACTS.glob(pattern)
    )
    return {
        relative: sha256(ROOT / relative)
        for relative in sorted(selected)
    }


def validate_sidecars(root: Path) -> tuple[int, bool]:
    checkpoint_files = sorted(root.glob("*/checkpoint_*.pt"))
    for checkpoint in checkpoint_files:
        sidecar = Path(str(checkpoint) + ".sha256")
        if not sidecar.exists():
            return len(checkpoint_files), False
        expected = sidecar.read_text(encoding="utf-8").split()[0]
        if sha256(checkpoint) != expected:
            return len(checkpoint_files), False
    return len(checkpoint_files), True


def run_counts(root: Path) -> dict[str, int | bool]:
    checkpoint_count, sidecars_valid = validate_sidecars(root)
    run_directories = sorted(path for path in root.iterdir() if path.is_dir())
    maximum_exposure = 0
    for run in run_directories:
        evaluation_log = run / "evaluations.jsonl"
        if not evaluation_log.exists():
            continue
        for line in evaluation_log.read_text(encoding="utf-8").splitlines():
            if line.strip():
                maximum_exposure = max(
                    maximum_exposure,
                    int(json.loads(line)["exposed_pairs"]),
                )
    return {
        "run_directories": len(run_directories),
        "checkpoints": checkpoint_count,
        "checkpoint_sidecars": len(list(root.glob("*/*.pt.sha256"))),
        "all_checkpoint_sidecars_valid": sidecars_valid,
        "evaluations": len(list(root.glob("*/*.cue.eval.json"))),
        "probe_reports": len(list(root.glob("*/*.cue.probes.json"))),
        "switch_invariance_records": len(list(root.glob("*/switch_invariance.json"))),
        "gradient_audit_records": len(list(root.glob("*/*.gradient.json"))),
        "maximum_exposure": maximum_exposure,
    }


def main() -> None:
    require_frozen_source_hashes()
    if git("rev-parse", FROZEN_TAG) != FROZEN_COMMIT:
        raise AssertionError("v3.6 confirmatory tag does not resolve to frozen commit")
    v35 = load_json(V35_RESULTS)
    v36 = load_json(V36_RESULTS)
    prior_freeze = load_json(ARTIFACTS / "v3_6_freeze_record.json")
    primary = v36["primary_confirmatory_test"]["ordered_onset_test"]
    record = {
        "project": "Project Kepler v3.6",
        "kind": "confirmatory_result_and_paper_packaging_freeze",
        "recorded_date": "2026-07-18",
        "final_v3_6_commit": FROZEN_COMMIT,
        "final_v3_6_tag": FROZEN_TAG,
        "paper_branch": "kepler-paper-v1",
        "all_git_tags": tags(),
        "manifest_sha256": manifest_hashes(),
        "result_sha256": {
            "v3.5_exploratory": sha256(V35_RESULTS),
            "v3.6_confirmatory": sha256(V36_RESULTS),
        },
        "environment_lock": prior_freeze["environment_lock"],
        "primary_test": {
            "definition": {
                "name": primary["test"],
                "alternative": primary["alternative"],
                "conditions_in_registered_order": primary[
                    "conditions_in_registered_order"
                ],
                "blocks": primary["seed_blocks"],
                "within_block_assignments": 6,
                "exact_assignment_count": primary["exact_permutation_count"],
                "tie_handling": primary["tie_handling"],
                "confirmatory_alpha": v36["primary_confirmatory_test"][
                    "confirmatory_alpha"
                ],
                "v3_5_combined_in_primary": False,
            },
            "result": {
                "observed_statistic": primary["observed_statistic"],
                "observed_statistic_doubled_integer": primary[
                    "observed_statistic_doubled_integer"
                ],
                "upper_tail_count": primary["upper_tail_count"],
                "exact_one_sided_p": primary["exact_one_sided_p"],
                "statistically_confirmatory": v36["primary_confirmatory_test"][
                    "statistically_confirmatory"
                ],
            },
        },
        "artifact_counts": {
            "v3.5_exploratory": run_counts(ARTIFACTS / "cue_dose_runs"),
            "v3.6_confirmatory": run_counts(ARTIFACTS / "confirmatory_runs"),
        },
        "integrity": {
            "v3.5_result_integrity": v35["integrity"],
            "v3.6_result_integrity": {
                key: value
                for key, value in v36["integrity"].items()
                if key != "switch_checks"
            },
            "frozen_source_hashes_verified": True,
            "previous_scientific_artifacts_modified": False,
            "optimizer_steps_run_during_packaging": 0,
        },
        "tracked_scientific_artifact_sha256_at_freeze": tracked_scientific_hashes(),
    }
    atomic_write_json(FREEZE_RECORD, record)
    print(FREEZE_RECORD.relative_to(ROOT))


if __name__ == "__main__":
    main()
