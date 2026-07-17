from __future__ import annotations

import argparse
import json
from pathlib import Path

from common import (
    ARTIFACTS,
    FREEZE_RECORD,
    ROOT,
    git,
    load_json,
    sha256,
)


FROZEN_TAG = "kepler-v3.6-confirm1"
ALLOWED_NEW_PREFIX = "paper/"
ALLOWED_NEW_ARTIFACT = "artifacts/v3_6_confirmatory_freeze_record.json"


def changed_paths() -> set[str]:
    changed = set(
        line
        for line in git("diff", "--name-only", FROZEN_TAG).splitlines()
        if line
    )
    for line in git("status", "--porcelain").splitlines():
        if not line:
            continue
        parts = line.split(maxsplit=1)
        path = parts[-1]
        if " -> " in path:
            path = path.split(" -> ", 1)[1]
        changed.add(path)
    return changed


def validate_sidecars(root: Path) -> tuple[int, bool]:
    checkpoints = sorted(root.glob("*/checkpoint_*.pt"))
    valid = True
    for checkpoint in checkpoints:
        sidecar = Path(str(checkpoint) + ".sha256")
        valid = valid and sidecar.exists()
        if not valid:
            break
        expected = sidecar.read_text(encoding="utf-8").split()[0]
        valid = valid and sha256(checkpoint) == expected
        if not valid:
            break
    return len(checkpoints), valid


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-checkpoint-bytes", action="store_true")
    args = parser.parse_args()
    freeze = load_json(FREEZE_RECORD)
    for relative, expected in freeze[
        "tracked_scientific_artifact_sha256_at_freeze"
    ].items():
        actual = sha256(ROOT / relative)
        if actual != expected:
            raise AssertionError(f"scientific artifact changed: {relative}")
    unexpected = sorted(
        path
        for path in changed_paths()
        if path != ALLOWED_NEW_ARTIFACT and not path.startswith(ALLOWED_NEW_PREFIX)
    )
    if unexpected:
        raise AssertionError(f"non-paper paths changed since freeze: {unexpected}")
    checkpoint_status = {}
    if not args.skip_checkpoint_bytes:
        for label, directory in (
            ("v3.5", ARTIFACTS / "cue_dose_runs"),
            ("v3.6", ARTIFACTS / "confirmatory_runs"),
        ):
            count, valid = validate_sidecars(directory)
            expected_key = (
                "v3.5_exploratory" if label == "v3.5" else "v3.6_confirmatory"
            )
            expected = freeze["artifact_counts"][expected_key]["checkpoints"]
            if count != expected or not valid:
                raise AssertionError((label, count, expected, valid))
            checkpoint_status[label] = {
                "checkpoint_count": count,
                "all_sidecars_valid": valid,
            }
    print(
        json.dumps(
            {
                "frozen_tag": FROZEN_TAG,
                "tracked_scientific_artifacts_unchanged": True,
                "allowed_packaging_paths_only": True,
                "checkpoint_status": checkpoint_status,
                "optimizer_steps": 0,
                "passed": True,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
