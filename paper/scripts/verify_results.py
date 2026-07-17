from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

from common import (
    ROOT,
    V35_AUDIT,
    V35_RESULTS,
    V36_AUDIT,
    V36_RESULTS,
    load_json,
    require_frozen_source_hashes,
)


sys.path.insert(0, str(ROOT / "src"))

from kepler.confirmatory_distribution import build_confirmatory_distribution_audit
from kepler.cue_distribution import build_cue_distribution_audit
from kepler.cue_dose_analysis import classify_transition, exact_blocked_page_test


def summary_records(cell: dict) -> list[dict]:
    records = []
    for item in cell["checkpoint_metrics"]:
        probes = item["probe_metrics"]
        layer6 = (
            None
            if probes is None
            else probes["6"]["linear"]["accuracy"]
        )
        records.append(
            {
                "exposed_pairs": item["exposed_pairs"],
                "phase": item["phase"],
                "R_full": item["R"],
                "center_accuracy_full": item["center_state_top1_accuracy"],
                "layer_6_linear_accuracy": layer6,
            }
        )
    return records


def verify_transition_coding(result: dict) -> int:
    verified = 0
    for cell in result["cells"]:
        recomputed = classify_transition(summary_records(cell))
        if recomputed != cell["transition"]:
            raise AssertionError(f"transition mismatch: {cell['run_id']}")
        verified += 1
    return verified


def verify_primary_test(v36: dict) -> dict:
    frozen = v36["primary_confirmatory_test"]["ordered_onset_test"]
    ordinal_blocks = {
        index: values
        for index, values in enumerate(
            frozen["onset_ordinals_by_block"].values()
        )
    }
    recomputed = exact_blocked_page_test(ordinal_blocks, ("R0", "R1", "R2"))
    fields = (
        "observed_statistic",
        "observed_statistic_doubled_integer",
        "exact_permutation_count",
        "upper_tail_count",
        "exact_one_sided_p",
        "block_ranks",
    )
    for field in fields:
        if recomputed[field] != frozen[field]:
            raise AssertionError(f"primary test mismatch: {field}")
    if recomputed["exact_permutation_count"] != 6**8:
        raise AssertionError("primary test did not reproduce all 6^8 assignments")
    if not math.isclose(
        recomputed["exact_one_sided_p"],
        recomputed["upper_tail_count"]
        / recomputed["exact_permutation_count"],
        rel_tol=0,
        abs_tol=0,
    ):
        raise AssertionError("exact p-value is not upper-tail count / assignments")
    return recomputed


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Read-only lightweight verification of frozen Kepler results."
    )
    parser.add_argument(
        "--skip-distribution-recomputation",
        action="store_true",
        help="Only verify hashes, coding, and exact test.",
    )
    args = parser.parse_args()
    source_hashes = require_frozen_source_hashes()
    v35 = load_json(V35_RESULTS)
    v36 = load_json(V36_RESULTS)
    if not args.skip_distribution_recomputation:
        if build_cue_distribution_audit() != load_json(V35_AUDIT):
            raise AssertionError("v3.5 distribution audit recomputation mismatch")
        if build_confirmatory_distribution_audit() != load_json(V36_AUDIT):
            raise AssertionError("v3.6 distribution audit recomputation mismatch")
    transition_cells = verify_transition_coding(v35) + verify_transition_coding(v36)
    test = verify_primary_test(v36)
    output = {
        "mode": "read_only_no_training",
        "optimizer_steps": 0,
        "frozen_source_hashes": source_hashes,
        "distribution_invariants_recomputed": not args.skip_distribution_recomputation,
        "transition_cells_recomputed": transition_cells,
        "exact_assignment_count": test["exact_permutation_count"],
        "upper_tail_count": test["upper_tail_count"],
        "exact_one_sided_p": test["exact_one_sided_p"],
        "passed": True,
    }
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

