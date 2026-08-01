from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
ARTIFACTS = ROOT / "artifacts"
PAPER = ROOT / "paper"
OUTPUT = ROOT / "ICLR2027_NUMERICAL_CLAIM_REGISTRY.json"
CHECKPOINT_OUTPUT = ARTIFACTS / "iclr2027_scale_robustness_checkpoint_hashes.sha256"

BASE_REGISTRY = PAPER / "numerical_claim_registry.json"
V36_MANIFEST = ARTIFACTS / "cue_dose_confirmatory_manifest.json"
V36_RESULTS = ARTIFACTS / "cue_dose_confirmatory_results.json"
SCALE_MANIFEST = ARTIFACTS / "iclr2027_scale_robustness_manifest.json"
SCALE_RESULTS = ARTIFACTS / "iclr2027_scale_robustness_results.json"
SPECTRAL_RESULTS = (
    PAPER / "iclr2027" / "frozen_sources" / "v1_3_spectral_transition_dynamics.json"
)


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def relative(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def actual_scale(manifest: dict, scale_id: str) -> dict:
    rows = {
        (run["width"], run["actual_core_parameters"], run["total_parameters"])
        for run in manifest["runs"]
        if run["scale_id"] == scale_id
    }
    if len(rows) != 1:
        raise AssertionError(f"scale {scale_id} has non-unique architecture records")
    width, core_parameters, total_parameters = rows.pop()
    return {
        "width": width,
        "actual_core_parameters": core_parameters,
        "total_parameters": total_parameters,
    }


def v36_row(manifest: dict, results: dict) -> dict:
    architectures = {
        (run["width"], run["actual_core_parameters"], run["total_parameters"])
        for run in manifest["runs"]
    }
    if len(architectures) != 1:
        raise AssertionError("v3.6 manifest has non-unique architecture records")
    width, core_parameters, total_parameters = architectures.pop()
    summaries = results["primary_confirmatory_test"]["condition_summaries"]
    differences = results["primary_confirmatory_test"]["paired_differences"][
        "R2_vs_R0"
    ]["onset_terminal_ordinal_delta"]
    deltas = differences["values_in_block_order"]
    return {
        "scale_label": "approximately_2m",
        "status": "confirmatory_v3_6",
        "width": width,
        "actual_core_parameters": core_parameters,
        "total_parameters": total_parameters,
        "paired_blocks": len(deltas),
        "verified_onset_counts": {
            "R0": sum(value is not None for value in summaries["R0"]["onset_exposure_by_block"]),
            "R2": sum(value is not None for value in summaries["R2"]["onset_exposure_by_block"]),
        },
        "paired_direction_counts": {
            "advance": sum(value < 0 for value in deltas),
            "tie": sum(value == 0 for value in deltas),
            "delay": sum(value > 0 for value in deltas),
        },
        "mean_R2_minus_R0_onset_ordinal": differences["mean"],
        "source_json": relative(V36_RESULTS),
        "source_fields": [
            "primary_confirmatory_test.condition_summaries.R0.onset_exposure_by_block",
            "primary_confirmatory_test.condition_summaries.R2.onset_exposure_by_block",
            "primary_confirmatory_test.paired_differences.R2_vs_R0.onset_terminal_ordinal_delta",
        ],
    }


def scale_rows(manifest: dict, results: dict) -> list[dict]:
    rows = []
    for scale_id in ("1m", "5m"):
        summary = results["by_scale"][scale_id]
        rows.append(
            {
                "scale_label": f"approximately_{scale_id}",
                "status": "post_confirmatory_robustness",
                **actual_scale(manifest, scale_id),
                **summary,
                "source_json": relative(SCALE_RESULTS),
                "source_field": f"by_scale.{scale_id}",
            }
        )
    return rows


def hessian_rows(results: dict) -> list[dict]:
    return [
        {
            "condition": row["condition"],
            "role": row["role"],
            "exposure": row["exposure"],
            "lambda_max": row["lambda_max"],
            "trace_estimate_mean": row["trace_estimate"]["mean"],
            "effective_spectral_dimension_10": row["effective_spectral_dimension"]["10"],
            "checkpoint_sha256": row["checkpoint_sha256"],
            "source_json": relative(SPECTRAL_RESULTS),
        }
        for row in results["checkpoints"]
    ]


def write_checkpoint_inventory(results: dict) -> int:
    records = []
    for cell in results["cells"]:
        verification = cell["checkpoint_verification"]
        if not verification["all_sidecars_valid"]:
            raise AssertionError(f"invalid checkpoint sidecar set: {cell['run_id']}")
        for check in verification["checks"]:
            if not check["passed"] or check["actual_sha256"] != check["sidecar_sha256"]:
                raise AssertionError(f"checkpoint hash mismatch: {check['checkpoint']}")
            records.append((check["checkpoint"], check["actual_sha256"]))
    records.sort()
    if len(records) != results["integrity"]["checkpoint_files_verified"]:
        raise AssertionError("checkpoint inventory count does not match frozen result")
    temporary = CHECKPOINT_OUTPUT.with_suffix(CHECKPOINT_OUTPUT.suffix + ".tmp")
    temporary.write_text(
        "".join(f"{digest}  {path}\n" for path, digest in records), encoding="utf-8"
    )
    temporary.replace(CHECKPOINT_OUTPUT)
    return len(records)


def main() -> None:
    base = load(BASE_REGISTRY)
    v36_manifest = load(V36_MANIFEST)
    v36_results = load(V36_RESULTS)
    scale_manifest = load(SCALE_MANIFEST)
    scale_results = load(SCALE_RESULTS)
    spectral_results = load(SPECTRAL_RESULTS)

    p_value = v36_results["primary_confirmatory_test"]["ordered_onset_test"][
        "exact_one_sided_p"
    ]
    base_p_value = next(
        claim["value"] for claim in base["claims"] if claim["claim_id"] == "v36_exact_p"
    )
    if p_value != base_p_value:
        raise AssertionError("original v3.6 p-value differs from the frozen registry")

    checkpoint_count = write_checkpoint_inventory(scale_results)
    source_paths = (
        BASE_REGISTRY,
        V36_MANIFEST,
        V36_RESULTS,
        SCALE_MANIFEST,
        SCALE_RESULTS,
        SPECTRAL_RESULTS,
    )
    payload = {
        "kind": "iclr2027_final_numerical_claim_registry",
        "source_policy": "frozen_json_only",
        "base_registry": {
            "path": relative(BASE_REGISTRY),
            "sha256": sha256(BASE_REGISTRY),
            "claim_count": len(base["claims"]),
            "claims": base["claims"],
        },
        "source_files": {
            relative(path): {"sha256": sha256(path), "bytes": path.stat().st_size}
            for path in source_paths
        },
        "original_v3_6_confirmatory_p_value": {
            "value": p_value,
            "unchanged_from_frozen_registry": True,
            "source_json": relative(V36_RESULTS),
            "source_field": "primary_confirmatory_test.ordered_onset_test.exact_one_sided_p",
        },
        "cross_scale_main_text_rows": [
            scale_rows(scale_manifest, scale_results)[0],
            v36_row(v36_manifest, v36_results),
            scale_rows(scale_manifest, scale_results)[1],
        ],
        "cross_scale_paired_blocks": scale_results["paired_blocks"],
        "cross_scale_paired_blocks_source": {
            "source_json": relative(SCALE_RESULTS),
            "source_field": "paired_blocks",
        },
        "exploratory_appendix_hessian_rows": hessian_rows(spectral_results),
        "scope_labels": {
            "approximately_2m": "confirmatory_v3_6",
            "approximately_1m_and_5m": "post_confirmatory_robustness",
            "combined_scale_p_value": None,
            "scaling_law_fit": False,
        },
        "checkpoint_hash_inventory": {
            "path": relative(CHECKPOINT_OUTPUT),
            "records": checkpoint_count,
            "all_sidecars_verified_in_frozen_result": scale_results["integrity"][
                "all_checkpoint_sidecars_valid"
            ],
        },
        "verification": {
            "builder_reads_frozen_json": True,
            "manual_numeric_entry_in_registry": False,
            "new_statistical_tests": 0,
            "optimizer_steps": 0,
            "original_v3_6_p_value_modified": False,
            "combined_scale_p_value_computed": False,
        },
    }
    atomic_json(OUTPUT, payload)
    print(OUTPUT)
    print(CHECKPOINT_OUTPUT)


if __name__ == "__main__":
    main()
