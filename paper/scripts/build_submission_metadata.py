from __future__ import annotations

from typing import Any

from common import (
    FREEZE_RECORD,
    PAPER,
    V35_AUDIT,
    V35_MANIFEST,
    V35_RESULTS,
    V36_AUDIT,
    V36_MANIFEST,
    V36_RESULTS,
    atomic_write_json,
    load_json,
    require_frozen_source_hashes,
)


PRE_TASK_PAPER_COMMIT = "cd25e702358f74121b7942c0354f8d3d257bd0b5"


def count(summary: dict[str, Any], fraction_key: str, values_key: str) -> int:
    values = summary[values_key]
    return int(round(float(summary[fraction_key]) * len(values)))


def claim(
    claim_id: str,
    description: str,
    value: Any,
    source_json: str,
    source_field: str,
    manuscript_locations: list[str],
    *,
    derivation: str | None = None,
) -> dict[str, Any]:
    result = {
        "claim_id": claim_id,
        "description": description,
        "value": value,
        "source_json": source_json,
        "source_field": source_field,
        "manuscript_locations": manuscript_locations,
    }
    if derivation is not None:
        result["derivation"] = derivation
    return result


def build_numerical_registry() -> dict[str, Any]:
    v35_audit = load_json(V35_AUDIT)
    v36_audit = load_json(V36_AUDIT)
    v35_manifest = load_json(V35_MANIFEST)
    v36_manifest = load_json(V36_MANIFEST)
    v35 = load_json(V35_RESULTS)
    v36 = load_json(V36_RESULTS)
    first = v36_audit["conditions"][0]
    primary = v36["primary_confirmatory_test"]
    summaries = primary["condition_summaries"]
    records: list[dict[str, Any]] = []

    def add(*args: Any, **kwargs: Any) -> None:
        records.append(claim(*args, **kwargs))

    add(
        "field_size",
        "Finite-field size p",
        int(v36_audit["p"]),
        "artifacts/confirmatory_distribution_audit.json",
        "p",
        ["Controlled finite-field task", "Limitations"],
    )
    add(
        "ordered_parent_pairs",
        "Complete ordered parent-pair space",
        int(v36_audit["p"]) ** 2,
        "artifacts/confirmatory_distribution_audit.json",
        "p",
        ["Supplementary methods / Task objective, evaluation, and probes"],
        derivation="p^2",
    )
    add(
        "noise_eta",
        "Additive-noise rate eta",
        float(v36_audit["eta"]),
        "artifacts/confirmatory_distribution_audit.json",
        "eta",
        ["Controlled finite-field task", "Limitations"],
    )
    add(
        "conditional_entropy_nats",
        "Bayes conditional entropy H(Y|X1,X2), nats",
        first["entropies_nats"]["H_Y_given_X1_X2"],
        "artifacts/confirmatory_distribution_audit.json",
        "conditions[0].entropies_nats.H_Y_given_X1_X2",
        ["Controlled finite-field task"],
    )
    add(
        "joint_predictable_information_nats",
        "Joint predictable information I(Y;X1,X2), nats",
        first["mutual_information_nats"]["I_Y_X1_X2"],
        "artifacts/confirmatory_distribution_audit.json",
        "conditions[0].mutual_information_nats.I_Y_X1_X2",
        ["Abstract", "Pure-synergy information structure"],
    )
    add(
        "confirmatory_cue_targets",
        "Prospectively frozen confirmatory C_single targets",
        sorted(
            {
                float(item["target_singleton_cue_index"])
                for item in v36_audit["conditions"]
            }
        ),
        "artifacts/confirmatory_distribution_audit.json",
        "conditions[*].target_singleton_cue_index",
        ["Prospectively frozen confirmatory replication"],
    )
    add(
        "v36_max_uniformity_error",
        "Maximum marginal-uniformity absolute error across v3.6 conditions",
        max(
            max(item["uniformity_max_absolute_error"].values())
            for item in v36_audit["conditions"]
        ),
        "artifacts/confirmatory_distribution_audit.json",
        "conditions[*].uniformity_max_absolute_error",
        ["Full-support cue-family construction"],
        derivation="maximum over all conditions and X1/X2/Z/Y marginals",
    )
    add(
        "v36_min_pair_probability",
        "Minimum pair probability across v3.6 conditions",
        min(item["minimum_pair_probability"] for item in v36_audit["conditions"]),
        "artifacts/confirmatory_distribution_audit.json",
        "conditions[*].minimum_pair_probability",
        ["Full-support cue-family construction"],
        derivation="minimum over all confirmatory conditions",
    )
    add(
        "v35_rule",
        "Exploratory v3.5 coefficient rule",
        list(v35_audit["coefficients"]),
        "artifacts/cue_distribution_audit.json",
        "coefficients",
        ["Exploratory discovery experiment"],
    )
    add(
        "v35_seed_count",
        "Exploratory paired initialization seeds",
        len(
            {
                int(run["model_initialization_seed"])
                for run in v35_manifest["runs"]
            }
        ),
        "artifacts/cue_dose_manifest.json",
        "runs[*].model_initialization_seed",
        ["Exploratory discovery experiment"],
        derivation="number of unique model-initialization seeds",
    )
    for condition in ("C0", "C1", "C2", "C3"):
        summary = v35["condition_summaries"][condition]
        add(
            f"v35_onset_{condition}",
            f"Exploratory verified-onset count for {condition}",
            count(
                summary,
                "fraction_onset_verified_by_10M",
                "onset_exposure_by_seed",
            ),
            "artifacts/cue_dose_results.json",
            f"condition_summaries.{condition}.fraction_onset_verified_by_10M",
            ["Abstract", "Exploratory discovery experiment"],
            derivation="fraction multiplied by five paired seeds",
        )
    v35_test = v35["ordered_dose_trend"]
    for identifier, description, field in (
        ("v35_exact_assignments", "Exploratory exact assignments", "exact_permutation_count"),
        ("v35_upper_tail", "Exploratory exact upper-tail count", "upper_tail_count"),
        ("v35_exact_p", "Exploratory one-sided exact p-value", "exact_one_sided_p"),
    ):
        add(
            identifier,
            description,
            v35_test[field],
            "artifacts/cue_dose_results.json",
            f"ordered_dose_trend.{field}",
            ["Exploratory discovery experiment"],
        )
    add(
        "v36_rules",
        "Held-out confirmatory coefficient rules",
        [item["coefficients"] for item in v36_audit["rules"]],
        "artifacts/confirmatory_distribution_audit.json",
        "rules[*].coefficients",
        ["Prospectively frozen confirmatory replication"],
    )
    add(
        "v36_blocks",
        "Primary rule x permutation x initialization inference blocks",
        int(primary["ordered_onset_test"]["seed_blocks"]),
        "artifacts/cue_dose_confirmatory_results.json",
        "primary_confirmatory_test.ordered_onset_test.seed_blocks",
        ["Abstract", "Prospectively frozen confirmatory replication"],
    )
    add(
        "v36_trajectories",
        "Paired confirmatory condition trajectories",
        int(v36["integrity"]["completed_cells"]),
        "artifacts/cue_dose_confirmatory_results.json",
        "integrity.completed_cells",
        ["Prospectively frozen confirmatory replication"],
    )
    for condition in ("R0", "R1", "R2"):
        summary = summaries[condition]
        add(
            f"v36_onset_{condition}",
            f"Confirmatory verified-onset blocks for {condition}",
            count(
                summary,
                "fraction_onset_verified_by_10M",
                "onset_exposure_by_block",
            ),
            "artifacts/cue_dose_confirmatory_results.json",
            f"primary_confirmatory_test.condition_summaries.{condition}.fraction_onset_verified_by_10M",
            ["Abstract", "Prospectively frozen confirmatory replication"],
            derivation="fraction multiplied by eight paired blocks",
        )
        add(
            f"v36_establishment_{condition}",
            f"Confirmatory establishment blocks for {condition}",
            count(
                summary,
                "fraction_established_by_10M",
                "onset_exposure_by_block",
            ),
            "artifacts/cue_dose_confirmatory_results.json",
            f"primary_confirmatory_test.condition_summaries.{condition}.fraction_established_by_10M",
            ["Prospectively frozen confirmatory replication"],
            derivation="fraction multiplied by eight paired blocks",
        )
    test = primary["ordered_onset_test"]
    for identifier, description, field in (
        ("v36_observed_statistic", "Primary Page-style observed statistic", "observed_statistic"),
        ("v36_exact_assignments", "Primary exact assignments", "exact_permutation_count"),
        ("v36_upper_tail", "Primary exact upper-tail count", "upper_tail_count"),
        ("v36_exact_p", "Primary one-sided exact p-value", "exact_one_sided_p"),
    ):
        add(
            identifier,
            description,
            test[field],
            "artifacts/cue_dose_confirmatory_results.json",
            f"primary_confirmatory_test.ordered_onset_test.{field}",
            ["Abstract", "Prospectively frozen confirmatory replication"],
        )
    retained = sum(
        int(
            summaries[condition]["post_switch_threshold_retention"][
                "pre_switch_threshold_positive_blocks"
            ]
        )
        for condition in ("R1", "R2")
    )
    add(
        "v36_persistence_cells",
        "Nonzero-cue cells positive at 5M and retained at 6M and 10M",
        retained,
        "artifacts/cue_dose_confirmatory_results.json",
        "primary_confirmatory_test.condition_summaries.{R1,R2}.post_switch_threshold_retention",
        ["Abstract", "Representation persistence after cue removal"],
        derivation="R1 plus R2 pre-switch-positive blocks; both retention fractions equal 1",
    )
    for comparison in ("R1_vs_R0", "R2_vs_R1"):
        for checkpoint in ("R_5M_pre_delta", "R_10M_delta"):
            add(
                f"{comparison}_{checkpoint}_mean",
                f"Mean paired {comparison} {checkpoint}",
                primary["paired_differences"][comparison][checkpoint]["mean"],
                "artifacts/cue_dose_confirmatory_results.json",
                f"primary_confirmatory_test.paired_differences.{comparison}.{checkpoint}.mean",
                ["Prospectively frozen confirmatory replication"],
            )
    add(
        "core_parameters",
        "Core model parameters",
        int(v36_manifest["runs"][0]["actual_core_parameters"]),
        "artifacts/cue_dose_confirmatory_manifest.json",
        "runs[0].actual_core_parameters",
        ["Supplementary methods / Model and optimization"],
    )
    add(
        "warmup_exposures",
        "Learning-rate warmup exposures",
        int(v36_manifest["frozen_protocol"]["optimizer"]["warmup_tokens"]),
        "artifacts/cue_dose_confirmatory_manifest.json",
        "frozen_protocol.optimizer.warmup_tokens",
        ["Limitations", "Supplementary methods / Model and optimization"],
    )
    add(
        "cue_switch_exposure",
        "Cue removal exposure",
        int(v36_manifest["frozen_protocol"]["switch_exposure"]),
        "artifacts/cue_dose_confirmatory_manifest.json",
        "frozen_protocol.switch_exposure",
        ["Figure 1", "Representation persistence after cue removal", "Limitations"],
    )
    add(
        "maximum_exposure",
        "Maximum confirmatory exposure",
        int(v36_manifest["frozen_protocol"]["maximum_exposure"]),
        "artifacts/cue_dose_confirmatory_manifest.json",
        "frozen_protocol.maximum_exposure",
        ["Figure 1", "Prospectively frozen confirmatory replication"],
    )
    return {
        "kind": "kepler_submission_numerical_claim_registry",
        "source_policy": "frozen_json_only",
        "headline_claim_count": len(records),
        "claims": records,
        "verification": {
            "builder_reads_frozen_json": True,
            "manual_numeric_entry_in_registry": False,
            "new_statistical_tests": 0,
            "optimizer_steps": 0,
        },
    }


def main() -> None:
    source_hashes = require_frozen_source_hashes()
    freeze = load_json(FREEZE_RECORD)
    baseline = {
        "kind": "kepler_paper_v1_2_submission_science_baseline",
        "pre_task_paper_commit": PRE_TASK_PAPER_COMMIT,
        "frozen_scientific_commit": freeze["final_v3_6_commit"],
        "frozen_scientific_tag": freeze["final_v3_6_tag"],
        "source_hashes": source_hashes,
        "all_manifest_sha256": freeze["manifest_sha256"],
        "result_sha256": freeze["result_sha256"],
        "tracked_scientific_artifact_sha256": freeze[
            "tracked_scientific_artifact_sha256_at_freeze"
        ],
        "tracked_scientific_artifact_count": len(
            freeze["tracked_scientific_artifact_sha256_at_freeze"]
        ),
        "verified_before_submission_hardening": True,
        "optimizer_steps": 0,
    }
    atomic_write_json(PAPER / "submission_science_baseline.json", baseline)
    atomic_write_json(
        PAPER / "numerical_claim_registry.json",
        build_numerical_registry(),
    )
    print(PAPER / "submission_science_baseline.json")
    print(PAPER / "numerical_claim_registry.json")


if __name__ == "__main__":
    main()
