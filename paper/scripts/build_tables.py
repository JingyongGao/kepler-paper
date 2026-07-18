from __future__ import annotations

from typing import Any, Sequence

from common import (
    FREEZE_RECORD,
    TABLES,
    V35_AUDIT,
    V35_MANIFEST,
    V35_RESULTS,
    V36_AUDIT,
    V36_MANIFEST,
    V36_RESULTS,
    atomic_write_text,
    format_float,
    latex_table,
    load_json,
    markdown_table,
    require_frozen_source_hashes,
    write_csv,
)


def display(value: Any) -> str:
    if value is None:
        return "—"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        return format_float(value)
    if isinstance(value, (list, tuple)):
        return ", ".join(display(item) for item in value)
    return str(value)


def write_table(
    stem: str,
    title: str,
    caption: str,
    headers: Sequence[tuple[str, str]],
    rows: list[dict[str, Any]],
) -> None:
    keys = [key for key, _label in headers]
    labels = [label for _key, label in headers]
    write_csv(TABLES / f"{stem}.csv", rows, keys)
    matrix = [[display(row[key]) for key in keys] for row in rows]
    markdown = f"# {title}\n\n{caption}\n\n" + markdown_table(labels, matrix)
    atomic_write_text(TABLES / f"{stem}.md", markdown)
    atomic_write_text(
        TABLES / f"{stem}.tex",
        latex_table(caption, f"tab:{stem.replace('_', '-')}", labels, matrix),
    )


def distribution_rows(v35_audit: dict, v36_audit: dict) -> list[dict[str, Any]]:
    rows = []
    for item in v35_audit["conditions"]:
        rows.append(
            {
                "version": "v3.5",
                "status": "exploratory",
                "rule": "45,36",
                "permutation": "v3.5",
                "condition": item["condition_id"],
                "m": item["m"],
                "target_C_single": item["target_singleton_cue_index"],
                "achieved_C_single": item["singleton_cue_index"],
                "lambda": item["lambda"],
                "minimum_pair_probability": item["minimum_pair_probability"],
                "maximum_uniformity_error": max(
                    item["uniformity_max_absolute_error"].values()
                ),
                "H_Y_given_X1_X2_nats": item["entropies_nats"]["H_Y_given_X1_X2"],
                "joint_information_absolute_error": item[
                    "joint_information_absolute_error"
                ],
                "strict_full_support": item["strictly_positive_full_support"],
            }
        )
    for item in v36_audit["conditions"]:
        rows.append(
            {
                "version": "v3.6",
                "status": "confirmatory",
                "rule": ",".join(str(value) for value in item["coefficients"]),
                "permutation": item["permutation_id"],
                "condition": item["condition_id"],
                "m": item["m"],
                "target_C_single": item["target_singleton_cue_index"],
                "achieved_C_single": item["singleton_cue_index"],
                "lambda": item["lambda"],
                "minimum_pair_probability": item["minimum_pair_probability"],
                "maximum_uniformity_error": max(
                    item["uniformity_max_absolute_error"].values()
                ),
                "H_Y_given_X1_X2_nats": item["entropies_nats"]["H_Y_given_X1_X2"],
                "joint_information_absolute_error": item[
                    "joint_information_absolute_error"
                ],
                "strict_full_support": item["strictly_positive_full_support"],
            }
        )
    return rows


def outcome_rows(
    summaries: dict[str, Any],
    *,
    version: str,
    status: str,
    unit: str,
) -> list[dict[str, Any]]:
    rows = []
    for condition, item in summaries.items():
        exposures = item[
            "onset_exposure_by_seed" if unit == "seed" else "onset_exposure_by_block"
        ]
        terminal = item[
            "onset_terminal_ordinal_by_seed"
            if unit == "seed"
            else "onset_terminal_ordinal_by_block"
        ]
        count = len(exposures)
        rows.append(
            {
                "version": version,
                "status": status,
                "condition": condition,
                "n": count,
                "onset_by_5M": int(round(item["fraction_onset_by_5M"] * count)),
                "onset_by_10M": int(
                    round(item["fraction_onset_verified_by_10M"] * count)
                ),
                "onset_exposures": [
                    None if value is None else float(value) / 1e6 for value in exposures
                ],
                "terminal_onset_ordinals": terminal,
                "established_by_5M": int(
                    round(item["fraction_established_by_5M"] * count)
                ),
                "established_by_10M": int(
                    round(item["fraction_established_by_10M"] * count)
                ),
                "mean_R_5M_pre": item["R_5M_pre"]["mean"],
                "mean_R_10M": item["R_10M"]["mean"],
            }
        )
    return rows


def exact_test_rows(v36: dict) -> list[dict[str, Any]]:
    rows = []
    strata = [
        ("primary", v36["primary_confirmatory_test"]),
        ("rule_0", v36["by_coefficient_rule"]["rule_0"]),
        ("rule_1", v36["by_coefficient_rule"]["rule_1"]),
        ("permutation_0", v36["by_cue_permutation"]["permutation_0"]),
        ("permutation_1", v36["by_cue_permutation"]["permutation_1"]),
    ]
    for name, item in strata:
        test = item["ordered_onset_test"]
        rows.append(
            {
                "analysis": name,
                "inference_status": (
                    "primary"
                    if name == "primary"
                    else "secondary_nominal_not_multiplicity_adjusted"
                ),
                "blocks": test["seed_blocks"],
                "exact_assignments": test["exact_permutation_count"],
                "observed_statistic": test["observed_statistic"],
                "upper_tail_count": test["upper_tail_count"],
                "one_sided_p": test["exact_one_sided_p"],
                "confirmatory_at_0_05": test["exact_one_sided_p"] < 0.05,
                "v3_5_included": False,
            }
        )
    return rows


def paired_rows(v36: dict) -> list[dict[str, Any]]:
    rows = []
    for key, item in v36["primary_confirmatory_test"]["paired_differences"].items():
        rows.append(
            {
                "comparison": item["comparison"],
                "blocks": len(item["blocks"]),
                "mean_R_delta_5M_pre": item["R_5M_pre_delta"]["mean"],
                "median_R_delta_5M_pre": item["R_5M_pre_delta"]["median"],
                "mean_R_delta_10M": item["R_10M_delta"]["mean"],
                "median_R_delta_10M": item["R_10M_delta"]["median"],
                "mean_onset_ordinal_delta": item[
                    "onset_terminal_ordinal_delta"
                ]["mean"],
                "median_onset_ordinal_delta": item[
                    "onset_terminal_ordinal_delta"
                ]["median"],
            }
        )
    return rows


def persistence_rows(v35: dict, v36: dict) -> list[dict[str, Any]]:
    rows = []
    for version, status, summaries, count_key in (
        (
            "v3.5",
            "exploratory",
            v35["condition_summaries"],
            "pre_switch_threshold_positive_seeds",
        ),
        (
            "v3.6",
            "confirmatory",
            v36["primary_confirmatory_test"]["condition_summaries"],
            "pre_switch_threshold_positive_blocks",
        ),
    ):
        for condition, item in summaries.items():
            retention = item["post_switch_threshold_retention"]
            count = int(retention[count_key])
            at6 = retention["fraction_of_pre_positive_retained_at_6M"]
            at10 = retention["fraction_of_pre_positive_retained_at_10M"]
            rows.append(
                {
                    "version": version,
                    "status": status,
                    "condition": condition,
                    "pre_switch_positive": count,
                    "retained_at_6M": (
                        None if at6 is None else int(round(count * float(at6)))
                    ),
                    "retained_at_10M": (
                        None if at10 is None else int(round(count * float(at10)))
                    ),
                    "fraction_retained_at_6M": at6,
                    "fraction_retained_at_10M": at10,
                    "sampler_after_5M": "uniform_full",
                }
            )
    return rows


def protocol_rows(
    freeze: dict,
    v35: dict,
    v36: dict,
    v35_manifest: dict,
    v36_manifest: dict,
) -> list[dict[str, Any]]:
    output = []
    for version, status, result, manifest in (
        ("v3.5", "exploratory", v35, v35_manifest),
        ("v3.6", "confirmatory", v36, v36_manifest),
    ):
        counts = freeze["artifact_counts"][
            "v3.5_exploratory" if version == "v3.5" else "v3.6_confirmatory"
        ]
        output.append(
            {
                "version": version,
                "status": status,
                "registered_trajectories": len(manifest["runs"]),
                "completed_trajectories": result["integrity"]["completed_cells"],
                "checkpoints": counts["checkpoints"],
                "valid_checkpoint_sidecars": counts[
                    "all_checkpoint_sidecars_valid"
                ],
                "exact_evaluations": counts["evaluations"],
                "probe_reports": counts["probe_reports"],
                "gradient_audits": counts["gradient_audit_records"],
                "switch_checks": counts["switch_invariance_records"],
                "all_switch_checks_passed": result["integrity"][
                    "all_switch_invariance_checks_passed"
                ],
                "maximum_exposure": counts["maximum_exposure"],
                "objective": "exact_soft_label_cross_entropy",
                "excluded_from_original_scaling_fit": True,
            }
        )
    return output


def main() -> None:
    require_frozen_source_hashes()
    TABLES.mkdir(parents=True, exist_ok=True)
    v35_audit = load_json(V35_AUDIT)
    v36_audit = load_json(V36_AUDIT)
    v35_manifest = load_json(V35_MANIFEST)
    v36_manifest = load_json(V36_MANIFEST)
    v35 = load_json(V35_RESULTS)
    v36 = load_json(V36_RESULTS)
    freeze = load_json(FREEZE_RECORD)

    write_table(
        "table01_distribution_invariants",
        "Distribution invariants",
        "Every registered cue distribution retained strict full support and the frozen information identities.",
        (
            ("version", "Version"),
            ("status", "Status"),
            ("rule", "Rule"),
            ("permutation", "Permutation"),
            ("condition", "Condition"),
            ("m", "m"),
            ("target_C_single", "Target C_single"),
            ("achieved_C_single", "Achieved C_single"),
            ("lambda", "Lambda"),
            ("minimum_pair_probability", "Min pair probability"),
            ("maximum_uniformity_error", "Max uniformity error"),
            ("H_Y_given_X1_X2_nats", "H(Y|X1,X2), nats"),
            ("joint_information_absolute_error", "Joint-I abs. error"),
            ("strict_full_support", "Strict support"),
        ),
        distribution_rows(v35_audit, v36_audit),
    )

    outcome_headers = (
        ("version", "Version"),
        ("status", "Status"),
        ("condition", "Condition"),
        ("n", "n"),
        ("onset_by_5M", "Onset by 5M"),
        ("onset_by_10M", "Onset by 10M"),
        ("onset_exposures", "Onset exposures (M)"),
        ("terminal_onset_ordinals", "Terminal ordinals"),
        ("established_by_5M", "Established by 5M"),
        ("established_by_10M", "Established by 10M"),
        ("mean_R_5M_pre", "Mean R, 5M pre"),
        ("mean_R_10M", "Mean R, 10M"),
    )
    write_table(
        "table02_v35_exploratory_outcomes",
        "v3.5 exploratory onset and establishment",
        "Exploratory discovery outcomes decoded with the frozen onset and establishment rules.",
        outcome_headers,
        outcome_rows(
            v35["condition_summaries"],
            version="v3.5",
            status="exploratory",
            unit="seed",
        ),
    )
    write_table(
        "table03_v36_confirmatory_outcomes",
        "v3.6 confirmatory onset outcomes",
        "Prospectively frozen outcomes across eight paired blocks. Each block crosses one held-out rule, cue permutation, and initialization; trajectories are paired observations, not independent inference units.",
        outcome_headers,
        outcome_rows(
            v36["primary_confirmatory_test"]["condition_summaries"],
            version="v3.6",
            status="confirmatory",
            unit="block",
        ),
    )
    write_table(
        "table04_exact_tests",
        "Primary and stratified exact tests",
        "The primary row is confirmatory. Rule- and permutation-stratified rows are secondary, nominal, and not multiplicity-adjusted; v3.5 is excluded throughout.",
        (
            ("analysis", "Analysis"),
            ("inference_status", "Inference status"),
            ("blocks", "Blocks"),
            ("exact_assignments", "Exact assignments"),
            ("observed_statistic", "Observed statistic"),
            ("upper_tail_count", "Upper-tail count"),
            ("one_sided_p", "One-sided p"),
            ("confirmatory_at_0_05", "p<0.05"),
            ("v3_5_included", "v3.5 included"),
        ),
        exact_test_rows(v36),
    )
    write_table(
        "table05_paired_R_differences",
        "Paired R differences",
        "Differences are higher registered dose minus lower registered dose within each v3.6 paired block.",
        (
            ("comparison", "Comparison"),
            ("blocks", "Blocks"),
            ("mean_R_delta_5M_pre", "Mean delta R, 5M"),
            ("median_R_delta_5M_pre", "Median delta R, 5M"),
            ("mean_R_delta_10M", "Mean delta R, 10M"),
            ("median_R_delta_10M", "Median delta R, 10M"),
            ("mean_onset_ordinal_delta", "Mean onset-ordinal delta"),
            ("median_onset_ordinal_delta", "Median onset-ordinal delta"),
        ),
        paired_rows(v36),
    )
    write_table(
        "table06_cue_removal_persistence",
        "Cue-removal persistence",
        "Retention is evaluated only among cells positive at the 5M pre-switch checkpoint.",
        (
            ("version", "Version"),
            ("status", "Status"),
            ("condition", "Condition"),
            ("pre_switch_positive", "Positive at 5M pre"),
            ("retained_at_6M", "Retained at 6M"),
            ("retained_at_10M", "Retained at 10M"),
            ("fraction_retained_at_6M", "Fraction at 6M"),
            ("fraction_retained_at_10M", "Fraction at 10M"),
            ("sampler_after_5M", "Sampler after 5M"),
        ),
        persistence_rows(v35, v36),
    )
    write_table(
        "table07_protocol_integrity",
        "Protocol and artifact-integrity audit",
        "Counts and integrity flags are read from the frozen result and confirmatory freeze records.",
        (
            ("version", "Version"),
            ("status", "Status"),
            ("registered_trajectories", "Registered runs"),
            ("completed_trajectories", "Completed runs"),
            ("checkpoints", "Checkpoints"),
            ("valid_checkpoint_sidecars", "Sidecars valid"),
            ("exact_evaluations", "Evaluations"),
            ("probe_reports", "Probe reports"),
            ("gradient_audits", "Gradient audits"),
            ("switch_checks", "Switch checks"),
            ("all_switch_checks_passed", "Switches passed"),
            ("maximum_exposure", "Max exposure"),
            ("objective", "Objective"),
            ("excluded_from_original_scaling_fit", "Excluded from scaling fit"),
        ),
        protocol_rows(freeze, v35, v36, v35_manifest, v36_manifest),
    )
    write_table(
        "table08_claim_boundaries",
        "Claim-boundary table",
        "The manuscript uses the allowed scoped statements and excludes the listed overclaims.",
        (
            ("topic", "Topic"),
            ("allowed_statement", "Allowed statement"),
            ("excluded_statement", "Excluded statement"),
        ),
        [
            {
                "topic": "Causal scope",
                "allowed_statement": "The registered transient cue-bearing full-support distributions changed onset timing or probability in the frozen parent-exposed task and model family.",
                "excluded_statement": "Singleton mutual information alone is the causal variable.",
            },
            {
                "topic": "Persistence",
                "allowed_statement": "Threshold-positive representations persisted after return to uniform full support.",
                "excluded_statement": "The cue is permanently required for prediction.",
            },
            {
                "topic": "Generality",
                "allowed_statement": "The ordered effect replicated across the two new rules and two new permutations tested.",
                "excluded_statement": "This is a universal Transformer property.",
            },
            {
                "topic": "Dynamics",
                "allowed_statement": "The registered checkpoints show abrupt or delayed representation formation in some trajectories.",
                "excluded_statement": "The result is a thermodynamic phase transition.",
            },
            {
                "topic": "Recurrent task",
                "allowed_statement": "The parent-exposed task isolates a mechanism relevant to the broader program.",
                "excluded_statement": "The recurrent production task has been solved.",
            },
            {
                "topic": "Scaling",
                "allowed_statement": "The experiments are mechanism-focused exploratory and confirmatory audits.",
                "excluded_statement": "The result belongs in the original scaling fit.",
            },
            {
                "topic": "C_single",
                "allowed_statement": "C_single is a registered scalar cue index used to order selected distributions.",
                "excluded_statement": "C_single is a unique-information percentage.",
            },
        ],
    )
    print(f"rendered 8 tables to {TABLES}")


if __name__ == "__main__":
    main()
