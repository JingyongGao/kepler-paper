from __future__ import annotations

from common import (
    PAPER,
    ROOT,
    V35_AUDIT,
    V35_MANIFEST,
    V35_RESULTS,
    V36_AUDIT,
    V36_MANIFEST,
    V36_RESULTS,
    atomic_write_json,
    atomic_write_text,
    checkpoint_seed_provenance,
    load_json,
    markdown_table,
    require_frozen_source_hashes,
    sha256,
)


METRICS = {
    "exact_conditional_kl": (
        "Mean over the registered full-support parent-pair evaluation set of "
        "D_KL(q_h || p_theta)."
    ),
    "normalized_R": (
        "Exact conditional KL divided by log(p) - H(nu_eta); a uniform "
        "predictor has R=1 and the analytic oracle has R=0."
    ),
    "row_bootstrap_interval": (
        "Registered 95% interval from 2,000 bootstrap resamples of first-parent rows."
    ),
    "center_accuracy": "Indicator that argmax_y p_theta(y|h) equals the true center z.",
    "mean_center_probability": "Mean model probability assigned to the true center z.",
    "predictive_entropy": "Mean entropy, in nats, of p_theta(.|h).",
    "representation_onset": (
        "First registered probe checkpoint with full-support R<0.9 and "
        "layer-6 linear-probe accuracy>0.5, with both retained at the next "
        "registered probe checkpoint."
    ),
    "representation_establishment": (
        "First registered probe checkpoint with full-support R<0.1, center "
        "accuracy>0.9, and layer-6 linear-probe accuracy>0.9."
    ),
    "page_style_test": (
        "One-sided exact blocked Page-style within-block label-permutation "
        "test using doubled within-block midranks; larger statistics align "
        "earlier onset with increasing registered cue dose."
    ),
    "persistence": (
        "Retention of the frozen onset threshold at 6M and 10M among cells "
        "that were threshold-positive at the 5M pre-switch evaluation."
    ),
}


def seed_summary(manifest: dict) -> dict:
    runs = manifest["runs"]
    return {
        "model_initialization_seeds": sorted(
            {int(run["model_initialization_seed"]) for run in runs}
        ),
        "data_seeds": sorted({int(run["data_seed"]) for run in runs}),
        "evaluation_seeds": sorted({int(run["evaluation_seed"]) for run in runs}),
        "probe_seeds": sorted({int(run["probe_seed"]) for run in runs}),
    }


def main() -> None:
    frozen = require_frozen_source_hashes()
    v35_audit = load_json(V35_AUDIT)
    v35_manifest = load_json(V35_MANIFEST)
    v35_results = load_json(V35_RESULTS)
    v36_audit = load_json(V36_AUDIT)
    v36_manifest = load_json(V36_MANIFEST)
    v36_results = load_json(V36_RESULTS)

    entries = [
        {
            "id": "v3.5-cue-distribution-audit",
            "originating_version": "Project Kepler v3.5",
            "status": "exploratory_discovery",
            "source_json": str(V35_AUDIT.relative_to(ROOT)),
            "sha256": sha256(V35_AUDIT),
            "figure_destinations": [
                "paper/figures/fig01_task_cue_schematic",
                "paper/figures/fig07_v35_geometry_comparison",
            ],
            "table_destinations": ["paper/tables/table01_distribution_invariants"],
            "metric_definitions": [
                "strict full support",
                "marginal uniformity of X1, X2, Z, and Y",
                "H(Y|X1,X2)",
                "I(Y;X1,X2)",
                "registered C_single",
            ],
            "checkpoint_and_seed_provenance": {
                "trained": v35_audit["trained"],
                "permutation_seed_sha256": v35_audit["permutation_seed_sha256"],
                "manifest_seed_summary": seed_summary(v35_manifest),
            },
        },
        {
            "id": "v3.5-cue-dose-results",
            "originating_version": "Project Kepler v3.5",
            "status": "exploratory_discovery",
            "source_json": str(V35_RESULTS.relative_to(ROOT)),
            "sha256": sha256(V35_RESULTS),
            "figure_destinations": [
                "paper/figures/fig02_v35_dose_trajectories",
                "paper/figures/fig07_v35_geometry_comparison",
            ],
            "table_destinations": [
                "paper/tables/table02_v35_exploratory_outcomes",
                "paper/tables/table06_cue_removal_persistence",
                "paper/tables/table07_protocol_integrity",
            ],
            "metric_definitions": list(METRICS),
            "checkpoint_and_seed_provenance": checkpoint_seed_provenance(
                v35_results, v35_manifest
            ),
        },
        {
            "id": "v3.6-confirmatory-distribution-audit",
            "originating_version": "Project Kepler v3.6",
            "status": "confirmatory_replication",
            "source_json": str(V36_AUDIT.relative_to(ROOT)),
            "sha256": sha256(V36_AUDIT),
            "figure_destinations": ["paper/figures/fig01_task_cue_schematic"],
            "table_destinations": ["paper/tables/table01_distribution_invariants"],
            "metric_definitions": [
                "strict full support",
                "marginal uniformity of X1, X2, Z, and Y",
                "H(Y|X1,X2)",
                "I(Y;X1,X2)",
                "registered C_single",
            ],
            "checkpoint_and_seed_provenance": {
                "trained": v36_audit["trained"],
                "generation_seeds_sha256": v36_audit["generation_seeds_sha256"],
                "manifest_seed_summary": seed_summary(v36_manifest),
            },
        },
        {
            "id": "v3.6-confirmatory-results",
            "originating_version": "Project Kepler v3.6",
            "status": "confirmatory_replication",
            "source_json": str(V36_RESULTS.relative_to(ROOT)),
            "sha256": sha256(V36_RESULTS),
            "figure_destinations": [
                "paper/figures/fig03_v36_paired_onset",
                "paper/figures/fig04_v36_exact_null",
                "paper/figures/fig05_v36_persistence",
                "paper/figures/fig06_v36_rule_heterogeneity",
            ],
            "table_destinations": [
                "paper/tables/table03_v36_confirmatory_outcomes",
                "paper/tables/table04_exact_tests",
                "paper/tables/table05_paired_R_differences",
                "paper/tables/table06_cue_removal_persistence",
                "paper/tables/table07_protocol_integrity",
                "paper/tables/table08_claim_boundaries",
            ],
            "metric_definitions": list(METRICS),
            "checkpoint_and_seed_provenance": checkpoint_seed_provenance(
                v36_results, v36_manifest
            ),
        },
    ]
    index = {
        "project": "Project Kepler manuscript package",
        "working_title": (
            "Transient Low-Order Cues Bootstrap High-Order Compositional "
            "Representations in Transformers"
        ),
        "frozen_source_sha256": frozen,
        "metric_definitions": METRICS,
        "separation_rule": {
            "v3.5": "exploratory discovery only",
            "v3.6": "independent confirmatory replication",
            "primary_confirmatory_p_value_combines_v3.5": False,
        },
        "artifacts": entries,
    }
    atomic_write_json(PAPER / "artifact_index.json", index)

    rows = [
        (
            entry["id"],
            entry["originating_version"],
            entry["status"],
            f"`{entry['source_json']}`",
            f"`{entry['sha256']}`",
        )
        for entry in entries
    ]
    sections = [
        "# Canonical Artifact Index",
        "",
        "This index separates the v3.5 exploratory discovery from the v3.6 "
        "independent confirmatory replication. The v3.5 result is never "
        "combined into the primary v3.6 p-value.",
        "",
        markdown_table(
            ("Artifact", "Version", "Status", "Frozen source", "SHA-256"),
            rows,
        ).rstrip(),
        "",
        "## Metric definitions",
        "",
    ]
    sections.extend(f"- **{name}:** {definition}" for name, definition in METRICS.items())
    sections.extend(["", "## Destinations and provenance", ""])
    for entry in entries:
        provenance = entry["checkpoint_and_seed_provenance"]
        sections.extend(
            [
                f"### {entry['id']}",
                "",
                f"- Figures: {', '.join(f'`{item}`' for item in entry['figure_destinations']) or 'none'}",
                f"- Tables: {', '.join(f'`{item}`' for item in entry['table_destinations']) or 'none'}",
                f"- Provenance: `{json_compact(provenance)}`",
                "",
            ]
        )
    atomic_write_text(PAPER / "ARTIFACT_INDEX.md", "\n".join(sections).rstrip() + "\n")
    print(PAPER / "artifact_index.json")


def json_compact(value: object) -> str:
    import json

    text = json.dumps(value, sort_keys=True, separators=(",", ":"))
    if len(text) > 600:
        return text[:597] + "..."
    return text


if __name__ == "__main__":
    main()
