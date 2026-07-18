from __future__ import annotations

from common import PAPER, atomic_write_json, git, load_json, sha256


SOURCE_PAPER_COMMIT = "d1377ee1a67c37da09aef89ca6ab0c68c9a4ee61"
OLD_PDF_SHA256 = "37e91cef9e9fc79b453083eaaeeda455dc9aef714e67857a4ecb9cfe74a4aa1c"


def main() -> None:
    if git("rev-parse", "HEAD") != SOURCE_PAPER_COMMIT:
        raise AssertionError("preflight baseline must be initialized at v1.2 HEAD")
    old_pdf = PAPER / "output" / "pdf" / "kepler_paper_v1_2_submission.pdf"
    if sha256(old_pdf) != OLD_PDF_SHA256:
        raise AssertionError("v1.2 submission PDF differs from recorded baseline")
    science = load_json(PAPER / "submission_science_baseline.json")
    output = {
        "kind": "kepler_paper_v1_2_1_final_preflight_baseline",
        "source_paper_commit": SOURCE_PAPER_COMMIT,
        "old_submission_pdf": str(old_pdf.relative_to(PAPER)),
        "old_submission_pdf_sha256": OLD_PDF_SHA256,
        "frozen_scientific_commit": science["frozen_scientific_commit"],
        "frozen_scientific_tag": science["frozen_scientific_tag"],
        "tracked_scientific_artifact_count": science[
            "tracked_scientific_artifact_count"
        ],
        "tracked_scientific_artifact_sha256": science[
            "tracked_scientific_artifact_sha256"
        ],
        "all_manifest_sha256": science["all_manifest_sha256"],
        "result_sha256": science["result_sha256"],
        "figure_source_data_sha256_before": {
            path.name: sha256(path)
            for path in sorted((PAPER / "figures").glob("fig*.source.*"))
        },
        "table_csv_sha256_before": {
            path.name: sha256(path)
            for path in sorted((PAPER / "tables").glob("table*.csv"))
        },
        "verified_before_final_preflight": True,
        "optimizer_steps": 0,
    }
    destination = PAPER / "final_preflight_baseline.json"
    atomic_write_json(destination, output)
    print(destination)


if __name__ == "__main__":
    main()
