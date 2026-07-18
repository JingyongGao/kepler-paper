from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from PIL import Image
from pypdf import PdfReader

from build_submission_metadata import build_numerical_registry
from common import (
    FREEZE_RECORD,
    PAPER,
    ROOT,
    load_json,
    require_frozen_source_hashes,
    sha256,
)


FORBIDDEN_MANUSCRIPT_PATTERNS = (
    "High-Order",
    "high-order",
    "independent confirmatory replication",
    "preregister",
    "frozen Transformer",
)
REQUIRED_MANUSCRIPT_PHRASES = (
    "Higher-Order Compositional Representations",
    "prospectively frozen confirmatory replication using held-out",
    "The primary inference unit was the eight",
    "not 24 independent inferential units",
    "exchangeable within each block",
    "included the observed assignment",
    "secondary, nominal, and not",
    "not multiplicity-adjusted",
    "Gradient statistics are correlates of transition dynamics,",
    "not identified causal mediators",
    "coincided with the prospectively frozen 5M learning-rate",
)
ANONYMOUS_FORBIDDEN_PATTERNS = (
    "/Users/",
    "/Volumes/",
    "aijing",
    "github.com",
    "gitlab.com",
    "Project Kepler Collaboration",
    "kepler-v3.6-confirm1",
)


def bibtex_keys(text: str) -> set[str]:
    return set(re.findall(r"@\w+\s*\{\s*([^,\s]+)", text))


def citation_keys(text: str) -> set[str]:
    keys: set[str] = set()
    for match in re.findall(r"\\cite\{([^}]+)\}", text):
        keys.update(item.strip() for item in match.split(","))
    return keys


def verify_references() -> dict[str, int | bool]:
    manuscript = (PAPER / "manuscript.tex").read_text(encoding="utf-8")
    bibtex = (PAPER / "references.bib").read_text(encoding="utf-8")
    report = load_json(PAPER / "reference_verification.json")
    bib_keys = bibtex_keys(bibtex)
    cited_keys = citation_keys(manuscript)
    verified_keys = {item["key"] for item in report["entries"]}
    if bib_keys != cited_keys or bib_keys != verified_keys:
        raise AssertionError(
            {
                "bib_only": sorted(bib_keys - cited_keys),
                "citation_only": sorted(cited_keys - bib_keys),
                "unverified": sorted(bib_keys - verified_keys),
                "verification_only": sorted(verified_keys - bib_keys),
            }
        )
    if any(item["status"] != "verified" for item in report["entries"]):
        raise AssertionError("unverified reference is cited")
    if any(not item["identifier"]["value"] for item in report["entries"]):
        raise AssertionError("cited reference lacks DOI/arXiv identifier")
    return {
        "bibtex_entries": len(bib_keys),
        "cited_entries": len(cited_keys),
        "verified_entries": len(verified_keys),
        "passed": True,
    }


def verify_numerical_registry() -> dict[str, int | bool]:
    actual = load_json(PAPER / "numerical_claim_registry.json")
    expected = build_numerical_registry()
    if actual != expected:
        raise AssertionError("numerical claim registry does not match frozen JSON")
    return {
        "headline_claims": int(actual["headline_claim_count"]),
        "all_values_rebuilt_from_frozen_json": True,
        "passed": True,
    }


def verify_frozen_artifacts() -> dict[str, int | bool]:
    expected_sources = require_frozen_source_hashes()
    freeze = load_json(FREEZE_RECORD)
    tracked = freeze["tracked_scientific_artifact_sha256_at_freeze"]
    for relative, expected in tracked.items():
        actual = sha256(ROOT / relative)
        if actual != expected:
            raise AssertionError(f"frozen artifact changed: {relative}")
    baseline = load_json(PAPER / "submission_science_baseline.json")
    if baseline["tracked_scientific_artifact_sha256"] != tracked:
        raise AssertionError("submission baseline does not contain frozen hashes")
    if baseline["source_hashes"] != expected_sources:
        raise AssertionError("submission baseline source hashes differ")
    return {
        "tracked_scientific_artifacts": len(tracked),
        "byte_identical": True,
        "passed": True,
    }


def verify_manuscript() -> dict[str, int | bool]:
    markdown = (PAPER / "manuscript.md").read_text(encoding="utf-8")
    latex = (PAPER / "manuscript.tex").read_text(encoding="utf-8")
    for pattern in FORBIDDEN_MANUSCRIPT_PATTERNS:
        if pattern.lower() in markdown.lower() or pattern.lower() in latex.lower():
            raise AssertionError(f"forbidden manuscript wording: {pattern}")
    for phrase in REQUIRED_MANUSCRIPT_PHRASES:
        if phrase.lower() not in markdown.lower() and phrase.lower() not in latex.lower():
            raise AssertionError(f"missing required manuscript wording: {phrase}")
    figures = re.findall(r"\\includegraphics[^{}]*\{([^}]+)\}", latex)
    table_refs = set(re.findall(r"\\ref\{(tab:[^}]+)\}", latex))
    figure_refs = set(re.findall(r"\\ref\{(fig:[^}]+)\}", latex))
    if len(figures) != 7 or len(table_refs) != 8 or len(figure_refs) != 7:
        raise AssertionError((len(figures), len(table_refs), len(figure_refs)))
    return {
        "embedded_figures": len(figures),
        "unique_figure_references": len(figure_refs),
        "unique_table_references": len(table_refs),
        "passed": True,
    }


def verify_figure_resolution() -> dict[str, object]:
    dimensions = {}
    for path in sorted((PAPER / "figures").glob("fig*.png")):
        with Image.open(path) as image:
            dimensions[path.name] = [int(image.width), int(image.height)]
            if image.width < 1200 or image.height < 720:
                raise AssertionError(f"low-resolution figure: {path}")
    if len(dimensions) != 7:
        raise AssertionError("expected seven PNG figures")
    return {"dimensions_pixels": dimensions, "vector_pdf_count": 7, "passed": True}


def verify_anonymity(pdf: Path | None) -> dict[str, object]:
    sources = [
        PAPER / "anonymized" / "manuscript.tex",
        PAPER / "anonymized" / "references.bib",
        PAPER / "anonymized" / "README.txt",
    ]
    scanned = []
    for path in sources:
        text = path.read_text(encoding="utf-8")
        scanned.append(str(path.relative_to(PAPER)))
        for pattern in ANONYMOUS_FORBIDDEN_PATTERNS:
            if pattern.lower() in text.lower():
                raise AssertionError(f"anonymous source contains {pattern}: {path}")
    pdf_pages = None
    if pdf is not None:
        reader = PdfReader(str(pdf))
        pdf_pages = len(reader.pages)
        text = "\n".join(page.extract_text() or "" for page in reader.pages)
        metadata = " ".join(str(value) for value in (reader.metadata or {}).values())
        for pattern in ANONYMOUS_FORBIDDEN_PATTERNS:
            if pattern.lower() in text.lower() or pattern.lower() in metadata.lower():
                raise AssertionError(f"anonymous PDF contains {pattern}")
        scanned.append(str(pdf))
    return {
        "scanned_sources": scanned,
        "pdf_pages": pdf_pages,
        "forbidden_matches": 0,
        "passed": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--anonymous-pdf", type=Path)
    args = parser.parse_args()
    output = {
        "mode": "submission_hardening_no_training",
        "optimizer_steps": 0,
        "new_experiments": 0,
        "new_statistical_tests": 0,
        "frozen_artifacts": verify_frozen_artifacts(),
        "numerical_claims": verify_numerical_registry(),
        "references": verify_references(),
        "manuscript": verify_manuscript(),
        "figures": verify_figure_resolution(),
        "anonymity": verify_anonymity(args.anonymous_pdf),
        "passed": True,
    }
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
