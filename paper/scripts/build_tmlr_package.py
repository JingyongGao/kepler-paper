from __future__ import annotations

import hashlib
import json
import shutil
import zipfile
from pathlib import Path
from typing import Any

from common import PAPER, ROOT, atomic_write_json, atomic_write_text, sha256


TMLR = PAPER / "tmlr"
TEMPLATE = TMLR / "template"
SUBMISSION = TMLR / "submission"
SUPPLEMENT_STAGE = PAPER / "tmp" / "tmlr_supplement"
OUTPUT = PAPER / "output" / "tmlr"
SUPPLEMENT_ZIP = OUTPUT / "kepler_tmlr_supplement_v1.zip"

TEMPLATE_SOURCE = (
    "https://github.com/JmlrOrg/tmlr-style-file/"
    "archive/refs/heads/main.zip"
)
TEMPLATE_COMMIT = "7bf90efe3a0debbba703c05c43f3ff7e4d4a2992"
TEMPLATE_ARCHIVE_SHA256 = (
    "48f9f972cc812d2cfc8032a78578c2f5d8939b92fbd4f61095e8cbeb3394f92d"
)
TEMPLATE_FILE_SHA256 = {
    "LICENSE": "c71d239df91726fc519c6eb72d318ec65820627232b2f796219e87dcf35d0ab4",
    "fancyhdr.sty": "3d2922548e0e5f1a6c5676eda6ebb6dc20d7d305b4d8c2be5f1c833fb1084e6d",
    "tmlr.bst": "306fd454cf40771bee01293eeb98d2c1cd5f4e11ed0cd7296b335f354fc45206",
    "tmlr.sty": "816214ff5919aa457b6b443bee52b15d9561421417b7f8a50cc84651519f0002",
}

CORE_ARTIFACTS = (
    "cue_distribution_audit.json",
    "cue_dose_manifest.json",
    "cue_dose_results.json",
    "confirmatory_distribution_audit.json",
    "cue_dose_confirmatory_manifest.json",
    "cue_dose_confirmatory_results.json",
    "v3_6_confirmatory_freeze_record.json",
)
HASHED_SCIENTIFIC_SOURCES = CORE_ARTIFACTS[:6]
SUPPORT_JSONS = (
    "artifact_index.json",
    "environment_spec.json",
    "experiment_genealogy.json",
    "numerical_claim_registry.json",
)
SUPPLEMENT_SCRIPTS = (
    "build_figures.py",
    "build_tables.py",
    "verify_results.py",
    "verify_tmlr_supplement.py",
)
IDENTITY_PATTERNS = (
    "/Users/",
    "/Volumes/",
    "aijing",
    "JingyongGao",
    "Meat-Body",
    "git@github.com",
    "github.com:JingyongGao",
    "Project Kepler Collaboration",
    "kepler-paper.git",
)
PROVENANCE_KEYS_TO_REMOVE = {
    "branch",
    "commit",
    "git_commit",
    "source_commit",
    "source_tag",
    "tag",
}
LLM_DISCLOSURE = (
    "LLM-based tools, including ChatGPT/Codex, Claude, and Gemini, were used "
    "as assistive tools for iterative experimental-design discussion, code "
    "generation and review, manuscript editing, and document preparation. "
    "The authors selected and approved all hypotheses, frozen protocols, "
    "analyses, interpretations, and claims; verified all reported results "
    "through the frozen artifact and reproduction audits; and take full "
    "responsibility for the manuscript."
)


def reset_directory(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True)


def verify_template() -> None:
    for name, expected in TEMPLATE_FILE_SHA256.items():
        path = TEMPLATE / name
        if not path.exists():
            raise AssertionError(f"missing official TMLR template file: {name}")
        actual = sha256(path)
        if actual != expected:
            raise AssertionError(
                f"official TMLR template file changed: {name}: {actual} != {expected}"
            )


def tmlr_manuscript() -> str:
    source = (PAPER / "manuscript.tex").read_text(encoding="utf-8")
    document_marker = "\\begin{document}\n\\maketitle\n"
    supplement_marker = "\\section{Supplementary methods}"
    references_marker = "\\section*{References}"
    if source.count(document_marker) != 1:
        raise AssertionError("unexpected canonical manuscript document preamble")
    if source.count(supplement_marker) != 1:
        raise AssertionError("unexpected canonical supplementary methods section")
    if source.count(references_marker) != 1:
        raise AssertionError("unexpected canonical references section")

    body = source.split(document_marker, 1)[1]
    supplement_index = body.index(supplement_marker)
    references_index = body.index(references_marker)
    if supplement_index >= references_index:
        raise AssertionError("canonical manuscript section order changed")
    main_body = body[:supplement_index].rstrip()
    supplement = body[supplement_index:references_index].rstrip()
    tagged_sentence = (
        "The final result is tagged \\texttt{kepler-v3.6-confirm1}. "
        "Frozen hashes,"
    )
    anonymous_sentence = (
        "The confirmatory result and its frozen hashes are recorded in the "
        "anonymous supplementary artifact package. Frozen hashes,"
    )
    if main_body.count(tagged_sentence) != 1:
        raise AssertionError("unexpected reproducibility tag sentence")
    main_body = main_body.replace(tagged_sentence, anonymous_sentence)

    preamble = rf"""\documentclass[10pt]{{article}}
\usepackage{{tmlr}}
\usepackage{{amsmath,amssymb}}
\usepackage{{booktabs}}
\usepackage{{graphicx}}
\usepackage{{hyperref}}
\usepackage{{url}}
\usepackage{{microtype}}
\hypersetup{{hidelinks}}

\title{{Transient Low-Order Cues Bootstrap Higher-Order Compositional
Representations in Transformers\thanks{{{LLM_DISCLOSURE}}}}}
\author{{\name Anonymous Authors}}

\def\month{{MM}}
\def\year{{YYYY}}
\def\openreview{{\url{{https://openreview.net/forum?id=XXXX}}}}

\begin{{document}}
\maketitle
"""
    references = r"""
\bibliographystyle{tmlr}
\bibliography{references}

\appendix
"""
    return (
        preamble
        + "\n"
        + main_body
        + "\n"
        + references
        + supplement
        + "\n\n\\end{document}\n"
    )


def build_submission_source() -> None:
    reset_directory(SUBMISSION)
    for name in TEMPLATE_FILE_SHA256:
        shutil.copy2(TEMPLATE / name, SUBMISSION / name)
    (SUBMISSION / "figures").mkdir()
    (SUBMISSION / "tables").mkdir()
    for source in sorted((PAPER / "figures").glob("fig*.pdf")):
        shutil.copy2(source, SUBMISSION / "figures" / source.name)
    for source in sorted((PAPER / "tables").glob("table*.tex")):
        table = source.read_text(encoding="utf-8")
        table = table.replace("—", "---").replace("–", "--").replace("‑", "-")
        atomic_write_text(SUBMISSION / "tables" / source.name, table)
    shutil.copy2(PAPER / "references.bib", SUBMISSION / "references.bib")
    atomic_write_text(SUBMISSION / "manuscript.tex", tmlr_manuscript())
    atomic_write_text(
        SUBMISSION / "README.txt",
        (
            "Anonymous TMLR submission source.\n"
            "Official TMLR style is included unmodified.\n"
            "Compile with: SOURCE_DATE_EPOCH=0 tectonic manuscript.tex\n"
        ),
    )


def pointer_escape(value: str) -> str:
    return value.replace("~", "~0").replace("/", "~1")


def sanitize_string(value: str) -> tuple[str, bool]:
    if value.startswith("/"):
        for anchor in ("/artifacts/", "/paper/", "/src/", "/tests/"):
            position = value.find(anchor)
            if position >= 0:
                return value[position + 1 :], True
    root_prefix = str(ROOT) + "/"
    if root_prefix in value:
        return value.replace(root_prefix, ""), True
    return value, False


def sanitize_value(
    value: Any,
    *,
    pointer: str,
    changes: list[dict[str, str]],
    remove_provenance: bool = False,
) -> Any:
    if isinstance(value, dict):
        output = {}
        for key, item in value.items():
            if remove_provenance and key in PROVENANCE_KEYS_TO_REMOVE:
                changes.append(
                    {
                        "json_pointer": f"{pointer}/{pointer_escape(str(key))}",
                        "action": "removed_non_scientific_git_provenance",
                        "original_value_sha256": hashlib.sha256(
                            json.dumps(
                                item,
                                sort_keys=True,
                                ensure_ascii=False,
                            ).encode("utf-8")
                        ).hexdigest(),
                    }
                )
                continue
            output[key] = sanitize_value(
                item,
                pointer=f"{pointer}/{pointer_escape(str(key))}",
                changes=changes,
                remove_provenance=remove_provenance,
            )
        return output
    if isinstance(value, list):
        return [
            sanitize_value(
                item,
                pointer=f"{pointer}/{index}",
                changes=changes,
                remove_provenance=remove_provenance,
            )
            for index, item in enumerate(value)
        ]
    if isinstance(value, str):
        sanitized, changed = sanitize_string(value)
        if changed:
            changes.append(
                {
                    "json_pointer": pointer or "/",
                    "action": "absolute_path_replaced_with_relative_path",
                    "original_value_sha256": hashlib.sha256(
                        value.encode("utf-8")
                    ).hexdigest(),
                    "replacement": sanitized,
                }
            )
        return sanitized
    return value


def sanitize_json(
    source: Path,
    destination: Path,
    *,
    remove_provenance: bool = False,
) -> dict[str, Any]:
    changes: list[dict[str, str]] = []
    value = json.loads(source.read_text(encoding="utf-8"))
    sanitized = sanitize_value(
        value,
        pointer="",
        changes=changes,
        remove_provenance=remove_provenance,
    )
    atomic_write_json(destination, sanitized)
    return {
        "source_path": str(source.relative_to(ROOT)),
        "packaged_path": str(destination.relative_to(SUPPLEMENT_STAGE)),
        "source_sha256": sha256(source),
        "packaged_sha256": sha256(destination),
        "string_or_provenance_changes": len(changes),
        "changes": changes,
    }


def patched_common(packaged_hashes: dict[str, str]) -> str:
    source = (PAPER / "scripts" / "common.py").read_text(encoding="utf-8")
    for name in HASHED_SCIENTIFIC_SOURCES:
        relative = f"artifacts/{name}"
        original = sha256(ROOT / relative)
        replacement = packaged_hashes[relative]
        if source.count(original) != 1:
            raise AssertionError(f"expected one frozen hash in common.py: {relative}")
        source = source.replace(original, replacement)
    return source


def supplement_readme() -> str:
    return """# Anonymous lightweight reproduction package

This package contains the frozen result tables, manifests, distribution
audits, figure/table source data, deterministic builders, and a lightweight
verification path. It contains no model checkpoints and performs no optimizer
steps.

The machine-readable result copies differ from the frozen source bytes only
where absolute local filesystem paths were replaced by relative paths for
double-blind review. `SOURCE_ARTIFACT_HASHES.json` records the original frozen
SHA-256 values, packaged SHA-256 values, and every sanitized JSON field.
Scientific numbers, conditions, coding, and statistical definitions are
unchanged.

## Install

```sh
uv sync --frozen
```

## Verify package hashes

```sh
uv run python paper/scripts/verify_tmlr_supplement.py
```

## Recompute distribution invariants, transition coding, and the exact test

```sh
uv run --frozen python paper/scripts/verify_results.py
```

## Rebuild every publication figure and table from packaged JSON

```sh
uv run --frozen --with reportlab==4.4.9 python paper/scripts/build_figures.py
uv run --frozen python paper/scripts/build_tables.py
```

All commands above are read-only with respect to model parameters and perform
zero training or optimizer steps.
"""


def copy_supplement_sources() -> list[dict[str, Any]]:
    reset_directory(SUPPLEMENT_STAGE)
    (SUPPLEMENT_STAGE / "artifacts").mkdir()
    (SUPPLEMENT_STAGE / "paper" / "scripts").mkdir(parents=True)
    (SUPPLEMENT_STAGE / "paper" / "figures").mkdir(parents=True)
    (SUPPLEMENT_STAGE / "paper" / "tables").mkdir(parents=True)
    (SUPPLEMENT_STAGE / "paper" / "environment").mkdir(parents=True)

    audits = []
    packaged_hashes: dict[str, str] = {}
    for name in CORE_ARTIFACTS:
        source = ROOT / "artifacts" / name
        destination = SUPPLEMENT_STAGE / "artifacts" / name
        record = sanitize_json(source, destination)
        audits.append(record)
        packaged_hashes[f"artifacts/{name}"] = record["packaged_sha256"]

    for name in SUPPORT_JSONS:
        source = PAPER / name
        destination = SUPPLEMENT_STAGE / "paper" / name
        audits.append(
            sanitize_json(
                source,
                destination,
                remove_provenance=name == "experiment_genealogy.json",
            )
        )

    for name in SUPPLEMENT_SCRIPTS:
        shutil.copy2(
            PAPER / "scripts" / name,
            SUPPLEMENT_STAGE / "paper" / "scripts" / name,
        )
    atomic_write_text(
        SUPPLEMENT_STAGE / "paper" / "scripts" / "common.py",
        patched_common(packaged_hashes),
    )
    shutil.copytree(
        ROOT / "src",
        SUPPLEMENT_STAGE / "src",
        dirs_exist_ok=True,
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )
    shutil.copy2(ROOT / "pyproject.toml", SUPPLEMENT_STAGE / "pyproject.toml")
    shutil.copy2(ROOT / "uv.lock", SUPPLEMENT_STAGE / "uv.lock")

    for source in sorted((PAPER / "figures").glob("fig*.source.*")):
        shutil.copy2(source, SUPPLEMENT_STAGE / "paper" / "figures" / source.name)
    for pattern in ("table*.csv", "table*.md"):
        for source in sorted((PAPER / "tables").glob(pattern)):
            shutil.copy2(source, SUPPLEMENT_STAGE / "paper" / "tables" / source.name)
    for name in ("ENVIRONMENT.md", "environment.yml", "FULL_COMPUTE_DISCLOSURE.md"):
        shutil.copy2(
            PAPER / name,
            SUPPLEMENT_STAGE / "paper" / "environment" / name,
        )

    atomic_write_text(SUPPLEMENT_STAGE / "README.md", supplement_readme())
    atomic_write_json(
        SUPPLEMENT_STAGE / "SOURCE_ARTIFACT_HASHES.json",
        {
            "kind": "anonymous_source_artifact_hash_and_sanitization_audit",
            "policy": (
                "Only absolute filesystem paths and non-scientific Git "
                "provenance are removed from review copies."
            ),
            "records": audits,
            "scientific_values_changed": False,
            "optimizer_steps": 0,
        },
    )
    return audits


def scan_identity(root: Path) -> dict[str, Any]:
    scanned = 0
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        scanned += 1
        for pattern in IDENTITY_PATTERNS:
            if pattern.casefold() in text.casefold():
                raise AssertionError(
                    f"anonymous package contains forbidden identity pattern "
                    f"{pattern!r}: {path}"
                )
    return {
        "text_files_scanned": scanned,
        "forbidden_identity_matches": 0,
        "passed": True,
    }


def file_manifest(root: Path) -> list[dict[str, Any]]:
    records = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.name == "SUPPLEMENT_MANIFEST.json":
            continue
        records.append(
            {
                "path": str(path.relative_to(root)),
                "sha256": sha256(path),
                "bytes": path.stat().st_size,
            }
        )
    return records


def deterministic_zip(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with zipfile.ZipFile(
        temporary,
        mode="w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=9,
    ) as archive:
        for path in sorted(source.rglob("*")):
            if not path.is_file():
                continue
            info = zipfile.ZipInfo(
                str(path.relative_to(source)),
                date_time=(1980, 1, 1, 0, 0, 0),
            )
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            archive.writestr(
                info,
                path.read_bytes(),
                compress_type=zipfile.ZIP_DEFLATED,
                compresslevel=9,
            )
    temporary.replace(destination)


def main() -> None:
    verify_template()
    atomic_write_json(
        TMLR / "template_provenance.json",
        {
            "kind": "official_tmlr_template_provenance",
            "retrieved_date": "2026-07-18",
            "source": TEMPLATE_SOURCE,
            "source_commit": TEMPLATE_COMMIT,
            "archive_sha256": TEMPLATE_ARCHIVE_SHA256,
            "file_sha256": TEMPLATE_FILE_SHA256,
            "official_style_files_modified": False,
        },
    )
    build_submission_source()
    audits = copy_supplement_sources()
    identity = scan_identity(SUPPLEMENT_STAGE)
    manifest = {
        "kind": "anonymous_tmlr_lightweight_supplement",
        "generated_date": "2026-07-18",
        "optimizer_steps": 0,
        "checkpoints_included": 0,
        "source_artifact_records": len(audits),
        "identity_audit": identity,
        "files": file_manifest(SUPPLEMENT_STAGE),
    }
    manifest["file_count"] = len(manifest["files"])
    atomic_write_json(SUPPLEMENT_STAGE / "SUPPLEMENT_MANIFEST.json", manifest)
    deterministic_zip(SUPPLEMENT_STAGE, SUPPLEMENT_ZIP)
    first_zip_sha256 = sha256(SUPPLEMENT_ZIP)
    deterministic_zip(SUPPLEMENT_STAGE, SUPPLEMENT_ZIP)
    second_zip_sha256 = sha256(SUPPLEMENT_ZIP)
    if first_zip_sha256 != second_zip_sha256:
        raise AssertionError(
            "supplemental ZIP is not deterministic: "
            f"{first_zip_sha256} != {second_zip_sha256}"
        )
    if SUPPLEMENT_ZIP.stat().st_size > 100 * 1024 * 1024:
        raise AssertionError("TMLR supplemental ZIP exceeds 100 MB")
    atomic_write_json(
        TMLR / "supplement_package.json",
        {
            "kind": "anonymous_tmlr_supplement_package_record",
            "path": str(SUPPLEMENT_ZIP.relative_to(PAPER)),
            "sha256": second_zip_sha256,
            "bytes": SUPPLEMENT_ZIP.stat().st_size,
            "maximum_bytes": 100 * 1024 * 1024,
            "within_tmlr_size_limit": True,
            "consecutive_builds": 2,
            "byte_identical": True,
            "file_count": manifest["file_count"] + 1,
            "identity_audit": identity,
            "optimizer_steps": 0,
        },
    )
    scan_identity(SUBMISSION)
    print(SUBMISSION)
    print(SUPPLEMENT_ZIP)


if __name__ == "__main__":
    main()
