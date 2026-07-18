from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from pathlib import Path

from pypdf import PdfReader

from common import PAPER, ROOT, atomic_write_json, sha256


TMLR = PAPER / "tmlr"
SUBMISSION = TMLR / "submission"
PDF = PAPER / "output" / "pdf" / "kepler_paper_tmlr_v1_submission.pdf"
SUPPLEMENT = PAPER / "output" / "tmlr" / "kepler_tmlr_supplement_v1.zip"
FROZEN_SUBMISSION_PDF = (
    PAPER / "output" / "pdf" / "kepler_paper_v1_2_1_submission.pdf"
)
FROZEN_SUBMISSION_PDF_SHA256 = (
    "3e70a70094f7e1b0a483b9c5e1633208fde314bc5f0a4e0f70f24a12906761dc"
)
STYLE_HASHES = {
    "LICENSE": "c71d239df91726fc519c6eb72d318ec65820627232b2f796219e87dcf35d0ab4",
    "fancyhdr.sty": "3d2922548e0e5f1a6c5676eda6ebb6dc20d7d305b4d8c2be5f1c833fb1084e6d",
    "tmlr.bst": "306fd454cf40771bee01293eeb98d2c1cd5f4e11ed0cd7296b335f354fc45206",
    "tmlr.sty": "816214ff5919aa457b6b443bee52b15d9561421417b7f8a50cc84651519f0002",
}
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


def digest_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def assert_anonymous(text: str, location: str) -> None:
    for pattern in IDENTITY_PATTERNS:
        if pattern.casefold() in text.casefold():
            raise AssertionError(
                f"identity pattern {pattern!r} found in {location}"
            )


def verify_style() -> dict[str, object]:
    for directory in (TMLR / "template", SUBMISSION):
        for name, expected in STYLE_HASHES.items():
            actual = sha256(directory / name)
            if actual != expected:
                raise AssertionError(
                    f"official style changed: {directory / name}: {actual}"
                )
    return {
        "official_template_commit":
            "7bf90efe3a0debbba703c05c43f3ff7e4d4a2992",
        "style_files_verified": len(STYLE_HASHES) * 2,
        "official_style_files_modified": False,
        "passed": True,
    }


def verify_source() -> dict[str, object]:
    manuscript = (SUBMISSION / "manuscript.tex").read_text(encoding="utf-8")
    assert_anonymous(manuscript, "TMLR manuscript source")
    requirements = (
        r"\usepackage{tmlr}",
        r"\author{\name Anonymous Authors}",
        "LLM-based tools, including ChatGPT/Codex, Claude, and Gemini",
        r"\bibliographystyle{tmlr}",
        r"\appendix",
        r"\section{Supplementary methods}",
    )
    for requirement in requirements:
        if manuscript.count(requirement) != 1:
            raise AssertionError(f"TMLR source requirement mismatch: {requirement}")
    if r"\usepackage[accepted]{tmlr}" in manuscript:
        raise AssertionError("anonymous submission uses accepted TMLR mode")
    if r"\usepackage[preprint]{tmlr}" in manuscript:
        raise AssertionError("anonymous submission uses preprint TMLR mode")
    if manuscript.index(r"\bibliographystyle{tmlr}") > manuscript.index(
        r"\appendix"
    ):
        raise AssertionError("appendix appears before references")
    return {
        "double_blind_style_mode": True,
        "anonymous_author_placeholder": True,
        "llm_disclosure_on_title_footnote": True,
        "references_before_appendix": True,
        "passed": True,
    }


def verify_supplement() -> dict[str, object]:
    if SUPPLEMENT.stat().st_size > 100 * 1024 * 1024:
        raise AssertionError("supplement exceeds TMLR 100 MB limit")
    with zipfile.ZipFile(SUPPLEMENT) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)):
            raise AssertionError("duplicate supplemental ZIP member")
        if any(
            name.endswith((".pt", ".pt.sha256", ".ckpt"))
            for name in names
        ):
            raise AssertionError("checkpoint bytes included in supplement")
        manifest = json.loads(
            archive.read("SUPPLEMENT_MANIFEST.json").decode("utf-8")
        )
        records = {record["path"]: record for record in manifest["files"]}
        if set(names) != set(records) | {"SUPPLEMENT_MANIFEST.json"}:
            raise AssertionError("supplement ZIP members differ from manifest")
        text_files = 0
        for name, record in records.items():
            value = archive.read(name)
            if len(value) != record["bytes"]:
                raise AssertionError(f"supplement size mismatch: {name}")
            if digest_bytes(value) != record["sha256"]:
                raise AssertionError(f"supplement hash mismatch: {name}")
            try:
                text = value.decode("utf-8")
            except UnicodeDecodeError:
                continue
            text_files += 1
            assert_anonymous(text, f"supplement/{name}")
        assert_anonymous(
            json.dumps(manifest, ensure_ascii=False),
            "supplement/SUPPLEMENT_MANIFEST.json",
        )
    return {
        "bytes": SUPPLEMENT.stat().st_size,
        "sha256": sha256(SUPPLEMENT),
        "within_100_mb_limit": True,
        "files": len(names),
        "text_files_scanned": text_files + 1,
        "checkpoint_files": 0,
        "identity_matches": 0,
        "passed": True,
    }


def verify_pdf() -> dict[str, object]:
    reader = PdfReader(str(PDF))
    texts = [page.extract_text() or "" for page in reader.pages]
    complete_text = "\n".join(texts)
    metadata = " ".join(str(value) for value in (reader.metadata or {}).values())
    assert_anonymous(complete_text, "TMLR submission PDF text")
    assert_anonymous(metadata, "TMLR submission PDF metadata")
    required = (
        "Under review as submission to TMLR",
        "Anonymous authors",
        "Paper under double-blind review",
        "LLM-based tools",
        "1,679,616",
        "9.5259869e",
    )
    for phrase in required:
        if phrase not in complete_text:
            raise AssertionError(f"missing TMLR PDF phrase: {phrase}")
    reference_pages = [
        index for index, text in enumerate(texts, start=1)
        if "References" in text
    ]
    appendix_pages = [
        index for index, text in enumerate(texts, start=1)
        if "Supplementary methods" in text
    ]
    if not reference_pages or not appendix_pages:
        raise AssertionError("references or appendix not found in PDF")
    first = reader.pages[0]
    width = float(first.mediabox.width)
    height = float(first.mediabox.height)
    if (round(width), round(height)) != (612, 792):
        raise AssertionError(f"TMLR PDF is not US Letter: {(width, height)}")
    return {
        "pages": len(reader.pages),
        "page_size_points": [width, height],
        "reference_start_page": min(reference_pages),
        "appendix_start_page": min(appendix_pages),
        "main_content_pages_at_most": min(reference_pages),
        "llm_disclosure_present": True,
        "forbidden_identity_matches": 0,
        "passed": True,
    }


def write_package_hashes() -> None:
    paths = set()
    for path in (PAPER / "tmlr").rglob("*"):
        if path.is_file() and path.name != "tmlr_package_hashes.json":
            paths.add(path)
    paths.update(
        {
            PDF,
            SUPPLEMENT,
            PAPER / "scripts" / "build_tmlr_package.py",
            PAPER / "scripts" / "compile_tmlr_pdf.py",
            PAPER / "scripts" / "verify_tmlr_package.py",
            PAPER / "scripts" / "verify_tmlr_supplement.py",
        }
    )
    records = [
        {
            "path": str(path.relative_to(ROOT)),
            "sha256": sha256(path),
            "bytes": path.stat().st_size,
        }
        for path in sorted(paths)
    ]
    atomic_write_json(
        TMLR / "tmlr_package_hashes.json",
        {
            "kind": "tmlr_submission_package_hash_manifest",
            "files": records,
            "file_count": len(records),
            "optimizer_steps": 0,
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write-report", action="store_true")
    args = parser.parse_args()
    if sha256(FROZEN_SUBMISSION_PDF) != FROZEN_SUBMISSION_PDF_SHA256:
        raise AssertionError("frozen v1.2.1 submission PDF changed")
    output = {
        "mode": "tmlr_submission_preparation_no_training",
        "optimizer_steps": 0,
        "new_experiments": 0,
        "frozen_submission_pdf_unchanged": True,
        "style": verify_style(),
        "source": verify_source(),
        "supplement": verify_supplement(),
        "pdf": verify_pdf(),
        "passed": True,
    }
    if args.write_report:
        atomic_write_json(TMLR / "tmlr_audit.json", output)
        write_package_hashes()
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
