from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
PAPER = ROOT / "paper"
ICLR = PAPER / "iclr2027"
SUBMISSION = ICLR / "submission"
SOURCE_PDF = SUBMISSION / "manuscript.pdf"
SUBMISSION_PDF = SUBMISSION / "submission.pdf"
OUTPUT_PDF = PAPER / "output" / "pdf" / "kepler_paper_iclr2027_anonymous_draft.pdf"
OUTPUT_ZIP = PAPER / "output" / "iclr2027" / "kepler_iclr2027_anonymous_code.zip"
BUILD_RECORD = ICLR / "build_record.json"

FORBIDDEN_IDENTITY_TOKENS = (
    "aijing",
    "gao yongjing",
    "yongjing",
    "github.com/jingyonggao",
    "openreview.net/forum?id=mhaj51wviy",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def find_binary(name: str) -> str:
    executable = shutil.which(name)
    if executable:
        return executable
    runtime = Path.home() / ".cache" / "codex-runtimes"
    candidates = sorted(
        runtime.glob(f"**/native/poppler/poppler/bin/{name}"),
        key=lambda item: item.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise RuntimeError(f"required executable not found: {name}")
    return str(candidates[0])


def compile_once(tectonic: str) -> str:
    environment = os.environ.copy()
    environment["SOURCE_DATE_EPOCH"] = "0"
    completed = subprocess.run(
        [tectonic, "--keep-logs", "manuscript.tex"],
        cwd=SUBMISSION,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout + completed.stderr


def audit_warnings(output: str) -> list[str]:
    warnings = [
        line.strip()
        for line in output.splitlines()
        if "warning:" in line.casefold()
    ]
    forbidden = ("overfull", "undefined", "missing character", "error:")
    for warning in warnings:
        if any(item in warning.casefold() for item in forbidden):
            raise AssertionError(f"LaTeX warning requires intervention: {warning}")
        if (
            "underfull" not in warning.casefold()
            and "warnings were issued by the tex engine" not in warning.casefold()
        ):
            raise AssertionError(f"unexpected LaTeX warning: {warning}")
    return sorted(set(re.sub(r"^warning:\s*", "", w, flags=re.I) for w in warnings))


def extract_pdf_text(pdf: Path) -> tuple[list[str], str]:
    pdftotext = find_binary("pdftotext")
    output = SUBMISSION / "manuscript.audit.txt"
    subprocess.run([pdftotext, "-layout", str(pdf), str(output)], check=True)
    text = output.read_text(encoding="utf-8")
    output.unlink()
    return text.split("\f"), text


def audit_pdf(pdf: Path) -> dict:
    pages, text = extract_pdf_text(pdf)
    if pages and not pages[-1].strip():
        pages = pages[:-1]
    references_page = next(i for i, page in enumerate(pages, 1) if "References" in page)
    appendix_page = next(
        i for i, page in enumerate(pages, 1) if "Supplementary methods" in page
    )
    ai_page = next(i for i, page in enumerate(pages, 1) if "AI use statement" in page)
    if references_page != 10:
        raise AssertionError(f"main text exceeds nine pages: references page {references_page}")
    if not appendix_page > references_page:
        raise AssertionError("appendix does not follow references")
    lowered = text.casefold()
    leaks = [token for token in FORBIDDEN_IDENTITY_TOKENS if token in lowered]
    if leaks:
        raise AssertionError(f"identity tokens found in PDF: {leaks}")
    required_phrases = (
        "post-confirmatory robustness extension",
        "excluded from the original v3.6 p-value",
        "not a scaling-law fit",
        "AI use statement",
    )
    missing = [phrase for phrase in required_phrases if phrase.casefold() not in lowered]
    if missing:
        raise AssertionError(f"required manuscript phrases missing: {missing}")
    pdfinfo = subprocess.run(
        [find_binary("pdfinfo"), str(pdf)], check=True, capture_output=True, text=True
    ).stdout
    if "Page size:       612 x 792 pts (letter)" not in pdfinfo:
        raise AssertionError("PDF is not US Letter")
    return {
        "pages": len(pages),
        "main_text_pages": 9,
        "references_start_page": references_page,
        "appendix_start_page": appendix_page,
        "ai_use_statement_page": ai_page,
        "letter_page_size": True,
        "anonymous_text_audit_passed": True,
    }


def audit_fonts(pdf: Path) -> dict:
    output = subprocess.run(
        [find_binary("pdffonts"), str(pdf)],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    lines = [line for line in output.splitlines() if line.strip()]
    if len(lines) < 3:
        raise AssertionError("pdffonts returned no font records")
    header = lines[0]
    emb_start = header.index("emb")
    sub_start = header.index("sub")
    embedded = [line[emb_start:sub_start].strip() for line in lines[2:]]
    if not embedded or any(item != "yes" for item in embedded):
        raise AssertionError("PDF contains a non-embedded font")
    return {"all_fonts_embedded": True, "font_records": len(embedded)}


def archive_inputs() -> list[Path]:
    paths: set[Path] = set()
    for name in ("README.md", "pyproject.toml", "uv.lock"):
        path = ROOT / name
        if path.exists():
            paths.add(path)
    for pattern in ("src/**/*.py", "tests/**/*.py"):
        paths.update(path for path in ROOT.glob(pattern) if path.is_file())
    for pattern in ("artifacts/*.json", "artifacts/*.sha256"):
        paths.update(path for path in ROOT.glob(pattern) if path.is_file())
    for name in (
        "ICLR2027_SCALE_ROBUSTNESS_FREEZE_RECORD.md",
        "ICLR2027_SCALE_ROBUSTNESS_PILOT_REPORT.md",
        "ICLR2027_SCALE_ROBUSTNESS_REPORT.md",
        "CONFIRMATORY_EXPERIMENT_PREREGISTRATION.md",
        "CUE_DOSE_PREREGISTRATION.md",
    ):
        path = ROOT / name
        if path.exists():
            paths.add(path)
    for name in (
        "README.md",
        "REPRODUCE.md",
        "ENVIRONMENT.md",
        "FULL_COMPUTE_DISCLOSURE.md",
        "environment.yml",
        "environment_spec.json",
        "requirements.txt",
    ):
        path = PAPER / name
        if path.exists():
            paths.add(path)
    for pattern in ("tables/*.csv", "tables/*.md"):
        paths.update(path for path in PAPER.glob(pattern) if path.is_file())
    return sorted(paths, key=lambda path: str(path.relative_to(ROOT)))


def write_deterministic_zip(output: Path) -> dict:
    output.parent.mkdir(parents=True, exist_ok=True)
    files = archive_inputs()
    root_name = "kepler_iclr2027_anonymous_code"
    readme = (ICLR / "ANONYMOUS_CODE_README.txt").read_bytes()
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        entries = [(f"{root_name}/README.txt", readme)]
        entries.extend(
            (f"{root_name}/{path.relative_to(ROOT).as_posix()}", path.read_bytes())
            for path in files
        )
        for name, data in sorted(entries):
            lowered = data.decode("utf-8", errors="ignore").casefold()
            leaks = [token for token in FORBIDDEN_IDENTITY_TOKENS if token in lowered]
            if leaks:
                raise AssertionError(f"identity tokens in archive entry {name}: {leaks}")
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            info.create_system = 3
            archive.writestr(info, data, compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)
    return {"files": len(files) + 1, "bytes": output.stat().st_size, "sha256": sha256(output)}


def main() -> None:
    tectonic = shutil.which("tectonic")
    if not tectonic:
        raise RuntimeError("tectonic is required")
    first = compile_once(tectonic)
    first_hash = sha256(SOURCE_PDF)
    second = compile_once(tectonic)
    second_hash = sha256(SOURCE_PDF)
    if first_hash != second_hash:
        raise AssertionError(f"PDF builds differ: {first_hash} != {second_hash}")
    warnings = audit_warnings(first + "\n" + second)
    pdf_audit = audit_pdf(SOURCE_PDF)
    font_audit = audit_fonts(SOURCE_PDF)
    OUTPUT_PDF.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(SOURCE_PDF, SUBMISSION_PDF)
    shutil.copy2(SOURCE_PDF, OUTPUT_PDF)
    archive = write_deterministic_zip(OUTPUT_ZIP)
    record = {
        "kind": "iclr2027_anonymous_draft_build_record",
        "official_template_provenance": "paper/iclr2027/template_provenance.json",
        "source_date_epoch": 0,
        "consecutive_pdf_builds": 2,
        "pdf_builds_byte_identical": True,
        "pdf": {
            "path": str(OUTPUT_PDF.relative_to(ROOT)),
            "bytes": OUTPUT_PDF.stat().st_size,
            "sha256": second_hash,
            **pdf_audit,
            **font_audit,
        },
        "anonymous_code_archive": {
            "path": str(OUTPUT_ZIP.relative_to(ROOT)),
            **archive,
            "checkpoint_binaries_included": False,
            "git_history_included": False,
            "identity_audit_passed": True,
        },
        "allowed_layout_warnings": warnings,
        "warnings_requiring_intervention": 0,
        "scientific_boundaries": {
            "v3_6_p_value_modified": False,
            "combined_p_value_computed": False,
            "scale_extension_post_confirmatory": True,
            "scale_extension_is_scaling_law_fit": False,
            "optional_early_removal_started": False,
            "v1_3_hessian_location": "exploratory appendix only",
        },
    }
    atomic_write_json(BUILD_RECORD, record)
    for suffix in (".pdf", ".log", ".blg"):
        path = SUBMISSION / f"manuscript{suffix}"
        if path.exists():
            path.unlink()
    print(OUTPUT_PDF)
    print(OUTPUT_ZIP)
    print(BUILD_RECORD)


if __name__ == "__main__":
    main()
