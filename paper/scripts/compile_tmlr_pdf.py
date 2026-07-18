from __future__ import annotations

import os
import re
import shutil
import subprocess
from pathlib import Path

from common import PAPER, atomic_write_json, sha256


SUBMISSION = PAPER / "tmlr" / "submission"
SOURCE_PDF = SUBMISSION / "manuscript.pdf"
SUBMISSION_PDF = SUBMISSION / "submission.pdf"
OUTPUT_PDF = PAPER / "output" / "pdf" / "kepler_paper_tmlr_v1_submission.pdf"
RECORD = PAPER / "tmlr" / "pdf_build_record.json"


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


def check_warnings(output: str) -> list[str]:
    warning_lines = [
        line.strip()
        for line in output.splitlines()
        if "warning:" in line.casefold()
    ]
    forbidden = (
        "missing character",
        "undefined",
        "overfull",
        "error:",
    )
    for line in warning_lines:
        if any(pattern in line.casefold() for pattern in forbidden):
            raise AssertionError(f"TMLR LaTeX warning requires intervention: {line}")
        if "underfull \\vbox" not in line.casefold():
            raise AssertionError(f"unexpected TMLR LaTeX warning: {line}")
    return sorted(
        {
            re.sub(r"^warning:\s*", "", line, flags=re.IGNORECASE)
            for line in warning_lines
        }
    )


def main() -> None:
    tectonic = shutil.which("tectonic")
    if tectonic is None:
        raise RuntimeError("tectonic is required to compile the TMLR PDF")
    first_output = compile_once(tectonic)
    first_hash = sha256(SOURCE_PDF)
    second_output = compile_once(tectonic)
    second_hash = sha256(SOURCE_PDF)
    if first_hash != second_hash:
        raise AssertionError(
            f"TMLR PDF is not deterministic: {first_hash} != {second_hash}"
        )
    warnings = check_warnings(first_output + "\n" + second_output)
    OUTPUT_PDF.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(SOURCE_PDF, SUBMISSION_PDF)
    shutil.copy2(SOURCE_PDF, OUTPUT_PDF)
    atomic_write_json(
        RECORD,
        {
            "kind": "deterministic_tmlr_pdf_build_record",
            "source_date_epoch": 0,
            "consecutive_builds": 2,
            "byte_identical": True,
            "pdf_sha256": second_hash,
            "pdf_bytes": OUTPUT_PDF.stat().st_size,
            "allowed_layout_warnings": warnings,
            "warnings_requiring_intervention": 0,
            "optimizer_steps": 0,
        },
    )
    for suffix in (".pdf", ".log", ".blg"):
        path = SUBMISSION / f"manuscript{suffix}"
        if path.exists():
            path.unlink()
    print(OUTPUT_PDF)
    print(RECORD)


if __name__ == "__main__":
    main()
