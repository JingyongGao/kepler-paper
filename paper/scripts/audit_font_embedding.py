from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path

from common import PAPER, atomic_write_text, sha256


def find_pdffonts() -> Path:
    executable = shutil.which("pdffonts")
    if executable:
        return Path(executable)
    runtime_root = Path.home() / ".cache" / "codex-runtimes"
    candidates = sorted(
        runtime_root.glob("**/native/poppler/poppler/bin/pdffonts"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise RuntimeError("pdffonts is required for the font-embedding audit")
    return candidates[0]


def inspect_pdf(pdffonts: Path, pdf: Path) -> tuple[str, list[tuple[str, str]]]:
    completed = subprocess.run(
        [str(pdffonts), str(pdf)],
        check=True,
        capture_output=True,
        text=True,
    )
    lines = completed.stdout.splitlines()
    if len(lines) < 2:
        raise AssertionError(f"unexpected pdffonts output for {pdf}")
    header = lines[0]
    emb_start = header.index("emb")
    sub_start = header.index("sub")
    rows = []
    for line in lines[2:]:
        if not line.strip():
            continue
        font_name = line[: header.index("type")].strip()
        embedded = line[emb_start:sub_start].strip()
        rows.append((font_name, embedded))
    if not rows:
        raise AssertionError(f"no fonts found in {pdf}")
    if any(embedded != "yes" for _name, embedded in rows):
        raise AssertionError(f"non-embedded font found in {pdf}: {rows}")
    return completed.stdout.rstrip(), rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pdf",
        type=Path,
        default=PAPER / "output" / "pdf" / "kepler_paper_v1_2_1_submission.pdf",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PAPER / "font_embedding_audit.txt",
    )
    args = parser.parse_args()
    pdffonts = find_pdffonts()
    paths = [args.pdf, *sorted((PAPER / "figures").glob("fig*.pdf"))]
    sections = [
        "Project Kepler v1.2.1 font embedding audit",
        "pdffonts: Poppler pdffonts (bundled runtime)",
        "all_fonts_embedded: yes",
        "",
    ]
    total_fonts = 0
    for path in paths:
        output, rows = inspect_pdf(pdffonts, path)
        total_fonts += len(rows)
        sections.extend(
            [
                f"file: {path.relative_to(PAPER)}",
                f"sha256: {sha256(path)}",
                f"font_records: {len(rows)}",
                output,
                "",
            ]
        )
    sections.insert(3, f"audited_font_records: {total_fonts}")
    atomic_write_text(args.output, "\n".join(sections).rstrip() + "\n")
    print(args.output)


if __name__ == "__main__":
    main()
