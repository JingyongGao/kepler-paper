from __future__ import annotations

import shutil
from pathlib import Path

from common import PAPER, atomic_write_text


PACKAGES = {
    "anonymized": {
        "author": "Anonymous Authors",
        "readme": (
            "Blinded conference submission source. Compile with:\n"
            "  tectonic manuscript.tex\n"
        ),
    },
    "arxiv": {
        "author": "Project Kepler Collaboration",
        "readme": (
            "arXiv source package. Compile with:\n"
            "  tectonic manuscript.tex\n"
        ),
    },
}


def copy_submission_assets(destination: Path) -> None:
    figures = destination / "figures"
    tables = destination / "tables"
    figures.mkdir(parents=True, exist_ok=True)
    tables.mkdir(parents=True, exist_ok=True)
    for source in sorted((PAPER / "figures").glob("fig*.pdf")):
        shutil.copy2(source, figures / source.name)
    for source in sorted((PAPER / "tables").glob("table*.tex")):
        shutil.copy2(source, tables / source.name)
    shutil.copy2(PAPER / "references.bib", destination / "references.bib")


def manuscript_for(author: str) -> str:
    source = (PAPER / "manuscript.tex").read_text(encoding="utf-8")
    expected = r"\author{Project Kepler Collaboration}"
    replacement = rf"\author{{{author}}}"
    if source.count(expected) != 1:
        raise AssertionError("unexpected manuscript author field")
    output = source.replace(expected, replacement)
    if author == "Anonymous Authors":
        output = output.replace(
            r"The final result is tagged \texttt{kepler-v3.6-confirm1}.",
            (
                "The confirmatory result and its frozen hashes are recorded in "
                "the supplementary artifact package."
            ),
        )
    return output


def main() -> None:
    for package_name, configuration in PACKAGES.items():
        destination = PAPER / package_name
        destination.mkdir(parents=True, exist_ok=True)
        copy_submission_assets(destination)
        atomic_write_text(
            destination / "manuscript.tex",
            manuscript_for(str(configuration["author"])),
        )
        atomic_write_text(destination / "README.txt", str(configuration["readme"]))
        print(destination)


if __name__ == "__main__":
    main()
