from __future__ import annotations

import csv
import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import Any, Iterable, Sequence


ROOT = Path(__file__).resolve().parents[2]
PAPER = ROOT / "paper"
ARTIFACTS = ROOT / "artifacts"
FIGURES = PAPER / "figures"
TABLES = PAPER / "tables"

V35_AUDIT = ARTIFACTS / "cue_distribution_audit.json"
V35_MANIFEST = ARTIFACTS / "cue_dose_manifest.json"
V35_RESULTS = ARTIFACTS / "cue_dose_results.json"
V36_AUDIT = ARTIFACTS / "confirmatory_distribution_audit.json"
V36_MANIFEST = ARTIFACTS / "cue_dose_confirmatory_manifest.json"
V36_RESULTS = ARTIFACTS / "cue_dose_confirmatory_results.json"
FREEZE_RECORD = ARTIFACTS / "v3_6_confirmatory_freeze_record.json"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def atomic_write_json(path: Path, value: Any) -> None:
    atomic_write_text(
        path,
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
    )


def write_csv(
    path: Path,
    rows: Sequence[dict[str, Any]],
    fieldnames: Sequence[str] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = list(rows[0]) if rows else []
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
            extrasaction="raise",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def git(*args: str) -> str:
    return subprocess.check_output(
        ["git", *args],
        cwd=ROOT,
        text=True,
    ).strip()


def require_frozen_source_hashes() -> dict[str, str]:
    expected = {
        "artifacts/cue_distribution_audit.json":
            "e6ce056320307dddd21691e475e7c604f7d02f880f89821f8c89545803fddc0a",
        "artifacts/cue_dose_manifest.json":
            "cfae2e4656c11cf37be387147e76b2f4f9585805320935893f6b27056d42957b",
        "artifacts/cue_dose_results.json":
            "74de64c64ae0ebf97be98611048b6a512fbbbcccbc60e37e1e72fafd8c146cf1",
        "artifacts/confirmatory_distribution_audit.json":
            "f67d261bc6e97fcb9fb7f25e12d7dfdae29fe0567acc930677507d5bf454c3c9",
        "artifacts/cue_dose_confirmatory_manifest.json":
            "04ca572e9367ab4216ec4ff1f0528e63f6a3f7b690ae960dc214fd0e4bfadb39",
        "artifacts/cue_dose_confirmatory_results.json":
            "5fd26ca916bcf73fd583d95b03fbb724f09536f88b22c848e8049c08e0c9b7a0",
    }
    for relative, digest in expected.items():
        actual = sha256(ROOT / relative)
        if actual != digest:
            raise AssertionError(f"frozen source changed: {relative}: {actual} != {digest}")
    return expected


def format_float(value: Any, digits: int = 6) -> str:
    if value is None:
        return "—"
    number = float(value)
    if number == 0:
        return "0"
    if abs(number) < 1e-4 or abs(number) >= 1e5:
        return f"{number:.3e}"
    return f"{number:.{digits}f}".rstrip("0").rstrip(".")


def markdown_table(headers: Sequence[str], rows: Iterable[Sequence[Any]]) -> str:
    materialized = [[str(item) for item in row] for row in rows]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in materialized)
    return "\n".join(lines) + "\n"


def latex_escape(value: Any) -> str:
    text = str(value)
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
    }
    return "".join(replacements.get(character, character) for character in text)


def latex_table(
    caption: str,
    label: str,
    headers: Sequence[str],
    rows: Iterable[Sequence[Any]],
) -> str:
    escaped_headers = [latex_escape(item) for item in headers]
    escaped_rows = [[latex_escape(item) for item in row] for row in rows]
    columns = "l" + "r" * (len(headers) - 1)
    lines = [
        r"\begin{table}[!htbp]",
        r"\centering",
        rf"\caption{{{latex_escape(caption)}}}",
        rf"\label{{{latex_escape(label)}}}",
        r"\resizebox{\linewidth}{!}{%",
        rf"\begin{{tabular}}{{{columns}}}",
        r"\toprule",
        " & ".join(escaped_headers) + r" \\",
        r"\midrule",
    ]
    lines.extend(" & ".join(row) + r" \\" for row in escaped_rows)
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}%",
            r"}",
            r"\end{table}",
            "",
        ]
    )
    return "\n".join(lines)


def checkpoint_seed_provenance(
    result: dict[str, Any],
    manifest: dict[str, Any],
) -> dict[str, Any]:
    cells = result["cells"]
    checkpoint_hashes = sorted(
        {
            metric["checkpoint_sha256"]
            for cell in cells
            for metric in cell["checkpoint_metrics"]
            if metric["checkpoint_sha256"] is not None
        }
    )
    if all("model_initialization_seed" in cell for cell in cells):
        model_seeds = sorted(
            {int(cell["model_initialization_seed"]) for cell in cells}
        )
    elif all("seed" in cell for cell in cells):
        model_seeds = sorted({int(cell["seed"]) for cell in cells})
    else:
        model_seeds = sorted(
            {int(run["model_initialization_seed"]) for run in manifest["runs"]}
        )
    exposures = sorted(
        {
            int(metric["exposed_pairs"])
            for cell in cells
            for metric in cell["checkpoint_metrics"]
        }
    )
    return {
        "cell_count": len(cells),
        "checkpoint_reference_count": sum(
            metric["checkpoint_sha256"] is not None
            for cell in cells
            for metric in cell["checkpoint_metrics"]
        ),
        "checkpoint_hash_count": len(checkpoint_hashes),
        "checkpoint_sha256": checkpoint_hashes,
        "evaluation_exposures": exposures,
        "model_or_pairing_seeds": model_seeds,
        "manifest_path": result["manifest"]["path"],
        "manifest_sha256": result["manifest"]["sha256"],
        "registered_runs": len(manifest["runs"]),
    }
