from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from common import PAPER, ROOT, atomic_write_json, load_json, sha256


def run_json(script: str, *args: str) -> dict:
    output = subprocess.check_output(
        [sys.executable, str(PAPER / "scripts" / script), *args],
        cwd=ROOT,
        text=True,
    )
    return json.loads(output)


def rebuild() -> None:
    subprocess.run(
        [sys.executable, str(PAPER / "scripts" / "build_all.py")],
        cwd=ROOT,
        check=True,
    )


def verify_hash_manifest() -> int:
    manifest = load_json(PAPER / "package_hashes.json")
    for item in manifest["files"]:
        path = PAPER / item["path"]
        if not path.exists():
            raise AssertionError(f"missing paper artifact: {item['path']}")
        actual = sha256(path)
        if actual != item["sha256"]:
            raise AssertionError(f"paper artifact hash mismatch: {item['path']}")
    return int(manifest["file_count"])


def output_counts() -> dict[str, int]:
    return {
        "figures_svg": len(list((PAPER / "figures").glob("fig*.svg"))),
        "figures_pdf": len(list((PAPER / "figures").glob("fig*.pdf"))),
        "figures_png": len(list((PAPER / "figures").glob("fig*.png"))),
        "figure_source_data": len(
            list((PAPER / "figures").glob("fig*.source.*"))
        ),
        "tables_markdown": len(list((PAPER / "tables").glob("table*.md"))),
        "tables_csv": len(list((PAPER / "tables").glob("table*.csv"))),
        "tables_latex": len(list((PAPER / "tables").glob("table*.tex"))),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rebuild", action="store_true")
    parser.add_argument("--with-tests", action="store_true")
    parser.add_argument("--write-report", action="store_true")
    parser.add_argument("--skip-checkpoint-bytes", action="store_true")
    args = parser.parse_args()
    if args.rebuild:
        rebuild()
    scientific = run_json("verify_results.py")
    frozen_args = (
        ("--skip-checkpoint-bytes",) if args.skip_checkpoint_bytes else ()
    )
    frozen = run_json("verify_frozen_artifacts.py", *frozen_args)
    tests = {"run": False, "passed": None, "summary": None}
    if args.with_tests:
        completed = subprocess.run(
            [sys.executable, "-m", "pytest"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        summary = completed.stdout.strip().splitlines()[-1]
        tests = {"run": True, "passed": True, "summary": summary}
    counts = output_counts()
    expected = {
        "figures_svg": 7,
        "figures_pdf": 7,
        "figures_png": 7,
        "figure_source_data": 8,
        "tables_markdown": 8,
        "tables_csv": 8,
        "tables_latex": 14,
    }
    if counts != expected:
        raise AssertionError((counts, expected))
    report = {
        "kind": "kepler_paper_reproduction_audit",
        "verified_date": "2026-07-18",
        "mode": "lightweight_no_training",
        "optimizer_steps": 0,
        "scientific_result_verification": scientific,
        "frozen_artifact_verification": frozen,
        "paper_output_counts": counts,
        "tests": tests,
        "passed": True,
    }
    if args.write_report:
        atomic_write_json(PAPER / "reproduction_audit.json", report)
        subprocess.run(
            [
                sys.executable,
                str(PAPER / "scripts" / "build_package_manifest.py"),
            ],
            cwd=ROOT,
            check=True,
        )
    report["package_files_hashed"] = verify_hash_manifest()
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
