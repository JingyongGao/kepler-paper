from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from common import ROOT


SCRIPTS = (
    "build_artifact_index.py",
    "build_figures.py",
    "build_tables.py",
    "build_manuscript.py",
    "build_reproducibility.py",
    "build_package_manifest.py",
)


def main() -> None:
    directory = Path(__file__).resolve().parent
    for name in SCRIPTS:
        subprocess.run(
            [sys.executable, str(directory / name)],
            cwd=ROOT,
            check=True,
        )
    print("paper package rebuilt from frozen JSON; optimizer steps: 0")


if __name__ == "__main__":
    main()

