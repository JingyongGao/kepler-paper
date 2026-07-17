from __future__ import annotations

from common import PAPER, atomic_write_json, sha256


EXCLUDED_NAMES = {"package_hashes.json", ".DS_Store"}


def role(path: str) -> str:
    if path.startswith("figures/"):
        return "publication_figure_or_source_data"
    if path.startswith("tables/"):
        return "publication_table"
    if path.startswith("scripts/"):
        return "deterministic_builder_or_verifier"
    if path.startswith("manuscript."):
        return "manuscript"
    return "reproducibility_or_index"


def main() -> None:
    files = []
    for path in sorted(PAPER.rglob("*")):
        if not path.is_file():
            continue
        if path.name in EXCLUDED_NAMES or "__pycache__" in path.parts:
            continue
        relative = str(path.relative_to(PAPER))
        files.append(
            {
                "path": relative,
                "sha256": sha256(path),
                "bytes": path.stat().st_size,
                "role": role(relative),
            }
        )
    manifest = {
        "kind": "kepler_paper_package_hash_manifest",
        "source_tag": "kepler-v3.6-confirm1",
        "source_commit": "f9fc22d85cd52863087afbb3f924466a63b30ba2",
        "file_count": len(files),
        "files": files,
    }
    atomic_write_json(PAPER / "package_hashes.json", manifest)
    print(PAPER / "package_hashes.json")


if __name__ == "__main__":
    main()

