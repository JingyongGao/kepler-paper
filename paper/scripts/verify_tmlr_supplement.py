from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "SUPPLEMENT_MANIFEST.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    for record in manifest["files"]:
        path = ROOT / record["path"]
        if not path.is_file():
            raise AssertionError(f"missing supplemental file: {record['path']}")
        if path.stat().st_size != record["bytes"]:
            raise AssertionError(f"size mismatch: {record['path']}")
        if sha256(path) != record["sha256"]:
            raise AssertionError(f"SHA-256 mismatch: {record['path']}")
    print(
        json.dumps(
            {
                "mode": "anonymous_supplement_hash_verification",
                "files_verified": len(manifest["files"]),
                "optimizer_steps": 0,
                "passed": True,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
