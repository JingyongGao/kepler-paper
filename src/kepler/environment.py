from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
from typing import Any, Mapping


LOCKED_MPS_ENV = {
    "PYTORCH_MPS_FAST_MATH": "0",
    "PYTORCH_ENABLE_MPS_FALLBACK": "0",
}

_ORIGINAL_MPS_ENV = {key: os.environ.get(key) for key in LOCKED_MPS_ENV}


def enforce_mps_environment() -> dict[str, str | None]:
    """Set the Phase-4 MPS lock before Kepler imports torch."""
    for key, value in LOCKED_MPS_ENV.items():
        os.environ[key] = value
    return dict(_ORIGINAL_MPS_ENV)


def canonical_hash(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_commit(workdir: str | Path | None = None) -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=workdir,
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except (FileNotFoundError, subprocess.CalledProcessError):
        return "unavailable:not-a-git-repository"


def environment_snapshot(
    config_payload: Mapping[str, Any],
    production_manifest: str | Path | None = None,
    pilot_manifest: str | Path | None = None,
    seeds: Mapping[str, Any] | None = None,
    workdir: str | Path | None = None,
) -> dict[str, Any]:
    # Environment locking happens in kepler.__init__, before this lazy import.
    import torch

    mps_environment = {
        key: value
        for key, value in sorted(os.environ.items())
        if "MPS" in key.upper() or key.startswith("PYTORCH_")
    }
    result: dict[str, Any] = {
        "macos_version": platform.mac_ver()[0] or "not-macos",
        "platform": platform.platform(),
        "python_version": sys.version,
        "pytorch_version": torch.__version__,
        "git_commit": git_commit(workdir),
        "config_hash": canonical_hash(config_payload),
        "mps_environment": mps_environment,
        "locked_mps_environment": dict(LOCKED_MPS_ENV),
        "original_mps_environment": dict(_ORIGINAL_MPS_ENV),
        "seeds": dict(seeds or {}),
    }
    if production_manifest is not None and Path(production_manifest).exists():
        result["production_manifest_hash"] = file_sha256(production_manifest)
    if pilot_manifest is not None and Path(pilot_manifest).exists():
        result["pilot_manifest_hash"] = file_sha256(pilot_manifest)
    return result

