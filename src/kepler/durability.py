from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Mapping

import torch

from .environment import file_sha256


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def atomic_write_bytes(path: str | Path, content: bytes) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp-{os.getpid()}")
    with temporary.open("wb") as handle:
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, output)
    _fsync_directory(output.parent)
    return output


def atomic_write_json(path: str | Path, payload: Any) -> Path:
    content = (
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    ).encode("utf-8")
    return atomic_write_bytes(path, content)


def transactional_append_jsonl(path: str | Path, payload: Mapping[str, Any]) -> Path:
    output = Path(path)
    existing = output.read_bytes() if output.exists() else b""
    line = (json.dumps(payload, sort_keys=True, ensure_ascii=False) + "\n").encode("utf-8")
    return atomic_write_bytes(output, existing + line)


def durable_torch_save(
    payload: dict[str, Any],
    path: str | Path,
    required_keys: set[str] | None = None,
) -> str:
    """Flush, validate, checksum, and atomically publish a checkpoint."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp-{os.getpid()}")
    with temporary.open("wb") as handle:
        torch.save(payload, handle)
        handle.flush()
        os.fsync(handle.fileno())

    validated = torch.load(temporary, map_location="cpu", weights_only=False)
    required = required_keys or set()
    missing = required - validated.keys()
    if missing:
        temporary.unlink(missing_ok=True)
        raise ValueError(f"checkpoint validation failed; missing keys={sorted(missing)}")
    del validated
    checksum = file_sha256(temporary)
    os.replace(temporary, output)
    _fsync_directory(output.parent)
    atomic_write_bytes(output.with_suffix(output.suffix + ".sha256"), f"{checksum}  {output.name}\n".encode())
    return checksum


def validate_checkpoint_checksum(path: str | Path) -> bool:
    checkpoint = Path(path)
    sidecar = checkpoint.with_suffix(checkpoint.suffix + ".sha256")
    if not sidecar.exists():
        return False
    expected = sidecar.read_text(encoding="utf-8").split()[0]
    return file_sha256(checkpoint) == expected

