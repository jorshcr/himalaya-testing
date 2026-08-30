"""Small, deterministic provenance records for every experiment."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
from typing import Any

from .config import ExperimentConfig


def tree_digest(root: Path) -> str:
    digest = hashlib.sha256()
    ignored = {
        ".git",
        ".mpl-cache",
        ".pytest_cache",
        ".test-tmp",
        ".test-tmp2",
        ".tools",
        ".venv",
        ".verify-venv",
        "__pycache__",
        "runs",
    }
    paths: list[Path] = []
    for current, directories, filenames in os.walk(root):
        directories[:] = sorted(name for name in directories if name not in ignored)
        current_path = Path(current)
        paths.extend(current_path / name for name in sorted(filenames))
    paths.sort()
    for path in paths:
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def checkpoint_digest(path: Path) -> str:
    if path.is_file():
        return file_digest(path)
    digest = hashlib.sha256()
    files = sorted(item for item in path.rglob("*") if item.is_file())
    if not files:
        raise ValueError(f"checkpoint has no files: {path}")
    for item in files:
        digest.update(item.relative_to(path).as_posix().encode())
        digest.update(b"\0")
        digest.update(item.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def runtime_contract(config: ExperimentConfig) -> dict[str, Any]:
    packages = {}
    for name in ("brax", "jax", "mujoco", "playground"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = "missing"
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": packages,
        "playground_revision": config.upstream.playground_revision,
        "menagerie_revision": config.upstream.menagerie_revision,
        "implementation": config.implementation,
    }


def write_run_manifest(
    output: Path,
    config: ExperimentConfig,
    *,
    command: list[str],
    checkpoint: Path | None = None,
    extra: dict[str, Any] | None = None,
) -> Path:
    project_root = Path(__file__).resolve().parents[1]
    runtime = runtime_contract(config)
    runtime_sha = hashlib.sha256(
        json.dumps(runtime, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    payload: dict[str, Any] = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "objective": "stationary four-contact slope balance",
        "humo_slope_inspired_not_reproduction": True,
        "simulation_only": True,
        "microspike_coefficients_uncalibrated": True,
        "config": config.as_dict(),
        "command": command,
        "source_sha256": tree_digest(project_root),
        "runtime": runtime,
        "runtime_sha256": runtime_sha,
    }
    if checkpoint is not None:
        payload["checkpoint"] = str(checkpoint)
        payload["checkpoint_sha256"] = checkpoint_digest(checkpoint)
    if extra:
        payload.update(extra)
    output.mkdir(parents=True, exist_ok=True)
    path = output / "run_manifest.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path
