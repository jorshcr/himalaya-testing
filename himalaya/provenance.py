"""Small, deterministic provenance records for every experiment."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import tarfile
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


def archive_checkpoint(output: Path, checkpoint: Path) -> Path:
    """Create a closed, portable archive for one numeric checkpoint."""
    output = output.resolve()
    checkpoint = checkpoint.resolve()
    artifacts = output / "artifacts"
    artifacts.mkdir(parents=True, exist_ok=True)
    archive = artifacts / f"checkpoint-{checkpoint.name}.tar.gz"
    with tarfile.open(archive, "w:gz") as bundle:
        bundle.add(checkpoint, arcname=checkpoint.relative_to(output).as_posix())
        manifest = output / "run_manifest.json"
        if manifest.is_file():
            bundle.add(manifest, arcname=manifest.relative_to(output).as_posix())
    return archive


def checkpoint_for_step(output: Path, step: int) -> Path:
    """Resolve Brax's zero-padded checkpoint directory for an integer step."""
    checkpoints = output.resolve() / "checkpoints"
    match = next(
        (
            item
            for item in checkpoints.iterdir()
            if item.is_dir() and item.name.isdigit() and int(item.name) == int(step)
        ),
        None,
    )
    if match is None:
        raise ValueError(f"checkpoint for step {step} is missing from {checkpoints}")
    return match


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
        "objective": (
            "soft phase-guided four-limb slope crawling"
            if config.stage == "posture-adapter"
            else "stationary four-contact slope balance"
        ),
        "humo_slope_inspired_not_reproduction": True,
        "wave_gait_sequence": (
            list(config.wave_gait.sequence)
            if config.stage == "posture-adapter"
            else None
        ),
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


def finalize_training_run(
    output: Path,
    config: ExperimentConfig,
    *,
    command: list[str],
    configured_timesteps: int,
    restore: str | None = None,
) -> Path:
    """Bind a completed run to its final checkpoint and portable local archive."""
    output = output.resolve()
    checkpoints = output / "checkpoints"
    candidates = sorted(
        (item for item in checkpoints.iterdir() if item.is_dir() and item.name.isdigit()),
        key=lambda item: int(item.name),
    )
    if not candidates:
        raise ValueError(f"completed training produced no checkpoints in {checkpoints}")
    checkpoint = candidates[-1]
    checkpoint_steps = int(checkpoint.name)
    if checkpoint_steps < configured_timesteps:
        raise ValueError(
            f"final checkpoint {checkpoint_steps} is below configured "
            f"timesteps {configured_timesteps}"
        )

    manifest = write_run_manifest(
        output,
        config,
        command=command,
        checkpoint=checkpoint,
        extra={
            "status": "completed",
            "restore": restore,
            "configured_timesteps": configured_timesteps,
            "checkpoint_steps": checkpoint_steps,
            "checkpoint_relative_path": checkpoint.relative_to(output).as_posix(),
        },
    )
    archive = archive_checkpoint(output, checkpoint)
    archive_sha256 = file_digest(archive)
    completion = {
        "schema_version": 1,
        "completed": True,
        "stage": config.stage,
        "slope_degrees": config.slope_degrees,
        "configured_timesteps": configured_timesteps,
        "checkpoint_steps": checkpoint_steps,
        "checkpoint": checkpoint.relative_to(output).as_posix(),
        "checkpoint_sha256": checkpoint_digest(checkpoint),
        "artifact": archive.relative_to(output).as_posix(),
        "artifact_sha256": archive_sha256,
    }
    path = output / "completion.json"
    path.write_text(json.dumps(completion, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path
