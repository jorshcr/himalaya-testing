import json
import tarfile

import pytest

from himalaya.config import ExperimentConfig
from himalaya.provenance import (
    checkpoint_digest,
    checkpoint_for_step,
    file_digest,
    finalize_training_run,
)


def test_finalize_training_run_writes_bound_local_artifact(tmp_path):
    checkpoint = tmp_path / "checkpoints" / "100761600"
    checkpoint.mkdir(parents=True)
    (checkpoint / "params").write_bytes(b"policy")

    completion_path = finalize_training_run(
        tmp_path,
        ExperimentConfig(),
        command=["himalaya", "train"],
        configured_timesteps=100_000_000,
    )

    completion = json.loads(completion_path.read_text(encoding="utf-8"))
    artifact = tmp_path / completion["artifact"]
    assert completion["completed"] is True
    assert completion["checkpoint_steps"] == 100_761_600
    assert completion["checkpoint_sha256"] == checkpoint_digest(checkpoint)
    assert completion["artifact_sha256"] == file_digest(artifact)
    with tarfile.open(artifact) as bundle:
        assert "checkpoints/100761600/params" in bundle.getnames()
        assert "run_manifest.json" in bundle.getnames()


def test_finalize_training_run_rejects_incomplete_checkpoint(tmp_path):
    checkpoint = tmp_path / "checkpoints" / "99"
    checkpoint.mkdir(parents=True)
    (checkpoint / "params").write_bytes(b"policy")
    with pytest.raises(ValueError, match="below configured timesteps"):
        finalize_training_run(
            tmp_path,
            ExperimentConfig(),
            command=["himalaya", "train"],
            configured_timesteps=100,
        )


def test_checkpoint_for_step_resolves_brax_zero_padding(tmp_path):
    checkpoint = tmp_path / "checkpoints" / "000025067520"
    checkpoint.mkdir(parents=True)
    assert checkpoint_for_step(tmp_path, 25_067_520) == checkpoint
