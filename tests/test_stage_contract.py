from pathlib import Path

import pytest

from himalaya.config import default_config
from himalaya.training import train


def test_stage_two_requires_stage_one_restore(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="requires a promoted 30-degree checkpoint"):
        train(default_config(stage="posture-adapter"), tmp_path, timesteps=1, num_envs=1)


def test_stage_one_rejects_restore(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="level balance-prior must start"):
        train(
            default_config(stage="balance-prior"),
            tmp_path,
            restore=tmp_path / "checkpoint",
            timesteps=1,
            num_envs=1,
        )
