"""One PPO training entry point with explicit two-stage restoration."""

from __future__ import annotations

import functools
import json
import math
from pathlib import Path
from typing import Any

from brax.training.agents.ppo import networks as ppo_networks
from brax.training.agents.ppo import train as ppo
from mujoco_playground import wrapper
from mujoco_playground.config import locomotion_params

from .config import ExperimentConfig
from .domain_randomization import make_domain_randomizer
from .environment import FourContactBalanceEnv
from .provenance import checkpoint_digest


def ppo_config(config: ExperimentConfig, *, timesteps: int | None = None):
    params = locomotion_params.brax_ppo_config("G1JoystickRoughTerrain")
    params.num_timesteps = int(timesteps or config.ppo.timesteps_per_stage)
    params.num_envs = int(config.ppo.num_envs)
    params.discounting = float(config.ppo.discounting)
    params.unroll_length = int(config.ppo.unroll_length)
    params.batch_size = int(config.ppo.batch_size)
    params.num_minibatches = int(config.ppo.num_minibatches)
    params.num_updates_per_batch = int(config.ppo.num_updates_per_batch)
    params.learning_rate = float(config.ppo.learning_rate)
    params.entropy_cost = float(config.ppo.entropy_cost)
    params.num_evals = (
        math.ceil(params.num_timesteps / config.ppo.checkpoint_interval_steps) + 1
    )
    return params


def train(
    config: ExperimentConfig,
    output: Path,
    *,
    restore: Path | None = None,
    timesteps: int | None = None,
    num_envs: int | None = None,
    allow_unpromoted_restore: bool = False,
    progress_fn=lambda *_: None,
):
    expected_restore_slope: float | None = None
    if config.stage == "posture-adapter":
        expected_restore_slope = 30.0
    elif config.slope_degrees > 0:
        index = config.curriculum_degrees.index(config.slope_degrees)
        expected_restore_slope = config.curriculum_degrees[index - 1]
    if expected_restore_slope is None and restore is not None:
        raise ValueError("level balance-prior must start without a restore")
    if expected_restore_slope is not None and restore is None:
        raise ValueError(
            f"{config.stage} at {config.slope_degrees:g} degrees requires a "
            f"promoted {expected_restore_slope:g}-degree checkpoint"
        )
    if allow_unpromoted_restore and config.stage != "posture-adapter":
        raise ValueError("unpromoted restores are only allowed for posture-adapter")
    if restore is not None and not allow_unpromoted_restore:
        validate_restore(
            restore,
            expected_stage="balance-prior",
            expected_slope=expected_restore_slope,
        )
    env = FourContactBalanceEnv(config)
    params = ppo_config(config, timesteps=timesteps)
    if num_envs is not None:
        params.num_envs = int(num_envs)
    smoke_run = params.num_timesteps < 100_000
    if smoke_run:
        # A real update with a small, internally consistent CPU smoke batch.
        params.episode_length = 32
        params.unroll_length = 2
        params.batch_size = int(params.num_envs)
        params.num_minibatches = 1
        params.num_updates_per_batch = 1
        params.num_evals = 1
    network_factory = functools.partial(
        ppo_networks.make_ppo_networks, **params.network_factory
    )
    kwargs: dict[str, Any] = dict(params)
    del kwargs["network_factory"]
    output.mkdir(parents=True, exist_ok=True)
    randomization_fn = None
    if config.domain_randomization.enabled:
        # Stage I spans equivalent inclines on the reference plane. Stage II
        # trains on the actual configured 30-degree terrain, so its dynamics
        # randomizer keeps gravity world-vertical and varies the other fields.
        slopes = None if config.stage == "balance-prior" else (0.0,)
        randomization_fn = make_domain_randomizer(
            env, config, int(params.num_envs), slope_degrees=slopes
        )
    return ppo.train(
        environment=env,
        eval_env=env,
        network_factory=network_factory,
        # Finite-difference CoM acceleration lives in info.  Cached-data-only
        # autoreset leaves that history stale and creates a spurious ZMP spike.
        wrap_env_fn=functools.partial(
            wrapper.wrap_for_brax_training,
            full_reset=True,
            randomization_fn=randomization_fn,
        ),
        save_checkpoint_path=str(output / "checkpoints"),
        restore_checkpoint_path=str(restore) if restore else None,
        # HumoSlope Stage II keeps the actor and resets the expanded critic.
        restore_value_fn=config.stage != "posture-adapter",
        num_eval_envs=min(int(params.num_envs), 4) if smoke_run else 128,
        seed=config.ppo.seed,
        progress_fn=progress_fn,
        **kwargs,
    )


def validate_restore(
    checkpoint: Path,
    *,
    expected_stage: str,
    expected_slope: float | None,
) -> None:
    checkpoint = checkpoint.resolve()
    manifest = next(
        (
            parent / "stage_result.json"
            for parent in (checkpoint.parent, *checkpoint.parents)
            if (parent / "stage_result.json").is_file()
        ),
        None,
    )
    if manifest is None:
        raise ValueError("restore checkpoint is missing stage_result.json")
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    valid = (
        payload.get("promotion_passed") is True
        and payload.get("stage") == expected_stage
        and float(payload.get("slope_degrees", -999)) == expected_slope
        and Path(payload.get("checkpoint", "")).resolve() == checkpoint
        and payload.get("checkpoint_sha256") == checkpoint_digest(checkpoint)
    )
    if not valid:
        raise ValueError("restore checkpoint does not satisfy the promoted-stage contract")


def load_policy(config: ExperimentConfig, checkpoint: Path):
    from brax.training.agents.ppo import checkpoint as ppo_checkpoint

    env = FourContactBalanceEnv(config)
    params = ppo_config(config, timesteps=1)
    networks = ppo_networks.make_ppo_networks(
        env.observation_size, env.action_size, **params.network_factory
    )
    restored = ppo_checkpoint.load(checkpoint.resolve())
    return ppo_networks.make_inference_fn(networks)(restored, deterministic=True)
