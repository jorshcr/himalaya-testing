"""Checkpoint-bound rendering with visible collision/contact evidence."""

from __future__ import annotations

from pathlib import Path
import shutil
from typing import Callable

import jax
import mediapy as media
import mujoco

from .config import ExperimentConfig
from .environment import FourContactBalanceEnv


def render_policy(
    config: ExperimentConfig,
    policy: Callable,
    output: Path,
    *,
    seconds: float = 20.0,
    seed: int | None = None,
) -> Path:
    env = FourContactBalanceEnv(config)
    reset = jax.jit(env.reset)
    step = jax.jit(env.step)
    act = jax.jit(policy)
    rng = jax.random.PRNGKey(seed if seed is not None else config.ppo.seed)
    rng, reset_rng = jax.random.split(rng)
    state = reset(reset_rng)
    trajectory = []
    for _ in range(min(config.episode_length, round(seconds / env.dt))):
        trajectory.append(state)
        rng, action_rng = jax.random.split(rng)
        state = step(state, act(state.obs, action_rng)[0])
        if float(state.done) > 0.5:
            trajectory.append(state)
            break
    option = mujoco.MjvOption()
    option.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = True
    option.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = True
    option.geomgroup[3] = True
    frames = env.render(
        trajectory[::2], height=540, width=960, camera="track", scene_option=option
    )
    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if not shutil.which("ffmpeg"):
        try:
            import imageio_ffmpeg

            media.set_ffmpeg(imageio_ffmpeg.get_ffmpeg_exe())
        except ImportError as exc:
            raise RuntimeError("ffmpeg is required for rendering") from exc
    media.write_video(output, frames, fps=1.0 / env.dt / 2, qp=18)
    return output

