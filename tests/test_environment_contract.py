import jax
import jax.numpy as jp
import pytest

from himalaya.config import default_config
from himalaya.environment import FourContactBalanceEnv


@pytest.fixture(scope="module")
def env() -> FourContactBalanceEnv:
    return FourContactBalanceEnv(default_config())


def test_actor_stays_upstream_and_critic_has_reserved_descriptor(env) -> None:
    state = env.reset(jax.random.PRNGKey(0))
    assert env.action_size == 29
    assert state.obs["state"].shape == (103,)
    assert state.obs["privileged_state"].shape == (233,)
    # Stage I keeps the descriptor ABI but does not expose active terrain cues.
    assert jp.all(state.obs["privileged_state"][216:221] == 0.0)
    height = jp.dot(state.data.qpos[:3], env._ramp_normal)
    assert env._init_q[2] <= height <= (
        env._init_q[2] + env.experiment.reset.drop_height_max_m
    )
    assert state.info["reset_drop_height_m"] == height - env._init_q[2]


def test_stationary_step_has_no_contact_count_or_progress_rewards(env) -> None:
    state = env.reset(jax.random.PRNGKey(1))
    state = env.step(state, jp.zeros(29))
    reward_metrics = {name.removeprefix("reward/") for name in state.metrics if name.startswith("reward/")}
    assert "uphill_progress" not in reward_metrics
    assert "four_contact" not in reward_metrics
    assert "terrain_zmp" in reward_metrics
    assert "root_height" in reward_metrics
    assert "com_height" in reward_metrics
    assert "action_magnitude" in reward_metrics
    assert "pose_deviation" in reward_metrics
