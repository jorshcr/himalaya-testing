import jax
import jax.numpy as jp
import pytest

from himalaya.config import default_config
from himalaya.environment import FourContactBalanceEnv


@pytest.fixture(scope="module")
def env() -> FourContactBalanceEnv:
    return FourContactBalanceEnv(default_config())


@pytest.fixture(scope="module")
def stage_two_env() -> FourContactBalanceEnv:
    return FourContactBalanceEnv(default_config(stage="posture-adapter"))


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


def test_stationary_step_disables_locomotion_and_contact_count_rewards(env) -> None:
    state = env.reset(jax.random.PRNGKey(1))
    reset_metric_keys = set(state.metrics)
    state = env.step(state, jp.zeros(29))
    assert set(state.metrics) == reset_metric_keys
    assert "validation/zmp_deviation_m" in state.metrics
    reward_metrics = {
        name.removeprefix("reward/")
        for name in state.metrics
        if name.startswith("reward/")
    }
    # The metrics pytree is fixed across both stages for JAX scan stability,
    # but the canonical Stage-I configuration gives every locomotion term a
    # zero effective weight.
    assert env._config.reward_config.scales.uphill_progress == 0.0
    assert env._config.reward_config.scales.wave_swing_clearance == 0.0
    assert "four_contact" not in reward_metrics
    assert "terrain_zmp" in reward_metrics
    assert "root_height" in reward_metrics
    assert "com_height" in reward_metrics
    assert "action_magnitude" in reward_metrics
    assert "pose_deviation" in reward_metrics


def test_stage_two_step_traces_time_bound_locomotion_rewards(stage_two_env) -> None:
    state = stage_two_env.reset(jax.random.PRNGKey(2))
    actor_shape = state.obs["state"].shape
    critic_shape = state.obs["privileged_state"].shape
    initial_phase = state.obs["state"][-4:]
    assert actor_shape == (103,)
    assert critic_shape == (233,)
    assert jp.allclose(initial_phase, jp.asarray([1.0, 0.0, 0.0, 1.0]), atol=1e-6)
    assert state.info["reset_drop_height_m"] == 0.0
    state = stage_two_env.step(state, jp.zeros(29))
    assert state.obs["state"].shape == actor_shape
    assert state.obs["privileged_state"].shape == critic_shape
    assert state.info["phase"][0] > 0.0
    # Upstream constructs obs before advancing its clock, so the new phase is
    # intentionally visible to the actor on the following environment step.
    state = stage_two_env.step(state, jp.zeros(29))
    assert not jp.allclose(state.obs["state"][-4:], initial_phase)
    reward_metrics = {
        name.removeprefix("reward/")
        for name in state.metrics
        if name.startswith("reward/")
    }
    assert "tracking_forward_velocity" in reward_metrics
    assert "progress_deficit" in reward_metrics
    assert "stagnation" in reward_metrics
    assert "wave_swing_clearance" in reward_metrics
    assert "wave_forward_placement" in reward_metrics
    assert "wave_recontact_ahead" in reward_metrics
    assert "wave_stance_stability" in reward_metrics
    assert "wave_missed_swing_window" in reward_metrics
