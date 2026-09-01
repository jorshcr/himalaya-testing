import math

import jax.numpy as jp
import numpy as np

from himalaya.config import ExperimentConfig, WaveGaitConfig, to_playground_config
from himalaya.training import expected_restore_contract
from himalaya.wave_gait import (
    WAVE_GAIT_SEQUENCE,
    update_no_progress,
    wave_gait_gates,
    wave_reward_terms,
)


def test_wave_gait_order_is_lh_rf_rh_lf_and_one_limb_at_a_time() -> None:
    config = WaveGaitConfig()
    centers = 2.0 * math.pi * (np.arange(4) + 0.5) / 4.0
    active = []
    for phase in centers:
        gates, _, index, _ = wave_gait_gates(jp.asarray(phase), config)
        active.append(int(jp.argmax(gates)))
        assert int(index) == active[-1]
    assert WAVE_GAIT_SEQUENCE == (
        "left_hand",
        "right_foot",
        "right_hand",
        "left_foot",
    )
    assert active == [0, 1, 2, 3]
    for phase in np.linspace(0.0, 2.0 * math.pi, 65, endpoint=False):
        gates, _, _, _ = wave_gait_gates(jp.asarray(phase), config)
        assert int(jp.sum(gates > 1.0e-6)) <= 1


def _terms(*, forward: float, stance_speed: float):
    config = WaveGaitConfig()
    positions = jp.asarray(
        [
            [forward, 0.0, config.swing_clearance_m],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
        ]
    )
    velocities = jp.asarray(
        [
            [0.0, 0.0, 0.0],
            [stance_speed, 0.0, 0.0],
            [stance_speed, 0.0, 0.0],
            [stance_speed, 0.0, 0.0],
        ]
    )
    return wave_reward_terms(
        positions=positions,
        velocities=velocities,
        contacts=jp.asarray([True, True, True, True]),
        normal_forces=jp.asarray([100.0, 100.0, 100.0, 100.0]),
        previous_contact_positions=jp.zeros((4, 3)),
        previous_contacts=jp.asarray([False, True, True, True]),
        tangent=jp.asarray([1.0, 0.0, 0.0]),
        normal=jp.asarray([0.0, 0.0, 1.0]),
        # Late in the left-hand quarter: recontact guidance is active.
        phase_radians=jp.asarray(2.0 * math.pi * 0.225),
        config=config,
    )


def test_wave_rewards_prefer_forward_recontact_and_stable_stance() -> None:
    forward = _terms(forward=0.10, stance_speed=0.0)
    backward = _terms(forward=-0.10, stance_speed=0.0)
    slipping = _terms(forward=0.10, stance_speed=0.30)
    assert forward["wave_forward_placement"] > backward["wave_forward_placement"]
    assert forward["wave_recontact_ahead"] > 0.0
    assert backward["wave_recontact_ahead"] == 0.0
    assert forward["wave_backward_placement"] == 0.0
    assert backward["wave_backward_placement"] > 0.0
    assert forward["wave_stance_stability"] > slipping["wave_stance_stability"]
    assert forward["wave_stance_slip"] < slipping["wave_stance_slip"]


def test_bootstrap_and_phase_reward_annealing_are_configured() -> None:
    config = ExperimentConfig(stage="posture-adapter", slope_degrees=0.0)
    upstream = to_playground_config(config)
    assert config.wave_gait.command_range(0.0) == (0.08, 0.25)
    assert config.wave_gait.command_range(5.0) == (0.08, 0.25)
    assert upstream.lin_vel_x == [0.08, 0.25]
    assert upstream.push_config.enable is False
    assert upstream.noise_config.level == config.wave_gait.bootstrap_sensor_noise_level
    scales = [
        config.wave_gait.reward_scale(slope)
        for slope in config.wave_gait.curriculum_degrees
    ]
    assert scales == sorted(scales, reverse=True)
    assert scales[0] == 1.0
    assert scales[-1] == 0.3


def test_no_progress_truncates_after_configured_window() -> None:
    anchor, stalled, truncated = update_no_progress(
        jp.asarray(0.0),
        jp.asarray(0.0),
        jp.asarray(99, dtype=jp.int32),
        minimum_delta_m=0.04,
        maximum_stalled_steps=100,
    )
    assert anchor == 0.0
    assert stalled == 100
    assert bool(truncated)
    anchor, stalled, truncated = update_no_progress(
        jp.asarray(0.05),
        anchor,
        stalled,
        minimum_delta_m=0.04,
        maximum_stalled_steps=100,
    )
    assert anchor == 0.05
    assert stalled == 0
    assert not bool(truncated)


def test_wave_curriculum_requires_measurable_promoted_predecessor() -> None:
    assert expected_restore_contract(
        ExperimentConfig(stage="posture-adapter", slope_degrees=0.0)
    ) == ("balance-prior", 30.0)
    expected = {5.0: 0.0, 10.0: 5.0, 15.0: 10.0, 20.0: 15.0, 30.0: 20.0}
    for slope, predecessor in expected.items():
        assert expected_restore_contract(
            ExperimentConfig(stage="posture-adapter", slope_degrees=slope)
        ) == ("posture-adapter", predecessor)


def test_no_hard_contact_count_reward_or_contact_pattern_gate() -> None:
    names = vars(ExperimentConfig().stage2_reward)
    assert not any("contact_count" in name or "four_contact" in name for name in names)
