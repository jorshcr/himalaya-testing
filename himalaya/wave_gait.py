"""Soft phase guidance for the canonical four-limb crawl sequence."""

from __future__ import annotations

import jax
import jax.numpy as jp

from .config import WaveGaitConfig


WAVE_GAIT_SEQUENCE = (
    "left_hand",
    "right_foot",
    "right_hand",
    "left_foot",
)


def wave_gait_gates(
    phase_radians: jax.Array,
    config: WaveGaitConfig,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Return soft swing/landing gates and the ordered active limb index."""

    cycle = jp.mod(phase_radians, 2.0 * jp.pi) / (2.0 * jp.pi)
    centers = (jp.arange(4, dtype=cycle.dtype) + 0.5) / 4.0
    distance = jp.abs(jp.mod(cycle - centers + 0.5, 1.0) - 0.5)
    half_window = config.swing_window_fraction / 2.0
    swing = jp.where(
        distance < half_window,
        0.5 * (1.0 + jp.cos(jp.pi * distance / half_window)),
        0.0,
    )
    active_index = jp.floor(cycle * 4.0).astype(jp.int32)
    local_phase = jp.mod(cycle * 4.0, 1.0)
    landing_ramp = jp.clip(
        (local_phase - config.landing_start_fraction)
        / (1.0 - config.landing_start_fraction),
        0.0,
        1.0,
    )
    landing = jax.nn.one_hot(active_index, 4) * jp.square(landing_ramp)
    return swing, landing, active_index, local_phase


def wave_reward_terms(
    *,
    positions: jax.Array,
    velocities: jax.Array,
    contacts: jax.Array,
    normal_forces: jax.Array,
    previous_contact_positions: jax.Array,
    previous_contacts: jax.Array,
    tangent: jax.Array,
    normal: jax.Array,
    phase_radians: jax.Array,
    config: WaveGaitConfig,
) -> dict[str, jax.Array]:
    """Compute soft gait rewards without imposing a contact-count contract."""

    swing, landing, active_index, local_phase = wave_gait_gates(
        phase_radians, config
    )
    epsilon = 1.0e-6
    swing_total = jp.sum(swing) + epsilon
    clearance = positions @ normal
    clearance_score = jp.exp(
        -jp.square(
            (clearance - config.swing_clearance_m)
            / config.swing_clearance_sigma_m
        )
    )
    forward_delta = (positions - previous_contact_positions) @ tangent
    placement_score = jp.exp(
        -jp.square(
            (forward_delta - config.forward_placement_target_m)
            / config.forward_placement_sigma_m
        )
    )
    backward = jp.square(
        jp.maximum(-forward_delta, 0.0)
        / config.forward_placement_target_m
    )

    first_contact = contacts & ~previous_contacts
    recontact_advance = jp.clip(
        (forward_delta - config.minimum_recontact_advance_m)
        / (
            config.forward_placement_target_m
            - config.minimum_recontact_advance_m
            + epsilon
        ),
        0.0,
        1.0,
    )
    recontact_ahead = jp.sum(
        landing * first_contact.astype(jp.float32) * recontact_advance
    )

    tangent_velocity = velocities - (velocities @ normal)[:, None] * normal
    tangent_speed = jp.linalg.norm(tangent_velocity, axis=-1)
    stance = jp.clip(1.0 - swing, 0.0, 1.0)
    # Force smoothing makes stance support a continuous preference rather than
    # a mandatory contact pattern or exact contact-count reward.
    loaded = normal_forces / (normal_forces + 25.0)
    stance_total = jp.sum(stance) + epsilon
    stance_stability = jp.sum(
        stance
        * loaded
        * jp.exp(-jp.square(tangent_speed / config.stance_velocity_sigma_mps))
    ) / stance_total
    stance_slip = jp.sum(
        stance
        * loaded
        * jp.square(tangent_speed / config.stance_velocity_sigma_mps)
    ) / stance_total

    landing_progress = jp.clip(
        (local_phase - config.landing_start_fraction)
        / (1.0 - config.landing_start_fraction),
        0.0,
        1.0,
    )
    lift_clearance_deficit = jp.clip(
        (config.swing_clearance_m - clearance)
        / config.swing_clearance_m,
        0.0,
        1.0,
    )
    missed_lift = jp.sum(
        swing * (1.0 - landing_progress) * lift_clearance_deficit
    ) / swing_total
    missed_landing = jp.sum(
        landing * (1.0 - contacts.astype(jp.float32))
    )

    return {
        "wave_swing_clearance": jp.sum(swing * clearance_score) / swing_total,
        "wave_forward_placement": jp.sum(swing * placement_score) / swing_total,
        "wave_recontact_ahead": recontact_ahead,
        "wave_stance_stability": stance_stability,
        "wave_backward_placement": jp.sum(swing * backward) / swing_total,
        "wave_missed_swing_window": missed_lift + missed_landing,
        "wave_stance_slip": stance_slip,
        "wave_active_limb_index": active_index.astype(jp.float32),
    }


def update_no_progress(
    progress_m: jax.Array,
    anchor_m: jax.Array,
    stalled_steps: jax.Array,
    *,
    minimum_delta_m: float,
    maximum_stalled_steps: int,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Advance the rolling progress anchor and return a soft time truncation."""

    advanced = progress_m >= anchor_m + minimum_delta_m
    next_anchor = jp.where(advanced, progress_m, anchor_m)
    next_steps = jp.where(advanced, 0, stalled_steps + 1).astype(jp.int32)
    truncated = next_steps >= maximum_stalled_steps
    return next_anchor, next_steps, truncated
