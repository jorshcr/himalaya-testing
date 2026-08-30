"""Backend-neutral HumoSlope-inspired balance calculations."""

from __future__ import annotations

from typing import Any

import numpy as np


def _xp(*values: Any):
    """Use JAX when passed JAX arrays, otherwise NumPy for cheap tests."""

    if any(type(value).__module__.startswith(("jax", "jaxlib")) for value in values):
        import jax.numpy as jnp

        return jnp
    return np


def force_weighted_support_anchor(
    positions: Any, normal_forces: Any, *, epsilon: float = 1.0e-3
):
    """Return a smooth interior anchor for any number of support contacts."""

    xp = _xp(positions, normal_forces)
    positions = xp.asarray(positions)
    forces = xp.asarray(normal_forces)
    finite = xp.isfinite(forces)
    weights = xp.where(finite, xp.maximum(forces, 0.0), 0.0) + epsilon
    return xp.sum(positions * weights[:, None], axis=0) / xp.sum(weights)


def terrain_aligned_zmp(
    com_position: Any,
    com_acceleration: Any,
    support_anchor: Any,
    plane_normal: Any,
    *,
    denominator_epsilon: float = 1.0e-5,
):
    """Point-mass apparent-force intersection used by HumoSlope Stage I."""

    xp = _xp(com_position, com_acceleration, support_anchor, plane_normal)
    com = xp.asarray(com_position)
    acceleration = xp.asarray(com_acceleration)
    support = xp.asarray(support_anchor)
    normal = xp.asarray(plane_normal)
    normal = normal / xp.maximum(xp.linalg.norm(normal), denominator_epsilon)
    apparent = xp.asarray([0.0, 0.0, -9.81]) - acceleration
    denominator = xp.dot(apparent, normal)
    sign = xp.where(denominator < 0.0, -1.0, 1.0)
    safe_denominator = xp.where(
        xp.abs(denominator) < denominator_epsilon,
        sign * denominator_epsilon,
        denominator,
    )
    distance = xp.dot(support - com, normal)
    zmp = com + (distance / safe_denominator) * apparent
    return xp.where(xp.isfinite(zmp), zmp, support)


def zmp_reward(zmp: Any, support_anchor: Any, *, sigma: float = 0.12):
    xp = _xp(zmp, support_anchor)
    distance = xp.linalg.norm(xp.asarray(zmp) - xp.asarray(support_anchor))
    return xp.exp(-distance / sigma)


def terrain_descriptor(
    longitudinal_slope_radians: Any,
    bank_radians: Any,
    *,
    facing_uphill: Any,
):
    """Five-value training-only descriptor from HumoSlope Stage II."""

    xp = _xp(longitudinal_slope_radians, bank_radians, facing_uphill)
    slope = xp.asarray(longitudinal_slope_radians)
    uphill = xp.asarray(facing_uphill, dtype=float)
    return xp.asarray([slope, bank_radians, xp.abs(slope), uphill, 1.0 - uphill])


def slope_conditioned_com_height(
    nominal_height: Any,
    slope_radians: Any,
    *,
    maximum_slope_radians: float,
    uphill_offset: float = 0.0,
    downhill_offset: float = 0.0,
    facing_uphill: Any = True,
):
    xp = _xp(nominal_height, slope_radians, facing_uphill)
    intensity = xp.clip(xp.abs(slope_radians) / maximum_slope_radians, 0.0, 1.0)
    uphill = xp.asarray(facing_uphill, dtype=float)
    offset = uphill * uphill_offset + (1.0 - uphill) * downhill_offset
    return nominal_height * xp.cos(xp.abs(slope_radians)) + intensity * offset

