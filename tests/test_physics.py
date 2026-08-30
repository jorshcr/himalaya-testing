import math

import numpy as np

from himalaya.physics import (
    force_weighted_support_anchor,
    slope_conditioned_com_height,
    terrain_aligned_zmp,
    terrain_descriptor,
    zmp_reward,
)


def test_force_weighted_four_contact_anchor() -> None:
    positions = np.asarray(
        [[-1.0, 1.0, 0.0], [-1.0, -1.0, 0.0], [1.0, 1.0, 0.0], [1.0, -1.0, 0.0]]
    )
    anchor = force_weighted_support_anchor(positions, np.ones(4))
    np.testing.assert_allclose(anchor, np.zeros(3), atol=1e-12)


def test_anchor_handles_zero_and_nonfinite_forces() -> None:
    positions = np.asarray([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    anchor = force_weighted_support_anchor(positions, np.asarray([np.nan, 2.0]))
    assert np.all(np.isfinite(anchor))
    assert anchor[0] > 1.99


def test_anchor_ignores_inactive_contact_even_with_large_sensor_force() -> None:
    positions = np.asarray([[0.0, 0.0, 0.0], [100.0, 0.0, 0.0]])
    anchor = force_weighted_support_anchor(
        positions, np.asarray([10.0, 10_000.0]), np.asarray([True, False])
    )
    np.testing.assert_allclose(anchor, positions[0], atol=1e-12)


def test_terrain_zmp_intersects_inclined_plane() -> None:
    slope = math.radians(30.0)
    normal = np.asarray([-math.sin(slope), 0.0, math.cos(slope)])
    support = np.asarray([0.0, 0.0, 0.0])
    zmp = terrain_aligned_zmp(
        np.asarray([0.0, 0.0, 1.0]), np.zeros(3), support, normal
    )
    assert abs(float(np.dot(zmp - support, normal))) < 1e-10
    assert 0.0 < float(zmp_reward(zmp, support)) <= 1.0


def test_zmp_singular_denominator_is_finite() -> None:
    zmp = terrain_aligned_zmp(
        np.asarray([0.0, 0.0, 1.0]),
        np.asarray([0.0, 0.0, -9.81]),
        np.zeros(3),
        np.asarray([0.0, 0.0, 1.0]),
    )
    assert np.all(np.isfinite(zmp))


def test_stage_two_descriptor_and_com_target() -> None:
    descriptor = terrain_descriptor(math.radians(15), 0.1, facing_uphill=True)
    assert descriptor.shape == (5,)
    np.testing.assert_allclose(descriptor[-2:], [1.0, 0.0])
    target = slope_conditioned_com_height(
        0.3, math.radians(30), maximum_slope_radians=math.radians(30)
    )
    assert target < 0.3
