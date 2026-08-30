import numpy as np

from himalaya.config import default_config
from himalaya.model import ALL_FOURS_CTRL, ALL_FOURS_QPOS, compile_model


def test_overlay_compiles_and_changes_only_declared_contract() -> None:
    model = compile_model(default_config())
    assert model.nu == 29
    assert model.nq == 36
    assert model.keyframe("four_contact_home").qpos.shape == (36,)
    np.testing.assert_allclose(
        model.keyframe("four_contact_home").qpos, ALL_FOURS_QPOS, atol=1e-9
    )
    np.testing.assert_allclose(
        model.keyframe("four_contact_home").ctrl, ALL_FOURS_CTRL, atol=1e-9
    )
    for side in ("left", "right"):
        np.testing.assert_allclose(model.pair(f"{side}_hand_floor").friction[:2], 0.9)
        np.testing.assert_allclose(model.pair(f"{side}_hand_floor").solref, [0.005, 1.0])
        np.testing.assert_allclose(model.pair(f"{side}_foot_floor").friction[:2], 1.0)
    assert model.geom("left_hand_collision").type == model.geom("right_hand_collision").type
    for name in ("pelvis", "torso", "head"):
        assert model.geom(f"himalaya_{name}_proxy").id >= 0


def test_positive_x_is_uphill_and_heightfield_datum_is_signed() -> None:
    import mujoco

    model = compile_model(default_config(slope_degrees=30.0))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    heights = []
    for x in (-1.0, 0.0, 1.0):
        distance = mujoco.mj_rayHfield(
            model,
            data,
            model.geom("floor").id,
            np.asarray([x, 0.0, 8.0]),
            np.asarray([0.0, 0.0, -1.0]),
        )
        heights.append(8.0 - distance)
    assert heights[0] < heights[1] < heights[2]
    assert abs(heights[1]) < 1.0e-3
