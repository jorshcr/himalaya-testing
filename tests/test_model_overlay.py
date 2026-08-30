import numpy as np

from himalaya.config import default_config
from himalaya.model import ALL_FOURS_QPOS, compile_model


def test_overlay_compiles_and_changes_only_declared_contract() -> None:
    model = compile_model(default_config())
    assert model.nu == 29
    assert model.nq == 36
    assert model.keyframe("four_contact_home").qpos.shape == (36,)
    np.testing.assert_allclose(
        model.keyframe("four_contact_home").qpos, ALL_FOURS_QPOS, atol=1e-9
    )
    for side in ("left", "right"):
        np.testing.assert_allclose(model.pair(f"{side}_hand_floor").friction[:2], 0.9)
        np.testing.assert_allclose(model.pair(f"{side}_foot_floor").friction[:2], 1.0)
    assert model.geom("left_hand_collision").type == model.geom("right_hand_collision").type
    for name in ("pelvis", "torso", "head"):
        assert model.geom(f"himalaya_{name}_proxy").id >= 0

