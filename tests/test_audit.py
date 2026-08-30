from himalaya.audit import audit_model
from himalaya.config import default_config


def test_30_degree_signed_clearance_and_support_gate() -> None:
    report = audit_model(default_config(slope_degrees=30.0), settle_seconds=0.25)
    assert report.passed
    assert report.minimum_hand_clearance_m >= -0.001
    assert report.maximum_hand_penetration_m <= 0.001
    assert set(report.support_contacts) == {
        "left_hand", "right_hand", "left_foot", "right_foot"
    }


def test_level_reset_is_load_bearing_for_full_acceptance_horizon() -> None:
    report = audit_model(default_config(), settle_seconds=20.0)
    assert report.passed
    assert report.settling_passed
    assert report.minimum_root_height_m >= 0.24
    assert report.prohibited_contacts == ()
