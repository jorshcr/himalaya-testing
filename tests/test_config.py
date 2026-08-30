from dataclasses import replace

import pytest

from himalaya.config import (
    MENAGERIE_REVISION,
    PLAYGROUND_REVISION,
    ExperimentConfig,
    default_config,
)


def test_canonical_upstream_and_microspikes() -> None:
    config = default_config()
    assert config.upstream.playground_revision == PLAYGROUND_REVISION
    assert config.upstream.menagerie_revision == MENAGERIE_REVISION
    assert config.contact.hand_sliding_friction == 0.9
    assert config.contact.foot_sliding_friction == 1.0
    assert config.contact.sensitivity_scale == 0.8


def test_only_reviewed_curriculum_is_accepted() -> None:
    assert ExperimentConfig(slope_degrees=30.0).slope_degrees == 30.0
    with pytest.raises(ValueError, match="slope must be one of"):
        ExperimentConfig(slope_degrees=35.0)


def test_actor_and_scientific_intent_are_stationary() -> None:
    config = default_config()
    payload = config.as_dict()
    text = repr(payload).lower()
    assert "uphill_progress" not in text
    assert "four_contact_reward" not in text
    assert config.stage == "balance-prior"
    assert config.reward.termination * 0.02 == -100.0


def test_drop_height_bounds_are_canonical_and_validated() -> None:
    config = default_config()
    assert config.reset.drop_height_min_m == 0.25
    assert config.reset.drop_height_max_m == 0.35
    with pytest.raises(ValueError, match="maximum drop height"):
        ExperimentConfig(
            reset=replace(
                config.reset, drop_height_min_m=0.35, drop_height_max_m=0.25
            )
        )


def test_canonical_stage_one_is_extensive_and_multi_slope() -> None:
    config = default_config()
    assert config.ppo.timesteps_per_stage == 200_000_000
    assert config.domain_randomization.enabled
    assert config.domain_randomization.slope_degrees == (0.0, 5.0, 10.0, 15.0, 30.0)
    assert config.domain_randomization.hand_friction_scale_range[0] == 0.75
    assert config.domain_randomization.push_magnitude_mps[1] == 0.75
