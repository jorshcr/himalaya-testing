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
