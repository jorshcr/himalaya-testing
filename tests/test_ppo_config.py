from himalaya.config import default_config
from himalaya.training import ppo_config


def test_balance_ppo_horizon_is_explicit_and_longer_than_upstream_default() -> None:
    config = default_config()
    params = ppo_config(config, timesteps=1_000_000)
    assert params.discounting == config.ppo.discounting == 0.997
    assert params.unroll_length == config.ppo.unroll_length == 64
    assert params.batch_size == config.ppo.batch_size
    assert params.num_minibatches == config.ppo.num_minibatches
