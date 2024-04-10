import numpy as np
import pytest

from cleanba.environments import BoxobanConfig, EnvConfig, EnvpoolBoxobanConfig, SokobanConfig

MAX_EPISODE_STEPS, NUM_ENVS, SEED = 20, 5, 1234


@pytest.mark.parametrize(
    "cfg",
    [
        BoxobanConfig(MAX_EPISODE_STEPS, NUM_ENVS, SEED, tinyworld_obs=True),
        SokobanConfig(MAX_EPISODE_STEPS, NUM_ENVS, SEED, tinyworld_obs=True, dim_room=(10, 10), num_boxes=1),
        # EnvpoolBoxobanConfig(MAX_EPISODE_STEPS, NUM_ENVS, SEED, min_episode_steps=MAX_EPISODE_STEPS * 3 // 4),
    ],
)
def test_environment_basics(cfg: EnvConfig):
    envs = cfg.make()

    envs.reset_async()
    next_obs, info = envs.reset_wait()
    # jax.lax convs are NCHW
    assert next_obs.shape == (NUM_ENVS, 3, 10, 10)

    for i in range(40):
        actions = np.stack([envs.single_action_space.sample() for _ in range(envs.num_envs)])
        envs.step_async(actions)
        next_obs, next_reward, terminated, truncated, info = envs.step_wait()

        assert next_obs.shape == (NUM_ENVS, 3, 10, 10)
