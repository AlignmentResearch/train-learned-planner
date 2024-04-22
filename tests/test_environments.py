import numpy as np
import pytest

from cleanba.environments import BoxobanConfig, EnvConfig, EnvpoolBoxobanConfig, SokobanConfig

MAX_EPISODE_STEPS, NUM_ENVS, SEED = 20, 5, 1234


@pytest.mark.parametrize(
    "cfg",
    [
        BoxobanConfig(
            MAX_EPISODE_STEPS,
            NUM_ENVS,
            SEED,
            tinyworld_obs=True,
            asynchronous=False,
            min_episode_steps=MAX_EPISODE_STEPS * 3 // 4,
        ),
        SokobanConfig(
            max_episode_steps=MAX_EPISODE_STEPS,
            num_envs=NUM_ENVS,
            seed=SEED,
            tinyworld_obs=True,
            dim_room=(10, 10),
            num_boxes=1,
            asynchronous=False,
            min_episode_steps=MAX_EPISODE_STEPS * 3 // 4,
        ),
        EnvpoolBoxobanConfig(MAX_EPISODE_STEPS, NUM_ENVS, SEED, min_episode_steps=MAX_EPISODE_STEPS * 3 // 4),
    ],
)
def test_environment_basics(cfg: EnvConfig):
    envs = cfg.make()
    assert envs.single_observation_space.shape == (3, 10, 10)
    assert envs.observation_space.shape == (NUM_ENVS, 3, 10, 10)

    envs.reset_async()
    next_obs, info = envs.reset_wait()
    assert next_obs.shape == (NUM_ENVS, 3, 10, 10), "jax.lax convs are NCHW but you sent NHWC"

    for i in range(40):
        actions = np.stack([envs.single_action_space.sample() for _ in range(envs.num_envs)])
        envs.step_async(actions)
        next_obs, next_reward, terminated, truncated, info = envs.step_wait()

        assert next_obs.shape == (NUM_ENVS, 3, 10, 10)
