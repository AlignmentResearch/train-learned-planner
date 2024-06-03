from pathlib import Path

import numpy as np
import pytest

from cleanba.environments import BoxobanConfig, EnvConfig, EnvpoolBoxobanConfig, SokobanConfig


def sokoban_has_reset(tile_size: int, old_obs: np.ndarray, new_obs: np.ndarray) -> np.ndarray:
    """In any sokoban step, at most 3 tiles can change (player's previous tile, player's current tile, possibly a
    pushed box).

    Check whether the environment has reset by checking whether more than 3 tiles just changed.
    """
    assert old_obs.shape[-3] == 3, "is not *CHW"
    assert new_obs.shape[-3] == 3, "is not *CHW"

    pixel_has_changed = old_obs != new_obs
    *batch_shape, c, h, w = pixel_has_changed.shape

    tiled_pixel_has_changed = np.reshape(
        pixel_has_changed, (*batch_shape, c, h // tile_size, tile_size, w // tile_size, tile_size)
    )

    tile_has_changed = np.any(tiled_pixel_has_changed, axis=(-5, -3, -1))
    assert tile_has_changed.shape == (*batch_shape, h // tile_size, w // tile_size)

    tile_changed_count = np.sum(tile_has_changed, axis=(-2, -1))
    return tile_changed_count > 3


MAX_EPISODE_STEPS, NUM_ENVS, SEED = 20, 5, 1234


@pytest.mark.parametrize(
    "cfg, shape",
    [
        (
            BoxobanConfig(
                MAX_EPISODE_STEPS,
                NUM_ENVS,
                SEED,
                tinyworld_obs=True,
                asynchronous=False,
                min_episode_steps=MAX_EPISODE_STEPS * 3 // 4,
                cache_path=Path(__file__).parent,
            ),
            (10, 10),
        ),
        (
            SokobanConfig(
                max_episode_steps=MAX_EPISODE_STEPS,
                num_envs=NUM_ENVS,
                seed=SEED,
                tinyworld_obs=True,
                dim_room=(10, 10),
                num_boxes=2,  # Make sure it's not solved just by going in one direction
                asynchronous=False,
                min_episode_steps=MAX_EPISODE_STEPS * 3 // 4,
            ),
            (10, 10),
        ),
        (
            BoxobanConfig(
                MAX_EPISODE_STEPS,
                NUM_ENVS,
                SEED,
                tinyworld_obs=False,
                asynchronous=False,
                min_episode_steps=MAX_EPISODE_STEPS * 3 // 4,
                cache_path=Path(__file__).parent,
            ),
            (80, 80),
        ),
        (
            SokobanConfig(
                max_episode_steps=MAX_EPISODE_STEPS,
                num_envs=NUM_ENVS,
                seed=SEED,
                tinyworld_obs=False,
                dim_room=(10, 10),
                num_boxes=2,  # Make sure it's not solved just by going in one direction1
                asynchronous=False,
                min_episode_steps=MAX_EPISODE_STEPS * 3 // 4,
            ),
            (80, 80),
        ),
        pytest.param(
            EnvpoolBoxobanConfig(
                MAX_EPISODE_STEPS,
                NUM_ENVS,
                SEED,
                min_episode_steps=MAX_EPISODE_STEPS * 3 // 4,
                cache_path=Path(__file__).parent,
            ),
            (10, 10),
            marks=pytest.mark.envpool,
        ),
    ],
)
def test_environment_basics(cfg: EnvConfig, shape: tuple[int, int]):
    envs = cfg.make()
    assert envs.single_observation_space.shape == (3, *shape)
    assert envs.observation_space.shape == (NUM_ENVS, 3, *shape)

    envs.reset_async()
    next_obs, info = envs.reset_wait()
    assert next_obs.shape == (NUM_ENVS, 3, *shape), "jax.lax convs are NCHW but you sent NHWC"

    assert (action_shape := envs.action_space.shape) is not None
    for i in range(50):
        prev_obs = next_obs
        actions = np.zeros(action_shape, dtype=np.int64)
        envs.step_async(actions)
        next_obs, next_reward, terminated, truncated, info = envs.step_wait()

        assert next_obs.shape == (NUM_ENVS, 3, *shape)

        # The environment should terminate | truncate in the same steps as it changes. In practice we're not solving
        # environments so it should always truncate.
        tile_size = shape[0] // 10  # Assume env is 10x10 sokoban
        assert np.array_equal(truncated, sokoban_has_reset(tile_size, prev_obs, next_obs))
