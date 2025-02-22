from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from cleanba.config import sokoban_drc33_59
from cleanba.env_trivial import MockSokobanEnv, MockSokobanEnvConfig
from cleanba.environments import BoxobanConfig, CraftaxEnvConfig, EnvConfig, EnvpoolBoxobanConfig, SokobanConfig


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
        (
            MockSokobanEnvConfig(
                max_episode_steps=MAX_EPISODE_STEPS,
                num_envs=NUM_ENVS,
                seed=SEED,
                min_episode_steps=MAX_EPISODE_STEPS * 3 // 4,
            ),
            (10, 10),
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

        tile_size = shape[0] // 10  # Assume env is 10x10 sokoban
        if isinstance(cfg, MockSokobanEnvConfig):
            done = terminated | truncated
            assert np.array_equal(done, sokoban_has_reset(tile_size, prev_obs, next_obs))
        else:
            # The environment should terminate | truncate in the same steps as it changes. In practice we're not solving
            # environments so it should always truncate.
            assert np.array_equal(truncated, sokoban_has_reset(tile_size, prev_obs, next_obs))


def test_craftax_environment_basics():
    cfg = CraftaxEnvConfig(max_episode_steps=20, num_envs=2, obs_flat=False)
    envs = cfg.make()
    envs.reset_async()
    next_obs, info = envs.reset_wait()

    assert (action_shape := envs.action_space.shape) is not None
    for i in range(50):
        actions = np.zeros(action_shape, dtype=np.int64)
        envs.step_async(actions)
        envs.step_wait()


@pytest.mark.parametrize("gamma", [1.0, 0.9])
def test_mock_sokoban_returns(gamma: float, num_envs: int = 7):
    max_episode_steps = 10
    envs = MockSokobanEnvConfig(
        max_episode_steps=max_episode_steps, num_envs=num_envs, seed=1234, min_episode_steps=8, gamma=gamma
    ).make()

    num_timesteps = 30
    values = np.zeros((num_timesteps + 1, num_envs))
    rewards = np.zeros((num_timesteps, num_envs))
    dones = np.zeros((num_timesteps + 1, num_envs), dtype=np.bool_)
    truncateds = np.zeros((num_timesteps, num_envs), dtype=np.bool_)
    terminateds = np.zeros((num_timesteps, num_envs), dtype=np.bool_)
    obs, _ = envs.reset()
    values[0] = MockSokobanEnv.compute_return(obs)
    dones[0, :] = True

    for t in range(num_timesteps):
        obs, reward, terminated, truncated, _ = envs.step(np.zeros(num_envs, dtype=np.int64))
        values[t + 1] = MockSokobanEnv.compute_return(obs)
        rewards[t] = reward
        dones[t + 1] = terminated | truncated
        truncateds[t] = truncated
        terminateds[t] = terminated

    discount = (~np.array(dones[1:])) * gamma
    td_errors = np.zeros((num_timesteps, num_envs))
    for t in reversed(range(num_timesteps)):
        td_errors[t] = rewards[t] + discount[t] * values[t + 1] - values[t]

    assert np.allclose(td_errors * (~truncateds), 0.0, atol=1e-6)
    if gamma == 1.0:
        assert np.all(values[dones] == max_episode_steps)


@pytest.mark.parametrize(
    "cfg",
    [
        BoxobanConfig(
            max_episode_steps=10,
            num_envs=5,
            seed=1234,
            tinyworld_obs=True,
            asynchronous=False,
            min_episode_steps=8,
            cache_path=Path(__file__).parent,
        ),
        pytest.param(
            EnvpoolBoxobanConfig(
                max_episode_steps=10,
                num_envs=5,
                seed=1234,
                min_episode_steps=8,
                cache_path=Path(__file__).parent,
            ),
            marks=pytest.mark.envpool,
        ),
    ],
)
@pytest.mark.parametrize("nn_without_noop", [True, False])
def test_loading_network_without_noop_action(cfg: EnvConfig, nn_without_noop: bool):
    assert isinstance(cfg, BoxobanConfig) or isinstance(cfg, EnvpoolBoxobanConfig)
    cfg.nn_without_noop = nn_without_noop
    envs = cfg.make()

    envs.reset_async()
    next_obs, info = envs.reset_wait()
    assert next_obs.shape == (cfg.num_envs, 3, 10, 10), "jax.lax convs are NCHW but you sent NHWC"

    args = sokoban_drc33_59()
    key = jax.random.PRNGKey(42)
    key, agent_params_subkey, carry_key = jax.random.split(key, 3)
    policy, _, agent_params = args.net.init_params(envs, agent_params_subkey)
    assert agent_params["params"]["actor_params"]["Output"]["kernel"].shape[1] == 4 + (
        not nn_without_noop
    ), "NOOP action not set correctly"
    carry = policy.apply(agent_params, carry_key, next_obs.shape, method=policy.initialize_carry)
    episode_starts_no = jnp.zeros(cfg.num_envs, dtype=jnp.bool_)

    assert envs.action_space.shape is not None
    # actions = np.zeros(action_shape, dtype=np.int64)
    carry, actions, _, key = policy.apply(agent_params, carry, next_obs, episode_starts_no, key, method=policy.get_action)
    actions = np.asarray(actions)
    envs.step_async(actions)
    next_obs, next_reward, terminated, truncated, info = envs.step_wait()

    assert next_obs.shape == (cfg.num_envs, 3, 10, 10)
