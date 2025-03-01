import dataclasses
from pathlib import Path
from typing import cast

import jax
import jax.numpy as jnp
import pytest

from cleanba.config import sokoban_drc33_59
from cleanba.env_trivial import MockSokobanEnv, MockSokobanEnvConfig
from cleanba.environments import (
    Box,
    BoxobanConfig,
    CraftaxEnvConfig,
    Discrete,
    EnvConfig,
    EnvpoolBoxobanConfig,
    EpisodeEvalWrapper,
    SokobanConfig,
)


def sokoban_has_reset(tile_size: int, old_obs: jnp.ndarray, new_obs: jnp.ndarray) -> jnp.ndarray:
    """In any sokoban step, at most 3 tiles can change (player's previous tile, player's current tile, possibly a
    pushed box).

    Check whether the environment has reset by checking whether more than 3 tiles just changed.
    """
    # Ensure inputs are jnp arrays
    old_obs = jnp.asarray(old_obs)
    new_obs = jnp.asarray(new_obs)

    assert old_obs.shape[-3] == 3, "is not *CHW"
    assert new_obs.shape[-3] == 3, "is not *CHW"

    pixel_has_changed = old_obs != new_obs
    batch_shape = pixel_has_changed.shape[:-3]
    c, h, w = pixel_has_changed.shape[-3:]

    tiled_pixel_has_changed = jnp.reshape(
        pixel_has_changed, (*batch_shape, c, h // tile_size, tile_size, w // tile_size, tile_size)
    )

    tile_has_changed = jnp.any(tiled_pixel_has_changed, axis=(-5, -3, -1))
    assert tile_has_changed.shape == (*batch_shape, h // tile_size, w // tile_size)

    tile_changed_count = jnp.sum(tile_has_changed, axis=(-2, -1))
    return tile_changed_count > 3


MAX_EPISODE_STEPS, NUM_ENVS = 20, 5


@pytest.mark.parametrize(
    "cfg, shape",
    [
        (
            BoxobanConfig(
                max_episode_steps=MAX_EPISODE_STEPS,
                num_envs=NUM_ENVS,
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
                max_episode_steps=MAX_EPISODE_STEPS,
                num_envs=NUM_ENVS,
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
                tinyworld_obs=False,
                dim_room=(10, 10),
                num_boxes=2,  # Make sure it's not solved just by going in one direction
                asynchronous=False,
                min_episode_steps=MAX_EPISODE_STEPS * 3 // 4,
            ),
            (80, 80),
        ),
        (
            MockSokobanEnvConfig(
                max_episode_steps=MAX_EPISODE_STEPS,
                num_envs=NUM_ENVS,
                min_episode_steps=MAX_EPISODE_STEPS * 3 // 4,
            ),
            (10, 10),
        ),
        pytest.param(
            EnvpoolBoxobanConfig(
                max_episode_steps=MAX_EPISODE_STEPS,
                num_envs=NUM_ENVS,
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

    obs_space = envs.single_observation_space
    if isinstance(obs_space, Box):
        assert obs_space.shape == (3, *shape)

    # Create a key for the environment
    key = jax.random.PRNGKey(0)
    reset_out = envs.reset_env(key)
    next_obs, state = reset_out.obs, reset_out.state
    assert next_obs.shape == (NUM_ENVS, 3, *shape), "jax.lax convs are NCHW but you sent NHWC"

    # Create zero actions
    actions = jnp.zeros_like(envs.example_action)

    # Run a few steps
    for i in range(10):
        prev_obs = next_obs
        key, step_key = jax.random.split(key)
        step_keys = jax.random.split(step_key, NUM_ENVS)

        # Step the environment
        next_obs, state, reward, terminated, truncated, info = jax.jit(envs.step_env)(step_keys, state, actions)

        assert next_obs.shape == (NUM_ENVS, 3, *shape)

        # Check for resets
        tile_size = shape[0] // 10  # Assume env is 10x10 sokoban
        if isinstance(cfg, MockSokobanEnvConfig):
            done = terminated | truncated
            assert jnp.array_equal(done, sokoban_has_reset(tile_size, prev_obs, next_obs))
        else:
            # The environment should terminate | truncate in the same steps as it changes. In practice we're not solving
            # environments so it should always truncate.
            assert jnp.array_equal(truncated, sokoban_has_reset(tile_size, prev_obs, next_obs))


def test_craftax_environment_basics():
    cfg = CraftaxEnvConfig(max_episode_steps=20, num_envs=2, obs_flat=False)
    env = cfg.make()
    envs = EpisodeEvalWrapper(env)

    # Create a key for the environment
    key = jax.random.PRNGKey(0)
    keys = jax.random.split(key, 2)  # 2 environments

    # Reset the environment
    reset_out = envs.reset_env(keys)
    _, state = reset_out.obs, reset_out.state

    # Get action shape from the environment
    action_space = envs.single_action_space
    if isinstance(action_space, Discrete):
        action_shape = (2,)  # 2 environments
    else:
        # For Box spaces with shape attribute
        action_shape = (2, *getattr(action_space, "shape", ()))

    # Run a few steps
    for i in range(5):
        key, step_key = jax.random.split(key)
        step_keys = jax.random.split(step_key, 2)  # 2 environments

        # Create zero actions
        actions = jnp.zeros(action_shape, dtype=jnp.int32)

        # Step the environment
        step_out = envs.step_env(step_keys, state, actions)
        _, state = step_out.obs, step_out.state


@pytest.mark.parametrize("gamma", [1.0, 0.9])
def test_mock_sokoban_returns(gamma: float, num_envs: int = 7):
    max_episode_steps = 10
    config = MockSokobanEnvConfig(max_episode_steps=max_episode_steps, num_envs=num_envs, min_episode_steps=8, gamma=gamma)
    envs = config.make()

    # Create arrays to store values
    num_timesteps = 30
    values = jnp.zeros((num_timesteps + 1, num_envs))
    rewards = jnp.zeros((num_timesteps, num_envs))
    dones = jnp.zeros((num_timesteps + 1, num_envs), dtype=jnp.bool_)
    truncateds = jnp.zeros((num_timesteps, num_envs), dtype=jnp.bool_)
    terminateds = jnp.zeros((num_timesteps, num_envs), dtype=jnp.bool_)

    # Create a key for the environment
    key = jax.random.PRNGKey(0)
    keys = jax.random.split(key, num_envs)

    # Reset the environment
    reset_out = envs.reset_env(keys)
    obs, state = reset_out.obs, reset_out.state

    # Store initial values
    values = values.at[0].set(MockSokobanEnv.compute_return(obs))
    dones = dones.at[0].set(True)

    # Run steps
    for t in range(num_timesteps):
        key, step_key = jax.random.split(key)
        step_keys = jax.random.split(step_key, num_envs)

        # Create zero actions
        actions = jnp.zeros((num_envs,), dtype=jnp.int32)

        # Step the environment
        step_out = envs.step_env(step_keys, state, actions)
        obs, state, reward, terminated, truncated, _ = (
            step_out.obs,
            step_out.state,
            step_out.reward,
            step_out.terminated,
            step_out.truncated,
            step_out.info,
        )

        # Store values
        values = values.at[t + 1].set(MockSokobanEnv.compute_return(obs))
        rewards = rewards.at[t].set(reward)
        dones = dones.at[t + 1].set(terminated | truncated)
        truncateds = truncateds.at[t].set(truncated)
        terminateds = terminateds.at[t].set(terminated)

    # Calculate TD errors
    discount = (~jnp.array(dones[1:])) * gamma
    td_errors = jnp.zeros((num_timesteps, num_envs))

    # Calculate TD errors in reverse
    for t in reversed(range(num_timesteps)):
        td_errors = td_errors.at[t].set(rewards[t] + discount[t] * values[t + 1] - values[t])

    # Check TD errors
    assert jnp.allclose(td_errors * (~truncateds), 0.0, atol=1e-6)
    if gamma == 1.0:
        assert jnp.all(values[dones] == max_episode_steps)


@pytest.mark.parametrize(
    "cfg",
    [
        BoxobanConfig(
            max_episode_steps=10,
            num_envs=5,
            tinyworld_obs=True,
            asynchronous=False,
            min_episode_steps=8,
            cache_path=Path(__file__).parent,
        ),
        pytest.param(
            EnvpoolBoxobanConfig(
                max_episode_steps=10,
                num_envs=5,
                env_id="Sokoban-v0",
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

    # Set nn_without_noop
    if hasattr(cfg, "nn_without_noop"):
        cfg = dataclasses.replace(cfg, nn_without_noop=nn_without_noop)

    # Create environment
    envs = cfg.make()

    # Create a key for the environment
    key = jax.random.PRNGKey(42)
    env_keys = jax.random.split(key, cfg.num_envs)

    # Reset the environment
    reset_out = envs.reset_env(env_keys)
    next_obs, state = reset_out.obs, reset_out.state

    assert next_obs.shape == (cfg.num_envs, 3, 10, 10), "jax.lax convs are NCHW but you sent NHWC"

    # Initialize network
    args = sokoban_drc33_59()
    key, agent_params_subkey, carry_key = jax.random.split(key, 3)

    # Use type cast to help the linter
    from gymnasium.vector import VectorEnv

    policy, _, agent_params = args.net.init_params(cast(VectorEnv, envs), agent_params_subkey)

    # Check that the network has the correct number of actions
    assert agent_params["params"]["actor_params"]["Output"]["kernel"].shape[1] == 4 + (not nn_without_noop), (
        "NOOP action not set correctly"
    )

    # Initialize carry
    carry = policy.apply(agent_params, carry_key, next_obs.shape, method=policy.initialize_carry)
    episode_starts_no = jnp.zeros(cfg.num_envs, dtype=jnp.bool_)

    # Get actions from policy
    key, action_key = jax.random.split(key)
    carry, actions, _, _, _ = policy.apply(
        agent_params, carry, next_obs, episode_starts_no, action_key, method=policy.get_action
    )

    # Step the environment
    step_out = envs.step_env(env_keys, state, actions)
    next_obs = step_out.obs

    assert next_obs.shape == (cfg.num_envs, 3, 10, 10)
