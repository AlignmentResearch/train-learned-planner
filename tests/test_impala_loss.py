from typing import Any, SupportsFloat

import gymnasium as gym
import jax.numpy as jnp
import numpy as np
import pytest
import rlax
from jax import ensure_compile_time_eval
from numpy.typing import NDArray


@pytest.mark.parametrize("gamma", [0.0, 0.9, 1.0])
@pytest.mark.parametrize("num_timesteps", [20, 2, 1])
@pytest.mark.parametrize("last_value", [0.0, 1.0])
def test_vtrace_alignment(np_rng: np.random.Generator, gamma: float, num_timesteps: int, last_value: float):
    rewards = np_rng.uniform(0.1, 2.0, size=num_timesteps)
    correct_returns = np.zeros(len(rewards) + 1)

    # Discount is gamma everywhere, except once in the middle of the episode
    discount = np.ones_like(rewards) * gamma
    if num_timesteps > 2:
        discount[num_timesteps // 2] = last_value

    # There are no more returns after the last step
    correct_returns[-1] = 0.0
    # Bellman equation to compute the correct returns
    for i in range(len(rewards) - 1, -1, -1):
        correct_returns[i] = rewards[i] + discount[i] * correct_returns[i + 1]

    # Now check that the vtrace error is zero
    rho_tm1 = np_rng.lognormal(0.0, 1.0, size=num_timesteps)

    v_tm1 = correct_returns[:-1]
    v_t = correct_returns[1:]
    vtrace_error = rlax.vtrace(v_tm1, v_t, rewards, discount, rho_tm1)

    assert np.allclose(vtrace_error, np.zeros(num_timesteps))


class TrivialEnv(gym.Env[NDArray, np.int64]):
    def __init__(self, gamma=0.9):
        self.gamma = gamma
        self.reward_range = (0.1, 2.0)
        self.action_space = gym.spaces.Discrete(1)
        self.observation_space = gym.spaces.Box(low=0.0, high=float("inf"), shape=())

    def step(self, action: NDArray) -> tuple[NDArray, SupportsFloat, bool, bool, dict[str, Any]]:
        # Pretend that we took the optimal action (there is only one action)
        reward = self._rewards[self._t]

        self._t += 1
        terminated = self._t == len(self._rewards)
        return (self._returns[self._t], reward, terminated, False, {})

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[NDArray, dict[str, Any]]:
        super().reset(seed=seed, options=options)

        num_timesteps = int(self.np_random.integers(1, 10, size=()))
        self._t = 0
        self._rewards = self.np_random.uniform(self.reward_range[0], self.reward_range[1], size=num_timesteps)
        returns = np.zeros(num_timesteps + 1)
        for i in range(num_timesteps - 1, -1, -1):
            returns[i] = self._rewards[i] + self.gamma * returns[i + 1]

        self._returns = returns
        return self._returns[self._t], {}
