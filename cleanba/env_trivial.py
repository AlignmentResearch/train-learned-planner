from functools import partial
from typing import Any, Callable, Iterable, List, Optional, SupportsFloat, Union

import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np
from numpy.typing import NDArray

from cleanba.environments import EnvConfig


class MockSokobanEnv(gym.Env[NDArray, np.int64]):
    """
    An environment with the same observation shape as Sokoban but which tells you the total remaining return in the [0,
    0] position.
    """

    def __init__(self, cfg: "MockSokobanEnvConfig"):
        self.cfg = cfg
        self.reward_range = (0.1, 2.0)
        self.action_space = gym.spaces.Discrete(2)  # Two actions so we can test importance ratios
        self.observation_space = gym.spaces.Box(low=0.0, high=float("inf"), shape=(3, 10, 10), dtype=np.float64)

    def step(self, action: np.int64) -> tuple[NDArray, SupportsFloat, bool, bool, dict[str, Any]]:
        # Pretend that we took the optimal action (there is only one action)
        reward = self._rewards[self._t]

        self._t += 1
        if self._t > self._ep_length:
            raise IndexError(f"time {self._t=} exceeded episode bounds")

        if self._t == self._ep_length:
            if self._ep_length == len(self._rewards):
                terminated = True
                truncated = False
            elif self._ep_length < len(self._rewards):
                terminated = False
                truncated = True
            else:
                raise ValueError(f"{self._ep_length=}, {len(self._rewards)=}")
        else:
            terminated = truncated = False
        return (self._make_obs(), reward, terminated, truncated, {})

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[NDArray, dict[str, Any]]:
        super().reset(seed=seed, options=options)
        if self.cfg.min_episode_steps < 1:
            raise ValueError(f"{self.cfg.min_episode_steps=}, but must be >=1")

        if self.cfg.min_episode_steps == self.cfg.max_episode_steps:
            self._ep_length = self.cfg.max_episode_steps
        else:
            self._ep_length = int(self.np_random.integers(self.cfg.min_episode_steps, self.cfg.max_episode_steps + 1, size=()))
        self._t = 0
        self._rewards = np.ones(self.cfg.max_episode_steps)  # Always send reward=1 for easy debuggability

        # Bellman equation to compute returns
        returns = np.zeros(self.cfg.max_episode_steps + 1)
        for i in range(self.cfg.max_episode_steps - 1, -1, -1):
            returns[i] = self._rewards[i] + self.cfg.gamma * returns[i + 1]
        self._returns = returns

        self._background = self.np_random.uniform(0.0, 255.0, self.observation_space.shape)
        self._foreground = np.copy(self._background)

        return self._make_obs(), {}

    def _make_obs(self) -> NDArray:
        self._foreground[0, 0, 0] = self._returns[self._t]
        return self._foreground

    @staticmethod
    def compute_return(obs: NDArray) -> NDArray:
        return jnp.copy(obs[..., 0, 0, 0])


class SeededSyncVectorEnv(gym.vector.SyncVectorEnv):
    """
    Seed the environments only when `reset_async()` is called, not every time the environment is reset. This way,
    the environment is not the same every time. This mimics the behavior of Envpool and makes sure it's not the same
    environment repeating over and over.
    """

    def __init__(
        self,
        env_fns: Iterable[Callable[[], gym.Env]],
        seed: int,
        observation_space: gym.Space = None,  # type: ignore
        action_space: gym.Space = None,  # type: ignore
        copy: bool = True,
    ):
        env_fns = list(env_fns)
        super().__init__(env_fns, observation_space, action_space, copy)
        seeds = np.random.default_rng(seed).integers(2**30 - 1, size=(len(env_fns),))
        self._seeds = list(seeds)

    def reset_async(self, seed: Optional[Union[int, List[int]]] = None, options: Optional[dict] = None):
        if seed is None:
            seed = self._seeds
        return super().reset_async(seed, options)

    def step(self, actions: np.ndarray | jax.Array) -> tuple[Any, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
        return super().step(np.asarray(actions))


class MockSokobanEnvConfig(EnvConfig):
    min_episode_steps: int = 1
    gamma: float = 1.0

    @property
    def make(self) -> Callable[[], gym.vector.VectorEnv]:
        env_fns = [partial(MockSokobanEnv, cfg=self)] * self.num_envs
        return partial(SeededSyncVectorEnv, env_fns, seed=self.seed)
