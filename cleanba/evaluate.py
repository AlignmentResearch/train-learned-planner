import dataclasses
from typing import Callable

import jax
import jax.numpy as jnp
import numpy as np

from cleanba.environments import EnvConfig


@dataclasses.dataclass
class EvalConfig:
    env: EnvConfig
    n_episode_multiple: int = 1
    steps_to_think: int = 0
    temperature: float = 0.0

    def __post_init__(self):
        if self.steps_to_think > 0:
            raise NotImplementedError(f"{self.steps_to_think=}")

    def run(self, get_action: Callable, agent_state, *, key: jnp.ndarray) -> dict[str, float]:
        key, env_key = jax.random.split(key, 2)
        env_seed = int(jax.random.randint(env_key, (), minval=0, maxval=2**31 - 2))
        envs = dataclasses.replace(self.env, seed=env_seed).make()

        try:
            all_episode_returns = []
            all_episode_lengths = []
            for _ in range(self.n_episode_multiple):
                obs, _ = envs.reset()

                eps_done = np.zeros(envs.num_envs, dtype=np.bool_)
                episode_returns = np.zeros(envs.num_envs, dtype=np.float64)
                episode_lengths = np.zeros(envs.num_envs, dtype=np.int64)

                while not np.all(eps_done):
                    action, _, key = get_action(
                        params=agent_state,
                        next_obs=obs,
                        key=key,
                        temperature=self.temperature,
                    )
                    cpu_action = np.asarray(action)
                    obs, rewards, truncated, terminated, infos = envs.step(cpu_action)
                    episode_returns[~eps_done] += rewards[~eps_done]
                    episode_lengths[~eps_done] += 1

                    # Set as done the episodes which are done
                    eps_done |= truncated | terminated

                all_episode_returns.append(episode_returns)
                all_episode_lengths.append(episode_lengths)

            return dict(
                episode_returns_mean=float(np.mean(all_episode_returns)),
                episode_returns_max=float(np.max(all_episode_returns)),
                episode_returns_min=float(np.min(all_episode_returns)),
                episode_lengths_mean=float(np.mean(all_episode_lengths)),
                episode_lengths_max=float(np.max(all_episode_lengths)),
                episode_lengths_min=float(np.min(all_episode_lengths)),
            )
        finally:
            envs.close()
