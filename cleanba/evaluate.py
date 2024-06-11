import contextlib
import dataclasses

import jax
import jax.numpy as jnp
import numpy as np

from cleanba.environments import EnvConfig
from cleanba.network import Policy


@dataclasses.dataclass
class EvalConfig:
    env: EnvConfig
    n_episode_multiple: int = 1
    steps_to_think: list[int] = dataclasses.field(default_factory=lambda: [0])
    temperature: float = 0.0

    safeguard_max_episode_steps: int = 30000

    def run(self, policy: Policy, get_action_fn, params, *, key: jnp.ndarray) -> dict[str, float]:
        key, carry_key = jax.random.split(key, 2)
        episode_starts_no = jnp.zeros(self.env.num_envs, dtype=jnp.bool_)

        metrics = {}
        for steps_to_think in self.steps_to_think:
            # Create the environments every time with the same seed so the levels are the exact same
            with contextlib.closing(self.env.make()) as envs:
                all_episode_returns = []
                all_episode_lengths = []
                all_episode_successes = []
                for _ in range(self.n_episode_multiple):
                    obs, _ = envs.reset()
                    # reset the carry here so we can use `episode_starts_no` later
                    carry = policy.apply(params, carry_key, obs.shape, method=policy.initialize_carry)

                    # Update the carry with the initial observation many times
                    for think_step in range(steps_to_think):
                        carry, _, _, key = get_action_fn(
                            params, carry, obs, episode_starts_no, key, temperature=self.temperature
                        )

                    eps_done = np.zeros(envs.num_envs, dtype=np.bool_)
                    episode_success = np.zeros(envs.num_envs, dtype=np.bool_)
                    episode_returns = np.zeros(envs.num_envs, dtype=np.float64)
                    episode_lengths = np.zeros(envs.num_envs, dtype=np.int64)

                    i = 0
                    while not np.all(eps_done):
                        i += 1
                        if i >= self.safeguard_max_episode_steps:
                            break
                        carry, action, _, key = get_action_fn(
                            params, carry, obs, episode_starts_no, key, temperature=self.temperature
                        )

                        cpu_action = np.asarray(action)
                        obs, rewards, terminated, truncated, infos = envs.step(cpu_action)
                        episode_returns[~eps_done] += rewards[~eps_done]
                        episode_lengths[~eps_done] += 1
                        episode_success[~eps_done] |= terminated[~eps_done]  # If episode terminates it's a success

                        # Set as done the episodes which are done
                        eps_done |= truncated | terminated

                    all_episode_returns.append(episode_returns)
                    all_episode_lengths.append(episode_lengths)
                    all_episode_successes.append(episode_success)

                metrics.update(
                    {
                        f"{steps_to_think:02d}_episode_returns": float(np.mean(all_episode_returns)),
                        f"{steps_to_think:02d}_episode_lengths": float(np.mean(all_episode_lengths)),
                        f"{steps_to_think:02d}_episode_successes": float(np.mean(all_episode_successes)),
                    }
                )
        return metrics
