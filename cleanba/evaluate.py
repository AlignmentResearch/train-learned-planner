import dataclasses
import random
from functools import partial

import flax
import jax
import jax.numpy as jnp
import numpy as np

from cleanba.config import random_seed
from cleanba.environments import EnvConfig


@partial(jax.jit, static_argnames=("temperature_is_zero", "network", "actor"))
def get_action_and_value(
    temperature_is_zero: bool,  # static
    temperature: float,
    network,  # static
    actor,  # static
    network_params: flax.core.FrozenDict,
    actor_params: flax.core.FrozenDict,
    obs: np.ndarray,
    key: jax.random.PRNGKey,
) -> tuple[jnp.ndarray, jax.random.PRNGKey]:
    hidden = network.apply(network_params, obs)
    logits = actor.apply(actor_params, hidden)
    assert len(logits.shape) == 2

    if temperature_is_zero:
        action = jnp.argmax(logits, axis=1)
    else:
        # sample action: Gumbel-max trick
        # see https://stats.stackexchange.com/questions/359442/sampling-from-a-categorical-distribution
        key, subkey = jax.random.split(key)
        u = jax.random.uniform(subkey, shape=logits.shape)
        action = jnp.argmax(logits / temperature - jnp.log(-jnp.log(u)), axis=1)
    return action, key


@dataclasses.dataclass
class EvalConfig:
    env: EnvConfig
    n_episode_multiple: int = 1
    steps_to_think: int = 0
    temperature: float = 0.0

    def __post_init__(self):
        if self.steps_to_think > 0:
            raise NotImplementedError(f"{self.steps_to_think=}")

    def run(self, network, actor, agent_state, *, key: jnp.ndarray) -> dict[str, float]:
        envs = self.env.make()
        try:
            all_episode_returns = []
            all_episode_lengths = []
            for _ in range(self.n_episode_multiple):
                key, env_seed = jax.random.split(key, 2)
                jax_randint = jax.random.randint(env_seed, (), minval=0, maxval=2**31 - 1)
                obs, _ = envs.reset(seed=int(jax_randint))

                eps_done = np.zeros(envs.num_envs, dtype=np.bool_)
                episode_returns = np.zeros(envs.num_envs, dtype=np.float64)
                episode_lengths = np.zeros(envs.num_envs, dtype=np.int64)

                while not np.all(eps_done):
                    action, key = get_action_and_value(
                        temperature_is_zero=(self.temperature == 0.0),
                        temperature=self.temperature,
                        network=network,
                        actor=actor,
                        network_params=agent_state.network_params,
                        actor_params=agent_state.actor_params,
                        obs=obs,
                        key=key,
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
                episode_returns_mean=np.mean(all_episode_returns),
                episode_returns_max=np.max(all_episode_returns),
                episode_returns_min=np.min(all_episode_returns),
                episode_lengths_mean=np.mean(all_episode_lengths),
                episode_lengths_max=np.max(all_episode_lengths),
                episode_lengths_min=np.min(all_episode_lengths),
            )
        finally:
            envs.close()
