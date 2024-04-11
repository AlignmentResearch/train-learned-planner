import dataclasses
import queue
from functools import partial
from typing import Any, Callable, SupportsFloat

import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np
import pytest
import rlax
from numpy.typing import NDArray

import cleanba.cleanba_impala as cleanba_impala
from cleanba.environments import EnvConfig
from cleanba.impala_loss import ImpalaLossConfig


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
    def __init__(self, cfg: "TrivialEnvConfig"):
        self.cfg = cfg
        self.reward_range = (0.1, 2.0)
        self.action_space = gym.spaces.Discrete(1)
        self.observation_space = gym.spaces.Box(low=0.0, high=float("inf"), shape=())

    def step(self, action: np.int64) -> tuple[NDArray, SupportsFloat, bool, bool, dict[str, Any]]:
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

        num_timesteps = int(self.np_random.integers(1, self.cfg.max_episode_steps, size=()))
        self._t = 0
        self._rewards = self.np_random.uniform(self.reward_range[0], self.reward_range[1], size=num_timesteps)

        # Bellman equation to compute returns
        returns = np.zeros(num_timesteps + 1)
        for i in range(num_timesteps - 1, -1, -1):
            returns[i] = self._rewards[i] + self.cfg.gamma * returns[i + 1]
        self._returns = returns

        return self._returns[self._t], {}


@dataclasses.dataclass
class TrivialEnvConfig(EnvConfig):
    gamma: float = 0.9

    @property
    def make(self) -> Callable[[], gym.vector.VectorEnv]:
        env_fn = partial(TrivialEnv, cfg=self)
        return partial(gym.vector.SyncVectorEnv, [env_fn] * self.num_envs)


def test_trivial_env_correct_returns(np_rng: np.random.Generator, num_envs: int = 7, gamma: float = 0.9):
    envs = TrivialEnvConfig(max_episode_steps=10, num_envs=num_envs, gamma=gamma).make()

    returns = []
    rewards = []
    terminateds = []
    obs, _ = envs.reset(seed=1234)
    returns.append(obs)

    num_timesteps = 30
    for t in range(num_timesteps):
        obs, reward, terminated, truncated, _ = envs.step(np.zeros(num_envs, dtype=np.int64))
        returns.append(obs)
        rewards.append(reward)
        terminateds.append(terminated)
        assert not np.any(truncated)

    rho_tm1 = np_rng.lognormal(0.0, 1.0, size=num_timesteps)

    v_t = np.array(returns[1:])
    v_tm1 = np.array(returns[:-1])
    discount_t = (~np.array(terminateds)) * gamma
    r_t = np.array(rewards)
    rho_tm1 = np_rng.lognormal(0.0, 1.0, size=(num_timesteps, num_envs))

    out = jax.vmap(rlax.vtrace, 1, 1)(v_tm1, v_t, r_t, discount_t, rho_tm1)
    assert np.allclose(out, np.zeros_like(out), atol=1e-6)


@jax.jit
def _get_zero_action(params, next_obs, key):
    assert params == {}
    actions = jnp.zeros(next_obs.shape[0], dtype=jnp.int32)
    logits = jnp.zeros((next_obs.shape[0], 1), dtype=jnp.float32)
    return next_obs, actions, logits, key


def test_loss_of_rollout(num_envs: int = 5, gamma: float = 0.9, num_timesteps: int = 30):
    args = cleanba_impala.Args(
        train_env=TrivialEnvConfig(max_episode_steps=10, num_envs=0, gamma=gamma),
        eval_envs={},
        loss=ImpalaLossConfig(
            gamma=0.9,
            vtrace_lambda=1.0,
        ),
        num_steps=num_timesteps,
        concurrency=False,
        local_num_envs=num_envs,
    )

    params_queue = queue.Queue(maxsize=5)
    for _ in range(5):
        params_queue.put({})

    rollout_queue = queue.Queue(maxsize=5)
    key = jax.random.PRNGKey(seed=1234)
    cleanba_impala.rollout(
        key=key,
        args=args,
        runtime_info=cleanba_impala.RuntimeInformation(0, [], 0, 0, 0, 0, 0, 0, 0, [], []),
        rollout_queue=rollout_queue,
        params_queue=params_queue,
        writer=None,  # OK because device_thread_id != 0
        learner_devices=jax.local_devices(),
        device_thread_id=1,
        actor_device=None,  # Currently unused
        get_action=_get_zero_action,
    )

    while True:
        try:
            (
                global_step,
                actor_policy_version,
                update,
                sharded_transition,
                params_queue_get_time,
                device_thread_id,
            ) = rollout_queue.get(timeout=1e-5)
        except queue.Empty:
            break  # we're done

        assert isinstance(global_step, int)
        assert isinstance(actor_policy_version, int)
        assert isinstance(update, int)
        assert isinstance(sharded_transition, cleanba_impala.Transition)
        assert isinstance(params_queue_get_time, float)
        assert device_thread_id == 1

        assert sharded_transition.obs_t.shape == (1, num_timesteps + 1, num_envs)

        tr = sharded_transition

        v_t = tr.obs_t[0, 1:]
        v_tm1 = tr.obs_t[0, :-1]

        # We have to use 1: with these because they represent the reward/discount of the *previous* step.
        r_t = tr.r_tm1[0, 1:]
        discount_t = (~tr.done_tm1[0, 1:]) * gamma

        rho_tm1 = np.ones((num_timesteps, num_envs))

        out = jax.vmap(rlax.vtrace, 1, 1)(v_tm1, v_t, r_t, discount_t, rho_tm1)
        assert np.allclose(out, np.zeros_like(out), atol=1e-6)
