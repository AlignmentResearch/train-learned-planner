import dataclasses
import queue
from functools import partial
from typing import Any, Callable, SupportsFloat

import distrax
import flax.linen as nn
import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np
import pytest
import rlax
from numpy.typing import NDArray

import cleanba.cleanba_impala as cleanba_impala
from cleanba.environments import EnvConfig
from cleanba.impala_loss import ImpalaLossConfig, Rollout, impala_loss
from cleanba.network import AgentParams, NetworkSpec


@pytest.mark.parametrize("gamma", [0.0, 0.9, 1.0])
@pytest.mark.parametrize("num_timesteps", [20, 2, 1])
@pytest.mark.parametrize("last_value", [0.0, 1.0])
def test_vtrace_alignment(gamma: float, num_timesteps: int, last_value: float):
    np_rng = np.random.default_rng(1234)

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


@pytest.mark.parametrize("gamma", [0.0, 0.9, 1.0])
@pytest.mark.parametrize("num_timesteps", [20, 2])  # Note: with 1 timesteps we get zero-length arrays
@pytest.mark.parametrize("last_value", [0.0, 1.0])
def test_impala_loss_zero_when_accurate(gamma: float, num_timesteps: int, last_value: float, batch_size: int = 5):
    np_rng = np.random.default_rng(1234)
    rewards = np_rng.uniform(0.1, 2.0, size=(num_timesteps, batch_size))
    correct_returns = np.zeros((num_timesteps + 1, batch_size))

    # Episodes change midway through the timesteps
    done_tm1 = np.zeros((num_timesteps, batch_size), dtype=np.bool_)
    if num_timesteps > 2:
        done_tm1[num_timesteps // 2] = 1

    # There are no more returns after the last step
    correct_returns[-1] = 0.0
    # Bellman equation to compute the correct returns
    for i in range(len(rewards) - 1, -1, -1):
        correct_returns[i] = rewards[i] + ((~done_tm1[i]) * gamma) * correct_returns[i + 1]

    obs_t = correct_returns  #  Mimic how actual rollouts collect observations
    logits_t = jnp.zeros((num_timesteps, batch_size, 1))
    a_t = jnp.zeros((num_timesteps, batch_size), dtype=jnp.int32)
    (total_loss, metrics_dict) = impala_loss(
        params=AgentParams((), (), ()),
        get_logits_and_value=lambda params, obs: (jnp.zeros((batch_size, 1)), obs, {}),
        args=ImpalaLossConfig(gamma=gamma),
        minibatch=Rollout(
            obs_t=jnp.array(obs_t),
            done_t=done_tm1,
            truncated_t=np.zeros_like(done_tm1),
            a_t=a_t,
            logits_t=logits_t,
            r_t=rewards,
        ),
    )

    assert np.allclose(metrics_dict["pg_loss"], 0.0)
    assert np.allclose(metrics_dict["v_loss"], 0.0)
    assert np.allclose(metrics_dict["ent_loss"], 0.0)
    assert np.allclose(total_loss, 0.0)


class TrivialEnv(gym.Env[NDArray, np.int64]):
    def __init__(self, cfg: "TrivialEnvConfig"):
        """
        truncation_probability: the probability that an entire *episode* is truncated prematurely
        """
        self.cfg = cfg
        self.reward_range = (0.1, 2.0)
        self.action_space = gym.spaces.Discrete(2)  # Two actions so we can test importance ratios
        self.observation_space = gym.spaces.Box(low=0.0, high=float("inf"), shape=())

        self.per_step_non_truncation_probability = 2 ** np.maximum(
            -100000.0, (np.log2(1 - cfg.truncation_probability) / self.cfg.max_episode_steps)
        )

    def step(self, action: np.int64) -> tuple[NDArray, SupportsFloat, bool, bool, dict[str, Any]]:
        # Pretend that we took the optimal action (there is only one action)
        reward = self._rewards[self._t]

        self._t += 1
        terminated = truncated = False
        if self._t > 1:
            if self.np_random.uniform(0.0, 1.0) < (1 - self.per_step_non_truncation_probability):
                terminated = False
                truncated = True

            # This goes after the previous if so takes precedence.
            if self._t == len(self._rewards):
                terminated = True
                truncated = False
        return (self._returns[self._t], reward, terminated, truncated, {})

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[NDArray, dict[str, Any]]:
        super().reset(seed=seed, options=options)

        num_timesteps = int(self.np_random.integers(2, self.cfg.max_episode_steps, size=()))
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
    truncation_probability: float = 0.5

    @property
    def make(self) -> Callable[[], gym.vector.VectorEnv]:
        seeds = np.random.default_rng(self.seed).integers(2**30 - 1, size=(self.num_envs,))
        print("The seeds are", seeds)
        env_fns = [partial(TrivialEnv, cfg=dataclasses.replace(self, seed=int(s))) for s in seeds]
        return partial(gym.vector.SyncVectorEnv, env_fns)


def test_trivial_env_correct_returns(num_envs: int = 7, gamma: float = 0.9):
    np_rng = np.random.default_rng(1234)
    envs = TrivialEnvConfig(max_episode_steps=10, num_envs=num_envs, gamma=gamma, truncation_probability=0.0, seed=1234).make()

    returns = []
    rewards = []
    terminateds = []
    obs, _ = envs.reset()
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


class ZeroActionNetwork(nn.Module):
    @nn.compact
    def __call__(self, x):
        return x


@dataclasses.dataclass(frozen=True)
class ZeroActionNetworkSpec(NetworkSpec):
    def make(self) -> nn.Module:
        return ZeroActionNetwork()

    @partial(jax.jit, static_argnames=["self"])
    def get_action(self, params: AgentParams, next_obs: jax.Array, key: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array]:
        actions = jnp.zeros(next_obs.shape[0], dtype=jnp.int32)
        logits = jnp.stack(
            [
                next_obs,  # Use next_obs itself to vary logits in a repeatable way
                jnp.zeros((next_obs.shape[0],), dtype=jnp.float32),
            ],
            axis=1,
        )
        return actions, logits, key

    @partial(jax.jit, static_argnames=["self"])
    def get_logits_and_value(self, params: AgentParams, x: jax.Array) -> tuple[jax.Array, jax.Array, dict[str, jax.Array]]:
        return self.get_action(params, x, None)[1], x, {}  # type: ignore


@pytest.mark.parametrize("truncation_probability", [0.0, 0.5])
def test_loss_of_rollout(truncation_probability: float, num_envs: int = 5, gamma: float = 0.9, num_timesteps: int = 30):
    np.random.seed(1234)

    args = cleanba_impala.Args(
        train_env=TrivialEnvConfig(
            max_episode_steps=10, num_envs=0, gamma=gamma, truncation_probability=truncation_probability, seed=4
        ),
        eval_envs={},
        net=ZeroActionNetworkSpec(),
        loss=ImpalaLossConfig(
            gamma=0.9,
            vtrace_lambda=1.0,
        ),
        num_steps=num_timesteps,
        concurrency=True,
        local_num_envs=num_envs,
        seed=3,
    )

    params_queue = queue.Queue(maxsize=5)
    for _ in range(5):
        params_queue.put((AgentParams({}, {}, {}), 1))

    rollout_queue = queue.Queue(maxsize=5)
    key = jax.random.PRNGKey(seed=1234)
    cleanba_impala.rollout(
        key=key,
        args=args,
        runtime_info=cleanba_impala.RuntimeInformation(0, [], 0, 1, 0, 0, 0, 0, 0, [], []),
        rollout_queue=rollout_queue,
        params_queue=params_queue,
        writer=None,  # OK because device_thread_id != 0
        learner_devices=jax.local_devices(),
        device_thread_id=1,
        actor_device=None,  # Currently unused
    )

    for iteration in range(100):
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
        assert isinstance(sharded_transition, cleanba_impala.Rollout)
        assert isinstance(params_queue_get_time, float)
        assert device_thread_id == 1

        assert sharded_transition.obs_t.shape == (1, num_timesteps + 1, num_envs)
        assert sharded_transition.r_t.shape == (1, num_timesteps, num_envs)
        assert sharded_transition.a_t.shape == (1, num_timesteps, num_envs)
        assert sharded_transition.logits_t.shape == (1, num_timesteps, num_envs, 2)

        transition = cleanba_impala.unreplicate(sharded_transition)

        v_t = transition.obs_t[1:]
        v_tm1 = transition.obs_t[:-1]

        # We have to use 1: with these because they represent the reward/discount of the *previous* step.
        r_t = transition.r_t
        discount_t = (~transition.done_t) * gamma

        rho_tm1 = np.ones((num_timesteps, num_envs))

        # We want the error to be 0 when the environment is truncated
        r_t = (~transition.truncated_t) * transition.r_t + transition.truncated_t * v_tm1
        out = jax.vmap(rlax.vtrace, 1, 1)(v_tm1, v_t, r_t, discount_t, rho_tm1)
        assert np.allclose(out, np.zeros_like(out), atol=1e-5), f"Return was incorrect at {iteration=}"

        # Now check that the impala loss works here, i.e. an integration test
        (total_loss, metrics_dict) = impala_loss(
            params=AgentParams({}, {}, {}),
            get_logits_and_value=args.net.get_logits_and_value,
            args=ImpalaLossConfig(gamma=gamma, logit_l2_coef=0.0),
            minibatch=transition,
        )
        logit_negentropy = -jnp.mean(distrax.Categorical(transition.logits_t).entropy() * (~transition.truncated_t))

        assert np.abs(metrics_dict["pg_loss"]) < 1e-6
        assert np.allclose(metrics_dict["v_loss"], 0.0)
        assert np.allclose(metrics_dict["ent_loss"], logit_negentropy)
        assert np.allclose(metrics_dict["max_ratio"], 1.0)
