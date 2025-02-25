import dataclasses
import queue
from functools import partial
from typing import Any

import distrax
import flax.linen as nn
import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np
import pytest
import rlax

import cleanba.cleanba_impala as cleanba_impala
from cleanba.env_trivial import MockSokobanEnv, MockSokobanEnvConfig
from cleanba.impala_loss import ActorCriticLossConfig, ImpalaLossConfig, PPOLossConfig, Rollout
from cleanba.network import Policy, PolicySpec


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
@pytest.mark.parametrize("gae_lambda", [0.0, 0.8, 1.0])
@pytest.mark.parametrize("num_timesteps", [20, 2, 1])
@pytest.mark.parametrize("last_value", [0.0, 1.0])
def test_gae_alignment(gamma: float, gae_lambda: float, num_timesteps: int, last_value: float):
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

    gae = rlax.truncated_generalized_advantage_estimation(rewards, discount, gae_lambda, correct_returns)

    assert np.allclose(gae, np.zeros(num_timesteps))


@pytest.mark.parametrize("cls", [ImpalaLossConfig, PPOLossConfig])
@pytest.mark.parametrize("gamma", [0.0, 0.9, 1.0])
@pytest.mark.parametrize("num_timesteps", [20, 2])  # Note: with 1 timesteps we get zero-length arrays
@pytest.mark.parametrize("last_value", [0.0, 1.0])
def test_impala_loss_zero_when_accurate(
    cls: type[ActorCriticLossConfig], gamma: float, num_timesteps: int, last_value: float, batch_size: int = 5
):
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
    (total_loss, metrics_dict) = cls(gamma=gamma).loss(
        params={},
        get_logits_and_value=lambda params, carry, obs, episode_starts: (
            carry,
            jnp.zeros((num_timesteps + 1, batch_size, 1)),
            obs,
            {},
        ),
        minibatch=Rollout(
            obs_t=jnp.array(obs_t),
            carry_t=(),
            episode_starts_t=np.concatenate([np.zeros((1, batch_size), dtype=np.bool_), done_tm1], axis=0),
            truncated_t=np.zeros_like(done_tm1),
            a_t=a_t,
            logits_t=logits_t,
            value_t=jnp.array(obs_t),
            r_t=rewards,
        ),
    )

    assert np.allclose(metrics_dict["pg_loss"], 0.0)
    assert np.allclose(metrics_dict["v_loss"], 0.0)
    assert np.allclose(metrics_dict["ent_loss"], 0.0)
    assert np.allclose(total_loss, 0.0)


class TrivialEnvPolicy(Policy):
    def get_action(
        self,
        carry: tuple[()],
        obs: jax.Array,
        episode_starts: jax.Array,
        key: jax.Array,
        *,
        temperature: float = 1.0,
    ) -> tuple[tuple[()], jax.Array, jax.Array, jax.Array, jax.Array]:
        actions = jnp.zeros(obs.shape[0], dtype=jnp.int32)
        logits = jnp.stack(
            [
                MockSokobanEnv.compute_return(obs),  # Use obs itself to vary logits in a repeatable way
                jnp.zeros((obs.shape[0],), dtype=jnp.float32),
            ],
            axis=1,
        )
        value = MockSokobanEnv.compute_return(obs)
        return (), actions, logits, value, key

    def get_logits_and_value(
        self,
        carry: tuple[()],
        obs: jax.Array,
        episode_starts: jax.Array,
    ) -> tuple[tuple[()], jax.Array, jax.Array, dict[str, jax.Array]]:
        carry, actions, logits, _, key = jax.vmap(self.get_action, in_axes=(None, 0, None, None))(
            carry,
            obs,
            None,  # type: ignore
            jax.random.PRNGKey(1234),
        )

        value = MockSokobanEnv.compute_return(obs)
        return carry, logits, value, {}


@dataclasses.dataclass(frozen=True)
class ZeroActionNetworkSpec(PolicySpec):
    def make(self) -> nn.Module:
        return None  # type: ignore

    def init_params(self, envs: gym.vector.VectorEnv, key: jax.Array) -> tuple["Policy", tuple[()], Any]:
        policy = TrivialEnvPolicy(2, self)
        return policy, (), {}


@pytest.mark.parametrize("cls", [ImpalaLossConfig, PPOLossConfig])
@pytest.mark.parametrize("min_episode_steps", (10, 7))
def test_loss_of_rollout(
    cls: type[ActorCriticLossConfig], min_episode_steps: int, num_envs: int = 5, gamma: float = 1.0, num_timesteps: int = 30
):
    np.random.seed(1234)

    args = cleanba_impala.Args(
        train_env=MockSokobanEnvConfig(
            max_episode_steps=10, num_envs=0, gamma=gamma, min_episode_steps=min_episode_steps, seed=4
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

    policy, carry, params = args.net.init_params(None, None)  # type: ignore
    get_logits_and_value_fn = jax.jit(partial(policy.apply, method=policy.get_logits_and_value))

    params_queue = queue.Queue(maxsize=5)
    for _ in range(5):
        params_queue.put((params, 1))

    rollout_queue = queue.Queue(maxsize=5)
    key = jax.random.PRNGKey(seed=1234)
    cleanba_impala.rollout(
        initial_update=1,
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

        assert sharded_transition.obs_t.shape == (1, num_timesteps + 1, num_envs, 3, 10, 10)
        assert sharded_transition.r_t.shape == (1, num_timesteps, num_envs)
        assert sharded_transition.a_t.shape == (1, num_timesteps, num_envs)
        assert sharded_transition.logits_t.shape == (1, num_timesteps, num_envs, 2)

        transition = cleanba_impala.unreplicate(sharded_transition)

        values = MockSokobanEnv.compute_return(transition.obs_t)
        v_t = values[1:]
        v_tm1 = values[:-1]

        # We have to use 1: with these because they represent the reward/discount of the *previous* step.
        r_t = transition.r_t
        discount_t = (~transition.episode_starts_t[1:]) * gamma

        rho_tm1 = np.ones((num_timesteps, num_envs))

        # We want the error to be 0 when the environment is truncated
        r_t = (~transition.truncated_t) * transition.r_t + transition.truncated_t * v_tm1
        out = jax.vmap(rlax.vtrace, 1, 1)(v_tm1, v_t, r_t, discount_t, rho_tm1)
        assert np.allclose(out, np.zeros_like(out), atol=1e-5), f"Return was incorrect at {iteration=}"

        # Now check that the impala loss works here, i.e. an integration test

        ## Check that the reward of truncated transitions affects nothing
        transition = Rollout(
            obs_t=transition.obs_t,
            carry_t=transition.carry_t,
            a_t=transition.a_t,
            logits_t=transition.logits_t,
            value_t=transition.value_t,
            r_t=transition.r_t.at[transition.truncated_t].set(9999.9),
            episode_starts_t=transition.episode_starts_t,
            truncated_t=transition.truncated_t,
        )
        (total_loss, metrics_dict) = cls(gamma=gamma).loss(
            params=params,
            get_logits_and_value=get_logits_and_value_fn,
            minibatch=transition,
        )
        logit_negentropy = -jnp.mean(distrax.Categorical(transition.logits_t).entropy() * (~transition.truncated_t))

        assert np.allclose(metrics_dict["pg_loss"], 0.0)
        assert np.allclose(metrics_dict["v_loss"], 0.0)
        assert np.allclose(metrics_dict["ent_loss"], logit_negentropy)
