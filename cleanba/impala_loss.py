import dataclasses
from functools import partial
from typing import Any, Callable, List, NamedTuple

import jax
import jax.numpy as jnp
import rlax
from flax.training.train_state import TrainState
from numpy.typing import NDArray


@dataclasses.dataclass(frozen=True)
class ImpalaLossConfig:
    gamma: float = 0.99  # the discount factor gamma
    ent_coef: float = 0.01  # coefficient of the entropy
    vf_coef: float = 0.5  # coefficient of the value function

    # Interpolate between VTrace (1.0) and monte-carlo function (0.0) estimates, for the estimate of targets, used in
    # both the value and policy losses. It's the parameter in Remark 2 of Espeholt et al.
    # (https://arxiv.org/pdf/1802.01561.pdf)
    vtrace_lambda: float = 0.95

    # Maximum importance ratio for the VTrace value estimates. This is \overline{rho} in eq. 1 of Espeholt et al.
    # (https://arxiv.org/pdf/1802.01561.pdf). \overline{c} is hardcoded to 1 in rlax.
    clip_rho_threshold: float = 1.0

    # Maximum importance ratio for policy gradient outputs. Clips the importance ratio in eq. 4 of Espeholt et al.
    # (https://arxiv.org/pdf/1802.01561.pdf)
    clip_pg_rho_threshold: float = 1.0


class Transition(NamedTuple):
    obs_t: jax.Array
    done_tm1: jax.Array | NDArray
    a_t: jax.Array
    logits_t: jax.Array
    r_tm1: jax.Array | NDArray
    trunc_tm1: jax.Array | NDArray
    term_tm1: jax.Array | NDArray


def policy_gradient_loss(logits, *args):
    """rlax.policy_gradient_loss, but with sum(loss) and [T, B, ...] inputs."""
    mean_per_batch = jax.vmap(rlax.policy_gradient_loss, in_axes=1)(logits, *args)
    total_loss_per_batch = mean_per_batch * logits.shape[0]
    return jnp.sum(total_loss_per_batch)


def entropy_loss_fn(logits, *args):
    """rlax.entropy_loss, but with sum(loss) and [T, B, ...] inputs."""
    mean_per_batch = jax.vmap(rlax.entropy_loss, in_axes=1)(logits, *args)
    total_loss_per_batch = mean_per_batch * logits.shape[0]
    return jnp.sum(total_loss_per_batch)


def impala_loss(
    params,
    get_logits_and_value: Callable[[Any, jax.Array], tuple[jax.Array, jax.Array]],
    args: ImpalaLossConfig,
    obs_t,
    a_t,
    logits_t,
    r_tm1,
    done_tm1,
):
    discount_tm1 = (~done_tm1) * args.gamma
    firststeps_t = done_tm1
    mask_t = ~firststeps_t

    logits_to_update, value_to_update = jax.vmap(get_logits_and_value, in_axes=(None, 0))(params, obs_t)

    v_t = value_to_update[1:]
    # Remove bootstrap timestep from non-timesteps.
    v_tm1 = value_to_update[:-1]

    logits_to_update = logits_to_update[:-1]
    logits_t = logits_t[:-1]
    a_t = a_t[:-1]
    rhos_tm1 = rlax.categorical_importance_sampling_ratios(logits_to_update, logits_t, a_t)

    mask_t = mask_t[:-1]
    float_mask_t = jnp.astype(mask_t, jnp.float32)
    r_tm1 = r_tm1[1:]
    discount_tm1 = discount_tm1[1:]
    vtrace_td_error_and_advantage = jax.vmap(
        partial(
            rlax.vtrace_td_error_and_advantage,
            lambda_=args.vtrace_lambda,
            clip_rho_threshold=args.clip_rho_threshold,
            clip_pg_rho_threshold=args.clip_pg_rho_threshold,
            stop_target_gradients=True,
        ),
        in_axes=1,
        out_axes=1,
    )

    """
    Some of these arguments are misnamed in `vtrace_td_error_and_advantage`:

    The argument `r_t` is paired with `v_t` and `v_tm1` to compute the TD error. But that's not the equation of
    the TD error, it is:

            td(t) = r_t + gamma_t*V(x_{t+1}) - V(x_{t})

    So arguably instead of r_t and discount_t, they should be r_tm1 and discount_tm1. And that's what we name
    them here.
    """
    vtrace_returns = vtrace_td_error_and_advantage(v_tm1, v_t, r_tm1, discount_tm1, rhos_tm1)
    pg_advs = vtrace_returns.pg_advantage
    pg_loss = policy_gradient_loss(logits_to_update, a_t, pg_advs, float_mask_t)

    baseline_loss = 0.5 * jnp.sum(jnp.square(vtrace_returns.errors) * mask_t)
    ent_loss = entropy_loss_fn(logits_to_update, float_mask_t)

    total_loss = pg_loss
    total_loss += args.vf_coef * baseline_loss
    total_loss += args.ent_coef * ent_loss
    return total_loss, (pg_loss, baseline_loss, ent_loss)


SINGLE_DEVICE_UPDATE_DEVICES_AXIS: str = "local_devices"


@partial(jax.jit, static_argnames=["num_batches", "get_logits_and_value", "impala_cfg"])
def single_device_update(
    agent_state: TrainState,
    sharded_storages: List[Transition],
    key: jax.Array,
    *,
    get_logits_and_value: Callable,
    num_batches: int,
    impala_cfg: ImpalaLossConfig,
):
    storage = jax.tree.map(lambda *x: jnp.hstack(x), *sharded_storages)
    impala_loss_grad_fn = jax.value_and_grad(impala_loss, has_aux=True)

    def update_minibatch(agent_state, minibatch):
        (
            mb_obs,
            mb_actions,
            mb_logitss,
            mb_rewards,
            mb_dones,
        ) = minibatch
        (loss, (pg_loss, v_loss, entropy_loss)), grads = impala_loss_grad_fn(
            agent_state.params,
            get_logits_and_value,
            impala_cfg,
            mb_obs,
            mb_actions,
            mb_logitss,
            mb_rewards,
            mb_dones,
        )
        grads = jax.lax.pmean(grads, axis_name="local_devices")
        agent_state = agent_state.apply_gradients(grads=grads)
        return agent_state, (loss, pg_loss, v_loss, entropy_loss)

    agent_state, (loss, pg_loss, v_loss, entropy_loss) = jax.lax.scan(
        update_minibatch,
        agent_state,
        (
            jnp.array(jnp.split(storage.obs_t, num_batches, axis=1)),
            jnp.array(jnp.split(storage.a_t, num_batches, axis=1)),
            jnp.array(jnp.split(storage.logits_t, num_batches, axis=1)),
            jnp.array(jnp.split(storage.r_tm1, num_batches, axis=1)),
            jnp.array(jnp.split(storage.done_tm1, num_batches, axis=1)),
        ),
    )
    loss = jax.lax.pmean(loss, axis_name=SINGLE_DEVICE_UPDATE_DEVICES_AXIS).mean()
    pg_loss = jax.lax.pmean(pg_loss, axis_name=SINGLE_DEVICE_UPDATE_DEVICES_AXIS).mean()
    v_loss = jax.lax.pmean(v_loss, axis_name=SINGLE_DEVICE_UPDATE_DEVICES_AXIS).mean()
    entropy_loss = jax.lax.pmean(entropy_loss, axis_name=SINGLE_DEVICE_UPDATE_DEVICES_AXIS).mean()
    return agent_state, loss, pg_loss, v_loss, entropy_loss, key
