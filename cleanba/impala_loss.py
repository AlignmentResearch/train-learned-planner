import dataclasses
from functools import partial
from typing import Any, Callable, List, NamedTuple

import flax.traverse_util
import jax
import jax.numpy as jnp
import rlax
from flax.training.train_state import TrainState
from numpy.typing import NDArray


@dataclasses.dataclass(frozen=True)
class ImpalaLossConfig:
    global_coef: float = 640.0  # Multiply the whole loss by this
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


class Rollout(NamedTuple):
    obs_t: jax.Array
    a_t: jax.Array
    logits_t: jax.Array
    r_t: jax.Array | NDArray
    # TODO: handle truncation vs. termination correctly. Truncated episodes should have the discounted estimated value
    # of the next observation, subtracted from the current value's observation. But that's annoying to support and we
    # don't use it for Sokoban anyways, so we won't support it.
    done_t: jax.Array | NDArray


def impala_loss(
    params,
    get_logits_and_value: Callable[[Any, jax.Array], tuple[jax.Array, jax.Array]],
    args: ImpalaLossConfig,
    minibatch: Rollout,
) -> tuple[jax.Array, dict[str, jax.Array]]:
    discount_t = (~minibatch.done_t) * args.gamma
    mask_t = jnp.ones_like(discount_t)  # Don't mask any loss timesteps -- we use them all.

    logits_to_update, value_to_update = jax.vmap(get_logits_and_value, in_axes=(None, 0))(params, minibatch.obs_t)

    # v_t does not enter the gradient in any way, because
    #   1. it's stop_grad()-ed in the `vtrace_td_error_and_advantage.errors`
    #   2. it intervenes in `vtrace_td_error_and_advantage.pg_advantage`, but that's stop_grad() ed by the pg loss.
    v_t = value_to_update[1:]

    # Discarding the last observation's logit corresponds to logits for the same timesteps as was observed in the
    # minibatch.
    logits_to_update = logits_to_update[:-1]

    # Remove bootstrap timestep from non-timesteps.
    v_tm1 = value_to_update[:-1]

    rhos_tm1 = rlax.categorical_importance_sampling_ratios(logits_to_update, minibatch.logits_t, minibatch.a_t)
    max_ratio = jnp.max(jnp.abs(rhos_tm1))

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
    vtrace_returns = vtrace_td_error_and_advantage(v_tm1, v_t, minibatch.r_t, discount_t, rhos_tm1)
    pg_advs = vtrace_returns.pg_advantage

    pg_loss = jnp.mean(jax.vmap(rlax.policy_gradient_loss, in_axes=1)(logits_to_update, minibatch.a_t, pg_advs, mask_t))

    baseline_loss = jnp.mean(jnp.square(vtrace_returns.errors) * mask_t)

    ent_loss = jnp.mean(jax.vmap(rlax.entropy_loss, in_axes=1)(logits_to_update, mask_t))

    total_loss = args.global_coef * pg_loss
    total_loss += (args.global_coef * args.vf_coef) * baseline_loss
    total_loss += (args.global_coef * args.ent_coef) * ent_loss
    return total_loss, dict(pg_loss=pg_loss, baseline_loss=baseline_loss, ent_loss=ent_loss, max_ratio=max_ratio)


SINGLE_DEVICE_UPDATE_DEVICES_AXIS: str = "local_devices"


def tree_flatten_and_concat(x) -> jax.Array:
    leaves, _ = jax.tree.flatten(x)
    return jnp.concat(list(map(jnp.ravel, leaves)))


@partial(jax.jit, static_argnames=["num_batches", "get_logits_and_value", "impala_cfg"])
def single_device_update(
    agent_state: TrainState,
    sharded_storages: List[Rollout],
    key: jax.Array,
    *,
    get_logits_and_value: Callable,
    num_batches: int,
    impala_cfg: ImpalaLossConfig,
) -> tuple[TrainState, jax.Array, dict[str, jax.Array]]:
    def update_minibatch(agent_state: TrainState, minibatch: Rollout):
        (loss, metrics_dict), grads = jax.value_and_grad(impala_loss, has_aux=True)(
            agent_state.params,
            get_logits_and_value,
            impala_cfg,
            minibatch,
        )
        metrics_dict["loss"] = loss
        grads = jax.lax.pmean(grads, axis_name=SINGLE_DEVICE_UPDATE_DEVICES_AXIS)

        for key, value in flax.traverse_util.flatten_dict(grads):
            key_str = "/".join(key)
            metrics_dict["grad_rms/" + key_str] = jnp.sqrt(jnp.mean(jnp.square(value)))

        for key, value in flax.traverse_util.flatten_dict(agent_state.params):
            key_str = "/".join(key)
            metrics_dict["param_rms/" + key_str] = jnp.sqrt(jnp.mean(jnp.square(value)))

        flat_param = tree_flatten_and_concat(
            dict(
                network=agent_state.params.network_params,
                actor=agent_state.params.actor_params,
                critic=agent_state.params.critic_params,
            )
        )
        metrics_dict["param_rms/avg"] = jnp.sqrt(jnp.mean(jnp.square(flat_param)))
        metrics_dict["param_rms/total"] = jnp.sqrt(jnp.sum(jnp.square(flat_param)))

        flat_grad = tree_flatten_and_concat(
            dict(
                network=grads.network_params,
                actor=grads.actor_params,
                critic=grads.critic_params,
            )
        )
        metrics_dict["grad_rms/avg"] = jnp.sqrt(jnp.mean(jnp.square(flat_grad)))
        metrics_dict["grad_rms/total"] = jnp.sqrt(jnp.sum(jnp.square(flat_grad)))

        agent_state = agent_state.apply_gradients(grads=grads)
        return agent_state, metrics_dict

    storage = jax.tree.map(lambda *x: jnp.hstack(x), *sharded_storages)
    storage_by_minibatches = jax.tree.map(lambda x: jnp.array(jnp.split(x, num_batches, axis=1)), storage)

    agent_state, loss_and_aux_per_step = jax.lax.scan(
        update_minibatch,
        agent_state,
        storage_by_minibatches,
    )

    aux_dict = jax.lax.pmean(loss_and_aux_per_step, axis_name=SINGLE_DEVICE_UPDATE_DEVICES_AXIS)
    aux_dict = jax.tree.map(jnp.mean, aux_dict)
    return (agent_state, key, aux_dict)
