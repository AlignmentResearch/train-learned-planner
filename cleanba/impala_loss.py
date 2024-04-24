import dataclasses
from functools import partial
from typing import Any, Callable, List, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import rlax
from flax.training.train_state import TrainState
from numpy.typing import NDArray

from cleanba.network import AgentParams


@dataclasses.dataclass(frozen=True)
class ImpalaLossConfig:
    gamma: float = 0.99  # the discount factor gamma
    ent_coef: float = 0.01  # coefficient of the entropy
    vf_coef: float = 0.25  # coefficient of the value function

    # Interpolate between VTrace (1.0) and monte-carlo function (0.0) estimates, for the estimate of targets, used in
    # both the value and policy losses. It's the parameter in Remark 2 of Espeholt et al.
    # (https://arxiv.org/pdf/1802.01561.pdf)
    vtrace_lambda: float = 1.0

    # Maximum importance ratio for the VTrace value estimates. This is \overline{rho} in eq. 1 of Espeholt et al.
    # (https://arxiv.org/pdf/1802.01561.pdf). \overline{c} is hardcoded to 1 in rlax.
    clip_rho_threshold: float = 1.0

    # Maximum importance ratio for policy gradient outputs. Clips the importance ratio in eq. 4 of Espeholt et al.
    # (https://arxiv.org/pdf/1802.01561.pdf)
    clip_pg_rho_threshold: float = 1.0

    normalize_advantage: bool = False

    logit_l2_coef: float = 0.0
    weight_l2_coef: float = 0.0


class Rollout(NamedTuple):
    obs_t: jax.Array
    a_t: jax.Array
    logits_t: jax.Array
    r_t: jax.Array | NDArray
    done_t: jax.Array | NDArray
    truncated_t: jax.Array | NDArray


def impala_loss(
    params: AgentParams,
    get_logits_and_value: Callable[[Any, jax.Array], tuple[jax.Array, jax.Array, dict[str, jax.Array]]],
    args: ImpalaLossConfig,
    minibatch: Rollout,
) -> tuple[jax.Array, dict[str, jax.Array]]:
    # If the episode has been truncated, the value of the next state (after truncation) would be some non-zero amount.
    # But we don't have access to that state, because the resetting code just throws it away. To compensate, we'll
    # actually truncate 1 step earlier than the time limit, use the value of the state we know, and just discard the
    # transition that actually caused truncation. That is, we receive:
    #
    #     s0, --r0-> s1 --r1-> s2 --r2-> s3 --r3-> ...
    #
    # and say the episode was truncated at s3. We don't know s4, so we can't calculate V(s4), which we need for the
    # objective. So instead we'll discard r3 and treat s3 as the final state. Now we can calculate V(s3).
    #
    # We can implement this by just not using the loss at the truncated steps.
    mask_t = jnp.float32(~minibatch.truncated_t)

    # If the episode has actually terminated, the outgoing state's value is known to be zero.
    #
    # If the episode was truncated or terminated, we don't want the value estimation for future steps to influence the
    # value estimation for the current one (or previous once). I.e., in the VTrace (or GAE) recurrence, we want to stop
    # the value at time t+1 from influencing the value at time t and before.
    #
    # Both of these aims can be served by setting the discount to zero when the episode is terminated or truncated.
    #
    # done_t = truncated_t | terminated_t
    discount_t = (~minibatch.done_t) * args.gamma

    nn_logits_from_obs, nn_value_from_obs, nn_metrics = jax.vmap(get_logits_and_value, in_axes=(None, 0))(
        params, minibatch.obs_t
    )

    # There's one extra timestep at the end for `obs_t` than logits and the rest of objects in `minibatch`, so we need
    # to cut these values to size.
    #
    # For the logits, we discard the last one, which makes the time-steps of `nn_logits_from_obs` match exactly with the
    # time-steps from `minibatch.logits_t`
    nn_logits_t = nn_logits_from_obs[:-1]

    ## Remark 1:
    # v_t does not enter the gradient in any way, because
    #   1. it's stop_grad()-ed in the `vtrace_td_error_and_advantage.errors`
    #   2. it intervenes in `vtrace_td_error_and_advantage.pg_advantage`, but that's stop_grad() ed by the pg loss.
    #
    # so we don't actually need to call stop_grad here.
    #
    ## Remark 2:
    # If we followed normal RL conventions, v_t corresponds to V(s_{t+1}) and v_tm1 corresponds to V(s_{t}). This can be
    # gleaned from looking at the implementation of the TD error in `rlax.vtrace`.
    #
    # We keep the name error from the `rlax` library for consistence.
    v_t = nn_value_from_obs[1:]
    v_tm1 = nn_value_from_obs[:-1]
    del nn_value_from_obs

    rhos_tm1 = rlax.categorical_importance_sampling_ratios(nn_logits_t, minibatch.logits_t, minibatch.a_t)

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
    vtrace_returns = vtrace_td_error_and_advantage(v_tm1, v_t, minibatch.r_t, discount_t, rhos_tm1)

    # Policy-gradient loss: stop_grad(advantage) * log_p(actions), with importance ratios. The importance ratios here
    # are implicit in `pg_advs`.
    norm_advantage = (vtrace_returns.pg_advantage - jnp.mean(vtrace_returns.pg_advantage)) / (
        jnp.std(vtrace_returns.pg_advantage, ddof=1) + 1e-8
    )
    pg_advs = jax.lax.select(args.normalize_advantage, norm_advantage, vtrace_returns.pg_advantage)
    pg_loss = jnp.mean(jax.vmap(rlax.policy_gradient_loss, in_axes=1)(nn_logits_t, minibatch.a_t, pg_advs, mask_t))

    # Value loss: MSE of VTrace-estimated errors
    v_loss = jnp.mean(jnp.square(vtrace_returns.errors) * mask_t)

    # Entropy loss: negative average entropy of the policy across timesteps and environments
    ent_loss = jnp.mean(jax.vmap(rlax.entropy_loss, in_axes=1)(nn_logits_t, mask_t))

    total_loss = pg_loss
    total_loss += args.vf_coef * v_loss
    total_loss += args.ent_coef * ent_loss
    total_loss += args.logit_l2_coef * jnp.sum(jnp.square(nn_logits_from_obs))
    total_loss += args.weight_l2_coef * sum(
        jnp.sum(jnp.square(p)) for p in [*jax.tree.leaves(params.actor_params), *jax.tree.leaves(params.critic_params)]
    )

    # Useful metrics to know
    targets_tm1 = vtrace_returns.errors + v_tm1
    metrics_dict = dict(
        pg_loss=pg_loss,
        v_loss=v_loss,
        ent_loss=ent_loss,
        max_ratio=jnp.max(rhos_tm1),
        min_ratio=jnp.min(rhos_tm1),
        avg_ratio=jnp.exp(jnp.mean(jnp.log(rhos_tm1))),
        var_explained=1 - jnp.var(vtrace_returns.errors, ddof=1) / jnp.var(targets_tm1, ddof=1),
        proportion_of_boxes=jnp.mean(minibatch.r_t > 0),
        mean_error=jnp.mean(vtrace_returns.errors),
        **nn_metrics,
    )
    return total_loss, metrics_dict


SINGLE_DEVICE_UPDATE_DEVICES_AXIS: str = "local_devices"


def tree_flatten_and_concat(x) -> jax.Array:
    leaves, _ = jax.tree.flatten(x)
    return jnp.concat(list(map(jnp.ravel, leaves)))


@partial(jax.jit, static_argnames=["num_batches", "get_logits_and_value", "impala_cfg"])
def single_device_update(
    agent_state: TrainState,
    sharded_storages: List[Rollout],
    *,
    get_logits_and_value: Callable[[Any, jax.Array], tuple[jax.Array, jax.Array, dict[str, jax.Array]]],
    num_batches: int,
    impala_cfg: ImpalaLossConfig,
) -> tuple[TrainState, dict[str, jax.Array]]:
    def update_minibatch(agent_state: TrainState, minibatch: Rollout):
        (loss, metrics_dict), grads = jax.value_and_grad(impala_loss, has_aux=True)(
            agent_state.params,
            get_logits_and_value,
            impala_cfg,
            minibatch,
        )
        metrics_dict["loss"] = loss
        grads = jax.lax.pmean(grads, axis_name=SINGLE_DEVICE_UPDATE_DEVICES_AXIS)

        flat_param = tree_flatten_and_concat(agent_state.params)
        metrics_dict["param_rms/avg"] = jnp.sqrt(jnp.mean(jnp.square(flat_param)))
        metrics_dict["param_rms/total"] = metrics_dict["param_rms/avg"] * np.sqrt(np.prod(flat_param.shape))

        flat_grad = tree_flatten_and_concat(grads)
        metrics_dict["grad_rms/avg"] = jnp.sqrt(jnp.mean(jnp.square(flat_grad)))
        metrics_dict["grad_rms/total"] = metrics_dict["grad_rms/avg"] * np.sqrt(np.prod(flat_grad.shape))

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
    return (agent_state, aux_dict)
