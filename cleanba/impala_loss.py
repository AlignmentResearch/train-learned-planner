import abc
import dataclasses
from functools import partial
from typing import Any, Callable, List, Literal, NamedTuple, Self

import jax
import jax.numpy as jnp
import numpy as np
import optax
import rlax
from flax.training.train_state import TrainState
from numpy.typing import NDArray


class Rollout(NamedTuple):
    obs_t: jax.Array
    carry_t: Any
    a_t: jax.Array
    logits_t: jax.Array
    value_t: jax.Array
    r_t: jax.Array | NDArray
    episode_starts_t: jax.Array | NDArray
    truncated_t: jax.Array | NDArray


GetLogitsAndValueFn = Callable[
    [Any, Any, jax.Array, jax.Array | NDArray], tuple[Any, jax.Array, jax.Array, dict[str, jax.Array]]
]


@dataclasses.dataclass(frozen=True)
class ActorCriticLossConfig(abc.ABC):
    gamma: float = 0.99  # the discount factor gamma
    ent_coef: float = 0.01  # coefficient of the entropy
    vf_coef: float = 0.25  # coefficient of the value function

    normalize_advantage: bool = False

    @abc.abstractmethod
    def loss(
        self: Self,
        params: Any,
        get_logits_and_value: GetLogitsAndValueFn,
        minibatch: Rollout,
    ) -> tuple[jax.Array, dict[str, jax.Array]]:
        ...

    def maybe_normalize_advantage(self, adv_t: jax.Array) -> jax.Array:
        def _norm_advantage():
            return (adv_t - jnp.mean(adv_t)) / (jnp.std(adv_t, ddof=1) + 1e-8)

        return jax.lax.cond(self.normalize_advantage, _norm_advantage, lambda: adv_t)


@dataclasses.dataclass(frozen=True)
class ImpalaLossConfig(ActorCriticLossConfig):
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

    logit_l2_coef: float = 0.0
    weight_l2_coef: float = 0.0

    max_vf_error: float = 1.0  # The maximum value-function error allowed for a particular policy gradient to step.
    vf_loss_type: Literal["square", "huber"] = "square"
    advantage_multiplier: Literal["one", "mean", "max", "elementwise"] | str = "one"

    def vf_loss_fn(self, x: jax.Array) -> jax.Array:
        if self.vf_loss_type == "square":
            return jnp.square(x)
        elif self.vf_loss_type == "huber":
            return optax.losses.huber_loss(x)
        else:
            raise ValueError(f"{self.vf_loss_type=}")

    def adv_multiplier(self, vtrace_errors: jax.Array) -> jax.Array | float:
        if self.advantage_multiplier == "one":
            return 1.0
        elif self.advantage_multiplier == "mean":
            return jax.lax.stop_gradient(jnp.clip(self.max_vf_error / jnp.mean(jnp.abs(vtrace_errors)), a_max=1.0))
        elif self.advantage_multiplier == "max":
            return jax.lax.stop_gradient(jnp.clip(self.max_vf_error / jnp.max(jnp.abs(vtrace_errors)), a_max=1.0))
        elif self.advantage_multiplier.startswith("p"):
            n = int(self.advantage_multiplier[1:])
            pXX_err = jnp.percentile(jnp.abs(vtrace_errors), n)
            return jax.lax.stop_gradient(jnp.clip(self.max_vf_error / pXX_err, a_max=1.0))
        elif self.advantage_multiplier == "elementwise":
            return jax.lax.stop_gradient(jnp.clip(self.max_vf_error / jnp.abs(vtrace_errors), a_max=1.0))
        else:
            raise ValueError(f"{self.advantage_multiplier=}")

    # The reason this loss function is peppered with `del` statements is so we don't accidentally use the wrong
    # (time-shifted) variable when coding
    def loss(
        self: Self,
        params: Any,
        get_logits_and_value: GetLogitsAndValueFn,
        minibatch: Rollout,
    ) -> tuple[jax.Array, dict[str, jax.Array]]:
        # If the episode has actually terminated, the outgoing state's value is known to be zero.
        #
        # If the episode was truncated or terminated, we don't want the value estimation for future steps to influence the
        # value estimation for the current one (or previous once). I.e., in the VTrace (or GAE) recurrence, we want to stop
        # the value at time t+1 from influencing the value at time t and before.
        #
        # Both of these aims can be served by setting the discount to zero when the episode is terminated or truncated.
        #
        # done_t = truncated_t | terminated_t
        done_t = minibatch.episode_starts_t[1:]
        discount_t = (~done_t) * self.gamma
        del done_t

        _final_carry, nn_logits_from_obs, nn_value_from_obs, nn_metrics = get_logits_and_value(
            params, jax.tree.map(lambda x: x[0], minibatch.carry_t), minibatch.obs_t, minibatch.episode_starts_t
        )
        del _final_carry

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
        # We could get the correct TD error just by ignoring the loss at the truncated steps. However, VTrace propagates
        # errors backward, so the truncated-episode error would propagate backward anyways. To solve this, we set the reward
        # at truncated timesteps to be equal to v_tm1. The discount for those steps also has to be 0, that's determined by
        # `discount_t` defined above.
        mask_t = jnp.float32(~minibatch.truncated_t)
        r_t = jnp.where(minibatch.truncated_t, jax.lax.stop_gradient(v_tm1), minibatch.r_t)

        rhos_tm1 = rlax.categorical_importance_sampling_ratios(nn_logits_t, minibatch.logits_t, minibatch.a_t)

        vtrace_td_error_and_advantage = jax.vmap(
            partial(
                rlax.vtrace_td_error_and_advantage,
                lambda_=self.vtrace_lambda,
                clip_rho_threshold=self.clip_rho_threshold,
                clip_pg_rho_threshold=self.clip_pg_rho_threshold,
                stop_target_gradients=True,
            ),
            in_axes=1,
            out_axes=1,
        )

        vtrace_returns = vtrace_td_error_and_advantage(v_tm1, v_t, r_t, discount_t, rhos_tm1)

        # We're going to multiply advantages by this value, so the policy doesn't change too much in situations where the
        # value error is large.
        adv_multiplier = self.adv_multiplier(vtrace_returns.errors)

        # Policy-gradient loss: stop_grad(advantage) * log_p(actions), with importance ratios. The importance ratios here
        # are implicit in `pg_advs`.
        norm_advantage = self.maybe_normalize_advantage(vtrace_returns.pg_advantage)
        pg_advs = jax.lax.stop_gradient(adv_multiplier * norm_advantage)
        pg_loss = jnp.mean(jax.vmap(rlax.policy_gradient_loss, in_axes=1)(nn_logits_t, minibatch.a_t, pg_advs, mask_t))

        # Value loss: MSE/Huber loss of VTrace-estimated errors
        ## Errors should be zero where mask_t is False, but we multiply anyways
        v_loss = jnp.mean(self.vf_loss_fn(vtrace_returns.errors) * mask_t)

        # Entropy loss: negative average entropy of the policy across timesteps and environments
        ent_loss = jnp.mean(jax.vmap(rlax.entropy_loss, in_axes=1)(nn_logits_t, mask_t))

        total_loss = pg_loss
        total_loss += self.vf_coef * v_loss
        total_loss += self.ent_coef * ent_loss
        total_loss += self.logit_l2_coef * jnp.sum(jnp.square(nn_logits_from_obs))

        actor_params = jax.tree.leaves(params.get("params", {}).get("actor_params", {}))
        critic_params = jax.tree.leaves(params.get("params", {}).get("critic_params", {}))

        total_loss += self.weight_l2_coef * sum(jnp.sum(jnp.square(p)) for p in [*actor_params, *critic_params])

        # Useful metrics to know
        targets_tm1 = vtrace_returns.errors + v_tm1
        metrics_dict = dict(
            pg_loss=pg_loss,
            v_loss=v_loss,
            ent_loss=ent_loss,
            var_explained=1 - jnp.var(vtrace_returns.errors, ddof=1) / jnp.var(targets_tm1, ddof=1),
            proportion_of_boxes=jnp.mean(minibatch.r_t > 0),
            **nn_metrics,
            adv_multiplier=jnp.mean(adv_multiplier),
        )
        return total_loss, metrics_dict


@dataclasses.dataclass(frozen=True)
class PPOLossConfig(ActorCriticLossConfig):
    gae_lambda: float = 0.8
    clip_eps: float = 0.2
    vf_clip_eps: float = 0.2

    def loss(
        self: Self, params: Any, get_logits_and_value: GetLogitsAndValueFn, minibatch: Rollout
    ) -> tuple[jax.Array, dict[str, jax.Array]]:
        done_t = minibatch.episode_starts_t[1:]
        discount_t = (~done_t) * self.gamma
        del done_t

        _final_carry, nn_logits_from_obs, nn_value_from_obs, nn_metrics = get_logits_and_value(
            params, jax.tree.map(lambda x: x[0], minibatch.carry_t), minibatch.obs_t, minibatch.episode_starts_t
        )
        del _final_carry

        # There's one extra timestep at the end for `obs_t` than logits and the rest of objects in `minibatch`, so we need
        # to cut the logits to size.
        nn_logits_t = nn_logits_from_obs[:-1]
        # We keep the name error (t vs tm1) from the `rlax` library for consistence.
        nn_value_tm1 = nn_value_from_obs[:-1]
        minibatch_value_tm1 = jax.lax.stop_gradient(minibatch.value_t[:-1])

        # Ignore truncated steps using the same technique as before
        mask_t = jnp.float32(~minibatch.truncated_t)
        # This r_t cancels out exactly at truncated steps in the GAE calculation
        r_t = jnp.where(minibatch.truncated_t, jax.lax.stop_gradient(nn_value_from_obs[:-1]), minibatch.r_t)

        # Compute advantage and clipped value loss
        gae = jax.vmap(rlax.truncated_generalized_advantage_estimation, in_axes=(1, 1, None, 1, None), out_axes=1)(
            r_t, discount_t, self.gae_lambda, minibatch.value_t, True
        )
        value_targets = gae + minibatch_value_tm1

        value_errors = nn_value_tm1 - value_targets
        value_pred_clipped = minibatch_value_tm1 + jnp.clip(
            nn_value_tm1 - minibatch_value_tm1, -self.vf_clip_eps, self.vf_clip_eps
        )
        value_clipped_errors = value_pred_clipped - value_targets
        v_loss = jnp.mean(jnp.maximum(jnp.square(value_errors), jnp.square(value_clipped_errors)) * mask_t)

        rhos_t = rlax.categorical_importance_sampling_ratios(nn_logits_t, minibatch.logits_t, minibatch.a_t)
        adv_t = self.maybe_normalize_advantage(gae)

        clip_rhos_t = jnp.clip(rhos_t, 1.0 - self.clip_eps, 1.0 + self.clip_eps)
        policy_gradient = jnp.fmin(rhos_t * adv_t, clip_rhos_t * adv_t)
        pg_loss = -jnp.mean(policy_gradient * mask_t)

        # Entropy loss: negative average entropy of the policy across timesteps and environments
        ent_loss = jnp.mean(jax.vmap(rlax.entropy_loss, in_axes=1)(nn_logits_t, mask_t))

        total_loss = pg_loss
        total_loss += self.vf_coef * v_loss
        total_loss += self.ent_coef * ent_loss
        metrics_dict = dict(
            pg_loss=pg_loss,
            v_loss=v_loss,
            ent_loss=ent_loss,
            var_explained=1 - jnp.var(value_errors, ddof=1) / jnp.var(value_targets, ddof=1),
        )
        return total_loss, metrics_dict


SINGLE_DEVICE_UPDATE_DEVICES_AXIS: str = "local_devices"


def tree_flatten_and_concat(x) -> jax.Array:
    leaves, _ = jax.tree.flatten(x)
    return jnp.concat(list(map(jnp.ravel, leaves)))


def single_device_update(
    agent_state: TrainState,
    sharded_storages: List[Rollout],
    *,
    get_logits_and_value: GetLogitsAndValueFn,
    num_batches: int,
    impala_cfg: ImpalaLossConfig,
) -> tuple[TrainState, dict[str, jax.Array]]:
    def update_minibatch(agent_state: TrainState, minibatch: Rollout):
        (loss, metrics_dict), grads = jax.value_and_grad(impala_cfg.loss, has_aux=True)(
            agent_state.params,
            get_logits_and_value,
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

        # [DISABLED] per-parameter grads
        # named_params, _ = jax.tree_util.tree_flatten_with_path(grads)

        # for k, v in named_params:
        #     metrics_dict[f"grad_rms/{jax.tree_util.keystr(k)}"] = jnp.sqrt(jnp.mean(jnp.square(v)))

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
