import collections
import dataclasses
import queue
from functools import partial
from pathlib import Path
from typing import Optional, Sequence

import flax
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax
import rlax
import torch.utils.data
from chex import Numeric
from flax.training.train_state import TrainState

import cleanba.cleanba_impala as cleanba_impala
from cleanba.config import Args, sokoban_resnet
from cleanba.environments import EnvpoolBoxobanConfig
from cleanba.impala_loss import (
    ImpalaLossConfig,
    Rollout,
    impala_loss,
)
from cleanba.network import RMSNorm, SokobanResNetConfig, label_and_learning_rate_for_params


def unreplicate(tree):
    """Returns a single instance of a replicated array."""
    return jax.tree_util.tree_map(lambda x: x[0], tree)


# %%


def collect_rollouts(args: Args, num_rollouts: int = 2, seed: int = 12345) -> list[Rollout]:
    key = jax.random.PRNGKey(seed)

    key, params_key = jax.random.split(key, 2)
    rollout_params = jax.jit(args.net.init_params, static_argnums=(0,))(
        dataclasses.replace(args.train_env, num_envs=1).make(),
        params_key,
    )
    param_queue = queue.Queue(num_rollouts)
    for _ in range(num_rollouts):
        param_queue.put_nowait(rollout_params)

    rollout_queue = queue.Queue(num_rollouts + 2)
    cleanba_impala.rollout(
        key,
        args,
        cleanba_impala.RuntimeInformation(
            0,
            [],
            0,
            1,
            0,
            args.local_num_envs,
            0,
            0,
            num_updates=num_rollouts,
            global_learner_devices=[],
            learner_devices=jax.devices(),
        ),
        rollout_queue=rollout_queue,
        params_queue=param_queue,
        writer=None,
        learner_devices=jax.devices(),
        device_thread_id=0,
        actor_device=None,
    )
    out = []
    while not rollout_queue.empty():
        (
            global_step,
            actor_policy_version,
            update,
            sharded_storage,
            _params_queue_get_time,
            device_thread_id,
        ) = rollout_queue.get_nowait()
        out.append(sharded_storage)

    return out


# %% Config

args: Args = sokoban_resnet()

args = dataclasses.replace(
    args,
    # train_env=SokobanConfig(max_episode_steps=120, num_envs=1, tinyworld_obs=True, dim_room=(10, 10), asynchronous=False),
    train_env=EnvpoolBoxobanConfig(
        max_episode_steps=30,
        min_episode_steps=20,
        cache_path=Path("/training/.sokoban_cache"),
        split="train",
        difficulty="unfiltered",
    ),
    eval_envs={},
    net=SokobanResNetConfig(yang_init=True, norm=RMSNorm(), mlp_hiddens=(256,)),
    local_num_envs=256,
    num_steps=20,
    log_frequency=1000000000,
)
# %% Collect rollouts

ROLLOUT_PATH: Path = Path("/workspace/rollouts/")
ROLLOUT_PATH.mkdir(exist_ok=True, parents=True)

rollout_files = list(ROLLOUT_PATH.iterdir())

NUM_ROLLOUTS = 3000
if len(rollout_files) < NUM_ROLLOUTS:
    rollouts = collect_rollouts(args, num_rollouts=NUM_ROLLOUTS - len(rollout_files))

    for i, r in enumerate(rollouts):
        rollout_idx = len(rollout_files) + i
        with (ROLLOUT_PATH / f"rollout_{rollout_idx:05d}.msgpack").open("wb") as f:
            leaves = jax.tree.leaves(r)
            f.write(flax.serialization.to_bytes(leaves))

print(f"{len(rollout_files)=}")

# %% Create training and test datasets


class RolloutDataset(torch.utils.data.Dataset):
    def __init__(self, fnames: Sequence[Path | str]):
        self.names = np.array([str(f) for f in fnames])
        self._structure = jax.tree.structure(Rollout(2, 2, 2, 2, 2, 2))
        assert len(self.names.shape) == 1

        self._leaves: Optional[dict[str, jax.Array]] = None
        self._last_loaded_file = -1

        self._rollouts_per_point = args.local_num_envs

    def __getitem__(self, i):
        file_idx = i // self._rollouts_per_point
        within_idx = i % self._rollouts_per_point

        if self._last_loaded_file != file_idx:
            with open(str(self.names[file_idx]), "rb") as f:
                leaves = flax.serialization.msgpack_restore(f.read())
            self._leaves = leaves
            self._last_loaded_file = file_idx
            assert leaves["0"].shape[2] == self._rollouts_per_point, leaves["0"].shape

        return jax.tree.unflatten(self._structure, [le[0, :, within_idx, ...] for le in self._leaves.values()])

    def __len__(self):
        return len(self.names) * self._rollouts_per_point


@jax.jit
def rollout_collate_fn(x: list[Rollout]) -> Rollout:
    out = jax.tree.map(lambda *xs: jnp.stack(xs, axis=1), *x)
    return out


np_random = np.random.default_rng(seed=12345)
_all_data = rollout_files
np_random.shuffle(_all_data)
_train_data_names = _all_data[: len(_all_data) * 9 // 10]

train_data = RolloutDataset(_train_data_names)
test_data = RolloutDataset(_all_data[len(_train_data_names) :])
print(f"{len(train_data)=}, {len(test_data)=}")

# %% Initialize parameters


def learning_rate_schedule(steps: Numeric) -> Numeric:
    return 0.01


def adam_with_parameters(learning_rate, b1=0.95, b2=0.99, eps=1e-8, eps_root=0.0) -> optax.GradientTransformation:
    return optax.inject_hyperparams(optax.adam)(learning_rate=learning_rate, b1=b1, b2=b2, eps=eps, eps_root=eps_root)


params = jax.jit(args.net.init_params, static_argnums=(0,))(
    dataclasses.replace(args.train_env, num_envs=1).make(), jax.random.PRNGKey(1234)
)

learning_rates, param_labels = label_and_learning_rate_for_params(params)

train_state = TrainState.create(
    apply_fn=None,
    params=params,
    tx=optax.MultiSteps(
        optax.chain(
            # optax.clip_by_global_norm(0.0625),
            optax.multi_transform(
                transforms={k: adam_with_parameters(lr) for k, lr in learning_rates.items()},
                param_labels=param_labels,
            ),
        ),
        every_k_schedule=1,
    ),
)
del params


@jax.jit
def update_minibatch_impala(train_state: TrainState, minibatch: Rollout):
    minibatch = unreplicate(minibatch)

    (loss, metrics_dict), grads = jax.value_and_grad(impala_loss, has_aux=True)(
        train_state.params,
        args.net.get_logits_and_value,
        ImpalaLossConfig(gamma=0.97, vf_coef=1.0, vtrace_lambda=0.97),
        minibatch,
    )
    metrics_dict["loss"] = loss
    train_state = train_state.apply_gradients(grads=grads)
    return train_state, metrics_dict


@jax.jit
def update_minibatch(train_state: TrainState, minibatch: Rollout):
    def loss_fn(params, get_logits_and_value, cfg: ImpalaLossConfig, minibatch: Rollout):
        mask_t = jnp.float32(~minibatch.truncated_t)
        discount_t = (~minibatch.done_t) * cfg.gamma
        nn_logits_from_obs, nn_value_from_obs, nn_metrics = jax.vmap(get_logits_and_value, in_axes=(None, 0))(
            params, minibatch.obs_t
        )
        nn_logits_t = nn_logits_from_obs[:-1]
        del nn_logits_from_obs
        v_t = nn_value_from_obs[1:]
        v_tm1 = nn_value_from_obs[:-1]
        del nn_value_from_obs
        rhos_tm1 = rlax.categorical_importance_sampling_ratios(nn_logits_t, minibatch.logits_t, minibatch.a_t)

        vtrace_td_error_and_advantage = jax.vmap(
            partial(
                rlax.vtrace_td_error_and_advantage,
                lambda_=cfg.vtrace_lambda,
                clip_rho_threshold=cfg.clip_rho_threshold,
                clip_pg_rho_threshold=cfg.clip_pg_rho_threshold,
                stop_target_gradients=True,
            ),
            in_axes=1,
            out_axes=1,
        )
        vtrace_returns = vtrace_td_error_and_advantage(v_tm1, v_t, minibatch.r_t, discount_t, rhos_tm1)

        pg_advs = vtrace_returns.pg_advantage
        pg_loss = jnp.mean(jax.vmap(rlax.policy_gradient_loss, in_axes=1)(nn_logits_t, minibatch.a_t, pg_advs, mask_t))
        v_loss = jnp.mean(jnp.square(vtrace_returns.errors) * mask_t)
        ent_loss = jnp.mean(jax.vmap(rlax.entropy_loss, in_axes=1)(nn_logits_t, mask_t))

        total_loss = pg_loss
        total_loss += cfg.vf_coef * v_loss
        total_loss += cfg.ent_coef * ent_loss

        # end copy original impala

        errors_ms = v_loss
        targets_tm1 = vtrace_returns.errors + v_tm1
        targets_ms = jnp.mean(jnp.square(targets_tm1) * mask_t)
        # It doesn't matter what the denominator for these two is so long as it's the same for every batch. We're just
        # going to add all of the batches and divide

        positive_mask = targets_tm1 > 0
        mean_positive = jnp.sum(v_tm1 * positive_mask) / jnp.sum(positive_mask)
        mean_negative = jnp.sum(v_tm1 * (~positive_mask)) / jnp.sum(~positive_mask)
        return total_loss, dict(
            errors_ms=errors_ms, targets_ms=targets_ms, mean_negative=mean_negative, mean_positive=mean_positive
        )

    (loss, metrics_dict), grads = jax.value_and_grad(loss_fn, has_aux=True)(
        train_state.params,
        args.net.get_logits_and_value,
        ImpalaLossConfig(gamma=0.97, vf_coef=1.0, vtrace_lambda=0.97),
        minibatch,
    )
    metrics_dict["loss"] = loss
    train_state = train_state.apply_gradients(grads=grads)
    return train_state, metrics_dict


BATCH_SIZE = 32
train_loader = torch.utils.data.DataLoader(
    train_data, batch_size=BATCH_SIZE, shuffle=False, collate_fn=rollout_collate_fn, num_workers=0
)


def evaluate(train_state, num_evaluate=5):
    test_loader = torch.utils.data.DataLoader(
        test_data, batch_size=BATCH_SIZE, shuffle=False, collate_fn=rollout_collate_fn, num_workers=0
    )

    errors_ms = 0.0
    targets_ms = 0.0

    out = collections.defaultdict(list)

    it = iter(test_loader)
    for _ in range(num_evaluate):
        rollout = next(it)
        _, metrics = update_minibatch(train_state, rollout)

        errors_ms += float(metrics.pop("errors_ms"))
        targets_ms += float(metrics.pop("targets_ms"))

        for k, v in metrics.items():
            out[k].append(v)

    out_metrics = {k: float(np.mean(v)) for k, v in out.items()}
    out_metrics["fve"] = 1.0 - errors_ms / targets_ms
    out_metrics["errors_ms"] = errors_ms / num_evaluate
    out_metrics["targets_ms"] = targets_ms / num_evaluate
    return out_metrics


metrics = collections.defaultdict(list)


def add_metrics(d: dict[str, jax.Array]):
    for k, v in d.items():
        metrics[k].append(v.item())


# Only do 1 epoch. We want to make sure we can learn within 1 epoch of all these steps -- this corresponds to 10 million
# steps
metrics_dict = {}
for i, rollout in enumerate(train_loader):
    if metrics_dict:
        metrics_dict["fve"] = 1.0 - metrics_dict["errors_ms"] / metrics_dict["targets_ms"]
    add_metrics(metrics_dict)  # Do it here, so we load the next points immediately.
    assert isinstance(rollout, Rollout)
    if i % 100 == 99:
        print(f"{i=}, loss={np.mean(metrics['loss'][-10:])}")
        # rich.pretty.pprint(evaluate(train_state, num_evaluate=5))

    train_state, metrics_dict = update_minibatch(train_state, rollout)

# %%

window = np.ones((50,)) / 50

_, axes = plt.subplots(2, 1)

steps_per_step = BATCH_SIZE * 20
ax = axes[0]
for metric in ["fve"]:
    ax.plot(np.arange(len(metrics[metric])) * steps_per_step, metrics[metric], label=metric)
ax.legend()

ax = axes[1]
ax.set_ylim(-0.1, 10)
for metric in ["loss", "errors_ms", "targets_ms"]:
    ax.plot(np.arange(len(metrics[metric])) * steps_per_step, metrics[metric], label=metric)
ax.legend()
