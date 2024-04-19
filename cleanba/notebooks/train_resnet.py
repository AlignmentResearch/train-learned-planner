import collections
from pathlib import Path

import flax
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax
import torch.utils.data
from chex import Numeric
from flax.training.train_state import TrainState

from cleanba.environments import SokobanConfig
from cleanba.impala_loss import (
    ImpalaLossConfig,
    Rollout,
    impala_loss,
)
from cleanba.network import IdentityNorm, SokobanResNetConfig, label_and_learning_rate_for_params


def unreplicate(tree):
    """Returns a single instance of a replicated array."""
    return jax.tree_util.tree_map(lambda x: x[0], tree)


# %%

train_env_cfg = SokobanConfig(max_episode_steps=120, num_envs=1, tinyworld_obs=True, dim_room=(10, 10), asynchronous=False)
envs = train_env_cfg.make()


# %%
rollout_files = []

for p in Path("/workspace/rollouts/devbox/wandb/").iterdir():
    if p.is_dir() and not p.is_symlink():
        for pp in (p / "local-files").iterdir():
            for f in pp.iterdir():
                if f.is_file():
                    rollout_files.append(f)

# %% Create training and test datasets


class RolloutDataset(torch.utils.data.Dataset):
    def __init__(self, fnames):
        self.names = np.array([str(f) for f in fnames])
        self._structure = jax.tree.structure(Rollout(2, 2, 2, 2, 2, 2))
        assert len(self.names.shape) == 1

    def __getitem__(self, i):
        with open(str(self.names[i]), "rb") as f:
            leaves = flax.serialization.msgpack_restore(f.read())
        return jax.tree.unflatten(self._structure, leaves)

    def __len__(self):
        return len(self.names)


np_random = np.random.default_rng(seed=12345)
_all_data = rollout_files
np_random.shuffle(_all_data)

train_data = RolloutDataset(_all_data[: len(_all_data) * 8 // 10])
test_data = RolloutDataset(_all_data[len(train_data) :])
print(f"{len(train_data)=}, {len(test_data)=}")

# %% Initialize parameters

net = SokobanResNetConfig(
    multiplicity=2, kernel_sizes=(3, 3), channels=(64, 64), strides=(1, 1), norm=IdentityNorm(), mlp_hiddens=(256,)
)


def learning_rate_schedule(steps: Numeric) -> Numeric:
    return 0.01


def adam_with_parameters(learning_rate, b1=0.95, b2=0.99, eps=1e-8, eps_root=0.0) -> optax.GradientTransformation:
    return optax.inject_hyperparams(optax.adam)(learning_rate=learning_rate, b1=b1, b2=b2, eps=eps, eps_root=eps_root)


params = jax.jit(net.init_params, static_argnums=(0,))(envs, jax.random.PRNGKey(1234), train_data[0].obs_t[0, 0, :1])

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

# %% Supervised training loop

metrics = collections.defaultdict(list)


def add_metrics(d: dict[str, jax.Array]):
    for k, v in d.items():
        metrics[k].append(v.item())


@jax.jit
def update_minibatch_impala(train_state: TrainState, minibatch: Rollout):
    minibatch = unreplicate(minibatch)

    (loss, metrics_dict), grads = jax.value_and_grad(impala_loss, has_aux=True)(
        train_state.params,
        net.get_logits_and_value,
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
        assert len(mask_t.shape) == 2 and mask_t.shape[0] == 20, mask_t.shape
        # discount_t = (~minibatch.done_t) * cfg.gamma

        nn_logits_from_obs, nn_value_from_obs = jax.vmap(get_logits_and_value, in_axes=(None, 0))(params, minibatch.obs_t)
        # nn_logits_t = nn_logits_from_obs[:-1]
        del nn_logits_from_obs

        # v_t = jnp.zeros_like(nn_value_from_obs[1:])  # KEY: make next state value estimate equal to 0
        v_tm1 = nn_value_from_obs[:-1]
        del nn_value_from_obs

        # rhos_tm1 = rlax.categorical_importance_sampling_ratios(nn_logits_t, minibatch.logits_t, minibatch.a_t)

        errors = minibatch.r_t - v_tm1
        # errors = jax.vmap(partial(rlax.vtrace, lambda_=cfg.vtrace_lambda), in_axes=1, out_axes=1)(
        #     v_tm1, v_t, minibatch.r_t, discount_t, rhos_tm1
        # )
        loss = jnp.mean(jnp.square(errors) * mask_t)

        targets_tm1 = errors + v_tm1
        var_explained = 1 - jnp.var(errors, ddof=1) / jnp.var(targets_tm1, ddof=1)

        positive_mask = minibatch.r_t > 0
        mean_positive = jnp.sum(v_tm1 * positive_mask) / jnp.sum(positive_mask)
        mean_negative = jnp.sum(v_tm1 * (~positive_mask)) / jnp.sum(~positive_mask)
        return loss, dict(var_explained=var_explained, mean_negative=mean_negative, mean_positive=mean_positive)

    (loss, metrics_dict), grads = jax.value_and_grad(loss_fn, has_aux=True)(
        train_state.params,
        net.get_logits_and_value,
        ImpalaLossConfig(gamma=0.97, vf_coef=1.0, vtrace_lambda=0.97),
        minibatch,
    )
    metrics_dict["loss"] = loss
    train_state = train_state.apply_gradients(grads=grads)
    return train_state, metrics_dict


@jax.jit
def collate_fn(x: list[Rollout]) -> Rollout:
    out = jax.tree.map(lambda *xs: jnp.concatenate([x.squeeze(0) for x in xs], axis=1), *x)
    return out


train_loader = torch.utils.data.DataLoader(train_data, batch_size=16, shuffle=True, collate_fn=collate_fn)

for epoch in range(1, 50):
    for i, rollout in enumerate(train_loader):
        assert isinstance(rollout, Rollout)
        train_state, metrics_dict = update_minibatch(train_state, rollout)
        add_metrics(metrics_dict)
    print(f"Epoch {epoch}, {metrics_dict}")

# %%
plt.ylim(-1, 10)
window = np.ones((50)) / 50
plt.plot(np.convolve(metrics["mean_positive"], window))

# %%

test_loader = torch.utils.data.DataLoader(test_data, batch_size=4, shuffle=False, collate_fn=collate_fn)


def print_total_metrics(name, loader):
    positive_count = 0.0
    negative_count = 0.0
    positive_total = 0.0
    negative_total = 0.0
    loss_total = 0.0

    for test_batch in loader:
        _, value = jax.jit(jax.vmap(net.get_logits_and_value, in_axes=(None, 0)))(train_state.params, test_batch.obs_t[:-1])
        positive = test_batch.r_t > 0
        positive_count += np.sum(positive)
        negative_count += np.sum(~positive)

        value = np.array(value)
        positive_total += np.sum(value * positive)
        negative_total += np.sum(value * (~positive))

        loss_total += jnp.mean(jnp.square(test_batch.r_t - value))

    print(f"{name} loss:", loss_total / len(test_loader))
    print(f"{name} positive avg:", positive_total / positive_count)
    print(f"{name} negative avg:", negative_total / negative_count)


print_total_metrics("test", test_loader)
print_total_metrics("train", train_loader)
