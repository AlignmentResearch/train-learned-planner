# %%
import dataclasses
import tempfile
from functools import partial
from pathlib import Path
from typing import Callable, Dict, Optional, Sequence

import flax.linen as nn
import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from gymnasium import spaces
from gymnasium.envs.classic_control.cartpole import CartPoleEnv
from gymnasium.wrappers import TimeLimit

import cleanba.cleanba_impala
from cleanba.cleanba_impala import WandbWriter, train
from cleanba.config import Args
from cleanba.convlstm import ConvConfig, ConvLSTMConfig
from cleanba.environments import EnvConfig
from cleanba.impala_loss import ImpalaConfig
from cleanba.network import GuezResNetConfig, PolicySpec


# %%
class DataFrameWriter(WandbWriter):
    def __init__(self, cfg: Args, save_dir: Path):
        self.metrics = pd.DataFrame()
        self.states = {}
        self._save_dir = save_dir

    def add_scalar(self, name: str, value: int | float, global_step: int):
        try:
            value = list(value)
        except TypeError:
            self.metrics.loc[global_step, name] = value
            return

        for i, v in enumerate(value):
            try:
                a = v.item()
                self.metrics.loc[global_step + 640 * i, name] = a
            except (TypeError, AttributeError, ValueError):
                self.states[global_step + 640 * i, name] = value


# %%
if "CartPoleNoVel-v0" not in gym.registry or "CartPoleCHW-v0" not in gym.registry:

    class CartPoleCHWEnv(CartPoleEnv):
        """Variant of CartPoleEnv with velocity information removed, and CHW-shaped observations.
        This task requires memory to solve."""

        def __init__(self):
            super().__init__()
            high = np.array(
                [
                    self.x_threshold * 2,
                    3.4028235e38,
                    self.theta_threshold_radians * 2,
                    3.4028235e38,
                ],
                dtype=np.float32,
            )[:, None, None]
            self.observation_space = spaces.Box(-high, high, dtype=np.float32)

        @staticmethod
        def _pos_obs(full_obs):
            return np.array(full_obs)[:, None, None] * 255.0

        def reset(self, *, seed: Optional[int] = None, options: Optional[Dict] = None):
            full_obs, info = super().reset(seed=seed, options=options)
            return CartPoleCHWEnv._pos_obs(full_obs), info

        def step(self, action):
            full_obs, rew, terminated, truncated, info = super().step(action)
            return CartPoleCHWEnv._pos_obs(full_obs), rew / 500, terminated, truncated, info

    class CartPoleNoVelEnv(CartPoleEnv):
        """Variant of CartPoleEnv with velocity information removed, and CHW-shaped observations.
        This task requires memory to solve."""

        def __init__(self):
            super().__init__()
            high = np.array(
                [
                    self.x_threshold * 2,
                    self.theta_threshold_radians * 2,
                ],
                dtype=np.float32,
            )[:, None, None]
            self.observation_space = spaces.Box(-high, high, dtype=np.float32)

        @staticmethod
        def _pos_obs(full_obs):
            xpos, _xvel, thetapos, _thetavel = full_obs
            return np.array([xpos, thetapos])[:, None, None] * 255.0

        def reset(self, *, seed: Optional[int] = None, options: Optional[Dict] = None):
            full_obs, info = super().reset(seed=seed, options=options)
            return CartPoleNoVelEnv._pos_obs(full_obs), info

        def step(self, action):
            full_obs, rew, terminated, truncated, info = super().step(action)
            return CartPoleNoVelEnv._pos_obs(full_obs), rew / 500, terminated, truncated, info

    gym.register(
        id="CartPoleNoVel-v0",
        entry_point=CartPoleNoVelEnv,
        max_episode_steps=500,
    )

    gym.register(
        id="CartPoleCHW-v0",
        entry_point=CartPoleCHWEnv,
        max_episode_steps=500,
    )


class CartPoleNoVelConfig(EnvConfig):
    @property
    def make(self) -> Callable[[], gym.vector.VectorEnv]:
        def tl_wrapper(env_fn):
            return TimeLimit(env_fn(), max_episode_steps=self.max_episode_steps)

        return partial(gym.vector.SyncVectorEnv, env_fns=[partial(tl_wrapper, CartPoleNoVelEnv)] * self.num_envs)


class CartPoleConfig(EnvConfig):
    @property
    def make(self) -> Callable[[], gym.vector.VectorEnv]:
        def tl_wrapper(env_fn):
            return TimeLimit(env_fn(), max_episode_steps=self.max_episode_steps)

        return partial(gym.vector.SyncVectorEnv, env_fns=[partial(tl_wrapper, CartPoleCHWEnv)] * self.num_envs)


class MountainCarNormalized(gym.envs.classic_control.MountainCarEnv):
    def step(self, action):
        full_obs, rew, terminated, truncated, info = super().step(action)
        return full_obs, rew, terminated, truncated, info


class MountainCarConfig(EnvConfig):
    max_episode_steps: int = 200

    @property
    def make(self) -> Callable[[], gym.vector.VectorEnv]:
        def tl_wrapper(env_fn):
            return TimeLimit(env_fn(), max_episode_steps=self.max_episode_steps)

        return partial(gym.vector.SyncVectorEnv, env_fns=[partial(tl_wrapper, MountainCarNormalized)] * self.num_envs)


# %%


@dataclasses.dataclass(frozen=True)
class MLPConfig(PolicySpec):
    hiddens: Sequence[int] = (64, 64)

    def make(self) -> nn.Module:
        return MLP(self)


class MLP(nn.Module):
    cfg: MLPConfig

    @nn.compact
    def __call__(self, x):
        for h in self.cfg.hiddens:
            x = nn.Dense(h)(x)
            x = nn.relu(x)
        return x


def train_cartpole_no_vel(policy="resnet", env="cartpole"):
    if policy == "resnet":
        net = GuezResNetConfig(
            channels=(),
            strides=(1,),
            kernel_sizes=(1,),
            mlp_hiddens=(256, 256),
            normalize_input=False,
        )
    else:
        net = ConvLSTMConfig(
            embed=[],
            recurrent=[ConvConfig(64, (1, 1), (1, 1), "SAME", True)],
            repeats_per_step=1,
            pool_and_inject=False,
            add_one_to_forget=True,
        )
    NUM_ENVS = 8
    if env == "cartpole":
        env_cfg = CartPoleConfig(num_envs=NUM_ENVS, max_episode_steps=500, seed=1234)
    else:
        env_cfg = CartPoleNoVelConfig(num_envs=NUM_ENVS, max_episode_steps=500, seed=1234)

    args = Args(
        seed=13246,
        train_env=env_cfg,
        eval_envs={},
        net=net,
        eval_frequency=int(1e9),
        save_model=False,
        log_frequency=50,
        local_num_envs=NUM_ENVS,
        num_actor_threads=1,
        num_minibatches=1,
        # If the whole thing deadlocks exit in some small multiple of 10 seconds
        queue_timeout=60,
        train_epochs=1,
        num_steps=32,
        learning_rate=0.001,
        anneal_lr=True,
        total_timesteps=1_000_000,
        max_grad_norm=0.5,
        base_fan_in=1,
        optimizer="adam",
        rmsprop_eps=1e-8,
        adam_b1=0.9,
        rmsprop_decay=0.95,
        # optimizer="rmsprop",
        # rmsprop_eps=1e-3,
        # loss=ImpalaLossConfig(logit_l2_coef=1e-6,),
        loss=ImpalaConfig(
            logit_l2_coef=0.0,
            weight_l2_coef=0.0,
            vf_coef=0.25,
            ent_coef=0,
            gamma=0.99,
            vtrace_lambda=0.97,
            max_vf_error=0.01,
        ),
        # loss=PPOConfig(
        #     # logit_l2_coef=0.0,
        #     # weight_l2_coef=0.0,
        #     vf_coef=0.5,
        #     ent_coef=0.0,
        #     gamma=0.98,
        #     gae_lambda=0.8,
        #     clip_vf=0.2,
        #     clip_rho=0.1,
        # ),
    )

    tmpdir = tempfile.TemporaryDirectory()
    tmpdir_path = Path(tmpdir.name)
    writer = DataFrameWriter(args, save_dir=tmpdir_path)

    cleanba.cleanba_impala.MUST_STOP_PROGRAM = False
    train(args, writer=writer)
    print("Done training")
    return writer


writer = train_cartpole_no_vel()


def perc_plot(ax, x, y, percentiles=[0.5, 0.75, 0.9, 0.95, 0.99, 1.00], outliers=False):
    y = np.asarray(y).reshape((len(y), -1))
    x = np.asarray(x)
    assert (y.shape[0],) == x.shape

    perc = np.asarray(percentiles)
    # perc = np.unique(np.concatenate([1 - perc, perc]))
    # np.sort(perc)

    to_plot = np.percentile(y, perc, axis=1)
    for i in range(to_plot.shape[0]):
        ax.plot(x, to_plot[i], alpha=1 - np.abs(perc[i] - 0.5), color="C0")

    if outliers:
        outlier_points = (y < np.min(to_plot, axis=0, keepdims=True).T) | (y > np.max(to_plot, axis=0, keepdims=True).T)
        outlier_i, _ = np.where(outlier_points)

        ax.plot(
            x[outlier_i],
            y[outlier_points],
            ls="",
            marker=".",
            color="C1",
        )


# Create a figure and axes
fig, axes = plt.subplots(6, 1, figsize=(6, 8), sharex="col")
writer.metrics = writer.metrics.sort_index()

# Plot var_explained
ax = axes[0]
writer.metrics["var_explained"].plot(ax=ax)
ax.set_ylabel("Variance")

# Plot avg_episode_return
ax = axes[1]
p_returns = writer.metrics["charts/0/avg_episode_lengths"]
p_returns.dropna().plot(ax=ax)
ax.set_ylabel("Ep lengths")

# Plot losses
ax = axes[2]
# writer.metrics["losses/loss"].plot(ax=ax, label="Total Loss")
writer.metrics["losses/value_loss"].plot(ax=ax, label="Value Loss")
# writer.metrics["pre_multiplier_v_loss"].plot(ax=ax, label="Pre-multiplier value loss")

ax.set_ylabel("Value loss")

ax = axes[3]
perc_plot(
    ax,
    writer.metrics.index[:-1],
    [(np.ravel(writer.states[i, "vtrace_errors"])) for i in writer.metrics.index[:-1]],
    percentiles=[0.0, 0.05, 0.25, 0.5, 0.75, 0.95, 1.0],
)
ax.set_ylabel("VTrace errors")

ax = axes[4]
# writer.metrics["multiplier"].plot(ax=ax, color="C1")
writer.metrics["losses/entropy"].plot(ax=ax, color="C1")
ax.set_ylabel("entropy loss")


ax = axes[5]
writer.metrics["losses/policy_loss"].plot(ax=ax, label="Policy Loss")
ax.set_ylabel("Policy loss")

# ax = axes[5]
# perc_plot(
#     ax,
#     writer.metrics.index[:-1],
#     [np.ravel(writer.states[i, "pg_loss_disagg"]) for i in writer.metrics.index[:-1]],
#     percentiles=[0.0, 0.05, 0.25, 0.5, 0.75, 0.95, 1.0],
# )
# ax.set_ylabel("PG loss")


# # Plot grad_rms/total
# ax = axes[-1]
# writer.metrics["grad_rms/total"].plot(ax=ax)
# ax.set_xlabel("Global Step")
# ax.set_ylabel("RMS")

# Adjust spacing between subplots
plt.tight_layout()

# LOW = 0
# HIGH = 1e6
# for ax in axes:
#     ax.set_xlim(LOW, HIGH)

# Display the plot
plt.show()
