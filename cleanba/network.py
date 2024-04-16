import abc
import dataclasses
from functools import partial

import flax
import flax.linen as nn
import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np
from flax.linen.initializers import constant, orthogonal
from flax.training.train_state import TrainState


@flax.struct.dataclass
class AgentParams:
    network_params: flax.core.FrozenDict
    actor_params: flax.core.FrozenDict
    critic_params: flax.core.FrozenDict

    def __contains__(self, item):
        return item in (f.name for f in dataclasses.fields(self))


@dataclasses.dataclass(frozen=True)
class NetworkSpec(abc.ABC):
    @abc.abstractmethod
    def make(self) -> nn.Module:
        ...

    def init_params(self, n_actions: int, key: jax.Array, example_obs: np.ndarray) -> AgentParams:
        net_key, actor_key, critic_key = jax.random.split(key, 3)

        net_obj = self.make()
        net_params = net_obj.init(net_key, example_obs)
        net_out_shape = jax.eval_shape(net_obj.apply, net_params, example_obs)

        actor_params = Actor(n_actions).init(actor_key, jnp.zeros(net_out_shape.shape))
        critic_params = Critic().init(critic_key, jnp.zeros(net_out_shape.shape))
        return AgentParams(net_params, actor_params, critic_params)

    @partial(jax.jit, static_argnames=["self", "n_actions"])
    def get_action(
        self,
        n_actions: int,
        params: AgentParams,
        next_obs: jax.Array,
        key: jax.Array,
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        hidden: jax.Array = self.make().apply(params.network_params, next_obs)

        logits: jax.Array = Actor(n_actions).apply(params.actor_params, hidden)
        # sample action: Gumbel-softmax trick
        # see https://stats.stackexchange.com/questions/359442/sampling-from-a-categorical-distribution
        key, subkey = jax.random.split(key)
        u = jax.random.uniform(subkey, shape=logits.shape)
        action = jnp.argmax(logits - jnp.log(-jnp.log(u)), axis=1)
        return action, logits, key

    @partial(jax.jit, static_argnames=["self", "n_actions"])
    def get_logits_and_value(
        self,
        n_actions: int,
        params: AgentParams,
        x: jax.Array,
    ) -> tuple[jax.Array, jax.Array]:
        hidden = self.make().apply(params.network_params, x)
        logits: jax.Array = Actor(n_actions).apply(params.actor_params, hidden)

        value = Critic().apply(params.critic_params, hidden).squeeze(-1)
        return logits, value


@dataclasses.dataclass(frozen=True)
class AtariCNNSpec(NetworkSpec):
    channels: tuple[int, ...] = (16, 32, 32)  # the channels of the CNN
    mlp_hiddens: tuple[int, ...] = (256,)  # the hiddens size of the MLP

    def make(self) -> "AtariCNN":
        return AtariCNN(self)


class ResidualBlock(nn.Module):
    channels: int

    @nn.compact
    def __call__(self, x):
        inputs = x
        x = nn.relu(x)
        x = nn.Conv(self.channels, kernel_size=(3, 3))(x)
        x = nn.relu(x)
        x = nn.Conv(self.channels, kernel_size=(3, 3))(x)
        return x + inputs


class ConvSequence(nn.Module):
    channels: int

    @nn.compact
    def __call__(self, x):
        x = nn.Conv(self.channels, kernel_size=(3, 3))(x)
        x = nn.max_pool(x, window_shape=(3, 3), strides=(2, 2), padding="SAME")
        x = ResidualBlock(self.channels)(x)
        x = ResidualBlock(self.channels)(x)
        return x


class AtariCNN(nn.Module):
    cfg: AtariCNNSpec

    @nn.compact
    def __call__(self, x):
        x = jnp.transpose(x, (0, 2, 3, 1))
        x = x / (255.0)
        for channels in self.cfg.channels:
            x = ConvSequence(channels)(x)
        x = nn.relu(x)
        x = x.reshape((x.shape[0], -1))
        for hidden in self.cfg.mlp_hiddens:
            x = nn.Dense(hidden, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x)
            x = nn.relu(x)
        return x


class Critic(nn.Module):
    @nn.compact
    def __call__(self, x):
        return nn.Dense(1, kernel_init=orthogonal(1), bias_init=constant(0.0))(x)


class Actor(nn.Module):
    action_dim: int

    @nn.compact
    def __call__(self, x):
        return nn.Dense(self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0))(x)
