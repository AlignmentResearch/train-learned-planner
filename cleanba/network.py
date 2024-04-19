import abc
import dataclasses
from functools import partial
from typing import Any, Literal, SupportsFloat

import flax
import flax.linen as nn
import flax.struct
import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np
from flax.linen.initializers import constant
from flax.typing import Axes, Shape


@flax.struct.dataclass
class AgentParams:
    network_params: flax.core.FrozenDict
    actor_params: flax.core.FrozenDict
    critic_params: flax.core.FrozenDict

    def __contains__(self, item):
        return item in (f.name for f in dataclasses.fields(self))

    def _n_actions(self) -> int:
        return self.actor_params["params"]["Output"]["kernel"].shape[-1]  # type: ignore


@dataclasses.dataclass(frozen=True)
class NetworkSpec(abc.ABC):
    yang_init: bool = False

    @abc.abstractmethod
    def make(self) -> nn.Module:
        ...

    def init_params(self, envs: gym.vector.VectorEnv, key: jax.Array, example_obs: np.ndarray) -> AgentParams:
        action_space = envs.single_action_space
        assert isinstance(action_space, gym.spaces.Discrete)
        n_actions = int(action_space.n)

        net_key, actor_key, critic_key = jax.random.split(key, 3)

        net_obj = self.make()
        net_params = net_obj.init(net_key, example_obs)
        net_out_shape = jax.eval_shape(net_obj.apply, net_params, example_obs)

        actor_params = Actor(n_actions, yang_init=self.yang_init).init(actor_key, jnp.zeros(net_out_shape.shape))
        critic_params = Critic(yang_init=self.yang_init).init(critic_key, jnp.zeros(net_out_shape.shape))
        out = AgentParams(net_params, actor_params, critic_params)  # type: ignore

        assert out._n_actions() == n_actions
        return out

    @partial(jax.jit, static_argnames=["self", "temperature"])
    def get_action(
        self,
        params: AgentParams,
        next_obs: jax.Array,
        key: jax.Array,
        *,
        temperature: float = 1.0,
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        hidden = self.make().apply(params.network_params, next_obs)

        logits = Actor(params._n_actions(), yang_init=self.yang_init).apply(params.actor_params, hidden)
        assert isinstance(logits, jax.Array)

        if temperature == 0.0:
            action = jnp.argmax(logits, axis=1)
        else:
            # sample action: Gumbel-softmax trick
            # see https://stats.stackexchange.com/questions/359442/sampling-from-a-categorical-distribution
            key, subkey = jax.random.split(key)
            u = jax.random.uniform(subkey, shape=logits.shape)
            action = jnp.argmax(logits / temperature - jnp.log(-jnp.log(u)), axis=1)
        return action, logits, key

    @partial(jax.jit, static_argnames=["self"])
    def get_logits_and_value(
        self,
        params: AgentParams,
        x: jax.Array,
    ) -> tuple[jax.Array, jax.Array]:
        hidden = self.make().apply(params.network_params, x)
        logits = Actor(params._n_actions(), yang_init=self.yang_init).apply(params.actor_params, hidden)
        value = Critic(yang_init=self.yang_init).apply(params.critic_params, hidden)

        assert isinstance(logits, jax.Array)
        assert isinstance(value, jax.Array)
        return logits, value.squeeze(-1)


class NormConfig(abc.ABC):
    @abc.abstractmethod
    def __call__(self, x: jax.Array) -> jax.Array:
        ...


@dataclasses.dataclass(frozen=True)
class RMSNorm(NormConfig):
    eps: float = 1e-6
    use_scale: bool = True
    reduction_axes: Axes = -1
    feature_axes: Axes = -1

    def __call__(self, x: jax.Array) -> jax.Array:
        return nn.RMSNorm(
            epsilon=self.eps, use_scale=self.use_scale, reduction_axes=self.reduction_axes, feature_axes=self.feature_axes
        )(x)


@dataclasses.dataclass(frozen=True)
class IdentityNorm(NormConfig):
    def __call__(self, x: jax.Array) -> jax.Array:
        return x


@dataclasses.dataclass(frozen=True)
class AtariCNNSpec(NetworkSpec):
    channels: tuple[int, ...] = (16, 32, 32)  # the channels of the CNN
    mlp_hiddens: tuple[int, ...] = (256,)  # the hiddens size of the MLP
    norm: NormConfig = RMSNorm()

    def make(self) -> "AtariCNN":
        return AtariCNN(self)


NONLINEARITY_GAINS: dict[str, SupportsFloat] = dict(
    relu=np.sqrt(2.0),
    identity=1.0,
    action_softmax=0.02,  # We want the actions to be pretty random at the start, so use a small scale
    tanh=5 / 3,
)


def yang_initializer(layer_type: Literal["input", "hidden", "output"], nonlinearity: str) -> jax.nn.initializers.Initializer:
    nonlinearity_gain = float(NONLINEARITY_GAINS[nonlinearity])

    def init(key: jax.Array, shape: Shape, dtype: Any = jnp.float_) -> jax.Array:
        dtype = jax.dtypes.canonicalize_dtype(dtype)
        fan_in = np.prod(shape[:-1])

        if layer_type in ["input", "hidden"]:
            stddev = nonlinearity_gain / np.sqrt(fan_in)
        elif layer_type == "output":
            stddev = nonlinearity_gain / fan_in
        else:
            raise ValueError(f"Unknown {layer_type=}")

        return jax.random.normal(key, shape, dtype) * stddev

    return init


class ResidualBlock(nn.Module):
    channels: int
    yang_init: bool

    @nn.compact
    def __call__(self, x, norm):
        if self.yang_init:
            bias_init = kernel_init = yang_initializer("input", "relu")
        else:
            kernel_init = nn.initializers.lecun_normal()
            bias_init = nn.initializers.zeros_init()

        inputs = x
        x = nn.relu(x)
        x = norm(x)
        x = nn.Conv(self.channels, kernel_size=(3, 3), kernel_init=kernel_init, bias_init=bias_init)(x)
        x = nn.relu(x)
        x = norm(x)
        x = nn.Conv(self.channels, kernel_size=(3, 3), kernel_init=kernel_init, bias_init=bias_init)(x)
        return x + inputs


class ConvSequence(nn.Module):
    channels: int
    max_pool: bool = True
    is_input: bool = False
    yang_init: bool = False

    @nn.compact
    def __call__(self, x, norm):
        x = norm(x)
        if self.yang_init:
            x = nn.Conv(
                self.channels,
                kernel_size=(3, 3),
                kernel_init=yang_initializer("input" if self.is_input else "hidden", "relu"),
                bias_init=yang_initializer("input" if self.is_input else "hidden", "relu"),
                name=INPUT_SENTINEL,
            )(x)
        else:
            x = nn.Conv(self.channels, kernel_size=(3, 3))(x)
        x = nn.max_pool(x, window_shape=(3, 3), strides=(2, 2), padding="SAME")
        x = ResidualBlock(self.channels, yang_init=self.yang_init)(x, norm=norm)
        x = ResidualBlock(self.channels, yang_init=self.yang_init)(x, norm=norm)
        return x


class AtariCNN(nn.Module):
    cfg: AtariCNNSpec

    @nn.compact
    def __call__(self, x):
        x = jnp.transpose(x, (0, 2, 3, 1))
        x = x / (255.0)
        for layer_i, channels in enumerate(self.cfg.channels):
            x = ConvSequence(channels, yang_init=self.cfg.yang_init, is_input=(layer_i == 0))(x, norm=self.cfg.norm)
        x = nn.relu(x)
        x = x.reshape((x.shape[0], -1))
        for hidden in self.cfg.mlp_hiddens:
            x = self.cfg.norm(x)
            if self.cfg.yang_init:
                x = nn.Dense(hidden, kernel_init=yang_initializer("hidden", "relu"), bias_init=constant(0.0))(x)
            else:
                x = nn.Dense(hidden)(x)
            x = nn.relu(x)
        return x


class Critic(nn.Module):
    yang_init: bool

    @nn.compact
    def __call__(self, x):
        if self.yang_init:
            bias_init = kernel_init = yang_initializer("output", "identity")
        else:
            kernel_init = nn.initializers.lecun_normal()
            bias_init = nn.initializers.zeros_init()
        return nn.Dense(1, kernel_init=kernel_init, bias_init=bias_init)(x)


class Actor(nn.Module):
    action_dim: int
    yang_init: bool

    @nn.compact
    def __call__(self, x):
        if self.yang_init:
            init = yang_initializer("output", "action_softmax")
        else:
            init = nn.initializers.lecun_normal()
        # Bias here is useless, because softmax is invariant to baseline.
        return nn.Dense(self.action_dim, kernel_init=init, use_bias=False, name="Output")(x)


# %%


@dataclasses.dataclass(frozen=True)
class SokobanResNetConfig(NetworkSpec):
    channels: tuple[int, ...] = (64, 64, 64) * 3
    kernel_sizes: tuple[int, ...] = (4, 4, 4) * 3
    strides: tuple[int, ...] = (1, 1, 1) * 3
    multiplicity: int = 1
    norm: NormConfig = IdentityNorm()

    mlp_hiddens: tuple[int, ...] = ()

    def make(self) -> nn.Module:
        return SokobanResNet(self)


class SokobanResNet(nn.Module):
    cfg: SokobanResNetConfig

    @nn.compact
    def __call__(self, x):
        x = jnp.transpose(x, (0, 2, 3, 1))
        x = x / (255.0)
        for layer_i, (chan, kern, stride) in enumerate(zip(self.cfg.channels, self.cfg.kernel_sizes, self.cfg.strides)):
            x = SokobanConvSequence(
                channels=chan, kernel_size=kern, strides=stride, multiplicity=self.cfg.multiplicity, is_input=layer_i == 0
            )(x, self.cfg.norm)
        x = x.reshape((x.shape[0], np.prod(x.shape[-3:])))
        assert x.shape[-1] == 64 * 10 * 10
        for hidden in self.cfg.mlp_hiddens:
            x = self.cfg.norm(x)
            x = nn.Dense(
                hidden,
                kernel_init=yang_initializer("hidden", "tanh"),
                bias_init=yang_initializer("hidden", "tanh"),
            )(x)
            x = nn.tanh(x)
        return x


class SokobanResidualBlock(nn.Module):
    channels: int
    kernel_size: int
    multiplicity: int

    @nn.compact
    def __call__(self, x, norm):
        inputs = x
        for _ in range(self.multiplicity):
            x = nn.relu(x)
            x = norm(x)
            x = nn.Conv(
                self.channels,
                kernel_size=(self.kernel_size, self.kernel_size),
                kernel_init=yang_initializer("hidden", "relu"),
                bias_init=yang_initializer("hidden", "relu"),
            )(x)
        return x + inputs


INPUT_SENTINEL: str = "xXx_Input_xXx"


class SokobanConvSequence(nn.Module):
    channels: int
    kernel_size: int
    strides: int
    multiplicity: int
    is_input: bool

    @nn.compact
    def __call__(self, x, norm):
        if self.is_input:
            x = nn.Conv(
                self.channels,
                kernel_size=(self.kernel_size, self.kernel_size),
                strides=(self.strides, self.strides),
                kernel_init=yang_initializer("input" if self.is_input else "hidden", "relu"),
                bias_init=yang_initializer("input" if self.is_input else "hidden", "relu"),
                name=INPUT_SENTINEL if self.is_input else None,
            )(x)
        x = SokobanResidualBlock(self.channels, self.kernel_size, self.multiplicity)(x, norm=norm)
        return x


def _fan_in_for_params(params: dict[str, dict[str, Any] | jax.Array]) -> dict[str, dict[str, Any] | str]:
    out: dict[str, dict[str, Any] | str] = {}
    for k, v in params.items():
        if k == INPUT_SENTINEL:
            out[k] = "input"

        elif isinstance(v, jax.Array):
            if k == "bias":
                out[k] = "bias"
            elif k == "kernel":
                fan_in = int(np.prod(v.shape[:-1]))
                out[k] = f"fan_in_{fan_in}"
            else:
                raise ValueError(f"Unknown parameter name {k=}")
        else:
            assert isinstance(v, dict)
            out[k] = _fan_in_for_params(v)
    return out


def label_and_learning_rate_for_params(params: AgentParams) -> tuple[dict[str, float], AgentParams]:
    param_labels: AgentParams = jax.tree.map(_fan_in_for_params, params, is_leaf=lambda v: isinstance(v, dict))
    key_set: set[str] = set(jax.tree.leaves(param_labels))

    possible_keys: dict[str, None] = {k: None for k in key_set}
    learning_rates: dict[str, float] = {}

    possible_keys.pop("input")
    learning_rates["input"] = 1.0

    possible_keys.pop("bias")
    learning_rates["bias"] = 1.0

    for k in possible_keys:
        fan_in = int(k.removeprefix("fan_in_"))
        learning_rates[k] = 1 / fan_in

    return learning_rates, param_labels
