import abc
import dataclasses
import math
from functools import partial
from typing import Literal, Sequence, Tuple

import flax.linen as nn
import flax.struct
import jax
import jax.numpy as jnp

from cleanba.network import PolicySpec


@dataclasses.dataclass(frozen=True)
class ConvConfig:
    features: int
    kernel_size: Tuple[int, ...]
    strides: Tuple[int, ...]
    padding: Literal["SAME", "VALID"] | Sequence[Tuple[int, int]] = "SAME"
    use_bias: bool = True
    initialization: Literal["torch", "lecun"] = "lecun"

    def make_conv(self, **kwargs):
        if self.initialization == "torch":
            kernel_init = nn.initializers.variance_scaling(1 / 3, "fan_in", "uniform")
        else:
            kernel_init = nn.initializers.lecun_normal()
        if "kernel_init" not in kwargs:
            kwargs["kernel_init"] = kernel_init
        if "use_bias" not in kwargs:
            kwargs["use_bias"] = self.use_bias
        return nn.Conv(self.features, self.kernel_size, self.strides, self.padding, **kwargs)


@dataclasses.dataclass(frozen=True)
class ConvLSTMCellConfig:
    conv: ConvConfig
    pool_and_inject: Literal["horizontal", "vertical", "no"] = "horizontal"
    pool_projection: Literal["full", "per-channel", "max", "mean"] = "full"

    output_activation: Literal["sigmoid", "tanh"] = "sigmoid"
    forget_bias: float = 0.0
    fence_pad: Literal["same", "valid", "no"] = "same"


@dataclasses.dataclass(frozen=True)
class BaseLSTMConfig(PolicySpec):
    n_recurrent: int = 1  # D in the paper
    repeats_per_step: int = 1  # N in the paper
    mlp_hiddens: Sequence[int] = (256,)
    skip_final: bool = True
    residual: bool = False

    @abc.abstractmethod
    def make(self) -> "BaseLSTM":
        ...


@dataclasses.dataclass(frozen=True)
class ConvLSTMConfig(BaseLSTMConfig):
    embed: Sequence[ConvConfig] = dataclasses.field(default_factory=list)
    recurrent: ConvLSTMCellConfig = ConvLSTMCellConfig(ConvConfig(32, (3, 3), (1, 1), "SAME", True))
    use_relu: bool = True

    def make(self) -> "ConvLSTM":
        return ConvLSTM(self)


@dataclasses.dataclass(frozen=True)
class LSTMConfig(BaseLSTMConfig):
    embed_hiddens: Sequence[int] = (200,)
    recurrent_hidden: int = 200

    def make(self) -> "LSTM":
        return LSTM(self)


def _broadcast_towards_the_left(target: jax.Array, src: jax.Array) -> jax.Array:
    """
    Broadcasts `src` towards the left-side of `target`'s shape.

    Example: if `target` is shape (2, 3, 4, 5), and `src` is shape(2, 3), it returns a `src` that is shape (2, 3, 1, 1)
    so it can be broadcasted with `target`.
    """

    assert len(src.shape) <= len(target.shape)
    if len(target.shape) == len(src.shape):
        return src

    # Check that the `target` and `src` have compatible broadcasting shapes
    _ = jax.eval_shape(partial(jnp.broadcast_to, shape=target.shape[: len(src.shape)]), src)

    dims_to_expand = tuple(range(len(src.shape), len(target.shape)))

    return jnp.expand_dims(src, axis=dims_to_expand)


class LSTMCellState(flax.struct.PyTreeNode):
    c: jax.Array
    h: jax.Array


LSTMState = list[LSTMCellState]


class BaseLSTM(nn.Module):
    cfg: BaseLSTMConfig
    cell_list: list["ConvLSTMCell"] = dataclasses.field(init=False)

    def setup(self):
        self.dense_list = [nn.Dense(hidden) for hidden in self.cfg.mlp_hiddens]

    @abc.abstractmethod
    def _compress_input(self, x: jax.Array) -> jax.Array:
        ...

    @nn.nowrap
    def initialize_carry(self, rng, input_shape) -> LSTMState:
        rng = jax.random.split(rng, len(self.cell_list))
        return [cell.initialize_carry(k, input_shape) for (cell, k) in zip(self.cell_list, rng)]

    def apply_cells_once(self, carry: LSTMState, inputs: jax.Array) -> tuple[LSTMState, tuple[()]]:
        """
        Applies all cells in `self.cell_list` once. `Inputs` gets passed as the input to every cell
        """
        assert len(inputs.shape) == 4
        carry = list(carry)  # copy

        # Top-down skip connection from previous time step
        # Importantly it's not residual like the rest of the carry-upwards hidden state
        prev_layer_state = carry[-1].h

        for d, cell in enumerate(self.cell_list):
            # c^n_d, h^n_d = MemoryModule_d(i_t, c^{n-1}_d, h^{n-1}_d, h^n_{d-1})
            #
            # equivalently
            # state[d] = cell_list[d](i_t, state[d], h_n{d-1}
            carry[d], new_state = cell(carry[d], inputs, prev_layer_state)
            if self.cfg.residual:
                prev_layer_state = prev_layer_state + new_state
            else:
                prev_layer_state = new_state
        if not self.cfg.skip_final:  # Pass the residual connection on to the next repetition
            carry[-1] = LSTMCellState(c=carry[-1].c, h=prev_layer_state)
        return carry, ()

    def _apply_cells(self, carry: LSTMState, inputs: jax.Array, episode_starts: jax.Array) -> tuple[LSTMState, jax.Array]:
        """
        Applies all cells in `self.cell_list`, several times: `self.cfg.repeats_per_step` times. Preprocesses the carry
        so it gets zeroed at the start of an episode
        """
        assert len(inputs.shape) == 4
        assert len(episode_starts.shape) == 1

        not_reset = ~episode_starts
        carry = jax.tree.map(lambda z: z * _broadcast_towards_the_left(z, not_reset), carry)

        apply_cells_once_fn = nn.scan(
            self.__class__.apply_cells_once, variable_broadcast="params", split_rngs={"params": False}
        )
        carry, _ = apply_cells_once_fn(self, carry, jnp.broadcast_to(inputs, (self.cfg.repeats_per_step, *inputs.shape)))

        out = carry[-1].h
        return carry, out

    def step(self, carry: LSTMState, observations: jax.Array, episode_starts: jax.Array) -> tuple[LSTMState, jax.Array]:
        """Applies the RNN for a single step"""
        embedded = self._compress_input(observations)
        out_carry, pre_mlp = self._apply_cells(carry, embedded, episode_starts)
        if self.cfg.skip_final:
            pre_mlp = pre_mlp + embedded
        return out_carry, self._mlp(pre_mlp)

    def scan(self, carry: LSTMState, observations: jax.Array, episode_starts: jax.Array) -> tuple[LSTMState, jax.Array]:
        """Applies the RNN over many time steps."""
        embedded = jax.vmap(self._compress_input)(observations)
        apply_cells_fn = nn.scan(self.__class__._apply_cells, variable_broadcast="params", split_rngs={"params": False})
        out_carry, pre_mlp = apply_cells_fn(self, carry, embedded, episode_starts)
        if self.cfg.skip_final:
            pre_mlp = pre_mlp + embedded
        out = jax.vmap(self._mlp)(pre_mlp)
        return out_carry, out

    def _mlp(self, x: jax.Array) -> jax.Array:
        x = jnp.reshape(x, (x.shape[0], -1))
        for dense in self.dense_list:
            x = self.cfg.norm(x)
            x = dense(x)
            x = nn.relu(x)
        return x


class ConvLSTM(BaseLSTM):
    cfg: ConvLSTMConfig

    def setup(self):
        super().setup()
        self.conv_list = [
            c.make_conv(kernel_init=nn.initializers.variance_scaling(1.0, "fan_in", "truncated_normal"))
            for c in self.cfg.embed
        ]
        self.cell_list = [ConvLSTMCell(self.cfg.recurrent) for _ in range(self.cfg.n_recurrent)]

    def _compress_input(self, x: jax.Array) -> jax.Array:
        """
        Embeds the inputs using `self.conv_list`
        """
        assert len(x.shape) == 4, f"observations shape must be [batch, c, h, w] but is {x.shape=}"

        for i, conv in enumerate(self.conv_list):
            x = conv(x)
            if self.cfg.use_relu and i < len(self.conv_list) - 1:
                x = nn.relu(x)
        return x

    @nn.nowrap
    def initialize_carry(self, rng, input_shape) -> LSTMState:
        n, h, w, c = input_shape
        for conv in self.conv_list:
            w //= conv.strides[0]
            h //= conv.strides[1]
        return super().initialize_carry(rng, (n, h, w, c))


class LSTM(BaseLSTM):
    cfg: LSTMConfig

    def setup(self):
        super().setup()
        self.compress_list = [nn.Dense(hidden) for hidden in self.cfg.embed_hiddens]
        self.cell_list = []  # LSTMCell(self.cfg.cell, features=self.cfg.recurrent_hidden) for _ in range(self.cfg.n_recurrent)]

    def _compress_input(self, x: jax.Array) -> jax.Array:
        assert len(x.shape) == 4, f"observations shape must be [batch, c, h, w] but is {x.shape=}"

        # Flatten input
        x = jnp.reshape(x, (x.shape[0], math.prod(x.shape[1:])))

        for c in self.compress_list:
            x = c(x)
            x = nn.relu(x)
        return x

    @nn.nowrap
    def initialize_carry(self, rng, input_shape) -> LSTMState:
        return super().initialize_carry(rng, (input_shape[0], self.cfg.embed_hiddens[-1]))


class ConvLSTMCell(nn.RNNCellBase):
    cfg: ConvLSTMCellConfig

    def pool_and_project(self, prev_layer_hidden: jax.Array) -> jax.Array:
        B, H, W, C = prev_layer_hidden.shape
        AXES_HW = (1, 2)
        h_max = jnp.max(prev_layer_hidden, axis=AXES_HW)
        h_mean = jnp.mean(prev_layer_hidden, axis=AXES_HW)
        if self.cfg.pool_projection == "max":
            pooled_h = h_max
        elif self.cfg.pool_projection == "mean":
            pooled_h = h_mean
        elif self.cfg.pool_projection == "full":
            h_max_and_mean = jnp.concatenate([h_max, h_mean], axis=-1)
            pooled_h = nn.Dense(C, use_bias=False)(h_max_and_mean)

        elif self.cfg.pool_projection == "per-channel":
            project = self.param(
                "project",
                nn.initializers.variance_scaling(1.0, "fan_in", "truncated_normal"),
                (2, self.cfg.conv.features),
                jnp.float32,
            )
            pooled_h = project[0] * h_max + project[1] * h_mean
        else:
            raise ValueError(f"{self.cfg.pool_projection=}")

        pooled_h_expanded = jnp.broadcast_to(pooled_h[:, None, None, :], (B, H, W, C))
        return pooled_h_expanded

    @nn.compact
    def __call__(
        self, carry: LSTMCellState, inputs: jax.Array, prev_layer_hidden: jax.Array
    ) -> tuple[LSTMCellState, jax.Array]:
        assert self.cfg.conv.padding == "SAME" and all(s == 1 for s in self.cfg.conv.strides), self.cfg

        batch, height, width, channels = inputs.shape
        if self.cfg.fence_pad == "same":
            ones = jnp.ones((batch, height, width, 1))
            fence = ones.at[:, 1:-1, 1:-1, :].set(0.0)
            processed_fence = dataclasses.replace(
                self.cfg.conv, features=4 * self.cfg.conv.features, use_bias=False, padding="SAME"
            ).make_conv(name="fence")(fence)
        elif self.cfg.fence_pad == "valid":
            valid_height = height + (self.cfg.conv.kernel_size[0] - 1)
            valid_width = width + (self.cfg.conv.kernel_size[1] - 1)
            ones = jnp.ones((batch, valid_height, valid_width, 1))
            fence = ones.at[:, 1:-1, 1:-1, :].set(0.0)
            processed_fence = dataclasses.replace(
                self.cfg.conv, features=4 * self.cfg.conv.features, use_bias=False, padding="VALID"
            ).make_conv(name="fence")(fence)
        elif self.cfg.fence_pad == "no":
            processed_fence = 0.0
        else:
            raise ValueError(f"{self.cfg.fence_pad=}")

        if self.cfg.pool_and_inject == "no":
            cell_inputs = jnp.concatenate([inputs, prev_layer_hidden], axis=-1)

        else:
            if self.cfg.pool_and_inject == "horizontal":
                to_pool = carry.h
            elif self.cfg.pool_and_inject == "vertical":
                to_pool = prev_layer_hidden
            else:
                raise ValueError(f"{self.cfg.pool_and_inject=}")
            pooled_h = self.pool_and_project(to_pool)
            cell_inputs = jnp.concatenate([inputs, prev_layer_hidden, pooled_h], axis=-1)

        make_conv_fn = dataclasses.replace(self.cfg.conv, features=4 * self.cfg.conv.features).make_conv

        gates = make_conv_fn(name="ih")(cell_inputs) + make_conv_fn(use_bias=False, name="hh")(carry.h) + processed_fence
        i, j, f, o = jnp.split(gates, indices_or_sections=4, axis=-1)

        i = jnp.tanh(i)
        j = nn.sigmoid(j)
        f = nn.sigmoid(f + self.cfg.forget_bias)
        if self.cfg.output_activation == "sigmoid":
            o = nn.sigmoid(o)
        elif self.cfg.output_activation == "tanh":
            o = jnp.tanh(o)
        else:
            raise ValueError(f"{self.cfg.output_activation=}")

        new_c = carry.c * f + i * j
        new_h = nn.tanh(new_c) * o
        return LSTMCellState(c=new_c, h=new_h), new_h

    @nn.nowrap
    def initialize_carry(self, rng: jax.Array, input_shape: tuple[int, ...]) -> LSTMCellState:
        shape = (*input_shape[:-1], self.cfg.conv.features)
        c_rng, h_rng = jax.random.split(rng, 2)
        return LSTMCellState(c=nn.zeros_init()(c_rng, shape), h=nn.zeros_init()(h_rng, shape))

    def num_feature_axes(self) -> int:
        return 3
