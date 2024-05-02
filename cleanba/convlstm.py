import abc
import dataclasses
import math
from functools import partial
from typing import Sequence

import flax.linen as nn
import flax.struct
import jax
import jax.numpy as jnp

from cleanba.network import PolicySpec


@dataclasses.dataclass(frozen=True)
class ConvConfig:
    features: int
    kernel_size: Sequence[int]
    strides: Sequence[int]
    padding: str | Sequence[tuple[int, int]] = "SAME"
    use_bias: bool = True

    def make_conv(self, **kwargs):
        return nn.Conv(self.features, self.kernel_size, self.strides, self.padding, self.use_bias, **kwargs)


@dataclasses.dataclass(frozen=True)
class BaseLSTMConfig(PolicySpec):
    n_recurrent: int = 1  # D in the paper
    repeats_per_step: int = 1  # N in the paper
    pool_and_inject: bool = True
    mlp_hiddens: Sequence[int] = (256,)

    @abc.abstractmethod
    def make(self) -> "BaseLSTM":
        ...


@dataclasses.dataclass(frozen=True)
class ConvLSTMConfig(BaseLSTMConfig):
    embed: Sequence[ConvConfig] = dataclasses.field(default_factory=list)
    recurrent: ConvConfig = ConvConfig(32, (3, 3), (1, 1), "SAME", True)

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
    cell_list: list["WrappedCellBase"] = dataclasses.field(init=False)

    def setup(self):
        self.dense_list = [nn.Dense(hidden) for hidden in self.cfg.mlp_hiddens]
        self._setup_compress_and_cells()

    @abc.abstractmethod
    def _setup_compress_and_cells(self) -> None:
        ...

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
        prev_layer_state = carry[-1].h  # Top-down skip connection from previous time step

        for d, cell in enumerate(self.cell_list):
            # c^n_d, h^n_d = MemoryModule_d(i_t, c^{n-1}_d, h^{n-1}_d, h^n_{d-1})
            #
            # equivalently
            # state[d] = cell_list[d](i_t, state[d], h_n{d-1}
            carry[d], prev_layer_state = cell(carry[d], inputs, prev_layer_state)
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
        flattened_out = jnp.reshape(out, (inputs.shape[0], -1))
        return carry, flattened_out

    def step(self, carry: LSTMState, observations: jax.Array, episode_starts: jax.Array) -> tuple[LSTMState, jax.Array]:
        """Applies the RNN for a single step"""
        embedded = self._compress_input(observations)
        out_carry, pre_mlp = self._apply_cells(carry, embedded, episode_starts)
        return out_carry, self._mlp(pre_mlp)

    def scan(self, carry: LSTMState, observations: jax.Array, episode_starts: jax.Array) -> tuple[LSTMState, jax.Array]:
        """Applies the RNN over many time steps."""
        embedded = jax.vmap(self._compress_input)(observations)
        apply_cells_fn = nn.scan(self.__class__._apply_cells, variable_broadcast="params", split_rngs={"params": False})
        out_carry, pre_mlp = apply_cells_fn(self, carry, embedded, episode_starts)
        out = jax.vmap(self._mlp)(pre_mlp)
        return out_carry, out

    def _mlp(self, x: jax.Array) -> jax.Array:
        for dense in self.dense_list:
            x = self.cfg.norm(x)
            x = dense(x)
            x = nn.relu(x)
        return x


class ConvLSTM(BaseLSTM):
    cfg: ConvLSTMConfig

    def setup(self):
        super().setup()
        self.conv_list = [c.make_conv() for c in self.cfg.embed]
        self.cell_list = [ConvLSTMCell(self.cfg.pool_and_inject, cfg=self.cfg.recurrent) for _ in range(self.cfg.n_recurrent)]

    def _compress_input(self, x: jax.Array) -> jax.Array:
        """
        Embeds the inputs using `self.conv_list`
        """
        assert len(x.shape) == 4, f"observations shape must be [batch, c, h, w] but is {x.shape=}"

        for c in self.conv_list:
            x = c(x)
            x = nn.relu(x)
        return x


class LSTM(BaseLSTM):
    cfg: LSTMConfig

    def _setup_compress_and_cells(self) -> None:
        self.compress_list = [nn.Dense(hidden) for hidden in self.cfg.embed_hiddens]
        self.cell_list = [
            LSTMCell(self.cfg.pool_and_inject, features=self.cfg.recurrent_hidden) for _ in range(self.cfg.n_recurrent)
        ]

    def _compress_input(self, x: jax.Array) -> jax.Array:
        assert len(x.shape) == 4, f"observations shape must be [batch, c, h, w] but is {x.shape=}"

        # Flatten input
        x = jnp.reshape(x, (x.shape[0], math.prod(x.shape[1:])))

        for c in self.compress_list:
            x = c(x)
            x = nn.relu(x)
        return x


class WrappedCellBase(nn.RNNCellBase):
    pool_and_inject: bool

    @staticmethod
    def pool_and_project(prev_layer_hidden: jax.Array) -> jax.Array:
        B, H, W, C = prev_layer_hidden.shape
        AXES_HW = (1, 2)
        h_max = jnp.max(prev_layer_hidden, axis=AXES_HW)
        h_mean = jnp.mean(prev_layer_hidden, axis=AXES_HW)
        h_max_and_mean = jnp.concatenate([h_max, h_mean], axis=-1)
        pooled_h = nn.Dense(C, use_bias=False)(h_max_and_mean)

        pooled_h_expanded = jnp.broadcast_to(pooled_h[:, None, None, :], (B, H, W, C))
        return pooled_h_expanded

    @nn.compact
    def __call__(
        self, carry: LSTMCellState, inputs: jax.Array, prev_layer_hidden: jax.Array
    ) -> tuple[LSTMCellState, jax.Array]:
        assert self.cfg.padding == "SAME" and all(s == 1 for s in self.cfg.strides), self.cfg

        if self.pool_and_inject:
            pooled_h = self.pool_and_project(prev_layer_hidden)
            cell_inputs = jnp.concatenate([inputs, prev_layer_hidden, pooled_h], axis=-1)
        else:
            cell_inputs = jnp.concatenate([inputs, prev_layer_hidden], axis=-1)

        (c, h), out = self.make_cell()((carry.c, carry.h), cell_inputs)
        return LSTMCellState(c=c, h=h), out

    @nn.nowrap
    def initialize_carry(self, rng: jax.Array, input_shape: tuple[int, ...]) -> LSTMCellState:
        shape = (*input_shape[:-1], self.cfg.features)
        c_rng, h_rng = jax.random.split(rng, 2)
        return LSTMCellState(c=nn.zeros_init()(c_rng, shape), h=nn.zeros_init()(h_rng, shape))


class ConvLSTMCell(WrappedCellBase):
    cfg: ConvConfig

    @nn.nowrap
    def make_cell(self):
        return nn.ConvLSTMCell(
            self.cfg.features, self.cfg.kernel_size, self.cfg.strides, self.cfg.padding, self.cfg.use_bias, name="convcell"
        )

    def num_feature_axes(self) -> int:
        return 3


class LSTMCell(WrappedCellBase):
    features: int

    @nn.nowrap
    def make_cell(self):
        return nn.LSTMCell(self.features)

    def num_feature_axes(self) -> int:
        return 1
