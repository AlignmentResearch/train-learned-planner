import dataclasses
from functools import partial
from typing import Any, Callable, Optional, Sequence

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

    def make_conv(
        self,
        kernel_init=nn.initializers.lecun_normal(),
        bias_init=nn.initializers.zeros_init(),
        *,
        name: Optional[str] = None,
    ):
        return nn.Conv(
            self.features,
            self.kernel_size,
            self.strides,
            self.padding,
            use_bias=self.use_bias,
            kernel_init=kernel_init,
            bias_init=bias_init,
            name=name,
        )


@dataclasses.dataclass(frozen=True)
class ConvLSTMConfig(PolicySpec):
    embed: Sequence[ConvConfig] = dataclasses.field(default_factory=list)
    recurrent: Sequence[ConvConfig] = dataclasses.field(default_factory=lambda: [ConvConfig(64, (3, 3), (1, 1))])

    repeats_per_step: int = 1
    pool_and_inject: bool = False
    add_one_to_forget: bool = True

    def make(self) -> "ConvLSTM":
        return ConvLSTM(self)


class ConvLSTMCellState(flax.struct.PyTreeNode):
    c: jax.Array
    h: jax.Array


ConvLSTMState = list[ConvLSTMCellState]


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
    _ = jax.eval_shape(partial(jnp.broadcast_to, shape=target.shape[: -len(src.shape)]), src)

    dims_to_expand = tuple(range(len(src.shape), len(target.shape)))

    return jnp.expand_dims(src, axis=dims_to_expand)


class ConvLSTM(nn.Module):
    cfg: ConvLSTMConfig

    def setup(self):
        self.conv_list = [c.make_conv() for c in self.cfg.embed]
        self.cell_list = [
            ConvLSTMCell(c, add_one_to_forget=self.cfg.add_one_to_forget, pool_and_inject=self.cfg.pool_and_inject)
            for c in self.cfg.recurrent
        ]

    def _preprocess_input(
        self, carry: ConvLSTMState, observations: jax.Array, episode_starts: jax.Array
    ) -> tuple[ConvLSTMState, jax.Array]:
        """
        Embeds the inputs using `self.conv_list`, and preprocesses `carry` so it is zeroed when `episode_starts` is True
        """
        assert len(episode_starts.shape) == 1
        assert len(observations.shape) == 4, f"observations shape must be [batch, h, w, c] but is {observations.shape=}"
        assert observations.shape[0] == episode_starts.shape[0]

        x = observations
        for c in self.conv_list:
            x = c(x)
            x = nn.relu(x)
        embedded = x
        del x

        not_reset = ~episode_starts
        carry = jax.tree.map(lambda z: z * _broadcast_towards_the_left(z, not_reset), carry)

        return carry, embedded

    def _apply_cells(self, carry: ConvLSTMState, inputs: jax.Array) -> tuple[Any, jax.Array]:
        """
        Applies all cells in `self.cell_list` once. `Inputs` gets passed as the input to every cell
        """

        def apply_layers(i: int, carry: ConvLSTMState) -> ConvLSTMState:
            prev_layer_state = carry[-1].h  # Top-down skip connection from previous time step

            for d, cell in enumerate(self.cell_list):
                # c^n_d, h^n_d = MemoryModule_d(i_t, c^{n-1}_d, h^{n-1}_d, h^n_{d-1})
                #
                # equivalently
                # state[d] = cell_list[d](i_t, state[d], h_n{d-1}
                carry[d], prev_layer_state = cell(carry[d], inputs, prev_layer_state)
            return carry

        out_carry = jax.lax.fori_loop(0, self.cfg.repeats_per_step, apply_layers, carry, unroll=None)
        return out_carry, out_carry[-1].h

    def step(
        self, carry: ConvLSTMState, observations: jax.Array, episode_starts: jax.Array
    ) -> tuple[ConvLSTMState, jax.Array]:
        """Applies the ConvLSTM for a single step"""
        carry, embedded = self._preprocess_input(carry, observations, episode_starts)
        return self._apply_cells(carry, embedded)

    def scan(
        self, carry: ConvLSTMState, observations: jax.Array, episode_starts: jax.Array
    ) -> tuple[ConvLSTMState, jax.Array]:
        """Applies the ConvLSTM over many time steps."""
        carry, embedded = jax.vmap(self._preprocess_input)(carry, observations, episode_starts)
        out_carry, out = jax.lax.scan(self._apply_cells, carry, embedded)
        return out_carry, out

    def initialize_carry(self, rng, input_shape):
        rng = jax.random.split(rng, len(self.cell_list))
        return [cell.initialize_carry(k, input_shape) for (cell, k) in zip(self.cell_list, rng)]


class ConvLSTMCell(nn.Module):
    cfg: ConvConfig
    add_one_to_forget: bool = False
    pool_and_inject: bool = False

    gate_fn: Callable[..., jax.Array] = nn.sigmoid
    activation_fn: Callable[..., jax.Array] = nn.tanh
    kernel_init: nn.initializers.Initializer = nn.initializers.lecun_normal()
    recurrent_kernel_init: nn.initializers.Initializer = nn.initializers.orthogonal()
    project_init: nn.initializers.Initializer = nn.initializers.lecun_normal()
    bias_init: nn.initializers.Initializer = nn.initializers.zeros_init()
    dtype: Optional[jnp.dtype] = None
    param_dtype: jnp.dtype = jnp.float32
    carry_init: nn.initializers.Initializer = nn.initializers.zeros_init()

    @nn.compact
    def __call__(
        self, carry: ConvLSTMCellState, inputs: jax.Array, prev_layer_hidden: jax.Array
    ) -> tuple[ConvLSTMCellState, jax.Array]:
        input_to_hidden = nn.Conv(
            features=4 * self.cfg.features,
            kernel_size=self.cfg.kernel_size,
            strides=self.cfg.strides,
            padding=self.cfg.padding,
            use_bias=self.cfg.use_bias,
            kernel_init=self.kernel_init,
            bias_init=self.bias_init,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            name="ih",
        )
        hidden_to_hidden = nn.Conv(
            features=4 * self.cfg.features,
            kernel_size=self.cfg.kernel_size,
            strides=1,  # Has to be 1 so the hidden size stays the same
            padding="SAME",  # Has to be same so the hidden size stays the same
            use_bias=False,  # input_to_hidden already uses bias
            kernel_init=self.recurrent_kernel_init,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            name="hh",
        )

        if self.pool_and_inject:
            AXES_HW = (1, 2)
            h_max = jnp.max(carry.h, axis=AXES_HW)
            h_mean = jnp.mean(carry.h, axis=AXES_HW)
            h_max_and_mean = jnp.concatenate([h_max, h_mean], axis=-1)
            pooled_h = nn.Dense(
                2 * self.cfg.features,
                use_bias=False,
                dtype=self.dtype,
                param_dtype=self.param_dtype,
                kernel_init=self.project_init,
            )(h_max_and_mean)

            B, H, W, _ = carry.h.shape

            pooled_h_expanded = jnp.broadcast_to(pooled_h[:, None, None, :], (B, H, W, pooled_h.shape[-1]))
            concat_hidden = [carry.h, prev_layer_hidden, pooled_h_expanded]
        else:
            concat_hidden = [carry.h, prev_layer_hidden]

        h = jnp.concatenate(concat_hidden, axis=-1)
        gates = input_to_hidden(inputs) + hidden_to_hidden(h)
        i, g, f, o = jnp.split(gates, indices_or_sections=4, axis=-1)

        i = self.gate_fn(i)
        f = self.gate_fn(f + (1 if self.add_one_to_forget else 0))
        g = self.activation_fn(g)
        o = self.gate_fn(o)
        new_c = f * carry.c + i * g
        new_h = o * self.activation_fn(new_c)
        return ConvLSTMCellState(c=new_c, h=new_h), new_h

    def initialize_carry(self, rng: jax.Array, input_shape: tuple[int, ...]) -> ConvLSTMCellState:
        """Initialize the RNN cell carry.

        Args:
          rng: random number generator passed to the init_fn.
          input_shape: a tuple providing the shape of the input to the cell.

        Returns:
          An initialized carry for the given RNN cell.
        """

        def _input_to_hidden_conv(input):
            kernel = jnp.ones((*self.cfg.kernel_size, input.shape[-1], self.cfg.features))
            assert isinstance(self.cfg.padding, str)
            return jax.lax.conv_general_dilated(
                input,
                kernel,
                window_strides=tuple(self.cfg.strides),
                padding=self.cfg.padding,
                dimension_numbers=("NHWC", "HWIO", "NHWC"),
            )

        hidden = _input_to_hidden_conv(jnp.ones((1, *input_shape[-3:])))
        # Take only H,W,C from the convolution (discard batch)
        mem_shape = input_shape[:-3] + hidden.shape[-3:]

        key1, key2 = jax.random.split(rng)
        c = self.carry_init(key1, mem_shape, self.param_dtype)
        h = self.carry_init(key2, mem_shape, self.param_dtype)
        return ConvLSTMCellState(c=c, h=h)
