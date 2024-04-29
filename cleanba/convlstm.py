import dataclasses
from typing import Any, Callable, Optional, Sequence

import flax.linen as nn
import jax
import jax.numpy as jnp


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
class ConvLSTMConfig:
    embed: Sequence[ConvConfig] = dataclasses.field(default_factory=list)
    recurrent: Sequence[ConvConfig] = dataclasses.field(default_factory=lambda: [ConvConfig(64, (3, 3), (1, 1))])

    repeats_per_step: int = 1
    fence_pad: bool = False
    pool_and_inject: bool = False
    add_one_to_forget: bool = False

    def make(self) -> "ConvLSTM":
        return ConvLSTM(self)


def _broadcast_left(a, b):
    assert len(b.shape) <= len(a.shape)
    if len(a.shape) == len(b.shape):
        return b
    dims_to_expand = tuple(range(len(b.shape), len(a.shape)))

    return jnp.expand_dims(b, axis=dims_to_expand)


class ConvLSTM(nn.Module):
    cfg: ConvLSTMConfig

    def setup(self):
        if self.cfg.pool_and_inject:
            raise NotImplementedError("pool_and_inject")
        if self.cfg.fence_pad:
            raise NotImplementedError("fence_pad")

        self.conv_list = [c.make_conv() for c in self.cfg.embed]
        self.cell_list = [ConvLSTMCell(c, add_one_to_forget=self.cfg.add_one_to_forget) for c in self.cfg.recurrent]

    def __call__(self, observations: jax.Array, carry: Any, episode_starts: jax.Array) -> tuple[Any, jax.Array]:
        assert len(episode_starts.shape) == 1
        assert len(observations.shape) == 4, f"observations shape must be [batch, h, w, c] but is {observations.shape=}"

        x = observations
        for c in self.conv_list:
            x = c(x)
            x = nn.relu(x)
        embedded = x
        del x

        not_reset = ~episode_starts
        carry = jax.tree.map(lambda z: z * _broadcast_left(z, not_reset), carry)

        def loop_all_layers(carry, x):
            assert isinstance(carry, list)
            assert isinstance(carry[-1], list)
            h_nd = carry[-1][0]  # Top-down skip connection from previous time step

            for cell in self.cell_list:
                carry, x = cell(carry, x, h_nd)
            return carry, x

        repeat_embedded = jnp.broadcast_to(embedded, (self.cfg.repeats_per_step, 1, 1, 1))
        new_carry, out = jax.lax.scan(loop_all_layers, carry, repeat_embedded)

        assert isinstance(out, jax.Array)
        assert jax.tree_structure(new_carry) == jax.tree_structure(carry)
        return new_carry, out


ConvLSTMCellState = tuple[jax.Array, jax.Array]


class ConvLSTMCell(nn.Module):
    cfg: ConvConfig
    add_one_to_forget: bool = False
    pool_and_inject: bool = False

    gate_fn: Callable[..., Any] = nn.sigmoid
    activation_fn: Callable[..., Any] = nn.tanh
    kernel_init: nn.initializers.Initializer = nn.initializers.lecun_normal()
    recurrent_kernel_init: nn.initializers.Initializer = nn.initializers.orthogonal()
    bias_init: nn.initializers.Initializer = nn.initializers.zeros_init()
    dtype: Optional[jnp.dtype] = None
    param_dtype: jnp.dtype = jnp.float32
    carry_init: nn.initializers.Initializer = nn.initializers.zeros_init()

    @nn.compact
    def __call__(
        self, carry: ConvLSTMCellState, inputs: jax.Array, prev_layer_hidden: jax.Array
    ) -> tuple[ConvLSTMCellState, jax.Array]:
        c, h = carry
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

        gates = input_to_hidden(inputs) + hidden_to_hidden(h)
        i, g, f, o = jnp.split(gates, indices_or_sections=4, axis=-1)

        i = self.gate_fn(i)
        f = self.gate_fn(f + (1 if self.add_one_to_forget else 0))
        g = self.activation_fn(g)
        o = self.gate_fn(o)
        new_c = f * c + i * g
        new_h = o * self.activation_fn(new_c)
        return (new_c, new_h), new_h

    def initialize_carry(self, rng, input_shape):
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

        batch_shape = input_shape[:-3]
        conv_shape = _input_to_hidden_conv(jnp.ones((1, *input_shape[-3:]))).shape
        mem_shape = batch_shape + conv_shape

        key1, key2 = jax.random.split(rng)
        c = self.carry_init(key1, mem_shape, self.param_dtype)
        h = self.carry_init(key2, mem_shape, self.param_dtype)
        return c, h
