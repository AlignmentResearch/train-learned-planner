from functools import partial
from typing import Any

import flax.linen as nn
import jax
import jax.numpy as jnp
import pytest

from cleanba.convlstm import ConvConfig, ConvLSTMCell, ConvLSTMConfig
from cleanba.environments import SokobanConfig


def _dict_copy(d: dict | Any) -> dict | Any:
    if isinstance(d, dict):
        return {k: _dict_copy(v) for k, v in d.items()}
    return d


def test_equivalent_to_lstm():
    cfg = ConvConfig(features=2, kernel_size=(3, 3), strides=(1, 1), padding="SAME", use_bias=True)
    cleanba_cell = ConvLSTMCell(cfg=cfg, add_one_to_forget=True, pool_and_inject=False)
    linen_cell = nn.ConvLSTMCell(cfg.features, cfg.kernel_size, cfg.strides, cfg.padding, cfg.use_bias)

    rng = jax.random.PRNGKey(1234)
    rng, k1, k2, k3 = jax.random.split(rng, 4)
    inputs = jax.random.normal(k1, (7, 5, 10, 10, 3))

    cleanba_carry = cleanba_cell.initialize_carry(k2, inputs[0].shape)
    linen_carry = linen_cell.initialize_carry(k3, inputs[0].shape)

    rng, k1 = jax.random.split(rng, 2)
    cleanba_params = cleanba_cell.init(k1, cleanba_carry, inputs[0], cleanba_carry.h)
    linen_params = _dict_copy(cleanba_params)

    ih_bias = cleanba_params["params"]["ih"]["bias"]
    hh_bias = jax.random.normal(rng, ih_bias.shape)

    linen_params["params"]["ih"]["bias"] = ih_bias - hh_bias
    linen_params["params"]["hh"]["bias"] = hh_bias

    linen_params["params"]["hh"]["kernel"] = linen_params["params"]["hh"]["kernel"][:, :, : cfg.features, :]

    for t in range(len(inputs)):
        cleanba_carry, cleanba_out = jax.jit(cleanba_cell.apply)(
            cleanba_params, cleanba_carry, inputs[t], jnp.zeros_like(cleanba_carry.h)
        )
        linen_carry, linen_out = jax.jit(linen_cell.apply)(linen_params, linen_carry, inputs[t])

        assert jnp.allclose(cleanba_out, linen_out, atol=1e-6)
        assert jnp.allclose(cleanba_carry.c, linen_carry[0], atol=1e-6)
        assert jnp.allclose(cleanba_carry.h, linen_carry[1], atol=1e-6)


@pytest.mark.parametrize("pool_and_inject", [True, False])
@pytest.mark.parametrize("add_one_to_forget", [True, False])
@pytest.mark.parametrize(
    "cfg",
    [
        ConvConfig(2, (3, 3), (1, 1), "VALID", True),
        ConvConfig(2, (3, 3), (2, 2), "SAME", True),
    ],
)
def test_convlstm_strides_padding(cfg: ConvConfig, add_one_to_forget: bool, pool_and_inject: bool):
    cell = ConvLSTMCell(cfg, add_one_to_forget, pool_and_inject)

    rng = jax.random.PRNGKey(1234)
    rng, k1, k2, k3 = jax.random.split(rng, 4)
    inputs = jax.random.normal(k1, (3, 5, 10, 10, 3))

    carry = cell.initialize_carry(k2, inputs[0].shape)
    params = cell.init(k3, carry, inputs[0], carry.h)

    for t in range(len(inputs)):
        carry, out = jax.jit(cell.apply)(params, carry, inputs[t], carry.h)


@pytest.mark.parametrize(
    "net",
    [
        ConvLSTMConfig(
            embed=[ConvConfig(3, (3, 3), (1, 1), "SAME", True)],
            recurrent=[ConvConfig(3, (3, 3), (2, 2), "SAME", True)],
            repeats_per_step=2,
            pool_and_inject=False,
            add_one_to_forget=True,
        ),
        ConvLSTMConfig(
            embed=[ConvConfig(3, (3, 3), (1, 1), "SAME", True)],
            recurrent=[ConvConfig(3, (3, 3), (2, 2), "SAME", True)],
            repeats_per_step=2,
            pool_and_inject=True,
            add_one_to_forget=True,
        ),
    ],
)
def test_convlstm_forward(net: ConvLSTMConfig):
    envs = SokobanConfig(
        max_episode_steps=40,
        num_envs=2,
        seed=1,
        min_episode_steps=20,
        tinyworld_obs=True,
        num_boxes=1,
        dim_room=(6, 6),
        asynchronous=False,
    ).make()
    k1, k2 = jax.random.split(jax.random.PRNGKey(1234))
    policy, carry, params = net.init_params(envs, k1)
    obs = envs.observation_space.sample()

    assert obs is not None
    _carry, _action, _logits, _key = (partial(policy.apply, method=policy.get_action))(
        params, carry, obs, jnp.zeros(envs.num_envs, dtype=jnp.bool_), k2
    )
