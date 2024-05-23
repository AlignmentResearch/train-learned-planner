from functools import partial
from typing import Any

import farconf
import flax.linen as nn
import gymnasium as gym
import jax
import jax.numpy as jnp
import pytest

from cleanba.convlstm import ConvConfig, ConvLSTM, ConvLSTMCell, ConvLSTMCellState, ConvLSTMConfig
from cleanba.environments import SokobanConfig
from cleanba.network import Policy, n_actions_from_envs


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


CONVLSTM_CONFIGS = [
    ConvLSTMConfig(
        embed=[ConvConfig(2, (3, 3), (1, 1), "SAME", True)],
        recurrent=[ConvConfig(2, (3, 3), (1, 1), "SAME", True)],
        repeats_per_step=1,
        pool_and_inject=False,
        add_one_to_forget=True,
    ),
    ConvLSTMConfig(
        embed=[ConvConfig(5, (3, 3), (1, 1), "SAME", True)],
        recurrent=[ConvConfig(5, (3, 3), (1, 1), "SAME", True)] * 2,
        repeats_per_step=2,
        pool_and_inject=True,
        add_one_to_forget=True,
    ),
]


def _sync_sokoban_envs(dim_room: tuple[int, int]) -> gym.vector.VectorEnv:
    return SokobanConfig(
        max_episode_steps=40,
        num_envs=2,
        seed=1,
        min_episode_steps=20,
        tinyworld_obs=True,
        num_boxes=1,
        dim_room=dim_room,
        asynchronous=False,
    ).make()


@pytest.mark.parametrize("net", CONVLSTM_CONFIGS)
def test_carry_shape(net: ConvLSTMConfig):
    dim_room = (6, 6)
    envs = _sync_sokoban_envs(dim_room)
    policy = Policy(n_actions_from_envs(envs), net)

    carry = policy.apply({}, jax.random.PRNGKey(0), envs.observation_space.shape, method=policy.initialize_carry)
    assert isinstance(carry, list)
    assert len(carry) == len(net.recurrent)

    for cell_carry, cfg in zip(carry, net.recurrent):
        if cfg.padding != "SAME":
            raise NotImplementedError

        shape = (envs.num_envs, dim_room[0] // cfg.strides[0], dim_room[1] // cfg.strides[1], cfg.features)
        assert cell_carry.c.shape == shape
        assert cell_carry.h.shape == shape


@pytest.mark.parametrize("net", CONVLSTM_CONFIGS)
def test_scan_correct(net: ConvLSTMConfig):
    num_envs = 5
    input_shape = (num_envs, 3, 60, 60)
    lstm = ConvLSTM(net)
    key, k1, k2 = jax.random.split(jax.random.PRNGKey(1234), 3)
    carry = lstm.apply({}, k1, input_shape, method=lstm.initialize_carry)
    params = lstm.init(k2, carry, jnp.ones((1, *input_shape)), jnp.ones((1, num_envs), dtype=jnp.bool_), method=lstm.scan)

    time_steps = 4
    key, k1, k2 = jax.random.split(key, 3)
    inputs_nchw = jax.random.uniform(k1, (time_steps, *input_shape), maxval=255)
    episode_starts = jnp.zeros((time_steps, num_envs))

    lstm_carry, lstm_out = lstm.apply(params, carry, inputs_nchw, episode_starts, method=lstm.scan)

    inputs = jnp.moveaxis(inputs_nchw, -3, -1)

    b_lstm = lstm.bind(params)

    cell_carry: list[ConvLSTMCellState] = list(carry)
    for t in range(time_steps):
        x = inputs[t]
        for conv in b_lstm.conv_list:
            x = conv(x)
            x = nn.relu(x)

        lstm_x = b_lstm._preprocess_input(inputs_nchw[t])
        assert jnp.allclose(x, lstm_x)

        h_nd = cell_carry[-1].h
        for _ in range(net.repeats_per_step):
            for d, cell in enumerate(b_lstm.cell_list):
                cell_carry[d], h_nd = cell(cell_carry[d], x, h_nd)

    for d in range(len(b_lstm.cell_list)):
        assert jnp.allclose(cell_carry[d].c, lstm_carry[d].c, atol=1e-5)
        assert jnp.allclose(cell_carry[d].h, lstm_carry[d].h, atol=1e-5)


@pytest.mark.parametrize("net", CONVLSTM_CONFIGS)
def test_policy_scan_correct(net: ConvLSTMConfig):
    dim_room = (6, 6)
    envs = _sync_sokoban_envs(dim_room)
    num_envs = envs.num_envs
    key, k1 = jax.random.split(jax.random.PRNGKey(1234))

    policy, carry, params = net.init_params(envs, k1)
    b_policy = policy.bind(params)

    time_steps = 4
    key, k1, k2 = jax.random.split(key, 3)
    inputs_nchw = jax.random.uniform(k1, (time_steps, num_envs, 3, *dim_room), maxval=255)
    episode_starts = jax.random.uniform(k2, (time_steps, num_envs)) < 0.4

    scan_carry, scan_logits, _, _ = b_policy.get_logits_and_value(carry, inputs_nchw, episode_starts)

    logits: list[Any] = [None] * time_steps
    for t in range(time_steps):
        carry, _, logits[t], key = b_policy.get_action(carry, inputs_nchw[t], episode_starts[t], key)

    assert jax.tree.all(jax.tree.map(partial(jnp.allclose, atol=1e-5), carry, scan_carry))
    assert jnp.allclose(scan_logits, jnp.stack(logits), atol=1e-5)


@pytest.mark.parametrize("net", CONVLSTM_CONFIGS)
def test_convlstm_forward(net: ConvLSTMConfig):
    dim_room = (6, 6)
    envs = _sync_sokoban_envs(dim_room)

    k1, k2 = jax.random.split(jax.random.PRNGKey(1234))
    policy, carry, params = net.init_params(envs, k1)
    obs = envs.observation_space.sample()

    assert obs is not None
    out_carry, actions, logits, _key = jax.jit(partial(policy.apply, method=policy.get_action))(
        params, carry, obs, jnp.zeros(envs.num_envs, dtype=jnp.bool_), k2
    )
    assert jax.tree.all(jax.tree.map(lambda x, y: x.shape == y.shape, carry, out_carry)), "Carries don't have the same shape"
    assert actions.shape == (envs.num_envs,)
    assert logits.shape == (envs.num_envs, n_actions_from_envs(envs))
    assert _key.shape == k2.shape

    timesteps = 4
    episode_starts = jnp.zeros((timesteps, envs.num_envs), dtype=jnp.bool_)
    obs = jnp.broadcast_to(obs, (timesteps, *obs.shape))

    out_carry, logits, value, metrics = jax.jit(partial(policy.apply, method=policy.get_logits_and_value))(
        params, carry, obs, episode_starts
    )

    assert jax.tree.all(jax.tree.map(lambda x, y: x.shape == y.shape, carry, out_carry)), "Carries don't have the same shape"
    assert logits.shape == (timesteps, envs.num_envs, n_actions_from_envs(envs))
    assert value.shape == (timesteps, envs.num_envs)

    for k, v in metrics.items():
        assert v.shape == (), f"{k} is not averaged over time steps, has {v.shape=}"


@pytest.mark.parametrize("net", CONVLSTM_CONFIGS)
def test_config_de_serialize(net: ConvLSTMConfig):
    d = farconf.to_dict(net, ConvLSTMConfig)
    net2 = farconf.from_dict(d, ConvLSTMConfig)
    assert net == net2
