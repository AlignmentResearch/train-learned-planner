from functools import partial
from pathlib import Path
from typing import Any, Literal

import farconf
import flax.serialization
import gymnasium as gym
import jax
import jax.numpy as jnp
import pytest

from cleanba.convlstm import (
    ConvConfig,
    ConvLSTM,
    ConvLSTMCell,
    ConvLSTMCellConfig,
    ConvLSTMConfig,
    LSTMCellState,
)
from cleanba.environments import SokobanConfig
from cleanba.network import Policy, n_actions_from_envs


@pytest.mark.parametrize("pool_and_inject", ["no", "horizontal"])
@pytest.mark.parametrize(
    "cfg",
    [
        ConvConfig(2, (3, 3), (1, 1), "SAME", True),
        ConvConfig(2, (2, 2), (1, 1), "SAME", True),
    ],
)
def test_convlstm_strides_padding(pool_and_inject: Literal["no", "horizontal"], cfg: ConvConfig):
    cell_cfg = ConvLSTMCellConfig(cfg, pool_and_inject)
    cell = ConvLSTMCell(cell_cfg)

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
        recurrent=ConvLSTMCellConfig(ConvConfig(2, (3, 3), (1, 1), "SAME", True), pool_and_inject="no"),
        repeats_per_step=1,
    ),
    ConvLSTMConfig(
        embed=[ConvConfig(5, (3, 3), (1, 1), "SAME", True)],
        recurrent=ConvLSTMCellConfig(ConvConfig(5, (3, 3), (1, 1), "SAME", True), pool_and_inject="horizontal"),
        repeats_per_step=2,
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
    assert len(carry) == net.n_recurrent

    cfg = net.recurrent
    for cell_carry in carry:
        if cfg.conv.padding != "SAME":
            raise NotImplementedError

        shape = (envs.num_envs, dim_room[0] // cfg.conv.strides[0], dim_room[1] // cfg.conv.strides[1], cfg.conv.features)
        assert cell_carry.c.shape == shape
        assert cell_carry.h.shape == shape


@pytest.mark.skip("Requires `convlstm_inout.msgpack` which is 50MB")
def test_scan_reference():
    net = ConvLSTMConfig(
        embed=[ConvConfig(5, (3, 3), (1, 1), "SAME", True)],
        recurrent=ConvLSTMCellConfig(ConvConfig(5, (3, 3), (1, 1), "SAME", True), pool_and_inject="horizontal"),
        n_recurrent=2,
        repeats_per_step=2,
    )
    lstm = ConvLSTM(net)

    with (Path(__file__).parent / "convlstm_inout.msgpack").open("rb") as f:
        inputs_and_outputs = flax.serialization.msgpack_restore(f.read())
    carry_structure = [LSTMCellState(jnp.zeros(()), jnp.zeros(()))] * 2

    carry = flax.serialization.from_state_dict(carry_structure, inputs_and_outputs["carry"])
    inputs = jnp.moveaxis(inputs_and_outputs["inputs_nchw"], 2, -1) / 255.0
    episode_starts = inputs_and_outputs["episode_starts"]
    lstm_carry2 = flax.serialization.from_state_dict(carry_structure, inputs_and_outputs["lstm_carry"])
    lstm_out2 = inputs_and_outputs["lstm_out"]

    params = inputs_and_outputs["params"]
    for layer_key in ["cell_list_0", "cell_list_1"]:
        conv_params = params["params"][layer_key].pop("Conv_0")  # type: ignore
        hidden_size = net.recurrent.features
        convcell = dict(
            hh=dict(
                kernel=conv_params["kernel"][:, :, hidden_size : hidden_size * 2, :],
                bias=conv_params["bias"],
            ),
            ih=dict(
                kernel=jnp.concatenate(
                    [
                        conv_params["kernel"][:, :, :hidden_size, :],
                        conv_params["kernel"][:, :, hidden_size * 2 :, :],
                    ],
                    axis=2,
                ),
                bias=jnp.zeros_like(conv_params["bias"]),
            ),
        )
        params["params"][layer_key]["convcell"] = convcell

    new_params = lstm.init(jax.random.PRNGKey(1234), carry, inputs, episode_starts, method=lstm.scan)
    assert jax.tree.structure(params) == jax.tree.structure(new_params)
    assert jax.tree.all(jax.tree.map(lambda x, y: x.shape == y.shape, params, new_params))

    lstm_carry, lstm_out = lstm.apply(params, carry, inputs, episode_starts, method=lstm.scan)

    assert jnp.allclose(lstm_out2, lstm_out, atol=1e-7)
    assert jax.tree.all(jax.tree.map(partial(jnp.allclose, atol=1e-7), lstm_carry2, lstm_carry))


@pytest.mark.parametrize("net", CONVLSTM_CONFIGS)
def test_scan_correct(net: ConvLSTMConfig):
    num_envs = 5
    input_shape = (num_envs, 60, 60, 3)
    lstm = ConvLSTM(net)
    key, k1, k2 = jax.random.split(jax.random.PRNGKey(1234), 3)
    carry = lstm.apply({}, k1, input_shape, method=lstm.initialize_carry)
    params = lstm.init(k2, carry, jnp.ones((1, *input_shape)), jnp.ones((1, num_envs), dtype=jnp.bool_), method=lstm.scan)

    time_steps = 4
    key, k1, k2 = jax.random.split(key, 3)
    inputs = jax.random.uniform(k1, (time_steps, *input_shape), maxval=255)
    episode_starts = jnp.zeros((time_steps, num_envs), dtype=jnp.bool_)

    lstm_carry, lstm_out = lstm.apply(params, carry, inputs, episode_starts, method=lstm.scan)

    b_lstm = lstm.bind(params)

    cell_carry: list[LSTMCellState] = list(carry)
    for t in range(time_steps):
        x = inputs[t]
        for conv in b_lstm.conv_list:
            x = conv(x)

        lstm_x = b_lstm._compress_input(inputs[t])
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

    scan_carry, scan_logits, scan_values, _ = b_policy.get_logits_and_value(carry, inputs_nchw, episode_starts)

    logits: list[Any] = [None] * time_steps
    values: list[Any] = [None] * time_steps
    for t in range(time_steps):
        carry, _, logits[t], values[t], key = b_policy.get_action(carry, inputs_nchw[t], episode_starts[t], key)

    assert jax.tree.all(jax.tree.map(partial(jnp.allclose, atol=1e-5), carry, scan_carry))
    assert jnp.allclose(scan_logits, jnp.stack(logits), atol=1e-5)
    assert jnp.allclose(scan_values, jnp.stack(values), atol=1e-5)


@pytest.mark.parametrize("net", CONVLSTM_CONFIGS)
def test_convlstm_forward(net: ConvLSTMConfig):
    dim_room = (6, 6)
    envs = _sync_sokoban_envs(dim_room)

    k1, k2 = jax.random.split(jax.random.PRNGKey(1234))
    policy, carry, params = net.init_params(envs, k1)
    obs = envs.observation_space.sample()

    assert obs is not None
    out_carry, actions, logits, values, _key = jax.jit(partial(policy.apply, method=policy.get_action))(
        params, carry, obs, jnp.zeros(envs.num_envs, dtype=jnp.bool_), k2
    )
    assert jax.tree.all(jax.tree.map(lambda x, y: x.shape == y.shape, carry, out_carry)), "Carries don't have the same shape"
    assert actions.shape == (envs.num_envs,)
    assert logits.shape == (envs.num_envs, n_actions_from_envs(envs))
    assert values.shape == (envs.num_envs,)
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


@pytest.mark.slow
@pytest.mark.parametrize("pool_and_inject", ["horizontal", "vertical", "no"])
@pytest.mark.parametrize("pool_projection", ["full", "per-channel", "max", "mean"])
@pytest.mark.parametrize("output_activation", ["sigmoid", "tanh"])
@pytest.mark.parametrize("fence_pad", ["same", "valid", "no"])
@pytest.mark.parametrize("forget_bias", [0.0, 1.0])
@pytest.mark.parametrize("skip_final", [True, False])
@pytest.mark.parametrize("residual", [True, False])
def do_test_count_params(pool_and_inject, pool_projection, output_activation, fence_pad, forget_bias, skip_final, residual):
    net = ConvLSTMConfig(
        n_recurrent=3,
        repeats_per_step=3,
        residual=residual,
        skip_final=skip_final,
        embed=[ConvConfig(32, (4, 4), (1, 1), "SAME", True)] * 2,
        recurrent=ConvLSTMCellConfig(
            ConvConfig(32, (3, 3), (1, 1), "SAME", True),
            pool_and_inject=pool_and_inject,
            pool_projection=pool_projection,
            output_activation=output_activation,
            fence_pad=fence_pad,
            forget_bias=forget_bias,
        ),
    )

    envs = SokobanConfig(
        max_episode_steps=120,
        num_envs=1,
        seed=1,
        tinyworld_obs=True,
        num_boxes=1,
        dim_room=(10, 10),
        asynchronous=False,
    ).make()

    net.count_params(envs)


@pytest.mark.parametrize("net", CONVLSTM_CONFIGS)
def test_config_de_serialize(net: ConvLSTMConfig):
    d = farconf.to_dict(net, ConvLSTMConfig)
    net2 = farconf.from_dict(d, ConvLSTMConfig)
    assert net == net2
