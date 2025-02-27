import collections
import contextlib
import threading
from pathlib import Path
from typing import Iterator

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax.training.train_state import TrainState

from cleanba.cleanba_impala import (
    WandbWriter,
    _concat_and_shard_rollout_internal,
    load_train_state,
    make_optimizer,
    save_train_state,
    train,
)
from cleanba.config import Args
from cleanba.convlstm import ConvConfig, ConvLSTMCellConfig, ConvLSTMConfig, LSTMCellState
from cleanba.environments import SokobanConfig
from cleanba.evaluate import EvalConfig
from cleanba.impala_loss import Rollout
from cleanba.network import GuezResNetConfig, IdentityNorm, PolicySpec


class TinyNetwork(PolicySpec):
    def make(self) -> nn.Module:
        return nn.Dense(10)


# TODO: use generic Writer interface, this is not correct inheritance
class CheckingWriter(WandbWriter):
    def __init__(self, cfg: Args, save_dir: Path, eval_keys):
        self.last_global_step = -1
        self.metrics = {}
        self.metrics_history = collections.defaultdict(list)
        self._save_dir = save_dir / "local-files"
        self._save_dir.mkdir()
        self.named_save_dir = save_dir

        self.eval_keys = set(eval_keys)
        assert len(self.eval_keys) > 0
        self.eval_events = {k: threading.Event() for k in self.eval_keys}

        assert cfg.save_model is True
        self._args = cfg
        self.step_digits = 4
        self.eval_metrics = {}
        self.eval_global_step = -1
        self.done_saving = threading.Event()
        self.done_saving.set()

    def add_scalar(self, name: str, value: int | float, global_step: int):
        if global_step == self.last_global_step:
            self.metrics.clear()

        self.last_global_step = global_step
        self.metrics[name] = float(value)
        self.metrics_history[name].append(float(value))

        if name in self.eval_events:
            if self.eval_global_step != global_step:
                self.done_saving.wait(10)
                self.eval_metrics.clear()

            self.eval_global_step = global_step
            self.eval_events[name].set()
            self.eval_metrics[name] = value

    def add_dict(self, metrics: dict[str, int | float], global_step: int):
        print(f"Adding {metrics=} at {global_step=}")
        for k, v in metrics.items():
            self.add_scalar(k, v, global_step)

    @contextlib.contextmanager
    def save_dir(self, global_step: int) -> Iterator[Path]:
        print(f"Saving at {global_step=}")
        for event in self.eval_events.values():
            event.wait(timeout=5)

        with super().save_dir(global_step) as dir:
            yield dir

            assert self.last_global_step == global_step, "we want to save with the same step as last metrics"
            assert all(k in self.eval_metrics for k in self.eval_keys), (
                f"One of {self.eval_keys=} not present in {list(self.eval_metrics.keys())=}"
            )

        # Clear for the next saving
        for event in self.eval_events.values():
            event.clear()
        self.done_saving.set()

        _, _, args, train_state, update = load_train_state(dir)
        assert args == self._args
        assert isinstance(train_state, TrainState)
        assert update == 1


@pytest.mark.parametrize(
    "net",
    [
        GuezResNetConfig(
            False,
            IdentityNorm(),
            channels=(2, 3),
            strides=(1, 1),
            kernel_sizes=(3, 3),
            mlp_hiddens=(16,),
            normalize_input=False,
        ),
    ],
)
def test_save_model_step(tmpdir: Path, net: PolicySpec):
    env_cfg = SokobanConfig(
        max_episode_steps=40,
        num_envs=1,
        seed=1,
        min_episode_steps=20,
        tinyworld_obs=True,
        num_boxes=1,
        dim_room=(6, 6),
        asynchronous=False,
    )

    eval_frequency = 4
    args = Args(
        train_env=env_cfg,
        eval_envs=dict(eval0=EvalConfig(env_cfg, steps_to_think=[0, 1]), eval1=EvalConfig(env_cfg, steps_to_think=[2])),
        net=net,
        eval_at_steps=frozenset(range(1, eval_frequency * 20, eval_frequency)),
        save_model=True,
        log_frequency=1,
        local_num_envs=1,
        num_actor_threads=2,  # Test multithreaded
        num_steps=2,
        num_minibatches=1,
        # If the whole thing deadlocks exit in some small multiple of 10 seconds
        queue_timeout=10,
    )

    args.total_timesteps = args.num_steps * args.num_actor_threads * args.local_num_envs * eval_frequency
    assert args.total_timesteps < 20

    writer = CheckingWriter(
        args, tmpdir, ["eval0/00_episode_successes", "eval0/01_episode_successes", "eval1/02_episode_successes"]
    )
    train(args, writer=writer)

    assert np.array_equal(
        writer.metrics_history["losses/learning_rate"],
        [0.0006000000284984708, 0.0004500000213738531, 0.0003000000142492354, 0.0001500000071246177],
    )


def test_concat_and_shard_rollout_internal():
    len_learner_devices = 2
    batch = 5 * len_learner_devices
    envs = SokobanConfig(
        max_episode_steps=40,
        num_envs=batch,
        seed=1,
        min_episode_steps=20,
        tinyworld_obs=True,
        num_boxes=1,
        dim_room=(7, 7),
        asynchronous=False,
    ).make()

    time = 4

    obs_t, _ = envs.reset()
    value_t = jnp.zeros((obs_t.shape[0]))
    episode_starts_t = np.ones((envs.num_envs,), dtype=np.bool_)
    carry_t = [LSTMCellState(obs_t, obs_t)]

    storage: list[Rollout] = []
    for _ in range(time):
        a_t = envs.action_space.sample()
        logits_t = jnp.zeros((*a_t.shape, 2), dtype=jnp.float32)
        obs_tplus1, r_t, term_t, trunc_t, _ = envs.step(a_t)
        storage.append(Rollout(obs_t, carry_t, a_t, logits_t, value_t, r_t, episode_starts_t, trunc_t))

        obs_t = obs_tplus1
        episode_starts_t = term_t | trunc_t

    out = _concat_and_shard_rollout_internal(storage, obs_t, episode_starts_t, value_t, len_learner_devices)
    assert isinstance(out, Rollout)

    assert out.obs_t[0].shape == (time + 1, batch // len_learner_devices, *storage[0].obs_t.shape[1:])
    assert out.a_t[0].shape == (time, batch // len_learner_devices)
    assert out.logits_t[0].shape == (time, batch // len_learner_devices, storage[0].logits_t.shape[1])
    assert out.value_t[0].shape == (time + 1, batch // len_learner_devices)
    assert out.r_t[0].shape == (time, batch // len_learner_devices)
    assert out.episode_starts_t[0].shape == (time + 1, batch // len_learner_devices)
    assert out.truncated_t[0].shape == (time, batch // len_learner_devices)
    assert jax.tree.all(
        jax.tree.map(
            lambda x_orig, x: x[0].shape == (1, batch // len_learner_devices, *x_orig.shape[1:]),
            storage[0].carry_t,
            out.carry_t,
        )
    )


@pytest.mark.parametrize(
    "frozen_at_all_steps",
    [True, False],
)
def test_finetune_noop(tmpdir: Path, frozen_at_all_steps: bool):
    env_cfg = SokobanConfig(
        max_episode_steps=4,
        num_envs=1,
        seed=1,
        min_episode_steps=4,  # keep a few steps for gradient to pass through max-pool or conv_hh
        tinyworld_obs=True,
        num_boxes=1,
        dim_room=(10, 10),
        asynchronous=False,
        nn_without_noop=True,
    )
    env = env_cfg.make()
    net = ConvLSTMConfig(
        embed=[ConvConfig(3, (4, 4), (1, 1), "SAME", True)],
        recurrent=ConvLSTMCellConfig(ConvConfig(3, (3, 3), (1, 1), "SAME", True), pool_and_inject="horizontal"),
        repeats_per_step=1,
    )
    args = Args(
        train_env=env_cfg,
        net=net,
        total_timesteps=10,
        local_num_envs=1,
        num_actor_threads=1,
        num_steps=4,
        train_epochs=3,
        gradient_accumulation_steps=1,
        num_minibatches=1,
        concurrency=False,
        learning_rate=1e-3,
        finetune_with_noop_head=False,
        final_learning_rate=0.0,
    )
    policy, _, agent_params = net.init_params(env, jax.random.PRNGKey(42))
    orig_num_actions = agent_params["params"]["actor_params"]["Output"]["kernel"].shape[1]
    assert orig_num_actions == 4, f"Expected 4 actions, got {orig_num_actions}"
    orig_agent_state = TrainState.create(
        apply_fn=None,
        params=agent_params,
        tx=make_optimizer(args, agent_params, total_updates=10),
    )
    orig_agent_state = jax.device_put_replicated(orig_agent_state, jax.devices("cpu"))
    policy_path = tmpdir / "policy"
    policy_path.mkdir()
    save_train_state(policy_path, args, orig_agent_state, 0)

    args.finetune_with_noop_head = True
    args.frozen_finetune_steps_ratio = 1.0 if frozen_at_all_steps else 0.2
    args.load_path = policy_path

    writer = CheckingWriter(args, tmpdir, ["eval0/00_episode_successes"])

    final_state = train(args, writer=writer)

    fs_flat = jax.tree_util.tree_leaves_with_path(final_state.params)
    os_flat = jax.tree_util.tree_leaves_with_path(orig_agent_state.params)

    violations = []
    for (fs_path, fs), (os_path, os) in zip(fs_flat, os_flat):
        assert fs_path == os_path
        concat_path = "/".join(map(lambda x: str(x.key), fs_path))

        if concat_path.startswith("params/actor_params/Output"):
            fs = fs[..., :orig_num_actions]

        if frozen_at_all_steps and not concat_path.startswith("params/actor_params/Output"):
            if not np.allclose(fs, os):
                violations.append(f"Path: {concat_path}")
        else:
            if np.allclose(fs, os):
                violations.append(f"Path: {concat_path}")

    assert not violations, "\n".join(violations)
