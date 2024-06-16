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

from cleanba.cleanba_impala import WandbWriter, _concat_and_shard_rollout_internal, load_train_state, train
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
        self._save_dir = save_dir

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
        self.metrics[name] = value

        if name in self.eval_events:
            if self.eval_global_step != global_step:
                self.done_saving.wait(10)
                self.eval_metrics.clear()

            self.eval_global_step = global_step
            self.eval_events[name].set()
            self.eval_metrics[name] = value

    @contextlib.contextmanager
    def save_dir(self, global_step: int) -> Iterator[Path]:
        for event in self.eval_events.values():
            event.wait(timeout=5)

        with super().save_dir(global_step) as dir:
            yield dir

            assert self.last_global_step == global_step, "we want to save with the same step as last metrics"
            assert all(
                k in self.eval_metrics for k in self.eval_keys
            ), f"One of {self.eval_keys=} not present in {list(self.eval_metrics.keys())=}"

        # Clear for the next saving
        for event in self.eval_events.values():
            event.clear()
        self.done_saving.set()

        args, train_state, update = load_train_state(dir)
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
        ConvLSTMConfig(
            embed=[ConvConfig(3, (4, 4), (1, 1), "SAME", True)],
            recurrent=ConvLSTMCellConfig(ConvConfig(3, (3, 3), (1, 1), "SAME", True), pool_and_inject="horizontal"),
            repeats_per_step=2,
        ),
        ConvLSTMConfig(
            embed=[ConvConfig(3, (4, 4), (1, 1), "SAME", True)],
            recurrent=ConvLSTMCellConfig(ConvConfig(3, (3, 3), (1, 1), "SAME", True), pool_and_inject="horizontal"),
            repeats_per_step=2,
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
        log_frequency=1234,
        local_num_envs=1,
        num_actor_threads=2,  # Test multithreaded
        num_steps=2,
        num_minibatches=1,
        # If the whole thing deadlocks exit in some small multiple of 10 seconds
        queue_timeout=4,
    )

    args.total_timesteps = args.num_steps * args.num_actor_threads * args.local_num_envs * eval_frequency
    assert args.total_timesteps < 20

    writer = CheckingWriter(
        args, tmpdir, ["eval0/00_episode_successes", "eval0/01_episode_successes", "eval1/02_episode_successes"]
    )
    train(args, writer=writer)


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
    episode_starts_t = np.ones((envs.num_envs,), dtype=np.bool_)
    carry_t = [LSTMCellState(obs_t, obs_t)]

    storage: list[Rollout] = []
    for _ in range(time):
        a_t = envs.action_space.sample()
        logits_t = jnp.zeros((*a_t.shape, 2), dtype=jnp.float32)
        obs_tplus1, r_t, term_t, trunc_t, _ = envs.step(a_t)
        storage.append(Rollout(obs_t, carry_t, a_t, logits_t, r_t, episode_starts_t, trunc_t))

        obs_t = obs_tplus1
        episode_starts_t = term_t | trunc_t

    out = _concat_and_shard_rollout_internal(storage, obs_t, episode_starts_t, len_learner_devices)
    assert isinstance(out, Rollout)

    assert out.obs_t[0].shape == (time + 1, batch // len_learner_devices, *storage[0].obs_t.shape[1:])
    assert out.a_t[0].shape == (time, batch // len_learner_devices)
    assert out.logits_t[0].shape == (time, batch // len_learner_devices, storage[0].logits_t.shape[1])
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
