import contextlib
import threading
from pathlib import Path
from typing import Iterator

import flax.linen as nn
from flax.training.train_state import TrainState

from cleanba.cleanba_impala import WandbWriter, load_train_state, train
from cleanba.config import Args
from cleanba.environments import SokobanConfig
from cleanba.evaluate import EvalConfig
from cleanba.network import NetworkSpec


class TinyNetwork(NetworkSpec):
    def make(self) -> nn.Module:
        return nn.Dense(10)


# TODO: use generic Writer interface, this is not correct inheritance
class CheckingWriter(WandbWriter):
    def __init__(self, cfg: Args, save_dir: Path):
        self.last_global_step = -1
        self.metrics = {}
        self._save_dir = save_dir

        self.eval_keys = {f"{k}/episode_success" for k in cfg.eval_envs.keys()}
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

        args, train_state = load_train_state(dir)
        assert args == self._args
        assert isinstance(train_state, TrainState)


def test_save_model_step(tmpdir: Path):
    env = SokobanConfig(
        max_episode_steps=40,
        num_envs=1,
        seed=1,
        min_episode_steps=20,
        tinyworld_obs=True,
        num_boxes=1,
        dim_room=(6, 6),
        asynchronous=False,
    )

    args = Args(
        train_env=env,
        eval_envs=dict(eval0=EvalConfig(env), eval1=EvalConfig(env)),
        eval_frequency=4,
        save_model=True,
        log_frequency=1234,
        local_num_envs=1,
        num_actor_threads=2,  # Test multithreaded
        num_steps=2,
        num_minibatches=1,
        # If the whole thing deadlocks exit in some small multiple of 10 seconds
        queue_timeout=10,
    )

    args.total_timesteps = args.num_steps * args.num_actor_threads * args.local_num_envs * args.eval_frequency
    assert args.total_timesteps < 20

    writer = CheckingWriter(args, tmpdir)
    train(args, writer=writer)
