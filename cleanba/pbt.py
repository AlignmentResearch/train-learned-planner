import argparse
import contextlib
import dataclasses
import json
import threading
from pathlib import Path
from typing import Any, Iterator, Optional

import farconf
import ray
from ray import train, tune
from ray.train import Checkpoint
from ray.tune.schedulers import PopulationBasedTraining

import cleanba.cleanba_impala
from cleanba import cleanba_impala as impala
from cleanba.config import Args, sokoban_drc_3_3


class RayWriter(cleanba.cleanba_impala.WandbWriter):
    steps_to_keep: int
    _last_global_step: int
    _metrics_by_step: dict[int, dict[str, float | int]]
    _metric_to_track: str

    def __init__(self, cfg: Args, metric_to_track: str, *, steps_to_keep: int = 2):
        super().__init__(cfg)

        self.steps_to_keep = steps_to_keep
        self._last_global_step = -1
        self._metrics_by_step = {}
        self._metric_to_track = metric_to_track
        self._save_barrier = threading.Barrier(1 + cfg.num_actor_threads, timeout=cfg.queue_timeout)

    def add_scalar(self, name: str, value: int | float, global_step: int):
        try:
            m = self._metrics_by_step[global_step]
        except KeyError:
            if len(self._metrics_by_step) > self.steps_to_keep:
                del self._metrics_by_step[min(self._metrics_by_step.keys())]

            m = self._metrics_by_step[global_step] = {}

        m[name] = value
        return super().add_scalar(name, value, global_step)

    @contextlib.contextmanager
    def save_dir(self, global_step: int) -> Iterator[Path]:
        with super().save_dir(global_step) as dir:
            yield dir

        m = self._metrics_by_step[global_step]
        if self._metric_to_track not in m:
            raise RuntimeError(f"Could not find the metric {self._metric_to_track} for {global_step=} in {m}")

        train.report(m, checkpoint=Checkpoint.from_directory(dir))

    def maybe_save_barrier(self) -> None:
        self._save_barrier.wait()

    def reset_save_barrier(self) -> None:
        self._save_barrier.reset()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--num-samples", type=int, default=1)
    return parser.parse_args()


cli_args = parse_args()


def load_args(checkpoint_dir: Path) -> dict[str, Any]:
    """Load the configuration from the checkpoint directory. Params (train_state) is loaded inside the training function."""
    with open(checkpoint_dir / "cfg.json", "r") as f:
        args_dict = json.load(f)
    args = farconf.from_dict(args_dict, Args)
    return args

metric_to_track = "valid_unfiltered/episode_success"
time_attr = "policy_versions/learner"


def trainable(config: dict[str, Any]):
    # If `train.get_checkpoint()` is populated, then we are resuming from a checkpoint.
    checkpoint = train.get_checkpoint()
    if checkpoint:
        with checkpoint.as_directory() as checkpoint_dir:
            args = load_args(checkpoint_dir)
            args.load_path = checkpoint_dir
    else:
        args = None

    args = custom_explore_fn(args, config)
    impala.train(
        args,
        writer=RayWriter(args, metric_to_track=metric_to_track),
    )


def custom_explore_fn(args: Optional[Args], config: dict[str, Any]) -> Args:
    if args is None:
        args = sokoban_drc_3_3()
    args = update_fn(args, config["optimizer"], config["rmsprop_decay"], config["max_grad_norm_mul"])

    return args


def update_fn(config: Args, optimizer, rmsprop_decay, max_grad_norm_mul):
    # TODO: this needs to be updated to the latest hyperparams for drc
    minibatch_size = 32
    n_envs = 256  # the paper says 200 actors
    assert n_envs % minibatch_size == 0

    logit_l2_coef = 1.5625e-6  # doesn't seem to matter much from now. May improve stability a tiny bit.

    world_size = 1
    len_actor_device_ids = 1
    num_actor_threads = 1

    train_epochs = 1
    actor_update_frequency = 1

    config.local_num_envs = n_envs
    config.num_steps = 20
    MUL = 3
    config.total_timesteps = 15632 * config.local_num_envs * config.num_steps * MUL

    global_step_multiplier = config.num_steps * config.local_num_envs * num_actor_threads * len_actor_device_ids * world_size
    assert config.total_timesteps % global_step_multiplier == 0
    num_updates = config.total_timesteps // global_step_multiplier

    config.eval_frequency = num_updates // (8 * MUL)

    config.actor_update_frequency = actor_update_frequency
    config.actor_update_cutoff = int(1e20)

    config.train_epochs = train_epochs
    config.num_actor_threads = 1
    config.num_minibatches = (config.local_num_envs * config.num_actor_threads) // minibatch_size

    config.sync_frequency = int(1e20)
    config.loss = dataclasses.replace(
        config.loss,
        vtrace_lambda=0.97,
        vf_coef=0.25,
        gamma=0.97,
        ent_coef=0.01,
        normalize_advantage=False,
        logit_l2_coef=logit_l2_coef,
        weight_l2_coef=logit_l2_coef / 100,
    )
    config.base_fan_in = 1
    config.anneal_lr = True

    config.optimizer = optimizer
    config.adam_b1 = 0.9
    config.rmsprop_decay = rmsprop_decay
    config.learning_rate = 4e-4
    config.max_grad_norm = 6.25e-5 * max_grad_norm_mul
    config.rmsprop_eps = 1.5625e-07
    config.optimizer_yang = False

    config.save_model = True
    config.base_run_dir = Path("/training/cleanba")
    return config


perturbation_interval = 5
hyperparam_mutations = {
    "optimizer": tune.choice(["adam", "rmsprop"]),
    "rmsprop_decay": tune.choice([0.99, 0.999]),
    "max_grad_norm_mul": tune.choice([4, 8, 16]),
}
param_space = dict(hyperparam_mutations, checkpoint_interval=perturbation_interval)
num_samples = cli_args.num_samples
max_failures = 1

scheduler = PopulationBasedTraining(
    time_attr=time_attr,
    perturbation_interval=perturbation_interval,
    metric=metric_to_track,
    mode="max",
    hyperparam_mutations=hyperparam_mutations,
)

ray.init()

trainable_with_resources = tune.with_resources(trainable, {"gpu": 1})
tuner = tune.Tuner(
    trainable=trainable_with_resources,
    run_config=train.RunConfig(
        name="pbt_test",
        # Stop when we've reached a threshold accuracy, or a maximum
        # training_iteration, whichever comes first
        stop={metric_to_track: 0.98, time_attr: 10**9},
        checkpoint_config=train.CheckpointConfig(
            checkpoint_score_attribute=metric_to_track,
            num_to_keep=4,
        ),
        storage_path="/training/cleanba",
        failure_config=train.FailureConfig(max_failures=max_failures),
    ),
    tune_config=tune.TuneConfig(
        scheduler=scheduler,
        num_samples=num_samples,
    ),
    param_space=param_space,
)
tuner.fit()
