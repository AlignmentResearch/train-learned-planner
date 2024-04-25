import contextlib
import threading
from pathlib import Path
from typing import Any, Iterator

import ray
from ray import train, tune
from ray.train import Checkpoint
from ray.tune.schedulers import PopulationBasedTraining

import cleanba.cleanba_impala
from cleanba.config import Args


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


def trainable(config: dict[str, Any]):
    # Create our data loaders, model, and optmizer.

    pass
    # step = 1

    # # If `train.get_checkpoint()` is populated, then we are resuming from a checkpoint.
    # checkpoint = train.get_checkpoint()
    # if checkpoint:
    #     with checkpoint.as_directory() as checkpoint_dir:
    #         checkpoint_dict = load(checkpoint_dir)

    #     # Load model state and iteration step from checkpoint.
    #     model.load_state_dict(checkpoint_dict["model_state_dict"])
    #     # Load optimizer state (needed since we're using momentum),
    #     # then set the `lr` and `momentum` according to the config.
    #     optimizer.load_state_dict(checkpoint_dict["optimizer_state_dict"])
    #     for param_group in optimizer.param_groups:
    #         if "lr" in config:
    #             param_group["lr"] = config["lr"]
    #         if "momentum" in config:
    #             param_group["momentum"] = config["momentum"]

    #     # Note: Make sure to increment the checkpointed step by 1 to get the current step.
    #     last_step = checkpoint_dict["step"]
    #     step = last_step + 1

    # while True:

    #     ray.tune.examples.mnist_pytorch.train_func(model, optimizer, train_loader)
    #     acc = test_func(model, test_loader)
    #     metrics = {"mean_accuracy": acc, "lr": config["lr"]}

    #     # Every `checkpoint_interval` steps, checkpoint our current state.
    #     if step % config["checkpoint_interval"] == 0:
    #         path = Path(f"/Users/adria/ray_tmp/{uuid.uuid4()}")
    #         path.mkdir()
    #         torch.save(
    #             {
    #                 "step": step,
    #                 "model_state_dict": model.state_dict(),
    #                 "optimizer_state_dict": optimizer.state_dict(),
    #             },
    #             path / "checkpoint.pt",
    #         )
    #         train.report(metrics, checkpoint=Checkpoint.from_directory(str(path)))
    #     else:
    #         train.report(metrics)

    #     step += 1


perturbation_interval = 5
scheduler = PopulationBasedTraining(
    time_attr="training_iteration",
    perturbation_interval=perturbation_interval,
    metric="mean_accuracy",
    mode="max",
    hyperparam_mutations={
        # distribution for resampling
        "lr": tune.uniform(0.0001, 1),
        # allow perturbations within this set of categorical values
        "momentum": [0.8, 0.9, 0.99],
    },
)

ray.init()

trainable_with_resources = tune.with_resources(trainable, {"gpu": 1})
tuner = tune.Tuner(
    trainable=trainable_with_resources,
    run_config=train.RunConfig(
        name="pbt_test",
        # Stop when we've reached a threshold accuracy, or a maximum
        # training_iteration, whichever comes first
        stop={"mean_accuracy": 0.96, "training_iteration": 50},
        checkpoint_config=train.CheckpointConfig(
            checkpoint_score_attribute="mean_accuracy",
            num_to_keep=4,
        ),
        storage_path="/Users/adria/ray_results",
        failure_config=train.FailureConfig(max_failures=3),
    ),
    tune_config=tune.TuneConfig(
        scheduler=scheduler,
        num_samples=4,
    ),
    param_space={
        "lr": tune.uniform(0.001, 1),
        "momentum": tune.uniform(0.001, 1),
        "checkpoint_interval": perturbation_interval,
    },
)
tuner.fit()
