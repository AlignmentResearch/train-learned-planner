import contextlib
import dataclasses
import json
import logging
import math
import os
import queue
import random
import re
import shutil
import sys
import threading
import time
import warnings
from collections import deque
from ctypes import cdll
from functools import partial
from pathlib import Path
from typing import Any, Callable, Hashable, Iterator, List, Mapping, NamedTuple, Optional

import chex
import databind.core.converter
import farconf
import flax
import jax
import jax.numpy as jnp
import numpy as np
import optax
import wandb
from flax.training.train_state import TrainState
from names_generator import generate_name
from rich.pretty import pprint
from typing_extensions import Self

from cleanba.config import Args
from cleanba.convlstm import ConvLSTMConfig
from cleanba.environments import EpisodeEvalWrapper, convert_to_cleanba_config, random_seed
from cleanba.evaluate import EvalConfig
from cleanba.impala_loss import (
    SINGLE_DEVICE_UPDATE_DEVICES_AXIS,
    Rollout,
    single_device_update,
    tree_flatten_and_concat,
)
from cleanba.network import AgentParams, Policy, PolicyCarryT, label_and_learning_rate_for_params
from cleanba.optimizer import rmsprop_pytorch_style

log = logging.getLogger(__file__)


class ParamsPayload(NamedTuple):
    """Structured data for the params queue."""

    params: Any  # device_params
    policy_version: int  # learner_policy_version


class RolloutPayload(NamedTuple):
    """Structured data for the rollout queue."""

    global_step: int
    policy_version: int  # actor_policy_version
    update: int
    storage: Rollout  # sharded_storage
    params_queue_get_time: float
    device_thread_id: int


libcudart = None
if os.getenv("NSIGHT_ACTIVE", "0") == "1":
    libcudart = cdll.LoadLibrary("libcudart.so")


def unreplicate(tree):
    """Returns a single instance of a replicated array."""
    return jax.tree_util.tree_map(lambda x: x[0], tree)


class WandbWriter:
    step_digits: int
    named_save_dir: Path

    def __init__(self, cfg: "Args", wandb_cfg_extra_data: dict[str, Any] = {}):
        wandb_kwargs: dict[str, Any] = dict(
            name=os.environ.get("WANDB_JOB_NAME", generate_name(style="hyphen")),
            mode=os.environ.get("WANDB_MODE", "online"),
            group=os.environ.get("WANDB_RUN_GROUP", "default"),
        )
        try:
            wandb_kwargs.update(
                dict(
                    entity=os.environ["WANDB_ENTITY"],
                    project=os.environ["WANDB_PROJECT"],
                )
            )
        except KeyError:
            # If any of the essential WANDB environment variables are missing,
            # simply don't upload this run.
            # It's fine to do this without giving any indication because Wandb already prints that the run is offline.
            wandb_kwargs["mode"] = os.environ.get("WANDB_MODE", "offline")
        job_name = wandb_kwargs["name"]

        run_dir = cfg.base_run_dir / wandb_kwargs["group"]
        run_dir.mkdir(parents=True, exist_ok=True)

        jax_compile_cache = cfg.base_run_dir / "kernel-cache"
        jax_compile_cache.mkdir(exist_ok=True, parents=True)

        jax.config.update("jax_compilation_cache_dir", str(jax_compile_cache))
        jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
        jax.config.update("jax_persistent_cache_min_compile_time_secs", 10)
        jax.config.update("jax_persistent_cache_enable_xla_caches", "xla_gpu_per_fusion_autotune_cache_dir")

        old_run_dir_sym = run_dir / "wandb" / job_name
        run_id = None
        if old_run_dir_sym.exists() and not cfg.finetune_with_noop_head:
            # check if run-{alphanumeric_id}.wandb exists in old_run_dir and fetch the alphanumeric part
            run_id = next((f.name for f in old_run_dir_sym.iterdir() if re.match(r"run-([a-zA-Z0-9]+)\.wandb", f.name)))
            run_id = run_id.split(".")[0].split("-")[1]
            print(f"Resuming run {run_id} found in {old_run_dir_sym}")

        cfg_dict = farconf.to_dict(cfg)
        assert isinstance(cfg_dict, dict)

        wandb.init(
            **wandb_kwargs,
            config={**cfg_dict, **wandb_cfg_extra_data},
            save_code=True,  # Make sure git diff is saved
            dir=run_dir,
            monitor_gym=False,  # Must manually log videos to wandb
            sync_tensorboard=False,  # Manually log tensorboard
            settings=wandb.Settings(code_dir=str(Path(__file__).parent.parent)),
            resume="allow",
            id=run_id,
        )

        assert wandb.run is not None
        save_dir_no_local_files = Path(wandb.run.dir).parent
        self._save_dir = save_dir_no_local_files / "local-files"
        self._save_dir.mkdir()

        self.named_save_dir = Path(wandb.run.dir).parent.parent / job_name
        assert old_run_dir_sym == self.named_save_dir
        if self.named_save_dir.exists():
            # copy all checkpoints to the new run dir
            for f in (self.named_save_dir / "local-files").iterdir():
                shutil.move(f, self._save_dir / f.name)
            self.named_save_dir.unlink()

        self.named_save_dir.symlink_to(save_dir_no_local_files.absolute(), target_is_directory=True)

        self.step_digits = math.ceil(math.log10(cfg.total_timesteps))

    def add_scalar(self, name: str, value: int | float, global_step: int):
        wandb.log({name: value}, step=global_step)

    def add_dict(self, metrics: dict[str, int | float], global_step: int):
        wandb.log(metrics, step=global_step)

    @contextlib.contextmanager
    def save_dir(self, global_step: int) -> Iterator[Path]:
        name = f"cp_{global_step:0{self.step_digits}d}"
        out = self._save_dir / name
        out.mkdir()
        yield out

    def maybe_save_barrier(self) -> None:
        pass

    def reset_save_barrier(self) -> None:
        pass


@dataclasses.dataclass(frozen=True)
class RuntimeInformation:
    local_batch_size: int
    local_devices: list[jax.Device]
    local_minibatch_size: int
    world_size: int
    local_rank: int
    num_envs: int
    batch_size: int
    minibatch_size: int
    num_updates: int
    global_learner_devices: list[jax.Device]
    learner_devices: list[jax.Device]


@contextlib.contextmanager
def initialize_multi_device(args: Args) -> Iterator[RuntimeInformation]:
    local_batch_size = int(args.local_num_envs * args.num_steps * args.num_actor_threads * len(args.actor_device_ids))
    local_minibatch_size = int(local_batch_size // args.num_minibatches)
    assert args.local_num_envs % len(args.learner_device_ids) == 0, (
        "local_num_envs must be divisible by len(learner_device_ids)"
    )

    assert int(args.local_num_envs / len(args.learner_device_ids)) * args.num_actor_threads % args.num_minibatches == 0, (
        "int(local_num_envs / len(learner_device_ids)) must be divisible by num_minibatches"
    )

    distributed = args.distributed  # guard agiainst edits to `args`
    if args.distributed:
        jax.distributed.initialize()

    world_size = jax.process_count()
    local_rank = jax.process_index()
    num_envs = args.local_num_envs * world_size * args.num_actor_threads * len(args.actor_device_ids)
    batch_size = local_batch_size * world_size
    minibatch_size = local_minibatch_size * world_size
    num_updates = args.total_timesteps // (local_batch_size * world_size)  # this shouldn't include args.train_epochs
    local_devices = jax.local_devices()
    global_devices = jax.devices()
    learner_devices = [local_devices[d_id] for d_id in args.learner_device_ids]
    global_learner_devices = [
        global_devices[d_id + process_index * len(local_devices)]
        for process_index in range(world_size)
        for d_id in args.learner_device_ids
    ]

    yield RuntimeInformation(
        local_batch_size=local_batch_size,
        local_minibatch_size=local_minibatch_size,
        num_updates=num_updates,
        local_devices=local_devices,
        world_size=world_size,
        local_rank=local_rank,
        num_envs=num_envs,
        batch_size=batch_size,
        minibatch_size=minibatch_size,
        global_learner_devices=global_learner_devices,
        learner_devices=learner_devices,
    )

    global MUST_STOP_PROGRAM
    MUST_STOP_PROGRAM = True
    if distributed:
        jax.distributed.shutdown()


@jax.jit
def log_parameter_differences(params) -> dict[str, jax.Array]:
    max_params = jax.tree.map(lambda x: np.max(x, axis=0), params)
    min_params = jax.tree.map(lambda x: np.min(x, axis=0), params)

    flat_max_params = tree_flatten_and_concat(max_params)
    flat_min_params = tree_flatten_and_concat(min_params)
    abs_diff = jnp.abs(flat_max_params - flat_min_params)
    return dict(
        max_diff=jnp.max(abs_diff),
        min_diff=jnp.min(abs_diff),
        mean_diff=jnp.mean(abs_diff),
    )


MUST_STOP_PROGRAM: bool = False


@dataclasses.dataclass
class LoggingStats:
    params_queue_get_time: list[float]
    rollout_time: list[float]
    create_rollout_time: list[float]
    rollout_queue_put_time: list[float]

    env_recv_time: list[float]
    inference_time: list[float]
    storage_time: list[float]
    update_time: list[float]

    @classmethod
    def new_empty(cls: type[Self]) -> Self:
        init_dict = {f.name: [] for f in dataclasses.fields(cls)}
        return cls(**init_dict)

    def avg_and_flush(self) -> dict[str, float]:
        field_names = [f.name for f in dataclasses.fields(self)]
        out = {}
        for n in field_names:
            this_list = getattr(self, n)
            out[f"avg_{n}"] = float(np.mean(getattr(self, n)))
            this_list.clear()  # Flush the stats
        return out


@contextlib.contextmanager
def time_and_append(stats: list[float], name: str, step_num: int):
    start_time = time.time()
    with jax.named_scope(name):
        yield
    stats.append(time.time() - start_time)


@dataclasses.dataclass(order=True)
class PrioritizedItem:
    priority: int
    item: Any = dataclasses.field(compare=False)


@partial(jax.jit, static_argnames=["len_learner_devices"])
def _concat_and_shard_rollout_internal(
    storage: List[Rollout],
    last_obs: jax.Array,
    last_episode_starts: np.ndarray,
    last_value: jax.Array,
    len_learner_devices: int,
) -> Rollout:
    """
    Stack the Rollout steps over time, splitting them for every learner device.

    If element of `storage` has shape (batch, *others)

    Then each returned element has shape (len_learner_devices, time, batch//len_learner_devices, *others)

    where time=len(storage) for most things except:
    - For `obs_t` and `episode_starts_t`, time=len(storage)+1
    - for `carry_t` the return shape is (len_learner_devices, batch, *others)
    """

    def _split_over_batches(x):
        """Split for every learner device over `num_envs`"""
        batch, *others = x.shape
        assert batch % len_learner_devices == 0, f"Number of envs {batch=} not divisible by {len_learner_devices=}"

        return jnp.reshape(x, (len_learner_devices, batch // len_learner_devices, *others))

    out = Rollout(
        # Add the `last_obs` on the end of this rollout
        obs_t=jnp.stack([*(_split_over_batches(r.obs_t) for r in storage), _split_over_batches(last_obs)], axis=1),
        # Only store the first recurrent state
        carry_t=jax.tree.map(lambda x: jnp.expand_dims(_split_over_batches(x), axis=1), storage[0].carry_t),
        a_t=jnp.stack([_split_over_batches(r.a_t) for r in storage], axis=1),
        logits_t=jnp.stack([_split_over_batches(r.logits_t) for r in storage], axis=1),
        value_t=jnp.stack([*(_split_over_batches(r.value_t) for r in storage), _split_over_batches(last_value)], axis=1),
        r_t=jnp.stack([_split_over_batches(r.r_t) for r in storage], axis=1),
        episode_starts_t=jnp.stack(
            [*(_split_over_batches(r.episode_starts_t) for r in storage), _split_over_batches(last_episode_starts)], axis=1
        ),
        truncated_t=jnp.stack([_split_over_batches(r.truncated_t) for r in storage], axis=1),
    )
    return out


def concat_and_shard_rollout(
    storage: list[Rollout],
    last_obs: jax.Array,
    last_episode_starts: jax.Array,
    last_value: jax.Array,
    learner_devices: list[jax.Device],
) -> Rollout:
    partitioned_storage = _concat_and_shard_rollout_internal(
        storage, last_obs, last_episode_starts, last_value, len(learner_devices)
    )
    sharded_storage = jax.tree.map(lambda x: jax.device_put_sharded(list(x), devices=learner_devices), partitioned_storage)
    return sharded_storage


def rollout(
    initial_update: int,
    key: jax.random.PRNGKey,
    args: Args,
    runtime_info: RuntimeInformation,
    rollout_queue: queue.Queue,
    params_queue: queue.Queue,
    metrics_queue: queue.PriorityQueue,
    learner_devices: list[jax.Device],
    device_thread_id: int,
    actor_device: jax.Device,
    global_step: int = 0,
):
    actor_id: int = device_thread_id + args.num_actor_threads * jax.process_index()

    envs = EpisodeEvalWrapper(
        dataclasses.replace(
            args.train_env,
            seed=args.train_env.seed + actor_id,
            num_envs=args.local_num_envs,
        ).make()
    )

    eval_envs: list[tuple[str, EvalConfig]] = list(args.eval_envs.items())
    # Spread various eval envs among the threads
    this_thread_eval_cfg = [
        eval_envs[i] for i in range(actor_id, len(args.eval_envs), runtime_info.world_size * args.num_actor_threads)
    ]
    key = jax.random.PRNGKey(args.train_env.seed + actor_id)
    key, eval_keys = jax.random.split(key)
    this_thread_eval_keys = list(jax.random.split(eval_keys, len(this_thread_eval_cfg)))

    len_actor_device_ids = len(args.actor_device_ids)
    start_time = None

    log_stats = LoggingStats.new_empty()
    info_t = {}
    actor_policy_version = 0
    storage = []
    metrics = {}

    # Store the first observation
    obs_t, _ = envs.reset()

    # Initialize carry_t and episode_starts_t
    key, carry_key = jax.random.split(key)
    policy, carry_t, _ = args.net.init_params(envs, carry_key)
    episode_starts_t = np.ones(envs.num_envs, dtype=np.bool_)

    get_action_fn = jax.jit(partial(policy.apply, method=policy.get_action), static_argnames="temperature")

    global MUST_STOP_PROGRAM
    global libcudart
    for update in range(initial_update, runtime_info.num_updates + 2):
        if MUST_STOP_PROGRAM:
            break

        param_frequency = args.actor_update_frequency if update <= args.actor_update_cutoff else 1
        if libcudart is not None and update == 4:
            libcudart.cudaProfilerStart()

        with time_and_append(log_stats.update_time, "update", global_step):
            with time_and_append(log_stats.params_queue_get_time, "params_queue_get", global_step):
                num_steps_with_bootstrap = args.num_steps

                if args.concurrency:
                    # NOTE: `update - 1 != args.actor_update_frequency` is actually IMPORTANT — it allows us to start
                    # running policy collection concurrently with the learning process. It also ensures the actor's
                    # policy version is only 1 step behind the learner's policy version
                    if ((update - 1) % param_frequency == 0 and (update - 1) != param_frequency) or (
                        (update - 2) == param_frequency
                    ):
                        payload = params_queue.get(timeout=args.queue_timeout)
                        # NOTE: block here is important because otherwise this thread will call
                        # the jitted `get_action` function that hangs until the params are ready.
                        # This blocks the `get_action` function in other actor threads.
                        # See https://excalidraw.com/#json=hSooeQL707gE5SWY8wOSS,GeaN1eb2r24PPi75a3n14Q for a visual explanation.
                        params, actor_policy_version = jax.block_until_ready(payload.params), payload.policy_version
                else:
                    if (update - 1) % args.actor_update_frequency == 0:
                        payload = params_queue.get(timeout=args.queue_timeout)
                        params, actor_policy_version = payload.params, payload.policy_version

            with time_and_append(log_stats.rollout_time, "rollout", global_step):
                for _ in range(1, num_steps_with_bootstrap + 1):
                    global_step += (
                        args.local_num_envs * args.num_actor_threads * len_actor_device_ids * runtime_info.world_size
                    )

                    with time_and_append(log_stats.inference_time, "inference", global_step):
                        obs_t, episode_starts_t = jax.device_put((obs_t, episode_starts_t), device=actor_device)
                        carry_tplus1, a_t, logits_t, value_t, key = get_action_fn(
                            params, carry_t, obs_t, episode_starts_t, key
                        )
                        assert a_t.shape == (args.local_num_envs,)

                    with time_and_append(log_stats.env_recv_time, "step", global_step):
                        obs_tplus1, r_t, term_t, trunc_t, info_t = envs.step(a_t)
                        done_t = term_t | trunc_t
                        assert r_t.shape == (args.local_num_envs,)
                        assert done_t.shape == (args.local_num_envs,)

                    with time_and_append(log_stats.create_rollout_time, "create_rollout", global_step):
                        storage.append(
                            Rollout(
                                obs_t=obs_t,
                                carry_t=carry_t,
                                a_t=a_t,
                                logits_t=logits_t,
                                value_t=value_t,
                                r_t=r_t,
                                episode_starts_t=episode_starts_t,
                                truncated_t=trunc_t,
                            )
                        )
                        obs_t = obs_tplus1
                        carry_t = carry_tplus1
                        episode_starts_t = done_t

            with time_and_append(log_stats.storage_time, "storage", global_step):
                obs_t, episode_starts_t = jax.device_put((obs_t, episode_starts_t), device=actor_device)
                if args.loss.needs_last_value:
                    # We can't roll this out of the loop. In the next loop iteration, we will use the updated parameters
                    # to gather rollouts.
                    _, _, _, value_t, _ = get_action_fn(params, carry_t, obs_t, episode_starts_t, key)
                else:
                    value_t = jnp.full(value_t.shape, jnp.nan, dtype=value_t.dtype, device=value_t.device)

                sharded_storage = concat_and_shard_rollout(storage, obs_t, episode_starts_t, value_t, learner_devices)
                storage.clear()
                payload = RolloutPayload(
                    global_step=global_step,
                    policy_version=actor_policy_version,
                    update=update,
                    storage=sharded_storage,
                    params_queue_get_time=np.mean(log_stats.params_queue_get_time),
                    device_thread_id=device_thread_id,
                )
            with time_and_append(log_stats.rollout_queue_put_time, "rollout_queue_put", global_step):
                rollout_queue.put(payload, timeout=args.queue_timeout)

        # Log on all rollout threads
        if update % args.log_frequency == 0:
            total_rollout_time = np.sum(log_stats.rollout_time)
            stats_dict: dict[str, float] = log_stats.avg_and_flush()

            if start_time is None:
                steps_per_second = 0
                start_time = time.time()
            else:
                steps_per_second = global_step / (time.time() - start_time)

            charts_dict = jax.tree.map(jnp.mean, {k: v for k, v in info_t.items() if k.startswith("returned")})
            print(
                f"{update=} {device_thread_id=}, SPS={steps_per_second:.2f}, {global_step=}, ep_returns={charts_dict['returned_episode_return']:.2f}, ep_length={charts_dict['returned_episode_length']:.2f}, avg_rollout_time={stats_dict['avg_rollout_time']:.5f}"
            )

            # Perf: Time performance metrics
            metrics.update(
                {
                    f"Perf/{device_thread_id}/rollout_total": total_rollout_time,
                    f"Perf/{device_thread_id}/SPS": steps_per_second,
                    f"policy_versions/{device_thread_id}/actor": actor_policy_version,
                }
            )
            for k, v in stats_dict.items():
                metrics[f"Perf/{device_thread_id}/{k}"] = v

            # Charts: RL performance-related metrics
            for k, v in charts_dict.items():
                metrics[f"Charts/{device_thread_id}/{k}"] = v.item()

        # Evaluate whenever configured to
        if update in args.eval_at_steps:
            for i, (eval_name, env_config) in enumerate(this_thread_eval_cfg):
                print("Evaluating ", eval_name)
                this_thread_eval_keys[i], eval_key = jax.random.split(this_thread_eval_keys[i], 2)
                log_dict = env_config.run(policy, get_action_fn, params, key=eval_key)

                metrics.update({f"{eval_name}/{k}": v for k, v in log_dict.items() if not k.endswith("_all_episode_info")})

        if metrics:
            # Flush the metrics at most once per global_step. This way, in the learner we can check that all actor
            # threads have sent the metrics by simply counting.
            metrics_queue.put(PrioritizedItem(global_step, metrics), timeout=args.queue_timeout)
            metrics = {}
    if libcudart is not None:
        libcudart.cudaProfilerStop()


def linear_schedule(
    count: chex.Numeric,
    *,
    initial_learning_rate: float,
    minibatches_per_update: int,
    total_updates: int,
    train_epochs: int,
    final_learning_rate: float = 0.0,
) -> chex.Numeric:
    # anneal learning rate linearly after one training iteration which contains
    # (args.num_minibatches) gradient updates
    frac = (count // minibatches_per_update) / (total_updates * train_epochs)
    return initial_learning_rate + frac * (final_learning_rate - initial_learning_rate)


def make_optimizer(args: Args, params: AgentParams, total_updates: int):
    _linear_schedule = partial(
        linear_schedule,
        initial_learning_rate=args.learning_rate,
        final_learning_rate=args.final_learning_rate,
        minibatches_per_update=args.num_minibatches,
        total_updates=total_updates,
        train_epochs=args.train_epochs,
    )

    def _linear_or_constant_schedule(count: chex.Numeric) -> chex.Numeric:
        return _linear_schedule(count) if args.anneal_lr else args.learning_rate

    def optimizer_with_learning_rate(learning_rate: chex.Numeric) -> optax.GradientTransformation:
        if args.optimizer_yang:
            learning_rates, agent_param_labels = label_and_learning_rate_for_params(params, base_fan_in=args.base_fan_in)
            transform_chain = [
                optax.multi_transform(
                    transforms={k: optax.scale(lr) for k, lr in learning_rates.items()}, param_labels=agent_param_labels
                ),
            ]
        else:
            transform_chain = []

        def get_transform_chain(learning_rate: chex.Numeric | Callable[[chex.Numeric], chex.Numeric]):
            return [
                optax.clip_by_global_norm(args.max_grad_norm),
                (
                    optax.inject_hyperparams(rmsprop_pytorch_style)(
                        learning_rate=learning_rate,
                        eps=args.rmsprop_eps,
                        decay=args.rmsprop_decay,
                    )
                    if args.optimizer == "rmsprop"
                    else (
                        optax.inject_hyperparams(optax.adam)(
                            learning_rate=learning_rate,
                            b1=args.adam_b1,
                            b2=args.rmsprop_decay,
                            eps=args.rmsprop_eps,
                            eps_root=0.0,
                        )
                    )
                ),
            ]

        transform_chain += get_transform_chain(learning_rate)

        labels_per_parameter = jax.tree_util.tree_map(lambda x: "trainable", params)
        if args.finetune_with_noop_head:
            # Label actor head parameters as 'trainable'
            labels_per_parameter = jax.tree_util.tree_map(lambda x: "frozen", params)
            labels_per_parameter["params"]["actor_params"]["Output"] = jax.tree_util.tree_map(
                lambda x: "trainable", params["params"]["actor_params"]["Output"]
            )

            def frozen_schedule(
                count: chex.Numeric,
                frozen_finetune_steps_ratio: float,
                minibatches_per_update: int,
                total_updates: int,
                train_epochs: int,
                otherwise_learning_rate: chex.Numeric,
            ) -> chex.Numeric:
                # Return 0 during frozen period, then transition to normal learning rate
                frac = (count // minibatches_per_update) / (total_updates * train_epochs)
                return jnp.where(frac < frozen_finetune_steps_ratio, 0.0, otherwise_learning_rate)

            frozen_transform_chain = get_transform_chain(
                partial(
                    frozen_schedule,
                    frozen_finetune_steps_ratio=args.frozen_finetune_steps_ratio,
                    minibatches_per_update=args.num_minibatches,
                    total_updates=total_updates,
                    train_epochs=args.train_epochs,
                    otherwise_learning_rate=learning_rate,
                )
            )

            transforms: Mapping[Hashable, optax.GradientTransformation] = {
                "frozen": optax.chain(*frozen_transform_chain),
                "trainable": optax.chain(*transform_chain),
            }
            optimizer = optax.MultiSteps(
                optax.multi_transform(transforms, labels_per_parameter), every_k_schedule=args.gradient_accumulation_steps
            )
        else:
            optimizer = optax.MultiSteps(optax.chain(*transform_chain), every_k_schedule=args.gradient_accumulation_steps)
        return optimizer  # type: ignore

    # Inject learning rate schedule at the top level so we can just get it from .hyperparams and log it.
    return optax.inject_hyperparams(optimizer_with_learning_rate)(_linear_or_constant_schedule)


def get_checkpoint_number(filename):
    if not filename.startswith("cp_"):
        return None
    try:
        return int(filename.split("_")[1])
    except (IndexError, ValueError):
        return None


def train(
    args: Args,
    *,
    writer: Optional[WandbWriter] = None,
):
    warnings.filterwarnings("ignore", "", UserWarning, module="gymnasium.vector")

    if args.finetune_with_noop_head:
        args.train_env.nn_without_noop = False
        for eval_env_cfg in args.eval_envs.values():
            eval_env_cfg.env.nn_without_noop = False
    train_env_cfg = dataclasses.replace(args.train_env, num_envs=args.local_num_envs)

    with initialize_multi_device(args) as runtime_info, contextlib.closing(train_env_cfg.make()) as envs:
        pprint(runtime_info)
        if writer is None:
            writer = WandbWriter(args)

        global_step = 0

        # seeding
        random.seed(args.seed)
        np.random.seed(random_seed())
        key = jax.random.PRNGKey(random_seed())

        key, agent_params_subkey = jax.random.split(key, 2)

        policy, _, agent_params = args.net.init_params(envs, agent_params_subkey)

        load_path = args.load_path
        if args.load_path is None:
            potential_load_path = writer.named_save_dir / "local-files"
            cp_paths = os.listdir(potential_load_path)
            # Filter and sort valid checkpoints only
            valid_checkpoints = [(f, get_checkpoint_number(f)) for f in cp_paths]
            valid_checkpoints = [(f, num) for f, num in valid_checkpoints if num is not None]

            if valid_checkpoints:
                latest_checkpoint, global_step = max(valid_checkpoints, key=lambda x: x[1])
                load_path = potential_load_path / Path(latest_checkpoint)
                assert load_path.exists()
                print(f"Set {load_path=}")

        if load_path is None:
            agent_state = TrainState.create(
                apply_fn=None,
                params=agent_params,
                tx=make_optimizer(args, agent_params, total_updates=runtime_info.num_updates),
            )
            update = 1
        else:
            _, _, old_args, agent_state, update = load_train_state(
                load_path,
                env_cfg=train_env_cfg,
                finetune_with_noop_head=args.finetune_with_noop_head,
            )
            # args.learner_policy_version = getattr(args if args.finetune_with_noop_head else old_args, "learner_policy_version")
            if args.finetune_with_noop_head:
                agent_state = TrainState.create(
                    apply_fn=None,
                    params=agent_state.params,
                    tx=make_optimizer(args, agent_state.params, total_updates=runtime_info.num_updates),
                )
                update = 1
            else:
                args.learner_policy_version = old_args.learner_policy_version
            print(
                f"Loaded TrainState at {update=} and {args.learner_policy_version=} from {load_path=}. Here are the differences from `args` "
                "to the loaded args:"
            )
            print(farconf.config_diff(farconf.to_dict(args), farconf.to_dict(old_args)))

        multi_device_update = jax.pmap(
            jax.jit(
                partial(
                    single_device_update,
                    num_batches=args.num_minibatches * args.gradient_accumulation_steps,
                    get_logits_and_value=partial(policy.apply, method=policy.get_logits_and_value),
                    impala_cfg=args.loss,
                ),
                donate_argnames=("agent_state", "key"),
            ),
            axis_name=SINGLE_DEVICE_UPDATE_DEVICES_AXIS,
            devices=runtime_info.global_learner_devices,
        )

        params_queues = []
        rollout_queues = []
        metrics_queue = queue.PriorityQueue()

        unreplicated_params = agent_state.params
        key, *actor_keys = jax.random.split(key, 1 + len(args.actor_device_ids))
        for d_idx, d_id in enumerate(args.actor_device_ids):
            # Copy device_params so we can donate the agent_state in the multi_device_update
            device_params = jax.tree.map(
                partial(jnp.array, copy=True),
                jax.device_put(unreplicated_params, runtime_info.local_devices[d_id]),
            )
            for thread_id in range(args.num_actor_threads):
                params_queues.append(queue.Queue(maxsize=1))
                rollout_queues.append(queue.Queue(maxsize=1))
                params_queues[-1].put(ParamsPayload(params=device_params, policy_version=args.learner_policy_version))
                threading.Thread(
                    target=rollout,
                    args=(
                        update,
                        jax.device_put(actor_keys[d_idx], runtime_info.local_devices[d_id]),
                        args,
                        runtime_info,
                        rollout_queues[-1],
                        params_queues[-1],
                        metrics_queue,
                        runtime_info.learner_devices,
                        d_idx * args.num_actor_threads + thread_id,
                        runtime_info.local_devices[d_id],
                        global_step,
                    ),
                ).start()

        rollout_queue_get_time = deque(maxlen=20)
        agent_state = jax.device_put_replicated(agent_state, devices=runtime_info.global_learner_devices)

        actor_policy_version = 0

        global MUST_STOP_PROGRAM
        MUST_STOP_PROGRAM = False  # setting here as well to False to ensure multiple test cases don't override it
        while not MUST_STOP_PROGRAM:
            print("train learner_policy_version", args.learner_policy_version)
            args.learner_policy_version += 1
            rollout_queue_get_time_start = time.time()
            sharded_storages = []
            for d_idx, d_id in enumerate(args.actor_device_ids):
                for thread_id in range(args.num_actor_threads):
                    payload = rollout_queues[d_idx * args.num_actor_threads + thread_id].get(timeout=args.queue_timeout)
                    global_step = payload.global_step
                    actor_policy_version = payload.policy_version
                    update = payload.update
                    sharded_storages.append(payload.storage)
            rollout_queue_get_time.append(time.time() - rollout_queue_get_time_start)
            training_time_start = time.time()

            key, *epoch_keys = jax.random.split(key, 1 + args.train_epochs)
            permutation_key = jax.random.split(epoch_keys[0], len(runtime_info.global_learner_devices))
            (agent_state, metrics_dict) = multi_device_update(agent_state, sharded_storages, permutation_key)
            for epoch in range(1, args.train_epochs):
                permutation_key = jax.random.split(epoch_keys[epoch], len(runtime_info.global_learner_devices))
                (agent_state, metrics_dict) = multi_device_update(agent_state, sharded_storages, permutation_key)

            unreplicated_params = unreplicate(agent_state.params)
            if update > args.actor_update_cutoff or update % args.actor_update_frequency == 0:
                for d_idx, d_id in enumerate(args.actor_device_ids):
                    # Copy device_params so we can donate the agent_state in the multi_device_update
                    device_params = jax.tree.map(
                        partial(jnp.array, copy=True),
                        jax.device_put(unreplicated_params, runtime_info.local_devices[d_id]),
                    )
                    for thread_id in range(args.num_actor_threads):
                        params_queues[d_idx * args.num_actor_threads + thread_id].put(
                            ParamsPayload(params=device_params, policy_version=args.learner_policy_version),
                            timeout=args.queue_timeout,
                        )

            # Copy the parameters from the first device to all other learner devices
            if len(runtime_info.global_learner_devices) > 1 and args.learner_policy_version % args.sync_frequency == 0:
                # Check the maximum parameter difference
                param_diff_stats = log_parameter_differences(agent_state.params)
                for k, v in param_diff_stats.items():
                    writer.add_scalar(f"diffs/{k}", v.item(), global_step)
                    print(f"diffs/{k}", v.item(), global_step)

                unreplicated_agent_state = unreplicate(agent_state)
                agent_state = jax.device_put_replicated(unreplicated_agent_state, devices=runtime_info.global_learner_devices)

            # record rewards for plotting purposes
            if args.learner_policy_version % args.log_frequency == 0:
                metrics = {
                    "Perf/rollout_queue_get_time": np.mean(rollout_queue_get_time),
                    "Perf/training_time": time.time() - training_time_start,
                    "Perf/rollout_queue_size": rollout_queues[-1].qsize(),
                    "Perf/params_queue_size": params_queues[-1].qsize(),
                    "losses/value_loss": metrics_dict.pop("v_loss")[0].item(),
                    "losses/policy_loss": metrics_dict.pop("pg_loss")[0].item(),
                    "losses/entropy": metrics_dict.pop("ent_loss")[0].item(),
                    "losses/loss": metrics_dict.pop("loss")[0].item(),
                    "policy_versions/learner": args.learner_policy_version,
                }
                metrics.update({k: v[0].item() for k, v in metrics_dict.items()})

                lr = unreplicate(agent_state.opt_state.hyperparams["learning_rate"])
                assert lr is not None
                metrics["losses/learning_rate"] = lr

                # Receive actors' metrics from the metrics_queue, and once we have all of them plot them together
                #
                # If we get metrics from a future step, we just put them back in the queue for next time.
                # If it is a previous step, we regretfully throw them away.
                add_back_later_metrics = []
                num_actor_metrics = 0
                while num_actor_metrics < len(rollout_queues):
                    item = metrics_queue.get(timeout=args.queue_timeout)
                    actor_global_step, actor_metrics = item.priority, item.item
                    print(f"Got metrics from {actor_global_step=}")

                    if actor_global_step == global_step:
                        metrics.update(
                            {k: (v.item() if isinstance(v, jnp.ndarray) else v) for (k, v) in actor_metrics.items()}
                        )
                        num_actor_metrics += 1
                    elif actor_global_step > global_step:
                        add_back_later_metrics.append(item)
                    else:
                        log.warning(
                            f"Had to throw away metrics for global_step {actor_global_step}, which is less than the current {global_step=}. {actor_metrics}"
                        )
                # We're done. Write metrics and add back the ones for the future.
                writer.add_dict(metrics, global_step=global_step)
                for item in add_back_later_metrics:
                    metrics_queue.put(item)

                print(
                    global_step,
                    f"actor_policy_version={actor_policy_version}, actor_update={update}, learner_policy_version={args.learner_policy_version}, training time: {time.time() - training_time_start}s",
                )

            if args.save_model and args.learner_policy_version in args.eval_at_steps:
                print("Learner thread entering save barrier (should be last)")
                writer.maybe_save_barrier()
                writer.reset_save_barrier()

                with writer.save_dir(global_step) as dir:
                    save_train_state(dir, args, agent_state, update)

            if args.learner_policy_version >= runtime_info.num_updates:
                # The program is finished!
                return agent_state
    return agent_state


def save_train_state(dir: Path, args: Args, train_state: TrainState, update_step: int):
    with open(dir / "cfg.json", "w") as f:
        json.dump({"cfg": farconf.to_dict(args, Args), "update_step": update_step}, f)

    with open(dir / "model", "wb") as f:
        f.write(flax.serialization.to_bytes(train_state))


def load_train_state(
    dir: Path,
    env_cfg=None,  # environment config from the learned_planner package are also supported
    finetune_with_noop_head: bool = False,
) -> tuple[Policy, PolicyCarryT, Args, TrainState, int]:
    with open(dir / "cfg.json", "r") as f:
        args_dict = json.load(f)
    try:
        update_step = args_dict["update_step"]
    except KeyError:
        update_step = 1
    try:
        loaded_cfg = args_dict["cfg"]
    except KeyError:
        loaded_cfg = args_dict

    try:
        args = farconf.from_dict(loaded_cfg, Args)
    except databind.core.converter.ConversionError as e:
        if (m := re.fullmatch(r"^encountered extra keys: \{(.*)\}$", e.message)) is not None:
            keys_to_remove = {k.strip("'") for k in m.group(1).split(",")}
            print("Removing keys ", keys_to_remove)
            for k in keys_to_remove:
                del loaded_cfg[k]
                args = farconf.from_dict(loaded_cfg, Args)
        else:
            raise

    if env_cfg is None:
        env_cfg = args.train_env
    env_cfg = convert_to_cleanba_config(env_cfg)  # converts environment config from the learned_planner package

    if finetune_with_noop_head:
        env_cfg = dataclasses.replace(env_cfg, nn_without_noop=False)

    env = env_cfg.make()
    policy, carry, params = args.net.init_params(env, jax.random.PRNGKey(1234))

    local_batch_size = int(args.local_num_envs * args.num_steps * args.num_actor_threads * len(args.actor_device_ids))

    target_state = TrainState.create(
        apply_fn=None,
        params=params,
        tx=make_optimizer(args, params, total_updates=args.total_timesteps // local_batch_size),
    )

    with open(dir / "model", "rb") as f:
        train_state = flax.serialization.from_bytes(target_state, f.read())
    assert isinstance(train_state, TrainState)
    try:
        train_state = unreplicate(train_state)
    except TypeError:
        pass  # must be already unreplicated
    if isinstance(args.net, ConvLSTMConfig):
        for i in range(args.net.n_recurrent):
            this_cell = train_state.params["params"]["network_params"][f"cell_list_{i}"]
            if "fence" in this_cell:
                this_cell["fence"]["kernel"] = jnp.sum(
                    this_cell["fence"]["kernel"],
                    axis=2,
                    keepdims=True,
                )

    if finetune_with_noop_head:
        loaded_head = train_state.params["params"]["actor_params"]["Output"]
        transfer_head = jax.tree_util.tree_map(np.array, target_state.params["params"]["actor_params"]["Output"])
        num_actions = loaded_head["kernel"].shape[1]

        transfer_head["kernel"][:, :num_actions] = loaded_head["kernel"]
        transfer_head["bias"][:num_actions] = loaded_head["bias"]

        train_state.params["params"]["actor_params"]["Output"] = transfer_head

    return policy, carry, args, train_state, update_step


if __name__ == "__main__":
    args = farconf.parse_cli(sys.argv[1:], Args)
    pprint(args)
    train(args)
