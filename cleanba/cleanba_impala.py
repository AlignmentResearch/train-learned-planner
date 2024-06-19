import contextlib
import dataclasses
import json
import math
import os
import queue
import random
import sys
import threading
import time
import warnings
from collections import deque
from functools import partial
from pathlib import Path
from typing import Any, Iterator, List, Optional

import chex
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
from cleanba.environments import random_seed
from cleanba.evaluate import EvalConfig
from cleanba.impala_loss import (
    SINGLE_DEVICE_UPDATE_DEVICES_AXIS,
    Rollout,
    single_device_update,
    tree_flatten_and_concat,
)
from cleanba.network import AgentParams, label_and_learning_rate_for_params
from cleanba.optimizer import rmsprop_pytorch_style

# Make Jax CPU use 1 thread only https://github.com/google/jax/issues/743
os.environ["XLA_FLAGS"] = (
    os.environ.get("XLA_FLAGS", "") + " --xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"
)
# Fix CUDNN non-determinism; https://github.com/google/jax/issues/4823#issuecomment-952835771
os.environ["TF_XLA_FLAGS"] = (
    os.environ.get("TF_XLA_FLAGS", "") + " --xla_gpu_autotune_level=2 --xla_gpu_deterministic_reductions"
)

# Fix CUDNN non-determinism; https://github.com/google/jax/issues/4823#issuecomment-952835771
os.environ["TF_CUDNN_DETERMINISTIC"] = "1"


def unreplicate(tree):
    """Returns a single instance of a replicated array."""
    return jax.tree_util.tree_map(lambda x: x[0], tree)


class WandbWriter:
    step_digits: int
    named_save_dir: Path

    def __init__(self, cfg: "Args"):
        wandb_kwargs: dict[str, Any]
        try:
            wandb_kwargs = dict(
                entity=os.environ["WANDB_ENTITY"],
                name=os.environ.get("WANDB_JOB_NAME", generate_name(style="hyphen")),
                project=os.environ["WANDB_PROJECT"],
                group=os.environ["WANDB_RUN_GROUP"],
                mode=os.environ.get("WANDB_MODE", "online"),  # Default to online here
            )
            job_name = wandb_kwargs["name"]
        except KeyError:
            # If any of the essential WANDB environment variables are missing,
            # simply don't upload this run.
            # It's fine to do this without giving any indication because Wandb already prints that the run is offline.

            wandb_kwargs = dict(mode=os.environ.get("WANDB_MODE", "offline"), group="default")
            job_name = "develop"

        run_dir = cfg.base_run_dir / wandb_kwargs["group"]
        run_dir.mkdir(parents=True, exist_ok=True)
        cfg_dict = farconf.to_dict(cfg, Args)
        assert isinstance(cfg_dict, dict)

        wandb.init(
            **wandb_kwargs,
            config=cfg_dict,
            save_code=True,  # Make sure git diff is saved
            dir=run_dir,
            monitor_gym=False,  # Must manually log videos to wandb
            sync_tensorboard=False,  # Manually log tensorboard
            settings=wandb.Settings(code_dir=str(Path(__file__).parent.parent)),
        )

        assert wandb.run is not None
        save_dir_no_local_files = Path(wandb.run.dir).parent
        self._save_dir = save_dir_no_local_files / "local-files"
        self._save_dir.mkdir()

        self.named_save_dir = Path(wandb.run.dir).parent.parent / job_name
        if not self.named_save_dir.exists():
            self.named_save_dir.symlink_to(save_dir_no_local_files, target_is_directory=True)

        self.step_digits = math.ceil(math.log10(cfg.total_timesteps))

    def add_scalar(self, name: str, value: int | float, global_step: int):
        wandb.log({name: value}, step=global_step)

    @contextlib.contextmanager
    def save_dir(self, global_step: int) -> Iterator[Path]:
        name = f"cp_{{step:0{self.step_digits}d}}".format(step=global_step)
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
    assert (
        args.local_num_envs % len(args.learner_device_ids) == 0
    ), "local_num_envs must be divisible by len(learner_device_ids)"

    assert (
        int(args.local_num_envs / len(args.learner_device_ids)) * args.num_actor_threads % args.num_minibatches == 0
    ), "int(local_num_envs / len(learner_device_ids)) must be divisible by num_minibatches"

    distributed = args.distributed  # guard agiainst edits to `args`
    if args.distributed:
        jax.distributed.initialize()

    world_size = jax.process_count()
    local_rank = jax.process_index()
    num_envs = args.local_num_envs * world_size * args.num_actor_threads * len(args.actor_device_ids)
    batch_size = local_batch_size * world_size
    minibatch_size = local_minibatch_size * world_size
    num_updates = args.total_timesteps // (local_batch_size * world_size)
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
    episode_returns: list[float]
    episode_lengths: list[float]
    episode_success: list[float]
    params_queue_get_time: list[float]
    rollout_time: list[float]
    create_rollout_time: list[float]
    rollout_queue_put_time: list[float]

    env_recv_time: list[float]
    inference_time: list[float]
    storage_time: list[float]
    device2host_time: list[float]
    env_send_time: list[float]
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
def time_and_append(stats: list[float]):
    start_time = time.time()
    yield
    stats.append(time.time() - start_time)


@partial(jax.jit, static_argnames=["len_learner_devices"])
def _concat_and_shard_rollout_internal(
    storage: List[Rollout], last_obs: jax.Array, last_episode_starts: np.ndarray, len_learner_devices: int
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
        r_t=jnp.stack([_split_over_batches(r.r_t) for r in storage], axis=1),
        episode_starts_t=jnp.stack(
            [*(_split_over_batches(r.episode_starts_t) for r in storage), _split_over_batches(last_episode_starts)], axis=1
        ),
        truncated_t=jnp.stack([_split_over_batches(r.truncated_t) for r in storage], axis=1),
    )
    return out


def concat_and_shard_rollout(
    storage: list[Rollout], last_obs: jax.Array, last_episode_starts: np.ndarray, learner_devices: list[jax.Device]
) -> Rollout:
    partitioned_storage = _concat_and_shard_rollout_internal(storage, last_obs, last_episode_starts, len(learner_devices))
    sharded_storage = jax.tree.map(lambda x: jax.device_put_sharded(list(x), devices=learner_devices), partitioned_storage)
    return sharded_storage


def rollout(
    initial_update: int,
    key: jax.random.PRNGKey,
    args: Args,
    runtime_info: RuntimeInformation,
    rollout_queue: queue.Queue,
    params_queue: queue.Queue,
    writer,
    learner_devices: list[jax.Device],
    device_thread_id: int,
    actor_device,
):
    actor_id: int = device_thread_id + args.num_actor_threads * jax.process_index()

    envs = dataclasses.replace(
        args.train_env,
        seed=args.train_env.seed + actor_id,
        num_envs=args.local_num_envs,
    ).make()

    eval_envs: list[tuple[str, EvalConfig]] = list(args.eval_envs.items())
    # Spread various eval envs among the threads
    this_thread_eval_cfg = [
        eval_envs[i] for i in range(actor_id, len(args.eval_envs), runtime_info.world_size * args.num_actor_threads)
    ]
    key = jax.random.PRNGKey(args.train_env.seed + actor_id)
    key, eval_keys = jax.random.split(key)
    this_thread_eval_keys = list(jax.random.split(eval_keys, len(this_thread_eval_cfg)))

    len_actor_device_ids = len(args.actor_device_ids)
    global_step = 0
    start_time = time.time()

    log_stats = LoggingStats.new_empty()
    # Counters for episode length and episode return
    episode_returns = np.zeros((args.local_num_envs,), dtype=np.float32)
    episode_lengths = np.zeros((args.local_num_envs,), dtype=np.float32)
    returned_episode_returns = np.zeros((args.local_num_envs,), dtype=np.float32)
    returned_episode_lengths = np.zeros((args.local_num_envs,), dtype=np.float32)
    returned_episode_success = np.zeros((args.local_num_envs,), dtype=np.bool_)

    actor_policy_version = 0
    storage = []

    # Store the first observation
    obs_t, _ = envs.reset()

    # Initialize carry_t and episode_starts_t
    key, carry_key = jax.random.split(key)
    policy, carry_t, _ = args.net.init_params(envs, carry_key)
    episode_starts_t = np.ones(envs.num_envs, dtype=np.bool_)
    get_action_fn = jax.jit(partial(policy.apply, method=policy.get_action), static_argnames="temperature")

    global MUST_STOP_PROGRAM
    for update in range(initial_update, runtime_info.num_updates + 2):
        if MUST_STOP_PROGRAM:
            break

        param_frequency = args.actor_update_frequency if update <= args.actor_update_cutoff else 1

        with time_and_append(log_stats.update_time):
            with time_and_append(log_stats.params_queue_get_time):
                num_steps_with_bootstrap = args.num_steps

                if args.concurrency:
                    # NOTE: `update - 1 != args.actor_update_frequency` is actually IMPORTANT — it allows us to start
                    # running policy collection concurrently with the learning process. It also ensures the actor's
                    # policy version is only 1 step behind the learner's policy version
                    if ((update - 1) % param_frequency == 0 and (update - 1) != param_frequency) or (
                        (update - 2) == param_frequency
                    ):
                        params, actor_policy_version = params_queue.get(timeout=args.queue_timeout)
                        # NOTE: block here is important because otherwise this thread will call
                        # the jitted `get_action` function that hangs until the params are ready.
                        # This blocks the `get_action` function in other actor threads.
                        # See https://excalidraw.com/#json=hSooeQL707gE5SWY8wOSS,GeaN1eb2r24PPi75a3n14Q for a visual explanation.
                        jax.block_until_ready(params)
                else:
                    if (update - 1) % args.actor_update_frequency == 0:
                        params, actor_policy_version = params_queue.get(timeout=args.queue_timeout)

            with time_and_append(log_stats.rollout_time):
                for _ in range(1, num_steps_with_bootstrap + 1):
                    global_step += (
                        args.local_num_envs * args.num_actor_threads * len_actor_device_ids * runtime_info.world_size
                    )

                    with time_and_append(log_stats.inference_time):
                        carry_tplus1, a_t, logits_t, key = get_action_fn(params, carry_t, obs_t, episode_starts_t, key)

                    with time_and_append(log_stats.device2host_time):
                        cpu_action = np.array(a_t)

                    with time_and_append(log_stats.env_send_time):
                        envs.step_async(cpu_action)

                    with time_and_append(log_stats.env_recv_time):
                        obs_tplus1, r_t, term_t, trunc_t, info_t = envs.step_wait()
                        done_t = term_t | trunc_t

                    with time_and_append(log_stats.create_rollout_time):
                        storage.append(
                            Rollout(
                                obs_t=obs_t,
                                carry_t=carry_t,
                                a_t=a_t,
                                logits_t=logits_t,
                                r_t=r_t,
                                episode_starts_t=episode_starts_t,
                                truncated_t=trunc_t,
                            )
                        )
                        obs_t = obs_tplus1
                        carry_t = carry_tplus1
                        episode_starts_t = done_t

                        # Atari envs clip their reward to [-1, 1], meaning we need to use the reward in `info` to get
                        # the true return.
                        non_clipped_reward = info_t.get("reward", r_t)

                        episode_returns[:] += non_clipped_reward
                        log_stats.episode_returns.extend(episode_returns[done_t])
                        returned_episode_returns[done_t] = episode_returns[done_t]
                        episode_returns[:] *= ~done_t

                        episode_lengths[:] += 1
                        log_stats.episode_lengths.extend(episode_lengths[done_t])
                        returned_episode_lengths[done_t] = episode_lengths[done_t]
                        episode_lengths[:] *= ~done_t

                        log_stats.episode_success.extend(map(float, term_t[done_t]))
                        returned_episode_success[done_t] = term_t[done_t]

            with time_and_append(log_stats.storage_time):
                sharded_storage = concat_and_shard_rollout(storage, obs_t, episode_starts_t, learner_devices)
                storage.clear()
                payload = (
                    global_step,
                    actor_policy_version,
                    update,
                    sharded_storage,
                    np.mean(log_stats.params_queue_get_time),
                    device_thread_id,
                )
            with time_and_append(log_stats.rollout_queue_put_time):
                rollout_queue.put(payload, timeout=args.queue_timeout)

        # Log on all rollout threads
        if update % args.log_frequency == 0:
            inner_loop_time = (
                np.sum(log_stats.env_recv_time)
                + np.sum(log_stats.create_rollout_time)
                + np.sum(log_stats.inference_time)
                + np.sum(log_stats.device2host_time)
                + np.sum(log_stats.env_send_time)
            )
            total_rollout_time = np.sum(log_stats.rollout_time)
            middle_loop_time = (
                total_rollout_time
                + np.sum(log_stats.storage_time)
                + np.sum(log_stats.params_queue_get_time)
                + np.sum(log_stats.rollout_queue_put_time)
            )
            outer_loop_time = np.sum(log_stats.update_time)

            stats_dict: dict[str, float] = log_stats.avg_and_flush()
            steps_per_second = global_step / (time.time() - start_time)
            print(
                f"{update=} {device_thread_id=}, SPS={steps_per_second:.2f}, {global_step=}, avg_episode_returns={stats_dict['avg_episode_returns']:.2f}, avg_episode_length={stats_dict['avg_episode_lengths']:.2f}, avg_rollout_time={stats_dict['avg_rollout_time']:.5f}"
            )

            for k, v in stats_dict.items():
                if k.endswith("_time"):
                    writer.add_scalar(f"stats/{device_thread_id}/{k}", v, global_step)
                else:
                    writer.add_scalar(f"charts/{device_thread_id}/{k}", v, global_step)

            writer.add_scalar(f"charts/{device_thread_id}/instant_avg_episode_length", np.mean(episode_lengths), global_step)
            writer.add_scalar(f"charts/{device_thread_id}/instant_avg_episode_return", np.mean(episode_returns), global_step)
            writer.add_scalar(
                f"charts/{device_thread_id}/returned_avg_episode_length", np.mean(returned_episode_lengths), global_step
            )
            writer.add_scalar(
                f"charts/{device_thread_id}/returned_avg_episode_return", np.mean(returned_episode_returns), global_step
            )
            writer.add_scalar(
                f"charts/{device_thread_id}/returned_avg_episode_success", np.mean(returned_episode_success), global_step
            )

            writer.add_scalar(
                f"stats/{device_thread_id}/inner_time_efficiency", inner_loop_time / total_rollout_time, global_step
            )
            writer.add_scalar(
                f"stats/{device_thread_id}/middle_time_efficiency", middle_loop_time / outer_loop_time, global_step
            )
            writer.add_scalar(f"charts/{device_thread_id}/SPS", steps_per_second, global_step)

            writer.add_scalar(f"policy_versions/actor_{device_thread_id}", actor_policy_version, global_step)

        if update in args.eval_at_steps:
            for i, (eval_name, env_config) in enumerate(this_thread_eval_cfg):
                print("Evaluating ", eval_name)
                this_thread_eval_keys[i], eval_key = jax.random.split(this_thread_eval_keys[i], 2)
                log_dict = env_config.run(policy, get_action_fn, params, key=eval_key)
                for k, v in log_dict.items():
                    writer.add_scalar(f"{eval_name}/{k}", v, global_step)


def linear_schedule(
    count: chex.Numeric,
    *,
    initial_learning_rate: float,
    minibatches_per_update: int,
    total_updates: int,
    final_learning_rate: float = 0.0,
) -> chex.Numeric:
    # anneal learning rate linearly after one training iteration which contains
    # (args.num_minibatches) gradient updates
    frac = (count // minibatches_per_update) / total_updates
    return initial_learning_rate + frac * (final_learning_rate - initial_learning_rate)


def make_optimizer(args: Args, params: AgentParams, total_updates: int):
    if args.optimizer_yang:
        learning_rates, agent_param_labels = label_and_learning_rate_for_params(params, base_fan_in=args.base_fan_in)
        transform_chain = [
            optax.multi_transform(
                transforms={k: optax.scale(lr) for k, lr in learning_rates.items()}, param_labels=agent_param_labels
            ),
        ]
    else:
        transform_chain = []

    _linear_schedule = partial(
        linear_schedule,
        initial_learning_rate=args.learning_rate,
        final_learning_rate=args.final_learning_rate,
        minibatches_per_update=args.num_minibatches,
        total_updates=total_updates,
    )

    transform_chain += [
        optax.clip_by_global_norm(args.max_grad_norm),
        (
            optax.inject_hyperparams(rmsprop_pytorch_style)(
                learning_rate=_linear_schedule if args.anneal_lr else args.learning_rate,
                eps=args.rmsprop_eps,
                decay=args.rmsprop_decay,
            )
            if args.optimizer == "rmsprop"
            else (
                optax.inject_hyperparams(optax.adam)(
                    learning_rate=_linear_schedule if args.anneal_lr else args.learning_rate,
                    b1=args.adam_b1,
                    b2=args.rmsprop_decay,
                    eps=args.rmsprop_eps,
                    eps_root=0.0,
                )
            )
        ),
    ]
    return optax.MultiSteps(optax.chain(*transform_chain), every_k_schedule=args.gradient_accumulation_steps)


def train(
    args: Args,
    *,
    writer: Optional[WandbWriter] = None,
):
    warnings.filterwarnings("ignore", "", UserWarning, module="gymnasium.vector")

    train_env_cfg = dataclasses.replace(args.train_env, num_envs=args.local_num_envs)
    with initialize_multi_device(args) as runtime_info, contextlib.closing(train_env_cfg.make()) as envs:
        pprint(runtime_info)
        if writer is None:
            writer = WandbWriter(args)

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
            cp_paths.sort()
            if cp_paths:
                load_path = potential_load_path / Path(cp_paths[-1])
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
            old_args, agent_state, update = load_train_state(load_path)
            print(
                f"Loaded TrainState at {update=} from {load_path=}. Here are the differences from `args` "
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
                )
            ),
            axis_name=SINGLE_DEVICE_UPDATE_DEVICES_AXIS,
            devices=runtime_info.global_learner_devices,
        )

        params_queues = []
        rollout_queues = []

        unreplicated_params = agent_state.params
        key, *actor_keys = jax.random.split(key, 1 + len(args.actor_device_ids))
        for d_idx, d_id in enumerate(args.actor_device_ids):
            device_params = jax.device_put(unreplicated_params, runtime_info.local_devices[d_id])
            for thread_id in range(args.num_actor_threads):
                params_queues.append(queue.Queue(maxsize=1))
                rollout_queues.append(queue.Queue(maxsize=1))
                params_queues[-1].put((device_params, args.learner_policy_version))
                threading.Thread(
                    target=rollout,
                    args=(
                        update,
                        jax.device_put(actor_keys[d_idx], runtime_info.local_devices[d_id]),
                        args,
                        runtime_info,
                        rollout_queues[-1],
                        params_queues[-1],
                        writer,
                        runtime_info.learner_devices,
                        d_idx * args.num_actor_threads + thread_id,
                        runtime_info.local_devices[d_id],
                    ),
                ).start()

        rollout_queue_get_time = deque(maxlen=10)
        agent_state = jax.device_put_replicated(agent_state, devices=runtime_info.global_learner_devices)

        actor_policy_version = 0

        global MUST_STOP_PROGRAM
        while not MUST_STOP_PROGRAM:
            args.learner_policy_version += 1
            rollout_queue_get_time_start = time.time()
            sharded_storages = []
            for d_idx, d_id in enumerate(args.actor_device_ids):
                for thread_id in range(args.num_actor_threads):
                    (
                        global_step,
                        actor_policy_version,
                        update,
                        sharded_storage,
                        avg_params_queue_get_time,
                        device_thread_id,
                    ) = rollout_queues[d_idx * args.num_actor_threads + thread_id].get(timeout=args.queue_timeout)
                    sharded_storages.append(sharded_storage)
            rollout_queue_get_time.append(time.time() - rollout_queue_get_time_start)
            training_time_start = time.time()
            for _ in range(args.train_epochs):
                (
                    agent_state,
                    metrics_dict,
                ) = multi_device_update(
                    agent_state,
                    sharded_storages,
                )
            unreplicated_params = unreplicate(agent_state.params)
            if update > args.actor_update_cutoff or update % args.actor_update_frequency == 0:
                for d_idx, d_id in enumerate(args.actor_device_ids):
                    device_params = jax.device_put(unreplicated_params, runtime_info.local_devices[d_id])
                    for thread_id in range(args.num_actor_threads):
                        params_queues[d_idx * args.num_actor_threads + thread_id].put(
                            (device_params, args.learner_policy_version), timeout=args.queue_timeout
                        )

            # Copy the parameters from the first device to all other learner devices
            if args.learner_policy_version % args.sync_frequency == 0:
                # Check the maximum parameter difference
                param_diff_stats = log_parameter_differences(agent_state.params)
                for k, v in param_diff_stats.items():
                    writer.add_scalar(f"diffs/{k}", v.item(), global_step)
                    print(f"diffs/{k}", v.item(), global_step)

                unreplicated_agent_state = unreplicate(agent_state)
                agent_state = jax.device_put_replicated(unreplicated_agent_state, devices=runtime_info.global_learner_devices)

            # record rewards for plotting purposes
            if args.learner_policy_version % args.log_frequency == 0:
                writer.add_scalar(
                    "stats/rollout_queue_get_time",
                    np.mean(rollout_queue_get_time),
                    global_step,
                )
                writer.add_scalar(
                    "stats/rollout_params_queue_get_time_diff",
                    np.mean(rollout_queue_get_time) - avg_params_queue_get_time,
                    global_step,
                )
                writer.add_scalar("stats/training_time", time.time() - training_time_start, global_step)
                writer.add_scalar("stats/rollout_queue_size", rollout_queues[-1].qsize(), global_step)
                writer.add_scalar("stats/params_queue_size", params_queues[-1].qsize(), global_step)
                print(
                    global_step,
                    f"actor_policy_version={actor_policy_version}, actor_update={update}, learner_policy_version={args.learner_policy_version}, training time: {time.time() - training_time_start}s",
                )
                writer.add_scalar("losses/value_loss", metrics_dict.pop("v_loss")[0].item(), global_step)
                writer.add_scalar("losses/policy_loss", metrics_dict.pop("pg_loss")[0].item(), global_step)
                writer.add_scalar("losses/entropy", metrics_dict.pop("ent_loss")[0].item(), global_step)
                writer.add_scalar("losses/loss", metrics_dict.pop("loss")[0].item(), global_step)

                for name, value in metrics_dict.items():
                    writer.add_scalar(name, value[0].item(), global_step)

                writer.add_scalar("policy_versions/learner", args.learner_policy_version, global_step)

            if args.save_model and args.learner_policy_version in args.eval_at_steps:
                print("Learner thread entering save barrier (should be last)")
                writer.maybe_save_barrier()
                writer.reset_save_barrier()

                with writer.save_dir(global_step) as dir:
                    save_train_state(dir, args, agent_state, update)

            if args.learner_policy_version >= runtime_info.num_updates:
                # The program is finished!
                return


def save_train_state(dir: Path, args: Args, train_state: TrainState, update_step: int):
    with open(dir / "cfg.json", "w") as f:
        json.dump({"cfg": farconf.to_dict(args, Args), "update_step": update_step}, f)

    with open(dir / "model", "wb") as f:
        f.write(flax.serialization.to_bytes(train_state))


def load_train_state(dir: Path) -> tuple[Args, TrainState, int]:
    with open(dir / "cfg.json", "r") as f:
        args_dict = json.load(f)
    try:
        update_step = args_dict["update_step"]
    except KeyError:
        update_step = 1
    args = farconf.from_dict(args_dict["cfg"], Args)

    _, _, params = args.net.init_params(args.train_env.make(), jax.random.PRNGKey(1234))

    local_batch_size = int(args.local_num_envs * args.num_steps * args.num_actor_threads * len(args.actor_device_ids))

    target_state = TrainState.create(
        apply_fn=None,
        params=params,
        tx=make_optimizer(args, params, total_updates=args.total_timesteps // local_batch_size),
    )

    with open(dir / "model", "rb") as f:
        train_state = flax.serialization.from_bytes(target_state, f.read())
    assert isinstance(train_state, TrainState)
    train_state = unreplicate(train_state)
    if isinstance(args.net, ConvLSTMConfig):
        for i in range(args.net.n_recurrent):
            train_state.params["params"]["network_params"][f"cell_list_{i}"]["fence"]["kernel"] = np.sum(
                train_state.params["params"]["network_params"][f"cell_list_{i}"]["fence"]["kernel"],
                axis=2,
                keepdims=True,
            )
    return args, train_state, update_step


if __name__ == "__main__":
    args = farconf.parse_cli(sys.argv[1:], Args)
    pprint(args)

    train(args)
