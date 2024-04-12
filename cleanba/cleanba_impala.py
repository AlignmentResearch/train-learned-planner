import contextlib
import dataclasses
import os
import queue
import random
import sys
import threading
import time
import warnings
from collections import deque
from dataclasses import field
from functools import partial
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterator, List, Sequence

import farconf
import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax
import wandb
from flax.linen.initializers import constant, orthogonal
from flax.training.train_state import TrainState
from names_generator import generate_name
from rich.pretty import pprint
from typing_extensions import Self

from cleanba.config import random_seed
from cleanba.environments import EnvConfig, SokobanConfig
from cleanba.evaluate import EvalConfig
from cleanba.impala_loss import (
    SINGLE_DEVICE_UPDATE_DEVICES_AXIS,
    ImpalaLossConfig,
    Rollout,
    single_device_update,
)
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
    save_dir: Path

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
        except KeyError:
            # If any of the essential WANDB environment variables are missing,
            # simply don't upload this run.
            # It's fine to do this without giving any indication because Wandb already prints that the run is offline.

            wandb_kwargs = dict(mode=os.environ.get("WANDB_MODE", "offline"), group="default")

        run_dir = args.base_run_dir / wandb_kwargs["group"]
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
        self.save_dir = Path(wandb.run.dir).parent / "local-files"
        self.save_dir.mkdir()

    def add_scalar(self, name: str, value: int | float, global_step: int):
        wandb.log({name: value}, step=global_step)


@dataclasses.dataclass
class Args:
    train_env: EnvConfig = dataclasses.field(  # Environment to do training, including seed
        default_factory=lambda: SokobanConfig(
            asynchronous=False, max_episode_steps=40, num_envs=64, tinyworld_obs=True, dim_room=(5, 5), num_boxes=1
        )
    )
    eval_envs: dict[str, EvalConfig] = dataclasses.field(  # How to evaluate the algorithm? Including envs and seeds
        default_factory=lambda: dict(
            eval=EvalConfig(
                SokobanConfig(
                    asynchronous=False,
                    max_episode_steps=40,
                    num_envs=64,
                    tinyworld_obs=True,
                    dim_room=(5, 5),
                    num_boxes=1,
                )
            )
        )
    )
    eval_frequency: int = 10  # How often to evaluate and maybe save the model

    seed: int = dataclasses.field(default_factory=random_seed)  # A seed to make the experiment deterministic

    save_model: bool = True  # whether to save model into the wandb run folder
    log_frequency: int = 10  # the logging frequency of the model performance (in terms of `updates`)
    sync_frequency: int = 100

    base_run_dir: Path = Path("/tmp/cleanba")

    loss: ImpalaLossConfig = ImpalaLossConfig()

    # Algorithm specific arguments
    total_timesteps: int = 50000000  # total timesteps of the experiments
    learning_rate: float = 0.0006  # the learning rate of the optimizer
    local_num_envs: int = 4  # the number of parallel game environments for every actor device
    num_steps: int = 3  # the number of steps to run in each environment per policy rollout
    anneal_lr: bool = True  # Toggle learning rate annealing for policy and value networks
    num_minibatches: int = 1  # the number of mini-batches
    gradient_accumulation_steps: int = 1  # the number of gradient accumulation steps before performing an optimization step
    max_grad_norm: float = 40.0  # the maximum norm for the gradient clipping
    channels: tuple[int, ...] = (32, 32, 32)  # the channels of the CNN
    hiddens: tuple[int, ...] = (256,)  # the hiddens size of the MLP

    queue_timeout: float = 300.0  # If any of the actor/learner queues takes at least this many seconds, crash training.

    num_actor_threads: int = 1  # The number of environment threads per actor device
    actor_device_ids: List[int] = field(default_factory=lambda: [0])  # the device ids that actor workers will use
    learner_device_ids: List[int] = field(default_factory=lambda: [0])  # the device ids that learner workers will use
    distributed: bool = False  # whether to use `jax.distributed`
    concurrency: bool = True  # whether to run the actor and learner concurrently


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


class ResidualBlock(nn.Module):
    channels: int

    @nn.compact
    def __call__(self, x):
        inputs = x
        x = nn.relu(x)
        x = nn.Conv(self.channels, kernel_size=(3, 3))(x)
        x = nn.relu(x)
        x = nn.Conv(self.channels, kernel_size=(3, 3))(x)
        return x + inputs


class ConvSequence(nn.Module):
    channels: int

    @nn.compact
    def __call__(self, x):
        x = nn.Conv(self.channels, kernel_size=(3, 3))(x)
        x = ResidualBlock(self.channels)(x)
        x = ResidualBlock(self.channels)(x)
        return x


class Network(nn.Module):
    channels: Sequence[int]
    hiddens: Sequence[int]

    @nn.compact
    def __call__(self, x):
        x = jnp.transpose(x, (0, 2, 3, 1))
        x = x / (255.0)
        for channels in self.channels:
            x = ConvSequence(channels)(x)
        x = nn.relu(x)
        x = x.reshape((x.shape[0], -1))
        for hidden in self.hiddens:
            x = nn.Dense(hidden, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x)
            x = nn.relu(x)
        return x


class Critic(nn.Module):
    @nn.compact
    def __call__(self, x):
        return nn.Dense(1, kernel_init=orthogonal(1), bias_init=constant(0.0))(x)


class Actor(nn.Module):
    action_dim: int

    @nn.compact
    def __call__(self, x):
        return nn.Dense(self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0))(x)


@flax.struct.dataclass
class AgentParams:
    network_params: flax.core.FrozenDict
    actor_params: flax.core.FrozenDict
    critic_params: flax.core.FrozenDict

    def __contains__(self, item):
        return item in (f.name for f in dataclasses.fields(self))


@jax.jit
def get_action(
    params: flax.core.FrozenDict,
    next_obs: jax.Array,
    key: jax.Array,
):
    hidden: jax.Array = Network(args.channels, args.hiddens).apply(params.network_params, next_obs)
    logits: jax.Array = Actor(envs.single_action_space.n).apply(params.actor_params, hidden)
    # sample action: Gumbel-softmax trick
    # see https://stats.stackexchange.com/questions/359442/sampling-from-a-categorical-distribution
    key, subkey = jax.random.split(key)
    u = jax.random.uniform(subkey, shape=logits.shape)
    action = jnp.argmax(logits - jnp.log(-jnp.log(u)), axis=1)
    return action, logits, key


def flatten_tree(x) -> jax.Array:
    leaves, _ = jax.tree.flatten(x)
    return jnp.concat(list(map(jnp.ravel, leaves)))


@jax.jit
def log_parameter_differences(params) -> dict[str, jax.Array]:
    max_params = jax.tree.map(lambda x: np.max(x, axis=0), params)
    min_params = jax.tree.map(lambda x: np.min(x, axis=0), params)

    flat_max_params = flatten_tree(max_params)
    flat_min_params = flatten_tree(min_params)
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
def _concat_and_shard_rollout_internal(storage: List[Rollout], last_obs: jax.Array, len_learner_devices: int) -> Rollout:
    partitioned_storage = jax.tree.map(lambda *xs: jnp.split(jnp.stack(xs), len_learner_devices, axis=1), *storage)
    return partitioned_storage


def concat_and_shard_rollout(storage: list[Rollout], last_obs: jax.Array, learner_devices: tuple[jax.Device, ...]) -> Rollout:
    partitioned_storage = _concat_and_shard_rollout_internal(storage, last_obs, len(learner_devices))
    sharded_storage = Rollout(
        *list(
            map(
                lambda x: jax.device_put_sharded(x, devices=learner_devices),
                partitioned_storage,
            )
        )
    )
    return sharded_storage


def rollout(
    key: jax.random.PRNGKey,
    args: Args,
    runtime_info: RuntimeInformation,
    rollout_queue: queue.Queue,
    params_queue: queue.Queue,
    writer,
    learner_devices: list[jax.Device],
    device_thread_id,
    actor_device,
    *,
    get_action,
):
    actor_id: int = device_thread_id + args.num_actor_threads * jax.process_index()

    envs = dataclasses.replace(
        args.train_env,
        seed=args.train_env.seed + actor_id,
        num_envs=args.local_num_envs,
    ).make()

    len_actor_device_ids = len(args.actor_device_ids)
    global_step = 0
    start_time = time.time()

    log_stats = LoggingStats.new_empty()
    # Counters for episode length and episode return
    episode_returns = np.zeros((args.local_num_envs,), dtype=np.float32)
    episode_lengths = np.zeros((args.local_num_envs,), dtype=np.float32)

    actor_policy_version = 0
    storage = []

    # Store the first observation
    obs_t, _ = envs.reset()

    global MUST_STOP_PROGRAM
    for update in range(1, runtime_info.num_updates + 2):
        if MUST_STOP_PROGRAM:
            break

        with time_and_append(log_stats.update_time):
            with time_and_append(log_stats.params_queue_get_time):
                num_steps_with_bootstrap = args.num_steps

                # NOTE: `update != 2` is actually IMPORTANT — it allows us to start running policy collection
                # concurrently with the learning process. It also ensures the actor's policy version is only 1 step
                # behind the learner's policy version
                if args.concurrency:
                    if update != 2:
                        params = params_queue.get(timeout=args.queue_timeout)
                        # NOTE: block here is important because otherwise this thread will call
                        # the jitted `get_action` function that hangs until the params are ready.
                        # This blocks the `get_action` function in other actor threads.
                        # See https://excalidraw.com/#json=hSooeQL707gE5SWY8wOSS,GeaN1eb2r24PPi75a3n14Q for a visual explanation.
                        params.network_params["params"]["Dense_0"][
                            "kernel"
                        ].block_until_ready()  # TODO: check if params.block_until_ready() is enough
                        actor_policy_version += 1
                else:
                    params = params_queue.get(timeout=args.queue_timeout)
                    actor_policy_version += 1

            with time_and_append(log_stats.rollout_time):
                for _ in range(1, num_steps_with_bootstrap):
                    global_step += (
                        args.local_num_envs * args.num_actor_threads * len_actor_device_ids * runtime_info.world_size
                    )

                    with time_and_append(log_stats.inference_time):
                        a_t, logits_t, key = get_action(params, obs_t, key)

                    with time_and_append(log_stats.device2host_time):
                        cpu_action = np.array(a_t)

                    with time_and_append(log_stats.env_send_time):
                        envs.step_async(cpu_action)

                    with time_and_append(log_stats.env_recv_time):
                        obs_t_plus_1, r_t, term_t, trunc_t, info_t = envs.step_wait()
                        done_t = term_t | trunc_t

                    with time_and_append(log_stats.create_rollout_time):
                        storage.append(
                            Rollout(
                                obs_t=obs_t,
                                done_t=done_t,
                                a_t=a_t,
                                logits_t=logits_t,
                                r_t=r_t,
                            )
                        )
                        obs_t = obs_t_plus_1

                        log_stats.episode_returns.extend(episode_returns[done_t])
                        episode_returns[:] += r_t
                        episode_returns[:] *= ~done_t
                        log_stats.episode_lengths.extend(episode_lengths[done_t])
                        episode_lengths[:] += 1
                        episode_lengths[:] *= ~done_t

            with time_and_append(log_stats.storage_time):
                sharded_storage = concat_and_shard_rollout(storage, obs_t, learner_devices)
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
            middle_loop_time = np.sum(log_stats.rollout_time) + np.sum(log_stats.storage_time)
            outer_loop_time = np.sum(log_stats.update_time)

            stats_dict: dict[str, float] = log_stats.avg_and_flush()
            steps_per_second = global_step / (time.time() - start_time)
            print(
                f"{device_thread_id=}, SPS={steps_per_second:.2f}, {global_step=}, avg_episode_returns={stats_dict['avg_episode_returns']:.2f}, avg_episode_length={stats_dict['avg_episode_lengths']:.2f}, avg_rollout_time={stats_dict['avg_rollout_time']:.5f}"
            )

            for k, v in stats_dict.items():
                if k.endswith("_time"):
                    writer.add_scalar(f"stats/{device_thread_id}/{k}", v, global_step)
                else:
                    writer.add_scalar(f"charts/{device_thread_id}/{k}", v, global_step)

            writer.add_scalar(
                f"charts/{device_thread_id}/inner_time_efficiency", inner_loop_time / outer_loop_time, global_step
            )
            writer.add_scalar(
                f"charts/{device_thread_id}/middle_time_efficiency", middle_loop_time / outer_loop_time, global_step
            )
            writer.add_scalar(f"charts/{device_thread_id}/SPS", steps_per_second, global_step)


if __name__ == "__main__":
    args = farconf.parse_cli(["--from-py-fn=cleanba.cleanba_impala:Args"] + sys.argv[1:], Args)
    pprint(args)

    warnings.filterwarnings("ignore", "", UserWarning, module="gymnasium.vector")

    train_env_cfg = dataclasses.replace(args.train_env, num_envs=args.local_num_envs)
    with initialize_multi_device(args) as runtime_info, contextlib.closing(train_env_cfg.make()) as envs:
        writer = WandbWriter(args)

        # seeding
        random.seed(args.seed)
        np.random.seed(args.seed)
        key = jax.random.PRNGKey(args.seed)
        key, network_key, actor_key, critic_key = jax.random.split(key, 4)
        learner_keys = jax.device_put_replicated(key, runtime_info.learner_devices)

        def linear_schedule(count):
            # anneal learning rate linearly after one training iteration which contains
            # (args.num_minibatches) gradient updates
            frac = 1.0 - (count // (args.num_minibatches)) / runtime_info.num_updates
            return args.learning_rate * frac

        network = Network(args.channels, args.hiddens)
        actor = Actor(action_dim=envs.single_action_space.n)
        critic = Critic()
        network_params = network.init(network_key, np.array([envs.single_observation_space.sample()]))
        agent_state = TrainState.create(
            apply_fn=None,
            params=AgentParams(
                network_params,
                actor.init(
                    actor_key,
                    network.apply(network_params, np.array([envs.single_observation_space.sample()])),
                ),
                critic.init(
                    critic_key,
                    network.apply(network_params, np.array([envs.single_observation_space.sample()])),
                ),
            ),
            tx=optax.MultiSteps(
                optax.chain(
                    optax.clip_by_global_norm(args.max_grad_norm),
                    optax.inject_hyperparams(rmsprop_pytorch_style)(
                        learning_rate=linear_schedule if args.anneal_lr else args.learning_rate,
                        eps=0.01,
                        decay=0.99,
                    ),
                ),
                every_k_schedule=args.gradient_accumulation_steps,
            ),
        )
        print(network.tabulate(network_key, np.array([envs.single_observation_space.sample()])))
        print(
            actor.tabulate(
                actor_key,
                network.apply(network_params, np.array([envs.single_observation_space.sample()])),
            )
        )
        print(
            critic.tabulate(
                critic_key,
                network.apply(network_params, np.array([envs.single_observation_space.sample()])),
            )
        )

        def get_logits_and_value(
            params: flax.core.FrozenDict,
            x: jax.Array,
        ):
            hidden = Network(args.channels, args.hiddens).apply(params.network_params, x)
            raw_logits = Actor(envs.single_action_space.n).apply(params.actor_params, hidden)
            value = Critic().apply(params.critic_params, hidden).squeeze(-1)
            return raw_logits, value

        multi_device_update = jax.pmap(
            partial(
                single_device_update,
                num_batches=args.num_minibatches * args.gradient_accumulation_steps,
                get_logits_and_value=get_logits_and_value,
                impala_cfg=args.loss,
            ),
            axis_name=SINGLE_DEVICE_UPDATE_DEVICES_AXIS,
            devices=runtime_info.global_learner_devices,
        )

        params_queues = []
        rollout_queues = []
        dummy_writer = SimpleNamespace()
        dummy_writer.add_scalar = lambda x, y, z: None

        unreplicated_params = agent_state.params
        key, *actor_keys = jax.random.split(key, 1 + len(args.actor_device_ids))
        for d_idx, d_id in enumerate(args.actor_device_ids):
            device_params = jax.device_put(unreplicated_params, runtime_info.local_devices[d_id])
            for thread_id in range(args.num_actor_threads):
                params_queues.append(queue.Queue(maxsize=1))
                rollout_queues.append(queue.Queue(maxsize=1))
                params_queues[-1].put(device_params)
                threading.Thread(
                    target=rollout,
                    args=(
                        jax.device_put(actor_keys[d_idx], runtime_info.local_devices[d_id]),
                        args,
                        runtime_info,
                        rollout_queues[-1],
                        params_queues[-1],
                        writer if d_idx == 0 and thread_id == 0 else dummy_writer,
                        runtime_info.learner_devices,
                        d_idx * args.num_actor_threads + thread_id,
                        runtime_info.local_devices[d_id],
                    ),
                    kwargs=dict(get_action=get_action),
                ).start()

        rollout_queue_get_time = deque(maxlen=10)
        data_transfer_time = deque(maxlen=10)
        learner_policy_version = 0
        agent_state = jax.device_put_replicated(agent_state, devices=runtime_info.global_learner_devices)

        global_step = 0
        actor_policy_version = 0

        while True:
            learner_policy_version += 1
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
            (
                agent_state,
                learner_keys,
                loss,
            ) = multi_device_update(
                agent_state,
                sharded_storages,
                learner_keys,
            )
            unreplicated_params = unreplicate(agent_state.params)
            for d_idx, d_id in enumerate(args.actor_device_ids):
                device_params = jax.device_put(unreplicated_params, runtime_info.local_devices[d_id])
                for thread_id in range(args.num_actor_threads):
                    params_queues[d_idx * args.num_actor_threads + thread_id].put(device_params, timeout=args.queue_timeout)

            # Copy the parameters from the first device to all other learner devices
            if learner_policy_version % args.sync_frequency == 0:
                # Check the maximum parameter difference
                param_diff_stats = log_parameter_differences(agent_state.params)
                for k, v in param_diff_stats.items():
                    writer.add_scalar(f"diffs/{k}", v.item(), global_step)
                    print(f"diffs/{k}", v.item(), global_step)

                unreplicated_agent_state = unreplicate(agent_state)
                agent_state = jax.device_put_replicated(unreplicated_agent_state, devices=runtime_info.global_learner_devices)

            # record rewards for plotting purposes
            if learner_policy_version % args.log_frequency == 0:
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
                    f"actor_policy_version={actor_policy_version}, actor_update={update}, learner_policy_version={learner_policy_version}, training time: {time.time() - training_time_start}s",
                )
                writer.add_scalar(
                    "charts/learning_rate",
                    agent_state.opt_state[2][1].hyperparams["learning_rate"][-1].item(),
                    global_step,
                )
                writer.add_scalar("losses/value_loss", loss.v_loss[0].item(), global_step)
                writer.add_scalar("losses/policy_loss", loss.pg_loss[0].item(), global_step)
                writer.add_scalar("losses/entropy", loss.entropy_loss[0].item(), global_step)
                writer.add_scalar("losses/loss", loss.loss[0].item(), global_step)
                print("loss difference:", np.abs(loss.loss[1] - loss.loss[0]))

            if learner_policy_version % args.eval_frequency == 0 and learner_policy_version != 0:
                if args.save_model:
                    saved_model_version: int = learner_policy_version // args.eval_frequency
                    with open(writer.save_dir / f"{saved_model_version:03d}.model", "wb") as f:
                        f.write(
                            flax.serialization.to_bytes(
                                [
                                    farconf.to_dict(args, Args),
                                    [
                                        agent_state.params.network_params,
                                        agent_state.params.actor_params,
                                        agent_state.params.critic_params,
                                    ],
                                ]
                            )
                        )
                    writer.add_scalar("eval/saved_model_idx", saved_model_version, global_step)

                for eval_name, eval_cfg in args.eval_envs.items():
                    key, eval_key = jax.random.split(key, 2)
                    log_dict: dict[str, float] = eval_cfg.run(network, actor, unreplicated_params, key=eval_key)

                    for k, v in log_dict.items():
                        writer.add_scalar(f"{eval_name}/{k}", v, global_step)

            if learner_policy_version >= runtime_info.num_updates:
                break
