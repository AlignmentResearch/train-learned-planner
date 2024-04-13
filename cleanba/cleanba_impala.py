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
from cleanba.environments import AtariEnv, EnvConfig
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
        # default_factory=lambda: SokobanConfig(
        #     asynchronous=False, max_episode_steps=40, num_envs=64, tinyworld_obs=True, dim_room=(5, 5), num_boxes=1
        # )
        default_factory=lambda: AtariEnv(env_id="Breakout-v5"),
    )
    eval_envs: dict[str, EvalConfig] = dataclasses.field(  # How to evaluate the algorithm? Including envs and seeds
        default_factory=lambda: dict(eval=EvalConfig(AtariEnv(env_id="Breakout-v5", num_envs=128)))
    )
    eval_frequency: int = 1000  # How often to evaluate and maybe save the model

    seed: int = dataclasses.field(default_factory=random_seed)  # A seed to make the experiment deterministic

    save_model: bool = True  # whether to save model into the wandb run folder
    log_frequency: int = 10  # the logging frequency of the model performance (in terms of `updates`)
    sync_frequency: int = 400

    base_run_dir: Path = Path("/tmp/cleanba")

    loss: ImpalaLossConfig = ImpalaLossConfig(vf_coef=0.5)

    # Algorithm specific arguments
    total_timesteps: int = 100_000_000  # total timesteps of the experiments
    learning_rate: float = 0.0006  # the learning rate of the optimizer
    local_num_envs: int = 64  # the number of parallel game environments for every actor device
    num_steps: int = 20  # the number of steps to run in each environment per policy rollout
    anneal_lr: bool = True  # Toggle learning rate annealing for policy and value networks
    num_minibatches: int = 4  # the number of mini-batches
    gradient_accumulation_steps: int = 1  # the number of gradient accumulation steps before performing an optimization step
    max_grad_norm: float = 40.0  # the maximum norm for the gradient clipping
    channels: tuple[int, ...] = (16, 32, 32)  # the channels of the CNN
    hiddens: tuple[int, ...] = (256,)  # the hiddens size of the MLP

    queue_timeout: float = 300.0  # If any of the actor/learner queues takes at least this many seconds, crash training.

    num_actor_threads: int = 4  # The number of environment threads per actor device
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
        x = nn.max_pool(x, window_shape=(3, 3), strides=(2, 2), padding="SAME")
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


def make_env(env_id, seed, num_envs):
    return AtariEnv(env_id=env_id, seed=seed, num_envs=num_envs).make


def rollout(
    key: jax.random.PRNGKey,
    args: Args,
    rollout_queue,
    params_queue: queue.Queue,
    writer,
    learner_devices,
    device_thread_id,
    actor_device,
):
    envs = make_env(
        args.train_env.env_id,
        args.seed + jax.process_index() + device_thread_id,
        args.local_num_envs,
    )()
    len_actor_device_ids = len(args.actor_device_ids)
    global_step = 0
    start_time = time.time()

    @jax.jit
    def get_action(
        params: flax.core.FrozenDict,
        next_obs: np.ndarray,
        key: jax.random.PRNGKey,
    ):
        next_obs = jnp.array(next_obs)
        hidden = Network(args.channels, args.hiddens).apply(params.network_params, next_obs)
        logits = Actor(envs.single_action_space.n).apply(params.actor_params, hidden)
        # sample action: Gumbel-softmax trick
        # see https://stats.stackexchange.com/questions/359442/sampling-from-a-categorical-distribution
        key, subkey = jax.random.split(key)
        u = jax.random.uniform(subkey, shape=logits.shape)
        action = jnp.argmax(logits - jnp.log(-jnp.log(u)), axis=1)
        return next_obs, action, logits, key

    # put data in the last index
    episode_returns = np.zeros((args.local_num_envs,), dtype=np.float32)
    returned_episode_returns = np.zeros((args.local_num_envs,), dtype=np.float32)
    episode_lengths = np.zeros((args.local_num_envs,), dtype=np.float32)
    returned_episode_lengths = np.zeros((args.local_num_envs,), dtype=np.float32)
    envs.reset()
    envs.step_async(envs.action_space.sample())

    params_queue_get_time = deque(maxlen=10)
    rollout_time = deque(maxlen=10)
    rollout_queue_put_time = deque(maxlen=10)
    actor_policy_version = 0
    storage = []

    @jax.jit
    def prepare_data(storage: List[Rollout]) -> Rollout:
        return jax.tree_map(lambda *xs: jnp.split(jnp.stack(xs), len(learner_devices), axis=1), *storage)

    global MUST_STOP_PROGRAM
    for update in range(1, args.num_updates + 2):
        if MUST_STOP_PROGRAM:
            break
        update_time_start = time.time()
        env_recv_time = 0
        inference_time = 0
        storage_time = 0
        d2h_time = 0
        env_send_time = 0
        num_steps_with_bootstrap = (
            args.num_steps + 1 + int(len(storage) == 0)
        )  # num_steps + 1 to get the states for value bootstrapping.
        # NOTE: `update != 2` is actually IMPORTANT — it allows us to start running policy collection
        # concurrently with the learning process. It also ensures the actor's policy version is only 1 step
        # behind the learner's policy version
        params_queue_get_time_start = time.time()
        if args.concurrency:
            if update != 2:
                params = params_queue.get()
                # NOTE: block here is important because otherwise this thread will call
                # the jitted `get_action` function that hangs until the params are ready.
                # This blocks the `get_action` function in other actor threads.
                # See https://excalidraw.com/#json=hSooeQL707gE5SWY8wOSS,GeaN1eb2r24PPi75a3n14Q for a visual explanation.
                params.network_params["params"]["Dense_0"][
                    "kernel"
                ].block_until_ready()  # TODO: check if params.block_until_ready() is enough
                actor_policy_version += 1
        else:
            params = params_queue.get()
            actor_policy_version += 1
        params_queue_get_time.append(time.time() - params_queue_get_time_start)
        rollout_time_start = time.time()
        for _ in range(1, num_steps_with_bootstrap):
            env_recv_time_start = time.time()
            next_obs, next_reward, next_trunc, next_term, info = envs.step_wait()
            next_done = next_trunc | next_term
            env_recv_time += time.time() - env_recv_time_start
            global_step += len(next_done) * args.num_actor_threads * len_actor_device_ids * args.world_size
            env_id = info["env_id"]

            inference_time_start = time.time()
            next_obs, action, logits, key = get_action(params, next_obs, key)
            inference_time += time.time() - inference_time_start

            d2h_time_start = time.time()
            cpu_action = np.array(action)
            d2h_time += time.time() - d2h_time_start
            env_send_time_start = time.time()
            envs.step_async(cpu_action)
            env_send_time += time.time() - env_send_time_start
            storage_time_start = time.time()

            # info["TimeLimit.truncated"] has a bug https://github.com/sail-sg/envpool/issues/239
            # so we use our own truncated flag
            truncated = info["elapsed_step"] >= envs.envs.spec.config.max_episode_steps
            storage.append(
                Rollout(
                    obs=next_obs,
                    dones=next_done,
                    actions=action,
                    logitss=logits,
                    env_ids=env_id,
                    rewards=next_reward,
                    truncations=truncated,
                    terminations=info["terminated"],
                    firststeps=info["elapsed_step"] == 0,
                )
            )
            episode_returns[env_id] += info["reward"]
            returned_episode_returns[env_id] = np.where(
                info["terminated"] + truncated, episode_returns[env_id], returned_episode_returns[env_id]
            )
            episode_returns[env_id] *= (1 - info["terminated"]) * (1 - truncated)
            episode_lengths[env_id] += 1
            returned_episode_lengths[env_id] = np.where(
                info["terminated"] + truncated, episode_lengths[env_id], returned_episode_lengths[env_id]
            )
            episode_lengths[env_id] *= (1 - info["terminated"]) * (1 - truncated)
            storage_time += time.time() - storage_time_start
        rollout_time.append(time.time() - rollout_time_start)

        avg_episodic_return = np.mean(returned_episode_returns)
        partitioned_storage = prepare_data(storage)
        sharded_storage = Rollout(
            *list(map(lambda x: jax.device_put_sharded(x, devices=learner_devices), partitioned_storage))
        )
        payload = (
            global_step,
            actor_policy_version,
            update,
            sharded_storage,
            np.mean(params_queue_get_time),
            device_thread_id,
        )
        rollout_queue_put_time_start = time.time()
        rollout_queue.put(payload)
        rollout_queue_put_time.append(time.time() - rollout_queue_put_time_start)

        # move bootstrapping step to the beginning of the next update
        storage = storage[-1:]

        if update % args.log_frequency == 0:
            if device_thread_id == 0:
                print(
                    f"global_step={global_step}, avg_episodic_return={avg_episodic_return}, rollout_time={np.mean(rollout_time)}"
                )
                print("SPS:", int(global_step / (time.time() - start_time)))
            writer.add_scalar(f"stats/{device_thread_id}/rollout_time", np.mean(rollout_time), global_step)
            writer.add_scalar(f"charts/{device_thread_id}/avg_episodic_return", avg_episodic_return, global_step)
            writer.add_scalar(f"charts/{device_thread_id}/avg_episodic_length", np.mean(returned_episode_lengths), global_step)
            writer.add_scalar(f"stats/{device_thread_id}/params_queue_get_time", np.mean(params_queue_get_time), global_step)
            writer.add_scalar(f"stats/{device_thread_id}/env_recv_time", env_recv_time, global_step)
            writer.add_scalar(f"stats/{device_thread_id}/inference_time", inference_time, global_step)
            writer.add_scalar(f"stats/{device_thread_id}/storage_time", storage_time, global_step)
            writer.add_scalar(f"stats/{device_thread_id}/d2h_time", d2h_time, global_step)
            writer.add_scalar(f"stats/{device_thread_id}/env_send_time", env_send_time, global_step)
            writer.add_scalar(f"stats/{device_thread_id}/rollout_queue_put_time", np.mean(rollout_queue_put_time), global_step)
            writer.add_scalar(f"charts/{device_thread_id}/SPS", int(global_step / (time.time() - start_time)), global_step)
            writer.add_scalar(
                f"charts/{device_thread_id}/SPS_update",
                int(
                    args.local_num_envs
                    * args.num_steps
                    * len_actor_device_ids
                    * args.num_actor_threads
                    * args.world_size
                    / (time.time() - update_time_start)
                ),
                global_step,
            )


if __name__ == "__main__":
    args = farconf.parse_cli(["--from-py-fn=cleanba.cleanba_impala:Args"] + sys.argv[1:], Args)
    pprint(args)

    warnings.filterwarnings("ignore", "", UserWarning, module="gymnasium.vector")

    train_env_cfg = dataclasses.replace(args.train_env, num_envs=args.local_num_envs)
    with initialize_multi_device(args) as runtime_info, contextlib.closing(train_env_cfg.make()) as envs:
        pprint(runtime_info)
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
                        writer,
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
