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
from typing import Any, ContextManager, Iterable, Iterator, List, NamedTuple, Sequence

import farconf
import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax
import rlax
import wandb
from flax.linen.initializers import constant, orthogonal
from flax.training.train_state import TrainState
from jaxlib import xla_client
from names_generator import generate_name
from rich.pretty import pprint
from typing_extensions import Self

from cleanba.config import random_seed
from cleanba.environments import EnvConfig, EnvpoolBoxobanConfig, SokobanConfig
from cleanba.evaluate import EvalConfig
from cleanba.optimizer import rmsprop_pytorch_style

# Fix weird OOM https://github.com/google/jax/discussions/6332#discussioncomment-1279991
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.6"
os.environ["XLA_FLAGS"] = "--xla_cpu_multi_thread_eigen=false " "intra_op_parallelism_threads=1"
# Fix CUDNN non-determinisim; https://github.com/google/jax/issues/4823#issuecomment-952835771
os.environ["TF_XLA_FLAGS"] = "--xla_gpu_autotune_level=2 --xla_gpu_deterministic_reductions"
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
            max_episode_steps=40, num_envs=64, tinyworld_obs=True, dim_room=(5, 5), num_boxes=1
        )
    )
    eval_envs: dict[str, EvalConfig] = dataclasses.field(  # How to evaluate the algorithm? Including envs and seeds
        default_factory=lambda: dict(
            eval=EvalConfig(
                SokobanConfig(
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

    base_run_dir: Path = Path("/tmp/cleanba")

    # Loss arguments
    gamma: float = 0.99  # the discount factor gamma
    ent_coef: float = 0.01  # coefficient of the entropy
    vf_coef: float = 0.5  # coefficient of the value function

    # Interpolate between VTrace (1.0) and monte-carlo function (0.0) estimates, for the estimate of targets, used in
    # both the value and policy losses. It's the parameter in Remark 2 of Espeholt et al.
    # (https://arxiv.org/pdf/1802.01561.pdf)
    vtrace_lambda: float = 0.95

    # Maximum importance ratio for the VTrace value estimates. This is \overline{rho} in eq. 1 of Espeholt et al.
    # (https://arxiv.org/pdf/1802.01561.pdf). \overline{c} is hardcoded to 1 in rlax.
    clip_rho_threshold: float = 1.0

    # Maximum importance ratio for policy gradient outputs. Clips the importanc ratio in eq. 4 of Espeholt et al.
    # (https://arxiv.org/pdf/1802.01561.pdf)
    clip_pg_rho_threshold: float = 1.0

    # Algorithm specific arguments
    total_timesteps: int = 50000000  # total timesteps of the experiments
    learning_rate: float = 0.0006  # the learning rate of the optimizer
    local_num_envs: int = 1  # the number of parallel game environments for every actor device
    num_actor_threads: int = 1  # the number of actor threads to use
    num_steps: int = 20  # the number of steps to run in each environment per policy rollout
    anneal_lr: bool = True  # Toggle learning rate annealing for policy and value networks
    num_minibatches: int = 1  # the number of mini-batches
    gradient_accumulation_steps: int = 1  # the number of gradient accumulation steps before performing an optimization step
    max_grad_norm: float = 40.0  # the maximum norm for the gradient clipping
    channels: tuple[int, ...] = (32, 32, 32)  # the channels of the CNN
    hiddens: tuple[int, ...] = (256,)  # the hiddens size of the MLP

    actor_device_ids: List[int] = field(default_factory=lambda: [0])  # the device ids that actor workers will use
    learner_device_ids: List[int] = field(default_factory=lambda: [0])  # the device ids that learner workers will use
    distributed: bool = False  # whether to use `jax.distributed`
    concurrency: bool = True  # whether to run the actor and learner concurrently


@dataclasses.dataclass(frozen=True)
class RuntimeInformation:
    local_batch_size: int
    local_devices: list[xla_client.Device]
    local_minibatch_size: int
    world_size: int
    local_rank: int
    num_envs: int
    batch_size: int
    minibatch_size: int
    num_updates: int
    global_learner_devices: Any
    learner_devices: Any


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
    actor_devices = [local_devices[d_id] for d_id in args.actor_device_ids]
    global_learner_devices = [
        global_devices[d_id + process_index * len(local_devices)]
        for process_index in range(world_size)
        for d_id in args.learner_device_ids
    ]
    global_learner_devices = global_learner_devices
    actor_devices = actor_devices
    learner_devices = learner_devices

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


class Transition(NamedTuple):
    obs_t: jax.Array
    done_tm1: jax.Array
    a_t: jax.Array
    logits_t: jax.Array
    r_tm1: jax.Array
    trunc_tm1: jax.Array
    term_tm1: jax.Array


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
    return next_obs, action, logits, key


MUST_STOP_PROGRAM: bool = False


def rollout(
    key: jax.random.PRNGKey,
    args: Args,
    runtime_info: RuntimeInformation,
    rollout_queue,
    params_queue: queue.Queue,
    writer,
    learner_devices,
    device_thread_id,
    actor_device,
    *,
    get_action,
):
    envs = dataclasses.replace(
        args.train_env,
        seed=args.train_env.seed + jax.process_index() + device_thread_id,
        num_envs=args.local_num_envs,
    ).make()

    len_actor_device_ids = len(args.actor_device_ids)
    global_step = 0
    start_time = time.time()

    # put data in the last index
    episode_returns = np.zeros((args.local_num_envs,), dtype=np.float32)
    returned_episode_returns = []
    episode_lengths = np.zeros((args.local_num_envs,), dtype=np.float32)
    returned_episode_lengths = []
    envs.reset()
    # Take a random action only once
    envs.step_async(envs.action_space.sample())

    params_queue_get_time = deque(maxlen=10)
    rollout_time = deque(maxlen=10)
    rollout_queue_put_time = deque(maxlen=10)
    actor_policy_version = 0
    storage = []

    @jax.jit
    def prepare_data(storage: List[Transition]) -> Transition:
        return jax.tree.map(lambda *xs: jnp.split(jnp.stack(xs), len(learner_devices), axis=1), *storage)

    global MUST_STOP_PROGRAM
    for update in range(1, runtime_info.num_updates + 2):
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
            obs_t, r_tm1, term_tm1, trunc_tm1, info = envs.step_wait()
            done_tm1 = term_tm1 | trunc_tm1
            env_recv_time += time.time() - env_recv_time_start
            global_step += len(done_tm1) * args.num_actor_threads * len_actor_device_ids * runtime_info.world_size

            inference_time_start = time.time()
            obs_t, a_t, logits_t, key = get_action(params, obs_t, key)
            inference_time += time.time() - inference_time_start

            d2h_time_start = time.time()
            cpu_action = np.array(a_t)
            d2h_time += time.time() - d2h_time_start
            env_send_time_start = time.time()
            envs.step_async(cpu_action)
            env_send_time += time.time() - env_send_time_start
            storage_time_start = time.time()

            storage.append(
                Transition(
                    obs_t=obs_t,
                    done_tm1=done_tm1,
                    a_t=a_t,
                    logits_t=logits_t,
                    r_tm1=r_tm1,
                    trunc_tm1=trunc_tm1,
                    term_tm1=term_tm1,
                )
            )
            returned_episode_returns.extend(episode_returns[done_tm1])
            episode_returns[:] += r_tm1
            episode_returns[:] *= ~done_tm1
            returned_episode_lengths.extend(episode_lengths[done_tm1])
            episode_lengths[:] += 1
            episode_lengths[:] *= ~done_tm1
            storage_time += time.time() - storage_time_start
        rollout_time.append(time.time() - rollout_time_start)

        partitioned_storage = prepare_data(storage)
        sharded_storage = Transition(
            *list(
                map(
                    lambda x: jax.device_put_sharded(x, devices=learner_devices),
                    partitioned_storage,
                )
            )
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

        if device_thread_id == 0 and update % args.log_frequency == 0:
            avg_episodic_return = np.mean(returned_episode_returns)
            print(
                f"global_step={global_step}, avg_episodic_return={avg_episodic_return}, rollout_time={np.mean(rollout_time)}"
            )
            print("SPS:", int(global_step / (time.time() - start_time)))
            writer.add_scalar("charts/avg_episodic_return", avg_episodic_return, global_step)
            writer.add_scalar("stats/rollout_time", np.mean(rollout_time), global_step)
            writer.add_scalar("charts/avg_episodic_return", avg_episodic_return, global_step)
            writer.add_scalar(
                "charts/avg_episodic_length",
                np.mean(returned_episode_lengths),
                global_step,
            )
            writer.add_scalar(
                "stats/params_queue_get_time",
                np.mean(params_queue_get_time),
                global_step,
            )
            writer.add_scalar("stats/env_recv_time", env_recv_time, global_step)
            writer.add_scalar("stats/inference_time", inference_time, global_step)
            writer.add_scalar("stats/storage_time", storage_time, global_step)
            writer.add_scalar("stats/d2h_time", d2h_time, global_step)
            writer.add_scalar("stats/env_send_time", env_send_time, global_step)
            writer.add_scalar(
                "stats/rollout_queue_put_time",
                np.mean(rollout_queue_put_time),
                global_step,
            )
            writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)
            writer.add_scalar(
                "charts/SPS_update",
                int(
                    args.local_num_envs
                    * args.num_steps
                    * len_actor_device_ids
                    * args.num_actor_threads
                    * runtime_info.world_size
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

        @jax.jit
        def get_logits_and_value(
            params: flax.core.FrozenDict,
            x: np.ndarray,
        ):
            hidden = Network(args.channels, args.hiddens).apply(params.network_params, x)
            raw_logits = Actor(envs.single_action_space.n).apply(params.actor_params, hidden)
            value = Critic().apply(params.critic_params, hidden).squeeze(-1)
            return raw_logits, value

        def policy_gradient_loss(logits, *args):
            """rlax.policy_gradient_loss, but with sum(loss) and [T, B, ...] inputs."""
            mean_per_batch = jax.vmap(rlax.policy_gradient_loss, in_axes=1)(logits, *args)
            total_loss_per_batch = mean_per_batch * logits.shape[0]
            return jnp.sum(total_loss_per_batch)

        def entropy_loss_fn(logits, *args):
            """rlax.entropy_loss, but with sum(loss) and [T, B, ...] inputs."""
            mean_per_batch = jax.vmap(rlax.entropy_loss, in_axes=1)(logits, *args)
            total_loss_per_batch = mean_per_batch * logits.shape[0]
            return jnp.sum(total_loss_per_batch)

        def impala_loss(params, obs_t, a_t, logits_t, r_tm1, done_tm1):
            discount_tm1 = (~done_tm1) * args.gamma
            firststeps_t = done_tm1
            mask_t = ~firststeps_t

            logits_to_update, value_to_update = jax.vmap(get_logits_and_value, in_axes=(None, 0))(params, obs_t)

            v_t = value_to_update[1:]
            # Remove bootstrap timestep from non-timesteps.
            v_tm1 = value_to_update[:-1]

            logits_to_update = logits_to_update[:-1]
            logits_t = logits_t[:-1]
            a_t = a_t[:-1]
            rhos_tm1 = rlax.categorical_importance_sampling_ratios(logits_to_update, logits_t, a_t)

            mask_t = mask_t[:-1]
            r_tm1 = r_tm1[:-1]
            discount_tm1 = discount_tm1[:-1]
            vtrace_td_error_and_advantage = jax.vmap(
                partial(
                    rlax.vtrace_td_error_and_advantage,
                    lambda_=args.vtrace_lambda,
                    clip_rho_threshold=args.clip_rho_threshold,
                    clip_pg_rho_threshold=args.clip_pg_rho_threshold,
                    stop_target_gradients=True,
                ),
                in_axes=1,
                out_axes=1,
            )

            """
            Some of these arguments are misnamed in `vtrace_td_error_and_advantage`:

            The argument `r_t` is paired with `v_t` and `v_tm1` to compute the TD error. But that's not the equation of
            the TD error, it is:

                 td(t) = r_t + gamma_t*V(x_{t+1}) - V(x_{t})

            So arguably instead of r_t and discount_t, they should be r_tm1 and discount_tm1. And that's what we name
            them here.
            """
            vtrace_returns = vtrace_td_error_and_advantage(v_tm1, v_t, r_tm1, discount_tm1, rhos_tm1)
            pg_advs = vtrace_returns.pg_advantage
            pg_loss = policy_gradient_loss(logits_to_update, a_t, pg_advs, mask_t)

            baseline_loss = 0.5 * jnp.sum(jnp.square(vtrace_returns.errors) * mask_t)
            ent_loss = entropy_loss_fn(logits_to_update, mask_t)

            total_loss = pg_loss
            total_loss += args.vf_coef * baseline_loss
            total_loss += args.ent_coef * ent_loss
            return total_loss, (pg_loss, baseline_loss, ent_loss)

        @jax.jit
        def single_device_update(
            agent_state: TrainState,
            sharded_storages: List[Transition],
            key: jax.Array,
        ):
            storage = jax.tree.map(lambda *x: jnp.hstack(x), *sharded_storages)
            impala_loss_grad_fn = jax.value_and_grad(impala_loss, has_aux=True)

            def update_minibatch(agent_state, minibatch):
                (
                    mb_obs,
                    mb_actions,
                    mb_logitss,
                    mb_rewards,
                    mb_dones,
                ) = minibatch
                (loss, (pg_loss, v_loss, entropy_loss)), grads = impala_loss_grad_fn(
                    agent_state.params,
                    mb_obs,
                    mb_actions,
                    mb_logitss,
                    mb_rewards,
                    mb_dones,
                )
                grads = jax.lax.pmean(grads, axis_name="local_devices")
                agent_state = agent_state.apply_gradients(grads=grads)
                return agent_state, (loss, pg_loss, v_loss, entropy_loss)

            agent_state, (loss, pg_loss, v_loss, entropy_loss) = jax.lax.scan(
                update_minibatch,
                agent_state,
                (
                    jnp.array(
                        jnp.split(
                            storage.obs_t,
                            args.num_minibatches * args.gradient_accumulation_steps,
                            axis=1,
                        )
                    ),
                    jnp.array(
                        jnp.split(
                            storage.a_t,
                            args.num_minibatches * args.gradient_accumulation_steps,
                            axis=1,
                        )
                    ),
                    jnp.array(
                        jnp.split(
                            storage.logits_t,
                            args.num_minibatches * args.gradient_accumulation_steps,
                            axis=1,
                        )
                    ),
                    jnp.array(
                        jnp.split(
                            storage.r_tm1,
                            args.num_minibatches * args.gradient_accumulation_steps,
                            axis=1,
                        )
                    ),
                    jnp.array(
                        jnp.split(
                            storage.done_tm1,
                            args.num_minibatches * args.gradient_accumulation_steps,
                            axis=1,
                        )
                    ),
                ),
            )
            loss = jax.lax.pmean(loss, axis_name="local_devices").mean()
            pg_loss = jax.lax.pmean(pg_loss, axis_name="local_devices").mean()
            v_loss = jax.lax.pmean(v_loss, axis_name="local_devices").mean()
            entropy_loss = jax.lax.pmean(entropy_loss, axis_name="local_devices").mean()
            return agent_state, loss, pg_loss, v_loss, entropy_loss, key

        multi_device_update = jax.pmap(
            single_device_update,
            axis_name="local_devices",
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
        agent_state = jax.device_put_replicated(agent_state, devices=runtime_info.learner_devices)

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
                    ) = rollout_queues[d_idx * args.num_actor_threads + thread_id].get()
                    sharded_storages.append(sharded_storage)
            rollout_queue_get_time.append(time.time() - rollout_queue_get_time_start)
            training_time_start = time.time()
            (
                agent_state,
                loss,
                pg_loss,
                v_loss,
                entropy_loss,
                learner_keys,
            ) = multi_device_update(
                agent_state,
                sharded_storages,
                learner_keys,
            )
            unreplicated_params = unreplicate(agent_state.params)
            for d_idx, d_id in enumerate(args.actor_device_ids):
                device_params = jax.device_put(unreplicated_params, runtime_info.local_devices[d_id])
                for thread_id in range(args.num_actor_threads):
                    params_queues[d_idx * args.num_actor_threads + thread_id].put(device_params)

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
                writer.add_scalar("losses/value_loss", v_loss[-1].item(), global_step)
                writer.add_scalar("losses/policy_loss", pg_loss[-1].item(), global_step)
                writer.add_scalar("losses/entropy", entropy_loss[-1].item(), global_step)
                writer.add_scalar("losses/loss", loss[-1].item(), global_step)

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
