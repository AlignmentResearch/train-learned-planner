import dataclasses
import os
import queue
import random
import sys
import threading
import time
from collections import deque
from dataclasses import field
from functools import partial
from pathlib import Path
from types import SimpleNamespace
from typing import Any, List, NamedTuple, Sequence

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
from names_generator import generate_name
from rich.pretty import pprint

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

    # Algorithm specific arguments
    total_timesteps: int = 50000000  # total timesteps of the experiments
    learning_rate: float = 0.0006  # the learning rate of the optimizer
    local_num_envs: int = 1  # the number of parallel game environments for every actor device
    num_actor_threads: int = 1  # the number of actor threads to use
    num_steps: int = 20  # the number of steps to run in each environment per policy rollout
    anneal_lr: bool = True  # Toggle learning rate annealing for policy and value networks
    gamma: float = 0.99  # the discount factor gamma
    num_minibatches: int = 1  # the number of mini-batches
    gradient_accumulation_steps: int = 1  # the number of gradient accumulation steps before performing an optimization step
    ent_coef: float = 0.01  # coefficient of the entropy
    vf_coef: float = 0.5  # coefficient of the value function
    max_grad_norm: float = 40.0  # the maximum norm for the gradient clipping
    channels: tuple[int, ...] = (32, 32, 32)  # the channels of the CNN
    hiddens: tuple[int, ...] = (256,)  # the hiddens size of the MLP

    actor_device_ids: List[int] = field(default_factory=lambda: [0])  # the device ids that actor workers will use
    learner_device_ids: List[int] = field(default_factory=lambda: [0])  # the device ids that learner workers will use
    distributed: bool = False  # whether to use `jax.distributed`
    concurrency: bool = True  # whether to run the actor and learner concurrently

    # runtime arguments to be filled in
    local_batch_size: int = 0
    local_minibatch_size: int = 0
    num_updates: int = 0
    world_size: int = 0
    local_rank: int = 0
    num_envs: int = 0
    batch_size: int = 0
    minibatch_size: int = 0
    num_updates: int = 0
    global_learner_devices: Any = None
    learner_devices: Any = None

    def to_dict(self) -> dict[str, Any]:
        return {f.name: getattr(self, f.name) for f in dataclasses.fields(self)}


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
    obs: list
    dones: list
    actions: list
    logitss: list
    rewards: list
    truncations: list
    terminations: list
    firststeps: list  # first step of an episode


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
    envs = dataclasses.replace(
        args.train_env,
        seed=args.train_env.seed + jax.process_index() + device_thread_id,
        num_envs=args.local_num_envs,
    ).make()

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

    for update in range(1, args.num_updates + 2):
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
            next_obs, next_reward, terminated, truncated, info = envs.step_wait()
            dones = terminated | truncated
            env_recv_time += time.time() - env_recv_time_start
            global_step += len(dones) * args.num_actor_threads * len_actor_device_ids * args.world_size

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

            storage.append(
                Transition(
                    obs=next_obs,
                    dones=dones,
                    actions=action,
                    logitss=logits,
                    rewards=next_reward,
                    truncations=truncated,
                    terminations=terminated,
                    firststeps=dones,
                )
            )
            returned_episode_returns.extend(episode_returns[dones])
            episode_returns[:] += next_reward
            episode_returns[:] *= ~dones
            returned_episode_lengths.extend(episode_lengths[dones])
            episode_lengths[:] += 1
            episode_lengths[:] *= ~dones
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
                    * args.world_size
                    / (time.time() - update_time_start)
                ),
                global_step,
            )


def parse_cli(cli: list[str]) -> tuple[Args, Any]:
    args = farconf.parse_cli(cli, Args)
    args.local_batch_size = int(args.local_num_envs * args.num_steps * args.num_actor_threads * len(args.actor_device_ids))
    args.local_minibatch_size = int(args.local_batch_size // args.num_minibatches)
    assert (
        args.local_num_envs % len(args.learner_device_ids) == 0
    ), "local_num_envs must be divisible by len(learner_device_ids)"
    assert (
        int(args.local_num_envs / len(args.learner_device_ids)) * args.num_actor_threads % args.num_minibatches == 0
    ), "int(local_num_envs / len(learner_device_ids)) must be divisible by num_minibatches"
    if args.distributed:
        jax.distributed.initialize(
            local_device_ids=range(len(args.learner_device_ids) + len(args.actor_device_ids)),
        )
        print(list(range(len(args.learner_device_ids) + len(args.actor_device_ids))))

    args.world_size = jax.process_count()
    args.local_rank = jax.process_index()
    args.num_envs = args.local_num_envs * args.world_size * args.num_actor_threads * len(args.actor_device_ids)
    args.batch_size = args.local_batch_size * args.world_size
    args.minibatch_size = args.local_minibatch_size * args.world_size
    args.num_updates = args.total_timesteps // (args.local_batch_size * args.world_size)
    local_devices = jax.local_devices()
    global_devices = jax.devices()
    learner_devices = [local_devices[d_id] for d_id in args.learner_device_ids]
    actor_devices = [local_devices[d_id] for d_id in args.actor_device_ids]
    global_learner_devices = [
        global_devices[d_id + process_index * len(local_devices)]
        for process_index in range(args.world_size)
        for d_id in args.learner_device_ids
    ]
    print("global_learner_devices", global_learner_devices)
    args.global_learner_devices = global_learner_devices
    args.actor_devices = actor_devices
    args.learner_devices = learner_devices
    pprint(args)
    return args, local_devices


if __name__ == "__main__":
    args, local_devices = parse_cli(["--from-py-fn=cleanba.cleanba_impala:Args"] + sys.argv[1:])

    writer = WandbWriter(args)

    # seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    key = jax.random.PRNGKey(args.seed)
    key, network_key, actor_key, critic_key = jax.random.split(key, 4)
    learner_keys = jax.device_put_replicated(key, args.learner_devices)

    # env setup
    envs = dataclasses.replace(args.train_env, num_envs=args.local_num_envs).make()

    def linear_schedule(count):
        # anneal learning rate linearly after one training iteration which contains
        # (args.num_minibatches) gradient updates
        frac = 1.0 - (count // (args.num_minibatches)) / args.num_updates
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

    def impala_loss(params, x, a, logitss, rewards, dones, firststeps):
        discounts = (1.0 - dones) * args.gamma
        mask = 1.0 - firststeps
        policy_logits, newvalue = jax.vmap(get_logits_and_value, in_axes=(None, 0))(params, x)

        v_t = newvalue[1:]
        # Remove bootstrap timestep from non-timesteps.
        v_tm1 = newvalue[:-1]
        policy_logits = policy_logits[:-1]
        logitss = logitss[:-1]
        a = a[:-1]
        mask = mask[:-1]
        rewards = rewards[:-1]
        discounts = discounts[:-1]

        rhos = rlax.categorical_importance_sampling_ratios(policy_logits, logitss, a)
        vtrace_td_error_and_advantage = jax.vmap(
            partial(
                rlax.vtrace_td_error_and_advantage,
                lambda_=0.95,
                clip_rho_threshold=1.0,
                clip_pg_rho_threshold=1.0,
                stop_target_gradients=True,
            ),
            in_axes=1,
            out_axes=1,
        )

        vtrace_returns = vtrace_td_error_and_advantage(v_tm1, v_t, rewards, discounts, rhos)
        pg_advs = vtrace_returns.pg_advantage
        pg_loss = policy_gradient_loss(policy_logits, a, pg_advs, mask)

        baseline_loss = 0.5 * jnp.sum(jnp.square(vtrace_returns.errors) * mask)
        ent_loss = entropy_loss_fn(policy_logits, mask)

        total_loss = pg_loss
        total_loss += args.vf_coef * baseline_loss
        total_loss += args.ent_coef * ent_loss
        return total_loss, (pg_loss, baseline_loss, ent_loss)

    @jax.jit
    def single_device_update(
        agent_state: TrainState,
        sharded_storages: List[Transition],
        key: jax.random.PRNGKey,
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
                mb_firststeps,
            ) = minibatch
            (loss, (pg_loss, v_loss, entropy_loss)), grads = impala_loss_grad_fn(
                agent_state.params,
                mb_obs,
                mb_actions,
                mb_logitss,
                mb_rewards,
                mb_dones,
                mb_firststeps,
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
                        storage.obs,
                        args.num_minibatches * args.gradient_accumulation_steps,
                        axis=1,
                    )
                ),
                jnp.array(
                    jnp.split(
                        storage.actions,
                        args.num_minibatches * args.gradient_accumulation_steps,
                        axis=1,
                    )
                ),
                jnp.array(
                    jnp.split(
                        storage.logitss,
                        args.num_minibatches * args.gradient_accumulation_steps,
                        axis=1,
                    )
                ),
                jnp.array(
                    jnp.split(
                        storage.rewards,
                        args.num_minibatches * args.gradient_accumulation_steps,
                        axis=1,
                    )
                ),
                jnp.array(
                    jnp.split(
                        storage.dones,
                        args.num_minibatches * args.gradient_accumulation_steps,
                        axis=1,
                    )
                ),
                jnp.array(
                    jnp.split(
                        storage.firststeps,
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
        devices=args.global_learner_devices,
    )

    params_queues = []
    rollout_queues = []
    dummy_writer = SimpleNamespace()
    dummy_writer.add_scalar = lambda x, y, z: None

    unreplicated_params = agent_state.params
    key, *actor_keys = jax.random.split(key, 1 + len(args.actor_device_ids))
    for d_idx, d_id in enumerate(args.actor_device_ids):
        device_params = jax.device_put(unreplicated_params, local_devices[d_id])
        for thread_id in range(args.num_actor_threads):
            params_queues.append(queue.Queue(maxsize=1))
            rollout_queues.append(queue.Queue(maxsize=1))
            params_queues[-1].put(device_params)
            threading.Thread(
                target=rollout,
                args=(
                    jax.device_put(actor_keys[d_idx], local_devices[d_id]),
                    args,
                    rollout_queues[-1],
                    params_queues[-1],
                    writer if d_idx == 0 and thread_id == 0 else dummy_writer,
                    args.learner_devices,
                    d_idx * args.num_actor_threads + thread_id,
                    local_devices[d_id],
                ),
            ).start()

    rollout_queue_get_time = deque(maxlen=10)
    data_transfer_time = deque(maxlen=10)
    learner_policy_version = 0
    agent_state = jax.device_put_replicated(agent_state, devices=args.learner_devices)
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
            device_params = jax.device_put(unreplicated_params, local_devices[d_id])
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
                with open(writer.save_dir / f"{saved_model_version:03d}.model") as f:
                    f.write(
                        flax.serialization.to_bytes(
                            [
                                vars(args),
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

        if learner_policy_version >= args.num_updates:
            break

    if args.distributed:
        jax.distributed.shutdown()

    envs.close()
    writer.close()
