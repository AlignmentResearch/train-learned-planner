import dataclasses
from dataclasses import field
from pathlib import Path
from typing import List

from cleanba.environments import AtariEnv, EnvConfig, EnvpoolBoxobanConfig, random_seed
from cleanba.evaluate import EvalConfig
from cleanba.impala_loss import (
    ImpalaLossConfig,
)
from cleanba.network import AtariCNNSpec, NetworkSpec, SokobanResNetConfig


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

    loss: ImpalaLossConfig = ImpalaLossConfig()

    net: NetworkSpec = AtariCNNSpec(channels=(16, 32, 32), mlp_hiddens=(256,))

    # Algorithm specific arguments
    total_timesteps: int = 100_000_000  # total timesteps of the experiments
    learning_rate: float = 0.0006  # the learning rate of the optimizer
    local_num_envs: int = 64  # the number of parallel game environments for every actor device
    num_steps: int = 20  # the number of steps to run in each environment per policy rollout
    train_epochs: int = 1  # Repetitions of going through the collected training
    anneal_lr: bool = True  # Toggle learning rate annealing for policy and value networks
    num_minibatches: int = 4  # the number of mini-batches
    gradient_accumulation_steps: int = 1  # the number of gradient accumulation steps before performing an optimization step
    max_grad_norm: float = 0.0625  # the maximum norm for the gradient clipping
    optimizer: str = "rmsprop"
    rmsprop_eps: float = 1.5625e-05
    rmsprop_decay: float = 0.99
    optimizer_yang: bool = False

    queue_timeout: float = 300.0  # If any of the actor/learner queues takes at least this many seconds, crash training.

    num_actor_threads: int = 2  # The number of environment threads per actor device
    actor_device_ids: List[int] = field(default_factory=lambda: [0])  # the device ids that actor workers will use
    learner_device_ids: List[int] = field(default_factory=lambda: [0])  # the device ids that learner workers will use
    distributed: bool = False  # whether to use `jax.distributed`
    concurrency: bool = True  # whether to run the actor and learner concurrently


def sokoban_resnet() -> Args:
    CACHE_PATH = Path("/training/.sokoban_cache")
    return Args(
        train_env=EnvpoolBoxobanConfig(
            max_episode_steps=120,
            num_envs=1,
            cache_path=CACHE_PATH,
            split="train",
            difficulty="unfiltered",
        ),
        eval_envs=dict(
            valid_unfiltered=EvalConfig(
                EnvpoolBoxobanConfig(
                    max_episode_steps=240,
                    num_envs=64,
                    cache_path=CACHE_PATH,
                    split="valid",
                    difficulty="unfiltered",
                ),
                n_episode_multiple=2,
            ),
            test_unfiltered=EvalConfig(
                EnvpoolBoxobanConfig(
                    max_episode_steps=240,
                    num_envs=64,
                    cache_path=CACHE_PATH,
                    split="test",
                    difficulty="unfiltered",
                ),
                n_episode_multiple=2,
            ),
            valid_medium=EvalConfig(
                EnvpoolBoxobanConfig(
                    max_episode_steps=240,
                    num_envs=64,
                    cache_path=CACHE_PATH,
                    split="valid",
                    difficulty="medium",
                ),
                n_episode_multiple=2,
            ),
            test_medium=EvalConfig(
                EnvpoolBoxobanConfig(
                    max_episode_steps=240,
                    num_envs=64,
                    cache_path=CACHE_PATH,
                    split="test",
                    difficulty="medium",
                ),
                n_episode_multiple=2,
            ),
        ),
        eval_frequency=400_000,
        seed=1234,
        save_model=False,
        log_frequency=10,
        sync_frequency=int(1e20),
        net=SokobanResNetConfig(),
        total_timesteps=int(1e9),
    )
