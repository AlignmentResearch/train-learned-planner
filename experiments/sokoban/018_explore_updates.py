import dataclasses
import random
import shlex
from pathlib import Path

from farconf import update_fns_to_cli

from cleanba.config import Args, sokoban_resnet
from cleanba.environments import SokobanConfig, random_seed
from cleanba.launcher import FlamingoRun, group_from_fname, launch_jobs
from cleanba.network import AtariCNNSpec, RMSNorm, SokobanResNetConfig

clis = []

for _ in range(48):
    env_seed, learn_seed = random_seed(), random_seed()

    learning_rate = 10 ** random.uniform(-4, -2)
    yang_init = random.random() < 0.8
    yang_optimizer = random.random() < 0.8

    update_freq = random.choice(list(range(1, 14)))

    norm = RMSNorm(eps=1e-06, use_scale=True, reduction_axes=-1, feature_axes=-1)

    minibatch_size = random.choice([32, 64, 128])
    n_envs = 256  # the paper says 200 actors
    assert n_envs % minibatch_size == 0

    vtrace_lambda = random.choice([0.95, 0.97, 1.0])
    vf_coef = 10 ** random.uniform(-2, -0.5)

    num_layers = random.choice([6, 9, 12])

    network = random.choice(
        [
            SokobanResNetConfig(
                yang_init=yang_init,
                norm=norm,
                channels=(64,) * num_layers,
                kernel_sizes=(4,) * num_layers,
                mlp_hiddens=(256,),
                last_activation="relu",
            ),
            AtariCNNSpec(
                yang_init=yang_init,
                norm=norm,
                channels=(64,) * num_layers,
                strides=(1,) * num_layers,
                mlp_hiddens=(256,),
                max_pool=False,
            ),
        ]
    )

    max_episode_steps = random.choice([20, 60, 88, 120])

    train_epochs = random.choice([1, 2])

    for dim_room, total_timesteps in [(7, int(1e7)), (8, int(3e7)), (10, int(6e7))]:

        def update_fn(config: Args) -> Args:
            config.train_env = SokobanConfig(
                max_episode_steps=max_episode_steps,
                num_envs=1,
                seed=env_seed,
                min_episode_steps=max_episode_steps * 3 // 4,
                tinyworld_obs=True,
                asynchronous=True,
                dim_room=(dim_room, dim_room),
                num_boxes=1,
            )
            config.eval_envs = {}
            config.actor_update_frequency = update_freq

            config.local_num_envs = n_envs
            config.train_epochs = train_epochs
            config.num_actor_threads = 1
            config.num_steps = 20
            config.num_minibatches = (config.local_num_envs * config.num_actor_threads) // minibatch_size
            config.total_timesteps = total_timesteps
            config.seed = learn_seed
            config.sync_frequency = 1000000000
            config.loss = dataclasses.replace(
                config.loss,
                vtrace_lambda=1.0,
                vf_coef=vf_coef,
                gamma=0.97,
                ent_coef=0.01,
                normalize_advantage=False,
            )
            config.max_grad_norm = 1.0
            config.learning_rate = learning_rate
            config.base_fan_in = 72
            config.anneal_lr = True

            config.optimizer = "adam"
            config.optimizer_yang = yang_optimizer

            config.net = network
            return config

        cli, _ = update_fns_to_cli(sokoban_resnet, update_fn)
        print(shlex.join(cli))
        clis.append(cli)

runs: list[FlamingoRun] = []
RUNS_PER_MACHINE = 2
for i in range(0, len(clis), RUNS_PER_MACHINE):
    runs.append(
        FlamingoRun(
            [["python", "-m", "cleanba.cleanba_impala", *clis[i + j]] for j in range(min(RUNS_PER_MACHINE, len(clis) - i))],
            CONTAINER_TAG="e5a58c4-main",
            CPU=10,
            MEMORY="70G",
            GPU=1,
            PRIORITY="normal-batch",
            XLA_PYTHON_CLIENT_MEM_FRACTION='".40"',
        )
    )


GROUP: str = group_from_fname(__file__)

if __name__ == "__main__":
    launch_jobs(
        runs,
        group=GROUP,
        job_template_path=Path(__file__).parent.parent.parent / "k8s/runner-no-nfs.yaml",
        project="cleanba",
    )
