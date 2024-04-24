import dataclasses
import shlex
from pathlib import Path

from farconf import update_fns_to_cli

from cleanba.config import Args, sokoban_resnet
from cleanba.environments import SokobanConfig, random_seed
from cleanba.launcher import FlamingoRun, group_from_fname, launch_jobs
from cleanba.network import GuezResNetConfig, IdentityNorm

minibatch_size = 32
n_envs = 256  # the paper says 200 actors
assert n_envs % minibatch_size == 0


logit_l2_coef = 1.5625e-6  # doesn't seem to matter much from now. May improve stability a tiny bit.

clis = []
for env_seed, learn_seed in [(random_seed(), random_seed()), (random_seed(), random_seed())]:
    for max_episode_steps in [20, 60, 80, 120]:
        for dim_room, num_boxes in [(8, 1), (9, 1), (10, 1)]:
            network = GuezResNetConfig(yang_init=True, norm=IdentityNorm())

            def update_fn(config: Args) -> Args:
                config.train_env = SokobanConfig(
                    max_episode_steps=max_episode_steps,
                    num_envs=1,
                    seed=env_seed,
                    min_episode_steps=max_episode_steps * 3 // 4,
                    tinyworld_obs=True,
                    asynchronous=True,
                    dim_room=(dim_room, dim_room),
                    num_boxes=num_boxes,
                )
                config.eval_envs = {}
                config.actor_update_frequency = 4
                config.actor_update_cutoff = int(1e20)

                config.local_num_envs = n_envs
                config.train_epochs = 1
                config.num_actor_threads = 1
                config.num_steps = 20
                config.num_minibatches = (config.local_num_envs * config.num_actor_threads) // minibatch_size
                config.total_timesteps = int(1e7)
                config.seed = learn_seed
                config.sync_frequency = int(1e20)
                config.loss = dataclasses.replace(
                    config.loss,
                    vtrace_lambda=0.97,
                    vf_coef=0.1,
                    gamma=0.97,
                    ent_coef=0.01,
                    normalize_advantage=False,
                    logit_l2_coef=logit_l2_coef,
                    weight_l2_coef=logit_l2_coef / 100,
                )
                config.base_fan_in = 1
                config.anneal_lr = True

                config.optimizer = "adam"
                config.adam_b1 = 0.9
                config.rmsprop_decay = 0.999
                config.learning_rate = 4e-4
                config.max_grad_norm = 6.25e-05
                config.rmsprop_eps = 1.5625e-07
                config.optimizer_yang = False

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
            CPU=14,
            MEMORY="70G",
            GPU=1,
            PRIORITY="normal-batch",
            XLA_PYTHON_CLIENT_MEM_FRACTION='".45"',
        )
    )


GROUP: str = group_from_fname(__file__, "4")

if __name__ == "__main__":
    launch_jobs(
        runs,
        group=GROUP,
        job_template_path=Path(__file__).parent.parent.parent / "k8s/runner-no-nfs.yaml",
        project="cleanba",
    )
