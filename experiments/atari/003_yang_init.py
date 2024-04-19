import dataclasses
from pathlib import Path

from farconf import update_fns_to_cli

from cleanba.config import Args, sokoban_resnet
from cleanba.environments import random_seed
from cleanba.launcher import FlamingoRun, group_from_fname, launch_jobs

runs: list[FlamingoRun] = []


for yang_init in [True, False]:
    for seed in [random_seed(), random_seed(), random_seed()]:

        def update_fn(config: Args) -> Args:
            config.local_num_envs = 128
            config.num_actor_threads = 1
            config.num_steps = 20
            config.num_minibatches = 4
            config.total_timesteps = int(1e9)
            config.seed = seed
            config.sync_frequency = 1000000000
            config.loss = dataclasses.replace(config.loss, vtrace_lambda=1.0, vf_coef=0.25)
            config.max_grad_norm = 0.0625
            config.rmsprop_decay = 0.99
            config.rmsprop_eps = 1.5625e-05
            config.learning_rate = 0.0006

            config.optimizer_yang = yang_init
            config.net = dataclasses.replace(config.net, yang_init=yang_init)
            return config

        cli, _ = update_fns_to_cli(sokoban_resnet, update_fn)
        print(cli)

        runs.append(
            FlamingoRun(
                [["python", "-m", "cleanba.cleanba_impala", *cli]],
                CONTAINER_TAG="e5a58c4-main",
                CPU=8,
                MEMORY="50G",
                GPU=1,
                PRIORITY="normal-batch",
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
