import dataclasses
import random
from pathlib import Path

from farconf import update_fns_to_cli

from cleanba.config import Args, sokoban_resnet
from cleanba.environments import random_seed
from cleanba.launcher import FlamingoRun, group_from_fname, launch_jobs
from cleanba.network import RMSNorm

runs: list[FlamingoRun] = []

random.seed(12345)

for train_epochs in [4, 6, 10]:

    def update_fn(config: Args) -> Args:
        config.train_env.seed = random_seed()
        config.seed = random_seed()
        config.num_minibatches = 16
        config.max_grad_norm = 0.12
        config.train_epochs = train_epochs

        config.eval_envs = {}
        config.loss = dataclasses.replace(config.loss, gamma=0.97, vtrace_lambda=0.97, vf_coef=0.5)
        config.net = dataclasses.replace(config.net, multiplicity=3, norm=RMSNorm(eps=1e-8))
        config.learning_rate = 0.002
        config.local_num_envs = 256
        config.num_actor_threads = 1
        config.num_steps = 120
        config.optimizer = "adam"
        config.rmsprop_decay = 0.95
        config.rmsprop_eps = 1e-8
        config.total_timesteps = 50_000_000

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
