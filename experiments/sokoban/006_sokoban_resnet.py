import dataclasses
import random
from pathlib import Path
from typing import Any

import numpy as np
from farconf import update_fns_to_cli

from cleanba.config import Args, sokoban_resnet
from cleanba.environments import SokobanConfig, random_seed
from cleanba.launcher import FlamingoRun, group_from_fname, launch_jobs
from cleanba.network import IdentityNorm, RMSNorm

runs: list[FlamingoRun] = []

random_params = dict(train_epochs=[4, 6, 10], num_minibatches=[32, 16, 8], max_grad_norm=[1.0, 0.3, 0.12, 0.06])

random.seed(12345)
np_random = np.random.default_rng(random_seed())

for run_idx in range(9):
    params: dict[str, Any] = {k: np_random.choice(v) for k, v in random_params.items()}
    seed = random_seed()

    def update_fn(config: Args) -> Args:
        config.eval_envs = {}

        config.loss = dataclasses.replace(config.loss, gamma=0.97, vtrace_lambda=0.97, vf_coef=0.5)

        config.net = dataclasses.replace(config.net, multiplicity=3, norm=RMSNorm(eps=1e-8))

        config.learning_rate = 0.002
        config.local_num_envs = 256
        config.max_grad_norm = params["max_grad_norm"]
        config.num_actor_threads = 1
        config.num_minibatches = params["num_minibatches"]
        config.train_epochs = params["train_epochs"]
        config.num_steps = 120
        config.optimizer = "adam"
        config.rmsprop_decay = 0.95
        config.rmsprop_eps = 1e-8
        config.total_timesteps = 50_000_000
        config.train_epochs = 1

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
