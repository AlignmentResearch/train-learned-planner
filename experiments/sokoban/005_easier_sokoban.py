import dataclasses
from pathlib import Path

from farconf import update_fns_to_cli

from cleanba.config import Args, sokoban_resnet
from cleanba.environments import SokobanConfig
from cleanba.launcher import FlamingoRun, group_from_fname, launch_jobs
from cleanba.network import IdentityNorm, RMSNorm

runs: list[FlamingoRun] = []


def update_fn(config: Args) -> Args:
    config.train_env = SokobanConfig(max_episode_steps=60, num_envs=1, tinyworld_obs=True, dim_room=(10, 10), num_boxes=1)
    config.eval_envs = {}

    config.loss = dataclasses.replace(config.loss, gamma=0.97, vtrace_lambda=0.97, vf_coef=1.0)

    config.net = dataclasses.replace(config.net, multiplicity=1, norm=IdentityNorm())

    config.learning_rate = 1e-4
    config.local_num_envs = 64
    config.max_grad_norm = 0.0625
    config.num_actor_threads = 2
    config.num_minibatches = 4
    config.num_steps = 20
    config.optimizer = "rmsprop"
    config.rmsprop_decay = 0.95
    config.rmsprop_eps = 1e-8
    config.total_timesteps = 500_000_000
    config.train_epochs = 1

    return config


cli, _ = update_fns_to_cli(sokoban_resnet, update_fn)

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
