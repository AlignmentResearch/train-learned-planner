import dataclasses
from pathlib import Path

from farconf import update_fns_to_cli

from cleanba.config import Args, sokoban_resnet
from cleanba.environments import random_seed
from cleanba.launcher import FlamingoRun, group_from_fname, launch_jobs
from cleanba.network import AtariCNNSpec, IdentityNorm, RMSNorm

runs: list[FlamingoRun] = []

for yang in [False, True]:
    for n in [1, 2]:
        for seed in [random_seed(), random_seed()]:

            def update_fn(config: Args) -> Args:
                config.eval_envs = {}

                config.local_num_envs = 128
                config.num_actor_threads = 1
                config.num_steps = 20
                config.num_minibatches = 4
                config.total_timesteps = int(40e6)
                config.seed = seed
                config.sync_frequency = 1000000000
                config.loss = dataclasses.replace(config.loss, vtrace_lambda=1.0, vf_coef=0.25)
                config.max_grad_norm = 0.625 if "yang" else 0.0625
                config.rmsprop_decay = 0.99
                config.rmsprop_eps = 1.5625e-05
                config.learning_rate = 0.001 if "yang" else 0.0006

                config.optimizer = "adam"
                config.optimizer_yang = yang

                norm = RMSNorm() if yang else IdentityNorm()
                config.net = AtariCNNSpec(
                    yang_init=yang, channels=(16 * n, 32 * n, 32 * n), strides=(1, 1, 1), mlp_hiddens=(256,), norm=norm
                )
                return config

            cli, _ = update_fns_to_cli(sokoban_resnet, update_fn)
            print(cli)

            runs.append(
                FlamingoRun(
                    [["python", "-m", "cleanba.cleanba_impala", *cli]],
                    CONTAINER_TAG="e5a58c4-main",
                    CPU=8,
                    MEMORY="10G",
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
