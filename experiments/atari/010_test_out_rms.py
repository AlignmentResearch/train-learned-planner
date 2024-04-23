import dataclasses
from pathlib import Path

from farconf import update_fns_to_cli

from cleanba.config import Args
from cleanba.environments import random_seed
from cleanba.launcher import FlamingoRun, group_from_fname, launch_jobs
from cleanba.network import RMSNorm

runs: list[FlamingoRun] = []
optimizer = "adam"

for norm in [RMSNorm()]:
    for base_fan_in in [x * 9 for x in [1, 8, 16, 32, 64, 128, 256, 512]]:
        for yang_init in [True]:
            for seed in [random_seed()]:  # , random_seed()]:

                def update_fn(config: Args) -> Args:
                    config.local_num_envs = 128
                    config.num_actor_threads = 1
                    config.num_steps = 20
                    config.num_minibatches = 4
                    config.total_timesteps = int(20e6)
                    config.seed = seed
                    config.sync_frequency = 1000000000
                    config.loss = dataclasses.replace(config.loss, vtrace_lambda=1.0, vf_coef=0.25)
                    config.max_grad_norm = 0.0625
                    config.rmsprop_decay = 0.99
                    config.rmsprop_eps = 1.5625e-05
                    config.learning_rate = 0.0006

                    config.optimizer = optimizer
                    config.optimizer_yang = yang_init
                    config.base_fan_in = base_fan_in
                    config.net = dataclasses.replace(config.net, yang_init=yang_init, norm=norm)
                    return config

                cli, _ = update_fns_to_cli(Args, update_fn)
                print(cli)

                runs.append(
                    FlamingoRun(
                        [["python", "-m", "cleanba.cleanba_impala", *cli]],
                        CONTAINER_TAG="6caee83-atari",
                        CPU=8,
                        MEMORY="7G",
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
