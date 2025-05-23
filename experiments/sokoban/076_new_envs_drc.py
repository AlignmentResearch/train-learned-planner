import dataclasses
import shlex
from pathlib import Path

from farconf import parse_cli, update_fns_to_cli

from cleanba.config import Args, minipacman_drc33
from cleanba.environments import random_seed
from cleanba.launcher import FlamingoRun, group_from_fname, launch_jobs

clis: list[list[str]] = []
all_args: list[Args] = []

for env_seed, learn_seed in [(random_seed(), random_seed()) for _ in range(1)]:
    # for base_fn in [boxworld_drc33, minipacman_drc33]:
    # for base_fn in [boxworld_drc33]:
    # for ent_coef in [1e-3, 1e-2, 5e-2]:
    #     for vtrace_lambda in [0.97, 1.0, 0.5]:
    for base_fn in [minipacman_drc33]:

        def update_seeds(config: Args) -> Args:
            config.train_env = dataclasses.replace(config.train_env, seed=env_seed)
            config.seed = learn_seed

            config.learning_rate = 4e-4
            config.final_learning_rate = 4e-6
            config.anneal_lr = True

            config.eval_envs = {}
            config.total_timesteps = 200_000_000

            # config.loss = dataclasses.replace(config.loss, ent_coef=ent_coef, vtrace_lambda=vtrace_lambda)

            return config

        cli, _ = update_fns_to_cli(base_fn, update_seeds)

        print(shlex.join(cli))
        # Check that parsing doesn't error
        out = parse_cli(cli, Args)

        all_args.append(out)
        clis.append(cli)

runs: list[FlamingoRun] = []
RUNS_PER_MACHINE = 1
for i in range(0, len(clis), RUNS_PER_MACHINE):
    this_run_clis = [
        ["python", "-m", "cleanba.cleanba_impala", *clis[i + j]] for j in range(min(RUNS_PER_MACHINE, len(clis) - i))
    ]
    runs.append(
        FlamingoRun(
            this_run_clis,
            CONTAINER_TAG="4f8513c-main",
            CPU=6,
            MEMORY="150G",
            GPU=1,
            PRIORITY="high-batch",
            XLA_PYTHON_CLIENT_MEM_FRACTION='".95"',
        )
    )


GROUP: str = group_from_fname(__file__)

if __name__ == "__main__":
    launch_jobs(
        runs,
        group=GROUP,
        job_template_path=Path(__file__).parent.parent.parent / "k8s/runner.yaml",
        project="cleanba",
    )
