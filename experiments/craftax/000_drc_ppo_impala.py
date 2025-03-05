import dataclasses
import shlex
from pathlib import Path

from farconf import parse_cli, update_fns_to_cli

from cleanba.config import Args, craftax_drc
from cleanba.environments import random_seed
from cleanba.launcher import FlamingoRun, group_from_fname, launch_jobs

clis: list[list[str]] = []
all_args: list[Args] = []

for gae_lambda in [0.8]:
    for env_seed, learn_seed in [(random_seed(), random_seed()) for _ in range(1)]:

        def update_seeds(config: Args) -> Args:
            config.train_env = dataclasses.replace(config.train_env, seed=env_seed)
            config.seed = learn_seed

            config.loss = dataclasses.replace(config.loss, gae_lambda=gae_lambda)
            config.base_run_dir = Path("/training/craftax")
            config.train_epochs = 4
            config.queue_timeout = 3000
            config.total_timesteps = 1000_000_000
            config.load_path = Path("/training/craftax/000-drc-ppo-impala/wandb/unruffled-golick-1/local-files/cp_063897600")
            return config

        cli, _ = update_fns_to_cli(craftax_drc, update_seeds)

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
            CONTAINER_TAG="4350d99-main",
            CPU=4 * RUNS_PER_MACHINE,
            MEMORY=f"{60 * RUNS_PER_MACHINE}G",
            GPU=1,
            PRIORITY="normal-batch",
            # PRIORITY="high-batch",
            XLA_PYTHON_CLIENT_MEM_FRACTION='".9"',  # Can go down to .48
        )
    )


GROUP: str = group_from_fname(__file__)

if __name__ == "__main__":
    launch_jobs(
        runs,
        group=GROUP,
        job_template_path=Path(__file__).parent.parent.parent / "k8s/runner.yaml",
        project="impala2",
        entity="matsrlgoals",
    )
