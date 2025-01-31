from pathlib import Path

import wandb
from farconf import obj_to_cli, parse_cli, to_dict

from cleanba.config import Args
from cleanba.launcher import FlamingoRun, launch_jobs

job_names: list[str] = []
clis: list[list[str]] = []

wandb_ids = ["npy2b4gj", "he1vyswe", "ene0mewu", "0lywx4th", "ccqhmkj2", "6mefystj", "hq3mur6i", "ughszi5c"]
group = "071-noop-training-check"

api = wandb.Api()
for i, wandb_id in enumerate(wandb_ids):
    run = api.run(f"lp-cleanba/{wandb_id}")
    cli = run.metadata["args"]
    args = parse_cli(cli, Args)
    args.total_timesteps = 2_002_944_000
    cli = obj_to_cli(to_dict(args))
    clis.append(cli)
    job_names.append(run.name)

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
            CPU=4 * RUNS_PER_MACHINE,
            MEMORY=f"{15 * RUNS_PER_MACHINE}G",
            GPU=1,
            PRIORITY="normal-batch",
            # PRIORITY="high-batch",
            XLA_PYTHON_CLIENT_MEM_FRACTION='".90"',
            job_names=job_names[i : i + RUNS_PER_MACHINE],
        )
    )

if __name__ == "__main__":
    launch_jobs(
        runs,
        group=group,
        job_template_path=Path(__file__).parent.parent.parent / "k8s/runner.yaml",
        project="lp-cleanba",
    )
