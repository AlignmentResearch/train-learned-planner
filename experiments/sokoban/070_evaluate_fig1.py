from pathlib import Path

from farconf import update_fns_to_cli

from cleanba.launcher import FlamingoRun, group_from_fname, launch_jobs
from cleanba.load_and_eval import LoadAndEvalArgs, default_load_and_eval

group_to_subdir = {
    "/training/cleanba/061-pfinal2/wandb/run": [
        "run-20240618_205932-syb50iz7",
        "run-20240618_205934-bkynosqi",
    ],
    "/training/cleanba/061-pfinal2-drc11/wandb/run": [
        "run-20240623_041343-eue6pax7",
    ],
}

runs_to_evaluate = [Path(k) / v for k, vs in group_to_subdir.items() for v in vs]


clis: list[list[str]] = []
for load_path in runs_to_evaluate:

    def update(config: LoadAndEvalArgs) -> LoadAndEvalArgs:
        config.load_other_run = load_path
        return config

    cli, _ = update_fns_to_cli(default_load_and_eval, update)
    clis.append(cli)


runs: list[FlamingoRun] = []
RUNS_PER_MACHINE = 1
for update_fns_i in range(0, len(clis), RUNS_PER_MACHINE):
    this_run_clis = [
        ["python", "-m", "cleanba.load_and_eval", *clis[update_fns_i + j]]
        for j in range(min(RUNS_PER_MACHINE, len(clis) - update_fns_i))
    ]
    print(this_run_clis)
    runs.append(
        FlamingoRun(
            this_run_clis,
            CONTAINER_TAG="6f8d92b-main",
            CPU=12,
            MEMORY="40G",
            GPU=1,
            PRIORITY="high-batch",
            XLA_PYTHON_CLIENT_MEM_FRACTION='".45"',
            parallel=True,
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
