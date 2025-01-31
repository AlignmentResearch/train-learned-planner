from pathlib import Path

from farconf import update_fns_to_cli

from cleanba.launcher import FlamingoRun, group_from_fname, launch_jobs
from cleanba.load_and_eval import LoadAndEvalArgs, default_load_and_eval

drc33_seeds = ["bkynosqi", "gobfm3wm", "jl6bq8ih", "q4mjldyy", "qqp0kn15"]
drc11_seeds = ["3a2pv9yr", "3i5nocf6", "eue6pax7", "nom9jda6", "v2fm2qze"]
resnet_seeds = ["13qckf6e", "28n07cac", "8ullb23e", "syb50iz7", "zgyp3v0o"]

base_dir = Path("/training/cleanba/061-pfinal2/")

runs_to_evaluate = [base_dir / "drc33/" / seed_name for seed_name in drc33_seeds]
runs_to_evaluate += [base_dir / "drc11/" / seed_name for seed_name in drc11_seeds]
runs_to_evaluate += [base_dir / "resnet/" / seed_name for seed_name in resnet_seeds]

clis: list[list[str]] = []
for load_path in runs_to_evaluate:

    def update(config: LoadAndEvalArgs) -> LoadAndEvalArgs:
        config.load_other_run = load_path
        config.save_logs = False
        return config

    cli, _ = update_fns_to_cli(default_load_and_eval, update)
    clis.append(cli)


runs: list[FlamingoRun] = []
RUNS_PER_MACHINE = 5
for update_fns_i in range(0, len(clis), RUNS_PER_MACHINE):
    this_run_clis = [
        ["python", "-m", "cleanba.load_and_eval", *clis[update_fns_i + j]]
        for j in range(min(RUNS_PER_MACHINE, len(clis) - update_fns_i))
    ]
    print(this_run_clis)
    runs.append(
        FlamingoRun(
            this_run_clis,
            CONTAINER_TAG="4f8513c-main",
            CPU=30,
            MEMORY="150G",
            GPU=1,
            PRIORITY="high-batch",
            XLA_PYTHON_CLIENT_MEM_FRACTION='".16"',
            parallel=True,
        )
    )


GROUP: str = group_from_fname(__file__)

if __name__ == "__main__":
    launch_jobs(
        runs,
        group=GROUP,
        job_template_path=Path(__file__).parent.parent.parent / "k8s/runner.yaml",
        project="lp-cleanba",
    )
