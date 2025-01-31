from pathlib import Path

from farconf import update_fns_to_cli

from cleanba.launcher import FlamingoRun, group_from_fname, launch_jobs
from cleanba.load_and_eval import LoadAndEvalArgs, default_load_and_eval

group_to_subdir = {
    "/training/cleanba/071-noop-training-check/wandb/": {
        "run-20250103_184242-9vqao1an/local-files": [
            "cp_1101619200",
            "cp_1201766400",
            "cp_1301913600",
            "cp_1402060800",
            "cp_1502208000",
        ],
        "run-20250102_211115-v7d711by/local-files": [
            "cp_1101619200",
            "cp_1201766400",
            "cp_1301913600",
            "cp_1402060800",
            "cp_1502208000",
        ],
        "run-20250103_184237-frnuw7jp/local-files": [
            "cp_1101619200",
            "cp_1201766400",
            "cp_1301913600",
            "cp_1402060800",
            "cp_1502208000",
        ],
        "run-20250104_012008-tvqm1z59/local-files": [
            "cp_1602355200",
            "cp_1702502400",
            "cp_1802649600",
            "cp_1902796800",
            "cp_2002944000",
        ],
    },
}

runs_to_evaluate = list(reversed([(Path(k) / v, cps) for k, vs in group_to_subdir.items() for v, cps in vs.items()]))

clis: list[list[str]] = []
for load_path, checkpoints in runs_to_evaluate:

    def update(config: LoadAndEvalArgs) -> LoadAndEvalArgs:
        config.load_other_run = load_path
        for env in config.eval_envs.values():
            env.env.nn_without_noop = False
        config.checkpoints_to_load = checkpoints
        return config

    cli, _ = update_fns_to_cli(default_load_and_eval, update)
    clis.append(cli)


runs: list[FlamingoRun] = []
RUNS_PER_MACHINE = 2
for update_fns_i in range(0, len(clis), RUNS_PER_MACHINE):
    this_run_clis = [
        ["python", "-m", "cleanba.load_and_eval", *clis[update_fns_i + j]]
        for j in range(min(RUNS_PER_MACHINE, len(clis) - update_fns_i))
    ]
    print(this_run_clis)
    runs.append(
        FlamingoRun(
            this_run_clis,
            CONTAINER_TAG="2369b92-main",
            CPU=12,
            MEMORY="80G",
            GPU=1,
            PRIORITY="high-batch",
            XLA_PYTHON_CLIENT_MEM_FRACTION='".46"',
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
