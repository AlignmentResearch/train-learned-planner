from pathlib import Path

from farconf import update_fns_to_cli

from cleanba.environments import EnvpoolBoxobanConfig
from cleanba.launcher import FlamingoRun, group_from_fname, launch_jobs
from cleanba.load_and_eval import LoadAndEvalArgs, default_load_and_eval

q1 = ["cp_0100147200", "cp_0200294400", "cp_0300441600", "cp_0400588800", "cp_0500736000"]
q2 = ["cp_0600883200", "cp_0701030400", "cp_0801177600", "cp_0901324800", "cp_1001472000"]
q3 = ["cp_1101619200", "cp_1201766400", "cp_1301913600", "cp_1402060800", "cp_1502208000"]
q4 = ["cp_1602355200", "cp_1702502400", "cp_1802649600", "cp_1902796800", "cp_2002944000"]

qall = q1 + q2 + q3 + q4

drc33_seeds = ["bkynosqi", "gobfm3wm", "jl6bq8ih", "q4mjldyy", "qqp0kn15"]

group_to_subdir = {
    "/training/cleanba/071-noop-training-check/wandb/": {
        "run-20250103_184242-9vqao1an/local-files": qall,
        "run-20250102_211115-v7d711by/local-files": qall,
        "run-20250103_184237-frnuw7jp/local-files": qall,
        "run-20250104_012008-tvqm1z59/local-files": qall,
    },
    # "/training/cleanba/061-pfinal2/drc33/": {k: qall for k in drc33_seeds},
}

runs_to_evaluate = list(reversed([(Path(k) / v, cps) for k, vs in group_to_subdir.items() for v, cps in vs.items()]))

clis: list[list[str]] = []
for load_path, checkpoints in runs_to_evaluate:

    def update(config: LoadAndEvalArgs) -> LoadAndEvalArgs:
        config.load_other_run = load_path
        config.eval_envs.pop("test_unfiltered")
        config.eval_envs.pop("hard")

        config.eval_envs["valid_medium"].env.n_levels_to_load = 5000
        config.eval_envs["valid_medium"].env.num_envs = 500
        config.eval_envs["valid_medium"].n_episode_multiple = 10
        config.eval_envs["valid_medium"].steps_to_think = [0]
        for env in config.eval_envs.values():
            assert isinstance(env.env, EnvpoolBoxobanConfig)
            env.env.nn_without_noop = "drc33" in str(load_path)
            print(env.env.nn_without_noop)
        config.checkpoints_to_load = checkpoints
        return config

    cli, _ = update_fns_to_cli(default_load_and_eval, update)
    clis.append(cli)


runs: list[FlamingoRun] = []
RUNS_PER_MACHINE = 4
for update_fns_i in range(0, len(clis), RUNS_PER_MACHINE):
    this_run_clis = [
        ["python", "-m", "cleanba.load_and_eval", *clis[update_fns_i + j]]
        for j in range(min(RUNS_PER_MACHINE, len(clis) - update_fns_i))
    ]

    runs.append(
        FlamingoRun(
            this_run_clis,
            CONTAINER_TAG="4f8513c-main",
            CPU=18,
            MEMORY="150G",
            GPU=1,
            PRIORITY="high-batch",
            XLA_PYTHON_CLIENT_MEM_FRACTION='".2"',
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
