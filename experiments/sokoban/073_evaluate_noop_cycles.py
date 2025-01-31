from pathlib import Path

from farconf import update_fns_to_cli

from cleanba.environments import EnvpoolBoxobanConfig
from cleanba.launcher import FlamingoRun, group_from_fname, launch_jobs
from cleanba.load_and_eval import LoadAndEvalArgs, default_load_and_eval

runs_to_evaluate = [
    # 0
    "/training/cleanba/071-noop-training-check/wandb/run-20250104_012008-tvqm1z59",
    "/training/cleanba/071-noop-training-check/wandb/run-20250128_083827-ughszi5c",
    "/training/cleanba/071-noop-training-check/wandb/run-20250128_083419-npy2b4gj",
    # 0.01
    "/training/cleanba/071-noop-training-check/wandb/run-20250103_184242-9vqao1an",
    "/training/cleanba/071-noop-training-check/wandb/run-20250128_083823-6mefystj",
    "/training/cleanba/071-noop-training-check/wandb/run-20250128_083826-he1vyswe",
    # 0.03
    "/training/cleanba/071-noop-training-check/wandb/run-20250130_164704-ozn4hh92",
    "/training/cleanba/071-noop-training-check/wandb/run-20250129_221050-2gav2nib",
    "/training/cleanba/071-noop-training-check/wandb/run-20250129_221046-d1zyc8xi",
    # 0.05
    "/training/cleanba/071-noop-training-check/wandb/run-20250121_062441-96czci5z",
    "/training/cleanba/071-noop-training-check/wandb/run-20250128_083825-hq3mur6i",
    "/training/cleanba/071-noop-training-check/wandb/run-20250128_083822-ccqhmkj2",
    # 0.09
    "/training/cleanba/071-noop-training-check/wandb/run-20250121_062441-yul7itdh",
    "/training/cleanba/071-noop-training-check/wandb/run-20250128_083830-0lywx4th",
    "/training/cleanba/071-noop-training-check/wandb/run-20250128_083826-ene0mewu",
]
runs_to_evaluate = [Path(k) for k in runs_to_evaluate]

clis: list[list[str]] = []
for load_path in runs_to_evaluate:

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
        return config

    cli, _ = update_fns_to_cli(default_load_and_eval, update)
    clis.append(cli)


runs: list[FlamingoRun] = []
RUNS_PER_MACHINE = 6
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
            XLA_PYTHON_CLIENT_MEM_FRACTION='".15"',
            parallel=True,
        )
    )


GROUP: str = group_from_fname(__file__) + "-new"

if __name__ == "__main__":
    launch_jobs(
        runs,
        group=GROUP,
        job_template_path=Path(__file__).parent.parent.parent / "k8s/runner.yaml",
        project="lp-cleanba",
    )
