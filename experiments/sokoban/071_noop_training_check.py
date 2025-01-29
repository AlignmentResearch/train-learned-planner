# Taken from 061_pfinal2.py which was the script that trained the drc3,3 used in the paper
# https://huggingface.co/AlignmentResearch/learned-planner/tree/main/drc33/bkynosqi
import dataclasses
import shlex
from pathlib import Path

from farconf import parse_cli, update_fns_to_cli

from cleanba.config import Args, sokoban_resnet59
from cleanba.convlstm import ConvConfig, ConvLSTMCellConfig, ConvLSTMConfig
from cleanba.environments import EnvpoolBoxobanConfig, random_seed
from cleanba.launcher import FlamingoRun, group_from_fname, launch_jobs

clis: list[list[str]] = []
all_args: list[Args] = []

drc_n_n = 3
arch_fns = [
    lambda cfg: dataclasses.replace(
        cfg,
        net=ConvLSTMConfig(
            n_recurrent=drc_n_n,
            repeats_per_step=drc_n_n,
            skip_final=True,
            residual=False,
            use_relu=False,
            embed=[ConvConfig(32, (4, 4), (1, 1), "SAME", True)] * 2,
            recurrent=ConvLSTMCellConfig(
                ConvConfig(32, (3, 3), (1, 1), "SAME", True),
                pool_and_inject="horizontal",
                pool_projection="per-channel",
                output_activation="tanh",
                fence_pad="valid",
                forget_bias=0.0,
            ),
            head_scale=1.0,
        ),
    ),
]


for i, env_seed, learn_seed in [(i, random_seed(), random_seed()) for i in range(2)]:
    for arch_fn in arch_fns:
        for reward_noop in [0.0, 0.01, 0.05, 0.09]:
            for ent_coef in [1e-2]:

                def update_seeds(config: Args) -> Args:
                    config.train_env = dataclasses.replace(config.train_env, seed=env_seed)
                    config.seed = learn_seed

                    config.learning_rate = 4e-4
                    config.final_learning_rate = 4e-6
                    config.anneal_lr = True

                    config.loss = dataclasses.replace(config.loss, ent_coef=ent_coef)
                    assert isinstance(config.train_env, EnvpoolBoxobanConfig)
                    config.train_env.reward_noop = reward_noop
                    config.train_env.nn_without_noop = False

                    config.total_timesteps = 2_002_944_000

                    for env in config.eval_envs.values():
                        env.steps_to_think = [0]

                    return config

                if i > 0 and reward_noop >= 1.0:  # only one run with reward_noop >= 1.0
                    continue

                cli, _ = update_fns_to_cli(sokoban_resnet59, arch_fn, update_seeds)

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
            CPU=4 * RUNS_PER_MACHINE,
            MEMORY=f"{15 * RUNS_PER_MACHINE}G",
            GPU=1,
            PRIORITY="normal-batch",
            # PRIORITY="high-batch",
            XLA_PYTHON_CLIENT_MEM_FRACTION='".90"',
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
