import dataclasses
import shlex
from pathlib import Path

from farconf import parse_cli, update_fns_to_cli

from cleanba.config import Args, sokoban_resnet59
from cleanba.convlstm import ConvConfig, ConvLSTMCellConfig, ConvLSTMConfig
from cleanba.environments import random_seed
from cleanba.launcher import FlamingoRun, group_from_fname, launch_jobs
from cleanba.network import GuezResNetConfig, IdentityNorm

clis: list[list[str]] = []
all_args: list[Args] = []

drc_n_n = 3
arch_fns = [
    lambda cfg: dataclasses.replace(cfg, net=GuezResNetConfig(yang_init=False, norm=IdentityNorm(), normalize_input=False)),
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


for env_seed, learn_seed in [(random_seed(), random_seed()) for _ in range(1)]:
    for arch_fn in arch_fns:
        cli, _ = update_fns_to_cli(sokoban_resnet59, arch_fn)

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
            CONTAINER_TAG="5d26ad1-main",
            CPU=6,
            MEMORY="20G",
            GPU=1,
            PRIORITY="normal-batch",
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
