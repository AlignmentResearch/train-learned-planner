from pathlib import Path

from farconf import parse_cli_into_dict

from cleanba.launcher import FlamingoRun, group_from_fname, launch_jobs

IMPLICIT_ARGS = [
    "train_env.min_episode_steps=90",
    "eval_at_steps=[0, 32000, 97800, 21000, 21516, 782, 195600, 10000, 293400, 31000, 1564, 15648, 391200, 20000, 9000, 2347, 37164, 30000, 11000, 9780, 58680, 3129, 19000, 31296, 156480, 8000, 3912, 254280, 586, 29000, 352080, 18000, 25428, 7000, 1369, 39000, 28000, 2151, 19560, 17000, 117360, 6000, 38000, 2934, 215160, 27000, 13692, 312960, 16000, 3716, 391, 35208, 5000, 37000, 7824, 26000, 1173, 15000, 29340, 78240, 4000, 36000, 1956, 176040, 25000, 23472, 273840, 2738, 14000, 371640, 3000, 35000, 3520, 24000, 195, 17604, 13000, 39120, 2000, 978, 34000, 11736, 136920, 23000, 1760, 234720, 12000, 33252, 332520, 1000, 33000, 5868, 2542, 22000, 27384, 3325]",
]

clis: list[list[str]] = [
    [
        "--from-py-fn=cleanba.config:sokoban_drc33_59",
        "rmsprop_eps=1.5625e-10",
        "eval_envs.valid_medium.env.seed=1581694829",
        "train_env.seed=1221409641",
        "loss.ent_coef=0.001",
        "total_timesteps=256000000",
        "seed=1485693912",
        "max_grad_norm=0.00015",
        "net.use_relu=false",
        "net.residual=true",
        "net.recurrent.forget_bias=1.0",
        "rmsprop_decay=0.999",
        "loss.vtrace_lambda=0.97",
        "loss.logit_l2_coef=1.5625e-05",
        "loss.weight_l2_coef=1.5625e-07",
        "net.recurrent.output_activation=sigmoid",
        *IMPLICIT_ARGS,
    ],
    [
        "--from-py-fn=cleanba.config:sokoban_drc33_59",
        "rmsprop_eps=1.5625e-10",
        "eval_envs.valid_medium.env.seed=1581694829",
        "train_env.seed=1221409641",
        "loss.ent_coef=0.001",
        "total_timesteps=256000000",
        "seed=1485693912",
        "max_grad_norm=0.00015",
        "net.use_relu=false",
        "net.residual=true",
        "net.recurrent.forget_bias=1.0",
        "rmsprop_decay=0.999",
        "loss.vtrace_lambda=0.97",
        "loss.logit_l2_coef=1.5625e-05",
        "loss.weight_l2_coef=1.5625e-07",
        "final_learning_rate=0.0",
        "net.recurrent.output_activation=sigmoid",
        *IMPLICIT_ARGS,
    ],
]


for cli in clis:
    parse_cli_into_dict(cli)


runs: list[FlamingoRun] = []
RUNS_PER_MACHINE = 1
for update_fns_i in range(0, len(clis), RUNS_PER_MACHINE):
    this_run_clis = [
        ["python", "-m", "cleanba.cleanba_impala", *clis[update_fns_i + j]]
        for j in range(min(RUNS_PER_MACHINE, len(clis) - update_fns_i))
    ]
    print(this_run_clis)
    runs.append(
        FlamingoRun(
            this_run_clis,
            CONTAINER_TAG="12db6d3-main",
            CPU=8,
            MEMORY="20G",
            GPU=1,
            PRIORITY="normal-batch",
            XLA_PYTHON_CLIENT_MEM_FRACTION='".95"',
        )
    )


GROUP: str = group_from_fname(__file__, "buggy-envpool")

if __name__ == "__main__":
    launch_jobs(
        runs,
        group=GROUP,
        job_template_path=Path(__file__).parent.parent.parent / "k8s/runner.yaml",
        project="cleanba",
    )
