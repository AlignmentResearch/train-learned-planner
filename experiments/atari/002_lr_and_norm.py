from pathlib import Path

from experiments.launcher import FlamingoRun, group_from_fname, launch_jobs

runs: list[FlamingoRun] = [
    FlamingoRun(
        [
            [
                "python",
                "-m",
                "cleanba.cleanba_impala",
                "local_num_envs=64",
                "num_actor_threads=2",
                "num_steps=20",
                "num_minibatches=4",
                "total_timesteps=10000000",  # reduced timesteps for speed
                "seed=1",
                "sync_frequency=1000000000",
                "eval_frequency=1000000000",
                "loss.vf_coef=0.25",
                "loss.vtrace_lambda=1.0",
                f"max_grad_norm={max_grad_norm}",
                "rmsprop_decay=0.99",
                f"rmsprop_eps={eps}",
                f"learning_rate={lr}",
                f"net.norm._type_=cleanba.network:{norm}",
            ]
        ],
        CONTAINER_TAG="latest-atari",
        CPU=12,
        MEMORY="60G",
        GPU=1,
        PRIORITY="normal-batch",
    )
    for max_grad_norm in [0.0625, 0.625]
    for lr in [0.0006, 0.002, 0.0002]
    for norm in ["RMSNorm", "IdentityNorm"]
    for eps in [1.5625e-05, 1e-8]
]

GROUP: str = group_from_fname(__file__)

if __name__ == "__main__":
    launch_jobs(
        runs,
        group=GROUP,
        job_template_path=Path(__file__).parent.parent.parent / "k8s/runner-no-nfs.yaml",
        project="cleanba",
    )
