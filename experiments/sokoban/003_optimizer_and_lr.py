from pathlib import Path

from experiments.launcher import FlamingoRun, group_from_fname, launch_jobs

runs: list[FlamingoRun] = [
    FlamingoRun(
        [
            [
                "python",
                "-m",
                "cleanba.cleanba_impala",
                "--from-py-fn=cleanba.config:sokoban_resnet",
                "net.multiplicity=1",
                "num_actor_threads=2",
                "local_num_envs=128",
                f"loss.vtrace_lambda=0.97",
                "loss.gamma=0.97",
                "num_minibatches=4",
                "rmsprop_decay=0.95",
                f"train_epochs={train_epochs}",
                f"max_grad_norm={max_grad_norm}",
                f"rmsprop_eps={eps}",
                f"learning_rate={lr}",
                f'net.norm={{"_type_": "cleanba.network:{norm}"}}',
                "total_timesteps=500000000",
                f"optimizer={optimizer}",
            ]
        ],
        CONTAINER_TAG="e5a58c4-main",
        CPU=8,
        MEMORY="50G",
        GPU=1,
        PRIORITY="normal-batch",
    )
    for max_grad_norm in [0.0625]
    for lr in [0.0006, 0.00006]
    for norm in ["RMSNorm", "IdentityNorm"]
    for optimizer in ["adam", "rmsprop"]
    for train_epochs in [2, 4, 8]
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
