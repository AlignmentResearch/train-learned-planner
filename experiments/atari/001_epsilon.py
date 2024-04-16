from pathlib import Path

from experiments.launcher import FlamingoRun, launch_jobs

runs: list[FlamingoRun] = [
    FlamingoRun(
        [
            [
                "python",
                "-m",
                "cleanba.cleanba_impala",
                "--from-py-fn=cleanba.config:Args",
                "local_num_envs=64",
                "num_actor_threads=2",
                "num_steps=20",
                "num_minibatches=4",
                "total_timesteps=50000000",
                "seed=1",
                "sync_frequency=1000000000",
                "loss.vf_coef=0.25",
                "loss.vtrace_lambda=1.0",
                "loss.global_coef=1.0",
                "max_grad_norm=0.0625",
                "rmsprop_decay=0.99",
                "rmsprop_eps=1.5625e-05",
                "learning_rate=0.0006",
            ]
        ],
        CONTAINER_TAG="latest-atari",
        CPU=12,
        MEMORY="60G",
        GPU=1,
    )
]

if __name__ == "__main__":
    launch_jobs(runs, group="atari", job_template_path=Path(__file__).parent.parent.parent / "k8s/runner-no-nfs.yaml")
