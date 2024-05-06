import shlex
from pathlib import Path

from cleanba.launcher import FlamingoRun, group_from_fname, launch_jobs

clis = []

a = shlex.split(
    """--from-py-fn=cleanba.config:sokoban_drc_3_3 loss.ent_coef=0.001 rmsprop_decay=0.999 max_grad_norm=0.0015 total_timesteps=51200000 rmsprop_eps=1.5625e-10 eval_frequency=625 net.residual=true net.recurrent.pool_projection="per-channel" net.recurrent.fence_pad="valid" net.recurrent.forget_bias=1.0 net.use_relu=false train_env.seed=438173579 seed=1692383821"""
)
a.append("load_path=/training/cleanba/041-more-tweaks-4/wandb/run-20240505_230928-uuz4mbp7/local-files/cp_28800000")
a.append("anneal_lr=true")
a.append("total_timesteps=256000000")
a.append("eval_frequency=3125")
clis.append(a)


a = shlex.split(
    """--from-py-fn=cleanba.config:sokoban_drc_3_3 loss.ent_coef=0.001 rmsprop_decay=0.999 max_grad_norm=0.00015 total_timesteps=51200000 rmsprop_eps=1.5625e-10 eval_frequency=625 net.residual=true net.recurrent.pool_projection="per-channel" net.recurrent.fence_pad="valid" net.recurrent.forget_bias=1.0 net.use_relu=false train_env.seed=869221144 seed=48729639"""
)
a.append("load_path=/training/cleanba/041-more-tweaks-4/wandb/run-20240505_230922-w9ptiwn0/local-files/cp_28800000")
a.append("anneal_lr=true")
a.append("total_timesteps=256000000")
a.append("eval_frequency=3125")
clis.append(a)

a = shlex.split(
    """--from-py-fn=cleanba.config:sokoban_drc_3_3 loss.ent_coef=0.001 rmsprop_decay=0.999 max_grad_norm=0.00015 total_timesteps=51200000 rmsprop_eps=1.5625e-10 eval_frequency=625 net.residual=true net.recurrent.pool_projection="per-channel" net.recurrent.fence_pad="valid" net.recurrent.forget_bias=1.0 net.use_relu=false train_env.seed=994894590 seed=884239727"""
)
a.append("load_path=/training/cleanba/041-more-tweaks-4/wandb/run-20240505_230921-ga0l65ji/local-files/cp_28800000")
a.append("anneal_lr=true")
a.append("total_timesteps=256000000")
a.append("eval_frequency=3125")
clis.append(a)


runs: list[FlamingoRun] = []
RUNS_PER_MACHINE = 1
for i in range(0, len(clis), RUNS_PER_MACHINE):
    runs.append(
        FlamingoRun(
            [["python", "-m", "cleanba.cleanba_impala", *clis[i + j]] for j in range(min(RUNS_PER_MACHINE, len(clis) - i))],
            CONTAINER_TAG="df13e85-main",
            CPU=6,
            MEMORY="20G",
            GPU=1,
            PRIORITY="high-batch",
            XLA_PYTHON_CLIENT_MEM_FRACTION='".95"',
            COMMIT_HASH="ac9b91935af801dfd558ede0869809486c5f0950",
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
