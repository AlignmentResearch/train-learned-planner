import shlex
from pathlib import Path

from cleanba.launcher import FlamingoRun, group_from_fname, launch_jobs

clis = []

a = shlex.split(
    """--from-py-fn=cleanba.config:sokoban_drc_3_3 total_timesteps=51200000 loss.ent_coef=0.001 train_env.seed=1551346042 eval_frequency=625 net.residual=true net.recurrent.fence_pad="valid" net.recurrent.forget_bias=1.0 net.recurrent.pool_projection="per-channel" net.use_relu=false rmsprop_decay=0.999 rmsprop_eps=1.5625e-10 max_grad_norm=0.00015 anneal_lr=false eval_envs={} seed=1420836954"""
)
a.append("load_path=/training/cleanba/040-residual-4/wandb/run-20240505_204952-sc20kyss/local-files/cp_51200000")
clis.append(a)


a = shlex.split(
    """--from-py-fn=cleanba.config:sokoban_drc_3_3 total_timesteps=51200000 loss.ent_coef=0.001 train_env.seed=1498572294 eval_frequency=625 net.residual=true net.recurrent.fence_pad="valid" net.recurrent.forget_bias=1.0 net.recurrent.pool_projection="per-channel" net.use_relu=false rmsprop_decay=0.999 rmsprop_eps=1.5625e-10 max_grad_norm=0.00015 anneal_lr=false eval_envs={} seed=605986226"""
)
a.append("load_path=/training/cleanba/040-residual-4/wandb/run-20240505_204954-5zp4ieok/local-files/cp_51200000")
clis.append(a)

a = shlex.split(
    """--from-py-fn=cleanba.config:sokoban_drc_3_3 total_timesteps=51200000 loss.ent_coef=0.001 train_env.seed=768284418 eval_frequency=625 net.residual=true net.recurrent.fence_pad="valid" net.recurrent.forget_bias=1.0 net.recurrent.pool_projection="per-channel" net.use_relu=false rmsprop_decay=0.999 rmsprop_eps=1.5625e-10 max_grad_norm=0.00015 anneal_lr=false eval_envs={} seed=1708301081"""
)
a.append("load_path=/training/cleanba/040-residual-4/wandb/run-20240505_204952-drm6t5l8/local-files/cp_51200000")
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
            PRIORITY="normal-batch",
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
