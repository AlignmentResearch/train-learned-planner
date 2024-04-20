import dataclasses
from pathlib import Path

from farconf import update_fns_to_cli

from cleanba.config import Args, sokoban_resnet
from cleanba.environments import random_seed
from cleanba.launcher import FlamingoRun, group_from_fname, launch_jobs
from cleanba.network import AtariCNNSpec, IdentityNorm, RMSNorm

runs: list[FlamingoRun] = []

yang = True
minibatch_size = 32
n_envs = 192  # the paper says 200 actors
assert n_envs % minibatch_size == 0
n_epochs = 4
mlp_layers = 1
seed = random_seed()

for lr in [0.0004]:
    for yang in [True, False]:

        def update_fn(config: Args) -> Args:
            config.eval_envs = {}

            config.local_num_envs = n_envs
            config.train_epochs = n_epochs
            config.num_actor_threads = 1
            config.num_steps = 20
            config.num_minibatches = n_envs // minibatch_size
            config.total_timesteps = int(40e6)
            config.seed = seed
            config.sync_frequency = 1000000000
            config.loss = dataclasses.replace(config.loss, vtrace_lambda=1.0, vf_coef=0.5, gamma=0.97, ent_coef=0.01)
            config.max_grad_norm = 1000.0
            config.adam_b1 = 0.9
            config.rmsprop_decay = 0.999
            config.rmsprop_eps = 1.5625e-07  # what's on the paper divided by 640
            config.learning_rate = lr
            config.base_fan_in = 72
            config.anneal_lr = False

            config.optimizer = "rmsprop"
            config.optimizer_yang = yang

            norm = RMSNorm() if yang else IdentityNorm()
            config.net = AtariCNNSpec(
                yang_init=yang, channels=(32, 32, 32), strides=(1, 1, 1), mlp_hiddens=(256,) * mlp_layers, norm=norm
            )
            return config

        cli, _ = update_fns_to_cli(sokoban_resnet, update_fn)
        print(cli)

        runs.append(
            FlamingoRun(
                [["python", "-m", "cleanba.cleanba_impala", *cli]],
                CONTAINER_TAG="e5a58c4-main",
                CPU=8,
                MEMORY="10G",
                GPU=1,
                PRIORITY="normal-batch",
            )
        )


GROUP: str = group_from_fname(__file__)

if __name__ == "__main__":
    launch_jobs(
        runs,
        group=GROUP,
        job_template_path=Path(__file__).parent.parent.parent / "k8s/runner-no-nfs.yaml",
        project="cleanba",
    )
