import dataclasses
import shlex
from pathlib import Path

from farconf import update_fns_to_cli

from cleanba.config import Args, sokoban_resnet
from cleanba.environments import random_seed
from cleanba.launcher import FlamingoRun, group_from_fname, launch_jobs
from cleanba.network import GuezResNetConfig, IdentityNorm, RMSNorm

minibatch_size = 32
n_envs = 256  # the paper says 200 actors
assert n_envs % minibatch_size == 0


logit_l2_coef = 1.5625e-6  # doesn't seem to matter much from now. May improve stability a tiny bit.
max_episode_steps = 120

world_size = 1
len_actor_device_ids = 1
num_actor_threads = 1

clis = []
train_epochs = 1

for norm in [IdentityNorm(), RMSNorm()]:
    for env_seed, learn_seed in [(random_seed(), random_seed()) for _ in range(2)]:
        for yang_init in [True, False]:
            for normalize_input in [True, False]:
                for actor_update_frequency in [3, 2, 1]:
                    for max_grad_norm_mul in [1, 2, 4]:
                        network = GuezResNetConfig(yang_init=yang_init, norm=norm, normalize_input=normalize_input)

                        def update_fn(config: Args) -> Args:
                            config.local_num_envs = n_envs
                            config.num_steps = 20
                            config.total_timesteps = 80_035_840

                            global_step_multiplier = (
                                config.num_steps
                                * config.local_num_envs
                                * num_actor_threads
                                * len_actor_device_ids
                                * world_size
                            )
                            assert config.total_timesteps % global_step_multiplier == 0
                            num_updates = config.total_timesteps // global_step_multiplier

                            config.eval_frequency = num_updates // 8

                            config.actor_update_frequency = actor_update_frequency
                            config.actor_update_cutoff = int(1e20)

                            config.train_epochs = train_epochs
                            config.num_actor_threads = 1
                            config.num_minibatches = (config.local_num_envs * config.num_actor_threads) // minibatch_size

                            config.seed = learn_seed
                            config.sync_frequency = int(1e20)
                            config.loss = dataclasses.replace(
                                config.loss,
                                vtrace_lambda=0.97,
                                vf_coef=0.25,
                                gamma=0.97,
                                ent_coef=0.01,
                                normalize_advantage=False,
                                logit_l2_coef=logit_l2_coef,
                                weight_l2_coef=logit_l2_coef / 100,
                            )
                            config.base_fan_in = 1
                            config.anneal_lr = True

                            config.optimizer = "adam"
                            config.adam_b1 = 0.9
                            config.rmsprop_decay = 0.999
                            config.learning_rate = 4e-4
                            config.max_grad_norm = 6.25e-5 * max_grad_norm_mul
                            config.rmsprop_eps = 1.5625e-07
                            config.optimizer_yang = False

                            config.net = network
                            return config

                        cli, _ = update_fns_to_cli(sokoban_resnet, update_fn)
                        print(shlex.join(cli))
                        clis.append(cli)

runs: list[FlamingoRun] = []
RUNS_PER_MACHINE = 1
for i in range(0, len(clis), RUNS_PER_MACHINE):
    runs.append(
        FlamingoRun(
            [["python", "-m", "cleanba.cleanba_impala", *clis[i + j]] for j in range(min(RUNS_PER_MACHINE, len(clis) - i))],
            CONTAINER_TAG="62d97f2-main",
            CPU=3,
            MEMORY="8G",
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
        job_template_path=Path(__file__).parent.parent.parent / "k8s/runner-no-nfs.yaml",
        project="cleanba",
    )
