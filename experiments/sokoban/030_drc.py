import dataclasses
import shlex
from pathlib import Path

from farconf import parse_cli, update_fns_to_cli

from cleanba.config import Args, sokoban_drc_3_3
from cleanba.environments import random_seed
from cleanba.launcher import FlamingoRun, group_from_fname, launch_jobs

clis = []
# Paper says 200 actors, we'll use 192 so it's evenly divided by 32
n_envs = 256
minibatch_size = 32
assert n_envs % minibatch_size == 0
for env_seed, learn_seed in [(random_seed(), random_seed()) for _ in range(2)]:
    for base_fn in [sokoban_drc_3_3]:

        def update_fn(config: Args) -> Args:
            config.local_num_envs = n_envs
            config.num_steps = 20

            world_size = 1
            len_actor_device_ids = 1
            num_actor_threads = 1
            global_step_multiplier = (
                config.num_steps * config.local_num_envs * num_actor_threads * len_actor_device_ids * world_size
            )
            config.total_timesteps = 80_117_760
            num_updates = config.total_timesteps // global_step_multiplier
            assert (
                num_updates * global_step_multiplier == config.total_timesteps
            ), f"{config.total_timesteps=} != {num_updates=}*{global_step_multiplier=}"

            # Evaluate (and save) EVAL_TIMES during training
            EVAL_TIMES = 16
            config.eval_frequency = num_updates // EVAL_TIMES

            config.save_model = True
            config.base_run_dir = Path("/training/cleanba")

            config.actor_update_frequency = 1
            config.actor_update_cutoff = int(1e20)  # disable

            config.train_epochs = 1
            config.num_actor_threads = 1
            config.num_minibatches = (config.local_num_envs * config.num_actor_threads) // minibatch_size

            config.seed = learn_seed
            config.sync_frequency = int(1e20)

            logit_l2_coef = 1.5625e-6  # doesn't seem to matter much from now. May improve stability a tiny bit.
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
            config.rmsprop_decay = 0.99
            config.learning_rate = 4e-4
            config.max_grad_norm = 2.5e-4
            config.rmsprop_eps = 1.5625e-07
            config.optimizer_yang = False

            return config

        cli, _ = update_fns_to_cli(base_fn, update_fn)
        # Check that parsing doesn't error
        _ = parse_cli(cli, Args)

        print(shlex.join(cli))
        clis.append(cli)

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
