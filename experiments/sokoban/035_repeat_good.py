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
for fence_pad in [True]:
    for env_seed, learn_seed in [(1007432561, 778268931), (random_seed(), random_seed()), (random_seed(), random_seed())]:

        def update_fn(config: Args) -> Args:
            config.local_num_envs = n_envs
            config.num_steps = 20
            config.net = dataclasses.replace(
                config.net,
                pool_and_inject=True,
                pool_and_inject_horizontal=True,
                fence_pad=fence_pad,
                skip_final=True,
                recurrent=dataclasses.replace(config.net.recurrent, initialization="lecun"),
            )

            world_size = 1
            len_actor_device_ids = 1
            num_actor_threads = 1
            global_step_multiplier = (
                config.num_steps * config.local_num_envs * num_actor_threads * len_actor_device_ids * world_size
            )
            config.total_timesteps = 80_117_760 * 5
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

            logit_l2_coef = 1.5625e-5
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
            config.max_grad_norm = 1.5e-2
            config.rmsprop_eps = 1.5625e-07
            config.optimizer_yang = False

            return config

        cli, _ = update_fns_to_cli(sokoban_drc_3_3, update_fn)
        print(shlex.join(cli))
        # Check that parsing doesn't error
        out = parse_cli(cli, Args)

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
