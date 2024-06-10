import dataclasses
import shlex
from pathlib import Path

from farconf import parse_cli, update_fns_to_cli

from cleanba.config import Args, sokoban_drc_3_3
from cleanba.environments import random_seed
from cleanba.launcher import FlamingoRun, group_from_fname, launch_jobs
from cleanba.network import GuezResNetConfig, IdentityNorm

clis = []
drc_n_n = 3

num_envs = 256
gamma = 0.97

for min_episode_steps in [30]:
    for grad_norm in [1.5e-4, 2.5e-4]:
        for ent_coef in [0.01, 0.001]:
            for vtrace_lambda in [0.5, 0.97]:
                for env_seed, learn_seed in [(random_seed(), random_seed()) for _ in range(2)]:

                    def update_fn(config: Args) -> Args:
                        config.train_env = dataclasses.replace(
                            config.train_env, seed=env_seed, min_episode_steps=min_episode_steps
                        )
                        config.local_num_envs = num_envs
                        config.num_steps = 20
                        config.net = GuezResNetConfig(yang_init=False, norm=IdentityNorm(), normalize_input=False)

                        world_size = 1
                        len_actor_device_ids = 1
                        num_actor_threads = 1
                        global_step_multiplier = (
                            config.num_steps * config.local_num_envs * num_actor_threads * len_actor_device_ids * world_size
                        )

                        config.total_timesteps = 1_001_472_000
                        num_updates = config.total_timesteps // global_step_multiplier
                        assert (
                            num_updates * global_step_multiplier == config.total_timesteps
                        ), f"{config.total_timesteps=} != {num_updates=}*{global_step_multiplier=}"

                        # Evaluate (and save) EVAL_TIMES during training
                        config.eval_at_steps = frozenset(
                            [195 * i for i in range(1, 10)]
                            + [1953 * i for i in range(10)]
                            + [19531 * i for i in range(10)]
                            + [195312 * i for i in range(10)]
                        )

                        config.save_model = True
                        config.base_run_dir = Path("/training/cleanba")

                        config.actor_update_frequency = 1
                        config.actor_update_cutoff = int(1e20)  # disable

                        config.train_epochs = 1
                        config.num_actor_threads = 1
                        config.num_minibatches = config.local_num_envs // 32

                        config.seed = learn_seed
                        config.sync_frequency = int(1e20)

                        logit_l2_coef = 1.5625e-5
                        config.loss = dataclasses.replace(
                            config.loss,
                            vtrace_lambda=vtrace_lambda,
                            vf_coef=0.25,
                            gamma=gamma,
                            ent_coef=ent_coef,
                            normalize_advantage=False,
                            logit_l2_coef=logit_l2_coef,
                            weight_l2_coef=logit_l2_coef / 100,
                            vf_loss_type="square",
                            advantage_multiplier="one",
                        )
                        config.base_fan_in = 1
                        config.anneal_lr = True

                        config.optimizer = "adam"
                        config.adam_b1 = 0.9
                        config.rmsprop_decay = 0.999
                        config.learning_rate = 4e-4
                        config.max_grad_norm = 1.5e-4
                        config.rmsprop_eps = 1.5625e-10
                        config.optimizer_yang = False

                        config.eval_envs["valid_medium"].steps_to_think = [0, 2, 4, 8, 12, 16, 24, 32]

                        return config

                    cli, _ = update_fns_to_cli(sokoban_drc_3_3, update_fn)
                    print(shlex.join(cli))
                    # Check that parsing doesn't error
                    out = parse_cli(cli, Args)

                    clis.append((cli, "normal-batch"))

runs: list[FlamingoRun] = []
RUNS_PER_MACHINE = 1
for i in range(0, len(clis), RUNS_PER_MACHINE):
    this_run_clis = [
        ["python", "-m", "cleanba.cleanba_impala", *clis[i + j][0]] for j in range(min(RUNS_PER_MACHINE, len(clis) - i))
    ]
    priority = clis[i][1]
    print("With priority", priority)
    runs.append(
        FlamingoRun(
            this_run_clis,
            CONTAINER_TAG="5d26ad1-main",
            CPU=6,
            MEMORY="20G",
            GPU=1,
            PRIORITY=priority,
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
