import dataclasses
import shlex
from pathlib import Path

from farconf import update_fns_to_cli

from cleanba.config import Args, sokoban_resnet
from cleanba.environments import BoxobanConfig, EnvpoolBoxobanConfig, random_seed
from cleanba.launcher import FlamingoRun, group_from_fname, launch_jobs
from cleanba.network import GuezResNetConfig, IdentityNorm

clis = []
# Paper says 200 actors, we'll use 192 so it's evenly divided by 32
for n_envs in [192, 256]:
    minibatch_size = 32
    assert n_envs % minibatch_size == 0
    for env_seed, learn_seed in [(random_seed(), random_seed()) for _ in range(2)]:
        for env_type in [
            BoxobanConfig(
                max_episode_steps=120,
                num_envs=1,
                seed=env_seed,
                tinyworld_obs=False,
                asynchronous=True,
                min_episode_steps=120 * 3 // 4,
            ),
            None,
        ]:
            for grad_norm_mul in [0.5, 1, 2]:
                if env_type is None and grad_norm_mul != 1:
                    continue

                def update_fn(config: Args) -> Args:
                    if env_type is None:
                        config.net = GuezResNetConfig(yang_init=False, norm=IdentityNorm(), normalize_input=False)
                    else:
                        config.net = GuezResNetConfig(
                            yang_init=False,
                            norm=IdentityNorm(),
                            channels=(32, 32, 64, 64, 64, 64, 64, 64, 64),
                            strides=(4, 2, 1, 1, 1, 1, 1, 1, 1),
                            kernel_sizes=(8, 4, 4, 4, 4, 4, 4, 4, 4),
                            normalize_input=False,
                        )
                        config.train_env = env_type

                        eval_envs = {}
                        for k, v in config.eval_envs.items():
                            assert isinstance(v.env, EnvpoolBoxobanConfig)
                            eval_envs[k] = dataclasses.replace(
                                v, env=dataclasses.replace(env_type, split=v.env.split, difficulty=v.env.difficulty)
                            )
                        config.eval_envs = eval_envs

                    config.local_num_envs = n_envs
                    config.num_steps = 20

                    world_size = 1
                    len_actor_device_ids = 1
                    num_actor_threads = 1
                    global_step_multiplier = (
                        config.num_steps * config.local_num_envs * num_actor_threads * len_actor_device_ids * world_size
                    )
                    # Approximately 8e7, but also divisible by (math.lcm(256, 192) * 20)
                    config.total_timesteps = 80_025_600
                    num_updates = config.total_timesteps // global_step_multiplier
                    assert (
                        num_updates * global_step_multiplier == config.total_timesteps
                    ), f"{config.total_timesteps=} != {num_updates=}*{global_step_multiplier=}"

                    # Evaluate (and save) EVAL_TIMES during training
                    EVAL_TIMES = 10
                    config.eval_frequency = num_updates // EVAL_TIMES
                    assert (
                        config.eval_frequency * EVAL_TIMES * global_step_multiplier > 8e7
                    ), "training and saving for insufficiently long"

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

                cli, _ = update_fns_to_cli(sokoban_resnet, update_fn)
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
            PRIORITY="normal-batch",
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
