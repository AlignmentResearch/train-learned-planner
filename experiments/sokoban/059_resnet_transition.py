import dataclasses
import shlex
from pathlib import Path

from farconf import parse_cli, update_fns_to_cli

from cleanba.config import Args, sokoban_drc_3_3
from cleanba.launcher import FlamingoRun, group_from_fname, launch_jobs
from cleanba.network import GuezResNetConfig, IdentityNorm

clis = []
all_args: list[Args] = []
drc_n_n = 3

num_envs = 256
gamma = 0.97
min_episode_steps = 30
vtrace_lambda = 0.5

update_fns_to_go_back = [
    lambda cfg: dataclasses.replace(cfg, rmsprop_decay=0.99),
    lambda cfg: dataclasses.replace(cfg, rmsprop_eps=1.5625e-07),
    lambda cfg: dataclasses.replace(cfg, max_grad_norm=0.00025),
    lambda cfg: dataclasses.replace(
        cfg, loss=dataclasses.replace(cfg.loss, logit_l2_coef=1.5625e-06, weight_l2_coef=1.5625e-08)
    ),
    lambda cfg: dataclasses.replace(cfg, loss=dataclasses.replace(cfg.loss, ent_coef=0.01)),
]

for i in range(len(update_fns_to_go_back) + 1):
    for env_seed, learn_seed in [(1234, 4242) for _ in range(2)]:

        def update_fn(config: Args) -> Args:
            config.train_env = dataclasses.replace(config.train_env, seed=env_seed, min_episode_steps=min_episode_steps)
            config.local_num_envs = num_envs
            config.num_steps = 20
            config.net = GuezResNetConfig(yang_init=False, norm=IdentityNorm(), normalize_input=False)

            world_size = 1
            len_actor_device_ids = 1
            num_actor_threads = 1
            global_step_multiplier = (
                config.num_steps * config.local_num_envs * num_actor_threads * len_actor_device_ids * world_size
            )

            config.total_timesteps = 256_000_000
            num_updates = config.total_timesteps // global_step_multiplier
            assert (
                num_updates * global_step_multiplier == config.total_timesteps
            ), f"{config.total_timesteps=} != {num_updates=}*{global_step_multiplier=}"

            # Evaluate (and save) EVAL_TIMES during training
            config.eval_at_steps = frozenset([0, 7810, 780, 156, 1562, 9372])

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
                ent_coef=0.001,
                normalize_advantage=False,
                logit_l2_coef=logit_l2_coef,
                weight_l2_coef=logit_l2_coef / 100,
                vf_loss_type="square",
                advantage_multiplier="one",
            )
            config.base_fan_in = 1
            config.anneal_lr = False

            config.optimizer = "adam"
            config.adam_b1 = 0.9
            config.rmsprop_decay = 0.999
            config.learning_rate = 4e-4
            config.max_grad_norm = 1.5e-4
            config.rmsprop_eps = 1.5625e-10
            config.optimizer_yang = False

            config.eval_envs["valid_medium"].steps_to_think = [0, 2, 4, 8, 12, 16, 24, 32]

            return config

        cli, _ = update_fns_to_cli(sokoban_drc_3_3, update_fn, *update_fns_to_go_back[:i])

        print(shlex.join(cli))
        # Check that parsing doesn't error
        out = parse_cli(cli, Args)

        all_args.append(out)
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
