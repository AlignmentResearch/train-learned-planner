import dataclasses
import shlex
from pathlib import Path

from farconf import parse_cli, update_fns_to_cli

from cleanba.config import Args, sokoban_drc33_59
from cleanba.environments import BoxobanConfig, random_seed
from cleanba.evaluate import EvalConfig
from cleanba.launcher import FlamingoRun, group_from_fname, launch_jobs

clis: list[list[str]] = []
all_args: list[Args] = []


"""
max_grad_norm: 0.00015 0.00025
rmsprop_decay: 0.999 0.99
rmsprop_eps: 1.5625e7 1.5625e10
ent_coef: 0.001 0.01
vtrace_lambda: 0.97 0.5
weight_l2_coef: 1.5625e7 1.5625e8
logit_l2_coef: 1.5625e5 1.5625e6
residual: true false
output_activation: "sigmoid" "tanh"
"""

update_fns_to_go_back = [
    lambda cfg: dataclasses.replace(cfg, loss=dataclasses.replace(cfg.loss, ent_coef=0.001)),
    lambda cfg: dataclasses.replace(cfg, max_grad_norm=0.00015),
    lambda cfg: dataclasses.replace(cfg, rmsprop_decay=0.999),
    lambda cfg: dataclasses.replace(cfg, rmsprop_eps=1.5625e-07),
    lambda cfg: dataclasses.replace(
        cfg, loss=dataclasses.replace(cfg.loss, logit_l2_coef=1.5625e-05, weight_l2_coef=1.5625e-07)
    ),
]

level_idxs_path = Path(__file__).absolute().parent.parent.parent / "val_medium_level_idxs.npy"
assert level_idxs_path.exists()


for env_seed, learn_seed in [(random_seed(), random_seed()) for _ in range(3)]:
    for output_activation in ["sigmoid", "tanh"]:
        for residual in [True, False]:
            for vtrace_lambda in [0.97, 0.5]:
                for update_fns_i in range(len(update_fns_to_go_back) + 1):

                    def update_seeds(config: Args) -> Args:
                        config.train_env = dataclasses.replace(config.train_env, seed=env_seed)
                        config.seed = learn_seed

                        config.loss = dataclasses.replace(
                            config.loss,
                            vtrace_lambda=vtrace_lambda,
                        )
                        config.net = dataclasses.replace(
                            config.net,
                            recurrent=dataclasses.replace(
                                config.net.recurrent,  # type: ignore
                                output_activation=output_activation,
                            ),
                            residual=residual,
                        )
                        config.total_timesteps = 200_294_400
                        config.learning_rate = 4e-4
                        config.final_learning_rate = 3.604e-4
                        config.anneal_lr = True

                        CACHE_PATH = Path("/opt/sokoban_cache")
                        config.eval_envs = {
                            "valid_medium": EvalConfig(
                                n_episode_multiple=20,
                                env=BoxobanConfig(
                                    max_episode_steps=120,
                                    min_episode_steps=120,
                                    num_envs=256,
                                    cache_path=CACHE_PATH,
                                    tinyworld_obs=True,
                                    tinyworld_render=True,
                                    split="valid",
                                    difficulty="medium",
                                    # level_idxs_path=level_idxs_path,
                                    seed=42,
                                ),
                                steps_to_think=[0, 2, 4, 8, 12, 16, 24, 32],
                            )
                        }

                        return config

                    cli, _ = update_fns_to_cli(sokoban_drc33_59, update_seeds, *update_fns_to_go_back[:update_fns_i])

                    print(shlex.join(cli))
                    # Check that parsing doesn't error
                    out = parse_cli(cli, Args)

                    all_args.append(out)
                    clis.append(cli)

runs: list[FlamingoRun] = []
RUNS_PER_MACHINE = 1
for update_fns_i in range(0, len(clis), RUNS_PER_MACHINE):
    this_run_clis = [
        ["python", "-m", "cleanba.cleanba_impala", *clis[update_fns_i + j]]
        for j in range(min(RUNS_PER_MACHINE, len(clis) - update_fns_i))
    ]
    runs.append(
        FlamingoRun(
            this_run_clis,
            CONTAINER_TAG="5d26ad1-main",
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
