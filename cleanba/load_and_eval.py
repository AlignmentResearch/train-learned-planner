import dataclasses
import re
import sys
from functools import partial
from pathlib import Path
from typing import Iterable, List, Tuple

import farconf
import jax
from rich.pretty import pprint

from cleanba.cleanba_impala import WandbWriter, load_train_state
from cleanba.environments import EnvpoolBoxobanConfig
from cleanba.evaluate import EvalConfig


def default_eval_envs(CACHE_PATH=Path("/opt/sokoban_cache")) -> dict[str, EvalConfig]:
    steps_to_think = [0, 1, 2, 4, 6, 8, 10, 12, 16, 24, 32]
    envs = dict(
        test_unfiltered=EvalConfig(
            EnvpoolBoxobanConfig(
                seed=0,
                load_sequentially=True,
                max_episode_steps=120,
                min_episode_steps=120,
                num_envs=500,
                cache_path=CACHE_PATH,
                split=None,
                difficulty="hard",
                n_levels_to_load=1000,
            ),
            n_episode_multiple=2,
            steps_to_think=steps_to_think,
        ),
        valid_medium=EvalConfig(
            EnvpoolBoxobanConfig(
                seed=0,
                load_sequentially=True,
                max_episode_steps=120,
                min_episode_steps=120,
                num_envs=500,
                cache_path=CACHE_PATH,
                split="valid",
                difficulty="medium",
                n_levels_to_load=50_000,
            ),
            n_episode_multiple=100,
            steps_to_think=steps_to_think,
        ),
        hard=EvalConfig(
            EnvpoolBoxobanConfig(
                seed=0,
                load_sequentially=True,
                max_episode_steps=120,
                min_episode_steps=120,
                num_envs=119,
                cache_path=CACHE_PATH,
                split=None,
                difficulty="hard",
                n_levels_to_load=3332,
            ),
            n_episode_multiple=28,
            steps_to_think=steps_to_think,
        ),
    )
    for env in envs.values():
        assert env.env.num_envs * env.n_episode_multiple == env.env.n_levels_to_load
    return envs


@dataclasses.dataclass
class LoadAndEvalArgs:
    load_other_run: Path
    eval_envs: dict[str, EvalConfig] = dataclasses.field(default_factory=default_eval_envs)
    only_last_checkpoint: bool = False

    # for Writer
    base_run_dir: Path = Path("/training/cleanba")

    @property
    def total_timesteps(self) -> int:
        return 1


def default_load_and_eval() -> LoadAndEvalArgs:
    return LoadAndEvalArgs(Path("/path/to/nowhere"))


def recursive_find_checkpoint(root: Path) -> Iterable[Path]:
    if (root / "cfg.json").exists():
        yield root
    for x in root.iterdir():
        if x.is_dir():
            yield from recursive_find_checkpoint(root / x)


cp_expr = re.compile("^.*/cp_([0-9]+)$")


def load_and_eval(args: LoadAndEvalArgs):
    checkpoints_to_load: List[Tuple[int, Path]] = []
    for cp_candidate in recursive_find_checkpoint(args.load_other_run):
        match = cp_expr.match(str(cp_candidate))
        if match is None:
            print("Skipping (not matching)", cp_candidate)
        else:
            checkpoints_to_load.append((int(match.group(1)), cp_candidate))
    checkpoints_to_load.sort()

    assert len(set(cp_candidate.parent for _, cp_candidate in checkpoints_to_load)) == 1
    if args.only_last_checkpoint:
        checkpoints_to_load = checkpoints_to_load[-1:]
    print("Going to load from checkpoints: ", checkpoints_to_load)
    policy, _, cp_cfg, train_state, _ = load_train_state(checkpoints_to_load[0][1])
    get_action_fn = jax.jit(partial(policy.apply, method=policy.get_action), static_argnames="temperature")

    writer = WandbWriter(cp_cfg, wandb_cfg_extra_data={"load_other_run": str(args.load_other_run)})
    for cp_step, cp_path in checkpoints_to_load:
        _, _, _, train_state, _ = load_train_state(cp_path)
        print("Evaluating", cp_path)
        for eval_name, evaluator in args.eval_envs.items():
            log_dict = evaluator.run(policy, get_action_fn, train_state.params, key=jax.random.PRNGKey(1234))
            for k, v in log_dict.items():
                writer.add_scalar(f"{eval_name}/{k}", v, cp_step)


if __name__ == "__main__":
    args = farconf.parse_cli(sys.argv[1:], LoadAndEvalArgs)
    pprint(args)
    load_and_eval(args)
