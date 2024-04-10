import abc
import dataclasses
import os
from functools import partial
from pathlib import Path
from typing import Any, Callable, List, Literal, Optional, Tuple, Union

import gym_sokoban  # noqa: F401
import gymnasium as gym
import numpy as np
from gymnasium.vector.utils.spaces import batch_space
from numpy.typing import NDArray

from cleanba.config import random_seed


@dataclasses.dataclass
class EnvConfig(abc.ABC):
    max_episode_steps: int
    num_envs: int
    seed: int = dataclasses.field(default_factory=random_seed)

    @abc.abstractmethod
    def make(self) -> gym.vector.VectorEnv:
        ...


class EnvpoolEnvConfig(EnvConfig):
    num_threads: int = 0
    thread_affinity_offset: int = -1
    max_num_players: int = 1


class EnvpoolVectorEnv(gym.vector.VectorEnv):
    def __init__(self, num_envs: int, envs):
        super().__init__(num_envs=num_envs, observation_space=envs.observation_space, action_space=envs.action_space)
        self.envs = envs

    def step_async(self, actions: np.ndarray):
        self.envs.send(actions)

    def step_wait(self, **kwargs) -> Tuple[Any, NDArray[Any], NDArray[Any], NDArray[Any], dict]:
        return self.envs.recv(**kwargs)

    def reset_async(self, seed: Optional[Union[int, List[int]]] = None, options: Optional[dict] = None):
        assert seed is None
        assert not options
        self.envs.async_reset()

    def reset_wait(self, seed: Optional[Union[int, List[int]]] = None, options: Optional[dict] = None):
        assert seed is None
        assert not options
        return self.envs.recv(reset=True, return_info=self.envs.config["gym_reset_return_info"])


@dataclasses.dataclass
class EnvpoolBoxobanConfig(EnvpoolEnvConfig):
    reward_finished: float = 10.0  # Reward for completing a level
    reward_box: float = 1.0  # Reward for putting a box on target
    reward_step: float = -0.1  # Reward for completing a step
    verbose: int = 0  # Verbosity level [0-2]
    min_episode_steps: int = 0  # The minimum length of an episode.
    load_sequentially: bool = False
    n_levels_to_load: int = -1  # -1 means "all levels". Used only when `load_sequentially` is True.

    # Not present in _SokobanEnvSpec
    cache_path: Path = Path(__file__).parent.parent / ".sokoban_cache"
    split: Literal["train", "valid", "test", None] = "train"
    difficulty: Literal["unfiltered", "medium", "hard"] = "unfiltered"

    def __post_init__(self):
        if self.difficulty == "hard":
            assert self.split is None
        else:
            assert self.split is not None
        assert self.min_episode_steps >= 0
        assert self.min_episode_steps <= self.max_episode_steps, f"{self.min_episode_steps=} {self.max_episode_steps=}"
        if not self.load_sequentially:
            assert self.n_levels_to_load == -1, "`n_levels_to_load` must be -1 when `load_sequentially` is False"

    @property
    def dim_room(self) -> int:
        return 10

    @property
    def levels_dir(self) -> str:
        levels_dir = self.cache_path / "boxoban-levels-master" / self.difficulty
        if self.difficulty == "hard":
            assert self.split is None
        else:
            assert self.split is not None
            levels_dir = levels_dir / self.split

        not_end_txt = [s for s in os.listdir(levels_dir) if not (levels_dir / s).is_dir() and not s.endswith(".txt")]
        if len(not_end_txt) > 0:
            raise ValueError(f"{levels_dir=} does not exist or some of its files don't end in .txt: {not_end_txt}")
        return str(levels_dir)

    def make(self) -> gym.vector.VectorEnv:
        # Import envpool only when needed so we can run on Mac OS
        import envpool

        env_id: str = "Sokoban-v0"
        dummy_spec = envpool.make_spec(env_id)
        special_kwargs = dict(
            batch_size=self.num_envs,
        )
        SPECIAL_KEYS = {"base_path", "gym_reset_return_info"}
        env_kwargs = {k: getattr(self, k) for k in dummy_spec._config_keys if not (k in special_kwargs or k in SPECIAL_KEYS)}

        envs = envpool.make_gymnasium("Sokoban-v0", **special_kwargs, **env_kwargs)
        vec_envs = EnvpoolVectorEnv(self.num_envs, envs)
        return vec_envs


@dataclasses.dataclass
class BaseSokobanEnvConfig(EnvConfig):
    tinyworld_obs: bool = False
    tinyworld_render: bool = False
    terminate_on_first_box: bool = False

    reward_finished: float = 10.0  # Reward for completing a level
    reward_box: float = 1.0  # Reward for putting a box on target
    reward_step: float = -0.1  # Reward for completing a step

    reset: bool = False
    asynchronous: bool = True

    def env_kwargs(self) -> dict[str, Any]:
        return dict(
            num_envs=self.num_envs,
            tinyworld_obs=self.tinyworld_obs,
            tinyworld_render=self.tinyworld_render,
            # Sokoban env uses `max_steps` internally
            max_steps=self.max_episode_steps,
            # Passing `max_episode_steps` to Gymnasium makes it add a TimeLimitWrapper
            terminate_on_first_box=self.terminate_on_first_box,
            reset_seed=self.seed,
            reset=self.reset,
            asynchronous=self.asynchronous,
        )

    def env_reward_kwargs(self):
        return dict(
            reward_finished=self.reward_finished,
            reward_box_on_target=self.reward_box,
            penalty_box_off_target=-self.reward_box,
            penalty_for_step=self.reward_step,
        )

    @property
    @abc.abstractmethod
    def make(self) -> Callable[[], gym.vector.VectorEnv]:
        ...


class VectorNHWCtoNCHWWrapper(gym.vector.VectorEnvWrapper):
    def __init__(self, env: gym.vector.VectorEnv):
        super().__init__(env)
        obs_space = env.single_observation_space
        if isinstance(obs_space, gym.spaces.Box):
            shape = (obs_space.shape[2], *obs_space.shape[:2], *obs_space.shape[3:])
            low = obs_space.low if isinstance(obs_space.low, float) else np.moveaxis(obs_space.low, 2, 0)
            high = obs_space.high if isinstance(obs_space.high, float) else np.moveaxis(obs_space.high, 2, 0)
            self.single_observation_space = gym.spaces.Box(low, high, shape)
        else:
            raise NotImplementedError(f"{type(obs_space)=}")

        self.observation_space = batch_space(self.single_observation_space, n=self.num_envs)

    def reset_wait(self, **kwargs):
        obs, info = super().reset_wait(**kwargs)
        return np.moveaxis(obs, 3, 1), info

    def step_wait(self):
        obs, *others = super().step_wait()
        return (np.moveaxis(obs, 3, 1), *others)

    @classmethod
    def from_fn(cls, fn: Callable[[], gym.vector.VectorEnv]) -> gym.vector.VectorEnv:
        return cls(fn())


@dataclasses.dataclass
class SokobanConfig(BaseSokobanEnvConfig):
    "Procedurally-generated Sokoban"

    dim_room: tuple[int, int] | None = None
    num_boxes: int = 4
    num_gen_steps: int | None = None

    @property
    def make(self) -> Callable[[], gym.Env]:
        kwargs = self.env_kwargs()
        for k in ["dim_room", "num_boxes", "num_gen_steps"]:
            if (a := getattr(self, k)) is not None:
                kwargs[k] = a
        make_fn = partial(
            VectorNHWCtoNCHWWrapper.from_fn,
            partial(
                # We use `gym.vector.make` even though it is deprecated because it gives us `gym.vector.SyncVectorEnv`
                # instead of `gym.experimental.vector.SyncVectorEnv`.
                gym.vector.make,
                "Sokoban-v2",
                **kwargs,
                **self.env_reward_kwargs(),
            ),
        )
        return make_fn


@dataclasses.dataclass
class BoxobanConfig(BaseSokobanEnvConfig):
    "Sokoban levels from the Boxoban data set"

    cache_path: Path = Path(__file__).parent.parent / ".sokoban_cache"
    split: Literal["train", "valid", "test", None] = "train"
    difficulty: Literal["unfiltered", "medium", "hard"] = "unfiltered"

    @property
    def make(self) -> Callable[[], gym.Env]:
        if self.difficulty == "hard":
            if self.split is not None:
                raise ValueError("`hard` levels have no splits")
        elif self.difficulty == "medium":
            if self.split == "test":
                raise ValueError("`medium` levels don't have a `test` split")

        make_fn = partial(
            VectorNHWCtoNCHWWrapper.from_fn,
            partial(
                # We use `gym.vector.make` even though it is deprecated because it gives us `gym.vector.SyncVectorEnv`
                # instead of `gym.experimental.vector.SyncVectorEnv`.
                #
                # This makes it so we can use the gym.vector.Wrapper above
                gym.vector.make,
                "Boxoban-Val-v1",
                cache_path=self.cache_path,
                split=self.split,
                difficulty=self.difficulty,
                **self.env_kwargs(),
                **self.env_reward_kwargs(),
            ),
        )
        return make_fn
