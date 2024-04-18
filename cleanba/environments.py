import abc
import dataclasses
import os
import warnings
from functools import partial
from pathlib import Path
from typing import Any, Callable, ClassVar, List, Literal, Optional, Tuple, Union

import gym_sokoban  # noqa: F401
import gymnasium as gym
import numpy as np
from gymnasium.vector.utils.spaces import batch_space
from numpy.typing import NDArray

from cleanba.config import random_seed


@dataclasses.dataclass
class EnvConfig(abc.ABC):
    max_episode_steps: int
    num_envs: int = 0  # gets overwritten anyways
    seed: int = dataclasses.field(default_factory=random_seed)

    @property
    @abc.abstractmethod
    def make(self) -> Callable[[], gym.vector.VectorEnv]:
        ...


@dataclasses.dataclass
class EnvpoolEnvConfig(EnvConfig):
    env_id: str | None = None

    num_threads: int = 0
    thread_affinity_offset: int = -1
    max_num_players: int = 1

    @property
    def make(self) -> Callable[[], gym.vector.VectorEnv]:
        # Import envpool only when needed so we can run on Mac OS
        import envpool

        if self.env_id is None:
            raise ValueError("env_id is None, I don't know what kind of environment to build.")

        dummy_spec = envpool.make_spec(self.env_id)
        special_kwargs = dict(
            batch_size=self.num_envs,
        )
        SPECIAL_KEYS = {"base_path", "gym_reset_return_info"}
        env_kwargs = {}
        for k in dummy_spec._config_keys:
            if not (k in special_kwargs or k in SPECIAL_KEYS):
                try:
                    env_kwargs[k] = getattr(self, k)
                except AttributeError as e:
                    warnings.warn(f"Could not get environment setting: {e}")

        vec_envs_fn = partial(
            EnvpoolVectorEnv,
            self.num_envs,
            partial(envpool.make_gymnasium, self.env_id, **special_kwargs, **env_kwargs),
        )
        return vec_envs_fn


class EnvpoolVectorEnv(gym.vector.VectorEnv):
    def __init__(self, num_envs: int, envs_fn: Callable[[], Any]):
        envs = envs_fn()
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
    env_id: ClassVar[str] = "Sokoban-v0"

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
        assert self.env_id is not None

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

        self.num_envs = env.num_envs
        self.observation_space = batch_space(self.single_observation_space, n=self.num_envs)
        self.single_action_space = env.single_action_space
        self.action_space = env.action_space

    def reset_wait(self, **kwargs) -> tuple[Any, dict]:
        obs, info = super().reset_wait(**kwargs)
        return np.moveaxis(obs, 3, 1), info

    def step_wait(self) -> tuple[Any, NDArray, NDArray, NDArray, dict]:
        obs, reward, terminated, truncated, info = super().step_wait()
        return np.moveaxis(obs, 3, 1), reward, terminated, truncated, info

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
    def make(self) -> Callable[[], gym.vector.VectorEnv]:
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
    def make(self) -> Callable[[], gym.vector.VectorEnv]:
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


ATARI_MAX_FRAMES = int(
    108000 / 4
)  # 108000 is the max number of frames in an Atari game, divided by 4 to account for frame skipping
# This equals 27k, which is the default max_episode_steps for Atari in Envpool


@dataclasses.dataclass
class AtariEnv(EnvpoolEnvConfig):
    max_episode_steps: int = ATARI_MAX_FRAMES  # Hessel et al. 2018 (Rainbow DQN), Table 3, Max frames per episode
    episodic_life: bool = False  # Machado et al. 2017 (Revisitng ALE: Eval protocols) p. 6
    repeat_action_probability: float = 0.25  # Machado et al. 2017 (Revisitng ALE: Eval protocols) p. 12
    # Machado et al. 2017 (Revisitng ALE: Eval protocols) p. 12 (no-op is deprecated in favor of sticky action, right?)
    noop_max: int = 1
    full_action_space: bool = True  # Machado et al. 2017 (Revisitng ALE: Eval protocols) Tab. 5
    reward_clip: bool = True
