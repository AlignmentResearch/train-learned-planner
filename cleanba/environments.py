import abc
import dataclasses
import os
import random
import warnings
from functools import partial
from pathlib import Path
from typing import Any, Callable, List, Literal, Optional, Tuple, Union

import gym_sokoban  # noqa: F401
import gymnasium as gym
import jax.numpy as jnp
import numpy as np
from gymnasium.envs.registration import register
from gymnasium.vector.utils.spaces import batch_space
from numpy.typing import NDArray

register(
    id="BoxWorld-v0",
    entry_point="cleanba.envs:BoxWorld",
)
register(
    id="MiniPacMan-v0",
    entry_point="cleanba.envs:MiniPacMan",
)


def random_seed() -> int:
    return random.randint(0, 2**31 - 2)


@dataclasses.dataclass
class EnvConfig(abc.ABC):
    max_episode_steps: int
    num_envs: int = 0  # gets overwritten anyways
    seed: int = dataclasses.field(default_factory=random_seed)

    @property
    @abc.abstractmethod
    def make(self) -> Callable[[], gym.vector.VectorEnv]:
        """Create a vector environment."""
        ...


@dataclasses.dataclass
class EnvpoolEnvConfig(EnvConfig):
    env_id: str | None = None

    num_threads: int = 0
    thread_affinity_offset: int = -1
    max_num_players: int = 1
    extra_env_kwargs: dict[str, Any] = dataclasses.field(default_factory=dict)

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
        env_kwargs = self.extra_env_kwargs.copy()
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
            remove_last_action=getattr(self, "nn_without_noop", False),
        )
        return vec_envs_fn


class EnvpoolVectorEnv(gym.vector.VectorEnv):
    def __init__(self, num_envs: int, envs_fn: Callable[[], Any], remove_last_action: bool = False):
        envs = envs_fn()
        if remove_last_action:
            envs.action_space.n -= 1  # can't set envs.action_space directly as it has no setter
        super().__init__(num_envs=num_envs, observation_space=envs.observation_space, action_space=envs.action_space)
        self.envs = envs

    def step(self, actions: np.ndarray) -> Tuple[Any, NDArray[Any], NDArray[Any], NDArray[Any], dict]:
        """Execute one step in the environment."""
        self.envs.send(np.array(actions))
        return self.envs.recv()

    def reset(self, seed: Optional[Union[int, List[int]]] = None, options: Optional[dict] = None) -> Tuple[Any, dict]:
        """Reset the environment."""
        assert seed is None
        assert not options
        self.envs.async_reset()
        return self.envs.recv(reset=True, return_info=self.envs.config["gym_reset_return_info"])


@dataclasses.dataclass
class EnvpoolBoxobanConfig(EnvpoolEnvConfig):
    env_id: str = "Sokoban-v0"

    reward_finished: float = 10.0  # Reward for completing a level
    reward_box: float = 1.0  # Reward for putting a box on target
    reward_step: float = -0.1  # Reward for completing a step
    reward_noop: float = 0.0  # Addtional Reward for doing nothing
    verbose: int = 0  # Verbosity level [0-2]
    min_episode_steps: int = 0  # The minimum length of an episode.
    load_sequentially: bool = False
    n_levels_to_load: int = -1  # -1 means "all levels". Used only when `load_sequentially` is True.
    nn_without_noop: bool = True  # Use a neural network without the noop action

    # Not present in _SokobanEnvSpec
    cache_path: Path = Path("/opt/sokoban_cache")
    split: Literal["train", "valid", "test", None] = "train"
    difficulty: Literal["unfiltered", "medium", "hard"] = "unfiltered"

    def __post_init__(self):
        assert self.env_id == "Sokoban-v0"

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
    min_episode_steps: int = 60
    tinyworld_obs: bool = False
    tinyworld_render: bool = False
    render_mode: str = "rgb_8x8"  # can be "rgb_array" or "rgb_8x8"
    terminate_on_first_box: bool = False

    reward_finished: float = 10.0  # Reward for completing a level
    reward_box: float = 1.0  # Reward for putting a box on target
    reward_step: float = -0.1  # Reward for completing a step
    reward_noop: float = 0.0  # Addtional reward for doing nothing
    nn_without_noop: bool = True  # Use a neural network without the noop action

    reset: bool = False
    asynchronous: bool = True
    dim_room: tuple[int, int] = (10, 10)

    def env_kwargs(self) -> dict[str, Any]:
        return dict(
            num_envs=self.num_envs,
            tinyworld_obs=self.tinyworld_obs,
            tinyworld_render=self.tinyworld_render,
            render_mode=self.render_mode,
            # Sokoban env uses `max_steps` internally
            max_steps=self.max_episode_steps,
            # Passing `max_episode_steps` to Gymnasium makes it add a TimeLimitWrapper
            terminate_on_first_box=self.terminate_on_first_box,
            reset_seed=self.seed,
            reset=self.reset,
            asynchronous=self.asynchronous,
            min_episode_steps=self.min_episode_steps,
            dim_room=self.dim_room,
        )

    def env_reward_kwargs(self):
        return dict(
            reward_finished=self.reward_finished,
            reward_box_on_target=self.reward_box,
            penalty_box_off_target=-self.reward_box,
            penalty_for_step=self.reward_step,
            reward_noop=self.reward_noop,
        )


class VectorNHWCtoNCHWWrapper(gym.vector.VectorEnvWrapper):
    def __init__(self, env: gym.vector.VectorEnv, nn_without_noop: bool = False, use_np_arrays: bool = False):
        super().__init__(env)
        self.use_np_arrays = use_np_arrays
        obs_space = env.single_observation_space
        if isinstance(obs_space, gym.spaces.Box):
            shape = (obs_space.shape[2], *obs_space.shape[:2], *obs_space.shape[3:])
            low = obs_space.low if isinstance(obs_space.low, float) else np.moveaxis(obs_space.low, 2, 0)
            high = obs_space.high if isinstance(obs_space.high, float) else np.moveaxis(obs_space.high, 2, 0)
            self.single_observation_space = gym.spaces.Box(low, high, shape, dtype=obs_space.dtype)
        else:
            raise NotImplementedError(f"{type(obs_space)=}")

        self.num_envs = env.num_envs
        self.observation_space = batch_space(self.single_observation_space, n=self.num_envs)

        if nn_without_noop:
            assert isinstance(env.single_action_space, gym.spaces.Discrete)
            env.single_action_space = gym.spaces.Discrete(env.single_action_space.n - 1)
            env.action_space = batch_space(env.single_action_space, n=self.num_envs)
        self.single_action_space = env.single_action_space
        self.action_space = env.action_space

    def reset(self, **kwargs) -> tuple[Any, dict]:
        obs, info = super().reset(**kwargs)
        return jnp.moveaxis(obs, 3, 1), info

    def step(self, actions: jnp.ndarray) -> tuple[Any, jnp.ndarray, jnp.ndarray, jnp.ndarray, dict]:
        if self.use_np_arrays:
            actions = np.asarray(actions)
        obs, reward, terminated, truncated, info = super().step(actions)
        return jnp.moveaxis(obs, 3, 1), reward, terminated, truncated, info

    @classmethod
    def from_fn(
        cls, fn: Callable[[], gym.vector.VectorEnv], nn_without_noop: bool, use_np_arrays: bool
    ) -> gym.vector.VectorEnv:
        return cls(fn(), nn_without_noop=nn_without_noop, use_np_arrays=use_np_arrays)


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
            self.nn_without_noop,
            use_np_arrays=True,
        )
        return make_fn


@dataclasses.dataclass
class BoxobanConfig(BaseSokobanEnvConfig):
    "Sokoban levels from the Boxoban data set"

    cache_path: Path = Path("/opt/sokoban_cache")
    split: Literal["train", "valid", "test", "planning", None] = "train"
    difficulty: Literal["unfiltered", "medium", "hard"] = "unfiltered"
    level_idxs_path: Path | None = None

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
            self.nn_without_noop,
            use_np_arrays=True,  # TODO: use the XLA interface for envpool and set this to false
        )
        return make_fn


@dataclasses.dataclass
class BoxWorldConfig(EnvConfig):
    dim_room: int = 12
    goal_length: int = 4
    num_distractor: int = 1
    distractor_length: int = 3
    step_cost: float = 0.1
    reward_gem: float = 10.0
    reward_key: float = 1.0
    reward_distractor: float = -1.0
    max_episode_steps: int = 120
    collect_key: bool = True
    nn_without_noop: bool = True
    asynchronous: bool = True

    @property
    def make(self) -> Callable[[], gym.vector.VectorEnv]:
        make_fn = partial(
            VectorNHWCtoNCHWWrapper.from_fn,
            partial(
                gym.vector.make,
                "BoxWorld-v0",
                n=self.dim_room,
                goal_length=self.goal_length,
                num_distractor=self.num_distractor,
                distractor_length=self.distractor_length,
                step_cost=self.step_cost,
                reward_gem=self.reward_gem,
                reward_key=self.reward_key,
                reward_distractor=self.reward_distractor,
                max_steps=self.max_episode_steps,
                collect_key=self.collect_key,
                asynchronous=self.asynchronous,
                num_envs=self.num_envs,
            ),
            self.nn_without_noop,
        )
        return make_fn


ATARI_MAX_FRAMES = int(
    108000 / 4
)  # 108000 is the max number of frames in an Atari game, divided by 4 to account for frame skipping
# This equals 27k, which is the default max_episode_steps for Atari in Envpool


@dataclasses.dataclass
class MiniPacManConfig(EnvConfig):
    env_id: str = "MiniPacMan-v0"
    mode: str = "regular"
    npills: int = 2
    pill_duration: int = 20
    stochasticity: float = 0.05
    nghosts_init: int = 1
    ghost_speed_init: float = 0.5
    ghost_speed_increase: float = 0.1
    max_episode_steps: int = 1000
    nn_without_noop: bool = True
    asynchronous: bool = True

    @property
    def make(self) -> Callable[[], gym.vector.VectorEnv]:
        make_fn = partial(
            VectorNHWCtoNCHWWrapper.from_fn,
            partial(
                gym.vector.make,
                self.env_id,
                mode=self.mode,
                npills=self.npills,
                pill_duration=self.pill_duration,
                stochasticity=self.stochasticity,
                nghosts_init=self.nghosts_init,
                ghost_speed_init=self.ghost_speed_init,
                ghost_speed_increase=self.ghost_speed_increase,
                frame_cap=self.max_episode_steps,
                num_envs=self.num_envs,
                asynchronous=self.asynchronous,
            ),
            self.nn_without_noop,
        )
        return make_fn


@dataclasses.dataclass
class AtariEnv(EnvpoolEnvConfig):
    max_episode_steps: int = ATARI_MAX_FRAMES  # Hessel et al. 2018 (Rainbow DQN), Table 3, Max frames per episode
    episodic_life: bool = False  # Machado et al. 2017 (Revisitng ALE: Eval protocols) p. 6
    repeat_action_probability: float = 0.25  # Machado et al. 2017 (Revisitng ALE: Eval protocols) p. 12
    # Machado et al. 2017 (Revisitng ALE: Eval protocols) p. 12 (no-op is deprecated in favor of sticky action, right?)
    noop_max: int = 1
    full_action_space: bool = True  # Machado et al. 2017 (Revisitng ALE: Eval protocols) Tab. 5
    reward_clip: bool = True


def convert_to_cleanba_config(env_config, asynchronous=False):
    """Converts an environment config from the learned_planner package to a cleanba environment config."""
    if isinstance(env_config, EnvConfig):
        return env_config
    env_classes_map = dict(
        EnvpoolSokobanVecEnvConfig=EnvpoolBoxobanConfig,
        BoxobanConfig=BoxobanConfig,
        SokobanConfig=SokobanConfig,
    )
    cls_name = env_config.__class__.__name__
    assert cls_name in env_classes_map, f"{cls_name=} not available in cleanba.environments"
    args = dataclasses.asdict(env_config)
    args["num_envs"] = args.pop("n_envs")
    args.pop("n_envs_to_render", None)
    if cls_name == "EnvpoolSokobanVecEnvConfig":
        args.pop("px_scale", None)
    else:
        args["asynchronous"] = asynchronous
    return env_classes_map[cls_name](**args)
