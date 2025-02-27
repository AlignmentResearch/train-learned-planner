import abc
import dataclasses
import os
import random
import warnings
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Iterable, List, Literal, Optional, Self, Tuple, Union

import flax.struct
import gym_sokoban  # noqa: F401
import gymnasium as gym
import jax
import jax.experimental.compilation_cache
import jax.numpy as jnp
import numpy as np
from gymnasium.vector.utils import batch_space
from numpy.typing import NDArray

if TYPE_CHECKING:
    from craftax.craftax.craftax_state import EnvParams
    from craftax.craftax.envs.craftax_symbolic_env import CraftaxSymbolicEnv


class EpisodeEvalState(flax.struct.PyTreeNode):
    episode_length: jax.Array
    episode_success: jax.Array
    episode_others: dict[str, jax.Array]

    returned_episode_length: jax.Array
    returned_episode_success: jax.Array
    returned_episode_others: dict[str, jax.Array]

    @classmethod
    def new(cls: type[Self], num_envs: int, others: Iterable[str]) -> Self:
        zero_float = jnp.zeros(())
        zero_int = jnp.zeros((), dtype=jnp.int32)
        zero_bool = jnp.zeros((), dtype=jnp.bool)
        others = set(others) | {"episode_return"}
        return jax.tree.map(
            partial(jnp.repeat, repeats=num_envs),
            cls(
                zero_int,
                zero_bool,
                {o: zero_float for o in others},
                zero_int,
                zero_bool,
                {o: zero_float for o in others},
            ),
        )

    @jax.jit
    def update(
        self: Self, reward: jnp.ndarray, terminated: jnp.ndarray, truncated: jnp.ndarray, others: dict[str, jnp.ndarray]
    ) -> Self:
        done = terminated | truncated

        new_episode_success = terminated
        new_episode_length = self.episode_length + 1
        new_others = jax.tree.map(lambda a, b: a + b, self.episode_others, {"episode_return": reward, **others})

        new_state = self.__class__(
            episode_length=new_episode_length * (1 - done),
            episode_success=new_episode_success * (1 - done),
            episode_others=jax.tree.map(lambda x: x * (1 - done), new_others),
            returned_episode_length=jax.lax.select(done, new_episode_length, self.returned_episode_length),
            returned_episode_success=jax.lax.select(done, new_episode_success, self.returned_episode_success),
            returned_episode_others=jax.tree.map(partial(jax.lax.select, done), new_others, self.returned_episode_others),
        )
        return new_state

    def update_info(self) -> dict[str, Any]:
        return {
            "returned_episode_length": self.returned_episode_length,
            "returned_episode_success": self.returned_episode_success,
            **{f"returned_{k}": v for k, v in self.returned_episode_others.items()},
        }


class EpisodeEvalWrapper(gym.vector.VectorEnvWrapper):
    """Log the episode returns and lengths."""

    state: EpisodeEvalState

    def __init__(self, env: gym.vector.VectorEnv):
        super().__init__(env)
        self._env = env

    @staticmethod
    def _info_achievements(info: dict[str, Any]) -> dict[str, Any]:
        return {k: v for k, v in info.items() if "achievement" in k}

    def reset(self, seed: Optional[Union[int, List[int]]] = None, options: Optional[dict] = None) -> Tuple[jnp.ndarray, dict]:
        obs, info = self._env.reset()
        self.state = EpisodeEvalState.new(self._env.num_envs, self._info_achievements(info).keys())
        return obs, {**info, **self.state.update_info()}

    def step(self, actions: jnp.ndarray) -> Tuple[Any, jnp.ndarray, jnp.ndarray, jnp.ndarray, dict]:
        self._state, other = self._step(actions)
        return other

    @partial(jax.jit, static_argnums=(0,))
    def _step(self, actions: jnp.ndarray) -> Tuple[Any, jnp.ndarray, jnp.ndarray, jnp.ndarray, dict]:
        obs, reward, terminated, truncated, info = self._env.step(actions)
        # Atari envs clip their reward to [-1, 1], meaning we need to use the reward in `info` to get
        # the true return.
        non_clipped_rewards = info.get("reward", reward)
        state = self.state.update(non_clipped_rewards, terminated, truncated, self._info_achievements(info))
        done = terminated | truncated
        return state, (obs, reward, done, truncated, {**info, **self.state.update_info()})


class CraftaxVectorEnv(gym.vector.VectorEnv):
    """
    Craftax environment with a generic VectorEnv interface.
    """

    cfg: "CraftaxEnvConfig"
    env: "CraftaxSymbolicEnv"
    rng_keys: jnp.ndarray
    state: Any
    obs: jnp.ndarray
    env_params: "EnvParams"

    def __init__(self, cfg: "CraftaxEnvConfig"):
        from craftax.craftax.envs.craftax_symbolic_env import CraftaxSymbolicEnv

        self.cfg = cfg
        self.env = CraftaxSymbolicEnv()

        obs_shape = (8268,) if cfg.obs_flat else (134, 9, 11)  # My guess is it should be (9, 11, 134) should be reversed
        single_observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=obs_shape, dtype=np.float32)
        single_action_space = gym.spaces.Discrete(self.env.action_space().n)
        super().__init__(cfg.num_envs, single_observation_space, single_action_space)

        self.env_params = self.env.default_params
        self.closed = False

        self.device, *_ = jax.devices(cfg.jit_backend)

        # set rng_keys, state, obs
        self.reset(self.cfg.seed)

    def _process_obs(self, obs_flat):
        if self.cfg.obs_flat:
            return obs_flat
        expected_size = 8268
        assert obs_flat.shape[0] == expected_size, (
            f"Observation size mismatch: got {obs_flat.shape[0]}, expected {expected_size}"
        )

        mapobs = obs_flat[:8217].reshape(9, 11, 83)
        invobs = obs_flat[8217:].reshape(51)
        invobs_spatial = invobs.reshape(1, 1, 51).repeat(9, axis=0).repeat(11, axis=1)
        obs_nhwc = jnp.concatenate([mapobs, invobs_spatial], axis=-1)  # (9, 11, 134)
        obs_nchw = jnp.transpose(obs_nhwc, (2, 0, 1))  # (134, 9, 11)

        return obs_nchw

    @partial(jax.jit, static_argnames=("self",))
    @partial(jax.vmap, in_axes=(None, 0))
    def _reset_wait_pure(self, key: jnp.ndarray) -> Tuple[jnp.ndarray, Any, jnp.ndarray]:
        key, reset_key = jax.random.split(key)
        obs_flat, state = self.env.reset_env(reset_key, self.env_params)
        obs_processed = self._process_obs(obs_flat)
        return obs_processed, state, key

    def reset(self, seed: Optional[Union[int, List[int]]] = None, options: Optional[dict] = None) -> Tuple[jnp.ndarray, dict]:
        """Reset the environment."""
        if isinstance(seed, int):
            self.rng_keys = jax.random.split(jax.random.PRNGKey(seed), self.num_envs)
        elif isinstance(seed, list):
            assert len(seed) == self.num_envs
            self.rng_keys = jax.jit(jax.vmap(jax.random.PRNGKey))(jnp.asarray(seed))
        self.rng_keys = jax.device_put(self.rng_keys, self.device)
        self.obs, self.state, self.rng_keys = self._reset_wait_pure(self.rng_keys)
        return self.obs, {}

    @partial(jax.jit, static_argnames=("self",))
    @partial(jax.vmap, in_axes=(None, 0, 0, 0))
    def _step_pure(self, key, state, action):
        key, step_key = jax.random.split(key)
        obs_flat, state, rewards, dones, info = self.env.step(step_key, state, action)
        terminated = dones
        # assume no truncation (basically true as agent does not survive long enough)
        truncated = jnp.zeros_like(dones, dtype=bool)
        assert terminated.dtype == truncated.dtype
        obs = self._process_obs(obs_flat)
        return key, obs, state, rewards, terminated, truncated, info

    def step(self, actions: jnp.ndarray) -> Tuple[Any, jnp.ndarray, jnp.ndarray, jnp.ndarray, dict]:
        """Execute one step in the environment."""
        actions = jax.device_put(actions, self.device)
        self.rng_keys, self.obs, self.state, rewards, terminated, truncated, info = self._step_pure(
            self.rng_keys, self.state, actions
        )
        return self.obs, rewards, terminated, truncated, info

    def close(self, **kwargs):
        self.closed = True


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
class CraftaxEnvConfig(EnvConfig):
    """Configuration class for integrating Craftax with IMPALA."""

    max_episode_steps: int
    num_envs: int = 1
    seed: int = dataclasses.field(default_factory=random_seed)
    obs_flat: bool = False
    jit_backend: str = "cuda"

    @property
    def make(self) -> Callable[[], CraftaxVectorEnv]:  # type: ignore
        # This property returns a function that creates the Craftax environment wrapper.
        return lambda: CraftaxVectorEnv(self)


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
