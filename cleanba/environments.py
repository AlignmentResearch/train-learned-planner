import abc
import dataclasses
import os
import warnings
from functools import partial
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Generic,
    Iterable,
    Literal,
    NamedTuple,
    Optional,
    Self,
    Tuple,
    TypeVar,
)

import chex
import flax.struct
import gym_sokoban  # noqa: F401
import gymnasium as gym
import jax
import jax.experimental.compilation_cache
import jax.numpy as jnp
import numpy as np
from gymnax.environments.environment import Environment, TEnvState
from gymnax.environments.spaces import Box, Discrete, Space

if TYPE_CHECKING:
    from craftax.craftax.craftax_state import EnvState as CraftaxEnvState
    from craftax.craftax.craftax_state import StaticEnvParams as CraftaxStaticEnvParams


@flax.struct.dataclass
class EpisodeEvalWrapperState:
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
        new_episode_success = terminated
        done = terminated | truncated
        new_episode_length = self.episode_length + 1

        # Populate things to do tree.map
        _episode_others = {k: jnp.zeros(new_episode_length.shape) for k in others.keys()}
        _episode_others.update(self.episode_others)
        _returned_episode_others = {k: jnp.zeros(new_episode_length.shape) for k in others.keys()}
        _returned_episode_others.update(self.returned_episode_others)

        new_others = jax.tree.map(lambda a, b: a + b, _episode_others, {"episode_return": reward, **others})

        new_state = self.__class__(
            episode_length=new_episode_length * (1 - done),
            episode_success=new_episode_success * (1 - done),
            episode_others=jax.tree.map(lambda x: x * (1 - done), new_others),
            returned_episode_length=jax.lax.select(done, new_episode_length, self.returned_episode_length),
            returned_episode_success=jax.lax.select(done, new_episode_success, self.returned_episode_success),
            returned_episode_others=jax.tree.map(partial(jax.lax.select, done), new_others, _returned_episode_others),
        )
        return new_state

    def update_info(self) -> dict[str, Any]:
        return {
            "returned_episode_length": self.returned_episode_length,
            "returned_episode_success": self.returned_episode_success,
            **{f"returned_{k}": v for k, v in self.returned_episode_others.items()},
        }


TState = TypeVar("TState")


class StepOutput(NamedTuple, Generic[TState]):
    obs: chex.Array
    state: TState
    reward: jnp.ndarray
    terminated: jnp.ndarray
    truncated: jnp.ndarray
    info: dict[str, jnp.ndarray]


class ResetOutput(NamedTuple, Generic[TState]):
    obs: chex.Array
    state: TState


class SimpleVectorizedEnvironment(abc.ABC, Generic[TState]):
    """Vectorized environment with five methods: reset and step (with truncation), obs and action spaces, and delete."""

    num_envs: int

    @abc.abstractmethod
    def reset_env(self, key: chex.PRNGKey, state: Optional[TState] = None) -> ResetOutput[TState]: ...

    @abc.abstractmethod
    def step_env(self, key: chex.PRNGKey, state: TState, action: chex.Array) -> StepOutput[TState]: ...

    @property
    @abc.abstractmethod
    def single_observation_space(self) -> Space: ...

    @property
    @abc.abstractmethod
    def single_action_space(self) -> Space: ...

    def delete(self, state: TState) -> None:
        pass

    def example_observation(self) -> chex.Array:
        return jax.eval_shape(
            jax.vmap(self.single_observation_space.sample), jax.random.split(jax.random.PRNGKey(0), self.num_envs)
        )

    def example_action(self) -> chex.Array:
        return jax.eval_shape(
            jax.vmap(self.single_action_space.sample), jax.random.split(jax.random.PRNGKey(0), self.num_envs)
        )


class EpisodeEvalWrapper(SimpleVectorizedEnvironment[Tuple[TState, EpisodeEvalWrapperState]]):
    """Log the episode returns and lengths."""

    def __init__(self, env: SimpleVectorizedEnvironment[TState]):
        self._env = env

    def reset_env(
        self, key: chex.PRNGKey, state: Optional[TState] = None
    ) -> ResetOutput[Tuple[TState, EpisodeEvalWrapperState]]:
        assert key.ndim >= 2, "key must have at least 2 dimensions. That is, the envs must be vectorized."
        num_envs = key.shape[0]
        obs, env_state = self._env.reset_env(key, state)
        step_out = jax.eval_shape(self._env.step_env, key, env_state, self.example_action())
        eval_state = EpisodeEvalWrapperState.new(num_envs, self._info_achievements(step_out.info).keys())
        return ResetOutput(obs, (env_state, eval_state))

    @staticmethod
    def _info_achievements(info: dict[str, Any]) -> dict[str, Any]:
        return {k: v for k, v in info.items() if "Achievement" in k}

    def step_env(
        self, key: chex.PRNGKey, state: Tuple[TState, EpisodeEvalWrapperState], action: chex.Array
    ) -> StepOutput[Tuple[TState, EpisodeEvalWrapperState]]:
        (env_state, eval_state) = state
        step_out = self._env.step_env(key, env_state, action)
        # Atari envs clip their reward to [-1, 1], meaning we need to use the reward in `info` to get
        # the true return.
        non_clipped_rewards = step_out.info.get("reward", step_out.reward)
        new_eval_state = eval_state.update(
            non_clipped_rewards, step_out.terminated, step_out.truncated, self._info_achievements(step_out.info)
        )

        new_state = (step_out.state, new_eval_state)
        info = {**step_out.info, **new_eval_state.update_info()}
        return StepOutput(step_out.obs, new_state, step_out.reward, step_out.terminated, step_out.truncated, info)


@flax.struct.dataclass
class EnvConfig(abc.ABC, Generic[TState]):
    num_envs: int
    max_episode_steps: int

    @abc.abstractmethod
    def make(self) -> SimpleVectorizedEnvironment[TState]: ...


# skip type checking for env params -- CraftaxEnvParams are not the same as EnvParams
class GymnaxSimpleVectorizedEnvironment(SimpleVectorizedEnvironment[TEnvState]):
    env: Environment[TEnvState, Any]
    num_envs: int
    params: Any
    obs_fn: Callable[[chex.Array], chex.Array]

    def __init__(
        self,
        env: Environment[TEnvState, Any],
        num_envs: int,
        params: Any,
        obs_fn: Optional[Callable[[chex.Array], chex.Array]] = None,
    ):
        self.env = env
        self.num_envs = num_envs
        self.params = params
        self.obs_fn = obs_fn or (lambda x: x)

    def reset_env(self, key: chex.PRNGKey, state: Optional[TEnvState] = None) -> ResetOutput[TEnvState]:
        many_keys = jax.random.split(key, self.num_envs)
        obs, state = jax.vmap(self.env.reset_env, in_axes=(0, None), out_axes=0)(many_keys, self.params)
        return ResetOutput(self.obs_fn(obs), state)

    def step_env(self, key: chex.PRNGKey, state: TEnvState, action: chex.Array) -> StepOutput[TEnvState]:
        many_keys = jax.random.split(key, self.num_envs)
        obs, state, reward, done, info = jax.vmap(self.env.step_env, in_axes=(0, 0, 0, None), out_axes=0)(
            many_keys, state, action, self.params
        )
        truncated = info.get("truncated", jnp.zeros_like(done))
        terminated = done & (~truncated)
        return StepOutput(self.obs_fn(obs), state, reward, terminated, truncated, info)

    @property
    def single_observation_space(self) -> Space:
        return self.env.observation_space(self.params)

    @property
    def single_action_space(self) -> Space:
        return self.env.action_space(self.params)


@flax.struct.dataclass
class CraftaxEnvConfig(EnvConfig["CraftaxEnvState", "CraftaxEnvParams"]):  # type: ignore
    static_params: Optional["CraftaxStaticEnvParams"] = None
    obs_flat: bool = False
    classic: bool = False
    symbolic: bool = True

    def make(self) -> SimpleVectorizedEnvironment[CraftaxEnvState]:
        if self.symbolic:
            from craftax.craftax.envs.craftax_symbolic_env import CraftaxSymbolicEnv

            craftax_env = CraftaxSymbolicEnv(self.static_params)
            obs_fn = self._make_obs_nchw if self.obs_flat else None
        else:
            from craftax.craftax.envs.craftax_pixels_env import CraftaxPixelsEnv

            if self.obs_flat:
                raise ValueError("You probably don't want to use the pixels env with obs_flat=True")
            obs_fn = None
            craftax_env = CraftaxPixelsEnv(self.static_params)

        default_params = craftax_env.default_params
        default_params.max_timesteps = self.max_episode_steps
        env = GymnaxSimpleVectorizedEnvironment(craftax_env, self.num_envs, default_params, obs_fn)
        return env

    @staticmethod
    def _make_obs_nchw(obs_flat):
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


@flax.struct.dataclass
class EnvpoolEnvConfig(EnvConfig[Any]):
    env_id: str | None = None

    num_threads: int = 0
    thread_affinity_offset: int = -1
    max_num_players: int = 1

    def make(self) -> SimpleVectorizedEnvironment[Any]:
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
        env = EnvpoolSimpleVectorizedEnvironment(env_id=self.env_id, batch_size=self.num_envs, **env_kwargs, **special_kwargs)  # type: ignore
        return env


def gym_to_gymnax_space(space: gym.Space) -> Space:
    if isinstance(space, gym.spaces.Discrete):
        return Discrete(int(space.n))
    elif isinstance(space, gym.spaces.Box):
        return Box(low=jnp.array(space.low), high=jnp.array(space.high), shape=jnp.array(space.shape))
    else:
        raise NotImplementedError(f"Envpool space {space} not implemented")


class EnvpoolSimpleVectorizedEnvironment(SimpleVectorizedEnvironment[Any]):
    def __init__(self, env_id: str, batch_size: int, remove_last_action: bool = False, **kwargs):
        self.env_id = env_id
        self.num_envs = batch_size
        self.kwargs = kwargs
        self.remove_last_action = remove_last_action

        # State of the envpools
        self.active_envs = {}
        handle, self._send, self._recv = self._effectful_make_envs(np.array(0))
        envs = self.active_envs[np.array(handle).tobytes()]
        self._single_observation_space = envs.observation_space
        self._single_action_space = envs.action_space

    def _effectful_make_envs(self, seed: np.ndarray, existing_handle: Optional[np.ndarray] = None) -> Any:
        import envpool

        assert seed.shape == ()
        seed_int = int(seed)
        if existing_handle is None:
            envs = envpool.make_gymnasium(self.env_id, seed=seed_int, batch_size=self.num_envs, **self.kwargs)
            handle, _send, _recv = envs.xla()
            self.active_envs[np.array(handle).tobytes()] = envs
            envs.async_reset()  # Send the reset signal
            return handle, _send, _recv
        else:
            envs = self.active_envs[np.array(existing_handle).tobytes()]
            envs.async_reset()  # Send the reset signal
            return existing_handle

    def reset_env(self, key: chex.PRNGKey, state: Optional[Any] = None) -> ResetOutput[Any]:
        chex.assert_rank(key, 1)
        seed = jax.random.randint(key, (), 0, 2**31 - 1)
        handle = jax.experimental.io_callback(
            lambda seed, existing_handle: self._effectful_make_envs(seed, existing_handle)[0],
            ordered=False,
            result_shape_dtypes=jnp.zeros((64 // 8,), dtype=jnp.uint8),
        )(seed, state)
        obs, _ = self._recv(handle)
        return ResetOutput(obs=obs, state=handle)

    def step_env(self, key: chex.PRNGKey, state: Any, action: chex.Array) -> StepOutput[Any]:
        """Execute one step in the environment."""
        handle = state
        obs, reward, terminated, truncated, info = self._send(handle, action, env_id=None)
        return StepOutput(obs=obs, state=handle, reward=reward, terminated=terminated, truncated=truncated, info=info)

    @property
    def single_observation_space(self) -> Space:
        return gym_to_gymnax_space(self._single_observation_space)

    @property
    def single_action_space(self) -> Space:
        return gym_to_gymnax_space(self._single_action_space)


@flax.struct.dataclass
class EnvpoolBoxobanConfig(EnvpoolEnvConfig):
    env_id: str = "Sokoban-v0"  # type: ignore

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


class GymnasiumSimpleVectorizedEnvironment(SimpleVectorizedEnvironment[None]):
    def __init__(
        self,
        make_fn: Callable[[], gym.Env],
        nn_without_noop: bool,
        num_envs: int,
        obs_fn: Optional[Callable[[chex.Array], chex.Array]] = None,
    ):
        self.envs = [make_fn() for _ in range(num_envs)]
        self.num_envs = num_envs

    def reset_env(self, key: chex.PRNGKey, state: Optional[None] = None) -> ResetOutput[None]:
        seeds = jax.random.randint(key, (self.num_envs,), 0, 2**31 - 1)
        obs = jax.experimental.io_callback(
            lambda x, seed: x.reset(seed=seed)[0], ordered=False, result_shape_dtypes=self.example_observation
        )(seeds)
        return ResetOutput(obs=obs, state=None)

    def step_env(self, key: chex.PRNGKey, state: None, action: chex.Array) -> StepOutput[None]:
        raise NotImplementedError("Not implemented")

    @property
    def single_observation_space(self) -> Space:
        return gym_to_gymnax_space(self.envs[0].observation_space)

    @property
    def single_action_space(self) -> Space:
        return gym_to_gymnax_space(self.envs[0].action_space)


@dataclasses.dataclass
class SokobanConfig(BaseSokobanEnvConfig):
    "Procedurally-generated Sokoban"

    dim_room: tuple[int, int] | None = None
    num_boxes: int = 4
    num_gen_steps: int | None = None

    def make(self) -> SimpleVectorizedEnvironment[None]:
        kwargs = self.env_kwargs()
        for k in ["dim_room", "num_boxes", "num_gen_steps"]:
            if (a := getattr(self, k)) is not None:
                kwargs[k] = a
        out = GymnasiumSimpleVectorizedEnvironment(
            partial(
                # We use `gym.vector.make` even though it is deprecated because it gives us `gym.vector.SyncVectorEnv`
                # instead of `gym.experimental.vector.SyncVectorEnv`.
                gym.vector.make,
                "Sokoban-v2",
                **kwargs,
                **self.env_reward_kwargs(),
            ),
            nn_without_noop=self.nn_without_noop,
            num_envs=self.num_envs,
            obs_fn=lambda x: np.moveaxis(x, 2, 0),
        )
        return out


@dataclasses.dataclass
class BoxobanConfig(BaseSokobanEnvConfig):
    "Sokoban levels from the Boxoban data set"

    cache_path: Path = Path("/opt/sokoban_cache")
    split: Literal["train", "valid", "test", "planning", None] = "train"
    difficulty: Literal["unfiltered", "medium", "hard"] = "unfiltered"
    level_idxs_path: Path | None = None

    def make(self) -> SimpleVectorizedEnvironment[None]:
        if self.difficulty == "hard":
            if self.split is not None:
                raise ValueError("`hard` levels have no splits")
        elif self.difficulty == "medium":
            if self.split == "test":
                raise ValueError("`medium` levels don't have a `test` split")

        out = GymnasiumSimpleVectorizedEnvironment(
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
            self.num_envs,
            obs_fn=lambda x: np.moveaxis(x, 2, 0),
        )
        return out


ATARI_MAX_FRAMES = int(
    108000 / 4
)  # 108000 is the max number of frames in an Atari game, divided by 4 to account for frame skipping
# This equals 27k, which is the default max_episode_steps for Atari in Envpool


@flax.struct.dataclass
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
