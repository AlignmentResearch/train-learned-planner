import contextlib
import dataclasses
import time
from typing import Optional

import jax
import jax.numpy as jnp
import numpy as np

from cleanba.environments import EnvConfig, EnvpoolBoxobanConfig
from cleanba.network import Policy


@dataclasses.dataclass
class EvalConfig:
    env: EnvConfig
    n_episode_multiple: int = 1
    steps_to_think: list[int] = dataclasses.field(default_factory=lambda: [0])
    temperature: float = 0.0

    safeguard_max_episode_steps: int = 30000

    def run(self, policy: Policy, get_action_fn, params, *, key: jnp.ndarray) -> dict[str, float]:
        key, carry_key = jax.random.split(key, 2)
        max_steps = min(self.safeguard_max_episode_steps, self.env.max_episode_steps)
        episode_starts_no = jnp.zeros(self.env.num_envs, dtype=jnp.bool_)

        reward_box_on_target, reward_box_off_target = None, None
        if isinstance(self.env, EnvpoolBoxobanConfig):
            reward_box_on_target = self.env.reward_step + self.env.reward_box
            reward_box_off_target = self.env.reward_step - self.env.reward_box

        metrics = {}
        for steps_to_think in self.steps_to_think:
            # Create the environments every time with the same seed so the levels are the exact same
            all_episode_returns = []
            all_episode_lengths = []
            all_episode_successes = []
            all_episode_num_boxes = []
            all_acts = []
            all_rewards = []
            all_cycles = []
            num_cycles, num_cycles_in_solved = 0, 0
            cycle_steps, cycle_steps_in_solved = 0, 0
            num_noops, num_noops_in_solved = 0, 0
            for minibatch_idx in range(self.n_episode_multiple):
                # Re-create the environments, so we start at the beginning of the batch
                with contextlib.closing(self.env.make()) as envs:
                    start_time = time.time()
                    obs, _ = envs.reset()
                    # Reset more than once so we get to the Nth batch of levels
                    for _ in range(minibatch_idx):
                        obs, _ = envs.reset()

                    # reset the carry here so we can use `episode_starts_no` later
                    carry = policy.apply(params, carry_key, obs.shape, method=policy.initialize_carry)

                    # Update the carry with the initial observation many times
                    for think_step in range(steps_to_think):
                        carry, _, _, _, key = get_action_fn(
                            params, carry, obs, episode_starts_no, key, temperature=self.temperature
                        )

                    eps_done = np.zeros(envs.num_envs, dtype=np.bool_)
                    episode_success = np.zeros(envs.num_envs, dtype=np.bool_)
                    episode_returns = np.zeros(envs.num_envs, dtype=np.float64)
                    episode_lengths = np.zeros(envs.num_envs, dtype=np.int64)
                    episode_acts = np.zeros((max_steps, envs.num_envs), dtype=np.int32)
                    episode_rewards = np.zeros((max_steps, envs.num_envs), dtype=np.float32)
                    episode_num_boxes = np.zeros(envs.num_envs, dtype=np.int64)

                    last_box_time_step = -1 * np.ones(envs.num_envs, dtype=np.int64)
                    noops_array = np.zeros((envs.num_envs, max_steps), dtype=bool)
                    i = 0
                    this_minibatch_obs = [obs]
                    while not np.all(eps_done):
                        if i >= self.safeguard_max_episode_steps:
                            break
                        carry, action, _, _, key = get_action_fn(
                            params, carry, obs, episode_starts_no, key, temperature=self.temperature
                        )

                        cpu_action = np.asarray(action)
                        obs, rewards, terminated, truncated, infos = envs.step(cpu_action)
                        this_minibatch_obs.append(obs)
                        episode_returns[~eps_done] += rewards[~eps_done]
                        episode_lengths[~eps_done] += 1
                        episode_success[~eps_done] |= terminated[~eps_done]  # If episode terminates it's a success

                        if reward_box_on_target is not None and reward_box_off_target is not None:
                            episode_num_boxes[~eps_done] += np.isclose(rewards[~eps_done], reward_box_on_target)
                            episode_num_boxes[~eps_done] -= np.isclose(rewards[~eps_done], reward_box_off_target)

                        episode_acts[i, ~eps_done] = cpu_action[~eps_done]
                        episode_rewards[i, ~eps_done] = rewards[~eps_done]
                        noops_array[:, i] = cpu_action == 4
                        # assumes only box pushes are positive rewards.
                        indices = np.where((~eps_done) & (rewards > 0))
                        last_box_time_step[indices] = episode_lengths[indices]

                        # Set as done the episodes which are done
                        eps_done |= truncated | terminated
                        i += 1

                    all_episode_returns.append(episode_returns)
                    all_episode_lengths.append(episode_lengths)
                    all_episode_successes.append(episode_success)
                    all_episode_num_boxes.append(episode_num_boxes)

                    all_acts += [episode_acts[: episode_lengths[i], i] for i in range(envs.num_envs)]
                    all_rewards += [episode_rewards[: episode_lengths[i], i] for i in range(envs.num_envs)]

                    for env_idx in range(envs.num_envs):
                        if last_box_time_step[env_idx] == -1:
                            all_cycles.append([])
                            continue
                        num_noops += np.sum(noops_array[env_idx, : last_box_time_step[env_idx]])
                        cycles = get_cycles(
                            np.stack([this_minibatch_obs[time_idx][env_idx] for time_idx in range(episode_lengths[env_idx])]),
                            last_box_time_step=last_box_time_step[env_idx],
                        )
                        all_cycles.append(cycles)
                        num_cycles += len(cycles)
                        cycle_steps += sum(cyc_len for _, cyc_len in cycles)

                        if episode_success[env_idx]:
                            num_noops_in_solved += np.sum(noops_array[env_idx, : last_box_time_step[env_idx]])
                            num_cycles_in_solved += len(cycles)
                            cycle_steps_in_solved += sum(cyc_len for _, cyc_len in cycles)

                    total_time = time.time() - start_time
                    print(f"To evaluate the {minibatch_idx}th batch, {round(total_time, ndigits=3)}s")

            total_episodes = self.n_episode_multiple * self.env.num_envs
            total_solved = np.sum(all_episode_successes)
            if total_solved == 0:
                total_solved = 1  # avoid division by zero
            all_episode_num_boxes = np.concatenate(all_episode_num_boxes)
            metrics.update(
                {
                    f"{steps_to_think:02d}_episode_returns": float(np.mean(all_episode_returns)),
                    f"{steps_to_think:02d}_episode_lengths": float(np.mean(all_episode_lengths)),
                    f"{steps_to_think:02d}_episode_successes": float(np.mean(all_episode_successes)),
                    f"{steps_to_think:02d}_episode_num_cycles": num_cycles / total_episodes,
                    f"{steps_to_think:02d}_cycles_steps_per_eps_incl_noops": cycle_steps / total_episodes,
                    f"{steps_to_think:02d}_cycles_steps_per_eps_excl_noops": (cycle_steps - num_noops) / total_episodes,
                    f"{steps_to_think:02d}_episode_num_noops_per_eps": num_noops / total_episodes,
                    f"{steps_to_think:02d}_episode_num_cycles_in_solved": num_cycles_in_solved / total_solved,
                    f"{steps_to_think:02d}_cycles_steps_per_eps_incl_noops_in_solved": cycle_steps_in_solved / total_solved,
                    f"{steps_to_think:02d}_cycles_steps_per_eps_excl_noops_in_solved": (
                        cycle_steps_in_solved - num_noops_in_solved
                    )
                    / total_solved,
                    f"{steps_to_think:02d}_episode_num_noops_per_eps_in_solved": num_noops_in_solved / total_solved,
                    f"{steps_to_think:02d}_episode_zero_boxes": np.sum(all_episode_num_boxes == 0) / total_episodes,
                    f"{steps_to_think:02d}_episode_one_box": np.sum(all_episode_num_boxes == 1) / total_episodes,
                    f"{steps_to_think:02d}_episode_two_boxes": np.sum(all_episode_num_boxes == 2) / total_episodes,
                    f"{steps_to_think:02d}_episode_three_boxes": np.sum(all_episode_num_boxes == 3) / total_episodes,
                    f"{steps_to_think:02d}_all_episode_info": dict(
                        episode_returns=all_episode_returns,
                        episode_lengths=all_episode_lengths,
                        episode_successes=all_episode_successes,
                        episode_acts=all_acts,
                        episode_rewards=all_rewards,
                        all_cycles=all_cycles,
                    ),
                }
            )
        return metrics


def get_cycles(
    all_obs: np.ndarray,
    last_box_time_step: int,
    cycle_starts_within: Optional[int] = None,
    min_cycle_length: int = 1,
) -> list[tuple[int, int]]:
    """
    Given a sequence of observations, find all cycles in the sequence (including noop actions).
    A cycle is a sequence of observations with the same starting and ending observations.

    Args:
        :param all_obs: A sequence of observations, shape (time, 3, H, W) where H == W.
        :param last_box_time_step: The last time step where a box was pushed onto a target.
        :param cycle_starts_within: Only consider cycles starting within the first `cycle_starts_within` time steps.
        :param min_cycle_length: Only consider cycles of length at least `min_cycle_length`.
    Returns:
        A list of tuples, where each tuple is a cycle (start, length).
    """
    assert all_obs.shape[1] == 3 and all_obs.shape[2] == all_obs.shape[3], all_obs.shape
    assert last_box_time_step is not None
    cycle_starts_within = cycle_starts_within or all_obs.shape[0]
    all_obs = all_obs[:last_box_time_step]
    all_obs = all_obs.reshape(all_obs.shape[0], 1, *all_obs.shape[1:])
    obs_repeat = np.all(all_obs == all_obs.transpose(1, 0, 2, 3, 4), axis=(2, 3, 4))
    np.fill_diagonal(obs_repeat, False)
    obs_repeat = [np.where(obs_repeat[j])[0] for j in range(min(cycle_starts_within, len(obs_repeat)))]
    dedup_obs_repeat = []
    i = 0
    # this way of deduplicating will break some 8 shaped cycles into two circles (at different starts).
    # such cycles occur rarely and this simplification is fine for our purpose.
    while i < len(obs_repeat):
        if obs_repeat[i].size > 0 and min_cycle_length <= obs_repeat[i][-1] - i:
            dedup_obs_repeat.append((i, obs_repeat[i][-1] - i))  # max length cycle starting at i
            i += dedup_obs_repeat[-1][1]
        i += 1

    return dedup_obs_repeat
