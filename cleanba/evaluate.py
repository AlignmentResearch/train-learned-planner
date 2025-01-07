import contextlib
import dataclasses
import time

import jax
import jax.numpy as jnp
import numpy as np

from cleanba.environments import EnvConfig
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

        metrics = {}
        for steps_to_think in self.steps_to_think:
            # Create the environments every time with the same seed so the levels are the exact same
            all_episode_returns = []
            all_episode_lengths = []
            all_episode_successes = []
            all_acts = []
            all_rewards = []
            all_cycles = []
            num_cycles = 0
            cycle_lens = 0
            num_noops = 0
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
                        carry, _, _, key = get_action_fn(
                            params, carry, obs, episode_starts_no, key, temperature=self.temperature
                        )

                    eps_done = np.zeros(envs.num_envs, dtype=np.bool_)
                    episode_success = np.zeros(envs.num_envs, dtype=np.bool_)
                    episode_returns = np.zeros(envs.num_envs, dtype=np.float64)
                    episode_lengths = np.zeros(envs.num_envs, dtype=np.int64)
                    episode_acts = np.zeros((max_steps, envs.num_envs), dtype=np.int32)
                    episode_rewards = np.zeros((max_steps, envs.num_envs), dtype=np.float32)

                    last_box_time_step = -1 * np.ones(envs.num_envs, dtype=np.int64)
                    noops_array = np.zeros((envs.num_envs, max_steps), dtype=bool)
                    i = 0
                    all_obs = [obs]
                    while not np.all(eps_done):
                        if i >= self.safeguard_max_episode_steps:
                            break
                        carry, action, _, key = get_action_fn(
                            params, carry, obs, episode_starts_no, key, temperature=self.temperature
                        )

                        cpu_action = np.asarray(action)
                        obs, rewards, terminated, truncated, infos = envs.step(cpu_action)
                        all_obs.append(obs)
                        episode_returns[~eps_done] += rewards[~eps_done]
                        episode_lengths[~eps_done] += 1
                        episode_success[~eps_done] |= terminated[~eps_done]  # If episode terminates it's a success

                        episode_acts[i, ~eps_done] = cpu_action[~eps_done]
                        episode_rewards[i, ~eps_done] = rewards[~eps_done]
                        noops_array[i] = (cpu_action == 4)
                        # assumes only box pushes are positive rewards.
                        indices = np.where((~eps_done) & (rewards > 0))
                        last_box_time_step[indices] = episode_lengths[indices]

                        # Set as done the episodes which are done
                        eps_done |= truncated | terminated
                        i += 1

                    all_episode_returns.append(episode_returns)
                    all_episode_lengths.append(episode_lengths)
                    all_episode_successes.append(episode_success)

                    all_acts += [episode_acts[: episode_lengths[i], i] for i in range(envs.num_envs)]
                    all_rewards += [episode_rewards[: episode_lengths[i], i] for i in range(envs.num_envs)]

                    for env_idx in range(envs.num_envs):
                        if last_box_time_step[env_idx] == -1:
                            all_cycles.append([])
                            continue
                        num_noops += np.sum(noops_array[env_idx, : last_box_time_step[env_idx]])
                        cycles = get_cycles(
                            np.stack([all_obs[time_idx][env_idx] for time_idx in range(episode_lengths[env_idx])]),
                            last_box_time_step=last_box_time_step[env_idx],
                        )
                        all_cycles.append(cycles)
                        num_cycles += len(cycles)
                        cycle_lens += sum(cyc_len for _, cyc_len in cycles)

                    total_time = time.time() - start_time
                    print(f"To evaluate the {minibatch_idx}th batch, {round(total_time, ndigits=3)}s")

            metrics.update(
                {
                    f"{steps_to_think:02d}_episode_returns": float(np.mean(all_episode_returns)),
                    f"{steps_to_think:02d}_episode_lengths": float(np.mean(all_episode_lengths)),
                    f"{steps_to_think:02d}_episode_successes": float(np.mean(all_episode_successes)),
                    f"{steps_to_think:02d}_episode_num_cycles": num_cycles / (self.n_episode_multiple * self.env.num_envs),
                    f"{steps_to_think:02d}_episode_cycle_lens": cycle_lens / (self.n_episode_multiple * self.env.num_envs),
                    f"{steps_to_think:02d}_episode_num_noops_per_eps": num_noops
                    / (self.n_episode_multiple * self.env.num_envs),
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
    all_obs,
    last_box_time_step,
    cycle_starts_within=None,
    min_cycle_length=1,
):
    assert all_obs.shape[1] == 3 and all_obs.shape[2] == all_obs.shape[3], all_obs.shape
    assert last_box_time_step is not None
    cycle_starts_within = cycle_starts_within or all_obs.shape[0]
    all_obs = all_obs[:last_box_time_step]
    all_obs = all_obs.reshape(all_obs.shape[0], 1, *all_obs.shape[1:])
    obs_repeat = np.all(all_obs == all_obs.transpose(1, 0, 2, 3, 4), axis=(2, 3, 4))
    np.fill_diagonal(obs_repeat, False)
    obs_repeat = [np.where(obs_repeat[j])[0] for j in range(min(cycle_starts_within, len(obs_repeat)))]
    # obs_repeat = [
    #     (j, arr[-1] - j)
    #     for j, arr in enumerate(obs_repeat)
    #     if arr.size > 0 and min_cycle_length <= arr[-1] - j
    # ]
    dedup_obs_repeat = []
    i = 0
    # this way of deduplicating will break some 8 shaped cycles into two circles (at different starts)
    while i < len(obs_repeat):
        if obs_repeat[i].size > 0 and min_cycle_length <= obs_repeat[i][-1] - i:
            dedup_obs_repeat.append((i, obs_repeat[i][-1] - i))  # max length cycle starting at i
            i += dedup_obs_repeat[-1][1]
        i += 1

    return dedup_obs_repeat
