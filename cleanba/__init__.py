from gymnasium.envs.registration import register

register(
    id="BoxWorld-v0",
    entry_point="cleanba.envs:BoxWorld",
)
