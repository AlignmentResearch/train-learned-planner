from gymnasium.envs.registration import register

register(
    id="BoxWorld-v0",
    entry_point="cleanba.envs:BoxWorld",
)
register(
    id="MiniPacMan-v0",
    entry_point="cleanba.envs:MiniPacMan",
)
