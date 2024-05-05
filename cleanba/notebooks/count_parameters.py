from cleanba.config import sokoban_drc_1_1, sokoban_drc_3_3
from cleanba.convlstm import ConvConfig, ConvLSTMConfig
from cleanba.environments import SokobanConfig
from cleanba.network import GuezResNetConfig

# %%
envs = SokobanConfig(
    max_episode_steps=120,
    num_envs=1,
    seed=1,
    tinyworld_obs=False,
    num_boxes=1,
    dim_room=(10, 10),
    asynchronous=False,
).make()
assert envs.single_observation_space.shape == (3, 80, 80)

# %% Define NN architectures and print their parameters

resnet = GuezResNetConfig(kernel_sizes=(8, 4, 4, 4, 4, 4, 4, 4, 4), strides=(4, 2, 1, 1, 1, 1, 1, 1, 1))

drc33 = ConvLSTMConfig(
    embed=[
        ConvConfig(32, (8, 8), (4, 4), "SAME", True),
        ConvConfig(32, (4, 4), (2, 2), "SAME", True),
    ],
    recurrent=ConvConfig(32, (3, 3), (1, 1), "SAME", True),
    n_recurrent=3,
    repeats_per_step=3,
    pool_and_inject=True,
)

drc11 = ConvLSTMConfig(
    embed=[
        ConvConfig(32, (8, 8), (4, 4), "SAME", True),
        ConvConfig(32, (4, 4), (2, 2), "SAME", True),
    ],
    recurrent=ConvConfig(32, (3, 3), (1, 1), "SAME", True),
    n_recurrent=1,
    repeats_per_step=1,
    pool_and_inject=True,
)


# %%

drc33.count_params(envs) - drc11.count_params(envs)


# %%

print("params diff", sokoban_drc_3_3().net.count_params(envs) - sokoban_drc_1_1().net.count_params(envs))


# %% New grad clipping
(sokoban_drc_3_3().net.count_params(envs) / GuezResNetConfig().count_params(envs)) ** 0.5 * 2.5e-4
