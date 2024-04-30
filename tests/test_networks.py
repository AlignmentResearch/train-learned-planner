from cleanba.convlstm import ConvConfig, ConvLSTMConfig
from cleanba.environments import SokobanConfig
from cleanba.network import GuezResNetConfig


def test_count_parameters():
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

    resnet_params = GuezResNetConfig(
        kernel_sizes=(8, 4, 4, 4, 4, 4, 4, 4, 4), strides=(4, 2, 1, 1, 1, 1, 1, 1, 1)
    ).count_params(envs)
    assert resnet_params == 3_171_333

    drc33_params = ConvLSTMConfig(
        embed=[
            ConvConfig(32, (8, 8), (4, 4), "SAME", True),
            ConvConfig(32, (4, 4), (2, 2), "SAME", True),
        ],
        recurrent=[ConvConfig(32, (3, 3), (1, 1), "SAME", True)] * 3,
        repeats_per_step=3,
        pool_and_inject=True,
        add_one_to_forget=True,
    ).count_params(envs)
    assert drc33_params == 2_042_376
