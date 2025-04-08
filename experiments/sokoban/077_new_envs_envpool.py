import dataclasses
import math
import shlex
from pathlib import Path
from typing import Optional, Sequence

from farconf import parse_cli, update_fns_to_cli

from cleanba.config import Args, boxworld_drc33
from cleanba.convlstm import ConvConfig, ConvLSTMCellConfig, ConvLSTMConfig
from cleanba.environments import EnvpoolEnvConfig, random_seed
from cleanba.launcher import FlamingoRun, group_from_fname, launch_jobs

clis: list[list[str]] = []
all_args: list[Args] = []


@dataclasses.dataclass
class EnvParameters:
    env_id: str
    max_steps: int
    embed_channels: Sequence[int]
    embed_kernel_sizes: Sequence[int]
    embed_strides: Sequence[int]
    paddings: Sequence[str | int]
    recurrent_layers: int = 3
    recurrent_kernel_size: int = 3
    recurrent_stride: int = 1
    recurrent_channels: int = 32
    repeats_per_step: int = 3
    lstm_trunk_size: Optional[int] = None
    obs_shape: Optional[tuple[int, int, int]] = None
    extra_embed_layers: int = 0

    def __post_init__(self):
        self.paddings = tuple(p.upper() if isinstance(p, str) else p for p in self.paddings)  # type: ignore


def calculate_encoded_shape(obs_shape, channels, kernel_sizes, strides, paddings, debug=False):
    """
    Calculates the output shape after a series of convolutional layers
    with potentially different padding types per layer.

    Padding types supported:
    - "same": Output spatial dimension = ceil(input_dimension / stride).
              Kernel size is not needed for shape calculation in this mode.
    - int (P): Explicit padding amount (applied symmetrically).
               Output spatial dimension =
               floor((input_dimension + 2*P - kernel_size) / stride) + 1.
               Kernel size for the layer is required.

    Args:
        obs_shape (tuple): The input observation shape in (C, H, W) format
                           (Channels, Height, Width).
        channels (list or tuple): List of output channels for each layer.
                                  The last value determines the final channel count.
        kernel_sizes (list or tuple): List of kernel sizes (int for square kernel)
                                      for each layer. Required for integer padding layers.
        strides (list or tuple): List of strides (int) for each layer.
        paddings (list or tuple): List of padding specifications for each layer.
                                  Each element must be the string "same" or a
                                  non-negative integer (P >= 0).
        debug (bool): If True, prints detailed debug information.

    Returns:
        tuple: The encoded shape after the convolutional layers in (C, H, W) format.

    Raises:
        ValueError: If obs_shape is not length 3, if lists are empty,
                    if list lengths don't match, if a padding value is invalid,
                    if stride/kernel size is invalid for a given calculation,
                    or if calculated dimensions become non-positive.
        TypeError: If inputs are not of expected types (tuple/list/int/str).
    """
    # --- Input Validation ---
    if not isinstance(obs_shape, tuple) or len(obs_shape) != 3:
        raise ValueError("obs_shape must be a tuple of length 3 (C, H, W)")
    if not isinstance(channels, (list, tuple)) or not channels:
        raise ValueError("channels must be a non-empty list or tuple")
    if not isinstance(kernel_sizes, (list, tuple)) or not kernel_sizes:
        raise ValueError("kernel_sizes must be a non-empty list or tuple")
    if not isinstance(strides, (list, tuple)) or not strides:
        raise ValueError("strides must be a non-empty list or tuple")
    if not isinstance(paddings, (list, tuple)) or not paddings:
        raise ValueError("paddings must be a non-empty list or tuple")

    num_layers = len(strides)
    if not (len(channels) == num_layers and len(kernel_sizes) == num_layers and len(paddings) == num_layers):
        raise ValueError(
            "Lists channels, kernel_sizes, strides, and paddings must have the same length. "
            f"Got lengths: channels={len(channels)}, kernel_sizes={len(kernel_sizes)}, "
            f"strides={len(strides)}, paddings={len(paddings)}"
        )

    # Assuming obs_shape is (C, H, W)
    current_h = obs_shape[1]
    current_w = obs_shape[2]

    if debug:
        print(f"Initial H: {current_h}, W: {current_w}")

    # --- Iterate Through Layers ---
    for i in range(num_layers):
        stride = strides[i]
        kernel_size = kernel_sizes[i]  # Assuming int for square kernel for simplicity
        padding_spec = paddings[i]

        # Validate stride (must be positive integer)
        if not isinstance(stride, int) or stride < 1:
            raise ValueError(f"Layer {i+1}: Strides must be positive integers. Got: {stride}")

        if debug:
            print(f"\nLayer {i+1}: Stride={stride}, Kernel={kernel_size}, Padding='{padding_spec}'")
            print(f"  Input H: {current_h}, W: {current_w}")

        # Calculate based on padding type
        if padding_spec == "SAME":
            # Formula for "same": ceil(N / S)
            current_h = math.ceil(current_h / stride)
            current_w = math.ceil(current_w / stride)
            if debug:
                print("  Using 'same' padding logic.")

        elif isinstance(padding_spec, int):
            # Validate integer padding value (must be non-negative)
            if padding_spec < 0:
                raise ValueError(f"Layer {i+1}: Integer padding cannot be negative. Got: {padding_spec}")

            # Validate kernel size for this calculation (must be positive integer)
            if not isinstance(kernel_size, int) or kernel_size < 1:
                raise ValueError(
                    f"Layer {i+1}: Kernel size must be a positive integer for explicit padding calculation. Got: {kernel_size}"
                )

            # Formula for explicit padding: floor((N + 2P - K) / S) + 1
            # Calculate for Height
            numerator_h = current_h + 2 * padding_spec - kernel_size
            current_h = math.floor(numerator_h / stride) + 1

            # Calculate for Width
            numerator_w = current_w + 2 * padding_spec - kernel_size
            current_w = math.floor(numerator_w / stride) + 1
            if debug:
                print(f"  Using integer padding ({padding_spec}) logic.")

            # Ensure dimensions don't become non-positive after calculation
            if current_h <= 0 or current_w <= 0:
                raise ValueError(
                    f"Layer {i+1}: Calculated dimensions are not positive (H={current_h}, W={current_w}). "
                    f"Check input shape ({obs_shape}), kernel size ({kernel_size}), "
                    f"padding ({padding_spec}), and stride ({stride})."
                )

        else:
            # Invalid padding type
            raise ValueError(
                f"Layer {i+1}: Invalid padding specification: '{padding_spec}'. " "Must be 'same' or a non-negative integer."
            )
        if debug:
            print(f"  Output H: {current_h}, W: {current_w}")

    # Ensure dimensions are integers after ceiling/floor operations
    final_h = int(current_h)
    final_w = int(current_w)

    # The final number of channels is the output channels of the *last* layer specified
    final_channels = channels[-1]

    return (final_channels, final_h, final_w)


env_params = [
    EnvParameters(
        "ChaserHard-v0",
        120,
        (32, 32, 32),
        (5, 5, 4),
        (2, 2, 1),
        ("same", "same", 0),
        3,
        3,
        1,
        lstm_trunk_size=13,
        obs_shape=(3, 64, 64),
    ),
    # EnvParameters("MazeHard-v0", 120, (32, 32, 32), (5, 5, 4), (2, 1, 1), ("same", 0, 0), 3, 3, 1, lstm_trunk_size=25, obs_shape=(3, 64, 64)),
    # EnvParameters("MinerHard-v0", 120, (32, 32, 32), (3, 3, 3), (3, 1, 1), ("same", "same", 0), 3, 3, 1, lstm_trunk_size=20, obs_shape=(3, 64, 64)),
    # EnvParameters("ChaserEasy-v0", 120, (32, 32, 32), (7, 4, 4), (2, 2, 1), (0, 0, 0), 3, 3, 1, lstm_trunk_size=11, obs_shape=(3, 64, 64)),
    # EnvParameters("MazeEasy-v0", 120, (32, 32, 32), (7, 3, 3), (2, 2, 1), (0, "same", "same"), 3, 3, 1, lstm_trunk_size=15, obs_shape=(3, 64, 64)),
    # EnvParameters("MinerEasy-v0", 120, (32, 32, 32), (7, 4, 4), (2, 2, 1), (0, 0, 0), 3, 3, 1, lstm_trunk_size=10, obs_shape=(3, 64, 64)),
    # atari
    # EnvParameters("Alien-v5", 120, (32, 32, 32), (5, 3, 3), (3, 2, 1), ("same", 0, "same"), 3, 3, 1),
    # EnvParameters("Amidar-v5", 120, (32, 32, 32), (5, 3, 3), (3, 2, 1), ("same", 0, "same"), 3, 3, 1),
    # EnvParameters("Freeway-v5", 120, (32, 32, 32), (5, 3, 3), (3, 2, 1), ("same", 0, "same"), 3, 3, 1),
]

for env_seed, learn_seed in [(random_seed(), random_seed()) for _ in range(1)]:
    for env_p in env_params:
        assert calculate_encoded_shape(
            env_p.obs_shape,
            env_p.embed_channels,
            env_p.embed_kernel_sizes,
            env_p.embed_strides,
            env_p.paddings,
        ) == (env_p.embed_channels[-1], env_p.lstm_trunk_size, env_p.lstm_trunk_size), f"env_p: {env_p}"

        def update_seeds(config: Args) -> Args:
            config.train_env = EnvpoolEnvConfig(env_id=env_p.env_id, max_episode_steps=env_p.max_steps, seed=env_seed)
            config.seed = learn_seed

            config.learning_rate = 4e-4
            config.final_learning_rate = 4e-6
            config.anneal_lr = True

            config.eval_envs = {}
            config.total_timesteps = 200_000_000

            extra_embed_layers = [
                ConvConfig(env_p.embed_channels[-1], (3, 3), (1, 1), "SAME", True)
            ] * env_p.extra_embed_layers

            config.net = ConvLSTMConfig(
                n_recurrent=env_p.recurrent_layers,
                repeats_per_step=env_p.repeats_per_step,
                skip_final=True,
                residual=False,
                use_relu=False,
                embed=[
                    ConvConfig(ch, (k, k), (s, s), p, True)
                    for ch, k, s, p in zip(env_p.embed_channels, env_p.embed_kernel_sizes, env_p.embed_strides, env_p.paddings)
                ]
                + extra_embed_layers,
                recurrent=ConvLSTMCellConfig(
                    ConvConfig(
                        env_p.recurrent_channels,
                        (env_p.recurrent_kernel_size, env_p.recurrent_kernel_size),
                        (env_p.recurrent_stride, env_p.recurrent_stride),
                        "SAME",
                        True,
                    ),
                    pool_and_inject="horizontal",
                    pool_projection="per-channel",
                    output_activation="tanh",
                    fence_pad="valid",
                    forget_bias=0.0,
                ),
            )
            print(config.net)

            return config

        cli, _ = update_fns_to_cli(boxworld_drc33, update_seeds)

        print(shlex.join(cli))
        # Check that parsing doesn't error
        out = parse_cli(cli, Args)

        all_args.append(out)
        clis.append(cli)

runs: list[FlamingoRun] = []
RUNS_PER_MACHINE = 1
for i in range(0, len(clis), RUNS_PER_MACHINE):
    this_run_clis = [
        ["python", "-m", "cleanba.cleanba_impala", *clis[i + j]] for j in range(min(RUNS_PER_MACHINE, len(clis) - i))
    ]
    runs.append(
        FlamingoRun(
            this_run_clis,
            CONTAINER_TAG="4f8513c-main",
            CPU=6,
            MEMORY="150G",
            GPU=1,
            PRIORITY="high-batch",
            XLA_PYTHON_CLIENT_MEM_FRACTION='".95"',
        )
    )


GROUP: str = group_from_fname(__file__)

if __name__ == "__main__":
    launch_jobs(
        runs,
        group=GROUP,
        job_template_path=Path(__file__).parent.parent.parent / "k8s/runner-pip-envpool.yaml",
        project="cleanba",
    )
