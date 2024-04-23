import dataclasses
from typing import Any

import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np

from cleanba.network import (
    GuezResNetConfig,
    IdentityNorm,
)

# %% envs


@dataclasses.dataclass
class Envs:
    single_action_space: Any
    single_observation_space: Any

    def __hash__(self):
        return 0


envs = Envs(gym.spaces.Discrete(4), gym.spaces.Box(0, 255, (3, 80, 80)))


# %% Check variance of untrained NN

# net = SokobanResNetConfig(
#     yang_init=True,
#     norm=RMSNorm(eps=1e-06, use_scale=True, reduction_axes=-1, feature_axes=-1),
#     channels=(64,) * 9,
#     kernel_sizes=(4,) * 9,
#     mlp_hiddens=(256,),
#     last_activation="relu",
# )
# net = AtariCNNSpec(yang_init=True, norm=RMSNorm(), channels=(64,) * 9, strides=(1,) * 9, mlp_hiddens=(256,), max_pool=False)

net = GuezResNetConfig(
    yang_init=True,
    norm=IdentityNorm(),
    channels=(32, 32, 64, 64, 64, 64, 64, 64, 64),
    strides=(4, 2, 1, 1, 1, 1, 1, 1, 1),
    kernel_sizes=(8, 4, 4, 4, 4, 4, 4, 4, 4),
    mlp_hiddens=(256,),
)


key, subk = jax.random.split(jax.random.PRNGKey(1236), 2)
params = jax.jit(net.init_params, static_argnums=(0,))(envs, subk)

obs_t = jax.random.normal(key, [20, 10, *envs.single_observation_space.shape])

logits, value, _ = jax.vmap(net.get_logits_and_value, in_axes=(None, 0))(params, obs_t)

logits_std = np.mean(np.std(logits, axis=(0, 1)))
print(f"{logits_std=}, {np.std(value)=}")


print("Total params:", sum(np.prod(x.shape) for x in jax.tree.leaves(params)) - 3_171_333)

# %%
matrix = jax.random.normal(jax.random.PRNGKey(1234), (3, 80, 80, 4)) / np.sqrt(3 * 80 * 80)

jnp.einsum("ijabc,abcd->ijd", obs_t, matrix).std()
