import dataclasses
import random
from typing import Any

import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np

from cleanba.config import random_seed
from cleanba.network import (
    Actor,
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
    strides=(1, 1, 1, 1, 1, 1, 1, 1, 1),
    kernel_sizes=(4, 4, 4, 4, 4, 4, 4, 4, 4),
    mlp_hiddens=(256,),
)


@jax.jit
def var_fn(key):
    key, subk = jax.random.split(key, 2)
    envs = Envs(gym.spaces.Discrete(4), gym.spaces.Box(0, 255, (3, 10, 10)))
    params = net.init_params(envs, subk)
    obs_t = jax.random.normal(key, [20, 10, *envs.single_observation_space.shape])

    print("Total params:", sum(np.prod(x.shape) for x in jax.tree.leaves(params)) - 3_171_333)

    logits, value, _ = jax.vmap(net.get_logits_and_value, in_axes=(None, 0))(params, obs_t)
    return jnp.var(logits, ddof=0)


print(np.mean([var_fn(jax.random.PRNGKey(random_seed())).item() for _ in range(20)]))

# %%
# matrix = jax.random.normal(jax.random.PRNGKey(1234), (3, 80, 80, 4)) / np.sqrt(3 * 80 * 80)

# jnp.einsum("ijabc,abcd->ijd", 2**0.5 * jax.nn.relu(obs_t), matrix).std()

# %%
layer = Actor(4, yang_init=True, norm=IdentityNorm())
key = jax.random.PRNGKey(random.randint(0, 1e9))
obs_t = jax.random.normal(key, [200, 10, 19200])
p = layer.init(key, obs_t)
layer.apply(p, 2**0.5 * jax.nn.relu(obs_t))[0].std()
