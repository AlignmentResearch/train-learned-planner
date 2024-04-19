import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import rlax

from cleanba.config import sokoban_resnet
from cleanba.impala_loss import (
    Rollout,
)


def unreplicate(tree):
    """Returns a single instance of a replicated array."""
    return jax.tree_util.tree_map(lambda x: x[0], tree)


# %%

args = sokoban_resnet()
envs = dataclasses.replace(args.train_env, num_envs=64).make()

obs_t, _ = envs.reset()
params = jax.jit(args.net.init_params, static_argnums=(0,))(envs, jax.random.PRNGKey(1234), obs_t)

# %%
key = jax.random.PRNGKey(1234)

storage: list[Rollout] = []
for t in range(40):
    a_t, logits_t, key = args.net.get_action(params, obs_t, key)
    obs_t_plus_1, r_t, term_t, trunc_t, info_t = envs.step(np.array(a_t))
    done_t = term_t | trunc_t
    storage.append(
        Rollout(
            obs_t=obs_t,
            a_t=a_t,
            logits_t=logits_t,
            r_t=r_t,
            done_t=done_t,
            truncated_t=trunc_t,
        )
    )
    obs_t = obs_t_plus_1

out = Rollout(
    # Add the `last_obs` on the end of this rollout
    obs_t=jnp.stack([*(r.obs_t for r in storage), obs_t]),
    a_t=jnp.stack([r.a_t for r in storage]),
    logits_t=jnp.stack([r.logits_t for r in storage]),
    r_t=jnp.stack([r.r_t for r in storage]),
    done_t=jnp.stack([r.done_t for r in storage]),
    truncated_t=jnp.stack([r.truncated_t for r in storage]),
)

# %%

discount_t = (~out.done_t) * 0.97
errors = jax.vmap(rlax.vtrace, in_axes=1, out_axes=1)(
    np.zeros((40, 64)) - 2.33333333, np.zeros((40, 64)) - 2.33333333, out.r_t, discount_t, np.ones((40, 64))
)

errors
