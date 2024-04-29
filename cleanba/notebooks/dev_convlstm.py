import typing

import flax.linen as nn
import jax
import jax.numpy as jnp

T = typing.TypeVar("T")


def f(acc: T, x) -> tuple[T, jax.Array]:
    return acc, x + x


# %% as-is

lowered = jax.jit(f).lower((), jnp.ones((2, 3)))

print("=== LOWERED ===")
print(lowered.as_text())
print("=== COMPILED ===")
print(lowered.compile().as_text())

# %% Vmapped


def g(acc, x):
    return jax.vmap(f)(acc, x)


lowered = jax.jit(g).lower((), jnp.ones((2, 3)))

print("=== LOWERED ===")
print(lowered.as_text())
print("=== COMPILED ===")
print(lowered.compile().as_text())

# %% Scanned


def g(acc, x):
    return jax.lax.scan(f, acc, x)


lowered = jax.jit(g).lower((), jnp.ones((2, 3)))

print("=== LOWERED ===")
print(lowered.as_text())
print("=== COMPILED ===")
print(lowered.compile().as_text())


# %% Conclusion
"""The jitted loop is significantly more complicated than the jitted vmap function. Thus, it would not be wise to force
all NNs (even non-recurrent ones) to be scanned.
"""


# %%


class MLP(nn.Module):
    def setup(self):
        # Submodule names are derived by the attributes you assign to. In this
        # case, "dense1" and "dense2". This follows the logic in PyTorch.
        self.dense1 = nn.Dense(32)
        self.dense2 = nn.Dense(32)

    def forward(self, x):
        x = self.dense1(x)
        x = nn.relu(x)
        x = self.dense2(x)
        return x


mod = MLP()
params = mod.init(jax.random.PRNGKey(1234), jnp.ones((3, 4)), method=MLP.forward)

mod.apply(params, jnp.ones((4, 4)), method=MLP.forward)
