import jax
import math
import jax.numpy as jnp
import numpy as np
import flax.linen as nn

from cwgf.solvers.models.activation import ActivationFactory


class TimeEmbedding(nn.Module):
    dim: int
    mul: int = 1
    act: str = 'celu'

    @nn.compact
    def __call__(self, inputs):
        time_dim = self.dim * self.mul

        se = SinusoidalEmbedding(self.dim)(inputs)

        x = nn.Dense(time_dim)(se)
        x = ActivationFactory.create(self.act)(x)
        x = nn.Dense(time_dim)(x)

        return x


class SpaceEmbedding(nn.Module):
    sigma: float
    in_dim: int
    out_dim: int
    seed: int = 123

    def setup(self):
        rng = jax.random.PRNGKey(self.seed)

        self.W = jax.random.normal(
            rng,
            [self.out_dim // 2, self.in_dim]) * self.sigma


    def __call__(self, inputs):
        assert(inputs.shape[0] == self.in_dim)

        tmp = self.W @ inputs
        tmp = 2 * np.pi * tmp
        return jnp.concatenate([inputs, jnp.sin(tmp), jnp.cos(tmp)])


class SinusoidalEmbedding(nn.Module):
    dim: int

    @nn.compact
    def __call__(self, inputs):
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        # emb = math.log(100) / (half_dim - 1)
        emb = jnp.exp(jnp.arange(half_dim) * -emb)
        emb = jnp.expand_dims(inputs, -1) * emb
        # emb = jnp.expand_dims(10 * inputs, -1) * emb
        assert(emb.ndim == 1)
        emb = jnp.concatenate([jnp.sin(emb), jnp.cos(emb)], -1)
        return emb

