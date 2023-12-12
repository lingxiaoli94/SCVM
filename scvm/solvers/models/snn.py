'''
Network architecture for parameterizing time-varying scores, similar to
VNN.
'''

# TODO: merge this with VNN

import jax.numpy as jnp
from flax import linen as nn

from cwgf.solvers.models.activation import ActivationFactory
from cwgf.solvers.models.time_emb import TimeEmbedding


class SNN(nn.Module):
    dim: int
    num_layer: int
    layer_size: int
    activation_layer: str
    embed_time_dim: int # 0 if not embedding_time

    @nn.compact
    def __call__(self, t, x):
        '''
        Note this is not batched.

        Args:
          t: A scalar, time.
          x: (D,), a point to evaluate the velocity at.
        Returns:
          (D,), velocity at (x, t).
        '''
        assert(x.ndim == 1)
        if self.embed_time_dim > 0:
            t = TimeEmbedding(self.embed_time_dim)(t)
        else:
            t = jnp.expand_dims(t, -1)
        x = jnp.concatenate([x, t])
        for _ in range(self.num_layer):
            x = nn.Dense(self.layer_size)(x)
            x = ActivationFactory.create(self.activation_layer)(x)
        x = nn.Dense(self.dim)(x)

        return x
