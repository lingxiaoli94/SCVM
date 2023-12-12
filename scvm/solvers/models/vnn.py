'''
Network architecture for parameterizing time-varying velocity fields.
'''

import jax.numpy as jnp
from flax import linen as nn

from cwgf.solvers.models.activation import ActivationFactory
from cwgf.solvers.models.time_emb import TimeEmbedding, SpaceEmbedding


class LinearSkipConnection(nn.Module):
    rank: int
    layer_size: int
    num_layer: int

    @nn.compact
    def __call__(self, t, x):
        '''
        Args:
          t: (T,) time after some embedding.
          x: (D,), a point to evaluate the velocity at.
        '''
        dim = x.shape[0]
        # t = TimeEmbedding(self.embed_time_dim)(t)
        out_size = (2 * self.rank + 1) * dim

        cur = t
        for i in range(self.num_layer):
            cur = nn.Dense(self.layer_size)(cur)
            cur = nn.silu(cur)
        Wb = nn.Dense(out_size)(cur)

        U, V, b = (Wb[:self.rank * dim],
                   Wb[self.rank * dim:2 * self.rank * dim],
                   Wb[-dim:])
        U = U.reshape(self.rank, dim)
        V = V.reshape(self.rank, dim)

        out = (jnp.transpose(U) @ (V @ jnp.expand_dims(x, -1))).squeeze(-1) + b
        return out


class VNN(nn.Module):
    dim: int
    num_layer: int
    layer_size: int
    activation_layer: str
    kernel_var: float
    ode_tol: float # used by VelocityView; put here for convenience
    use_diffrax: bool # whether use diffrax library or jax's odeint
    log_p_ode_mul: float
    embed_time_dim: int # 0 if not embedding_time
    embed_space_dim: int
    use_skip: bool
    use_residual: bool
    skip_only: bool
    layer_norm: bool

    @nn.compact
    def __call__(self, t, x, reverse):
        '''
        Note this is not batched.

        Args:
          t: A scalar, time.
          x: (D,), a point to evaluate the velocity at.
        Returns:
          (D,), velocity at (x, t).
        '''
        kernel_init = nn.initializers.variance_scaling(
            self.kernel_var, 'fan_in', 'truncated_normal')

        assert(x.ndim == 1)
        if self.skip_only:
            assert(self.use_skip)
        if self.embed_time_dim > 0:
            t = TimeEmbedding(self.embed_time_dim)(t)
        else:
            t = jnp.expand_dims(t, -1)
        if self.use_skip:
            x_skip = LinearSkipConnection(rank=20,
                                          layer_size=self.layer_size,
                                          num_layer=self.num_layer)(t, x)
        if self.embed_space_dim > 0:
            x = SpaceEmbedding(sigma=1.0,
                               in_dim=self.dim,
                               out_dim=self.embed_space_dim)(x)

        x_t_ori = jnp.concatenate([x, t])
        x = x_t_ori
        for i in range(self.num_layer):
            x = nn.Dense(self.layer_size, kernel_init=kernel_init)(x)
            if self.layer_norm:
                x = nn.LayerNorm()(x)
            if self.use_residual and i > 0:
                y = nn.Dense(self.layer_size, kernel_init=kernel_init)(x_t_ori)
                x += y
            x = ActivationFactory.create(self.activation_layer)(x)

        x = nn.Dense(self.dim, kernel_init=kernel_init)(x)

        if self.use_skip:
            if self.skip_only:
                x = x_skip
            else:
                x += x_skip

        if reverse:
            x = -x
        return x
