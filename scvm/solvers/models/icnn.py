'''
Time-dependent input convex neural network from Amos et al. 2017.

Code partially adapted from https://github.com/facebookresearch/w2ot.
'''

import numpy as np
from flax import linen as nn
import jax
import jax.numpy as jnp
from typing import Any, Sequence

from cwgf.solvers.models.activation import ActivationFactory

class PositiveDense(nn.Module):
    out_dim: int
    kernel_init: Any = None

    @nn.compact
    def __call__(self, inputs):
        '''
        Args:
           inputs: (..., D_in)
        Returns:
          (..., D_out)
        '''
        kernel = self.param(
            'kernel', self.kernel_init, (inputs.shape[-1], self.out_dim))
        kernel = nn.softplus(kernel)
        out = jax.lax.dot_general(
            inputs, kernel, (((inputs.ndim - 1,), (0,)), ((), ())))

        gain = 1. / inputs.shape[-1]
        out *= gain
        return out


'''
Convex quadratic layers from Korotin et al. 2021.
'''
class ConvexQuadratic(nn.Module):
    out_dim: int
    rank: int
    use_bias: bool

    @nn.compact
    def __call__(self, inputs):
        '''
        Args:
           inputs: (..., D_in)
        Returns:
          (..., D_out)
        '''
        in_dim = inputs.shape[-1]
        qd = self.param('quad_decomposed', nn.initializers.normal(),
                        (in_dim, self.rank, self.out_dim)) # (D_in, R, D_out)
        linear = nn.Dense(self.out_dim, use_bias=self.use_bias)
        quad = jnp.tensordot(inputs, qd, axes=(-1, 0)) # (..., R, D_out)
        quad = (quad ** 2).sum(-2) # (..., D_out)
        return quad + linear(inputs)


class ICNN(nn.Module):
    dim: int
    hidden_dims: Sequence[int]
    time_hidden_dims: Sequence[int]
    activation_layer: str
    quadratic_rank: int # if -1, don't use ConvexQuadratic layers
    icnn_tol: float # tolerance for backward solve
    soft_init: float # hard parameterization not implemented
    ignore_time: bool # whether to remove time dependency

    def setup(self):
        assert(len(self.hidden_dims) == len(self.time_hidden_dims))
        assert(self.ignore_time or self.soft_init > 0)
        kernel_init = nn.initializers.variance_scaling(
            1., 'fan_in', 'truncated_normal')

        # Notations follow Amos et al. 2017.
        self.w_ts = [nn.Dense(dim, kernel_init=kernel_init)
                     for dim in self.time_hidden_dims]
        self.w_zs = [PositiveDense(dim, kernel_init=kernel_init)
                     for dim in self.hidden_dims[1:] + (1,)]
        if not self.ignore_time:
            self.w_zus = [nn.Dense(dim, kernel_init=kernel_init)
                          for dim in self.hidden_dims]
            self.w_yus = [nn.Dense(self.dim, kernel_init=kernel_init)
                          for _ in self.hidden_dims + (1,)]
            self.w_us = [nn.Dense(dim, kernel_init=kernel_init, use_bias=False)
                         for dim in self.hidden_dims + (1,)]
        if self.quadratic_rank > 0:
            self.w_ys = [ConvexQuadratic(out_dim=dim,
                                         rank=min(self.quadratic_rank, dim),
                                         use_bias=True)
                         for dim in self.hidden_dims + (1,)]
        else:
            self.w_ys = [nn.Dense(dim, kernel_init=kernel_init)
                         for dim in self.hidden_dims + (1,)]

        self.log_alpha = self.param('log_alpha',
                                    nn.initializers.constant(0), [])
        self.activation = ActivationFactory.create(self.activation_layer)


    def __call__(self, t, x):
        '''
        Args:
          t: (...), time
          x: (..., D), inputs
        Returns:
          (...), a scalar
        '''
        num_hidden = len(self.hidden_dims)
        if not self.ignore_time:
            u = jnp.expand_dims(t, -1) # (..., -1)
        for i in range(num_hidden + 1):
            z_next = 0
            if i > 0:
                if self.ignore_time:
                    tmp = jnp.ones_like(z)
                else:
                    tmp = nn.softplus(self.w_zus[i-1](u))
                z_next += self.w_zs[i-1](z * tmp)
            if self.ignore_time:
                tmp = x
            else:
                tmp = x * self.w_yus[i](u)
            z_next += self.w_ys[i](tmp)
            if not self.ignore_time:
                z_next += self.w_us[i](u)
            z_next = self.activation(z_next)
            z = z_next

            if not self.ignore_time and i < num_hidden:
                u = self.activation(self.w_ts[i](u))

        z = jnp.squeeze(z, -1)
        z += jnp.exp(self.log_alpha) / 2 * (x * x).sum(-1)

        return z


