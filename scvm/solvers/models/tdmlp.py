from cwgf.solvers.models.tdpf_base import TDPFBase

from flax import linen as nn
import jax
import jax.numpy as jnp

from cwgf.solvers.models.activation import ActivationFactory

class TDMLP(TDPFBase):
    num_layer: int
    layer_size: int
    activation_layer: str
    soft_init: float

    @nn.compact
    def __call__(self, t, x, reverse=False):
        '''
        Note this is not batched.

        Args:
          t: A scalar, time.
          x: (D,), a point to evaluate the velocity at.
        Returns:
          (), potential at (x, t).
        '''
        assert(x.ndim == 1)
        y = x 
        for _ in range(self.num_layer):
            y = nn.Dense(self.layer_size)(jnp.append(y, t))
            y = ActivationFactory.create(self.activation_layer)(y)
        y = nn.Dense(self.dim)(y)
        if self.soft_init == 0.:
            return x + t * y, None
        else :
            return y, None


