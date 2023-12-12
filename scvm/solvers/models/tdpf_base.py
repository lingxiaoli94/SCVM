from abc import ABC, abstractmethod

import jax
import jax.numpy as jnp
from flax import linen as nn


class TDPFBase(nn.Module, ABC):
    '''
    Base class for a jax model of time-dependent push forward (TDPF) of
    the form (x, t) -> (y).
    '''
    dim: int

    @abstractmethod
    def __call__(self, t, x0, reverse):
        '''
        Args:
          t: ()
          x0: (D,)
          reverse: bool
        Returns:
          A tuple, (xt, ldj) where
            xt: (D,), pushed samples
            ldj:
              () or None, log|detJT|(x).
              Hence density at xt is log_p(x) - ldj.
        '''
        pass

