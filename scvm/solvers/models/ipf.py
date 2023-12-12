'''
Identity push forward. Used for debugging.
'''

from cwgf.solvers.models.tdpf_base import TDPFBase

import flax.linen as nn

class IPF(TDPFBase):
    @nn.compact
    def __call__(self, T, X, reverse=False):
        Z = nn.Dense(1)(X)
        return X, (Z - Z)[..., 0]
