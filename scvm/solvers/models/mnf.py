'''
Masked normalizing flow of Dinh et al. 2017.
'''

from scvm.solvers.models.tdpf_base import TDPFBase

import numpy as np
from flax import linen as nn
import jax
import jax.numpy as jnp
from typing import Any

from scvm.solvers.models.activation import ActivationFactory
from scvm.solvers.models.time_emb import TimeEmbedding

class BasicMLP(nn.Module):
    out_dim: int
    act: str

    @nn.compact
    def __call__(self, X):
        out = nn.Sequential([
            nn.Dense(64),
            ActivationFactory.create(self.act),
            nn.Dense(128),
            ActivationFactory.create(self.act),
            nn.Dense(128),
            ActivationFactory.create(self.act),
            nn.Dense(self.out_dim),
        ])(X)
        return out


class CouplingLayer(nn.Module):
    mask : np.ndarray # coordinates to keep identical
    soft_init: float
    ignore_time: bool
    act: str
    time_emb: Any

    def setup(self):
        dim = self.mask.shape[0]
        self.scaling_factor = self.param('scaling_factor',
                                         nn.initializers.zeros,
                                         (dim,))
        self.scale_net = BasicMLP(out_dim=dim,
                                  act=self.act)
        self.translate_net = BasicMLP(
            out_dim=dim, act=self.act)


    def __call__(self, t, x, reverse):
        assert(x.ndim == 1)

        if not self.ignore_time:
            if self.time_emb is not None:
                xt_cat = jnp.concatenate([x * self.mask, self.time_emb(t)])
            else:
                xt_cat = jnp.append(x * self.mask, t)
        else:
            xt_cat = x * self.mask
        # Modify scale/translate so that at t=0 the resulting map is identity.
        scale = self.scale_net(xt_cat)
        translate = self.translate_net(xt_cat)
        if not self.ignore_time and self.soft_init == 0.:
            scale = t * scale
            translate = t * translate

        sf = jnp.exp(self.scaling_factor)
        scale = nn.tanh(scale / sf) * sf

        scale = scale * (1 - self.mask)
        translate = translate * (1 - self.mask)

        if reverse:
            x = (x + translate) * jnp.exp(scale)
            ldj = scale.sum()
        else:
            x = (x * jnp.exp(-scale)) - translate
            ldj = -scale.sum()

        return x, ldj


class MNF(TDPFBase):
    '''
    Masked normalizing flow by Dinh et al.
    '''
    couple_mul: int
    mask_type: str # ['loop', 'random']
    soft_init: float # if 0, then use hard parameterization
    ignore_time: bool # whether to remove time dependency
    activation_layer: str
    embed_time_dim: int # 0 if not embedding_time

    def setup(self):
        if self.embed_time_dim > 0:
            self.time_emb = TimeEmbedding(self.embed_time_dim)
        else:
            self.time_emb = None
        couple_layers = []
        num_layer = (self.couple_mul if self.mask_type == 'random'
                     else self.dim * self.couple_mul)
        if self.mask_type == 'random':
            rng_state = np.random.RandomState(seed=888)
            prev_mask = np.zeros(self.dim, dtype=int)
        for i in range(num_layer):
            if self.mask_type == 'loop':
                # Change one coordinate at a time; okay in low dimensions.
                mask = np.ones(self.dim)
                mask[i % self.dim] = 0
            else:
                while True:
                    mask = rng_state.binomial(1, p=0.5, size=[self.dim])
                    if not (mask.sum() in [0, self.dim] or (mask == prev_mask).all()):
                        prev_mask = mask
                        break
            # print(f'mask {i} = {mask}')
            couple_layers.append(CouplingLayer(
                time_emb=self.time_emb,
                mask=mask,
                act=self.activation_layer,
                soft_init=self.soft_init,
                ignore_time=self.ignore_time))
        self.couple_layers = couple_layers


    def __call__(self, t, x0, reverse=False):
        ldj_sum = 0

        couple_layers = self.couple_layers
        if reverse:
            couple_layers = reversed(couple_layers)
        x = x0
        for layer in couple_layers:
            x, ldj = layer(t, x, reverse)
            ldj_sum += ldj
        return x, ldj_sum

